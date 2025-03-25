import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import build_encoder
from servers.build import SERVER_REGISTRY
from utils.logging_utils import AverageMeter

logger = logging.getLogger(__name__)

@SERVER_REGISTRY.register()
class Server:
    def __init__(self, args):
        self.args = args
        
        # Trust-based client filtering configuration
        trust_config = getattr(args.server, "trust_filtering", {})
        self.trust_threshold = getattr(trust_config, "threshold", 0.5)
        self.enable_trust_filtering = getattr(trust_config, "enable", False)
        self.min_trusted_clients = getattr(trust_config, "min_trusted_clients", 1)
        self.soft_filtering = getattr(trust_config, "soft_filtering", True)
        
        # Personalization configuration
        personalization_config = getattr(args.server, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.personalization_layers = getattr(personalization_config, "layers", 2)
        
        # Adaptive learning rate configuration
        learning_rate_config = getattr(args.server, "adaptive_lr", {})
        self.enable_adaptive_lr = getattr(learning_rate_config, "enable", False)
        self.base_lr = getattr(learning_rate_config, "base_lr", 0.01)
        self.max_lr = getattr(learning_rate_config, "max_lr", 0.1)
        
        # Multi-level contrastive learning configuration
        contrastive_config = getattr(args.server, "contrastive", {})
        self.enable_multi_level_cl = getattr(contrastive_config, "multi_level", False)
        self.layer_weights = getattr(contrastive_config, "layer_weights", [0.2, 0.3, 0.5])
        
        # Server state tracking
        self.trust_scores = {}
        self.round_stats = {}
        self.client_history = {}
        self.round_num = 0
        
        # Models
        self.model = None
        self.global_model = None
        self.global_model_state_dict = None

    def setup(self, model):
        """Initialize server with a model"""
        self.model = model
        self.global_model = copy.deepcopy(model)
        
        # Ensure both models are on the same device
        device = next(model.parameters()).device
        self.global_model = self.global_model.to(device)
        
        # Copy state dict after ensuring device match
        self.global_model_state_dict = copy.deepcopy(self.global_model.state_dict())
        
        # Initialize personalized head if enabled
        if self.enable_personalization and hasattr(model, 'enable_personalized_mode'):
            model.enable_personalized_mode()
            logger.info("Personalized mode enabled for the global model")

    def select_clients(self, clients, num_clients):
        """Select clients for the current round based on trust scores if available"""
        if num_clients >= len(clients):
            return clients
        
        if self.enable_trust_filtering and len(self.trust_scores) > 0:
            # Select clients based on trust scores with probability proportional to score
            client_scores = np.array([self.trust_scores.get(c, 1.0) for c in clients])
            client_scores = np.maximum(client_scores, 0.01)  # Ensure minimum probability
            client_probs = client_scores / client_scores.sum()
            
            selected_clients = np.random.choice(
                clients, 
                size=num_clients, 
                replace=False, 
                p=client_probs
            ).tolist()
            
            logger.info(f"Selected {num_clients} clients based on trust scores")
            return selected_clients
        else:
            # Random selection
            return np.random.choice(clients, num_clients, replace=False).tolist()

    def aggregate(self, client_models, client_weights=None, client_stats=None):
        """Aggregate client models using trust-based weighted averaging"""
        # Handle empty client models case
        if not client_models:
            logger.warning("No client models to aggregate")
            return self.global_model.state_dict()
            
        self.round_num += 1
        
        # Determine device to use
        device = next(self.global_model.parameters()).device
        
        # Apply trust-based filtering if enabled
        if self.enable_trust_filtering and client_stats:
            trusted_clients = {}
            trusted_weights = {}
            trusted_stats = {}
            
            # Calculate average trust score for reference
            all_trust_scores = [float(stats.get('trust_score', 0.5)) for client_id, stats in client_stats.items() 
                               if client_id in client_models]
            avg_trust_score = sum(all_trust_scores) / len(all_trust_scores) if all_trust_scores else 0.5
            
            for client_id, stats in client_stats.items():
                if client_id in client_models:
                    trust_score = float(stats.get('trust_score', 0.5))
                    self.trust_scores[client_id] = trust_score
                    
                    # Track client trust history
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append(trust_score)
                    if len(self.client_history[client_id]) > 10:
                        self.client_history[client_id].pop(0)
                    
                    # Dynamic trust threshold based on average score
                    dynamic_threshold = max(0.3, min(0.7, avg_trust_score * 0.8))
                    
                    # Apply soft or hard filtering based on configuration
                    if self.soft_filtering:
                        # Soft filtering: use client with adjusted weight based on trust
                        trusted_clients[client_id] = client_models[client_id]
                        
                        if client_weights:
                            base_weight = float(client_weights.get(client_id, 1.0))
                            # Sigmoid function to make weights more representative of trust differences
                            # This creates more separation between high and low trust clients
                            trust_factor = 1.0 / (1.0 + np.exp(-10 * (trust_score - dynamic_threshold)))
                            adjusted_weight = base_weight * max(0.2, trust_factor)
                            trusted_weights[client_id] = adjusted_weight
                            
                            # Log significant weight adjustments
                            if trust_factor < 0.5:
                                logger.info(f"Client {client_id} weight reduced to {trust_factor:.2f} (trust={trust_score:.3f}, threshold={dynamic_threshold:.3f})")
                        
                        trusted_stats[client_id] = stats
                    else:
                        # Hard filtering: only include clients above threshold
                        if trust_score >= self.trust_threshold:
                            trusted_clients[client_id] = client_models[client_id]
                            if client_weights:
                                trusted_weights[client_id] = float(client_weights.get(client_id, 1.0))
                            trusted_stats[client_id] = stats
                            logger.info(f"Client {client_id} trusted with score {trust_score:.4f}")
                        else:
                            logger.info(f"Client {client_id} filtered out with score {trust_score:.4f}")
            
            # Log trust distribution statistics
            if all_trust_scores:
                logger.info(f"Trust scores - Avg: {avg_trust_score:.3f}, Min: {min(all_trust_scores):.3f}, Max: {max(all_trust_scores):.3f}")
            
            # Ensure minimum number of clients for aggregation
            if len(trusted_clients) < self.min_trusted_clients:
                # Sort clients by trust score and take top N
                sorted_clients = sorted(
                    [(cid, self.trust_scores.get(cid, 0.0)) for cid in client_models.keys()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Use at least min_trusted_clients
                for cid, score in sorted_clients[:self.min_trusted_clients]:
                    if cid not in trusted_clients and cid in client_models:
                        trusted_clients[cid] = client_models[cid]
                        if client_weights:
                            # Apply minimum weight for low-trust but needed clients
                            trusted_weights[cid] = client_weights.get(cid, 1.0) * max(0.2, score)
                        if cid in client_stats:
                            trusted_stats[cid] = client_stats[cid]
                
                logger.warning(f"Only {len(trusted_clients)-self.min_trusted_clients} trusted clients (min: {self.min_trusted_clients}). Added additional clients.")
            
            # Log participation statistics
            participation_rate = len(trusted_clients) / len(client_models) if client_models else 0
            logger.info(f"Client participation: {len(trusted_clients)}/{len(client_models)} ({participation_rate:.2%})")
            
            client_models = trusted_clients
            if client_weights:
                client_weights = trusted_weights
            client_stats = trusted_stats
        
        # Return current model if no models to aggregate
        if not client_models:
            logger.warning("No trusted clients after filtering. Using current global model.")
            return self.global_model.state_dict()
        
        # Normalize client weights
        if client_weights:
            weight_sum = sum(client_weights.values())
            if weight_sum > 0:
                client_weights = {k: v / weight_sum for k, v in client_weights.items()}
            else:
                # Equal weights if sum is zero
                client_weights = {k: 1.0 / len(client_models) for k in client_models.keys()}
        else:
            # Default to equal weights
            client_weights = {k: 1.0 / len(client_models) for k in client_models.keys()}
        
        # Weighted averaging of model parameters
        avg_state_dict = {}
        reference_state_dict = next(iter(client_models.values()))
        
        # Initialize average state dict with zeros
        for key in reference_state_dict.keys():
            # Skip personalized head parameters if personalization is enabled
            if self.enable_personalization and 'personalized_head' in key:
                continue
                
            # Get tensor from reference and ensure it's on the correct device
            tensor = reference_state_dict[key].to(device)
            # Ensure tensor is float for aggregation
            if tensor.dtype == torch.int64 or tensor.dtype == torch.long:
                tensor = tensor.float()
            avg_state_dict[key] = torch.zeros_like(tensor)
        
        # Add up weighted parameters
        for client_id, state_dict in client_models.items():
            weight = float(client_weights.get(client_id, 1.0 / len(client_models)))
            
            for key in avg_state_dict.keys():
                if key in state_dict:
                    # Move client parameter to the correct device and apply weight
                    tensor = state_dict[key].to(device)
                    # Ensure tensor is float for aggregation
                    if tensor.dtype == torch.int64 or tensor.dtype == torch.long:
                        tensor = tensor.float()
                    avg_state_dict[key] += weight * tensor
                else:
                    logger.warning(f"Key {key} missing from client {client_id}")
        
        # Apply model averaging stabilization for BatchNorm layers
        for key in avg_state_dict.keys():
            # Special handling for batch norm running statistics
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # Use exponential moving average for running statistics
                current_value = self.global_model_state_dict.get(key, None)
                if current_value is not None:
                    decay = min(0.9, 1.0 - 1.0 / (self.round_num + 1))
                    current_value = current_value.to(device)
                    avg_state_dict[key] = decay * current_value + (1 - decay) * avg_state_dict[key]
        
        # Update round statistics
        self.round_stats = {
            'num_clients': len(client_models),
            'trust_scores': {k: v for k, v in self.trust_scores.items() if k in client_models},
            'client_stats': client_stats,
            'client_history': self.client_history,
            'round': self.round_num
        }
        
        # Update global model state dict
        self.global_model_state_dict = avg_state_dict
        self.global_model.load_state_dict(avg_state_dict, strict=False)
        
        return avg_state_dict

    def update_global_model(self, client_models, client_weights=None, client_stats=None):
        """Update the global model using aggregated client models"""
        if not client_models:
            logger.warning("No client models received. Global model unchanged.")
            return self.global_model.state_dict()
            
        local_weights = {}
        local_deltas = {}
        client_ids = list(client_models.keys())
        
        for i, client_id in enumerate(client_ids):
            local_weights[i] = client_models[client_id]
            local_deltas[i] = {}
            
        model_dict = self.global_model.state_dict()
        
        # Get current learning rate for clients
        current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        # If adaptive learning rate is enabled, calculate new rates based on trust scores
        if self.enable_adaptive_lr and client_stats:
            trust_avg = np.mean([stats.get('trust_score', 1.0) for stats in client_stats.values()])
            adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * trust_avg
            logger.info(f"Adjusted global learning rate to {adjusted_lr:.6f} based on average trust {trust_avg:.4f}")
            # Store for next round
            self.args.trainer.local_lr = adjusted_lr
        
        # Perform aggregation
        aggregated_params = self.aggregate(client_models, client_weights, client_stats)
        
        # Apply aggregated parameters to global model
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            if key in aggregated_params:
                global_state_dict[key] = aggregated_params[key]
        
        self.global_model.load_state_dict(global_state_dict, strict=False)
        self.global_model_state_dict = global_state_dict
        
        return global_state_dict
        
    def get_global_model(self):
        """Return the current global model"""
        return self.global_model
        
    def get_round_stats(self):
        """Return statistics from the current round"""
        return self.round_stats

    def initialize(self, model):
        """Alias for setup to maintain compatibility"""
        return self.setup(model)


@SERVER_REGISTRY.register()
class ServerM(Server):
    """Server with momentum-based aggregation"""
    def __init__(self, args):
        super(ServerM, self).__init__(args)
        self.momentum = getattr(args.server, "momentum", 0.0)
        self.use_fedacg = getattr(args.server, "fedacg", False)
        self.dampening = getattr(args.server, "dampening", 0.0)
        self.nesterov = getattr(args.server, "nesterov", False)
        
    def setup(self, model):
        """Initialize server with a model and setup momentum buffers"""
        super().setup(model)
        self.set_momentum(model)
    
    def set_momentum(self, model):
        """Initialize momentum buffers for the model"""
        device = next(model.parameters()).device
        self.global_delta = {key: torch.zeros_like(val).to(device) for key, val in model.state_dict().items()}
        self.global_momentum = {key: torch.zeros_like(val).to(device) for key, val in model.state_dict().items()}

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        """Apply FedACG lookahead step to the model"""
        sending_model_dict = copy.deepcopy(model.state_dict())
        device = next(model.parameters()).device
        
        for key in self.global_momentum:
            # Ensure both tensors are on the same device
            momentum = self.global_momentum[key].to(device)
            if key in sending_model_dict:
                sending_model_dict[key] = sending_model_dict[key].to(device) + self.momentum * momentum
                
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        """Aggregate client models with momentum"""
        if client_metrics is None:
            client_metrics = {cid: {"trust_score": 1.0} for cid in client_ids}
            
        trust_scores = {cid: client_metrics.get(cid, {}).get("trust_score", 1.0) for cid in client_ids}
        
        # Determine device to use
        device = next(iter(model_dict.values())).device
        
        if self.enable_trust_filtering:
            if self.soft_filtering:
                # Soft filtering: adjust weights by trust score
                trusted_clients = client_ids
                # Update trust scores for soft weighting
                for cid in trusted_clients:
                    if cid in trust_scores:
                        if trust_scores[cid] < self.trust_threshold:
                            # Scale down impact of less trusted clients
                            trust_scores[cid] *= trust_scores[cid]/self.trust_threshold
            else:
                # Hard filtering: only include clients above threshold
                trusted_clients = [cid for cid in client_ids if trust_scores.get(cid, 0) >= self.trust_threshold]
                if not trusted_clients:
                    trusted_clients = client_ids
                    logger.warning("No trusted clients found, using all clients")
        else:
            trusted_clients = client_ids

        # Calculate sum of trust scores for normalization
        trust_sum = sum(trust_scores.get(cid, 1.0) for cid in trusted_clients)
        if trust_sum <= 0:
            trust_sum = len(trusted_clients)
            trust_scores = {cid: 1.0 for cid in trusted_clients}

        # Initialize weighted average dictionary
        avg_weights = {}
        
        # Compute trust-weighted average of parameters
        for param_key in model_dict.keys():
            # Skip personalized head parameters if personalization is enabled
            if isinstance(param_key, str) and self.enable_personalization and 'personalized_head' in param_key:
                continue
            
            # Check if all trusted clients have this parameter
            valid_clients = [i for i, cid in enumerate(client_ids) 
                               if cid in trusted_clients and 
                               i in local_weights and 
                               param_key in local_weights[i]]
            
            if valid_clients:
                # Compute weighted sum across valid clients, ensuring all tensors are on the same device
                weighted_sum = sum(
                    local_weights[i][param_key].to(device) * trust_scores.get(client_ids[i], 1.0) / trust_sum
                    for i in valid_clients
                )
                avg_weights[param_key] = weighted_sum

        # Apply momentum if enabled
        if self.momentum > 0:
            if not self.use_fedacg:
                # Standard momentum update
                for param_key in avg_weights:
                    if param_key in self.global_momentum:
                        momentum = self.global_momentum[param_key].to(device)
                        avg_weights[param_key] += self.momentum * momentum

            # Update momentum buffers
            for param_key in model_dict.keys():
                if isinstance(param_key, str) and self.enable_personalization and 'personalized_head' in param_key:
                    continue
                
                # Compute weighted delta
                valid_clients = [i for i, cid in enumerate(client_ids) 
                                if cid in trusted_clients and 
                                i in local_deltas and 
                                param_key in local_deltas[i]]
                
                if valid_clients:
                    # Ensure all tensors are on the same device
                    self.global_delta[param_key] = sum(
                        local_deltas[i][param_key].to(device) * trust_scores.get(client_ids[i], 1.0) / trust_sum
                        for i in valid_clients
                    )
                    
                    # Apply momentum update to buffer
                    self.global_momentum[param_key] = (
                        self.momentum * self.global_momentum[param_key].to(device) +
                        (1 - self.dampening) * self.global_delta[param_key]
                    )
                    
                    # Apply Nesterov momentum if enabled
                    if self.nesterov and param_key in avg_weights:
                        avg_weights[param_key] += self.global_delta[param_key]
        
        # Update model dictionary with averaged weights
        for key in avg_weights:
            if key in model_dict:
                model_dict[key] = avg_weights[key]
        
        # Update trust scores and round statistics
        self.trust_scores = trust_scores
        self.round_stats = {
            'num_clients': len(trusted_clients),
            'trust_scores': {k: v for k, v in trust_scores.items() if k in trusted_clients},
            'client_metrics': client_metrics,
            'round': self.round_num
        }
        
        # Update global model state dict
        self.global_model_state_dict = model_dict
        self.global_model.load_state_dict(model_dict, strict=False)
                
        return model_dict


@SERVER_REGISTRY.register()
class PersonalizedServer(ServerM):
    """Server supporting personalized client models"""
    def __init__(self, args):
        super(PersonalizedServer, self).__init__(args)
        self.enable_personalization = True
        
        # Personalization configuration
        personalization_config = getattr(args.server, "personalization", {})
        self.knowledge_distillation = getattr(personalization_config, "knowledge_distillation", True)
        self.kd_temperature = getattr(personalization_config, "kd_temperature", 2.0)
        
    def setup(self, model):
        """Initialize personalized server with a model"""
        super().setup(model)
        if hasattr(self, 'set_momentum'):
            self.set_momentum(model)
        
        # Enable personalized mode if the model supports it
        if hasattr(model, 'enable_personalized_mode'):
            model.enable_personalized_mode()
            logger.info("Personalized mode enabled for the global model")
        
    def update_global_model(self, client_models, client_weights=None, client_stats=None):
        """Update global model excluding personalized head parameters"""
        if not client_models:
            logger.warning("No client models received. Global model unchanged.")
            return self.global_model.state_dict()
            
        local_weights = {}
        local_deltas = {}
        client_ids = list(client_models.keys())
        
        # Determine which device to use (prefer CUDA if available)
        device = next(self.global_model.parameters()).device
        
        # Ensure global model state dict is on the same device
        global_state_dict = {}
        for k, v in self.global_model_state_dict.items():
            global_state_dict[k] = v.to(device)
        
        # Extract non-personalized parameters from client models
        for i, client_id in enumerate(client_ids):
            local_weights[i] = {}
            local_deltas[i] = {}
            
            # Filter out personalized head parameters and ensure all tensors are on the same device
            for key, value in client_models[client_id].items():
                if 'personalized_head' not in key:
                    # Ensure value is on the same device as the global model
                    value_on_device = value.to(device)
                    local_weights[i][key] = value_on_device
                    
                    # Calculate deltas from previous global model
                    if key in global_state_dict:
                        local_deltas[i][key] = value_on_device - global_state_dict[key]
        
        # Ensure model_dict is on the correct device
        model_dict = {}
        for k, v in self.global_model.state_dict().items():
            model_dict[k] = v.to(device)
            
        current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        # Ensure client_stats has the required format
        if client_stats:
            for client_id in client_stats:
                # Add 'loss' field if missing, using global_loss or default 0.0
                if 'loss' not in client_stats[client_id]:
                    client_stats[client_id]['loss'] = client_stats[client_id].get('global_loss', 0.0)
        
        # Aggregate using the momentum-based aggregation method
        aggregated_params = self.aggregate(
            local_weights, 
            local_deltas, 
            client_ids, 
            model_dict, 
            current_lr, 
            client_stats
        )
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        self.global_model_state_dict = copy.deepcopy(aggregated_params)
        
        return aggregated_params
