import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import build_encoder
from servers.build import SERVER_REGISTRY
from utils.logging_utils import AverageMeter
from collections import defaultdict

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
        """Initialize server with model and hyperparameters"""
        # Move model to CPU for safe deepcopy operation
        model_device = next(model.parameters()).device
        cpu_model = model.to('cpu')
        
        # Use safer copy method to avoid OOM
        try:
            self.global_model = copy.deepcopy(cpu_model)
        except Exception as e:
            logger.error(f"Error during model copy: {str(e)}")
            # Fallback method - create new model and load state dict
            if hasattr(cpu_model, '__class__'):
                model_class = cpu_model.__class__
                self.global_model = model_class(**getattr(cpu_model, 'init_args', {}))
                self.global_model.load_state_dict(cpu_model.state_dict())
            else:
                raise e
        
        # Move models back to original device
        model.to(model_device)
        self.global_model.to(model_device)
        
        # Initialize metrics and tracking
        self.client_trust_scores = {}
        self.client_histories = defaultdict(list)
        self.client_contributions = defaultdict(float)
        self.central_momentum_buffer = {}
        
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

    def aggregate(self, local_weights, client_ids, trust_scores=None, device='cpu'):
        """
        Aggregate local client models with trust-based weighting
        Args:
            local_weights: list of client model state dictionaries
            client_ids: list of client IDs corresponding to the weights
            trust_scores: dict mapping client IDs to their trust scores
            device: device to perform aggregation on (default: cpu to save GPU memory)
        Returns:
            aggregated model parameters
        """
        if not local_weights:
            logger.warning("No client weights to aggregate")
            return self.global_model.state_dict()
        
        if trust_scores is None:
            trust_scores = {client_id: 1.0 for client_id in client_ids}
        
        # Calculate trust score sum for normalization
        trust_sum = sum(trust_scores.get(client_id, 1.0) for client_id in client_ids)
        if trust_sum == 0:
            logger.warning("Sum of trust scores is 0, using uniform weighting")
            trust_scores = {client_id: 1.0 for client_id in client_ids}
            trust_sum = len(client_ids)
        
        # Get keys from the first client's model
        keys = list(local_weights[0].keys())
        
        # Create a CPU copy of the global model state dict as template
        global_state = {k: v.clone().detach().to('cpu') for k, v in self.global_model.state_dict().items()}
        
        # Perform aggregation on CPU to avoid OOM
        for param_key in keys:
            try:
                # First move all tensors to CPU and perform weighted sum
                weighted_sum = sum(
                    local_weights[i][param_key].to('cpu') * trust_scores.get(client_ids[i], 1.0) / trust_sum
                    for i in range(len(local_weights))
                )
                global_state[param_key] = weighted_sum
            except Exception as e:
                logger.error(f"Error aggregating parameter {param_key}: {str(e)}")
                # Keep global model's parameter if aggregation fails
                global_state[param_key] = self.global_model.state_dict()[param_key].clone().detach().to('cpu')
        
        # Apply global momentum if enabled
        if hasattr(self, 'global_momentum') and self.global_momentum > 0:
            self._apply_global_momentum(global_state, device='cpu')
        
        return global_state

    def update_global_model(self, client_models, client_weights=None, client_stats=None):
        """Update global model with aggregated client models"""
        self.round_num = getattr(self, 'round_num', 0) + 1
        
        # Convert client models dictionary to lists for the new aggregate method
        client_ids = list(client_models.keys())
        local_weights = [client_models[client_id] for client_id in client_ids]
        
        # Prepare trust scores if client_stats is available
        trust_scores = {}
        if client_stats:
            for client_id, stats in client_stats.items():
                if client_id in client_models:
                    trust_scores[client_id] = float(stats.get('trust_score', 0.5))
        
        # Determine device to use
        device = next(self.global_model.parameters()).device
        
        # Apply trust-based filtering if enabled
        if self.enable_trust_filtering and client_stats:
            # Log trust distribution statistics
            if trust_scores:
                avg_trust_score = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0.5
                min_trust = min(trust_scores.values()) if trust_scores else 0.0
                max_trust = max(trust_scores.values()) if trust_scores else 0.0
                logger.info(f"Trust scores - Avg: {avg_trust_score:.3f}, Min: {min_trust:.3f}, Max: {max_trust:.3f}")
            
            # Track client trust history
            for client_id, score in trust_scores.items():
                if not hasattr(self, 'client_history'):
                    self.client_history = {}
                if client_id not in self.client_history:
                    self.client_history[client_id] = []
                self.client_history[client_id].append(score)
                if len(self.client_history[client_id]) > 10:
                    self.client_history[client_id].pop(0)
        
        # Perform aggregation on CPU to avoid OOM errors
        aggregated_params = self.aggregate(
            local_weights=local_weights,
            client_ids=client_ids,
            trust_scores=trust_scores,
            device='cpu'  # Force CPU aggregation to save GPU memory
        )
        
        # Move aggregated params to the correct device and update global model
        try:
            for key in aggregated_params:
                aggregated_params[key] = aggregated_params[key].to(device)
            self.global_model.load_state_dict(aggregated_params, strict=False)
        except Exception as e:
            logger.error(f"Error loading aggregated parameters: {str(e)}")
            # Fallback: keep current global model if there's an error
        
        # Update round statistics
        self.round_stats = {
            'num_clients': len(client_models),
            'trust_scores': trust_scores,
            'client_stats': client_stats,
            'client_history': getattr(self, 'client_history', {}),
            'round': self.round_num
        }
        
        return aggregated_params

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
