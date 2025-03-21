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
        self.global_model_state_dict = copy.deepcopy(model.state_dict())
        
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
        
        # Apply trust-based filtering if enabled
        if self.enable_trust_filtering and client_stats:
            trusted_clients = {}
            trusted_weights = {}
            trusted_stats = {}
            
            for client_id, stats in client_stats.items():
                if client_id in client_models:
                    trust_score = stats.get('trust_score', 1.0)
                    self.trust_scores[client_id] = trust_score
                    
                    # Track client trust history
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append(trust_score)
                    if len(self.client_history[client_id]) > 10:
                        self.client_history[client_id].pop(0)
                    
                    # Apply soft or hard filtering based on configuration
                    if self.soft_filtering:
                        # Soft filtering: use client with adjusted weight based on trust
                        trusted_clients[client_id] = client_models[client_id]
                        if client_weights:
                            trusted_weights[client_id] = client_weights.get(client_id, 1.0) * trust_score
                        trusted_stats[client_id] = stats
                        logger.info(f"Client {client_id} weighted by trust score {trust_score:.4f}")
                    else:
                        # Hard filtering: only include clients above threshold
                        if trust_score >= self.trust_threshold:
                            trusted_clients[client_id] = client_models[client_id]
                            if client_weights:
                                trusted_weights[client_id] = client_weights.get(client_id, 1.0)
                            trusted_stats[client_id] = stats
                            logger.info(f"Client {client_id} trusted with score {trust_score:.4f}")
                        else:
                            logger.info(f"Client {client_id} filtered out with score {trust_score:.4f}")
            
            # Ensure minimum number of clients for aggregation
            if len(trusted_clients) < self.min_trusted_clients:
                logger.warning(f"Only {len(trusted_clients)} trusted clients (min: {self.min_trusted_clients}). Using all clients.")
                trusted_clients = client_models
                trusted_weights = client_weights
                trusted_stats = client_stats
            
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
        
        # Weighted averaging of model parameters
        avg_state_dict = {}
        reference_state_dict = next(iter(client_models.values()))
        
        # Initialize average state dict with zeros
        for key in reference_state_dict.keys():
            # Skip personalized head parameters if personalization is enabled
            if self.enable_personalization and 'personalized_head' in key:
                continue
                
            avg_state_dict[key] = torch.zeros_like(reference_state_dict[key])
        
        # Weighted sum of model parameters
        for client_id, state_dict in client_models.items():
            weight = client_weights.get(client_id, 1.0 / len(client_models)) if client_weights else 1.0 / len(client_models)
            
            for key in avg_state_dict.keys():
                if key in state_dict:
                    avg_state_dict[key] += weight * state_dict[key]
        
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
        self.global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        """Apply FedACG lookahead step to the model"""
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum:
            sending_model_dict[key] += self.momentum * self.global_momentum[key]
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        """Aggregate client models with momentum"""
        if client_metrics is None:
            client_metrics = {cid: {"trust_score": 1.0} for cid in client_ids}
            
        trust_scores = {cid: client_metrics.get(cid, {}).get("trust_score", 1.0) for cid in client_ids}
        
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
                # Compute weighted sum across valid clients
                weighted_sum = sum(
                    local_weights[i][param_key] * trust_scores.get(client_ids[i], 1.0) / trust_sum
                    for i in valid_clients
                )
                avg_weights[param_key] = weighted_sum

        # Apply momentum if enabled
        if self.momentum > 0:
            if not self.use_fedacg:
                # Standard momentum update
                for param_key in avg_weights:
                    if param_key in self.global_momentum:
                        avg_weights[param_key] += self.momentum * self.global_momentum[param_key]

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
                    self.global_delta[param_key] = sum(
                        local_deltas[i][param_key] * trust_scores.get(client_ids[i], 1.0) / trust_sum
                        for i in valid_clients
                    )
                    
                    # Apply momentum update to buffer
                    self.global_momentum[param_key] = (
                        self.momentum * self.global_momentum[param_key] +
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
        
        # Extract non-personalized parameters from client models
        for i, client_id in enumerate(client_ids):
            local_weights[i] = {}
            local_deltas[i] = {}
            
            # Filter out personalized head parameters
            for key, value in client_models[client_id].items():
                if 'personalized_head' not in key:
                    local_weights[i][key] = value
                    
                    # Calculate deltas from previous global model
                    if key in self.global_model_state_dict:
                        local_deltas[i][key] = value - self.global_model_state_dict[key]
            
        model_dict = self.global_model.state_dict()
        current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        # Aggregate using the momentum-based aggregation method
        aggregated_params = self.aggregate(
            local_weights, 
            local_deltas, 
            client_ids, 
            model_dict, 
            current_lr, 
            client_stats
        )
        
        self.global_model.load_state_dict(aggregated_params, strict=False)
        self.global_model_state_dict = aggregated_params
        
        return aggregated_params
