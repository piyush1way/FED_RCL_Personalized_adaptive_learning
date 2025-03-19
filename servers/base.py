import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from models import build_encoder
from servers.build import SERVER_REGISTRY
from utils.logging_utils import AverageMeter

logger = logging.getLogger(__name__)

@SERVER_REGISTRY.register()
class Server:
    def __init__(self, args):
        self.args = args
        # Initialize trust filtering settings
        trust_config = getattr(args.server, "trust_filtering", {})
        self.trust_threshold = getattr(trust_config, "threshold", 0.5)
        self.enable_trust_filtering = getattr(trust_config, "enable", False)
        self.min_trusted_clients = getattr(trust_config, "min_trusted_clients", 1)
        
        # Personalization settings
        personalization_config = getattr(args.server, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        
        # Metrics tracking
        self.trust_scores = {}
        self.round_stats = {}

    def setup(self, model):
        """Initialize server model"""
        self.model = model
        self.global_model = copy.deepcopy(model)

    def select_clients(self, clients, num_clients):
        """Select clients for the current round"""
        if num_clients >= len(clients):
            return clients
        
        return np.random.choice(clients, num_clients, replace=False).tolist()

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        """
        Trust-based Federated Averaging (FedAvg).
        Clients with higher trust scores contribute more to the global update.
        
        Args:
            local_weights: Dict of model weights from each client
            local_deltas: Dict of weight deltas from each client
            client_ids: List of client IDs that participated in this round
            model_dict: Current global model state dict
            current_lr: Current learning rate
            client_metrics: Dict of client metrics including trust scores
            
        Returns:
            Updated model state dict
        """
        if client_metrics is None:
            client_metrics = {cid: {"trust_score": 1.0} for cid in client_ids}
            
        trust_scores = {cid: client_metrics[cid].get("trust_score", 1.0) for cid in client_ids}
        
        # Apply trust filtering if enabled
        if self.enable_trust_filtering:
            trusted_clients = [cid for cid in client_ids if trust_scores[cid] >= self.trust_threshold]
            if not trusted_clients:  # If no clients meet threshold, use all clients
                trusted_clients = client_ids
        else:
            trusted_clients = client_ids

        # Calculate trust-weighted average
        trust_sum = sum(trust_scores[cid] for cid in trusted_clients)
        if trust_sum == 0:  # Avoid division by zero
            weights = {cid: 1.0/len(trusted_clients) for cid in trusted_clients}
        else:
            weights = {cid: trust_scores[cid]/trust_sum for cid in trusted_clients}

        # Create aggregated weights dictionary
        avg_weights = {}
        for param_key in local_weights:
            # Skip personalized parameters if personalization is enabled
            if self.enable_personalization and 'personalized_head' in param_key:
                continue
                
            # Apply weighted averaging only to parameters that exist in all client updates
            if all(param_key in local_weights[i] for i, _ in enumerate(trusted_clients)):
                trusted_weight_sum = sum(
                    local_weights[i][param_key] * weights[cid] 
                    for i, cid in enumerate(client_ids) 
                    if cid in trusted_clients
                )
                avg_weights[param_key] = trusted_weight_sum
        
        # Update global model with aggregated weights
        for key in avg_weights:
            if key in model_dict:
                model_dict[key] = avg_weights[key]
                
        # Track aggregation statistics
        self.trust_scores = trust_scores
        self.round_stats = {
            'num_clients': len(trusted_clients),
            'trust_scores': {k: v for k, v in trust_scores.items() if k in trusted_clients},
            'client_metrics': client_metrics
        }
                
        return model_dict

    def get_global_model(self):
        """Return the current global model"""
        return self.global_model
        
    def get_round_stats(self):
        """Return statistics from the last aggregation round"""
        return self.round_stats


@SERVER_REGISTRY.register()
class ServerM(Server):    
    def set_momentum(self, model):
        """Initialize momentum terms for the server."""
        self.global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        """Applies momentum-based lookahead for FedACG."""
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum:
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        """
        Implements ServerM aggregation with trust-weighted averaging and momentum
        """
        if client_metrics is None:
            client_metrics = {cid: {"trust_score": 1.0} for cid in client_ids}
            
        trust_scores = {cid: client_metrics[cid].get("trust_score", 1.0) for cid in client_ids}
        
        # Apply trust filtering if enabled
        if self.enable_trust_filtering:
            trusted_clients = [cid for cid in client_ids if trust_scores[cid] >= self.trust_threshold]
            if not trusted_clients:  # If no clients meet threshold, use all clients
                trusted_clients = client_ids
        else:
            trusted_clients = client_ids

        # Calculate trust-weighted average
        trust_sum = sum(trust_scores[cid] for cid in trusted_clients)
        if trust_sum == 0:  # Avoid division by zero
            trust_sum = len(trusted_clients)
            trust_scores = {cid: 1.0 for cid in trusted_clients}

        # Create aggregated weights dictionary
        avg_weights = {}
        for param_key in local_weights:
            # Skip personalized parameters if personalization is enabled
            if self.enable_personalization and 'personalized_head' in param_key:
                continue
                
            # Check if this key exists in all trusted clients' local weights
            if all(param_key in local_weights[i] for i, cid in enumerate(client_ids) if cid in trusted_clients):
                weighted_sum = sum(
                    local_weights[i][param_key] * trust_scores[cid] / trust_sum
                    for i, cid in enumerate(client_ids) 
                    if cid in trusted_clients
                )
                avg_weights[param_key] = weighted_sum

        # Apply momentum if configured
        if hasattr(self.args.server, 'momentum') and self.args.server.momentum > 0:
            if not hasattr(self.args.server, 'FedACG') or not self.args.server.FedACG:
                # Add momentum to the weights directly
                for param_key in avg_weights:
                    if param_key in self.global_momentum:
                        avg_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]

            # Update delta and momentum
            for param_key in local_deltas:
                if self.enable_personalization and 'personalized_head' in param_key:
                    continue
                    
                if all(param_key in local_deltas[i] for i, cid in enumerate(client_ids) if cid in trusted_clients):
                    self.global_delta[param_key] = sum(
                        local_deltas[i][param_key] * trust_scores[cid] / trust_sum
                        for i, cid in enumerate(client_ids) 
                        if cid in trusted_clients
                    )
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        
        # Update global model with aggregated weights
        for key in avg_weights:
            if key in model_dict:
                model_dict[key] = avg_weights[key]
        
        # Track aggregation statistics
        self.trust_scores = trust_scores
        self.round_stats = {
            'num_clients': len(trusted_clients),
            'trust_scores': {k: v for k, v in trust_scores.items() if k in trusted_clients},
            'client_metrics': client_metrics
        }
                
        return model_dict


@SERVER_REGISTRY.register()
class PersonalizedServer(ServerM):
    """Server with trust-based client filtering and personalization support"""
    
    def __init__(self, args):
        super(PersonalizedServer, self).__init__(args)
        self.enable_personalization = True
        
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        """
        Personalized aggregation that only aggregates global parameters
        """
        # Filter out personalized parameters before aggregation
        filtered_weights = {}
        filtered_deltas = {}
        
        for i, client_id in enumerate(client_ids):
            filtered_weights[i] = {
                k: v for k, v in local_weights[i].items() 
                if 'personalized_head' not in k
            }
            
            filtered_deltas[i] = {
                k: v for k, v in local_deltas[i].items() 
                if 'personalized_head' not in k
            }
        
        # Call parent class aggregation with filtered parameters
        return super().aggregate(filtered_weights, filtered_deltas, client_ids, model_dict, current_lr, client_metrics)
