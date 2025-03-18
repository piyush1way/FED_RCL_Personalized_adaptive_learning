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
class BaseServer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.global_model = None
        self.client_models = {}
        self.client_stats = {}
        
        # Trust filtering settings
        trust_filtering_config = getattr(args.server, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        self.min_trusted_clients = getattr(trust_filtering_config, "min_trusted_clients", 1)
        
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

    def aggregate(self, client_models, client_weights=None, client_stats=None):
        """Aggregate client models with trust-based filtering"""
        if not client_models:
            logger.warning("No client models to aggregate")
            return self.global_model.state_dict()
            
        # Apply trust-based filtering if enabled
        if self.enable_trust_filtering and client_stats:
            trusted_clients = {}
            trusted_weights = {}
            trusted_stats = {}
            
            # Extract trust scores and filter clients
            for client_id, stats in client_stats.items():
                if client_id in client_models:
                    trust_score = stats.get('trust_score', 1.0)
                    self.trust_scores[client_id] = trust_score
                    
                    if trust_score >= self.trust_threshold:
                        trusted_clients[client_id] = client_models[client_id]
                        if client_weights:
                            trusted_weights[client_id] = client_weights[client_id]
                        trusted_stats[client_id] = stats
                        logger.info(f"Client {client_id} trusted with score {trust_score:.4f}")
                    else:
                        logger.info(f"Client {client_id} filtered out with score {trust_score:.4f}")
            
            # Ensure minimum number of trusted clients
            if len(trusted_clients) < self.min_trusted_clients:
                logger.warning(f"Only {len(trusted_clients)} trusted clients (min: {self.min_trusted_clients}). Using all clients.")
                trusted_clients = client_models
                trusted_weights = client_weights
                trusted_stats = client_stats
            
            # Update references for aggregation
            client_models = trusted_clients
            if client_weights:
                client_weights = trusted_weights
            client_stats = trusted_stats
        
        # If no clients remain after filtering, return current global model
        if not client_models:
            logger.warning("No trusted clients after filtering. Using current global model.")
            return self.global_model.state_dict()
            
        # Normalize weights if provided
        if client_weights:
            weight_sum = sum(client_weights.values())
            if weight_sum > 0:
                client_weights = {k: v / weight_sum for k, v in client_weights.items()}
        
        # Perform weighted averaging of model parameters
        avg_state_dict = {}
        
        # Get reference model structure from first client
        reference_state_dict = next(iter(client_models.values()))
        
        # Initialize with zeros
        for key in reference_state_dict.keys():
            if self.enable_personalization and 'personalized_head' in key:
                # Skip personalized parameters
                continue
                
            avg_state_dict[key] = torch.zeros_like(reference_state_dict[key])
        
        # Weighted average
        for client_id, state_dict in client_models.items():
            weight = client_weights.get(client_id, 1.0 / len(client_models)) if client_weights else 1.0 / len(client_models)
            
            for key in avg_state_dict.keys():
                if key in state_dict:
                    avg_state_dict[key] += weight * state_dict[key]
        
        # Track aggregation statistics
        self.round_stats = {
            'num_clients': len(client_models),
            'trust_scores': {k: v for k, v in self.trust_scores.items() if k in client_models},
            'client_stats': client_stats
        }
        
        return avg_state_dict

    def update_global_model(self, client_models, client_weights=None, client_stats=None):
        """Update global model with aggregated parameters"""
        if not client_models:
            logger.warning("No client models received. Global model unchanged.")
            return self.global_model.state_dict()
            
        # Aggregate client models
        aggregated_params = self.aggregate(client_models, client_weights, client_stats)
        
        # Update global model
        global_state_dict = self.global_model.state_dict()
        
        # Only update keys that exist in the aggregated params
        for key in global_state_dict.keys():
            if key in aggregated_params:
                global_state_dict[key] = aggregated_params[key]
        
        self.global_model.load_state_dict(global_state_dict)
        
        return global_state_dict
        
    def get_global_model(self):
        """Return the current global model"""
        return self.global_model
        
    def get_round_stats(self):
        """Return statistics from the last aggregation round"""
        return self.round_stats
