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
        trust_config = getattr(args.server, "trust_filtering", {})
        self.trust_threshold = getattr(trust_config, "threshold", 0.5)
        self.enable_trust_filtering = getattr(trust_config, "enable", False)
        self.min_trusted_clients = getattr(trust_config, "min_trusted_clients", 1)
        
        personalization_config = getattr(args.server, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        
        self.trust_scores = {}
        self.round_stats = {}
        self.client_history = {}
        
        self.model = None
        self.global_model = None

    def setup(self, model):
        self.model = model
        self.global_model = copy.deepcopy(model)

    def select_clients(self, clients, num_clients):
        if num_clients >= len(clients):
            return clients
        
        return np.random.choice(clients, num_clients, replace=False).tolist()

    def aggregate(self, client_models, client_weights=None, client_stats=None):
        if not client_models:
            logger.warning("No client models to aggregate")
            return self.global_model.state_dict()
            
        if self.enable_trust_filtering and client_stats:
            trusted_clients = {}
            trusted_weights = {}
            trusted_stats = {}
            
            for client_id, stats in client_stats.items():
                if client_id in client_models:
                    trust_score = stats.get('trust_score', 1.0)
                    self.trust_scores[client_id] = trust_score
                    
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append(trust_score)
                    
                    if trust_score >= self.trust_threshold:
                        trusted_clients[client_id] = client_models[client_id]
                        if client_weights:
                            trusted_weights[client_id] = client_weights[client_id]
                        trusted_stats[client_id] = stats
                        logger.info(f"Client {client_id} trusted with score {trust_score:.4f}")
                    else:
                        logger.info(f"Client {client_id} filtered out with score {trust_score:.4f}")
            
            if len(trusted_clients) < self.min_trusted_clients:
                logger.warning(f"Only {len(trusted_clients)} trusted clients (min: {self.min_trusted_clients}). Using all clients.")
                trusted_clients = client_models
                trusted_weights = client_weights
                trusted_stats = client_stats
            
            client_models = trusted_clients
            if client_weights:
                client_weights = trusted_weights
            client_stats = trusted_stats
        
        if not client_models:
            logger.warning("No trusted clients after filtering. Using current global model.")
            return self.global_model.state_dict()
            
        if client_weights:
            weight_sum = sum(client_weights.values())
            if weight_sum > 0:
                client_weights = {k: v / weight_sum for k, v in client_weights.items()}
        
        avg_state_dict = {}
        
        reference_state_dict = next(iter(client_models.values()))
        
        for key in reference_state_dict.keys():
            if self.enable_personalization and 'personalized_head' in key:
                continue
                
            avg_state_dict[key] = torch.zeros_like(reference_state_dict[key])
        
        for client_id, state_dict in client_models.items():
            weight = client_weights.get(client_id, 1.0 / len(client_models)) if client_weights else 1.0 / len(client_models)
            
            for key in avg_state_dict.keys():
                if key in state_dict:
                    avg_state_dict[key] += weight * state_dict[key]
        
        self.round_stats = {
            'num_clients': len(client_models),
            'trust_scores': {k: v for k, v in self.trust_scores.items() if k in client_models},
            'client_stats': client_stats,
            'client_history': self.client_history
        }
        
        return avg_state_dict

    def update_global_model(self, client_models, client_weights=None, client_stats=None):
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
        
        current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        aggregated_params = self.aggregate(client_models, client_weights, client_stats)
        
        global_state_dict = self.global_model.state_dict()
        
        for key in global_state_dict.keys():
            if key in aggregated_params:
                global_state_dict[key] = aggregated_params[key]
        
        self.global_model.load_state_dict(global_state_dict)
        
        return global_state_dict
        
    def get_global_model(self):
        return self.global_model
        
    def get_round_stats(self):
        return self.round_stats


@SERVER_REGISTRY.register()
class ServerM(Server):    
    def set_momentum(self, model):
        self.global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum:
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        if client_metrics is None:
            client_metrics = {cid: {"trust_score": 1.0} for cid in client_ids}
            
        trust_scores = {cid: client_metrics[cid].get("trust_score", 1.0) for cid in client_ids}
        
        if self.enable_trust_filtering:
            trusted_clients = [cid for cid in client_ids if trust_scores[cid] >= self.trust_threshold]
            if not trusted_clients:
                trusted_clients = client_ids
        else:
            trusted_clients = client_ids

        trust_sum = sum(trust_scores[cid] for cid in trusted_clients)
        if trust_sum == 0:
            trust_sum = len(trusted_clients)
            trust_scores = {cid: 1.0 for cid in trusted_clients}

        avg_weights = {}
        for param_key in local_weights:
            if isinstance(param_key, str) and self.enable_personalization and 'personalized_head' in param_key:
                continue
                
            if all(param_key in local_weights[i] for i, cid in enumerate(client_ids) if cid in trusted_clients):
                weighted_sum = sum(
                    local_weights[i][param_key] * trust_scores[cid] / trust_sum
                    for i, cid in enumerate(client_ids) 
                    if cid in trusted_clients
                )
                avg_weights[param_key] = weighted_sum

        if hasattr(self.args.server, 'momentum') and self.args.server.momentum > 0:
            if not hasattr(self.args.server, 'FedACG') or not self.args.server.FedACG:
                for param_key in avg_weights:
                    if param_key in self.global_momentum:
                        avg_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]

            for param_key in local_deltas:
                if isinstance(param_key, str) and self.enable_personalization and 'personalized_head' in param_key:
                    continue
                    
                if all(param_key in local_deltas[i] for i, cid in enumerate(client_ids) if cid in trusted_clients):
                    self.global_delta[param_key] = sum(
                        local_deltas[i][param_key] * trust_scores[cid] / trust_sum
                        for i, cid in enumerate(client_ids) 
                        if cid in trusted_clients
                    )
                    self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        
        for key in avg_weights:
            if key in model_dict:
                model_dict[key] = avg_weights[key]
        
        self.trust_scores = trust_scores
        self.round_stats = {
            'num_clients': len(trusted_clients),
            'trust_scores': {k: v for k, v in trust_scores.items() if k in trusted_clients},
            'client_metrics': client_metrics
        }
                
        return model_dict


@SERVER_REGISTRY.register()
class PersonalizedServer(ServerM):
    def __init__(self, args):
        super(PersonalizedServer, self).__init__(args)
        self.enable_personalization = True
        
    def update_global_model(self, client_models, client_weights=None, client_stats=None):
        if not client_models:
            logger.warning("No client models received. Global model unchanged.")
            return self.global_model.state_dict()
            
        local_weights = {}
        local_deltas = {}
        client_ids = list(client_models.keys())
        
        for i, client_id in enumerate(client_ids):
            local_weights[i] = {}
            for key, value in client_models[client_id].items():
                if 'personalized_head' not in key:
                    local_weights[i][key] = value
            
            local_deltas[i] = {}
            
        model_dict = self.global_model.state_dict()
        
        current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        aggregated_params = self.aggregate(
            local_weights, 
            local_deltas, 
            client_ids, 
            model_dict, 
            current_lr, 
            client_stats
        )
        
        self.global_model.load_state_dict(aggregated_params)
        
        return aggregated_params
        
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics=None):
        filtered_weights = {}
        filtered_deltas = {}
        
        for i, client_id in enumerate(client_ids):
            if i in local_weights:
                filtered_weights[i] = {}
                for key, value in local_weights[i].items():
                    if isinstance(key, str) and 'personalized_head' not in key:
                        filtered_weights[i][key] = value
                
                if i in local_deltas:
                    filtered_deltas[i] = {}
                    for key, value in local_deltas[i].items():
                        if isinstance(key, str) and 'personalized_head' not in key:
                            filtered_deltas[i][key] = value
        
        return super().aggregate(filtered_weights, filtered_deltas, client_ids, model_dict, current_lr, client_metrics)
