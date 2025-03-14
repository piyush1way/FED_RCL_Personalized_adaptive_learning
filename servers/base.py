# #!/usr/bin/env python
# # coding: utf-8
# import copy
# import time

# import matplotlib.pyplot as plt
# import torch.multiprocessing as mp
# from sklearn.manifold import TSNE

# from utils import *
# from utils.metrics import evaluate
# from models import build_encoder
# from typing import Callable, Dict, Tuple, Union, List

# import torch
# from servers.build import SERVER_REGISTRY


# @SERVER_REGISTRY.register()
# class Server:
#     def __init__(self, args):
#         self.args = args
#         return
    
#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         """
#         Federated Averaging (FedAvg) aggregation.
#         """
#         C = len(client_ids)
#         for param_key in local_weights:
#             local_weights[param_key] = sum(local_weights[param_key]) / C
#         return local_weights
    

# @SERVER_REGISTRY.register()
# class ServerM(Server):    
    
#     def set_momentum(self, model):
#         global_delta = copy.deepcopy(model.state_dict())
#         for key in global_delta.keys():
#             global_delta[key] = torch.zeros_like(global_delta[key])

#         global_momentum = copy.deepcopy(model.state_dict())
#         for key in global_momentum.keys():
#             global_momentum[key] = torch.zeros_like(global_momentum[key])

#         self.global_delta = global_delta
#         self.global_momentum = global_momentum


#     @torch.no_grad()
#     def FedACG_lookahead(self, model):
#         sending_model_dict = copy.deepcopy(model.state_dict())
#         for key in self.global_momentum.keys():
#             sending_model_dict[key] += self.args.server.server.momentum * self.global_momentum[key]  # Updated to handle nested server config

#         model.load_state_dict(sending_model_dict)
#         return copy.deepcopy(model)
    

#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         C = len(client_ids)
#         for param_key in local_weights:
#             local_weights[param_key] = sum(local_weights[param_key]) / C
#         if self.args.server.server.momentum > 0:  # Updated to handle nested server config
#             if not self.args.server.server.get("FedACG"):  # Updated to handle nested server config
#                 for param_key in local_weights:               
#                     local_weights[param_key] += self.args.server.server.momentum * self.global_momentum[param_key]  # Updated to handle nested server config
                    
#             for param_key in local_deltas:
#                 self.global_delta[param_key] = sum(local_deltas[param_key]) / C
#                 self.global_momentum[param_key] = self.args.server.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]  # Updated to handle nested server config
            

#         return local_weights


# @SERVER_REGISTRY.register()
# class ServerAdam(Server):    
    
#     def set_momentum(self, model):
#         global_delta = copy.deepcopy(model.state_dict())
#         for key in global_delta.keys():
#             global_delta[key] = torch.zeros_like(global_delta[key])

#         global_momentum = copy.deepcopy(model.state_dict())
#         for key in global_momentum.keys():
#             global_momentum[key] = torch.zeros_like(global_momentum[key])

#         global_v = copy.deepcopy(model.state_dict())
#         for key in global_v.keys():
#             global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.server.tau * self.args.server.server.tau)  # Updated to handle nested server config

#         self.global_delta = global_delta
#         self.global_momentum = global_momentum
#         self.global_v = global_v

    
#     def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
#         C = len(client_ids)
#         server_lr = self.args.trainer.global_lr
        
#         for param_key in local_deltas:
#             self.global_delta[param_key] = sum(local_deltas[param_key]) / C
#             self.global_momentum[param_key] = self.args.server.server.momentum * self.global_momentum[param_key] + (1 - self.args.server.server.momentum) * self.global_delta[param_key]  # Updated to handle nested server config
#             self.global_v[param_key] = self.args.server.server.beta * self.global_v[param_key] + (1 - self.args.server.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])  # Updated to handle nested server config

#         for param_key in model_dict.keys():
#             model_dict[param_key] += server_lr * self.global_momentum[param_key] / ((self.global_v[param_key] ** 0.5) + self.args.server.server.tau)  # Updated to handle nested server config
            
#         return model_dict

import copy
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn.manifold import TSNE
from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Dict, List
from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():
    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        """
        Aggregates client updates using Federated Averaging (FedAvg).
        Clients with lower trust scores contribute less to the global update.

        Args:
            local_weights: Dict of client weight updates.
            local_deltas: Dict of client model deltas.
            client_ids: List of selected client IDs.
            model_dict: Current global model state.
            current_lr: Learning rate for server updates.
            trust_scores: Trust scores for each client.

        Returns:
            Updated global model state.
        """
        avg_weights = {}
        trusted_clients = [cid for cid in client_ids if trust_scores[cid] > self.args.trainer.trust_threshold]
        if len(trusted_clients) == 0:
            trusted_clients = client_ids  # If all are below threshold, aggregate all
        
        for param_key in local_weights:
            stacked_weights = torch.stack([local_weights[param_key][i] for i, cid in enumerate(client_ids) if cid in trusted_clients], dim=0)
            avg_weights[param_key] = torch.mean(stacked_weights, dim=0)

        return avg_weights


@SERVER_REGISTRY.register()
class ServerM(Server):    
    def set_momentum(self, model):
        """Initialize momentum terms for the server."""
        global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_delta = global_delta
        self.global_momentum = global_momentum

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        """Applies momentum-based lookahead for FedACG."""
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]
        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        trusted_clients = [cid for cid in client_ids if trust_scores[cid] > self.args.trainer.trust_threshold]
        if len(trusted_clients) == 0:
            trusted_clients = client_ids
        
        avg_weights = {}
        for param_key in local_weights:
            stacked_weights = torch.stack([local_weights[param_key][i] for i, cid in enumerate(client_ids) if cid in trusted_clients], dim=0)
            avg_weights[param_key] = torch.mean(stacked_weights, dim=0)
        
        if self.args.server.momentum > 0:
            if not self.args.server.get('FedACG'): 
                for param_key in avg_weights:
                    avg_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
            for param_key in local_deltas:
                self.global_delta[param_key] = torch.mean(torch.stack(local_deltas[param_key], dim=0), dim=0)
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        
        return avg_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    def set_momentum(self, model):
        """Initialize momentum and adaptive learning terms for Adam-based aggregation."""
        global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        global_v = {key: torch.zeros_like(val) + (self.args.server.tau * self.args.server.tau) for key, val in model.state_dict().items()}
        
        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        trusted_clients = [cid for cid in client_ids if trust_scores[cid] > self.args.trainer.trust_threshold]
        if len(trusted_clients) == 0:
            trusted_clients = client_ids

        server_lr = self.args.trainer.global_lr
        avg_weights = {}

        for param_key in local_deltas:
            stacked_deltas = torch.stack([local_deltas[param_key][i] for i, cid in enumerate(client_ids) if cid in trusted_clients], dim=0)
            self.global_delta[param_key] = torch.mean(stacked_deltas, dim=0)
            self.global_momentum[param_key] = (self.args.server.momentum * self.global_momentum[param_key] +
                                               (1 - self.args.server.momentum) * self.global_delta[param_key])
            self.global_v[param_key] = (self.args.server.beta * self.global_v[param_key] +
                                        (1 - self.args.server.beta) * (self.global_delta[param_key] ** 2))

        for param_key in model_dict.keys():
            avg_weights[param_key] = model_dict[param_key] + server_lr * self.global_momentum[param_key] / (torch.sqrt(self.global_v[param_key]) + self.args.server.tau)
        
        return avg_weights

