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
import torch
import numpy as np
from utils import *
from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():
    def __init__(self, args):
        self.args = args
        # Initialize trust filtering settings
        self.trust_threshold = getattr(args.server, "trust_threshold", 0.5)
        self.enable_trust_filtering = getattr(args.server, "enable_trust_filtering", True)

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        """
        Trust-based Federated Averaging (FedAvg).
        Clients with higher trust scores contribute more to the global update.
        """
        avg_weights = {}
        
        # Apply trust filtering if enabled
        if self.enable_trust_filtering:
            trusted_clients = [cid for cid in client_ids if trust_scores[cid] > self.trust_threshold]
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

        for param_key in local_weights:
            weighted_sum = sum(
                local_weights[param_key][i] * weights[cid] 
                for i, cid in enumerate(client_ids) 
                if cid in trusted_clients
            )
            avg_weights[param_key] = weighted_sum

        return avg_weights


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

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        trust_sum = sum(trust_scores[cid] for cid in client_ids)
        if trust_sum == 0:
            trust_scores = {cid: 1.0 for cid in client_ids}
            trust_sum = len(client_ids)

        avg_weights = {}
        for param_key in local_weights:
            weighted_sum = sum(local_weights[param_key][i] * trust_scores[cid] for i, cid in enumerate(client_ids))
            avg_weights[param_key] = weighted_sum / trust_sum

        if self.args.server.momentum > 0:
            if not self.args.server.get('FedACG'): 
                for param_key in avg_weights:
                    avg_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]

            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key][i] * trust_scores[cid] for i, cid in enumerate(client_ids)) / trust_sum
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        
        return avg_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    def set_momentum(self, model):
        """Initialize momentum and adaptive learning terms for Adam-based aggregation."""
        self.global_delta = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_momentum = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
        self.global_v = {key: torch.zeros_like(val) + (self.args.server.tau ** 2) for key, val in model.state_dict().items()}

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, trust_scores):
        trust_sum = sum(trust_scores[cid] for cid in client_ids)
        if trust_sum == 0:
            trust_scores = {cid: 1.0 for cid in client_ids}
            trust_sum = len(client_ids)

        server_lr = self.args.trainer.global_lr
        avg_weights = {}

        for param_key in local_deltas:
            weighted_deltas = sum(local_deltas[param_key][i] * trust_scores[cid] for i, cid in enumerate(client_ids)) / trust_sum
            self.global_delta[param_key] = weighted_deltas
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1 - self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1 - self.args.server.beta) * (self.global_delta[param_key] ** 2)

        for param_key in model_dict.keys():
            avg_weights[param_key] = model_dict[param_key] + server_lr * self.global_momentum[param_key] / (torch.sqrt(self.global_v[param_key]) + self.args.server.tau)
        
        return avg_weights


