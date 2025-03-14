#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List

import torch
from servers.build import SERVER_REGISTRY


@SERVER_REGISTRY.register()
class Server:
    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        """
        Federated Averaging (FedAvg) aggregation.
        """
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key]) / C
        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] += self.args.server.server.momentum * self.global_momentum[key]  # Updated to handle nested server config

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key]) / C
        if self.args.server.server.momentum > 0:  # Updated to handle nested server config
            if not self.args.server.server.get("FedACG"):  # Updated to handle nested server config
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.server.momentum * self.global_momentum[param_key]  # Updated to handle nested server config
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key]) / C
                self.global_momentum[param_key] = self.args.server.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]  # Updated to handle nested server config
            

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.server.tau * self.args.server.server.tau)  # Updated to handle nested server config

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key]) / C
            self.global_momentum[param_key] = self.args.server.server.momentum * self.global_momentum[param_key] + (1 - self.args.server.server.momentum) * self.global_delta[param_key]  # Updated to handle nested server config
            self.global_v[param_key] = self.args.server.server.beta * self.global_v[param_key] + (1 - self.args.server.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])  # Updated to handle nested server config

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr * self.global_momentum[param_key] / ((self.global_v[param_key] ** 0.5) + self.args.server.server.tau)  # Updated to handle nested server config
            
        return model_dict
