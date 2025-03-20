import copy
import logging
import time
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils.logging_utils import AverageMeter
from utils.metrics import evaluate
from trainers.build import TRAINER_REGISTRY

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class BaseTrainer:
    def __init__(self, model=None, client_type=None, server=None, evaler_type=None, 
                datasets=None, device=None, args=None, config=None):
        self.model = model
        self.client_type = client_type
        self.server = server
        self.evaler_type = evaler_type
        self.datasets = datasets
        self.device = device
        self.args = args
        self.config = config
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Check for personalization config
        if hasattr(args, 'client') and hasattr(args.client, 'personalization'):
            personalization_config = args.client.personalization
            self.enable_personalization = getattr(personalization_config, "enable", False)
        else:
            self.enable_personalization = False
            
        # Check for cyclical LR config
        if hasattr(args, 'cyclical_lr'):
            cyclical_lr_config = args.cyclical_lr
            self.enable_cyclical_lr = getattr(cyclical_lr_config, "enable", False)
        else:
            self.enable_cyclical_lr = False
            
        # Check for trust filtering config
        if hasattr(args, 'client') and hasattr(args.client, 'trust_filtering'):
            trust_config = args.client.trust_filtering
            self.enable_trust_filtering = getattr(trust_config, "enable", False)
        else:
            self.enable_trust_filtering = False
            
        # Initialize best metrics
        self.best_global_acc = 0.0
        self.best_personalized_acc = 0.0
        self.best_global_model = None
        self.best_personalized_model = None
        
    def train(self):
        """Main training loop for federated learning"""
        logger.info("Starting federated training...")
        
        # Initialize server and clients
        self.server.setup(self.model)
        
        num_clients = self.args.trainer.num_clients
        clients_per_round = max(1, int(num_clients * self.args.trainer.participation_rate))
        
        # Check for num_rounds in config, with fallback to global_rounds
        if hasattr(self.args.trainer, 'num_rounds'):
            num_rounds = self.args.trainer.num_rounds
        elif hasattr(self.args.trainer, 'global_rounds'):
            num_rounds = self.args.trainer.global_rounds
        else:
            num_rounds = 100  # Default fallback
        
        # Set eval_every if not present
        if not hasattr(self.args.trainer, 'eval_every'):
            setattr(self.args.trainer, 'eval_every', 5)
        
        logger.info(f"Training with {num_clients} clients, {clients_per_round} per round for {num_rounds} rounds")
        
        # Create clients
        clients = {}
        for i in range(num_clients):
            try:
                # Ensure model is on CPU before deepcopy to avoid CUDA issues
                cpu_model = copy.deepcopy(self.model).cpu()
                clients[i] = self.client_type(self.args, i, cpu_model)
                # Move model back to device after client initialization
                clients[i].model = clients[i].model.to(self.device)
            except Exception as e:
                logger.error(f"Error creating client {i}: {str(e)}")
                raise e
        
        # Training loop
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Round {round_num}/{num_rounds}")
            
            # Select clients for this round
            selected_clients = np.random.choice(range(num_clients), clients_per_round, replace=False)
            logger.info(f"Selected clients: {selected_clients.tolist()}")
            
            # Train selected clients
            client_models = {}
            client_stats = {}
            client_trust_scores = {}
            
            for client_id in selected_clients:
                try:
                    # Send global model to client
                    if hasattr(self.server, 'global_model_state_dict'):
                        clients[client_id].model.load_state_dict(self.server.global_model_state_dict)
                    elif hasattr(self.server, 'get_global_model'):
                        global_model = self.server.get_global_model()
                        clients[client_id].model.load_state_dict(global_model.state_dict())
                    else:
                        # Fallback to the model stored in the server object
                        clients[client_id].model.load_state_dict(self.server.model.state_dict())
                    
                    # Perform local training
                    client_state_dict, stats = clients[client_id].local_train(round_num)
                    
                    # Store client results
                    if client_state_dict is not None:
                        client_models[client_id] = client_state_dict
                        client_stats[client_id] = stats
                    
                    # Get trust score if trust filtering is enabled
                    if self.enable_trust_filtering and stats and 'trust_score' in stats:
                        client_trust_scores[client_id] = stats['trust_score']
                    
                except Exception as e:
                    logger.error(f" Training Failed: {str(e)}")
                    raise e
            
            # Aggregate client models
            if client_models:
                client_ids = list(client_models.keys())
                local_deltas = {i: {} for i in range(len(client_ids))}
                local_weights = {i: client_models[cid] for i, cid in enumerate(client_ids)}
                model_dict = self.server.get_global_model().state_dict()
                current_lr = getattr(self.args.trainer, 'local_lr', 0.01)
                
                if self.enable_trust_filtering and client_trust_scores:
                    # Convert client_stats to format expected by server
                    client_metrics = {i: {"trust_score": client_trust_scores.get(cid, 0)} 
                                     for i, cid in enumerate(client_ids)}
                    
                    self.server.aggregate(local_weights, local_deltas, client_ids, model_dict, current_lr, client_metrics)
                else:
                    self.server.aggregate(local_weights, local_deltas, client_ids, model_dict, current_lr)
            
            # Evaluate global model periodically
            if round_num % self.args.trainer.eval_every == 0:
                global_acc, personalized_acc = self.evaluate(round_num, clients)
                
                # Save best models
                if global_acc > self.best_global_acc:
                    self.best_global_acc = global_acc
                    self.best_global_model = copy.deepcopy(self.server.get_global_model().state_dict())
                    torch.save(self.best_global_model, f"./checkpoints/model_{round_num}_best_global.pt")
                    logger.info(f"Model saved to ./checkpoints/model_{round_num}_best_global.pt")
                
                if personalized_acc > self.best_personalized_acc:
                    self.best_personalized_acc = personalized_acc
                    self.best_personalized_model = copy.deepcopy(self.server.get_global_model().state_dict())
                    torch.save(self.best_personalized_model, f"./checkpoints/model_{round_num}_best_personalized.pt")
                    logger.info(f"Model saved to ./checkpoints/model_{round_num}_best_personalized.pt")
        
        # Return best model
        if self.enable_personalization:
            return self.best_personalized_model
        else:
            return self.best_global_model
    
    def evaluate(self, round_num, clients):
        """Evaluate the global and personalized models"""
        # Create a test model with the global parameters
        test_model = copy.deepcopy(self.model).to(self.device)
        if hasattr(self.server, 'global_model_state_dict'):
            test_model.load_state_dict(self.server.global_model_state_dict)
        elif hasattr(self.server, 'get_global_model'):
            global_model = self.server.get_global_model()
            test_model.load_state_dict(global_model.state_dict())
        else:
            test_model.load_state_dict(self.server.model.state_dict())
        
        # Evaluate global model - use model as first argument explicitly
        global_acc = self.evaler_type.evaluate(model=test_model, test_dataset=self.datasets['test'])
        
        # Evaluate personalized model if enabled
        personalized_acc = 0.0
        if self.enable_personalization:
            # Sample a subset of clients for personalization evaluation
            eval_clients = np.random.choice(list(clients.keys()), 
                                          min(10, len(clients)), 
                                          replace=False)
            
            personalized_accs = []
            for client_id in eval_clients:
                # Create personalized model for this client
                personalized_model = copy.deepcopy(test_model)
                
                # Fine-tune on client's data
                clients[client_id].personalize(personalized_model)
                
                # Evaluate personalized model - use model as first argument explicitly
                client_acc = self.evaler_type.evaluate(model=personalized_model, test_dataset=self.datasets['test'])
                personalized_accs.append(client_acc)
            
            personalized_acc = np.mean(personalized_accs)
        
        # Log results
        logger.info(f"Round {round_num} - Global Acc: {global_acc:.2f}%, Personalized Acc: {personalized_acc:.2f}%")
        
        # Also evaluate on balanced test set if available
        if 'balanced_test' in self.datasets:
            balanced_global_acc = self.evaler_type.evaluate(model=test_model, test_dataset=self.datasets['balanced_test'])
            
            balanced_personalized_acc = 0.0
            if self.enable_personalization:
                balanced_personalized_accs = []
                for client_id in eval_clients:
                    personalized_model = copy.deepcopy(test_model)
                    clients[client_id].personalize(personalized_model)
                    client_acc = self.evaler_type.evaluate(model=personalized_model, test_dataset=self.datasets['balanced_test'])
                    balanced_personalized_accs.append(client_acc)
                
                balanced_personalized_acc = np.mean(balanced_personalized_accs)
            
            logger.info(f"Round {round_num} - Balanced Global Acc: {balanced_global_acc:.2f}%, Balanced Personalized Acc: {balanced_personalized_acc:.2f}%")
        
        logger.info(f"Round {round_num} evaluation: Global Acc = {global_acc:.2f}%, Personalized Acc = {personalized_acc:.2f}%")
        
        return global_acc, personalized_acc
