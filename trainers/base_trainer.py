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
    """Base trainer class for federated learning experiments.
    
    This class provides the foundation for training models in a federated learning setup,
    with support for personalization, adaptive learning rates, and trust-based filtering.
    """
    
    def __init__(self, model=None, client_type=None, server=None, evaler_type=None, 
                datasets=None, device=None, args=None, config=None):
        """Initialize the trainer with configuration parameters."""
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
        
        # Personalization settings
        if hasattr(args, 'client') and hasattr(args.client, 'personalization'):
            personalization_config = args.client.personalization
            self.enable_personalization = getattr(personalization_config, "enable", False)
        else:
            self.enable_personalization = False
            
        # Adaptive learning rate settings
        if hasattr(args, 'client') and hasattr(args.client, 'adaptive_lr'):
            adaptive_lr_config = args.client.adaptive_lr
            self.enable_adaptive_lr = getattr(adaptive_lr_config, "enable", False)
        else:
            self.enable_adaptive_lr = False
            
        # Trust filtering settings
        if hasattr(args, 'client') and hasattr(args.client, 'trust_filtering'):
            trust_config = args.client.trust_filtering
            self.enable_trust_filtering = getattr(trust_config, "enable", False)
        else:
            self.enable_trust_filtering = False
        
        # Initialize model if provided
        if self.model is not None and self.device is not None:
            self.setup(self.model, self.device)
        
    def setup(self, model, device, optimizer=None):
        """Set up the trainer with a model and device."""
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        if optimizer is None and hasattr(self.args, 'optimizer') and hasattr(self.args.optimizer, 'lr'):
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.optimizer.lr,
                momentum=getattr(self.args.optimizer, 'momentum', 0.9),
                weight_decay=getattr(self.args.optimizer, 'weight_decay', 1e-4)
            )
        else:
            self.optimizer = optimizer
            
    def evaluate(self, epoch):
        """Evaluate the model on the test dataset"""
        self.model.eval()
        
        # Create test loader
        if isinstance(self.datasets['test'], DataLoader):
            test_loader = self.datasets['test']
        else:
            test_loader = DataLoader(
                self.datasets['test'],
                batch_size=self.args.eval.batch_size,
                shuffle=False,
                num_workers=getattr(self.args, 'num_workers', 0),
                pin_memory=True
            )
        
        global_correct = 0
        personalized_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # Global model accuracy
                    if "global_logit" in output:
                        global_pred = output["global_logit"].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    elif "logit" in output:
                        global_pred = output["logit"].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    else:
                        # Fallback if no specific logits found
                        global_pred = list(output.values())[0].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    
                    # Personalized model accuracy
                    if "personalized_logit" in output:
                        personalized_pred = output["personalized_logit"].argmax(dim=1)
                        personalized_correct += personalized_pred.eq(target).sum().item()
                    elif "logit" in output and self.enable_personalization:
                        # If personalization is enabled, the default logit should be personalized
                        personalized_pred = output["logit"].argmax(dim=1)
                        personalized_correct += personalized_pred.eq(target).sum().item()
                    else:
                        # Fallback if no personalized logits
                        personalized_pred = global_pred
                        personalized_correct += personalized_pred.eq(target).sum().item()
                else:
                    # Handle case where output is not a dictionary
                    pred = output.argmax(dim=1)
                    global_correct += pred.eq(target).sum().item()
                    personalized_correct += pred.eq(target).sum().item()
                    
                total += target.size(0)
        
        global_acc = 100. * global_correct / total if total > 0 else 0
        personalized_acc = 100. * personalized_correct / total if total > 0 else 0
        
        logger.info(f'Round {epoch} - Global Acc: {global_acc:.2f}%, Personalized Acc: {personalized_acc:.2f}%')
        
        return {
            'acc': global_acc,
            'acc_personalized': personalized_acc,
            'total': total
        }
    
    def train(self):
        """Main training method for the federated learning process"""
        logger.info("Starting federated training...")
        
        # Initialize clients and server
        if self.server is None or self.client_type is None:
            raise ValueError("Server and client_type must be provided")
            
        # Setup server with the model
        self.server.setup(self.model)
        
        # Get training parameters
        num_rounds = self.args.trainer.global_rounds
        num_clients = self.args.trainer.num_clients
        
        # Calculate number of participating clients per round
        if hasattr(self.args.trainer, 'participating_clients'):
            clients_per_round = self.args.trainer.participating_clients
        elif hasattr(self.args.trainer, 'participation_rate'):
            clients_per_round = max(1, int(num_clients * self.args.trainer.participation_rate))
        else:
            clients_per_round = max(1, int(num_clients * 0.1))  # Default 10% participation
            
        logger.info(f"Training with {num_clients} clients, {clients_per_round} per round for {num_rounds} rounds")
        
        # Create client instances
        clients = {}
        for i in range(num_clients):
            clients[i] = self.client_type(self.args, i, copy.deepcopy(self.model))
            
        # Training loop
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Round {round_num}/{num_rounds}")
            
            # Select clients for this round
            selected_clients = self.server.select_clients(list(clients.keys()), clients_per_round)
            logger.info(f"Selected clients: {selected_clients}")
            
            # Get global model state
            global_model = self.server.get_global_model()
            global_state = global_model.state_dict()
            
            # Train selected clients
            client_models = {}
            client_stats = {}
            
            for client_id in selected_clients:
                # Setup client with global model and dataset
                clients[client_id].setup(
                    state_dict=global_state,
                    device=self.device,
                    local_dataset=self.datasets['train'][client_id] if client_id in self.datasets['train'] else None,
                    global_epoch=round_num,
                    local_lr=self.args.trainer.local_lr,
                    trainer=self
                )
                
                # Train client
                client_state_dict, stats = clients[client_id].local_train(round_num)
                
                # Store client results if valid
                if client_state_dict is not None:
                    client_models[client_id] = client_state_dict
                    client_stats[client_id] = stats
            
            # Aggregate client models
            if client_models:
                updated_state = self.server.update_global_model(client_models, client_stats=client_stats)
                self.model.load_state_dict(updated_state)
            else:
                logger.warning("No valid client models returned for aggregation")
            
            # Evaluate global model periodically
            if round_num % self.args.eval.freq == 0 or round_num == num_rounds:
                eval_results = self.evaluate(round_num)
                logger.info(f"Round {round_num} evaluation: Global Acc = {eval_results['acc']:.2f}%, Personalized Acc = {eval_results['acc_personalized']:.2f}%")
        
        logger.info("Federated training completed")
        return self.model
