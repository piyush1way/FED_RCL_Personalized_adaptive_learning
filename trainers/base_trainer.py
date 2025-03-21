import copy
import logging
import time
import gc
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.logging_utils import AverageMeter
from utils.metrics import evaluate
from trainers.build import TRAINER_REGISTRY

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class BaseTrainer:
    def __init__(self, args, model=None, trainset=None, testset=None, clients=None, server=None, evaler=None):
        self.args = args
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.clients = clients
        self.server = server
        self.evaler = evaler
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test loader
        self.testloader = DataLoader(
            self.testset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ) if self.testset is not None else None
        
        # Basic setup
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Personalization configuration
        if hasattr(args, 'client') and hasattr(args.client, 'personalization'):
            personalization_config = args.client.personalization
            self.enable_personalization = getattr(personalization_config, "enable", False)
            self.personalization_layers = getattr(personalization_config, "layers", 2)
            self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
            self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)
            self.freeze_ratio = getattr(personalization_config, "freeze_ratio", 0.5)
            self.personalization_epochs = getattr(personalization_config, "epochs", 5)
            self.personalization_lr = getattr(personalization_config, "lr", 0.01)
        else:
            self.enable_personalization = False
            self.personalization_layers = 2
            self.freeze_backbone = False
            self.adaptive_layer_freezing = False
            self.freeze_ratio = 0.5
            self.personalization_epochs = 5
            self.personalization_lr = 0.01
            
        # Trust-based adaptive learning rate configuration
        if hasattr(args, 'client') and hasattr(args.client, 'cyclical_lr'):
            cyclical_lr_config = args.client.cyclical_lr
            self.enable_cyclical_lr = getattr(cyclical_lr_config, "enable", True)
            self.base_lr = getattr(cyclical_lr_config, "base_lr", 0.001)
            self.max_lr = getattr(cyclical_lr_config, "max_lr", 0.1)
            self.step_size = getattr(cyclical_lr_config, "step_size", 10)
        else:
            self.enable_cyclical_lr = True
            self.base_lr = 0.001
            self.max_lr = 0.1
            self.step_size = 10
            
        # Trust-based client filtering configuration
        if hasattr(args, 'client') and hasattr(args.client, 'trust_filtering'):
            trust_config = args.client.trust_filtering
            self.enable_trust_filtering = getattr(trust_config, "enable", False)
            self.trust_threshold = getattr(trust_config, "threshold", 0.5)
            self.min_trust_clients = getattr(trust_config, "min_clients", 1)
        else:
            self.enable_trust_filtering = False
            self.trust_threshold = 0.5
            self.min_trust_clients = 1
        
        # Multi-level contrastive learning configuration
        if hasattr(args, 'client') and hasattr(args.client, 'multi_level_rcl'):
            self.multi_level_rcl = getattr(args.client, "multi_level_rcl", True)
            self.layer_weights = getattr(args.client, "layer_weights", [0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            self.multi_level_rcl = True
            self.layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Knowledge distillation configuration
        if hasattr(args, 'client') and hasattr(args.client, 'distillation'):
            self.enable_distillation = getattr(args.client, "distillation", True)
            self.distillation_temp = getattr(args.client, "distillation_temp", 3.0)
            self.distillation_weight = getattr(args.client, "distillation_weight", 0.7)
        else:
            self.enable_distillation = True
            self.distillation_temp = 3.0
            self.distillation_weight = 0.7
        
        # Initialize training metrics trackers
        self.best_global_acc = 0.0
        self.best_personalized_acc = 0.0
        self.best_global_model = None
        self.best_personalized_model = None
        
        # Create directories for checkpoints and visualizations
        os.makedirs("./checkpoints", exist_ok=True)
        os.makedirs("./visualizations", exist_ok=True)
        
        # Initialize metrics tracking
        self.round_metrics = defaultdict(list)
        
    def train(self):
        """Generic training loop"""
        # Setup trackers
        metrics_history = defaultdict(list)
        best_acc = 0.0
        
        # Setup server
        if self.server is not None:
            self.server.setup(self.model)
        
        # Training loop
        for round_idx in range(self.args.trainer.global_rounds):
            logger.info(f"=== Round {round_idx+1}/{self.args.trainer.global_rounds} ===")
            
            # Select clients for this round
            selected_clients = self.select_clients(round_idx)
            logger.info(f"Selected {len(selected_clients)} clients for training")
            
            # Client updates
            client_models = {}
            client_weights = {}
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                
                # Setup client
                if hasattr(client, 'setup'):
                    current_global_model = self.server.get_global_model() if self.server is not None else self.model
                    client.setup(
                        state_dict=current_global_model.state_dict(),
                        device=self.device,
                        local_dataset=self.trainset[client_idx],
                        global_epoch=round_idx,
                        local_lr=self.args.trainer.local_lr  # Use base learning rate
                    )
                
                # Train client
                client_model_dict, stats = client.local_train(round_idx)
                
                # Store results
                if client_model_dict is not None:
                    client_models[client_idx] = client_model_dict
                    client_weights[client_idx] = len(self.trainset[client_idx])
            
            # Server update
            if self.server is not None and client_models:
                updated_model_dict = self.server.update_global_model(client_models, client_weights)
                self.model.load_state_dict(updated_model_dict)
            
            # Evaluate
            if (round_idx + 1) % self.args.trainer.eval_every == 0:
                metrics = self.evaluate(round_idx)
                
                # Track metrics
                for k, v in metrics.items():
                    metrics_history[k].append(v)
                
                # Save best model
                if metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    self.save_model(f"best_model_r{round_idx}.pt")
        
        return self.model, metrics_history
    
    def evaluate(self, round_idx):
        """Evaluate the model on test data"""
        metrics = {}
        
        # Prepare model for evaluation
        test_model = copy.deepcopy(self.model)
        test_model.eval()
        test_model.to(self.device)
        
        # Global model evaluation
        with torch.no_grad():
            if self.testloader is not None:
                correct = 0
                total = 0
                
                for data in self.testloader:
                    if isinstance(data, (tuple, list)) and len(data) >= 2:
                        images, labels = data[0], data[1]
                    else:
                        images = data
                        labels = torch.zeros(images.size(0))  # Default labels if not provided
                        
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = test_model(images)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        logits = outputs.get("logit", outputs.get("global_logit", None))
                        if logits is None:
                            # If no logits found in dict, use the first tensor
                            logits = next(tensor for tensor in outputs.values() if isinstance(tensor, torch.Tensor))
                    else:
                        logits = outputs
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                global_acc = 100.0 * correct / total
                metrics['acc'] = global_acc
                logger.info(f"Round {round_idx} | Global Model Accuracy: {global_acc:.2f}%")
            
        # Personalized evaluation if supported
        if hasattr(self.args.trainer, 'personalization') and getattr(self.args.trainer.personalization, 'enable', False):
            client_accs = []
            num_eval_clients = min(10, len(self.clients))  # Limit number of clients to evaluate
            
            for client_idx in list(self.clients.keys())[:num_eval_clients]:
                # Create personalized model
                personalized_model = copy.deepcopy(test_model)
                if hasattr(personalized_model, 'enable_personalized_mode'):
                    personalized_model.enable_personalized_mode()
                
                # Get client dataset
                client_dataset = self.trainset.get(client_idx, None)
                if client_dataset and len(client_dataset) > 0:
                    # Train on client data
                    try:
                        # Simple fine-tuning
                        client_loader = DataLoader(
                            client_dataset, 
                            batch_size=min(32, len(client_dataset)),  # Ensure batch size isn't larger than dataset
                            shuffle=True,
                            drop_last=True  # Drop last batch if incomplete to avoid BN issues
                        )
                        
                        optimizer = torch.optim.SGD(
                            personalized_model.parameters(),
                            lr=0.01,
                            momentum=0.9
                        )
                        
                        # Set model to eval mode for batch norm (use accumulated statistics)
                        personalized_model.eval()
                        for param in personalized_model.parameters():
                            param.requires_grad = True
                            
                        for _ in range(5):  # Few epochs of fine-tuning
                            for batch in client_loader:
                                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                                    inputs, targets = batch[0], batch[1]
                                else:
                                    inputs = batch
                                    targets = torch.zeros(inputs.size(0))  # Default labels if not provided
                                
                                inputs = inputs.to(self.device)
                                targets = targets.to(self.device).long()  # Ensure targets are long type
                                
                                optimizer.zero_grad()
                                outputs = personalized_model(inputs)
                                
                                # Handle different output formats
                                if isinstance(outputs, dict):
                                    logits = outputs.get("logit", outputs.get("personalized_logit", None))
                                    if logits is None:
                                        logits = next(tensor for tensor in outputs.values() if isinstance(tensor, torch.Tensor))
                                else:
                                    logits = outputs
                                
                                loss = self.criterion(logits, targets)
                                loss.backward()
                                optimizer.step()
                        
                        # Evaluate personalized model
                        personalized_model.eval()
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for batch in self.testloader:
                                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                                    inputs, targets = batch[0], batch[1]
                                else:
                                    inputs = batch
                                    targets = torch.zeros(inputs.size(0))  # Default labels if not provided
                                
                                inputs = inputs.to(self.device)
                                targets = targets.to(self.device).long()  # Ensure targets are long type
                                
                                outputs = personalized_model(inputs)
                                
                                # Handle different output formats
                                if isinstance(outputs, dict):
                                    logits = outputs.get("logit", outputs.get("personalized_logit", None))
                                    if logits is None:
                                        logits = next(tensor for tensor in outputs.values() if isinstance(tensor, torch.Tensor))
                                else:
                                    logits = outputs
                                
                                _, predicted = torch.max(logits.data, 1)
                                total += targets.size(0)
                                correct += (predicted == targets).sum().item()
                            
                            client_acc = 100.0 * correct / total
                            client_accs.append(client_acc)
                            
                    except Exception as e:
                        logger.error(f"Error personalizing for client {client_idx}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            if client_accs:
                personalized_acc = sum(client_accs) / len(client_accs)
                metrics['acc_personalized'] = personalized_acc
                logger.info(f"Round {round_idx} | Personalized Model Avg Accuracy: {personalized_acc:.2f}%")
        
        return metrics
    
    def select_clients(self, round_idx):
        """Select clients for the current round"""
        num_clients = len(self.clients)
        clients_per_round = max(1, int(num_clients * self.args.trainer.participation_rate))
        return np.random.choice(list(self.clients.keys()), clients_per_round, replace=False).tolist()
    
    def save_model(self, filename):
        """Save the current model"""
        save_path = os.path.join(self.args.log_dir, filename)
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    def generate_visualizations(self):
        """Generate visualizations of training metrics"""
        try:
            # Create figures directory if it doesn't exist
            os.makedirs("./visualizations", exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            
            # Plot accuracy curves
            if "global_acc" in self.round_metrics and len(self.round_metrics["global_acc"]) > 0:
                plt.plot(
                    range(len(self.round_metrics["global_acc"])), 
                    self.round_metrics["global_acc"], 
                    label="Global Model Accuracy"
                )
            
            if "personalized_acc" in self.round_metrics and len(self.round_metrics["personalized_acc"]) > 0:
                plt.plot(
                    range(len(self.round_metrics["personalized_acc"])), 
                    self.round_metrics["personalized_acc"], 
                    label="Personalized Model Accuracy"
                )
            
            plt.xlabel("Evaluation Round")
            plt.ylabel("Accuracy (%)")
            plt.title("Model Accuracy Over Training Rounds")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("./visualizations/accuracy_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot trust scores if available
            if "avg_trust_score" in self.round_metrics and len(self.round_metrics["avg_trust_score"]) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    range(len(self.round_metrics["avg_trust_score"])), 
                    self.round_metrics["avg_trust_score"], 
                    label="Average Trust Score", 
                    color='green'
                )
                plt.xlabel("Training Round")
                plt.ylabel("Trust Score")
                plt.title("Average Client Trust Score Over Training Rounds")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig("./visualizations/trust_scores.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot loss curves
            loss_types = ["ce_loss", "rcl_loss", "distillation_loss", "fedprox_loss", "ewc_loss"]
            loss_labels = ["Cross Entropy", "Relaxed Contrastive", "Distillation", "FedProx", "EWC"]
            
            plt.figure(figsize=(12, 6))
            
            for loss_type, label in zip(loss_types, loss_labels):
                key = f"avg_{loss_type}"
                if key in self.round_metrics and len(self.round_metrics[key]) > 0:
                    plt.plot(
                        range(len(self.round_metrics[key])), 
                        self.round_metrics[key], 
                        label=f"{label} Loss"
                    )
            
            plt.xlabel("Training Round")
            plt.ylabel("Loss Value")
            plt.title("Average Loss Values Over Training Rounds")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("./visualizations/loss_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Generated visualizations of training metrics")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
