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
        """Main training loop for federated learning with FedRCL novelties"""
        logger.info("Starting FedRCL training with novel features...")
        
        # Initialize server and clients
        self.server.setup(self.model)
        
        num_clients = self.args.trainer.num_clients
        clients_per_round = max(1, int(num_clients * self.args.trainer.participation_rate))
        
        # Determine number of training rounds
        if hasattr(self.args.trainer, 'num_rounds'):
            num_rounds = self.args.trainer.num_rounds
        elif hasattr(self.args.trainer, 'global_rounds'):
            num_rounds = self.args.trainer.global_rounds
        else:
            num_rounds = 100  # Default fallback
        
        # Set evaluation frequency
        if not hasattr(self.args.trainer, 'eval_every'):
            setattr(self.args.trainer, 'eval_every', 5)
        
        logger.info(f"Training with {num_clients} clients, {clients_per_round} per round for {num_rounds} rounds")
        logger.info(f"Personalization enabled: {self.enable_personalization}")
        logger.info(f"Trust-based filtering enabled: {self.enable_trust_filtering}")
        logger.info(f"Trust-based adaptive LR enabled: {self.enable_cyclical_lr}")
        logger.info(f"Multi-level contrastive learning enabled: {self.multi_level_rcl}")
        
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
        
        # Client trust score tracking for adaptive sampling
        client_trust_history = {i: 1.0 for i in range(num_clients)}
        
        # Training loop
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Round {round_num}/{num_rounds}")
            
            # Select clients for this round - use trust-based selection if enabled
            if round_num > 10 and self.enable_trust_filtering:
                # Use trust scores to bias selection (higher trust = higher chance)
                selection_weights = np.array([max(0.1, client_trust_history.get(i, 1.0)) for i in range(num_clients)])
                selection_weights = selection_weights / selection_weights.sum()
                selected_clients = np.random.choice(
                    range(num_clients), 
                    clients_per_round, 
                    replace=False, 
                    p=selection_weights
                )
            else:
                # Standard random selection
                selected_clients = np.random.choice(range(num_clients), clients_per_round, replace=False)
                
            logger.info(f"Selected clients: {selected_clients.tolist()}")
            
            # Train selected clients
            client_models = {}
            client_stats = {}
            client_trust_scores = {}
            
            local_lr = self.args.trainer.local_lr
            
            for client_id in selected_clients:
                try:
                    # Compute trust-based adaptive learning rate if enabled
                    if self.enable_cyclical_lr and round_num > 1:
                        # Get trust score from history or default to high trust
                        trust_score = client_trust_history.get(client_id, 0.8)
                        
                        # Calculate cyclical learning rate with trust adjustment
                        cycle_step = round_num % (2 * self.step_size)
                        x = abs(cycle_step / self.step_size - 1)
                        
                        # Adjust max_lr based on trust score
                        adjusted_max_lr = self.max_lr * (0.5 + 0.5 * trust_score)
                        
                        # Calculate final LR
                        adjusted_lr = self.base_lr + (adjusted_max_lr - self.base_lr) * max(0, (1 - x))
                        
                        logger.info(f"Client {client_id} Trust-Adjusted LR: {adjusted_lr:.6f} (trust: {trust_score:.2f})")
                        local_lr = adjusted_lr
                    
                    # Get latest global model state dict
                    global_state_dict = None
                    if hasattr(self.server, 'global_model_state_dict'):
                        global_state_dict = self.server.global_model_state_dict
                    elif hasattr(self.server, 'get_global_model'):
                        global_model = self.server.get_global_model()
                        global_state_dict = global_model.state_dict()
                    else:
                        global_state_dict = self.server.model.state_dict()
                    
                    # Setup client for training
                    clients[client_id].setup(
                        state_dict=global_state_dict,
                        device=self.device,
                        local_dataset=self.datasets.get(f'train_{client_id}', None),
                        global_epoch=round_num,
                        local_lr=local_lr,
                        trainer=self
                    )
                    
                    # Perform local training
                    client_state_dict, stats = clients[client_id].local_train(round_num)
                    
                    # Store client results if training was successful
                    if client_state_dict is not None:
                        client_models[client_id] = client_state_dict
                        client_stats[client_id] = stats
                    
                    # Update trust score history if available
                    if stats and 'trust_score' in stats:
                        trust_score = stats['trust_score']
                        client_trust_history[client_id] = trust_score
                        client_trust_scores[client_id] = trust_score
                        
                        # Log metrics
                        if stats:
                            for key, value in stats.items():
                                if isinstance(value, (int, float)):
                                    self.round_metrics[f"client_{client_id}_{key}"].append(value)
                    
                except Exception as e:
                    logger.error(f"Client {client_id} Training Failed: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Aggregate client models with trust-based weighting
            if client_models:
                client_ids = list(client_models.keys())
                local_weights = {i: client_models[cid] for i, cid in enumerate(client_ids)}
                
                # Prepare for aggregation
                model_dict = None
                if hasattr(self.server, 'get_global_model'):
                    model_dict = self.server.get_global_model().state_dict()
                else:
                    model_dict = self.server.model.state_dict()
                
                # Prepare client metrics for trust-based aggregation
                if self.enable_trust_filtering and client_trust_scores:
                    # Filter clients with trust scores below threshold
                    trusted_ids = []
                    trusted_weights = {}
                    trusted_metrics = {}
                    
                    for i, cid in enumerate(client_ids):
                        if client_trust_scores.get(cid, 0) >= self.trust_threshold:
                            trusted_ids.append(i)
                            trusted_weights[len(trusted_ids)-1] = local_weights[i]
                            trusted_metrics[len(trusted_ids)-1] = {"trust_score": client_trust_scores.get(cid, 0)}
                    
                    # Ensure we have minimum number of clients
                    if len(trusted_ids) >= self.min_trust_clients:
                        logger.info(f"Using {len(trusted_ids)} trusted clients for aggregation")
                        
                        # Check if server supports trust-based aggregation
                        if hasattr(self.server, 'aggregate_with_trust_scores'):
                            self.server.aggregate_with_trust_scores(
                                trusted_weights, trusted_ids, model_dict, 
                                local_lr, trusted_metrics
                            )
                        else:
                            # Fallback to standard aggregation but with only trusted clients
                            local_deltas = {i: {} for i in range(len(trusted_ids))}
                            self.server.aggregate(
                                trusted_weights, local_deltas, trusted_ids, 
                                model_dict, local_lr
                            )
                    else:
                        # Not enough trusted clients, use all clients
                        logger.warning(f"Not enough trusted clients ({len(trusted_ids)}), using all {len(client_ids)} clients")
                        local_deltas = {i: {} for i in range(len(client_ids))}
                        self.server.aggregate(
                            local_weights, local_deltas, client_ids,
                            model_dict, local_lr
                        )
                else:
                    # Standard aggregation without trust filtering
                    local_deltas = {i: {} for i in range(len(client_ids))}
                    self.server.aggregate(
                        local_weights, local_deltas, client_ids,
                        model_dict, local_lr
                    )
                    
                # Track metrics for visualization
                self.round_metrics["round"].append(round_num)
                if client_trust_scores:
                    self.round_metrics["avg_trust_score"].append(np.mean(list(client_trust_scores.values())))
                    
                # Track loss metrics if available
                loss_types = ["ce_loss", "rcl_loss", "distillation_loss", "fedprox_loss", "ewc_loss"]
                for loss_type in loss_types:
                    losses = [stats.get(loss_type, 0) for stats in client_stats.values() if stats]
                    if losses:
                        self.round_metrics[f"avg_{loss_type}"].append(np.mean(losses))
            
            # Evaluate global and personalized models periodically
            if round_num % self.args.trainer.eval_every == 0:
                global_acc, personalized_acc = self.evaluate(round_num, clients)
                
                # Track accuracy metrics
                self.round_metrics["global_acc"].append(global_acc)
                if self.enable_personalization:
                    self.round_metrics["personalized_acc"].append(personalized_acc)
                
                # Save best models
                if global_acc > self.best_global_acc:
                    self.best_global_acc = global_acc
                    if hasattr(self.server, 'get_global_model'):
                        self.best_global_model = copy.deepcopy(self.server.get_global_model().state_dict())
                    else:
                        self.best_global_model = copy.deepcopy(self.server.model.state_dict())
                    torch.save(self.best_global_model, f"./checkpoints/model_{round_num}_best_global.pt")
                    logger.info(f"New best global model saved: {global_acc:.2f}%")
                
                if self.enable_personalization and personalized_acc > self.best_personalized_acc:
                    self.best_personalized_acc = personalized_acc
                    if hasattr(self.server, 'get_global_model'):
                        self.best_personalized_model = copy.deepcopy(self.server.get_global_model().state_dict())
                    else:
                        self.best_personalized_model = copy.deepcopy(self.server.model.state_dict())
                    torch.save(self.best_personalized_model, f"./checkpoints/model_{round_num}_best_personalized.pt")
                    logger.info(f"New best personalized model saved: {personalized_acc:.2f}%")
                
                # Generate visualizations periodically
                if round_num % (self.args.trainer.eval_every * 5) == 0:
                    self.generate_visualizations()
        
        # Generate final visualizations
        self.generate_visualizations()
        
        # Save final model
        final_model = None
        if hasattr(self.server, 'get_global_model'):
            final_model = copy.deepcopy(self.server.get_global_model().state_dict())
        else:
            final_model = copy.deepcopy(self.server.model.state_dict())
            
        torch.save(final_model, f"./checkpoints/model_final.pt")
        
        # Return best model based on configuration
        if self.enable_personalization:
            return self.best_personalized_model if self.best_personalized_model is not None else final_model
        else:
            return self.best_global_model if self.best_global_model is not None else final_model
    
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
        
        # Evaluate global model
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
                
                # Setup personalization mode if available
                if hasattr(personalized_model, 'enable_personalized_mode'):
                    personalized_model.enable_personalized_mode()
                elif hasattr(personalized_model, 'use_personalized_head'):
                    personalized_model.use_personalized_head = True
                
                # Get client's training data
                client_dataset = self.datasets.get(f'train_{client_id}', None)
                if client_dataset is None or len(client_dataset) == 0:
                    logger.warning(f"Client {client_id} has no training data for personalization")
                    continue
                
                # Create data loader for personalization
                train_loader = DataLoader(
                    client_dataset, 
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False
                )
                
                # Fine-tune personalized head on client's data
                personalized_model.train()
                optimizer = torch.optim.SGD(
                    [p for n, p in personalized_model.named_parameters() if 'personalized_head' in n],
                    lr=self.personalization_lr,
                    momentum=0.9
                )
                
                # Freeze backbone if specified
                if self.freeze_backbone:
                    for name, param in personalized_model.named_parameters():
                        if 'personalized_head' not in name:
                            param.requires_grad = False
                
                # Personalization training loop
                for epoch in range(self.personalization_epochs):
                    for images, labels in train_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = personalized_model(images)
                        
                        if isinstance(outputs, dict):
                            logits = outputs.get("personalized_logit", outputs.get("logit", None))
                        else:
                            logits = outputs
                            
                        if logits is not None:
                            loss = self.criterion(logits, labels)
                            loss.backward()
                            optimizer.step()
                
                # Evaluate personalized model
                personalized_model.eval()
                client_acc = self.evaler_type.evaluate(model=personalized_model, test_dataset=self.datasets['test'])
                personalized_accs.append(client_acc)
                
                # Free memory
                del personalized_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if personalized_accs:
                personalized_acc = np.mean(personalized_accs)
        
        # Log results
        logger.info(f"Round {round_num} - " +
                    f"Global Acc: {global_acc:.2f}%, " +
                    f"Personalized Acc: {personalized_acc:.2f}%")
        
        # Also evaluate on balanced test set if available
        if 'balanced_test' in self.datasets:
            balanced_global_acc = self.evaler_type.evaluate(model=test_model, test_dataset=self.datasets['balanced_test'])
            
            balanced_personalized_acc = 0.0
            if self.enable_personalization and 'eval_clients' in locals():
                balanced_personalized_accs = []
                for client_id in eval_clients:
                    personalized_model = copy.deepcopy(test_model)
                    
                    # Setup personalization mode
                    if hasattr(personalized_model, 'enable_personalized_mode'):
                        personalized_model.enable_personalized_mode()
                    elif hasattr(personalized_model, 'use_personalized_head'):
                        personalized_model.use_personalized_head = True
                    
                    # Personalize
                    client_dataset = self.datasets.get(f'train_{client_id}', None)
                    if client_dataset is not None and len(client_dataset) > 0:
                        # Create data loader
                        train_loader = DataLoader(
                            client_dataset, 
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=False
                        )
                        
                        # Fine-tune
                        personalized_model.train()
                        optimizer = torch.optim.SGD(
                            [p for n, p in personalized_model.named_parameters() if 'personalized_head' in n],
                            lr=self.personalization_lr,
                            momentum=0.9
                        )
                        
                        if self.freeze_backbone:
                            for name, param in personalized_model.named_parameters():
                                if 'personalized_head' not in name:
                                    param.requires_grad = False
                        
                        for epoch in range(self.personalization_epochs):
                            for images, labels in train_loader:
                                images = images.to(self.device)
                                labels = labels.to(self.device)
                                
                                optimizer.zero_grad()
                                outputs = personalized_model(images)
                                
                                if isinstance(outputs, dict):
                                    logits = outputs.get("personalized_logit", outputs.get("logit", None))
                                else:
                                    logits = outputs
                                    
                                if logits is not None:
                                    loss = self.criterion(logits, labels)
                                    loss.backward()
                                    optimizer.step()
                        
                        # Evaluate
                        personalized_model.eval()
                        client_acc = self.evaler_type.evaluate(
                            model=personalized_model, 
                            test_dataset=self.datasets['balanced_test']
                        )
                        balanced_personalized_accs.append(client_acc)
                        
                        # Free memory
                        del personalized_model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                if balanced_personalized_accs:
                    balanced_personalized_acc = np.mean(balanced_personalized_accs)
            
            logger.info(f"Round {round_num} - " +
                        f"Balanced Global Acc: {balanced_global_acc:.2f}%, " +
                        f"Balanced Personalized Acc: {balanced_personalized_acc:.2f}%")
            
            # Track balanced metrics
            self.round_metrics["balanced_global_acc"].append(balanced_global_acc)
            if self.enable_personalization:
                self.round_metrics["balanced_personalized_acc"].append(balanced_personalized_acc)
        
        # Free memory
        del test_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return global_acc, personalized_acc
    
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
