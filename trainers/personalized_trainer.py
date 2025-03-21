import logging
import torch
import copy
import numpy as np
from collections import defaultdict
from trainers.base_trainer import BaseTrainer
from trainers.build import TRAINER_REGISTRY
from utils.metrics import evaluate, track_trust_scores, evaluate_personalization_benefits
from utils.helper import save_dict_to_json, setup_adaptive_learning_rate
import os
import time

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class PersonalizedTrainer(BaseTrainer):
    """Trainer with personalization support for FedRCL"""
    
    def __init__(self, args, model, trainset, testset, clients, server, evaler):
        super().__init__(args, model, trainset, testset, clients, server, evaler)
        
        # Ensure server is properly initialized
        if isinstance(server, dict):
            from servers.build import build_server
            logger.warning("Server passed as dict, converting to proper server instance")
            self.server = build_server(args)
            if self.model is not None:
                self.server.setup(self.model)
        
        # Personalization settings
        personalization_config = getattr(args.trainer, "personalization", {})
        self.enable_personalization = personalization_config.get("enable", True)
        self.adaptive_lr = personalization_config.get("adaptive_lr", True)
        self.trust_filtering = personalization_config.get("trust_filtering", True)
        
        # Enable personalized mode in model if available
        if hasattr(self.model, 'enable_personalized_mode'):
            self.model.enable_personalized_mode()
            logger.info("Personalized mode enabled for the model")
        
        # Adaptive freezing settings
        adaptive_freezing_config = getattr(args.trainer, "adaptive_freezing", {})
        self.adaptive_freezing = adaptive_freezing_config.get("enable", False)
        self.freeze_ratio = adaptive_freezing_config.get("initial_freeze_ratio", 0.5)
        self.freeze_decay_rate = adaptive_freezing_config.get("decay_rate", 0.05)
        
        # Training tracking
        self.global_round = 0
        self.best_acc = 0.0
        self.best_personalized_acc = 0.0
        self.best_model_state = None
        self.best_personalized_state = None
        
        # Client performance tracking
        self.client_performance = {}
        self.personalization_metrics = {}
        
        logger.info(f"Initialized PersonalizedTrainer with personalization={self.enable_personalization}, "
                   f"adaptive_lr={self.adaptive_lr}, trust_filtering={self.trust_filtering}")
    
    def train(self):
        """Training process with personalization support"""
        # Initialize metrics tracking
        metrics_history = defaultdict(list)
        self.global_round = 0
        
        # Setup initial model
        if self.server is not None:
            self.server.setup(self.model)
        
        # Training loop
        for round_idx in range(self.args.trainer.global_rounds):
            self.global_round = round_idx
            logger.info(f"=== Round {round_idx+1}/{self.args.trainer.global_rounds} ===")
            
            # Update adaptive freezing if enabled
            if self.adaptive_freezing and hasattr(self.model, 'setup_adaptive_freezing'):
                current_freeze_ratio = max(0.0, self.freeze_ratio - (round_idx * self.freeze_decay_rate))
                self.model.setup_adaptive_freezing(freeze_ratio=current_freeze_ratio)
                logger.info(f"Adaptive freezing ratio: {current_freeze_ratio:.3f}")
            
            # Select clients for this round
            selected_clients = self.select_clients(round_idx)
            logger.info(f"Selected {len(selected_clients)} clients for training")
            
            # Get current global model
            global_model = self.server.get_global_model() if self.server is not None else self.model
            global_model_dict = global_model.state_dict()
            
            # Client training
            client_models = {}
            client_weights = {}
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                
                # Setup client for training
                if hasattr(client, 'setup'):
                    client.setup(
                        state_dict=global_model_dict,
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
                logger.info(f"Global model updated with {len(client_models)} client models")
            
            # Evaluate
            if (round_idx + 1) % self.args.trainer.eval_every == 0:
                metrics = self.evaluate(round_idx)
                
                # Track metrics
                for k, v in metrics.items():
                    metrics_history[k].append(v)
                
                # Update best accuracies
                current_acc = metrics.get('acc', 0.0)
                current_personalized_acc = metrics.get('acc_personalized', 0.0)
                
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.save_model(f"best_model_round_{round_idx+1}.pt")
                
                if current_personalized_acc > self.best_personalized_acc:
                    self.best_personalized_acc = current_personalized_acc
                    self.best_personalized_state = copy.deepcopy(self.model.state_dict())
                    self.save_model(f"best_personalized_model_round_{round_idx+1}.pt")
        
        return self.model, metrics_history
    
    def evaluate(self, round_idx):
        """Evaluate both global and personalized models"""
        logger.info(f"Evaluating models at round {round_idx+1}")
        
        # Basic evaluation
        metrics = super().evaluate(round_idx)
        
        # Track trust scores if available
        if hasattr(self.server, 'trust_scores') and len(self.server.trust_scores) > 0:
            trust_stats = track_trust_scores(self)
            metrics.update({"trust_stats": trust_stats})
            
        # Evaluate personalization benefits
        if self.enable_personalization:
            personalization_metrics = evaluate_personalization_benefits(
                self.args, self.model, self.testloader, self.device
            )
            metrics.update({"personalization": personalization_metrics})
            
            # Log key personalization metrics
            if 'bop' in personalization_metrics:
                logger.info(f"Benefit of Personalization: {personalization_metrics['bop']:.4f}")
                logger.info(f"Global Acc: {personalization_metrics['acc_global']:.4f}, "
                           f"Personalized Acc: {personalization_metrics['acc_personalized']:.4f}")
            
        return metrics
    
    def select_clients(self, round_idx):
        """Select clients with trust-based selection if available"""
        if self.trust_filtering and hasattr(self.server, 'select_clients') and callable(self.server.select_clients):
            # Use server's trust-based client selection if available
            num_clients = max(1, int(self.args.trainer.participation_rate * len(self.clients)))
            selected_clients = self.server.select_clients(list(self.clients.keys()), num_clients)
            logger.info(f"Trust-based client selection: {len(selected_clients)} clients")
            return selected_clients
        else:
            # Fall back to random selection
            return super().select_clients(round_idx)
    
    def save_model(self, filename):
        """Save model checkpoint with training metadata"""
        try:
            save_path = os.path.join(self.args.log_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare checkpoint
            checkpoint = {
                'model_state': self.model.state_dict(),
                'round': self.global_round,
                'best_acc': self.best_acc,
                'best_personalized_acc': self.best_personalized_acc,
                'args': self.args,
                'personalization_enabled': self.enable_personalization,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            # Add personalization metrics if available
            if self.personalization_metrics:
                checkpoint['personalization_metrics'] = self.personalization_metrics
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
            # Save additional metadata
            metadata = {
                'round': self.global_round,
                'accuracy': self.best_acc,
                'personalized_accuracy': self.best_personalized_acc,
                'timestamp': checkpoint['timestamp']
            }
            metadata_path = os.path.join(os.path.dirname(save_path), 'model_metadata.json')
            save_dict_to_json(metadata, metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
