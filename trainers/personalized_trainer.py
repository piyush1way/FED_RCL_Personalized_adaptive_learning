import logging
import torch
import copy
import numpy as np
from collections import defaultdict
from trainers.base_trainer import BaseTrainer
from trainers.build import TRAINER_REGISTRY
from utils.metrics import evaluate, track_trust_scores, evaluate_personalization_benefits
from utils.helper import save_dict_to_json, setup_adaptive_learning_rate

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class PersonalizedTrainer(BaseTrainer):
    """Trainer with personalization support for FedRCL"""
    
    def __init__(self, args, model, trainset, testset, clients, server, evaler):
        super().__init__(args, model, trainset, testset, clients, server, evaler)
        
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
        
        # Client performance tracking
        self.client_performance = {}
        self.personalization_metrics = {}
        
        logger.info(f"Initialized PersonalizedTrainer with personalization={self.enable_personalization}, "
                   f"adaptive_lr={self.adaptive_lr}, trust_filtering={self.trust_filtering}")
    
    def train(self):
        """Training process with personalization support"""
        # Initialize metrics tracking
        metrics_history = defaultdict(list)
        best_acc = 0.0
        
        # Setup initial model
        self.server.setup(self.model)
        
        # Training loop
        for round_idx in range(self.args.trainer.global_rounds):
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
            global_model = self.server.get_global_model()
            global_model_dict = global_model.state_dict()
            
            # Client training
            client_models = {}
            client_weights = {}
            client_stats = {}
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                
                # Determine learning rate (adaptive if enabled)
                local_lr = self.args.trainer.local_lr
                if self.adaptive_lr and hasattr(client, 'trust_score'):
                    trust_score = getattr(client, 'trust_score', 1.0)
                    local_lr = setup_adaptive_learning_rate(
                        base_lr=0.001,
                        max_lr=0.1,
                        trust_score=trust_score,
                        step=round_idx,
                        step_size=10
                    )
                    logger.debug(f"Client {client_idx} adaptive LR: {local_lr:.6f} (trust: {trust_score:.2f})")
                
                # Setup client for training
                client.setup(
                    state_dict=global_model_dict,
                    device=self.device,
                    local_dataset=self.trainset[client_idx],
                    global_epoch=round_idx,
                    local_lr=local_lr,
                    trainer=self
                )
                
                # Train client
                client_model_dict, stats = client.local_train(round_idx)
                
                # Store results if client returned a model
                if client_model_dict is not None:
                    client_models[client_idx] = client_model_dict
                    client_weights[client_idx] = len(self.trainset[client_idx])
                    client_stats[client_idx] = stats
                    
                    # Track client performance
                    if stats is not None:
                        if client_idx not in self.client_performance:
                            self.client_performance[client_idx] = []
                        self.client_performance[client_idx].append({
                            'round': round_idx,
                            'loss': stats.get('loss', 0.0),
                            'trust_score': stats.get('trust_score', 1.0)
                        })
            
            # Update global model
            if client_models:
                updated_model_dict = self.server.update_global_model(client_models, client_weights, client_stats)
                self.model.load_state_dict(updated_model_dict, strict=False)
                logger.info(f"Global model updated with {len(client_models)} client models")
            else:
                logger.warning("No client models returned. Global model unchanged.")
            
            # Evaluate periodically
            if (round_idx + 1) % self.args.trainer.eval_every == 0 or round_idx == 0:
                metrics = self.evaluate(round_idx)
                
                # Track metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metrics_history[k].append(v)
                
                # Save best model
                current_acc = metrics.get('acc_personalized', metrics.get('acc', 0.0))
                if current_acc > best_acc:
                    best_acc = current_acc
                    self.save_model(f"best_model_round_{round_idx+1}.pt")
                    logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")
                
                # Save metrics
                save_dict_to_json(metrics, f"{self.args.log_dir}/metrics_round_{round_idx+1}.json")
                
                # Save overall metrics history
                save_dict_to_json(dict(metrics_history), f"{self.args.log_dir}/metrics_history.json")
                
                # Save client performance
                save_dict_to_json(self.client_performance, f"{self.args.log_dir}/client_performance.json")
                
                # Save personalization metrics
                if 'personalization' in metrics:
                    self.personalization_metrics[round_idx+1] = metrics['personalization']
                    save_dict_to_json(self.personalization_metrics, f"{self.args.log_dir}/personalization_metrics.json")
            
            # Save checkpoint periodically
            if (round_idx + 1) % self.args.save_freq == 0:
                self.save_model(f"model_round_{round_idx+1}.pt")
        
        # Save final model
        self.save_model("final_model.pt")
        logger.info("Training completed!")
        
        return metrics_history
    
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
        """Save model to file"""
        save_path = f"{self.args.checkpoint_path}/{filename}"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'round': self.global_round,
            'args': self.args
        }, save_path)
        logger.info(f"Model saved to {save_path}")
