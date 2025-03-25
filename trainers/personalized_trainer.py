# import logging
# import torch
# import copy
# import numpy as np
# from collections import defaultdict
# from trainers.base_trainer import BaseTrainer
# from trainers.build import TRAINER_REGISTRY
# from utils.metrics import evaluate, track_trust_scores, evaluate_personalization_benefits
# from utils.helper import save_dict_to_json, setup_adaptive_learning_rate
# import os
# import time

# logger = logging.getLogger(__name__)

# @TRAINER_REGISTRY.register()
# class PersonalizedTrainer(BaseTrainer):
#     """Trainer with personalization support for FedRCL"""
    
#     def __init__(self, args, model, trainset, testset, clients, server, evaler):
#         super().__init__(args, model, trainset, testset, clients, server, evaler)
        
#         # Ensure server is properly initialized
#         if isinstance(server, dict):
#             from servers.build import build_server
#             logger.warning("Server passed as dict, converting to proper server instance")
#             self.server = build_server(args)
#             if self.model is not None:
#                 self.server.setup(self.model)
        
#         # Personalization settings
#         personalization_config = getattr(args.trainer, "personalization", {})
#         self.enable_personalization = personalization_config.get("enable", True)
#         self.adaptive_lr = personalization_config.get("adaptive_lr", True)
#         self.trust_filtering = personalization_config.get("trust_filtering", True)
        
#         # Enable personalized mode in model if available
#         if hasattr(self.model, 'enable_personalized_mode'):
#             self.model.enable_personalized_mode()
#             logger.info("Personalized mode enabled for the model")
        
#         # Adaptive freezing settings
#         adaptive_freezing_config = getattr(args.trainer, "adaptive_freezing", {})
#         self.adaptive_freezing = adaptive_freezing_config.get("enable", False)
#         self.freeze_ratio = adaptive_freezing_config.get("initial_freeze_ratio", 0.5)
#         self.freeze_decay_rate = adaptive_freezing_config.get("decay_rate", 0.05)
        
#         # Training tracking
#         self.global_round = 0
#         self.best_acc = 0.0
#         self.best_personalized_acc = 0.0
#         self.best_model_state = None
#         self.best_personalized_state = None
        
#         # Client performance tracking
#         self.client_performance = {}
#         self.personalization_metrics = {}
        
#         self.trust_ema = 0.5  # Initialize exponential moving average for trust
#         self.trust_history = []  # Keep track of trust scores
#         self.acc_history = {'global': [], 'personalized': []}  # Track accuracy history
#         self.trust_momentum = 0.9  # Momentum factor for trust score updates
#         self.min_trust = 0.1  # Minimum trust score
#         self.max_trust = 1.0  # Maximum trust score
        
#         logger.info(f"Initialized PersonalizedTrainer with personalization={self.enable_personalization}, "
#                    f"adaptive_lr={self.adaptive_lr}, trust_filtering={self.trust_filtering}")
    
#     def calculate_trust_score(self, global_acc, personalized_acc, round_number):
#         """Calculate dynamic trust score based on performance metrics and training progress"""
#         # Get base trust from accuracy ratio
#         acc_ratio = personalized_acc / (global_acc + 1e-8)  # Prevent division by zero
#         base_trust = min(acc_ratio, 2.0) / 2.0  # Normalize to [0, 1]
        
#         # Add round-dependent factor to encourage exploration in early rounds
#         round_factor = min(round_number / 100, 1.0)  # Scales up to 1.0 over first 100 rounds
        
#         # Add small random noise to break plateaus
#         noise = torch.randn(1).item() * 0.02  # Using randn instead of normal
        
#         # Calculate new trust score with momentum
#         new_trust = self.trust_momentum * self.trust_ema + (1 - self.trust_momentum) * base_trust
#         new_trust = new_trust * round_factor + noise
        
#         # Ensure trust score stays within bounds
#         new_trust = max(self.min_trust, min(self.max_trust, new_trust))
        
#         # Update EMA
#         self.trust_ema = new_trust
#         self.trust_history.append(new_trust)
        
#         return new_trust
    
#     def train(self):
#         """Training process with personalization support"""
#         # Initialize metrics tracking
#         metrics_history = defaultdict(list)
#         self.global_round = 0
        
#         # Setup initial model
#         if self.server is not None:
#             self.server.setup(self.model)
        
#         # Training loop
#         for round_idx in range(self.args.trainer.global_rounds):
#             self.global_round = round_idx
#             logger.info(f"=== Round {round_idx+1}/{self.args.trainer.global_rounds} ===")
            
#             # Update adaptive freezing if enabled
#             if self.adaptive_freezing and hasattr(self.model, 'setup_adaptive_freezing'):
#                 current_freeze_ratio = max(0.0, self.freeze_ratio - (round_idx * self.freeze_decay_rate))
#                 self.model.setup_adaptive_freezing(freeze_ratio=current_freeze_ratio)
#                 logger.info(f"Adaptive freezing ratio: {current_freeze_ratio:.3f}")
            
#             # Select clients for this round
#             selected_clients = self.select_clients(round_idx)
#             logger.info(f"Selected {len(selected_clients)} clients for training")
            
#             # Get current global model
#             global_model = self.server.get_global_model() if self.server is not None else self.model
#             global_model_dict = global_model.state_dict()
            
#             # Client training
#             client_models = {}
#             client_weights = {}
#             client_stats = {}
            
#             for client_idx in selected_clients:
#                 client = self.clients[client_idx]
                
#                 # Determine learning rate (adaptive if enabled)
#                 local_lr = self.args.trainer.local_lr
#                 if self.adaptive_lr and hasattr(client, 'trust_score'):
#                     trust_score = getattr(client, 'trust_score', 1.0)
#                     local_lr = setup_adaptive_learning_rate(
#                         base_lr=self.args.trainer.local_lr,
#                         max_lr=self.args.trainer.local_lr * 2,
#                         trust_score=trust_score,
#                         step=round_idx,
#                         step_size=10
#                     )
#                     logger.debug(f"Client {client_idx} adaptive LR: {local_lr:.6f} (trust: {trust_score:.2f})")
                
#                 # Setup client for training
#                 if hasattr(client, 'setup'):
#                     client.setup(
#                         state_dict=global_model_dict,
#                         device=self.device,
#                         local_dataset=self.trainset[client_idx],
#                         global_epoch=round_idx,
#                         local_lr=local_lr,
#                         trainer=self
#                     )
                
#                 # Train client
#                 client_model_dict, stats = client.local_train(round_idx)
                
#                 # Store results
#                 if client_model_dict is not None:
#                     client_models[client_idx] = client_model_dict
#                     client_weights[client_idx] = len(self.trainset[client_idx])
#                     client_stats[client_idx] = stats
                    
#                     # Track client performance
#                     if stats is not None:
#                         if client_idx not in self.client_performance:
#                             self.client_performance[client_idx] = []
#                         self.client_performance[client_idx].append({
#                             'round': round_idx,
#                             'loss': stats.get('global_loss', stats.get('loss', 0.0)),
#                             'trust_score': stats.get('trust_score', 1.0)
#                         })
            
#             # Server update
#             if self.server is not None and client_models:
#                 updated_model_dict = self.server.update_global_model(client_models, client_weights, client_stats)
#                 self.model.load_state_dict(updated_model_dict)
#                 logger.info(f"Global model updated with {len(client_models)} client models")
            
#             # Log trust scores if available
#             if hasattr(self.server, 'trust_scores') and self.server.trust_scores:
#                 avg_trust = sum(self.server.trust_scores.values()) / len(self.server.trust_scores)
#                 logger.info(f"Average trust score: {avg_trust:.4f}")
            
#             # Evaluate
#             if (round_idx + 1) % self.args.trainer.eval_every == 0:
#                 metrics = self.evaluate(round_idx)
                
#                 # Track metrics
#                 for k, v in metrics.items():
#                     metrics_history[k].append(v)
                
#                 # Update best accuracies
#                 current_acc = metrics.get('acc', 0.0)
#                 current_personalized_acc = metrics.get('acc_personalized', 0.0)
                
#                 if current_acc > self.best_acc:
#                     self.best_acc = current_acc
#                     self.best_model_state = copy.deepcopy(self.model.state_dict())
#                     self.save_model(f"best_model_round_{round_idx+1}.pt")
                
#                 if current_personalized_acc > self.best_personalized_acc:
#                     self.best_personalized_acc = current_personalized_acc
#                     self.best_personalized_state = copy.deepcopy(self.model.state_dict())
#                     self.save_model(f"best_personalized_model_round_{round_idx+1}.pt")
                
#                 # Save metrics
#                 save_dict_to_json(metrics, f"{self.args.log_dir}/metrics_round_{round_idx+1}.json")
                
#                 # Save overall metrics history
#                 save_dict_to_json(dict(metrics_history), f"{self.args.log_dir}/metrics_history.json")
                
#                 # Save client performance
#                 save_dict_to_json(self.client_performance, f"{self.args.log_dir}/client_performance.json")
                
#                 # Save personalization metrics
#                 if 'personalization' in metrics:
#                     self.personalization_metrics[round_idx+1] = metrics['personalization']
#                     save_dict_to_json(self.personalization_metrics, f"{self.args.log_dir}/personalization_metrics.json")
        
#         return self.model, metrics_history
    
#     def evaluate(self, round_idx):
#         """Evaluate both global and personalized models"""
#         logger.info(f"Evaluating models at round {round_idx+1}")
        
#         # Create a copy of the model for evaluation
#         eval_model = copy.deepcopy(self.model)
#         eval_model.eval()
#         eval_model.to(self.device)
        
#         # Evaluate global model
#         if hasattr(eval_model, 'disable_personalized_mode'):
#             eval_model.disable_personalized_mode()
#         metrics = evaluate(self.args, eval_model, self.testloader, self.device)
        
#         # Track trust scores if available
#         if hasattr(self.server, 'trust_scores') and len(self.server.trust_scores) > 0:
#             trust_stats = track_trust_scores(self)
#             metrics.update({"trust_stats": trust_stats})
        
#         # Evaluate personalization if enabled
#         if self.enable_personalization:
#             # Enable personalized mode
#             if hasattr(eval_model, 'enable_personalized_mode'):
#                 eval_model.enable_personalized_mode()
            
#             # Evaluate personalization benefits
#             personalization_metrics = evaluate_personalization_benefits(
#                 self.args, eval_model, self.testloader, self.device
#             )
#             metrics.update({"personalization": personalization_metrics})
            
#             # Update personalized accuracy
#             if 'acc_personalized' in personalization_metrics:
#                 metrics['acc_personalized'] = personalization_metrics['acc_personalized']
            
#             # Log key personalization metrics
#             if 'bop' in personalization_metrics:
#                 logger.info(f"Benefit of Personalization: {personalization_metrics['bop']:.4f}")
#                 logger.info(f"Global Acc: {personalization_metrics['acc_global']:.4f}, "
#                           f"Personalized Acc: {personalization_metrics['acc_personalized']:.4f}")
        
#         # Clean up
#         eval_model.to('cpu')
#         del eval_model
#         torch.cuda.empty_cache()
        
#         return metrics
    
#     def select_clients(self, round_idx):
#         """Select clients with trust-based selection if available"""
#         if self.trust_filtering and hasattr(self.server, 'select_clients') and callable(self.server.select_clients):
#             # Use server's trust-based client selection if available
#             num_clients = max(1, int(self.args.trainer.participation_rate * len(self.clients)))
#             selected_clients = self.server.select_clients(list(self.clients.keys()), num_clients)
#             logger.info(f"Trust-based client selection: {len(selected_clients)} clients")
#             return selected_clients
#         else:
#             # Fall back to random selection
#             return super().select_clients(round_idx)
    
#     def save_model(self, filename):
#         """Save model checkpoint with training metadata"""
#         try:
#             save_path = os.path.join(self.args.log_dir, filename)
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
#             # Prepare checkpoint
#             checkpoint = {
#                 'model_state': self.model.state_dict(),
#                 'round': self.global_round,
#                 'best_acc': self.best_acc,
#                 'best_personalized_acc': self.best_personalized_acc,
#                 'args': self.args,
#                 'personalization_enabled': self.enable_personalization,
#                 'timestamp': time.strftime("%Y%m%d-%H%M%S")
#             }
            
#             # Add personalization metrics if available
#             if self.personalization_metrics:
#                 checkpoint['personalization_metrics'] = self.personalization_metrics
            
#             # Save checkpoint
#             torch.save(checkpoint, save_path)
#             logger.info(f"Model saved to {save_path}")
            
#             # Save additional metadata
#             metadata = {
#                 'round': self.global_round,
#                 'accuracy': self.best_acc,
#                 'personalized_accuracy': self.best_personalized_acc,
#                 'timestamp': checkpoint['timestamp']
#             }
#             metadata_path = os.path.join(os.path.dirname(save_path), 'model_metadata.json')
#             save_dict_to_json(metadata, metadata_path)
            
#         except Exception as e:
#             logger.error(f"Failed to save model: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
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
        
        self.trust_ema = 0.5  # Initialize exponential moving average for trust
        self.trust_history = []  # Keep track of trust scores
        self.acc_history = {'global': [], 'personalized': []}  # Track accuracy history
        self.trust_momentum = 0.9  # Momentum factor for trust score updates
        self.min_trust = 0.1  # Minimum trust score
        self.max_trust = 1.0  # Maximum trust score
        
        logger.info(f"Initialized PersonalizedTrainer with personalization={self.enable_personalization}, "
                   f"adaptive_lr={self.adaptive_lr}, trust_filtering={self.trust_filtering}")
    
    def calculate_trust_score(self, global_acc, personalized_acc, round_number):
        """Calculate dynamic trust score based on performance metrics and training progress"""
        acc_ratio = personalized_acc / (global_acc + 1e-8)  # Prevent division by zero
        base_trust = min(acc_ratio, 2.0) / 2.0  # Normalize to [0, 1]
        round_factor = min(round_number / 100, 1.0)  # Scales up to 1.0 over first 100 rounds
        noise = torch.randn(1).item() * 0.02  # Using randn instead of normal
        new_trust = self.trust_momentum * self.trust_ema + (1 - self.trust_momentum) * base_trust
        new_trust = new_trust * round_factor + noise
        new_trust = max(self.min_trust, min(self.max_trust, new_trust))
        self.trust_ema = new_trust
        self.trust_history.append(new_trust)
        return new_trust
    
    def train(self):
        """Training process with personalization support"""
        metrics_history = defaultdict(list)
        self.global_round = 0
        
        if self.server is not None:
            self.server.setup(self.model)
        
        for round_idx in range(self.args.trainer.global_rounds):
            self.global_round = round_idx
            logger.info(f"=== Round {round_idx+1}/{self.args.trainer.global_rounds} ===")
            
            if self.adaptive_freezing and hasattr(self.model, 'setup_adaptive_freezing'):
                current_freeze_ratio = max(0.0, self.freeze_ratio - (round_idx * self.freeze_decay_rate))
                self.model.setup_adaptive_freezing(freeze_ratio=current_freeze_ratio)
                logger.info(f"Adaptive freezing ratio: {current_freeze_ratio:.3f}")
            
            selected_clients = self.select_clients(round_idx)
            logger.info(f"Selected {len(selected_clients)} clients for training")
            
            global_model = self.server.get_global_model() if self.server is not None else self.model
            global_model_dict = global_model.state_dict()
            
            client_models = {}
            client_weights = {}
            client_stats = {}
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                
                local_lr = self.args.trainer.local_lr
                if self.adaptive_lr and hasattr(client, 'trust_score'):
                    trust_score = getattr(client, 'trust_score', 1.0)
                    local_lr = setup_adaptive_learning_rate(
                        base_lr=self.args.trainer.local_lr,
                        max_lr=self.args.trainer.local_lr * 2,
                        trust_score=trust_score,
                        step=round_idx,
                        step_size=10
                    )
                    logger.debug(f"Client {client_idx} adaptive LR: {local_lr:.6f} (trust: {trust_score:.2f})")
                
                if hasattr(client, 'setup'):
                    client.setup(
                        state_dict=global_model_dict,
                        device=self.device,
                        local_dataset=self.trainset[client_idx],
                        global_epoch=round_idx,
                        local_lr=local_lr,
                        trainer=self
                    )
                
                client_model_dict, stats = client.local_train(round_idx)
                
                if client_model_dict is not None:
                    client_models[client_idx] = client_model_dict
                    client_weights[client_idx] = len(self.trainset[client_idx])
                    client_stats[client_idx] = stats
                    
                    if stats is not None:
                        if client_idx not in self.client_performance:
                            self.client_performance[client_idx] = []
                        self.client_performance[client_idx].append({
                            'round': round_idx,
                            'loss': stats.get('global_loss', stats.get('loss', 0.0)),
                            'trust_score': stats.get('trust_score', 1.0)
                        })
            
            if self.server is not None and client_models:
                updated_model_dict = self.server.update_global_model(client_models, client_weights, client_stats)
                self.model.load_state_dict(updated_model_dict)
                logger.info(f"Global model updated with {len(client_models)} client models")
            
            if hasattr(self.server, 'trust_scores') and self.server.trust_scores:
                avg_trust = sum(self.server.trust_scores.values()) / len(self.server.trust_scores)
                logger.info(f"Average trust score: {avg_trust:.4f}")
            
            if (round_idx + 1) % self.args.trainer.eval_every == 0:
                metrics = self.evaluate(round_idx)
                
                for k, v in metrics.items():
                    metrics_history[k].append(v)
                
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
                
                save_dict_to_json(metrics, f"{self.args.log_dir}/metrics_round_{round_idx+1}.json")
                save_dict_to_json(dict(metrics_history), f"{self.args.log_dir}/metrics_history.json")
                save_dict_to_json(self.client_performance, f"{self.args.log_dir}/client_performance.json")
                
                if 'personalization' in metrics:
                    self.personalization_metrics[round_idx+1] = metrics['personalization']
                    save_dict_to_json(self.personalization_metrics, f"{self.args.log_dir}/personalization_metrics.json")
        
        return self.model, metrics_history
    
    def evaluate(self, round_idx):
        """Evaluate both global and personalized models"""
        logger.info(f"Evaluating models at round {round_idx+1}")
        
        eval_model = copy.deepcopy(self.model)
        eval_model.eval()
        eval_model.to(self.device)
        
        if hasattr(eval_model, 'disable_personalized_mode'):
            eval_model.disable_personalized_mode()
        metrics = evaluate(self.args, eval_model, self.testloader, self.device)
        
        if hasattr(self.server, 'trust_scores') and len(self.server.trust_scores) > 0:
            trust_stats = track_trust_scores(self)
            metrics.update({"trust_stats": trust_stats})
        
        if self.enable_personalization:
            if hasattr(eval_model, 'enable_personalized_mode'):
                eval_model.enable_personalized_mode()
            
            personalization_metrics = evaluate_personalization_benefits(
                self.args, eval_model, self.testloader, self.device
            )
            metrics.update({"personalization": personalization_metrics})
            
            if 'acc_personalized' in personalization_metrics:
                metrics['acc_personalized'] = personalization_metrics['acc_personalized']
            
            if 'bop' in personalization_metrics:
                logger.info(f"Benefit of Personalization: {personalization_metrics['bop']:.4f}")
                logger.info(f"Global Acc: {personalization_metrics['acc_global']:.4f}, "
                          f"Personalized Acc: {personalization_metrics['acc_personalized']:.4f}")
        
        eval_model.to('cpu')
        del eval_model
        torch.cuda.empty_cache()
        
        return metrics
    
    def select_clients(self, round_idx):
        """Select clients with trust-based selection if available"""
        if self.trust_filtering and hasattr(self.server, 'select_clients') and callable(self.server.select_clients):
            num_clients = max(1, int(self.args.trainer.participation_rate * len(self.clients)))
            selected_clients = self.server.select_clients(list(self.clients.keys()), num_clients)
            logger.info(f"Trust-based client selection: {len(selected_clients)} clients")
            return selected_clients
        else:
            return super().select_clients(round_idx)
    
    def save_model(self, filename):
        """Save model checkpoint with training metadata"""
        try:
            save_path = os.path.join(self.args.log_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            checkpoint = {
                'model_state': self.model.state_dict(),
                'round': self.global_round,
                'best_acc': self.best_acc,
                'best_personalized_acc': self.best_personalized_acc,
                'args': self.args,
                'personalization_enabled': self.enable_personalization,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            if self.personalization_metrics:
                checkpoint['personalization_metrics'] = self.personalization_metrics
            
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
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
