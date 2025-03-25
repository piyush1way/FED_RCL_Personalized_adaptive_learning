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
from torch.utils.data import DataLoader
import traceback
import gc

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class PersonalizedTrainer(BaseTrainer):
    """Trainer with personalization support for FedRCL"""
    
    def __init__(self, args, model, trainset, testset, clients, server, evaler):
        super().__init__(args, model, trainset, testset, clients, server, evaler)
        
        # Set device based on CUDA availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        # Initialize training
        self.train_init()
        metrics = {}
        metrics_history = defaultdict(list)
        
        try:
            # Train for specified number of rounds
            for round_idx in range(self.args.trainer.global_rounds):
                logger.info(f"=== Round {round_idx+1}/{self.args.trainer.global_rounds} ===")
                
                # Pre-training steps (client selection, layer freezing, etc.)
                selected_clients = self._pre_training_steps(round_idx)
                
                # Train selected clients
                updated_models, client_stats = self._train_clients(selected_clients)
                
                # Update global model
                self._update_global_model(selected_clients, updated_models, client_stats)
                
                # Post-training steps (evaluation, saving, etc.)
                self._post_training_steps(round_idx)
                
                # Clear cache periodically to avoid memory leaks
                if hasattr(self.args, 'memory') and hasattr(self.args.memory, 'empty_cache_freq'):
                    if (round_idx+1) % self.args.memory.empty_cache_freq == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("Cleared CUDA cache to prevent memory issues")
                
                # Save metrics for tracking
                if hasattr(self, 'round_accs'):
                    metrics['acc_history'] = self.round_accs
                if hasattr(self, 'round_personalized_accs'):
                    metrics['personalized_acc_history'] = self.round_personalized_accs
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            logger.error(traceback.format_exc())
            raise e
        
        # Final evaluation
        final_metrics = self.evaluate(self.args.trainer.global_rounds - 1)
        if metrics:
            metrics.update(final_metrics)
        
        return self.model, metrics
    
    def evaluate(self, round_idx=None, eval_model=None):
        """Evaluate model on test data"""
        if eval_model is None:
            eval_model = self.model.to(self.device)
            
        # Create a criterion for loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        
        # Evaluate global model by default
        if hasattr(eval_model, 'disable_personalized_mode'):
            eval_model.disable_personalized_mode()
        
        # Evaluate model and get metrics
        metrics = evaluate(self.args, eval_model, self.testloader, self.device, criterion)
        
        # Store best accuracy so far
        current_acc = metrics.get('acc', 0.0)
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            logger.info(f"New best accuracy: {self.best_acc:.4f}")
            if round_idx is not None:
                self.save_model(suffix=f"best_model_round_{round_idx}")
        
        # Track metrics
        self.round_accs.append(current_acc)
        
        # Evaluate personalization benefits if enabled
        if self.enable_personalization:
            try:
                if hasattr(eval_model, 'enable_personalized_mode'):
                    eval_model.enable_personalized_mode()
                
                personalization_metrics = evaluate_personalization_benefits(
                    self.args, eval_model, self.testloader, self.device, criterion=criterion
                )
                metrics.update({"personalization": personalization_metrics})
                    
                if 'acc_personalized' in personalization_metrics:
                    metrics['acc_personalized'] = personalization_metrics['acc_personalized']
                    
                if 'bop' in personalization_metrics:
                    logger.info(f"Benefit of Personalization: {personalization_metrics['bop']:.4f}")
                    
                    # Fix KeyError by checking for different possible keys or using get() with default value
                    global_acc = personalization_metrics.get('acc_global', 
                                personalization_metrics.get('global_acc', metrics.get('acc', 0.0)))
                    
                    personalized_acc = personalization_metrics.get('acc_personalized', 
                                      personalization_metrics.get('personalized_acc', global_acc))
                    
                    logger.info(f"Global Acc: {global_acc:.4f}, Personalized Acc: {personalized_acc:.4f}")
                    
                    # Store best personalized accuracy
                    if personalized_acc > self.best_personalized_acc:
                        self.best_personalized_acc = personalized_acc
                        logger.info(f"New best personalized accuracy: {self.best_personalized_acc:.4f}")
                        if round_idx is not None:
                            self.save_model(suffix=f"best_personalized_model_round_{round_idx}")
                            
                    # Track personalized metrics
                    self.round_personalized_accs.append(personalized_acc)
                    self.personalization_metrics = personalization_metrics
            except Exception as e:
                logger.error(f"Error in personalization evaluation: {str(e)}")
                logger.error(f"Personalization metrics keys: {list(personalization_metrics.keys()) if 'personalization_metrics' in locals() else 'N/A'}")
                # Continue with evaluation without personalization results
                if 'acc_personalized' not in metrics:
                    metrics['acc_personalized'] = metrics.get('acc', 0.0)  # Use global accuracy as fallback
        
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
    
    def save_model(self, suffix=None):
        """Save model checkpoint with training metadata"""
        try:
            if suffix is None:
                suffix = "final"
            
            # Ensure we have a log_dir
            if not hasattr(self.args, 'log_dir'):
                self.args.log_dir = './checkpoints'
                os.makedirs(self.args.log_dir, exist_ok=True)
            
            # Create filename from suffix
            filename = f"{suffix}.pt"
            save_path = os.path.join(self.args.log_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Check for global_round attribute
            if not hasattr(self, 'global_round'):
                self.global_round = getattr(self, 'current_round', 0)
            
            checkpoint = {
                'model_state': self.model.state_dict(),
                'round': self.global_round,
                'best_acc': self.best_acc,
                'best_personalized_acc': self.best_personalized_acc,
                'args': self.args,
                'personalization_enabled': self.enable_personalization,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            if hasattr(self, 'personalization_metrics') and self.personalization_metrics:
                checkpoint['personalization_metrics'] = self.personalization_metrics
            
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
            metadata = {
                'round': self.global_round,
                'accuracy': float(self.best_acc),  # Convert tensors to float for JSON
                'personalized_accuracy': float(self.best_personalized_acc),
                'timestamp': checkpoint['timestamp']
            }
            metadata_path = os.path.join(os.path.dirname(save_path), 'model_metadata.json')
            save_dict_to_json(metadata, metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            logger.error(traceback.format_exc())

    def train_init(self):
        """Initialize training by setting up models and optimizer"""
        # Set up model and optimizer
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = getattr(self.args, 'enable_benchmark', True)
            
            # Configure memory management
            max_split_size_mb = getattr(self.args, 'max_split_size_mb', 128)
            torch.cuda.set_per_process_memory_fraction(0.85)  # Limit memory usage to 85%
            
            # Set environment variables for PyTorch memory management
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'
            
            logger.info(f"CUDA setup: max_split_size_mb={max_split_size_mb}")
        
        # Initialize model with personalization support
        if hasattr(self.model, 'enable_personalized_mode') and self.enable_personalization:
            self.model.enable_personalized_mode()
            logger.info("Personalized mode enabled for training")
        
        # Initialize server with the model
        if self.server and hasattr(self.server, 'setup'):
            # Move model to CPU first to reduce memory during setup
            model_device = next(self.model.parameters()).device
            self.model = self.model.to('cpu')
            
            try:
                self.server.setup(self.model)
                # Restore model to original device
                self.model = self.model.to(model_device)
                
                # Get global model from server if available
                if hasattr(self.server, 'get_global_model'):
                    self.global_model = self.server.get_global_model()
                else:
                    # Create global model with minimal memory usage
                    self.global_model = self.create_model_copy(self.model)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA OOM during server setup. Using alternative initialization.")
                    # Clear cache and retry with more aggressive memory management
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Create global model directly with minimal memory usage
                    self.global_model = self.create_model_copy(self.model)
                    
                    # Restore model to original device
                    self.model = self.model.to(model_device)
                else:
                    raise e
        else:
            self.global_model = self.create_model_copy(self.model)
        
        # Move models to the right device
        self.model = self.model.to(self.device)
        self.global_model = self.global_model.to(self.device)
        
        # Set up evaluation metrics
        self.best_acc = 0.
        self.best_personalized_acc = 0.
        self.round_losses = []
        self.round_accs = []
        self.round_personalized_accs = []
        
        # Get important training parameters from config
        self.num_clients = self.args.trainer.num_clients
        self.batch_size = getattr(self.args, 'batch_size', 32)
        self.local_epochs = getattr(self.args.trainer, 'local_epochs', 3)
        self.local_lr = getattr(self.args.trainer, 'local_lr', 0.01)
        
        # Setup trust-based weighting 
        self.trust_based_weighting = getattr(self.args.client.trust_filtering, 'enable', False) if hasattr(self.args.client, 'trust_filtering') else False
        
        # Initialize client data loaders
        self.trainloaders = {}
        for client_idx in range(self.num_clients):
            if client_idx in self.trainset and len(self.trainset[client_idx]) > 0:
                self.trainloaders[client_idx] = DataLoader(
                    self.trainset[client_idx],
                    batch_size=min(self.batch_size, len(self.trainset[client_idx])),
                    shuffle=True,
                    num_workers=getattr(self.args, 'num_workers', 2),
                    pin_memory=getattr(self.args, 'pin_memory', True),
                    drop_last=False
                )
        
        # Initialize client data distribution information
        self.client_data_distributions = {}
        for client_idx in range(self.num_clients):
            if client_idx in self.trainset:
                # Count labels in the dataset
                label_counts = {}
                for _, label in self.trainset[client_idx]:
                    label_val = label.item() if hasattr(label, 'item') else label
                    label_counts[label_val] = label_counts.get(label_val, 0) + 1
                self.client_data_distributions[client_idx] = {
                    'total_samples': len(self.trainset[client_idx]),
                    'label_counts': label_counts
                }
        
        # Track trust scores if using trust-based client selection
        self.trust_scores = {}
        self.client_performances = {client_idx: {"global_acc": [], "personalized_acc": []} 
                                  for client_idx in range(self.num_clients)}

    def create_model_copy(self, model):
        """Create a copy of model with minimal memory usage"""
        try:
            # First try to create a new instance of the same class
            if hasattr(model, '__class__'):
                model_class = model.__class__
                if hasattr(model, 'init_params') and model.init_params:
                    # If model has stored initialization parameters
                    new_model = model_class(**model.init_params)
                    new_model.load_state_dict(model.state_dict())
                    return new_model
            
            # Fallback to deepcopy but with careful memory management
            model_device = next(model.parameters()).device
            model = model.to('cpu')  # Move to CPU first
            
            # Clear CUDA cache before copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            new_model = copy.deepcopy(model)
            
            # Restore original model device
            model.to(model_device)
            
            return new_model
        
        except Exception as e:
            logger.error(f"Error in model copy: {str(e)}")
            # Last resort: create new model and copy state dict
            if hasattr(model, '_get_name'):
                logger.info(f"Attempting alternative copy for {model._get_name()}")
            
            # Create new state dict to copy values one by one
            new_state_dict = {}
            for key, param in model.state_dict().items():
                new_state_dict[key] = param.clone().detach().cpu()
            
            # Create a new instance (this assumes model has a constructor that takes no args)
            try:
                new_model = model.__class__()
                new_model.load_state_dict(new_state_dict)
                return new_model
            except:
                logger.error("Failed to create model copy")
                # Return original model as last resort
                return model

    def _train_clients(self, selected_clients):
        """Train selected clients and return updated models and metrics"""
        updated_models = {}
        client_stats = {}
        
        for idx, client_idx in enumerate(selected_clients):
            # Skip clients with insufficient data
            if not hasattr(self.trainloaders[client_idx], 'dataset') or len(self.trainloaders[client_idx].dataset) < 2:
                logger.warning(f"Skipping client {client_idx} due to insufficient data")
                continue
            
            # Ensure client model exists
            if not hasattr(self.clients[client_idx], 'model') or self.clients[client_idx].model is None:
                logger.info(f"Initializing model for client {client_idx}")
                self.clients[client_idx].model = copy.deepcopy(self.model)
                
            # Setup client for training
            try:
                self.clients[client_idx].setup(
                    state_dict=self.global_model.state_dict(),
                    device=self.device,
                    local_dataset=self.trainloaders[client_idx].dataset,
                    global_epoch=self.current_round,
                    local_lr=self.local_lr,
                    local_ep=self.local_epochs,
                    local_bs=self.batch_size,
                    trainer=self,
                    num_workers=self.args.num_workers,
                    pin_memory=self.args.pin_memory
                )
            except Exception as e:
                logger.error(f"Error setting up client {client_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # Check if client is ready for training
            if not hasattr(self.clients[client_idx], 'trainloader') or self.clients[client_idx].trainloader is None:
                logger.warning(f"Client {client_idx} has no trainloader, skipping")
                continue
            
            # Train the client
            try:
                start_time = time.time()
                updated_model, client_metrics = self.clients[client_idx].local_train(self.current_round)
                
                training_time = time.time() - start_time
                if client_metrics is not None:
                    client_metrics['training_time'] = training_time
                
                if updated_model is not None:
                    updated_models[client_idx] = updated_model
                if client_metrics is not None:
                    client_stats[client_idx] = client_metrics
                    
                logger.info(f"Client {client_idx} training completed in {training_time:.2f}s. "
                          f"Trust score: {client_metrics.get('trust_score', 'N/A')}")
                    
            except Exception as e:
                logger.error(f"Error training client {client_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Try to recover by cleaning up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        # Check if any models were updated
        if not updated_models:
            logger.warning("No updated models received from clients")
            
        return updated_models, client_stats

    def _update_global_model(self, selected_clients, updated_models, client_stats):
        """Update the global model based on client updates"""
        if not updated_models:
            logger.warning("No updated models received from clients")
            return
        
        # Calculate client weights based on dataset sizes if not using trust-based weighting
        if not self.trust_based_weighting:
            client_weights = {
                client_idx: len(self.trainloaders[client_idx].dataset) 
                for client_idx in updated_models.keys()
            }
            total_samples = sum(client_weights.values())
            client_weights = {k: v/total_samples for k, v in client_weights.items()} if total_samples > 0 else None
        else:
            # Use trust scores for weighting if available
            client_weights = {
                client_idx: client_stats[client_idx].get('trust_score', 1.0) 
                for client_idx in updated_models.keys() if client_idx in client_stats
            }
            total_weight = sum(client_weights.values())
            client_weights = {k: v/total_weight for k, v in client_weights.items()} if total_weight > 0 else None
        
        # Update global model with weighted client models
        global_model_state = self.server.update_global_model(updated_models, client_weights, client_stats)
        self.global_model.load_state_dict(global_model_state)
        
        # Update trust scores
        if hasattr(self.server, 'trust_scores'):
            self.trust_scores = self.server.trust_scores
        
        # Log aggregation statistics
        logger.info(f"Global model updated with {len(updated_models)} client models")
        if self.trust_scores:
            avg_trust = sum(self.trust_scores.values()) / len(self.trust_scores) if self.trust_scores else 0
            logger.info(f"Average trust score: {avg_trust:.4f}")

    def _pre_training_steps(self, round_idx):
        """Perform steps before training begins for the current round"""
        # Adjust learning rate based on round
        self.current_round = round_idx
        
        # Ensure we have num_clients defined
        if not hasattr(self, 'num_clients'):
            self.num_clients = self.args.trainer.num_clients
        
        # Perform adaptive layer freezing if enabled
        if self.enable_personalization and hasattr(self.model, 'setup_adaptive_freezing'):
            if self.adaptive_freezing:
                # Get global rounds from config
                global_rounds = self.args.trainer.global_rounds
                # Gradually unfreeze more layers as training progresses
                freeze_ratio = max(0.0, self.freeze_ratio - (round_idx / (global_rounds * 2)))
                self.model.setup_adaptive_freezing(freeze_ratio=freeze_ratio)
                logger.info(f"Adaptive freezing ratio: {freeze_ratio:.3f}")
        
        # Select clients for this round using trust-based selection if enabled
        if self.trust_filtering and hasattr(self.server, 'select_clients'):
            selected_clients = self.server.select_clients(
                list(range(self.num_clients)), 
                max(1, int(self.num_clients * self.args.trainer.participation_rate))
            )
            logger.info(f"Trust-based client selection: {len(selected_clients)} clients")
        else:
            # Random selection
            selected_clients = np.random.choice(
                range(self.num_clients), 
                max(1, int(self.num_clients * self.args.trainer.participation_rate)), 
                replace=False
            ).tolist()
        
        logger.info(f"Selected {len(selected_clients)} clients for training")
        return selected_clients

    def _post_training_steps(self, round_idx):
        """Perform steps after training completes for the current round"""
        # Evaluate global model
        if round_idx % self.args.trainer.eval_every == 0 or round_idx == self.args.trainer.global_rounds - 1:
            metrics = self.evaluate(round_idx)
            
            # Save best model
            if metrics["acc"] > self.best_acc:
                self.best_acc = metrics["acc"]
                self.save_model(suffix=f"best_model_round_{round_idx}")
            
            # Save best personalized model
            if "acc_personalized" in metrics and metrics["acc_personalized"] > self.best_personalized_acc:
                self.best_personalized_acc = metrics["acc_personalized"]
                self.save_model(suffix=f"best_personalized_model_round_{round_idx}")
            
            # Save metrics
            self._save_metrics(round_idx, metrics)

    def _save_metrics(self, round_idx, metrics):
        """Save metrics to disk and update histories"""
        try:
            # Ensure we have log_dir
            if not hasattr(self.args, 'log_dir'):
                self.args.log_dir = './checkpoints'
                os.makedirs(self.args.log_dir, exist_ok=True)
            
            # Initialize tracking dictionaries if they don't exist
            if not hasattr(self, 'round_accs'):
                self.round_accs = []
            if not hasattr(self, 'round_personalized_accs'):
                self.round_personalized_accs = []
            if not hasattr(self, 'personalization_metrics'):
                self.personalization_metrics = {}
            
            # Update tracking
            if 'acc' in metrics:
                self.round_accs.append(float(metrics['acc']))
            
            if 'acc_personalized' in metrics:
                self.round_personalized_accs.append(float(metrics['acc_personalized']))
            
            # Handle saving the metrics files
            try:
                # Convert tensors to Python types for JSON serialization
                clean_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        clean_metrics[k] = v.item() if v.numel() == 1 else v.tolist()
                    elif isinstance(v, np.ndarray):
                        clean_metrics[k] = v.item() if v.size == 1 else v.tolist()
                    elif isinstance(v, dict):
                        clean_metrics[k] = {
                            kk: vv.item() if isinstance(vv, torch.Tensor) and vv.numel() == 1 else 
                               vv.tolist() if isinstance(vv, torch.Tensor) else vv
                            for kk, vv in v.items()
                        }
                    else:
                        clean_metrics[k] = v
                
                # Save to individual round files
                save_dict_to_json(clean_metrics, os.path.join(self.args.log_dir, f"metrics_round_{round_idx}.json"))
                
                # Save history
                if hasattr(self, 'round_accs') and self.round_accs:
                    history = {
                        'global_acc': self.round_accs,
                        'personalized_acc': self.round_personalized_accs if hasattr(self, 'round_personalized_accs') else []
                    }
                    save_dict_to_json(history, os.path.join(self.args.log_dir, "metrics_history.json"))
                
                # Save client performance if available
                if hasattr(self, 'client_performances'):
                    save_dict_to_json(self.client_performances, os.path.join(self.args.log_dir, "client_performance.json"))
                
                # Save personalization metrics if available
                if 'personalization' in metrics:
                    self.personalization_metrics[round_idx+1] = metrics['personalization']
                    save_dict_to_json(self.personalization_metrics, os.path.join(self.args.log_dir, "personalization_metrics.json"))
                
            except Exception as e:
                logger.warning(f"Error saving metrics files: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            logger.error(traceback.format_exc())
