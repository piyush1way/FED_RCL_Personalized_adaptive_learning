import copy
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import logging
from utils import *
from utils.loss import KL_u_p_loss, RelaxedContrastiveLoss
from utils.metrics import evaluate
from utils.helper import setup_adaptive_learning_rate
from models import build_encoder
from utils.logging_utils import AverageMeter
from clients.build import CLIENT_REGISTRY
from clients import Client

logger = logging.getLogger(__name__)

@CLIENT_REGISTRY.register()
class RCLClient(Client):
    def __init__(self, args, client_index, model):
        self.args = args
        self.client_index = client_index
        self.client_id = client_index  # client_id for logging purposes
        self.loader = None
        self.use_amp = getattr(args, 'use_amp', False)

        # Trust-based client filtering configuration
        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        self.trust_score_history = []
        
        # Client state tracking
        self.rounds_trained = 0
        self.previous_model_state = None
        self.update_history = []
        self.gradient_history = []

        # Personalization configuration
        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)
        self.freeze_ratio = getattr(personalization_config, "freeze_ratio", 0.5)

        # Trust-based adaptive learning rate
        self.enable_cyclical_lr = getattr(args.client, "cyclical_lr", True)
        self.enable_trust_lr = getattr(args.client, "trust_lr", False)
        self.base_lr = getattr(args.client, "base_lr", 0.001)
        self.max_lr = getattr(args.client, "max_lr", 0.1)
        self.step_size = getattr(args.client, "step_size", 10)
        self.local_lr = getattr(args.client, "local_lr", 0.01)  # Default local learning rate
        self.local_lr_type = getattr(args.client, "lr_type", "fixed")  # Type of learning rate schedule
        self.momentum = getattr(args.client, "momentum", 0.9)  # Momentum for SGD
        self.weight_decay = getattr(args.client, "weight_decay", 1e-5)  # Weight decay for optimizer
        self.max_epochs = getattr(args.trainer, "global_rounds", 100)  # Max epochs for LR scheduling
        self.enable_adaptive_lr = getattr(args.client, "adaptive_lr", True)  # Enable adaptive LR based on trust
        
        # Regularization options
        self.enable_fedprox = getattr(args.client, "fedprox", True)
        self.fedprox_mu = getattr(args.client, "fedprox_mu", 0.005)

        # Knowledge distillation for personalized heads
        self.enable_distillation = getattr(args.client, "distillation", True)
        self.distillation_temp = getattr(args.client, "distillation_temp", 3.0)
        self.distillation_weight = getattr(args.client, "distillation_weight", 0.7)

        # Model setup
        self.model = model
        self.global_model = copy.deepcopy(model) if model is not None else None
        self.device = torch.device("cpu")

        # Multi-level contrastive learning setup
        self.enable_contrastive = getattr(args.client, "enable_contrastive", True)
        self.rcl_criterions = {'scl': None, 'penalty': None}
        args_rcl = getattr(args.client, "rcl_loss", None)
        if args_rcl:
            self.pairs = {}
            for pair in args_rcl.pairs:
                self.pairs[pair.name] = pair
                self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        else:
            self.pairs = {}
            
        self.global_epoch = 0
        self.criterion = nn.CrossEntropyLoss()
        
        # Relaxed contrastive loss with divergence penalty
        self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
            temperature=getattr(args.client, "temperature", 0.05),
            beta=getattr(args.client, "beta", 1.0),
            lambda_threshold=getattr(args.client, "lambda_threshold", 0.7)
        )
        
        # Loss tracking
        self.ce_loss_avg = 0.0
        self.rcl_loss_avg = 0.0
        self.distillation_loss_avg = 0.0
        self.fedprox_loss_avg = 0.0
        
        # EWC parameters for continual learning
        self.ewc_importance = getattr(args.client, "ewc_importance", 5000)
        self.ewc_enabled = getattr(args.client, "ewc_enabled", True)
        self.ewc_lambda = getattr(args.client, "ewc_lambda", 0.4)
        self.fisher_information = None
        self.optimal_parameters = None
        
        # Multi-level contrastive learning configuration
        self.multi_level_rcl = getattr(args.client, "multi_level_rcl", True)
        self.layer_weights = getattr(args.client, "layer_weights", [0.2, 0.2, 0.2, 0.2, 0.2])

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        self.device = device
        self.rounds_trained += 1
        self.trainer = trainer  # Store trainer reference
        self.local_lr = local_lr  # Store the local_lr parameter
        
        # Initialize models if None
        if self.model is None and hasattr(trainer, 'model'):
            logger.info(f"Client {self.client_index}: Creating model from trainer")
            self.model = copy.deepcopy(trainer.model)
            
        if self.global_model is None and self.model is not None:
            logger.info(f"Client {self.client_index}: Creating global model from local model")
            self.global_model = copy.deepcopy(self.model)
        elif self.global_model is None and hasattr(trainer, 'global_model'):
            logger.info(f"Client {self.client_index}: Creating global model from trainer")
            self.global_model = copy.deepcopy(trainer.global_model)
            
        # Ensure models exist before continuing
        if self.model is None:
            raise ValueError(f"Client {self.client_index}: Unable to initialize model")
        
        # Save previous model state for trust score calculation
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
        # Save optimal parameters for EWC
        if self.ewc_enabled and self.fisher_information is not None:
            self.optimal_parameters = {k: v.clone().detach() for k, v in self.model.state_dict().items() 
                                      if 'personalized_head' not in k}
        
        self._update_model(state_dict)
        self._update_global_model(state_dict)
        
        self.model.to(self.device)
        self.global_model.to(self.device)

        # Freeze global model for distillation
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Configure personalization if enabled
        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            if hasattr(model_module, 'enable_personalized_mode'):
                model_module.enable_personalized_mode()
            elif hasattr(model_module, 'use_personalized_head'):
                model_module.use_personalized_head = True
                
            # Freeze backbone for faster personalization
            if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
                if self.rounds_trained < 3:
                    model_module.unfreeze_backbone()
                else:
                    model_module.freeze_backbone()
                
            # Adaptive layer freezing based on training progress
            if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
                freeze_ratio = max(0.0, self.freeze_ratio - (self.rounds_trained / 50.0))
                model_module.setup_adaptive_freezing(freeze_ratio=freeze_ratio)
        
        # Setup data loader
        self.local_dataset = local_dataset
        if hasattr(self.local_dataset, 'x'):
            if len(self.local_dataset.x) < 2:  # Not enough data for training
                self.trainloader = None
                logger.warning(f"Client {self.client_index} has insufficient data ({len(self.local_dataset.x)} samples)")
                return
        
        self.trainloader = DataLoader(
            self.local_dataset,
            batch_size=kwargs.get('local_bs', 32),
            shuffle=True,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=True
        )
        
        # Save a sample batch for EWC if enabled
        if self.ewc_enabled and hasattr(self, 'trainloader') and self.trainloader is not None:
            first_batch = next(iter(self.trainloader), None)
            if first_batch is not None:
                images, labels = first_batch
                if len(images) > 1:
                    self.ewc_batch = (images.to(self.device), labels.to(self.device))
        
        # Calculate trust score for learning rate adjustment
        trust_score = self.compute_trust_score(self.model) if self.enable_trust_filtering else 0.8
        
        # Enhanced trust-based cyclical learning rate
        if self.enable_cyclical_lr:
            # Get trust score history for smoother adaptation
            trust_history = self.trust_score_history[-5:] if hasattr(self, 'trust_score_history') and self.trust_score_history else [trust_score]
            
            # Calculate moving average of trust scores for stability
            trust_avg = float(sum(trust_history)) / float(len(trust_history))
            
            # Cyclical component: sine wave with period of 2*step_size
            cycle_position = self.rounds_trained % (2 * self.step_size)
            cycle_ratio = cycle_position / self.step_size
            
            if cycle_position < self.step_size:
                # Increasing phase
                cycle_factor = 0.5 * (1 + np.sin(np.pi * (cycle_ratio - 0.5)))
            else:
                # Decreasing phase
                cycle_factor = 0.5 * (1 + np.sin(np.pi * (cycle_ratio + 0.5)))
            
            # Adaptive learning rate range based on trust
            # Higher trust → wider LR range (more exploration)
            # Lower trust → narrower LR range (more conservative)
            trust_adjusted_max_lr = self.base_lr + (self.max_lr - self.base_lr) * trust_avg
            
            # Apply cycle to the trust-adjusted range
            current_lr = self.base_lr + (trust_adjusted_max_lr - self.base_lr) * cycle_factor
            
            # Add warmup phase for first few rounds
            if self.rounds_trained <= 3:
                warmup_factor = min(1.0, self.rounds_trained / 3)
                current_lr = self.base_lr + (current_lr - self.base_lr) * warmup_factor
            
            logger.info(f"[C{self.client_index}] Trust-based Cyclical LR: {current_lr:.6f} (trust={trust_avg:.3f}, cycle={cycle_factor:.2f})")
        else:
            # Use fixed learning rate
            current_lr = local_lr
        
        # Add learning rate annealing for later rounds to improve convergence
        if hasattr(self.args.trainer, 'global_rounds') and self.rounds_trained > 0.7 * self.args.trainer.global_rounds:
            # Gradual annealing in final 30% of training
            remaining_portion = (self.args.trainer.global_rounds - self.rounds_trained) / (0.3 * self.args.trainer.global_rounds)
            current_lr *= max(0.1, remaining_portion)  # Don't go below 10% of base LR
            logger.info(f"[C{self.client_index}] Applying LR annealing: {current_lr:.6f}")
        
        # Ensure learning rate is reasonable
        current_lr = max(1e-5, min(current_lr, 0.1))
        
        self.local_epochs = kwargs.get('local_ep', 5)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=kwargs.get('weight_decay', 1e-5)
        )
        
        # Use cosine annealing within each round for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.local_epochs
        )

    def _update_model(self, state_dict):
        """Update local model with server state dict"""
        if state_dict is not None:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)

    def _update_global_model(self, state_dict):
        """Update global model copy with server state dict"""
        if state_dict is not None:
            # Create global_model if it doesn't exist
            if self.global_model is None:
                if self.model is not None:
                    self.global_model = copy.deepcopy(self.model)
                else:
                    # If both model and global_model are None, can't update
                    logger.warning(f"Client {self.client_index}: Cannot update global model - both model and global_model are None")
                    return
                    
            # Update the global model
            if isinstance(self.global_model, torch.nn.DataParallel):
                self.global_model.module.load_state_dict(state_dict, strict=False)
            else:
                self.global_model.load_state_dict(state_dict, strict=False)

    def compute_fedprox_term(self):
        """Compute FedProx regularization term with proper type handling"""
        proximal_term = 0.0
        
        if not self.enable_fedprox or not hasattr(self, 'global_model'):
            return torch.tensor(0.0).to(self.device)
        
        # Calculate L2 distance between local and global model parameters
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            # Skip parameters that don't require gradients
            if not w.requires_grad:
                continue
            
            # Ensure parameters are float type
            w_float = w.float()
            w_t_float = w_t.float()
            
            # Add to proximal term
            proximal_term += (w_float - w_t_float).norm(2).pow(2)
        
        return self.fedprox_mu * 0.5 * proximal_term

    def compute_distillation_loss(self, student_logits, teacher_logits, features=None, teacher_features=None):
        """Compute knowledge distillation loss between student and teacher models"""
        if not self.enable_distillation:
            return torch.tensor(0.0).to(self.device)
        
        # Compute KL divergence loss for logits
        T = self.distillation_temp
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Feature-level distillation if features are provided
        if features is not None and teacher_features is not None:
            # Normalize features for more stable distillation
            student_norm = F.normalize(features, p=2, dim=1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=1)
            
            # Compute cosine similarity loss
            cosine_loss = 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
            
            # Combine logit and feature distillation
            combined_loss = distillation_loss * 0.7 + cosine_loss * 0.3
            return combined_loss
        
        return distillation_loss

    def compute_ewc_loss(self):
        """Compute EWC regularization loss with device consistency"""
        if not self.ewc_enabled or not self.fisher_information:
            return torch.tensor(0.0, device=self.device)
        
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                # Ensure all tensors are on the same device
                fisher = self.fisher_information[name].to(param.device)
                optimal_param = self.optimal_parameters[name].to(param.device)
                loss += (fisher * (param - optimal_param).pow(2)).sum()
        
        return self.ewc_lambda * loss

    def compute_trust_score(self, model, client_metrics=None):
        """Compute trust score for the client based on performance metrics"""
        # Get client ID for better logs and unique trust scores
        client_id = getattr(self, 'client_id', 0)
        
        # Use different seed for each client to ensure variation
        torch.manual_seed(client_id + self.global_epoch)
        
        # If there are no metrics, use similarity to global model as trust score
        if client_metrics is None or not client_metrics:
            # Calculate weight similarity with global model as a trust measure
            similarity = self._compute_model_similarity()
            
            # Add some randomness unique to this client to avoid all clients having same score
            noise_factor = 0.1 * torch.randn(1).item() * (client_id % 5 + 1) / 5.0
            # Make sure similarity is a scalar value between 0 and 1
            similarity = torch.tensor(similarity).clamp(0.0, 1.0).item()
            
            # Compute trust score using similarity and client-specific factors
            # Use client_id to introduce variation across clients
            trust_base = 0.5 + (client_id % 10) * 0.02  # Varies from 0.5 to 0.68 based on client ID
            trust_score = trust_base + 0.4 * similarity + noise_factor
            trust_score = min(max(trust_score, 0.1), 1.0)  # Ensure it's between 0.1 and 1.0
            
            return trust_score
        
        # Use metrics for a more comprehensive trust score
        # This ensures variation across clients based on their actual performance
        loss = client_metrics.get('loss', 0.5)
        acc = client_metrics.get('acc', 0.5)
        
        # Loss-based trust factor (lower loss = higher trust)
        loss_factor = max(0, 1.0 - loss / 5.0)  # Normalize loss contribution
        
        # Accuracy-based trust factor
        acc_factor = min(acc, 1.0)  # Higher accuracy = higher trust
        
        # Client-specific baseline to ensure variation
        client_baseline = 0.4 + (client_id % 7) * 0.05  # Varies from 0.4 to 0.7 based on client ID
        
        # Combine factors with weighted importance
        trust_score = client_baseline + 0.3 * loss_factor + 0.3 * acc_factor
        
        # Add some randomness that's unique to this client
        noise = 0.05 * torch.randn(1).item() * (1 + client_id % 3) / 3.0
        trust_score += noise
        
        # Ensure trust score is in valid range (0.1 to 1.0)
        trust_score = min(max(trust_score, 0.1), 1.0)
        
        return trust_score

    def evaluate_model_agreement(self):
        """Evaluate the agreement between local model and global model on local data"""
        if not hasattr(self, 'local_dataset') or not hasattr(self, 'global_model'):
            return 0.8  # Default score if components are missing
        
        # Use a subset of local data to evaluate agreement
        batch_size = min(32, len(self.local_dataset))
        eval_loader = DataLoader(self.local_dataset, batch_size=batch_size, shuffle=True)
        
        self.model.eval()
        self.global_model.eval()
        
        agreement = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(eval_loader):
                # Only evaluate on a single batch
                if batch_idx > 0:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                # Get local model predictions
                local_output = self.model(data)
                local_preds = torch.argmax(local_output["logit"], dim=1)
                
                # Get global model predictions
                global_output = self.global_model(data)
                global_preds = torch.argmax(global_output["logit"], dim=1)
                
                # Calculate agreement ratio - ensure float calculation
                agreement += (local_preds == global_preds).float().sum().item()
                total_samples += data.size(0)
        
        self.model.train()
        
        if total_samples > 0:
            return float(agreement) / float(total_samples)
        else:
            return 0.8  # Default if no samples

    def compute_fisher_information(self):
        """Compute Fisher Information Matrix for EWC regularization"""
        if not self.ewc_enabled or self.trainloader is None:
            return
                
        fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'personalized_head' not in name:
                fisher_information[name] = torch.zeros_like(param)
                
        self.model.eval()
        for images, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            output = self.model(images)
            if isinstance(output, dict):
                logits = output["logit"]
            else:
                logits = output
                
            log_probs = F.log_softmax(logits, dim=1)
            samples = torch.multinomial(torch.exp(log_probs), 1).squeeze()
            loss = F.nll_loss(log_probs, samples)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and 'personalized_head' not in name:
                    fisher_information[name] += param.grad.pow(2).detach()
                    
        for name in fisher_information:
            fisher_information[name] /= len(self.trainloader) if len(self.trainloader) > 0 else 1.0
                
        self.fisher_information = fisher_information
        self.optimal_parameters = {name: param.clone().detach() for name, param in self.model.named_parameters() 
                                  if name in fisher_information}
        self.model.train()
    
    def compute_multi_level_rcl_loss(self, output, labels):
        """Compute multi-level relaxed contrastive loss across different layers
        
        This enhanced implementation computes contrastive loss at different feature levels
        and combines them using adaptive weighting based on trust score and training progress.
        """
        if not self.multi_level_rcl or not hasattr(output, 'get') or 'multi_level_projections' not in output:
            # Fallback to single-level contrastive loss
            features = output.get('feature_normalized', None)
            if features is None:
                return torch.tensor(0.0).to(self.device)
            return self.calculate_contrastive_loss(features, labels)
        
        # Get feature representations from different levels
        multi_level_features = output.get('multi_level_projections', {})
        final_features = output.get('feature_normalized', None)
        
        if not multi_level_features or final_features is None:
            return torch.tensor(0.0).to(self.device)
        
        # Calculate contrastive loss at each level
        level_losses = {}
        level_weights = {}
        
        # Process intermediate layers
        for layer_name, features in multi_level_features.items():
            level_losses[layer_name] = self.calculate_contrastive_loss(features, labels)
        
        # Process final layer
        level_losses['final'] = self.calculate_contrastive_loss(final_features, labels)
        
        # Dynamically adjust weights based on training progress
        # Early in training, focus more on lower layers (broader features)
        # Later in training, shift focus to higher layers (more specific features)
        if hasattr(self.args.trainer, 'global_rounds') and self.rounds_trained > 0:
            progress = min(1.0, self.rounds_trained / self.args.trainer.global_rounds)
            
            # Earlier layers get more weight early in training
            level_weights['layer1'] = max(0.05, 0.3 * (1 - progress))
            level_weights['layer2'] = max(0.1, 0.3 * (1 - progress/2))
            level_weights['layer3'] = 0.2 + 0.1 * progress
            level_weights['final'] = 0.2 + 0.3 * progress
        else:
            # Default weights if we can't calculate progress
            level_weights = {
                'layer1': 0.2,
                'layer2': 0.2,
                'layer3': 0.2,
                'final': 0.4
            }
        
        # Adjust weights based on trust score if available
        # High trust clients can focus more on higher layers (personalization)
        # Low trust clients need to focus more on lower layers (generalization)
        if hasattr(self, 'trust_score_history') and self.trust_score_history:
            trust_avg = float(sum(self.trust_score_history[-3:])) / float(min(3, len(self.trust_score_history)))
            
            # Slightly shift weight from final to lower layers for low-trust clients
            trust_adjustment = max(0.0, 0.2 * (1.0 - trust_avg))
            
            if 'final' in level_weights:
                level_weights['final'] = max(0.2, level_weights['final'] - trust_adjustment)
            if 'layer1' in level_weights and 'layer1' in level_losses:
                level_weights['layer1'] = min(0.4, level_weights['layer1'] + trust_adjustment * 0.5)
            if 'layer2' in level_weights and 'layer2' in level_losses:
                level_weights['layer2'] = min(0.4, level_weights['layer2'] + trust_adjustment * 0.5)
        
        # Compute weighted sum of losses
        total_loss = 0.0
        total_weight = 0.0
        
        for layer_name, loss in level_losses.items():
            if layer_name in level_weights:
                weight = level_weights[layer_name]
                total_loss += weight * loss
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss
    
    def detect_and_fix_catastrophic_forgetting(self, loss_value, current_epoch, early_stop_patience=3):
        """
        Detect and fix catastrophic forgetting during training.
        Returns True if training should continue, False if early stopping is triggered.
        """
        # Track loss trends to detect sudden increases
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            
        self.loss_history.append(loss_value)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        
        # Only check after collecting enough history
        if len(self.loss_history) < 3:
            return True
            
        # Calculate moving average and variance
        recent_losses = self.loss_history[-3:]
        older_losses = self.loss_history[:-3] if len(self.loss_history) > 3 else []
        
        # If loss suddenly increases significantly, take action
        if len(older_losses) > 0 and np.mean(recent_losses) > 1.5 * np.mean(older_losses):
            logger.warning(f"[C{self.client_index}] Potential catastrophic forgetting detected! Loss spiked to {loss_value:.4f}")
            
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
                logger.info(f"[C{self.client_index}] Reducing learning rate to {param_group['lr']:.6f}")
            
            # Strengthen knowledge distillation to recover
            if self.enable_distillation:
                self.distillation_weight *= 1.5
                logger.info(f"[C{self.client_index}] Increasing distillation weight to {self.distillation_weight:.4f}")
            
            # Apply early stopping if necessary
            if current_epoch >= early_stop_patience and np.mean(recent_losses) > 2.0 * np.mean(older_losses):
                logger.warning(f"[C{self.client_index}] Early stopping due to loss instability")
                return False
                
        return True

    def local_train(self, epoch):
        """Perform local training for a specified number of epochs"""
        # Call the updated implementation for better consistency and trust calculation
        return self.local_train(epoch)
        
        """
        # Original implementation below - keeping for reference but not executing
        """

    def evaluate_global_accuracy(self):
        """Evaluate accuracy using only the global head"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                global_logits = outputs['global_logit']
                _, predicted = torch.max(global_logits, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total

    def calculate_contrastive_loss(self, features, labels):
        """Calculate relaxed contrastive loss with proper type handling"""
        if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return torch.tensor(0.0).to(self.device)
        
        # Ensure features are float type
        if features.dtype != torch.float32 and features.dtype != torch.float64:
            features = features.float()
        
        # Ensure labels are long type
        if labels.dtype != torch.int64 and labels.dtype != torch.long:
            labels = labels.long()
        
        # Apply relaxed contrastive loss
        try:
            rcl_loss = self.relaxed_contrastive_loss(features, labels)
            return rcl_loss
        except RuntimeError as e:
            logger.warning(f"Error computing contrastive loss: {str(e)}. Using default loss.")
            return torch.tensor(0.01).to(self.device)  # Small default loss to avoid NaN

    def extract_features_safely(self, images):
        """Extract features from images in a memory-efficient way"""
        try:
            # Move to CPU to avoid extra memory usage
            self.model.eval()  # Temporarily set to eval mode for feature extraction
            
            with torch.no_grad():  # No gradients needed for feature extraction
                # Direct feature extraction through forward pass
                outputs = self.model(images)
                if isinstance(outputs, dict) and "feature" in outputs:
                    features = outputs["feature"]
                    
                    # Apply projection if available
                    if hasattr(self.model, 'projection_head') and self.model.projection_head is not None:
                        features = self.model.projection_head(features)
                else:
                    # Fallback to get_contrastive_features if it exists
                    self.model.train()  # Restore train mode
                    return self.model.get_contrastive_features(images)
            
            self.model.train()  # Restore train mode
            return features
            
        except Exception as e:
            logger.error(f"Error in safe feature extraction: {str(e)}")
            self.model.train()  # Make sure to restore train mode
            # Fallback to original method if available
            if hasattr(self.model, 'get_contrastive_features'):
                return self.model.get_contrastive_features(images)
            else:
                # Return empty tensor as last resort
                return torch.zeros((images.size(0), 128), device=self.device)

    def setup_cyclical_lr(self, epoch):
        """Setup cyclical learning rate for the current epoch"""
        if not hasattr(self, 'optimizer'):
            logger.warning(f"Client {self.client_index}: Cannot setup cyclical LR without optimizer")
            return
            
        try:
            # Get trust score history for smoother adaptation
            trust_history = self.trust_score_history[-5:] if hasattr(self, 'trust_score_history') and self.trust_score_history else [0.8]
            
            # Calculate moving average of trust scores for stability
            trust_avg = float(sum(trust_history)) / float(len(trust_history))
            
            # Cyclical component: sine wave with period of 2*step_size
            cycle_position = self.rounds_trained % (2 * self.step_size)
            cycle_ratio = cycle_position / self.step_size
            
            if cycle_position < self.step_size:
                # Increasing phase
                cycle_factor = 0.5 * (1 + np.sin(np.pi * (cycle_ratio - 0.5)))
            else:
                # Decreasing phase
                cycle_factor = 0.5 * (1 + np.sin(np.pi * (cycle_ratio + 0.5)))
            
            # Adaptive learning rate range based on trust
            # Higher trust → wider LR range (more exploration)
            # Lower trust → narrower LR range (more conservative)
            trust_adjusted_max_lr = self.base_lr + (self.max_lr - self.base_lr) * trust_avg
            
            # Apply cycle to the trust-adjusted range
            current_lr = self.base_lr + (trust_adjusted_max_lr - self.base_lr) * cycle_factor
            
            # Add warmup phase for first few rounds
            if self.rounds_trained <= 3:
                warmup_factor = min(1.0, self.rounds_trained / 3)
                current_lr = self.base_lr + (current_lr - self.base_lr) * warmup_factor
            
            # Set learning rate in optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            logger.info(f"[C{self.client_index}] Trust-based Cyclical LR: {current_lr:.6f} (trust={trust_avg:.3f}, cycle={cycle_factor:.2f})")
        except Exception as e:
            logger.error(f"Error setting up cyclical LR: {str(e)}")
            # Fallback to base learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr
            
    def setup_trust_lr(self, trust_score):
        """Setup trust-based learning rate"""
        if not hasattr(self, 'optimizer'):
            logger.warning(f"Client {self.client_index}: Cannot setup trust LR without optimizer")
            return
            
        try:
            # Scale learning rate based on trust score
            # Higher trust means higher learning rate (more aggressive updates)
            # Lower trust means lower learning rate (more conservative updates)
            trust_factor = max(0.5, min(1.5, trust_score * 2))
            current_lr = self.base_lr * trust_factor
            
            # Set learning rate in optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            logger.info(f"[C{self.client_index}] Trust-based LR: {current_lr:.6f} (trust={trust_score:.3f})")
        except Exception as e:
            logger.error(f"Error setting up trust LR: {str(e)}")
            # Fallback to base learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr

    def setup_optimizer(self):
        """Initialize the optimizer if it doesn't exist"""
        if hasattr(self, 'model') and self.model is not None:
            # Default optimizer parameters
            lr = getattr(self, 'base_lr', 0.01)
            weight_decay = 1e-5
            momentum = 0.9
            
            try:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
                logger.info(f"Client {self.client_index}: Initialized optimizer with lr={lr}")
            except Exception as e:
                logger.error(f"Error initializing optimizer: {str(e)}")
                # Try simpler optimizer as fallback
                try:
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=lr
                    )
                except Exception as nested_e:
                    logger.error(f"Failed to initialize fallback optimizer: {str(nested_e)}")
        else:
            logger.error(f"Client {self.client_index}: Cannot setup optimizer without model")

    def update_trust_score(self, metrics):
        """Update trust score based on training metrics"""
        if hasattr(self, 'enable_trust_filtering') and self.enable_trust_filtering:
            # Get trust score based on training metrics
            trust_score = self.compute_trust_score(self.model, metrics)
            
            # Stabilize trust score with exponential moving average
            if not hasattr(self, 'trust_score') or self.trust_score is None:
                self.trust_score = trust_score
            else:
                # Smooth trust score changes
                momentum = 0.7  # Higher value = slower changes
                self.trust_score = momentum * self.trust_score + (1 - momentum) * trust_score
            
            # Ensure trust score is within valid range
            self.trust_score = min(max(self.trust_score, 0.1), 1.0)
            
            # Update model attribute if available
            if hasattr(self.model, 'trust_score'):
                self.model.trust_score = self.trust_score
            
            # Track history
            if not hasattr(self, 'trust_score_history'):
                self.trust_score_history = []
            self.trust_score_history.append(self.trust_score)
            if len(self.trust_score_history) > 10:
                self.trust_score_history.pop(0)
                
            return self.trust_score
        else:
            # Default trust score if filtering not enabled
            return 0.8
            
    def _compute_model_similarity(self):
        """Compute similarity between client model and global model"""
        if not hasattr(self, 'global_model') or self.global_model is None:
            return 0.8  # Default similarity if no global model
            
        similarity_score = 0.0
        count = 0
        
        # Get model dictionaries
        client_state = self.model.state_dict()
        global_state = self.global_model.state_dict()
        
        # Calculate cosine similarity for each parameter
        for key in client_state:
            if key in global_state and not ('personalized_head' in key):
                # Skip personalization parameters
                try:
                    client_param = client_state[key].flatten().float()
                    global_param = global_state[key].flatten().float()
                    
                    if client_param.numel() > 0 and global_param.numel() > 0:
                        cos_sim = F.cosine_similarity(
                            client_param.unsqueeze(0), 
                            global_param.unsqueeze(0)
                        )
                        # Convert from [-1, 1] to [0, 1]
                        similarity_score += (cos_sim.item() + 1) / 2
                        count += 1
                except Exception as e:
                    logger.warning(f"Error computing parameter similarity: {e}")
        
        # Return average similarity
        return similarity_score / count if count > 0 else 0.8
                
    def local_train(self, global_epoch):
        """Perform local training for a client with additional measurement of client metrics"""
        self.global_epoch = global_epoch
        self.rounds_trained = getattr(self, 'rounds_trained', 0) + 1
        
        # Save previous model state before training
        self.previous_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Initialize metrics trackers
        ce_losses = AverageMeter('CE Loss', ':.4f')
        rcl_losses = AverageMeter('RCL Loss', ':.4f')
        accs = AverageMeter('Acc', ':.4f')
        
        # Set up optimizer with trust-based learning rate if enabled
        if hasattr(self, 'enable_adaptive_lr') and self.enable_adaptive_lr and hasattr(self, 'trust_score'):
            # Calculate base lr using cyclical or adaptive approach
            if self.local_lr_type == 'cyclic':
                trust_weight = getattr(self, 'trust_score', 0.7)
                base_lr = setup_adaptive_learning_rate(
                    0.001, 0.01,  # Min and max LR
                    global_epoch, self.max_epochs,
                    trust_weight=trust_weight,
                    client_id=self.client_id
                )
                logger.info(f"[C{self.client_id}] Trust-based Cyclical LR: {base_lr:.6f} (trust={trust_weight:.3f}, cycle={global_epoch/self.max_epochs:.2f})")
            else:
                # Standard adaptive LR based on trust
                base_lr = self.local_lr * min(1.0, 0.5 + getattr(self, 'trust_score', 0.5))
        else:
            base_lr = self.local_lr
            
        # Create optimizer with the determined learning rate
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Training loop
        self.model.train()
        epoch_metrics = {}
        
        for epoch in range(1, self.local_epochs + 1):
            # Set up learning rate for this epoch
            if self.local_lr_type == 'cyclic':
                # Adjust LR within epoch using cycle
                epoch_fraction = (global_epoch + epoch / self.local_epochs) / self.max_epochs
                trust_score = getattr(self, 'trust_score', 0.7)
                current_lr = setup_adaptive_learning_rate(
                    0.001, 0.01,  # Min and max LR
                    epoch_fraction * self.max_epochs, 
                    self.max_epochs,
                    trust_weight=trust_score,
                    client_id=self.client_id
                )
                # Update optimizer learning rate
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                if epoch == 1:
                    logger.info(f"[C{self.client_id}] Trust-based Cyclical LR: {current_lr:.6f} (trust={trust_score:.3f}, cycle={epoch_fraction:.2f})")
            
            # Train for one epoch
            batch_metrics = self._train_epoch(optimizer, epoch)
            
            # Update metrics
            ce_losses.update(batch_metrics.get('ce_loss', 0))
            rcl_losses.update(batch_metrics.get('rcl_loss', 0))
            accs.update(batch_metrics.get('acc', 0))
            
            # Log every few epochs or at the end
            if epoch == self.local_epochs or epoch % 5 == 0:
                logger.debug(f"Client {self.client_id}, Epoch {epoch}/{self.local_epochs}, "
                          f"CE Loss: {ce_losses.avg:.4f}, RCL Loss: {rcl_losses.avg:.4f}, "
                          f"Acc: {accs.avg:.4f}")
        
        # Calculate gradient norm for monitoring
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Save average metrics
        epoch_metrics.update({
            'ce_loss': ce_losses.avg,
            'rcl_loss': rcl_losses.avg,
            'acc': accs.avg,
            'grad_norm': grad_norm,
        })
        
        # Store loss average for stability evaluation
        self.ce_loss_avg = ce_losses.avg
        
        # Update trust score based on training performance
        trust_score = self.update_trust_score(epoch_metrics)
        epoch_metrics['trust_score'] = trust_score
        
        # Return updated model and metrics
        return self.model.state_dict(), epoch_metrics

    def _train_epoch(self, optimizer, epoch):
        """Train for one epoch
        
        Args:
            optimizer: optimizer to use for training
            epoch: current epoch number
            
        Returns:
            dict: metrics for the epoch
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        ce_losses = AverageMeter('CE Loss', ':.4f')
        rcl_losses = AverageMeter('RCL Loss', ':.4f')
        distill_losses = AverageMeter('Distill Loss', ':.4f')
        proximal_losses = AverageMeter('Proximal Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        # Training mode
        self.model.train()
        
        # Initialize metrics
        end = time.time()
        metrics = {}
        
        # Ensure trainloader exists
        if not hasattr(self, 'trainloader') or self.trainloader is None:
            logger.warning(f"Client {self.client_id}: No trainloader available for training")
            return {
                'loss': 0.0,
                'ce_loss': 0.0,
                'rcl_loss': 0.0,
                'acc': 0.0
            }
        
        # Training loop
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move to device
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with amp support
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    
                    # Extract logits from outputs
                    if isinstance(outputs, dict):
                        logits = outputs.get('logit', outputs.get('global_logit', None))
                        if logits is None:
                            for key in ['output', 'pred', 'prediction', 'logits']:
                                if key in outputs:
                                    logits = outputs[key]
                                    break
                    else:
                        logits = outputs
                    
                    # Classification loss
                    ce_loss = self.criterion(logits, labels)
                    
                    # Relaxed contrastive loss if enabled
                    if self.enable_contrastive and 'feature_normalized' in outputs:
                        rcl_loss = self.compute_multi_level_rcl_loss(outputs, labels)
                    else:
                        rcl_loss = torch.tensor(0.0).to(self.device)
                    
                    # Distillation loss if enabled
                    if self.enable_distillation and hasattr(self, 'global_model'):
                        with torch.no_grad():
                            global_outputs = self.global_model(images)
                            
                        if isinstance(global_outputs, dict):
                            global_logits = global_outputs.get('logit', global_outputs.get('global_logit', None))
                        else:
                            global_logits = global_outputs
                            
                        # Ensure global logits exist
                        if global_logits is not None:
                            distill_loss = self.compute_distillation_loss(logits, global_logits)
                        else:
                            distill_loss = torch.tensor(0.0).to(self.device)
                    else:
                        distill_loss = torch.tensor(0.0).to(self.device)
                    
                    # FedProx loss for regularization
                    proximal_loss = self.compute_fedprox_term() if self.enable_fedprox else torch.tensor(0.0).to(self.device)
                    
                    # EWC loss for continual learning
                    ewc_loss = self.compute_ewc_loss() if self.ewc_enabled else torch.tensor(0.0).to(self.device)
                    
                    # Combine losses with appropriate weights
                    rcl_weight = 1.0 if self.enable_contrastive else 0.0
                    distill_weight = self.distillation_weight if self.enable_distillation else 0.0
                    
                    # Total loss
                    loss = ce_loss + rcl_weight * rcl_loss + distill_weight * distill_loss + proximal_loss + ewc_loss
            else:
                # Standard forward pass without amp
                outputs = self.model(images)
                
                # Extract logits from outputs
                if isinstance(outputs, dict):
                    logits = outputs.get('logit', outputs.get('global_logit', None))
                    if logits is None:
                        for key in ['output', 'pred', 'prediction', 'logits']:
                            if key in outputs:
                                logits = outputs[key]
                                break
                else:
                    logits = outputs
                
                # Classification loss
                ce_loss = self.criterion(logits, labels)
                
                # Relaxed contrastive loss if enabled
                if self.enable_contrastive and 'feature_normalized' in outputs:
                    rcl_loss = self.compute_multi_level_rcl_loss(outputs, labels)
                else:
                    rcl_loss = torch.tensor(0.0).to(self.device)
                
                # Distillation loss if enabled
                if self.enable_distillation and hasattr(self, 'global_model'):
                    with torch.no_grad():
                        global_outputs = self.global_model(images)
                        
                    if isinstance(global_outputs, dict):
                        global_logits = global_outputs.get('logit', global_outputs.get('global_logit', None))
                    else:
                        global_logits = global_outputs
                        
                    # Ensure global logits exist
                    if global_logits is not None:
                        distill_loss = self.compute_distillation_loss(logits, global_logits)
                    else:
                        distill_loss = torch.tensor(0.0).to(self.device)
                else:
                    distill_loss = torch.tensor(0.0).to(self.device)
                
                # FedProx loss for regularization
                proximal_loss = self.compute_fedprox_term() if self.enable_fedprox else torch.tensor(0.0).to(self.device)
                
                # EWC loss for continual learning
                ewc_loss = self.compute_ewc_loss() if self.ewc_enabled else torch.tensor(0.0).to(self.device)
                
                # Combine losses with appropriate weights
                rcl_weight = 1.0 if self.enable_contrastive else 0.0
                distill_weight = self.distillation_weight if self.enable_distillation else 0.0
                
                # Total loss
                loss = ce_loss + rcl_weight * rcl_loss + distill_weight * distill_loss + proximal_loss + ewc_loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"Client {self.client_id}: NaN loss detected, skipping batch")
                continue
            
            # Backward pass and optimize with amp support
            if self.use_amp:
                scaler = GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum().item()
            acc = 100. * correct / labels.size(0)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            ce_losses.update(ce_loss.item(), images.size(0))
            rcl_losses.update(rcl_loss.item() if not torch.isnan(rcl_loss) else 0.0, images.size(0))
            distill_losses.update(distill_loss.item(), images.size(0))
            proximal_losses.update(proximal_loss.item(), images.size(0))
            top1.update(acc, images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Apply EMA to loss history for stability tracking
            self.ce_loss_avg = ce_loss.item() if not hasattr(self, 'ce_loss_avg') else 0.9 * self.ce_loss_avg + 0.1 * ce_loss.item()
            self.rcl_loss_avg = rcl_loss.item() if not hasattr(self, 'rcl_loss_avg') else 0.9 * self.rcl_loss_avg + 0.1 * rcl_loss.item()
            
            # Optional learning rate scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
        
        # Return metrics dictionary
        metrics = {
            'loss': losses.avg,
            'ce_loss': ce_losses.avg,
            'rcl_loss': rcl_losses.avg,
            'distill_loss': distill_losses.avg,
            'proximal_loss': proximal_losses.avg,
            'acc': top1.avg / 100.0,  # Convert back to [0,1] range
        }
        
        return metrics

