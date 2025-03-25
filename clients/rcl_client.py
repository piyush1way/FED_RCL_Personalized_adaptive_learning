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
        self.loader = None

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
        self.base_lr = getattr(args.client, "base_lr", 0.001)
        self.max_lr = getattr(args.client, "max_lr", 0.1)
        self.step_size = getattr(args.client, "step_size", 10)
        
        # Regularization options
        self.enable_fedprox = getattr(args.client, "fedprox", True)
        self.fedprox_mu = getattr(args.client, "fedprox_mu", 0.005)

        # Knowledge distillation for personalized heads
        self.enable_distillation = getattr(args.client, "distillation", True)
        self.distillation_temp = getattr(args.client, "distillation_temp", 3.0)
        self.distillation_weight = getattr(args.client, "distillation_weight", 0.7)

        # Model setup
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.device = torch.device("cpu")

        # Multi-level contrastive learning setup
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
        trust_score = self.compute_trust_score() if self.enable_trust_filtering else 0.8
        
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

    def compute_trust_score(self):
        """Compute trust score based on multiple metrics"""
        trust_score = 0.8  # Default trust score
        
        # Only compute trust if we have previous model state and current model
        if self.previous_model_state is not None and hasattr(self.model, 'state_dict'):
            # 1. Gradient consistency with previous updates
            if len(self.gradient_history) > 0:
                current_state = self.model.state_dict()
                current_grads = {}
                
                # Compute current gradient
                for key in current_state:
                    if key in self.previous_model_state:
                        # Skip personalized layers in gradient consistency check
                        if 'personalized_head' in key:
                            continue
                        current_grads[key] = current_state[key] - self.previous_model_state[key]
                
                # Compare with gradient history
                grad_sim_scores = []
                for past_grad in self.gradient_history[-3:]:  # Compare with last 3 gradients
                    sim_score = 0.0
                    num_layers = 0
                    
                    for key in current_grads:
                        if key in past_grad:
                            flat_current = current_grads[key].flatten()
                            flat_past = past_grad[key].flatten()
                            
                            # Compute cosine similarity if tensors are not empty
                            if flat_current.shape[0] > 0 and flat_past.shape[0] > 0:
                                # Convert tensors to float for cosine similarity
                                flat_current = flat_current.float()
                                flat_past = flat_past.float()
                                
                                cos_sim = F.cosine_similarity(flat_current.unsqueeze(0), flat_past.unsqueeze(0))
                                # Convert from [-1, 1] to [0, 1] range
                                sim_score += (cos_sim + 1) / 2
                                num_layers += 1
                
                if num_layers > 0:
                    avg_sim = sim_score / num_layers
                    grad_sim_scores.append(avg_sim.item())
                
                # Update gradient history
                if len(current_grads) > 0:
                    # Limit history size
                    if len(self.gradient_history) >= 5:
                        self.gradient_history.pop(0)
                    self.gradient_history.append(current_grads)
                
                # Compute final gradient consistency score
                if len(grad_sim_scores) > 0:
                    grad_consistency = sum(grad_sim_scores) / len(grad_sim_scores)
                    # Low consistency should not completely zero out trust,
                    # so we scale from 0.3 to 1.0
                    grad_trust = 0.3 + 0.7 * grad_consistency
                else:
                    grad_trust = 0.8  # Default if no history
            else:
                # First round, initialize gradient history
                current_state = self.model.state_dict()
                current_grads = {}
                
                for key in current_state:
                    if key in self.previous_model_state:
                        current_grads[key] = current_state[key] - self.previous_model_state[key]
                
                if len(current_grads) > 0:
                    self.gradient_history.append(current_grads)
                
                grad_trust = 0.8  # Default for first round
            
            # 2. Model agreement with global model
            if hasattr(self, 'local_dataset') and hasattr(self, 'global_model'):
                # Evaluate on a small subset of local data
                model_agreement = self.evaluate_model_agreement()
                
                # Scale from 0.4 to 1.0 (even low agreement should have some trust)
                model_trust = 0.4 + 0.6 * model_agreement
            else:
                model_trust = 0.8  # Default if can't evaluate
            
            # 3. Consider training stability
            stability_trust = 1.0
            if hasattr(self, 'ce_loss_avg') and self.ce_loss_avg > 0:
                # High loss may indicate unstable training
                stability_trust = min(1.0, 2.0 / (1.0 + self.ce_loss_avg))
            
            # Combine trust factors with appropriate weights
            # Gradient consistency is important but should not dominate
            trust_score = 0.4 * grad_trust + 0.4 * model_trust + 0.2 * stability_trust
            
            # Add slight trust increase for older clients to avoid cold start issues
            rounds_bonus = min(0.1, 0.01 * self.rounds_trained)
            trust_score = min(1.0, trust_score + rounds_bonus)
            
            # Track trust score history
            self.trust_score_history.append(trust_score)
            if len(self.trust_score_history) > 10:
                self.trust_score_history.pop(0)
            
            # Update model's trust score attribute if it exists
            if hasattr(self.model, 'trust_score'):
                self.model.trust_score = trust_score
        
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

    def local_train(self, round_idx):
        """Train the model locally with improved balance between global and personalized objectives"""
        self.model.train()
        
        # Initialize tracking variables
        global_losses = []
        personalized_losses = []
        contrastive_losses = []
        
        # Get initial global model state for distillation
        global_model_state = copy.deepcopy(self.model.state_dict())
        
        # Training loop
        for epoch in range(self.args.trainer.local_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                features = outputs['feature']
                global_logits = outputs['global_logit']
                personalized_logits = outputs['personalized_logit']
                
                # Calculate losses
                # 1. Global classification loss
                global_loss = self.criterion(global_logits, labels)
                
                # 2. Personalized classification loss
                personalized_loss = self.criterion(personalized_logits, labels)
                
                # 3. Knowledge distillation loss
                if self.enable_distillation:
                    with torch.no_grad():
                        global_model = copy.deepcopy(self.model)
                        global_model.load_state_dict(global_model_state)
                        global_model.eval()
                        teacher_outputs = global_model(images)
                        teacher_logits = teacher_outputs['global_logit']
                    
                    distill_temp = self.distillation_temp
                    soft_targets = F.softmax(teacher_logits / distill_temp, dim=1)
                    distill_loss = F.kl_div(
                        F.log_softmax(personalized_logits / distill_temp, dim=1),
                        soft_targets,
                        reduction='batchmean'
                    ) * (distill_temp ** 2)
                else:
                    distill_loss = torch.tensor(0.0).to(self.device)
                
                # 4. Contrastive loss for feature alignment
                if hasattr(self.model, 'get_contrastive_features'):
                    proj_features = self.model.get_contrastive_features(images)
                    contrastive_loss = self.calculate_contrastive_loss(proj_features, labels)
                else:
                    contrastive_loss = torch.tensor(0.0).to(self.device)
                
                # Combine losses with dynamic weighting
                trust_score = getattr(self.model, 'trust_score', 0.8)  # Default to 0.8 if not set
                global_weight = max(0.3, 1.0 - trust_score)  # Ensure minimum global weight
                personalized_weight = trust_score
                
                total_loss = (
                    global_weight * global_loss +
                    personalized_weight * personalized_loss +
                    self.distillation_weight * distill_loss +  # Use client's distillation weight
                    0.1 * contrastive_loss  # Small weight for contrastive loss
                )
                
                # Backward pass and optimize
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # Add gradient clipping
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                _, predicted = torch.max(personalized_logits, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Track individual losses
                global_losses.append(global_loss.item())
                personalized_losses.append(personalized_loss.item())
                contrastive_losses.append(contrastive_loss.item())
            
            # Log epoch metrics
            accuracy = 100. * correct / total
            logger.debug(f'Epoch {epoch}: Loss: {epoch_loss/len(self.trainloader):.3f}, '
                        f'Acc: {accuracy:.2f}%, Trust: {trust_score:.3f}')
            
            # Update trust score based on performance
            if hasattr(self.trainer, 'calculate_trust_score'):
                global_acc = self.evaluate_global_accuracy()
                personalized_acc = accuracy / 100.0  # Convert to decimal
                new_trust = self.trainer.calculate_trust_score(global_acc, personalized_acc, round_idx)
                self.model.trust_score = new_trust
        
        # Create state_dict dictionary to return - only include non-personalized parameters
        if self.enable_personalization and hasattr(self.model, 'get_global_params'):
            return_dict = self.model.get_global_params()
        else:
            return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
                          if 'personalized_head' not in k}
        
        # Create stats dictionary
        stats_dict = {
            'trust_score': getattr(self.model, 'trust_score', 0.8),
            'global_loss': np.mean(global_losses),
            'personalized_loss': np.mean(personalized_losses),
            'contrastive_loss': np.mean(contrastive_losses)
        }
        
        # Return both the model state and stats (as two separate values)
        return return_dict, stats_dict

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

