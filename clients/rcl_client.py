# import copy
# import time
# import gc
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler

# import logging
# from utils import *
# from utils.loss import KL_u_p_loss, RelaxedContrastiveLoss
# from utils.metrics import evaluate
# from models import build_encoder
# from utils.logging_utils import AverageMeter
# from clients.build import CLIENT_REGISTRY
# from clients import Client

# logger = logging.getLogger(__name__)

# @CLIENT_REGISTRY.register()
# class RCLClient(Client):
#     def __init__(self, args, client_index, model):
#         self.args = args
#         self.client_index = client_index
#         self.loader = None

#         # Trust-based client filtering configuration
#         trust_filtering_config = getattr(args.client, "trust_filtering", {})
#         self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
#         self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
#         self.trust_score_history = []
        
#         # Client state tracking
#         self.rounds_trained = 0
#         self.previous_model_state = None
#         self.update_history = []
#         self.gradient_history = []

#         # Personalization configuration
#         personalization_config = getattr(args.client, "personalization", {})
#         self.enable_personalization = getattr(personalization_config, "enable", False)
#         self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
#         self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)
#         self.freeze_ratio = getattr(personalization_config, "freeze_ratio", 0.5)

#         # Trust-based adaptive learning rate
#         self.enable_cyclical_lr = getattr(args.client, "cyclical_lr", True)
#         self.base_lr = getattr(args.client, "base_lr", 0.001)
#         self.max_lr = getattr(args.client, "max_lr", 0.1)
#         self.step_size = getattr(args.client, "step_size", 10)
        
#         # Regularization options
#         self.enable_fedprox = getattr(args.client, "fedprox", True)
#         self.fedprox_mu = getattr(args.client, "fedprox_mu", 0.005)

#         # Knowledge distillation for personalized heads
#         self.enable_distillation = getattr(args.client, "distillation", True)
#         self.distillation_temp = getattr(args.client, "distillation_temp", 3.0)
#         self.distillation_weight = getattr(args.client, "distillation_weight", 0.7)

#         # Model setup
#         self.model = model
#         self.global_model = copy.deepcopy(model)
#         self.device = torch.device("cpu")

#         # Multi-level contrastive learning setup
#         self.rcl_criterions = {'scl': None, 'penalty': None}
#         args_rcl = getattr(args.client, "rcl_loss", None)
#         if args_rcl:
#             self.pairs = {}
#             for pair in args_rcl.pairs:
#                 self.pairs[pair.name] = pair
#                 self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
#         else:
#             self.pairs = {}
            
#         self.global_epoch = 0
#         self.criterion = nn.CrossEntropyLoss()
        
#         # Relaxed contrastive loss with divergence penalty
#         self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
#             temperature=getattr(args.client, "temperature", 0.05),
#             beta=getattr(args.client, "beta", 1.0),
#             lambda_threshold=getattr(args.client, "lambda_threshold", 0.7)
#         )
        
#         # Loss tracking
#         self.ce_loss_avg = 0.0
#         self.rcl_loss_avg = 0.0
#         self.distillation_loss_avg = 0.0
#         self.fedprox_loss_avg = 0.0
        
#         # EWC parameters for continual learning
#         self.ewc_importance = getattr(args.client, "ewc_importance", 5000)
#         self.ewc_enabled = getattr(args.client, "ewc_enabled", True)
#         self.ewc_lambda = getattr(args.client, "ewc_lambda", 0.4)
#         self.fisher_information = None
#         self.optimal_parameters = None
        
#         # Multi-level contrastive learning configuration
#         self.multi_level_rcl = getattr(args.client, "multi_level_rcl", True)
#         self.layer_weights = getattr(args.client, "layer_weights", [0.2, 0.2, 0.2, 0.2, 0.2])

#     def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
#         self.device = device
#         self.rounds_trained += 1
#         self.trainer = trainer  # Store trainer reference
        
#         # Save previous model state for trust score calculation
#         if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
#             self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
#         # Save optimal parameters for EWC
#         if self.ewc_enabled and self.fisher_information is not None:
#             self.optimal_parameters = {k: v.clone().detach() for k, v in self.model.state_dict().items() 
#                                       if 'personalized_head' not in k}
        
#         self._update_model(state_dict)
#         self._update_global_model(state_dict)
        
#         self.model.to(self.device)
#         self.global_model.to(self.device)

#         # Freeze global model for distillation
#         for param in self.global_model.parameters():
#             param.requires_grad = False

#         # Configure personalization if enabled
#         if self.enable_personalization:
#             if isinstance(self.model, torch.nn.DataParallel):
#                 model_module = self.model.module
#             else:
#                 model_module = self.model
                
#             if hasattr(model_module, 'enable_personalized_mode'):
#                 model_module.enable_personalized_mode()
#             elif hasattr(model_module, 'use_personalized_head'):
#                 model_module.use_personalized_head = True
                
#             # Freeze backbone for faster personalization
#             if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
#                 if self.rounds_trained < 3:
#                     model_module.unfreeze_backbone()
#                 else:
#                     model_module.freeze_backbone()
                
#             # Adaptive layer freezing based on training progress
#             if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
#                 freeze_ratio = max(0.0, self.freeze_ratio - (self.rounds_trained / 50.0))
#                 model_module.setup_adaptive_freezing(freeze_ratio=freeze_ratio)
        
#         # Setup data loader
#         self.local_dataset = local_dataset
#         if hasattr(self.local_dataset, 'x'):
#             if len(self.local_dataset.x) < 2:  # Not enough data for training
#                 self.trainloader = None
#                 logger.warning(f"Client {self.client_index} has insufficient data ({len(self.local_dataset.x)} samples)")
#                 return
        
#         self.trainloader = DataLoader(
#             self.local_dataset,
#             batch_size=kwargs.get('local_bs', 32),
#             shuffle=True,
#             num_workers=kwargs.get('num_workers', 4),
#             pin_memory=kwargs.get('pin_memory', True),
#             drop_last=True
#         )
        
#         # Save a sample batch for EWC if enabled
#         if self.ewc_enabled and hasattr(self, 'trainloader') and self.trainloader is not None:
#             first_batch = next(iter(self.trainloader), None)
#             if first_batch is not None:
#                 images, labels = first_batch
#                 if len(images) > 1:
#                     self.ewc_batch = (images.to(self.device), labels.to(self.device))
        
#         # Calculate trust score for learning rate adjustment if enabled
#         trust_score = self.compute_trust_score() if self.enable_trust_filtering else 0.8
        
#         # Setup optimizer with trust-based adaptive learning rate
#         if self.enable_cyclical_lr:
#             # Adapt learning rate based on trust score if available
#             if hasattr(self, 'trust_score_history') and len(self.trust_score_history) > 0:
#                 # Use trust score to adjust learning rate
#                 trust_avg = np.mean(self.trust_score_history[-3:]) if len(self.trust_score_history) >= 3 else trust_score
                
#                 # Base learning rate on trust score - higher trust = higher LR
#                 trust_adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * trust_avg
                
#                 # Apply cyclical pattern based on rounds trained
#                 cycle = np.sin(np.pi * (self.rounds_trained % self.step_size) / self.step_size)
#                 cycle_factor = 0.5 * (1 + cycle)
                
#                 # Combine trust adjustment with cycle
#                 current_lr = self.base_lr + (trust_adjusted_lr - self.base_lr) * cycle_factor
                
#                 logger.info(f"[C{self.client_index}] Trust-adjusted LR: {current_lr:.6f} (trust={trust_avg:.3f}, cycle={cycle_factor:.2f})")
#             else:
#                 # Default cyclical LR if no trust score history
#                 cycle = np.sin(np.pi * (self.rounds_trained % self.step_size) / self.step_size)
#                 cycle_factor = 0.5 * (1 + cycle)
#                 current_lr = self.base_lr + (self.max_lr - self.base_lr) * cycle_factor
#                 logger.info(f"[C{self.client_index}] Cyclical LR: {current_lr:.6f} (cycle={cycle_factor:.2f})")
#         else:
#             # No cyclical LR, use constant rate
#             current_lr = local_lr
#             logger.info(f"[C{self.client_index}] Fixed LR: {current_lr:.6f}")
            
#         # Ensure learning rate is reasonable
#         current_lr = max(1e-5, min(current_lr, 0.1))
        
#         # Special case for early rounds to prevent cold start
#         if self.rounds_trained <= 2:
#             current_lr = min(current_lr * 1.5, 0.1)  # Slightly higher LR at the start

#         self.local_epochs = kwargs.get('local_ep', 5)
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=current_lr,
#             momentum=0.9,
#             weight_decay=kwargs.get('weight_decay', 1e-5)
#         )
        
#         # Reset scheduler for each round of training
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer, 
#             T_max=self.local_epochs
#         )

#     def _update_model(self, state_dict):
#         """Update local model with server state dict"""
#         if state_dict is not None:
#             if isinstance(self.model, torch.nn.DataParallel):
#                 self.model.module.load_state_dict(state_dict, strict=False)
#             else:
#                 self.model.load_state_dict(state_dict, strict=False)

#     def _update_global_model(self, state_dict):
#         """Update global model copy with server state dict"""
#         if state_dict is not None:
#             if isinstance(self.global_model, torch.nn.DataParallel):
#                 self.global_model.module.load_state_dict(state_dict, strict=False)
#             else:
#                 self.global_model.load_state_dict(state_dict, strict=False)

#     def compute_fedprox_term(self):
#         """Compute FedProx regularization term"""
#         proximal_term = 0.0
#         for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
#             proximal_term += (w - w_t).norm(2)**2
#         return self.fedprox_mu * proximal_term / 2

#     def compute_distillation_loss(self, student_logits, teacher_logits, features=None, teacher_features=None):
#         """
#         Compute distillation loss between teacher and student models.
#         Improved to prevent overfitting to the teacher model.
#         """
#         # Logit-based distillation with temperature scaling
#         temp = self.distillation_temp
#         soft_targets = F.softmax(teacher_logits / temp, dim=1)
#         log_probs = F.log_softmax(student_logits / temp, dim=1)
        
#         # Apply temperature scaling and normalize
#         kd_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temp * temp)
        
#         # Feature distillation (optional)
#         feature_loss = 0.0
#         if features is not None and teacher_features is not None:
#             # Normalize features
#             student_features = F.normalize(features, p=2, dim=1)
#             teacher_features = F.normalize(teacher_features, p=2, dim=1)
            
#             # Apply L2 loss with a smaller weight for feature distillation
#             feature_loss = F.mse_loss(student_features, teacher_features) * 0.1
        
#         # Combine losses with a safeguard to prevent overfitting to teacher
#         # Dynamically reduce distillation impact as training progresses
#         distillation_weight_factor = max(0.2, 1.0 - 0.05 * self.rounds_trained)
#         combined_loss = (kd_loss + feature_loss) * distillation_weight_factor
        
#         # Add a safeguard against excessive distillation
#         if self.rounds_trained > 10 and hasattr(self, 'distillation_loss_avg') and self.distillation_loss_avg > 0:
#             if combined_loss > 2.0 * self.distillation_loss_avg:
#                 combined_loss = self.distillation_loss_avg
#                 logger.warning(f"[C{self.client_index}] Limiting excessive distillation loss")
                
#         return combined_loss

#     def compute_ewc_loss(self):
#         """Compute EWC regularization loss with device consistency"""
#         if not self.ewc_enabled or not self.fisher_information:
#             return torch.tensor(0.0, device=self.device)
        
#         loss = torch.tensor(0.0, device=self.device)
#         for name, param in self.model.named_parameters():
#             if name in self.fisher_information:
#                 # Ensure all tensors are on the same device
#                 fisher = self.fisher_information[name].to(param.device)
#                 optimal_param = self.optimal_parameters[name].to(param.device)
#                 loss += (fisher * (param - optimal_param).pow(2)).sum()
        
#         return self.ewc_lambda * loss

#     def compute_trust_score(self):
#         """Compute trust score based on update magnitude and consistency"""
#         if self.previous_model_state is None:
#             return 0.8  # Initialize with a neutral score instead of 1.0
        
#         current_state = self.model.state_dict()
        
#         update_norm = 0.0
#         param_count = 0
        
#         for key in current_state:
#             if 'personalized_head' not in key and key in self.previous_model_state:
#                 current_param = current_state[key].to(self.device).float()
#                 prev_param = self.previous_model_state[key].to(self.device).float()
#                 diff = current_param - prev_param
#                 update_norm += torch.norm(diff).item() ** 2
#                 param_count += diff.numel()
        
#         if param_count > 0:
#             update_norm = (update_norm / param_count) ** 0.5
        
#         # Ensure we have a more dynamic update history
#         self.update_history.append(update_norm)
#         if len(self.update_history) > 5:
#             self.update_history.pop(0)
            
#         # Add small epsilon to prevent division by zero
#         update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
#         # Compute trust score components with improved scaling
#         magnitude_score = 1.0 / (1.0 + 10.0 * update_norm)  # Adjust sensitivity
#         consistency_score = 1.0 / (1.0 + 5.0 * update_variance)  # Adjust sensitivity
        
#         # Dynamic weighting based on training progress
#         mag_weight = max(0.5, min(0.8, 0.8 - 0.01 * self.rounds_trained))  # Reduce weight over time
#         cons_weight = 1.0 - mag_weight
        
#         # Combine scores with dynamic weights
#         trust_score = mag_weight * magnitude_score + cons_weight * consistency_score
        
#         # Add random noise to break any potential plateaus
#         trust_score += np.random.normal(0, 0.02)  # Small random noise
        
#         # Save current state for next round
#         self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
#         # Add to history for tracking
#         self.trust_score_history.append(trust_score)
#         if len(self.trust_score_history) > 10:
#             self.trust_score_history.pop(0)
        
#         # Ensure the trust score is within [0.1, 1.0] to avoid getting stuck
#         return max(0.1, min(1.0, trust_score))

#     def compute_fisher_information(self):
#         """Compute Fisher Information Matrix for EWC regularization"""
#         if not self.ewc_enabled or self.trainloader is None:
#             return
                
#         fisher_information = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and 'personalized_head' not in name:
#                 fisher_information[name] = torch.zeros_like(param)
                
#         self.model.eval()
#         for images, labels in self.trainloader:
#             images = images.to(self.device)
#             labels = labels.to(self.device)
            
#             self.model.zero_grad()
#             output = self.model(images)
#             if isinstance(output, dict):
#                 logits = output["logit"]
#             else:
#                 logits = output
                
#             log_probs = F.log_softmax(logits, dim=1)
#             samples = torch.multinomial(torch.exp(log_probs), 1).squeeze()
#             loss = F.nll_loss(log_probs, samples)
#             loss.backward()
            
#             for name, param in self.model.named_parameters():
#                 if param.requires_grad and param.grad is not None and 'personalized_head' not in name:
#                     fisher_information[name] += param.grad.pow(2).detach()
                    
#         for name in fisher_information:
#             fisher_information[name] /= len(self.trainloader) if len(self.trainloader) > 0 else 1.0
                
#         self.fisher_information = fisher_information
#         self.optimal_parameters = {name: param.clone().detach() for name, param in self.model.named_parameters() 
#                                   if name in fisher_information}
#         self.model.train()
    
#     def compute_multi_level_rcl_loss(self, output, labels):
#         """Compute RCL loss across multiple feature levels"""
#         if not self.multi_level_rcl or not isinstance(output, dict):
#             return 0.0
                
#         total_loss = 0.0
        
#         # Get features from different layers
#         layer_features = []
#         for i in range(5):  # Assuming 5 layers (0-4)
#             layer_key = f"layer{i}"
#             if layer_key in output:
#                 layer_features.append(output[layer_key])
                
#         # If no layer features found, return 0
#         if not layer_features:
#             return 0.0
                
#         # Apply RCL to each layer with weights
#         for i, features in enumerate(layer_features):
#             if i < len(self.layer_weights):
#                 weight = self.layer_weights[i]
#                 # Reshape features if needed
#                 if len(features.shape) > 2:
#                     # Global average pooling for conv features
#                     features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
                
#                 # Apply L2 normalization
#                 features = F.normalize(features, p=2, dim=1)
                
#                 # Compute RCL loss for this layer
#                 layer_loss = self.relaxed_contrastive_loss(features, labels)
#                 total_loss += weight * layer_loss
                
#         return total_loss
    
#     def detect_and_fix_catastrophic_forgetting(self, loss_value, current_epoch, early_stop_patience=3):
#         """
#         Detect and fix catastrophic forgetting during training.
#         Returns True if training should continue, False if early stopping is triggered.
#         """
#         # Track loss trends to detect sudden increases
#         if not hasattr(self, 'loss_history'):
#             self.loss_history = []
            
#         self.loss_history.append(loss_value)
#         if len(self.loss_history) > 10:
#             self.loss_history.pop(0)
        
#         # Only check after collecting enough history
#         if len(self.loss_history) < 3:
#             return True
            
#         # Calculate moving average and variance
#         recent_losses = self.loss_history[-3:]
#         older_losses = self.loss_history[:-3] if len(self.loss_history) > 3 else []
        
#         # If loss suddenly increases significantly, take action
#         if len(older_losses) > 0 and np.mean(recent_losses) > 1.5 * np.mean(older_losses):
#             logger.warning(f"[C{self.client_index}] Potential catastrophic forgetting detected! Loss spiked to {loss_value:.4f}")
            
#             # Reduce learning rate
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] *= 0.5
#                 logger.info(f"[C{self.client_index}] Reducing learning rate to {param_group['lr']:.6f}")
            
#             # Strengthen knowledge distillation to recover
#             if self.enable_distillation:
#                 self.distillation_weight *= 1.5
#                 logger.info(f"[C{self.client_index}] Increasing distillation weight to {self.distillation_weight:.4f}")
            
#             # Apply early stopping if necessary
#             if current_epoch >= early_stop_patience and np.mean(recent_losses) > 2.0 * np.mean(older_losses):
#                 logger.warning(f"[C{self.client_index}] Early stopping due to loss instability")
#                 return False
                
#         return True

#     def local_train(self, round_idx):
#         """Train the model locally with improved balance between global and personalized objectives"""
#         self.model.train()
        
#         # Initialize tracking variables
#         global_losses = []
#         personalized_losses = []
#         contrastive_losses = []
        
#         # Get initial global model state for distillation
#         global_model_state = copy.deepcopy(self.model.state_dict())
        
#         # Training loop
#         for epoch in range(self.args.trainer.local_epochs):
#             epoch_loss = 0
#             correct = 0
#             total = 0
            
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 self.optimizer.zero_grad()
                
#                 # Forward pass
#                 outputs = self.model(images)
#                 features = outputs['feature']
#                 global_logits = outputs['global_logit']
#                 personalized_logits = outputs['personalized_logit']
                
#                 # Calculate losses
#                 # 1. Global classification loss
#                 global_loss = self.criterion(global_logits, labels)
                
#                 # 2. Personalized classification loss
#                 personalized_loss = self.criterion(personalized_logits, labels)
                
#                 # 3. Knowledge distillation loss
#                 if self.enable_distillation:
#                     with torch.no_grad():
#                         global_model = copy.deepcopy(self.model)
#                         global_model.load_state_dict(global_model_state)
#                         global_model.eval()
#                         teacher_outputs = global_model(images)
#                         teacher_logits = teacher_outputs['global_logit']
                    
#                     distill_temp = self.distillation_temp
#                     soft_targets = F.softmax(teacher_logits / distill_temp, dim=1)
#                     distill_loss = F.kl_div(
#                         F.log_softmax(personalized_logits / distill_temp, dim=1),
#                         soft_targets,
#                         reduction='batchmean'
#                     ) * (distill_temp ** 2)
#                 else:
#                     distill_loss = torch.tensor(0.0).to(self.device)
                
#                 # 4. Contrastive loss for feature alignment
#                 if hasattr(self.model, 'get_contrastive_features'):
#                     proj_features = self.model.get_contrastive_features(images)
#                     contrastive_loss = self.calculate_contrastive_loss(proj_features, labels)
#                 else:
#                     contrastive_loss = torch.tensor(0.0).to(self.device)
                
#                 # Combine losses with dynamic weighting
#                 trust_score = getattr(self.model, 'trust_score', 0.8)  # Default to 0.8 if not set
#                 global_weight = max(0.3, 1.0 - trust_score)  # Ensure minimum global weight
#                 personalized_weight = trust_score
                
#                 total_loss = (
#                     global_weight * global_loss +
#                     personalized_weight * personalized_loss +
#                     self.distillation_weight * distill_loss +  # Use client's distillation weight
#                     0.1 * contrastive_loss  # Small weight for contrastive loss
#                 )
                
#                 # Backward pass and optimize
#                 total_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # Add gradient clipping
#                 self.optimizer.step()
                
#                 # Update metrics
#                 epoch_loss += total_loss.item()
#                 _, predicted = torch.max(personalized_logits, 1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
                
#                 # Track individual losses
#                 global_losses.append(global_loss.item())
#                 personalized_losses.append(personalized_loss.item())
#                 contrastive_losses.append(contrastive_loss.item())
            
#             # Log epoch metrics
#             accuracy = 100. * correct / total
#             logger.debug(f'Epoch {epoch}: Loss: {epoch_loss/len(self.trainloader):.3f}, '
#                         f'Acc: {accuracy:.2f}%, Trust: {trust_score:.3f}')
            
#             # Update trust score based on performance
#             if hasattr(self.trainer, 'calculate_trust_score'):
#                 global_acc = self.evaluate_global_accuracy()
#                 personalized_acc = accuracy / 100.0  # Convert to decimal
#                 new_trust = self.trainer.calculate_trust_score(global_acc, personalized_acc, round_idx)
#                 self.model.trust_score = new_trust
        
#         # Create state_dict dictionary to return - only include non-personalized parameters
#         if self.enable_personalization and hasattr(self.model, 'get_global_params'):
#             return_dict = self.model.get_global_params()
#         else:
#             return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
#                           if 'personalized_head' not in k}
        
#         # Create stats dictionary
#         stats_dict = {
#             'trust_score': getattr(self.model, 'trust_score', 0.8),
#             'global_loss': np.mean(global_losses),
#             'personalized_loss': np.mean(personalized_losses),
#             'contrastive_loss': np.mean(contrastive_losses)
#         }
        
#         # Return both the model state and stats (as two separate values)
#         return return_dict, stats_dict

#     def evaluate_global_accuracy(self):
#         """Evaluate accuracy using only the global head"""
#         self.model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in self.trainloader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.model(images)
#                 global_logits = outputs['global_logit']
#                 _, predicted = torch.max(global_logits, 1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
        
#         return correct / total

#     def calculate_contrastive_loss(self, features, labels):
#         """Calculate supervised contrastive loss"""
#         temperature = self.model.temperature
#         batch_size = features.size(0)
        
#         # Normalize features
#         features = F.normalize(features, p=2, dim=1)
        
#         # Compute similarity matrix
#         similarity_matrix = torch.matmul(features, features.T) / temperature
        
#         # Create mask for positive pairs
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float()
        
#         # Remove diagonal elements
#         mask = mask - torch.eye(batch_size).to(mask.device)
        
#         # Compute loss
#         exp_sim = torch.exp(similarity_matrix)
#         log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
#         mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
#         return -mean_log_prob.mean()

import copy
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

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
        
        # Force CPU usage and optimize threading
        self.device = torch.device("cpu")
        torch.set_num_threads(4)  # Optimize CPU usage

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
        # Override device to ensure CPU usage
        self.device = torch.device("cpu")
        self.rounds_trained += 1
        self.trainer = trainer  # Store trainer reference
        
        # Save previous model state for trust score calculation
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            self.previous_model_state = {k: v.to(self.device) for k, v in self.model.state_dict().items()}
        
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
        
        # Setup data loader with CPU-optimized settings
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
            num_workers=0,  # Set to 0 for CPU to avoid multiprocessing issues
            pin_memory=False,
            drop_last=True
        )
        
        # Calculate trust score for learning rate adjustment if enabled
        trust_score = self.compute_trust_score() if self.enable_trust_filtering else 0.8
        
        # Setup optimizer with trust-based adaptive learning rate
        if self.enable_cyclical_lr:
            # Adapt learning rate based on trust score if available
            if hasattr(self, 'trust_score_history') and len(self.trust_score_history) > 0:
                # Use trust score to adjust learning rate
                trust_avg = np.mean(self.trust_score_history[-3:]) if len(self.trust_score_history) >= 3 else trust_score
                
                # Base learning rate on trust score - higher trust = higher LR
                trust_adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * trust_avg
                
                # Apply cyclical pattern based on rounds trained
                cycle = np.sin(np.pi * (self.rounds_trained % self.step_size) / self.step_size)
                cycle_factor = 0.5 * (1 + cycle)
                
                # Combine trust adjustment with cycle
                current_lr = self.base_lr + (trust_adjusted_lr - self.base_lr) * cycle_factor
                
                logger.info(f"[C{self.client_index}] Trust-adjusted LR: {current_lr:.6f} (trust={trust_avg:.3f}, cycle={cycle_factor:.2f})")
            else:
                # Default cyclical LR if no trust score history
                cycle = np.sin(np.pi * (self.rounds_trained % self.step_size) / self.step_size)
                cycle_factor = 0.5 * (1 + cycle)
                current_lr = self.base_lr + (self.max_lr - self.base_lr) * cycle_factor
                logger.info(f"[C{self.client_index}] Cyclical LR: {current_lr:.6f} (cycle={cycle_factor:.2f})")
        else:
            # No cyclical LR, use constant rate
            current_lr = local_lr
            logger.info(f"[C{self.client_index}] Fixed LR: {current_lr:.6f}")
            
        # Ensure learning rate is reasonable
        current_lr = max(1e-5, min(current_lr, 0.1))
        
        # Special case for early rounds to prevent cold start
        if self.rounds_trained <= 2:
            current_lr = min(current_lr * 1.5, 0.1)  # Slightly higher LR at the start

        self.local_epochs = kwargs.get('local_ep', 5)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=kwargs.get('weight_decay', 1e-5)
        )
        
        # Reset scheduler for each round of training
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
        """Compute FedProx regularization term"""
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)**2
        return self.fedprox_mu * proximal_term / 2

    def compute_distillation_loss(self, student_logits, teacher_logits, features=None, teacher_features=None):
        """
        Compute distillation loss between teacher and student models.
        Improved to prevent overfitting to the teacher model.
        """
        # Logit-based distillation with temperature scaling
        temp = self.distillation_temp
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        log_probs = F.log_softmax(student_logits / temp, dim=1)
        
        # Apply temperature scaling and normalize
        kd_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temp * temp)
        
        # Feature distillation (optional)
        feature_loss = 0.0
        if features is not None and teacher_features is not None:
            # Normalize features
            student_features = F.normalize(features, p=2, dim=1)
            teacher_features = F.normalize(teacher_features, p=2, dim=1)
            
            # Apply L2 loss with a smaller weight for feature distillation
            feature_loss = F.mse_loss(student_features, teacher_features) * 0.1
        
        # Combine losses with a safeguard to prevent overfitting to teacher
        # Dynamically reduce distillation impact as training progresses
        distillation_weight_factor = max(0.2, 1.0 - 0.05 * self.rounds_trained)
        combined_loss = (kd_loss + feature_loss) * distillation_weight_factor
        
        # Add a safeguard against excessive distillation
        if self.rounds_trained > 10 and hasattr(self, 'distillation_loss_avg') and self.distillation_loss_avg > 0:
            if combined_loss > 2.0 * self.distillation_loss_avg:
                combined_loss = self.distillation_loss_avg
                logger.warning(f"[C{self.client_index}] Limiting excessive distillation loss")
                
        return combined_loss

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
        """Compute trust score based on update magnitude and consistency"""
        if self.previous_model_state is None:
            return 0.8  # Initialize with a neutral score instead of 1.0
        
        current_state = self.model.state_dict()
        
        update_norm = 0.0
        param_count = 0
        
        for key in current_state:
            if 'personalized_head' not in key and key in self.previous_model_state:
                current_param = current_state[key].to(self.device).float()
                prev_param = self.previous_model_state[key].to(self.device).float()
                diff = current_param - prev_param
                update_norm += torch.norm(diff).item() ** 2
                param_count += diff.numel()
        
        if param_count > 0:
            update_norm = (update_norm / param_count) ** 0.5
        
        # Ensure we have a more dynamic update history
        self.update_history.append(update_norm)
        if len(self.update_history) > 5:
            self.update_history.pop(0)
            
        # Add small epsilon to prevent division by zero
        update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
        # Compute trust score components with improved scaling
        magnitude_score = 1.0 / (1.0 + 10.0 * update_norm)  # Adjust sensitivity
        consistency_score = 1.0 / (1.0 + 5.0 * update_variance)  # Adjust sensitivity
        
        # Dynamic weighting based on training progress
        mag_weight = max(0.5, min(0.8, 0.8 - 0.01 * self.rounds_trained))  # Reduce weight over time
        cons_weight = 1.0 - mag_weight
        
        # Combine scores with dynamic weights
        trust_score = mag_weight * magnitude_score + cons_weight * consistency_score
        
        # Add random noise to break any potential plateaus
        trust_score += np.random.normal(0, 0.02)  # Small random noise
        
        # Save current state for next round
        self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
        # Add to history for tracking
        self.trust_score_history.append(trust_score)
        if len(self.trust_score_history) > 10:
            self.trust_score_history.pop(0)
        
        # Ensure the trust score is within [0.1, 1.0] to avoid getting stuck
        return max(0.1, min(1.0, trust_score))

    def compute_fisher_information(self):
        """Compute Fisher Information Matrix for EWC regularization"""
        if not self.ewc_enabled or self.trainloader is None:
            return
                
        fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'personalized_head' not in name:
                fisher_information[name] = torch.zeros_like(param)
                
        self.model.eval()
        
        # Use a small subset of data for efficiency on CPU
        sample_count = 0
        max_samples = 50  # Limit samples for CPU efficiency
        
        for images, labels in self.trainloader:
            if sample_count >= max_samples:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            output = self.model(images)
            if isinstance(output, dict):
                logits = output.get("logit", output.get("global_logit", None))
            else:
                logits = output
                
            if logits is None:
                continue
                
            log_probs = F.log_softmax(logits, dim=1)
            samples = torch.multinomial(torch.exp(log_probs), 1).squeeze()
            loss = F.nll_loss(log_probs, samples)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and 'personalized_head' not in name:
                    fisher_information[name] += param.grad.pow(2).detach()
            
            sample_count += len(images)
                    
        # Normalize by the number of samples
        for name in fisher_information:
            fisher_information[name] /= sample_count if sample_count > 0 else 1.0
                
        self.fisher_information = fisher_information
        self.optimal_parameters = {name: param.clone().detach() for name, param in self.model.named_parameters() 
                                  if name in fisher_information}
        self.model.train()
    
    def compute_multi_level_rcl_loss(self, output, labels):
        """Compute RCL loss across multiple feature levels"""
        if not self.multi_level_rcl or not isinstance(output, dict):
            return torch.tensor(0.0, device=self.device)
                
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Get features from different layers
        layer_features = []
        for i in range(5):  # Assuming 5 layers (0-4)
            layer_key = f"layer{i}"
            if layer_key in output:
                layer_features.append(output[layer_key])
                
        # If no layer features found, return 0
        if not layer_features:
            return total_loss
                
        # Apply RCL to each layer with weights
        for i, features in enumerate(layer_features):
            if i < len(self.layer_weights):
                weight = self.layer_weights[i]
                # Reshape features if needed
                if len(features.shape) > 2:
                    # Global average pooling for conv features
                    features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
                
                # Apply L2 normalization
                features = F.normalize(features, p=2, dim=1)
                
                # Compute RCL loss for this layer
                layer_loss = self.relaxed_contrastive_loss(features, labels)
                total_loss += weight * layer_loss
                
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
        
        # Check if trainloader is available
        if not hasattr(self, 'trainloader') or self.trainloader is None:
            logger.warning(f"[C{self.client_index}] No trainloader available, skipping training")
            return None, None
        
        # Get initial global model state for distillation
        global_model_state = copy.deepcopy(self.model.state_dict())
        
        # Setup gradient accumulation to reduce memory usage on CPU
        accumulation_steps = 2  # Adjust based on batch size and CPU memory
        self.optimizer.zero_grad()
        
        # Training loop
        for epoch in range(self.args.trainer.local_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Extract necessary outputs
                if isinstance(outputs, dict):
                    features = outputs.get('feature', None)
                    global_logits = outputs.get('global_logit', None)
                    personalized_logits = outputs.get('personalized_logit', None)
                    
                    # If structure doesn't match expected output, adapt
                    if global_logits is None and 'logit' in outputs:
                        global_logits = outputs['logit']
                    if personalized_logits is None and global_logits is not None:
                        personalized_logits = global_logits  # Fallback
                else:
                    # Handle case where model returns logits directly
                    global_logits = outputs
                    personalized_logits = outputs
                    features = None
                
                # Calculate losses with proper error handling
                try:
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
                            teacher_logits = teacher_outputs['global_logit'] if isinstance(teacher_outputs, dict) else teacher_outputs
                        
                        distill_loss = self.compute_distillation_loss(
                            personalized_logits, 
                            teacher_logits,
                            features=features if features is not None else None,
                            teacher_features=None
                        )
                    else:
                        distill_loss = torch.tensor(0.0, device=self.device)
                    
                    # 4. Contrastive loss for feature alignment
                    contrastive_loss = self.compute_multi_level_rcl_loss(outputs, labels) if isinstance(outputs, dict) else torch.tensor(0.0, device=self.device)
                    
                    # 5. Add FedProx regularization if enabled
                    fedprox_loss = self.compute_fedprox_term() if self.enable_fedprox else torch.tensor(0.0, device=self.device)
                    
                    # 6. Add EWC regularization if enabled
                    ewc_loss = self.compute_ewc_loss() if self.ewc_enabled else torch.tensor(0.0, device=self.device)
                    
                    # Combine losses with dynamic weighting
                    trust_score = getattr(self.model, 'trust_score', 0.8)  # Default to 0.8 if not set
                    global_weight = max(0.3, 1.0 - trust_score)  # Ensure minimum global weight
                    personalized_weight = trust_score
                    
                    total_loss = (
                        global_weight * global_loss +
                        personalized_weight * personalized_loss +
                        self.distillation_weight * distill_loss +
                        0.1 * contrastive_loss +
                        fedprox_loss +
                        ewc_loss
                    ) / accumulation_steps
                    
                    # Backward pass with gradient accumulation
                    total_loss.backward()
                    
                    # Update only after accumulation steps or at the last batch
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.trainloader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # Add gradient clipping
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += total_loss.item() * accumulation_steps
                    _, predicted = torch.max(personalized_logits, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Track individual losses
                    global_losses.append(global_loss.item())
                    personalized_losses.append(personalized_loss.item())
                    contrastive_losses.append(contrastive_loss.item())
                    
                except Exception as e:
                    logger.error(f"[C{self.client_index}] Error in loss computation: {str(e)}")
                    continue
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch metrics
            if total > 0:
                accuracy = 100. * correct / total
                logger.debug(f'Epoch {epoch}: Loss: {epoch_loss/len(self.trainloader):.3f}, '
                            f'Acc: {accuracy:.2f}%, Trust: {trust_score:.3f}')
                
                # Update trust score based on performance
                if hasattr(self.trainer, 'calculate_trust_score'):
                    global_acc = self.evaluate_global_accuracy()
                    personalized_acc = accuracy / 100.0  # Convert to decimal
                    new_trust = self.trainer.calculate_trust_score(global_acc, personalized_acc, round_idx)
                    self.model.trust_score = new_trust
            
            # Check for catastrophic forgetting
            if not self.detect_and_fix_catastrophic_forgetting(epoch_loss, epoch):
                break
        
        # Create state_dict dictionary to return - only include non-personalized parameters
        try:
            if self.enable_personalization and hasattr(self.model, 'get_global_params'):
                return_dict = self.model.get_global_params()
            else:
                return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
                              if 'personalized_head' not in k}
            
            # Create stats dictionary
            stats_dict = {
                'trust_score': getattr(self.model, 'trust_score', 0.8),
                'global_loss': np.mean(global_losses) if global_losses else 0.0,
                'personalized_loss': np.mean(personalized_losses) if personalized_losses else 0.0,
                'contrastive_loss': np.mean(contrastive_losses) if contrastive_losses else 0.0
            }
            
            # Return both the model state and stats (as two separate values)
            return return_dict, stats_dict
            
        except Exception as e:
            logger.error(f"[C{self.client_index}] Error creating return values: {str(e)}")
            return None, None

    def evaluate_global_accuracy(self):
        """Evaluate accuracy using only the global head"""
        self.model.eval()
        correct = 0
        total = 0
        
        # Limit evaluation to a small subset for CPU efficiency
        max_samples = 100
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in self.trainloader:
                if sample_count >= max_samples:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                if isinstance(outputs, dict) and 'global_logit' in outputs:
                    global_logits = outputs['global_logit']
                elif isinstance(outputs, dict) and 'logit' in outputs:
                    global_logits = outputs['logit']
                else:
                    global_logits = outputs
                
                _, predicted = torch.max(global_logits, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                sample_count += len(images)
        
        accuracy = correct / total if total > 0 else 0.0
        self.model.train()
        return accuracy

    def calculate_contrastive_loss(self, features, labels):
        """Calculate supervised contrastive loss"""
        if features is None or not hasattr(self.model, 'temperature'):
            return torch.tensor(0.0, device=self.device)
            
        temperature = getattr(self.model, 'temperature', 0.05)
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # For small batch sizes, add a check
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(batch_size, device=self.device)
        
        # Check for valid pairs
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return -mean_log_prob.mean()
