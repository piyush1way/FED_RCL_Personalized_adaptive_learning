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

#         # Initialize trust filtering settings
#         trust_filtering_config = getattr(args.client, "trust_filtering", {})
#         self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
#         self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        
#         # Track gradient history for adaptive learning rate
#         self.grad_history = []
#         self.grad_variance = 0.0
#         self.previous_model_state = None
#         self.update_history = []

#         # Initialize personalization settings
#         personalization_config = getattr(args.client, "personalization", {})
#         self.enable_personalization = getattr(personalization_config, "enable", False)
#         self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
#         self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)

#         # Initialize adaptive learning rate settings
#         adaptive_lr_config = getattr(args.client, "adaptive_lr", {})
#         self.enable_adaptive_lr = getattr(adaptive_lr_config, "enable", False)
#         self.adaptive_lr_beta = getattr(adaptive_lr_config, "beta", 0.1)
#         self.adaptive_lr_min = getattr(adaptive_lr_config, "min_lr", 0.001)
#         self.adaptive_lr_max = getattr(adaptive_lr_config, "max_lr", 0.1)
#         self.warmup_rounds = getattr(adaptive_lr_config, "warmup_rounds", 5)
#         self.rounds_trained = 0

#         self.model = model
#         self.global_model = copy.deepcopy(model)
#         self.device = torch.device("cpu")  # Default to CPU

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
        
#         # Add relaxed contrastive loss with more conservative parameters
#         self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
#             temperature=0.1,  # Increased from 0.05 for more stable gradients
#             lambda_penalty=0.05,  # Reduced to prevent dominating the loss
#             similarity_threshold=0.5  # Reduced from 0.7 to be less aggressive
#         )
        
#         # Track metrics for debugging
#         self.ce_loss_avg = 0.0
#         self.rcl_loss_avg = 0.0

#     def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
#         """Initialize client model, dataset, and optimizer"""
#         # Store device for later use
#         self.device = device
#         self.rounds_trained += 1
        
#         # Store previous model state for trust score calculation
#         if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
#             # Make sure previous model state is on the same device as the current model
#             self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
#         self._update_model(state_dict)
#         self._update_global_model(state_dict)

#         # Move models to the correct device
#         self.model.to(self.device)
#         self.global_model.to(self.device)

#         # Freeze the global model
#         for param in self.global_model.parameters():
#             param.requires_grad = False

#         # Apply personalization settings if enabled
#         if self.enable_personalization:
#             if isinstance(self.model, torch.nn.DataParallel):
#                 model_module = self.model.module
#             else:
#                 model_module = self.model
                
#             # Enable personalized mode if the model supports it
#             if hasattr(model_module, 'enable_personalized_mode'):
#                 model_module.enable_personalized_mode()
#             elif hasattr(model_module, 'use_personalized_head'):
#                 model_module.use_personalized_head = True
                
#             # Apply freezing strategies
#             if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
#                 model_module.freeze_backbone()
                
#             if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
#                 model_module.setup_adaptive_freezing()

#         self.trainer = trainer
#         self.global_epoch = global_epoch

#         # Check if dataset is empty and handle gracefully
#         if local_dataset is None or len(local_dataset) == 0:
#             logger.warning(f"[C{self.client_index}] Empty dataset provided. Creating a dummy loader.")
#             self.loader = None
#         else:
#             # Create DataLoader with proper settings
#             train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
#                 if hasattr(self.args.dataset, 'num_instances') and self.args.dataset.num_instances > 0 else None
                
#             # Configure DataLoader to drop incomplete batches
#             self.loader = DataLoader(
#                 local_dataset,
#                 batch_size=self.args.batch_size,
#                 sampler=train_sampler,
#                 shuffle=train_sampler is None,
#                 num_workers=0,  # Reduced workers to avoid issues
#                 pin_memory=False,  # Disabled pin_memory for stability
#                 drop_last=True  # Drop incomplete batches to ensure consistent sizes
#             )

#         # Apply adaptive learning rate if enabled
#         if self.enable_adaptive_lr and self.rounds_trained > self.warmup_rounds:
#             # Adjust learning rate based on gradient variance
#             adjusted_lr = self.adaptive_learning_rate(local_lr)
#             logger.info(f"[C{self.client_index}] Adjusted LR: {adjusted_lr:.6f} (base: {local_lr:.6f}, variance: {self.grad_variance:.6f})")
#             local_lr = adjusted_lr
#         else:
#             logger.info(f"[C{self.client_index}] Using base LR: {local_lr:.6f}")

#         # Set up optimizer based on personalization mode
#         if self.enable_personalization:
#             # Only optimize personalized parameters
#             if hasattr(model_module, 'get_local_params'):
#                 # Get personalized parameters
#                 personalized_params = []
#                 for name, param in self.model.named_parameters():
#                     if 'personalized_head' in name:
#                         personalized_params.append(param)
                
#                 if personalized_params:
#                     self.optimizer = torch.optim.SGD(
#                         personalized_params,
#                         lr=local_lr,
#                         momentum=self.args.optimizer.momentum,
#                         weight_decay=self.args.optimizer.wd
#                     )
#                 else:
#                     # Fallback to all parameters if no personalized params found
#                     self.optimizer = torch.optim.SGD(
#                         self.model.parameters(),
#                         lr=local_lr,
#                         momentum=self.args.optimizer.momentum,
#                         weight_decay=self.args.optimizer.wd
#                     )
#             else:
#                 # Fallback to all parameters
#                 self.optimizer = torch.optim.SGD(
#                     self.model.parameters(),
#                     lr=local_lr,
#                     momentum=self.args.optimizer.momentum,
#                     weight_decay=self.args.optimizer.wd
#                 )
#         else:
#             # Regular mode: optimize all parameters
#             self.optimizer = torch.optim.SGD(
#                 self.model.parameters(),
#                 lr=local_lr,
#                 momentum=self.args.optimizer.momentum,
#                 weight_decay=self.args.optimizer.wd
#             )
            
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer=self.optimizer,
#             lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
#         )
        
#     def adaptive_learning_rate(self, base_lr):
#         """Adjust learning rate based on gradient variance"""
#         if len(self.grad_history) < 2:
#             return base_lr
            
#         # More conservative adjustment factor
#         lr_factor = 1.0 / (1.0 + self.adaptive_lr_beta * min(self.grad_variance, 10.0))
        
#         # Limit the adaptive range - more conservative limits
#         adjusted_lr = base_lr * max(min(lr_factor, 1.0), 0.1)
        
#         # Ensure LR stays within absolute bounds
#         adjusted_lr = max(min(adjusted_lr, self.adaptive_lr_max), self.adaptive_lr_min)
        
#         return adjusted_lr

#     def compute_trust_score(self):
#         """Compute trust score for this client based on model update consistency"""
#         if self.previous_model_state is None:
#             return 1.0  # Default high trust if no previous state
        
#         # Get current model state
#         current_state = self.model.state_dict()
        
#         # Calculate update norm (L2 distance between current and previous model)
#         update_norm = 0.0
#         param_count = 0
        
#         for key in current_state:
#             if 'personalized_head' not in key and key in self.previous_model_state:
#                 # Ensure both tensors are on the same device
#                 current_param = current_state[key].to(self.device).float()
#                 prev_param = self.previous_model_state[key].to(self.device).float()
#                 diff = current_param - prev_param
#                 update_norm += torch.norm(diff).item() ** 2
#                 param_count += diff.numel()
        
#         if param_count > 0:
#             update_norm = (update_norm / param_count) ** 0.5
        
#         # Track update norms for variance calculation
#         self.update_history.append(update_norm)
#         if len(self.update_history) > 5:
#             self.update_history.pop(0)
            
#         # Calculate variance of updates
#         update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
#         # Trust score combines magnitude and consistency
#         magnitude_score = 1.0 / (1.0 + update_norm)
#         consistency_score = 1.0 / (1.0 + update_variance)
        
#         # Weighted combination
#         trust_score = 0.7 * magnitude_score + 0.3 * consistency_score
        
#         # Update previous model state for next calculation
#         self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
#         return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

#     def store_gradients(self):
#         """Store current gradients for variance calculation"""
#         grad_norm = 0.0
#         param_count = 0
        
#         for param in self.model.parameters():
#             if param.grad is not None:
#                 grad_norm += torch.norm(param.grad).item() ** 2
#                 param_count += param.grad.numel()
        
#         if param_count > 0:
#             grad_norm = (grad_norm / param_count) ** 0.5
#             self.grad_history.append(grad_norm)
            
#             # Keep history manageable
#             if len(self.grad_history) > 5:  # Reduced from 10 to be more responsive
#                 self.grad_history.pop(0)
            
#             # Update variance calculation
#             self.grad_variance = np.var(self.grad_history) if len(self.grad_history) > 1 else 0.0

#     def local_train(self, global_epoch, **kwargs):
#         """Performs local training on the client's dataset"""
#         self.global_epoch = global_epoch

#         scaler = GradScaler(enabled=self.device.type == "cuda")  # Use AMP only if CUDA is available
#         start_time = time.time()
#         loss_meter = AverageMeter('Loss', ':.4f')
#         ce_loss_meter = AverageMeter('CE Loss', ':.4f')
#         rcl_loss_meter = AverageMeter('RCL Loss', ':.4f')

#         # Skip training if loader is empty
#         if self.loader is None or len(self.loader) == 0:
#             logger.warning(f"[C{self.client_index}] No data for training!")
#             return None, None

#         for local_epoch in range(self.args.trainer.local_epochs):
#             epoch_loss = 0.0
#             samples_processed = 0
            
#             for images, labels in self.loader:
#                 # Skip batches that are too small for contrastive learning
#                 if len(images) < 2:
#                     continue
                    
#                 images = images.to(self.device, non_blocking=True)
#                 labels = labels.to(self.device, non_blocking=True)

#                 self.model.zero_grad()

#                 with autocast(enabled=self.device.type == "cuda"):
#                     # Forward pass
#                     output = self.model(images, get_projection=True)
                    
#                     # Extract logits from the output dictionary
#                     if isinstance(output, dict):
#                         logits = output["logit"] if "logit" in output else output.get("personalized_logit", output)
#                         features = output.get("feature", None)
#                         projection = output.get("projection", None)
#                     else:
#                         logits = output
#                         features = None
#                         projection = None
                    
#                     # Compute cross-entropy loss
#                     ce_loss = self.criterion(logits, labels)
                    
#                     # Compute relaxed contrastive loss if enabled
#                     rcl_loss = 0.0
#                     if hasattr(self.args.client, 'rcl_loss') and getattr(self.args.client.rcl_loss, 'weight', 0) > 0:
#                         # Use projection features if available, otherwise use regular features
#                         contrastive_features = projection if projection is not None else features
                        
#                         if contrastive_features is not None and len(contrastive_features) >= 2:
#                             # Apply relaxed contrastive loss with a small weight
#                             rcl_weight = getattr(self.args.client.rcl_loss, 'weight', 0.1)
#                             rcl_loss = self.relaxed_contrastive_loss(contrastive_features, labels) * rcl_weight
                    
#                     # Total loss with proper weighting
#                     loss = ce_loss + rcl_loss
                    
#                     # Update loss meters
#                     ce_loss_meter.update(ce_loss.item(), images.size(0))
#                     if rcl_loss > 0:
#                         rcl_loss_meter.update(rcl_loss.item(), images.size(0))
                
#                 # Check if loss is valid
#                 if not torch.isfinite(loss):
#                     logger.warning(f"[C{self.client_index}] Loss is {loss}, skipping batch")
#                     continue
                    
#                 # Backward and optimize
#                 if self.device.type == "cuda":
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(self.optimizer)
#                 else:
#                     loss.backward()
                
#                 # Store gradients for adaptive learning rate
#                 if self.enable_adaptive_lr:
#                     self.store_gradients()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # Reduced from 10.0

#                 if self.device.type == "cuda":
#                     scaler.step(self.optimizer)
#                     scaler.update()
#                 else:
#                     self.optimizer.step()

#                 # Update metrics
#                 batch_size = images.size(0)
#                 loss_meter.update(loss.item(), batch_size)
#                 epoch_loss += loss.item() * batch_size
#                 samples_processed += batch_size

#             # Log per-epoch statistics
#             if samples_processed > 0:
#                 logger.info(f"[C{self.client_index}] Epoch {local_epoch+1}/{self.args.trainer.local_epochs}, Loss: {epoch_loss/samples_processed:.4f}")
#             else:
#                 logger.warning(f"[C{self.client_index}] No samples processed in epoch {local_epoch+1}")
            
#             self.scheduler.step()

#         end_time = time.time()
#         training_time = end_time - start_time
        
#         # Store average losses for reporting
#         self.ce_loss_avg = ce_loss_meter.avg
#         self.rcl_loss_avg = rcl_loss_meter.avg
        
#         # Compute trust score for client updates
#         trust_score = self.compute_trust_score() if self.enable_trust_filtering else 1.0

#         logger.info(f"[C{self.client_index}] Training Complete. Time: {training_time:.2f}s, Loss: {loss_meter.avg:.4f}, Trust Score: {trust_score:.3f}")

#         # Prepare model parameters for aggregation
#         if self.enable_personalization:
#             # Only share non-personalized parameters
#             if isinstance(self.model, torch.nn.DataParallel):
#                 model_module = self.model.module
#             else:
#                 model_module = self.model
                
#             # If the model supports separate global/personal params
#             if hasattr(model_module, 'get_global_params'):
#                 # Extract only the global parameters for server aggregation
#                 global_params = model_module.get_global_params()
#                 return_dict = {k: v.cpu() for k, v in global_params.items()}
#             else:
#                 # Fallback: filter out personalized head parameters
#                 return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
#                               if 'personalized_head' not in k}
#         else:
#             # Regular mode: return the full model
#             return_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

#         # Filter out low-trust clients
#         if self.enable_trust_filtering and trust_score < self.trust_threshold:
#             logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
#             return None, None

#         # Move models back to CPU to save memory
#         self.model = self.model.cpu()
#         self.global_model = self.global_model.cpu()

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()  # Clear CUDA cache

#         return return_dict, {"loss": float(loss_meter.avg), "trust_score": trust_score, "grad_variance": float(self.grad_variance)}


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

        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        
        self.rounds_trained = 0
        self.previous_model_state = None
        self.update_history = []

        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)

        self.enable_cyclical_lr = True
        self.base_lr = 0.001
        self.max_lr = 0.1
        self.step_size = 10
        
        self.enable_fedprox = True
        self.fedprox_mu = 0.01

        self.enable_distillation = True
        self.distillation_temp = 2.0
        self.distillation_weight = 0.5

        self.model = model
        self.global_model = copy.deepcopy(model)
        self.device = torch.device("cpu")

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
        
        self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
            temperature=0.1,
            lambda_penalty=0.05,
            similarity_threshold=0.5
        )
        
        self.ce_loss_avg = 0.0
        self.rcl_loss_avg = 0.0
        self.distillation_loss_avg = 0.0
        self.fedprox_loss_avg = 0.0

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        self.device = device
        self.rounds_trained += 1
        
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        self.model.to(self.device)
        self.global_model.to(self.device)

        for param in self.global_model.parameters():
            param.requires_grad = False

        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            if hasattr(model_module, 'enable_personalized_mode'):
                model_module.enable_personalized_mode()
            elif hasattr(model_module, 'use_personalized_head'):
                model_module.use_personalized_head = True
                
            if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
                model_module.freeze_backbone()
                
            if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
                freeze_ratio = max(0.0, 0.8 - (self.rounds_trained / 50.0))
                model_module.setup_adaptive_freezing(freeze_ratio=freeze_ratio)

        self.trainer = trainer
        self.global_epoch = global_epoch

        if local_dataset is None or len(local_dataset) == 0:
            logger.warning(f"[C{self.client_index}] Empty dataset provided. Creating a dummy loader.")
            self.loader = None
        else:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
                if hasattr(self.args.dataset, 'num_instances') and self.args.dataset.num_instances > 0 else None
                
            self.loader = DataLoader(
                local_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                num_workers=0,
                pin_memory=False,
                drop_last=True
            )

        if self.enable_cyclical_lr:
            cycle_step = self.rounds_trained % (2 * self.step_size)
            x = abs(cycle_step / self.step_size - 1)
            adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
            logger.info(f"[C{self.client_index}] Cyclical LR: {adjusted_lr:.6f} (cycle step: {cycle_step})")
            local_lr = adjusted_lr
        else:
            logger.info(f"[C{self.client_index}] Using base LR: {local_lr:.6f}")

        if self.enable_personalization:
            if hasattr(model_module, 'get_local_params'):
                personalized_params = []
                for name, param in self.model.named_parameters():
                    if 'personalized_head' in name or 'projection_head' in name:
                        personalized_params.append(param)
                
                if personalized_params:
                    self.optimizer = torch.optim.SGD(
                        personalized_params,
                        lr=local_lr,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
                else:
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=local_lr,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=local_lr,
                    momentum=self.args.optimizer.momentum,
                    weight_decay=self.args.optimizer.wd
                )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=local_lr,
                momentum=self.args.optimizer.momentum,
                weight_decay=self.args.optimizer.wd
            )
            
        if self.rounds_trained < 5:
            self.local_epochs = min(20, self.args.trainer.local_epochs * 2)
        else:
            self.local_epochs = self.args.trainer.local_epochs
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
        )

    def compute_fedprox_term(self):
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)**2
        return self.fedprox_mu * proximal_term / 2

    def compute_distillation_loss(self, student_logits, teacher_logits):
        temp = self.distillation_temp
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        log_probs = F.log_softmax(student_logits / temp, dim=1)
        distillation_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return distillation_loss * (temp ** 2)

    def compute_trust_score(self):
        if self.previous_model_state is None:
            return 1.0
        
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
        
        self.update_history.append(update_norm)
        if len(self.update_history) > 5:
            self.update_history.pop(0)
            
        update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
        magnitude_score = 1.0 / (1.0 + update_norm)
        consistency_score = 1.0 / (1.0 + update_variance)
        
        trust_score = 0.7 * magnitude_score + 0.3 * consistency_score
        
        self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
        return max(0.0, min(1.0, trust_score))

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        scaler = GradScaler(enabled=self.device.type == "cuda")
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.4f')
        ce_loss_meter = AverageMeter('CE Loss', ':.4f')
        rcl_loss_meter = AverageMeter('RCL Loss', ':.4f')
        distillation_loss_meter = AverageMeter('Distill Loss', ':.4f')
        fedprox_loss_meter = AverageMeter('FedProx Loss', ':.4f')

        if self.loader is None or len(self.loader) == 0:
            logger.warning(f"[C{self.client_index}] No data for training!")
            return None, None

        for local_epoch in range(self.local_epochs):
            epoch_loss = 0.0
            samples_processed = 0
            
            for images, labels in self.loader:
                if len(images) < 2:
                    continue
                    
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.model.zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    output = self.model(images, get_projection=True)
                    
                    if isinstance(output, dict):
                        logits = output["logit"] if "logit" in output else output.get("personalized_logit", output)
                        features = output.get("feature", None)
                        projection = output.get("projection", None)
                    else:
                        logits = output
                        features = None
                        projection = None
                    
                    ce_loss = self.criterion(logits, labels)
                    
                    rcl_loss = 0.0
                    if hasattr(self.args.client, 'rcl_loss') and getattr(self.args.client.rcl_loss, 'weight', 0) > 0:
                        contrastive_features = projection if projection is not None else features
                        
                        if contrastive_features is not None and len(contrastive_features) >= 2:
                            rcl_weight = getattr(self.args.client.rcl_loss, 'weight', 0.1)
                            rcl_loss = self.relaxed_contrastive_loss(contrastive_features, labels) * rcl_weight
                    
                    distillation_loss = 0.0
                    if self.enable_distillation:
                        with torch.no_grad():
                            global_output = self.global_model(images)
                            if isinstance(global_output, dict):
                                global_logits = global_output.get("global_logit", global_output.get("logit", None))
                            else:
                                global_logits = global_output
                                
                        if global_logits is not None:
                            distillation_loss = self.compute_distillation_loss(logits, global_logits) * self.distillation_weight
                    
                    fedprox_loss = 0.0
                    if self.enable_fedprox:
                        fedprox_loss = self.compute_fedprox_term()
                    
                    loss = ce_loss + rcl_loss + distillation_loss + fedprox_loss
                    
                    ce_loss_meter.update(ce_loss.item(), images.size(0))
                    if rcl_loss > 0:
                        rcl_loss_meter.update(rcl_loss.item(), images.size(0))
                    if distillation_loss > 0:
                        distillation_loss_meter.update(distillation_loss.item(), images.size(0))
                    if fedprox_loss > 0:
                        fedprox_loss_meter.update(fedprox_loss.item(), images.size(0))
                
                if not torch.isfinite(loss):
                    logger.warning(f"[C{self.client_index}] Loss is {loss}, skipping batch")
                    continue
                    
                if self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                if self.device.type == "cuda":
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                epoch_loss += loss.item() * batch_size
                samples_processed += batch_size

            if samples_processed > 0:
                logger.info(f"[C{self.client_index}] Epoch {local_epoch+1}/{self.local_epochs}, Loss: {epoch_loss/samples_processed:.4f}")
            else:
                logger.warning(f"[C{self.client_index}] No samples processed in epoch {local_epoch+1}")
            
            self.scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        
        self.ce_loss_avg = ce_loss_meter.avg
        self.rcl_loss_avg = rcl_loss_meter.avg
        self.distillation_loss_avg = distillation_loss_meter.avg
        self.fedprox_loss_avg = fedprox_loss_meter.avg
        
        trust_score = self.compute_trust_score() if self.enable_trust_filtering else 1.0

        logger.info(f"[C{self.client_index}] Training Complete. Time: {training_time:.2f}s, Loss: {loss_meter.avg:.4f}, Trust Score: {trust_score:.3f}")

        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            if hasattr(model_module, 'get_global_params'):
                global_params = model_module.get_global_params()
                return_dict = {k: v.cpu() for k, v in global_params.items()}
            else:
                return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
                              if 'personalized_head' not in k}
        else:
            return_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if self.enable_trust_filtering and trust_score < self.trust_threshold:
            logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
            return None, None

        self.model = self.model.cpu()
        self.global_model = self.global_model.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return return_dict, {"loss": float(loss_meter.avg), 
                            "trust_score": trust_score, 
                            "ce_loss": float(self.ce_loss_avg),
                            "rcl_loss": float(self.rcl_loss_avg),
                            "distillation_loss": float(self.distillation_loss_avg),
                            "fedprox_loss": float(self.fedprox_loss_avg)}

