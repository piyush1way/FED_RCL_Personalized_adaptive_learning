# #!/usr/bin/env python
# # coding: utf-8
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
# from utils.loss import KL_u_p_loss
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

#         # Initialize personalization settings
#         personalization_config = getattr(args.client, "personalization", {})
#         self.enable_personalization = getattr(personalization_config, "enable", False)
#         self.freeze_backbone = getattr(personalization_config, "freeze_backbone", True)
#         self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", True)

#         self.model = model
#         self.global_model = copy.deepcopy(model)

#         self.rcl_criterions = {'scl': None, 'penalty': None, }
#         args_rcl = args.client.rcl_loss
#         self.global_epoch = 0

#         self.pairs = {}
#         for pair in args_rcl.pairs:
#             self.pairs[pair.name] = pair
#             self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        
#         self.criterion = nn.CrossEntropyLoss()

#     def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
#         """Initialize client model, dataset, and optimizer"""
#         self._update_model(state_dict)
#         self._update_global_model(state_dict)

#         # Ensure CUDA is available
#         if torch.cuda.is_available():
#             self.device = device
#         else:
#             logger.warning(f"Client {self.client_index}: CUDA not available, using CPU.")
#             self.device = torch.device("cpu")

#         self.model.to(self.device)
#         self.global_model.to(self.device)

#         # Freeze the global model
#         for param in self.global_model.parameters():
#             param.requires_grad = False

#         # Apply personalization settings
#         if self.enable_personalization:
#             if self.freeze_backbone:
#                 self.model.freeze_backbone()
#             if self.adaptive_layer_freezing:
#                 self.model.setup_adaptive_freezing()

#         self.trainer = trainer
#         self.global_epoch = global_epoch

#         # Create DataLoader
#         train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
#             if self.args.dataset.num_instances > 0 else None

#         self.loader = DataLoader(
#             local_dataset,
#             batch_size=self.args.batch_size,
#             sampler=train_sampler,
#             shuffle=train_sampler is None,
#             num_workers=self.args.num_workers,
#             pin_memory=self.device.type == "cuda"
#         )

#         # Optimizer & Scheduler
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=local_lr,
#             momentum=self.args.optimizer.momentum,
#             weight_decay=self.args.optimizer.wd
#         )
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer=self.optimizer,
#             lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
#         )

#     def compute_trust_score(self, local_state_dict):
#         """Compute trust score for this client"""
#         trust_score = 1.0  # Default trust score
#         return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

#     def local_train(self, global_epoch, **kwargs):
#         """Performs local training on the client's dataset"""
#         self.global_epoch = global_epoch

#         scaler = GradScaler(enabled=self.device.type == "cuda")  # Use AMP only if CUDA is available
#         start_time = time.time()
#         loss_meter = AverageMeter('Loss', ':.2f')

#         gradients = []

#         for local_epoch in range(self.args.trainer.local_epochs):
#             for images, labels in self.loader:
#                 images = images.to(self.device, non_blocking=True)
#                 labels = labels.to(self.device, non_blocking=True)

#                 self.model.zero_grad()

#                 with autocast(enabled=self.device.type == "cuda"):
#                     # Forward pass
#                     output = self.model(images)
                    
#                     # Extract logits from the output dictionary
#                     if isinstance(output, dict):
#                         logits = output["logit"]  # Extract logits tensor
#                     else:
#                         logits = output  # Fallback if output is not a dictionary
                    
#                     # Compute loss
#                     loss = self.criterion(logits, labels)

#                 if self.device.type == "cuda":
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(self.optimizer)
#                 else:
#                     loss.backward()

#                 # Collect gradients for debugging
#                 gradients.append([p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None])
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

#                 if self.device.type == "cuda":
#                     scaler.step(self.optimizer)
#                     scaler.update()
#                 else:
#                     self.optimizer.step()

#                 loss_meter.update(loss.item(), images.size(0))

#             self.scheduler.step()

#         end_time = time.time()

#         # Compute trust score for client updates
#         trust_score = self.compute_trust_score(self.model.state_dict())

#         logger.info(f"[C{self.client_index}] Training Complete. Time: {end_time - start_time:.2f}s, Loss: {loss_meter.avg:.3f}, Trust Score: {trust_score:.3f}")

#         # Filter out low-trust clients
#         if self.enable_trust_filtering and trust_score < self.trust_threshold:
#             logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
#             return None, None

#         # Move models back to CPU to save memory
#         self.model = self.model.cpu()
#         self.global_model = self.global_model.cpu()

#         gc.collect()
#         torch.cuda.empty_cache()  # Clear CUDA cache

#         return self.model.state_dict(), {"loss": float(loss_meter.avg), "trust_score": trust_score}
#!/usr/bin/env python
# coding: utf-8
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
from utils.loss import KL_u_p_loss
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

        # Initialize trust filtering settings
        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        
        # Track gradient history for trust score calculation
        self.grad_history = []
        self.grad_variance = 0.0

        # Initialize personalization settings
        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", True)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", True)

        # Initialize adaptive learning rate settings
        adaptive_lr_config = getattr(args.client, "adaptive_lr", {})
        self.enable_adaptive_lr = getattr(adaptive_lr_config, "enable", False)
        self.adaptive_lr_beta = getattr(adaptive_lr_config, "beta", 0.9)
        self.adaptive_lr_min = getattr(adaptive_lr_config, "min_lr", 0.001)
        self.adaptive_lr_max = getattr(adaptive_lr_config, "max_lr", 0.1)

        self.model = model
        self.global_model = copy.deepcopy(model)

        self.rcl_criterions = {'scl': None, 'penalty': None, }
        args_rcl = args.client.rcl_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_rcl.pairs:
            self.pairs[pair.name] = pair
            self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        
        self.criterion = nn.CrossEntropyLoss()

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        """Initialize client model, dataset, and optimizer"""
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        # Ensure CUDA is available
        if torch.cuda.is_available():
            self.device = device
        else:
            logger.warning(f"Client {self.client_index}: CUDA not available, using CPU.")
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.global_model.to(self.device)

        # Freeze the global model
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Apply personalization settings if enabled
        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            # Enable personalized mode if the model supports it
            if hasattr(model_module, 'enable_personalized_mode'):
                model_module.enable_personalized_mode()
                
            # Apply freezing strategies
            if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
                model_module.freeze_backbone()
                
            if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
                model_module.setup_adaptive_freezing()

        self.trainer = trainer
        self.global_epoch = global_epoch

        # Create DataLoader
        train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
            if self.args.dataset.num_instances > 0 else None

        self.loader = DataLoader(
            local_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda"
        )

        # Apply adaptive learning rate if enabled
        if self.enable_adaptive_lr:
            # Adjust learning rate based on gradient variance
            adjusted_lr = self.adaptive_learning_rate(local_lr)
            logger.info(f"[C{self.client_index}] Adjusted LR: {adjusted_lr:.6f} (base: {local_lr:.6f})")
            local_lr = adjusted_lr

        # Optimizer & Scheduler
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=local_lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.wd
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
        )
        
    def adaptive_learning_rate(self, base_lr):
        """Adjust learning rate based on gradient variance"""
        if len(self.grad_history) == 0:
            return base_lr
            
        # Limit the adaptive range
        lr_factor = 1.0 / (1.0 + self.grad_variance)
        adjusted_lr = base_lr * max(min(lr_factor, self.adaptive_lr_max/base_lr), self.adaptive_lr_min/base_lr)
        return adjusted_lr

    def compute_trust_score(self, local_state_dict):
        """Compute trust score for this client based on gradient stability"""
        if len(self.grad_history) < 2:
            return 1.0  # Default high trust if not enough history
            
        # Calculate variance across recent gradient norms
        # Lower variance = higher trust (more stable updates)
        variance = np.var([g for g in self.grad_history[-5:]])
        normalized_variance = min(1.0, variance / (1.0 + variance))  # Normalize to [0,1]
        
        # Trust score is inversely related to variance
        trust_score = 1.0 - normalized_variance
        
        return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

    def local_train(self, global_epoch, **kwargs):
        """Performs local training on the client's dataset"""
        self.global_epoch = global_epoch

        scaler = GradScaler(enabled=self.device.type == "cuda")  # Use AMP only if CUDA is available
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')

        gradients = []
        batch_grad_norms = []

        for local_epoch in range(self.args.trainer.local_epochs):
            epoch_grad_norms = []
            
            for images, labels in self.loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.model.zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    # Forward pass
                    output = self.model(images)
                    
                    # Extract logits from the output dictionary
                    if isinstance(output, dict):
                        logits = output["logit"]  # Extract logits tensor
                    else:
                        logits = output  # Fallback if output is not a dictionary
                    
                    # Compute loss
                    loss = self.criterion(logits, labels)

                if self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                # Calculate gradient norm for trust scoring
                grad_norm = torch.norm(torch.stack([
                    torch.norm(p.grad.detach()) 
                    for p in self.model.parameters() 
                    if p.grad is not None
                ]))
                epoch_grad_norms.append(grad_norm.item())
                
                # Collect gradients for debugging
                gradients.append([p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

                if self.device.type == "cuda":
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                loss_meter.update(loss.item(), images.size(0))

            self.scheduler.step()
            
            # Store the mean gradient norm for this epoch
            if epoch_grad_norms:
                batch_grad_norms.append(np.mean(epoch_grad_norms))

        end_time = time.time()
        
        # Update gradient history for adaptive learning rate
        if batch_grad_norms:
            mean_grad_norm = np.mean(batch_grad_norms)
            self.grad_history.append(mean_grad_norm)
            # Keep history manageable
            if len(self.grad_history) > 10:
                self.grad_history = self.grad_history[-10:]
            
            # Update variance calculation
            self.grad_variance = np.var(self.grad_history) if len(self.grad_history) > 1 else 0.0

        # Compute trust score for client updates
        trust_score = self.compute_trust_score(self.model.state_dict())

        logger.info(f"[C{self.client_index}] Training Complete. Time: {end_time - start_time:.2f}s, Loss: {loss_meter.avg:.3f}, Trust Score: {trust_score:.3f}")

        # Prepare model parameters for aggregation
        if self.enable_personalization:
            # Only share non-personalized parameters
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            # If the model supports separate global/personal params
            if hasattr(model_module, 'get_global_params'):
                # Extract only the global parameters for server aggregation
                global_params = model_module.get_global_params()
                return_dict = {k: v.cpu() for k, v in global_params.items()}
            else:
                # Fallback: return full model
                return_dict = self.model.state_dict()
        else:
            # Regular mode: return the full model
            return_dict = self.model.state_dict()

        # Filter out low-trust clients
        if self.enable_trust_filtering and trust_score < self.trust_threshold:
            logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
            return None, None

        # Move models back to CPU to save memory
        self.model = self.model.cpu()
        self.global_model = self.global_model.cpu()

        gc.collect()
        torch.cuda.empty_cache()  # Clear CUDA cache

        return return_dict, {"loss": float(loss_meter.avg), "trust_score": trust_score, "grad_variance": float(self.grad_variance)}
