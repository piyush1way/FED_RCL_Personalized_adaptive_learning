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

        # Initialize trust filtering settings
        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        
        # Track gradient history for adaptive learning rate
        self.grad_history = []
        self.grad_variance = 0.0
        self.previous_model_state = None

        # Initialize personalization settings
        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)

        # Initialize adaptive learning rate settings
        adaptive_lr_config = getattr(args.client, "adaptive_lr", {})
        self.enable_adaptive_lr = getattr(adaptive_lr_config, "enable", False)
        self.adaptive_lr_beta = getattr(adaptive_lr_config, "beta", 0.1)
        self.adaptive_lr_min = getattr(adaptive_lr_config, "min_lr", 0.001)
        self.adaptive_lr_max = getattr(adaptive_lr_config, "max_lr", 0.1)

        self.model = model
        self.global_model = copy.deepcopy(model)

        self.rcl_criterions = {'scl': None, 'penalty': None}
        args_rcl = args.client.rcl_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_rcl.pairs:
            self.pairs[pair.name] = pair
            self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Add relaxed contrastive loss
        self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
            temperature=0.05,
            lambda_penalty=1.0,
            similarity_threshold=0.7
        )

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        """Initialize client model, dataset, and optimizer"""
        # Store previous model state for trust score calculation
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            self.previous_model_state = copy.deepcopy(self.model.state_dict())
        
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
            elif hasattr(model_module, 'use_personalized_head'):
                model_module.use_personalized_head = True
                
            # Apply freezing strategies
            if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
                model_module.freeze_backbone()
                
            if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
                model_module.setup_adaptive_freezing()

        self.trainer = trainer
        self.global_epoch = global_epoch

        # Check if dataset is empty and handle gracefully
        if local_dataset is None or len(local_dataset) == 0:
            logger.warning(f"[C{self.client_index}] Empty dataset provided. Creating a dummy loader.")
            self.loader = None
        else:
            # Create DataLoader with proper settings
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
                if self.args.dataset.num_instances > 0 else None

            # Configure DataLoader to drop incomplete batches
            self.loader = DataLoader(
                local_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                num_workers=0,  # Reduced workers to avoid issues
                pin_memory=False,  # Disabled pin_memory for stability
                drop_last=True  # Drop incomplete batches to ensure consistent sizes
            )

        # Apply adaptive learning rate if enabled
        if self.enable_adaptive_lr:
            # Adjust learning rate based on gradient variance
            adjusted_lr = self.adaptive_learning_rate(local_lr)
            logger.info(f"[C{self.client_index}] Adjusted LR: {adjusted_lr:.6f} (base: {local_lr:.6f}, variance: {self.grad_variance:.6f})")
            local_lr = adjusted_lr

        # Set up optimizer based on personalization mode
        if self.enable_personalization:
            # Only optimize personalized parameters
            if hasattr(model_module, 'get_local_params'):
                # Get personalized parameters
                personalized_params = []
                for name, param in self.model.named_parameters():
                    if 'personalized_head' in name:
                        personalized_params.append(param)
                
                if personalized_params:
                    self.optimizer = torch.optim.SGD(
                        personalized_params,
                        lr=local_lr,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
                else:
                    # Fallback to all parameters if no personalized params found
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=local_lr,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
            else:
                # Fallback to all parameters
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=local_lr,
                    momentum=self.args.optimizer.momentum,
                    weight_decay=self.args.optimizer.wd
                )
        else:
            # Regular mode: optimize all parameters
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
        if len(self.grad_history) < 2:
            return base_lr
            
        # Calculate adjustment factor based on gradient variance
        lr_factor = 1.0 / (1.0 + self.adaptive_lr_beta * self.grad_variance)
        
        # Limit the adaptive range
        adjusted_lr = base_lr * max(min(lr_factor, self.adaptive_lr_max/base_lr), self.adaptive_lr_min/base_lr)
        return adjusted_lr

    def compute_trust_score(self):
        """Compute trust score for this client based on model update consistency"""
        if self.previous_model_state is None:
            return 1.0  # Default high trust if no previous state
        
        # Get current model state
        current_state = self.model.state_dict()
        
        # Calculate update norm (L2 distance between current and previous model)
        update_norm = 0.0
        param_count = 0
        
        for key in current_state:
            if 'personalized_head' not in key and key in self.previous_model_state:
                diff = current_state[key] - self.previous_model_state[key]
                update_norm += torch.norm(diff).item() ** 2
                param_count += diff.numel()
        
        if param_count > 0:
            update_norm = (update_norm / param_count) ** 0.5
        
        # Trust score is inversely related to update norm
        trust_score = 1.0 / (1.0 + update_norm)
        
        return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

    def store_gradients(self):
        """Store current gradients for variance calculation"""
        grad_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += torch.norm(param.grad).item() ** 2
                param_count += param.grad.numel()
        
        if param_count > 0:
            grad_norm = (grad_norm / param_count) ** 0.5
            self.grad_history.append(grad_norm)
            
            # Keep history manageable
            if len(self.grad_history) > 10:
                self.grad_history = self.grad_history[-10:]
            
            # Update variance calculation
            self.grad_variance = np.var(self.grad_history) if len(self.grad_history) > 1 else 0.0

    def local_train(self, global_epoch, **kwargs):
        """Performs local training on the client's dataset"""
        self.global_epoch = global_epoch

        scaler = GradScaler(enabled=self.device.type == "cuda")  # Use AMP only if CUDA is available
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')

        # Skip training if loader is empty
        if self.loader is None or len(self.loader) == 0:
            logger.warning(f"[C{self.client_index}] No data for training!")
            return None, None

        for local_epoch in range(self.args.trainer.local_epochs):
            epoch_loss = 0.0
            samples_processed = 0
            
            for images, labels in self.loader:
                # Skip batches that are too small for contrastive learning
                if len(images) < 2:
                    continue
                    
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.model.zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    # Forward pass
                    output = self.model(images)
                    
                    # Extract logits from the output dictionary
                    if isinstance(output, dict):
                        logits = output["logit"] if "logit" in output else output.get("personalized_logit", output)
                        features = output.get("feature", None)
                    else:
                        logits = output
                        features = None
                    
                    # Compute cross-entropy loss
                    ce_loss = self.criterion(logits, labels)
                    
                    # Compute relaxed contrastive loss if enabled
                    rcl_loss = 0.0
                    if hasattr(self.args.client, 'rcl_loss') and self.args.client.rcl_loss.weight > 0:
                        if features is not None and len(features) >= 2:
                            # Use relaxed contrastive loss for better representation learning
                            rcl_loss = self.relaxed_contrastive_loss(features, labels)
                            
                            # Also use traditional contrastive loss if configured
                            for pair_name, criterion in self.rcl_criterions.items():
                                if pair_name in self.pairs:
                                    pair = self.pairs[pair_name]
                                    try:
                                        # Get lambda_weight with fallback to 1.0 if not present
                                        lambda_weight = getattr(pair, 'weight', 1.0)
                                        pair_loss = criterion(features, features, labels)
                                        rcl_loss += pair_loss * lambda_weight
                                    except Exception as e:
                                        logger.warning(f"[C{self.client_index}] Error in contrastive loss: {e}")
                    
                    # Total loss
                    loss = ce_loss + rcl_loss
                
                # Check if loss is valid
                if not torch.isfinite(loss):
                    logger.warning(f"[C{self.client_index}] Loss is {loss}, skipping batch")
                    continue
                    
                # Backward and optimize
                if self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                
                # Store gradients for adaptive learning rate
                if self.enable_adaptive_lr:
                    self.store_gradients()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

                if self.device.type == "cuda":
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                # Update metrics
                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                epoch_loss += loss.item() * batch_size
                samples_processed += batch_size

            # Log per-epoch statistics
            if samples_processed > 0:
                logger.info(f"[C{self.client_index}] Epoch {local_epoch+1}/{self.args.trainer.local_epochs}, Loss: {epoch_loss/samples_processed:.4f}")
            else:
                logger.warning(f"[C{self.client_index}] No samples processed in epoch {local_epoch+1}")
            
            self.scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        
        # Compute trust score for client updates
        trust_score = self.compute_trust_score() if self.enable_trust_filtering else 1.0

        logger.info(f"[C{self.client_index}] Training Complete. Time: {training_time:.2f}s, Loss: {loss_meter.avg:.4f}, Trust Score: {trust_score:.3f}")

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
                # Fallback: filter out personalized head parameters
                return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
                              if 'personalized_head' not in k}
        else:
            # Regular mode: return the full model
            return_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

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
