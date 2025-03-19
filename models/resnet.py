# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.build import ENCODER_REGISTRY


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=True, Conv2d=nn.Conv2d):
#         super(BasicBlock, self).__init__()
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.downsample(x)
#         if not no_relu:
#             out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=True, Conv2d=nn.Conv2d):
#         super(Bottleneck, self).__init__()
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.downsample(x)
#         if not no_relu:
#             out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=True,
#                  last_feature_dim=512, **kwargs):
#         super(ResNet, self).__init__()
#         self.l2_norm = l2_norm
#         self.in_planes = 64
#         conv1_kernel_size = 3 if not use_pretrained else 7

#         Conv2d = self.get_conv()
#         self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)

#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
#         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

#         self.num_layers = 6
#         self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)
        
#         # Initialize weights properly
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def get_conv(self):
#         return nn.Conv2d
    
#     def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=True):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, return_feature=False):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         out = F.adaptive_avg_pool2d(out, 1)
#         features = out.view(out.size(0), -1)
        
#         if self.l2_norm:
#             features = F.normalize(features, p=2, dim=1)
            
#         logit = self.fc(features)

#         if return_feature:
#             return features, logit
            
#         return {"feature": features, "logit": logit}


# class PersonalizedResNet(ResNet):
#     def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, 
#                  personalization_layers=1, use_bn_layer=True, **kwargs):
#         super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, 
#                          l2_norm=l2_norm, use_bn_layer=use_bn_layer, **kwargs)
        
#         # Feature dimension after pooling
#         self.feature_dim = last_feature_dim * block.expansion
        
#         # Create personalized head with proper initialization
#         self._create_personalized_head(personalization_layers, num_classes)
        
#         # Flag to control which head to use
#         self.use_personalized_head = True  # Default to using personalized head
#         self.frozen_layers = []
        
#         # Projection head for contrastive learning
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.feature_dim, self.feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.feature_dim, 128)
#         )
        
#         # Initialize projection head
#         for m in self.projection_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def _create_personalized_head(self, personalization_layers, num_classes):
#         if personalization_layers == 1:
#             self.personalized_head = nn.Linear(self.feature_dim, num_classes)
#             # Initialize with small weights for stability
#             nn.init.normal_(self.personalized_head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(self.personalized_head.bias)
#         else:
#             layers = []
#             for i in range(personalization_layers-1):
#                 if i == 0:
#                     layers.append(nn.Linear(self.feature_dim, self.feature_dim))
#                 else:
#                     layers.append(nn.Linear(self.feature_dim, self.feature_dim))
#                 layers.append(nn.ReLU(inplace=True))
#                 layers.append(nn.BatchNorm1d(self.feature_dim))  # Add BN for stability
#             layers.append(nn.Linear(self.feature_dim, num_classes))
#             self.personalized_head = nn.Sequential(*layers)
            
#             # Initialize with small weights for stability
#             for m in self.personalized_head.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)

#     def forward(self, x, return_feature=False, get_projection=False):
#         """Forward pass with support for contrastive learning"""
#         # Extract features through backbone
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
        
#         # Global average pooling
#         out = F.adaptive_avg_pool2d(out, 1)
#         features = out.view(out.size(0), -1)
        
#         # Apply L2 normalization if needed (only to features, not weights)
#         if self.l2_norm:
#             features = F.normalize(features, p=2, dim=1)
        
#         # Get projection features for contrastive learning
#         projection_features = None
#         if get_projection:
#             projection_features = self.projection_head(features)
#             projection_features = F.normalize(projection_features, p=2, dim=1)
        
#         # Global classification
#         global_logit = self.fc(features)
        
#         # Personalized classification
#         personalized_logit = self.personalized_head(features)
        
#         # Choose which logit to use as the default based on personalization flag
#         default_logit = personalized_logit if self.use_personalized_head else global_logit
        
#         # Return a dictionary with all outputs
#         output_dict = {
#             "feature": features,  # Feature representation
#             "global_logit": global_logit,  # Global logits
#             "personalized_logit": personalized_logit,  # Personalized logits
#             "logit": default_logit,  # Default logit based on current mode
#             "use_personalized": self.use_personalized_head  # Flag for debugging
#         }
        
#         if projection_features is not None:
#             output_dict["projection"] = projection_features
        
#         if return_feature:
#             return output_dict, {"pooled": features}
#         return output_dict
    
#     def freeze_backbone(self):
#         """Freeze all layers except the classifier"""
#         for name, param in self.named_parameters():
#             if 'fc' not in name and 'personalized_head' not in name and 'projection_head' not in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True

#     def unfreeze_backbone(self):
#         """Unfreeze all layers"""
#         for param in self.parameters():
#             param.requires_grad = True
    
#     def enable_personalized_mode(self):
#         """Enable using the personalized head instead of the global head"""
#         self.use_personalized_head = True
        
#     def disable_personalized_mode(self):
#         """Disable using the personalized head, revert to global head"""
#         self.use_personalized_head = False
        
#     def get_global_params(self):
#         """Get parameters that should be shared globally"""
#         global_params = {}
#         for name, param in self.named_parameters():
#             if 'personalized_head' not in name:  # Everything except personalized head
#                 global_params[name] = param.data.clone()
#         return global_params
    
#     def get_local_params(self):
#         """Get parameters that should be kept local to the client"""
#         local_params = {}
#         for name, param in self.named_parameters():
#             if 'personalized_head' in name:  # Only personalized head
#                 local_params[name] = param.data.clone()
#         return local_params
    
#     def setup_adaptive_freezing(self, freeze_ratio=0.5):
#         """Setup adaptive layer freezing based on training progress"""
#         # Calculate how many layers to freeze based on ratio
#         all_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
#         num_to_freeze = int(len(all_layers) * freeze_ratio)
#         self.frozen_layers = all_layers[:num_to_freeze]
#         self.freeze_layers(self.frozen_layers)
        
#     def freeze_layers(self, layer_names):
#         """Freeze specific layers by name"""
#         for name, param in self.named_parameters():
#             if any(layer in name for layer in layer_names):
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
    
#     def get_contrastive_features(self, x):
#         """Extract features for contrastive learning"""
#         # Extract features through backbone
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
        
#         # Global average pooling
#         out = F.adaptive_avg_pool2d(out, 1)
#         features = out.view(out.size(0), -1)
        
#         # Project features for contrastive learning
#         projected = self.projection_head(features)
#         projected = F.normalize(projected, p=2, dim=1)
        
#         return projected


# @ENCODER_REGISTRY.register()
# class ResNet18(ResNet):
#     def __init__(self, args, num_classes=10, **kwargs):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


# @ENCODER_REGISTRY.register()
# class PersonalizedResNet18(PersonalizedResNet):
#     def __init__(self, args, num_classes=10, **kwargs):
#         # First check if personalization_layers is in kwargs
#         personalization_layers = kwargs.pop('personalization_layers', None)
        
#         # If not in kwargs, try to get from args
#         if personalization_layers is None:
#             personalization_layers = getattr(args, 'personalization_layers', 1)
        
#         # Now pass it explicitly to the parent class
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
#                          personalization_layers=personalization_layers, **kwargs)

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
        
        # Track gradient history for cyclical learning rate
        self.rounds_trained = 0
        self.previous_model_state = None
        self.update_history = []

        # Initialize personalization settings
        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)

        # Initialize cyclical learning rate settings
        self.enable_cyclical_lr = True
        self.base_lr = 0.001
        self.max_lr = 0.1
        self.step_size = 10  # Number of rounds for half cycle
        
        # Initialize FedProx regularization
        self.enable_fedprox = True
        self.fedprox_mu = 0.01  # Regularization strength

        # Initialize knowledge distillation
        self.enable_distillation = True
        self.distillation_temp = 2.0
        self.distillation_weight = 0.5

        self.model = model
        self.global_model = copy.deepcopy(model)
        self.device = torch.device("cpu")  # Default to CPU

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
        
        # Add relaxed contrastive loss with more conservative parameters
        self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
            temperature=0.1,  # Increased for more stable gradients
            lambda_penalty=0.05,  # Reduced to prevent dominating the loss
            similarity_threshold=0.5  # Reduced to be less aggressive
        )
        
        # Track metrics for debugging
        self.ce_loss_avg = 0.0
        self.rcl_loss_avg = 0.0
        self.distillation_loss_avg = 0.0
        self.fedprox_loss_avg = 0.0

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        """Initialize client model, dataset, and optimizer"""
        # Store device for later use
        self.device = device
        self.rounds_trained += 1
        
        # Store previous model state for trust score calculation
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            # Make sure previous model state is on the same device as the current model
            self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        # Move models to the correct device
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
                freeze_ratio = max(0.0, 0.8 - (self.rounds_trained / 50.0))  # Gradually unfreeze
                model_module.setup_adaptive_freezing(freeze_ratio=freeze_ratio)

        self.trainer = trainer
        self.global_epoch = global_epoch

        # Check if dataset is empty and handle gracefully
        if local_dataset is None or len(local_dataset) == 0:
            logger.warning(f"[C{self.client_index}] Empty dataset provided. Creating a dummy loader.")
            self.loader = None
        else:
            # Create DataLoader with proper settings
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
                if hasattr(self.args.dataset, 'num_instances') and self.args.dataset.num_instances > 0 else None
                
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

        # Apply cyclical learning rate if enabled
        if self.enable_cyclical_lr:
            # Calculate current position in cycle
            cycle_step = self.rounds_trained % (2 * self.step_size)
            # Apply cyclical learning rate formula
            x = abs(cycle_step / self.step_size - 1)
            adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
            logger.info(f"[C{self.client_index}] Cyclical LR: {adjusted_lr:.6f} (cycle step: {cycle_step})")
            local_lr = adjusted_lr
        else:
            logger.info(f"[C{self.client_index}] Using base LR: {local_lr:.6f}")

        # Set up optimizer based on personalization mode
        if self.enable_personalization:
            # Only optimize personalized parameters
            if hasattr(model_module, 'get_local_params'):
                # Get personalized parameters
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
            
        # Increase local epochs for early rounds to establish better representations
        if self.rounds_trained < 5:
            self.local_epochs = min(20, self.args.trainer.local_epochs * 2)
        else:
            self.local_epochs = self.args.trainer.local_epochs
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
        )

    def compute_fedprox_term(self):
        """Compute FedProx regularization term"""
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)**2
        return self.fedprox_mu * proximal_term / 2

    def compute_distillation_loss(self, student_logits, teacher_logits):
        """Compute knowledge distillation loss"""
        temp = self.distillation_temp
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        log_probs = F.log_softmax(student_logits / temp, dim=1)
        distillation_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return distillation_loss * (temp ** 2)

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
                # Ensure both tensors are on the same device
                current_param = current_state[key].to(self.device).float()
                prev_param = self.previous_model_state[key].to(self.device).float()
                diff = current_param - prev_param
                update_norm += torch.norm(diff).item() ** 2
                param_count += diff.numel()
        
        if param_count > 0:
            update_norm = (update_norm / param_count) ** 0.5
        
        # Track update norms for variance calculation
        self.update_history.append(update_norm)
        if len(self.update_history) > 5:
            self.update_history.pop(0)
            
        # Calculate variance of updates
        update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
        # Trust score combines magnitude and consistency
        magnitude_score = 1.0 / (1.0 + update_norm)
        consistency_score = 1.0 / (1.0 + update_variance)
        
        # Weighted combination
        trust_score = 0.7 * magnitude_score + 0.3 * consistency_score
        
        # Update previous model state for next calculation
        self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
        return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

    def local_train(self, global_epoch, **kwargs):
        """Performs local training on the client's dataset"""
        self.global_epoch = global_epoch

        scaler = GradScaler(enabled=self.device.type == "cuda")  # Use AMP only if CUDA is available
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.4f')
        ce_loss_meter = AverageMeter('CE Loss', ':.4f')
        rcl_loss_meter = AverageMeter('RCL Loss', ':.4f')
        distillation_loss_meter = AverageMeter('Distill Loss', ':.4f')
        fedprox_loss_meter = AverageMeter('FedProx Loss', ':.4f')

        # Skip training if loader is empty
        if self.loader is None or len(self.loader) == 0:
            logger.warning(f"[C{self.client_index}] No data for training!")
            return None, None

        for local_epoch in range(self.local_epochs):
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
                    # Forward pass with model
                    output = self.model(images, get_projection=True)
                    
                    # Extract logits from the output dictionary
                    if isinstance(output, dict):
                        logits = output["logit"] if "logit" in output else output.get("personalized_logit", output)
                        features = output.get("feature", None)
                        projection = output.get("projection", None)
                    else:
                        logits = output
                        features = None
                        projection = None
                    
                    # Compute cross-entropy loss
                    ce_loss = self.criterion(logits, labels)
                    
                    # Compute relaxed contrastive loss if enabled
                    rcl_loss = 0.0
                    if hasattr(self.args.client, 'rcl_loss') and getattr(self.args.client.rcl_loss, 'weight', 0) > 0:
                        # Use projection features if available, otherwise use regular features
                        contrastive_features = projection if projection is not None else features
                        
                        if contrastive_features is not None and len(contrastive_features) >= 2:
                            # Apply relaxed contrastive loss with a small weight
                            rcl_weight = getattr(self.args.client.rcl_loss, 'weight', 0.1)
                            rcl_loss = self.relaxed_contrastive_loss(contrastive_features, labels) * rcl_weight
                    
                    # Compute knowledge distillation loss if enabled
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
                    
                    # Compute FedProx regularization if enabled
                    fedprox_loss = 0.0
                    if self.enable_fedprox:
                        fedprox_loss = self.compute_fedprox_term()
                    
                    # Total loss with proper weighting
                    loss = ce_loss + rcl_loss + distillation_loss + fedprox_loss
                    
                    # Update loss meters
                    ce_loss_meter.update(ce_loss.item(), images.size(0))
                    if rcl_loss > 0:
                        rcl_loss_meter.update(rcl_loss.item(), images.size(0))
                    if distillation_loss > 0:
                        distillation_loss_meter.update(distillation_loss.item(), images.size(0))
                    if fedprox_loss > 0:
                        fedprox_loss_meter.update(fedprox_loss.item(), images.size(0))
                
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # Reduced from 10.0

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
                logger.info(f"[C{self.client_index}] Epoch {local_epoch+1}/{self.local_epochs}, Loss: {epoch_loss/samples_processed:.4f}")
            else:
                logger.warning(f"[C{self.client_index}] No samples processed in epoch {local_epoch+1}")
            
            self.scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        
        # Store average losses for reporting
        self.ce_loss_avg = ce_loss_meter.avg
        self.rcl_loss_avg = rcl_loss_meter.avg
        self.distillation_loss_avg = distillation_loss_meter.avg
        self.fedprox_loss_avg = fedprox_loss_meter.avg
        
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache
        
            return return_dict, {"loss": float(loss_meter.avg), 
                                "trust_score": trust_score, 
                                "ce_loss": float(self.ce_loss_avg),
                                "rcl_loss": float(self.rcl_loss_avg),
                                "distillation_loss": float(self.distillation_loss_avg),
                                "fedprox_loss": float(self.fedprox_loss_avg)}


