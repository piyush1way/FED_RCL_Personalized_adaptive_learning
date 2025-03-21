# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.build import ENCODER_REGISTRY
# from typing import Dict, List, Optional
# from omegaconf import DictConfig
# import logging

# logger = logging.getLogger(__name__)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=True):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()
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

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()
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
#     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_bn_layer=True,
#                  last_feature_dim=512, personalization_layers=2, **kwargs):
#         super(ResNet, self).__init__()
#         self.l2_norm = l2_norm
#         self.in_planes = 64
#         self.num_classes = num_classes
#         self.use_bn_layer = use_bn_layer

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64) if use_bn_layer else nn.Identity()
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2)
        
#         self.feature_dim = last_feature_dim * block.expansion
#         self.fc = nn.Linear(self.feature_dim, num_classes)
        
#         self.temperature = nn.Parameter(torch.ones(1) * 0.07)
#         self._create_personalized_head(personalization_layers)
        
#         self.use_personalized_head = True
#         self.frozen_layers = []
#         self.trust_score = 1.0  # Initialize trust score for adaptive learning rate
        
#         self.projection_head = self._create_projection_head()
#         self.multi_level_projections = self._create_multi_level_projections()
        
#         self._initialize_weights()

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, use_bn_layer=self.use_bn_layer))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def _create_personalized_head(self, personalization_layers):
#         if personalization_layers == 1:
#             self.personalized_head = nn.Linear(self.feature_dim, self.num_classes)
#         else:
#             layers = []
#             for i in range(personalization_layers-1):
#                 layers.extend([
#                     nn.Linear(self.feature_dim, self.feature_dim),
#                     nn.BatchNorm1d(self.feature_dim) if self.use_bn_layer else nn.Identity(),
#                     nn.ReLU(inplace=True)
#                 ])
#             layers.append(nn.Linear(self.feature_dim, self.num_classes))
#             self.personalized_head = nn.Sequential(*layers)
        
#         self._initialize_head(self.personalized_head)

#     def _create_projection_head(self):
#         return nn.Sequential(
#             nn.Linear(self.feature_dim, self.feature_dim),
#             nn.BatchNorm1d(self.feature_dim) if self.use_bn_layer else nn.Identity(),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.feature_dim, 256),
#             nn.BatchNorm1d(256) if self.use_bn_layer else nn.Identity(),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128)
#         )
    
#     def _create_multi_level_projections(self):
#         """Create projection heads for intermediate layers to support multi-level contrastive learning"""
#         projections = nn.ModuleDict({
#             'layer1': nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(64 * self.expansion, 128),
#                 nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(128, 128)
#             ),
#             'layer2': nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(128 * self.expansion, 128),
#                 nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(128, 128)
#             ),
#             'layer3': nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(),
#                 nn.Linear(256 * self.expansion, 128),
#                 nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(128, 128)
#             )
#         })
#         return projections

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

#     def _initialize_head(self, head):
#         if isinstance(head, nn.Linear):
#             nn.init.normal_(head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(head.bias)
#         elif isinstance(head, nn.Sequential):
#             for m in head.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor, return_feature: bool = False, get_projection: bool = False, 
#                 get_multi_level: bool = False) -> Dict[str, torch.Tensor]:
#         results = {}

#         out = F.relu(self.bn1(self.conv1(x)))
#         results['layer0'] = out

#         for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
#             out = layer(out)
#             results[f'layer{i+1}'] = out

#         out = F.adaptive_avg_pool2d(out, 1)
#         features = out.view(out.size(0), -1)
        
#         features_normalized = F.normalize(features, p=2, dim=1) if self.l2_norm else features
        
#         projection_features = None
#         if get_projection:
#             projection_features = self.projection_head(features)
#             projection_features = F.normalize(projection_features, p=2, dim=1)
        
#         # Get multi-level projections if requested
#         if get_multi_level:
#             multi_level_projections = {}
#             for layer_name in ['layer1', 'layer2', 'layer3']:
#                 if layer_name in results:
#                     proj = self.multi_level_projections[layer_name](results[layer_name])
#                     multi_level_projections[layer_name] = F.normalize(proj, p=2, dim=1)
#             results['multi_level_projections'] = multi_level_projections
        
#         global_logit = self.fc(features)
#         personalized_logit = self.personalized_head(features)
#         default_logit = personalized_logit if self.use_personalized_head else global_logit
        
#         results.update({
#             "feature": features,
#             "feature_normalized": features_normalized,
#             "global_logit": global_logit,
#             "personalized_logit": personalized_logit,
#             "logit": default_logit,
#             "use_personalized": self.use_personalized_head,
#             "temperature": self.temperature,
#             "trust_score": self.trust_score
#         })
        
#         if projection_features is not None:
#             results["projection"] = projection_features
        
#         if return_feature:
#             return results, {"pooled": features}
#         return results

#     def freeze_backbone(self):
#         for name, param in self.named_parameters():
#             if 'fc' not in name and 'personalized_head' not in name and 'projection_head' not in name and 'temperature' not in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
#         logger.info('Froze backbone parameters (except fc, personalized_head, projection_head, and temperature)')

#     def unfreeze_backbone(self):
#         for param in self.parameters():
#             param.requires_grad = True
#         logger.info('Unfroze all parameters')
    
#     def enable_personalized_mode(self):
#         self.use_personalized_head = True
        
#     def disable_personalized_mode(self):
#         self.use_personalized_head = False
        
#     def get_global_params(self):
#         return {name: param.data.clone() for name, param in self.named_parameters() if 'personalized_head' not in name}
    
#     def get_local_params(self):
#         return {name: param.data.clone() for name, param in self.named_parameters() if 'personalized_head' in name}
    
#     def setup_adaptive_freezing(self, freeze_ratio=0.5):
#         all_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
#         num_to_freeze = int(len(all_layers) * freeze_ratio)
#         self.frozen_layers = all_layers[:num_to_freeze]
#         self.freeze_layers(self.frozen_layers)
#         logger.info(f"Adaptively freezing layers: {self.frozen_layers}")
        
#     def freeze_layers(self, layer_names):
#         for name, param in self.named_parameters():
#             param.requires_grad = not any(layer in name for layer in layer_names)
    
#     def get_contrastive_features(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
        
#         out = F.adaptive_avg_pool2d(out, 1)
#         features = out.view(out.size(0), -1)
        
#         projected = self.projection_head(features)
#         projected = F.normalize(projected, p=2, dim=1)
        
#         return projected
    
#     def get_multi_level_contrastive_features(self, x):
#         """Extract contrastive features from multiple network levels"""
#         results = {}
        
#         out = F.relu(self.bn1(self.conv1(x)))
#         layer_features = {}
        
#         # Get features from each layer
#         out1 = self.layer1(out)
#         layer_features['layer1'] = self.multi_level_projections['layer1'](out1)
        
#         out2 = self.layer2(out1)
#         layer_features['layer2'] = self.multi_level_projections['layer2'](out2)
        
#         out3 = self.layer3(out2)
#         layer_features['layer3'] = self.multi_level_projections['layer3'](out3)
        
#         out4 = self.layer4(out3)
#         out = F.adaptive_avg_pool2d(out4, 1)
#         features = out.view(out.size(0), -1)
        
#         # Final layer projection
#         layer_features['layer4'] = self.projection_head(features)
        
#         # Normalize all features
#         for layer_name in layer_features:
#             layer_features[layer_name] = F.normalize(layer_features[layer_name], p=2, dim=1)
            
#         return layer_features
    
#     def update_trust_score(self, new_score):
#         """Update the trust score for adaptive learning rate"""
#         self.trust_score = new_score
#         logger.debug(f"Updated trust score to {self.trust_score}")

# @ENCODER_REGISTRY.register()
# class ResNet18(ResNet):
#     def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
#                          l2_norm=args.model.l2_norm,
#                          use_bn_layer=args.model.use_bn_layer,
#                          personalization_layers=args.model.personalization_layers,
#                          **kwargs)

# @ENCODER_REGISTRY.register()
# class ResNet34(ResNet):
#     def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
#         super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, 
#                          l2_norm=args.model.l2_norm,
#                          use_bn_layer=args.model.use_bn_layer,
#                          personalization_layers=args.model.personalization_layers,
#                          **kwargs)

# @ENCODER_REGISTRY.register()
# class PersonalizedResNet18(ResNet18):
#     expansion = 1
#     def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
#         super().__init__(args, num_classes=num_classes, **kwargs)
#         # Enable personalized mode by default
#         # self.expansion = 1 
#         self.use_personalized_head = True
        
#         # Enhanced personalization options
#         personalization_config = getattr(args.model, "personalization", {})
#         self.use_distillation = getattr(personalization_config, "knowledge_distillation", True)
#         self.distillation_temp = getattr(personalization_config, "kd_temperature", 3.0)
        
#         # Trust-based settings
#         self.trust_score = 1.0  # Initialize with full trust
        
#         # Adaptive layer freezing
#         adaptive_config = getattr(args.model, "adaptive_freezing", {})
#         self.use_adaptive_freezing = getattr(adaptive_config, "enable", False)
#         self.initial_freeze_ratio = getattr(adaptive_config, "initial_freeze_ratio", 0.5)
#         self.freeze_decay_rate = getattr(adaptive_config, "decay_rate", 0.05)
        
#         if self.use_adaptive_freezing:
#             self.setup_adaptive_freezing(self.initial_freeze_ratio)
            
#         logger.info(f"Initialized PersonalizedResNet18 with personalization_layers={args.model.personalization_layers}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY
from typing import Dict, List, Optional
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()
            )

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if use_bn_layer else nn.Identity()
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_bn_layer else nn.Identity()
            )

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_bn_layer=True,
                 last_feature_dim=512, personalization_layers=2, **kwargs):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        self.num_classes = num_classes
        self.use_bn_layer = use_bn_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_bn_layer else nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2)
        
        self.feature_dim = last_feature_dim * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self._create_personalized_head(personalization_layers)
        
        self.use_personalized_head = True
        self.frozen_layers = []
        self.trust_score = 1.0  # Initialize trust score for adaptive learning rate
        
        self.projection_head = self._create_projection_head()
        self.multi_level_projections = self._create_multi_level_projections()
        
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=self.use_bn_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _create_personalized_head(self, personalization_layers):
        if personalization_layers == 1:
            self.personalized_head = nn.Linear(self.feature_dim, self.num_classes)
        else:
            layers = []
            for i in range(personalization_layers-1):
                layers.extend([
                    nn.Linear(self.feature_dim, self.feature_dim),
                    nn.BatchNorm1d(self.feature_dim) if self.use_bn_layer else nn.Identity(),
                    nn.ReLU(inplace=True)
                ])
            layers.append(nn.Linear(self.feature_dim, self.num_classes))
            self.personalized_head = nn.Sequential(*layers)
        
        self._initialize_head(self.personalized_head)

    def _create_projection_head(self):
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim) if self.use_bn_layer else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256) if self.use_bn_layer else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
    
    def _create_multi_level_projections(self):
        """Create projection heads for intermediate layers to support multi-level contrastive learning"""
        projections = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64 * self.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer2': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128 * self.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer3': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256 * self.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            )
        })
        return projections

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_head(self, head):
        if isinstance(head, nn.Linear):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
        elif isinstance(head, nn.Sequential):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_feature: bool = False, get_projection: bool = False, 
                get_multi_level: bool = False) -> Dict[str, torch.Tensor]:
        results = {}

        out = F.relu(self.bn1(self.conv1(x)))
        results['layer0'] = out

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            out = layer(out)
            results[f'layer{i+1}'] = out

        out = F.adaptive_avg_pool2d(out, 1)
        features = out.view(out.size(0), -1)
        
        features_normalized = F.normalize(features, p=2, dim=1) if self.l2_norm else features
        
        projection_features = None
        if get_projection:
            projection_features = self.projection_head(features)
            projection_features = F.normalize(projection_features, p=2, dim=1)
        
        # Get multi-level projections if requested
        if get_multi_level:
            multi_level_projections = {}
            for layer_name in ['layer1', 'layer2', 'layer3']:
                if layer_name in results:
                    proj = self.multi_level_projections[layer_name](results[layer_name])
                    multi_level_projections[layer_name] = F.normalize(proj, p=2, dim=1)
            results['multi_level_projections'] = multi_level_projections
        
        global_logit = self.fc(features)
        personalized_logit = self.personalized_head(features)
        default_logit = personalized_logit if self.use_personalized_head else global_logit
        
        results.update({
            "feature": features,
            "feature_normalized": features_normalized,
            "global_logit": global_logit,
            "personalized_logit": personalized_logit,
            "logit": default_logit,
            "use_personalized": self.use_personalized_head,
            "temperature": self.temperature,
            "trust_score": self.trust_score
        })
        
        if projection_features is not None:
            results["projection"] = projection_features
        
        if return_feature:
            return results, {"pooled": features}
        return results

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'fc' not in name and 'personalized_head' not in name and 'projection_head' not in name and 'temperature' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        logger.info('Froze backbone parameters (except fc, personalized_head, projection_head, and temperature)')

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info('Unfroze all parameters')
    
    def enable_personalized_mode(self):
        self.use_personalized_head = True
        
    def disable_personalized_mode(self):
        self.use_personalized_head = False
        
    def get_global_params(self):
        return {name: param.data.clone() for name, param in self.named_parameters() if 'personalized_head' not in name}
    
    def get_local_params(self):
        return {name: param.data.clone() for name, param in self.named_parameters() if 'personalized_head' in name}
    
    def setup_adaptive_freezing(self, freeze_ratio=0.5):
        all_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        num_to_freeze = int(len(all_layers) * freeze_ratio)
        self.frozen_layers = all_layers[:num_to_freeze]
        self.freeze_layers(self.frozen_layers)
        logger.info(f"Adaptively freezing layers: {self.frozen_layers}")
        
    def freeze_layers(self, layer_names):
        for name, param in self.named_parameters():
            param.requires_grad = not any(layer in name for layer in layer_names)
    
    def get_contrastive_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.view(out.size(0), -1)
        
        projected = self.projection_head(features)
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected
    
    def get_multi_level_contrastive_features(self, x):
        """Extract contrastive features from multiple network levels"""
        results = {}
        
        out = F.relu(self.bn1(self.conv1(x)))
        layer_features = {}
        
        # Get features from each layer
        out1 = self.layer1(out)
        layer_features['layer1'] = self.multi_level_projections['layer1'](out1)
        
        out2 = self.layer2(out1)
        layer_features['layer2'] = self.multi_level_projections['layer2'](out2)
        
        out3 = self.layer3(out2)
        layer_features['layer3'] = self.multi_level_projections['layer3'](out3)
        
        out4 = self.layer4(out3)
        out = F.adaptive_avg_pool2d(out4, 1)
        features = out.view(out.size(0), -1)
        
        # Final layer projection
        layer_features['layer4'] = self.projection_head(features)
        
        # Normalize all features
        for layer_name in layer_features:
            layer_features[layer_name] = F.normalize(layer_features[layer_name], p=2, dim=1)
            
        return layer_features
    
    def update_trust_score(self, new_score):
        """Update the trust score for adaptive learning rate"""
        self.trust_score = new_score
        logger.debug(f"Updated trust score to {self.trust_score}")

@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                         l2_norm=args.model.l2_norm,
                         use_bn_layer=args.model.use_bn_layer,
                         personalization_layers=args.model.personalization_layers,
                         **kwargs)

@ENCODER_REGISTRY.register()
class ResNet34(ResNet):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, 
                         l2_norm=args.model.l2_norm,
                         use_bn_layer=args.model.use_bn_layer,
                         personalization_layers=args.model.personalization_layers,
                         **kwargs)

@ENCODER_REGISTRY.register()
class PersonalizedResNet18(ResNet18):
    expansion = 1
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(args, num_classes=num_classes, **kwargs)
        
        # Personalization settings
        personalization_config = getattr(args.model, "personalization", {})
        self.use_personalized_head = True
        self.personalization_layers = getattr(personalization_config, "layers", 2)
        
        # Create separate heads for global and personalized predictions
        self.global_head = nn.Sequential(
            nn.Linear(512 * self.expansion, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self.personalized_head = nn.Sequential(
            nn.Linear(512 * self.expansion, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Enhanced personalization options
        self.use_distillation = getattr(personalization_config, "knowledge_distillation", True)
        self.distillation_temp = getattr(personalization_config, "kd_temperature", 3.0)
        
        # Trust-based settings
        self.trust_score = 1.0
        
        # Adaptive layer freezing
        adaptive_config = getattr(args.model, "adaptive_freezing", {})
        self.use_adaptive_freezing = getattr(adaptive_config, "enable", False)
        self.initial_freeze_ratio = getattr(adaptive_config, "initial_freeze_ratio", 0.5)
        self.freeze_decay_rate = getattr(adaptive_config, "decay_rate", 0.05)
        
        if self.use_adaptive_freezing:
            self.setup_adaptive_freezing(self.initial_freeze_ratio)
            
        logger.info(f"Initialized PersonalizedResNet18 with personalization_layers={self.personalization_layers}")
    
    def forward(self, x):
        # Use parent class's feature extraction
        results = super().forward(x)
        
        # Extract features
        features = results['feature']
        
        # Get predictions from both heads
        global_logits = self.global_head(features)
        personalized_logits = self.personalized_head(features)
        
        # Determine which logits to use as default
        default_logits = personalized_logits if self.use_personalized_head else global_logits
        
        # Return consistent dictionary structure
        return {
            "logit": default_logits,  # Always include default logits
            "global_logit": global_logits,
            "personalized_logit": personalized_logits,
            "features": features,
            "feature": features,  # Keep both feature keys for compatibility
            "feature_normalized": F.normalize(features, p=2, dim=1) if hasattr(self, 'l2_norm') and self.l2_norm else features,
            "use_personalized": self.use_personalized_head,
            "trust_score": self.trust_score
        }
    
    def enable_personalized_mode(self):
        """Enable personalized mode"""
        self.use_personalized_head = True
        # Freeze backbone layers if configured
        if self.use_adaptive_freezing:
            self.setup_adaptive_freezing(self.initial_freeze_ratio)
    
    def disable_personalized_mode(self):
        """Disable personalized mode"""
        self.use_personalized_head = False
        # Unfreeze all layers
        for param in self.parameters():
            param.requires_grad = True
    
    def setup_adaptive_freezing(self, freeze_ratio=0.5):
        """Setup adaptive layer freezing based on trust scores"""
        if not self.use_adaptive_freezing:
            return
            
        # List of layers to potentially freeze
        layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        num_layers_to_freeze = int(len(layers_to_freeze) * freeze_ratio)
        
        # Freeze selected layers
        frozen_layers = layers_to_freeze[:num_layers_to_freeze]
        for name, param in self.named_parameters():
            if any(layer in name for layer in frozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        if frozen_layers:
            logger.info(f"Adaptively freezing layers: {frozen_layers}")
            
    def get_personalization_params(self):
        """Get parameters for personalization"""
        if self.use_personalized_head:
            return list(self.personalized_head.parameters())
        return list(self.parameters())


