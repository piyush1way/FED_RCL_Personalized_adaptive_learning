import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from utils import *
import numpy as np
from models.resnet import BasicBlock, Bottleneck, ResNet
from models.build import ENCODER_REGISTRY
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

class ResNet_base(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_bn_layer=True,
                 last_feature_dim=512, personalization_layers=2, **kwargs):
        super().__init__(block, num_blocks, num_classes=num_classes, l2_norm=l2_norm,
                         use_bn_layer=use_bn_layer, last_feature_dim=last_feature_dim, 
                         personalization_layers=personalization_layers, **kwargs)
        
        self.feature_dim = last_feature_dim * block.expansion
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self._create_personalized_head(personalization_layers, num_classes)
        
        self.use_personalized_head = True
        self.frozen_layers = []
        self.trust_score = 1.0  # Initialize trust score for adaptive learning rate
        
        self.projection_head = self._create_projection_head()
        self.multi_level_projections = self._create_multi_level_projections()

    def _create_personalized_head(self, personalization_layers, num_classes):
        if personalization_layers == 1:
            self.personalized_head = nn.Linear(self.feature_dim, num_classes)
        else:
            layers = []
            for i in range(personalization_layers-1):
                layers.extend([
                    nn.Linear(self.feature_dim, self.feature_dim),
                    nn.BatchNorm1d(self.feature_dim) if self.use_bn_layer else nn.Identity(),
                    nn.ReLU(inplace=True)
                ])
            layers.append(nn.Linear(self.feature_dim, num_classes))
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
                nn.Linear(64 * block.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer2': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128 * block.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer3': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256 * block.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            )
        })
        return projections

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
        logger.warning('Freeze backbone parameters (except fc, personalized_head, projection_head, and temperature)')

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
    
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
class ResNet18_base(ResNet_base):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                         l2_norm=args.model.l2_norm,
                         use_bn_layer=args.model.use_bn_layer,
                         personalization_layers=args.model.personalization_layers,
                         **kwargs)

@ENCODER_REGISTRY.register()
class ResNet34_base(ResNet_base):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                         l2_norm=args.model.l2_norm,
                         use_bn_layer=args.model.use_bn_layer,
                         personalization_layers=args.model.personalization_layers,
                         **kwargs)
