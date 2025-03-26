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
        return out if no_relu else F.relu(out)

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
        return out if no_relu else F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_bn_layer=True,
                 last_feature_dim=512, personalization_layers=2, **kwargs):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        self.num_classes = num_classes
        self.use_bn_layer = use_bn_layer
        self.expansion = block.expansion

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
        self.trust_score = 1.0
        
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
        
        # Initialize with slightly different weights than global head for better specialization
        if isinstance(self.personalized_head, nn.Linear):
            nn.init.normal_(self.personalized_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.personalized_head.bias)
        elif isinstance(self.personalized_head, nn.Sequential):
            for m in self.personalized_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
        # Hybrid head for combining global and personalized knowledge
        self.hybrid_factor = nn.Parameter(torch.tensor(0.5))  # Learnable hybridization factor
        self.hybrid_mode = False  # Toggle for hybrid mode

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
        return nn.ModuleDict({
            f'layer{i}': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim * self.expansion, 128),
                nn.BatchNorm1d(128) if self.use_bn_layer else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ) for i, dim in enumerate([64, 128, 256], start=1)
        })

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
        
        # Apply L2 normalization to features for more stable representation
        features_normalized = F.normalize(features, p=2, dim=1)
        
        projection_features = None
        if get_projection:
            projection_features = self.projection_head(features)
            projection_features = F.normalize(projection_features, p=2, dim=1)
        
        if get_multi_level:
            multi_level_projections = {}
            for layer_name in ['layer1', 'layer2', 'layer3']:
                if layer_name in results:
                    proj = self.multi_level_projections[layer_name](results[layer_name])
                    multi_level_projections[layer_name] = F.normalize(proj, p=2, dim=1)
            results['multi_level_projections'] = multi_level_projections
        
        # Global logits from original features
        global_logit = self.fc(features)
        
        # Personalized logits from normalized features for more stable learning
        personalized_logit = self.personalized_head(features_normalized)
        
        # Hybrid mode combines global and personalized logits using a learnable factor
        if self.hybrid_mode:
            hybrid_factor = torch.sigmoid(self.hybrid_factor)  # Convert to [0,1] range
            default_logit = hybrid_factor * personalized_logit + (1 - hybrid_factor) * global_logit
        else:
            # Use personalized head if enabled, otherwise use global head
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
        """Freeze backbone layers for personalization"""
        for name, param in self.named_parameters():
            if 'fc' not in name and 'personalized_head' not in name and 'projection_head' not in name and 'temperature' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        logger.info('Froze backbone parameters (except fc, personalized_head, projection_head, and temperature)')

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full training"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info('Unfroze all parameters')
    
    def enable_personalized_mode(self):
        """Enable personalized mode to use the personalized head"""
        self.use_personalized_head = True
        logger.info('Enabled personalized mode')
        
    def disable_personalized_mode(self):
        """Disable personalized mode to use the global head"""
        self.use_personalized_head = False
        logger.info('Disabled personalized mode')
        
    def get_global_params(self):
        """Get only the global model parameters (excluding personalized head)"""
        return {k: v.cpu() for k, v in self.state_dict().items() if 'personalized_head' not in k}
    
    def get_local_params(self):
        return {name: param.data.clone() for name, param in self.named_parameters() if 'personalized_head' in name}
    
    def setup_adaptive_freezing(self, freeze_ratio=0.5):
        """Selectively freeze layers based on the freeze ratio"""
        if freeze_ratio <= 0:
            self.unfreeze_backbone()
            return
        
        if freeze_ratio >= 1.0:
            self.freeze_backbone()
            return
        
        # Define layers in order of freezing priority
        freezable_layers = [
            'conv1', 'bn1', 
            'layer1.0', 'layer1.1',
            'layer2.0', 'layer2.1',
            'layer3.0', 'layer3.1', 
            'layer4.0'
        ]
        
        # Calculate how many layers to freeze
        num_to_freeze = int(len(freezable_layers) * freeze_ratio)
        to_freeze = freezable_layers[:num_to_freeze]
        
        # Unfreeze all parameters first
        self.unfreeze_backbone()
        
        # Then freeze selected parameters
        frozen_layers = []
        for name, param in self.named_parameters():
            if any(layer in name for layer in to_freeze):
                param.requires_grad = False
                if name.split('.')[0] not in frozen_layers:
                    frozen_layers.append(name.split('.')[0])
        
        logger.info(f"Adaptively freezing layers: {frozen_layers}")
        
    def get_contrastive_features(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(x)))))))
        features = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return F.normalize(self.projection_head(features), p=2, dim=1)
    
    def get_multi_level_contrastive_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        layer_features = {}
        
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):
            out = layer(out)
            if i < 4:
                layer_features[f'layer{i}'] = self.multi_level_projections[f'layer{i}'](out)
        
        features = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        layer_features['layer4'] = self.projection_head(features)
        
        return {k: F.normalize(v, p=2, dim=1) for k, v in layer_features.items()}
    
    def update_trust_score(self, new_score):
        self.trust_score = new_score
        logger.debug(f"Updated trust score to {self.trust_score}")

    def enable_hybrid_mode(self):
        """Enable hybrid mode that combines global and personalized heads"""
        self.hybrid_mode = True
        logger.info("Enabled hybrid personalized head mode")
        
    def disable_hybrid_mode(self):
        """Disable hybrid mode"""
        self.hybrid_mode = False
        logger.info("Disabled hybrid personalized head mode")

@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                        l2_norm=args.model.l2_norm,
                        use_bn_layer=args.model.use_bn_layer,
                        personalization_layers=args.model.personalization_layers,
                        **kwargs)
        self.use_personalized_head = False

@ENCODER_REGISTRY.register()
class PersonalizedResNet18(ResNet):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                        l2_norm=args.model.l2_norm,
                        use_bn_layer=args.model.use_bn_layer,
                        personalization_layers=args.model.personalization_layers,
                        **kwargs)
        self.use_personalized_head = True
        self.enable_personalized_mode()

