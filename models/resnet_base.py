# import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from utils import *
import numpy as np
from models.build import ENCODER_REGISTRY
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=True, Conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=True, Conv2d=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=True,
                 last_feature_dim=512, logit_detach=False, **kwargs):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        self.logit_detach = logit_detach
        conv1_kernel_size = 3 if not use_pretrained else 7

        Conv2d = self.get_conv()
        self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        self.num_layers = 6
        self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)
        
        # Multi-level contrastive learning projectors
        self.projectors = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64 * block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer2': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128 * block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer3': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256 * block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            ),
            'layer4': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(last_feature_dim * block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            )
        })
        
        # Trust-based adaptive learning rate parameters
        self.trust_score = nn.Parameter(torch.ones(1), requires_grad=False)
        self.base_lr = 0.1
        self.min_lr = 0.001
        self.max_lr = 0.5
        
        self._initialize_weights()

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

    def get_conv(self):
        return nn.Conv2d
    
    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_multilevel=False):
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Multi-level feature extraction
        layer1_out = self.layer1(out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Final pooling
        pooled = F.adaptive_avg_pool2d(layer4_out, 1)
        features = pooled.view(pooled.size(0), -1)
        
        features_normalized = F.normalize(features, p=2, dim=1) if self.l2_norm else features
        
        if self.logit_detach:
            logit = self.fc(features.detach())
        else:
            logit = self.fc(features)
        
        result = {"feature": features, "feature_normalized": features_normalized, "logit": logit}
        
        if return_multilevel:
            # Get multi-level features for contrastive learning
            multilevel_features = {
                'layer1': self.projectors['layer1'](layer1_out),
                'layer2': self.projectors['layer2'](layer2_out),
                'layer3': self.projectors['layer3'](layer3_out),
                'layer4': self.projectors['layer4'](layer4_out)
            }
            
            # Normalize all projections
            for key in multilevel_features:
                multilevel_features[key] = F.normalize(multilevel_features[key], p=2, dim=1)
                
            result["multilevel_features"] = multilevel_features
        
        if return_feature:
            return features, logit
            
        return result
    
    def update_trust_score(self, score):
        """Update trust score for adaptive learning rate"""
        self.trust_score.data = torch.clamp(score, 0.1, 1.0)
        
    def get_adaptive_lr(self):
        """Calculate adaptive learning rate based on trust score"""
        return self.min_lr + (self.max_lr - self.min_lr) * self.trust_score.item()
    
    def get_multilevel_features(self, x):
        """Extract features from multiple layers for contrastive learning"""
        out = F.relu(self.bn1(self.conv1(x)))
        
        layer1_out = self.layer1(out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        multilevel_features = {
            'layer1': self.projectors['layer1'](layer1_out),
            'layer2': self.projectors['layer2'](layer2_out),
            'layer3': self.projectors['layer3'](layer3_out),
            'layer4': self.projectors['layer4'](layer4_out)
        }
        
        # Normalize all projections
        for key in multilevel_features:
            multilevel_features[key] = F.normalize(multilevel_features[key], p=2, dim=1)
            
        return multilevel_features


class ResNet_base(ResNet):

    def forward_layer(self, layer, x, no_relu=True):
        if isinstance(layer, nn.Linear):
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            out = layer(x)
        else:
            if no_relu:
                out = x
                for sublayer in layer[:-1]:
                    out = sublayer(out)
                out = layer[-1](out, no_relu=no_relu)
            else:
                out = layer(x)

        return out
    
    def forward_layer_by_name(self, layer_name, x, no_relu=True):
        layer = getattr(self, layer_name)
        return self.forward_layer(layer, x, no_relu)

    def forward_layer0(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out0 = self.bn1(self.conv1(x))
        if not no_relu:
            out0 = F.relu(out0)
        return out0

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'fc' not in n:
            # if True:
                p.requires_grad = False
        logger.warning('Freeze backbone parameters (except fc)')
        return
    
    def forward(self, x: torch.Tensor, no_relu: bool = True, return_multilevel: bool = False) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.logit_detach:
            logit = self.fc(out.detach())
        else:
            logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit
        
        if return_multilevel:
            # Get multi-level features for contrastive learning
            multilevel_features = {
                'layer1': self.projectors['layer1'](results['layer1']),
                'layer2': self.projectors['layer2'](results['layer2']),
                'layer3': self.projectors['layer3'](results['layer3']),
                'layer4': self.projectors['layer4'](results['layer4'])
            }
            
            # Normalize all projections
            for key in multilevel_features:
                multilevel_features[key] = F.normalize(multilevel_features[key], p=2, dim=1)
                
            results["multilevel_features"] = multilevel_features

        return results


@ENCODER_REGISTRY.register()
class ResNet18_base(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    


@ENCODER_REGISTRY.register()
class ResNet34_base(ResNet_base):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
