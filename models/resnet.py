
# # '''ResNet in PyTorch.
# # For Pre-activation ResNet, see 'preact_resnet.py'.
# # Reference:
# # [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
# #     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# # '''
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torchvision
# # from models.build import ENCODER_REGISTRY


# # class BasicBlock(nn.Module):
# #     expansion = 1

# #     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
# #         super(BasicBlock, self).__init__()
# #         self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
# #         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
# #         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

# #         self.downsample = nn.Sequential()
# #         if stride != 1 or in_planes != self.expansion * planes:
# #             self.downsample = nn.Sequential(
# #                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
# #                 nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
# #             )

# #     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
# #         out = F.relu(self.bn1(self.conv1(x)))
# #         out = self.bn2(self.conv2(out))
# #         out += self.downsample(x)
# #         if not no_relu:
# #             out = F.relu(out)
# #         return out


# # class Bottleneck(nn.Module):
# #     expansion = 4

# #     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
# #         super(Bottleneck, self).__init__()
# #         self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
# #         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
# #         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
# #         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
# #         self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
# #         self.bn3 = nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)

# #         self.downsample = nn.Sequential()
# #         if stride != 1 or in_planes != self.expansion * planes:
# #             self.downsample = nn.Sequential(
# #                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
# #                 nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
# #             )

# #     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
# #         out = F.relu(self.bn1(self.conv1(x)))
# #         out = F.relu(self.bn2(self.conv2(out)))
# #         out = self.bn3(self.conv3(out))
# #         out += self.downsample(x)
# #         if not no_relu:
# #             out = F.relu(out)
# #         return out


# # class ResNet(nn.Module):
# #     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
# #                  last_feature_dim=512, **kwargs):
# #         super(ResNet, self).__init__()
# #         self.l2_norm = l2_norm
# #         self.in_planes = 64
# #         conv1_kernel_size = 3 if not use_pretrained else 7

# #         Conv2d = self.get_conv()
# #         self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
# #         self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64)

# #         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
# #         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
# #         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
# #         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

# #         self.num_layers = 6
# #         self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)

# #     def get_conv(self):
# #         return nn.Conv2d
    
# #     def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
# #         strides = [stride] + [1] * (num_blocks - 1)
# #         layers = []
# #         for stride in strides:
# #             layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
# #             self.in_planes = planes * block.expansion
# #         return nn.Sequential(*layers)

# #     def forward(self, x, return_feature=False):
# #         out = F.relu(self.bn1(self.conv1(x)))
# #         out = self.layer1(out)
# #         out = self.layer2(out)
# #         out = self.layer3(out)
# #         out = self.layer4(out)

# #         out = F.adaptive_avg_pool2d(out, 1)
# #         out = out.view(out.size(0), -1)
# #         logit = self.fc(out)

# #         if return_feature:
# #             return out, logit
# #         return logit


# # #  Personalized FedRCL Model with a Client-Specific Head
# # class PersonalizedResNet(ResNet):
# #     def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, **kwargs):
# #         super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, l2_norm=l2_norm, **kwargs)
# #         self.frozen_layers = []
# #         self.adaptive_freezing_enabled = False

# #     def setup_adaptive_freezing(self):
# #         """Setup adaptive layer freezing for personalization"""
# #         self.adaptive_freezing_enabled = True
# #         # Initially freeze all layers except the last one
# #         self.frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
# #         self.freeze_layers(self.frozen_layers)
        
# #     def freeze_layers(self, layer_names):
# #         """Freeze specific layers by name"""
# #         for name, param in self.named_parameters():
# #             if any(layer in name for layer in layer_names):
# #                 param.requires_grad = False
# #             else:
# #                 param.requires_grad = True

# #     def freeze_backbone(self):
# #         """Freeze all layers except the classifier"""
# #         for name, param in self.named_parameters():
# #             if 'fc' not in name:  # fc is the classifier layer
# #                 param.requires_grad = False
# #             else:
# #                 param.requires_grad = True

# #     def unfreeze_backbone(self):
# #         """Unfreeze all layers"""
# #         for param in self.parameters():
# #             param.requires_grad = True

# #     def forward(self, x, return_feature=False):
# #         """Forward pass with layer-wise feature extraction"""
# #         features = {}
        
# #         # Layer 0
# #         out = F.relu(self.bn1(self.conv1(x)))
# #         features['layer0'] = out
        
# #         # Layers 1-4
# #         out = self.layer1(out)
# #         features['layer1'] = out
        
# #         out = self.layer2(out)
# #         features['layer2'] = out
        
# #         out = self.layer3(out)
# #         features['layer3'] = out
        
# #         out = self.layer4(out)
# #         features['layer4'] = out
        
# #         # Global average pooling
# #         out = F.adaptive_avg_pool2d(out, 1)
# #         out = out.view(out.size(0), -1)
# #         features['pooled'] = out
        
# #         # Classification
# #         if self.l2_norm:
# #             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
# #             out = F.normalize(out, dim=1)
# #         logit = self.fc(out)
# #         features['logit'] = logit
        
# #         # Return a dictionary with 'feature' and 'logit' keys
# #         output_dict = {
# #             "feature": features['pooled'],  # Feature representation
# #             "logit": logit,                # Logits for classification
# #         }
        
# #         if return_feature:
# #             return output_dict, features
# #         return output_dict


# # @ENCODER_REGISTRY.register()
# # class ResNet18(ResNet):
# #     def __init__(self, args, num_classes=10, **kwargs):
# #         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


# # @ENCODER_REGISTRY.register()
# # class PersonalizedResNet18(PersonalizedResNet):
# #     def __init__(self, args, num_classes=10, **kwargs):
# #         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

# #     def forward_classifier(self, x):
# #         """
# #         Forward pass for the classifier head.
# #         Args:
# #             x (torch.Tensor): Input features (e.g., from the global model).
# #         Returns:
# #             torch.Tensor: Output logits.
# #         """
# #         # Pass the input through the classifier (fully connected layer)
# #         if self.l2_norm:
# #             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
# #             x = F.normalize(x, dim=1)
# #         return self.fc(x)


# '''ResNet in PyTorch.
# For Pre-activation ResNet, see 'preact_resnet.py'.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from models.build import ENCODER_REGISTRY


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
#         super(BasicBlock, self).__init__()
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
#         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
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

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
#         super(Bottleneck, self).__init__()
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
#         self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
#         self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
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
#     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
#                  last_feature_dim=512, **kwargs):
#         super(ResNet, self).__init__()
#         self.l2_norm = l2_norm
#         self.in_planes = 64
#         conv1_kernel_size = 3 if not use_pretrained else 7

#         Conv2d = self.get_conv()
#         self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64)

#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
#         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

#         self.num_layers = 6
#         self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)

#     def get_conv(self):
#         return nn.Conv2d
    
#     def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
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
#         out = out.view(out.size(0), -1)
#         logit = self.fc(out)

#         if return_feature:
#             return out, logit
#         return logit


# #  Personalized FedRCL Model with a Client-Specific Head
# class PersonalizedResNet(ResNet):
#     def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, **kwargs):
#         super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, l2_norm=l2_norm, **kwargs)
#         self.frozen_layers = []
#         self.adaptive_freezing_enabled = False

#         # Add a personalized head (client-specific layer)
#         self.personalized_head = nn.Linear(last_feature_dim * block.expansion, num_classes)

#     def setup_adaptive_freezing(self):
#         """Setup adaptive layer freezing for personalization"""
#         self.adaptive_freezing_enabled = True
#         # Initially freeze all layers except the last one
#         self.frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
#         self.freeze_layers(self.frozen_layers)
        
#     def freeze_layers(self, layer_names):
#         """Freeze specific layers by name"""
#         for name, param in self.named_parameters():
#             if any(layer in name for layer in layer_names):
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True

#     def freeze_backbone(self):
#         """Freeze all layers except the classifier"""
#         for name, param in self.named_parameters():
#             if 'fc' not in name and 'personalized_head' not in name:  # Freeze backbone only
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True

#     def unfreeze_backbone(self):
#         """Unfreeze all layers"""
#         for param in self.parameters():
#             param.requires_grad = True

#     def forward(self, x, return_feature=False):
#         """Forward pass with layer-wise feature extraction"""
#         features = {}
        
#         # Layer 0
#         out = F.relu(self.bn1(self.conv1(x)))
#         features['layer0'] = out
        
#         # Layers 1-4
#         out = self.layer1(out)
#         features['layer1'] = out
        
#         out = self.layer2(out)
#         features['layer2'] = out
        
#         out = self.layer3(out)
#         features['layer3'] = out
        
#         out = self.layer4(out)
#         features['layer4'] = out
        
#         # Global average pooling
#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)
#         features['pooled'] = out
        
#         # Global classification
#         if self.l2_norm:
#             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
#             out = F.normalize(out, dim=1)
#         global_logit = self.fc(out)
#         features['global_logit'] = global_logit
        
#         # Personalized classification
#         personalized_logit = self.personalized_head(out)
#         features['personalized_logit'] = personalized_logit
        
#         # Return a dictionary with global and personalized outputs
#         output_dict = {
#             "global_feature": features['pooled'],  # Global feature representation
#             "global_logit": global_logit,         # Global logits
#             "personalized_logit": personalized_logit,  # Personalized logits
#         }
        
#         if return_feature:
#             return output_dict, features
#         return output_dict


# @ENCODER_REGISTRY.register()
# class ResNet18(ResNet):
#     def __init__(self, args, num_classes=10, **kwargs):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


# @ENCODER_REGISTRY.register()
# class PersonalizedResNet18(PersonalizedResNet):
#     def __init__(self, args, num_classes=10, **kwargs):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

#     def forward_classifier(self, x):
#         """
#         Forward pass for the classifier head.
#         Args:
#             x (torch.Tensor): Input features (e.g., from the global model).
#         Returns:
#             torch.Tensor: Output logits.
#         """
#         # Pass the input through the classifier (fully connected layer)
#         if self.l2_norm:
#             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
#             x = F.normalize(x, dim=1)
#         return self.fc(x)

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.build import ENCODER_REGISTRY


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
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

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion * planes) if not use_bn_layer else nn.BatchNorm2d(planes)
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
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
                 last_feature_dim=512, **kwargs):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 3 if not use_pretrained else 7

        Conv2d = self.get_conv()
        self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        self.num_layers = 6
        self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)

    def get_conv(self):
        return nn.Conv2d
    
    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logit = self.fc(out)

        if return_feature:
            return out, logit
        return logit


#  Personalized FedRCL Model with a Client-Specific Head
class PersonalizedResNet(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, **kwargs):
        super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, l2_norm=l2_norm, **kwargs)
        self.frozen_layers = []
        self.adaptive_freezing_enabled = False
        
        # Add a personalized head (client-specific layer)
        self.personalized_head = nn.Linear(last_feature_dim * block.expansion, num_classes)
        
        # Flags to control which head to use
        self.use_personalized_head = False
        self.feature_dim = last_feature_dim * block.expansion

    def setup_adaptive_freezing(self):
        """Setup adaptive layer freezing for personalization"""
        self.adaptive_freezing_enabled = True
        # Initially freeze all layers except the last one
        self.frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        self.freeze_layers(self.frozen_layers)
        
    def freeze_layers(self, layer_names):
        """Freeze specific layers by name"""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_backbone(self):
        """Freeze all layers except the classifier"""
        for name, param in self.named_parameters():
            if 'fc' not in name and 'personalized_head' not in name:  # fc and personalized_head are classifier layers
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
    
    def enable_personalized_mode(self):
        """Enable using the personalized head instead of the global head"""
        self.use_personalized_head = True
        
    def disable_personalized_mode(self):
        """Disable using the personalized head, revert to global head"""
        self.use_personalized_head = False
        
    def get_global_params(self):
        """Get parameters that should be shared globally"""
        global_params = {}
        for name, param in self.named_parameters():
            if 'personalized_head' not in name:  # Everything except personalized head
                global_params[name] = param
        return global_params
    
    def get_personalized_params(self):
        """Get parameters that should be kept local to the client"""
        personal_params = {}
        for name, param in self.named_parameters():
            if 'personalized_head' in name:  # Only personalized head
                personal_params[name] = param
        return personal_params

    def forward(self, x, return_feature=False):
        """Forward pass with layer-wise feature extraction"""
        features = {}
        
        # Layer 0
        out = F.relu(self.bn1(self.conv1(x)))
        features['layer0'] = out
        
        # Layers 1-4
        out = self.layer1(out)
        features['layer1'] = out
        
        out = self.layer2(out)
        features['layer2'] = out
        
        out = self.layer3(out)
        features['layer3'] = out
        
        out = self.layer4(out)
        features['layer4'] = out
        
        # Global average pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        features['pooled'] = out
        
        # Classification - choose between global or personalized head
        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            if self.use_personalized_head:
                self.personalized_head.weight.data = F.normalize(self.personalized_head.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
        
        # Use either the global FC layer or the personalized head
        if self.use_personalized_head:
            logit = self.personalized_head(out)
        else:
            logit = self.fc(out)
            
        features['logit'] = logit
        
        # Return a dictionary with 'feature' and 'logit' keys
        output_dict = {
            "feature": features['pooled'],  # Feature representation
            "logit": logit,                # Logits for classification
            "use_personalized": self.use_personalized_head,  # Flag for debugging
        }
        
        if return_feature:
            return output_dict, features
        return output_dict


@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


@ENCODER_REGISTRY.register()
class PersonalizedResNet18(PersonalizedResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        
    def forward_classifier(self, x):
        """
        Forward pass for the classifier head.
        Args:
            x (torch.Tensor): Input features (e.g., from the global model).
        Returns:
            torch.Tensor: Output logits.
        """
        if self.use_personalized_head:
            return self.personalized_head(x)
        else:
            return self.fc(x)

