# '''ResNet in PyTorch.
# For Pre-activation ResNet, see 'preact_resnet.py'.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import torchvision
# from models.build import ENCODER_REGISTRY


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
#         super(BasicBlock, self).__init__()
#         self.conv1 = Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
#         self.conv2 = Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(self.expansion*planes) 
#             )

#     def forward_intermediate(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out_i = self.bn1(self.conv1(x))
#         out = F.relu(out_i)
#         out = self.bn2(self.conv2(out))
#         out += self.downsample(x)
#         if not no_relu:
#             out = F.relu(out)
#         else:
#             out = out
#         return out, out_i

#     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.downsample(x)
#         if not no_relu:
#             out = F.relu(out)
#         else:
#             out = out
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
#         super(Bottleneck, self).__init__()
#         self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
#         self.conv2 = Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
#         self.conv3 = Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.downsample = nn.Sequential(
#                 Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)
#             )

#     def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.downsample(x)
#         if not no_relu:
#             out = F.relu(out)
#         else:
#             out = out
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
#                  last_feature_dim=512, **kwargs):
        
#         #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
#         super(ResNet, self).__init__()
#         self.l2_norm = l2_norm
#         self.in_planes = 64
#         conv1_kernel_size = 3
#         if use_pretrained:
#             conv1_kernel_size = 7

#         Conv2d = self.get_conv()
#         Linear = self.get_linear()   
#         self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
#         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

#         self.logit_detach = False        

#         if use_pretrained:
#             resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
#             self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
#             self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
#             self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
#             self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)

#         self.num_layers = 6 # layer0 to layer5 (fc)

#         if l2_norm:
#             self.fc = Linear(last_feature_dim * block.expansion, num_classes, bias=False)
#         else:
#             self.fc = Linear(last_feature_dim * block.expansion, num_classes)


#     def get_conv(self):
#         return nn.Conv2d
    
#     def get_linear(self):
#         return nn.Linear

#     def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
#         strides = [stride] + [1]*(num_blocks-1)
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

#         if self.l2_norm:
#             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
#             out = F.normalize(out, dim=1)
#             logit = self.fc(out)
#         else:
#             logit = self.fc(out)
            
#         if return_feature==True:
#             return out, logit
#         else:
#             return logit
        
    
#     def forward_classifier(self,x):
#         logit = self.fc(x)
#         return logit        
    
    
#     def sync_online_and_global(self):
#         state_dict=self.state_dict()
#         for key in state_dict:
#             if 'global' in key:
#                 x=(key.split("_global"))
#                 online=(x[0]+x[1])
#                 state_dict[key]=state_dict[online]
#         self.load_state_dict(state_dict)

    


# @ENCODER_REGISTRY.register()
# class ResNet18(ResNet):
#     def __init__(self, args, num_classes=10, **kwargs):
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
#                          )
        


# def ResNet34(num_classes=10, l2_norm=False):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, l2_norm=l2_norm)


# def ResNet101(num_classes=10, l2_norm=False):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, l2_norm=l2_norm)


# def ResNet152(num_classes=10, l2_norm=False):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, l2_norm=l2_norm)


# models/resnet.py

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
            if 'fc' not in name:  # fc is the classifier layer
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True

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
        
        # Classification
        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
        logit = self.fc(out)
        features['logit'] = logit
        
        if return_feature:
            return features, logit
        return logit


@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


@ENCODER_REGISTRY.register()
class PersonalizedResNet18(PersonalizedResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
