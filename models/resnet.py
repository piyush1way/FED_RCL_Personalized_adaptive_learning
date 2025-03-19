import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # Changed to BatchNorm for better performance
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # Changed to BatchNorm for better performance

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)  # Changed to BatchNorm for better performance
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
        self.bn1 = nn.BatchNorm2d(planes)  # Changed to BatchNorm for better performance
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # Changed to BatchNorm for better performance
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)  # Changed to BatchNorm for better performance

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)  # Changed to BatchNorm for better performance
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
                 last_feature_dim=512, **kwargs):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 3 if not use_pretrained else 7

        Conv2d = self.get_conv()
        self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Changed to BatchNorm for better performance

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        self.num_layers = 6
        self.fc = nn.Linear(last_feature_dim * block.expansion, num_classes)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_conv(self):
        return nn.Conv2d
    
    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=True):
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
        features = out.view(out.size(0), -1)
        
        if self.l2_norm:
            features = F.normalize(features, p=2, dim=1)
            
        logit = self.fc(features)

        if return_feature:
            return features, logit
            
        return {"feature": features, "logit": logit}


class PersonalizedResNet(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, 
                 personalization_layers=1, use_bn_layer=True, **kwargs):
        super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, 
                         l2_norm=l2_norm, use_bn_layer=use_bn_layer, **kwargs)
        
        # Feature dimension after pooling
        self.feature_dim = last_feature_dim * block.expansion
        
        # Create personalized head (can be multiple layers)
        if personalization_layers == 1:
            self.personalized_head = nn.Linear(self.feature_dim, num_classes)
            # Initialize weights properly
            nn.init.kaiming_normal_(self.personalized_head.weight, mode='fan_out')
            nn.init.zeros_(self.personalized_head.bias)
        else:
            layers = []
            for i in range(personalization_layers-1):
                if i == 0:
                    layers.append(nn.Linear(self.feature_dim, self.feature_dim))
                else:
                    layers.append(nn.Linear(self.feature_dim, self.feature_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.feature_dim, num_classes))
            self.personalized_head = nn.Sequential(*layers)
            
            # Initialize weights properly
            for m in self.personalized_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Flag to control which head to use
        self.use_personalized_head = False
        self.frozen_layers = []

    def forward(self, x, return_feature=False):
        """Forward pass with layer-wise feature extraction"""
        # Extract features through backbone
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.view(out.size(0), -1)
        
        # Apply L2 normalization if needed
        if self.l2_norm:
            features = F.normalize(features, p=2, dim=1)
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            if isinstance(self.personalized_head, nn.Linear):
                self.personalized_head.weight.data = F.normalize(self.personalized_head.weight.data, p=2, dim=1)
        
        # Global classification
        global_logit = self.fc(features)
        
        # Personalized classification
        personalized_logit = self.personalized_head(features)
        
        # Choose which logit to use as the default based on personalization flag
        default_logit = personalized_logit if self.use_personalized_head else global_logit
        
        # Return a dictionary with all outputs
        output_dict = {
            "feature": features,  # Feature representation
            "global_logit": global_logit,  # Global logits
            "personalized_logit": personalized_logit,  # Personalized logits
            "logit": default_logit,  # Default logit based on current mode
            "use_personalized": self.use_personalized_head  # Flag for debugging
        }
        
        if return_feature:
            return output_dict, {"pooled": features}
        return output_dict
    
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
                global_params[name] = param.data.clone()
        return global_params
    
    def get_local_params(self):
        """Get parameters that should be kept local to the client"""
        local_params = {}
        for name, param in self.named_parameters():
            if 'personalized_head' in name:  # Only personalized head
                local_params[name] = param.data.clone()
        return local_params
    
    def setup_adaptive_freezing(self):
        """Setup adaptive layer freezing for personalization"""
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


@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


@ENCODER_REGISTRY.register()
class PersonalizedResNet18(PersonalizedResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        # First check if personalization_layers is in kwargs
        personalization_layers = kwargs.pop('personalization_layers', None)
        
        # If not in kwargs, try to get from args
        if personalization_layers is None:
            personalization_layers = getattr(args, 'personalization_layers', 1)
        
        # Now pass it explicitly to the parent class
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                         personalization_layers=personalization_layers, **kwargs)
