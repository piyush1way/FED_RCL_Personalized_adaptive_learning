import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PersonalizedResNet(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, last_feature_dim=512, l2_norm=False, 
                 personalization_layers=1, **kwargs):
        super().__init__(block, num_blocks, num_classes=num_classes, last_feature_dim=last_feature_dim, 
                         l2_norm=l2_norm, **kwargs)
        
        # Feature dimension after pooling
        self.feature_dim = last_feature_dim * block.expansion
        
        # Create personalized head (can be multiple layers)
        if personalization_layers == 1:
            self.personalized_head = nn.Linear(self.feature_dim, num_classes)
        else:
            layers = []
            for i in range(personalization_layers-1):
                layers.append(nn.Linear(self.feature_dim, self.feature_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.feature_dim, num_classes))
            self.personalized_head = nn.Sequential(*layers)
        
        # Adaptive learning rate parameters
        self.grad_history = []
        self.learning_rate_scale = 1.0
        
        # Trust score for client filtering
        self.update_norm = 0.0
        self.trust_score = 1.0
        
        # Flag to control which head to use
        self.use_personalized_head = True

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
        
        # Apply L2 normalization if needed
        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            if self.use_personalized_head:
                if isinstance(self.personalized_head, nn.Linear):
                    self.personalized_head.weight.data = F.normalize(self.personalized_head.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
        
        # Choose between global and personalized head
        if self.use_personalized_head:
            logit = self.personalized_head(out)
        else:
            logit = self.fc(out)
        
        if return_feature:
            return features, logit
        return logit
    
    def get_global_params(self):
        """Return only global parameters for aggregation"""
        global_params = {}
        for name, param in self.named_parameters():
            if 'personalized_head' not in name:
                global_params[name] = param.data.clone()
        return global_params
    
    def get_local_params(self):
        """Return only personalized parameters"""
        local_params = {}
        for name, param in self.named_parameters():
            if 'personalized_head' in name:
                local_params[name] = param.data.clone()
        return local_params
    
    def update_global_params(self, global_params):
        """Update only global parameters"""
        for name, param in self.named_parameters():
            if name in global_params:
                param.data.copy_(global_params[name])
    
    def compute_trust_score(self, prev_model=None):
        """Compute trust score based on update consistency"""
        if prev_model is None:
            return 1.0
            
        # Calculate update norm
        update_norm = 0.0
        total_params = 0
        
        for name, param in self.named_parameters():
            if 'personalized_head' not in name and name in prev_model:
                diff = param.data - prev_model[name]
                update_norm += torch.norm(diff).item() ** 2
                total_params += diff.numel()
        
        if total_params > 0:
            update_norm = (update_norm / total_params) ** 0.5
            
        self.update_norm = update_norm
        
        # Trust score is inversely related to update norm
        # Normalize it to be between 0 and 1
        self.trust_score = 1.0 / (1.0 + update_norm)
        
        return self.trust_score
    
    def update_learning_rate_scale(self, grad_variance=None):
        """Update learning rate scale based on gradient variance"""
        if grad_variance is None:
            # Calculate gradient variance from history
            if len(self.grad_history) > 1:
                # Flatten all gradients
                flat_grads = []
                for grads in self.grad_history:
                    flat_grad = torch.cat([g.flatten() for g in grads if g is not None])
                    flat_grads.append(flat_grad)
                
                # Calculate variance
                stacked_grads = torch.stack(flat_grads)
                grad_variance = torch.var(stacked_grads, dim=0).mean().item()
            else:
                grad_variance = 0.0
        
        # Update learning rate scale
        self.learning_rate_scale = 1.0 / (1.0 + grad_variance)
        
        return self.learning_rate_scale
    
    def store_gradient(self):
        """Store current gradients for variance calculation"""
        current_grads = []
        for param in self.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.detach().clone())
            else:
                current_grads.append(None)
        
        self.grad_history.append(current_grads)
        
        # Keep only recent history
        if len(self.grad_history) > 5:
            self.grad_history.pop(0)


@ENCODER_REGISTRY.register()
class ResNet18(ResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


@ENCODER_REGISTRY.register()
class PersonalizedResNet18(PersonalizedResNet):
    def __init__(self, args, num_classes=10, **kwargs):
        personalization_layers = getattr(args, 'personalization_layers', 1)
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                         personalization_layers=personalization_layers, **kwargs)
