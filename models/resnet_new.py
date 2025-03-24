import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY
from typing import Dict, List, Optional
from omegaconf import DictConfig
import logging
from models.resnet_base import ResNet_base, BasicBlock

logger = logging.getLogger(__name__)

@ENCODER_REGISTRY.register()
class ResNet18(ResNet_base):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                        l2_norm=args.model.l2_norm,
                        use_bn_layer=args.model.use_bn_layer,
                        personalization_layers=args.model.personalization_layers,
                        **kwargs)
        self.use_personalized_head = False

@ENCODER_REGISTRY.register()
class PersonalizedResNet18(ResNet_base):
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                        l2_norm=args.model.l2_norm,
                        use_bn_layer=args.model.use_bn_layer,
                        personalization_layers=args.model.personalization_layers,
                        **kwargs)
        
        # Basic settings
        self.use_personalized_head = True
        self.trust_score = 0.8  # Initialize trust score
        
        # Get configurations
        personalization_config = getattr(args.model, "personalization", {})
        distillation_config = getattr(args.model, "distillation", {})
        adaptive_config = getattr(args.model, "adaptive_freezing", {})
        
        # Personalization settings
        self.enable_personalized_mode()
        
        # Knowledge distillation settings
        self.use_distillation = getattr(distillation_config, "enable", True)
        self.distillation_temp = getattr(distillation_config, "temperature", 3.0)
        self.distillation_weight = getattr(distillation_config, "weight", 0.7)
        
        # Adaptive layer freezing settings
        self.use_adaptive_freezing = getattr(adaptive_config, "enable", False)
        self.initial_freeze_ratio = getattr(adaptive_config, "initial_freeze_ratio", 0.5)
        self.freeze_decay_rate = getattr(adaptive_config, "decay_rate", 0.05)
        
        if self.use_adaptive_freezing:
            self.setup_adaptive_freezing(self.initial_freeze_ratio)
            
        logger.info(f"Initialized PersonalizedResNet18 with:"
                   f"\n - personalization_layers={args.model.personalization_layers}"
                   f"\n - use_distillation={self.use_distillation}"
                   f"\n - distillation_temp={self.distillation_temp}"
                   f"\n - use_adaptive_freezing={self.use_adaptive_freezing}") 
