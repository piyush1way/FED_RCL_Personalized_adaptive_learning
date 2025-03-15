# models/__init__.py

# Expose build_encoder and other necessary functions/classes
from .build import build_encoder, ENCODER_REGISTRY

# Expose specific models (optional, but useful for direct imports)
from .resnet import ResNet18, PersonalizedResNet18
from .VGG9 import VGG9
from .basic import *
from .resnet_base import *
from .MobileNet import *
from .SqueezeNet import *
from .ShuffleNet import *

# Define __all__ to control what gets imported with `from models import *`
__all__ = [
    "build_encoder",
    "ENCODER_REGISTRY",
    "ResNet18",
    "PersonalizedResNet18",
    "VGG9",
    # Add other models as needed
]
