# models/build.py

from utils import get_numclasses
from utils.registry import Registry
import models

# Initialize the encoder registry
ENCODER_REGISTRY = Registry("ENCODER")

# Expose build_encoder and ENCODER_REGISTRY
__all__ = ["get_model", "build_encoder", "ENCODER_REGISTRY"]

# Import models before registering
from models.resnet import ResNet18, PersonalizedResNet18

# Register models only if not already registered
if "ResNet18" not in ENCODER_REGISTRY._obj_map:
    ENCODER_REGISTRY.register(ResNet18)

if "PersonalizedResNet18" not in ENCODER_REGISTRY._obj_map:
    ENCODER_REGISTRY.register(PersonalizedResNet18)

def build_encoder(args):
    # Get the number of classes from the dataset configuration
    num_classes = get_numclasses(args)
    
    # Get the encoder class from the registry
    encoder_type = args.model.get('type', args.model.name)
    encoder_class = ENCODER_REGISTRY.get(encoder_type)
    
    if encoder_class is None:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create a copy of the model args and remove 'num_classes' if it exists
    model_args = dict(args.model)
    if 'num_classes' in model_args:
        del model_args['num_classes']
    
    # Pass num_classes as a separate argument to avoid duplication
    return encoder_class(args, num_classes=num_classes, **model_args)
