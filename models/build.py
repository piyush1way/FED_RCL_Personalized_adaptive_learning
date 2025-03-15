from utils import get_numclasses
from utils.registry import Registry
import models

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder models.
"""

__all__ = ['get_model', 'build_encoder']


def build_encoder(args):
    """
    Builds an encoder model based on the provided arguments.

    Args:
        args: Configuration arguments for model selection.

    Returns:
        An instance of the selected encoder model.
    """
    num_classes = get_numclasses(args)

    if args.verbose:
        print(ENCODER_REGISTRY)

    print(f"=> Creating model '{args.model.name}, pretrained={args.model.pretrained}'")

    # Ensure model exists in the registry
    if args.model.name not in ENCODER_REGISTRY._obj_map:
        raise KeyError(
            f"Model '{args.model.name}' not found in ENCODER_REGISTRY. Available models: {list(ENCODER_REGISTRY._obj_map.keys())}"
        )

    encoder_class = ENCODER_REGISTRY.get(args.model.name)
    return encoder_class(args, num_classes, **args.model)


# ✅ Import and register models
from models.resnet import ResNet18, PersonalizedResNet18

# ✅ Register models if not already registered
if "ResNet18" not in ENCODER_REGISTRY._obj_map:
    ENCODER_REGISTRY.register(ResNet18)

if "personalized_resnet18" not in ENCODER_REGISTRY._obj_map:
    ENCODER_REGISTRY.register(PersonalizedResNet18)
