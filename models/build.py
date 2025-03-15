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

    encoder_class = ENCODER_REGISTRY.get(args.model.name)
    if encoder_class is None:
        raise ValueError(
            f"Model '{args.model.name}' not found in ENCODER_REGISTRY. Available models: {list(ENCODER_REGISTRY._obj_map.keys())}"
        )

    return encoder_class(args, num_classes, **args.model)


# Import ResNet models for registration
from models.resnet import ResNet18, PersonalizedResNet18

# Register models safely
ENCODER_REGISTRY.register(ResNet18)
ENCODER_REGISTRY.register(PersonalizedResNet18, "personalized_resnet18")  # Ensure correct key
