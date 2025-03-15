from utils import get_numclasses
from utils.registry import Registry

# Import models directly instead of from models/__init__.py
from models.resnet import ResNet18, PersonalizedResNet18

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder models.
"""

# ✅ Register models only if they are not already registered
if "ResNet18" not in ENCODER_REGISTRY:
    ENCODER_REGISTRY.register(ResNet18)

if "PersonalizedResNet18" not in ENCODER_REGISTRY:
    ENCODER_REGISTRY.register(PersonalizedResNet18)

def build_encoder(args):
    """
    Builds an encoder model based on the provided arguments.

    Args:
        args: Configuration arguments for model selection.

    Returns:
        An instance of the selected encoder model.
    """
    num_classes = get_numclasses(args)

    print(f"=> Creating model '{args.model.name}, pretrained={args.model.pretrained}'")

    # Ensure model exists in the registry
    if args.model.name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Model '{args.model.name}' not found in ENCODER_REGISTRY. "
            f"Available models: {list(ENCODER_REGISTRY._obj_map.keys())}"
        )

    encoder_class = ENCODER_REGISTRY.get(args.model.name)
    return encoder_class(args, num_classes, **args.model)
