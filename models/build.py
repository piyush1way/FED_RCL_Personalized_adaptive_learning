# models/build.py

from utils import get_numclasses
from utils.registry import Registry
import logging

logger = logging.getLogger(__name__)

# Initialize the encoder registry
ENCODER_REGISTRY = Registry("ENCODER")

# Expose build_encoder and ENCODER_REGISTRY
__all__ = ["build_encoder", "ENCODER_REGISTRY"]

def build_encoder(args):
    """
    Build an encoder model based on the provided arguments.
    
    Args:
        args: Configuration object containing model specifications
        
    Returns:
        A model instance of the specified type
    """
    # Get the number of classes from the dataset configuration
    num_classes = get_numclasses(args)
    
    # Get the encoder class from the registry
    encoder_type = args.model.get('type', args.model.name)
    
    # Log the model being built
    logger.info(f"Building model: {encoder_type} with {num_classes} classes")
    
    # Check if the model type exists in the registry
    if encoder_type not in ENCODER_REGISTRY._obj_map:
        available_models = list(ENCODER_REGISTRY._obj_map.keys())
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available models: {available_models}")
    
    # Get the encoder class
    encoder_class = ENCODER_REGISTRY.get(encoder_type)
    
    # Create the model instance
    try:
        model = encoder_class(args, num_classes=num_classes)
        logger.info(f"Successfully built {encoder_type} model")
        return model
    except Exception as e:
        logger.error(f"Error building model {encoder_type}: {str(e)}")
        raise
