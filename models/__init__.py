# from models.basic import *
# from models.VGG9 import *
# from models.resnet import *
# # from models.generator import *

# from models.resnet_base import *

# from models.MobileNet import *
# from models.SqueezeNet import *
# from models.ShuffleNet import *

# from models.build import build_encoder
# Import Basic Models
from models.basic import BasicModel  # Replace with actual class name

# Import CNN-Based Models
from models.VGG9 import VGG9
from models.resnet import ResNet18, ResNet34, ResNet50, PersonalizedResNet18
from models.resnet_base import ResNetBase  # If needed

# Import Lightweight Models (if used)
from models.MobileNet import MobileNetV2  # Replace if needed
from models.SqueezeNet import SqueezeNet
from models.ShuffleNet import ShuffleNetV2

# Import Model Builder
from models.build import build_encoder
