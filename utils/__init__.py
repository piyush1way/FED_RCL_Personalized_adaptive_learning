"""Useful utils
"""
from .data import *
from .helper import *
from .loss import *
from .sampler import *
from .misc import *
from .io_utils import *
from .meter import *

# Adjust steepness (higher = sharper cutoff, lower = smoother transition)
steepness = 6.0  # More gradual than current 8.0

