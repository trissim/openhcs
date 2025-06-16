"""
Image processors for different backends.

This module contains image processing functions implemented for different
computational backends (numpy, cupy, torch, tensorflow, jax).

Each processor module provides the same set of functions but optimized
for its specific backend.
"""

# Import all processor modules to ensure they're available for function registry scanning
from . import numpy_processor
from . import cupy_processor
from . import torch_processor
from . import tensorflow_processor
from . import jax_processor
from . import pyclesperanto_processor

__all__ = [
    'numpy_processor',
    'cupy_processor',
    'torch_processor',
    'tensorflow_processor',
    'jax_processor',
    'pyclesperanto_processor'
]
