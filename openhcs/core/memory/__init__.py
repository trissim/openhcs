"""
Memory module for OpenHCS.

This module provides classes and utilities for working with in-memory data arrays
with explicit type declarations and conversion methods, enforcing Clause 251
(Declarative Memory Conversion Interface) and Clause 106-A (Declared Memory Types).
"""

from openhcs.constants.constants import MemoryType

from .decorators import cupy, jax, memory_types, numpy, tensorflow, torch
from .wrapper import MemoryWrapper

# Define memory type constants
MEMORY_TYPE_NUMPY = MemoryType.NUMPY.value
MEMORY_TYPE_CUPY = MemoryType.CUPY.value
MEMORY_TYPE_TORCH = MemoryType.TORCH.value
MEMORY_TYPE_TENSORFLOW = MemoryType.TENSORFLOW.value
MEMORY_TYPE_JAX = MemoryType.JAX.value

__all__ = [
    'MemoryWrapper',
    'MEMORY_TYPE_NUMPY',
    'MEMORY_TYPE_CUPY',
    'MEMORY_TYPE_TORCH',
    'MEMORY_TYPE_TENSORFLOW',
    'MEMORY_TYPE_JAX',
    'memory_types',
    'numpy',
    'cupy',
    'torch',
    'tensorflow',
    'jax',
]
