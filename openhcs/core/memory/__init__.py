"""Memory module for OpenHCS.

This module re-exports from arraybridge for memory type conversion utilities.
MemoryType and DtypeConversion are kept in openhcs.constants for backward compatibility.
"""

# Re-export from arraybridge
from arraybridge import (
    # Converters
    convert_memory,
    detect_memory_type,
    DtypeConversion,
    # Stack utilities
    stack_slices,
    unstack_slices,
    # Slice processing
    process_slices,
    # GPU cleanup
    cleanup_all_gpu_frameworks,
    # Exceptions
    MemoryConversionError,
    # Scaling
    SCALING_FUNCTIONS,
    # Framework config
    _FRAMEWORK_CONFIG,
    _FRAMEWORK_OPS,
    # OOM recovery
    _execute_with_oom_recovery,
    # Utils
    _ensure_module,
    _supports_dlpack,
    _get_device_id,
)

# OpenHCS wraps arraybridge decorators to preserve compiler metadata while
# leaving conversion semantics in arraybridge.
from . import decorators as decorators
from .decorators import (
    cupy,
    jax,
    memory_types,
    numpy,
    pyclesperanto,
    tensorflow,
    torch,
)

# Keep MemoryType from openhcs constants for backward compatibility
from openhcs.constants.constants import MemoryType

# Define memory type constants
MEMORY_TYPE_NUMPY = MemoryType.NUMPY.value
MEMORY_TYPE_CUPY = MemoryType.CUPY.value
MEMORY_TYPE_TORCH = MemoryType.TORCH.value
MEMORY_TYPE_TENSORFLOW = MemoryType.TENSORFLOW.value
MEMORY_TYPE_JAX = MemoryType.JAX.value
MEMORY_TYPE_PYCLESPERANTO = MemoryType.PYCLESPERANTO.value

__all__ = [
    # Converters
    'convert_memory',
    'detect_memory_type',
    # Memory type constants
    'MEMORY_TYPE_NUMPY',
    'MEMORY_TYPE_CUPY',
    'MEMORY_TYPE_TORCH',
    'MEMORY_TYPE_TENSORFLOW',
    'MEMORY_TYPE_JAX',
    'MEMORY_TYPE_PYCLESPERANTO',
    # Decorators
    'memory_types',
    'numpy',
    'cupy',
    'torch',
    'tensorflow',
    'jax',
    'pyclesperanto',
    'decorators',
    'DtypeConversion',
    # Stack utilities
    'stack_slices',
    'unstack_slices',
    # Slice processing
    'process_slices',
    # GPU cleanup
    'cleanup_all_gpu_frameworks',
    # Exceptions
    'MemoryConversionError',
    # Types
    'MemoryType',
    # Scaling
    'SCALING_FUNCTIONS',
    # Framework config
    '_FRAMEWORK_CONFIG',
    '_FRAMEWORK_OPS',
    # OOM recovery
    '_execute_with_oom_recovery',
    # Utils
    '_ensure_module',
    '_supports_dlpack',
    '_get_device_id',
]
