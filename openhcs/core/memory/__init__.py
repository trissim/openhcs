"""Memory module for OpenHCS.

This module re-exports from arraybridge for memory type conversion utilities.
MemoryType and DtypeConversion are kept in openhcs.constants for backward compatibility.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

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


def stack_runtime_slices(
    slices: Sequence[Any],
    memory_type: str,
    gpu_id: int,
) -> Any:
    """Stack an explicitly declared runtime-slice sequence along axis zero."""

    slice_values = tuple(slices)
    if not slice_values:
        raise ValueError("Runtime-slice stacking requires at least one slice.")
    numpy_slices = tuple(
        np.asarray(
            value
            if detect_memory_type(value) == MEMORY_TYPE_NUMPY
            else convert_memory(
                data=value,
                source_type=detect_memory_type(value),
                target_type=MEMORY_TYPE_NUMPY,
                gpu_id=gpu_id,
            )
        )
        for value in slice_values
    )
    shapes = tuple(tuple(value.shape) for value in numpy_slices)
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            "Runtime slices must have one exact shape before stacking; "
            f"got {shapes!r}."
        )
    stacked = np.stack(numpy_slices, axis=0)
    if memory_type == MEMORY_TYPE_NUMPY:
        return stacked
    return convert_memory(
        data=stacked,
        source_type=MEMORY_TYPE_NUMPY,
        target_type=memory_type,
        gpu_id=gpu_id,
    )


def unstack_runtime_slices(
    stack: Any,
    memory_type: str,
    gpu_id: int,
    *,
    expected_count: int | None = None,
) -> tuple[Any, ...]:
    """Split an explicitly declared leading runtime-slice axis."""

    source_type = detect_memory_type(stack)
    converted = (
        stack
        if source_type == memory_type
        else convert_memory(
            data=stack,
            source_type=source_type,
            target_type=memory_type,
            gpu_id=gpu_id,
        )
    )
    shape = tuple(int(value) for value in converted.shape)
    if not shape:
        raise ValueError("Runtime-slice unstacking requires a leading array axis.")
    if expected_count is not None and shape[0] != expected_count:
        raise ValueError(
            "Runtime-slice stack cardinality does not match its declaration: "
            f"{shape[0]} != {expected_count}."
        )
    return tuple(converted[index] for index in range(shape[0]))

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
    'stack_runtime_slices',
    'unstack_runtime_slices',
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
