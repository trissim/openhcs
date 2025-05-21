"""
Stack utilities module for OpenHCS.

This module provides functions for stacking 2D slices into a 3D array
and unstacking a 3D array into 2D slices, with explicit memory type handling.

This module enforces Clause 278 — Mandatory 3D Output Enforcement:
All functions must return a 3D array of shape [Z, Y, X], even when operating
on a single 2D slice. No logic may check, coerce, or infer rank at unstack time.
"""

import logging
from typing import Any, List

import numpy as np

from openhcs.constants.constants import (GPU_MEMORY_TYPES, MEMORY_TYPE_CUPY,
                                            MEMORY_TYPE_JAX, MEMORY_TYPE_NUMPY,
                                            MEMORY_TYPE_TENSORFLOW,
                                            MEMORY_TYPE_TORCH, MemoryType)
from openhcs.core.memory import MemoryWrapper

logger = logging.getLogger(__name__)


def _is_2d(data: Any) -> bool:
    """
    Check if data is a 2D array.

    Args:
        data: Data to check

    Returns:
        True if data is 2D, False otherwise
    """
    # Check if data has a shape attribute
    if not hasattr(data, 'shape'):
        return False

    # Check if shape has length 2
    return len(data.shape) == 2


def _is_3d(data: Any) -> bool:
    """
    Check if data is a 3D array.

    Args:
        data: Data to check

    Returns:
        True if data is 3D, False otherwise
    """
    # Check if data has a shape attribute
    if not hasattr(data, 'shape'):
        return False

    # Check if shape has length 3
    return len(data.shape) == 3


def _detect_memory_type(data: Any) -> str:
    """
    Detect the memory type of the data.

    STRICT VALIDATION: Fails loudly if the memory type cannot be detected.
    No automatic fallback to a default memory type.

    Args:
        data: The data to detect the memory type of

    Returns:
        The detected memory type

    Raises:
        ValueError: If the memory type cannot be detected
    """
    # Check if it's a MemoryWrapper
    if isinstance(data, MemoryWrapper):
        return data.memory_type

    # Check if it's a numpy array
    if isinstance(data, np.ndarray):
        return MemoryType.NUMPY.value

    # Check if it's a cupy array
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return MemoryType.CUPY.value
    except ImportError:
        pass

    # Check if it's a torch tensor
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return MemoryType.TORCH.value
    except ImportError:
        pass

    # Check if it's a tensorflow tensor
    try:
        import tensorflow as tf
        if isinstance(data, tf.Tensor):
            return MemoryType.TENSORFLOW.value
    except ImportError:
        pass

    # Check if it's a JAX array
    try:
        import jax
        import jax.numpy as jnp
        if isinstance(data, jnp.ndarray):
            return MemoryType.JAX.value
    except ImportError:
        pass

    # Fail loudly if we can't detect the type
    raise ValueError(f"Could not detect memory type of {type(data)}")


def _enforce_gpu_device_requirements(memory_type: str, gpu_id: int) -> None:
    """
    Enforce GPU device requirements.

    Args:
        memory_type: The memory type
        gpu_id: The GPU device ID

    Raises:
        ValueError: If gpu_id is negative
    """
    # For GPU memory types, validate gpu_id
    if memory_type in {mem_type.value for mem_type in GPU_MEMORY_TYPES}:
        if gpu_id < 0:
            raise ValueError(f"Invalid GPU device ID: {gpu_id}. Must be a non-negative integer.")


def stack_slices(slices: List[Any], memory_type: str, gpu_id: int, allow_single_slice: bool = False) -> Any:
    """
    Stack 2D slices into a 3D array with the specified memory type.

    STRICT VALIDATION: Assumes all slices are 2D arrays.
    No automatic handling of improper inputs.

    Args:
        slices: List of 2D slices (numpy arrays, cupy arrays, torch tensors, etc.)
        memory_type: The memory type to use for the stacked array (REQUIRED)
        gpu_id: The target GPU device ID (REQUIRED)
        allow_single_slice: If True, allows stacking a single slice into a 3D array with shape [1, Y, X].
                           If False (default), raises an error when only one slice is provided.

    Returns:
        A 3D array with the specified memory type of shape [Z, Y, X]

    Raises:
        ValueError: If memory_type is not supported or slices is empty
        ValueError: If slices contains only one element and allow_single_slice is False
        ValueError: If gpu_id is negative for GPU memory types
        ValueError: If slices are not 2D arrays
        MemoryConversionError: If conversion fails
    """
    if not slices:
        raise ValueError("Cannot stack empty list of slices")

    # Check for single slice case
    if len(slices) == 1 and not allow_single_slice:
        raise ValueError("Cannot stack a single slice unless allow_single_slice=True. "
                         "This prevents silent shape coercion and enforces explicit intent.")

    # Verify all slices are 2D
    for i, slice_data in enumerate(slices):
        if not _is_2d(slice_data):
            raise ValueError(f"Slice at index {i} is not a 2D array. All slices must be 2D.")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert each slice to the target memory type
    converted_slices = []
    for i, slice_data in enumerate(slices):
        # Convert to target memory type using MemoryWrapper (Clause 290)
        # This improves traceability, error validation, and GPU discipline
        wrapped = MemoryWrapper(slice_data, memory_type=_detect_memory_type(slice_data), gpu_id=gpu_id)

        # Use the appropriate conversion method based on the target memory type
        if memory_type == MEMORY_TYPE_NUMPY:
            converted_slice = wrapped.to_numpy()
        elif memory_type == MEMORY_TYPE_CUPY:
            converted_slice = wrapped.to_cupy(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_TORCH:
            converted_slice = wrapped.to_torch(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_TENSORFLOW:
            converted_slice = wrapped.to_tensorflow(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_JAX:
            converted_slice = wrapped.to_jax(allow_cpu_roundtrip=False)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        converted_slices.append(converted_slice)

    # Stack the converted slices using the appropriate function
    if memory_type == MemoryType.NUMPY.value:
        return np.stack(converted_slices)
    elif memory_type == MemoryType.CUPY.value:
        try:
            import cupy as cp
            return cp.stack(converted_slices)
        except ImportError:
            raise ValueError(f"CuPy is required for memory type {memory_type}")
    elif memory_type == MemoryType.TORCH.value:
        try:
            import torch
            return torch.stack(converted_slices)
        except ImportError:
            raise ValueError(f"PyTorch is required for memory type {memory_type}")
    elif memory_type == MemoryType.TENSORFLOW.value:
        try:
            import tensorflow as tf
            return tf.stack(converted_slices)
        except ImportError:
            raise ValueError(f"TensorFlow is required for memory type {memory_type}")
    elif memory_type == MemoryType.JAX.value:
        try:
            import jax
            import jax.numpy as jnp
            return jnp.stack(converted_slices)
        except ImportError:
            raise ValueError(f"JAX is required for memory type {memory_type}")
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")


def unstack_slices(array: Any, memory_type: str, gpu_id: int, validate_slices: bool = True) -> List[Any]:
    """
    Split a 3D array into 2D slices along axis 0 and convert to the specified memory type.

    STRICT VALIDATION: Input must be a 3D array. No automatic handling of improper inputs.

    Args:
        array: 3D array to split - MUST BE 3D
        memory_type: The memory type to use for the output slices (REQUIRED)
        gpu_id: The target GPU device ID (REQUIRED)
        validate_slices: If True, validates that each extracted slice is 2D

    Returns:
        List of 2D slices in the specified memory type

    Raises:
        ValueError: If array is not 3D
        ValueError: If validate_slices is True and any extracted slice is not 2D
        ValueError: If gpu_id is negative for GPU memory types
        ValueError: If memory_type is not supported
        MemoryConversionError: If conversion fails
    """
    # Verify the array is 3D - fail loudly if not
    if not _is_3d(array):
        raise ValueError(f"Array must be 3D, got shape {getattr(array, 'shape', 'unknown')}")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert to target memory type using MemoryWrapper (Clause 290)
    # This improves traceability, error validation, and GPU discipline
    wrapped = MemoryWrapper(array, memory_type=_detect_memory_type(array), gpu_id=gpu_id)

    # Use the appropriate conversion method based on the target memory type
    if memory_type == MemoryType.NUMPY.value:
        array = wrapped.to_numpy()
    elif memory_type == MemoryType.CUPY.value:
        array = wrapped.to_cupy(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.TORCH.value:
        array = wrapped.to_torch(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.TENSORFLOW.value:
        array = wrapped.to_tensorflow(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.JAX.value:
        array = wrapped.to_jax(allow_cpu_roundtrip=False)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    # Extract slices along axis 0 (already in the target memory type)
    slices = [array[i] for i in range(array.shape[0])]

    # Validate that all extracted slices are 2D if requested
    if validate_slices:
        for i, slice_data in enumerate(slices):
            if not _is_2d(slice_data):
                raise ValueError(f"Extracted slice at index {i} is not 2D. This indicates a malformed 3D array.")

    return slices
