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
                                            MEMORY_TYPE_PYCLESPERANTO, MEMORY_TYPE_TENSORFLOW,
                                            MEMORY_TYPE_TORCH, MemoryType)
from openhcs.core.memory import MemoryWrapper
from openhcs.core.utils import optional_import

logger = logging.getLogger(__name__)

# 🔍 MEMORY CONVERSION LOGGING: Test log to verify logger is working
logger.info("🔄 STACK_UTILS: Module loaded - memory conversion logging enabled")


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
    cp = optional_import("cupy")
    if cp is not None and isinstance(data, cp.ndarray):
        return MemoryType.CUPY.value

    # Check if it's a torch tensor
    torch = optional_import("torch")
    if torch is not None and isinstance(data, torch.Tensor):
        return MemoryType.TORCH.value

    # Check if it's a tensorflow tensor
    tf = optional_import("tensorflow")
    if tf is not None and isinstance(data, tf.Tensor):
        return MemoryType.TENSORFLOW.value

    # Check if it's a JAX array
    jax = optional_import("jax")
    jnp = optional_import("jax.numpy") if jax is not None else None
    if jnp is not None and isinstance(data, jnp.ndarray):
        return MemoryType.JAX.value

    # Check if it's a pyclesperanto array
    cle = optional_import("pyclesperanto")
    if cle is not None and hasattr(cle, 'Array') and isinstance(data, cle.Array):
        return MemoryType.PYCLESPERANTO.value

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


def stack_slices(slices: List[Any], memory_type: str, gpu_id: int) -> Any:
    """
    Stack 2D slices into a 3D array with the specified memory type.

    STRICT VALIDATION: Assumes all slices are 2D arrays.
    No automatic handling of improper inputs.

    Args:
        slices: List of 2D slices (numpy arrays, cupy arrays, torch tensors, etc.)
        memory_type: The memory type to use for the stacked array (REQUIRED)
        gpu_id: The target GPU device ID (REQUIRED)

    Returns:
        A 3D array with the specified memory type of shape [Z, Y, X]

    Raises:
        ValueError: If memory_type is not supported or slices is empty
        ValueError: If gpu_id is negative for GPU memory types
        ValueError: If slices are not 2D arrays
        MemoryConversionError: If conversion fails
    """
    if not slices:
        raise ValueError("Cannot stack empty list of slices")

    # 🔍 MEMORY CONVERSION LOGGING: Log stacking operation start
    logger.info(f"🔄 STACK_SLICES: Starting stack operation - {len(slices)} slices → target_memory_type: {memory_type}, gpu_id: {gpu_id}")

    # Verify all slices are 2D
    for i, slice_data in enumerate(slices):
        if not _is_2d(slice_data):
            raise ValueError(f"Slice at index {i} is not a 2D array. All slices must be 2D.")

    # 🔍 MEMORY CONVERSION LOGGING: Log input slice types
    input_types = []
    for i, slice_data in enumerate(slices):
        detected_type = _detect_memory_type(slice_data)
        input_types.append(detected_type)
        if i < 3:  # Log first 3 slices for brevity
            logger.info(f"🔄 STACK_SLICES: Input slice[{i}] - type: {detected_type}, shape: {getattr(slice_data, 'shape', 'unknown')}")

    unique_input_types = set(input_types)
    logger.info(f"🔄 STACK_SLICES: Input types summary - unique_types: {unique_input_types}, total_slices: {len(slices)}")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert each slice to the target memory type
    converted_slices = []
    conversion_count = 0
    for i, slice_data in enumerate(slices):
        # Convert to target memory type using MemoryWrapper (Clause 290)
        # This improves traceability, error validation, and GPU discipline
        source_type = _detect_memory_type(slice_data)
        wrapped = MemoryWrapper(slice_data, memory_type=source_type, gpu_id=gpu_id)

        # 🔍 MEMORY CONVERSION LOGGING: Log conversion attempt
        if source_type != memory_type:
            logger.info(f"🔄 STACK_SLICES: Converting slice[{i}] - {source_type} → {memory_type}")
            conversion_count += 1
        else:
            logger.debug(f"🔄 STACK_SLICES: No conversion needed for slice[{i}] - already {memory_type}")

        # Use the appropriate conversion method based on the target memory type
        if memory_type == MEMORY_TYPE_NUMPY:
            converted_wrapper = wrapped.to_numpy()
        elif memory_type == MEMORY_TYPE_CUPY:
            converted_wrapper = wrapped.to_cupy(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_TORCH:
            converted_wrapper = wrapped.to_torch(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_TENSORFLOW:
            converted_wrapper = wrapped.to_tensorflow(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_JAX:
            converted_wrapper = wrapped.to_jax(allow_cpu_roundtrip=False)
        elif memory_type == MEMORY_TYPE_PYCLESPERANTO:
            converted_wrapper = wrapped.to_pyclesperanto(allow_cpu_roundtrip=False)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # 🔍 MEMORY CONVERSION LOGGING: Log successful conversion
        converted_type = _detect_memory_type(converted_wrapper.data)
        if source_type != memory_type:
            logger.info(f"🔄 STACK_SLICES: Conversion complete slice[{i}] - {source_type} → {converted_type} ✓")

        # Extract the raw data from the MemoryWrapper for stacking
        converted_slices.append(converted_wrapper.data)

    # 🔍 MEMORY CONVERSION LOGGING: Log conversion summary
    logger.info(f"🔄 STACK_SLICES: Conversion summary - {conversion_count}/{len(slices)} slices converted to {memory_type}")

    # Stack the converted slices using the appropriate function
    # 🔍 MEMORY CONVERSION LOGGING: Log stacking operation
    logger.info(f"🔄 STACK_SLICES: Stacking {len(converted_slices)} converted slices using {memory_type} backend")

    if memory_type == MemoryType.NUMPY.value:
        result = np.stack(converted_slices)
    elif memory_type == MemoryType.CUPY.value:
        cp = optional_import("cupy")
        if cp is None:
            raise ValueError(f"CuPy is required for memory type {memory_type}")
        result = cp.stack(converted_slices)
    elif memory_type == MemoryType.TORCH.value:
        torch = optional_import("torch")
        if torch is None:
            raise ValueError(f"PyTorch is required for memory type {memory_type}")
        result = torch.stack(converted_slices)
    elif memory_type == MemoryType.TENSORFLOW.value:
        tf = optional_import("tensorflow")
        if tf is None:
            raise ValueError(f"TensorFlow is required for memory type {memory_type}")
        result = tf.stack(converted_slices)
    elif memory_type == MemoryType.JAX.value:
        jax = optional_import("jax")
        jnp = optional_import("jax.numpy") if jax is not None else None
        if jnp is None:
            raise ValueError(f"JAX is required for memory type {memory_type}")
        result = jnp.stack(converted_slices)
    elif memory_type == MemoryType.PYCLESPERANTO.value:
        cle = optional_import("pyclesperanto")
        if cle is None:
            raise ValueError(f"pyclesperanto is required for memory type {memory_type}")

        # pyclesperanto doesn't have a direct stack function, so we need to create and copy
        if not converted_slices:
            raise ValueError("Cannot stack empty list of slices")

        # Get shape from first slice
        first_slice = converted_slices[0]
        stack_shape = (len(converted_slices), first_slice.shape[0], first_slice.shape[1])

        # Create result array
        result = cle.create(stack_shape, dtype=first_slice.dtype)

        # Copy each slice into the result
        for i, slice_data in enumerate(converted_slices):
            cle.copy_slice(slice_data, result, i)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    # 🔍 MEMORY CONVERSION LOGGING: Log final result
    result_type = _detect_memory_type(result)
    result_shape = getattr(result, 'shape', 'unknown')
    logger.info(f"🔄 STACK_SLICES: Complete - result_type: {result_type}, result_shape: {result_shape}")

    return result


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
    # 🔍 MEMORY CONVERSION LOGGING: Log unstacking operation start
    input_type = _detect_memory_type(array)
    input_shape = getattr(array, 'shape', 'unknown')
    logger.info(f"🔄 UNSTACK_SLICES: Starting unstack operation - input_type: {input_type}, input_shape: {input_shape} → target_memory_type: {memory_type}, gpu_id: {gpu_id}")

    # Verify the array is 3D - fail loudly if not
    if not _is_3d(array):
        raise ValueError(f"Array must be 3D, got shape {getattr(array, 'shape', 'unknown')}")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert to target memory type using MemoryWrapper (Clause 290)
    # This improves traceability, error validation, and GPU discipline
    source_type = _detect_memory_type(array)
    wrapped = MemoryWrapper(array, memory_type=source_type, gpu_id=gpu_id)

    # 🔍 MEMORY CONVERSION LOGGING: Log conversion attempt
    if source_type != memory_type:
        logger.info(f"🔄 UNSTACK_SLICES: Converting array - {source_type} → {memory_type}")
    else:
        logger.debug(f"🔄 UNSTACK_SLICES: No conversion needed - already {memory_type}")

    # Use the appropriate conversion method based on the target memory type
    if memory_type == MemoryType.NUMPY.value:
        array_wrapper = wrapped.to_numpy()
    elif memory_type == MemoryType.CUPY.value:
        array_wrapper = wrapped.to_cupy(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.TORCH.value:
        array_wrapper = wrapped.to_torch(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.TENSORFLOW.value:
        array_wrapper = wrapped.to_tensorflow(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.JAX.value:
        array_wrapper = wrapped.to_jax(allow_cpu_roundtrip=False)
    elif memory_type == MemoryType.PYCLESPERANTO.value:
        array_wrapper = wrapped.to_pyclesperanto(allow_cpu_roundtrip=False)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    # 🔍 MEMORY CONVERSION LOGGING: Log successful conversion
    converted_type = _detect_memory_type(array_wrapper.data)
    if source_type != memory_type:
        logger.info(f"🔄 UNSTACK_SLICES: Conversion complete - {source_type} → {converted_type} ✓")

    # Extract the raw data from the MemoryWrapper
    array = array_wrapper.data

    # Extract slices along axis 0 (already in the target memory type)
    slices = [array[i] for i in range(array.shape[0])]

    # 🔍 MEMORY CONVERSION LOGGING: Log slice extraction
    logger.info(f"🔄 UNSTACK_SLICES: Extracted {len(slices)} slices from 3D array")

    # Validate that all extracted slices are 2D if requested
    if validate_slices:
        for i, slice_data in enumerate(slices):
            if not _is_2d(slice_data):
                raise ValueError(f"Extracted slice at index {i} is not 2D. This indicates a malformed 3D array.")

    # 🔍 MEMORY CONVERSION LOGGING: Log final result
    if slices:
        first_slice_type = _detect_memory_type(slices[0])
        first_slice_shape = getattr(slices[0], 'shape', 'unknown')
        logger.info(f"🔄 UNSTACK_SLICES: Complete - output_slices: {len(slices)}, slice_type: {first_slice_type}, slice_shape: {first_slice_shape}")
    else:
        logger.warning(f"🔄 UNSTACK_SLICES: Complete - no slices extracted (empty array)")

    return slices
