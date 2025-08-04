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
logger.debug("🔄 STACK_UTILS: Module loaded - memory conversion logging enabled")


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

    # Verify all slices are 2D
    for i, slice_data in enumerate(slices):
        if not _is_2d(slice_data):
            raise ValueError(f"Slice at index {i} is not a 2D array. All slices must be 2D.")

    # Analyze input types for conversion planning (minimal logging)
    input_types = [_detect_memory_type(slice_data) for slice_data in slices]
    unique_input_types = set(input_types)
    needs_conversion = memory_type not in unique_input_types or len(unique_input_types) > 1

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Pre-allocate the final 3D array to avoid intermediate list and final stack operation
    first_slice = slices[0]
    stack_shape = (len(slices), first_slice.shape[0], first_slice.shape[1])

    # Create pre-allocated result array in target memory type
    if memory_type == MEMORY_TYPE_NUMPY:
        import numpy as np

        # Handle torch dtypes by converting a sample slice first
        first_slice_source_type = _detect_memory_type(first_slice)
        if first_slice_source_type == MEMORY_TYPE_TORCH:
            # Convert torch tensor to numpy to get compatible dtype
            from openhcs.core.memory.converters import convert_memory
            sample_converted = convert_memory(
                data=first_slice,
                source_type=first_slice_source_type,
                target_type=memory_type,
                gpu_id=gpu_id,
                allow_cpu_roundtrip=True  # Allow CPU roundtrip for numpy conversion
            )
            result = np.empty(stack_shape, dtype=sample_converted.dtype)
        else:
            # Use dtype directly for non-torch types
            result = np.empty(stack_shape, dtype=first_slice.dtype)
    elif memory_type == MEMORY_TYPE_CUPY:
        cupy = optional_import("cupy")
        if cupy is None:
            raise ValueError(f"CuPy is required for memory type {memory_type}")
        with cupy.cuda.Device(gpu_id):
            result = cupy.empty(stack_shape, dtype=first_slice.dtype)
    elif memory_type == MEMORY_TYPE_TORCH:
        torch = optional_import("torch")
        if torch is None:
            raise ValueError(f"PyTorch is required for memory type {memory_type}")

        # Convert first slice to get the correct torch dtype
        from openhcs.core.memory.converters import convert_memory
        first_slice_source_type = _detect_memory_type(first_slice)
        sample_converted = convert_memory(
            data=first_slice,
            source_type=first_slice_source_type,
            target_type=memory_type,
            gpu_id=gpu_id,
            allow_cpu_roundtrip=False
        )

        result = torch.empty(stack_shape, dtype=sample_converted.dtype, device=sample_converted.device)
    elif memory_type == MEMORY_TYPE_TENSORFLOW:
        tf = optional_import("tensorflow")
        if tf is None:
            raise ValueError(f"TensorFlow is required for memory type {memory_type}")
        with tf.device(f"/device:GPU:{gpu_id}"):
            result = tf.zeros(stack_shape, dtype=first_slice.dtype)  # TF doesn't have empty()
    elif memory_type == MEMORY_TYPE_JAX:
        jax = optional_import("jax")
        if jax is None:
            raise ValueError(f"JAX is required for memory type {memory_type}")
        jnp = optional_import("jax.numpy")
        if jnp is None:
            raise ValueError(f"JAX is required for memory type {memory_type}")
        result = jnp.empty(stack_shape, dtype=first_slice.dtype)
    elif memory_type == MEMORY_TYPE_PYCLESPERANTO:
        cle = optional_import("pyclesperanto")
        if cle is None:
            raise ValueError(f"pyclesperanto is required for memory type {memory_type}")
        # For pyclesperanto, we'll build the result using concatenate_along_z
        # Don't pre-allocate here, we'll handle it in the loop below
        result = None
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    # Convert each slice and assign to result array
    conversion_count = 0

    # Special handling for pyclesperanto - build using concatenate_along_z
    if memory_type == MEMORY_TYPE_PYCLESPERANTO:
        cle = optional_import("pyclesperanto")
        converted_slices = []

        for i, slice_data in enumerate(slices):
            source_type = _detect_memory_type(slice_data)

            # Track conversions for batch logging
            if source_type != memory_type:
                conversion_count += 1

            # Convert slice to pyclesperanto
            if source_type == memory_type:
                converted_data = slice_data
            else:
                from openhcs.core.memory.converters import convert_memory
                converted_data = convert_memory(
                    data=slice_data,
                    source_type=source_type,
                    target_type=memory_type,
                    gpu_id=gpu_id,
                    allow_cpu_roundtrip=False
                )

            # Ensure slice is 2D, expand to 3D single slice if needed
            if converted_data.ndim == 2:
                # Convert 2D slice to 3D single slice using expand_dims equivalent
                converted_data = cle.push(cle.pull(converted_data)[None, ...])

            converted_slices.append(converted_data)

        # Build 3D result using efficient batch concatenation
        if len(converted_slices) == 1:
            result = converted_slices[0]
        else:
            # Use divide-and-conquer approach for better performance
            # This reduces O(N²) copying to O(N log N)
            slices_to_concat = converted_slices[:]
            while len(slices_to_concat) > 1:
                new_slices = []
                for i in range(0, len(slices_to_concat), 2):
                    if i + 1 < len(slices_to_concat):
                        # Concatenate pair
                        combined = cle.concatenate_along_z(slices_to_concat[i], slices_to_concat[i + 1])
                        new_slices.append(combined)
                    else:
                        # Odd one out
                        new_slices.append(slices_to_concat[i])
                slices_to_concat = new_slices
            result = slices_to_concat[0]

    else:
        # Standard handling for other memory types
        for i, slice_data in enumerate(slices):
            source_type = _detect_memory_type(slice_data)

            # Track conversions for batch logging
            if source_type != memory_type:
                conversion_count += 1

            # Direct conversion without MemoryWrapper overhead
            if source_type == memory_type:
                converted_data = slice_data
            else:
                from openhcs.core.memory.converters import convert_memory
                converted_data = convert_memory(
                    data=slice_data,
                    source_type=source_type,
                    target_type=memory_type,
                    gpu_id=gpu_id,
                    allow_cpu_roundtrip=False
                )

            # Assign converted slice directly to pre-allocated result array
            # Handle JAX immutability
            if memory_type == MEMORY_TYPE_JAX:
                result = result.at[i].set(converted_data)
            else:
                result[i] = converted_data

    # 🔍 MEMORY CONVERSION LOGGING: Only log when conversions happen or issues occur
    if conversion_count > 0:
        logger.debug(f"🔄 STACK_SLICES: Converted {conversion_count}/{len(slices)} slices to {memory_type}")
    # Silent success for no-conversion cases to reduce log pollution

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
    # Detect input type and check if conversion is needed
    input_type = _detect_memory_type(array)
    input_shape = getattr(array, 'shape', 'unknown')
    needs_conversion = input_type != memory_type

    # Verify the array is 3D - fail loudly if not
    if not _is_3d(array):
        raise ValueError(f"Array must be 3D, got shape {getattr(array, 'shape', 'unknown')}")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert to target memory type using direct convert_memory call
    # Bypass MemoryWrapper to eliminate object creation overhead
    source_type = input_type  # Reuse already detected type from line 286

    # Direct conversion without MemoryWrapper overhead
    if source_type == memory_type:
        # No conversion needed - silent success to reduce log pollution
        pass
    else:
        # Use direct convert_memory call and log the conversion
        from openhcs.core.memory.converters import convert_memory
        logger.debug(f"🔄 UNSTACK_SLICES: Converting array - {source_type} → {memory_type}")
        array = convert_memory(
            data=array,
            source_type=source_type,
            target_type=memory_type,
            gpu_id=gpu_id,
            allow_cpu_roundtrip=False
        )

    # Extract slices along axis 0 (already in the target memory type)
    slices = [array[i] for i in range(array.shape[0])]

    # Validate that all extracted slices are 2D if requested
    if validate_slices:
        for i, slice_data in enumerate(slices):
            if not _is_2d(slice_data):
                raise ValueError(f"Extracted slice at index {i} is not 2D. This indicates a malformed 3D array.")

    # 🔍 MEMORY CONVERSION LOGGING: Only log conversions or issues
    if source_type != memory_type:
        logger.debug(f"🔄 UNSTACK_SLICES: Converted and extracted {len(slices)} slices")
    elif len(slices) == 0:
        logger.warning(f"🔄 UNSTACK_SLICES: No slices extracted (empty array)")
    # Silent success for no-conversion cases to reduce log pollution

    return slices
