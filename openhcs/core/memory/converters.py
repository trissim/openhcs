"""
Memory conversion functions for OpenHCS.

This module provides functions for converting between different memory types,
enforcing Clause 65 (Fail Loudly), Clause 88 (No Inferred Capabilities),
and Clause 251 (Declarative Memory Conversion).
"""

from typing import Any

from openhcs.constants.constants import MemoryType

from .conversion_functions import (_cupy_to_jax, _cupy_to_numpy,
                                   _cupy_to_pyclesperanto, _cupy_to_tensorflow, _cupy_to_torch,
                                   _jax_to_cupy, _jax_to_jax, _jax_to_numpy, _jax_to_pyclesperanto,
                                   _jax_to_tensorflow, _jax_to_torch,
                                   _numpy_to_cupy, _numpy_to_jax,
                                   _numpy_to_pyclesperanto, _numpy_to_tensorflow, _numpy_to_torch,
                                   _pyclesperanto_to_cupy, _pyclesperanto_to_jax, _pyclesperanto_to_numpy,
                                   _pyclesperanto_to_pyclesperanto, _pyclesperanto_to_tensorflow, _pyclesperanto_to_torch,
                                   _tensorflow_to_cupy, _tensorflow_to_jax, _tensorflow_to_pyclesperanto,
                                   _tensorflow_to_numpy, _tensorflow_to_torch,
                                   _torch_to_cupy, _torch_to_jax, _torch_to_pyclesperanto,
                                   _torch_to_numpy, _torch_to_tensorflow)


def validate_memory_type(memory_type: str) -> None:
    """
    Validate that a memory type is supported.

    Args:
        memory_type: The memory type to validate

    Raises:
        ValueError: If the memory type is not supported
    """
    if memory_type not in [m.value for m in MemoryType]:
        raise ValueError(
            f"Unsupported memory type: {memory_type}. "
            f"Supported types are: {', '.join([m.value for m in MemoryType])}"
        )


def validate_data_compatibility(data: Any, memory_type: str) -> None:
    """
    Validate that data is compatible with a memory type.

    Args:
        data: The data to validate
        memory_type: The memory type to validate against

    Raises:
        ValueError: If the data is not compatible with the memory type
    """
    # This is a placeholder for future validation logic
    # Currently, we don't have a way to validate data compatibility
    # without importing the memory type modules
    pass


def convert_memory(
    data: Any,
    source_type: str,
    target_type: str,
    gpu_id: int,
    allow_cpu_roundtrip: bool = False,
) -> Any:
    """
    Convert data between memory types.

    Args:
        data: The data to convert
        source_type: The source memory type
        target_type: The target memory type
        gpu_id: The target GPU device ID (required)
        allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

    Returns:
        The converted data

    Raises:
        ValueError: If source_type or target_type is not supported
        MemoryConversionError: If conversion fails
    """
    # If source and target types are the same, return the data as is
    if source_type == target_type:
        return data

    # NumPy to X conversions
    if source_type == MemoryType.NUMPY.value:
        if target_type == MemoryType.CUPY.value:
            return _numpy_to_cupy(data, gpu_id)
        elif target_type == MemoryType.TORCH.value:
            return _numpy_to_torch(data, gpu_id)
        elif target_type == MemoryType.TENSORFLOW.value:
            return _numpy_to_tensorflow(data, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _numpy_to_jax(data, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _numpy_to_pyclesperanto(data, gpu_id)

    # CuPy to X conversions
    elif source_type == MemoryType.CUPY.value:
        if target_type == MemoryType.NUMPY.value:
            return _cupy_to_numpy(data)
        elif target_type == MemoryType.TORCH.value:
            return _cupy_to_torch(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TENSORFLOW.value:
            return _cupy_to_tensorflow(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _cupy_to_jax(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _cupy_to_pyclesperanto(data, allow_cpu_roundtrip, gpu_id)

    # PyTorch to X conversions
    elif source_type == MemoryType.TORCH.value:
        if target_type == MemoryType.NUMPY.value:
            return _torch_to_numpy(data)
        elif target_type == MemoryType.CUPY.value:
            return _torch_to_cupy(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TENSORFLOW.value:
            return _torch_to_tensorflow(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _torch_to_jax(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _torch_to_pyclesperanto(data, allow_cpu_roundtrip, gpu_id)

    # TensorFlow to X conversions
    elif source_type == MemoryType.TENSORFLOW.value:
        if target_type == MemoryType.NUMPY.value:
            return _tensorflow_to_numpy(data)
        elif target_type == MemoryType.CUPY.value:
            return _tensorflow_to_cupy(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TORCH.value:
            return _tensorflow_to_torch(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _tensorflow_to_jax(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _tensorflow_to_pyclesperanto(data, allow_cpu_roundtrip, gpu_id)

    # JAX to X conversions
    elif source_type == MemoryType.JAX.value:
        if target_type == MemoryType.NUMPY.value:
            return _jax_to_numpy(data)
        elif target_type == MemoryType.CUPY.value:
            return _jax_to_cupy(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TORCH.value:
            return _jax_to_torch(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TENSORFLOW.value:
            return _jax_to_tensorflow(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _jax_to_jax(data, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _jax_to_pyclesperanto(data, allow_cpu_roundtrip, gpu_id)

    # pyclesperanto to X conversions
    elif source_type == MemoryType.PYCLESPERANTO.value:
        if target_type == MemoryType.NUMPY.value:
            return _pyclesperanto_to_numpy(data)
        elif target_type == MemoryType.CUPY.value:
            return _pyclesperanto_to_cupy(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TORCH.value:
            return _pyclesperanto_to_torch(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.TENSORFLOW.value:
            return _pyclesperanto_to_tensorflow(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.JAX.value:
            return _pyclesperanto_to_jax(data, allow_cpu_roundtrip, gpu_id)
        elif target_type == MemoryType.PYCLESPERANTO.value:
            return _pyclesperanto_to_pyclesperanto(data)

    # If we get here, the conversion is not supported
    raise ValueError(
        f"Unsupported memory conversion: {source_type} -> {target_type}"
    )
