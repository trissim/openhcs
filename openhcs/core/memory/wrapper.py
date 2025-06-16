"""
Memory wrapper implementation for OpenHCS.

This module provides the MemoryWrapper class for encapsulating in-memory data arrays
with explicit type declarations and conversion methods, enforcing Clause 251
(Declarative Memory Conversion Interface) and Clause 106-A (Declared Memory Types).
"""

from typing import Any, Optional

from openhcs.constants.constants import MemoryType

from .converters import (convert_memory, validate_data_compatibility,
                         validate_memory_type)
from .exceptions import MemoryConversionError
from .utils import _ensure_module, _get_device_id


class MemoryWrapper:
    """
    Immutable wrapper for in-memory data arrays with explicit type declarations.

    This class enforces Clause 251 (Declarative Memory Conversion Interface) and
    Clause 106-A (Declared Memory Types) by requiring explicit memory type declarations
    and providing declarative conversion methods.

    Attributes:
        memory_type: The declared memory type (e.g., "numpy", "cupy")
        data: The wrapped data array (read-only)
        gpu_id: The GPU device ID (for GPU memory types) or None for CPU
        input_memory_type: Alias for memory_type (for canonical access pattern)
        output_memory_type: Alias for memory_type (for canonical access pattern)
    """

    def __init__(self, data: Any, memory_type: str, gpu_id: int):
        """
        Initialize a MemoryWrapper with data and explicit memory type.

        Args:
            data: The in-memory data array (numpy, cupy, torch, tensorflow)
            memory_type: The explicit memory type declaration (e.g., "numpy", "cupy")
            gpu_id: The GPU device ID (required for GPU memory types)

        Raises:
            ValueError: If memory_type is not supported or data is incompatible
            MemoryConversionError: If gpu_id is invalid
        """
        # Validate memory type
        validate_memory_type(memory_type)

        # Validate data compatibility
        validate_data_compatibility(data, memory_type)

        # Store data and memory type
        self._data = data
        self._memory_type = memory_type

        # Store the provided gpu_id for all memory types
        # We need gpu_id even for numpy data when converting TO GPU memory types
        if gpu_id is not None and gpu_id < 0:
            raise ValueError(f"Invalid GPU ID: {gpu_id}. Must be a non-negative integer.")
        self._gpu_id = gpu_id

    @property
    def memory_type(self) -> str:
        """
        Get the declared memory type.

        Returns:
            The memory type as a string
        """
        return self._memory_type

    @property
    def data(self) -> Any:
        """
        Get the wrapped data array.

        Returns:
            The wrapped data array
        """
        return self._data

    @property
    def gpu_id(self) -> Optional[int]:
        """
        Get the GPU device ID.

        Returns:
            The GPU device ID or None for CPU memory types
        """
        return self._gpu_id

    @property
    def input_memory_type(self) -> str:
        """
        Get input memory type (same as memory_type).

        This property is provided for compatibility with the canonical memory type
        access pattern defined in Clause 106-A.2.

        Returns:
            The memory type as a string
        """
        return self._memory_type

    @property
    def output_memory_type(self) -> str:
        """
        Get output memory type (same as memory_type).

        This property is provided for compatibility with the canonical memory type
        access pattern defined in Clause 106-A.2.

        Returns:
            The memory type as a string
        """
        return self._memory_type

    def to_numpy(self) -> "MemoryWrapper":
        """
        Convert to numpy array and return a new MemoryWrapper.

        Returns:
            A new MemoryWrapper with numpy array data

        Raises:
            ValueError: If conversion to numpy is not supported for this memory type
            MemoryConversionError: If conversion fails
        """
        if self._memory_type == MemoryType.NUMPY.value:
            # Already numpy, return a copy
            # Use 0 as a placeholder for gpu_id since it's ignored for numpy
            return MemoryWrapper(self._data.copy(), MemoryType.NUMPY.value, 0)

        # Convert to numpy (always goes to CPU)
        # Always allow CPU roundtrip for to_numpy since it's explicitly going to CPU
        numpy_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.NUMPY.value,
            allow_cpu_roundtrip=True,
            gpu_id=0  # Use 0 as a placeholder since it's ignored for numpy
        )
        # Use 0 as a placeholder for gpu_id since it's ignored for numpy
        return MemoryWrapper(numpy_data, MemoryType.NUMPY.value, 0)

    def to_cupy(self, allow_cpu_roundtrip: bool = False) -> "MemoryWrapper":
        """
        Convert to cupy array and return a new MemoryWrapper.

        Preserves the GPU device ID if possible.

        Args:
            allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

        Returns:
            A new MemoryWrapper with cupy array data

        Raises:
            ValueError: If conversion to cupy is not supported for this memory type
            ImportError: If cupy is not installed
            MemoryConversionError: If conversion fails and CPU fallback is not authorized
        """
        if self._memory_type == MemoryType.CUPY.value:
            # Already cupy, return a copy
            return MemoryWrapper(self._data.copy(), MemoryType.CUPY.value, self._gpu_id)

        # Convert to cupy, preserving GPU ID if possible
        cupy_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.CUPY.value,
            gpu_id=self._gpu_id,
            allow_cpu_roundtrip=allow_cpu_roundtrip
        )

        # Get the GPU ID from the result (may have changed during conversion)
        result_gpu_id = _get_device_id(cupy_data, MemoryType.CUPY.value)

        # Ensure we have a GPU ID for GPU memory
        if result_gpu_id is None:
            raise MemoryConversionError(
                source_type=self._memory_type,
                target_type=MemoryType.CUPY.value,
                method="device_detection",
                reason="Failed to detect GPU ID for CuPy array after conversion"
            )

        return MemoryWrapper(cupy_data, MemoryType.CUPY.value, result_gpu_id)

    def to_torch(self, allow_cpu_roundtrip: bool = False) -> "MemoryWrapper":
        """
        Convert to torch tensor and return a new MemoryWrapper.

        Preserves the GPU device ID if possible.

        Args:
            allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

        Returns:
            A new MemoryWrapper with torch tensor data

        Raises:
            ValueError: If conversion to torch is not supported for this memory type
            ImportError: If torch is not installed
            MemoryConversionError: If conversion fails and CPU fallback is not authorized
        """
        if self._memory_type == MemoryType.TORCH.value:
            # Already torch, return a copy
            return MemoryWrapper(self._data.clone(), MemoryType.TORCH.value, self._gpu_id)

        # Convert to torch, preserving GPU ID if possible
        torch_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.TORCH.value,
            gpu_id=self._gpu_id,
            allow_cpu_roundtrip=allow_cpu_roundtrip
        )

        # Get the GPU ID from the result (may have changed during conversion)
        result_gpu_id = _get_device_id(torch_data, MemoryType.TORCH.value)

        # For GPU tensors, ensure we have a GPU ID
        if torch_data.is_cuda and result_gpu_id is None:
            raise MemoryConversionError(
                source_type=self._memory_type,
                target_type=MemoryType.TORCH.value,
                method="device_detection",
                reason="Failed to detect GPU ID for CUDA tensor after conversion"
            )

        return MemoryWrapper(torch_data, MemoryType.TORCH.value, result_gpu_id)

    def to_tensorflow(self, allow_cpu_roundtrip: bool = False) -> "MemoryWrapper":
        """
        Convert to tensorflow tensor and return a new MemoryWrapper.

        Preserves the GPU device ID if possible.

        Args:
            allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

        Returns:
            A new MemoryWrapper with tensorflow tensor data

        Raises:
            ValueError: If conversion to tensorflow is not supported for this memory type
            ImportError: If tensorflow is not installed
            MemoryConversionError: If conversion fails and CPU fallback is not authorized
        """
        if self._memory_type == MemoryType.TENSORFLOW.value:
            # Already tensorflow, return a copy
            tf = _ensure_module("tensorflow")
            return MemoryWrapper(tf.identity(self._data), MemoryType.TENSORFLOW.value, self._gpu_id)

        # Convert to tensorflow, preserving GPU ID if possible
        tf_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.TENSORFLOW.value,
            gpu_id=self._gpu_id,
            allow_cpu_roundtrip=allow_cpu_roundtrip
        )

        # Get the GPU ID from the result (may have changed during conversion)
        result_gpu_id = _get_device_id(tf_data, MemoryType.TENSORFLOW.value)

        # Check if this is a GPU tensor and ensure we have a GPU ID
        device_str = tf_data.device.lower()
        is_gpu_tensor = "gpu" in device_str

        if is_gpu_tensor and result_gpu_id is None:
            raise MemoryConversionError(
                source_type=self._memory_type,
                target_type=MemoryType.TENSORFLOW.value,
                method="device_detection",
                reason="Failed to detect GPU ID for TensorFlow GPU tensor after conversion"
            )

        return MemoryWrapper(tf_data, MemoryType.TENSORFLOW.value, result_gpu_id)

    def to_jax(self, allow_cpu_roundtrip: bool = False) -> "MemoryWrapper":
        """
        Convert to JAX array and return a new MemoryWrapper.

        Preserves the GPU device ID if possible.

        Args:
            allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

        Returns:
            A new MemoryWrapper with JAX array data

        Raises:
            ValueError: If conversion to JAX is not supported for this memory type
            ImportError: If JAX is not installed
            MemoryConversionError: If conversion fails and CPU fallback is not authorized
        """
        if self._memory_type == MemoryType.JAX.value:
            # Already JAX, return a copy
            jax = _ensure_module("jax")
            return MemoryWrapper(jax.numpy.array(self._data), MemoryType.JAX.value, self._gpu_id)

        # Convert to JAX, preserving GPU ID if possible
        jax_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.JAX.value,
            gpu_id=self._gpu_id,
            allow_cpu_roundtrip=allow_cpu_roundtrip
        )

        # Get GPU ID from JAX array
        result_gpu_id = _get_device_id(jax_data, MemoryType.JAX.value)

        # Check if this is a GPU array and ensure we have a GPU ID
        device_str = str(jax_data.device).lower()
        is_gpu_array = "gpu" in device_str

        if is_gpu_array and result_gpu_id is None:
            raise MemoryConversionError(
                source_type=self._memory_type,
                target_type=MemoryType.JAX.value,
                method="device_detection",
                reason="Failed to detect GPU ID for JAX GPU array after conversion"
            )

        return MemoryWrapper(jax_data, MemoryType.JAX.value, result_gpu_id)

    def to_pyclesperanto(self, allow_cpu_roundtrip: bool = False) -> "MemoryWrapper":
        """
        Convert to pyclesperanto array and return a new MemoryWrapper.

        Preserves the GPU device ID if possible.

        Args:
            allow_cpu_roundtrip: Whether to allow fallback to CPU roundtrip

        Returns:
            A new MemoryWrapper with pyclesperanto array data

        Raises:
            ValueError: If conversion to pyclesperanto is not supported for this memory type
            ImportError: If pyclesperanto is not installed
            MemoryConversionError: If conversion fails and CPU fallback is not authorized
        """
        if self._memory_type == MemoryType.PYCLESPERANTO.value:
            # Already pyclesperanto, return a copy
            cle = _ensure_module("pyclesperanto")
            result = cle.create_like(self._data)
            cle.copy(self._data, result)
            return MemoryWrapper(result, MemoryType.PYCLESPERANTO.value, self._gpu_id)

        # Convert to pyclesperanto, preserving GPU ID if possible
        pyclesperanto_data = convert_memory(
            self._data,
            self._memory_type,
            MemoryType.PYCLESPERANTO.value,
            gpu_id=self._gpu_id,
            allow_cpu_roundtrip=allow_cpu_roundtrip
        )

        # Get the GPU ID from the result (may have changed during conversion)
        result_gpu_id = _get_device_id(pyclesperanto_data, MemoryType.PYCLESPERANTO.value)

        # Ensure we have a GPU ID for GPU memory
        if result_gpu_id is None:
            raise MemoryConversionError(
                source_type=self._memory_type,
                target_type=MemoryType.PYCLESPERANTO.value,
                method="device_detection",
                reason="Failed to detect GPU ID for pyclesperanto array after conversion"
            )

        return MemoryWrapper(pyclesperanto_data, MemoryType.PYCLESPERANTO.value, result_gpu_id)

    def __repr__(self) -> str:
        """
        Get a string representation of the MemoryWrapper.

        Returns:
            A string representation
        """
        return f"MemoryWrapper(memory_type='{self._memory_type}', shape={self._get_shape()})"

    def _get_shape(self) -> tuple:
        """
        Get the shape of the wrapped data array.

        Returns:
            The shape as a tuple
        """
        if self._memory_type == MemoryType.NUMPY.value:
            return self._data.shape
        if self._memory_type == MemoryType.CUPY.value:
            return self._data.shape
        if self._memory_type == MemoryType.TORCH.value:
            return tuple(self._data.shape)
        if self._memory_type == MemoryType.TENSORFLOW.value:
            return tuple(self._data.shape)
        if self._memory_type == MemoryType.PYCLESPERANTO.value:
            return tuple(self._data.shape)

        # This should never happen if validate_memory_type is called in __init__
        return tuple()
