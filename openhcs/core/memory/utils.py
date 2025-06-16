"""
Memory conversion utility functions for OpenHCS.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
"""

import importlib
import logging
from typing import Any, Optional

from openhcs.constants.constants import MemoryType

from .exceptions import MemoryConversionError

logger = logging.getLogger(__name__)


def _ensure_module(module_name: str) -> Any:
    """
    Ensure a module is imported and meets version requirements.

    Args:
        module_name: The name of the module to import

    Returns:
        The imported module

    Raises:
        ImportError: If the module cannot be imported or does not meet version requirements
        RuntimeError: If the module has known issues with specific versions
    """
    try:
        module = importlib.import_module(module_name)

        # Check TensorFlow version for DLPack compatibility
        if module_name == "tensorflow":
            import pkg_resources
            tf_version = pkg_resources.parse_version(module.__version__)
            min_version = pkg_resources.parse_version("2.12.0")

            if tf_version < min_version:
                raise RuntimeError(
                    f"TensorFlow version {module.__version__} is not supported for DLPack operations. "
                    f"Version 2.12.0 or higher is required for stable DLPack support. "
                    f"Clause 88 (No Inferred Capabilities) violation: Cannot infer DLPack capability."
                )

        return module
    except ImportError:
        raise ImportError(f"Module {module_name} is required for this operation but is not installed")


def _supports_cuda_array_interface(obj: Any) -> bool:
    """
    Check if an object supports the CUDA Array Interface.

    Args:
        obj: The object to check

    Returns:
        True if the object supports the CUDA Array Interface, False otherwise
    """
    return hasattr(obj, "__cuda_array_interface__")


def _supports_dlpack(obj: Any) -> bool:
    """
    Check if an object supports DLPack.

    Args:
        obj: The object to check

    Returns:
        True if the object supports DLPack, False otherwise

    Note:
        For TensorFlow tensors, this function enforces Clause 88 (No Inferred Capabilities)
        by explicitly checking:
        1. TensorFlow version must be 2.12+ for stable DLPack support
        2. Tensor must be on GPU (CPU tensors might succeed even without proper DLPack support)
        3. tf.experimental.dlpack module must exist
    """
    # Check for PyTorch, CuPy, or JAX DLPack support
    # PyTorch: __dlpack__ method, CuPy: toDlpack method, JAX: __dlpack__ method
    if hasattr(obj, "toDlpack") or hasattr(obj, "to_dlpack") or hasattr(obj, "__dlpack__"):
        # Special handling for TensorFlow to enforce Clause 88
        if 'tensorflow' in str(type(obj)):
            try:
                import tensorflow as tf

                # Check TensorFlow version - DLPack is only stable in TF 2.12+
                tf_version = tf.__version__
                major, minor = map(int, tf_version.split('.')[:2])

                if major < 2 or (major == 2 and minor < 12):
                    # Explicitly fail for TF < 2.12 to prevent silent fallbacks
                    raise RuntimeError(
                        f"TensorFlow version {tf_version} does not support stable DLPack operations. "
                        f"Version 2.12.0 or higher is required. "
                        f"Clause 88 violation: Cannot infer DLPack capability."
                    )

                # Check if tensor is on GPU - CPU tensors might succeed even without proper DLPack support
                device_str = obj.device.lower()
                if "gpu" not in device_str:
                    # Explicitly fail for CPU tensors to prevent deceptive behavior
                    raise RuntimeError(
                        "TensorFlow tensor on CPU cannot use DLPack operations reliably. "
                        "Only GPU tensors are supported for DLPack operations. "
                        "Clause 88 violation: Cannot infer GPU capability."
                    )

                # Check if experimental.dlpack module exists
                if not hasattr(tf.experimental, "dlpack"):
                    raise RuntimeError(
                        "TensorFlow installation missing experimental.dlpack module. "
                        "Clause 88 violation: Cannot infer DLPack capability."
                    )

                return True
            except (ImportError, AttributeError) as e:
                # Re-raise with more specific error message
                raise RuntimeError(
                    f"TensorFlow DLPack support check failed: {str(e)}. "
                    f"Clause 88 violation: Cannot infer DLPack capability."
                ) from e

        # For non-TensorFlow types, return True if they have DLPack methods
        return True

    return False


def _get_device_id(data: Any, memory_type: str) -> Optional[int]:
    """
    Get the GPU device ID from a data object.

    Args:
        data: The data object
        memory_type: The memory type

    Returns:
        The GPU device ID or None if not applicable

    Raises:
        MemoryConversionError: If the device ID cannot be determined for a GPU memory type
    """
    if memory_type == MemoryType.NUMPY.value:
        return None

    if memory_type == MemoryType.CUPY.value:
        try:
            return data.device.id
        except AttributeError:
            # Default to device 0 if not available
            # This is a special case because CuPy arrays are always on a GPU
            return 0
        except Exception as e:
            logger.warning(f"Failed to get device ID for CuPy array: {str(e)}")
            return 0

    if memory_type == MemoryType.TORCH.value:
        try:
            if data.is_cuda:
                return data.device.index
            # CPU tensor, no device ID
            return None
        except Exception as e:
            logger.warning(f"Failed to get device ID for PyTorch tensor: {str(e)}")
            return None

    if memory_type == MemoryType.TENSORFLOW.value:
        try:
            device_str = data.device.lower()
            if "gpu" in device_str:
                # Extract device ID from string like "/device:gpu:0"
                return int(device_str.split(":")[-1])
            # CPU tensor, no device ID
            return None
        except Exception as e:
            logger.warning(f"Failed to get device ID for TensorFlow tensor: {str(e)}")
            return None

    if memory_type == MemoryType.JAX.value:
        try:
            device_str = str(data.device).lower()
            if "gpu" in device_str:
                # Extract device ID from string like "gpu:0"
                return int(device_str.split(":")[-1])
            # CPU array, no device ID
            return None
        except Exception as e:
            logger.warning(f"Failed to get device ID for JAX array: {str(e)}")
            return None

    if memory_type == MemoryType.PYCLESPERANTO.value:
        try:
            cle = _ensure_module("pyclesperanto")
            current_device = cle.get_device()
            # pyclesperanto device is an object, try to extract ID
            if hasattr(current_device, 'id'):
                return current_device.id
            # Fallback: try to get device index from device list
            devices = cle.list_available_devices()
            for i, device in enumerate(devices):
                if str(device) == str(current_device):
                    return i
            # Default to 0 if we can't determine
            return 0
        except Exception as e:
            logger.warning(f"Failed to get device ID for pyclesperanto array: {str(e)}")
            return 0

    return None


def _set_device(memory_type: str, device_id: int) -> None:
    """
    Set the current device for a specific memory type.

    Args:
        memory_type: The memory type
        device_id: The GPU device ID

    Raises:
        MemoryConversionError: If the device cannot be set
    """
    if memory_type == MemoryType.CUPY.value:
        try:
            cupy = _ensure_module("cupy")
            cupy.cuda.Device(device_id).use()
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_selection",
                reason=f"Failed to set CuPy device to {device_id}: {str(e)}"
            ) from e

    if memory_type == MemoryType.PYCLESPERANTO.value:
        try:
            cle = _ensure_module("pyclesperanto")
            devices = cle.list_available_devices()
            if device_id >= len(devices):
                raise ValueError(f"Device ID {device_id} not available. Available devices: {len(devices)}")
            cle.select_device(device_id)
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_selection",
                reason=f"Failed to set pyclesperanto device to {device_id}: {str(e)}"
            ) from e

    # JAX doesn't have a global device setting mechanism
    # Device selection happens at array creation or device_put time

    # PyTorch and TensorFlow handle device placement at tensor creation time
    # No need to set a global device


def _move_to_device(data: Any, memory_type: str, device_id: int) -> Any:
    """
    Move data to a specific GPU device.

    Args:
        data: The data to move
        memory_type: The memory type
        device_id: The target GPU device ID

    Returns:
        The data on the target device

    Raises:
        MemoryConversionError: If the data cannot be moved to the specified device
    """
    if memory_type == MemoryType.CUPY.value:
        cupy = _ensure_module("cupy")
        try:
            if data.device.id != device_id:
                with cupy.cuda.Device(device_id):
                    return data.copy()
            return data
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_movement",
                reason=f"Failed to move CuPy array to device {device_id}: {str(e)}"
            ) from e

    if memory_type == MemoryType.TORCH.value:
        try:
            if data.is_cuda and data.device.index != device_id:
                return data.to(f"cuda:{device_id}")
            if not data.is_cuda:
                return data.to(f"cuda:{device_id}")
            return data
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_movement",
                reason=f"Failed to move PyTorch tensor to device {device_id}: {str(e)}"
            ) from e

    if memory_type == MemoryType.TENSORFLOW.value:
        try:
            tf = _ensure_module("tensorflow")
            with tf.device(f"/device:GPU:{device_id}"):
                return tf.identity(data)
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_movement",
                reason=f"Failed to move TensorFlow tensor to device {device_id}: {str(e)}"
            ) from e

    if memory_type == MemoryType.JAX.value:
        try:
            jax = _ensure_module("jax")
            # JAX uses different device notation
            return jax.device_put(data, jax.devices("gpu")[device_id])
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_movement",
                reason=f"Failed to move JAX array to device {device_id}: {str(e)}"
            ) from e

    if memory_type == MemoryType.PYCLESPERANTO.value:
        try:
            cle = _ensure_module("pyclesperanto")
            # Get current device of the array
            current_device_id = _get_device_id(data, memory_type)

            if current_device_id != device_id:
                # Select target device and copy data
                cle.select_device(device_id)
                result = cle.create_like(data)
                cle.copy(data, result)
                return result
            return data
        except Exception as e:
            raise MemoryConversionError(
                source_type=memory_type,
                target_type=memory_type,
                method="device_movement",
                reason=f"Failed to move pyclesperanto array to device {device_id}: {str(e)}"
            ) from e

    return data
