"""
BaSiC (Background and Shading Correction) Processor Factory

This module provides a factory function to select the appropriate BaSiC
processor implementation based on the available backends and user preferences.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
"""

import logging
from typing import Any, Callable, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import processor implementations
try:
    import cupy as cp

    from openhcs.processing.backends.enhance.basic_processor_cupy import (
        basic_flatfield_correction_batch_cupy, basic_flatfield_correction_cupy)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy processor not available for BaSiC. Using NumPy fallback.")

from openhcs.processing.backends.enhance.basic_processor_numpy import (
    basic_flatfield_correction_batch_numpy, basic_flatfield_correction_numpy)


def _supports_dlpack(array: Any) -> bool:
    """
    Check if an array supports DLPack protocol.

    Args:
        array: Array to check

    Returns:
        True if the array supports DLPack, False otherwise
    """
    return hasattr(array, "__dlpack__") or hasattr(array, "toDlpack")


def _ensure_same_memory_type(array: Any, expected_type: type) -> None:
    """
    Ensure that an array is of the expected memory type.

    Args:
        array: Array to check
        expected_type: Expected memory type

    Raises:
        TypeError: If the array is not of the expected memory type
        ValueError: If the array is of the expected type but doesn't support DLPack
    """
    if not isinstance(array, expected_type):
        raise TypeError(
            f"Array must be of type {expected_type.__name__}, got {type(array).__name__}. "
            f"No automatic conversion is performed to maintain explicit contracts. "
            f"Use DLPack for zero-copy GPU-to-GPU transfers."
        )

    # For GPU types, ensure DLPack support
    if expected_type.__name__ != "ndarray" and not _supports_dlpack(array):
        raise ValueError(
            f"Array of type {type(array).__name__} does not support DLPack. "
            f"DLPack is required for GPU memory conversions."
        )


def get_basic_processor(
    prefer_gpu: bool = True,
    fallback_to_cpu: bool = True
) -> Callable:
    """
    Get the appropriate BaSiC processor function based on availability and preferences.

    Args:
        prefer_gpu: Whether to prefer GPU backends over CPU backends
        fallback_to_cpu: Whether to fall back to CPU if GPU is not available
            If False and GPU is not available, raises ValueError

    Returns:
        BaSiC processor function

    Raises:
        ValueError: If GPU is requested but not available and fallback_to_cpu is False
    """
    if prefer_gpu and HAS_CUPY:
        return basic_flatfield_correction_cupy

    if prefer_gpu and not HAS_CUPY and not fallback_to_cpu:
        raise ValueError(
            "GPU-accelerated BaSiC processor was requested, but CuPy is not available "
            "and fallback_to_cpu is False."
        )

    # Fall back to NumPy
    return basic_flatfield_correction_numpy


def get_basic_batch_processor(
    prefer_gpu: bool = True,
    fallback_to_cpu: bool = True
) -> Callable:
    """
    Get the appropriate batch BaSiC processor function based on availability and preferences.

    Args:
        prefer_gpu: Whether to prefer GPU backends over CPU backends
        fallback_to_cpu: Whether to fall back to CPU if GPU is not available
            If False and GPU is not available, raises ValueError

    Returns:
        Batch BaSiC processor function

    Raises:
        ValueError: If GPU is requested but not available and fallback_to_cpu is False
    """
    if prefer_gpu and HAS_CUPY:
        return basic_flatfield_correction_batch_cupy

    if prefer_gpu and not HAS_CUPY and not fallback_to_cpu:
        raise ValueError(
            "GPU-accelerated batch BaSiC processor was requested, but CuPy is not available "
            "and fallback_to_cpu is False."
        )

    # Fall back to NumPy
    return basic_flatfield_correction_batch_numpy


def basic_flatfield_correction(
    image: Union[np.ndarray, "cp.ndarray"],
    *,
    allow_cpu_roundtrip: bool = False,
    **kwargs
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Perform BaSiC-style illumination correction on a 3D image stack.

    This function selects the appropriate backend based on the input type
    and availability. For GPU memory types, it enforces DLPack protocol
    for zero-copy transfers.

    Args:
        image: 3D array of shape (Z, Y, X)
        allow_cpu_roundtrip: Whether to allow CPU roundtrip for GPU-to-GPU transfers
            (only used when converting between different GPU memory types)
        **kwargs: Additional parameters passed to the processor function

    Returns:
        Corrected 3D array of shape (Z, Y, X)

    Raises:
        TypeError: If the input type is not supported
        ValueError: If a GPU array doesn't support DLPack
    """
    # Select the appropriate processor based on input type
    if HAS_CUPY and isinstance(image, cp.ndarray):
        # Ensure the CuPy array supports DLPack
        _ensure_same_memory_type(image, cp.ndarray)
        return basic_flatfield_correction_cupy(image, **kwargs)

    if isinstance(image, np.ndarray):
        return basic_flatfield_correction_numpy(image, **kwargs)

    # If we get here, the input type is not supported
    raise TypeError(
        f"Unsupported input type: {type(image)}. "
        f"Must be NumPy array or CuPy array. "
        f"No automatic conversion is performed to maintain explicit contracts."
    )


def basic_flatfield_correction_batch(
    image_batch: Union[np.ndarray, "cp.ndarray"],
    *,
    allow_cpu_roundtrip: bool = False,
    **kwargs
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Apply BaSiC flatfield correction to a batch of 3D image stacks.

    This function selects the appropriate backend based on the input type
    and availability. For GPU memory types, it enforces DLPack protocol
    for zero-copy transfers.

    Args:
        image_batch: 4D array of shape (B, Z, Y, X) or (Z, B, Y, X)
        allow_cpu_roundtrip: Whether to allow CPU roundtrip for GPU-to-GPU transfers
            (only used when converting between different GPU memory types)
        **kwargs: Additional parameters passed to the processor function

    Returns:
        Corrected 4D array of the same shape as input

    Raises:
        TypeError: If the input type is not supported
        ValueError: If a GPU array doesn't support DLPack
    """
    # Select the appropriate processor based on input type
    if HAS_CUPY and isinstance(image_batch, cp.ndarray):
        # Ensure the CuPy array supports DLPack
        _ensure_same_memory_type(image_batch, cp.ndarray)
        return basic_flatfield_correction_batch_cupy(image_batch, **kwargs)

    if isinstance(image_batch, np.ndarray):
        return basic_flatfield_correction_batch_numpy(image_batch, **kwargs)

    # If we get here, the input type is not supported
    raise TypeError(
        f"Unsupported input type: {type(image_batch)}. "
        f"Must be NumPy array or CuPy array. "
        f"No automatic conversion is performed to maintain explicit contracts."
    )
