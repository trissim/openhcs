"""
Image Processor Factory

This module provides a factory function to create the appropriate image processor
based on the available backends and user preferences.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
"""

import logging
from typing import Dict, List, Optional, Type

from openhcs.processing.processor import ImageProcessorInterface

logger = logging.getLogger(__name__)

# Import all processor implementations
try:
    from openhcs.processing.backends.processors.numpy_processor import \
        NumPyImageProcessor
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy processor not available")

try:
    from openhcs.processing.backends.processors.cupy_processor import \
        CuPyImageProcessor
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy processor not available")

try:
    from openhcs.processing.backends.processors.torch_processor import \
        TorchImageProcessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch processor not available")

try:
    from openhcs.processing.backends.processors.tensorflow_processor import \
        TensorFlowImageProcessor
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow processor not available")

try:
    from openhcs.processing.backends.processors.jax_processor import JAXImageProcessor
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    logger.warning("JAX processor not available")


def get_available_processors() -> Dict[str, Type[ImageProcessorInterface]]:
    """
    Get a dictionary of available image processors.

    Returns:
        Dictionary mapping backend names to processor classes
    """
    processors = {}

    if HAS_NUMPY:
        processors["numpy"] = NumPyImageProcessor

    if HAS_CUPY:
        processors["cupy"] = CuPyImageProcessor

    if HAS_TORCH:
        processors["torch"] = TorchImageProcessor

    if HAS_TENSORFLOW:
        processors["tensorflow"] = TensorFlowImageProcessor

    if HAS_JAX:
        processors["jax"] = JAXImageProcessor

    return processors


def get_processor(
    backend: Optional[str] = None,
    prefer_gpu: bool = True,
    fallback_to_cpu: bool = True
) -> Type[ImageProcessorInterface]:
    """
    Get the appropriate image processor based on the available backends and user preferences.

    Args:
        backend: Specific backend to use. If None, will select based on preferences.
        prefer_gpu: Whether to prefer GPU backends over CPU backends.
        fallback_to_cpu: Whether to fall back to CPU if the requested backend is not available.
            If False and the requested backend is not available, raises ValueError.

    Returns:
        Image processor class

    Raises:
        ValueError: If no processors are available or the requested backend is not available
            and fallback_to_cpu is False.
    """
    processors = get_available_processors()

    if not processors:
        raise ValueError("No image processors are available. Please install at least NumPy.")

    # If a specific backend is requested, try to use it
    if backend is not None:
        if backend in processors:
            return processors[backend]

        if not fallback_to_cpu:
            raise ValueError(
                f"Requested backend '{backend}' is not available. "
                f"Available backends: {list(processors.keys())}"
            )

        logger.warning(
            "Requested backend '%s' is not available. Falling back to CPU backend.",
            backend
        )

    # If no specific backend is requested or the requested backend is not available,
    # select based on preferences
    gpu_backends = ["cupy", "torch", "tensorflow", "jax"]
    cpu_backends = ["numpy"]

    # Order backends based on preference
    if prefer_gpu:
        preferred_order = gpu_backends + cpu_backends
    else:
        preferred_order = cpu_backends + gpu_backends

    # Select the first available backend in the preferred order
    for backend_name in preferred_order:
        if backend_name in processors:
            return processors[backend_name]

    # If we get here, no preferred backends are available
    # Just return the first available processor
    return next(iter(processors.values()))


def list_available_backends() -> List[str]:
    """
    List the names of all available backends.

    Returns:
        List of backend names
    """
    return list(get_available_processors().keys())
