"""
GPU Utility Functions

This module provides utility functions for GPU-accelerated code analysis.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from semantic_matrix_analyzer.utils import optional_import

# Import PyTorch as an optional dependency
torch = optional_import("torch")

# Set up logging
logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Get the default device for GPU-accelerated analysis.

    Returns:
        Device string ("cuda" or "cpu")
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def set_device(device: str) -> str:
    """
    Set the default device for GPU-accelerated analysis.

    Args:
        device: Device to use ("cuda" or "cpu")

    Returns:
        The actual device that will be used ("cuda" or "cpu")
    """
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'.")

    if device == "cuda":
        if torch is None:
            logger.warning("PyTorch not available. Using CPU instead.")
            return "cpu"
        elif not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU instead.")
            return "cpu"
        else:
            return "cuda"
    else:
        return "cpu"


def is_gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise
    """
    return torch is not None and torch.cuda.is_available()


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        NumPy array

    Raises:
        ImportError: If PyTorch is not available
        TypeError: If the input is not a PyTorch tensor
    """
    if torch is None:
        raise ImportError("PyTorch is required for tensor_to_numpy")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected PyTorch tensor, got {type(tensor)}")

    return tensor.cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: Optional[str] = None) -> Any:
    """
    Convert a NumPy array to a PyTorch tensor.

    Args:
        array: NumPy array
        device: Device to place tensor on

    Returns:
        PyTorch tensor

    Raises:
        ImportError: If PyTorch is not available
    """
    if torch is None:
        raise ImportError("PyTorch is required for numpy_to_tensor")

    if device is None:
        device = get_device()

    return torch.tensor(array, device=device)


def batch_process(
    items: List[Any],
    process_fn: Callable[[List[Any]], List[Any]],
    batch_size: int = 16
) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: Items to process
        process_fn: Function to process a batch of items
        batch_size: Size of each batch

    Returns:
        List of processed items
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)

    return results
