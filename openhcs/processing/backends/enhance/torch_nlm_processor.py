"""
Non-Local Means Denoising Implementation using torch_nlm

This module provides OpenHCS-decorated wrapper functions for the torch_nlm library,
which implements memory-efficient non-local means denoising with GPU acceleration.

Non-local means is an advanced denoising algorithm that preserves fine details
and textures by comparing patches across the entire image rather than just
local neighborhoods. The torch_nlm implementation provides significant speedup
over traditional CPU implementations, especially for large 3D volumes.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit PyTorch and torch_nlm dependency
- Clause 273 — Memory Backend Restrictions: GPU-only implementation
"""
from __future__ import annotations

import logging
from typing import Optional

from openhcs.utils.import_utils import optional_import
from openhcs.core.memory.decorators import torch as torch_func

# Import torch modules as optional dependencies
torch = optional_import("torch")

# Import torch_nlm as optional dependency
# Note: The PyPI package is named 'nlm-torch' but imports as 'torch_nlm'
torch_nlm = optional_import("torch_nlm")
if torch_nlm is not None:
    nlm2d = torch_nlm.nlm2d
    nlm3d = torch_nlm.nlm3d
else:
    nlm2d = None
    nlm3d = None

logger = logging.getLogger(__name__)


def _validate_3d_array(image: "torch.Tensor") -> None:
    """Validate that input is a 3D torch tensor."""
    if torch is None:
        raise ImportError("PyTorch is required for torch_nlm functions")
    
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(image)}")
    
    if image.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor (Z, Y, X), got {image.ndim}D tensor")


@torch_func
def non_local_means_denoise_torch(
    image: "torch.Tensor",
    *,
    kernel_size: int = 11,
    std: float = 1.0,
    kernel_size_mean: int = 3,
    sub_filter_size: int = 32,
    slice_by_slice: bool = True,
    **kwargs
) -> "torch.Tensor":
    """
    Apply Non-Local Means denoising to a 3D image stack using torch_nlm.

    Non-Local Means is an advanced denoising algorithm that preserves fine details
    and textures by comparing patches across the entire image rather than just
    local neighborhoods. This implementation uses torch_nlm for GPU acceleration.

    Args:
        image: 3D PyTorch tensor of shape (Z, Y, X)
        kernel_size: Size of the neighborhood for patch comparison (default: 11)
        std: Standard deviation for weight calculation (default: 1.0)
        kernel_size_mean: Kernel size for initial mean filtering (default: 3)
        sub_filter_size: Number of neighborhoods computed per iteration for memory efficiency (default: 32)
        slice_by_slice: Process each Z-slice independently using 2D NLM (default: True).
                       If False, uses 3D NLM processing across all dimensions.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Denoised 3D PyTorch tensor of shape (Z, Y, X)

    Raises:
        ImportError: If torch_nlm is not available
        TypeError: If input is not a torch.Tensor
        ValueError: If input is not 3D
        RuntimeError: If tensor is not on CUDA device

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: True)
        If True, process 3D arrays slice-by-slice using 2D non-local means to avoid
        cross-slice contamination. If False, use 3D non-local means processing.
        Recommended for stitched microscopy data to prevent artifacts at field boundaries.
    """
    _validate_3d_array(image)
    
    if torch_nlm is None:
        raise ImportError(
            "torch_nlm is required for this function. "
            "Install with: pip install nlm-torch"
        )
    
    # FAIL LOUDLY if not on CUDA - no CPU fallback allowed
    if image.device.type != "cuda":
        raise RuntimeError(
            f"torch_nlm requires CUDA tensor, got device: {image.device}. "
            "Move tensor to CUDA with: tensor.cuda()"
        )
    
    # Store original dtype for conversion back
    original_dtype = image.dtype
    device = image.device

    # Convert to float32 for processing if needed
    if image.dtype != torch.float32:
        image_float = image.float()
    else:
        image_float = image

    # Handle slice_by_slice processing using OpenHCS pattern
    if slice_by_slice:
        # Process each Z-slice independently using 2D non-local means
        from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

        # Detect memory type and use proper OpenHCS utilities
        memory_type = _detect_memory_type(image_float)
        gpu_id = 0  # Default GPU ID for slice processing

        # Unstack 3D array into 2D slices
        slices_2d = unstack_slices(image_float, memory_type, gpu_id)

        # Process each slice
        processed_slices = []
        for slice_2d in slices_2d:
            # Apply 2D non-local means to this slice
            denoised_slice = nlm2d(
                slice_2d,
                kernel_size=kernel_size,
                std=std,
                kernel_size_mean=kernel_size_mean,
                sub_filter_size=sub_filter_size
            )
            processed_slices.append(denoised_slice)

        # Stack results back to 3D
        result = stack_slices(processed_slices, memory_type, gpu_id)
    else:
        # Use 3D processing directly (fallback to nlm3d)
        result = nlm3d(
            image_float,
            kernel_size=kernel_size,
            std=std,
            kernel_size_mean=kernel_size_mean,
            sub_filter_size=sub_filter_size
        )

    # Convert back to original dtype
    result = result.to(original_dtype)

    return result


# Alias for convenience
torch_nlm_denoise = non_local_means_denoise_torch
