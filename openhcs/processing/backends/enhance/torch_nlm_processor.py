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
from enum import Enum

from openhcs.core.memory import torch as torch_func
from openhcs.utils.import_utils import optional_import

# Import torch modules as optional dependencies
from openhcs.core.lazy_gpu_imports import torch

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


def _denoise_2d(image: "torch.Tensor", **kwargs) -> "torch.Tensor":
    return nlm2d(image, **kwargs)


def _denoise_3d(image: "torch.Tensor", **kwargs) -> "torch.Tensor":
    return nlm3d(image, **kwargs)


class TorchNlmInputDimensionality(Enum):
    """Input dimensionalities carrying their torch-nlm execution leaves."""

    IMAGE_2D = (2, _denoise_2d)
    VOLUME_3D = (3, _denoise_3d)

    def __new__(cls, ndim, denoiser):
        member = object.__new__(cls)
        member._value_ = ndim
        member._denoiser = denoiser
        return member

    @classmethod
    def from_ndim(cls, ndim: int) -> "TorchNlmInputDimensionality":
        try:
            return cls(ndim)
        except ValueError as exc:
            raise ValueError(
                f"Input must be a 2D image or 3D volume, got {ndim}D"
            ) from exc

    def denoise(self, image: "torch.Tensor", **kwargs) -> "torch.Tensor":
        return self._denoiser(image, **kwargs)


def _validate_image(image: "torch.Tensor") -> TorchNlmInputDimensionality:
    """Validate a torch-nlm input and return its dimensionality declaration."""
    if torch is None:
        raise ImportError("PyTorch is required for torch_nlm functions")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(image)}")

    return TorchNlmInputDimensionality.from_ndim(image.ndim)


@torch_func(slice_by_slice_default=True)
def non_local_means_denoise_torch(
    image: "torch.Tensor",
    *,
    kernel_size: int = 11,
    std: float = 1.0,
    kernel_size_mean: int = 3,
    sub_filter_size: int = 32,
) -> "torch.Tensor":
    """
    Apply Non-Local Means denoising to an image or volume using torch_nlm.

    Non-Local Means is an advanced denoising algorithm that preserves fine details
    and textures by comparing patches across the entire image rather than just
    local neighborhoods. This implementation uses torch_nlm for GPU acceleration.

    Args:
        image: 2D PyTorch image or 3D PyTorch volume
        kernel_size: Size of the neighborhood for patch comparison (default: 11)
        std: Standard deviation for weight calculation (default: 1.0)
        kernel_size_mean: Kernel size for initial mean filtering (default: 3)
        sub_filter_size: Number of neighborhoods computed per iteration for memory efficiency (default: 32)

    Returns:
        Denoised PyTorch tensor with the same shape as the input

    Raises:
        ImportError: If torch_nlm is not available
        TypeError: If input is not a torch.Tensor
        ValueError: If input is not 2D or 3D
        RuntimeError: If tensor is not on CUDA device
    """
    dimensionality = _validate_image(image)

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

    # Convert to float32 for processing if needed
    if image.dtype != torch.float32:
        image_float = image.float()
    else:
        image_float = image

    return dimensionality.denoise(
        image_float,
        kernel_size=kernel_size,
        std=std,
        kernel_size_mean=kernel_size_mean,
        sub_filter_size=sub_filter_size,
    )


# Alias for convenience
torch_nlm_denoise = non_local_means_denoise_torch
