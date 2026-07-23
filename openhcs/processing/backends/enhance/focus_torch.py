
from __future__ import annotations 
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openhcs.core.utils import optional_import
from openhcs.core.memory import torch as torch_decorator

# Import torch modules as optional dependencies
from openhcs.core.lazy_gpu_imports import torch
F = optional_import("torch.nn.functional") if torch else None


class FocusSharpnessMethod(Enum):
    """Sharpness metrics supported by focus stacking."""

    LAPLACIAN = "laplacian"
    GRADIENT = "gradient"

    @classmethod
    def from_value(cls, value: str) -> "FocusSharpnessMethod":
        try:
            return cls(value)
        except ValueError as error:
            raise ValueError(
                f"Invalid method: {value}. Use 'laplacian' or 'gradient'"
            ) from error


def _laplacian_sharpness(image_stack: "torch.Tensor") -> "torch.Tensor":
    return torch.abs(laplacian(image_stack.unsqueeze(1))).squeeze(1)


def _gradient_sharpness(image_stack: "torch.Tensor") -> "torch.Tensor":
    gx, gy = torch.gradient(image_stack, dim=(1, 2))
    return torch.sqrt(gx**2 + gy**2)


FOCUS_SHARPNESS_METHODS: dict[
    FocusSharpnessMethod, Callable[["torch.Tensor"], "torch.Tensor"]
] = {
    FocusSharpnessMethod.LAPLACIAN: _laplacian_sharpness,
    FocusSharpnessMethod.GRADIENT: _gradient_sharpness,
}


@dataclass(frozen=True, slots=True)
class LaplacianImageProjection:
    """Project supported 2D/3D/4D inputs to conv2d image layout."""

    original_ndim: int
    image: "torch.Tensor"

    @classmethod
    def from_image(cls, image: "torch.Tensor") -> "LaplacianImageProjection":
        if image.ndim == 2:
            return cls(original_ndim=2, image=image.unsqueeze(0).unsqueeze(0))
        if image.ndim in (3, 4):
            return cls(original_ndim=image.ndim, image=image)
        raise ValueError(f"Unsupported image dimension for laplacian: {image.ndim}")

    def restore(self, laplacian_img: "torch.Tensor") -> "torch.Tensor":
        if self.original_ndim == 2:
            return laplacian_img.squeeze(0).squeeze(0)
        if self.original_ndim == 3 and self.image.shape[1] == 1:
            return laplacian_img.squeeze(1)
        return laplacian_img


@dataclass(frozen=True, slots=True)
class FocusStackProjection:
    """Validate and expose the focus-stack dimensional contract."""

    z: int
    height: int
    width: int
    image_stack: "torch.Tensor"

    @classmethod
    def from_stack(cls, image_stack: "torch.Tensor") -> "FocusStackProjection":
        if image_stack.ndim != 3 or image_stack.device.type != "cuda":
            raise ValueError(
                f"Input must be 3D tensor [Z,H,W]. Got {image_stack.ndim}D"
            )
        z, height, width = image_stack.shape
        return cls(z=z, height=height, width=width, image_stack=image_stack)


def laplacian(image: "torch.Tensor") -> "torch.Tensor":
    """Applies a 2D Laplacian filter."""
    # Input image is expected to be [N, C, H, W] or [C, H, W] or [H, W]
    # Kernel is [out_channels, in_channels/groups, kH, kW]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=image.dtype, device=image.device)
    kernel = kernel.reshape(1, 1, 3, 3) # For a single channel input/output

    projection = LaplacianImageProjection.from_image(image)

    # Apply convolution. Assuming input channel is 1.
    # If input has multiple channels, need to apply laplacian to each or convert to grayscale.
    # The calling context passes [Z, 1, H, W], so in_channels is 1.
    laplacian_img = F.conv2d(projection.image, kernel, padding=1)
    return projection.restore(laplacian_img)

@torch_decorator
def focus_stack_max_sharpness(
    image_stack: "torch.Tensor",
    method: str = "laplacian",
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    normalize_sharpness: bool = False
) -> "torch.Tensor":
    """
    GPU-accelerated focus stacking using PyTorch. Selects sharpest regions from a Z-stack.

    Args:
        image_stack: Input tensor of shape [Z, H, W]
        method: Sharpness metric ('laplacian' or 'gradient')
        patch_size: Size of analysis patches. Default: max(H,W)//8
        stride: Stride between patches. Default: patch_size//2
        normalize_sharpness: Normalize sharpness scores per patch

    Returns:
        Composite image of shape [1, H, W] with maximal sharpness regions
    """
    stack_projection = FocusStackProjection.from_stack(image_stack)
    Z = stack_projection.z
    H = stack_projection.height
    W = stack_projection.width
    image_stack = stack_projection.image_stack
    device = image_stack.device
    dtype = image_stack.dtype

    # Set adaptive defaults based on image dimensions
    patch_size = patch_size or max(H, W) // 8
    stride = stride or patch_size // 2

    sharpness_method = FocusSharpnessMethod.from_value(method)
    sharpness = FOCUS_SHARPNESS_METHODS[sharpness_method](image_stack)

    if normalize_sharpness:
        sharpness = (sharpness - sharpness.mean(dim=0)) / (sharpness.std(dim=0) + 1e-6)

    # Generate sliding window patches
    patches = F.unfold(
        sharpness.unsqueeze(1),
        kernel_size=patch_size,
        stride=stride
    ).view(Z, -1, H//stride, W//stride)

    # Find sharpest z-index per patch
    _, max_indices = torch.max(patches, dim=0)

    # Create composite image using max sharpness indices
    composite = torch.zeros_like(image_stack[0])
    weights = torch.zeros_like(composite)

    for i in range(max_indices.shape[1]):
        for j in range(max_indices.shape[2]):
            z_idx = max_indices[0,i,j]
            h_start = i * stride
            w_start = j * stride

            composite_slice = composite[h_start:h_start+patch_size, w_start:w_start+patch_size]
            weight_slice = weights[h_start:h_start+patch_size, w_start:w_start+patch_size]

            composite_slice += image_stack[z_idx, h_start:h_start+patch_size, w_start:w_start+patch_size]
            weight_slice += torch.ones_like(weight_slice)

    # Avoid division by zero in overlapping regions
    return (composite / torch.clamp_min(weights, 1)).unsqueeze(0)
