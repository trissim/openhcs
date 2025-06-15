
from __future__ import annotations 
from typing import Optional

from openhcs.core.utils import optional_import
from openhcs.core.memory.decorators import torch as torch_decorator

# Import torch modules as optional dependencies
torch = optional_import("torch")
F = optional_import("torch.nn.functional") if torch is not None else None


def laplacian(image: "torch.Tensor") -> "torch.Tensor":
    """Applies a 2D Laplacian filter."""
    # Input image is expected to be [N, C, H, W] or [C, H, W] or [H, W]
    # Kernel is [out_channels, in_channels/groups, kH, kW]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=image.dtype, device=image.device)
    kernel = kernel.reshape(1, 1, 3, 3) # For a single channel input/output

    # Handle different input dimensions by adding/removing batch/channel dims
    original_ndim = image.ndim
    if original_ndim == 2: # [H, W]
        image = image.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    elif original_ndim == 3: # [C, H, W] or [Z, H, W] - assuming [C, H, W] for conv2d
         # If it's [Z, H, W] as in focus_stack_max_sharpness, need to process each slice
         # This laplacian is for a single 2D image or batch of 2D images.
         # The calling function focus_stack_max_sharpness passes image_stack.unsqueeze(1) -> [Z, 1, H, W]
         # So input to this laplacian function will be [Z, 1, H, W].
         pass # Already in [N, C, H, W] format where N=Z, C=1
    elif original_ndim == 4: # [N, C, H, W]
        pass
    else:
        raise ValueError(f"Unsupported image dimension for laplacian: {original_ndim}")

    # Apply convolution. Assuming input channel is 1.
    # If input has multiple channels, need to apply laplacian to each or convert to grayscale.
    # The calling context passes [Z, 1, H, W], so in_channels is 1.
    laplacian_img = F.conv2d(image, kernel, padding=1)

    # Restore original dimensions
    if original_ndim == 2:
        laplacian_img = laplacian_img.squeeze(0).squeeze(0)
    # If original_ndim was 3 ([Z, H, W]), the input was [Z, 1, H, W], output is [Z, 1, H, W]. Squeeze channel.
    elif original_ndim == 3 and image.shape[1] == 1:
         laplacian_img = laplacian_img.squeeze(1) # [Z, H, W]

    return laplacian_img

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
    if not (str(image_stack.ndim) == '3' and str(image_stack.device.type) == 'cuda'):
        raise ValueError(f"Input must be 3D tensor [Z,H,W]. Got {image_stack.ndim}D")

    Z, H, W = image_stack.shape
    device = image_stack.device
    dtype = image_stack.dtype

    # Set adaptive defaults based on image dimensions
    patch_size = patch_size or max(H, W) // 8
    stride = stride or patch_size // 2

    # Calculate sharpness maps
    if method == "laplacian":
        sharpness = torch.abs(laplacian(image_stack.unsqueeze(1))).squeeze(1)
    elif method == "gradient":
        gx, gy = torch.gradient(image_stack, dim=(1,2))
        sharpness = torch.sqrt(gx**2 + gy**2)
    else:
        raise ValueError(f"Invalid method: {method}. Use 'laplacian' or 'gradient'")

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
