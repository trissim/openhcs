"""
N2V2-inspired denoising implementation using PyTorch.

This module implements a 3D GPU-native, planner-compatible, self-supervised denoising
step based on the N2V2 paper (Höck et al., 2022) using PyTorch.

The implementation follows the OpenHCS doctrinal principles:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit PyTorch dependency
- Clause 273 — Memory Backend Restrictions: GPU-only implementation
"""
from __future__ import annotations 

import logging
import math
from typing import List, Optional, Tuple

from openhcs.utils.import_utils import optional_import, create_placeholder_class # Updated import path
from openhcs.core.memory.decorators import torch as torch_func

# Import torch modules as optional dependencies
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None

# Create placeholder for nn.Module
# If nn (and thus nn.Module) is available, Module will be nn.Module.
# Otherwise, Module will be a placeholder class.
Module = create_placeholder_class(
    "Module",
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)

logger = logging.getLogger(__name__)


class BlurPool3d(Module): # Inherit from Module (placeholder or actual nn.Module)
    """
    3D BlurPool layer for anti-aliasing as described in N2V2 paper.

    This layer applies a Gaussian blur before downsampling to prevent
    checkerboard artifacts in the U-Net architecture.
    """

    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3):
        """
        Initialize the BlurPool3d layer.

        Args:
            channels: Number of input channels
            stride: Stride for downsampling
            kernel_size: Size of the blur kernel
        """
        super().__init__()

        # Create a 3D Gaussian kernel
        if kernel_size == 3:
            # Simple 3x3x3 blur kernel
            kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        else:
            # Larger kernel with Gaussian weights
            sigma = 0.8 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            kernel_range = torch.arange(kernel_size, dtype=torch.float32)
            kernel = torch.exp(-(kernel_range - (kernel_size - 1) / 2)**2 / (2 * sigma**2))

        # Normalize the kernel
        kernel = kernel / kernel.sum()

        # Create 3D kernel by outer product
        kernel_x = kernel.view(1, 1, -1, 1, 1)
        kernel_y = kernel.view(1, 1, 1, -1, 1)
        kernel_z = kernel.view(1, 1, 1, 1, -1)

        # Register the kernel as a buffer (not a parameter)
        self.register_buffer('kernel', kernel)
        self.register_buffer('kernel_3d', kernel_x * kernel_y * kernel_z)

        # Create a depthwise convolution for each channel
        self.conv = nn.Conv3d(
            channels, channels, kernel_size,
            stride=stride, padding=kernel_size // 2,
            groups=channels, bias=False
        )

        # Initialize the convolution weights with the blur kernel
        with torch.no_grad():
            for i in range(channels):
                self.conv.weight[i, 0] = self.kernel_3d[0, 0]

        # Make weights non-trainable
        self.conv.weight.requires_grad = False

        self.stride = stride

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply blur and downsample."""
        return self.conv(x)


class DoubleConv3d(Module): # Inherit from Module
    """
    Double 3D convolution block with ReLU activation.

    This block consists of two Conv3d layers with batch normalization and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the DoubleConv3d block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply double convolution."""
        return self.double_conv(x)


class Down3d(Module): # Inherit from Module
    """
    Downsampling block with BlurPool for N2V2.

    This block applies a double convolution followed by BlurPool downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the Down3d block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        self.conv = DoubleConv3d(in_channels, out_channels)
        self.pool = BlurPool3d(out_channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply convolution and downsampling."""
        x = self.conv(x)
        return self.pool(x)


class Up3d(Module): # Inherit from Module
    """
    Upsampling block for N2V2.

    This block applies transposed convolution for upsampling, followed by a double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, skip_connection: bool = True):
        """
        Initialize the Up3d block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            skip_connection: Whether to use skip connections
        """
        super().__init__()

        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Double convolution
        if skip_connection:
            self.conv = DoubleConv3d(in_channels, out_channels)
        else:
            self.conv = DoubleConv3d(in_channels // 2, out_channels)

        self.skip_connection = skip_connection

    def forward(self, x: "torch.Tensor", skip: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Apply upsampling and convolution with optional skip connection."""
        x = self.up(x)

        if self.skip_connection and skip is not None:
            # Ensure the dimensions match for concatenation
            diff_z = skip.size(2) - x.size(2)
            diff_y = skip.size(3) - x.size(3)
            diff_x = skip.size(4) - x.size(4)

            # Pad if necessary
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2,
                          diff_z // 2, diff_z - diff_z // 2])

            # Concatenate along the channel dimension
            x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class DoubleConv2d(Module):
    """Double convolution block for 2D N2V2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.double_conv(x)


class Down2d(Module):
    """Downsampling block with max blur pooling for N2V2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv2d(in_channels, out_channels)
        # Max blur pooling: max pool followed by blur
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.conv(x)
        return self.pool(x)


class Up2d(Module):
    """Upsampling block for 2D N2V2."""

    def __init__(self, in_channels: int, out_channels: int, skip_connection: bool = True, skip_channels: int = 0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        if skip_connection:
            # After upsampling and concatenation: (in_channels // 2) + skip_channels
            conv_input_channels = (in_channels // 2) + skip_channels
            self.conv = DoubleConv2d(conv_input_channels, out_channels)
        else:
            self.conv = DoubleConv2d(in_channels // 2, out_channels)

        self.skip_connection = skip_connection

    def forward(self, x: "torch.Tensor", skip: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        x = self.up(x)

        if self.skip_connection and skip is not None:
            # Ensure dimensions match
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)

            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

            x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class N2V2UNet(Module):
    """
    2D U-Net architecture with N2V2 modifications.

    Modifications include:
    - Max blur pooling instead of MaxPool
    - No top-level skip connection
    - 2D convolutions for processing 2D images
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = None):
        """
        Initialize the N2V2UNet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
        """
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        # Input convolution
        self.inc = DoubleConv2d(in_channels, features[0])

        # Downsampling path
        self.down1 = Down2d(features[0], features[1])
        self.down2 = Down2d(features[1], features[2])
        self.down3 = Down2d(features[2], features[3])

        # Bottom convolution
        self.bottom = DoubleConv2d(features[3], features[3] * 2)

        # Upsampling path - account for skip connection concatenation
        self.up1 = Up2d(features[3] * 2, features[2], skip_channels=features[2])
        self.up2 = Up2d(features[2], features[1], skip_channels=features[1])
        self.up3 = Up2d(features[1], features[0], skip_connection=False)  # No top-level skip connection

        # Output convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the 2D U-Net."""
        # Downsampling path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottom
        x5 = self.bottom(x4)

        # Upsampling path with skip connections
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x)  # No skip connection at top level

        # Output
        return self.outc(x)


def generate_blindspot_mask(shape: Tuple[int, ...], prob: float, device: torch.device) -> "torch.Tensor":
    """
    Generate a random binary mask for blind-spot training.

    Args:
        shape: Shape of the mask - can be (Y, X) for single mask or (batch_size, Y, X) for batch
        prob: Probability of masking a pixel
        device: Device to create the mask on

    Returns:
        Binary mask where True indicates pixels to mask
    """
    return (torch.rand(shape, device=device) < prob)


def extract_random_patches_2d(
    image: "torch.Tensor",
    patch_size: int,
    num_patches: int
) -> torch.Tensor:
    """
    Extract random 2D patches from the input image stack using vectorized GPU operations.

    Args:
        image: Input image tensor of shape (Z, Y, X)
        patch_size: Size of the square patches
        num_patches: Number of patches to extract

    Returns:
        Tensor of patches with shape (num_patches, patch_size, patch_size)
    """
    device = image.device  # Get device from input tensor
    z, y, x = image.shape

    # Ensure patch_size is not larger than the spatial dimensions
    if patch_size > min(y, x):
        raise ValueError(f"Patch size {patch_size} is larger than the smallest spatial dimension of the image {min(y, x)}")

    # OPTIMIZED: Generate all random indices at once (NO CPU SYNC)
    # Generate random indices for all patches in a single GPU operation
    z_indices = torch.randint(0, z, (num_patches,), device=device)
    y_starts = torch.randint(0, y - patch_size + 1, (num_patches,), device=device)
    x_starts = torch.randint(0, x - patch_size + 1, (num_patches,), device=device)

    # Pre-allocate output tensor on GPU
    patches = torch.zeros((num_patches, patch_size, patch_size), device=device, dtype=image.dtype)

    # Extract patches using advanced indexing (still requires loop but no CPU sync)
    for i in range(num_patches):
        # Use tensor indexing (no .item() calls - stays on GPU)
        z_idx = z_indices[i]
        y_start = y_starts[i]
        x_start = x_starts[i]

        # Extract 2D patch from selected slice
        patches[i] = image[z_idx, y_start:y_start + patch_size, x_start:x_start + patch_size]

    return patches


def apply_n2v2_masking(patches: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """
    Apply N2V2 masking strategy using median of neighborhood (Höck et al., 2022).

    N2V2 paper specifies: "median replacement" to fix checkerboard artifacts.
    This replaces masked pixels with the median of their 3x3 neighborhood.

    Args:
        patches: Input patches of shape (batch_size, patch_size, patch_size)
        mask: Binary mask of shape (batch_size, patch_size, patch_size)

    Returns:
        Masked patches where masked pixels are replaced with local median
    """
    # Validate input dimensions
    if patches.ndim != 3:
        raise RuntimeError(f"🔥 N2V2 MASKING ERROR: patches must be 3D (batch, H, W), got {patches.ndim}D shape={patches.shape}")
    if mask.ndim != 3:
        raise RuntimeError(f"🔥 N2V2 MASKING ERROR: mask must be 3D (batch, H, W), got {mask.ndim}D shape={mask.shape}")

    batch_size, patch_size, _ = patches.shape
    device = patches.device

    # N2V2 PAPER IMPLEMENTATION: Median replacement for masked pixels
    # Use a simpler, more robust approach that avoids unfold padding issues

    masked_patches = patches.clone()

    # Process each batch item individually to avoid dimension issues
    for b in range(batch_size):
        # Get current patch and mask
        current_patch = patches[b]  # Shape: (patch_size, patch_size)
        current_mask = mask[b]     # Shape: (patch_size, patch_size)

        # Find masked pixel locations
        mask_indices = torch.where(current_mask)

        # For each masked pixel, replace with median of 3x3 neighborhood
        for i, j in zip(mask_indices[0], mask_indices[1]):
            # Define 3x3 neighborhood bounds with boundary handling
            y_min = max(0, i - 1)
            y_max = min(patch_size, i + 2)
            x_min = max(0, j - 1)
            x_max = min(patch_size, j + 2)

            # Extract neighborhood (excluding the center pixel)
            neighborhood = current_patch[y_min:y_max, x_min:x_max].flatten()

            # Remove the center pixel from neighborhood if it exists
            center_y, center_x = i - y_min, j - x_min
            if 0 <= center_y < (y_max - y_min) and 0 <= center_x < (x_max - x_min):
                center_idx = center_y * (x_max - x_min) + center_x
                if center_idx < len(neighborhood):
                    neighborhood = torch.cat([neighborhood[:center_idx], neighborhood[center_idx+1:]])

            # Replace with median (N2V2 paper specification)
            if len(neighborhood) > 0:
                masked_patches[b, i, j] = torch.median(neighborhood)

    return masked_patches


@torch_func
def n2v2_denoise_torch(
    image: "torch.Tensor",
    model_path: Optional[str] = None,
    *,
    random_seed: int = 42,

    blindspot_prob: float = 0.05,
    max_epochs: int = 10,
    batch_size: int = 4,
    patch_size: int = 64,
    learning_rate: float = 1e-4,
    save_model_path: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Denoise a 3D image using N2V2-inspired self-supervised learning.

    This function implements a 3D GPU-native, planner-compatible, self-supervised
    denoising step based on the N2V2 paper (Höck et al., 2022) using PyTorch.

    Args:
        image: Input 3D tensor of shape (Z, Y, X) - MUST be on CUDA device
        model_path: Path to a pre-trained model (optional)
        random_seed: Random seed for reproducibility
        blindspot_prob: Probability of masking a pixel for blind-spot training
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        patch_size: Size of the patches for training
        learning_rate: Learning rate for the optimizer
        save_model_path: Path to save the trained model (optional)
        verbose: Whether to print progress information
        **kwargs: Additional parameters for the model or training

    Returns:
        Denoised 3D tensor of shape (Z, Y, X) on same device as input

    Raises:
        ValueError: If the input is not a 3D tensor
        RuntimeError: If input tensor is not on CUDA device (NO CPU FALLBACK)
    """
    # Get device from input tensor - NO CPU FALLBACK ALLOWED
    device = image.device

    # Validate input
    if image.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor, got {image.ndim}D")

    # FAIL LOUDLY if not on CUDA - no CPU fallback allowed
    if device.type != "cuda":
        raise RuntimeError(f"@torch_func requires CUDA tensor, got device: {device}")

    # OPTIMIZED: Cache shape information early to avoid repeated queries
    z_size, y_size, x_size = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Normalize image (already on correct device)
    image = image.float()
    max_val = image.max()
    image = image / max_val  # Normalize to [0, 1]

    # Create model with N2V2 specifications (depth=3, 64 initial features for microscopy)
    model = N2V2UNet(features=[64, 128, 256, 512], **kwargs).to(device)

    # Load pre-trained model if provided
    if model_path is not None:
        if verbose:
            logger.info(f"Loading model from {model_path}")
        # FAIL LOUDLY if model loading fails - no CPU fallback
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Train model on-the-fly
        if verbose:
            logger.info("Training model on-the-fly")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss(reduction="none")

        model.train()
        for epoch in range(max_epochs):
            # OPTIMIZED: Accumulate losses as tensors to avoid CPU sync
            epoch_losses = []
            num_batches = max(1, min(100, int((z_size * y_size * x_size) / (patch_size**2 * batch_size))))

            for _ in range(num_batches):
                # Extract random 2D patches
                patches = extract_random_patches_2d(image, patch_size, batch_size)

                # OPTIMIZED: Generate all masks at once instead of list comprehension
                masks = generate_blindspot_mask((batch_size, patch_size, patch_size), blindspot_prob, device)

                # Apply N2V2 masking (replace with median, not zero)
                masked_input = apply_n2v2_masking(patches, masks)

                # Add channel dimension for network: (batch_size, 1, patch_size, patch_size)
                patches_input = patches.unsqueeze(1)
                masked_input = masked_input.unsqueeze(1)

                # Forward pass
                prediction = model(masked_input)

                # Compute loss only on masked pixels
                loss = loss_fn(prediction.squeeze(1), patches)
                masked_loss = loss[masks].mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                masked_loss.backward()
                optimizer.step()

                # OPTIMIZED: Store loss tensor instead of calling .item()
                epoch_losses.append(masked_loss.detach())

            # OPTIMIZED: Single CPU sync per epoch for logging
            if verbose and (epoch % 2 == 0 or epoch == max_epochs - 1):
                avg_loss = torch.stack(epoch_losses).mean().item()  # Single .item() call
                logger.info(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.6f}")

        # Save trained model if path is provided
        if save_model_path is not None:
            if verbose:
                logger.info(f"Saving model to {save_model_path}")
            torch.save(model.state_dict(), save_model_path)

    # Inference: Process each 2D slice individually using the learned 2D model
    model.eval()
    with torch.no_grad():
        # OPTIMIZED: Cache shape values to avoid repeated queries
        z_size, y_size, x_size = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        denoised = torch.zeros_like(image)

        # Process each 2D slice separately
        for slice_idx in range(z_size):
            slice_2d = image[slice_idx]  # Shape: (y, x)

            if max(y_size, x_size) <= 256:  # Small enough to process at once
                # Add batch and channel dimensions: (1, 1, y, x) for 2D model
                slice_input = slice_2d.unsqueeze(0).unsqueeze(0)
                prediction = model(slice_input)
                denoised[slice_idx] = prediction.squeeze()  # Remove extra dimensions
            else:
                # Process in overlapping 2D patches
                stride = patch_size // 2
                slice_denoised = torch.zeros_like(slice_2d)
                count = torch.zeros_like(slice_2d)

                # Pad 2D slice - ensure slice_2d is 2D before padding
                if slice_2d.ndim != 2:
                    raise RuntimeError(f"🔥 N2V2 INFERENCE ERROR: slice_2d must be 2D, got {slice_2d.ndim}D shape={slice_2d.shape}")

                # Pad with reflection for edge handling: [left, right, top, bottom]
                pad_size = patch_size // 2
                padded_2d = F.pad(slice_2d, [pad_size, pad_size, pad_size, pad_size], mode='reflect')

                for y_start in range(0, y_size, stride):
                    for x_start in range(0, x_size, stride):
                        # Extract 2D patch
                        y_end = min(y_start + patch_size, y_size)
                        x_end = min(x_start + patch_size, x_size)

                        # Get 2D patch and add dimensions for 2D model: (1, 1, patch_size, patch_size)
                        patch_2d = padded_2d[
                            y_start:y_start + patch_size,
                            x_start:x_start + patch_size
                        ]
                        patch_input = patch_2d.unsqueeze(0).unsqueeze(0)

                        pred_patch = model(patch_input)

                        # Add to result
                        slice_denoised[
                            y_start:y_end,
                            x_start:x_end
                        ] += pred_patch.squeeze()[
                            :y_end-y_start,
                            :x_end-x_start
                        ]

                        count[
                            y_start:y_end,
                            x_start:x_end
                        ] += 1

                # Average overlapping regions
                denoised[slice_idx] = slice_denoised / count.clamp(min=1)

    # Rescale to original range and convert to uint16
    denoised = denoised.clip(0, 1) * max_val

    return denoised.to(dtype=torch.uint16)
