from openhcs.core.memory.decorators import torch as torch_func

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

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BlurPool3d(nn.Module):
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


class DoubleConv3d(nn.Module):
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


class Down3d(nn.Module):
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


class Up3d(nn.Module):
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


class N2V2UNet(nn.Module):
    """
    3D U-Net architecture with N2V2 modifications.
    
    Modifications include:
    - BlurPool instead of MaxPool
    - No top-level skip connection
    - No residual connections
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
        self.inc = DoubleConv3d(in_channels, features[0])
        
        # Downsampling path
        self.down1 = Down3d(features[0], features[1])
        self.down2 = Down3d(features[1], features[2])
        self.down3 = Down3d(features[2], features[3])
        
        # Bottom convolution
        self.bottom = DoubleConv3d(features[3], features[3] * 2)
        
        # Upsampling path
        self.up1 = Up3d(features[3] * 2, features[2])
        self.up2 = Up3d(features[2], features[1])
        self.up3 = Up3d(features[1], features[0], skip_connection=False)  # No top-level skip connection
        
        # Output convolution
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the U-Net."""
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


def generate_blindspot_mask(shape: Tuple[int, ...], prob: float, device: str) -> "torch.Tensor":
    """
    Generate a random binary mask for blind-spot training.
    
    Args:
        shape: Shape of the mask (Z, Y, X)
        prob: Probability of masking a pixel
        device: Device to create the mask on
        
    Returns:
        Binary mask where True indicates pixels to mask
    """
    return (torch.rand(shape, device=device) < prob)


def extract_random_patches(
    image: "torch.Tensor",
    patch_size: int,
    num_patches: int,
    device: str
) -> torch.Tensor:
    """
    Extract random 3D patches from the input image.
    
    Args:
        image: Input image tensor of shape (Z, Y, X)
        patch_size: Size of the cubic patches
        num_patches: Number of patches to extract
        device: Device to create the patches on
        
    Returns:
        Tensor of patches with shape (num_patches, patch_size, patch_size, patch_size)
    """
    z, y, x = image.shape
    
    # Ensure patch_size is not larger than any dimension
    if patch_size > min(z, y, x):
        raise ValueError(f"Patch size {patch_size} is larger than the smallest dimension of the image {min(z, y, x)}")
    
    patches = []
    for _ in range(num_patches):
        # Random starting indices
        z_start = torch.randint(0, z - patch_size + 1, (1,)).item()
        y_start = torch.randint(0, y - patch_size + 1, (1,)).item()
        x_start = torch.randint(0, x - patch_size + 1, (1,)).item()
        
        # Extract patch
        patch = image[
            z_start:z_start + patch_size,
            y_start:y_start + patch_size,
            x_start:x_start + patch_size
        ]
        
        patches.append(patch)
    
    return torch.stack(patches)


@torch_func
def n2v2_denoise_torch(
    image: "torch.Tensor",
    model_path: Optional[str] = None,
    *,
    random_seed: int = 42,
    device: str = "cuda",
    blindspot_prob: float = 0.05,
    max_epochs: int = 10,
    batch_size: int = 4,
    patch_size: int = 64,
    learning_rate: float = 1e-4,
    save_model_path: Optional[str] = None,
    verbose: bool = False,
    denoise: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Denoise a 3D image using N2V2-inspired self-supervised learning.
    
    This function implements a 3D GPU-native, planner-compatible, self-supervised
    denoising step based on the N2V2 paper (Höck et al., 2022) using PyTorch.
    
    Args:
        image: Input 3D tensor of shape (Z, Y, X)
        model_path: Path to a pre-trained model (optional)
        random_seed: Random seed for reproducibility
        device: Device to run the model on ("cuda" or "cpu")
        blindspot_prob: Probability of masking a pixel for blind-spot training
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        patch_size: Size of the patches for training
        learning_rate: Learning rate for the optimizer
        save_model_path: Path to save the trained model (optional)
        verbose: Whether to print progress information
        **kwargs: Additional parameters for the model or training
        
    Returns:
        Denoised 3D tensor of shape (Z, Y, X)
        
    Raises:
        ValueError: If the input is not a 3D tensor
        RuntimeError: If CUDA is not available when device="cuda"
    """
    # Validate input
    if image.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor, got {image.ndim}D")
    
    # Check if CUDA is available when device is "cuda"
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device='cuda' was specified")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    if str(device) == "cuda":
        torch.cuda.manual_seed(random_seed)
    
    # Move image to device and normalize
    image = image.to(device).float()
    max_val = image.max()
    image = image / max_val  # Normalize to [0, 1]
    
    # Create model
    model = N2V2UNet(**kwargs).to(device)
    
    # Load pre-trained model if provided
    if model_path is not None:
        if verbose:
            logger.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train model on-the-fly
        if verbose:
            logger.info("Training model on-the-fly")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss(reduction="none")
        
        model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_batches = max(1, min(100, int(math.prod(image.shape) / (patch_size**3 * batch_size))))
            
            for _ in range(num_batches):
                # Extract random patches
                patches = extract_random_patches(image, patch_size, batch_size, device)
                
                # Generate blind-spot masks
                masks = torch.stack([
                    generate_blindspot_mask((patch_size, patch_size, patch_size), blindspot_prob, device)
                    for _ in range(batch_size)
                ])
                
                # Create masked input (set masked pixels to 0)
                masked_input = patches.clone()
                masked_input[masks] = 0
                
                # Forward pass
                prediction = model(masked_input.unsqueeze(1))  # Add channel dimension
                
                # Compute loss only on masked pixels
                loss = loss_fn(prediction.squeeze(1), patches)
                masked_loss = loss[masks].mean()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                masked_loss.backward()
                optimizer.step()
                
                epoch_loss += masked_loss.item()
            
            if verbose and (epoch % 2 == 0 or epoch == max_epochs - 1):
                logger.info(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss/num_batches:.6f}")
        
        # Save trained model if path is provided
        if save_model_path is not None:
            if verbose:
                logger.info(f"Saving model to {save_model_path}")
            torch.save(model.state_dict(), save_model_path)
    
    # Inference
    model.eval()
    with torch.no_grad():
        # Process the entire image at once if it fits in memory
        # Otherwise, process in patches and stitch the results
        z, y, x = image.shape
        
        if max(z, y, x) <= 256:  # Small enough to process at once
            prediction = model(image.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
            denoised = prediction.squeeze()  # Remove batch and channel dimensions
        else:
            # Process in overlapping patches
            stride = patch_size // 2
            denoised = torch.zeros_like(image)
            count = torch.zeros_like(image)
            
            # Pad image to ensure all pixels are processed
            padded = F.pad(image, [patch_size//2] * 6, mode='reflect')
            padded = padded.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            for z_start in range(0, z, stride):
                for y_start in range(0, y, stride):
                    for x_start in range(0, x, stride):
                        # Extract patch with padding
                        z_end = min(z_start + patch_size, z)
                        y_end = min(y_start + patch_size, y)
                        x_end = min(x_start + patch_size, x)
                        
                        # Process patch
                        patch = padded[
                            :, :,
                            z_start:z_start + patch_size,
                            y_start:y_start + patch_size,
                            x_start:x_start + patch_size
                        ]
                        
                        pred_patch = model(patch)
                        
                        # Add to result with proper alignment
                        denoised[
                            z_start:z_end,
                            y_start:y_end,
                            x_start:x_end
                        ] += pred_patch.squeeze()[
                            :z_end-z_start,
                            :y_end-y_start,
                            :x_end-x_start
                        ]
                        
                        count[
                            z_start:z_end,
                            y_start:y_end,
                            x_start:x_end
                        ] += 1
            
            # Average overlapping regions
            denoised = denoised / count.clamp(min=1)
    
    # Rescale to original range and convert to uint16
    denoised = denoised.clip(0, 1) * max_val
    
    return denoised.to(dtype=torch.uint16)
