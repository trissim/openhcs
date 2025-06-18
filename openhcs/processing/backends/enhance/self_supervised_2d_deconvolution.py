from __future__ import annotations 
import logging
from typing import Optional, Tuple

# Import torch decorator and optional_import utility
from openhcs.utils.import_utils import optional_import, create_placeholder_class
from openhcs.core.memory.decorators import torch as torch_func

# --- PyTorch Imports as optional dependencies ---
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None
if torch is not None:
    from torch.fft import irfft2, rfft2
else:
    irfft2 = None
    rfft2 = None

logger = logging.getLogger(__name__)

nnModule = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)
# --- PyTorch Specific Models and Helpers for 2D ---
class _Simple2DCNN_torch(nnModule):
    """Simple 2D CNN for deconvolution - optimized for 2D data per paper."""
    def __init__(self, in_channels=1, out_channels=1, features=(96, 192)):  # Paper: 96 initial features for 2D
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, H, W)
        return self.conv_block(x)

class _LearnedBlur2D_torch(nnModule):
    """Learned blur for 2D deconvolution."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.blur_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # Initialize weights to be somewhat like a Gaussian blur
        if kernel_size > 0:
            weights = torch.ones(kernel_size, kernel_size)
            weights = weights / weights.sum()
            self.blur_conv.weight.data = weights.reshape(1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, H, W)
        return self.blur_conv(x)

def _gaussian_kernel_2d_torch(shape: Tuple[int, int], sigma: Tuple[float, float], device) -> torch.Tensor:
    """Generate 2D Gaussian kernel."""
    coords_h = torch.arange(shape[0], dtype=torch.float32, device=device) - (shape[0] - 1) / 2.0
    coords_w = torch.arange(shape[1], dtype=torch.float32, device=device) - (shape[1] - 1) / 2.0

    kernel_h = torch.exp(-coords_h**2 / (2 * sigma[0]**2))
    kernel_w = torch.exp(-coords_w**2 / (2 * sigma[1]**2))

    kernel = torch.outer(kernel_h, kernel_w)
    return kernel / torch.sum(kernel)

def _blur_fft_2d_torch(image: torch.Tensor, kernel: torch.Tensor, device) -> torch.Tensor:
    """FFT-based 2D blur convolution."""
    # image: (B, 1, H, W), kernel: (kH, kW)
    B, C, H, W = image.shape
    kH, kW = kernel.shape

    # Pad kernel to image size for FFT
    kernel_padded = F.pad(kernel, (
        (W - kW) // 2, (W - kW + 1) // 2,
        (H - kH) // 2, (H - kH + 1) // 2,
    ))
    kernel_padded = kernel_padded.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    img_fft = rfft2(image, dim=(-2, -1))
    ker_fft = rfft2(kernel_padded.to(device), dim=(-2, -1))

    blurred_fft = img_fft * ker_fft
    blurred_img = irfft2(blurred_fft, s=(H, W), dim=(-2, -1))
    return blurred_img

def _extract_random_patches_2d_torch(
    image_single_batch_channel: torch.Tensor,  # (H, W)
    patch_size_hw: Tuple[int, int],
    num_patches: int,
    device
) -> torch.Tensor:  # (num_patches, 1, pH, pW)
    """Extract random 2D patches - GPU-native."""
    H, W = image_single_batch_channel.shape
    pH, pW = patch_size_hw
    patches = torch.empty((num_patches, 1, pH, pW), device=device)
    for i in range(num_patches):
        # Force GPU device for random operations - NO CPU FALLBACK
        h_start = torch.randint(0, H - pH + 1, (1,), device=device).item()
        w_start = torch.randint(0, W - pW + 1, (1,), device=device).item()
        patch = image_single_batch_channel[
            h_start:h_start+pH, w_start:w_start+pW
        ]
        patches[i, 0, ...] = patch
    return patches

# --- Main 2D Deconvolution Function ---
@torch_func
def self_supervised_2d_deconvolution(
    image: torch.Tensor,  # Expected (H, W) or (1, H, W)
    apply_deconvolution: bool = True,
    n_epochs: int = 10,  # Reduced for testing
    patch_size_hw: Tuple[int, int] = (128, 128),  # Paper: 128x128 for 2D
    mask_fraction: float = 0.005,  # Paper: 0.5%
    sigma_noise: float = 0.2,
    lambda_rec: float = 1.0,
    lambda_inv_d: float = 2.0,  # Paper: deconvolved invariance for 2D
    lambda_bound_d: float = 0.1,  # Paper: boundary loss for 2D
    min_val: float = 0.0,
    max_val: float = 1.0,
    learning_rate: float = 4e-4,  # Paper: Adam 4e-4
    blur_mode: str = "gaussian",  # 'fft', 'gaussian', 'learned'
    blur_sigma_spatial: float = 1.5,
    blur_kernel_size: int = 5,
    **kwargs
) -> torch.Tensor:
    """
    Self-supervised 2D deconvolution optimized for 2D imaging data.
    
    Based on the paper's optimal 2D configuration:
    - 96 initial features (vs 48 for 3D)
    - 128x128 patches (vs 64x64x64 for 3D)
    - Batch size 16 (vs 4 for 3D)
    - Loss (4) with deconvolved invariance (vs reconvolved for 3D)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image must be a PyTorch Tensor. Got {type(image)}")


    if not apply_deconvolution:
        return image

    # --- PyTorch Backend Implementation ---
    device = image.device

    # FAIL LOUDLY if not on CUDA - no CPU fallback allowed
    if device.type != "cuda":
        raise RuntimeError(f"@torch_func requires CUDA tensor, got device: {device}")

    # Ensure input is (1, 1, H, W)
    if image.ndim == 2:  # (H, W)
        img_norm = image.unsqueeze(0).unsqueeze(0).float()
    elif image.ndim == 3:  # (1, H, W)
        img_norm = image.unsqueeze(1).float()
    elif image.ndim == 4:  # (1, 1, H, W)
        img_norm = image.float()
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    # Normalize to [min_val, max_val]
    img_min_orig, img_max_orig = torch.min(img_norm), torch.max(img_norm)
    if img_max_orig > img_min_orig:
        img_norm = (img_norm - img_min_orig) / (img_max_orig - img_min_orig)
        img_norm = img_norm * (max_val - min_val) + min_val
    else:
        img_norm = torch.full_like(img_norm, min_val)

    # Create 2D model with paper's optimal architecture
    f_model = _Simple2DCNN_torch().to(device)

    g_model_blur: Optional[nn.Module] = None
    fixed_blur_kernel: Optional[torch.Tensor] = None

    if blur_mode == "learned":
        g_model_blur = _LearnedBlur2D_torch(kernel_size=blur_kernel_size).to(device)
        optimizer = torch.optim.Adam(list(f_model.parameters()) + list(g_model_blur.parameters()), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(f_model.parameters(), lr=learning_rate)
        if blur_mode in ["gaussian", "fft"]:
            fixed_blur_kernel = _gaussian_kernel_2d_torch(
                (blur_kernel_size, blur_kernel_size),
                (blur_sigma_spatial, blur_sigma_spatial), device
            )

    # Training Loop
    for epoch in range(n_epochs):
        f_model.train()
        if g_model_blur:
            g_model_blur.train()

        # Extract 2D patch
        current_patch_orig = _extract_random_patches_2d_torch(
            img_norm.squeeze(0).squeeze(0), patch_size_hw, 1, device
        )

        # Create masked variant
        mask = (torch.rand_like(current_patch_orig) < mask_fraction).bool()
        noise = (torch.randn_like(current_patch_orig) * sigma_noise).clamp(min_val, max_val)
        current_patch_masked = torch.where(mask, noise, current_patch_orig)

        # Forward pass f(x)
        f_x_orig = f_model(current_patch_orig).clamp(min_val, max_val)
        f_x_masked = f_model(current_patch_masked).clamp(min_val, max_val)

        # Apply blur g(f(x))
        if blur_mode == "learned":
            g_f_x_orig = g_model_blur(f_x_orig)
            g_f_x_masked = g_model_blur(f_x_masked)
        elif blur_mode == "fft":
            g_f_x_orig = _blur_fft_2d_torch(f_x_orig, fixed_blur_kernel, device)
            g_f_x_masked = _blur_fft_2d_torch(f_x_masked, fixed_blur_kernel, device)
        elif blur_mode == "gaussian":
            conv_kernel = fixed_blur_kernel.unsqueeze(0).unsqueeze(0).to(device)
            pad_size = blur_kernel_size // 2
            g_f_x_orig = F.conv2d(f_x_orig, conv_kernel, padding=pad_size)
            g_f_x_masked = F.conv2d(f_x_masked, conv_kernel, padding=pad_size)
        else:
            raise ValueError(f"Unknown blur_mode: {blur_mode}")

        # Losses - Paper's optimal Loss (4) for 2D: deconvolved invariance
        loss_rec = F.mse_loss(g_f_x_masked, current_patch_orig)

        # Deconvolved invariance loss (before PSF) - optimal for 2D per paper
        loss_inv_d = torch.tensor(0.0, device=device)
        if mask.sum() > 0:
            loss_inv_d = F.mse_loss(f_x_orig[mask], f_x_masked[mask])

        # Boundary loss on deconvolved output
        loss_bound_d = (torch.relu(f_x_masked - max_val) + torch.relu(min_val - f_x_masked)).mean()
        loss_bound_d += (torch.relu(f_x_orig - max_val) + torch.relu(min_val - f_x_orig)).mean()
        loss_bound_d /= 2.0

        total_loss = lambda_rec * loss_rec + lambda_inv_d * loss_inv_d + lambda_bound_d * loss_bound_d

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % (n_epochs // 10 if n_epochs >= 10 else 1) == 0:
            logger.info(f"2D Deconv Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.4f} "
                       f"(Rec: {loss_rec.item():.4f}, Inv_d: {loss_inv_d.item():.4f}, Bound_d: {loss_bound_d.item():.4f})")

    # Inference
    f_model.eval()
    with torch.no_grad():
        deconvolved_norm = f_model(img_norm).clamp(min_val, max_val)

    # Denormalize
    if img_max_orig > img_min_orig:
        deconvolved_final = (deconvolved_norm - min_val) / (max_val - min_val)
        deconvolved_final = deconvolved_final * (img_max_orig - img_min_orig) + img_min_orig
    else:
        deconvolved_final = torch.full_like(deconvolved_norm, img_min_orig)

    # Return in original input shape
    if image.ndim == 2:
        return deconvolved_final.squeeze(0).squeeze(0)
    elif image.ndim == 3:
        return deconvolved_final.squeeze(1)
    return deconvolved_final
