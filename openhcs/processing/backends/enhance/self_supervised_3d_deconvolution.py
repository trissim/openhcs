from __future__ import annotations
import logging
from typing import Optional, Tuple, Tuple

# Import torch decorator and optional_import utility
from openhcs.utils.import_utils import optional_import, create_placeholder_class
from openhcs.core.memory.decorators import torch as torch_func

# --- PyTorch Imports as optional dependencies ---
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None
if torch is not None:
    from torch.fft import irfftn, rfftn
else:
    irfftn = None
    rfftn = None

logger = logging.getLogger(__name__)

nnModule = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)

# --- PyTorch Specific Models and Helpers ---
class _Simple3DCNN_torch(nnModule):
    """Simple 3D CNN for deconvolution - optimized for 3D data per paper."""
    def __init__(self, in_channels=1, out_channels=1, features=(48, 96)):  # Paper: 48 initial features for 3D
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features[1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, D, H, W)
        return self.conv_block(x)

class _LearnedBlur3D_torch(nnModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.blur_conv = nn.Conv3d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # Initialize weights to be somewhat like a Gaussian blur
        # For simplicity, using default initialization or a simple averaging kernel
        if kernel_size > 0 :
            weights = torch.ones(kernel_size, kernel_size, kernel_size)
            weights = weights / weights.sum()
            self.blur_conv.weight.data = weights.reshape(1,1,kernel_size,kernel_size,kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (B, 1, D, H, W)
        return self.blur_conv(x)

def _gaussian_kernel_3d_torch(shape: Tuple[int, int, int], sigma: Tuple[float, float, float], device) -> torch.Tensor:
    coords_d = torch.arange(shape[0], dtype=torch.float32, device=device) - (shape[0] - 1) / 2.0
    coords_h = torch.arange(shape[1], dtype=torch.float32, device=device) - (shape[1] - 1) / 2.0
    coords_w = torch.arange(shape[2], dtype=torch.float32, device=device) - (shape[2] - 1) / 2.0

    kernel_d = torch.exp(-coords_d**2 / (2 * sigma[0]**2))
    kernel_h = torch.exp(-coords_h**2 / (2 * sigma[1]**2))
    kernel_w = torch.exp(-coords_w**2 / (2 * sigma[2]**2))

    # Create 3D kernel using broadcasting instead of nested outer products
    kernel = kernel_d[:, None, None] * kernel_h[None, :, None] * kernel_w[None, None, :]
    return kernel / torch.sum(kernel)

def _blur_fft_torch(volume: torch.Tensor, kernel: torch.Tensor, device) -> torch.Tensor:
    # volume: (B, 1, D, H, W), kernel: (kD, kH, kW)
    B, C, D, H, W = volume.shape
    kD, kH, kW = kernel.shape

    # Pad kernel to volume size for FFT
    kernel_padded = F.pad(kernel, (
        (W - kW) // 2, (W - kW + 1) // 2,
        (H - kH) // 2, (H - kH + 1) // 2,
        (D - kD) // 2, (D - kD + 1) // 2,
    ))
    kernel_padded = kernel_padded.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)

    vol_fft = rfftn(volume, dim=(-3, -2, -1))
    ker_fft = rfftn(kernel_padded.to(device), dim=(-3, -2, -1))

    blurred_fft = vol_fft * ker_fft
    blurred_vol = irfftn(blurred_fft, s=(D, H, W), dim=(-3, -2, -1))
    return blurred_vol

def _blur_gaussian_conv_torch(volume: torch.Tensor, sigma_spatial: float, sigma_depth: float, kernel_size: int, device) -> torch.Tensor:
    kernel_d_1d = _gaussian_kernel_3d_torch((kernel_size,1,1), (sigma_depth,1,1), device)[:,0,0]
    kernel_h_1d = _gaussian_kernel_3d_torch((1,kernel_size,1), (1,sigma_spatial,1), device)[0,:,0]
    kernel_w_1d = _gaussian_kernel_3d_torch((1,1,kernel_size), (1,1,sigma_spatial), device)[0,0,:]

    kernel = kernel_d_1d[:,None,None] * kernel_h_1d[None,:,None] * kernel_w_1d[None,None,:]
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, kD, kH, kW)

    padding = kernel_size // 2
    return F.conv3d(volume, kernel, padding=padding)

def _extract_random_patches_torch(
    volume_single_batch_channel: torch.Tensor, # (D, H, W)
    patch_size_dhw: Tuple[int, int, int],
    num_patches: int,
    device
) -> torch.Tensor: # (num_patches, 1, pD, pH, pW)
    D, H, W = volume_single_batch_channel.shape
    pD, pH, pW = patch_size_dhw
    patches = torch.empty((num_patches, 1, pD, pH, pW), device=device)
    for i in range(num_patches):
        # Force GPU device for random operations - NO CPU FALLBACK
        d_start = torch.randint(0, D - pD + 1, (1,), device=device).item()
        h_start = torch.randint(0, H - pH + 1, (1,), device=device).item()
        w_start = torch.randint(0, W - pW + 1, (1,), device=device).item()
        patch = volume_single_batch_channel[
            d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW
        ]
        patches[i, 0, ...] = patch
    return patches

# --- Main Deconvolution Function ---
@torch_func
def self_supervised_3d_deconvolution(
    image_volume: torch.Tensor, # Expected (1, Z, H, W) or (Z,H,W)
    apply_deconvolution: bool = True,
    n_epochs: int = 10,  # Reduced default for quick test (was 100)
    patch_size: Tuple[int, int, int] = (16, 32, 32),  # Reduced for small test images (paper: 64x64x64)
    mask_fraction: float = 0.005,  # Paper: 0.5%
    sigma_noise: float = 0.2,
    lambda_rec: float = 1.0,
    lambda_inv: float = 2.0,  # Paper: reconvolved invariance for 3D
    lambda_bound: float = 0.0,  # Paper: λbound = 0 for 3D
    min_val: float = 0.0,
    max_val: float = 1.0,
    learning_rate: float = 4e-4,
    blur_mode: str = "gaussian",  # 'fft', 'gaussian', 'learned'
    blur_sigma_spatial: float = 1.5,
    blur_sigma_depth: float = 1.5,
    blur_kernel_size: int = 5,  # For gaussian/learned conv blur
    **kwargs
) -> torch.Tensor:

    if not isinstance(image_volume, torch.Tensor):
        raise TypeError(f"Input image_volume must be a PyTorch Tensor. Got {type(image_volume)}")

    # --- Parameters already extracted from function signature ---
    patch_size_dhw = tuple(patch_size)  # Convert to tuple for consistency

    if not apply_deconvolution:
        return image_volume

    # --- PyTorch Backend Implementation ---
    device = image_volume.device

    # FAIL LOUDLY if not on CUDA - no CPU fallback allowed
    if device.type != "cuda":
        raise RuntimeError(f"@torch_func requires CUDA tensor, got device: {device}")

    # Ensure input is (1, 1, D, H, W)
    if image_volume.ndim == 3:  # (D, H, W)
        img_vol_norm = image_volume.unsqueeze(0).unsqueeze(0).float()
    elif image_volume.ndim == 4:  # (1, D, H, W)
        img_vol_norm = image_volume.unsqueeze(1).float()
    elif image_volume.ndim == 5:  # (1, 1, D, H, W)
        img_vol_norm = image_volume.float()
    else:
        raise ValueError(f"Unsupported image_volume ndim: {image_volume.ndim}")

    # Normalize to [min_val, max_val] (typically [0,1])
    img_min_orig, img_max_orig = torch.min(img_vol_norm), torch.max(img_vol_norm)
    if img_max_orig > img_min_orig:
        img_vol_norm = (img_vol_norm - img_min_orig) / (img_max_orig - img_min_orig) # to [0,1]
        img_vol_norm = img_vol_norm * (max_val - min_val) + min_val # to [min_val, max_val]
    else: # Constant image
        img_vol_norm = torch.full_like(img_vol_norm, min_val)

    f_model = _Simple3DCNN_torch().to(device)

    g_model_blur: Optional[nn.Module] = None
    fixed_blur_kernel: Optional[torch.Tensor] = None

    if blur_mode == "learned":
        g_model_blur = _LearnedBlur3D_torch(kernel_size=blur_kernel_size).to(device)
        optimizer = torch.optim.Adam(list(f_model.parameters()) + list(g_model_blur.parameters()), lr=learning_rate)
    else: # fft or gaussian (fixed kernel)
        optimizer = torch.optim.Adam(f_model.parameters(), lr=learning_rate)
        if blur_mode == "gaussian" or blur_mode == "fft": # FFT also needs a kernel
            fixed_blur_kernel = _gaussian_kernel_3d_torch(
                (blur_kernel_size, blur_kernel_size, blur_kernel_size),
                (blur_sigma_depth, blur_sigma_spatial, blur_sigma_spatial), device
            )

    # Training Loop
    for epoch in range(n_epochs):
        f_model.train()
        if g_model_blur: g_model_blur.train()

        # Extract one patch for this step (batch_size=1 for patches)
        # Input to _extract_random_patches_torch is (D,H,W)
        current_patch_orig = _extract_random_patches_torch(img_vol_norm.squeeze(0).squeeze(0), patch_size_dhw, 1, device)
        # current_patch_orig shape: (1, 1, pD, pH, pW)

        # Create masked variant
        mask = (torch.rand_like(current_patch_orig) < mask_fraction).bool()
        noise = (torch.randn_like(current_patch_orig) * sigma_noise).clamp(min_val, max_val)
        current_patch_masked = torch.where(mask, noise, current_patch_orig)

        # Forward pass f(x)
        f_x_orig = f_model(current_patch_orig).clamp(min_val, max_val)
        f_x_masked = f_model(current_patch_masked).clamp(min_val, max_val)

        # Apply blur g(f(x))
        if blur_mode == "learned":
            assert g_model_blur is not None
            g_f_x_orig = g_model_blur(f_x_orig)
            g_f_x_masked = g_model_blur(f_x_masked)
        elif blur_mode == "fft":
            assert fixed_blur_kernel is not None
            g_f_x_orig = _blur_fft_torch(f_x_orig, fixed_blur_kernel, device)
            g_f_x_masked = _blur_fft_torch(f_x_masked, fixed_blur_kernel, device)
        elif blur_mode == "gaussian": # Conv based Gaussian
            assert fixed_blur_kernel is not None # Re-use kernel logic for conv
            # For conv3d, kernel needs to be (out_c, in_c/groups, kD, kH, kW)
            conv_kernel = fixed_blur_kernel.unsqueeze(0).unsqueeze(0).to(device)
            pad_size = blur_kernel_size // 2
            g_f_x_orig = F.conv3d(f_x_orig, conv_kernel, padding=pad_size)
            g_f_x_masked = F.conv3d(f_x_masked, conv_kernel, padding=pad_size)
        else:
            raise ValueError(f"Unknown blur_mode: {blur_mode}")

        # Losses - Paper's optimal Loss (3) for 3D: reconvolved invariance
        loss_rec = F.mse_loss(g_f_x_masked, current_patch_orig)

        # Reconvolved invariance loss (after PSF) - optimal for 3D per paper
        loss_inv = torch.tensor(0.0, device=device)
        if mask.sum() > 0: # Only if some pixels were masked
            loss_inv = F.mse_loss(g_f_x_orig[mask], g_f_x_masked[mask])

        # Boundary loss on reconvolved output (λbound = 0 for 3D per paper)
        loss_bound = torch.tensor(0.0, device=device)
        if lambda_bound > 0:
            loss_bound_f_masked = (torch.relu(g_f_x_masked - max_val) + torch.relu(min_val - g_f_x_masked)).mean()
            loss_bound_f_orig = (torch.relu(g_f_x_orig - max_val) + torch.relu(min_val - g_f_x_orig)).mean()
            loss_bound = (loss_bound_f_masked + loss_bound_f_orig) / 2.0

        total_loss = lambda_rec * loss_rec + lambda_inv * loss_inv + lambda_bound * loss_bound

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % (n_epochs // 10 if n_epochs >=10 else 1) == 0:
            logger.info(f"3D Deconv Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.4f} "
                       f"(Rec: {loss_rec.item():.4f}, Inv: {loss_inv.item():.4f}, Bound: {loss_bound.item():.4f})")

    # Inference
    f_model.eval()
    with torch.no_grad():
        # Full volume inference - may need patching for large volumes on limited GPU
        # Assuming img_vol_norm is (1, 1, D, H, W)
        deconvolved_norm = f_model(img_vol_norm).clamp(min_val, max_val)

    # Denormalize from [min_val, max_val] back to original image range [0, orig_max_val_if_uint]
    if img_max_orig > img_min_orig:
        deconvolved_final = (deconvolved_norm - min_val) / (max_val - min_val) # to [0,1]
        deconvolved_final = deconvolved_final * (img_max_orig - img_min_orig) + img_min_orig
    else: # Constant image
        deconvolved_final = torch.full_like(deconvolved_norm, img_min_orig)

    # Return in the original input shape format if it was (1,D,H,W) or (D,H,W)
    if image_volume.ndim == 3: return deconvolved_final.squeeze(0).squeeze(0)
    if image_volume.ndim == 4: return deconvolved_final.squeeze(1)
    return deconvolved_final # (1,1,D,H,W)
