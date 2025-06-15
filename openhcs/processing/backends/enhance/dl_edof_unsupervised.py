from __future__ import annotations 
from typing import TYPE_CHECKING, List, Optional

from openhcs.core.memory.decorators import torch as torch_func
from openhcs.utils.import_utils import optional_import, create_placeholder_class



# For type checking only
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

# Import torch modules using optional_import
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None

nnModule = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)

# Helper for sharpness loss
def laplacian_filter_torch(image_batch: "torch.Tensor") -> "torch.Tensor":
    """
    Applies a Laplacian filter to a batch of 2D images.
    Input: (N, 1, H, W)
    Output: (N, 1, H, W)
    """
    kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]],
                          dtype=image_batch.dtype, device=image_batch.device).reshape(1, 1, 3, 3)
    return F.conv2d(image_batch, kernel, padding=1)

def extract_patches_2d_from_3d_stack(
    stack_3d: "torch.Tensor", patch_size: int, stride: int
) -> torch.Tensor:
    """
    Extracts 2D patches from a 3D stack.
    Input stack_3d: [Z, H, W]
    Output patches: [N, Z, patch_size, patch_size], where N is num_patches.
    """
    Z, H, W = stack_3d.shape
    patches = stack_3d.unfold(1, patch_size, stride)
    patches = patches.unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4)
    patches = patches.reshape(-1, Z, patch_size, patch_size)
    return patches

def blend_patches_to_2d_image(
    patch_outputs: List["torch.Tensor"],  # List of [1, patch_size, patch_size]
    target_h: int,
    target_w: int,
    patch_size: int,
    stride: int,
    device: torch.device
) -> torch.Tensor:
    """
    Blends 2D fused patches back into a single 2D image.
    Input patch_outputs: List of [1, patch_size, patch_size] tensors.
    Output: [1, target_h, target_w]
    """
    fused_image = torch.zeros((target_h, target_w), dtype=torch.float32, device=device)
    count_map = torch.zeros((target_h, target_w), dtype=torch.float32, device=device)

    num_blocks_h = (target_h - patch_size) // stride + 1
    num_blocks_w = (target_w - patch_size) // stride + 1

    patch_idx = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if patch_idx >= len(patch_outputs):
                # This case should ideally not be reached if inputs are consistent
                break

            patch_content = patch_outputs[patch_idx].squeeze(0) # [patch_size, patch_size]

            h_start = i * stride
            w_start = j * stride

            h_end = h_start + patch_size
            w_end = w_start + patch_size

            fused_image[h_start:h_end, w_start:w_end] += patch_content
            count_map[h_start:h_end, w_start:w_end] += 1.0 # Use float for count_map
            patch_idx += 1

    fused_image /= count_map.clamp(min=1.0)
    return fused_image.unsqueeze(0)

class UNetLite(nnModule):
    def __init__(self, in_channels_z: int, model_config_depth: int):
        super().__init__()
        multiplier = 1 if model_config_depth == 3 else 2
        ch1 = 32 * multiplier
        ch2 = 64 * multiplier

        self.conv1 = nn.Conv2d(in_channels_z, ch1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, stride=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.upconv = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(ch1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.relu3(self.upconv(x2))
        out = self.sigmoid(self.conv_out(x3))
        return out

def sharpness_loss_fn(fused_patch: "torch.Tensor") -> "torch.Tensor":
    laplacian_response = laplacian_filter_torch(fused_patch)
    return -torch.var(laplacian_response, dim=(-1, -2), unbiased=False).mean()

def consistency_loss_fn(fused_patch: "torch.Tensor", input_patch_stack: "torch.Tensor") -> "torch.Tensor":
    diff_sq = (fused_patch - input_patch_stack)**2
    min_diff_sq_over_z = torch.min(diff_sq, dim=1)[0]
    return torch.mean(min_diff_sq_over_z)

@torch_func
def dl_edof_unsupervised(
    image_stack: "torch.Tensor",
    model_depth: Optional[int] = None,
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    denoise: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("PyTorch is required for this function")
    if not (image_stack.ndim == 3 and str(image_stack.device.type) == 'cuda'):
        raise ValueError("Input image_stack must be a 3D CUDA tensor [Z, H, W]. "
                         f"Got {image_stack.ndim}D tensor on {image_stack.device.type}.")

    Z_orig, H_orig, W_orig = image_stack.shape
    device = image_stack.device
    original_dtype = image_stack.dtype

    # Memory usage warning for large images
    total_elements = Z_orig * H_orig * W_orig
    if total_elements > 100_000_000:  # 100M elements
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️  Large image stack ({total_elements:,} elements) may cause high memory usage in deep learning EDoF. "
                      f"Consider using smaller patch sizes or processing smaller regions.")
        logger.warning(f"Current image size: {Z_orig}×{H_orig}×{W_orig}")

    # Estimate patch memory usage
    current_patch_size = patch_size or max(H_orig, W_orig) // 8
    current_stride = stride or current_patch_size // 2
    num_patches_h = (H_orig - current_patch_size) // current_stride + 1
    num_patches_w = (W_orig - current_patch_size) // current_stride + 1
    total_patches = num_patches_h * num_patches_w

    if total_patches > 1000:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️  Large number of patches ({total_patches:,}) may cause high memory usage. "
                      f"Consider increasing stride or reducing patch size.")

    current_patch_size = patch_size
    if current_patch_size is None:
        current_patch_size = max(H_orig, W_orig) // 8

    current_patch_size = max(current_patch_size, 16) # Min patch size
    if current_patch_size % 2 != 0: # Ensure even for CNN
        current_patch_size +=1
    current_patch_size = min(current_patch_size, H_orig, W_orig)


    current_stride = stride
    if current_stride is None:
        current_stride = current_patch_size // 2
    if current_stride <=0: current_stride = 1


    current_model_depth_config = model_depth
    if current_model_depth_config is None:
        current_model_depth_config = 3 if H_orig < 1024 else 5

    if normalize:
        stack_f32 = image_stack.float() / 65535.0
    else:
        stack_f32 = image_stack.float()

    if denoise:
        stack_to_blur = stack_f32.unsqueeze(1)
        blurred_stack = F.gaussian_blur(stack_to_blur, kernel_size=(3,3), sigma=(0.5,0.5))
        stack_f32 = blurred_stack.squeeze(1)

    patches = extract_patches_2d_from_3d_stack(stack_f32, current_patch_size, current_stride)

    fused_patch_outputs = []
    num_epochs_per_patch = 10

    for i in range(patches.shape[0]):
        patch_stack_z = patches[i]
        model_input = patch_stack_z.unsqueeze(0).to(device)
        Z_patch = model_input.shape[1]

        model = UNetLite(in_channels_z=Z_patch, model_config_depth=current_model_depth_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs_per_patch):
            model.train()
            optimizer.zero_grad()
            fused_output_patch = model(model_input)
            loss_s = sharpness_loss_fn(fused_output_patch)
            loss_c = consistency_loss_fn(fused_output_patch, model_input)
            total_loss = loss_s + loss_c
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_fused_patch = model(model_input)
        fused_patch_outputs.append(final_fused_patch.detach().squeeze(0))

    fused_2d_normalized = blend_patches_to_2d_image(
        fused_patch_outputs, H_orig, W_orig, current_patch_size, current_stride, device
    )

    fused_uint16 = fused_2d_normalized.clamp(0, 1).mul(65535.0).to(original_dtype)
    return fused_uint16
