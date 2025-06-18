from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, List, Tuple, Union

from openhcs.utils.import_utils import optional_import, create_placeholder_class
from openhcs.core.memory.decorators import torch as torch_func # Changed from numpy_func

# --- Backend Imports as optional dependencies ---
# PyTorch
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None
HAS_TORCH = torch is not None

# CuPy
if TYPE_CHECKING:
    import cupy as cp

cp = optional_import("cupy")
HAS_CUPY = cp is not None

# JAX
if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

jax = optional_import("jax")
jnp = optional_import("jax.numpy") if jax is not None else None
HAS_JAX = jax is not None

# TensorFlow
if TYPE_CHECKING:
    import tensorflow as tf

tf = optional_import("tensorflow")
HAS_TENSORFLOW = tf is not None

logger = logging.getLogger(__name__)

# --- PyTorch Specific Helpers ---
# Create placeholder for nn.Module
# If nn (and thus nn.Module) is available, ModulePlaceholder will be nn.Module.
# Otherwise, ModulePlaceholder will be a placeholder class.
ModulePlaceholder = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)

if HAS_TORCH: # Keep nn.Module definition conditional on PyTorch availability
    class _RegistrationCNN_torch(ModulePlaceholder): # Inherit from placeholder or actual nn.Module
        def __init__(self):
            super().__init__() # This will call nn.Module.__init__ or Placeholder.__init__
            self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor": # x shape: [B, 2, H, W]
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.tanh(self.conv4(x)) # Output in [-1, 1] range for displacement
            return x

    def _rasterize_polygons_slice_torch(
        polygons_gpu: List["torch.Tensor"], H: int, W: int, device: "torch.device"
    ) -> "torch.Tensor":
        """
        Simplified GPU rasterization for a single slice using point-in-polygon.
        WARNING: This implementation is basic and slow for many/complex polygons.
        A production system would use optimized rasterization libraries or custom kernels.
        """
        mask_slice = torch.zeros((H, W), dtype=torch.bool, device=device)

        for poly_tensor in polygons_gpu: # poly_tensor shape [N_points, 2] (x, y)
            if poly_tensor.shape[0] < 3: continue

            min_xy = torch.min(poly_tensor, dim=0)[0]
            max_xy = torch.max(poly_tensor, dim=0)[0]
            min_x, min_y = torch.floor(min_xy).long()
            max_x, max_y = torch.ceil(max_xy).long()

            min_x = torch.clamp(min_x, 0, W - 1)
            max_x = torch.clamp(max_x, 0, W - 1)
            min_y = torch.clamp(min_y, 0, H - 1)
            max_y = torch.clamp(max_y, 0, H - 1)

            if max_x < min_x or max_y < min_y: continue

            # Create grid for the bounding box
            bb_H, bb_W = max_y - min_y + 1, max_x - min_x + 1
            yy_bb, xx_bb = torch.meshgrid(
                torch.arange(min_y, max_y + 1, device=device),
                torch.arange(min_x, max_x + 1, device=device),
                indexing='ij'
            ) # yy_bb, xx_bb shapes [bb_H, bb_W]

            # Points to test within bounding box, shape [bb_H, bb_W, 2]
            test_points = torch.stack((xx_bb.float(), yy_bb.float()), dim=-1)

            # Ray casting algorithm (vectorized attempt for one polygon)
            num_poly_pts = poly_tensor.shape[0]
            poly_x = poly_tensor[:, 0]
            poly_y = poly_tensor[:, 1]

            # Replicate points for comparison with all edges
            # test_points: [bb_H, bb_W, 2] -> [bb_H, bb_W, 1, 2]
            # poly_x/y: [num_poly_pts]
            # j_indices: [num_poly_pts] (0, 1, ..., N-1)
            # k_indices: [num_poly_pts] (N-1, 0, ..., N-2) (previous vertex)
            j_indices = torch.arange(num_poly_pts, device=device)
            k_indices = (j_indices - 1 + num_poly_pts) % num_poly_pts

            # Edges: (poly_x[j], poly_y[j]) to (poly_x[k], poly_y[k])
            # Test point: (test_x, test_y) from test_points
            test_x = test_points[..., 0].unsqueeze(-1) # [bb_H, bb_W, 1]
            test_y = test_points[..., 1].unsqueeze(-1) # [bb_H, bb_W, 1]

            # Compare test_y with y-coordinates of polygon vertices
            # Shape of poly_y[j_indices] is [num_poly_pts]
            # Need to broadcast: test_y [bb_H, bb_W, 1], poly_y[j_indices] [1, 1, num_poly_pts]
            cond1 = (poly_y[j_indices] <= test_y) & (test_y < poly_y[k_indices]) # Upward edge
            cond2 = (poly_y[k_indices] <= test_y) & (test_y < poly_y[j_indices]) # Downward edge

            # Intersection x-coordinate: test_x < (poly_x[k] - poly_x[j]) * (test_y - poly_y[j]) / (poly_y[k] - poly_y[j]) + poly_x[j]
            # Avoid division by zero if poly_y[k] == poly_y[j] (horizontal edge)
            # delta_y = poly_y[k_indices] - poly_y[j_indices]
            # delta_x = poly_x[k_indices] - poly_x[j_indices]

            # Simplified: this vectorized PIP is complex. Using iterative for clarity here.
            # The iterative version from sandbox was more direct to write under constraints.
            # Reverting to iterative for bounding box for now.
            current_poly_mask_bb = torch.zeros((bb_H, bb_W), dtype=torch.bool, device=device)
            for r_idx in range(bb_H):
                for c_idx in range(bb_W):
                    abs_y, abs_x = min_y + r_idx, min_x + c_idx
                    intersections = 0
                    p1x, p1y = poly_tensor[-1, 0], poly_tensor[-1, 1]
                    for i in range(num_poly_pts):
                        p2x, p2y = poly_tensor[i, 0], poly_tensor[i, 1]
                        if abs_y > min(p1y, p2y) and abs_y <= max(p1y, p2y) and abs_x <= max(p1x, p2x):
                            if p1y != p2y: # Non-horizontal edge
                                xinters = (abs_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or abs_x <= xinters: # Point is to the left of edge
                                    intersections += 1
                        p1x, p1y = p2x, p2y
                    if intersections % 2 == 1:
                        current_poly_mask_bb[r_idx, c_idx] = True

            mask_slice[min_y:max_y+1, min_x:max_x+1] = mask_slice[min_y:max_y+1, min_x:max_x+1] | current_poly_mask_bb
        return mask_slice

    def _apply_displacement_field_torch(data_slice: "torch.Tensor", displacement_field: "torch.Tensor") -> "torch.Tensor":
        H_data, W_data = data_slice.shape[-2:]

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_data, device=data_slice.device),
            torch.linspace(-1, 1, W_data, device=data_slice.device),
            indexing='ij'
        )
        identity_grid = torch.stack((grid_x, grid_y), dim=-1) # Shape [H, W, 2] (x,y) for grid_sample

        # displacement_field is [2, H, W] (dx, dy), needs to be [H, W, 2] (dx, dy) then (x+dx, y+dy)
        # The displacement field from CNN is already scaled by tanh to [-1, 1].
        # This range corresponds to displacement from -image_size/2 to +image_size/2.
        # grid_sample expects final coordinates in [-1, 1].
        # So, new_x_norm = old_x_norm + dx_norm
        displaced_grid = identity_grid + displacement_field.permute(1, 2, 0) # [H, W, 2]
        displaced_grid = torch.clamp(displaced_grid, -1, 1) # Ensure grid is within bounds

        data_slice_unsqueezed = data_slice.unsqueeze(0) # [1, C, H, W] or [1, H, W]
        if data_slice_unsqueezed.ndim == 3: # Was [H,W] -> [1,H,W], need [1,1,H,W]
            data_slice_unsqueezed = data_slice_unsqueezed.unsqueeze(1)

        warped_slice = F.grid_sample(
            data_slice_unsqueezed,
            displaced_grid.unsqueeze(0), # [1, H, W, 2]
            mode='bilinear', padding_mode='zeros', align_corners=False # Zeros for outside mask
        )
        return warped_slice.squeeze(0) # [C,H,W] or [H,W]

    def _smooth_field_z_torch(displacement_field_stack: "torch.Tensor", sigma_z: float) -> "torch.Tensor":
        if sigma_z <= 0: return displacement_field_stack
        Z, C, H, W = displacement_field_stack.shape # [Z, 2, H, W]

        kernel_size_z = max(3, int(2 * 2 * sigma_z + 1))
        if kernel_size_z % 2 == 0: kernel_size_z +=1

        coords_z = torch.arange(kernel_size_z, dtype=torch.float32, device=displacement_field_stack.device)
        coords_z -= (kernel_size_z - 1) / 2
        kernel_1d_z = torch.exp(-(coords_z**2) / (2 * sigma_z**2))
        kernel_1d_z /= kernel_1d_z.sum()

        kernel_1d_z_reshaped = kernel_1d_z.view(1, 1, kernel_size_z).repeat(C * H * W, 1, 1) # For grouped conv

        # Reshape for 1D convolution: (N, C_in, L_in) -> (C*H*W, 1, Z)
        field_permuted = displacement_field_stack.permute(1,2,3,0).contiguous() # [C,H,W,Z]
        field_reshaped = field_permuted.view(-1, 1, Z) # [C*H*W, 1, Z]

        padding_z = kernel_size_z // 2
        smoothed_reshaped = F.conv1d(field_reshaped, kernel_1d_z_reshaped, padding=padding_z, groups=C*H*W)

        smoothed_permuted = smoothed_reshaped.view(C,H,W,Z)
        smoothed_stack = smoothed_permuted.permute(3,0,1,2).contiguous() # [Z,C,H,W]
        return smoothed_stack

# --- Main Pipeline Function ---
@torch_func # Decorate with torch_func
def dxf_mask_pipeline(
    image_stack, # Expected to be a torch.Tensor if torch_func is used
    dxf_polygons: List[List[Tuple[float, float]]],
    apply_mask: bool = False,
    masking_mode: str = "zero_out",
    smoothing_sigma_z: float = 0.0,
    **kwargs
) -> Union["torch.Tensor", "cp.ndarray", "jnp.ndarray", "tf.Tensor"]: # type: ignore

    # Assuming image_stack is (Z, H, W) or (Z, C, H, W)
    # If (Z,C,H,W), C is usually 1 for grayscale, or we take the first channel.
    if image_stack.ndim == 4: # Z, C, H, W
        Z, C_img, H, W = image_stack.shape
        if C_img > 1: logger.warning("Multi-channel image stack provided, using first channel for registration.")
        image_stack_reg = image_stack[:, 0, :, :] # Use first channel for registration: (Z, H, W)
    elif image_stack.ndim == 3: # Z, H, W
        Z, H, W = image_stack.shape
        image_stack_reg = image_stack
    else:
        raise ValueError(f"image_stack has unsupported ndim: {image_stack.ndim}. Expected 3 or 4.")

        device = image_stack.device # image_stack is now expected to be a torch.Tensor
        polygons_gpu = [torch.tensor(p, dtype=torch.float32, device=device) for p in dxf_polygons]

        initial_rasterized_masks_float = torch.zeros((Z, H, W), device=device, dtype=torch.float32)
        displacement_field_slices = []

        registration_cnn = _RegistrationCNN_torch().to(device)
        registration_cnn.eval()

        for z_idx in range(Z):
            image_slice_gray = image_stack_reg[z_idx] # Shape [H, W]

            img_min, img_max = torch.min(image_slice_gray), torch.max(image_slice_gray)
            image_slice_norm = (image_slice_gray - img_min) / (img_max - img_min + 1e-6) if img_max > img_min else torch.zeros_like(image_slice_gray)

            raster_slice = _rasterize_polygons_slice_torch(polygons_gpu, H, W, device).float()
            initial_rasterized_masks_float[z_idx] = raster_slice

            cnn_input = torch.stack([image_slice_norm, raster_slice], dim=0).unsqueeze(0) # [1, 2, H, W]
            with torch.no_grad():
                displacement_field_slice = registration_cnn(cnn_input).squeeze(0) # [2, H, W]
            displacement_field_slices.append(displacement_field_slice)

        displacement_field_stack = torch.stack(displacement_field_slices, dim=0) # [Z, 2, H, W]

        if smoothing_sigma_z > 0:
            displacement_field_stack = _smooth_field_z_torch(displacement_field_stack, smoothing_sigma_z)

        aligned_mask_slices_list = []
        for z_idx in range(Z):
            aligned_slice = _apply_displacement_field_torch(
                initial_rasterized_masks_float[z_idx],
                displacement_field_stack[z_idx]
            ) # Output can be [1,H,W] or [H,W]
            if aligned_slice.ndim == 3 and aligned_slice.shape[0] == 1:
                 aligned_slice = aligned_slice.squeeze(0) # to [H,W]
            aligned_mask_slices_list.append(aligned_slice > 0.5) # Binarize

        aligned_mask_stack_bool = torch.stack(aligned_mask_slices_list, dim=0) # [Z, H, W] bool

        if apply_mask:
            original_dtype = image_stack.dtype
            # Prepare mask for broadcasting if image_stack is (Z,C,H,W)
            mask_to_apply = aligned_mask_stack_bool.float()
            if image_stack.ndim == 4: # Z,C,H,W
                mask_to_apply = mask_to_apply.unsqueeze(1) # -> (Z,1,H,W)

            if masking_mode == "zero_out" or masking_mode == "multiply":
                masked_img = image_stack.float() * mask_to_apply
                return masked_img.to(original_dtype)
            elif masking_mode == "nan_out":
                masked_img_float = image_stack.float()
                nans = torch.full_like(masked_img_float, float('nan'))
                return torch.where(mask_to_apply.bool(), masked_img_float, nans) # Nan where mask is False
            else:
                raise ValueError(f"Unknown masking_mode: {masking_mode}")
        else:
            return aligned_mask_stack_bool
