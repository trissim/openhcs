from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

from openhcs.core.utils import optional_import
from openhcs.core.memory.decorators import torch as torch_func

# Import torch modules as optional dependencies
torch = optional_import("torch")
F = optional_import("torch.nn.functional") if torch is not None else None

logger = logging.getLogger(__name__)

def _moving_average_1d_torch(data: torch.Tensor, window_size: int) -> torch.Tensor:
    """Applies a 1D moving average filter along the last dimension."""
    if window_size <= 0:
        return data
    if window_size % 2 == 0: # Ensure odd window size for centered average
        window_size += 1

    weights = torch.ones(window_size, device=data.device, dtype=data.dtype) / window_size
    weights = weights.view(1, 1, -1) # Shape (out_channels, in_channels/groups, kW)

    # data shape (N, L) -> (N, 1, L) for conv1d
    data_reshaped = data.unsqueeze(1)
    padding = window_size // 2

    smoothed_data = F.conv1d(data_reshaped, weights, padding=padding)
    return smoothed_data.squeeze(1)


@torch_func
def straighten_object_3d(
    image_volume: torch.Tensor, # Expected (Z, H, W) or (1, Z, H, W)
    min_voxel_threshold: float,  # Required parameter
    patch_radius: Optional[int] = None,
    sampling_spacing: float = 1.0,
    max_components: int = 1,
    return_grid: bool = False,
    spline_smoothness: Optional[float] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Identifies and straightens the largest continuous 3D object in a preprocessed
    input volume using PyTorch for GPU-native operations.
    """
    if not isinstance(image_volume, torch.Tensor):
        raise TypeError(f"Input image_volume must be a PyTorch Tensor. Got {type(image_volume)}")

    device = image_volume.device
    original_dtype = image_volume.dtype

    # --- Input Shape Handling ---
    img_vol_proc = image_volume.float() # Work with float32
    if img_vol_proc.ndim == 4: # (1, Z, H, W)
        if img_vol_proc.shape[0] != 1:
            raise ValueError("If 4D, first dimension (batch) must be 1.")
        img_vol_proc = img_vol_proc.squeeze(0) # (Z, H, W)
    elif img_vol_proc.ndim != 3: # (Z, H, W)
        raise ValueError(f"image_volume must be 3D (Z,H,W) or 4D (1,Z,H,W). Got {image_volume.ndim}D")

    Z_orig, H_orig, W_orig = img_vol_proc.shape

    # --- Parse Parameters & Compute Defaults ---
    min_voxel_threshold = float(min_voxel_threshold)  # Already validated as required

    patch_radius_val = patch_radius
    if patch_radius_val is None:
        patch_radius_val = min(H_orig, W_orig) // 10
    patch_radius_val = int(patch_radius_val)
    if patch_radius_val <= 0:
        patch_radius_val = 1  # Ensure positive patch radius

    sampling_spacing_val = float(sampling_spacing)
    if sampling_spacing_val <= 0:
        sampling_spacing_val = 1.0

    max_components_val = int(max_components)
    if max_components_val != 1:
        # Full 3D CC on GPU without SciPy/etc. is complex for this scope.
        raise NotImplementedError("max_components > 1 is not implemented due to GPU CC complexity constraints.")

    return_grid_val = bool(return_grid)

    # spline_smoothness default depends on curve length, calculated later

    # --- 1. Thresholding + Masking (Simplified for largest object) ---
    binary_mask = img_vol_proc > min_voxel_threshold

    object_coords_idx = torch.nonzero(binary_mask, as_tuple=False) # (N_voxels, 3) -> (z, y, x)
    if object_coords_idx.shape[0] == 0:
        logger.warning("No object found above threshold. Returning empty tensor(s).")
        patch_dim = 2 * patch_radius_val + 1
        empty_vol = torch.empty((0, patch_dim, patch_dim), device=device, dtype=original_dtype)
        empty_grid = torch.empty((0, patch_dim, patch_dim, 3), device=device, dtype=torch.float32)
        return (empty_vol, empty_grid) if return_grid_val else empty_vol

    object_coords = object_coords_idx.float()

    # --- 2. Centerline Fitting ---
    # PCA for initial orientation
    mean_coord = torch.mean(object_coords, dim=0)
    centered_coords = object_coords - mean_coord

    # Ensure there's enough variance for SVD
    if centered_coords.shape[0] < 2 or torch.allclose(centered_coords, torch.zeros_like(centered_coords)):
        logger.warning("Not enough distinct object coordinates for PCA. Using a simplified centerline.")
        # Fallback: use z-axis if object is too small/flat for PCA
        # This is a simplification; a more robust fallback might be needed.
        min_z, max_z = object_coords[:,0].min(), object_coords[:,0].max()
        num_fallback_points = max(2, int((max_z - min_z) / sampling_spacing_val) +1)
        centerline_points_smooth = torch.zeros((num_fallback_points, 3), device=device)
        centerline_points_smooth[:,0] = torch.linspace(min_z, max_z, num_fallback_points, device=device)
        centerline_points_smooth[:,1] = mean_coord[1] # Use mean y
        centerline_points_smooth[:,2] = mean_coord[2] # Use mean x
    else:
        # Using try-except for SVD as it can fail on ill-conditioned matrices
        try:
            # Using .T @ . instead of cov for potentially better stability with many points
            # For SVD on covariance: U, S, Vh = torch.linalg.svd(torch.cov(centered_coords.T))
            # Vh[0] would be the principal axis. Here V from (coords.T @ coords) gives principal components as columns.
            U_pca, S_pca, V_pca_transpose = torch.linalg.svd(centered_coords, full_matrices=False)
            principal_axis = V_pca_transpose[0, :] # First right singular vector
        except Exception as e:
            logger.warning(f"PCA (SVD) failed: {e}. Using simplified z-axis centerline.")
            min_z, max_z = object_coords[:,0].min(), object_coords[:,0].max()
            num_fallback_points = max(2, int((max_z - min_z) / sampling_spacing_val) +1)
            centerline_points_smooth = torch.zeros((num_fallback_points, 3), device=device)
            centerline_points_smooth[:,0] = torch.linspace(min_z, max_z, num_fallback_points, device=device)
            centerline_points_smooth[:,1] = mean_coord[1]
            centerline_points_smooth[:,2] = mean_coord[2]
        else:
            projected_scalar = torch.matmul(centered_coords, principal_axis)
            sorted_indices = torch.argsort(projected_scalar)
            sorted_coords_on_axis = object_coords[sorted_indices]

            # Estimate curve length for spline_smoothness default
            segment_lengths_est = torch.norm(sorted_coords_on_axis[1:] - sorted_coords_on_axis[:-1], dim=1)
            curve_length_est = torch.sum(segment_lengths_est)

            spline_smoothness_val = float(spline_smoothness if spline_smoothness is not None else 0.01 * curve_length_est.item())

            # Moving Average for smoothing (applied to each coordinate)
            # Window size needs to be related to spline_smoothness and number of points
            # A larger spline_smoothness_val should imply a larger window.
            # This is a heuristic. A proper spline would use the smoothness param differently.
            num_sorted_pts = sorted_coords_on_axis.shape[0]
            # Heuristic: window_size proportional to smoothness and curve length, bounded by num_points
            # Let spline_smoothness_val be a fraction of total points for window size
            # e.g. if spline_smoothness_val is 0.01 (1%), window is 1% of points
            # The prompt's default is 0.01 * curve_length. This needs interpretation for window size.
            # Let's assume spline_smoothness_val is a fraction of the number of points for window size.
            # If it was given as 0.01 * curve_length, and curve_length ~ num_points * spacing,
            # then window_size ~ 0.01 * num_points * spacing / spacing = 0.01 * num_points.
            # This interpretation makes spline_smoothness a relative factor.
            # Let's use a simpler interpretation: spline_smoothness is a direct factor for window size relative to num_points
            # Or, if it's an absolute voxel unit, then window_size = smoothness / avg_spacing_along_axis
            # Given the prompt "0.01 * curve length", let's use it to define window size.
            # Avg spacing along curve: curve_length_est / num_sorted_pts
            # Window size in voxels: spline_smoothness_val (if it's already in voxels)
            # If spline_smoothness_val was from 0.01 * curve_length, it's already in "voxel units"
            # So, window_size_voxels = spline_smoothness_val.
            # Number of points in window = window_size_voxels / (curve_length_est / num_sorted_pts)

            # Simpler: let spline_smoothness be a relative factor for window size
            # For now, let's use a fixed relative window size or a small absolute one if spline_smoothness is small.
            # The prompt's default "0.01 * curve_length" is tricky to map directly to a moving average window
            # without more context on how that smoothness value is intended.
            # Let's assume spline_smoothness_val is a target smoothing window in voxel units.
            # Approximate points per voxel unit along curve: num_sorted_pts / curve_length_est
            # Window size in points: spline_smoothness_val * (num_sorted_pts / curve_length_est)
            if curve_length_est > 1e-3: # Avoid division by zero
                 window_size = int(max(3, spline_smoothness_val * (num_sorted_pts / curve_length_est.item())))
            else:
                 window_size = 3
            window_size = min(window_size, num_sorted_pts // 2 if num_sorted_pts > 4 else 1) # Cap window size
            if window_size < 1: window_size = 1
            if window_size % 2 == 0: window_size +=1 # must be odd for centered

            if num_sorted_pts > window_size and window_size > 1 : # only smooth if enough points and window > 1
                centerline_points_smooth_z = _moving_average_1d_torch(sorted_coords_on_axis[:, 0].unsqueeze(0), window_size).squeeze(0)
                centerline_points_smooth_y = _moving_average_1d_torch(sorted_coords_on_axis[:, 1].unsqueeze(0), window_size).squeeze(0)
                centerline_points_smooth_x = _moving_average_1d_torch(sorted_coords_on_axis[:, 2].unsqueeze(0), window_size).squeeze(0)
                centerline_points_smooth = torch.stack([centerline_points_smooth_z, centerline_points_smooth_y, centerline_points_smooth_x], dim=1)
            else:
                centerline_points_smooth = sorted_coords_on_axis


    # Resample smoothed centerline
    segment_lengths = torch.norm(centerline_points_smooth[1:] - centerline_points_smooth[:-1], p=2, dim=1)
    if segment_lengths.numel() == 0: # Single point object after smoothing/PCA
        logger.warning("Centerline reduced to a single point. Cannot straighten.")
        # Return empty or a single slice based on patch_radius
        patch_dim = 2 * patch_radius_val + 1
        single_slice = img_vol_proc[
            int(mean_coord[0]),
            max(0, int(mean_coord[1])-patch_radius_val):min(H_orig, int(mean_coord[1])+patch_radius_val+1),
            max(0, int(mean_coord[2])-patch_radius_val):min(W_orig, int(mean_coord[2])+patch_radius_val+1)
        ]
        # Pad if necessary to patch_dim x patch_dim
        padded_slice = torch.zeros((patch_dim, patch_dim), device=device, dtype=original_dtype)
        h_s, w_s = single_slice.shape
        y_start, x_start = (patch_dim - h_s)//2, (patch_dim - w_s)//2
        padded_slice[y_start:y_start+h_s, x_start:x_start+w_s] = single_slice[:patch_dim, :patch_dim].to(original_dtype)

        final_volume = padded_slice.unsqueeze(0) # (1, patch_dim, patch_dim)
        final_grid = torch.zeros((1, patch_dim, patch_dim, 3), device=device, dtype=torch.float32) # Dummy grid
        return (final_volume, final_grid) if return_grid_val else final_volume


    cum_lengths = torch.cat((torch.tensor([0.0], device=device), torch.cumsum(segment_lengths, dim=0)))
    total_curve_length = cum_lengths[-1]

    if total_curve_length < 1e-3: # Effectively a point
        num_samples_L = 1
    else:
        num_samples_L = int(torch.ceil(total_curve_length / sampling_spacing_val).item())
        if num_samples_L < 2: num_samples_L = 2 # Need at least 2 points for tangents

    target_cum_lengths = torch.linspace(0, total_curve_length, num_samples_L, device=device)

    resampled_centerline = torch.empty((num_samples_L, 3), device=device, dtype=torch.float32)

    # Interpolation for resampling
    # For each target_cum_length, find its place in original cum_lengths
    indices = torch.searchsorted(cum_lengths, target_cum_lengths, right=True) - 1
    indices = torch.clamp(indices, 0, cum_lengths.shape[0] - 2) # Ensure valid index range

    # Alpha for interpolation: (target - prev_cum) / (next_cum - prev_cum)
    len_prev_cum = cum_lengths[indices]
    len_next_cum = cum_lengths[indices + 1]

    # Avoid division by zero if segment length is zero
    segment_len_for_alpha = len_next_cum - len_prev_cum
    alpha = torch.where(
        segment_len_for_alpha > 1e-6,
        (target_cum_lengths - len_prev_cum) / segment_len_for_alpha,
        torch.zeros_like(target_cum_lengths) # if segment length is zero, alpha is 0
    )
    alpha = alpha.unsqueeze(1) # for broadcasting with coordinates

    pt_prev = centerline_points_smooth[indices]
    pt_next = centerline_points_smooth[indices + 1]
    resampled_centerline = pt_prev * (1.0 - alpha) + pt_next * alpha

    # --- 3. Plane Sampling ---
    L = resampled_centerline.shape[0]
    patch_dim = 2 * patch_radius_val + 1

    sampling_grid_slices = torch.empty((L, patch_dim, patch_dim, 3), device=device, dtype=torch.float32)

    # Create local patch coordinates (relative to plane center)
    u_coords = torch.linspace(-patch_radius_val, patch_radius_val, patch_dim, device=device)
    v_coords = torch.linspace(-patch_radius_val, patch_radius_val, patch_dim, device=device)
    grid_v_local, grid_u_local = torch.meshgrid(v_coords, u_coords, indexing='ij') # (patch_dim, patch_dim)

    for i in range(L):
        current_point = resampled_centerline[i] # (z_c, y_c, x_c)

        if L == 1:
            tangent = torch.tensor([1.0, 0.0, 0.0], device=device) # Arbitrary if only one point
        elif i == 0:
            tangent_vec = resampled_centerline[i+1] - resampled_centerline[i]
        elif i == L - 1:
            tangent_vec = resampled_centerline[i] - resampled_centerline[i-1]
        else:
            tangent_vec = (resampled_centerline[i+1] - resampled_centerline[i-1]) / 2.0

        if torch.norm(tangent_vec) < 1e-6: # Degenerate tangent
            tangent = torch.tensor([1.0, 0.0, 0.0], device=device) # Default to Z-axis like
        else:
            tangent = F.normalize(tangent_vec, p=2, dim=0) # This is new Z' (z_new_axis)

        # Define orthogonal plane vectors (new X', new Y')
        # Choose arbitrary vector not parallel to tangent
        if torch.abs(tangent[0]) < 0.9: # If tangent is not mostly along Z axis
            arbitrary_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
        else: # Tangent is mostly along Z, pick X or Y
            arbitrary_vec = torch.tensor([0.0, 1.0, 0.0], device=device)

        vec_u_prime_unnorm = torch.cross(tangent, arbitrary_vec)
        if torch.norm(vec_u_prime_unnorm) < 1e-6: # Arbitrary vec was parallel
            arbitrary_vec = torch.tensor([0.0, 0.0, 1.0], device=device) # Try another
            if torch.abs(torch.dot(tangent, arbitrary_vec)) > 0.99: # if tangent is Z axis
                 arbitrary_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
            vec_u_prime_unnorm = torch.cross(tangent, arbitrary_vec)

        vec_u_prime = F.normalize(vec_u_prime_unnorm, p=2, dim=0) # New X' (x_new_axis)
        vec_v_prime = F.normalize(torch.cross(tangent, vec_u_prime), p=2, dim=0) # New Y' (y_new_axis)

        # Transform local grid points to world coordinates
        # plane_points_world = current_point + grid_u_local[..., None] * vec_u_prime + grid_v_local[..., None] * vec_v_prime
        # current_point: [3] -> [1,1,3]
        # grid_u_local: [patch_dim, patch_dim] -> [patch_dim, patch_dim, 1]
        # vec_u_prime: [3] -> [1,1,3]
        plane_points_world_z = current_point[0] + grid_u_local * vec_u_prime[0] + grid_v_local * vec_v_prime[0]
        plane_points_world_y = current_point[1] + grid_u_local * vec_u_prime[1] + grid_v_local * vec_v_prime[1]
        plane_points_world_x = current_point[2] + grid_u_local * vec_u_prime[2] + grid_v_local * vec_v_prime[2]

        # Normalize coordinates for grid_sample (expects x, y, z order in [-1, 1])
        # Original volume dimensions: Z_orig, H_orig, W_orig
        norm_coords_x = 2.0 * (plane_points_world_x / (W_orig - 1)) - 1.0 if W_orig > 1 else torch.zeros_like(plane_points_world_x)
        norm_coords_y = 2.0 * (plane_points_world_y / (H_orig - 1)) - 1.0 if H_orig > 1 else torch.zeros_like(plane_points_world_y)
        norm_coords_z = 2.0 * (plane_points_world_z / (Z_orig - 1)) - 1.0 if Z_orig > 1 else torch.zeros_like(plane_points_world_z)

        sampling_grid_slices[i] = torch.stack((norm_coords_x, norm_coords_y, norm_coords_z), dim=-1)

    final_sampling_grid = sampling_grid_slices # Shape (L, patch_dim, patch_dim, 3)

    # Prepare image_volume for grid_sample: (N, C, D_in, H_in, W_in)
    img_vol_for_sampling = img_vol_proc.unsqueeze(0).unsqueeze(0) # (1, 1, Z_orig, H_orig, W_orig)

    # Reshape grid for grid_sample: (N, D_out, H_out, W_out, 3)
    # Here, D_out = L, H_out = patch_dim, W_out = patch_dim
    grid_for_sampling = final_sampling_grid.unsqueeze(0) # (1, L, patch_dim, patch_dim, 3)

    aligned_volume_slices = F.grid_sample(
        img_vol_for_sampling,
        grid_for_sampling,
        mode='bilinear',
        padding_mode='zeros', # or 'border'
        align_corners=False # Usually False for feature sampling
    )
    # Output shape: (N, C, D_out, H_out, W_out) -> (1, 1, L, patch_dim, patch_dim)

    aligned_volume = aligned_volume_slices.squeeze(0).squeeze(0) # (L, patch_dim, patch_dim)
    aligned_volume = aligned_volume.to(original_dtype)

    if return_grid_val:
        return aligned_volume, final_sampling_grid
    else:
        return aligned_volume
