from __future__ import annotations 

from typing import Any, Dict, List, Tuple

from openhcs.core.utils import optional_import
from openhcs.core.memory.decorators import torch as torch_func

# Import torch as an optional dependency
torch = optional_import("torch")

@torch_func
def trace_neurites_rrs_vectorized(
    image: torch.Tensor,
    seed_density: float = 0.01,
    max_path_length: int = 300,
    intensity_threshold: float = 0.2,
    angle_tolerance: float = 0.4,  # Radians for angular change tolerance
    reaction_retries: int = 2,
    trace_radius: float = 1.0  # Step size for trace propagation
) -> Dict[str, List[Tuple[float, ...]]]:
    """
    Vectorized Random-Reaction-Seed (RRS) neurite tracing algorithm on GPU.
    Traces are output as a dictionary where keys are trace IDs and values are
    lists of (x, y, [z]) coordinate tuples.
    """
    device = image.device
    D = image.ndim # Number of spatial dimensions

    if D not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {D}D.")
    if max_path_length <= 0:
        return {}

    # 1. Seed Generation
    num_pixels = image.numel()
    N = int(seed_density * num_pixels)
    if N == 0:
        return {} # No seeds to trace

    # Initial seed positions (scaled to image dimensions)
    # Seeds are [x, y] or [x, y, z] where x is width-dim, y is height-dim, z is depth-dim.
    # PyTorch convention for image shapes: (H, W) for 2D, (D_img, H, W) for 3D.
    # Coordinates for sampling: (x,y) for 2D, (x,y,z) for 3D, where x maps to W, y to H, z to D_img.
    seeds = torch.rand(N, D, device=device, dtype=torch.float32)
    img_dims_tensor = torch.tensor(image.shape, device=device, dtype=torch.float32)

    if D == 2: # image shape (H, W)
        seeds[:, 0] = seeds[:, 0] * (img_dims_tensor[1] - 1) # W (x-coordinate)
        seeds[:, 1] = seeds[:, 1] * (img_dims_tensor[0] - 1) # H (y-coordinate)
    else: # image shape (Z_img, H, W)
        seeds[:, 0] = seeds[:, 0] * (img_dims_tensor[2] - 1) # W (x-coordinate)
        seeds[:, 1] = seeds[:, 1] * (img_dims_tensor[1] - 1) # H (y-coordinate)
        seeds[:, 2] = seeds[:, 2] * (img_dims_tensor[0] - 1) # Z_img (z-coordinate)

    # Buffers
    trace_buffer = torch.zeros(N, max_path_length, D, device=device, dtype=torch.float32)
    mask_buffer = torch.zeros(N, max_path_length, device=device, dtype=torch.bool) # Valid points in traces

    direction_buffer = torch.randn(N, D, device=device, dtype=torch.float32)
    direction_buffer = direction_buffer / (torch.linalg.norm(direction_buffer, dim=1, keepdim=True) + 1e-9) # Avoid div by zero

    active_mask = torch.ones(N, device=device, dtype=torch.bool) # Active traces
    current_path_lengths = torch.ones(N, device=device, dtype=torch.long) # Start with 1 point (seed)
    reaction_counts = torch.zeros(N, device=device, dtype=torch.long)

    current_positions = seeds.clone()
    trace_buffer[:, 0, :] = current_positions
    mask_buffer[:, 0] = True

    # Prepare image for grid_sample: (B, C, [Depth_img,] Height, Width)
    img_for_sample = image.unsqueeze(0).unsqueeze(0).float()

    # Main propagation loop
    for _t_step in range(1, max_path_length): # t_step is the current length being considered (from 1 to max_path_length-1)
        if not active_mask.any():
            break

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        if active_indices.numel() == 0:
            break

        num_active = active_indices.shape[0]

        current_pos_active = current_positions[active_indices]
        current_dirs_active = direction_buffer[active_indices]

        num_candidate_dirs_K = 8
        if D == 2:
            base_angles = torch.atan2(current_dirs_active[:, 1], current_dirs_active[:, 0]) # y then x for atan2
            angular_offsets = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=num_candidate_dirs_K, device=device)
            candidate_angles = base_angles.unsqueeze(1) + angular_offsets.unsqueeze(0)
            cand_dx = torch.cos(candidate_angles)
            cand_dy = torch.sin(candidate_angles)
            candidate_direction_vectors = torch.stack([cand_dx, cand_dy], dim=-1)
        else: # D == 3
            perturbs = torch.randn(num_active, num_candidate_dirs_K, D, device=device) * 0.5
            candidate_direction_vectors = current_dirs_active.unsqueeze(1) + perturbs
            candidate_direction_vectors = candidate_direction_vectors / (torch.linalg.norm(candidate_direction_vectors, dim=-1, keepdim=True) + 1e-9)

        candidate_next_positions = current_pos_active.unsqueeze(1) + candidate_direction_vectors * trace_radius

        # Normalize candidate positions for grid_sample: range [-1, 1]
        # Grid sample expects (x,y,z) where x is W, y is H, z is D_img
        normalized_cand_pos = torch.empty_like(candidate_next_positions)
        if D == 2: # image (H,W), cand_pos (x,y)
            normalized_cand_pos[..., 0] = 2 * candidate_next_positions[..., 0] / (img_dims_tensor[1] - 1) - 1 # X for W
            normalized_cand_pos[..., 1] = 2 * candidate_next_positions[..., 1] / (img_dims_tensor[0] - 1) - 1 # Y for H
        else: # image (Z_img,H,W), cand_pos (x,y,z)
            normalized_cand_pos[..., 0] = 2 * candidate_next_positions[..., 0] / (img_dims_tensor[2] - 1) - 1 # X for W
            normalized_cand_pos[..., 1] = 2 * candidate_next_positions[..., 1] / (img_dims_tensor[1] - 1) - 1 # Y for H
            normalized_cand_pos[..., 2] = 2 * candidate_next_positions[..., 2] / (img_dims_tensor[0] - 1) - 1 # Z for Z_img

        sampled_intensities = torch.nn.functional.grid_sample(
            img_for_sample,
            normalized_cand_pos.view(1, num_active, num_candidate_dirs_K, D), # B, N_traces, N_candidates, D_spatial
            mode='bilinear', padding_mode='zeros', align_corners=True # align_corners=True is often legacy
        ).view(num_active, num_candidate_dirs_K)

        intensity_scores = sampled_intensities
        dot_products_continuity = torch.einsum('nd,nkd->nk', current_dirs_active, candidate_direction_vectors)
        angle_continuity_scores = (dot_products_continuity + 1) / 2
        total_scores = intensity_scores + angle_continuity_scores

        best_score_indices = torch.argmax(total_scores, dim=1)

        chosen_directions = candidate_direction_vectors[torch.arange(num_active), best_score_indices]
        chosen_next_positions = candidate_next_positions[torch.arange(num_active), best_score_indices]
        intensities_at_chosen = sampled_intensities[torch.arange(num_active), best_score_indices]

        term_intensity = intensities_at_chosen < intensity_threshold
        cos_sim_step = torch.einsum('nd,nd->n', chosen_directions, current_dirs_active)
        max_angle_change_cosine = torch.cos(torch.tensor(angle_tolerance, device=device))
        term_angle = cos_sim_step < max_angle_change_cosine
        term_length = current_path_lengths[active_indices] >= (max_path_length - 1)

        terminate_this_step_mask_local = term_intensity | term_angle | term_length

        continuing_mask_local = ~terminate_this_step_mask_local
        continuing_global_indices = active_indices[continuing_mask_local]

        if continuing_global_indices.numel() > 0:
            path_idx_to_write = current_path_lengths[continuing_global_indices] # This is the index for the new point

            trace_buffer[continuing_global_indices, path_idx_to_write] = chosen_next_positions[continuing_mask_local]
            mask_buffer[continuing_global_indices, path_idx_to_write] = True
            current_positions[continuing_global_indices] = chosen_next_positions[continuing_mask_local]
            direction_buffer[continuing_global_indices] = chosen_directions[continuing_mask_local] # Already normalized
            current_path_lengths[continuing_global_indices] += 1

        terminating_mask_local = terminate_this_step_mask_local
        terminating_global_indices = active_indices[terminating_mask_local]

        if terminating_global_indices.numel() > 0:
            active_mask[terminating_global_indices] = False

            for i_global in terminating_global_indices:
                if reaction_counts[i_global] < reaction_retries:
                    reaction_counts[i_global] += 1
                    len_of_terminated_path = current_path_lengths[i_global]

                    if len_of_terminated_path > 0:
                        restart_idx_in_trace = torch.randint(0, len_of_terminated_path.item(), (1,), device=device).item()
                        new_seed_pos_reaction = trace_buffer[i_global, restart_idx_in_trace, :]

                        current_positions[i_global] = new_seed_pos_reaction

                        new_reaction_dir = torch.randn(D, device=device, dtype=torch.float32)
                        new_reaction_dir_norm = torch.linalg.norm(new_reaction_dir)
                        if new_reaction_dir_norm > 1e-9:
                           direction_buffer[i_global] = new_reaction_dir / new_reaction_dir_norm
                        else:
                           direction_buffer[i_global,0]=1.0
                           if D > 1: direction_buffer[i_global,1:]=0.0
                           if D > 2: direction_buffer[i_global,2:]=0.0


                        trace_buffer[i_global, 0, :] = new_seed_pos_reaction
                        if max_path_length > 1:
                            trace_buffer[i_global, 1:, :] = 0.0

                        mask_buffer[i_global, 0] = True
                        if max_path_length > 1:
                            mask_buffer[i_global, 1:] = False

                        current_path_lengths[i_global] = 1
                        active_mask[i_global] = True

    output_traces: Dict[str, List[Tuple[float, ...]]] = {}
    for i in range(N):
        trace_len = current_path_lengths[i].item()
        # Ensure we only consider points marked True by mask_buffer up to trace_len
        valid_points_in_segment = mask_buffer[i, :trace_len]
        actual_points_for_trace = trace_buffer[i, :trace_len][valid_points_in_segment]

        if actual_points_for_trace.shape[0] > 1:
            trace_list = [tuple(coord.item() for coord in point_coords) for point_coords in actual_points_for_trace]
            output_traces[f"trace_{i:03d}"] = trace_list

    return output_traces
