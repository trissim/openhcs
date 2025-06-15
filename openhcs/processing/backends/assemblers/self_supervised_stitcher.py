from __future__ import annotations 

import math
from typing import Any, Dict, List, Optional, Tuple

from openhcs.utils.import_utils import optional_import, create_placeholder_class
from openhcs.core.memory.decorators import torch as torch_backend_func

# Import torch modules as optional dependencies
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None
models = optional_import("torchvision.models") if optional_import("torchvision") is not None else None

nnModule = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)
# --- Helper Modules and Functions (Placeholders or Simplified) ---

class FeatureEncoder(nnModule):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Modify ResNet for grayscale input (1 channel)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Adapt weights from RGB to grayscale for the first layer if possible
            # Simple averaging of RGB weights:
            rgb_weights = resnet.conv1.weight.data
            self.conv1.weight.data = rgb_weights.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # self.layer4 = resnet.layer4 # Often excluded for smaller feature maps in stitching

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x

class HomographyPredictionNet(nnModule):
    def __init__(self, feature_dim: int):
        super().__init__()
        # Placeholder: A simple network to predict 8 parameters for homography (last one is 1)
        # Input would be concatenated features of two tiles
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512), # Assuming features are flattened and concatenated
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8) # For du, dv parameters of homography matrix corners
        )
        # Initialize weights to output near-identity transform initially
        # For du, dv, this means initializing biases to zero.
        # The last layer's weights should also be small.
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.xavier_uniform_(m.weight, gain=0.01)


    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Assuming features1, features2 are [B, C, Hf, Wf]
        # Flatten and concatenate
        batch_size = features1.shape[0]
        flat_features1 = features1.mean(dim=[2,3]) # Global average pooling as a simple feature vector
        flat_features2 = features2.mean(dim=[2,3])

        combined_features = torch.cat((flat_features1, flat_features2), dim=1)
        params_8 = self.fc(combined_features) # [B, 8]

        # Construct 3x3 homography matrix from 8 parameters
        # H = [[h00, h01, h02], [h10, h11, h12], [h20, h21, 1]]
        # params_8 = [h00-1, h01, h02, h10, h11-1, h12, h20, h21] (delta from identity)
        homography = torch.eye(3, device=params_8.device).repeat(batch_size, 1, 1)
        homography[:, 0, 0] += params_8[:, 0]
        homography[:, 0, 1] = params_8[:, 1]
        homography[:, 0, 2] = params_8[:, 2]
        homography[:, 1, 0] = params_8[:, 3]
        homography[:, 1, 1] += params_8[:, 4]
        homography[:, 1, 2] = params_8[:, 5]
        homography[:, 2, 0] = params_8[:, 6]
        homography[:, 2, 1] = params_8[:, 7]
        # homography[:, 2, 2] is already 1
        return homography

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_coeff: float = 5e-3) -> torch.Tensor:
    # Placeholder for Barlow Twins loss
    # z1, z2 are [B, FeatureDim]
    # Normalize features
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

    N, D = z1_norm.shape
    c = (z1_norm.T @ z2_norm) / N  # Cross-correlation matrix

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.masked_fill(torch.eye(D, device=c.device, dtype=torch.bool), 0).pow_(2).sum()
    loss = on_diag + lambda_coeff * off_diag
    return loss

def geometry_consistency_loss(H_ab: torch.Tensor, H_ba: torch.Tensor) -> torch.Tensor:
    # H_ab: transform from b to a, H_ba: transform from a to b
    # H_ab @ H_ba should be close to Identity
    identity = torch.eye(3, device=H_ab.device).unsqueeze(0).repeat(H_ab.shape[0], 1, 1)
    product = H_ab @ H_ba
    loss = F.mse_loss(product, identity)
    return loss

def photometric_loss(tile_warped: torch.Tensor, tile_target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(tile_warped, tile_target)

def warp_tile_homography(tile: torch.Tensor, H: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """ Warps a tile using a homography.
    tile: [1, 1, H_in, W_in]
    H: [1, 3, 3] homography matrix
    output_shape: (H_out, W_out)
    """
    H_out, W_out = output_shape
    # Create grid for sampling
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H_out, device=tile.device),
                                    torch.linspace(-1, 1, W_out, device=tile.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1) # [H_out, W_out, 3]
    grid = grid.view(1, H_out * W_out, 3) # [1, H_out*W_out, 3]

    # We need to warp from output grid to input tile's coordinate system.
    # So we need H_inv. If H transforms points from tile_src to tile_dst,
    # grid_sample needs a grid in tile_src's coordinates.
    # If H maps tile_src to tile_dst, then H_inv maps tile_dst (output) to tile_src (input).
    try:
        H_inv = torch.inverse(H)
    except RuntimeError: # Singular matrix
        H_inv = H # Fallback, or handle error
        print("Warning: Singular homography matrix encountered during inverse.")


    # Transform grid points
    # grid_transformed = grid @ H_inv.transpose(1, 2) # [1, H_out*W_out, 3]
    # grid_transformed = torch.matmul(grid, H_inv.transpose(1,2))
    grid_transformed = torch.bmm(grid, H_inv.transpose(1,2))


    # Normalize to [-1, 1] for grid_sample
    # grid_transformed[:, :, 0] = grid_transformed[:, :, 0] / grid_transformed[:, :, 2] # x' = x/w
    # grid_transformed[:, :, 1] = grid_transformed[:, :, 1] / grid_transformed[:, :, 2] # y' = y/w

    # Avoid division by zero or very small w
    w_coords = grid_transformed[:, :, 2].unsqueeze(2)
    safe_w_coords = torch.where(torch.abs(w_coords) < 1e-6, torch.sign(w_coords) * 1e-6 + 1e-9, w_coords)

    grid_transformed_normalized = grid_transformed[:, :, :2] / safe_w_coords # [1, H_out*W_out, 2]

    # Reshape for grid_sample
    sampling_grid = grid_transformed_normalized.view(1, H_out, W_out, 2) # [B, H_out, W_out, 2] (x,y order)

    warped_tile = F.grid_sample(tile, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_tile


def get_adjacency_from_layout(layout_rows: int, layout_cols: int, num_tiles: int, device: torch.device) -> List[Tuple[int, int]]:
    """
    Generates a list of adjacent tile index pairs based on a grid layout.
    Considers 4-connectivity (up, down, left, right).
    """
    adjacency_pairs = []
    for r in range(layout_rows):
        for c in range(layout_cols):
            current_idx = r * layout_cols + c
            if current_idx >= num_tiles:
                continue

            # Check right neighbor
            if c + 1 < layout_cols:
                right_idx = r * layout_cols + (c + 1)
                if right_idx < num_tiles:
                    adjacency_pairs.append(tuple(sorted((current_idx, right_idx))))

            # Check bottom neighbor
            if r + 1 < layout_rows:
                bottom_idx = (r + 1) * layout_cols + c
                if bottom_idx < num_tiles:
                    adjacency_pairs.append(tuple(sorted((current_idx, bottom_idx))))

    # Remove duplicates that might arise from sorted tuples if order doesn't matter for pairs
    return sorted(list(set(adjacency_pairs)))

def optimize_pose_graph(
    pairwise_homographies: Dict[Tuple[int, int], torch.Tensor],
    num_tiles: int,
    device: torch.device,
    initial_global_transforms: Optional[List[torch.Tensor]] = None
) -> List[torch.Tensor]:
    """
    Placeholder for pose graph optimization.
    Takes pairwise homographies and refines global tile transforms.
    Returns a list of [3,3] global homography matrices for each tile.
    """
    print("Pose Graph Optimization: SKIPPED (using initial/placeholder transforms).")
    # TODO: Implement actual pose graph optimization (e.g., using least squares on log-transforms, or a spring model).
    # For now, if initial_global_transforms are provided, use them, otherwise return identity.
    if initial_global_transforms:
        if len(initial_global_transforms) == num_tiles:
            return initial_global_transforms

    # Fallback to identity if no initial transforms or mismatch
    global_transforms = [torch.eye(3, device=device) for _ in range(num_tiles)]
    if num_tiles > 0: # Anchor the first tile
        global_transforms[0] = torch.eye(3, device=device)

    # Simplistic chaining if pairwise_homographies were available (illustrative, not robust)
    # This part should be replaced by a real graph solver.
    # Example: if H_ij transforms tile j to tile i's frame.
    # for i in range(1, num_tiles):
    #     if (i-1, i) in pairwise_homographies:
    #         H_prev_curr = pairwise_homographies[(i-1,i)] # H that maps tile i to tile i-1 frame
    #         global_transforms[i] = global_transforms[i-1] @ torch.inverse(H_prev_curr)
    #     elif (i, i-1) in pairwise_homographies:
    #         H_curr_prev = pairwise_homographies[(i,i-1)] # H that maps tile i-1 to tile i frame
    #         global_transforms[i] = global_transforms[i-1] @ H_curr_prev


    return global_transforms


# --- Main Stitcher Function ---
@torch_backend_func
def self_supervised_stitcher_func(
    tile_stack: torch.Tensor,  # shape: [Z, Y, X]
    *,
    tile_shape_override: Optional[Tuple[int, int]] = None,   # (tile_height, tile_width)
    layout_shape_override: Optional[Tuple[int, int]] = None,  # (rows, cols)
    learn: bool = False,
    num_train_iterations: int = 100, # Only if learn=True
    overlap_percent: float = 0.1, # For global transform normalization
    return_homographies: bool = False,
    # For pre-trained model paths
    encoder_path: Optional[str] = None,
    homography_net_path: Optional[str] = None
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[int, int]] | Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Self-supervised image stitching module.
    Learns relative alignment of tiles using unsupervised geometry matching.
    Infers pairwise transformations and composes them into global tile offsets (x, y).
    Returns:
        - position tensor [1, Z, 2]
        - (Optionally) global homographies [Z, 3, 3]
        - canvas dimensions (canvas_H, canvas_W)
    """
    device = tile_stack.device
    Z, Y, X = tile_stack.shape

    if Z == 0:
        empty_positions = torch.empty((1, 0, 2), device=device, dtype=torch.float32)
        canvas_dims = (0, 0)
        if return_homographies:
            empty_homographies = torch.empty((0, 3, 3), device=device, dtype=torch.float32)
            return empty_positions, empty_homographies, canvas_dims
        return empty_positions, canvas_dims

    # 1. Input Description & Defaults
    tile_shape: Tuple[int, int] = tile_shape_override if tile_shape_override else (Y, X)
    H_tile, W_tile = tile_shape

    if layout_shape_override:
        layout_rows, layout_cols = layout_shape_override
        if layout_rows * layout_cols < Z:
            raise ValueError(f"Provided layout_shape {layout_shape_override} is too small for {Z} tiles.")
    else:
        # Infer layout_shape to be as square as possible
        layout_rows = math.ceil(math.sqrt(Z))
        layout_cols = math.ceil(Z / layout_rows)
        # Ensure it's at least Z
        while layout_rows * layout_cols < Z:
            layout_cols +=1 # or layout_rows, depending on preference for aspect ratio

    layout_shape: Tuple[int, int] = (layout_rows, layout_cols)

    print(f"Using tile_shape: {tile_shape}, layout_shape: {layout_shape} for {Z} tiles.")

    # 2. Tile Reshaping
    # Reshape [Z, Y, X] -> [Z, 1, H_tile, W_tile] for CNN
    tiles_for_cnn = tile_stack.unsqueeze(1).float() # Add channel dim

    # Create a 2D grid representation of tiles for adjacency
    # Pad tile_stack if Z < layout_rows * layout_cols
    num_layout_slots = layout_rows * layout_cols
    if Z < num_layout_slots:
        padding_count = num_layout_slots - Z
        padding_tensor = torch.zeros(padding_count, Y, X, device=device, dtype=tile_stack.dtype)
        padded_tile_stack = torch.cat((tile_stack, padding_tensor), dim=0)
    else:
        padded_tile_stack = tile_stack

    tile_grid = padded_tile_stack.view(layout_rows, layout_cols, Y, X)

    # 3. Feature Encoder
    feature_encoder = FeatureEncoder().to(device)
    if learn:
        feature_encoder.train()
    else:
        feature_encoder.eval()

    # 4. Unsupervised Alignment (AltO-inspired)
    # This is a highly complex part. Placeholder logic:

    # Store pairwise homographies, e.g., from tile i to tile j
    pairwise_H_matrices: Dict[Tuple[int, int], torch.Tensor] = {}
    # global_transforms will be populated by pose graph optimization or fallback
    global_transforms: List[torch.Tensor] = [torch.eye(3, device=device) for _ in range(Z)]


    if learn:
        print("Starting learning phase for pairwise alignments...")
        dummy_features = feature_encoder(tiles_for_cnn[:1])
        feature_dim_encoder = dummy_features.shape[1]
        homography_net = HomographyPredictionNet(feature_dim_encoder).to(device)
        optimizer = torch.optim.Adam(list(feature_encoder.parameters()) + list(homography_net.parameters()), lr=1e-4)

        # Get adjacent pairs based on layout
        adjacent_tile_pairs = get_adjacency_from_layout(layout_rows, layout_cols, Z, device)

        if Z > 1 and not adjacent_tile_pairs :
            print("Warning: No adjacent pairs from layout for training, using sequential pairs as fallback.")
            for i_rand_pair in range(Z-1): # Create simple chain pairs
                 adjacent_tile_pairs.append(tuple(sorted((i_rand_pair, i_rand_pair+1))))
            adjacent_tile_pairs = sorted(list(set(adjacent_tile_pairs)))

        for iter_idx in range(num_train_iterations):
            optimizer.zero_grad()
            total_loss_iter = torch.tensor(0.0, device=device)

            if not adjacent_tile_pairs or Z < 2:
                print("Not enough tiles or pairs for training iteration.")
                break

            # Create batches from adjacent_tile_pairs
            # For simplicity, process a few pairs per iteration or all if few
            # Ensure num_pairs_batch is at least 1 if adjacent_tile_pairs is not empty
            num_pairs_in_batch = min(len(adjacent_tile_pairs), 8) if adjacent_tile_pairs else 0
            if num_pairs_in_batch == 0:
                print("No pairs to process in this iteration.")
                continue

            # Use randint for memory efficiency (though len(adjacent_tile_pairs) is typically small)
            current_batch_pairs_indices = torch.randint(0, len(adjacent_tile_pairs), (num_pairs_in_batch,))

            batch_idx1_list = []
            batch_idx2_list = []
            for perm_idx in current_batch_pairs_indices:
                p1, p2 = adjacent_tile_pairs[perm_idx.item()]
                batch_idx1_list.append(p1)
                batch_idx2_list.append(p2)

            idx1 = torch.tensor(batch_idx1_list, device=device, dtype=torch.long)
            idx2 = torch.tensor(batch_idx2_list, device=device, dtype=torch.long)

            tiles1_batch = tiles_for_cnn[idx1]
            tiles2_batch = tiles_for_cnn[idx2]

            features1 = feature_encoder(tiles1_batch)
            features2 = feature_encoder(tiles2_batch)

            H_12 = homography_net(features1, features2) # tile2 -> tile1
            H_21 = homography_net(features2, features1) # tile1 -> tile2

            # Store these for later graph optimization
            for i_pair in range(idx1.shape[0]):
                p_idx1, p_idx2 = idx1[i_pair].item(), idx2[i_pair].item()
                pairwise_H_matrices[(p_idx1, p_idx2)] = H_12[i_pair].detach() # H mapping p_idx2 to p_idx1 frame
                pairwise_H_matrices[(p_idx2, p_idx1)] = H_21[i_pair].detach() # H mapping p_idx1 to p_idx2 frame

            loss_bt = barlow_twins_loss(features1.mean(dim=[2,3]), features2.mean(dim=[2,3]))
            loss_geom = geometry_consistency_loss(H_12, H_21)

            # TODO: For photometric loss, consider replacing grid_sample in warp_tile_homography
            # with GPU-efficient tensor tiling (block aggregation) for significant speedup,
            # especially if overlap regions are known or can be estimated.
            tiles2_warped_to_1 = torch.stack([
                warp_tile_homography(tiles2_batch[i].unsqueeze(0), H_12[i].unsqueeze(0), tile_shape)
                for i in range(tiles2_batch.shape[0])]).squeeze(1)
            loss_photo = photometric_loss(tiles2_warped_to_1, tiles1_batch)

            loss_total_batch = loss_bt + loss_geom + loss_photo
            loss_total_batch.backward()
            optimizer.step()
            total_loss_iter += loss_total_batch.item()

            if num_pairs_in_batch > 0 and (iter_idx + 1) % max(1, (num_train_iterations // 10)) == 0 :
                 print(f"Iter {iter_idx+1}/{num_train_iterations}, Loss: {total_loss_iter / num_pairs_in_batch:.4f}")

        print("Learning phase finished.")
        # 5. Graph-Based Global Alignment (after learning all pairwise)
        # Ensure all necessary pairs for graph are computed if not covered by training batches
        with torch.no_grad():
            feature_encoder.eval()
            homography_net.eval()
            all_tile_features_final = feature_encoder(tiles_for_cnn[:Z])

            all_graph_pairs = get_adjacency_from_layout(layout_rows, layout_cols, Z, device)
            # Optionally add more pairs (e.g., random, next-nearest) for graph robustness
            # ... (logic for adding more pairs can be inserted here) ...

            for p1_g, p2_g in all_graph_pairs:
                # Only compute if not already in pairwise_H_matrices from training
                if (p1_g, p2_g) not in pairwise_H_matrices:
                     feat1_g = all_tile_features_final[p1_g].unsqueeze(0)
                     feat2_g = all_tile_features_final[p2_g].unsqueeze(0)
                     H_p1_p2 = homography_net(feat1_g, feat2_g).squeeze(0) # p2 -> p1 frame
                     pairwise_H_matrices[(p1_g, p2_g)] = H_p1_p2
                if (p2_g, p1_g) not in pairwise_H_matrices: # And the reverse
                     feat1_g = all_tile_features_final[p1_g].unsqueeze(0)
                     feat2_g = all_tile_features_final[p2_g].unsqueeze(0)
                     H_p2_p1 = homography_net(feat2_g, feat1_g).squeeze(0) # p1 -> p2 frame
                     pairwise_H_matrices[(p2_g, p1_g)] = H_p2_p1

        # Initial global transforms for optimization (e.g., identity or grid-based estimate)
        initial_transforms_for_opt = [torch.eye(3, device=device) for _ in range(Z)]
        # A simple grid layout can be a better start than pure identity for all
        for i_init in range(Z):
            row_idx_init, col_idx_init = i_init // layout_cols, i_init % layout_cols
            dx_init = col_idx_init * W_tile * (1.0 - overlap_percent)
            dy_init = row_idx_init * H_tile * (1.0 - overlap_percent)
            translate_matrix_init = torch.eye(3, device=device)
            translate_matrix_init[0, 2] = dx_init
            translate_matrix_init[1, 2] = dy_init
            initial_transforms_for_opt[i_init] = translate_matrix_init

        global_transforms = optimize_pose_graph(pairwise_H_matrices, Z, device, initial_transforms_for_opt)

    else: # learn=False
        print("Inference mode: Using placeholder global transforms (grid layout).")
        # TODO: Load pre-trained feature_encoder and homography_net
        # For now, use a simple grid layout for global transforms
        for i in range(Z):
            row_idx = i // layout_cols
            col_idx = i % layout_cols
            dx = col_idx * W_tile * (1.0 - overlap_percent) # Assume some overlap
            dy = row_idx * H_tile * (1.0 - overlap_percent)

            translate_matrix = torch.eye(3, device=device)
            translate_matrix[0, 2] = dx
            translate_matrix[1, 2] = dy
            global_transforms[i] = translate_matrix

    # 5. Graph-Based Global Alignment
    # TODO: Implement a proper graph solver (e.g., spring model or least-squares on H_matrices)
    # This step would refine `global_transforms`. The current `global_transforms` are placeholders.
    print("Graph-Based Global Alignment: SKIPPED (using placeholder transforms).")


    # 6. Finalize Global Transforms and Extract Positions
    # The global_transforms list currently holds [3,3] homography matrices for each tile,
    # mapping its local coordinates to a common global frame.
    # We need to ensure the coordinate system's origin (0,0) is sensible,
    # e.g., the top-leftmost point of the stitched layout.

    # Transform corners of all tiles to the current global frame to find bounds
    all_corners_global_frame = []
    # Define corners of a tile in its local coordinate system (homogeneous)
    # (0,0), (W-1,0), (0,H-1), (W-1,H-1)
    tile_local_corners_homog = torch.tensor([
        [0, W_tile - 1, 0         , W_tile - 1], # x-coordinates
        [0, 0         , H_tile - 1, H_tile - 1], # y-coordinates
        [1, 1         , 1         , 1         ]  # w-coordinates
    ], dtype=torch.float32, device=device) # Shape: [3, 4]

    for i in range(Z):
        H_global_i = global_transforms[i] # Shape [3, 3]
        # Transform local corners to global frame: H_global @ local_corners
        corners_transformed_homog_i = H_global_i @ tile_local_corners_homog # Shape [3, 4]

        # Perspective divide (x/w, y/w)
        w_coords_i = corners_transformed_homog_i[2, :]
        # Avoid division by zero for w: if w is close to 0, it's problematic.
        # For affine transforms (like translation used in placeholder), w is always 1.
        # For full homographies, w can vary.
        safe_w_coords_i = torch.where(torch.abs(w_coords_i) < 1e-6, torch.sign(w_coords_i) * 1e-6 + 1e-9, w_coords_i)

        corners_global_frame_i = corners_transformed_homog_i[:2, :] / safe_w_coords_i # Shape [2, 4] (x,y)
        all_corners_global_frame.append(corners_global_frame_i)

    all_corners_stacked = torch.cat(all_corners_global_frame, dim=1) # Shape [2, Z*4]

    # Find min x and min y to define the top-left of the bounding box of all tiles
    min_global_coords = torch.min(all_corners_stacked, dim=1).values # Shape [2] (min_x, min_y)

    # Create an offset matrix to shift the entire layout so that min_global_coords becomes (0,0)
    offset_x_to_origin = -min_global_coords[0]
    offset_y_to_origin = -min_global_coords[1]

    normalization_offset_matrix = torch.eye(3, device=device)
    normalization_offset_matrix[0, 2] = offset_x_to_origin
    normalization_offset_matrix[1, 2] = offset_y_to_origin

    # Apply this normalization to all global transforms
    final_global_transforms_list = []
    for i in range(Z):
        final_H_i = normalization_offset_matrix @ global_transforms[i]
        final_global_transforms_list.append(final_H_i)

    # Extract (x, y) positions (top-left corner of each tile) from the final homographies
    # The translation part of H (H[0,2], H[1,2]) gives the position of the tile's origin (0,0)
    # in the global (canvas) frame.
    tile_positions_xy = torch.stack(
        [H[0:2, 2] for H in final_global_transforms_list], dim=0
    ) # Shape [Z, 2]

    # Reshape to [1, Z, 2] as per output spec
    output_positions = tile_positions_xy.unsqueeze(0)

    if return_homographies:
        # Stack the list of [3,3] homography tensors into a single [Z, 3, 3] tensor
        output_homographies = torch.stack(final_global_transforms_list, dim=0)
        return output_positions, output_homographies
    else:
        return output_positions


if __name__ == '__main__':
    # Example Usage (for testing within this file)
    print("Running self_supervised_stitcher_func example...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic jittered tiles
    Z_tiles = 4
    tile_H, tile_W = 64, 64

    # Create a base image to extract tiles from
    base_img_H, base_img_W = tile_H * 2 + 20, tile_W * 2 + 20 # Larger base to allow jitter
    base_image = torch.zeros(base_img_H, base_img_W, device=device)
    # Add some features to the base image (e.g., a cross)
    base_image[base_img_H//2 - 10 : base_img_H//2 + 10, :] = 0.7
    base_image[:, base_img_W//2 - 10 : base_img_W//2 + 10] = 0.7
    base_image[base_img_H//4 : 3*base_img_H//4, base_img_W//4 : 3*base_img_W//4] += 0.3
    base_image = torch.clamp(base_image, 0, 1)

    synthetic_tiles_list = []
    overlap = 10 # pixels

    # Expected layout: 2x2
    # Tile 0: top-left, Tile 1: top-right, Tile 2: bottom-left, Tile 3: bottom-right
    # Define ideal top-left corners for each tile in the base image
    ideal_starts = [
        (5, 5),
        (5, 5 + tile_W - overlap),
        (5 + tile_H - overlap, 5),
        (5 + tile_H - overlap, 5 + tile_W - overlap)
    ]

    for i in range(Z_tiles):
        start_y, start_x = ideal_starts[i]

        # Add some random jitter to actual extraction
        jitter_y = torch.randint(-3, 4, (1,)).item()
        jitter_x = torch.randint(-3, 4, (1,)).item()

        current_start_y = start_y + jitter_y
        current_start_x = start_x + jitter_x

        tile = base_image[current_start_y : current_start_y + tile_H,
                          current_start_x : current_start_x + tile_W].clone()

        synthetic_tiles_list.append(tile)

    synthetic_tile_stack = torch.stack(synthetic_tiles_list).to(device)
    print(f"Synthetic tile stack shape: {synthetic_tile_stack.shape}")

    # Test with learn=True (will be slow and likely not converge well with few iterations)
    print("\nTesting with learn=True...")
    tile_positions_learn, homographies_learn = self_supervised_stitcher(
        synthetic_tile_stack.clone(),
        learn=True,
        num_train_iterations=10, # Very few iterations for a quick test
        return_homographies=True,
        layout_shape_override=(2,2) # Explicit layout for test
    )
    print(f"Tile positions (learn=True) shape: {tile_positions_learn.shape}")
    print(f"Tile positions (learn=True, tile 0): {tile_positions_learn[0,0,:]}")
    print(f"Homographies (learn=True) shape: {homographies_learn.shape}")
    # print(f"Homography (learn=True, tile 0):\n{homographies_learn[0]}")


    # Test with learn=False (uses placeholder grid transforms)
    print("\nTesting with learn=False...")
    tile_positions_infer = self_supervised_stitcher(
        synthetic_tile_stack.clone(),
        learn=False,
        return_homographies=False, # Test this path too
        layout_shape_override=(2,2)
    )
    print(f"Tile positions (learn=False) shape: {tile_positions_infer.shape}")
    print(f"Tile positions (learn=False, tile 0): {tile_positions_infer[0,0,:]}")


    # Try to visualize the layout if possible
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        def plot_layout(ax, tile_stack_cpu, positions_cpu, title, H_tile, W_tile, homographies_cpu=None):
            ax.clear()
            ax.set_title(title)
            ax.set_aspect('equal', 'box')

            all_x = []
            all_y = []

            for i in range(positions_cpu.shape[0]):
                x, y = positions_cpu[i, 0], positions_cpu[i, 1]
                all_x.extend([x, x + W_tile]) # Approximate bounding box
                all_y.extend([y, y + H_tile])

                # Display tile image at its position (approximate, as homography might skew)
                # For simplicity, just place the tile image's top-left at (x,y)
                # A more accurate plot would use the homography to warp the tile outline

                ax.imshow(tile_stack_cpu[i], cmap='gray', alpha=0.7,
                          extent=(x, x + W_tile, y + H_tile, y)) # extent is (left, right, bottom, top)

                # Draw a rectangle border for the tile
                rect = Rectangle((x, y), W_tile, H_tile, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, f"{i}", color='cyan', fontsize=8)

            if all_x and all_y:
                ax.set_xlim(min(all_x) - W_tile*0.1, max(all_x) + W_tile*0.1)
                ax.set_ylim(max(all_y) + H_tile*0.1, min(all_y) - H_tile*0.1) # Flipped for imshow
            else: # Default if no positions
                ax.set_xlim(0, W_tile * 2)
                ax.set_ylim(H_tile * 2, 0)


        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        plot_layout(axes[0],
                    synthetic_tile_stack.cpu().numpy(),
                    tile_positions_learn.squeeze(0).cpu().detach().numpy(),
                    "Layout (learn=True, 10 iter)",
                    tile_H, tile_W,
                    homographies_learn.cpu().detach().numpy() if homographies_learn is not None else None)

        plot_layout(axes[1],
                    synthetic_tile_stack.cpu().numpy(),
                    tile_positions_infer.squeeze(0).cpu().detach().numpy(),
                    "Layout (learn=False, Grid)",
                    tile_H, tile_W)

        plt.tight_layout()
        plt.savefig("self_supervised_stitcher_layout_test_output.png")
        print("\nSaved test layout plot to self_supervised_stitcher_layout_test_output.png")

    except ImportError:
        print("\nMatplotlib not available. Skipping layout visualization.")
    except Exception as e:
        print(f"\nError during layout visualization: {e}")

    print("Self_supervised_stitcher example finished.")
