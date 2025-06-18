from __future__ import annotations 

import logging
from typing import Tuple, Union

from openhcs.utils.import_utils import optional_import, create_placeholder_class
from openhcs.core.memory.decorators import torch as torch_func

# Import torch modules as optional dependencies
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None

nnModule = create_placeholder_class(
    "Module", # Name for the placeholder if generated
    base_class=nn.Module if nn else None,
    required_library="PyTorch"
)
logger = logging.getLogger(__name__)

# --- PyTorch Models and Helper Functions ---

class Encoder3D(nnModule):
    def __init__(self, in_channels=1, features=(32, 64), embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(features[0], features[1], kernel_size=3, padding=1)
        # Stride in conv layers can reduce spatial dimensions before pooling
        # For simplicity here, keeping full spatial resolution until pooling
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(features[1], embedding_dim)
        self.features_channels_before_pool = features[1] # To know feature depth for K-Means

    def forward(self, x: torch.Tensor, return_features_before_pool: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, 1, D, H, W]
        features_conv1 = F.relu(self.conv1(x))
        features_conv2 = F.relu(self.conv2(features_conv1)) # [B, features[1], D, H, W]

        pooled_features = self.pool(features_conv2) # [B, features[1], 1, 1, 1]
        flattened_features = torch.flatten(pooled_features, 1) # [B, features[1]]
        embedding = self.fc(flattened_features) # [B, embedding_dim]
        normalized_embedding = F.normalize(embedding, p=2, dim=1)

        if return_features_before_pool:
            return normalized_embedding, features_conv2 # Return features from last conv layer
        return normalized_embedding

class Decoder3D(nnModule):
    def __init__(self, embedding_dim=128, features=(64, 32), out_channels=1, patch_size_dhw=(64,64,64)):
        super().__init__()
        self.patch_d, self.patch_h, self.patch_w = patch_size_dhw

        # Determine initial dimensions for unflattening.
        # This depends on the encoder's spatial reduction. If encoder doesn't reduce much,
        # these initial dimensions might need to be larger or upsampling more aggressive.
        # Assuming encoder's convs are padding=1, kernel=3, stride=1, spatial dim is preserved.
        # If encoder had pooling/strides, this would need to match the smallest feature map size.
        # For simplicity, let's assume the decoder reconstructs to the patch size directly
        # from a smaller latent representation.
        # A common strategy is to project embedding to features * D/s * H/s * W/s where s is total downsample factor.
        # Here, let's make it simple: project to features[0] * small_d * small_h * small_w
        self.init_d, self.init_h, self.init_w = patch_size_dhw[0] // 4, patch_size_dhw[1] // 4, patch_size_dhw[2] // 4
        if self.init_d < 1: self.init_d = 1
        if self.init_h < 1: self.init_h = 1
        if self.init_w < 1: self.init_w = 1

        self.fc = nn.Linear(embedding_dim, features[0] * self.init_d * self.init_h * self.init_w)
        self.unflatten_channels = features[0]

        # Upsample to patch_size/2
        self.upconv1 = nn.ConvTranspose3d(features[0], features[1], kernel_size=4, stride=2, padding=1)
        # Upsample to patch_size
        self.upconv2 = nn.ConvTranspose3d(features[1], out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: [B, embedding_dim]
        x = self.fc(x)
        x = x.view(-1, self.unflatten_channels, self.init_d, self.init_h, self.init_w)
        x = F.relu(self.upconv1(x))
        x = self.upconv2(x) # Output reconstruction
        return x

def _extract_random_patches(
    volume: torch.Tensor, patch_size_dhw: Tuple[int,int,int], num_patches: int
) -> torch.Tensor: # volume: [1, 1, D, H, W] -> output: [num_patches, 1, pD, pH, pW]
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size_dhw
    patches = torch.empty((num_patches, 1, pD, pH, pW), device=volume.device, dtype=volume.dtype)
    for i in range(num_patches):
        d_start = torch.randint(0, D - pD + 1, (1,)).item() if D > pD else 0
        h_start = torch.randint(0, H - pH + 1, (1,)).item() if H > pH else 0
        w_start = torch.randint(0, W - pW + 1, (1,)).item() if W > pW else 0
        patches[i] = volume[0, :, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
    return patches

def _affine_augment_patch(patch: torch.Tensor) -> torch.Tensor: # patch: [1, pD, pH, pW]
    # Random flips
    if torch.rand(1).item() > 0.5: patch = torch.flip(patch, dims=[1]) # D
    if torch.rand(1).item() > 0.5: patch = torch.flip(patch, dims=[2]) # H
    if torch.rand(1).item() > 0.5: patch = torch.flip(patch, dims=[3]) # W

    # Random 90-degree rotations (example: rotate in DH plane)
    if torch.rand(1).item() > 0.5:
        k = torch.randint(0, 4, (1,)).item()
        patch = torch.rot90(patch, k, dims=[1, 2]) # Rotate D-H plane
    return patch

def _nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z_i.shape[0]
    # z_i, z_j are already normalized by encoder

    # Concatenate embeddings from two views
    z = torch.cat([z_i, z_j], dim=0) # Shape: [2*B, E]

    # Calculate similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature # Shape: [2*B, 2*B]

    # Create mask to identify positive pairs (i-th sample from view 1 with i-th sample from view 2)
    # And exclude self-similarity (i-th sample with itself)
    identity_mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)

    # Positive pairs are (i, i+B) and (i+B, i)
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
    pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True

    # Numerator: similarity of positive pairs
    numerator = torch.exp(sim_matrix[pos_mask]) # Shape: [2*B] (actually B pairs, repeated)

    # Denominator: sum of similarities with all other samples (excluding self)
    # For each row, sum exp(sim) over all columns except the diagonal (self-similarity)
    exp_sim_no_self = torch.exp(sim_matrix.masked_fill(identity_mask, -float('inf'))) # Mask self-similarity
    denominator = exp_sim_no_self.sum(dim=1) # Shape: [2*B]

    # Calculate log probabilities
    log_probs = torch.log(numerator / denominator[pos_mask.any(dim=1)]) # Select relevant denominators

    # Loss is the negative mean of these log probabilities
    loss = -log_probs.mean()
    return loss

def _kmeans_torch(X: torch.Tensor, K: int, n_iters: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    N, D_feat = X.shape
    if N == 0: return torch.empty(0, dtype=torch.long, device=X.device), torch.empty((K,D_feat), device=X.device, dtype=X.dtype)
    if N < K : K = N

    # Use randint for memory efficiency (though N is typically small for K-means)
    centroids = X[torch.randint(0, N, (K,), device=X.device)]

    for _ in range(n_iters):
        dists_sq = torch.sum((X[:, None, :] - centroids[None, :, :])**2, dim=2)
        labels = torch.argmin(dists_sq, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for k_idx in range(K):
            assigned_points = X[labels == k_idx]
            if assigned_points.shape[0] > 0:
                new_centroids[k_idx] = assigned_points.mean(dim=0)
            else:
                new_centroids[k_idx] = X[torch.randint(0,N,(1,)).item()] if N > 0 else centroids[k_idx]

        if torch.allclose(centroids, new_centroids, atol=1e-5): break
        centroids = new_centroids
    return labels, centroids

# --- Main Segmentation Function ---
@torch_func
def self_supervised_segmentation_3d(
    image_volume: torch.Tensor,
    apply_segmentation: bool = True,
    min_val: float = 0.0,
    max_val: float = 1.0,
    patch_size: Optional[Tuple[int, int, int]] = None,
    n_epochs: int = 500,
    embedding_dim: int = 128,
    temperature: float = 0.1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    reconstruction_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    cluster_k: int = 2,
    mask_fraction: float = 0.01,
    sigma_noise: float = 0.2,
    lambda_bound: float = 0.1,
    **kwargs
) -> torch.Tensor:

    if not isinstance(image_volume, torch.Tensor):
        raise TypeError(f"Input image_volume must be a PyTorch Tensor. Got {type(image_volume)}")

    device = image_volume.device
    original_input_shape_len = image_volume.ndim
    original_dtype = image_volume.dtype

    img_vol_proc = image_volume.float()
    if img_vol_proc.ndim == 3:
        Z_orig, H_orig, W_orig = img_vol_proc.shape
        img_vol_proc = img_vol_proc.unsqueeze(0).unsqueeze(0)
    elif img_vol_proc.ndim == 4:
        _, Z_orig, H_orig, W_orig = img_vol_proc.shape
        img_vol_proc = img_vol_proc.unsqueeze(1)
    elif img_vol_proc.ndim == 5:
        _, _, Z_orig, H_orig, W_orig = img_vol_proc.shape
    else:
        raise ValueError(f"image_volume must be 3D, 4D or 5D. Got {image_volume.ndim}D")

    min_val_norm = float(min_val)
    max_val_norm = float(max_val)

    img_min_orig, img_max_orig = torch.min(img_vol_proc), torch.max(img_vol_proc)
    if img_max_orig > img_min_orig:
        img_vol_norm = (img_vol_proc - img_min_orig) / (img_max_orig - img_min_orig)
        img_vol_norm = img_vol_norm * (max_val_norm - min_val_norm) + min_val_norm
    else:
        img_vol_norm = torch.full_like(img_vol_proc, min_val_norm)

    # Use provided patch_size or compute default
    if patch_size is None:
        patch_size_dhw = (max(16, Z_orig // 8), max(16, H_orig // 8), max(16, W_orig // 8))  # Ensure min size
    else:
        patch_size_dhw = patch_size

    patch_size_dhw = (min(patch_size_dhw[0], Z_orig), min(patch_size_dhw[1], H_orig), min(patch_size_dhw[2], W_orig))
    if any(p <= 0 for p in patch_size_dhw):
        raise ValueError(f"Patch dimensions must be positive. Got {patch_size_dhw} for volume {Z_orig,H_orig,W_orig}")

    encoder = Encoder3D(in_channels=1, embedding_dim=embedding_dim).to(device)
    decoder = Decoder3D(embedding_dim=embedding_dim, patch_size_dhw=patch_size_dhw).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()

        patches_orig_batch = _extract_random_patches(img_vol_norm, patch_size_dhw, batch_size)

        patches_mvm_list = []
        patches_affine_list = []
        masks_for_loss_list = []

        for i in range(batch_size):
            current_patch_orig = patches_orig_batch[i] # [1, pD, pH, pW]

            mask_mvm = (torch.rand_like(current_patch_orig) < mask_fraction).bool()
            masks_for_loss_list.append(mask_mvm.clone()) # Store for loss calculation
            noise = (torch.randn_like(current_patch_orig) * sigma_noise).clamp(min_val_norm, max_val_norm)
            patch_mvm = torch.where(mask_mvm, noise, current_patch_orig)
            patches_mvm_list.append(patch_mvm)

            patch_affine = _affine_augment_patch(current_patch_orig.clone())
            patches_affine_list.append(patch_affine)

        patches_mvm_batch = torch.stack(patches_mvm_list)
        patches_affine_batch = torch.stack(patches_affine_list)
        masks_batch = torch.stack(masks_for_loss_list) # [B, 1, pD, pH, pW]

        emb_mvm = encoder(patches_mvm_batch)
        emb_affine = encoder(patches_affine_batch)
        reconstructed_patches = decoder(emb_mvm)

        loss_rec = torch.tensor(0.0, device=device)
        if masks_batch.any(): # Only compute if there are masked voxels
            # Ensure shapes match for masked selection
            masked_reconstruction = reconstructed_patches[masks_batch]
            masked_original = patches_orig_batch[masks_batch]
            if masked_reconstruction.numel() > 0: # If any elements were actually masked and selected
                 loss_rec = F.mse_loss(masked_reconstruction, masked_original)

        loss_contrastive = _nt_xent_loss(emb_mvm, emb_affine, temperature)
        loss_bound = (torch.relu(reconstructed_patches - max_val_norm) + \
                      torch.relu(min_val_norm - reconstructed_patches)).mean()

        total_loss = (reconstruction_weight * loss_rec +
                      contrastive_weight * loss_contrastive +
                      lambda_bound * loss_bound)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % (n_epochs // 10 if n_epochs >=10 else 1) == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4f} "
                        f"(Rec: {loss_rec.item():.4f}, Contr: {loss_contrastive.item():.4f}, Bound: {loss_bound.item():.4f})")

    if not apply_segmentation:
        return image_volume

    encoder.eval()
    with torch.no_grad():
        _, dense_features_full_vol = encoder(img_vol_norm, return_features_before_pool=True)

        if dense_features_full_vol.shape[2:] != (Z_orig, H_orig, W_orig):
            dense_features_upsampled = F.interpolate(
                dense_features_full_vol, size=(Z_orig, H_orig, W_orig),
                mode='trilinear', align_corners=False
            )
        else:
            dense_features_upsampled = dense_features_full_vol

        features_for_kmeans = dense_features_upsampled.squeeze(0).permute(1,2,3,0).reshape(-1, encoder.features_channels_before_pool)

        if features_for_kmeans.shape[0] == 0:
             logger.warning("No features extracted for K-Means, returning empty segmentation.")
             return torch.zeros((Z_orig, H_orig, W_orig), dtype=torch.long, device=device)

        voxel_labels_flat, _ = _kmeans_torch(features_for_kmeans, cluster_k)
        segmentation_mask = voxel_labels_flat.reshape(Z_orig, H_orig, W_orig)

    if original_input_shape_len == 3:
        return segmentation_mask.to(original_dtype)
    elif original_input_shape_len == 4:
        return segmentation_mask.unsqueeze(0).to(original_dtype)
    else:
        return segmentation_mask.unsqueeze(0).unsqueeze(0).to(original_dtype)
