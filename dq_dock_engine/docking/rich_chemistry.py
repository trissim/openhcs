import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.scoring import (
    CertifiedScreenedCoulombSpec,
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
)
from dq_dock_engine.docking.core import LigandContext

def find_k_nearest_neighbors(coords: np.ndarray, k: int) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    if dists.shape[0] <= k:
        return np.tile(np.arange(dists.shape[0]), (dists.shape[0], 1))[:, :k]
    return np.argsort(dists, axis=-1)[:, :k]

def build_screened_coulomb_spec(
    receptor_charges: np.ndarray, 
    ligand_charges: np.ndarray,
    kappa: float = 1.0,
    cutoff: float = 8.0,
    dielectric: float = 4.0
) -> CertifiedScreenedCoulombSpec:
    return CertifiedScreenedCoulombSpec(
        receptor_charges=jnp.array(receptor_charges, dtype=jnp.float32),
        ligand_charges=jnp.array(ligand_charges, dtype=jnp.float32),
        kappa=kappa,
        cutoff=cutoff,
        dielectric=dielectric,
    )

def build_contact_surrogate_spec(
    receptor_elements: tuple[str, ...],
    ligand_elements: tuple[str, ...],
    beta: float = 0.6,
    cutoff: float = 6.0
) -> CertifiedContactSurrogateSpec:
    # A simple builder: weight is 1.0 for polar (N,O,S), 0.5 for others
    polars = {"N", "O", "S", "F", "Cl", "Br", "I"}
    r_weights = np.array([1.0 if e in polars else 0.5 for e in receptor_elements], dtype=np.float32)
    l_weights = np.array([1.0 if e in polars else 0.5 for e in ligand_elements], dtype=np.float32)
    return CertifiedContactSurrogateSpec(
        receptor_weights=jnp.array(r_weights),
        ligand_weights=jnp.array(l_weights),
        beta=beta,
        cutoff=cutoff,
    )

def build_directional_hbond_spec(
    receptor_coords: np.ndarray,
    receptor_elements: tuple[str, ...],
    ligand_coords: np.ndarray,
    ligand_elements: tuple[str, ...],
    ideal_distance: float = 2.8,
    distance_width: float = 0.8,
    cutoff: float = 4.0
) -> CertifiedDirectionalHBondSpec:
    polars = {"N", "O", "F"}
    
    # 1. Strengths (1.0 for polar, 0.0 otherwise)
    r_strengths = np.array([1.0 if e in polars else 0.0 for e in receptor_elements], dtype=np.float32)
    l_strengths = np.array([1.0 if e in polars else 0.0 for e in ligand_elements], dtype=np.float32)
    
    # 2. Receptor directions (rigid, so precomputed)
    if receptor_coords.shape[0] > 0:
        r_neighbors = find_k_nearest_neighbors(receptor_coords, k=3)
        r_neighbor_coords = receptor_coords[np.arange(receptor_coords.shape[0])[:, None], r_neighbors]
        r_mean_neighbor = np.mean(r_neighbor_coords, axis=1)
        r_directions = receptor_coords - r_mean_neighbor
        r_norms = np.linalg.norm(r_directions, axis=-1, keepdims=True)
        r_directions = np.where(r_norms > 1e-6, r_directions / r_norms, 0.0)
    else:
        r_directions = np.zeros((0, 3))
    
    # 3. Ligand neighbor indices
    if ligand_coords.shape[0] > 0:
        l_neighbors = find_k_nearest_neighbors(ligand_coords, k=3)
    else:
        l_neighbors = np.zeros((0, 3), dtype=np.int32)
    
    return CertifiedDirectionalHBondSpec(
        receptor_directions=jnp.array(r_directions, dtype=jnp.float32),
        ligand_neighbor_indices=jnp.array(l_neighbors, dtype=jnp.int32),
        receptor_strengths=jnp.array(r_strengths, dtype=jnp.float32),
        ligand_strengths=jnp.array(l_strengths, dtype=jnp.float32),
        ideal_distance=ideal_distance,
        distance_width=distance_width,
        cutoff=cutoff,
    )

def build_all_rich_chemistry_specs(
    receptor_coords: np.ndarray,
    receptor_elements: tuple[str, ...],
    receptor_charges: np.ndarray,
    ligand_ctx: LigandContext,
) -> tuple[CertifiedScreenedCoulombSpec, CertifiedContactSurrogateSpec, CertifiedDirectionalHBondSpec]:
    ligand_charges = ligand_ctx.charges if ligand_ctx.charges is not None else np.zeros((ligand_ctx.base_coords.shape[0],))
    screened = build_screened_coulomb_spec(receptor_charges, ligand_charges)
    contact = build_contact_surrogate_spec(receptor_elements, ligand_ctx.elements)
    hbond = build_directional_hbond_spec(
        receptor_coords, receptor_elements, 
        ligand_ctx.base_coords, ligand_ctx.elements
    )
    return screened, contact, hbond
