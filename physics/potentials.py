import jax.numpy as jnp
from jax import jit
from .kernels import lennard_jones_potential, pairwise_distances, sum_pair_potentials

"""
Potential functions for molecular interactions.
Verified in ArrayDSL.lean and documented in BornOppenheimer.lean.
"""

@jit
def lennard_jones(
    q_ligand: jnp.ndarray, 
    q_protein: jnp.ndarray, 
    epsilon: float = 1.0, 
    sigma: float = 1.0
) -> float:
    """LJ potential between ligand and protein atoms."""
    def pot(r):
        return lennard_jones_potential(r, epsilon, sigma)
    return sum_pair_potentials(q_ligand, q_protein, pot)

@jit
def electrostatic_potential(
    q_ligand: jnp.ndarray, 
    q_protein: jnp.ndarray, 
    charges_ligand: jnp.ndarray, 
    charges_protein: jnp.ndarray
) -> float:
    """
    Coulomb potential.
    U = Σ Σ (k * q_i * q_j) / r_ij
    """
    dists = pairwise_distances(q_ligand, q_protein)
    # Using 1e-10 guard for safety
    dists_safe = jnp.where(dists > 1e-10, dists, 1e-10)
    interaction_matrix = (charges_ligand[:, None] * charges_protein[None, :]) / dists_safe
    # Mask out the diagonal if same array but here we assume different arrays
    return jnp.sum(interaction_matrix)

@jit
def hydrophobic_potential(
    q_ligand: jnp.ndarray, 
    q_protein: jnp.ndarray, 
    hydrophobicity_ligand: jnp.ndarray, 
    hydrophobicity_protein: jnp.ndarray,
    rc: float = 4.0
) -> float:
    """
    Simplified hydrophobic potential based on contact surface and distance.
    U = Σ Σ h_i * h_j * f(r_ij)
    """
    dists = pairwise_distances(q_ligand, q_protein)
    # Contact is 1 if r < rc, else 0
    contact = jnp.where(dists < rc, 1.0, 0.0)
    return jnp.sum(hydrophobicity_ligand[:, None] * hydrophobicity_protein[None, :] * contact)
