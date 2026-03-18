import jax.numpy as jnp
from jax import jit
from .kernels import lennard_jones_potential, pairwise_distances, apply_cutoff
from abc import ABC, abstractmethod

"""
Potential functions for molecular interactions.
ABC contract enforces consistent interface for all potentials.
"""

class Potential(ABC):
    """ABC Contract: all potentials expose a single `energy` method."""
    @abstractmethod
    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> float:
        """Compute total interaction energy between two atom sets."""

class LennardJones(Potential):
    """LJ potential. Verified in ArrayDSL.lean::lennardJones."""
    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, cutoff: float = 10.0):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> float:
        dists = pairwise_distances(q_a, q_b)
        masked = apply_cutoff(dists, self.cutoff)
        # Reuse DSL primitive directly
        return jnp.sum(lennard_jones_potential(masked, self.epsilon, self.sigma))

class Electrostatic(Potential):
    """Coulomb potential. Uses DSL: pairwiseDistances, applyCutoff."""
    def __init__(self, charges_a: jnp.ndarray, charges_b: jnp.ndarray, cutoff: float = 12.0):
        self.charges_a = charges_a
        self.charges_b = charges_b
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> float:
        dists = pairwise_distances(q_a, q_b)
        masked = apply_cutoff(dists, self.cutoff)
        # Singularity guard consistent with ArrayDSL.lean::lennardJones pattern
        r_safe = jnp.where(masked > 1e-10, masked, 1e-10)
        charge_product = self.charges_a[:, None] * self.charges_b[None, :]
        return jnp.sum(jnp.where(masked > 0, charge_product / r_safe, 0.0))

class Hydrophobic(Potential):
    """Contact-based hydrophobic potential. Uses DSL: pairwiseDistances, applyCutoff."""
    def __init__(self, h_a: jnp.ndarray, h_b: jnp.ndarray, cutoff: float = 4.0):
        self.h_a = h_a
        self.h_b = h_b
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> float:
        dists = pairwise_distances(q_a, q_b)
        contact = jnp.where(dists < self.cutoff, 1.0, 0.0)
        return jnp.sum(self.h_a[:, None] * self.h_b[None, :] * contact)

class CompositePotential(Potential):
    """Sum of multiple potentials. Genericism enforcement."""
    def __init__(self, potentials: list[Potential]):
        self.potentials = potentials

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> float:
        return sum(p.energy(q_a, q_b) for p in self.potentials)
