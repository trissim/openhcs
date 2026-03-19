import jax.numpy as jnp
from jax import jit
from .kernels import (
    apply_cutoff,
    coulomb_cutoff,
    lennard_jones_potential,
    pairwise_distances,
    typed_lennard_jones_cutoff,
)
from abc import ABC, abstractmethod

"""
Potential functions for molecular interactions.

PROOF STATUS SUMMARY:
  - LennardJones.energy: CONDITIONALLY_CERTIFIED (form proven, weights are HEURISTIC)
  - Electrostatic.energy: CONDITIONALLY_CERTIFIED (form proven, cutoff is HEURISTIC)
  - Hydrophobic.energy: HEURISTIC (contact model is empirical)
"""

from dq_dock_engine.proof_status import certified, conditionally_certified, heuristic


class Potential(ABC):
    """ABC Contract: all potentials expose a single `energy` method."""

    @abstractmethod
    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        """Compute total interaction energy between two atom sets."""


class LennardJones(Potential):
    """
    LJ potential form.

    PROOF STATUS: CONDITIONALLY_CERTIFIED
      - Form: ArrayDSL.lean::lennardJones (verified)
      - WEIGHTS ARE HEURISTIC: epsilon, sigma ratios are empirical
      - Cutoff: HEURISTIC (typically 10-12Å in practice)
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, cutoff: float = 10.0):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        dists = pairwise_distances(q_a, q_b)
        epsilon_matrix = jnp.full_like(dists, self.epsilon)
        sigma_matrix = jnp.full_like(dists, self.sigma)
        return typed_lennard_jones_cutoff(
            dists, epsilon_matrix, sigma_matrix, self.cutoff
        )


class Electrostatic(Potential):
    """
    Coulomb potential with cutoff.

    PROOF STATUS: CONDITIONALLY_CERTIFIED
      - Form: ArrayDSL.lean (verified)
      - Cutoff approximation: CONDITIONALLY_CERTIFIED via EwaldSummation.lean
      - Charge assignment: HEURISTIC (Gasteiger, AM1-BCC, etc.)
    """

    def __init__(
        self, charges_a: jnp.ndarray, charges_b: jnp.ndarray, cutoff: float = 12.0
    ):
        self.charges_a = charges_a
        self.charges_b = charges_b
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        dists = pairwise_distances(q_a, q_b)
        return coulomb_cutoff(
            self.charges_a,
            self.charges_b,
            dists,
            self.cutoff,
            1.0,
        )


class Hydrophobic(Potential):
    """
    Contact-based hydrophobic potential.

    PROOF STATUS: HEURISTIC
      - Contact model is empirical
      - No Lean backing for this functional form
    """

    def __init__(self, h_a: jnp.ndarray, h_b: jnp.ndarray, cutoff: float = 4.0):
        self.h_a = h_a
        self.h_b = h_b
        self.cutoff = cutoff

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        dists = pairwise_distances(q_a, q_b)
        contact = jnp.where(dists < self.cutoff, 1.0, 0.0)
        return jnp.sum(self.h_a[:, None] * self.h_b[None, :] * contact)


class CompositePotential(Potential):
    """
    Sum of multiple potentials.

    PROOF STATUS: CONDITIONALLY_CERTIFIED (if all components are)
    """

    def __init__(self, potentials: list[Potential]):
        self.potentials = potentials

    def energy(self, q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.stack([p.energy(q_a, q_b) for p in self.potentials]))
