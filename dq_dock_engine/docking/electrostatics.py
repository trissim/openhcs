"""
Electrostatic energy with literature-validated parameters.

Reference: Allinger (1977) for ε=4 baseline

OpenHCS Compliance:
- Frozen dataclasses for configuration
- ABC contract for electrostatics calculators
- Pure JAX functions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ElectrostaticsConfig:
    """
    Electrostatics configuration.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Tuple for sweep values
    - Fail-loud validation
    """

    dielectric: float = 4.0
    dielectric_sweep: tuple[float, ...] = (4.0, 8.0, 12.0)
    cutoff: float = 8.0
    use_distance_dependent: bool = False

    def validate(self) -> None:
        """Fail-loud validation."""
        if self.dielectric <= 0:
            raise ValueError(f"Dielectric must be positive, got {self.dielectric}")
        if not (6.0 <= self.cutoff <= 15.0):
            raise ValueError(f"Cutoff {self.cutoff} outside reasonable range [6, 15]")


class ElectrostaticsCalculator(ABC):
    """ABC contract for electrostatics calculators."""

    @abstractmethod
    def compute(
        self,
        pose_coords: jnp.ndarray,
        pose_charges: jnp.ndarray,
        receptor_coords: jnp.ndarray,
        receptor_charges: jnp.ndarray,
    ) -> float:
        """Compute electrostatic energy."""

    @property
    @abstractmethod
    def config(self) -> ElectrostaticsConfig:
        """Direct access to config."""


class CoulombCalculator(ElectrostaticsCalculator):
    """Coulomb electrostatic calculator."""

    def __init__(self, config: ElectrostaticsConfig | None = None) -> None:
        self._config = config if config is not None else ElectrostaticsConfig()
        self._config.validate()

    @property
    def config(self) -> ElectrostaticsConfig:
        return self._config

    def compute(
        self,
        pose_coords: jnp.ndarray,
        pose_charges: jnp.ndarray,
        receptor_coords: jnp.ndarray,
        receptor_charges: jnp.ndarray,
    ) -> float:
        """Coulomb electrostatic energy."""
        diffs = pose_coords[:, None, :] - receptor_coords[None, :, :]
        dists: jnp.ndarray = jnp.linalg.norm(diffs, axis=-1)

        within_cutoff = dists < self._config.cutoff
        cutoff_val = jnp.full_like(dists, self._config.cutoff)
        dists_masked: jnp.ndarray = jnp.where(within_cutoff, dists, cutoff_val)
        min_val = jnp.array(0.5, dtype=dists.dtype)
        dists_safe: jnp.ndarray = jnp.maximum(dists_masked, min_val)

        q_i_q_j = pose_charges[:, None] * receptor_charges[None, :]
        energy = q_i_q_j / (self._config.dielectric * dists_safe)

        zero_energy = jnp.zeros_like(energy)
        return float(jnp.sum(jnp.where(within_cutoff, energy, zero_energy)))


def create_electrostatics_calculator(
    config: ElectrostaticsConfig | None = None,
) -> ElectrostaticsCalculator:
    """Factory function."""
    cfg = config if config is not None else ElectrostaticsConfig()
    cfg.validate()
    return CoulombCalculator(cfg)
