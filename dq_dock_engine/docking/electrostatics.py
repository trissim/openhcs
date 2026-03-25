"""
Electrostatic energy with literature-validated parameters.

Reference:
- Allinger (1977) for ε=4 baseline
- Kirkwood-Fröhlich theory (DielectricBounds.lean)

OpenHCS Compliance:
- Frozen dataclasses for configuration
- ABC contract for electrostatics calculators
- Pure JAX functions
- Variable dielectric from Lean-proven bounds
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass


# Physical constants from DielectricBounds.lean
_EPSILON_WATER: float = 80.0  # Bulk water dielectric
_EPSILON_PROTEIN: float = 2.0  # Dry protein interior dielectric
_EPSILON_INTERFACE: float = 4.0  # Binding site interface (derived)


def kirkwood_effective_dielectric(
    eps_in: float, eps_out: float, f_geom: float
) -> float:
    """
    Kirkwood-Fröhlich effective dielectric constant.

    From DielectricBounds.lean:
        ε_eff = ε_in × ε_out / [ε_out + (ε_in - ε_out) × f_geom]

    Args:
        eps_in: Internal dielectric (protein interior, typically 2.0)
        eps_out: External dielectric (bulk solvent, typically 80.0)
        f_geom: Geometric factor in [0, 1]
                - 0: fully buried (ε_eff = ε_in)
                - 1: fully exposed (ε_eff = ε_out)
                - 0.5: interface (ε_eff ≈ 4.0)

    Returns:
        Effective dielectric constant bounded between eps_in and eps_out

    Proof: DielectricBounds.lean theorems prove:
        - dielectric_lower_bound: ε_in ≤ ε_eff
        - dielectric_upper_bound: ε_eff ≤ ε_out
    """
    denominator = eps_out + (eps_in - eps_out) * f_geom
    return (eps_in * eps_out) / denominator


def compute_f_geom_from_distance(
    distance: jnp.ndarray,
    cavity_radius: float,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """
    Compute geometric factor f_geom from distance to binding site.

    From DielectricBounds.lean:
        f(r/a) depends on position relative to protein-solvent interface.
        When distance >> cavity_radius + probe_radius, fully exposed (f_geom → 1)
        When distance << cavity_radius, fully buried (f_geom → 0)

    Args:
        distance: Distance from atom to binding site center
        cavity_radius: Radius of protein cavity (e.g., 5.0 Å)
        probe_radius: Solvent probe radius (default 1.4 Å for water)

    Returns:
        f_geom in [0, 1]
    """
    # Transition region spans probe_radius around cavity surface
    transition_start = cavity_radius - probe_radius
    transition_end = cavity_radius + probe_radius

    # Normalize to [0, 1] within transition region
    f_geom = (distance - transition_start) / (transition_end - transition_start)

    # Clamp to [0, 1]
    return jnp.clip(f_geom, 0.0, 1.0)


@dataclass(frozen=True)
class ElectrostaticsConfig:
    """
    Electrostatics configuration.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Tuple for sweep values
    - Fail-loud validation

    Variable dielectric follows DielectricBounds.lean:
    - Uses Kirkwood-Fröhlich formula with proven bounds
    - ε_in = 2.0 (protein), ε_out = 80.0 (water)
    - f_geom computed from distance to binding site
    """

    dielectric: float = 4.0  # Constant fallback
    dielectric_sweep: tuple[float, ...] = (4.0, 8.0, 12.0)
    cutoff: float = 8.0

    # Variable dielectric parameters (from DielectricBounds.lean)
    use_variable_dielectric: bool = False
    cavity_radius: float = 5.0  # Å - binding site cavity radius

    def validate(self) -> None:
        """Fail-loud validation."""
        if self.dielectric <= 0:
            raise ValueError(f"Dielectric must be positive, got {self.dielectric}")
        if not (6.0 <= self.cutoff <= 15.0):
            raise ValueError(f"Cutoff {self.cutoff} outside reasonable range [6, 15]")
        if self.cavity_radius <= 0:
            raise ValueError(
                f"Cavity radius must be positive, got {self.cavity_radius}"
            )


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
    """Coulomb electrostatic calculator with optional variable dielectric.

    When use_variable_dielectric=True, uses Kirkwood-Fröhlich formula from
    DielectricBounds.lean for distance-dependent ε_eff.
    """

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
        """Coulomb electrostatic energy with optional variable dielectric."""
        diffs = pose_coords[:, None, :] - receptor_coords[None, :, :]
        dists: jnp.ndarray = jnp.linalg.norm(diffs, axis=-1)

        within_cutoff = dists < self._config.cutoff
        cutoff_val = jnp.full_like(dists, self._config.cutoff)
        dists_masked: jnp.ndarray = jnp.where(within_cutoff, dists, cutoff_val)
        min_val = jnp.array(0.5, dtype=dists.dtype)
        dists_safe: jnp.ndarray = jnp.maximum(dists_masked, min_val)

        q_i_q_j = pose_charges[:, None] * receptor_charges[None, :]

        if self._config.use_variable_dielectric:
            # Variable dielectric from DielectricBounds.lean
            # Compute per-atom distance to binding site center (use receptor centroid)
            binding_site_center = jnp.mean(receptor_coords, axis=0)
            atom_distances = jnp.linalg.norm(pose_coords - binding_site_center, axis=-1)

            # Compute f_geom for each ligand atom
            f_geom = compute_f_geom_from_distance(
                atom_distances,
                self._config.cavity_radius,
            )

            # Per-atom effective dielectric (broadcast to receptor pairs)
            # f_geom shape: (n_lig_atoms,) → (n_lig, n_rec)
            f_geom_broadcast = f_geom[:, None] * jnp.ones_like(dists)

            # Compute ε_eff using Kirkwood formula (vectorized)
            # ε_eff = ε_in × ε_out / (ε_out + (ε_in - ε_out) × f_geom)
            eps_eff = (_EPSILON_PROTEIN * _EPSILON_WATER) / (
                _EPSILON_WATER + (_EPSILON_PROTEIN - _EPSILON_WATER) * f_geom_broadcast
            )

            # Energy with variable dielectric
            energy = q_i_q_j / (eps_eff * dists_safe)
        else:
            # Constant dielectric (original behavior)
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
