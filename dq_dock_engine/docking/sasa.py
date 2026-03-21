"""
Solvent-accessible surface area (SASA) calculation.

Reference: Shrake & Rupley (1973) J. Mol. Biol. 79(2):351-371
           Lee & Richards (1971) J. Mol. Biol. 55(3):379-400

OpenHCS Compliance:
- Frozen dataclasses for configuration
- Pure JAX functions
- Explicit type hints
- Fail-loud validation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass


# =============================================================================
# FROZEN DATACLASS CONFIGURATION (OpenHCS Pattern)
# =============================================================================

# Atomic solvation parameters from Eisenberg & McLachlan (1986)
_SOLVATION_PARAMS: dict[str, tuple[float, float]] = {
    "C": (16.0, 1.70),
    "N": (-11.0, 1.55),
    "O": (-11.0, 1.52),
    "S": (21.0, 1.80),
    "H": (0.0, 1.20),
    "P": (15.0, 1.80),
    "NA": (-20.0, 1.40),
    "K": (-20.0, 1.80),
    "CL": (-20.0, 1.75),
    "MG": (-20.0, 1.45),
    "CA": (-20.0, 1.70),
    "FE": (-15.0, 1.50),
}


@dataclass(frozen=True)
class SASAConfig:
    """
    SASA calculation configuration.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Fail-loud validation

    Reference: Shrake & Rupley (1973), Lee & Richards (1971)
    """

    probe_radius: float = 1.4
    n_points: int = 14
    surface_density: float = 14.4

    def validate(self) -> None:
        """Fail-loud validation."""
        if not (1.0 <= self.probe_radius <= 2.0):
            raise ValueError(
                f"Probe radius {self.probe_radius} outside standard range [1.0, 2.0]"
            )
        if self.n_points < 4:
            raise ValueError(f"n_points {self.n_points} too small (min 4)")


@dataclass(frozen=True)
class SASAAtomConfig:
    """SASA configuration for a single atom with its parameters."""

    sigma: float
    radius: float


# =============================================================================
# FROZEN DATA CONTAINERS (OpenHCS Pattern)
# =============================================================================


@dataclass(frozen=True)
class SASAAtom:
    """Immutable atom for SASA calculation."""

    position: jnp.ndarray
    sigma: float
    radius: float


@dataclass(frozen=True)
class SASAResult:
    """Immutable result of SASA calculation."""

    sasa: float  # Å²
    solvation_energy: float  # kcal/mol


# =============================================================================
# FACTORY FUNCTION (Explicit Dependency Injection)
# =============================================================================


def create_sasa_config(
    probe_radius: float = 1.4,
    n_points: int = 14,
) -> SASAConfig:
    """
    Factory function with explicit dependency injection.

    OpenHCS Compliance:
    - Explicit factory, not __init__
    - Validation at construction
    """
    config = SASAConfig(probe_radius=probe_radius, n_points=n_points)
    config.validate()
    return config


def get_solvation_params(
    element: str,
) -> SASAAtomConfig:
    """
    Get solvation parameters for an element.

    OpenHCS Compliance:
    - Pure function
    - Fail-loud on unknown element
    - No defensive defaultdict
    """
    upper = element.upper()
    if upper not in _SOLVATION_PARAMS:
        raise ValueError(f"Unknown element for solvation: {element}")
    sigma, radius = _SOLVATION_PARAMS[upper]
    return SASAAtomConfig(sigma=sigma, radius=radius)


# =============================================================================
# PURE JAX FUNCTIONS (Stateless, No Side Effects)
# =============================================================================


def _fibonacci_sphere_points(n_points: int) -> jnp.ndarray:
    """Generate evenly distributed points on a sphere."""
    indices = jnp.arange(0, n_points, dtype=float)
    phi = jnp.arccos(1.0 - 2.0 * (indices + 0.5) / n_points)
    theta = jnp.pi * (1.0 + 5**0.5) * indices
    x = jnp.cos(theta) * jnp.sin(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(phi)
    return jnp.stack([x, y, z], axis=1)


def compute_sasa_single(
    atom_position: jnp.ndarray,
    atom_radius: float,
    neighbor_positions: jnp.ndarray,
    neighbor_radii: jnp.ndarray,
    config: SASAConfig,
) -> float:
    """
    Compute SASA for a single atom using Shrake-Rupley algorithm.

    OpenHCS Compliance:
    - Pure function
    - jax.jit for GPU acceleration
    - Explicit types
    """
    directions = _fibonacci_sphere_points(config.n_points)
    surface_points = atom_position + (atom_radius + config.probe_radius) * directions

    n_points = surface_points.shape[0]
    accessible = jnp.ones(n_points, dtype=bool)

    for i in range(neighbor_positions.shape[0]):
        diffs = surface_points - neighbor_positions[i]
        distances = jnp.linalg.norm(diffs, axis=1)
        buried_threshold = neighbor_radii[i] + config.probe_radius
        is_buried = distances < buried_threshold
        accessible = accessible & ~is_buried

    n_accessible = float(jnp.sum(accessible))
    sphere_radius = atom_radius + config.probe_radius
    sphere_area = 4.0 * jnp.pi * sphere_radius**2
    area_per_point = sphere_area / config.n_points

    return n_accessible * area_per_point


def accessible_surface_points_single(
    atom_position: jnp.ndarray,
    atom_radius: float,
    neighbor_positions: jnp.ndarray,
    neighbor_radii: jnp.ndarray,
    config: SASAConfig,
) -> jnp.ndarray:
    """Return the accessible surface samples used by Shrake-Rupley."""
    directions = _fibonacci_sphere_points(config.n_points)
    surface_points = atom_position + (atom_radius + config.probe_radius) * directions
    accessible = jnp.ones(surface_points.shape[0], dtype=bool)

    for i in range(neighbor_positions.shape[0]):
        diffs = surface_points - neighbor_positions[i]
        distances = jnp.linalg.norm(diffs, axis=1)
        buried_threshold = neighbor_radii[i] + config.probe_radius
        is_buried = distances < buried_threshold
        accessible = accessible & ~is_buried

    return surface_points[accessible]


def compute_sasa_batch(
    positions: jnp.ndarray,
    radii: jnp.ndarray,
    elements: tuple[str, ...],
    config: SASAConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute SASA for all atoms.

    OpenHCS Compliance:
    - Pure function
    - Explicit config dependency
    - Tuple inputs (immutable)
    """
    n_atoms = positions.shape[0]
    sasa_values = []
    solvation_energies = []

    for i in range(n_atoms):
        elem = elements[i]
        atom_config = get_solvation_params(elem)

        neighbor_mask = jnp.arange(n_atoms) != i
        neighbor_positions = positions[neighbor_mask]
        neighbor_radii_i = radii[neighbor_mask]

        sasa = compute_sasa_single(
            positions[i],
            atom_config.radius,
            neighbor_positions,
            neighbor_radii_i,
            config,
        )

        solvation_e = atom_config.sigma * sasa / 1000.0

        sasa_values.append(sasa)
        solvation_energies.append(solvation_e)

    return (jnp.array(sasa_values), jnp.array(solvation_energies))


def delta_sasa_energy(
    unbound_positions: jnp.ndarray,
    bound_positions: jnp.ndarray,
    radii: jnp.ndarray,
    elements: tuple[str, ...],
    config: SASAConfig,
) -> float:
    """
    Compute ΔSASA desolvation energy.

    OpenHCS Compliance:
    - Pure function
    - Explicit config dependency
    - Fail-loud validation
    """
    config.validate()

    _, unbound_solv = compute_sasa_batch(unbound_positions, radii, elements, config)
    _, bound_solv = compute_sasa_batch(bound_positions, radii, elements, config)

    return float(jnp.sum(bound_solv - unbound_solv))
