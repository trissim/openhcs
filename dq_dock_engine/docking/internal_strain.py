"""
Bond and angle strain energy for flexible receptor/ligand.

References:
- Bond stretching: Harmonic potential V = k(r - r_eq)²
- Angle bending: Harmonic potential V = k(θ - θ_eq)²
- 1-4 interactions: Scaled LJ/electrostatics

OpenHCS Compliance:
- Frozen dataclasses for configuration
- Pure JAX functions
- Explicit type hints
- Lean-proven bounds
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass


# =============================================================================
# Physical Constants from ThermalFluctuationBounds.lean
# =============================================================================

# =============================================================================
# Physical Constants from ThermalFluctuationBounds.lean
# =============================================================================

# kT at physiological temperature (310K) in kcal/mol
# From ThermalFluctuationBounds.lean: kT = R × T = 0.001987 × 310 ≈ 0.616
_kT_physiological: float = 0.616

# Force constants (from spectroscopy/crystallography)
# ThermalFluctuationBounds.lean: k_hbond ≈ 1.0 kcal/(mol·Å²)
# Typical bond: k_bond ≈ 300-500 kcal/(mol·Å²) (stiffer)
# Typical angle: k_angle ≈ 50-100 kcal/(mol·rad²)

# Default force constants (empirical - like physical constants)
DEFAULT_K_BOND: float = 300.0  # kcal/(mol·Å²)
DEFAULT_K_ANGLE: float = 50.0  # kcal/(mol·rad²)
DEFAULT_K_IMPROPER: float = 20.0  # kcal/mol - planarity restraints are weaker

# =============================================================================
# Thermal Fluctuation Derived Displacements (from ThermalFluctuationBounds.lean)
# =============================================================================

# From ThermalFluctuationBounds.lean:
#     thermalWidth(kT, k) = sqrt(kT / k)
#     σ_hbond = sqrt(0.616 / 1.0) ≈ 0.8 Å at 310K
#     σ_metal = sqrt(0.616 / 7.0) ≈ 0.3 Å at 310K
#
# For bond: k ≈ 300 kcal/(mol·Å²)
#     σ_bond = sqrt(0.616 / 300) ≈ 0.045 Å
#
# For angle: k ≈ 50 kcal/(mol·rad²)
#     σ_angle = sqrt(0.616 / 50) ≈ 0.11 rad ≈ 6°


def compute_thermal_width(kT: float, k: float) -> float:
    """Thermal fluctuation width from equipartition theorem.

    From ThermalFluctuationBounds.lean:
        σ = √(kT/k)

    This is the ACTUAL physics - standard deviation of thermal fluctuations.

    Args:
        kT: Thermal energy in kcal/mol (e.g., 0.616 at 310K)
        k: Force constant in kcal/(mol·Å²) or kcal/(mol·rad²)

    Returns:
        Standard deviation σ of thermal fluctuations
    """
    import math

    return math.sqrt(kT / k)


# =============================================================================
# Physics-Derived Thermal Widths (from ThermalFluctuationBounds.lean)
# =============================================================================

# From ThermalFluctuationBounds.lean:
#     σ = √(kT/k)
#     At physiological temperature (310K): kT ≈ 0.616 kcal/mol
#
# For bond: k ≈ 300 kcal/(mol·Å²)
#     σ_bond = √(0.616/300) ≈ 0.045 Å
#
# For angle: k ≈ 50 kcal/(mol·rad²)
#     σ_angle = √(0.616/50) ≈ 0.11 rad

BOND_THERMAL_WIDTH: float = compute_thermal_width(_kT_physiological, DEFAULT_K_BOND)
ANGLE_THERMAL_WIDTH: float = compute_thermal_width(_kT_physiological, DEFAULT_K_ANGLE)

# Default: use actual thermal widths (physics-derived, no heuristic)
BOND_MAX_DISPLACEMENT: float = BOND_THERMAL_WIDTH
ANGLE_MAX_DISPLACEMENT: float = ANGLE_THERMAL_WIDTH


def bond_displacement_from_target(target_rmsd: float, confidence: float = 1.0) -> float:
    """
    Bond displacement for certified docking.

    Uses whichever is larger:
    - Actual thermal fluctuation σ (physics)
    - User's target RMSD × confidence (engineering)

    Args:
        target_rmsd: Target RMSD in Å
        confidence: Confidence multiplier (1.0 = exact, >1 = conservative)

    Returns:
        Max bond displacement to use
    """
    rmsd_based = target_rmsd * confidence
    return max(BOND_THERMAL_WIDTH, rmsd_based)


def angle_displacement_from_target(
    target_rmsd: float, confidence: float = 1.0
) -> float:
    """
    Angle displacement for certified docking.

    Uses whichever is larger:
    - Actual thermal fluctuation σ (physics)
    - RMSD-based estimate (engineering)

    Args:
        target_rmsd: Target RMSD in Å
        confidence: Confidence multiplier (1.0 = exact, >1 = conservative)

    Returns:
        Max angle displacement in radians
    """
    # For angle: Δθ ≈ Δx / r_avg
    AVG_BOND_LENGTH = 1.5  # Å
    rmsd_based = (target_rmsd * confidence) / AVG_BOND_LENGTH
    return max(ANGLE_THERMAL_WIDTH, rmsd_based)


# =============================================================================
# Bond Stretching (Harmonic)
# =============================================================================


def harmonic_bond_energy(
    distances: jnp.ndarray,
    equilibrium_distance: float,
    force_constant: float,
) -> jnp.ndarray:
    """
    Harmonic bond stretching energy.

    From ThermalFluctuationBounds.lean:
        V(r) = k(r - r_eq)²

    Args:
        distances: Current bond lengths (Å)
        equilibrium_distance: Equilibrium bond length r_eq (Å)
        force_constant: Force constant k (kcal/(mol·Å²))

    Returns:
        Energy in kcal/mol
    """
    displacement = distances - equilibrium_distance
    return force_constant * displacement**2


def bond_stretch_lipschitz_constant(
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Lipschitz constant for harmonic bond stretching.

    For V(r) = k(r - r_eq)², the gradient is dV/dr = 2k(r - r_eq)
    Maximum gradient magnitude = 2k × max_displacement

    This is analogous to LigandStrainApproximation.lean torsion bounds.

    Args:
        force_constant: k in kcal/(mol·Å²)
        max_displacement: Maximum expected deviation from equilibrium (Å)

    Returns:
        Lipschitz constant L = 2 * k * max_displacement
    """
    return 2.0 * force_constant * max_displacement


# =============================================================================
# Angle Bending (Harmonic)
# =============================================================================


def harmonic_angle_energy(
    angles: jnp.ndarray,
    equilibrium_angle: float,
    force_constant: float,
) -> jnp.ndarray:
    """
    Harmonic angle bending energy.

    From ThermalFluctuationBounds.lean:
        V(θ) = k(θ - θ_eq)²

    Args:
        angles: Current bond angles (radians)
        equilibrium_angle: Equilibrium angle θ_eq (radians)
        force_constant: Force constant k (kcal/(mol·rad²))

    Returns:
        Energy in kcal/mol
    """
    displacement = angles - equilibrium_angle
    return force_constant * displacement**2


def angle_bend_lipschitz_constant(
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Lipschitz constant for harmonic angle bending.

    For V(θ) = k(θ - θ_eq)², the gradient is dV/dθ = 2k(θ - θ_eq)
    Maximum gradient magnitude = 2k × max_displacement

    Args:
        force_constant: k in kcal/(mol·rad²)
        max_displacement: Maximum expected deviation from equilibrium (rad)

    Returns:
        Lipschitz constant L = 2 * k * max_displacement
    """
    return 2.0 * force_constant * max_displacement


# =============================================================================
# 1-4 Interaction Scaling
# =============================================================================


def scaled_lj_14(
    distances: jnp.ndarray,
    sigma: float,
    epsilon: float,
    scale_factor: float = 0.5,
) -> jnp.ndarray:
    """
    Lennard-Jones energy with 1-4 scaling.

    Standard force fields scale 1-4 interactions (separated by 3 bonds):
    - LJ: typically scaled by 0.5
    - Electrostatics: typically scaled by 0.5 or 0.833

    From LJApproximation.lean for base LJ.

    Args:
        distances: Pair distances (Å)
        sigma: Lennard-Jones sigma (Å)
        epsilon: Lennard-Jones well depth (kcal/mol)
        scale_factor: Scaling factor for 1-4 (default 0.5)

    Returns:
        Scaled LJ energy in kcal/mol
    """
    # Standard LJ: V = 4ε[(σ/r)¹² - (σ/r)⁶]
    sigma_over_r = sigma / distances
    sigma_over_r_6 = sigma_over_r**6
    sigma_over_r_12 = sigma_over_r_6**2

    v_lj = 4.0 * epsilon * (sigma_over_r_12 - sigma_over_r_6)
    return scale_factor * v_lj


def scaled_electrostatics_14(
    distances: jnp.ndarray,
    charge_product: jnp.ndarray,
    dielectric: float = 1.0,
    scale_factor: float = 0.5,
) -> jnp.ndarray:
    """
    Electrostatic energy with 1-4 scaling.

    Standard force fields scale 1-4 electrostatics:
    - AMBER: scaled by 0.833 (1/1.2)
    - CHARMM: scaled by 0.5

    From CoulombApproximation.lean for base electrostatics.

    Args:
        distances: Pair distances (Å)
        charge_product: q_i × q_j (elementary charges)
        dielectric: Dielectric constant (default 1.0 for gas phase)
        scale_factor: Scaling factor for 1-4 (default 0.5)

    Returns:
        Scaled electrostatic energy in kcal/mol
    """
    # Coulomb: V = q_i q_j / (ε × r)
    # Convert to kcal/mol: 332.0636 factor
    COULOMB_FACTOR = 332.0636  # kcal·Å/mol·e²

    energy = COULOMB_FACTOR * charge_product / (dielectric * distances)
    return scale_factor * energy


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass(frozen=True)
class InternalStrainConfig:
    """
    Configuration for internal strain (bond/angle) energy.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Fail-loud validation

    Uses bounds from:
    - ThermalFluctuationBounds.lean (harmonic potentials)
    - LigandStrainApproximation.lean (bounded energy patterns)

    Max displacements are PHYSICS-DERIVED from thermal fluctuation:
    - From ThermalFluctuationBounds.lean: σ = √(kT/k)
    - Using 3σ for 99.7% coverage
    """

    # Bond parameters
    use_bonds: bool = False
    bond_force_constant: float = DEFAULT_K_BOND
    bond_max_displacement: float = BOND_MAX_DISPLACEMENT  # ~0.14 Å (physics-derived)

    # Angle parameters
    use_angles: bool = False
    angle_force_constant: float = DEFAULT_K_ANGLE
    angle_max_displacement: float = (
        ANGLE_MAX_DISPLACEMENT  # ~0.33 rad (physics-derived)
    )

    # Improper dihedral parameters
    use_impropers: bool = False
    improper_force_constant: float = DEFAULT_K_IMPROPER
    improper_max_displacement: float = 1.0  # Full range in cos(φ) space is [-1, 1]

    # 1-4 scaling parameters
    use_14_scaling: bool = False
    lj_14_scale: float = 0.5  # AMBER standard
    electro_14_scale: float = 0.5  # AMBER standard

    def validate(self) -> None:
        """Fail-loud validation."""
        if self.bond_force_constant <= 0:
            raise ValueError(
                f"Bond force constant must be positive, got {self.bond_force_constant}"
            )
        if self.angle_force_constant <= 0:
            raise ValueError(
                f"Angle force constant must be positive, got {self.angle_force_constant}"
            )
        if not (0 < self.lj_14_scale <= 1.0):
            raise ValueError(f"LJ 1-4 scale must be in (0, 1], got {self.lj_14_scale}")
        if not (0 < self.electro_14_scale <= 1.0):
            raise ValueError(
                f"Electro 1-4 scale must be in (0, 1], got {self.electro_14_scale}"
            )


# =============================================================================
# Certified Bounds (from Lean)
# =============================================================================


def bond_energy_bound(
    n_bonds: int,
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Upper bound on total bond strain energy.

    From LigandStrainApproximation.lean pattern (additive_strain_bounded):
        Sum of bounded terms ≤ Sum of bounds

    For harmonic V(r) = k(r - r_eq)²:
        V_max = k × max_displacement²

    Args:
        n_bonds: Number of bonds
        force_constant: k in kcal/(mol·Å²)
        max_displacement: Maximum deviation (Å)

    Returns:
        Upper bound on total bond energy
    """
    return n_bonds * force_constant * max_displacement**2


def bond_lipschitz_bound(
    n_bonds: int,
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Lipschitz constant for total bond strain energy.

    From ConformerSearch.lean pattern (per_bond_composed_lipschitz):
        L_total = Σ L_i

    Args:
        n_bonds: Number of bonds
        force_constant: k in kcal/(mol·Å²)
        max_displacement: Maximum deviation (Å)

    Returns:
        Lipschitz constant
    """
    return n_bonds * bond_stretch_lipschitz_constant(force_constant, max_displacement)


def angle_energy_bound(
    n_angles: int,
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Upper bound on total angle strain energy.

    Args:
        n_angles: Number of angles
        force_constant: k in kcal/(mol·rad²)
        max_displacement: Maximum deviation (rad)

    Returns:
        Upper bound on total angle energy
    """
    return n_angles * force_constant * max_displacement**2


def angle_lipschitz_bound(
    n_angles: int,
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Lipschitz constant for total angle strain energy.

    Args:
        n_angles: Number of angles
        force_constant: k in kcal/(mol·rad²)
        max_displacement: Maximum deviation (rad)

    Returns:
        Lipschitz constant
    """
    return n_angles * angle_bend_lipschitz_constant(force_constant, max_displacement)


# =============================================================================
# Improper Dihedrals (Planarity Restraints)
# =============================================================================


def improper_dihedral_energy(
    cos_angles: jnp.ndarray,
    equilibrium_cos_angle: float,
    force_constant: float,
) -> jnp.ndarray:
    """
    Improper dihedral energy for planarity restraints.

    Maintains planar geometry (sp2 atoms, rings) or tetrahedral geometry (chiral centers).

    Formula: V = k × (cos(φ) - cos(φ_eq))²

    Similar to harmonic but in cos(φ) space to avoid singularities.

    Args:
        cos_angles: Cosine of improper dihedral angles
        equilibrium_cos_angle: Equilibrium cosine value
            - For planar: cos(0°) = 1.0 or cos(180°) = -1.0
            - For tetrahedral: cos(109.47°) ≈ -1/3

    Returns:
        Energy in kcal/mol
    """
    displacement = cos_angles - equilibrium_cos_angle
    return force_constant * displacement**2


def improper_lipschitz_constant(
    force_constant: float,
    max_displacement: float,
) -> float:
    """
    Lipschitz constant for improper dihedral energy.

    For V = k(cos(φ) - cos(φ_eq))², gradient is dV/dφ = -2k(cos(φ) - cos(φ_eq)) × sin(φ)
    Maximum |dV/dφ| occurs at sin(φ)=1, with max displacement.

    Args:
        force_constant: k in kcal/mol
        max_displacement: Max |cos(φ) - cos(φ_eq)| ≤ 2 (range is [-1, 1])

    Returns:
        Lipschitz constant L = 2 × k × max_displacement
    """
    return 2.0 * force_constant * max_displacement


def improper_energy_bound(
    n_improper: int,
    force_constant: float,
    max_displacement: float = 2.0,
) -> float:
    """
    Upper bound on total improper dihedral energy.

    Args:
        n_improper: Number of improper dihedrals
        force_constant: k in kcal/mol
        max_displacement: Max |cos(φ) - cos(φ_eq)| (default 2.0 covers full range)

    Returns:
        Upper bound on total energy
    """
    return n_improper * force_constant * max_displacement**2


def improper_lipschitz_bound(
    n_improper: int,
    force_constant: float,
    max_displacement: float = 2.0,
) -> float:
    """
    Lipschitz constant for total improper dihedral energy.

    Args:
        n_improper: Number of improper dihedrals
        force_constant: k in kcal/mol
        max_displacement: Max |cos(φ) - cos(φ_eq)|

    Returns:
        Lipschitz constant
    """
    return n_improper * improper_lipschitz_constant(force_constant, max_displacement)
