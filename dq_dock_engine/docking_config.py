"""
Certified Docking Configuration
==============================

Two modes:
1. CERTIFIED: Only Lean-proven algorithms + NIST empirical constants
2. HEURISTIC: Uses ad-hoc approximations (fast screening only)

The CERTIFIED mode is mathematically sound. The HEURISTIC mode
may be faster but has no formal guarantee.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Literal

from dq_dock_engine.proof_status import ProofStatus


class DockingMode(Enum):
    """Docking computation mode with proof guarantees."""

    #: Lean-proven algorithms + NIST constants only
    CERTIFIED = "certified"

    #: Heuristic approximations (fast but no formal guarantee)
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class DockingConfig:
    """
    Configuration for molecular docking with proof status awareness.

    Modes:
    -----
    CERTIFIED:
        - Lattice-bounded LJ cutoff errors
        - Ewald electrostatics (conditionally certified)
        - NIST VdW radii
        - NO ad-hoc scoring weights
        - NO external binaries

    HEURISTIC:
        - Ad-hoc LJ weights (4.0, 0.4)
        - Optional SMINA/Vina for "ground truth"
        - May use different cutoff strategies
    """

    mode: DockingMode

    # Cutoff radius (Angstroms)
    # CERTIFIED: Computed from error bound
    # HEURISTIC: User-specified
    cutoff_radius: float = 10.0

    # Energy gap threshold for certification
    # CERTIFIED: Must exceed 2 × lattice_tail_bound(R)
    min_energy_gap: float = 0.0

    # Use external SMINA/Vina scoring?
    # CERTIFIED: Never
    # HEURISTIC: Optional for comparison
    use_external_scorer: bool = False

    @property
    def is_certified(self) -> bool:
        """True if running in certified mode."""
        return self.mode == DockingMode.CERTIFIED

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration consistency.

        Returns:
            (is_valid, warnings)
        """
        warnings = []

        if self.is_certified:
            if self.use_external_scorer:
                warnings.append(
                    "CERTIFIED mode: use_external_scorer=True conflicts with "
                    "formal guarantee. External scorers are HEURISTIC."
                )
            if self.cutoff_radius <= 0:
                warnings.append("CERTIFIED mode: cutoff_radius must be positive.")

        return len(warnings) == 0, warnings


# Predefined configurations
CERTIFIED_DOCKING = DockingConfig(
    mode=DockingMode.CERTIFIED,
    cutoff_radius=0.001,  # Target error bound in kcal/mol
    min_energy_gap=0.0,
    use_external_scorer=False,
)

HEURISTIC_SCREENING = DockingConfig(
    mode=DockingMode.HEURISTIC,
    cutoff_radius=8.0,  # Typical heuristic cutoff
    min_energy_gap=0.0,
    use_external_scorer=True,
)


def create_config(mode: Literal["certified", "heuristic"], **kwargs) -> DockingConfig:
    """
    Factory for creating docking configurations.

    Examples:
    --------
    >>> config = create_config("certified", cutoff_radius=12.0)
    >>> config = create_config("heuristic", use_external_scorer=True)
    """
    mode_enum = DockingMode.CERTIFIED if mode == "certified" else DockingMode.HEURISTIC
    return DockingConfig(mode=mode_enum, **kwargs)


def compute_certified_cutoff(target_error: float, exponent: float = 6.0) -> float:
    """
    Compute minimum cutoff radius for target error bound.

    From LatticeSum.lean: error(R) ≤ M/R^(s-3)

    CERTIFIED: Uses proven lattice tail bound.
    """
    from dq_dock_engine.physics.lattice_sum import optimal_cutoff

    return optimal_cutoff(target_error, exponent)


def compute_certified_energy_bound(
    delta_E: float, R: float, exponent: float = 6.0
) -> tuple[bool, float]:
    """
    Check if energy gap exceeds certified threshold.

    CERTIFIED: gap must exceed 2 × lattice_tail_bound(R)

    Returns:
        (is_certified, max_uncertainty)
    """
    from dq_dock_engine.physics.lattice_sum import lj6_cutoff_error

    max_uncertainty = 2.0 * lj6_cutoff_error(R)
    is_certified = delta_E > max_uncertainty

    return is_certified, max_uncertainty
