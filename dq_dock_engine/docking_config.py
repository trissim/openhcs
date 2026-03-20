"""Docking configuration with explicit certified/heuristic boundaries."""

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


class OptimizerBackend(Enum):
    """Optimizer backend for local pose refinement."""

    #: JAX gradient descent on translation/quaternion (batch-parallel)
    GRADIENT = "gradient"

    #: Formal multi-round Bayesian action selection (per-pose, certified bounds)
    FORMAL = "formal"


class FormalRoundStrategy(Enum):
    """Certified local-round strategy for the formal optimizer."""

    EXACT = "exact"
    SINGLETON_HYBRID = "singleton_hybrid"


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

    #: CERTIFIED: Target error bound in kcal/mol (passed to optimal_cutoff)
    #: HEURISTIC: Cutoff radius in Angstroms for ad-hoc scoring
    target_error: float = 0.001

    #: Energy gap threshold for certification
    #: CERTIFIED: Must exceed 2 × lattice_tail_bound(R)
    min_energy_gap: float = 0.0

    #: Use external SMINA/Vina scoring?
    #: CERTIFIED: Never
    #: HEURISTIC: Optional for comparison
    use_external_scorer: bool = False

    #: Which optimizer backend to use for local refinement
    optimizer_backend: OptimizerBackend = OptimizerBackend.GRADIENT

    #: Which certified local-round strategy to use when optimizer_backend is FORMAL
    formal_round_strategy: FormalRoundStrategy = FormalRoundStrategy.SINGLETON_HYBRID

    def __post_init__(self) -> None:
        if (
            self.mode == DockingMode.CERTIFIED
            and self.optimizer_backend != OptimizerBackend.FORMAL
        ):
            raise ValueError(
                "CERTIFIED mode requires OptimizerBackend.FORMAL; gradient refinement is heuristic."
            )

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
            if self.target_error <= 0:
                warnings.append("CERTIFIED mode: target_error must be positive.")

        return len(warnings) == 0, warnings


# Predefined configurations
CERTIFIED_DOCKING = DockingConfig(
    mode=DockingMode.CERTIFIED,
    target_error=0.001,
    min_energy_gap=0.0,
    use_external_scorer=False,
    optimizer_backend=OptimizerBackend.FORMAL,
    formal_round_strategy=FormalRoundStrategy.SINGLETON_HYBRID,
)

CERTIFIED_DOCKING_FORMAL = DockingConfig(
    mode=DockingMode.CERTIFIED,
    target_error=0.001,
    min_energy_gap=0.0,
    use_external_scorer=False,
    optimizer_backend=OptimizerBackend.FORMAL,
    formal_round_strategy=FormalRoundStrategy.SINGLETON_HYBRID,
)

HEURISTIC_SCREENING = DockingConfig(
    mode=DockingMode.HEURISTIC,
    target_error=8.0,  # Angstroms, heuristic cutoff radius
    min_energy_gap=0.0,
    use_external_scorer=True,
)


def create_config(
    mode: Literal["certified", "heuristic"],
    optimizer: Literal["gradient", "formal"] = "formal",
    formal_round_strategy: FormalRoundStrategy = FormalRoundStrategy.SINGLETON_HYBRID,
    **kwargs,
) -> DockingConfig:
    """Factory for creating docking configurations."""
    mode_enum = DockingMode.CERTIFIED if mode == "certified" else DockingMode.HEURISTIC
    backend = (
        OptimizerBackend.FORMAL if optimizer == "formal" else OptimizerBackend.GRADIENT
    )
    return DockingConfig(
        mode=mode_enum,
        optimizer_backend=backend,
        formal_round_strategy=formal_round_strategy,
        **kwargs,
    )


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
