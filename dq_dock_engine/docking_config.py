"""Docking configuration with explicit certified/heuristic boundaries."""

from enum import Enum
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

from jax.tree_util import register_pytree_node_class

from dq_dock_engine.docking.core import CertifiedBindingSite
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


class CertifiedScoringFamily(Enum):
    """Certified scoring family used by the formal/runtime path."""

    LJ = "lj"
    LJ_REALSPACE_EWALD = "lj_realspace_ewald"


class ExactChemistryMode(Enum):
    """Exact chemistry family used for certified exact rescoring/refinement."""

    NONE = "none"
    EXTENDED_RICH = "extended_rich"


class ConformerSearchMode(Enum):
    """Whether to run certified conformer (torsion) search for ligands."""

    DISABLED = "disabled"
    ENABLED = "enabled"


class RefinementCertificationMode(Enum):
    """Strategy for certified refinement budget derivation.

    OBSERVED:     SE(3) GD with scan-recorded energy trajectory → empirical q,
                  certified post-hoc via Hessian + Jacobian bridge (Approach A).
    CERTIFIED_GD: Standard gradient descent with theorem-derived step size and
                  budget from SE(3) Hessian eigenvalues (Approach B).
    """

    OBSERVED = "observed"
    CERTIFIED_GD = "certified_gd"


@register_pytree_node_class
@dataclass(frozen=True)
class DockingConfig:
    """
    Configuration for molecular docking with proof status awareness.
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

    #: Certified coarse surrogate error budget used by formal round pruning.
    #: Larger values shrink the certified cutoff and can reduce compute.
    coarse_target_error: float = 0.004

    #: Optional theorem-backed softened-LJ coarse prefilter.
    #: Disabled by default because the current aggregate softening bound is often
    #: too conservative to improve pruning in practice.
    use_softened_coarse_prefilter: bool = False

    #: Optional adaptive certified coarse schedule for singleton acceptance.
    #: Tried from loosest to tightest until a theorem-backed singleton decision
    #: succeeds, then the exact fallback handles the remainder.
    adaptive_coarse_target_errors: Tuple[float, ...] | None = (0.05, 0.01, 0.004)

    #: The exact chemistry family used for certified exact rescoring/refinement.
    exact_chemistry_mode: ExactChemistryMode = ExactChemistryMode.EXTENDED_RICH

    #: Certified physical score family for both route_scoring and formal rounds
    certified_scoring_family: CertifiedScoringFamily = (
        CertifiedScoringFamily.LJ_REALSPACE_EWALD
    )

    #: Whether to run certified conformer search (torsional flexibility)
    conformer_search: ConformerSearchMode = ConformerSearchMode.ENABLED

    #: Whether branch-and-bound may treat the parsed input conformer as an
    #: explicit incumbent/returned candidate. Disabled by default so blind
    #: conformer docking does not privilege the crystallographic input geometry.
    reuse_initial_conformer: bool = False

    #: Target RMSD precision in Angstroms. This is THE primary specification:
    #: the certified iteration budget, seed budget, and pruning thresholds
    #: are all derived from this value via the Hessian + Jacobian bridge.
    target_rmsd: float = 0.5

    #: Probability of capturing the global minimum basin with the initial
    #: seed set. The pipeline uses a small probe phase to estimate a certified
    #: local contraction rate, then derives the main seed budget (n_poses) from
    #: this confidence level, the search volume (box + torsion space), and the
    #: implied capture basin volume. Higher confidence = more seeds = more compute.
    confidence: float = 0.999

    #: Strategy for certified refinement budget derivation.
    #: CERTIFIED_GD derives per-pose step budgets from the SE(3) Hessian,
    #: replacing the fixed n_opt_steps with a theorem-backed adaptive budget.
    refinement_certification: RefinementCertificationMode = (
        RefinementCertificationMode.CERTIFIED_GD
    )

    #: Wall-clock timeout in seconds. The pipeline will stop and return the
    #: best results found so far if this budget is exceeded.
    timeout_seconds: float = 300.0

    #: Fixed padding limits for JAX JIT stability.
    #: All receptor/ligand atom sets are padded to these sizes with ghost atoms.
    max_receptor_atoms: int = 1024
    max_ligand_atoms: int = 128

    #: Optional theorem-backed binding site used to restrict blind docking to a
    #: certified pocket region.
    certified_binding_site: CertifiedBindingSite | None = None

    def tree_flatten(self):
        children = (
            self.target_error,
            self.min_energy_gap,
            self.coarse_target_error,
            self.target_rmsd,
            self.certified_binding_site,
        )
        aux_data = (
            self.mode,
            self.use_external_scorer,
            self.optimizer_backend,
            self.formal_round_strategy,
            self.use_softened_coarse_prefilter,
            self.adaptive_coarse_target_errors,
            self.exact_chemistry_mode,
            self.certified_scoring_family,
            self.max_receptor_atoms,
            self.max_ligand_atoms,
            self.conformer_search,
            self.reuse_initial_conformer,
            self.refinement_certification,
            self.timeout_seconds,
            self.confidence,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            mode=aux_data[0],
            target_error=children[0],
            min_energy_gap=children[1],
            use_external_scorer=aux_data[1],
            optimizer_backend=aux_data[2],
            formal_round_strategy=aux_data[3],
            coarse_target_error=children[2],
            use_softened_coarse_prefilter=aux_data[4],
            adaptive_coarse_target_errors=aux_data[5],
            exact_chemistry_mode=aux_data[6],
            certified_scoring_family=aux_data[7],
            target_rmsd=children[3],
            certified_binding_site=children[4],
            max_receptor_atoms=aux_data[8],
            max_ligand_atoms=aux_data[9],
            conformer_search=aux_data[10],
            reuse_initial_conformer=aux_data[11],
            refinement_certification=aux_data[12],
            timeout_seconds=aux_data[13],
            confidence=aux_data[14],
        )

    def __post_init__(self) -> None:
        pass

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
            if self.coarse_target_error <= 0:
                warnings.append("CERTIFIED mode: coarse_target_error must be positive.")
            if self.coarse_target_error < self.target_error:
                warnings.append(
                    "CERTIFIED mode: coarse_target_error < target_error tightens the surrogate instead of coarsening it."
                )
            if self.adaptive_coarse_target_errors is not None:
                if len(self.adaptive_coarse_target_errors) == 0:
                    warnings.append(
                        "CERTIFIED mode: adaptive_coarse_target_errors cannot be empty."
                    )
                elif any(err <= 0 for err in self.adaptive_coarse_target_errors):
                    warnings.append(
                        "CERTIFIED mode: adaptive_coarse_target_errors entries must be positive."
                    )
            if self.exact_chemistry_mode == ExactChemistryMode.NONE:
                warnings.append(
                    "CERTIFIED mode: exact_chemistry_mode=NONE disables theorem-backed exact chemistry refinement."
                )

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
