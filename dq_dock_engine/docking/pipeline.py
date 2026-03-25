"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring
with multi-stage filtering and pocket-guided sampling.
"""

import inspect
import math
from abc import ABC, abstractmethod
from dataclasses import (
    MISSING,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Optional,
    Self,
    TypeVar,
    Union,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import (
    BindingSite,
    BlindDockingPlan,
    CertifiedBindingSite,
    CertifiedBlindDockingResult,
    CertifiedBlindDockingPlan,
    CertifiedPocketFailureReason,
    DockingBox,
    GeometricBindingSite,
    GeometricBlindDockingPlan,
    GeometricBlindDockingResult,
    LigandContext,
    SamplingStrategy,
    ScoringEngine,
    ScoredPose,
    PoseVector,
    GapCertification,
    NativeCertification,
    CertificationDecision,
)
from dq_dock_engine.docking.charges import ChargeMethod, create_charge_assigner
from dq_dock_engine.docking.scoring import (
    CertifiedRealSpaceEwaldSpec,
    route_scoring,
    score_certified_lj,
)
from dq_dock_engine.docking.pocket_analysis import (
    CertifiedDetectedPocket,
    GeometricDetectedPocket,
    detect_certified_pocket,
    detect_geometric_pocket,
)
from dq_dock_engine.docking.formal_pruning import (
    certified_pruning_certificate,
    coarse_top1_ambiguity_mask,
)
from dq_dock_engine.docking.scoring import (
    score_certified_softened_lj,
    score_certified_softened_lj_realspace_ewald,
)
from dq_dock_engine.docking.pocket_sampling import (
    extract_local_pocket_region,
    extract_local_pocket_region_view,
)
from dq_dock_engine.docking.se3_refinement import (
    RefinementCertificate,
    make_se3_energy_fn,
    make_se3_kinematics_fn,
    observe_gd_trajectory,
    optimize_certified_gd,
    _pose_to_se3_params,
    _se3_params_to_pose,
)
from dq_dock_engine.docking_config import RefinementCertificationMode
from dq_dock_engine.docking.scoring_context import (
    CertifiedScoringContext,
    build_certified_scoring_context,
)
from dq_dock_engine.docking.conformer_search import (
    BranchAndBoundConfig,
    RotatableBond,
    TorsionStrainParams,
    compute_raw_lj_lipschitz,
    compute_softened_lipschitz_constant,
    detect_rotatable_bonds,
    derive_uff_torsion_barrier_heights,
    search_conformers,
)
from dq_dock_engine.docking.formal_handles import (
    branch_and_bound_cross_docking_handles,
    pocket_cross_docking_handles,
    receptor_flex_cross_docking_handles,
    receptor_flexibility_theorem_handles,
)
from dq_dock_engine.physics.kernels import rigid_transform_3d
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    ConformerSearchMode,
    compute_certified_cutoff,
    DockingConfig,
    DockingMode,
    ExactChemistryMode,
    FormalRoundStrategy,
    OptimizerBackend,
    SofteningPolicy,
)

if TYPE_CHECKING:
    from dq_dock_engine.docking.formal_sampling import CertifiedGlobalActionFamily


# Certified Pruning Constants
# We use a fixed power-of-two size for the survivor set to stabilize XLA caching.
# Two-phase seed-budget derivation uses a small probe set to estimate a certified
# local contraction rate and derive the full seed budget mechanically from the
# target RMSD. These constants control only the calibration phase size.
SEED_BUDGET_PROBE_POSES = 16
SEED_BUDGET_PROBE_TOP_K = 4

# Exact certified rescoring can be invoked over the full canonical retain survivor set.
# Keep it chunked so theorem-honest Top-K pruning remains usable on large pose sets
# without allocating one giant batched exact-score tensor.
EXACT_RESCORING_CHUNK_SIZE = 2048

# Formal local refinement expands each survivor into a finite local action family.
# Chunk that stage as well so large certified survivor sets do not need one giant
# action tensor resident on device at once.
FORMAL_REFINEMENT_CHUNK_SIZE = 512

RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED = "RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED"
RUNTIME_ORIENTATION_SIGNAL_INACTIVE = RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED
RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY = "RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY"


def _merge_theorem_handles(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return tuple(dict.fromkeys(merged))


def _detect_rigid_equivalence_ambiguity(
    coords_batch: jnp.ndarray,
    scores: jnp.ndarray | np.ndarray,
    *,
    error_bound: float | None,
    target_rmsd: float,
    max_candidates: int = 8,
) -> bool:
    if (
        coords_batch.shape[0] < 2
        or error_bound is None
        or not math.isfinite(error_bound)
    ):
        return False

    from dq_dock_engine.docking.metrics import (
        compute_docking_rmsd_batched,
        compute_rmsd_batched,
    )

    limit = min(int(coords_batch.shape[0]), max_candidates)
    ranked = jnp.argsort(scores)[:limit]
    ranked_coords = coords_batch[ranked]
    ranked_scores = scores[ranked]
    if ranked_coords.shape[0] < 2:
        return False

    best_coords = ranked_coords[0]
    others = ranked_coords[1:]
    kabsch = np.asarray(jax.device_get(compute_rmsd_batched(others, best_coords)))
    raw = np.asarray(jax.device_get(compute_docking_rmsd_batched(others, best_coords)))
    score_gaps = np.abs(
        np.asarray(
            jax.device_get(ranked_scores[1:] - ranked_scores[0]), dtype=np.float64
        )
    )
    aligned_tol = target_rmsd / 2.0
    separated_tol = target_rmsd
    ambiguity_gap = 2.0 * float(error_bound)
    return bool(
        np.any(
            (kabsch <= aligned_tol)
            & (raw >= separated_tol)
            & (score_gaps <= ambiguity_gap)
        )
    )


def _certified_top1_gap(
    scores: jnp.ndarray | np.ndarray, error_bound: float | None
) -> bool:
    """Finite certified singleton top-1 check from a top-2 score gap.

    Lean: TK16 (`exact_top1_eq_singleton_of_coarse_energy_gap_margin`).
    Lower energies are better, so the theorem is applied to negated utilities.
    """
    if scores.shape[0] < 2 or error_bound is None or not math.isfinite(error_bound):
        return False
    scores_np = np.asarray(jax.device_get(scores), dtype=np.float64)
    top2 = np.argsort(scores_np)[:2]
    cert = GapCertification.from_energies(
        float(scores_np[top2[0]]),
        float(scores_np[top2[1]]),
        float(error_bound),
    )
    return cert.is_certified


def _shared_certified_singleton_top1(
    scores_a: jnp.ndarray | np.ndarray,
    error_bound_a: float | None,
    scores_b: jnp.ndarray | np.ndarray,
    error_bound_b: float | None,
) -> bool:
    """Both score families certify the same singleton top-1 winner.

    Lean: FLO22 (`exact_energy_gap_certified_choice_agreement`), combining TK16
    singleton certification for each score family with FLO21 witness agreement.
    """
    if scores_a.shape[0] == 0 or scores_b.shape[0] == 0:
        return False
    top1_a = int(jnp.argmin(scores_a))
    top1_b = int(jnp.argmin(scores_b))
    return (
        top1_a == top1_b
        and _certified_top1_gap(scores_a, error_bound_a)
        and _certified_top1_gap(scores_b, error_bound_b)
    )


@dataclass(frozen=True)
class CertifiedPoseGeneration:
    pose_vecs: PoseVector
    family: "CertifiedGlobalActionFamily | None"


BindingSiteT = TypeVar("BindingSiteT", bound=BindingSite)
DetectedPocketT = TypeVar("DetectedPocketT")
PlanT = TypeVar("PlanT", bound=BlindDockingPlan)


@dataclass(frozen=True, kw_only=True)
class BlindDockingPreparation:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    box: DockingBox


@dataclass(frozen=True, kw_only=True)
class CertifiedPocketPreparation(BlindDockingPreparation):
    detected_pocket: CertifiedDetectedPocket | None
    plan: CertifiedBlindDockingPlan


@dataclass(frozen=True, kw_only=True)
class GeometricPocketPreparation(BlindDockingPreparation):
    detected_pocket: GeometricDetectedPocket | None
    plan: GeometricBlindDockingPlan


def derive_seed_budget(
    confidence: float,
    box_size: jnp.ndarray,
    target_rmsd: float,
    ligand_radius: float,
    n_torsions: int = 0,
    probe_certificate: RefinementCertificate | None = None,
) -> int:
    """Derive the rigid initial-seed count from capture probability.

    Uses the SE(3) covering number argument: the search volume (translation,
    rotation, and optionally torsion for a *joint* sampler) divided by a
    conservative capture-basin volume gives the expected number of trials needed.
    The confidence level converts this to a concrete seed count via the
    geometric distribution CDF inversion:

        N ≥ ln(1 - P) / ln(1 - V_basin / V_total)

    Lean: SeedBudgetDerivation.lean — sufficient_seed_budget (SB2),
    minSeedBudget_antitone (SB5), composed_two_phase_seed_budget (SB7).

    In the current docking pipeline, random seeds are rigid SE(3) seeds *per
    conformer*; conformer search is budgeted separately by the branch-and-bound
    support theorems. That means runtime callers should pass `n_torsions=0`
    unless they truly sample the joint rigid+tortion space at this stage.

    Without a probe certificate, this uses the theorem-honest zero-step capture
    basin: a seed is a hit only if it already lies within `target_rmsd` of the
    optimum. With a probe certificate, we only expand this radius when the probe
    certifies both a coordinate-space lower curvature `μ`, an upper curvature
    `M`, and a linear contraction factor `q` over `T` steps. The certified
    larger capture radius is then

        capture_rmsd = target_rmsd * sqrt(μ / (M * q^T)).

    Lean: EnergyRMSDConvergence.lean —
      rmsd_target_of_initial_rmsd_and_linear_energy_convergence.

    This removes the old ad hoc 2Å / π/4 / π/6 constants. The probe-based
    protocol remains conservative by underestimating the true basin volume.
    """
    import math

    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if target_rmsd <= 0.0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    if n_torsions < 0:
        raise ValueError(f"n_torsions must be nonnegative, got {n_torsions}")

    capture_rmsd = target_rmsd
    if probe_certificate is not None and probe_certificate.n_steps > 0:
        q = float(probe_certificate.q)
        mu_coord = float(probe_certificate.spectral.mu_coord)
        M_coord = float(probe_certificate.spectral.M_coord)
        if (
            0.0 < q < 1.0
            and math.isfinite(mu_coord)
            and math.isfinite(M_coord)
            and mu_coord > 0.0
            and M_coord > 0.0
        ):
            contraction = math.pow(q, probe_certificate.n_steps)
            if contraction > 0.0 and math.isfinite(contraction):
                amplification_sq = mu_coord / (M_coord * contraction)
                if amplification_sq > 1.0 and math.isfinite(amplification_sq):
                    capture_rmsd = target_rmsd * math.sqrt(amplification_sq)

    # --- Search volumes ---

    # Translation: box volume in Å³
    box_vol = float(jnp.prod(box_size))

    # Rotation: SO(3) has volume 2π² ≈ 19.739
    so3_volume = 2.0 * math.pi**2

    # Torsion: [-π, π]^n_torsions
    torsion_search = (2.0 * math.pi) ** n_torsions if n_torsions > 0 else 1.0

    v_total = box_vol * so3_volume * torsion_search

    # --- Basin capture volumes ---
    # Translation: a pure translation by r shifts every atom by r, so RMSD=r.
    trans_capture = min(
        box_vol,
        (4.0 / 3.0) * math.pi * capture_rmsd**3,
    )

    # Rotation: a rigid rotation by angle θ moves atoms by at most ligand_radius * θ.
    if ligand_radius > 1e-6 and math.isfinite(capture_rmsd):
        rot_capture_radius = min(math.pi, capture_rmsd / ligand_radius)
        rot_capture = min(
            so3_volume,
            (4.0 / 3.0) * math.pi * rot_capture_radius**3,
        )
    else:
        rot_capture = so3_volume

    # Torsion: require every torsion component to fit inside an RMSD ball using
    # the ligand radius as a conservative arm-length bound.
    if n_torsions > 0:
        if ligand_radius > 1e-6 and math.isfinite(capture_rmsd):
            torsion_capture_radius = min(
                math.pi,
                capture_rmsd / (ligand_radius * math.sqrt(float(n_torsions))),
            )
            torsion_capture = min(
                torsion_search,
                (2.0 * torsion_capture_radius) ** n_torsions,
            )
        else:
            torsion_capture = torsion_search
    else:
        torsion_capture = 1.0

    v_basin = trans_capture * rot_capture * torsion_capture

    # --- Geometric CDF inversion (SB2/SB4) ---
    ratio = min(1.0, v_basin / v_total)
    if ratio >= 1.0:
        return 1  # basin covers the entire search space
    if ratio <= 0.0:
        raise ValueError("derived basin volume fraction must be positive")

    n_poses = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - ratio))
    return max(1, n_poses)


def _ligand_radius(ligand_ctx: LigandContext) -> float:
    # `base_coords` are already centered in build_ligand_context, so the ligand
    # radius is just the farthest atom from the origin in that centered frame.
    return float(jnp.max(jnp.linalg.norm(ligand_ctx.base_coords, axis=-1)))


def _fit_pose_vector_from_coords(
    base_coords: jnp.ndarray,
    world_coords: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recover the rigid translation/quaternion mapping `base_coords` to `world_coords`.

    Formal round refinement operates directly on coordinates. The theorem-backed
    SE(3) refinement layer expects the equivalent rigid pose vector, so we recover
    it deterministically with the Kabsch solution.
    """
    from scipy.spatial.transform import Rotation

    base_np = np.asarray(base_coords, dtype=np.float64)
    world_np = np.asarray(world_coords, dtype=np.float64)
    base_mean = base_np.mean(axis=0)
    world_mean = world_np.mean(axis=0)
    base_centered = base_np - base_mean
    world_centered = world_np - world_mean
    covariance = base_centered.T @ world_centered
    u, _, vt = np.linalg.svd(covariance)
    # `rigid_transform_3d` applies coordinates as `coords @ rotation.T + translation`.
    # For this row-vector convention, the Kabsch solution is R = V U^T.
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = world_mean - base_mean @ rotation.T
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    dtype = world_coords.dtype
    return (
        jnp.asarray(translation, dtype=dtype),
        jnp.asarray(quat_wxyz, dtype=dtype),
    )


def _fit_pose_vectors_from_coords_batch(
    base_coords: jnp.ndarray,
    world_coords_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    translations: list[jnp.ndarray] = []
    quaternions: list[jnp.ndarray] = []
    for coords in np.asarray(world_coords_batch):
        translation, quaternion = _fit_pose_vector_from_coords(
            base_coords,
            jnp.asarray(coords, dtype=world_coords_batch.dtype),
        )
        translations.append(translation)
        quaternions.append(quaternion)
    return jnp.stack(translations), jnp.stack(quaternions)


def _rotatable_bond_count(ligand_ctx: LigandContext) -> int:
    adjacency = ligand_ctx.adjacency
    elements = ligand_ctx.elements
    if adjacency is None or not elements:
        return 0
    coords_np = np.asarray(ligand_ctx.base_coords, dtype=np.float32)
    return len(detect_rotatable_bonds(adjacency, coords_np, elements))


def _min_pairwise_sigma_from_radii(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> float:
    if receptor_radii.size == 0 or ligand_radii.size == 0:
        raise ValueError("Cannot derive sigma without receptor and ligand radii")
    return float(jnp.min(receptor_radii) + jnp.min(ligand_radii))


def _derive_local_rotation_step_rad(base_step: float, ligand_radius: float) -> float:
    """Derive a local rigid-rotation step from the translational cell scale.

    The formal local optimizer uses rigid moves whose translation and
    rotation-induced pointwise displacements should live on the same geometric
    scale. If the translation shell starts at `base_step`, the matching rigid
    rotation shell is the angle whose maximal atomic displacement is also
    `base_step`, i.e. `angle = base_step / ligand_radius`.

    We cap at `pi/2`, the coarsest global-orientation shell already assumed by
    the certified SE(3) refinement round budget.
    """
    if base_step <= 0.0:
        raise ValueError(f"base_step must be positive, got {base_step}")
    if ligand_radius <= 1e-8:
        return float(jnp.pi / 2.0)
    return min(float(jnp.pi / 2.0), base_step / ligand_radius)


def _derive_realspace_ewald_lipschitz_constant(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> float:
    """Conservative pairwise Lipschitz bound for real-space Ewald electrostatics.

    For a single pair with charge product `Q` and kernel

        f(r) = Q * erfc(alpha * r) / (dielectric * r),

    the radial derivative satisfies

        |f'(r)| <= |Q| / dielectric * (2 * alpha / (sqrt(pi) * r) + 1 / r^2)

    because `exp(-(alpha r)^2) <= 1` and `erfc(alpha r) <= 1`. Summing the
    per-pair bounds gives a conservative coordinate-space Lipschitz constant for
    the rigid local-optimization objective.

    Closest Lean support today lives in
    `ScreenedCoulombApproximation.lean` / `ConditionalComposition.lean` for the
    exact screened-Coulomb kernel and cutoff/tail control; this explicit radial
    derivative enclosure is the runtime-side bridge still to mechanize.
    """
    if electrostatics is None:
        return 0.0
    r_min = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if r_min <= 0.0:
        raise ValueError(f"r_min must be positive, got {r_min}")
    charge_sum = float(
        jnp.sum(
            jnp.abs(
                electrostatics.receptor_charges[:, None]
                * electrostatics.ligand_charges[None, :]
            )
        )
    )
    if charge_sum <= 0.0:
        return 0.0
    alpha = float(electrostatics.alpha)
    dielectric = float(electrostatics.dielectric)
    if alpha <= 0.0 or dielectric <= 0.0:
        raise ValueError(
            f"real-space Ewald requires positive alpha/dielectric, got {alpha}, {dielectric}"
        )
    radial_bound = (2.0 * alpha) / (math.sqrt(math.pi) * r_min) + 1.0 / (r_min**2)
    return (charge_sum / dielectric) * radial_bound


def _derive_optimization_score_lipschitz_constant(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
) -> float:
    """Physics-derived Lipschitz constant for the formal local optimizer score."""
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
    )
    lj_lipschitz = compute_raw_lj_lipschitz(_EPSILON_KCAL_MOL, min_sigma)
    electro_lipschitz = _derive_realspace_ewald_lipschitz_constant(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
        scoring_context.electrostatics,
    )
    return lj_lipschitz + electro_lipschitz


def _dyadic_action_path_displacement_bound(
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
) -> float:
    """Worst-case pointwise displacement over a dyadic rigid local-search path.

    One certified local action is taken per round. The action family offers
    either a pure translation step or a pure rotation step, so the maximal
    pointwise displacement at round `r` is

        max(trans_r, ligand_radius * rot_r)

    rather than their sum. Summing that dyadic schedule over the remaining
    rounds yields a theorem-aligned bound on total coordinate motion available to
    the local optimizer.
    """
    if base_translation_step <= 0.0:
        raise ValueError(
            f"base_translation_step must be positive, got {base_translation_step}"
        )
    if n_rounds <= 0:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}")
    base_rotation_displacement = ligand_radius * base_rotation_step_rad
    base_round_displacement = max(base_translation_step, base_rotation_displacement)
    return float(
        sum(
            base_round_displacement / (2.0**round_index)
            for round_index in range(n_rounds)
        )
    )


def _rigid_local_improvement_bound(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
    *,
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
) -> float:
    """Bound the total possible score improvement from rigid local refinement.

    Runtime composition of existing proof families:

      - `LipschitzStepBounds.lean::descent_bounded_change`
      - `LipschitzStepBounds.lean::n_step_error_bound`
      - `SupportExpansion.lean::dyadicTranslationStep`

    applied to the exact optimization score and the dyadic rigid-action family.
    """
    lipschitz = _derive_optimization_score_lipschitz_constant(request, scoring_context)
    path_displacement = _dyadic_action_path_displacement_bound(
        base_translation_step,
        base_rotation_step_rad,
        ligand_radius,
        n_rounds,
    )
    return lipschitz * path_displacement


def _derive_exact_score_lipschitz_constant(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    exact_chemistry_mode: ExactChemistryMode,
    softening_policy: SofteningPolicy,
) -> float:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if (
        exact_chemistry_mode == ExactChemistryMode.EXTENDED_RICH
        and softening_policy != SofteningPolicy.NONE
    ):
        return 24.0 * _EPSILON_KCAL_MOL / min_sigma
    return compute_raw_lj_lipschitz(_EPSILON_KCAL_MOL, min_sigma)


def _derive_target_error_from_rmsd(
    target_rmsd: float,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    exact_chemistry_mode: ExactChemistryMode,
    softening_policy: SofteningPolicy,
) -> float:
    """Derive a certified score-resolution budget from target RMSD.

    Lean: CSC48 (`energyBudget_of_rmsdTarget`).
    Using a certified physical Lipschitz constant `L`, an RMSD tolerance `eps`
    induces the energy resolution budget `L * eps`.
    """
    if target_rmsd <= 0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    return (
        _derive_exact_score_lipschitz_constant(
            receptor_radii,
            ligand_radii,
            exact_chemistry_mode,
            softening_policy,
        )
        * target_rmsd
    )


def _resolve_softening_radius(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    *,
    policy: SofteningPolicy,
    mode: DockingMode,
    heuristic_ratio: float,
) -> float | None:
    sigma_min = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if policy == SofteningPolicy.NONE:
        return None
    if policy in (
        SofteningPolicy.CANONICAL_MAX_SIGMA,
        SofteningPolicy.DERIVED_FROM_ERROR_BUDGET,
    ):
        return sigma_min
    if mode == DockingMode.CERTIFIED:
        raise ValueError("Certified mode does not allow EMPIRICAL_RATIO softening")
    if heuristic_ratio <= 0:
        raise ValueError(
            f"heuristic_softening_ratio must be positive, got {heuristic_ratio}"
        )
    return sigma_min * heuristic_ratio


def _seed_budget_torsion_count(request: "DockingRequestBase") -> int:
    """Count torsions included in the rigid seed-budget volume model.

    The current pipeline samples rigid SE(3) seeds first and performs conformer
    search only after a rigid pose has already been selected/refined. Theorem
    `derive_seed_budget()` therefore applies *per conformer*, not over the full
    joint rigid+torsion volume, so torsional degrees of freedom must not inflate
    the initial rigid seed count.

    Keeping this at zero is what prevents flexible ligands like `1hk4` from
    exploding into billions of rigid seeds before the separate conformer-search
    budget even runs.
    """
    del request
    return 0


def _strictly_smaller_positive(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    strict_value = math.nextafter(value, 0.0)
    if not math.isfinite(strict_value) or strict_value <= 0.0:
        raise ValueError(
            f"{name}={value} is too small to derive a strict positive bound"
        )
    return strict_value


def _argmax_subdivision_convergence_max_cells(
    per_bond_lipschitz: tuple[float, ...],
    target_slack: float,
) -> int:
    """Bound the full argmax-subdivision tree size from the CS14 contraction rate.

    CS14 gives a worst-case one-step contraction factor

        slack' <= (1 - 1 / (2m)) * slack

    for `m` active torsions when we always bisect the coordinate with maximal
    weighted-L1 slack contribution. A complete binary tree of depth `d` therefore
    suffices once the root slack `pi * sum(L_i)` contracts below `target_slack`.
    """
    if target_slack <= 0.0:
        raise ValueError(f"target_slack must be positive, got {target_slack}")
    if not per_bond_lipschitz:
        return 1
    n_active = len(per_bond_lipschitz)
    initial_slack = math.pi * sum(max(0.0, float(li)) for li in per_bond_lipschitz)
    if initial_slack <= target_slack:
        return 1
    contraction = 1.0 - 1.0 / (2.0 * n_active)
    if not (0.0 < contraction < 1.0):
        raise ValueError(f"invalid argmax-subdivision contraction factor {contraction}")
    depth = max(
        0,
        int(
            math.ceil(
                math.log(initial_slack / target_slack) / math.log(1.0 / contraction)
            )
        ),
    )
    while initial_slack * (contraction**depth) > target_slack:
        depth += 1
    return max(1, (1 << (depth + 1)) - 1)


def _derive_adaptive_torsion_support_spec(
    per_bond_lipschitz: tuple[float, ...],
    target_delta: float,
    target_rmsd: float,
    max_arm: float,
) -> tuple[int, float, tuple[int, ...]]:
    """Derive theorem-backed torsion search budgets from slack and radius certificates.

    The runtime stopping radius must align with the actual B&B cell metric:
    `TorsionCell.radius()` is an L2 radius, so the old `min(half_widths)` derivation
    did not match the Lean `bb_stopping_radius_yields_coverage` / CSC50 bridge.

    We therefore derive the stopping radius from the composed parameter-space
    Lipschitz constant,

        min_cell_radius < target_error / L_param,

    where `L_param = max_i L_i = score_lipschitz * kinematics_lipschitz`.
    When `target_error` itself came from CSC48 (`target_error = score_lipschitz *
    target_rmsd`), this is exactly the CSC50 rule `target_rmsd / K`.

    To turn that stopping radius into a finite worst-case tree budget, we tighten to
    a weighted-L1 slack target that is strict enough to force the L2 cell radius below
    `min_cell_radius`, then take the tighter of:

      - CSC44's support-cardinality bound from canonical adaptive segments, and
      - CS14's geometric argmax-subdivision contraction bound.

    The strict `math.nextafter(..., 0.0)` steps are only there to match the runtime's
    strict `< config.min_cell_radius` stopping predicate without overshooting the
    certified radius.
    """
    if target_delta <= 0.0:
        raise ValueError(f"target_delta must be positive, got {target_delta}")
    if not per_bond_lipschitz:
        return 1, float(2.0 * np.pi), ()
    if target_rmsd <= 0.0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    if max_arm <= 0.0:
        raise ValueError(f"max_arm must be positive, got {max_arm}")

    lipschitz = tuple(max(0.0, float(li)) for li in per_bond_lipschitz)
    max_lipschitz = max(lipschitz)
    min_lipschitz = min(lipschitz)
    if max_lipschitz <= 0.0:
        raise ValueError(
            "per_bond_lipschitz must contain a positive composed Lipschitz constant"
        )
    if min_lipschitz <= 0.0:
        raise ValueError(
            "all active torsions must have positive per-bond Lipschitz constants"
        )

    raw_min_cell_radius = min(target_delta / max_lipschitz, target_rmsd / max_arm)
    min_cell_radius = _strictly_smaller_positive(
        raw_min_cell_radius,
        name="min_cell_radius",
    )
    support_target_delta = _strictly_smaller_positive(
        min(target_delta, min_lipschitz * min_cell_radius),
        name="support_target_delta",
    )

    n_active = len(lipschitz)
    segments = tuple(
        max(
            1,
            int(math.ceil((2.0 * math.pi * n_active * li) / support_target_delta)),
        )
        for li in lipschitz
    )
    support_size = 1
    for seg in segments:
        support_size *= seg + 1
    # Theorem CSC44: a full binary refinement tree with `support_size` certified
    # leaf cells evaluates at most `2 * support_size - 1` cell centers.
    support_max_cells = max(1, 2 * support_size - 1)
    convergence_max_cells = _argmax_subdivision_convergence_max_cells(
        lipschitz,
        support_target_delta,
    )
    max_cells = min(support_max_cells, convergence_max_cells)

    return max_cells, min_cell_radius, segments


def _active_rotatable_bonds_for_pose(
    world_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float = 6.0,
) -> tuple[tuple[RotatableBond, ...], np.ndarray]:
    if not rotatable_bonds:
        return (), np.zeros((0,), dtype=bool)
    active_mask = np.asarray(
        jax.device_get(
            _posewise_active_torsion_mask(
                poses_coords=jnp.expand_dims(world_coords, axis=0),
                receptor_coords=receptor_coords,
                bonds=rotatable_bonds,
                scoring_cutoff=scoring_cutoff,
            )[0]
        ),
        dtype=bool,
    )
    active_bonds = tuple(
        bond for bond, is_active in zip(rotatable_bonds, active_mask) if is_active
    )
    return active_bonds, active_mask


def _restrict_strain_params(
    strain_params: TorsionStrainParams | None,
    active_mask: np.ndarray,
) -> TorsionStrainParams | None:
    def _empty_params() -> TorsionStrainParams:
        return TorsionStrainParams(
            barrier_heights=jnp.zeros((0,), dtype=jnp.float32),
            multiplicities=jnp.zeros((0,), dtype=jnp.float32),
            phases=jnp.zeros((0,), dtype=jnp.float32),
        )

    if strain_params is None or active_mask.size == 0:
        return None if strain_params is None else _empty_params()
    if not np.any(active_mask):
        return _empty_params()
    idx = np.flatnonzero(active_mask)
    return TorsionStrainParams(
        barrier_heights=jnp.asarray(strain_params.barrier_heights)[idx],
        multiplicities=jnp.asarray(strain_params.multiplicities)[idx],
        phases=jnp.asarray(strain_params.phases)[idx],
    )


def _probe_seed_budget_certificate(
    request: "PipelineDockingRequest",
    route: "PipelineRoute",
) -> tuple["PipelineDockingRequest", RefinementCertificate | None]:
    from dq_dock_engine.docking.placement import apply_poses

    probe_request = cast(
        "PipelineDockingRequest",
        request.with_updates(n_poses_override=SEED_BUDGET_PROBE_POSES),
    )
    probe_batch = route.generate_pose_batch(probe_request)
    probe_request = probe_batch.request
    probe_coords = apply_poses(probe_request.ligand_ctx, probe_batch.pose_vecs)
    probe_scores = route.score_pose_batch(
        probe_request, probe_coords, probe_batch.pose_vecs
    ).final_scores
    probe_scores_np = np.asarray(jax.device_get(probe_scores), dtype=np.float64)
    ranked = np.argsort(probe_scores_np)

    best_cert: RefinementCertificate | None = None
    best_seed_budget: int | None = None
    n_torsions = _seed_budget_torsion_count(probe_request)
    ligand_radius = _ligand_radius(probe_request.ligand_ctx)

    for idx in ranked[: min(SEED_BUDGET_PROBE_TOP_K, ranked.shape[0])]:
        opt_t, opt_q, certs = _certified_refinement(
            request=probe_request,
            initial_translations=probe_batch.pose_vecs.translation[idx : idx + 1],
            initial_quaternions=probe_batch.pose_vecs.quaternion[idx : idx + 1],
            mode_override=RefinementCertificationMode.OBSERVED,
        )
        del opt_t, opt_q
        cert = certs[0]
        if cert is None:
            continue
        derived_budget = derive_seed_budget(
            confidence=cast(DockingConfig, probe_request.config).confidence,
            box_size=probe_request.box.size,
            target_rmsd=cast(DockingConfig, probe_request.config).target_rmsd,
            ligand_radius=ligand_radius,
            n_torsions=n_torsions,
            probe_certificate=cert,
        )
        # SB7 only needs one conservative basin underestimate. Among multiple
        # successful probe certificates, keep the largest certified basin, i.e.
        # the smallest derived full-run seed budget.
        if best_seed_budget is None or derived_budget < best_seed_budget:
            best_seed_budget = derived_budget
            best_cert = cert

    if best_seed_budget is None:
        return request, None
    return cast(
        "PipelineDockingRequest",
        probe_request.with_updates(n_poses_override=best_seed_budget),
    ), best_cert


@dataclass(frozen=True, kw_only=True)
class DockingRequestBase:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    ligand_ctx: LigandContext
    box: DockingBox
    n_poses_override: int | None = None
    key: jax.Array
    receptor_elements: tuple[str, ...] | None = None
    charge_method: ChargeMethod | None = None
    receptor_file: str | Path | None = None
    ligand_source_path: str | Path | None = None
    precomputed_receptor_charges: jnp.ndarray | None = None
    config: DockingConfig | None = None
    optimize: bool = True
    include_native: bool = False
    scoring_kwargs: dict[str, object] = field(default_factory=dict)

    def with_updates(self, **changes: object) -> Self:
        return replace(self, **changes)

    @property
    def normalized_key(self) -> jax.Array:
        return _normalize_sampling_key(self.key)

    @property
    def n_poses(self) -> int:
        """Seed budget: explicit if set, otherwise derived from confidence."""
        if self.n_poses_override is not None:
            return self.n_poses_override
        config = self.config
        if config is None:
            raise ValueError(
                "n_poses is None and no DockingConfig provided — "
                "cannot derive seed budget without confidence + target_rmsd"
            )
        ligand_radius = _ligand_radius(self.ligand_ctx)
        return derive_seed_budget(
            confidence=config.confidence,
            box_size=self.box.size,
            target_rmsd=config.target_rmsd,
            ligand_radius=ligand_radius,
            n_torsions=_seed_budget_torsion_count(self),
        )

    @property
    def resolved_target_error(self) -> float:
        if self.config is None:
            raise ValueError("Certified docking requires an explicit docking config")
        if self.config.target_error <= 0:
            return _derive_target_error_from_rmsd(
                self.config.target_rmsd,
                self.receptor_radii,
                self.ligand_radii,
                self.config.exact_chemistry_mode,
                self.config.softening_policy,
            )
        return self.config.target_error

    @property
    def target_error(self) -> float:
        return self.resolved_target_error

    @property
    def certified_binding_site(self) -> CertifiedBindingSite | None:
        return None if self.config is None else self.config.certified_binding_site

    @property
    def coarse_target_error(self) -> float:
        if self.config is None or self.config.coarse_target_error <= 0:
            return self.target_error
        return self.config.coarse_target_error

    @property
    def adaptive_coarse_target_errors(self) -> tuple[float, ...] | None:
        if self.config is None:
            return (self.coarse_target_error,)
        if self.config.adaptive_coarse_target_errors is None:
            return (self.coarse_target_error,)
        return self.config.adaptive_coarse_target_errors

    @property
    def use_softened_coarse_prefilter(self) -> bool:
        if self.config is None:
            return False
        if self.config.mode == DockingMode.CERTIFIED:
            return self.config.softening_policy != SofteningPolicy.NONE
        return self.config.use_softened_coarse_prefilter or (
            self.config.softening_policy != SofteningPolicy.NONE
        )

    @property
    def softening_radius(self) -> float | None:
        if self.config is None:
            return None
        return _resolve_softening_radius(
            self.receptor_radii,
            self.ligand_radii,
            policy=self.config.softening_policy,
            mode=self.config.mode,
            heuristic_ratio=self.config.heuristic_softening_ratio,
        )

    @property
    def reuse_initial_conformer(self) -> bool:
        return False if self.config is None else self.config.reuse_initial_conformer

    @property
    def target_rmsd(self) -> float:
        if self.config is None:
            raise ValueError(
                "target_rmsd is only defined when a docking config is present"
            )
        return self.config.target_rmsd

    @property
    def receptor_coords(self) -> jnp.ndarray:
        return self.protein_coords

    @property
    def ligand_radii(self) -> jnp.ndarray:
        return self.ligand_ctx.base_radii


@dataclass(frozen=True, kw_only=True)
class BlindDockingRequest(DockingRequestBase):
    pass


@dataclass(frozen=True, kw_only=True)
class CertifiedBlindDockingRequest(BlindDockingRequest):
    pass


@dataclass(frozen=True, kw_only=True)
class RoutedDockingRequest(DockingRequestBase):
    engine: ScoringEngine = ScoringEngine.INTERNAL_LJ


@dataclass(frozen=True, kw_only=True)
class GeometricBlindDockingRequest(RoutedDockingRequest):
    pass


@dataclass(frozen=True, kw_only=True)
class BlindDockingPreparationRequest:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    ligand_ctx: LigandContext
    box: DockingBox
    target_error: float


@dataclass(frozen=True, kw_only=True)
class CertifiedPreparationRequest(BlindDockingPreparationRequest):
    explicit_binding_site: CertifiedBindingSite | None = None
    coarse_target_error: float = 0.0
    adaptive_coarse_target_errors: tuple[float, ...] | None = None
    use_softened_coarse_prefilter: bool = False

    @classmethod
    def from_request(
        cls, request: "PipelineDockingRequest | CertifiedBlindDockingRequest"
    ) -> "CertifiedPreparationRequest":
        return derive_request(
            cls,
            request,
            target_error=request.target_error,
            explicit_binding_site=request.certified_binding_site,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=request.use_softened_coarse_prefilter,
        )


@dataclass(frozen=True, kw_only=True)
class GeometricPreparationRequest(BlindDockingPreparationRequest):
    @classmethod
    def from_request(
        cls, request: "PipelineDockingRequest | GeometricBlindDockingRequest"
    ) -> "GeometricPreparationRequest":
        return derive_request(
            cls,
            request,
            target_error=request.target_error,
        )


@dataclass(frozen=True, kw_only=True)
class PipelineDockingRequest(RoutedDockingRequest):
    use_pocket_guided: bool = True
    use_multi_stage: bool = False
    certified_pocket_prep: CertifiedPocketPreparation | None = None

    @property
    def is_certified_mode(self) -> bool:
        return self.config is not None and self.config.mode == DockingMode.CERTIFIED

    @property
    def requires_fixed_size_padding(self) -> bool:
        return self.config is not None and self.config.mode != DockingMode.CERTIFIED

    @property
    def certified_scoring_family(self) -> CertifiedScoringFamily | None:
        return None if self.config is None else self.config.certified_scoring_family

    @property
    def effective_engine(self) -> ScoringEngine:
        if not self.is_certified_mode:
            return self.engine
        if self.certified_scoring_family == CertifiedScoringFamily.LJ:
            return ScoringEngine.CERTIFIED_LJ
        return ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD

    @property
    def formal_backend(self) -> OptimizerBackend:
        return (
            self.config.optimizer_backend
            if self.config is not None
            else OptimizerBackend.GRADIENT
        )

    @property
    def formal_round_strategy(self) -> FormalRoundStrategy:
        return (
            self.config.formal_round_strategy
            if self.config is not None
            else FormalRoundStrategy.SINGLETON_HYBRID
        )

    @property
    def exact_chemistry_mode(self) -> ExactChemistryMode:
        return (
            self.config.exact_chemistry_mode
            if self.config is not None
            else ExactChemistryMode.NONE
        )

    def with_scoring_override(
        self, **scoring_overrides: object
    ) -> "PipelineDockingRequest":
        return self.with_updates(
            scoring_kwargs=dict(self.scoring_kwargs) | dict(scoring_overrides)
        )

    def with_preparation(
        self,
        prep: BlindDockingPreparation,
        *,
        certified_pocket_prep: CertifiedPocketPreparation | None = None,
    ) -> "PipelineDockingRequest":
        return self.with_updates(
            protein_coords=prep.protein_coords,
            receptor_radii=prep.receptor_radii,
            receptor_elements=prep.receptor_elements,
            precomputed_receptor_charges=prep.precomputed_receptor_charges,
            box=prep.box,
            certified_pocket_prep=certified_pocket_prep,
        )

    def with_fixed_size_padding(self) -> "PipelineDockingRequest":
        if not self.requires_fixed_size_padding:
            return self
        assert self.config is not None
        padded_receptor_charges = self.precomputed_receptor_charges
        if padded_receptor_charges is not None:
            padded_receptor_charges = _pad_to_size(
                padded_receptor_charges,
                self.config.max_receptor_atoms,
                axis=0,
                value=0.0,
            )
        padded_ligand_elements = _pad_tuple_to_size(
            self.ligand_ctx.elements,
            self.config.max_ligand_atoms,
            value="C",
        )
        padded_ligand_charges = None
        if self.ligand_ctx.charges is not None:
            padded_ligand_charges = _pad_to_size(
                self.ligand_ctx.charges,
                self.config.max_ligand_atoms,
                axis=0,
                value=0.0,
            )
        return self.with_updates(
            protein_coords=_pad_to_size(
                self.protein_coords,
                self.config.max_receptor_atoms,
                axis=0,
                value=1e4,
            ),
            receptor_radii=_pad_to_size(
                self.receptor_radii,
                self.config.max_receptor_atoms,
                axis=0,
                value=0.0,
            ),
            receptor_elements=_pad_tuple_to_size(
                self.receptor_elements,
                self.config.max_receptor_atoms,
                value="C",
            ),
            precomputed_receptor_charges=padded_receptor_charges,
            ligand_ctx=LigandContext(
                base_coords=_pad_to_size(
                    self.ligand_ctx.base_coords,
                    self.config.max_ligand_atoms,
                    axis=0,
                    value=1e4,
                ),
                base_radii=_pad_to_size(
                    self.ligand_ctx.base_radii,
                    self.config.max_ligand_atoms,
                    axis=0,
                    value=0.0,
                ),
                elements=()
                if padded_ligand_elements is None
                else padded_ligand_elements,
                charges=padded_ligand_charges,
                center_of_mass=self.ligand_ctx.center_of_mass,
            ),
        )


class RequestMatchBase:
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        del request
        return True


class CertifiedModeMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.is_certified_mode


class NonCertifiedModeMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.is_certified_mode


class GuidedSamplingMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.use_pocket_guided


class BoxSamplingMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.use_pocket_guided


class DirectScoringMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.use_multi_stage


class MultiStageScoringMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.use_multi_stage


class CertifiedPreparationMixin:
    def prepare(self) -> "PreparedCertifiedDirectPipelineRequest":
        request = cast(Any, self)
        prep = request.certified_pocket_prep
        if prep is None:
            prep = _prepare_certified_blind_docking(
                CertifiedPreparationRequest.from_request(request)
            )
        return derive_request(
            PreparedCertifiedDirectPipelineRequest,
            request.with_preparation(prep, certified_pocket_prep=prep),
            certified_pocket_prep=prep,
        )


class GeometricPreparationMixin:
    def prepare(self) -> "PreparedGeometricPipelineRequest":
        request = cast(Any, self)
        prep = _prepare_geometric_blind_docking(
            GeometricPreparationRequest.from_request(request)
        )
        return derive_request(
            PreparedGeometricPipelineRequest,
            request.with_preparation(prep),
            geometric_pocket_prep=prep,
            sampling_plan=derive_geometric_sampling_plan(prep),
        )


@dataclass(frozen=True, kw_only=True)
class NominalPipelineDockingRequest(RequestMatchBase, PipelineDockingRequest):
    route_type_name: ClassVar[str | None] = None
    _registered_types: ClassVar[list[type["NominalPipelineDockingRequest"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("route_type_name") is not None:
            cls._registered_types.append(cls)

    @classmethod
    def from_request(
        cls, request: PipelineDockingRequest
    ) -> "NominalPipelineDockingRequest":
        matches = [
            candidate
            for candidate in cls._registered_types
            if candidate.matches_request(request)
        ]
        if not matches:
            raise ValueError(
                "CERTIFIED mode does not support the heuristic multi-stage scoring pipeline."
            )
        if len(matches) != 1:
            raise TypeError(
                f"Ambiguous nominal pipeline request refinement for {type(request).__name__}: {[candidate.__name__ for candidate in matches]}"
            )
        return derive_request(matches[0], request)

    def create_route(self) -> "PipelineRoute":
        if self.route_type_name is None:
            raise TypeError(
                f"Nominal request type {type(self).__name__} does not declare a route type."
            )
        return cast(type[PipelineRoute], globals()[self.route_type_name])()


@dataclass(frozen=True, kw_only=True)
class DirectPipelineDockingRequest(DirectScoringMatch, NominalPipelineDockingRequest):
    use_multi_stage: bool = False


@dataclass(frozen=True, kw_only=True)
class MultiStagePipelineDockingRequest(
    MultiStageScoringMatch, NominalPipelineDockingRequest
):
    use_multi_stage: bool = True
    charge_method: ChargeMethod | None = None

    def __post_init__(self) -> None:
        _require_present_fields(self, "charge_method")


@dataclass(frozen=True, kw_only=True)
class CertifiedDirectPipelineRequest(
    CertifiedPreparationMixin,
    CertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "CertifiedPipelineRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class GeometricDirectPipelineRequest(
    GeometricPreparationMixin,
    GuidedSamplingMatch,
    NonCertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "GeometricPocketRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class BoxDirectPipelineRequest(
    BoxSamplingMatch,
    NonCertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "BoxSamplingRoute"
    use_pocket_guided: bool = False


@dataclass(frozen=True, kw_only=True)
class GeometricMultiStagePipelineRequest(
    GeometricPreparationMixin,
    GuidedSamplingMatch,
    NonCertifiedModeMatch,
    MultiStagePipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "GeometricMultiStageRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class BoxMultiStagePipelineRequest(
    BoxSamplingMatch,
    NonCertifiedModeMatch,
    MultiStagePipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "BoxMultiStageRoute"
    use_pocket_guided: bool = False


def _require_present_fields(instance: object, *field_names: str) -> None:
    missing = [name for name in field_names if getattr(instance, name) is None]
    if missing:
        raise ValueError(
            f"{type(instance).__name__} requires non-null fields: {', '.join(missing)}"
        )


@dataclass(frozen=True, kw_only=True)
class PreparedCertifiedDirectPipelineRequest(CertifiedDirectPipelineRequest):
    certified_pocket_prep: CertifiedPocketPreparation | None = None

    def __post_init__(self) -> None:
        _require_present_fields(self, "certified_pocket_prep")

    def prepare(self) -> "PreparedCertifiedDirectPipelineRequest":
        return self


class GeometricSamplingPlan(ABC):
    @abstractmethod
    def sample(
        self, request: "PreparedGeometricPipelineRequest"
    ) -> tuple[jax.Array, PoseVector]:
        """Sample a pose batch for the prepared geometric request."""


class DerivedSamplingPlan(GeometricSamplingPlan, ABC):
    @property
    @abstractmethod
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        """Concrete sampler function."""

    def sample(
        self, request: "PreparedGeometricPipelineRequest"
    ) -> tuple[jax.Array, PoseVector]:
        return call_with_derived_kwargs(
            self.sampler,
            request,
            aliases=None,
            **self.sampling_kwargs(),
        )

    def sampling_kwargs(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, kw_only=True)
class PreparedGeometricPipelineRequest(PipelineDockingRequest):
    geometric_pocket_prep: GeometricPocketPreparation
    sampling_plan: GeometricSamplingPlan

    def prepare(self) -> "PreparedGeometricPipelineRequest":
        return self

    def sample_pose_batch(self) -> tuple[jax.Array, PoseVector]:
        return self.sampling_plan.sample(self)


@dataclass(frozen=True)
class PocketGuidedSamplingPlan(DerivedSamplingPlan):
    geometric_detected_pocket: GeometricDetectedPocket

    @property
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        return _sample_geometric_pocket_guided_pose_vectors

    def sampling_kwargs(self) -> dict[str, object]:
        return {"geometric_detected_pocket": self.geometric_detected_pocket}


@dataclass(frozen=True)
class BoxFallbackSamplingPlan(DerivedSamplingPlan):
    @property
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        return _sample_box_guided_pose_vectors


def derive_geometric_sampling_plan(
    prep: GeometricPocketPreparation,
) -> GeometricSamplingPlan:
    if prep.detected_pocket is None:
        return BoxFallbackSamplingPlan()
    return PocketGuidedSamplingPlan(geometric_detected_pocket=prep.detected_pocket)


RequestTypeT = TypeVar("RequestTypeT")


def derive_request_kwargs(
    request_type: type[RequestTypeT],
    source: object | dict[str, Any],
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if isinstance(source, dict):
        source_values = source
    elif is_dataclass(source):
        source_values = {
            field.name: getattr(source, field.name)
            for field in dataclass_fields(source)
        }
    else:
        raise TypeError(
            f"Cannot derive {request_type.__name__} from non-dataclass source {type(source).__name__}."
        )
    derived = {
        field.name: source_values[field.name]
        for field in dataclass_fields(cast(Any, request_type))
        if field.init and field.name in source_values
    }
    derived.update(overrides)
    return derived


def derive_request(
    request_type: type[RequestTypeT],
    source: object | dict[str, Any],
    /,
    **overrides: Any,
) -> RequestTypeT:
    return request_type(**derive_request_kwargs(request_type, source, **overrides))


def derive_callable_kwargs(
    func: Callable[..., Any],
    source: object | dict[str, Any],
    /,
    *,
    aliases: dict[str, str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    alias_map = {} if aliases is None else aliases
    signature = inspect.signature(func)
    if isinstance(source, dict):
        source_values = source
    elif is_dataclass(source):
        source_values = {
            field.name: getattr(source, field.name)
            for field in dataclass_fields(source)
        }
    else:
        source_values = None
    kwargs: dict[str, Any] = {}
    accepts_var_keyword = False
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,):
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        if name in overrides:
            kwargs[name] = overrides[name]
            continue
        source_name = alias_map.get(name, name)
        if source_values is not None and source_name in source_values:
            kwargs[name] = source_values[source_name]
            continue
        if (
            not hasattr(source, source_name)
            and parameter.default is not inspect.Parameter.empty
        ):
            continue
        kwargs[name] = getattr(source, source_name)
    if accepts_var_keyword:
        for name, value in overrides.items():
            if name not in kwargs:
                kwargs[name] = value
    return kwargs


def call_with_derived_kwargs(
    func: Callable[..., Any],
    source: object | dict[str, Any],
    /,
    *,
    aliases: dict[str, str] | None = None,
    **overrides: Any,
) -> Any:
    return func(**derive_callable_kwargs(func, source, aliases=aliases, **overrides))


def resolve_request_electrostatics(
    request: RoutedDockingRequest,
    *,
    engine: ScoringEngine | None = None,
) -> CertifiedRealSpaceEwaldSpec | None:
    return call_with_derived_kwargs(
        _resolve_route_scoring_electrostatics,
        request,
        engine=request.engine if engine is None else engine,
    )


def resolve_request_scoring_context(
    request: RoutedDockingRequest,
    *,
    engine: ScoringEngine | None = None,
) -> CertifiedScoringContext:
    effective_engine = request.engine if engine is None else engine
    electrostatics = resolve_request_electrostatics(request, engine=effective_engine)
    if not isinstance(request, PipelineDockingRequest):
        return CertifiedScoringContext(
            exact_chemistry_mode=ExactChemistryMode.NONE,
            electrostatics=electrostatics,
            rich_chemistry_plan=None,
        )
    return build_certified_scoring_context(
        exact_chemistry_mode=request.exact_chemistry_mode,
        electrostatics=electrostatics,
        receptor_coords=request.protein_coords,
        receptor_radii=request.receptor_radii,
        receptor_elements=request.receptor_elements,
        receptor_file=request.receptor_file,
        ligand_source_path=request.ligand_source_path,
        ligand_ctx=request.ligand_ctx,
        target_electrostatic_error=request.target_error,
    )


def derive_route_scoring_kwargs(
    request: RoutedDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    engine: ScoringEngine | None = None,
    **extra_overrides: object,
) -> dict[str, Any]:
    return {
        "engine": request.engine if engine is None else engine,
        "receptor_coords": request.receptor_coords,
        "receptor_radii": request.receptor_radii,
        "ligand_radii": request.ligand_radii,
        "poses_coords": poses_coords,
        "electrostatics": electrostatics,
        **dict(request.scoring_kwargs),
        **dict(extra_overrides),
    }


def _ligand_extent_radius(ligand_ctx: LigandContext) -> float:
    centered = ligand_ctx.base_coords - ligand_ctx.center_of_mass
    if centered.shape[0] == 0:
        return 0.0
    return float(jnp.max(jnp.linalg.norm(centered, axis=1)))


def _apply_binding_site_restriction(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    binding_site: BindingSite,
    target_error: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    tuple[str, ...] | None,
    jnp.ndarray | None,
    DockingBox,
]:
    interaction_cutoff = compute_certified_cutoff(target_error)
    restriction_radius = (
        binding_site.radius + interaction_cutoff + _ligand_extent_radius(ligand_ctx)
    )
    distances = jnp.linalg.norm(protein_coords - binding_site.center, axis=1)
    keep_mask = distances <= restriction_radius
    if not bool(jnp.any(keep_mask)):
        return (
            protein_coords,
            receptor_radii,
            receptor_elements,
            precomputed_receptor_charges,
            box,
        )

    kept_indices = jnp.nonzero(keep_mask, size=int(jnp.sum(keep_mask)))[0]
    restricted_coords = protein_coords[kept_indices]
    restricted_radii = receptor_radii[kept_indices]
    restricted_elements = (
        None
        if receptor_elements is None
        else tuple(receptor_elements[int(i)] for i in np.asarray(kept_indices))
    )
    restricted_charges = (
        None
        if precomputed_receptor_charges is None
        else precomputed_receptor_charges[kept_indices]
    )
    restricted_box = DockingBox(
        center=binding_site.center,
        size=jnp.full((3,), 2.0 * binding_site.radius),
    )
    return (
        restricted_coords,
        restricted_radii,
        restricted_elements,
        restricted_charges,
        restricted_box,
    )


def _derive_certified_binding_site_from_box(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    box: DockingBox,
) -> tuple[CertifiedDetectedPocket | None, CertifiedPocketFailureReason | None]:
    region = extract_local_pocket_region_view(
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        box_center=box.center,
        box_size=float(jnp.max(box.size)),
    )
    if region.coords.shape[0] == 0:
        return None, CertifiedPocketFailureReason.NO_LOCAL_REGION
    pocket_radii = receptor_radii[region.indices]
    pocket = detect_certified_pocket(
        region.coords, region.elements, pocket_radii=pocket_radii
    )
    if pocket is None:
        return None, CertifiedPocketFailureReason.NO_CERTIFIED_POCKET
    return pocket, None


def _derive_geometric_pocket_from_box(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    box: DockingBox,
) -> GeometricDetectedPocket | None:
    region = extract_local_pocket_region_view(
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        box_center=box.center,
        box_size=float(jnp.max(box.size)),
    )
    pocket_radii = receptor_radii[region.indices]
    return detect_geometric_pocket(
        region.coords,
        region.elements,
        pocket_radii=pocket_radii,
    )


def _create_certified_pose_vectors(
    box: DockingBox,
    n_poses: int,
    certified_binding_site: CertifiedBindingSite | None,
) -> CertifiedPoseGeneration:
    from dq_dock_engine.docking.formal_sampling import (
        create_certified_binding_site_action_family,
        create_certified_global_action_family,
    )

    certified_family = (
        create_certified_global_action_family(box, n_poses)
        if certified_binding_site is None
        else create_certified_binding_site_action_family(
            certified_binding_site, n_poses
        )
    )
    pose_vecs = PoseVector(
        translation=certified_family.translations,
        quaternion=certified_family.quaternions,
    )
    return CertifiedPoseGeneration(pose_vecs=pose_vecs, family=certified_family)


def _sample_box_guided_pose_vectors(
    key: jax.Array,
    box: DockingBox,
    n_poses: int,
    protein_coords: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses(
        key=key_samp,
        box_center=box.center,
        box_size=float(box.size[0]),
        n_poses=n_poses,
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


def _sample_certified_pocket_guided_pose_vectors(
    key: jax.Array,
    n_poses: int,
    certified_detected_pocket: CertifiedDetectedPocket,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses_from_certified_pocket,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses_from_certified_pocket(
        key=key_samp,
        n_poses=n_poses,
        certified_pocket=certified_detected_pocket,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


def _sample_geometric_pocket_guided_pose_vectors(
    key: jax.Array,
    n_poses: int,
    geometric_detected_pocket: GeometricDetectedPocket,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses_from_geometric_pocket,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses_from_geometric_pocket(
        key=key_samp,
        n_poses=n_poses,
        geometric_pocket=geometric_detected_pocket,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


class BlindDockingPreparer(ABC, Generic[BindingSiteT, DetectedPocketT, PlanT]):
    preparation_type: ClassVar[type[BlindDockingPreparation]]
    plan_type: ClassVar[type[BlindDockingPlan]]

    def prepare_request(
        self,
        request: BlindDockingPreparationRequest,
    ) -> BlindDockingPreparation:
        return self.prepare(**derive_callable_kwargs(self.prepare, request))

    def prepare(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        precomputed_receptor_charges: jnp.ndarray | None,
        ligand_ctx: LigandContext,
        box: DockingBox,
        target_error: float,
        explicit_binding_site: BindingSiteT | None = None,
        coarse_target_error: float = 0.0,
        adaptive_coarse_target_errors: tuple[float, ...] | None = None,
        use_softened_coarse_prefilter: bool = False,
    ) -> BlindDockingPreparation:
        detected_pocket: DetectedPocketT | None = None
        failure_reason: CertifiedPocketFailureReason | None = None
        binding_site = explicit_binding_site
        if binding_site is None:
            detected_pocket, failure_reason = self.detect_pocket_from_box(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                box=box,
            )
            if detected_pocket is not None:
                binding_site = self.binding_site_from_detected_pocket(detected_pocket)

        restricted_coords = protein_coords
        restricted_radii = receptor_radii
        restricted_elements = receptor_elements
        restricted_charges = precomputed_receptor_charges
        restricted_box = box
        theorem_handles = self.binding_site_theorem_handles(binding_site)
        if binding_site is not None:
            (
                restricted_coords,
                restricted_radii,
                restricted_elements,
                restricted_charges,
                restricted_box,
            ) = _apply_binding_site_restriction(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                precomputed_receptor_charges=precomputed_receptor_charges,
                ligand_ctx=ligand_ctx,
                box=box,
                binding_site=binding_site,
                target_error=target_error,
            )
        theorem_handles = self.merge_theorem_handles(detected_pocket, theorem_handles)
        plan = self.build_plan(
            binding_site=binding_site,
            restricted_box=restricted_box,
            restricted_atom_count=int(restricted_coords.shape[0]),
            detected_pocket=detected_pocket,
            failure_reason=failure_reason,
            theorem_handles=theorem_handles,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=use_softened_coarse_prefilter,
        )
        return self.build_preparation(
            protein_coords=restricted_coords,
            receptor_radii=restricted_radii,
            receptor_elements=restricted_elements,
            precomputed_receptor_charges=restricted_charges,
            box=restricted_box,
            detected_pocket=detected_pocket,
            plan=plan,
        )

    @abstractmethod
    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[DetectedPocketT | None, CertifiedPocketFailureReason | None]:
        """Derive a local pocket object from the docking box."""

    @abstractmethod
    def binding_site_from_detected_pocket(
        self, detected_pocket: DetectedPocketT
    ) -> BindingSiteT:
        """Project a detected pocket down to its binding-site abstraction."""

    def build_plan(
        self,
        *,
        binding_site: BindingSiteT | None,
        restricted_box: DockingBox,
        restricted_atom_count: int,
        detected_pocket: DetectedPocketT | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> PlanT:
        return cast(
            PlanT,
            self.plan_type(
                binding_site=binding_site,
                restricted_box=restricted_box,
                restricted_atom_count=restricted_atom_count,
                **self.plan_extras(
                    detected_pocket=detected_pocket,
                    failure_reason=failure_reason,
                    theorem_handles=theorem_handles,
                    coarse_target_error=coarse_target_error,
                    adaptive_coarse_target_errors=adaptive_coarse_target_errors,
                    use_softened_coarse_prefilter=use_softened_coarse_prefilter,
                ),
            ),
        )

    def build_preparation(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        precomputed_receptor_charges: jnp.ndarray | None,
        box: DockingBox,
        detected_pocket: DetectedPocketT | None,
        plan: PlanT,
    ) -> BlindDockingPreparation:
        return cast(Any, self.preparation_type)(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            box=box,
            detected_pocket=detected_pocket,
            plan=plan,
        )

    @abstractmethod
    def plan_extras(
        self,
        *,
        detected_pocket: DetectedPocketT | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        """Route-specific plan fields beyond the shared blind-docking skeleton."""

    def binding_site_theorem_handles(
        self, binding_site: BindingSiteT | None
    ) -> tuple[str, ...]:
        return ()

    def merge_theorem_handles(
        self,
        detected_pocket: DetectedPocketT | None,
        theorem_handles: tuple[str, ...],
    ) -> tuple[str, ...]:
        return theorem_handles


class CertifiedBlindDockingPreparer(
    BlindDockingPreparer[
        CertifiedBindingSite,
        CertifiedDetectedPocket,
        CertifiedBlindDockingPlan,
    ]
):
    preparation_type = CertifiedPocketPreparation
    plan_type = CertifiedBlindDockingPlan

    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[CertifiedDetectedPocket | None, CertifiedPocketFailureReason | None]:
        return _derive_certified_binding_site_from_box(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            box=box,
        )

    def binding_site_from_detected_pocket(
        self, detected_pocket: CertifiedDetectedPocket
    ) -> CertifiedBindingSite:
        return detected_pocket.binding_site

    def plan_extras(
        self,
        *,
        detected_pocket: CertifiedDetectedPocket | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        return dict(
            certified_pocket_found=detected_pocket is not None,
            certified_failure_reason=failure_reason,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=use_softened_coarse_prefilter,
            theorem_handles=theorem_handles,
        )

    def binding_site_theorem_handles(
        self, binding_site: CertifiedBindingSite | None
    ) -> tuple[str, ...]:
        return (
            ()
            if binding_site is None
            else _merge_theorem_handles(
                binding_site.theorem_handles,
                pocket_cross_docking_handles(),
            )
        )

    def merge_theorem_handles(
        self,
        detected_pocket: CertifiedDetectedPocket | None,
        theorem_handles: tuple[str, ...],
    ) -> tuple[str, ...]:
        if detected_pocket is None:
            return theorem_handles
        return _merge_theorem_handles(
            detected_pocket.theorem_handles,
            theorem_handles,
            pocket_cross_docking_handles(),
        )


class GeometricBlindDockingPreparer(
    BlindDockingPreparer[
        GeometricBindingSite,
        GeometricDetectedPocket,
        GeometricBlindDockingPlan,
    ]
):
    preparation_type = GeometricPocketPreparation
    plan_type = GeometricBlindDockingPlan

    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[GeometricDetectedPocket | None, CertifiedPocketFailureReason | None]:
        return (
            _derive_geometric_pocket_from_box(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                box=box,
            ),
            None,
        )

    def binding_site_from_detected_pocket(
        self, detected_pocket: GeometricDetectedPocket
    ) -> GeometricBindingSite:
        return detected_pocket.binding_site

    def plan_extras(
        self,
        *,
        detected_pocket: GeometricDetectedPocket | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        del detected_pocket
        del failure_reason
        del theorem_handles
        del coarse_target_error
        del adaptive_coarse_target_errors
        del use_softened_coarse_prefilter
        return dict(
            sampling_strategy=SamplingStrategy.HYBRID,
        )


_CERTIFIED_PREPARER = CertifiedBlindDockingPreparer()
_GEOMETRIC_PREPARER = GeometricBlindDockingPreparer()


def _prepare_certified_blind_docking(
    request: CertifiedPreparationRequest,
) -> CertifiedPocketPreparation:
    prep = _CERTIFIED_PREPARER.prepare_request(request)
    return cast(CertifiedPocketPreparation, prep)


def _prepare_geometric_blind_docking(
    request: GeometricPreparationRequest,
) -> GeometricPocketPreparation:
    prep = _GEOMETRIC_PREPARER.prepare_request(request)
    return cast(GeometricPocketPreparation, prep)


def _resolve_route_scoring_electrostatics(
    effective_engine: ScoringEngine,
    ligand_ctx: LigandContext,
    receptor_elements: tuple[str, ...] | None,
    charge_method: ChargeMethod | None,
    receptor_file: str | Path | None,
    precomputed_receptor_charges: jnp.ndarray | None = None,
) -> CertifiedRealSpaceEwaldSpec | None:
    if effective_engine != ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD:
        return None

    if precomputed_receptor_charges is not None:
        if ligand_ctx.charges is None:
            raise ValueError(
                "Precomputed receptor charges require ligand_ctx.charges for electrostatic scoring."
            )
        return CertifiedRealSpaceEwaldSpec(
            receptor_charges=precomputed_receptor_charges,
            ligand_charges=ligand_ctx.charges,
        )

    if charge_method is None:
        raise ValueError(
            "CERTIFIED_LJ_REALSPACE_EWALD requires an explicit ChargeMethod."
        )

    assigner = create_charge_assigner(charge_method)

    if assigner.method == ChargeMethod.SIMPLE:
        if receptor_elements is None:
            raise ValueError("SIMPLE electrostatic scoring requires receptor_elements.")
        if not ligand_ctx.elements and ligand_ctx.charges is None:
            raise ValueError(
                "SIMPLE electrostatic scoring requires ligand elements or precomputed ligand charges."
            )
        receptor_charges = assigner.assign(receptor_elements).charges
        ligand_charges = (
            ligand_ctx.charges
            if ligand_ctx.charges is not None
            else assigner.assign(ligand_ctx.elements).charges
        )
        return CertifiedRealSpaceEwaldSpec(
            receptor_charges=receptor_charges,
            ligand_charges=ligand_charges,
        )

    if receptor_file is None:
        raise ValueError(
            f"ChargeMethod {assigner.method.name} requires receptor_file for electrostatic scoring."
        )
    if ligand_ctx.charges is None:
        raise ValueError(
            f"ChargeMethod {assigner.method.name} requires precomputed ligand_ctx.charges for electrostatic scoring."
        )

    receptor_charges = assigner.assign(receptor_file).charges
    return CertifiedRealSpaceEwaldSpec(
        receptor_charges=receptor_charges,
        ligand_charges=ligand_ctx.charges,
    )


def _posewise_active_torsion_mask(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float = 6.0,
) -> jnp.ndarray:
    """For each pose, which torsion bonds have rotating atoms in receptor range?

    Returns shape (B, n_bonds) boolean mask.

    Theorem CSC45 (torsion_locality_radius): If a scoring function evaluates to
    exactly 0 beyond a cutoff radius R_c, and a kinematic rotation keeps an atom
    strictly beyond R_c from the receptor for ALL possible rotation angles,
    then that torsion has a Lipschitz constant of exactly 0 with respect to that
    receptor atom.

    Therefore, torsion bonds whose rotating atoms are ALL > scoring_cutoff from
    ALL receptor atoms are "inactive" and can be excluded from conformer search.

    The cutoff should be the maximum across all scoring terms:
      - Gaussian contact surrogate: certified cutoff from GaussianDecayBounds
      - Softened LJ: R_cutoff from optimal_cutoff
      - Electrostatics: Coulomb/screened Coulomb cutoff
      Default 6.0 Å is conservative overestimate for typical parameters.
    """
    masks = []
    for bond in bonds:
        rotating_coords = poses_coords[:, list(bond.rotating_atom_indices), :]
        dists = jnp.linalg.norm(
            rotating_coords[:, :, None, :] - receptor_coords[None, None, :, :],
            axis=-1,
        )
        min_dist_per_pose = jnp.min(dists, axis=(1, 2))
        masks.append(min_dist_per_pose < scoring_cutoff)
    return jnp.stack(masks, axis=-1)


def _posewise_conformer_correction(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_local_bounds: jnp.ndarray | None,
    scoring_cutoff: float = 6.0,
) -> jnp.ndarray:
    """Per-pose conformer improvement bound from active torsion subsets.

    Lean: pose_specific_improvement_bound_of_active_subset — active subsets
    give no larger improvement budgets than the global bound.

    Returns shape (B,) per-pose correction.
    """
    n_poses = poses_coords.shape[0]
    if per_bond_local_bounds is None or not rotatable_bonds:
        return jnp.zeros(n_poses, dtype=poses_coords.dtype)
    active_mask = _posewise_active_torsion_mask(
        poses_coords,
        receptor_coords,
        rotatable_bonds,
        scoring_cutoff=scoring_cutoff,
    )
    return jnp.sum(active_mask * per_bond_local_bounds[None, :], axis=-1)


def _certified_pruning_pass(
    request: PipelineDockingRequest,
    poses_coords: jnp.ndarray,
    electrostatics: Optional[CertifiedRealSpaceEwaldSpec],
    *,
    scoring_context: CertifiedScoringContext | None = None,
    rotatable_bonds: tuple[RotatableBond, ...] = (),
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Perform a formally justified pruning pass on the global pose set.

    Uses the Lean-proven top-1 coarse ambiguity band (TK11, BD5) to eliminate
    poses that cannot possibly be the global minimum under the exact engine.
    """
    n_total = int(poses_coords.shape[0])
    if n_total == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return jnp.asarray(empty.astype(bool)), jnp.asarray(empty), 0.0

    # Runtime-execution detail only: evaluate the certified pruning quantities in
    # host-side chunks so large theorem-backed seed budgets do not require one
    # monolithic receptor x pose tensor on device.
    chunk_size = EXACT_RESCORING_CHUNK_SIZE
    coarse_scores_np = np.empty((n_total,), dtype=np.float32)
    use_conf = _request_uses_conformer_search(request)
    needs_receptor_flex = (
        scoring_context is not None
        and scoring_context.receptor_conformations is not None
    )
    additive_correction_np = (
        np.empty((n_total,), dtype=np.float32)
        if (use_conf or needs_receptor_flex)
        else None
    )

    conformer_config = None
    per_bond_bounds = None
    if use_conf:
        conformer_config = _build_conformer_search_config(
            request,
            rotatable_bonds=rotatable_bonds,
        )
        per_bond_bounds = _conformer_local_improvement_bounds(
            request,
            rotatable_bonds,
            conformer_config.per_bond_lipschitz,
        )

    delta: float | None = None
    flex_delta_val: float | None = None

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        chunk_coords = poses_coords[start:stop]
        chunk_scores, chunk_delta = _score_softened_pose_batch(
            request,
            poses_coords=chunk_coords,
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
        chunk_scores_np = np.asarray(jax.device_get(chunk_scores), dtype=np.float32)
        coarse_scores_np[start:stop] = chunk_scores_np

        if delta is None:
            delta = float(chunk_delta)
        else:
            delta = max(delta, float(chunk_delta))

        if additive_correction_np is None:
            continue

        chunk_correction_np = np.zeros((stop - start,), dtype=np.float32)
        if use_conf:
            chunk_correction_np += np.asarray(
                jax.device_get(
                    _posewise_conformer_correction(
                        chunk_coords,
                        request.protein_coords,
                        rotatable_bonds,
                        per_bond_bounds,
                        scoring_cutoff=compute_certified_cutoff(request.target_error),
                    )
                ),
                dtype=np.float32,
            )
        if needs_receptor_flex:
            assert scoring_context is not None
            posewise_flex_error, coarse_phys_delta = (
                scoring_context.posewise_receptor_flex_error_softened_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                    softening_radius=request.softening_radius,
                )
            )
            chunk_correction_np += np.asarray(
                jax.device_get(posewise_flex_error),
                dtype=np.float32,
            )
            chunk_flex_delta = float(np.asarray(jax.device_get(coarse_phys_delta)))
            if flex_delta_val is None:
                flex_delta_val = chunk_flex_delta
            else:
                flex_delta_val = max(flex_delta_val, chunk_flex_delta)
            chunk_correction_np += np.float32(2.0 * chunk_flex_delta)
        additive_correction_np[start:stop] = chunk_correction_np

    assert delta is not None
    best_score = float(np.min(coarse_scores_np))
    tau = best_score + 2.0 * delta

    if additive_correction_np is None:
        survivor_mask_np = coarse_scores_np <= tau
    else:
        survivor_mask_np = (coarse_scores_np - additive_correction_np) <= tau

    n_surv_val = int(np.count_nonzero(survivor_mask_np))
    efficiency = 100.0 * (1.0 - n_surv_val / n_total)
    print(
        f"[CERTIFIED PRUNING] Pruned {n_total} -> {n_surv_val} poses "
        f"({efficiency:.1f}% reduction, delta={delta:.3f} kcal/mol)"
    )

    return jnp.asarray(survivor_mask_np), jnp.asarray(coarse_scores_np), delta


def _request_uses_conformer_search(request: "PipelineDockingRequest") -> bool:
    """Check if conformer search is enabled for this request.

    Requires: config with ConformerSearchMode.ENABLED AND ligand adjacency.
    """
    return (
        request.config is not None
        and request.config.conformer_search == ConformerSearchMode.ENABLED
        and request.ligand_ctx.adjacency is not None
    )


def _request_rotatable_bonds(
    request: "PipelineDockingRequest",
) -> tuple[RotatableBond, ...]:
    adjacency = request.ligand_ctx.adjacency
    elements = request.ligand_ctx.elements
    if adjacency is None or elements is None:
        return ()
    coords_np = np.asarray(request.ligand_ctx.base_coords, dtype=np.float32)
    return detect_rotatable_bonds(adjacency, coords_np, elements)


def _build_conformer_search_config(
    request: "PipelineDockingRequest",
    *,
    rotatable_bonds: tuple[RotatableBond, ...],
) -> BranchAndBoundConfig:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
    )
    score_lipschitz_constant = _derive_exact_score_lipschitz_constant(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
        request.exact_chemistry_mode,
        request.config.softening_policy
        if request.config is not None
        else SofteningPolicy.NONE,
    )
    r_soft = request.softening_radius
    if r_soft is not None:
        if r_soft <= 0:
            raise ValueError(
                f"resolved softening radius must be positive, got {r_soft}"
            )
        softened = compute_softened_lipschitz_constant(
            _EPSILON_KCAL_MOL,
            min_sigma,
            r_soft,
        )
        if 0 < softened < score_lipschitz_constant:
            score_lipschitz_constant = softened

    per_bond_lipschitz = None
    if rotatable_bonds:
        per_bond_lipschitz = tuple(
            score_lipschitz_constant * bond.max_arm_length for bond in rotatable_bonds
        )
        max_arm = max(bond.max_arm_length for bond in rotatable_bonds)
        max_cells, min_cell_radius, _ = _derive_adaptive_torsion_support_spec(
            per_bond_lipschitz,
            request.target_error,
            request.target_rmsd,
            max_arm,
        )
    else:
        max_cells = 1
        min_cell_radius = float(2.0 * np.pi)

    return BranchAndBoundConfig(
        max_cells=max_cells,
        min_cell_radius=min_cell_radius,
        score_lipschitz_constant=score_lipschitz_constant,
        max_conformers=1,
        per_bond_lipschitz=per_bond_lipschitz,
        reuse_initial_conformer=request.reuse_initial_conformer,
    )


def _conformer_runtime_theorem_handles(
    *,
    has_rotatable_bonds: bool,
    include_pocket_handles: bool,
    include_receptor_flex_handles: bool,
) -> tuple[str, ...]:
    return _merge_theorem_handles(
        (
            "CS2",
            "CS5",
            "CS6",
            "CS8",
            "CS9",
            "CS10",
            "CS12",
            "CS13",
            "CS14",
            "GAP2",
        )
        if has_rotatable_bonds
        else (),
        (
            "CSC14",
            "CSC15",
            "CSC16",
            "CSC29",
            "CSC30",
            "CSC31",
            "CSC32",
            "CSC33",
            "CSC34",
            "CSC35",
            "CSC36",
            "CSC44",
            "CSC50",
            "BCRC1",
            "BCRC2",
            "BCRC3",
        )
        if has_rotatable_bonds
        else (),
        branch_and_bound_cross_docking_handles() if has_rotatable_bonds else (),
        pocket_cross_docking_handles() if include_pocket_handles else (),
        receptor_flexibility_theorem_handles() if include_receptor_flex_handles else (),
        receptor_flex_cross_docking_handles() if include_receptor_flex_handles else (),
    )


def _per_bond_lipschitz_improvement_bounds(
    per_bond_lipschitz: tuple[float, ...] | None,
) -> jnp.ndarray | None:
    if per_bond_lipschitz is None:
        return None
    if len(per_bond_lipschitz) == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    return (2.0 * math.pi) * jnp.asarray(per_bond_lipschitz, dtype=jnp.float32)


def _conformer_local_improvement_bounds(
    request: "PipelineDockingRequest",
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_lipschitz: tuple[float, ...] | None,
) -> jnp.ndarray | None:
    if not rotatable_bonds:
        return jnp.zeros((0,), dtype=jnp.float32)

    physical_barriers = derive_uff_torsion_barrier_heights(
        request.ligand_source_path,
        request.ligand_ctx.elements,
        rotatable_bonds,
    )
    if physical_barriers is not None:
        return 2.0 * jnp.asarray(physical_barriers, dtype=jnp.float32)

    return _per_bond_lipschitz_improvement_bounds(per_bond_lipschitz)


def _conformer_improvement_bound(per_bond_bounds: jnp.ndarray | None) -> float:
    if per_bond_bounds is None:
        return 0.0
    return float(jnp.sum(jnp.asarray(per_bond_bounds, dtype=jnp.float32)))


def _canonical_retain_mask(
    reference_scores: jnp.ndarray,
    *,
    delta: float,
    additive_correction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Retain all poses whose lower bound is within δ of the best score.

    The retain set size k is DERIVED from the scores and δ — not specified.
    Lean: canonicalRetain_certifiedSafe guarantees the true energy minimum
    is in this set. A fixed k would either miss winners (too small) or
    waste compute (too large).

    Returns (mask, tau, lower_bounds).
    """
    if reference_scores.ndim != 1:
        raise ValueError("reference_scores must be 1D")
    if additive_correction.shape != reference_scores.shape:
        raise ValueError("additive_correction must match reference_scores shape")
    if reference_scores.shape[0] == 0:
        return (
            jnp.zeros_like(reference_scores, dtype=bool),
            jnp.array(jnp.inf, dtype=reference_scores.dtype),
            jnp.full_like(reference_scores, jnp.inf),
        )
    delta_arr = jnp.asarray(delta, dtype=reference_scores.dtype)
    best_score = jnp.min(reference_scores)
    tau = best_score + delta_arr
    lower_bounds = reference_scores - additive_correction - delta_arr
    return lower_bounds <= tau, tau, lower_bounds


def _score_softened_pose_batch(
    request: "PipelineDockingRequest",
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None,
) -> tuple[jnp.ndarray, float]:
    softening_radius = request.softening_radius
    if scoring_context is not None:
        coarse_batch = scoring_context.score_softened_batch(
            receptor_coords=request.protein_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            target_error=request.target_error,
            epsilon=0.2,
            softening_radius=softening_radius,
        )
        exact_softening = float(
            jnp.min(request.receptor_radii) + jnp.min(request.ligand_ctx.base_radii)
        )
        if getattr(scoring_context, "uses_batch_pruning_delta", lambda: False)() and (
            softening_radius is None
            or math.isclose(
                softening_radius,
                exact_softening,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            delta = 0.0
        elif (
            scoring_context.uses_extended_rich
            and not getattr(
                scoring_context, "uses_batch_pruning_delta", lambda: False
            )()
        ):
            delta = scoring_context.analytic_pruning_delta()
        else:
            delta = float(
                np.asarray(jax.device_get(coarse_batch.softening_error_bound))
            )
        return coarse_batch.scores, delta

    if electrostatics is not None:
        coarse_batch = score_certified_softened_lj_realspace_ewald(
            receptor_coords=request.receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_radii,
            electrostatics=electrostatics,
            target_error=request.target_error,
            compute_error_bound=True,
            softening_radius=softening_radius,
        )
    else:
        coarse_batch = score_certified_softened_lj(
            receptor_coords=request.receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_radii,
            target_error=request.target_error,
            compute_error_bound=True,
            softening_radius=softening_radius,
        )
    delta = float(np.asarray(jax.device_get(coarse_batch.softening_error_bound)))
    return coarse_batch.scores, delta


def _build_conformer_score_fns(
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
) -> tuple[
    Callable[[jnp.ndarray], float],
    Callable[[jnp.ndarray], jnp.ndarray] | None,
]:
    """Build scoring closures for conformer search at a given rigid pose.

    Returns (score_fn, score_fn_batch):
      - score_fn: (N_atoms, 3) → float  (single-pose fallback)
      - score_fn_batch: (PAD, N_atoms, 3) → (PAD,) scores  (fixed-size batched path)
        Caller MUST always pass the same padded shape to avoid JIT recompilation.
    """

    def score_fn(candidate_coords: jnp.ndarray) -> float:
        transformed = rigid_transform_3d(candidate_coords, quaternion, translation)
        posed = jnp.expand_dims(jnp.asarray(transformed), 0)
        if scoring_context is not None:
            scores = scoring_context.score_rigid_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=posed,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            ).scores
        else:
            scores = route_scoring(
                **derive_route_scoring_kwargs(
                    request,
                    engine=request.effective_engine,
                    poses_coords=posed,
                    electrostatics=electrostatics,
                )
            )
        return float(np.asarray(scores[0]))

    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if scoring_context is not None:

        def _score_batch(coords_batch: jnp.ndarray) -> jnp.ndarray:
            """Score a fixed-size padded batch: (PAD, N_atoms, 3) → (PAD,)."""
            posed = jax.vmap(lambda c: rigid_transform_3d(c, quaternion, translation))(
                coords_batch
            )
            return scoring_context.score_rigid_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=posed,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            ).scores

        score_fn_batch = _score_batch

    return score_fn, score_fn_batch


def _run_conformer_search_for_pose(
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    *,
    rotatable_bonds: tuple[RotatableBond, ...] | None = None,
    conformer_config: BranchAndBoundConfig | None = None,
    strain_params: TorsionStrainParams | None = None,
) -> tuple[jnp.ndarray, float, tuple[str, ...]] | None:
    """Run conformer search for a single posed ligand.

    Returns (best_conformer_coords_in_world, energy, theorem_handles) if a
    conformer improves over the rigid baseline, else None.
    Fails loud — no silent fallback.
    """
    score_fn, score_fn_batch = _build_conformer_score_fns(
        request,
        quaternion,
        translation,
        scoring_context,
        electrostatics,
    )
    adjacency = request.ligand_ctx.adjacency
    if adjacency is None:
        return None
    adjacency = cast(tuple[tuple[int, ...], ...], adjacency)

    bonds = (
        _request_rotatable_bonds(request)
        if rotatable_bonds is None
        else rotatable_bonds
    )
    active_mask = np.zeros((len(bonds),), dtype=bool)
    if bonds:
        base_world_coords = rigid_transform_3d(
            request.ligand_ctx.base_coords,
            quaternion,
            translation,
        )
        bonds, active_mask = _active_rotatable_bonds_for_pose(
            base_world_coords,
            request.protein_coords,
            bonds,
            scoring_cutoff=compute_certified_cutoff(request.target_error),
        )
    if not bonds:
        world_coords = rigid_transform_3d(
            request.ligand_ctx.base_coords,
            quaternion,
            translation,
        )
        best_energy = float(score_fn(request.ligand_ctx.base_coords))
        return (
            jnp.asarray(world_coords),
            best_energy,
            _conformer_runtime_theorem_handles(
                has_rotatable_bonds=False,
                include_pocket_handles=request.certified_binding_site is not None,
                include_receptor_flex_handles=(
                    scoring_context is not None
                    and scoring_context.receptor_conformations is not None
                ),
            ),
        )
    active_strain_params = _restrict_strain_params(strain_params, active_mask)
    config = (
        _build_conformer_search_config(request, rotatable_bonds=bonds)
        if conformer_config is None
        else conformer_config
    )
    result = search_conformers(
        base_coords=request.ligand_ctx.base_coords,
        adjacency=adjacency,
        elements=request.ligand_ctx.elements,
        score_fn=score_fn,
        config=config,
        strain_params=active_strain_params,
        score_fn_batch=score_fn_batch,
        rotatable_bonds=bonds,
    )
    if not result.conformer_coords:
        return None
    best_conf = result.conformer_coords[0]
    world_coords = rigid_transform_3d(best_conf, quaternion, translation)
    best_energy = float(result.conformer_energies[0])
    theorem_handles = _merge_theorem_handles(
        result.theorem_handles,
        _conformer_runtime_theorem_handles(
            has_rotatable_bonds=bool(bonds),
            include_pocket_handles=request.certified_binding_site is not None,
            include_receptor_flex_handles=(
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ),
        ),
    )
    return world_coords, best_energy, theorem_handles


def _score_exact_pose_batch(
    request: PipelineDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None = None,
    chunk_size: int = EXACT_RESCORING_CHUNK_SIZE,
) -> jnp.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if poses_coords.shape[0] == 0:
        return jnp.zeros((0,), dtype=request.protein_coords.dtype)

    if request.is_certified_mode:
        resolved_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if scoring_context is None
            else scoring_context
        )

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                resolved_scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                ).scores
            )
    else:

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                route_scoring(
                    **derive_route_scoring_kwargs(
                        request,
                        engine=request.effective_engine,
                        poses_coords=chunk_coords,
                        electrostatics=electrostatics,
                    )
                )
            )

    if poses_coords.shape[0] <= chunk_size:
        return score_chunk(poses_coords)

    scored_chunks: list[jnp.ndarray] = []
    for start in range(0, poses_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, poses_coords.shape[0])
        scored_chunks.append(score_chunk(poses_coords[start:stop]))

    return jnp.concatenate(scored_chunks, axis=0)


def _score_rigid_exact_pose_batch(
    request: PipelineDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None = None,
    chunk_size: int = EXACT_RESCORING_CHUNK_SIZE,
) -> jnp.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if poses_coords.shape[0] == 0:
        return jnp.zeros((0,), dtype=request.protein_coords.dtype)

    if request.is_certified_mode:
        resolved_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if scoring_context is None
            else scoring_context
        )

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                resolved_scoring_context.score_rigid_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                ).scores
            )

    else:

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                route_scoring(
                    **derive_route_scoring_kwargs(
                        request,
                        engine=request.effective_engine,
                        poses_coords=chunk_coords,
                        electrostatics=electrostatics,
                    )
                )
            )

    if poses_coords.shape[0] <= chunk_size:
        return score_chunk(poses_coords)

    scored_chunks: list[jnp.ndarray] = []
    for start in range(0, poses_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, poses_coords.shape[0])
        scored_chunks.append(score_chunk(poses_coords[start:stop]))

    return jnp.concatenate(scored_chunks, axis=0)


def _pad_to_size(
    arr: jax.Array, size: int, axis: int = 0, value: float = 0.0
) -> jax.Array:
    """Pad or clip an array to a fixed size along a specific axis."""
    current_size = arr.shape[axis]
    if current_size == size:
        return arr
    if current_size > size:
        return jax.lax.dynamic_slice_in_dim(arr, 0, size, axis=axis)

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, size - current_size)
    return jnp.pad(arr, pad_width, constant_values=value)


def _pad_tuple_to_size(
    tup: tuple[str, ...] | None, size: int, value: str = "G"
) -> tuple[str, ...] | None:
    """Pad or clip a tuple to a fixed size."""
    if tup is None:
        return None
    current_size = len(tup)
    if current_size == size:
        return tup
    if current_size > size:
        return tup[:size]
    return tup + (value,) * (size - current_size)


def _normalize_sampling_key(key: jax.Array | None) -> jax.Array:
    return jax.random.PRNGKey(0) if key is None else key


@dataclass(frozen=True)
class PipelinePoseBatch:
    request: PipelineDockingRequest
    pose_vecs: PoseVector
    certified_family: "CertifiedGlobalActionFamily | None" = None


@dataclass(frozen=True)
class PipelineInitialScores:
    final_scores: jnp.ndarray | np.ndarray
    survivor_pose_vecs: PoseVector | None = None
    survivor_exact_scores: jnp.ndarray | np.ndarray | None = None
    valid_survivor_mask: jnp.ndarray | np.ndarray | None = None
    survivor_coords: jnp.ndarray | np.ndarray | None = None


class PipelineRoute(ABC):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        return request

    @abstractmethod
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        """Generate the initial pose batch for the route."""

    @abstractmethod
    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        """Score the initial pose batch."""

    def best_index_limit(
        self, request: PipelineDockingRequest, initial_scores: PipelineInitialScores
    ) -> int:
        return request.n_poses

    def optimization_inputs(
        self,
        request: PipelineDockingRequest,
        pose_vecs: PoseVector,
        initial_scores: PipelineInitialScores,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        # Optimize all canonical retain survivors — the retain set IS the
        # optimization set. Its size is emergent from delta, not a heuristic.
        n_to_opt = request.n_poses
        return (
            pose_vecs.translation,
            pose_vecs.quaternion,
            n_to_opt,
        )

    def validate_backend(
        self, request: PipelineDockingRequest, backend: OptimizerBackend
    ) -> None:
        del request, backend


class DirectScoringRoute(PipelineRoute, ABC):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        return request.with_fixed_size_padding()

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del pose_vecs
        kwargs = derive_route_scoring_kwargs(
            request,
            engine=request.effective_engine,
            poses_coords=batched_coords,
            electrostatics=resolve_request_electrostatics(
                request,
                engine=request.effective_engine,
            ),
        )
        return PipelineInitialScores(final_scores=route_scoring(**kwargs))


class MultiStageScoringRoute(PipelineRoute, ABC):
    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del pose_vecs
        from dq_dock_engine.docking.scoring_stages import (
            StageLevel,
            create_pipeline,
            create_receptor_data,
        )
        from dq_dock_engine.docking.charges import create_charge_assigner, ChargeMethod

        typed_request = cast(MultiStagePipelineDockingRequest, request)
        receptor_elements = typed_request.receptor_elements
        if receptor_elements is None:
            receptor_elements = tuple(["C"] * len(typed_request.protein_coords))

        assigner = create_charge_assigner(
            cast(ChargeMethod, typed_request.charge_method)
        )
        if assigner.method == ChargeMethod.SIMPLE:
            receptor_charges = assigner.assign(receptor_elements).charges
        else:
            if typed_request.receptor_file is None:
                raise ValueError(
                    f"ChargeMethod {assigner.method.name} requires a receptor file path or RDKit Mol to assign charges."
                )
            receptor_charges = assigner.assign(typed_request.receptor_file).charges

        receptor_data = create_receptor_data(
            coords=typed_request.protein_coords,
            radii=typed_request.receptor_radii,
            charges=receptor_charges,
            elements=receptor_elements,
        )
        pipeline = create_pipeline(
            (
                StageLevel.STAGE1_GEOMETRIC,
                StageLevel.STAGE2_MEDIUM,
                StageLevel.STAGE3_FULL,
            )
        )
        stage_results, validation = pipeline.run(
            receptor_data,
            batched_coords,
            validate=False,
            ligand_radii=typed_request.ligand_ctx.base_radii,
            ligand_charges=typed_request.ligand_ctx.charges,
            ligand_elements=typed_request.ligand_ctx.elements,
        )
        if validation is not None:
            print(
                f"Stage validation: Spearman 1-3={validation.spearman_1_3:.2f}, Top-10 overlap={validation.top10_overlap_1_3:.2f}"
            )
        return PipelineInitialScores(final_scores=stage_results[-1].scores)


class CertifiedPipelineRoute(PipelineRoute):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        prepared_request = cast(CertifiedPreparationMixin, request).prepare()
        return prepared_request.with_scoring_override(target_error=request.target_error)

    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        prepared_request = cast(PreparedCertifiedDirectPipelineRequest, request)
        generation = _create_certified_pose_vectors(
            box=prepared_request.box,
            n_poses=prepared_request.n_poses,
            certified_binding_site=cast(
                CertifiedPocketPreparation,
                prepared_request.certified_pocket_prep,
            ).plan.binding_site,
        )
        return PipelinePoseBatch(
            request=prepared_request,
            pose_vecs=generation.pose_vecs,
            certified_family=generation.family,
        )

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        electrostatics = resolve_request_electrostatics(
            request,
            engine=request.effective_engine,
        )
        full_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if request.is_certified_mode
            else None
        )
        pruning_scoring_context = (
            None
            if full_scoring_context is None
            else full_scoring_context.pruning_context()
        )
        scoring_context = (
            None
            if full_scoring_context is None
            else full_scoring_context.optimization_context()
            if full_scoring_context.uses_extended_rich
            else full_scoring_context
        )
        do_conf = _request_uses_conformer_search(request)
        rotatable_bonds = _request_rotatable_bonds(request) if do_conf else ()
        survivor_mask, coarse_scores, delta = _certified_pruning_pass(
            request,
            poses_coords=batched_coords,
            electrostatics=electrostatics,
            scoring_context=pruning_scoring_context,
            rotatable_bonds=rotatable_bonds,
        )
        valid_survivor_indices = jnp.asarray(
            np.flatnonzero(np.asarray(jax.device_get(survivor_mask), dtype=bool))
        )
        survivor_coords = batched_coords[valid_survivor_indices]
        del coarse_scores, delta
        survivor_pose_vecs = PoseVector(
            translation=pose_vecs.translation[valid_survivor_indices],
            quaternion=pose_vecs.quaternion[valid_survivor_indices],
        )

        if do_conf:
            survivor_rigid_scores = _score_rigid_exact_pose_batch(
                request,
                poses_coords=survivor_coords,
                electrostatics=electrostatics,
                scoring_context=scoring_context,
            )
            conformer_config = _build_conformer_search_config(
                request,
                rotatable_bonds=rotatable_bonds,
            )
            per_bond_bounds = _conformer_local_improvement_bounds(
                request,
                rotatable_bonds,
                conformer_config.per_bond_lipschitz,
            )
            conf_improvement_bound = _conformer_improvement_bound(per_bond_bounds)
            strain_params = None
            exact_additive_correction = jnp.full_like(
                survivor_rigid_scores,
                conf_improvement_bound,
                dtype=survivor_rigid_scores.dtype,
            )
            if (
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ):
                exact_additive_correction = (
                    exact_additive_correction
                    + scoring_context.posewise_receptor_flex_error_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=survivor_coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                )
            exact_survivor_mask, _, _ = _canonical_retain_mask(
                survivor_rigid_scores,
                delta=0.0,
                additive_correction=exact_additive_correction,
            )
            valid_survivor_indices = valid_survivor_indices[exact_survivor_mask]
            survivor_coords = survivor_coords[exact_survivor_mask]
            survivor_pose_vecs = PoseVector(
                translation=survivor_pose_vecs.translation[exact_survivor_mask],
                quaternion=survivor_pose_vecs.quaternion[exact_survivor_mask],
            )
        else:
            strain_params = None

        survivor_exact_scores = _score_exact_pose_batch(
            request,
            poses_coords=survivor_coords,
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
        if (
            full_scoring_context is not None
            and full_scoring_context.uses_extended_rich
            and survivor_coords.shape[0] > 0
        ):
            rich_survivor_batch = full_scoring_context.score_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=survivor_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            disambiguation_batch = full_scoring_context.score_flip_disambiguation_batch(
                receptor_coords=request.protein_coords,
                poses_coords=survivor_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            if _shared_certified_singleton_top1(
                rich_survivor_batch.scores,
                float(np.asarray(jax.device_get(rich_survivor_batch.error_bound))),
                disambiguation_batch.scores,
                float(np.asarray(jax.device_get(disambiguation_batch.error_bound))),
            ):
                survivor_exact_scores = rich_survivor_batch.scores
            else:
                survivor_exact_scores = disambiguation_batch.scores
        if do_conf:
            updated_scores = np.asarray(jax.device_get(survivor_exact_scores)).copy()
            updated_coords = np.asarray(jax.device_get(survivor_coords)).copy()
            conformer_config = _build_conformer_search_config(
                request, rotatable_bonds=rotatable_bonds
            )
            for i in range(updated_scores.shape[0]):
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=survivor_pose_vecs.quaternion[i],
                    translation=survivor_pose_vecs.translation[i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    rotatable_bonds=rotatable_bonds,
                    conformer_config=conformer_config,
                    strain_params=strain_params,
                )
                if conf is not None and conf[1] < float(updated_scores[i]):
                    updated_coords[i] = np.asarray(conf[0])
                    updated_scores[i] = conf[1]
            survivor_exact_scores = jnp.asarray(updated_scores)
            survivor_coords = jnp.asarray(updated_coords)

        final_scores = (
            jnp.full((batched_coords.shape[0],), 1e6)
            .at[valid_survivor_indices]
            .set(survivor_exact_scores, indices_are_sorted=False)
        )
        return PipelineInitialScores(
            final_scores=final_scores,
            survivor_pose_vecs=survivor_pose_vecs,
            survivor_exact_scores=survivor_exact_scores,
            valid_survivor_mask=jnp.ones(survivor_exact_scores.shape, dtype=bool),
            survivor_coords=survivor_coords,
        )

    def best_index_limit(
        self, request: PipelineDockingRequest, initial_scores: PipelineInitialScores
    ) -> int:
        del request
        assert initial_scores.survivor_exact_scores is not None
        return int(initial_scores.survivor_exact_scores.shape[0])

    def optimization_inputs(
        self,
        request: PipelineDockingRequest,
        pose_vecs: PoseVector,
        initial_scores: PipelineInitialScores,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        del request, pose_vecs
        assert initial_scores.survivor_pose_vecs is not None
        assert initial_scores.survivor_exact_scores is not None
        n_valid_survivors = initial_scores.survivor_pose_vecs.translation.shape[0]
        survivor_ranked = jnp.argsort(initial_scores.survivor_exact_scores)
        return (
            initial_scores.survivor_pose_vecs.translation[survivor_ranked],
            initial_scores.survivor_pose_vecs.quaternion[survivor_ranked],
            n_valid_survivors,
        )

    def validate_backend(
        self, request: PipelineDockingRequest, backend: OptimizerBackend
    ) -> None:
        if backend != OptimizerBackend.FORMAL:
            raise ValueError(
                "CERTIFIED mode requires the formal optimizer backend; gradient refinement is heuristic."
            )


class GeometricPocketRoute(DirectScoringRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        prepared_request = cast(GeometricPreparationMixin, request).prepare()
        key, pose_vecs = prepared_request.sample_pose_batch()
        return PipelinePoseBatch(
            request=prepared_request.with_updates(key=key),
            pose_vecs=pose_vecs,
        )


class BoxSamplingRoute(DirectScoringRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        from dq_dock_engine.docking.placement import sample_random_poses

        return PipelinePoseBatch(
            request=request,
            pose_vecs=sample_random_poses(
                request.normalized_key,
                request.box,
                request.n_poses,
            ),
        )


class GeometricMultiStageRoute(MultiStageScoringRoute, GeometricPocketRoute):
    pass


class BoxMultiStageRoute(MultiStageScoringRoute, BoxSamplingRoute):
    pass


def nominalize_pipeline_request(
    request: PipelineDockingRequest,
) -> PipelineDockingRequest:
    if isinstance(request, NominalPipelineDockingRequest):
        return request
    return NominalPipelineDockingRequest.from_request(request)


def derive_pipeline_route(request: PipelineDockingRequest) -> PipelineRoute:
    if not isinstance(request, NominalPipelineDockingRequest):
        raise TypeError(
            f"Cannot derive a pipeline route from non-nominal request type {type(request).__name__}."
        )
    return request.create_route()


def _run_docking_pipeline_request(
    request: PipelineDockingRequest,
) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
    """
    Run a two-stage pose prediction pipeline.
    """
    request = nominalize_pipeline_request(
        request.with_updates(key=request.normalized_key)
    )
    route = derive_pipeline_route(request)
    request = route.prepare_request(request)
    if request.n_poses_override is None and request.config is not None:
        request, _ = _probe_seed_budget_certificate(request, route)
    pose_batch = route.generate_pose_batch(request)
    request = pose_batch.request

    from dq_dock_engine.docking.placement import apply_poses

    batched_coords = apply_poses(request.ligand_ctx, pose_batch.pose_vecs)

    initial_scores = route.score_pose_batch(
        request, batched_coords, pose_batch.pose_vecs
    )

    final_scores = initial_scores.final_scores

    best_indices = jnp.argsort(final_scores)[
        : route.best_index_limit(request, initial_scores)
    ]

    if not request.optimize:
        do_conf = _request_uses_conformer_search(request)
        scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            ).ranking_context()
            if do_conf and request.is_certified_mode
            else None
        )
        electrostatics = (
            resolve_request_electrostatics(request, engine=request.effective_engine)
            if do_conf and not request.is_certified_mode
            else None
        )
        outputs = []
        for idx in best_indices:
            idx_i = int(idx)
            th_handles: tuple[str, ...] = ()
            coords_out = batched_coords[idx_i]
            energy_out = float(final_scores[idx_i])

            if do_conf:
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=pose_batch.pose_vecs.quaternion[idx_i],
                    translation=pose_batch.pose_vecs.translation[idx_i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                )
                if conf is not None and conf[1] < energy_out:
                    coords_out, energy_out, th_handles = conf

            outputs.append(
                ScoredPose(
                    coords=coords_out,
                    energy=energy_out,
                    engine=request.effective_engine,
                    theorem_handles=th_handles,
                )
            )
        cert = _compute_native_certification(
            config=request.config,
            protein_coords=request.protein_coords,
            coords=batched_coords,
            pre_opt_scores=final_scores,
            receptor_radii=request.receptor_radii,
            ligand_ctx=request.ligand_ctx,
            include_native=request.include_native,
        )
        return outputs, cert

    opt_translations, opt_quaternions, n_to_opt = route.optimization_inputs(
        request,
        pose_batch.pose_vecs,
        initial_scores,
    )

    pre_opt_scores = final_scores

    backend = request.formal_backend
    route.validate_backend(request, backend)

    scoring_context = resolve_request_scoring_context(
        request,
        engine=request.effective_engine,
    )
    ranking_scoring_context = scoring_context.ranking_context()
    # The formal local optimizer searches on the certified base physics
    # objective, while final pose scoring/ranking can still use the richer exact
    # chemistry context.
    optimization_scoring_context = scoring_context.optimization_context()
    electrostatics = scoring_context.electrostatics
    opt_pose_translations = opt_translations
    opt_pose_quaternions = opt_quaternions
    opt_coords = apply_poses(
        request.ligand_ctx,
        PoseVector(translation=opt_pose_translations, quaternion=opt_pose_quaternions),
    )
    local_refine_indices = np.arange(int(opt_coords.shape[0]), dtype=np.int32)
    binding_site_center: jnp.ndarray | None = None
    binding_site_radius = -1.0

    refinement_certificates: list[RefinementCertificate | None] = []
    rich_ranking_certified = False

    if backend == OptimizerBackend.FORMAL:
        from dq_dock_engine.docking.formal_optimizer import (
            _run_exact_formal_refinement,
            _run_singleton_hybrid_formal_refinement,
        )

        initial_opt_vecs = PoseVector(
            translation=opt_translations,
            quaternion=opt_quaternions,
        )
        if initial_scores.survivor_coords is not None:
            assert initial_scores.survivor_exact_scores is not None
            survivor_ranked = jnp.argsort(initial_scores.survivor_exact_scores)
            initial_coords = jnp.asarray(initial_scores.survivor_coords)[
                survivor_ranked
            ]
        else:
            initial_coords = apply_poses(request.ligand_ctx, initial_opt_vecs)
        assert pose_batch.certified_family is not None
        translation_cell_width = float(jnp.min(request.box.size)) / float(
            pose_batch.certified_family.lattice_resolution
        )
        # Keep the formal local-search translation step tied to the certified pose
        # lattice itself. The current formal optimizer searches exact candidate
        # energies while softened scoring only appears in the coarse pruning lane;
        # inflating the local action family by the softened-Lipschitz speedup can
        # jump over narrow native basins (observed on 1hk4 known-conformer runs).
        # Use the half-cell exact lattice step here and leave PERF5-style scaling
        # to explicitly softened local search paths instead.
        base_step = translation_cell_width / 2.0
        # Use the canonical least positive joint dyadic refinement round from the
        # Lean support-expansion semantics: repeatedly halve both the translation
        # step and the radius-scaled rotation displacement until they are each at
        # most the target RMSD scale, and stop at the first such round.
        from dq_dock_engine.docking.formal_actions import (
            least_positive_joint_adequate_dyadic_round,
        )

        assert request.config is not None
        target_rmsd = request.config.target_rmsd
        ligand_radius = float(
            jnp.max(jnp.linalg.norm(request.ligand_ctx.base_coords, axis=-1))
        )
        rotation_displacement_step = ligand_radius * float(jnp.pi / 2.0)
        base_rotation_step_rad = _derive_local_rotation_step_rad(
            base_step, ligand_radius
        )
        n_search_rounds = least_positive_joint_adequate_dyadic_round(
            base_step,
            rotation_displacement_step,
            target_rmsd,
        )
        binding_site = (
            None
            if request.certified_pocket_prep is None
            else cast(
                CertifiedPocketPreparation,
                request.certified_pocket_prep,
            ).plan.binding_site
        )
        binding_site_center = None if binding_site is None else binding_site.center
        binding_site_radius = -1.0 if binding_site is None else binding_site.radius
        local_improvement_bound = _rigid_local_improvement_bound(
            request,
            optimization_scoring_context,
            base_translation_step=base_step,
            base_rotation_step_rad=base_rotation_step_rad,
            ligand_radius=ligand_radius,
            n_rounds=n_search_rounds,
        )
        initial_local_scores = np.asarray(
            jax.device_get(
                _score_exact_pose_batch(
                    request,
                    poses_coords=initial_coords,
                    electrostatics=electrostatics,
                    scoring_context=optimization_scoring_context,
                )
            ),
            dtype=np.float32,
        )
        best_local_score = float(np.min(initial_local_scores))
        local_refine_mask = initial_local_scores <= (
            best_local_score + local_improvement_bound
        )
        if not np.any(local_refine_mask):
            local_refine_mask[int(np.argmin(initial_local_scores))] = True
        local_refine_indices = np.flatnonzero(local_refine_mask).astype(np.int32)
        if local_refine_indices.size < int(initial_coords.shape[0]):
            print(
                "[LOCAL REFINEMENT] "
                f"Retaining {int(local_refine_indices.size)}/{int(initial_coords.shape[0])} "
                f"poses (Delta_max={local_improvement_bound:.3f} kcal/mol)",
                flush=True,
            )
        refinement_input_coords = initial_coords[jnp.asarray(local_refine_indices)]

        refinement_kwargs: dict[str, object] = dict(
            coords_batch=refinement_input_coords,
            receptor_coords=request.protein_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            n_rounds=n_search_rounds,
            target_error=request.target_error,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            # Keep the formal local optimizer on exact local-family scoring.
            # On 1hk4, softened local coarse scoring makes the ambiguity band so
            # wide that the singleton selector repeatedly chooses the noop action,
            # even when exact local candidates are much better than the incumbent.
            use_softened_coarse=False,
            base_translation_step=base_step,
            base_rotation_step_rad=base_rotation_step_rad,
            scoring_context=optimization_scoring_context,
            binding_site_center=binding_site_center,
            binding_site_radius=binding_site_radius,
        )
        formal_refiners = {
            FormalRoundStrategy.EXACT: _run_exact_formal_refinement,
            FormalRoundStrategy.SINGLETON_HYBRID: _run_singleton_hybrid_formal_refinement,
        }
        formal_refiner = formal_refiners[request.formal_round_strategy]
        if refinement_input_coords.shape[0] <= FORMAL_REFINEMENT_CHUNK_SIZE:
            refined_subset = formal_refiner(**refinement_kwargs)
        else:
            refined_chunks: list[jnp.ndarray] = []
            for start in range(
                0,
                int(refinement_input_coords.shape[0]),
                FORMAL_REFINEMENT_CHUNK_SIZE,
            ):
                stop = min(
                    start + FORMAL_REFINEMENT_CHUNK_SIZE,
                    int(refinement_input_coords.shape[0]),
                )
                chunk_kwargs = dict(refinement_kwargs)
                chunk_kwargs["coords_batch"] = refinement_input_coords[start:stop]
                refined_chunks.append(formal_refiner(**chunk_kwargs))
            refined_subset = jnp.concatenate(refined_chunks, axis=0)
        opt_coords = jnp.array(initial_coords)
        opt_coords = opt_coords.at[jnp.asarray(local_refine_indices)].set(
            refined_subset
        )
        opt_pose_translations = jnp.array(opt_translations)
        opt_pose_quaternions = jnp.array(opt_quaternions)
        refined_translations, refined_quaternions = _fit_pose_vectors_from_coords_batch(
            request.ligand_ctx.base_coords,
            refined_subset,
        )
        opt_pose_translations = opt_pose_translations.at[
            jnp.asarray(local_refine_indices)
        ].set(refined_translations)
        opt_pose_quaternions = opt_pose_quaternions.at[
            jnp.asarray(local_refine_indices)
        ].set(refined_quaternions)
        # FORMAL search proof complete — now add the refinement proof.
        # The Bayesian rounds certify basin selection; SE(3) refinement
        # certifies convergence to within ε RMSD of the basin minimum.
    # --- Certified SE(3) refinement (rigid-only routes) ---
    # When conformer search is active the optimization state is no longer a pure
    # rigid transform of the original ligand frame, so the theorem-backed rigid
    # SE(3) certificate does not directly apply. In that case we keep the formal
    # coordinate-space refinement result and only recover a best-fit rigid pose for
    # downstream bookkeeping / optional conformer re-evaluation.
    do_conf = _request_uses_conformer_search(request)
    if do_conf:
        refinement_certificates = [None] * int(opt_coords.shape[0])
    else:
        site_center = binding_site_center
        site_radius = binding_site_radius
        refinement_certificates = [None] * int(opt_coords.shape[0])
        if local_refine_indices.size > 0:
            pre_se3_translations = opt_pose_translations[
                jnp.asarray(local_refine_indices)
            ]
            pre_se3_quaternions = opt_pose_quaternions[
                jnp.asarray(local_refine_indices)
            ]
            pre_se3_coords = opt_coords[jnp.asarray(local_refine_indices)]
            opt_t_subset, opt_q_subset, subset_certificates = _certified_refinement(
                request=request,
                initial_translations=pre_se3_translations,
                initial_quaternions=pre_se3_quaternions,
            )
            subset_coords = apply_poses(
                request.ligand_ctx,
                PoseVector(translation=opt_t_subset, quaternion=opt_q_subset),
            )
            subset_in_site = np.ones((int(local_refine_indices.size),), dtype=bool)
            if site_center is not None and site_radius > 0.0:
                subset_centers = np.asarray(
                    jax.device_get(jnp.mean(subset_coords, axis=1))
                )
                subset_in_site = np.linalg.norm(
                    subset_centers - np.asarray(site_center),
                    axis=1,
                ) <= float(site_radius)
                subset_coords = jnp.where(
                    jnp.asarray(subset_in_site)[:, None, None],
                    subset_coords,
                    pre_se3_coords,
                )
                opt_t_subset = jnp.where(
                    jnp.asarray(subset_in_site)[:, None],
                    opt_t_subset,
                    pre_se3_translations,
                )
                opt_q_subset = jnp.where(
                    jnp.asarray(subset_in_site)[:, None],
                    opt_q_subset,
                    pre_se3_quaternions,
                )
            opt_coords = opt_coords.at[jnp.asarray(local_refine_indices)].set(
                subset_coords
            )
            opt_pose_translations = opt_pose_translations.at[
                jnp.asarray(local_refine_indices)
            ].set(opt_t_subset)
            opt_pose_quaternions = opt_pose_quaternions.at[
                jnp.asarray(local_refine_indices)
            ].set(opt_q_subset)
            for local_idx, pose_idx in enumerate(local_refine_indices.tolist()):
                if subset_in_site[local_idx]:
                    refinement_certificates[pose_idx] = subset_certificates[local_idx]

    if request.is_certified_mode:
        if scoring_context.uses_extended_rich:
            rich_final_batch = scoring_context.score_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=opt_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            disambiguation_batch = scoring_context.score_flip_disambiguation_batch(
                receptor_coords=request.protein_coords,
                poses_coords=opt_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            # Only let the richer orientation-sensitive score decide the winner
            # when it shares the same certified singleton top-1 action as the
            # simpler H-bond-backed disambiguation score. Lean: FLO21.
            rich_ranking_certified = _shared_certified_singleton_top1(
                rich_final_batch.scores,
                float(np.asarray(jax.device_get(rich_final_batch.error_bound))),
                disambiguation_batch.scores,
                float(np.asarray(jax.device_get(disambiguation_batch.error_bound))),
            )
            final_batch = (
                rich_final_batch if rich_ranking_certified else disambiguation_batch
            )
        else:
            rich_ranking_certified = False
            final_batch = ranking_scoring_context.score_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=opt_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
        final_scores = final_batch.scores
        final_error_bound = float(np.asarray(jax.device_get(final_batch.error_bound)))
    else:
        final_scores = route_scoring(
            **derive_route_scoring_kwargs(
                request,
                engine=request.effective_engine,
                poses_coords=opt_coords,
                electrostatics=electrostatics,
            )
        )
        final_error_bound = None

    cert = _compute_native_certification(
        config=request.config,
        protein_coords=request.protein_coords,
        coords=opt_coords,
        pre_opt_scores=pre_opt_scores,
        receptor_radii=request.receptor_radii,
        ligand_ctx=request.ligand_ctx,
        include_native=request.include_native,
    )

    best_final_indices = jnp.argsort(final_scores)[:n_to_opt]

    runtime_diagnostic_handles: tuple[str, ...] = ()
    if (
        request.is_certified_mode
        and scoring_context.uses_extended_rich
        and not rich_ranking_certified
    ):
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (RUNTIME_ORIENTATION_SIGNAL_INACTIVE,),
        )
    if rich_ranking_certified:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            ("TK16", "FLO21", "FLO22"),
        )
    if request.is_certified_mode and _detect_rigid_equivalence_ambiguity(
        opt_coords,
        final_scores,
        error_bound=final_error_bound,
        target_rmsd=request.target_rmsd,
    ):
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY,),
        )

    rotatable_bonds: tuple[RotatableBond, ...] = ()
    conformer_config: BranchAndBoundConfig | None = None
    conf_improvement_bound = 0.0
    strain_params: TorsionStrainParams | None = None
    conformer_handles = ()
    if do_conf:
        rotatable_bonds = _request_rotatable_bonds(request)
        conformer_config = _build_conformer_search_config(
            request, rotatable_bonds=rotatable_bonds
        )
        per_bond_bounds = _conformer_local_improvement_bounds(
            request,
            rotatable_bonds,
            conformer_config.per_bond_lipschitz,
        )
        conf_improvement_bound = _conformer_improvement_bound(per_bond_bounds)
        strain_params = None
        conformer_handles = _conformer_runtime_theorem_handles(
            has_rotatable_bonds=bool(rotatable_bonds),
            include_pocket_handles=request.certified_binding_site is not None,
            include_receptor_flex_handles=(
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ),
        )
    best_energy_with_conf = (
        float(final_scores[best_final_indices[0]])
        if best_final_indices.shape[0] > 0
        else float("inf")
    )
    best_poses = []
    for idx in best_final_indices:
        idx_i = int(idx)
        th_handles: tuple[str, ...] = _merge_theorem_handles(
            conformer_handles if do_conf else (),
            runtime_diagnostic_handles,
        )
        coords_out = opt_coords[idx_i]
        energy_out = float(final_scores[idx_i])

        if do_conf:
            if energy_out <= best_energy_with_conf + conf_improvement_bound:
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=opt_pose_quaternions[idx_i],
                    translation=opt_pose_translations[idx_i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    rotatable_bonds=rotatable_bonds,
                    conformer_config=conformer_config,
                    strain_params=strain_params,
                )
                if conf is not None and conf[1] < energy_out:
                    coords_out, energy_out, th_handles = conf
            best_energy_with_conf = min(best_energy_with_conf, energy_out)

        best_poses.append(
            ScoredPose(
                coords=coords_out,
                energy=energy_out,
                engine=request.effective_engine,
                theorem_handles=th_handles,
            )
        )

    return best_poses, cert


def run_docking_pipeline_request(
    request: PipelineDockingRequest,
) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
    return _run_docking_pipeline_request(request)


def run_certified_blind_docking_request(
    request: CertifiedBlindDockingRequest,
) -> CertifiedBlindDockingResult:
    effective_request = request.with_updates(
        config=(
            DockingConfig(
                mode=DockingMode.CERTIFIED, optimizer_backend=OptimizerBackend.FORMAL
            )
            if request.config is None
            else request.config
        )
    )
    prep = _prepare_certified_blind_docking(
        CertifiedPreparationRequest.from_request(effective_request)
    )
    if not prep.plan.certified_pocket_found and prep.plan.binding_site is None:
        raise ValueError(
            "Certified blind docking could not derive a theorem-backed pocket/binding-site plan"
            f" ({prep.plan.certified_failure_reason.name if prep.plan.certified_failure_reason is not None else 'UNKNOWN'})."
        )
    poses, certification = run_docking_pipeline_request(
        derive_request(
            PipelineDockingRequest,
            effective_request,
            engine=ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
            use_pocket_guided=True,
            certified_pocket_prep=prep,
            scoring_kwargs=dict(effective_request.scoring_kwargs),
        )
    )
    return CertifiedBlindDockingResult(
        plan=prep.plan,
        poses=tuple(poses),
        certification=certification,
    )


def run_geometric_blind_docking_request(
    request: GeometricBlindDockingRequest,
) -> GeometricBlindDockingResult:
    prep = _prepare_geometric_blind_docking(
        GeometricPreparationRequest.from_request(request)
    )
    poses, _ = run_docking_pipeline_request(
        derive_request(
            PipelineDockingRequest,
            request,
            use_pocket_guided=True,
            scoring_kwargs=dict(request.scoring_kwargs),
        )
    )
    return GeometricBlindDockingResult(plan=prep.plan, poses=tuple(poses))


@dataclass(frozen=True)
class GeneratedRequestWrapperSpec:
    name: str
    request_type: type[DockingRequestBase]
    runner: Callable[[Any], Any]
    middle_positional_fields: tuple[str, ...] = ()
    signature_defaults: dict[str, object] = field(default_factory=dict)

    @property
    def positional_fields(self) -> tuple[str, ...]:
        return (
            "protein_coords",
            "receptor_radii",
            "ligand_ctx",
            "box",
            *self.middle_positional_fields,
            "key",
            "receptor_elements",
        )


def _build_request_wrapper_signature(
    spec: GeneratedRequestWrapperSpec,
) -> inspect.Signature:
    parameters: list[inspect.Parameter] = []
    ordered_fields = {
        field_info.name: field_info
        for field_info in dataclass_fields(spec.request_type)
        if field_info.init and field_info.name != "scoring_kwargs"
    }
    parameter_names = list(spec.positional_fields) + [
        name for name in ordered_fields if name not in spec.positional_fields
    ]
    for name in parameter_names:
        field_info = ordered_fields[name]
        default = inspect._empty
        if name in spec.signature_defaults:
            default = spec.signature_defaults[name]
        elif field_info.default is not MISSING:
            default = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            default = field_info.default_factory()  # type: ignore[misc]
        kind = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
            if name in spec.positional_fields
            else inspect.Parameter.KEYWORD_ONLY
        )
        parameters.append(
            inspect.Parameter(
                name,
                kind,
                default=default,
                annotation=field_info.type,
            )
        )
    parameters.append(
        inspect.Parameter("scoring_kwargs", inspect.Parameter.VAR_KEYWORD)
    )
    return inspect.Signature(
        parameters,
        return_annotation=inspect.signature(spec.runner).return_annotation,
    )


def _make_request_wrapper(spec: GeneratedRequestWrapperSpec) -> Callable[..., Any]:
    signature = _build_request_wrapper_signature(spec)

    def wrapper(*args: object, **kwargs: object) -> Any:
        bound = signature.bind(*args, **kwargs)
        request_kwargs = dict(bound.arguments)
        scoring_kwargs = request_kwargs.pop("scoring_kwargs", {})
        if "key" in request_kwargs:
            request_kwargs["key"] = _normalize_sampling_key(
                cast(jax.Array | None, request_kwargs["key"])
            )
        request_kwargs["scoring_kwargs"] = dict(cast(dict[str, object], scoring_kwargs))
        return spec.runner(derive_request(spec.request_type, request_kwargs))

    wrapper.__name__ = spec.name
    wrapper.__qualname__ = spec.name
    wrapper.__doc__ = (
        f"Auto-generated convenience wrapper for `{spec.request_type.__name__}`."
    )
    setattr(wrapper, "__signature__", signature)
    return wrapper


REQUEST_WRAPPER_SPECS = (
    GeneratedRequestWrapperSpec(
        name="run_docking_pipeline",
        request_type=PipelineDockingRequest,
        runner=run_docking_pipeline_request,
        signature_defaults={"key": None},
    ),
    GeneratedRequestWrapperSpec(
        name="run_certified_blind_docking",
        request_type=CertifiedBlindDockingRequest,
        runner=run_certified_blind_docking_request,
    ),
    GeneratedRequestWrapperSpec(
        name="run_geometric_blind_docking",
        request_type=GeometricBlindDockingRequest,
        runner=run_geometric_blind_docking_request,
        middle_positional_fields=("engine",),
        signature_defaults={"engine": inspect._empty},
    ),
)

globals().update(
    {spec.name: _make_request_wrapper(spec) for spec in REQUEST_WRAPPER_SPECS}
)


def _certified_refinement(
    request: "PipelineDockingRequest",
    initial_translations: jnp.ndarray,
    initial_quaternions: jnp.ndarray,
    *,
    mode_override: RefinementCertificationMode | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[RefinementCertificate | None]]:
    """Certified SE(3) refinement: replaces fixed n_opt_steps with per-pose
    theorem-derived budgets.

    Both approaches start from the initial pose and run SE(3) GD in
    axis-angle parameterization:

      OBSERVED:     Fixed-α GD with scan-recorded energy trajectory →
                    empirical q from worst-case contraction ratio.
      CERTIFIED_GD: Two-phase (probe + Hessian-derived α and budget) →
                    analytic q = (lmax - lmin) / (lmax + lmin).

    Returns (optimized_translations, optimized_quaternions, certificates).
    The optimized poses come from the certified optimizer itself — this is
    "fixed-precision docking" not "fixed-cost docking".
    """
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    config = request.config
    assert config is not None
    mode = config.refinement_certification if mode_override is None else mode_override
    target_rmsd = config.target_rmsd

    cutoff = jnp.array(compute_certified_cutoff(request.target_error))
    base_coords = request.ligand_ctx.base_coords
    receptor_coords = request.protein_coords
    receptor_radii = request.receptor_radii
    ligand_radii = request.ligand_ctx.base_radii
    ligand_radius = float(jnp.max(jnp.linalg.norm(base_coords, axis=-1)))
    n_atoms = base_coords.shape[0]

    energy_fn = make_se3_energy_fn(
        base_coords,
        receptor_coords,
        receptor_radii,
        ligand_radii,
        cutoff,
        _EPSILON_KCAL_MOL,
    )
    kinematics_fn = make_se3_kinematics_fn(base_coords)

    from dq_dock_engine.docking.formal_actions import (
        least_positive_joint_adequate_dyadic_round,
    )

    translation_extent = float(jnp.max(request.box.size)) / 2.0
    rotation_displacement_extent = ligand_radius * float(jnp.pi / 2.0)
    # Use the same canonical joint dyadic adequacy budget that drives the formal
    # local optimizer shells, rather than a fixed heuristic probe length.
    observed_probe_steps = least_positive_joint_adequate_dyadic_round(
        translation_extent,
        rotation_displacement_extent,
        target_rmsd,
    )

    n_poses = initial_translations.shape[0]
    opt_translations_list: list[jnp.ndarray] = []
    opt_quaternions_list: list[jnp.ndarray] = []
    certificates: list[RefinementCertificate | None] = []

    for i in range(n_poses):
        initial_params = _pose_to_se3_params(
            initial_translations[i],
            initial_quaternions[i],
        )

        if mode == RefinementCertificationMode.OBSERVED:
            optimized_params, cert = observe_gd_trajectory(
                initial_params=initial_params,
                energy_fn=energy_fn,
                kinematics_fn=kinematics_fn,
                n_steps=observed_probe_steps,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
                ligand_radius=ligand_radius,
            )
        elif mode == RefinementCertificationMode.CERTIFIED_GD:
            optimized_params, cert = optimize_certified_gd(
                initial_params=initial_params,
                energy_fn=energy_fn,
                kinematics_fn=kinematics_fn,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
                ligand_radius=ligand_radius,
                max_probe_steps=observed_probe_steps,
            )
        else:
            raise ValueError(f"Unexpected refinement certification mode: {mode}")

        t_opt, q_opt = _se3_params_to_pose(optimized_params)
        opt_translations_list.append(t_opt)
        opt_quaternions_list.append(q_opt)
        certificates.append(cert)

    return (
        jnp.stack(opt_translations_list),
        jnp.stack(opt_quaternions_list),
        certificates,
    )


def _compute_native_certification(
    config: DockingConfig | None,
    protein_coords: jnp.ndarray,
    coords: jnp.ndarray,
    pre_opt_scores: jnp.ndarray | np.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    include_native: bool,
) -> Union[NativeCertification, GapCertification, None]:
    if config is None or config.mode != DockingMode.CERTIFIED:
        return None

    target_error = (
        config.target_error
        if config.target_error > 0
        else _derive_target_error_from_rmsd(
            config.target_rmsd,
            receptor_radii,
            ligand_ctx.base_radii,
            config.exact_chemistry_mode,
            config.softening_policy,
        )
    )
    _, error_bound = score_certified_lj(
        protein_coords,
        coords[:1],
        receptor_radii,
        ligand_ctx.base_radii,
        target_error=target_error,
    )
    error_bound = float(error_bound)

    # Convert to numpy once to avoid tracer issues in list comprehensions
    # We use jax.device_get() to pull values from the device concretely.
    try:
        pre_opt_scores_np = jax.device_get(pre_opt_scores)
    except Exception:
        # If we can't device_get, we are likely in a transformation where we shouldn't be anyway.
        # But for robustness in the benchmark, we'll try to convert.
        pre_opt_scores_np = np.asarray(pre_opt_scores)

    pre_scores_list = pre_opt_scores_np.tolist()
    best_energy = float(np.min(pre_opt_scores_np))

    if include_native:
        native_coords = ligand_ctx.base_coords + ligand_ctx.center_of_mass
        native_score_arr, _ = score_certified_lj(
            protein_coords,
            native_coords[None],
            receptor_radii,
            ligand_ctx.base_radii,
            target_error=target_error,
        )
        native_energy = float(native_score_arr[0])
        native_rank = int(np.sum(pre_opt_scores_np < native_energy)) + 1
        gap = abs(native_energy - best_energy)

        two_bound = 2 * error_bound
        decision = (
            CertificationDecision.CERTIFIED_BETTER
            if gap > two_bound
            else CertificationDecision.UNCERTIFIED
        )
        return NativeCertification(
            decision=decision,
            energy_gap=gap,
            error_bound=error_bound,
            native_rank=native_rank,
        )
    else:
        sorted_indices = sorted(
            range(len(pre_scores_list)), key=lambda i: pre_scores_list[i]
        )
        if len(sorted_indices) < 2:
            return None
        return GapCertification.from_energies(
            pre_scores_list[sorted_indices[0]],
            pre_scores_list[sorted_indices[1]],
            error_bound,
        )
