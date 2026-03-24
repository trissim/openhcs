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
    detect_rotatable_bonds,
    default_torsion_strain_params,
    search_conformers,
)
from dq_dock_engine.docking.formal_handles import (
    branch_and_bound_cross_docking_handles,
    pocket_cross_docking_handles,
    receptor_flex_cross_docking_handles,
    receptor_flexibility_theorem_handles,
    strain_augmented_cross_docking_handles,
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


def _merge_theorem_handles(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return tuple(dict.fromkeys(merged))


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
    """Derive the number of initial seeds from capture probability.

    Uses the SE(3) covering number argument: the search volume (translation,
    rotation, torsion) divided by a conservative capture-basin volume gives the
    expected number of trials needed. The confidence level converts this to a
    concrete seed count via the geometric distribution CDF inversion:

        N ≥ ln(1 - P) / ln(1 - V_basin / V_total)

    Lean: SeedBudgetDerivation.lean — sufficient_seed_budget (SB2),
    minSeedBudget_antitone (SB5), composed_two_phase_seed_budget (SB7).

    Without a probe certificate, this uses the theorem-honest zero-step capture
    basin: a seed is a hit only if it already lies within `target_rmsd` of the
    optimum. With a probe certificate, the certified linear contraction rate `q`
    and budget `T` amplify this to the larger convergence radius

        capture_rmsd = target_rmsd * (1 / q)^(T / 2).

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
        if q <= 0.0:
            capture_rmsd = float("inf")
        elif q < 1.0:
            capture_rmsd = target_rmsd * math.pow(
                1.0 / q, probe_certificate.n_steps / 2.0
            )

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


def _seed_budget_torsion_count(request: "DockingRequestBase") -> int:
    config = request.config
    if config is None or config.conformer_search != ConformerSearchMode.ENABLED:
        return 0
    return _rotatable_bond_count(request.ligand_ctx)


def _derive_adaptive_torsion_support_spec(
    per_bond_lipschitz: tuple[float, ...],
    target_delta: float,
) -> tuple[int, float, tuple[int, ...]]:
    """Derive a proof-backed adaptive torsion support from the Lean support theorems.

    Lean now provides canonical adaptive segment counts from per-dimension Lipschitz
    constants and target slack. For `m` active torsions, choose

        segments_i = ceil(2 * pi * m * L_i / target_delta) ∨ 1

    so the arithmetic center half-width is `2*pi/segments_i` and the total support
    slack is certified to be at most `target_delta`. The finite support size is the
    tensor-product cardinality `prod_i (segments_i + 1)`.
    """
    if target_delta <= 0.0:
        raise ValueError(f"target_delta must be positive, got {target_delta}")
    if not per_bond_lipschitz:
        return 1, float(2.0 * np.pi), ()
    n_active = len(per_bond_lipschitz)
    segments = tuple(
        max(
            1,
            int(
                math.ceil(
                    (2.0 * math.pi * n_active * max(0.0, float(li))) / target_delta
                )
            ),
        )
        for li in per_bond_lipschitz
    )
    half_widths = [float((2.0 * math.pi) / seg) for seg in segments]
    support_size = 1
    for seg in segments:
        support_size *= seg + 1
    min_cell_radius = min(half_widths) if half_widths else float(2.0 * np.pi)
    return support_size, min_cell_radius, segments


def _active_rotatable_bonds_for_pose(
    world_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
) -> tuple[tuple[RotatableBond, ...], np.ndarray]:
    if not rotatable_bonds:
        return (), np.zeros((0,), dtype=bool)
    active_mask = np.asarray(
        jax.device_get(
            _posewise_active_torsion_mask(
                poses_coords=jnp.expand_dims(world_coords, axis=0),
                receptor_coords=receptor_coords,
                bonds=rotatable_bonds,
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
    if strain_params is None or active_mask.size == 0:
        return None if strain_params is None else default_torsion_strain_params(0)
    if not np.any(active_mask):
        return default_torsion_strain_params(0)
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
        if best_seed_budget is None or derived_budget > best_seed_budget:
            best_seed_budget = derived_budget
            best_cert = cert

    if best_seed_budget is None:
        return probe_request, None
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
        if self.config is None or self.config.target_error <= 0:
            return 0.001
        return self.config.target_error

    @property
    def target_error(self) -> float:
        return self.resolved_target_error

    @property
    def certified_binding_site(self) -> CertifiedBindingSite | None:
        return None if self.config is None else self.config.certified_binding_site

    @property
    def coarse_target_error(self) -> float:
        return 0.004 if self.config is None else self.config.coarse_target_error

    @property
    def adaptive_coarse_target_errors(self) -> tuple[float, ...] | None:
        return (
            None if self.config is None else self.config.adaptive_coarse_target_errors
        )

    @property
    def use_softened_coarse_prefilter(self) -> bool:
        return (
            False if self.config is None else self.config.use_softened_coarse_prefilter
        )

    @property
    def reuse_initial_conformer(self) -> bool:
        return False if self.config is None else self.config.reuse_initial_conformer

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
    coarse_target_error: float = 0.004
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
        coarse_target_error: float = 0.004,
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
    interaction_radius: float = 6.0,
) -> jnp.ndarray:
    """For each pose, which torsion bonds have rotating atoms in receptor range?

    Returns shape (B, n_bonds) boolean mask.

    Lean: pose_specific_improvement_bound_of_active_subset — a torsion bond
    is "active" for a pose if its rotating atoms are within interaction range
    of the receptor.
    """
    masks = []
    for bond in bonds:
        rotating_coords = poses_coords[:, list(bond.rotating_atom_indices), :]
        dists = jnp.linalg.norm(
            rotating_coords[:, :, None, :] - receptor_coords[None, None, :, :],
            axis=-1,
        )
        min_dist_per_pose = jnp.min(dists, axis=(1, 2))
        masks.append(min_dist_per_pose < interaction_radius)
    return jnp.stack(masks, axis=-1)


def _posewise_conformer_correction(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    strain_params: TorsionStrainParams | None,
) -> jnp.ndarray:
    """Per-pose conformer improvement bound from active torsion subsets.

    Lean: pose_specific_improvement_bound_of_active_subset — active subsets
    give no larger improvement budgets than the global bound.

    Returns shape (B,) per-pose correction.
    """
    n_poses = poses_coords.shape[0]
    if strain_params is None or not rotatable_bonds:
        return jnp.zeros(n_poses, dtype=poses_coords.dtype)
    per_bond_barriers = 2.0 * jnp.asarray(strain_params.barrier_heights)
    active_mask = _posewise_active_torsion_mask(
        poses_coords,
        receptor_coords,
        rotatable_bonds,
    )
    return jnp.sum(active_mask * per_bond_barriers[None, :], axis=-1)


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
    coarse_scores, delta = _score_softened_pose_batch(
        request,
        poses_coords=poses_coords,
        electrostatics=electrostatics,
        scoring_context=scoring_context,
    )

    if _request_uses_conformer_search(request):
        _, strain_params = _conformer_improvement_bound(rotatable_bonds)
        additive_correction = _posewise_conformer_correction(
            poses_coords=poses_coords,
            receptor_coords=request.protein_coords,
            rotatable_bonds=rotatable_bonds,
            strain_params=strain_params,
        )
        if (
            scoring_context is not None
            and scoring_context.receptor_conformations is not None
        ):
            posewise_flex_error, coarse_phys_delta = (
                scoring_context.posewise_receptor_flex_error_softened_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=poses_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                    softening_radius=None,
                )
            )
            coarse_phys_delta_val = float(np.asarray(jax.device_get(coarse_phys_delta)))
            additive_correction = (
                additive_correction
                + posewise_flex_error
                + jnp.asarray(2.0 * coarse_phys_delta_val, dtype=coarse_scores.dtype)
            )
        survivor_mask, _, _ = _canonical_retain_mask(
            coarse_scores,
            delta=delta,
            additive_correction=additive_correction,
        )
    else:
        # 2. Compute the survivor mask from the proved pruning certificate.
        survivor_mask = coarse_top1_ambiguity_mask(coarse_scores, delta)

    n_total = poses_coords.shape[0]

    # 3. Log efficiency (safe for both JAX and vanilla numpy)
    # We use jax.device_get to ensure we have a concrete value for printing if we are not in JIT.
    # If we ARE in JIT, this print is skipped/safe.
    try:
        n_surv_val = int(jax.device_get(jnp.sum(survivor_mask)))
        delta_val = float(delta)
        efficiency = 100.0 * (1.0 - n_surv_val / n_total)
        print(
            f"[CERTIFIED PRUNING] Pruned {n_total} -> {n_surv_val} poses "
            f"({efficiency:.1f}% reduction, delta={delta_val:.3f} kcal/mol)"
        )
    except Exception:
        # Fallback for JIT context where device_get is forbidden
        jax.debug.print("[CERTIFIED PRUNING] Pruning completed (tracing context).")

    return survivor_mask, coarse_scores, delta


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
    score_lipschitz_constant = 22.0
    if request.receptor_elements is not None and request.ligand_ctx.elements:
        from dq_dock_engine.docking.physics_params import get_pairwise_contact_sigma
        from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

        min_sigma = min(
            get_pairwise_contact_sigma(re, le)
            for re in request.receptor_elements
            for le in request.ligand_ctx.elements
        )
        score_lipschitz_constant = compute_raw_lj_lipschitz(
            _EPSILON_KCAL_MOL, min_sigma
        )

    per_bond_lipschitz = None
    if rotatable_bonds:
        per_bond_lipschitz = tuple(
            score_lipschitz_constant * bond.max_arm_length for bond in rotatable_bonds
        )
        max_cells, min_cell_radius, _ = _derive_adaptive_torsion_support_spec(
            per_bond_lipschitz,
            request.target_error,
        )
    else:
        max_cells = 1
        min_cell_radius = float(2.0 * np.pi)

    return BranchAndBoundConfig(
        max_cells=max_cells,
        min_cell_radius=min_cell_radius,
        score_lipschitz_constant=score_lipschitz_constant,
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
        ("CS2", "CS5", "CS6", "CS8", "CS9", "GAP2") if has_rotatable_bonds else (),
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
        )
        if has_rotatable_bonds
        else (),
        branch_and_bound_cross_docking_handles() if has_rotatable_bonds else (),
        ("LSA1", "LSA3", "LSA5", "LSA7") if has_rotatable_bonds else (),
        strain_augmented_cross_docking_handles() if has_rotatable_bonds else (),
        pocket_cross_docking_handles() if include_pocket_handles else (),
        receptor_flexibility_theorem_handles() if include_receptor_flex_handles else (),
        receptor_flex_cross_docking_handles() if include_receptor_flex_handles else (),
    )


def _conformer_improvement_bound(
    rotatable_bonds: tuple[RotatableBond, ...],
) -> tuple[float, TorsionStrainParams | None]:
    if not rotatable_bonds:
        return 0.0, None
    strain_params = default_torsion_strain_params(len(rotatable_bonds))
    improvement_bound = float(2.0 * np.sum(np.asarray(strain_params.barrier_heights)))
    return improvement_bound, strain_params


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
    if scoring_context is not None:
        coarse_batch = scoring_context.score_softened_batch(
            receptor_coords=request.protein_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            target_error=request.target_error,
            epsilon=0.2,
            softening_radius=None,
        )
        delta = scoring_context.analytic_pruning_delta()
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
        )
    else:
        coarse_batch = score_certified_softened_lj(
            receptor_coords=request.receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_radii,
            target_error=request.target_error,
            compute_error_bound=True,
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
        scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if request.is_certified_mode
            else None
        )
        do_conf = _request_uses_conformer_search(request)
        rotatable_bonds = _request_rotatable_bonds(request) if do_conf else ()
        survivor_mask, coarse_scores, delta = _certified_pruning_pass(
            request,
            poses_coords=batched_coords,
            electrostatics=electrostatics,
            scoring_context=scoring_context,
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
            conf_improvement_bound, strain_params = _conformer_improvement_bound(
                rotatable_bonds
            )
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
            resolve_request_scoring_context(request, engine=request.effective_engine)
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
    electrostatics = scoring_context.electrostatics
    opt_pose_translations = opt_translations
    opt_pose_quaternions = opt_quaternions
    opt_coords = apply_poses(
        request.ligand_ctx,
        PoseVector(translation=opt_pose_translations, quaternion=opt_pose_quaternions),
    )

    refinement_certificates: list[RefinementCertificate | None] = []

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
        translation_cell_width = 1.0
        if pose_batch.certified_family is not None:
            translation_cell_width = float(jnp.min(request.box.size)) / float(
                pose_batch.certified_family.lattice_resolution
            )
        # PERF5 (softened_grid_speedup_ratio): when softened coarse scoring
        # has a tighter Lipschitz constant, allow larger optimizer steps.
        # Certified by PerformanceCertificates.lean::softened_grid_speedup_ratio.
        # All constants derived from molecular data:
        #   - epsilon_lj: scoring._EPSILON_KCAL_MOL (LJ well depth)
        #   - min_sigma: min pairwise contact σ from Alvarez/Bondi table
        #   - r_soft: 0.5 × (min_rec_radius + min_lig_radius) (same as scoring)
        base_step = translation_cell_width / 2.0
        if (
            request.use_softened_coarse_prefilter
            and request.receptor_elements is not None
        ):
            from dq_dock_engine.docking.formal_actions import (
                compute_adaptive_translation_step,
            )
            from dq_dock_engine.docking.physics_params import get_pairwise_contact_sigma
            from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

            # Derive min pairwise sigma from actual receptor/ligand elements
            min_sigma = min(
                get_pairwise_contact_sigma(re, le)
                for re in request.receptor_elements
                for le in request.ligand_ctx.elements
            )
            r_soft = 0.5 * float(
                jnp.min(request.receptor_radii) + jnp.min(request.ligand_ctx.base_radii)
            )
            base_step = compute_adaptive_translation_step(
                base_step,
                _EPSILON_KCAL_MOL,
                min_sigma,
                r_soft,
            )
        # Derive search rounds from initial step size and target RMSD.
        # Each round contracts the search by ~2×, so we need
        # ceil(log2(base_step / target_rmsd)) rounds to reach RMSD precision.
        import math as _math

        target_rmsd = request.config.target_rmsd if request.config is not None else 0.5
        n_search_rounds = max(1, _math.ceil(_math.log2(base_step / target_rmsd)))

        refinement_kwargs: dict[str, object] = dict(
            coords_batch=initial_coords,
            receptor_coords=request.protein_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            n_rounds=n_search_rounds,
            target_error=request.target_error,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            use_softened_coarse=request.use_softened_coarse_prefilter,
            base_translation_step=base_step,
            base_rotation_step_rad=float(jnp.pi / 2.0),
            scoring_context=scoring_context,
        )
        formal_refiners = {
            FormalRoundStrategy.EXACT: _run_exact_formal_refinement,
            FormalRoundStrategy.SINGLETON_HYBRID: _run_singleton_hybrid_formal_refinement,
        }
        opt_coords = formal_refiners[request.formal_round_strategy](**refinement_kwargs)
        opt_pose_translations, opt_pose_quaternions = (
            _fit_pose_vectors_from_coords_batch(
                request.ligand_ctx.base_coords,
                opt_coords,
            )
        )
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
        opt_t, opt_q, refinement_certificates = _certified_refinement(
            request=request,
            initial_translations=opt_pose_translations,
            initial_quaternions=opt_pose_quaternions,
        )
        opt_coords = apply_poses(
            request.ligand_ctx,
            PoseVector(translation=opt_t, quaternion=opt_q),
        )
        opt_pose_translations = opt_t
        opt_pose_quaternions = opt_q

    if request.is_certified_mode:
        final_scores = scoring_context.score_exact_batch(
            receptor_coords=request.protein_coords,
            poses_coords=opt_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            target_error=request.target_error,
            epsilon=0.2,
        ).scores
    else:
        final_scores = route_scoring(
            **derive_route_scoring_kwargs(
                request,
                engine=request.effective_engine,
                poses_coords=opt_coords,
                electrostatics=electrostatics,
            )
        )

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
        conf_improvement_bound, strain_params = _conformer_improvement_bound(
            rotatable_bonds
        )
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
        th_handles: tuple[str, ...] = conformer_handles if do_conf else ()
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
    mode = config.refinement_certification
    target_rmsd = config.target_rmsd

    cutoff = jnp.array(compute_certified_cutoff(request.target_error))
    base_coords = request.ligand_ctx.base_coords
    receptor_coords = request.protein_coords
    receptor_radii = request.receptor_radii
    ligand_radii = request.ligand_ctx.base_radii
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

    # OBSERVED mode probe budget: 50 steps is a conservative default.
    # The certificate tells us post-hoc whether this was sufficient.
    _OBSERVED_PROBE_STEPS = 50

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
                n_steps=_OBSERVED_PROBE_STEPS,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
            )
        elif mode == RefinementCertificationMode.CERTIFIED_GD:
            optimized_params, cert = optimize_certified_gd(
                initial_params=initial_params,
                energy_fn=energy_fn,
                kinematics_fn=kinematics_fn,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
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

    target_error = config.target_error if config.target_error > 0 else 0.001
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
