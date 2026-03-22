"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring
with multi-stage filtering and pocket-guided sampling.
"""

import inspect
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
from dq_dock_engine.docking.formal_pruning import coarse_top1_ambiguity_mask
from dq_dock_engine.docking.scoring import (
    score_certified_softened_lj,
    score_certified_softened_lj_realspace_ewald,
)
from dq_dock_engine.docking.pocket_sampling import (
    extract_local_pocket_region,
    extract_local_pocket_region_view,
)
from dq_dock_engine.docking.optimization import optimize_poses_batched
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    compute_certified_cutoff,
    DockingConfig,
    DockingMode,
    FormalRoundStrategy,
    OptimizerBackend,
)

if TYPE_CHECKING:
    from dq_dock_engine.docking.formal_sampling import CertifiedGlobalActionFamily


# Certified Pruning Constants
# We use a fixed power-of-two size for the survivor set to stabilize XLA caching.
# According to the BD5/TK11 theorems, the survivor set size is bounded by O(K+L).
# 256 is an ample bound for typical drug-like docking scenarios.
SURVIVOR_BATCH_SIZE = 256


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


@dataclass(frozen=True, kw_only=True)
class DockingRequestBase:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    ligand_ctx: LigandContext
    box: DockingBox
    n_poses: int
    key: jax.Array
    receptor_elements: tuple[str, ...] | None = None
    charge_method: ChargeMethod | None = None
    receptor_file: str | Path | None = None
    precomputed_receptor_charges: jnp.ndarray | None = None
    config: DockingConfig | None = None
    top_k: int = 10
    optimize: bool = True
    n_opt_steps: int = 50
    top_k_to_optimize: int = 200
    include_native: bool = False
    scoring_kwargs: dict[str, object] = field(default_factory=dict)

    def with_updates(self, **changes: object) -> Self:
        return replace(self, **changes)

    @property
    def normalized_key(self) -> jax.Array:
        return _normalize_sampling_key(self.key)

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
        return () if binding_site is None else binding_site.theorem_handles

    def merge_theorem_handles(
        self,
        detected_pocket: CertifiedDetectedPocket | None,
        theorem_handles: tuple[str, ...],
    ) -> tuple[str, ...]:
        if detected_pocket is None:
            return theorem_handles
        return tuple(dict.fromkeys(detected_pocket.theorem_handles + theorem_handles))


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


def _certified_pruning_pass(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    electrostatics: Optional[CertifiedRealSpaceEwaldSpec],
    target_error: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Perform a formally justified pruning pass on the global pose set.

    Uses the Lean-proven top-1 coarse ambiguity band (TK11, BD5) to eliminate
    poses that cannot possibly be the global minimum under the exact engine.
    """
    # 1. Compute coarse (softened) scores and the associated error bound delta
    if electrostatics is not None:
        coarse_batch = score_certified_softened_lj_realspace_ewald(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_ctx.base_radii,
            electrostatics=electrostatics,
            target_error=target_error,
            compute_error_bound=False,
        )
    else:
        from dq_dock_engine.docking.scoring import score_certified_softened_lj

        coarse_batch = score_certified_softened_lj(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_ctx.base_radii,
            target_error=target_error,
            compute_error_bound=False,
        )

    # 2. Compute the survivor mask based on the theorem bound (TK11).
    # Since the global optimum is guaranteed to be non-clashing, its softening
    # error is 0.0. The only remaining error is the cutoff target_error.
    delta = target_error
    survivor_mask = coarse_top1_ambiguity_mask(coarse_batch.scores, delta)

    n_total = poses_coords.shape[0]

    # 3. Log efficiency (safe for both JAX and vanilla numpy)
    # We use jax.device_get to ensure we have a concrete value for printing if we are not in JIT.
    # If we ARE in JIT, this print is skipped/safe.
    try:
        import jax
        import jax.debug

        n_surv_val = int(jax.device_get(jnp.sum(survivor_mask)))
        delta_val = float(jax.device_get(delta))
        efficiency = 100.0 * (1.0 - n_surv_val / n_total)
        print(
            f"[CERTIFIED PRUNING] Pruned {n_total} -> {n_surv_val} poses "
            f"({efficiency:.1f}% reduction, delta={delta_val:.3f} kcal/mol)"
        )
    except Exception:
        # Fallback for JIT context where device_get is forbidden
        import jax.debug

        jax.debug.print("[CERTIFIED PRUNING] Pruning completed (tracing context).")

    return survivor_mask, coarse_batch.scores, delta


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
        n_to_opt = min(request.top_k_to_optimize, request.n_poses)
        opt_indices = jnp.argsort(initial_scores.final_scores)[:n_to_opt]
        return (
            pose_vecs.translation[opt_indices],
            pose_vecs.quaternion[opt_indices],
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
        survivor_mask, coarse_scores, delta = call_with_derived_kwargs(
            _certified_pruning_pass,
            request,
            poses_coords=batched_coords,
            electrostatics=electrostatics,
        )
        survivor_indices = jnp.where(
            survivor_mask, size=SURVIVOR_BATCH_SIZE, fill_value=-1
        )[0]
        survivor_coords = batched_coords[survivor_indices]
        del coarse_scores, delta
        if request.config is not None and getattr(request.config, "use_rich_exact_rescoring", False):
            from dq_dock_engine.docking.rich_chemistry import build_all_rich_chemistry_specs
            from dq_dock_engine.docking.scoring import score_certified_rich_chemistry_batch
            
            receptor_charges = electrostatics.receptor_charges if electrostatics else jnp.zeros(request.protein_coords.shape[0])
            screened, contact, hbond = build_all_rich_chemistry_specs(
                np.asarray(request.protein_coords),
                request.receptor_elements or tuple(["C"] * request.protein_coords.shape[0]),
                np.asarray(receptor_charges),
                request.ligand_ctx
            )
            rich_batch = score_certified_rich_chemistry_batch(
                receptor_coords=request.protein_coords,
                poses_coords=survivor_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                screened_coulomb=screened,
                contact=contact,
                directional_hbond=hbond,
                target_error=request.target_error
            )
            survivor_exact_scores = rich_batch.scores
        else:
            exact_kwargs = derive_route_scoring_kwargs(
                request,
                engine=request.effective_engine,
                poses_coords=survivor_coords,
                electrostatics=electrostatics.receptor_subset(
                    jnp.arange(request.protein_coords.shape[0])
                )
                if electrostatics
                else None,
            )
            survivor_exact_scores = route_scoring(**exact_kwargs)
        valid_survivor_mask = survivor_indices != -1
        valid_survivor_indices = survivor_indices[valid_survivor_mask]
        padded_survivor_scores = jnp.where(
            valid_survivor_mask, survivor_exact_scores, 1e6
        )
        final_scores = (
            jnp.full((batched_coords.shape[0],), 1e6)
            .at[survivor_indices]
            .set(padded_survivor_scores, indices_are_sorted=False)
        )
        return PipelineInitialScores(
            final_scores=final_scores,
            survivor_pose_vecs=PoseVector(
                translation=pose_vecs.translation[valid_survivor_indices],
                quaternion=pose_vecs.quaternion[valid_survivor_indices],
            ),
            survivor_exact_scores=padded_survivor_scores,
            valid_survivor_mask=valid_survivor_mask,
        )

    def best_index_limit(
        self, request: PipelineDockingRequest, initial_scores: PipelineInitialScores
    ) -> int:
        del request, initial_scores
        return SURVIVOR_BATCH_SIZE

    def optimization_inputs(
        self,
        request: PipelineDockingRequest,
        pose_vecs: PoseVector,
        initial_scores: PipelineInitialScores,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        del pose_vecs
        assert initial_scores.survivor_pose_vecs is not None
        assert initial_scores.survivor_exact_scores is not None
        assert initial_scores.valid_survivor_mask is not None
        n_valid_survivors = initial_scores.survivor_pose_vecs.translation.shape[0]
        n_to_opt = min(request.top_k_to_optimize, n_valid_survivors)
        survivor_ranked = jnp.argsort(
            initial_scores.survivor_exact_scores[initial_scores.valid_survivor_mask]
        )[:n_to_opt]
        return (
            initial_scores.survivor_pose_vecs.translation[survivor_ranked],
            initial_scores.survivor_pose_vecs.quaternion[survivor_ranked],
            n_to_opt,
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
    pose_batch = route.generate_pose_batch(request)
    request = pose_batch.request

    from dq_dock_engine.docking.placement import apply_poses

    batched_coords = apply_poses(request.ligand_ctx, pose_batch.pose_vecs)

    initial_scores = route.score_pose_batch(
        request, batched_coords, pose_batch.pose_vecs
    )

    final_scores = initial_scores.final_scores

    best_indices = jnp.argsort(final_scores)[
        : min(request.top_k, route.best_index_limit(request, initial_scores))
    ]

    if not request.optimize:
        outputs = []
        for idx in best_indices:
            idx_i = int(idx)
            outputs.append(
                ScoredPose(
                    coords=batched_coords[idx_i],
                    energy=float(final_scores[idx_i]),
                    engine=request.effective_engine,
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

    if backend == OptimizerBackend.FORMAL:
        from dq_dock_engine.docking.formal_optimizer import (
            _run_exact_formal_refinement,
            _run_singleton_hybrid_formal_refinement,
        )

        initial_opt_vecs = PoseVector(
            translation=opt_translations,
            quaternion=opt_quaternions,
        )
        initial_coords = apply_poses(request.ligand_ctx, initial_opt_vecs)
        translation_cell_width = 1.0
        if pose_batch.certified_family is not None:
            translation_cell_width = float(jnp.min(request.box.size)) / float(
                pose_batch.certified_family.lattice_resolution
            )
        refinement_kwargs: dict[str, object] = dict(
            coords_batch=initial_coords,
            receptor_coords=request.protein_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            n_rounds=request.n_opt_steps,
            target_error=request.target_error,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            use_softened_coarse=request.use_softened_coarse_prefilter,
            base_translation_step=translation_cell_width / 2.0,
            base_rotation_step_rad=float(jnp.pi / 2.0),
        )
        formal_electrostatics = resolve_request_electrostatics(
            request,
            engine=request.effective_engine,
        )
        refinement_kwargs["electrostatics"] = formal_electrostatics
        formal_refiners = {
            FormalRoundStrategy.EXACT: _run_exact_formal_refinement,
            FormalRoundStrategy.SINGLETON_HYBRID: _run_singleton_hybrid_formal_refinement,
        }
        opt_coords = formal_refiners[request.formal_round_strategy](**refinement_kwargs)
    else:
        opt_t, opt_q = optimize_poses_batched(
            translations=opt_translations,
            quaternions=opt_quaternions,
            ligand_base_coords=request.ligand_ctx.base_coords,
            receptor_coords=request.protein_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            n_steps=request.n_opt_steps,
            lr_t=0.05,
            lr_q=0.05,
            config=request.config,
        )
        opt_coords = apply_poses(
            request.ligand_ctx,
            PoseVector(translation=opt_t, quaternion=opt_q),
        )

    electrostatics = resolve_request_electrostatics(
        request,
        engine=request.effective_engine,
    )
    if request.config is not None and getattr(request.config, "use_rich_exact_rescoring", False):
        from dq_dock_engine.docking.rich_chemistry import build_all_rich_chemistry_specs
        from dq_dock_engine.docking.scoring import score_certified_rich_chemistry_batch
        
        receptor_charges = electrostatics.receptor_charges if electrostatics else jnp.zeros(request.protein_coords.shape[0])
        screened, contact, hbond = build_all_rich_chemistry_specs(
            np.asarray(request.protein_coords),
            request.receptor_elements or tuple(["C"] * request.protein_coords.shape[0]),
            np.asarray(receptor_charges),
            request.ligand_ctx
        )
        final_scores = score_certified_rich_chemistry_batch(
            receptor_coords=request.protein_coords,
            poses_coords=opt_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            screened_coulomb=screened,
            contact=contact,
            directional_hbond=hbond,
            target_error=request.target_error
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

    best_final_indices = jnp.argsort(final_scores)[: min(request.top_k, n_to_opt)]

    best_poses = []
    for idx in best_final_indices:
        idx_i = int(idx)
        best_poses.append(
            ScoredPose(
                coords=opt_coords[idx_i],
                energy=float(final_scores[idx_i]),
                engine=request.effective_engine,
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
            "n_poses",
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
