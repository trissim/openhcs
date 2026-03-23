from __future__ import annotations

"""
Core OpenHCS Domain Types for Pose Prediction.

Enforces strict immutability, frozen dataclass structures.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class ScoringEngine(Enum):
    """Supported scoring backends."""

    INTERNAL_LJ = auto()
    SMINA_EXACT = auto()
    VINARDO = auto()
    SOFT_LJ = auto()
    CERTIFIED_LJ = auto()
    CERTIFIED_LJ_REALSPACE_EWALD = auto()


class FormalProofStatus(Enum):
    """Formal status of a runtime docking path."""

    CERTIFIED = auto()
    CONDITIONALLY_CERTIFIED = auto()
    HEURISTIC = auto()


class CertificationDecision(Enum):
    CERTIFIED_BETTER = auto()
    CERTIFIED_WORSE = auto()
    UNCERTIFIED = auto()


class SamplingStrategy(Enum):
    """Pocket-guided sampling strategies."""

    RANDOM = auto()
    GUIDED = auto()
    HYBRID = auto()

    def keep_ratio(self) -> float:
        match self:
            case SamplingStrategy.RANDOM:
                return 0.0
            case SamplingStrategy.GUIDED:
                return 1.0
            case SamplingStrategy.HYBRID:
                return 0.5

    @staticmethod
    def recommended() -> "SamplingStrategy":
        return SamplingStrategy.HYBRID


class CertifiedPocketFailureReason(Enum):
    NO_LOCAL_REGION = auto()
    NO_CERTIFIED_POCKET = auto()
    NO_CERTIFIED_BINDING_SITE = auto()


class BlindDockingExecutionPath(Enum):
    STRICT_CERTIFIED = auto()
    GEOMETRIC_FALLBACK = auto()
    GENERIC_PIPELINE = auto()


@dataclass(frozen=True)
class GapCertification:
    decision: CertificationDecision
    energy_gap: float
    error_bound: float

    @property
    def is_certified(self) -> bool:
        return self.decision in (
            CertificationDecision.CERTIFIED_BETTER,
            CertificationDecision.CERTIFIED_WORSE,
        )

    @property
    def confidence(self) -> float:
        if self.is_certified:
            return 1.0
        two_bound = 2 * self.error_bound
        return self.energy_gap / two_bound if two_bound > 0 else 0.0

    def summary(self) -> str:
        if self.decision == CertificationDecision.CERTIFIED_BETTER:
            return f"CERTIFIED BETTER (gap={self.energy_gap:.3f} > 2×{self.error_bound:.6f})"
        elif self.decision == CertificationDecision.CERTIFIED_WORSE:
            return f"CERTIFIED WORSE (gap={self.energy_gap:.3f} > 2×{self.error_bound:.6f})"
        else:
            return f"UNCERTIFIED (gap={self.energy_gap:.3f} ≤ 2×{self.error_bound:.6f})"

    @staticmethod
    def from_energies(
        energy_a: float, energy_b: float, error_bound: float
    ) -> "GapCertification":
        gap = abs(energy_a - energy_b)
        two_bound = 2 * error_bound
        if gap > two_bound:
            decision = (
                CertificationDecision.CERTIFIED_BETTER
                if energy_a < energy_b
                else CertificationDecision.CERTIFIED_WORSE
            )
        else:
            decision = CertificationDecision.UNCERTIFIED
        return GapCertification(
            decision=decision,
            energy_gap=gap,
            error_bound=error_bound,
        )


@dataclass(frozen=True)
class NativeCertification(GapCertification):
    native_rank: int = 1

    @property
    def is_native_ranked_first(self) -> bool:
        return self.native_rank == 1

    def summary(self) -> str:
        base = super().summary()
        return f"Native #{self.native_rank}: {base}"


@dataclass(frozen=True)
class BenchmarkResult:
    success: bool
    energy: float
    rmsd: float
    time: float
    n_atoms: int
    formal_status: str
    error: Optional[str] = None
    dock_time_s: Optional[float] = None
    checkpoint_time_s: Optional[float] = None
    total_wall_time_s: Optional[float] = None
    certified: Optional[bool] = None
    confidence: Optional[float] = None
    energy_gap: Optional[float] = None
    native_rank: Optional[int] = None
    execution_path: Optional[BlindDockingExecutionPath] = None
    certified_failure_reason: Optional[CertifiedPocketFailureReason] = None
    theorem_handles: Tuple[str, ...] = ()

    @classmethod
    def from_certification(
        cls,
        pose_energy: float,
        pose_rmsd: float,
        elapsed: float,
        n_atoms: int,
        formal_status: str,
        cert: Optional[Union[NativeCertification, GapCertification]] = None,
        execution_path: Optional[BlindDockingExecutionPath] = None,
        certified_failure_reason: Optional[CertifiedPocketFailureReason] = None,
        theorem_handles: Tuple[str, ...] = (),
    ) -> "BenchmarkResult":
        certified = None if cert is None else cert.is_certified
        confidence = None if cert is None else cert.confidence
        energy_gap = None if cert is None else cert.energy_gap
        native_rank = (
            cert.native_rank if isinstance(cert, NativeCertification) else None
        )
        return cls(
            success=True,
            energy=pose_energy,
            rmsd=pose_rmsd,
            time=elapsed,
            n_atoms=n_atoms,
            formal_status=formal_status,
            error=None,
            dock_time_s=elapsed,
            checkpoint_time_s=None,
            total_wall_time_s=elapsed,
            certified=certified,
            confidence=confidence,
            energy_gap=energy_gap,
            native_rank=native_rank,
            execution_path=execution_path,
            certified_failure_reason=certified_failure_reason,
            theorem_handles=theorem_handles,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class DockingBox:
    """Bounding box for geometric constraints."""

    center: jnp.ndarray  # shape (3,)
    size: jnp.ndarray  # shape (3,)

    def validate(self) -> bool:
        return self.center.shape == (3,) and self.size.shape == (3,)

    def tree_flatten(self):
        children = (self.center, self.size)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclass(frozen=True)
class BindingSite:
    """Runtime representation of a docking binding site."""

    center: jnp.ndarray  # shape (3,)
    radius: float

    def tree_flatten(self):
        children = (self.center, self.radius)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedBindingSite(BindingSite):
    """Theorem-backed binding site with proof metadata."""

    theorem_handles: Tuple[str, ...] = ()

    def tree_flatten(self):
        children = (self.center, self.radius)
        aux_data = (self.theorem_handles,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


@dataclass(frozen=True)
class GeometricBindingSite(BindingSite):
    """Runtime geometric binding-site abstraction without theorem guarantees."""


@dataclass(frozen=True)
class BlindDockingPlan:
    """Shared runtime plan for blind docking."""

    binding_site: Optional[BindingSite]
    restricted_box: DockingBox
    restricted_atom_count: int


@dataclass(frozen=True)
class CertifiedBlindDockingPlan(BlindDockingPlan):
    """Theorem-directed runtime plan for certified blind docking."""

    binding_site: Optional[CertifiedBindingSite]
    certified_pocket_found: bool
    certified_failure_reason: Optional[CertifiedPocketFailureReason]
    coarse_target_error: float
    adaptive_coarse_target_errors: Optional[Tuple[float, ...]]
    use_softened_coarse_prefilter: bool
    theorem_handles: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BlindDockingResult:
    plan: BlindDockingPlan
    poses: tuple[ScoredPose, ...]


@dataclass(frozen=True)
class CertifiedBlindDockingResult(BlindDockingResult):
    plan: CertifiedBlindDockingPlan
    certification: Optional[Union[NativeCertification, GapCertification]] = None


@dataclass(frozen=True)
class GeometricBlindDockingPlan(BlindDockingPlan):
    binding_site: Optional[GeometricBindingSite]
    sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID


@dataclass(frozen=True)
class GeometricBlindDockingResult(BlindDockingResult):
    plan: GeometricBlindDockingPlan


@register_pytree_node_class
@dataclass(frozen=True)
class LigandContext:
    """Immutable context for a ligand prior to geometric transformation."""

    base_coords: jnp.ndarray  # shape (N, 3) relative to origin
    base_radii: jnp.ndarray  # shape (N,)
    center_of_mass: jnp.ndarray  # shape (3,)
    # Per-atom element symbols in the same order as base_coords. Empty tuple if not provided.
    elements: Tuple[str, ...] = ()
    # Optional per-atom partial charges (shape (N,)). None if not assigned.
    charges: Optional[jnp.ndarray] = None
    # Optional adjacency graph per-atom (tuple of neighbor index tuples). None if not available.
    adjacency: Optional[Tuple[Tuple[int, ...], ...]] = None

    def tree_flatten(self):
        children = (
            self.base_coords,
            self.base_radii,
            self.center_of_mass,
            self.charges,
        )
        aux_data = (self.elements, self.adjacency)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        base_coords, base_radii, center_of_mass, charges = children
        elements = aux_data[0]
        adjacency = aux_data[1]
        return cls(
            base_coords=base_coords,
            base_radii=base_radii,
            center_of_mass=center_of_mass,
            elements=elements,
            charges=charges,
            adjacency=adjacency,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class PoseVector:
    """Represents a batched set of rigid transformations."""

    translation: jnp.ndarray  # shape (N_poses, 3)
    quaternion: jnp.ndarray  # shape (N_poses, 4)

    def tree_flatten(self):
        children = (self.translation, self.quaternion)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass(frozen=True)
class ScoredPose:
    """An immutable generated and scored structure."""

    coords: jnp.ndarray  # shape (N, 3)
    energy: float
    engine: ScoringEngine
    theorem_handles: Tuple[str, ...] = ()
