"""
Core OpenHCS Domain Types for Pose Prediction.

Enforces strict immutability, frozen dataclass structures.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import jax.numpy as jnp


class ScoringEngine(Enum):
    """Supported scoring backends."""

    INTERNAL_LJ = auto()
    SMINA_EXACT = auto()
    VINARDO = auto()
    SOFT_LJ = auto()
    CERTIFIED_LJ = auto()


class FormalProofStatus(Enum):
    """Formal status of a runtime docking path."""

    CERTIFIED = auto()
    CONDITIONALLY_CERTIFIED = auto()
    HEURISTIC = auto()


class CertificationDecision(Enum):
    CERTIFIED_BETTER = auto()
    CERTIFIED_WORSE = auto()
    UNCERTIFIED = auto()


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
    certified: Optional[bool] = None
    confidence: Optional[float] = None
    energy_gap: Optional[float] = None
    native_rank: Optional[int] = None

    @classmethod
    def from_certification(
        cls,
        pose_energy: float,
        pose_rmsd: float,
        elapsed: float,
        n_atoms: int,
        formal_status: str,
        cert: Optional[Union[NativeCertification, GapCertification]] = None,
    ) -> "BenchmarkResult":
        kwargs = dict(
            success=True,
            energy=pose_energy,
            rmsd=pose_rmsd,
            time=elapsed,
            n_atoms=n_atoms,
            formal_status=formal_status,
        )
        if cert is not None:
            kwargs.update(
                certified=cert.is_certified,
                confidence=cert.confidence,
                energy_gap=cert.energy_gap,
            )
        if isinstance(cert, NativeCertification):
            kwargs["native_rank"] = cert.native_rank
        return cls(**kwargs)


@dataclass(frozen=True)
class DockingBox:
    """Bounding box for geometric constraints."""

    center: jnp.ndarray  # shape (3,)
    size: jnp.ndarray  # shape (3,)

    def validate(self) -> bool:
        return self.center.shape == (3,) and self.size.shape == (3,)


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


@dataclass(frozen=True)
class PoseVector:
    """Represents a batched set of rigid transformations."""

    translation: jnp.ndarray  # shape (N_poses, 3)
    quaternion: jnp.ndarray  # shape (N_poses, 4)


@dataclass(frozen=True)
class ScoredPose:
    """An immutable generated and scored structure."""

    coords: jnp.ndarray  # shape (N, 3)
    energy: float
    engine: ScoringEngine
