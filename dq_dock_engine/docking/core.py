"""
Core OpenHCS Domain Types for Pose Prediction.

Enforces strict immutability, frozen dataclass structures.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional
import jax.numpy as jnp


class ScoringEngine(Enum):
    """Supported scoring backends."""

    INTERNAL_LJ = auto()
    SMINA_EXACT = auto()
    VINARDO = auto()
    SOFT_LJ = auto()


class FormalProofStatus(Enum):
    """Formal status of a runtime docking path."""

    CERTIFIED = auto()
    CONDITIONALLY_CERTIFIED = auto()
    HEURISTIC = auto()


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
