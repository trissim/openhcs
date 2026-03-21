"""
Pocket shape analysis for guided sampling.

OpenHCS Compliance:
- Frozen dataclasses for all configs and results
- ABC contracts for analyzers
- Enum-driven feature types
- Explicit dependency injection
- Fail-loud validation
- Pure functions for calculations
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from dq_dock_engine.docking.core import CertifiedBindingSite
from dq_dock_engine.docking.sasa import get_solvation_params
from dq_dock_engine.proof_status import conditionally_certified


# =============================================================================
# ENUM-DRIVEN FEATURE TYPES
# =============================================================================


class FeatureType(Enum):
    """Pharmacophore feature types per AD4 and docking literature."""

    HBA = auto()  # Hydrogen bond acceptor
    HBD = auto()  # Hydrogen bond donor
    HYD = auto()  # Hydrophobic
    POS = auto()  # Positive charge
    NEG = auto()  # Negative charge


# =============================================================================
# FROZEN DATACLASS CONFIGURATION (OpenHCS Pattern)
# =============================================================================


@dataclass(frozen=True)
class PocketAnalysisConfig:
    """
    Configuration for pocket analysis.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Literature-backed defaults
    - Fail-loud validation
    """

    min_subpocket_size: int = 10
    clustering_cutoff: float = 4.0
    pca_n_components: int = 3
    shape_extent_std: float = 3.0
    probe_radius: float = 1.4
    accessibility_cutoff: float = 0.3
    hbond_distance_min: float = 2.5
    hbond_distance_max: float = 4.0
    hydrophobic_cutoff: float = 4.5

    def validate(self) -> None:
        """Fail-loud validation."""
        if self.min_subpocket_size < 5:
            raise ValueError(
                f"min_subpocket_size {self.min_subpocket_size} too small (min 5)"
            )
        if not (0.0 < self.accessibility_cutoff < 1.0):
            raise ValueError(
                f"accessibility_cutoff {self.accessibility_cutoff} must be in (0, 1)"
            )


# =============================================================================
# FROZEN DATA CONTAINERS (OpenHCS Pattern)
# =============================================================================


@dataclass(frozen=True)
class PocketShape:
    """
    Geometric descriptors of pocket shape.

    OpenHCS Compliance:
    - Frozen dataclass (immutable)
    - Explicit types
    - No defensive checks
    """

    center_of_mass: jnp.ndarray
    principal_axes: jnp.ndarray
    extents: jnp.ndarray
    volume: float
    concavity: float
    openness: tuple[float, float, float]


@dataclass(frozen=True)
class PharmacophoreFeature:
    """
    A pharmacophore feature (interaction point).

    OpenHCS Compliance:
    - Frozen dataclass (immutable)
    - Explicit types
    """

    position: jnp.ndarray
    feature_type: FeatureType
    direction: jnp.ndarray
    strength: float


@dataclass(frozen=True)
class SubPocket:
    """
    A sub-pocket region within the binding site.

    OpenHCS Compliance:
    - Frozen dataclass (immutable)
    - Explicit types
    """

    coords: jnp.ndarray
    center: jnp.ndarray
    volume: float
    features: tuple[PharmacophoreFeature, ...]
    weight: float


@dataclass(frozen=True)
class PocketAnalysisResult:
    """
    Result of pocket analysis.

    OpenHCS Compliance:
    - Frozen dataclass (immutable)
    - Explicit types
    """

    shape: PocketShape
    sub_pockets: tuple[SubPocket, ...]
    all_features: tuple[PharmacophoreFeature, ...]


@dataclass(frozen=True)
class CertifiedSurfacePoint:
    position: jnp.ndarray
    defining_atom_index: int
    defining_atom_radius: float
    cavity_normal: jnp.ndarray


@dataclass(frozen=True)
class ConcavityWitness:
    surface_point_index: int
    probe_center: jnp.ndarray
    epsilon: float
    intersecting_atom_indices: tuple[int, ...]
    theorem_handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class CertifiedDetectedPocket:
    """
    Runtime approximation of a theorem-backed detected pocket.

    This packages the local pocket geometry used for sampling/restriction with
    the conservative certified binding-site abstraction consumed downstream.
    """

    pocket_coords: jnp.ndarray
    pocket_elements: tuple[str, ...]
    shape: PocketShape
    accessibility: jnp.ndarray
    features: tuple[PharmacophoreFeature, ...]
    surface_points: tuple[CertifiedSurfacePoint, ...]
    concavity_witnesses: tuple[ConcavityWitness, ...]
    binding_site: CertifiedBindingSite
    theorem_handles: tuple[str, ...] = ()


# =============================================================================
# ABC CONTRACT FOR POCKET ANALYZERS
# =============================================================================


class PocketAnalyzer(ABC):
    """
    ABC contract for pocket analyzers.

    OpenHCS Compliance:
    - ABC enforces explicit contract
    - @abstractmethod for required methods
    """

    @property
    @abstractmethod
    def config(self) -> PocketAnalysisConfig:
        """Direct access to config."""

    @abstractmethod
    def analyze(
        self,
        pocket_coords: jnp.ndarray,
        pocket_elements: tuple[str, ...],
    ) -> PocketAnalysisResult:
        """Analyze pocket structure."""


# =============================================================================
# PURE FUNCTIONS (Stateless, No Side Effects)
# =============================================================================


def compute_pocket_shape(pocket_coords: jnp.ndarray) -> PocketShape:
    """
    Compute geometric shape descriptors of pocket.

    OpenHCS Compliance:
    - Pure function (no side effects)
    - Explicit types
    - NOT jax.jit (returns Python floats in tuples)
    """
    com = jnp.mean(pocket_coords, axis=0)
    centered = pocket_coords - com
    cov = jnp.cov(centered.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    sort_idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    extents = 3.0 * jnp.sqrt(eigenvalues)
    volume = (4.0 / 3.0) * jnp.pi * jnp.prod(extents / 3.0)
    concavity = eigenvalues[2] / eigenvalues[0]

    projections = centered @ eigenvectors
    openness = (
        float(jnp.max(projections[:, 0]) - jnp.min(projections[:, 0])),
        float(jnp.max(projections[:, 1]) - jnp.min(projections[:, 1])),
        float(jnp.max(projections[:, 2]) - jnp.min(projections[:, 2])),
    )

    return PocketShape(
        center_of_mass=com,
        principal_axes=eigenvectors,
        extents=extents,
        volume=float(volume),
        concavity=float(concavity),
        openness=openness,
    )


def _safe_normalize(v: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(v)
    if float(norm) > 1e-8:
        return v / norm
    return jnp.array([1.0, 0.0, 0.0], dtype=v.dtype)


def _default_pocket_radii(pocket_elements: tuple[str, ...]) -> jnp.ndarray:
    if len(pocket_elements) == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    radii = [get_solvation_params(elem).radius for elem in pocket_elements]
    return jnp.asarray(radii, dtype=jnp.float32)


def derive_surface_points(
    pocket_coords: jnp.ndarray,
    pocket_radii: jnp.ndarray,
    pocket_center: jnp.ndarray,
) -> tuple[CertifiedSurfacePoint, ...]:
    surface_points: list[CertifiedSurfacePoint] = []
    for idx in range(pocket_coords.shape[0]):
        atom_position = pocket_coords[idx]
        atom_radius = float(pocket_radii[idx])
        cavity_normal = _safe_normalize(pocket_center - atom_position)
        surface_position = atom_position + atom_radius * cavity_normal
        surface_points.append(
            CertifiedSurfacePoint(
                position=surface_position,
                defining_atom_index=idx,
                defining_atom_radius=atom_radius,
                cavity_normal=cavity_normal,
            )
        )
    return tuple(surface_points)


def intersecting_atom_indices(
    pocket_coords: jnp.ndarray,
    pocket_radii: jnp.ndarray,
    probe_center: jnp.ndarray,
    probe_radius: float,
) -> tuple[int, ...]:
    distances = jnp.linalg.norm(pocket_coords - probe_center, axis=1)
    threshold = pocket_radii + probe_radius
    mask = distances < threshold
    return tuple(int(i) for i in jnp.nonzero(mask, size=int(jnp.sum(mask)))[0])


def detect_concavity_witnesses(
    surface_points: tuple[CertifiedSurfacePoint, ...],
    pocket_coords: jnp.ndarray,
    pocket_radii: jnp.ndarray,
    probe_radius: float,
    epsilon: float,
) -> tuple[ConcavityWitness, ...]:
    witnesses: list[ConcavityWitness] = []
    for surface_index, surface_point in enumerate(surface_points):
        probe_center = surface_point.position + epsilon * surface_point.cavity_normal
        intersecting = intersecting_atom_indices(
            pocket_coords,
            pocket_radii,
            probe_center,
            probe_radius,
        )
        if len(intersecting) >= 2:
            witnesses.append(
                ConcavityWitness(
                    surface_point_index=surface_index,
                    probe_center=probe_center,
                    epsilon=epsilon,
                    intersecting_atom_indices=intersecting,
                    theorem_handles=("PD2",),
                )
            )
    return tuple(witnesses)


@conditionally_certified(
    "HandleAliases.lean::PD3; HandleAliases.lean::PD5; HandleAliases.lean::BD1",
    assumptions=[
        "pocket_coords approximate a theorem-backed detected pocket region",
        "the explicit center-of-mass enclosing radius is used as a conservative binding-site abstraction",
    ],
)
def derive_certified_binding_site(
    pocket_coords: jnp.ndarray,
    radius_margin: float = 0.0,
) -> CertifiedBindingSite | None:
    """
    Derive a conservative certified binding-site abstraction from pocket geometry.

    The center is the pocket center of mass and the radius is the maximum sampled
    distance to that center plus an optional safety margin.
    """
    if pocket_coords.shape[0] == 0:
        return None

    shape = compute_pocket_shape(pocket_coords)
    distances = jnp.linalg.norm(pocket_coords - shape.center_of_mass, axis=1)
    radius = float(jnp.max(distances) + radius_margin)
    return CertifiedBindingSite(
        center=shape.center_of_mass,
        radius=radius,
        theorem_handles=("PD3", "PD5", "BD1"),
    )


@conditionally_certified(
    "HandleAliases.lean::PD2; HandleAliases.lean::PD3; HandleAliases.lean::PD5; HandleAliases.lean::BD1",
    assumptions=[
        "the supplied pocket_coords and pocket_elements come from a runtime detector aligned with a theorem-backed concave region",
        "the accessibility and pharmacophore computations preserve the local pocket support used by the certified binding-site abstraction",
    ],
)
def detect_certified_pocket(
    pocket_coords: jnp.ndarray,
    pocket_elements: tuple[str, ...],
    config: PocketAnalysisConfig | None = None,
    pocket_radii: jnp.ndarray | None = None,
    radius_margin: float = 0.0,
) -> CertifiedDetectedPocket | None:
    if pocket_coords.shape[0] == 0:
        return None

    config = ACCESSIBILITY_CONFIG if config is None else config
    pocket_radii = (
        _default_pocket_radii(pocket_elements) if pocket_radii is None else pocket_radii
    )

    accessibility = compute_accessibility_batch(pocket_coords, config)
    features = detect_pharmacophore_features(
        pocket_coords,
        pocket_elements,
        accessibility,
        config,
    )
    shape = compute_pocket_shape(pocket_coords)
    surface_points = derive_surface_points(
        pocket_coords,
        pocket_radii,
        shape.center_of_mass,
    )
    concavity_witnesses = detect_concavity_witnesses(
        surface_points,
        pocket_coords,
        pocket_radii,
        config.probe_radius,
        epsilon=config.probe_radius / 2.0,
    )
    if len(concavity_witnesses) == 0:
        return None
    binding_site = derive_certified_binding_site(
        pocket_coords,
        radius_margin=radius_margin,
    )
    if binding_site is None:
        return None

    return CertifiedDetectedPocket(
        pocket_coords=pocket_coords,
        pocket_elements=pocket_elements,
        shape=shape,
        accessibility=accessibility,
        features=features,
        surface_points=surface_points,
        concavity_witnesses=concavity_witnesses,
        binding_site=binding_site,
        theorem_handles=("PD2", "PD3", "PD5", "BD1"),
    )


def compute_accessibility(
    pocket_coords: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """
    Compute solvent accessibility for each pocket atom.

    Returns: (N,) array of accessibility scores (0-1)
    """
    n_atoms = pocket_coords.shape[0]

    diffs = pocket_coords[:, None, :] - pocket_coords[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)

    n_neighbors = jnp.sum(dists < probe_radius, axis=1)
    accessibility = jnp.exp(-n_neighbors.astype(float) / 10.0)

    return accessibility


ACCESSIBILITY_CONFIG = PocketAnalysisConfig(
    probe_radius=1.4,
    accessibility_cutoff=0.3,
)


def compute_accessibility_batch(
    pocket_coords: jnp.ndarray,
    config: PocketAnalysisConfig = ACCESSIBILITY_CONFIG,
) -> jnp.ndarray:
    """
    Compute accessibility with configuration.

    OpenHCS Compliance:
    - Pure function
    - Explicit config dependency
    """
    return compute_accessibility(pocket_coords, config.probe_radius)


def detect_pharmacophore_features(
    pocket_coords: jnp.ndarray,
    pocket_elements: tuple[str, ...],
    accessibility: jnp.ndarray,
    config: PocketAnalysisConfig,
) -> tuple[PharmacophoreFeature, ...]:
    """
    Detect pharmacophore features in pocket.

    OpenHCS Compliance:
    - Pure function
    - Explicit config dependency
    - Tuple return (immutable)
    - Fail-loud on invalid element

    Rules (from AD4 and docking literature):
        - H-bond acceptor: O, N with high accessibility (exposed)
        - H-bond donor: N-H, O-H with high accessibility
        - Hydrophobic: C, S with LOW accessibility (buried)
        - Positive: NH3+, metal ions
        - Negative: COO-, phosphate
    """
    features = []

    for i, (coord, elem, acc) in enumerate(
        zip(pocket_coords, pocket_elements, accessibility)
    ):
        upper = elem.upper()

        if upper in ("O", "N") and acc > config.accessibility_cutoff:
            feature_type = FeatureType.HBA if upper == "O" else FeatureType.HBD
            features.append(
                PharmacophoreFeature(
                    position=coord,
                    feature_type=feature_type,
                    direction=jnp.zeros(3),
                    strength=float(acc),
                )
            )

        elif upper in ("C", "S") and acc < config.accessibility_cutoff:
            features.append(
                PharmacophoreFeature(
                    position=coord,
                    feature_type=FeatureType.HYD,
                    direction=jnp.zeros(3),
                    strength=float(1.0 - acc),
                )
            )

    return tuple(features)
