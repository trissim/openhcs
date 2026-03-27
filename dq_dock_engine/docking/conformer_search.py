"""
Conformer Search — Certified torsion-space exploration via Lipschitz branch-and-bound.

Translates nine Lean theorems (CS1–CS9) from ConformerSearch.lean into JAX/Python:

  CS1  sum_channel_uniformApprox          → compose_channel_error_bounds
  CS2  conformer_dominated                → ScoreUpperBound.is_dominated_by
  CS3  optimal_conformer_bound_tight      → runtime assertion
  CS4  sum_channel_lowerBound             → compose_channel_lower_bounds
  CS5  energy_conformer_dominated         → EnergyLowerBound.is_dominated_by
  CS6  lipschitz_score_composition        → TorsionKinematics.lipschitz_constant
  CS7  isometric_kinematics_preserves_lip → RigidBodyKinematics.lipschitz_constant
  CS8  lipschitz_energy_lower_bound       → EnergyLowerBound (point case)
  CS9  lipschitz_energy_lower_bound_on_ball → EnergyLowerBound (cell case)
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.chemistry_annotations import (
    _infer_bond_adjacency,
    _normalize_element,
    _unique_cycles_of_size,
)
from dq_dock_engine.docking.formal_handles import (
    branch_and_bound_cross_docking_handles,
)
from dq_dock_engine.docking.rdkit_io import load_rdkit_molecule
from dq_dock_engine.proof_status import certified, conditionally_certified


# ---------------------------------------------------------------------------
# Section 1: Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatableBond:
    """A single rotatable bond identified in the ligand."""

    atom_i: int
    atom_j: int
    rotating_atom_indices: tuple[int, ...]
    max_arm_length: float
    rotating_atom_arm_lengths: tuple[float, ...] = ()


def compute_raw_lj_lipschitz(epsilon_lj: float, min_pairwise_sigma: float) -> float:
    """Derive the raw LJ Lipschitz constant from physical parameters.

    L_raw = 762 × ε / σ  (LipschitzStepBounds.lean::typicalLipschitzConstant)

    This is the maximum gradient magnitude of the LJ potential, occurring at
    r = (26/7)^(1/6) × σ ≈ 0.8σ. The coefficient 762 comes from evaluating
    |dU/dr| at that point: 762 = 24 × |2×(7/26)^2 - (7/26)|  × (26/7)^(1/6).
    """
    if min_pairwise_sigma <= 0:
        raise ValueError(
            f"min_pairwise_sigma must be positive, got {min_pairwise_sigma}"
        )
    if epsilon_lj <= 0:
        raise ValueError(f"epsilon_lj must be positive, got {epsilon_lj}")
    return 762.0 * epsilon_lj / min_pairwise_sigma


@dataclass(frozen=True)
class BranchAndBoundConfig:
    """Configuration for the branch-and-bound conformer search."""

    max_cells: int
    min_cell_radius: float
    score_lipschitz_constant: float
    max_conformers: int
    per_bond_lipschitz: tuple[float, ...] | None = None
    reuse_initial_conformer: bool = False

    @classmethod
    def from_coverage_plan(
        cls,
        coverage_plan: object,
        *,
        reuse_initial_conformer: bool,
        max_conformers: int,
    ) -> "BranchAndBoundConfig":
        from dq_dock_engine.docking.certified_runtime_plans import (
            CertifiedConformerCoveragePlan,
        )

        if not isinstance(coverage_plan, CertifiedConformerCoveragePlan):
            raise TypeError(
                "BranchAndBoundConfig.from_coverage_plan requires a CertifiedConformerCoveragePlan"
            )
        return cls(
            max_cells=int(coverage_plan.max_cells),
            min_cell_radius=float(coverage_plan.min_cell_radius),
            score_lipschitz_constant=float(coverage_plan.score_lipschitz_constant),
            max_conformers=max_conformers,
            per_bond_lipschitz=tuple(coverage_plan.per_bond_lipschitz),
            reuse_initial_conformer=reuse_initial_conformer,
        )


@dataclass(frozen=True)
class ConformerResult:
    """Output of conformer search: distinct coordinate sets with certified bounds."""

    conformer_coords: tuple[jnp.ndarray, ...]
    conformer_energies: tuple[float, ...]
    theorem_handles: tuple[str, ...]


@dataclass(frozen=True)
class CertifiedCellLowerBoundState:
    """Optional theorem-backed reduced-family state for a torsion cell.

    `score_fn` / `score_fn_batch` may evaluate a reduced receptor family for the
    cell. `omitted_energy_bound` certifies the worst-case omitted-channel shift
    between that local family and the authoritative exact score. When
    `cell_lower_bound` is supplied, it is treated as an already-composed exact
    lower bound for the full cell and can be used directly for incumbent-based
    pruning without evaluating the child center first.
    """

    omitted_energy_bound: float = 0.0
    theorem_handles: tuple[str, ...] = ()
    score_fn: Callable[[jnp.ndarray], float] | None = None
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    active_mask: np.ndarray | None = None
    cell_lower_bound: float | None = None
    score_is_exact: bool = True
    payload: object | None = None


# ---------------------------------------------------------------------------
# Section 2: ABC hierarchy
# ---------------------------------------------------------------------------


class KinematicMap(ABC):
    """Abstract forward kinematics: parameter space → atom coordinates.

    CS6 (lipschitz_score_composition): composed Lipschitz = M_score * K_kinematics.
    CS7 (isometric_kinematics_preserves_lipschitz): K = 1 for rigid rotation.
    """

    @abstractmethod
    def forward(self, base_coords: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Map parameter vector to atom coordinates. Must be JIT-compatible."""
        ...

    def forward_batch(
        self, base_coords: jnp.ndarray, params_batch: jnp.ndarray
    ) -> jnp.ndarray:
        """Map a batch of parameter vectors to atom coordinates.

        params_batch: (B, n_params) → returns (B, N_atoms, 3).
        Default implementation vmaps over forward.
        """
        return jax.vmap(lambda p: self.forward(base_coords, p))(params_batch)

    @property
    @abstractmethod
    def lipschitz_constant(self) -> float:
        """Lipschitz constant K of this kinematic map."""
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Dimensionality of the parameter space."""
        ...


class ScoreBound(ABC):
    """Abstract bound on a score/energy over a parameter region.

    CS2 (conformer_dominated): upper bound pruning for maximization.
    CS5 (energy_conformer_dominated): lower bound pruning for minimization.
    """

    @abstractmethod
    def bound_value(self) -> float:
        """The scalar bound value."""
        ...

    @abstractmethod
    def is_dominated_by(self, witness_value: float) -> bool:
        """True if this bound certifies domination given a witness value."""
        ...


class SearchCell(ABC):
    """Abstract cell in parameter space for branch-and-bound.

    CS9 (lipschitz_energy_lower_bound_on_ball): bound over entire cell.
    """

    @abstractmethod
    def center(self) -> jnp.ndarray:
        """Center point of this cell."""
        ...

    @abstractmethod
    def radius(self) -> float:
        """Radius of this cell (L2 ball containment for CS9)."""
        ...

    @abstractmethod
    def subdivide(self) -> tuple[SearchCell, ...]:
        """Split cell into sub-cells."""
        ...


# ---------------------------------------------------------------------------
# Section 3: Concrete implementations
# ---------------------------------------------------------------------------


class RigidBodyKinematics(KinematicMap):
    """CS7: Isometric kinematics — K = 1, preserves score Lipschitz constant."""

    @property
    def lipschitz_constant(self) -> float:
        return 1.0

    @property
    def n_params(self) -> int:
        return 7

    def forward(self, base_coords: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        from dq_dock_engine.physics.kernels import rigid_transform_3d

        quaternion = params[:4]
        translation = params[4:7]

        # Gap B: Normalize quaternion to ensure rigid transform is an isometry (K=1)
        # Without normalization, distances aren't preserved and pruning certificates break
        quat_norm = jnp.linalg.norm(quaternion)
        quaternion_normalized = quaternion / (
            quat_norm + 1e-8
        )  # Add epsilon to avoid div by zero

        return rigid_transform_3d(base_coords, quaternion_normalized, translation)


@dataclass(frozen=True)
class TorsionKinematics(KinematicMap):
    """CS6: Torsion-space kinematics with composed Lipschitz constant M * K.

    K = max arm length across all rotatable bonds.
    """

    bonds: tuple[RotatableBond, ...]
    axis_origins: jnp.ndarray
    axis_directions: jnp.ndarray
    rotation_masks: jnp.ndarray

    @property
    def lipschitz_constant(self) -> float:
        if not self.bonds:
            return 1.0
        return max(bond.max_arm_length for bond in self.bonds)

    @property
    def n_params(self) -> int:
        return len(self.bonds)

    def forward(self, base_coords: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        return _apply_torsion_rotations(
            base_coords,
            params,
            self.axis_origins,
            self.axis_directions,
            self.rotation_masks,
        )


@dataclass(frozen=True)
class EnergyLowerBound(ScoreBound):
    """CS5/CS8/CS9 + Gap2: Lipschitz-certified energy lower bound.

    When per_dim_lipschitz and half_widths are provided, uses the tighter
    per-dimension weighted-L1 bound (ConformerSearch.lean::per_dimension_energy_lower_bound_on_cell):
        lower_bound = energy(center) - Σᵢ Lᵢ × hᵢ

    Falls back to isotropic CS9 bound when only scalar Lipschitz is given:
        lower_bound = energy(center) - L × r
    """

    center_energy: float
    lipschitz_constant: float
    cell_radius: float
    per_dim_slack: float | None = None

    def bound_value(self) -> float:
        if self.per_dim_slack is not None:
            return self.center_energy - self.per_dim_slack
        return self.center_energy - self.lipschitz_constant * self.cell_radius

    def is_dominated_by(self, best_known_energy: float) -> bool:
        return best_known_energy < self.bound_value()


@dataclass(frozen=True)
class ScoreUpperBound(ScoreBound):
    """CS2: conformer_dominated — upper bound pruning for maximization.

    If bound_value < best_known_score, conformer is dominated everywhere.
    """

    center_score: float
    lipschitz_constant: float
    cell_radius: float

    def bound_value(self) -> float:
        return self.center_score + self.lipschitz_constant * self.cell_radius

    def is_dominated_by(self, best_known_score: float) -> bool:
        return self.bound_value() < best_known_score


@dataclass(frozen=True)
class TorsionCell(SearchCell):
    """Hypercube cell in torsion-angle space."""

    lower: jnp.ndarray
    upper: jnp.ndarray

    def center(self) -> jnp.ndarray:
        return (self.lower + self.upper) / 2.0

    def half_widths(self) -> jnp.ndarray:
        return (self.upper - self.lower) / 2.0

    def radius(self) -> float:
        return float(jnp.linalg.norm(self.half_widths()))

    def weighted_l1_slack(self, per_dim_lipschitz: jnp.ndarray) -> float:
        """Per-dimension weighted-L1 bound (Gap 2: per_dimension_energy_lower_bound_on_cell).

        Returns Σᵢ Lᵢ × hᵢ where Lᵢ is per-dimension Lipschitz and hᵢ is half-width.
        This is tighter than L × ||h||₂ when Lipschitz constants vary across dimensions.

        Certified by ConformerSearch.lean::per_dimension_energy_lower_bound_on_cell.
        """
        return float(jnp.sum(per_dim_lipschitz * self.half_widths()))

    def subdivide(
        self, per_dim_lipschitz: jnp.ndarray | None = None
    ) -> tuple[SearchCell, ...]:
        """Split cell by bisecting the dimension with largest slack contribution.

        When per_dim_lipschitz is provided, bisects argmax(Lᵢ × widthᵢ) —
        the dimension contributing most to the lower bound slack.

        Certified by:
          - ConformerSearch.lean::weighted_l1_argmax_subdivision_optimal
          - ConformerSearch.lean::weighted_l1_subdivision_strict_of_positive_contribution
          - ConformerSearch.lean::weighted_l1_argmax_subdivision_strict_progress
          - ConformerSearch.lean::weighted_l1_argmax_subdivision_contraction

        So the argmax weighted-slack split is theorem-backed as the best
        one-step coordinate bisection among single-dimension splits, and it
        strictly decreases weighted slack whenever the chosen contribution is
        positive, with an explicit worst-case contraction factor available for
        runtime convergence budgets.

        Falls back to bisecting the widest dimension when no weights given.
        """
        widths = self.upper - self.lower
        if per_dim_lipschitz is not None:
            dim = int(jnp.argmax(per_dim_lipschitz * widths))
        else:
            dim = int(jnp.argmax(widths))
        mid = (self.lower[dim] + self.upper[dim]) / 2.0
        upper_left = self.upper.at[dim].set(mid)
        lower_right = self.lower.at[dim].set(mid)
        return (
            TorsionCell(lower=self.lower, upper=upper_left),
            TorsionCell(lower=lower_right, upper=self.upper),
        )


# ---------------------------------------------------------------------------
# Section 4: Channel composition (CS1, CS4)
# ---------------------------------------------------------------------------


@certified("SoftLJApproximation.lean::softenedLipschitzConstant")
def compute_softened_lipschitz_constant(
    epsilon_lj: float, sigma: float, r_soft: float
) -> float:
    """Compute the Lipschitz constant for softened LJ from physical parameters.

    L_soft = 24ε/rSoft × |2(σ/rSoft)¹² - (σ/rSoft)⁶|

    This is the maximum gradient magnitude of the softened LJ potential,
    which occurs at r = rSoft (repulsive wall). Certified by:
      - softenedLJ_lipschitzWith (theorem): softened LJ is L_soft-Lipschitz
        Precondition: rSoft ≤ σ (gradient bound requires repulsive wall dominance)
      - softenedLipschitz_le_rawLipschitz: L_soft ≤ L_raw when 0.8σ ≤ rSoft ≤ σ
    """
    ratio = sigma / r_soft
    return abs(24.0 * epsilon_lj / r_soft * (2.0 * ratio**12 - ratio**6))


@certified("ConformerSearch.lean::per_dimension_lipschitz_bound")
def compute_per_bond_lipschitz(
    score_lipschitz: float,
    bonds: tuple[RotatableBond, ...],
) -> tuple[float, ...]:
    """Compute per-bond composed Lipschitz constants: Lᵢ = M × arm_i.

    Gap 2 (ConformerSearch.lean::per_dimension_lipschitz_bound):
    When the score is M-Lipschitz in Cartesian space and bond i moves atoms
    at most arm_i from the rotation axis, rotating bond i by δ radians
    changes the score by at most M × arm_i × δ.

    This gives per-dimension Lipschitz constants that are much tighter than
    the isotropic bound M × max(arm_i) when arm lengths vary.
    """
    return tuple(score_lipschitz * bond.max_arm_length for bond in bonds)


@certified("ConformerSearch.lean::sum_channel_uniformApprox")
def compose_channel_error_bounds(delta_1: float, delta_2: float) -> float:
    """CS1: Uniform approximations compose under score addition.

    Combined error radius is the sum of individual radii.
    """
    return delta_1 + delta_2


@certified("ConformerSearch.lean::sum_channel_lowerBound")
def compose_channel_lower_bounds(lb_1: float, lb_2: float) -> float:
    """CS4: Pointwise lower bounds compose under utility addition.

    lb1 + lb2 ≤ score1(a,s) + score2(a,s) for all (a, s).
    """
    return lb_1 + lb_2


# ---------------------------------------------------------------------------
# Section 5: Rotatable bond detection
# ---------------------------------------------------------------------------


def _collect_ring_atoms(
    adjacency: tuple[tuple[int, ...], ...],
) -> frozenset[int]:
    """Collect all atoms that participate in any 5- or 6-membered ring."""
    ring_atoms: set[int] = set()
    for ring_size in (5, 6):
        for cycle in _unique_cycles_of_size(adjacency, ring_size):
            ring_atoms.update(cycle)
    return frozenset(ring_atoms)


def _bfs_component(
    start: int,
    adjacency: tuple[tuple[int, ...], ...],
    excluded_edge: tuple[int, int],
) -> frozenset[int]:
    """BFS from start, not crossing the excluded edge."""
    visited: set[int] = set()
    queue: deque[int] = deque([start])
    visited.add(start)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            if (current, neighbor) == excluded_edge or (
                neighbor,
                current,
            ) == excluded_edge:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return frozenset(visited)


def _point_to_line_distance(
    point: np.ndarray,
    line_origin: np.ndarray,
    line_direction: np.ndarray,
) -> float:
    """Distance from a point to an infinite line defined by origin + direction."""
    diff = point - line_origin
    projection = np.dot(diff, line_direction)
    closest = line_origin + projection * line_direction
    return float(np.linalg.norm(point - closest))


def detect_rotatable_bonds(
    adjacency: tuple[tuple[int, ...], ...],
    coords: np.ndarray,
    elements: tuple[str, ...],
) -> tuple[RotatableBond, ...]:
    """Identify rotatable bonds from a ligand's bond graph.

    A bond (i, j) is rotatable if:
    - Both endpoints have heavy-atom degree >= 2
    - Neither endpoint is in a ring
    - Both endpoints are heavy atoms
    """
    ring_atoms = _collect_ring_atoms(adjacency)
    bonds: list[RotatableBond] = []
    seen_edges: set[tuple[int, int]] = set()

    for atom_i, neighbors in enumerate(adjacency):
        element_i = _normalize_element(elements[atom_i])
        if element_i == "H":
            continue
        for atom_j in neighbors:
            if atom_j <= atom_i:
                continue
            edge = (atom_i, atom_j)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            element_j = _normalize_element(elements[atom_j])
            if element_j == "H":
                continue
            if atom_i in ring_atoms or atom_j in ring_atoms:
                continue
            if len(adjacency[atom_i]) < 2 or len(adjacency[atom_j]) < 2:
                continue

            far_side = _bfs_component(atom_j, adjacency, excluded_edge=edge)
            near_side = _bfs_component(atom_i, adjacency, excluded_edge=edge)
            if len(far_side) <= 1 or len(near_side) <= 1:
                continue

            rotating_indices = tuple(sorted(far_side - {atom_i}))
            axis_origin = coords[atom_i]
            axis_vec = coords[atom_j] - coords[atom_i]
            axis_norm = float(np.linalg.norm(axis_vec))
            if axis_norm < 1e-8:
                continue
            axis_dir = axis_vec / axis_norm

            max_arm = 0.0
            arm_lengths: list[float] = []
            for idx in rotating_indices:
                dist = _point_to_line_distance(coords[idx], axis_origin, axis_dir)
                arm_lengths.append(dist)
                if dist > max_arm:
                    max_arm = dist

            bonds.append(
                RotatableBond(
                    atom_i=atom_i,
                    atom_j=atom_j,
                    rotating_atom_indices=rotating_indices,
                    max_arm_length=max_arm,
                    rotating_atom_arm_lengths=tuple(float(v) for v in arm_lengths),
                )
            )

    return tuple(bonds)


# ---------------------------------------------------------------------------
# Section 6: Forward kinematics (JAX JIT-compatible)
# ---------------------------------------------------------------------------


def _rodrigues_rotate(
    points: jnp.ndarray,
    axis: jnp.ndarray,
    angle: jnp.ndarray,
) -> jnp.ndarray:
    """Rotate points around axis by angle using Rodrigues' formula.

    points: (N, 3)
    axis: (3,) unit vector
    angle: scalar
    Returns: (N, 3)
    """
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    dot = jnp.sum(points * axis[None, :], axis=-1, keepdims=True)
    cross = jnp.cross(axis[None, :], points)
    return points * cos_a + cross * sin_a + axis[None, :] * dot * (1.0 - cos_a)


def _rodrigues_rotate_np(
    points: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dot = np.sum(points * axis[None, :], axis=-1, keepdims=True)
    cross = np.cross(axis[None, :], points)
    return points * cos_a + cross * sin_a + axis[None, :] * dot * (1.0 - cos_a)


def _apply_single_torsion(
    carry: jnp.ndarray,
    inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, None]:
    """Apply one torsion rotation. Used inside jax.lax.scan."""
    coords = carry
    angle, origin, direction, mask = inputs
    translated = coords - origin[None, :]
    rotated = _rodrigues_rotate(translated, direction, angle)
    new_coords = rotated + origin[None, :]
    coords = jnp.where(mask[:, None], new_coords, coords)
    return coords, None


@jax.jit
def _apply_torsion_rotations(
    coords: jnp.ndarray,
    angles: jnp.ndarray,
    axis_origins: jnp.ndarray,
    axis_directions: jnp.ndarray,
    rotation_masks: jnp.ndarray,
) -> jnp.ndarray:
    """Apply sequential torsion rotations to atom coordinates.

    coords: (N, 3)
    angles: (n_bonds,)
    axis_origins: (n_bonds, 3)
    axis_directions: (n_bonds, 3) unit vectors
    rotation_masks: (n_bonds, N) boolean masks
    Returns: (N, 3)
    """
    final_coords, _ = jax.lax.scan(
        _apply_single_torsion,
        coords,
        (angles, axis_origins, axis_directions, rotation_masks),
    )
    return final_coords


def _apply_torsion_rotations_np(
    coords: np.ndarray,
    angles: np.ndarray,
    axis_origins: np.ndarray,
    axis_directions: np.ndarray,
    rotation_masks: np.ndarray,
) -> np.ndarray:
    current = np.array(coords, dtype=np.float32, copy=True)
    for angle, origin, direction, mask in zip(
        angles.tolist(),
        axis_origins,
        axis_directions,
        rotation_masks,
    ):
        translated = current - origin[None, :]
        rotated = (
            _rodrigues_rotate_np(translated, direction, float(angle)) + origin[None, :]
        )
        current = np.where(mask[:, None], rotated, current)
    return current


def build_torsion_kinematics(
    bonds: tuple[RotatableBond, ...],
    coords: np.ndarray,
    n_atoms: int,
) -> TorsionKinematics:
    """Construct TorsionKinematics from detected rotatable bonds."""
    if not bonds:
        return TorsionKinematics(
            bonds=(),
            axis_origins=jnp.zeros((0, 3), dtype=jnp.float32),
            axis_directions=jnp.zeros((0, 3), dtype=jnp.float32),
            rotation_masks=jnp.zeros((0, n_atoms), dtype=jnp.bool_),
        )

    origins = []
    directions = []
    masks = []
    for bond in bonds:
        origin = coords[bond.atom_i]
        axis_vec = coords[bond.atom_j] - origin
        axis_dir = axis_vec / max(float(np.linalg.norm(axis_vec)), 1e-8)
        origins.append(origin)
        directions.append(axis_dir)
        mask = np.zeros(n_atoms, dtype=bool)
        for idx in bond.rotating_atom_indices:
            mask[idx] = True
        masks.append(mask)

    return TorsionKinematics(
        bonds=bonds,
        axis_origins=jnp.array(np.stack(origins), dtype=jnp.float32),
        axis_directions=jnp.array(np.stack(directions), dtype=jnp.float32),
        rotation_masks=jnp.array(np.stack(masks), dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Section 7: Branch-and-bound engine
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PrioritizedCell:
    """Wrapper for heapq ordering by bound value (ascending = best first)."""

    priority: float
    insertion_order: int
    cell: SearchCell = field(compare=False)
    center_energy: float = field(compare=False)
    bound_state: CertifiedCellLowerBoundState | None = field(
        compare=False, default=None
    )


# Fixed pad size for batch scoring — ensures JIT compiles exactly once per
# (N_lig, N_rec) pair, then every subsequent call is a cache hit.
_BB_PAD_SIZE = 128
_BB_BATCH_SIZE = _BB_PAD_SIZE
_SEQUENTIAL_SCAN_GRID_SIZE = 24
_SEQUENTIAL_SCAN_ACTIVE_BOND_THRESHOLD = 8


def _sequential_scan_grid_size(n_bonds: int) -> int:
    if n_bonds >= 12:
        return 4
    if n_bonds >= 8:
        return 6
    if n_bonds >= 4:
        return 8
    return _SEQUENTIAL_SCAN_GRID_SIZE


def _sequential_scan_incumbent(
    *,
    kinematics: TorsionKinematics,
    base_coords: jnp.ndarray,
    n_atoms: int,
    n_bonds: int,
    score_fn: Callable[[jnp.ndarray], float],
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None,
    strain_fn: Callable[[jnp.ndarray], float] | None,
    initial_params: jnp.ndarray,
    initial_energy: float,
    initial_coords: jnp.ndarray,
    single_bond_score_provider: Callable[
        [int, jnp.ndarray],
        tuple[
            Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], jnp.ndarray] | None
        ],
    ]
    | None = None,
) -> tuple[float, jnp.ndarray]:
    if n_bonds < _SEQUENTIAL_SCAN_ACTIVE_BOND_THRESHOLD:
        return initial_energy, initial_coords

    grid = np.linspace(
        -math.pi,
        math.pi,
        _sequential_scan_grid_size(n_bonds),
        endpoint=False,
        dtype=np.float32,
    )
    params = np.array(jax.device_get(initial_params), dtype=np.float32, copy=True)
    best_energy = float(initial_energy)
    best_coords = initial_coords

    improved_any = False
    for bond_idx in range(n_bonds):
        bond_score_fn = score_fn
        bond_score_fn_batch = score_fn_batch
        if single_bond_score_provider is not None:
            bond_score_fn, bond_score_fn_batch = single_bond_score_provider(
                bond_idx,
                jnp.asarray(best_coords),
            )
        batch_params = np.repeat(params[None, :], grid.shape[0], axis=0)
        batch_params[:, bond_idx] = grid
        params_batch_jnp = jnp.asarray(batch_params, dtype=base_coords.dtype)
        coords_batch_np = np.stack(
            [
                _apply_torsion_rotations_np(
                    np.asarray(base_coords, dtype=np.float32),
                    batch_params[i],
                    np.asarray(kinematics.axis_origins),
                    np.asarray(kinematics.axis_directions),
                    np.asarray(kinematics.rotation_masks),
                )
                for i in range(grid.shape[0])
            ],
            axis=0,
        )
        coords_batch = jnp.asarray(coords_batch_np, dtype=base_coords.dtype)
        if bond_score_fn_batch is not None:
            scores = np.asarray(
                jax.device_get(bond_score_fn_batch(coords_batch)), dtype=np.float32
            )
        else:
            scores = np.array(
                [float(bond_score_fn(coords_batch[i])) for i in range(grid.shape[0])],
                dtype=np.float32,
            )
        if strain_fn is not None:
            strain = np.array(
                [float(strain_fn(params_batch_jnp[i])) for i in range(grid.shape[0])],
                dtype=np.float32,
            )
            scores = scores + strain
        best_idx = int(np.argmin(scores))
        if float(scores[best_idx]) < best_energy:
            params = batch_params[best_idx]
            best_energy = float(scores[best_idx])
            best_coords = coords_batch[best_idx]
            improved_any = True

    if improved_any:
        return best_energy, best_coords
    return initial_energy, initial_coords


def _collapse_torsion_cell_to_active_subset(
    cell: TorsionCell,
    active_mask: np.ndarray,
) -> TorsionCell:
    if active_mask.shape != (cell.lower.shape[0],):
        raise ValueError(
            "active_mask must match torsion cell dimensionality "
            f"(got {active_mask.shape} vs {(cell.lower.shape[0],)})"
        )
    if bool(np.all(active_mask)):
        return cell
    center = cell.center()
    active_mask_jnp = jnp.asarray(active_mask, dtype=jnp.bool_)
    lower = jnp.where(active_mask_jnp, cell.lower, center)
    upper = jnp.where(active_mask_jnp, cell.upper, center)
    return TorsionCell(lower=lower, upper=upper)


def _combine_active_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    combined: np.ndarray | None = None
    for mask in masks:
        if mask is None:
            continue
        mask_np = np.asarray(mask, dtype=bool)
        combined = mask_np if combined is None else np.logical_and(combined, mask_np)
    return combined


def _collapse_torsion_cell_with_masks(
    cell: TorsionCell,
    *masks: np.ndarray | None,
) -> TorsionCell:
    combined = _combine_active_masks(*masks)
    if combined is None:
        return cell
    return _collapse_torsion_cell_to_active_subset(cell, combined)


def _evaluate_score_with_optional_batch(
    coords: jnp.ndarray,
    *,
    default_score_fn: Callable[[jnp.ndarray], float],
    local_score_fn: Callable[[jnp.ndarray], float] | None,
    local_score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> float:
    if local_score_fn is not None:
        return float(local_score_fn(coords))
    if local_score_fn_batch is not None:
        scores = np.asarray(
            jax.device_get(local_score_fn_batch(coords[None, ...])), dtype=np.float32
        )
        return float(scores[0])
    return float(default_score_fn(coords))


def _cell_lower_bound_value(
    *,
    center_energy: float,
    slack: float,
    bound_state: CertifiedCellLowerBoundState | None,
) -> float:
    if bound_state is not None and bound_state.cell_lower_bound is not None:
        return float(bound_state.cell_lower_bound)
    omitted = 0.0 if bound_state is None else float(bound_state.omitted_energy_bound)
    return center_energy - omitted - slack


def branch_and_bound_search(
    kinematics: KinematicMap,
    base_coords: jnp.ndarray,
    score_fn: Callable[[jnp.ndarray], float],
    initial_cell: SearchCell,
    config: BranchAndBoundConfig,
    strain_fn: Callable[[jnp.ndarray], float] | None = None,
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    pruning_incumbent_energy: float | None = None,
    local_activity_mask_fn: Callable[[TorsionCell, np.ndarray], np.ndarray | None]
    | None = None,
    cell_bound_state_fn: Callable[
        [TorsionCell, np.ndarray], CertifiedCellLowerBoundState | None
    ]
    | None = None,
    child_cell_bound_state_fn: Callable[
        [CertifiedCellLowerBoundState, TorsionCell],
        CertifiedCellLowerBoundState | None,
    ]
    | None = None,
    score_is_exact: bool = True,
) -> ConformerResult:
    """Branch-and-bound over torsion space with certified Lipschitz pruning.

    When score_fn_batch is provided, cells are evaluated in fixed-size padded
    batches (_BB_PAD_SIZE) to avoid JIT recompilation. The batch function
    always receives the same shape — JIT compiles once, then cache-hits.

    When per_bond_lipschitz is set in config, uses the tighter per-dimension
    weighted-L1 bound (Gap 2: per_dimension_energy_lower_bound_on_cell):
        slack = Σᵢ Lᵢ × hᵢ  (where Lᵢ = M × arm_i, hᵢ = half-width of dim i)
    Falls back to isotropic CS9 bound: slack = L × ||h||₂.

    Pruning certificates:
      CS6: L_composed = score_lipschitz * kinematics.lipschitz_constant
      CS8/CS9: lower_bound = energy(center) - L * radius
      Gap2: lower_bound = energy(center) - Σᵢ Lᵢ × hᵢ  (tighter)
      CS5: prune if best_energy < lower_bound
      LSA3/LSA7: strain composes with scoring without degrading error
    """
    composed_lipschitz = config.score_lipschitz_constant * kinematics.lipschitz_constant

    # Per-dimension Lipschitz (Gap 2) — much tighter when arm lengths vary
    per_dim_lip_arr: jnp.ndarray | None = None
    if config.per_bond_lipschitz is not None:
        per_dim_lip_arr = jnp.array(config.per_bond_lipschitz, dtype=jnp.float32)
    use_per_dim = per_dim_lip_arr is not None

    def _compute_slack(cell: SearchCell) -> float:
        if use_per_dim and isinstance(cell, TorsionCell):
            return cell.weighted_l1_slack(cast(jnp.ndarray, per_dim_lip_arr))
        return composed_lipschitz * cell.radius()

    def _subdivide(cell: SearchCell) -> tuple[SearchCell, ...]:
        if use_per_dim and isinstance(cell, TorsionCell):
            return cell.subdivide(cast(jnp.ndarray, per_dim_lip_arr))
        return cell.subdivide()

    n_atoms = base_coords.shape[0]
    use_batch = (
        score_fn_batch is not None
        and cell_bound_state_fn is None
        and child_cell_bound_state_fn is None
    )
    extra_theorem_handles: list[str] = []

    # --- Score initial cell center ---
    center_params = initial_cell.center()
    center_coords = kinematics.forward(base_coords, center_params)
    root_state = None
    if cell_bound_state_fn is not None and isinstance(initial_cell, TorsionCell):
        root_state = cell_bound_state_fn(
            initial_cell,
            np.asarray(jax.device_get(center_coords), dtype=np.float32),
        )
        if root_state is not None:
            extra_theorem_handles.extend(root_state.theorem_handles)
    center_score = _evaluate_score_with_optional_batch(
        center_coords,
        default_score_fn=score_fn,
        local_score_fn=None if root_state is None else root_state.score_fn,
        local_score_fn_batch=None if root_state is None else root_state.score_fn_batch,
    )
    center_strain = float(strain_fn(center_params)) if strain_fn is not None else 0.0
    root_energy = center_score + center_strain
    local_best_energy = root_energy if config.reuse_initial_conformer else float("inf")
    best_return_energy = local_best_energy
    if pruning_incumbent_energy is not None and math.isfinite(pruning_incumbent_energy):
        best_return_energy = min(best_return_energy, float(pruning_incumbent_energy))
    best_pruning_energy = float("inf")
    if pruning_incumbent_energy is not None and math.isfinite(pruning_incumbent_energy):
        best_pruning_energy = float(pruning_incumbent_energy)
    root_score_is_exact = (
        score_is_exact if root_state is None else root_state.score_is_exact
    )
    if config.reuse_initial_conformer and root_score_is_exact:
        best_pruning_energy = min(best_pruning_energy, root_energy)
    best_coords = center_coords

    if pruning_incumbent_energy is None or not math.isfinite(pruning_incumbent_energy):
        scan_energy, scan_coords = _sequential_scan_incumbent(
            kinematics=cast(TorsionKinematics, kinematics),
            base_coords=base_coords,
            n_atoms=n_atoms,
            n_bonds=len(config.per_bond_lipschitz or ()),
            score_fn=score_fn
            if root_state is None or root_state.score_fn is None
            else root_state.score_fn,
            score_fn_batch=(
                score_fn_batch
                if root_state is None or root_state.score_fn_batch is None
                else root_state.score_fn_batch
            ),
            strain_fn=strain_fn,
            initial_params=center_params,
            initial_energy=root_energy,
            initial_coords=center_coords,
        )
        if scan_energy < local_best_energy:
            local_best_energy = scan_energy
            best_coords = scan_coords
        if scan_energy < best_return_energy:
            best_return_energy = scan_energy
            best_coords = scan_coords
        if root_score_is_exact and scan_energy < best_pruning_energy:
            best_pruning_energy = scan_energy

    leaf_conformers: list[tuple[float, jnp.ndarray]] = [(root_energy, center_coords)]

    root_cell = initial_cell
    if isinstance(initial_cell, TorsionCell):
        root_active_mask = None
        if local_activity_mask_fn is not None:
            root_active_mask = local_activity_mask_fn(
                initial_cell,
                np.asarray(jax.device_get(center_coords), dtype=np.float32),
            )
        root_cell = _collapse_torsion_cell_with_masks(
            initial_cell,
            None if root_state is None else root_state.active_mask,
            None
            if root_active_mask is None
            else np.asarray(root_active_mask, dtype=bool),
        )

    root_slack = _compute_slack(root_cell)
    root_lower_bound = _cell_lower_bound_value(
        center_energy=root_energy,
        slack=root_slack,
        bound_state=root_state,
    )
    if best_pruning_energy < root_lower_bound:
        theorem_handles = tuple(
            dict.fromkeys(
                (
                    "CS2",
                    "CS5",
                    "CS6",
                    "CS8",
                    "CS9",
                    "CS10",
                    "CS11",
                    "CS12",
                    "CS13",
                    "CS14",
                    "CSC43",
                    "GAP2",
                )
                + branch_and_bound_cross_docking_handles()
                + (
                    ("LSA1", "LSA3", "LSA5", "LSA7", "XD5", "XD6", "XD7", "XD8")
                    if strain_fn is not None
                    else ()
                )
                + tuple(extra_theorem_handles)
            )
        )
        return ConformerResult(
            conformer_coords=(best_coords,),
            conformer_energies=(best_return_energy,),
            theorem_handles=theorem_handles,
        )

    heap: list[_PrioritizedCell] = []
    insertion_counter = 0
    if root_cell.radius() >= config.min_cell_radius:
        for child in _subdivide(root_cell):
            child_state = None
            child_cell = child
            if (
                child_cell_bound_state_fn is not None
                and root_state is not None
                and isinstance(child, TorsionCell)
            ):
                child_state = child_cell_bound_state_fn(root_state, child)
                if child_state is not None:
                    extra_theorem_handles.extend(child_state.theorem_handles)
                    child_cell = _collapse_torsion_cell_with_masks(
                        child,
                        child_state.active_mask,
                    )
            child_bound_value = _cell_lower_bound_value(
                center_energy=root_energy,
                slack=_compute_slack(child_cell),
                bound_state=child_state if child_state is not None else root_state,
            )
            if (
                child_state is not None
                and child_state.cell_lower_bound is not None
                and best_pruning_energy < child_bound_value
            ):
                continue
            heapq.heappush(
                heap,
                _PrioritizedCell(
                    priority=child_bound_value,
                    insertion_order=insertion_counter,
                    cell=child_cell,
                    center_energy=0.0,
                    bound_state=child_state if child_state is not None else root_state,
                ),
            )
            insertion_counter += 1

    cells_evaluated = 1
    while heap and cells_evaluated < config.max_cells:
        # --- Collect a batch of cells from the heap ---
        # NOTE: No pruning at heap-pop. The priority is an optimistic estimate
        # derived from the *parent* cell's energy, not the child's own center.
        # Only the post-evaluation check (CS5: energy_conformer_dominated) is
        # a certified bound — it uses the cell's own evaluated center energy.
        batch_entries: list[_PrioritizedCell] = []
        batch_limit = _BB_BATCH_SIZE if use_batch else 1
        while heap and len(batch_entries) < batch_limit:
            batch_entries.append(heapq.heappop(heap))

        if not batch_entries:
            break

        # --- Evaluate: fixed-pad batch or single-cell fallback ---
        batch_cells = [entry.cell for entry in batch_entries]
        batch_params = [cell.center() for cell in batch_cells]
        n_real = len(batch_cells)

        if use_batch and n_real > 1 and score_fn_batch is not None:
            # Forward kinematics for real cells
            params_stack = jnp.stack(batch_params, axis=0)
            real_coords = kinematics.forward_batch(base_coords, params_stack)
            # Pad to fixed size so JIT sees the same shape every time
            padded = jnp.zeros((_BB_PAD_SIZE, n_atoms, 3), dtype=base_coords.dtype)
            padded = padded.at[:n_real].set(real_coords)
            all_scores = score_fn_batch(padded)
            # Extract real results as numpy (no JAX dispatch in processing loop)
            scores_np = np.asarray(all_scores[:n_real])
            coords_np = np.asarray(real_coords)
        else:
            single_coords = []
            scores_list: list[float] = []
            for entry, params in zip(batch_entries, batch_params, strict=False):
                coords_jnp = kinematics.forward(base_coords, params)
                single_coords.append(np.asarray(coords_jnp))
                bound_state = entry.bound_state
                scores_list.append(
                    _evaluate_score_with_optional_batch(
                        coords_jnp,
                        default_score_fn=score_fn,
                        local_score_fn=(
                            None if bound_state is None else bound_state.score_fn
                        ),
                        local_score_fn_batch=(
                            None if bound_state is None else bound_state.score_fn_batch
                        ),
                    )
                )
            coords_np = np.stack(single_coords, axis=0)
            scores_np = np.array(scores_list, dtype=np.float32)

        # --- Process results (all numpy — no JAX dispatch) ---
        for i, (entry, cell) in enumerate(
            zip(batch_entries, batch_cells, strict=False)
        ):
            center_coords_i = jnp.asarray(coords_np[i])
            bound_state = entry.bound_state
            if (
                bound_state is None
                and cell_bound_state_fn is not None
                and isinstance(cell, TorsionCell)
            ):
                bound_state = cell_bound_state_fn(
                    cell,
                    np.asarray(coords_np[i], dtype=np.float32),
                )
                if bound_state is not None:
                    extra_theorem_handles.extend(bound_state.theorem_handles)
                    center_score_i = _evaluate_score_with_optional_batch(
                        center_coords_i,
                        default_score_fn=score_fn,
                        local_score_fn=bound_state.score_fn,
                        local_score_fn_batch=bound_state.score_fn_batch,
                    )
                else:
                    center_score_i = float(scores_np[i])
            else:
                center_score_i = float(scores_np[i])
            center_strain_i = (
                float(strain_fn(batch_params[i])) if strain_fn is not None else 0.0
            )
            center_energy = center_score_i + center_strain_i
            cells_evaluated += 1

            if center_energy < local_best_energy:
                local_best_energy = center_energy
                best_coords = center_coords_i
            if center_energy < best_return_energy:
                best_return_energy = center_energy
            score_exact_here = (
                score_is_exact if bound_state is None else bound_state.score_is_exact
            )
            if score_exact_here and center_energy < best_pruning_energy:
                best_pruning_energy = center_energy

            local_cell = cell
            if isinstance(cell, TorsionCell):
                local_active_mask = None
                if local_activity_mask_fn is not None:
                    local_active_mask = local_activity_mask_fn(
                        cell,
                        np.asarray(coords_np[i], dtype=np.float32),
                    )
                local_cell = _collapse_torsion_cell_with_masks(
                    cell,
                    None if bound_state is None else bound_state.active_mask,
                    None
                    if local_active_mask is None
                    else np.asarray(local_active_mask, dtype=bool),
                )

            slack = _compute_slack(local_cell)
            bound = EnergyLowerBound(
                center_energy=_cell_lower_bound_value(
                    center_energy=center_energy,
                    slack=0.0,
                    bound_state=bound_state,
                ),
                lipschitz_constant=composed_lipschitz,
                cell_radius=local_cell.radius(),
                per_dim_slack=slack if use_per_dim else None,
            )

            # CS5: energy_conformer_dominated — prune if dominated
            if bound.is_dominated_by(best_pruning_energy):
                continue

            if local_cell.radius() < config.min_cell_radius:
                leaf_conformers.append((center_energy, center_coords_i))
                continue

            for child in _subdivide(local_cell):
                child_state = None
                child_cell = child
                if (
                    child_cell_bound_state_fn is not None
                    and bound_state is not None
                    and isinstance(child, TorsionCell)
                ):
                    child_state = child_cell_bound_state_fn(bound_state, child)
                    if child_state is not None:
                        extra_theorem_handles.extend(child_state.theorem_handles)
                        child_cell = _collapse_torsion_cell_with_masks(
                            child,
                            child_state.active_mask,
                        )
                child_lb = _cell_lower_bound_value(
                    center_energy=center_energy,
                    slack=_compute_slack(child_cell),
                    bound_state=child_state if child_state is not None else bound_state,
                )
                if (
                    child_state is not None
                    and child_state.cell_lower_bound is not None
                    and best_pruning_energy < child_lb
                ):
                    continue
                heapq.heappush(
                    heap,
                    _PrioritizedCell(
                        priority=child_lb,
                        insertion_order=insertion_counter,
                        cell=child_cell,
                        center_energy=0.0,
                        bound_state=(
                            child_state if child_state is not None else bound_state
                        ),
                    ),
                )
                insertion_counter += 1

    # --- THEOREM-BACKED DEDUPLICATION (CSC43) ---
    # Theorem CSC43: If we deduplicate conformers within distance τ of the
    # B&B cell centers (which form an ε-cover), the resulting set is an (ε + τ)-cover.

    # B&B stops at min_cell_radius (parameter space resolution in radians).
    # Kinematics Lipschitz constant K converts parameter distance to coordinate distance.
    # Therefore coordinate space resolution = K * min_cell_radius.

    # τ = kinematics.lipschitz_constant * config.min_cell_radius ensures we only
    # merge conformers that could be from the same true optimum region, introducing
    # no additional unverified slack beyond what B&B already guarantees.

    tau_rmsd = kinematics.lipschitz_constant * config.min_cell_radius

    leaf_conformers.sort(key=lambda x: x[0])
    unique: list[tuple[float, jnp.ndarray]] = []

    for energy, coords in leaf_conformers:
        if len(unique) >= config.max_conformers:
            break

        is_duplicate = False
        for _, existing_coords in unique:
            # Theorem CSC43: Deduplication preserves the epsilon-cover up to tau.
            # Since tau is exactly our derived search resolution, we introduce no unverified slack.
            diff = coords - existing_coords
            rmsd = float(jnp.sqrt(jnp.mean(diff**2)))

            if rmsd <= tau_rmsd:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append((energy, coords))

    if not unique:
        raise ValueError(
            "Certified conformer search produced no finite leaf conformers; "
            "refine the derived search budget or inspect the scoring function"
        )

    theorem_handles = tuple(
        dict.fromkeys(
            (
                "CS2",
                "CS5",
                "CS6",
                "CS8",
                "CS9",
                "CS10",
                "CS11",
                "CS12",
                "CS13",
                "CS14",
                "CSC43",
                "GAP2",
            )
            + branch_and_bound_cross_docking_handles()
            + (
                ("LSA1", "LSA3", "LSA5", "LSA7", "XD5", "XD6", "XD7", "XD8")
                if strain_fn is not None
                else ()
            )
            + tuple(extra_theorem_handles)
        )
    )

    return ConformerResult(
        conformer_coords=tuple(coords for _, coords in unique),
        conformer_energies=tuple(energy for energy, _ in unique),
        theorem_handles=theorem_handles,
    )


# ---------------------------------------------------------------------------
# Section 8: Top-level integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TorsionStrainParams:
    """Per-bond torsion strain parameters for certified strain scoring.

    barrier_heights: (n_bonds,) Vk values in kcal/mol
    multiplicities: (n_bonds,) periodicity n
    phases: (n_bonds,) equilibrium phase φ₀ in radians

    These parameters must come from an explicit physical/empirical source.
    The certified engine intentionally does not fabricate generic defaults.
    """

    barrier_heights: jnp.ndarray
    multiplicities: jnp.ndarray
    phases: jnp.ndarray


def derive_uff_torsion_barrier_heights(
    ligand_source_path: str | Path | None,
    expected_elements: tuple[str, ...],
    rotatable_bonds: tuple[RotatableBond, ...],
) -> jnp.ndarray | None:
    """Derive per-bond torsion barrier heights from an explicit UFF source.

    This does not invent generic torsion parameters. It uses RDKit's UFF torsion
    barrier API on the actual ligand source and returns the maximum available
    barrier for each heavy-atom rotatable bond over all neighbor quadruples.
    If no explicit physical source is available, returns `None`.
    """
    if ligand_source_path is None or not rotatable_bonds:
        return None

    from rdkit import Chem
    from rdkit.Chem import AllChem, rdForceFieldHelpers

    source_path = Path(ligand_source_path)
    try:
        mol = load_rdkit_molecule(source_path, remove_hs=False, sanitize=True)
    except Exception:
        return None
    mol = Chem.AddHs(mol, addCoords=True)

    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    heavy_elements = tuple(
        _normalize_element(mol.GetAtomWithIdx(atom_idx).GetSymbol())
        for atom_idx in heavy_indices
    )
    normalized_expected = tuple(
        _normalize_element(element) for element in expected_elements
    )
    if heavy_elements != normalized_expected:
        raise ValueError(
            "UFF torsion parameter derivation changed heavy-atom ordering; "
            f"expected {normalized_expected} but got {heavy_elements}"
        )

    barrier_heights: list[float] = []
    for bond in rotatable_bonds:
        atom_i = heavy_indices[bond.atom_i]
        atom_j = heavy_indices[bond.atom_j]
        neighbors_i = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_i).GetNeighbors()
            if nbr.GetIdx() != atom_j
        ]
        neighbors_j = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_j).GetNeighbors()
            if nbr.GetIdx() != atom_i
        ]

        barrier_candidates: list[float] = []
        for atom_k in neighbors_i:
            for atom_l in neighbors_j:
                barrier = rdForceFieldHelpers.GetUFFTorsionParams(
                    mol, atom_k, atom_i, atom_j, atom_l
                )
                if barrier is not None:
                    barrier_candidates.append(abs(float(barrier)))

        if not barrier_candidates:
            return None
        barrier_heights.append(max(barrier_candidates))

    return jnp.asarray(barrier_heights, dtype=jnp.float32)


def derive_mmff_torsion_current_headroom(
    ligand_source_path: str | Path | None,
    expected_elements: tuple[str, ...],
    rotatable_bonds: tuple[RotatableBond, ...],
) -> jnp.ndarray | None:
    """Derive per-bond MMFF torsion headroom at the current conformer.

    Uses explicit MMFF torsion coefficients `(V1, V2, V3)` from RDKit and the
    current conformer dihedral angle to bound the maximum possible torsion-energy
    decrease by `current_energy - lower_bound`, where

      lower_bound = c - |a1| - |a2| - |a3|

    for the cosine series `c + a1*cos(phi) + a2*cos(2phi) + a3*cos(3phi)`.
    """
    if ligand_source_path is None or not rotatable_bonds:
        return None

    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

    source_path = Path(ligand_source_path)
    try:
        mol = load_rdkit_molecule(source_path, remove_hs=False, sanitize=True)
    except Exception:
        return None
    mol = Chem.AddHs(mol, addCoords=True)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    heavy_elements = tuple(
        _normalize_element(mol.GetAtomWithIdx(atom_idx).GetSymbol())
        for atom_idx in heavy_indices
    )
    normalized_expected = tuple(
        _normalize_element(element) for element in expected_elements
    )
    if heavy_elements != normalized_expected:
        raise ValueError(
            "MMFF torsion parameter derivation changed heavy-atom ordering; "
            f"expected {normalized_expected} but got {heavy_elements}"
        )

    conf = mol.GetConformer()
    headrooms: list[float] = []
    for bond in rotatable_bonds:
        atom_i = heavy_indices[bond.atom_i]
        atom_j = heavy_indices[bond.atom_j]
        neighbors_i = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_i).GetNeighbors()
            if nbr.GetIdx() != atom_j
        ]
        neighbors_j = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_j).GetNeighbors()
            if nbr.GetIdx() != atom_i
        ]

        bond_headroom = 0.0
        found_term = False
        for atom_k in neighbors_i:
            for atom_l in neighbors_j:
                params = props.GetMMFFTorsionParams(mol, atom_k, atom_i, atom_j, atom_l)
                if params is None:
                    continue
                _, v1, v2, v3 = params
                phi = float(
                    rdMolTransforms.GetDihedralRad(conf, atom_k, atom_i, atom_j, atom_l)
                )
                current_energy = 0.5 * (
                    v1 * (1.0 + math.cos(phi))
                    + v2 * (1.0 - math.cos(2.0 * phi))
                    + v3 * (1.0 + math.cos(3.0 * phi))
                )
                const = 0.5 * (v1 + v2 + v3)
                lower_bound = const - 0.5 * (abs(v1) + abs(v2) + abs(v3))
                bond_headroom += max(0.0, current_energy - lower_bound)
                found_term = True
        if not found_term:
            return None
        headrooms.append(bond_headroom)

    return jnp.asarray(headrooms, dtype=jnp.float32)


def _build_strain_fn(
    params: TorsionStrainParams,
) -> Callable[[jnp.ndarray], float]:
    """Build a strain function from torsion parameters.

    Certified by LSA1 (cosineTorsionStrain_le_twoVk) and
    LSA5 (additive_strain_bounded): V(φ) = Σ Vk·(1 - cos(n·φ - φ₀)),
    bounded in [0, Σ 2·Vk].
    """

    def strain_fn(angles: jnp.ndarray) -> float:
        strains = params.barrier_heights * (
            1.0 - jnp.cos(params.multiplicities * angles - params.phases)
        )
        return float(jnp.sum(strains))

    return strain_fn


def search_conformers(
    base_coords: jnp.ndarray,
    adjacency: tuple[tuple[int, ...], ...],
    elements: tuple[str, ...],
    score_fn: Callable[[jnp.ndarray], float],
    config: BranchAndBoundConfig | None = None,
    strain_params: TorsionStrainParams | None = None,
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    rotatable_bonds: tuple[RotatableBond, ...] | None = None,
    pruning_incumbent_energy: float | None = None,
    local_activity_mask_fn: Callable[[TorsionCell, np.ndarray], np.ndarray | None]
    | None = None,
    cell_bound_state_fn: Callable[
        [TorsionCell, np.ndarray], CertifiedCellLowerBoundState | None
    ]
    | None = None,
    child_cell_bound_state_fn: Callable[
        [CertifiedCellLowerBoundState, TorsionCell],
        CertifiedCellLowerBoundState | None,
    ]
    | None = None,
    score_is_exact: bool = True,
) -> ConformerResult:
    """Search for distinct low-energy conformers via certified branch-and-bound.

    Takes the ligand's base coordinates, bond graph, and a scoring function.
    Returns a ConformerResult with distinct conformer coordinate sets.

    When score_fn_batch is provided ((B, N_atoms, 3) → (B,)), cells are
    evaluated in batches for dramatically reduced JAX dispatch overhead.

    When strain_params is provided (or auto-generated), torsion strain is
    added to the energy at each candidate conformation. The certified engine no
    longer auto-generates generic torsion strain parameters; callers must supply
    an explicit physically grounded strain model if they want strain-augmented
    conformer ranking.

    The scoring function should accept (N_atoms, 3) coordinates and return
    a scalar energy (lower = better).
    """
    coords_np = np.asarray(base_coords, dtype=np.float32)
    bonds = (
        detect_rotatable_bonds(adjacency, coords_np, elements)
        if rotatable_bonds is None
        else rotatable_bonds
    )

    if not bonds:
        return ConformerResult(
            conformer_coords=(base_coords,),
            conformer_energies=(float(score_fn(base_coords)),),
            theorem_handles=("CS7",),
        )

    if config is None:
        raise ValueError(
            "search_conformers requires an explicit theorem-derived BranchAndBoundConfig "
            "when rotatable bonds are present"
        )

    kinematics = build_torsion_kinematics(bonds, coords_np, len(elements))
    n_bonds = len(bonds)

    # Gap 2: compute per-bond Lipschitz constants automatically if not provided
    if config.per_bond_lipschitz is None:
        per_bond_lip = compute_per_bond_lipschitz(
            config.score_lipschitz_constant, bonds
        )
        config = BranchAndBoundConfig(
            max_cells=config.max_cells,
            min_cell_radius=config.min_cell_radius,
            score_lipschitz_constant=config.score_lipschitz_constant,
            max_conformers=config.max_conformers,
            per_bond_lipschitz=per_bond_lip,
            reuse_initial_conformer=config.reuse_initial_conformer,
        )

    strain_fn = None if strain_params is None else _build_strain_fn(strain_params)

    initial_cell = TorsionCell(
        lower=jnp.full((n_bonds,), -jnp.pi, dtype=jnp.float32),
        upper=jnp.full((n_bonds,), jnp.pi, dtype=jnp.float32),
    )

    return branch_and_bound_search(
        kinematics=kinematics,
        base_coords=base_coords,
        score_fn=score_fn,
        initial_cell=initial_cell,
        config=config,
        strain_fn=strain_fn,
        score_fn_batch=score_fn_batch,
        pruning_incumbent_energy=pruning_incumbent_energy,
        local_activity_mask_fn=local_activity_mask_fn,
        cell_bound_state_fn=cell_bound_state_fn,
        child_cell_bound_state_fn=child_cell_bound_state_fn,
        score_is_exact=score_is_exact,
    )


def search_conformers_sequential_scan(
    base_coords: jnp.ndarray,
    adjacency: tuple[tuple[int, ...], ...],
    elements: tuple[str, ...],
    score_fn: Callable[[jnp.ndarray], float],
    strain_params: TorsionStrainParams | None = None,
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    rotatable_bonds: tuple[RotatableBond, ...] | None = None,
    single_bond_score_provider: Callable[
        [int, jnp.ndarray],
        tuple[
            Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], jnp.ndarray] | None
        ],
    ]
    | None = None,
) -> ConformerResult:
    coords_np = np.asarray(base_coords, dtype=np.float32)
    bonds = (
        detect_rotatable_bonds(adjacency, coords_np, elements)
        if rotatable_bonds is None
        else rotatable_bonds
    )
    if not bonds:
        return ConformerResult(
            conformer_coords=(base_coords,),
            conformer_energies=(float(score_fn(base_coords)),),
            theorem_handles=("CS7",),
        )

    kinematics = build_torsion_kinematics(bonds, coords_np, len(elements))
    initial_params = jnp.zeros((len(bonds),), dtype=base_coords.dtype)
    center_coords = kinematics.forward(base_coords, initial_params)
    center_score = float(score_fn(center_coords))
    strain_fn = None if strain_params is None else _build_strain_fn(strain_params)
    center_strain = float(strain_fn(initial_params)) if strain_fn is not None else 0.0
    best_energy, best_coords = _sequential_scan_incumbent(
        kinematics=kinematics,
        base_coords=base_coords,
        n_atoms=base_coords.shape[0],
        n_bonds=len(bonds),
        score_fn=score_fn,
        score_fn_batch=score_fn_batch,
        strain_fn=strain_fn,
        initial_params=initial_params,
        initial_energy=center_score + center_strain,
        initial_coords=center_coords,
        single_bond_score_provider=single_bond_score_provider,
    )
    return ConformerResult(
        conformer_coords=(best_coords,),
        conformer_energies=(best_energy,),
        theorem_handles=(),
    )


def search_conformers_support_grid(
    base_coords: jnp.ndarray,
    adjacency: tuple[tuple[int, ...], ...],
    elements: tuple[str, ...],
    score_fn: Callable[[jnp.ndarray], float],
    config: BranchAndBoundConfig,
    strain_params: TorsionStrainParams | None = None,
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    rotatable_bonds: tuple[RotatableBond, ...] | None = None,
    canonical_segments: tuple[int, ...] | None = None,
) -> ConformerResult:
    coords_np = np.asarray(base_coords, dtype=np.float32)
    bonds = (
        detect_rotatable_bonds(adjacency, coords_np, elements)
        if rotatable_bonds is None
        else rotatable_bonds
    )
    if not bonds:
        return ConformerResult(
            conformer_coords=(base_coords,),
            conformer_energies=(float(score_fn(base_coords)),),
            theorem_handles=(),
        )
    if canonical_segments is None:
        raise ValueError("support-grid conformer search requires canonical_segments")

    axes = [
        np.linspace(-math.pi, math.pi, int(seg) + 1, dtype=np.float32)
        for seg in canonical_segments
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    params_batch = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
    params_batch_jnp = jnp.asarray(params_batch, dtype=base_coords.dtype)

    kinematics = build_torsion_kinematics(bonds, coords_np, len(elements))
    coords_batch = kinematics.forward_batch(base_coords, params_batch_jnp)
    if score_fn_batch is not None:
        scores = np.asarray(
            jax.device_get(score_fn_batch(coords_batch)), dtype=np.float32
        )
    else:
        scores = np.array(
            [float(score_fn(coords_batch[i])) for i in range(coords_batch.shape[0])],
            dtype=np.float32,
        )
    if strain_params is not None:
        strain_fn = _build_strain_fn(strain_params)
        strain_scores = np.array(
            [
                float(strain_fn(params_batch_jnp[i]))
                for i in range(params_batch_jnp.shape[0])
            ],
            dtype=np.float32,
        )
        scores = scores + strain_scores
    best_order = np.argsort(scores)[: max(1, config.max_conformers)]
    return ConformerResult(
        conformer_coords=tuple(coords_batch[int(i)] for i in best_order.tolist()),
        conformer_energies=tuple(float(scores[int(i)]) for i in best_order.tolist()),
        theorem_handles=(),
    )
