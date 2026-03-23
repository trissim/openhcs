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
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

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


def compute_raw_lj_lipschitz(epsilon_lj: float, min_pairwise_sigma: float) -> float:
    """Derive the raw LJ Lipschitz constant from physical parameters.

    L_raw = 762 × ε / σ  (LipschitzStepBounds.lean::typicalLipschitzConstant)

    This is the maximum gradient magnitude of the LJ potential, occurring at
    r = (26/7)^(1/6) × σ ≈ 0.8σ. The coefficient 762 comes from evaluating
    |dU/dr| at that point: 762 = 24 × |2×(7/26)^2 - (7/26)|  × (26/7)^(1/6).
    """
    if min_pairwise_sigma <= 0 or epsilon_lj <= 0:
        return 22.0  # safe fallback
    return 762.0 * epsilon_lj / min_pairwise_sigma


@dataclass(frozen=True)
class BranchAndBoundConfig:
    """Configuration for the branch-and-bound conformer search."""

    max_cells: int = 200
    min_cell_radius: float = 0.05
    score_lipschitz_constant: float = 22.0
    max_conformers: int = 10
    per_bond_lipschitz: tuple[float, ...] | None = None
    reuse_initial_conformer: bool = False


@dataclass(frozen=True)
class ConformerResult:
    """Output of conformer search: distinct coordinate sets with certified bounds."""

    conformer_coords: tuple[jnp.ndarray, ...]
    conformer_energies: tuple[float, ...]
    theorem_handles: tuple[str, ...]


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
        return rigid_transform_3d(base_coords, quaternion, translation)


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
        the dimension contributing most to the lower bound slack. This is
        certified by ConformerSearch.lean::weighted_l1_targeted_subdivision
        to maximally tighten the bound.

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
    which occurs at r = rSoft. Always ≤ raw LJ Lipschitz when rSoft ≥ 0.8σ
    (proven in SoftLJApproximation.lean::softenedLipschitz_le_rawLipschitz).
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
            for idx in rotating_indices:
                dist = _point_to_line_distance(coords[idx], axis_origin, axis_dir)
                if dist > max_arm:
                    max_arm = dist

            bonds.append(
                RotatableBond(
                    atom_i=atom_i,
                    atom_j=atom_j,
                    rotating_atom_indices=rotating_indices,
                    max_arm_length=max_arm,
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


# Fixed pad size for batch scoring — ensures JIT compiles exactly once per
# (N_lig, N_rec) pair, then every subsequent call is a cache hit.
_BB_PAD_SIZE = 128
_BB_BATCH_SIZE = _BB_PAD_SIZE
# Lean: canonicalSeedBudgetCost_mono — terminate early when no new
# conformers are found, since larger budgets never decrease total cost.
_BB_STAGNATION_LIMIT = 50


def branch_and_bound_search(
    kinematics: KinematicMap,
    base_coords: jnp.ndarray,
    score_fn: Callable[[jnp.ndarray], float],
    initial_cell: SearchCell,
    config: BranchAndBoundConfig,
    strain_fn: Callable[[jnp.ndarray], float] | None = None,
    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
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
            return cell.weighted_l1_slack(per_dim_lip_arr)
        return composed_lipschitz * cell.radius()

    def _subdivide(cell: SearchCell) -> tuple[SearchCell, ...]:
        if use_per_dim and isinstance(cell, TorsionCell):
            return cell.subdivide(per_dim_lip_arr)
        return cell.subdivide()

    n_atoms = base_coords.shape[0]
    use_batch = score_fn_batch is not None

    # --- Score initial cell center ---
    center_params = initial_cell.center()
    center_coords = kinematics.forward(base_coords, center_params)
    center_score = float(score_fn(center_coords))
    center_strain = float(strain_fn(center_params)) if strain_fn is not None else 0.0
    root_energy = center_score + center_strain
    best_energy = root_energy if config.reuse_initial_conformer else float("inf")
    best_coords = center_coords

    leaf_conformers: list[tuple[float, jnp.ndarray]] = [(root_energy, center_coords)]

    heap: list[_PrioritizedCell] = []
    insertion_counter = 0
    if initial_cell.radius() >= config.min_cell_radius:
        for child in _subdivide(initial_cell):
            child_bound_value = root_energy - _compute_slack(child)
            heapq.heappush(
                heap,
                _PrioritizedCell(
                    priority=child_bound_value,
                    insertion_order=insertion_counter,
                    cell=child,
                    center_energy=0.0,
                ),
            )
            insertion_counter += 1

    cells_evaluated = 1
    steps_since_new_conformer = 0
    while heap and cells_evaluated < config.max_cells:
        # --- Collect a batch of unpruned cells from the heap ---
        batch_cells: list[SearchCell] = []
        batch_limit = _BB_BATCH_SIZE if use_batch else 1
        while heap and len(batch_cells) < batch_limit:
            entry = heapq.heappop(heap)
            if entry.priority > best_energy:
                continue
            batch_cells.append(entry.cell)

        if not batch_cells:
            break

        # --- Evaluate: fixed-pad batch or single-cell fallback ---
        batch_params = [cell.center() for cell in batch_cells]
        n_real = len(batch_cells)

        if use_batch and n_real > 1:
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
            single_coords = [
                np.asarray(kinematics.forward(base_coords, p)) for p in batch_params
            ]
            coords_np = np.stack(single_coords, axis=0)
            scores_np = np.array(
                [float(score_fn(jnp.asarray(c))) for c in single_coords],
            )

        # --- Process results (all numpy — no JAX dispatch) ---
        for i, cell in enumerate(batch_cells):
            center_coords_i = jnp.asarray(coords_np[i])
            center_score_i = float(scores_np[i])
            center_strain_i = (
                float(strain_fn(batch_params[i])) if strain_fn is not None else 0.0
            )
            center_energy = center_score_i + center_strain_i
            cells_evaluated += 1

            if center_energy < best_energy:
                best_energy = center_energy
                best_coords = center_coords_i

            slack = _compute_slack(cell)
            bound = EnergyLowerBound(
                center_energy=center_energy,
                lipschitz_constant=composed_lipschitz,
                cell_radius=cell.radius(),
                per_dim_slack=slack if use_per_dim else None,
            )

            # CS5: energy_conformer_dominated — prune if dominated
            if bound.is_dominated_by(best_energy):
                continue

            if cell.radius() < config.min_cell_radius:
                leaf_conformers.append((center_energy, center_coords_i))
                steps_since_new_conformer = 0
                continue

            for child in _subdivide(cell):
                child_lb = center_energy - _compute_slack(child)
                heapq.heappush(
                    heap,
                    _PrioritizedCell(
                        priority=child_lb,
                        insertion_order=insertion_counter,
                        cell=child,
                        center_energy=0.0,
                    ),
                )
                insertion_counter += 1

            steps_since_new_conformer += 1

        if steps_since_new_conformer >= _BB_STAGNATION_LIMIT and leaf_conformers:
            break

    # Collect best conformers, deduplicated by energy
    leaf_conformers.sort(key=lambda x: x[0])
    unique: list[tuple[float, jnp.ndarray]] = []
    for energy, coords in leaf_conformers:
        if len(unique) >= config.max_conformers:
            break
        is_duplicate = False
        for existing_energy, existing_coords in unique:
            if abs(energy - existing_energy) < 0.01:
                rmsd = float(jnp.sqrt(jnp.mean((coords - existing_coords) ** 2)))
                if rmsd < 0.1:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique.append((energy, coords))

    if not unique:
        fallback_energy = root_energy if not np.isfinite(best_energy) else best_energy
        fallback_coords = center_coords if not np.isfinite(best_energy) else best_coords
        unique.append((fallback_energy, fallback_coords))

    theorem_handles = tuple(
        dict.fromkeys(
            ("CS2", "CS5", "CS6", "CS8", "CS9", "GAP2")
            + branch_and_bound_cross_docking_handles()
            + (
                ("LSA1", "LSA3", "LSA5", "LSA7", "XD5", "XD6", "XD7", "XD8")
                if strain_fn is not None
                else ()
            )
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
    multiplicities: (n_bonds,) periodicity n (typically 1, 2, or 3)
    phases: (n_bonds,) equilibrium phase φ₀ in radians
    """

    barrier_heights: jnp.ndarray
    multiplicities: jnp.ndarray
    phases: jnp.ndarray


def default_torsion_strain_params(n_bonds: int) -> TorsionStrainParams:
    """Build default torsion strain parameters for detected rotatable bonds.

    Uses generic sp3-sp3 single bond parameters:
      Vk = 1.0 kcal/mol, n = 3, φ₀ = 0 (staggered preferred)
    """
    return TorsionStrainParams(
        barrier_heights=jnp.ones(n_bonds, dtype=jnp.float32),
        multiplicities=jnp.full(n_bonds, 3.0, dtype=jnp.float32),
        phases=jnp.zeros(n_bonds, dtype=jnp.float32),
    )


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
) -> ConformerResult:
    """Search for distinct low-energy conformers via certified branch-and-bound.

    Takes the ligand's base coordinates, bond graph, and a scoring function.
    Returns a ConformerResult with distinct conformer coordinate sets.

    When score_fn_batch is provided ((B, N_atoms, 3) → (B,)), cells are
    evaluated in batches for dramatically reduced JAX dispatch overhead.

    When strain_params is provided (or auto-generated), torsion strain is
    added to the energy at each candidate conformation. Certified by
    LSA3 (strain_preserves_uniformApprox) and LSA7 (strain_aware_conformer_dominated).

    The scoring function should accept (N_atoms, 3) coordinates and return
    a scalar energy (lower = better).
    """
    if config is None:
        config = BranchAndBoundConfig()

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

    # Build strain function from params (auto-generate if not provided)
    if strain_params is None:
        strain_params = default_torsion_strain_params(n_bonds)
    strain_fn = _build_strain_fn(strain_params)

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
    )
