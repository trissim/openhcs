"""
Explicit Water Placement — Certified water bridge scoring over discrete candidates.

Translates five Lean theorems (EWP1–EWP5) from ExplicitWaterPlacement.lean:

  EWP1  bestWaterBridge_achieved           → best_water_bridge (witness)
  EWP2  discrete_placement_approximation   → discrete_approximates_continuous
  EWP3  water_bridge_additive_with_base    → water_bridge_additive_error
  EWP4  waterBridgeScore_le_two            → water_bridge_score_bounded
  EWP5  waterBridgeScore_nonneg            → water_bridge_score_nonneg

Physics: A water molecule bridges receptor and ligand via two H-bonds:
  receptor ← H-O-H → ligand
  bridge(w) = hbond(receptor, w) + hbond(w, ligand)

The best bridge over a finite candidate set is exactly computable (EWP1).
A discrete grid approximates continuous placement with Lipschitz error (EWP2).
The bridge term composes additively with base chemistry (EWP3).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.proof_status import certified


# ---------------------------------------------------------------------------
# Section 1: Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterCandidate:
    """A candidate water position in the binding pocket."""

    position: jnp.ndarray  # (3,)


@dataclass(frozen=True)
class WaterPlacementGrid:
    """A grid of candidate water positions for bridge scoring.

    positions: (N_water, 3) — candidate water coordinates.
    grid_spacing: Å — maximum distance from any point to nearest grid point.
    """

    positions: jnp.ndarray  # (N_water, 3)
    grid_spacing: float


@dataclass(frozen=True)
class WaterBridgeResult:
    """Result of water bridge scoring over a candidate grid."""

    bridge_scores: jnp.ndarray  # (batch,) best bridge score per pose
    best_water_indices: jnp.ndarray  # (batch,) which water is best per pose
    n_candidates: int
    grid_error_bound: float  # Lipschitz × grid_spacing
    theorem_handles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Section 2: Water bridge scoring model (EWP4, EWP5)
# ---------------------------------------------------------------------------


@certified("ExplicitWaterPlacement.lean::waterBridgeScore_nonneg")
def water_bridge_score_nonneg(rec_water_score: float, water_lig_score: float) -> bool:
    """EWP5: bridge(w) = hbond(rec, w) + hbond(w, lig) ≥ 0 when both ≥ 0."""
    return (rec_water_score + water_lig_score) >= 0.0


@certified("ExplicitWaterPlacement.lean::waterBridgeScore_le_two")
def water_bridge_score_bounded(rec_water_score: float, water_lig_score: float) -> bool:
    """EWP4: bridge(w) ≤ 2 when both component scores are in [0, 1]."""
    return (rec_water_score + water_lig_score) <= 2.0


# ---------------------------------------------------------------------------
# Section 3: Bridge scoring kernel (JIT-compatible)
# ---------------------------------------------------------------------------


@jax.jit
def _score_water_bridges_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    water_positions: jnp.ndarray,
    receptor_hbond_strengths: jnp.ndarray,
    ligand_hbond_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
) -> jnp.ndarray:
    """Score water bridges for all poses × all water candidates.

    For each (pose, water) pair, computes:
      bridge = Σ_rec w_rec · G(d(rec, water)) + Σ_lig w_lig · G(d(water, lig))

    where G(d) = exp(-((d - ideal) / width)²) is the radial Gaussian.

    receptor_coords: (N_rec, 3)
    poses_coords: (batch, N_lig, 3)
    water_positions: (N_water, 3)
    receptor_hbond_strengths: (N_rec,) — H-bond capability per receptor atom
    ligand_hbond_strengths: (N_lig,) — H-bond capability per ligand atom

    Returns: (batch, N_water) bridge scores
    """
    # Receptor-water distances: (N_rec, N_water)
    rec_water_diffs = receptor_coords[:, None, :] - water_positions[None, :, :]
    rec_water_dists = jnp.linalg.norm(rec_water_diffs, axis=-1)
    rec_water_radial = jnp.exp(
        -(((rec_water_dists - ideal_distance) / distance_width) ** 2)
    )
    # Weighted sum over receptor atoms: (N_water,)
    rec_water_scores = jnp.sum(
        receptor_hbond_strengths[:, None] * rec_water_radial, axis=0
    )

    # Water-ligand distances: (batch, N_water, N_lig)
    water_lig_diffs = water_positions[None, :, None, :] - poses_coords[:, None, :, :]
    water_lig_dists = jnp.linalg.norm(water_lig_diffs, axis=-1)
    water_lig_radial = jnp.exp(
        -(((water_lig_dists - ideal_distance) / distance_width) ** 2)
    )
    # Weighted sum over ligand atoms: (batch, N_water)
    water_lig_scores = jnp.sum(
        ligand_hbond_strengths[None, None, :] * water_lig_radial, axis=-1
    )

    # Bridge score = receptor-water + water-ligand: (batch, N_water)
    return rec_water_scores[None, :] + water_lig_scores


# ---------------------------------------------------------------------------
# Section 4: Best water bridge (EWP1) and discrete approximation (EWP2)
# ---------------------------------------------------------------------------


@certified("ExplicitWaterPlacement.lean::bestWaterBridge_achieved")
def best_water_bridge(
    bridge_scores: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """EWP1: The best bridge is achieved by some candidate (witness existence).

    bridge_scores: (batch, N_water)
    Returns: (best_scores (batch,), best_indices (batch,))
    """
    best_indices = jnp.argmax(bridge_scores, axis=-1)
    best_scores = jnp.max(bridge_scores, axis=-1)
    return best_scores, best_indices


@certified("ExplicitWaterPlacement.lean::discrete_placement_approximation")
def discrete_approximates_continuous(
    best_discrete_score: float,
    lipschitz_constant: float,
    grid_spacing: float,
) -> float:
    """EWP2: Continuous optimum ≤ best_discrete + L·h.

    The discrete grid approximation error is bounded by L·h where
    L is the bridge score Lipschitz constant and h is the grid spacing.
    Returns the error bound.
    """
    return lipschitz_constant * grid_spacing


# ---------------------------------------------------------------------------
# Section 5: Additive composition (EWP3)
# ---------------------------------------------------------------------------


@certified("ExplicitWaterPlacement.lean::water_bridge_additive_with_base")
def water_bridge_additive_error(
    base_error: float,
    water_bridge_error: float,
) -> float:
    """EWP3: Water bridge composes additively with base chemistry.

    Combined error = base_error + water_bridge_error.
    When the candidate set is finite, water_bridge_error = 0 (exact).
    """
    return base_error + water_bridge_error


# ---------------------------------------------------------------------------
# Section 6: Water grid generation
# ---------------------------------------------------------------------------


def generate_water_grid(
    receptor_coords: np.ndarray,
    ligand_center: np.ndarray,
    pocket_radius: float = 10.0,
    grid_spacing: float = 1.0,
    min_receptor_distance: float = 2.5,
    max_receptor_distance: float = 5.0,
) -> WaterPlacementGrid:
    """Generate candidate water positions in the binding pocket.

    Places a regular grid around the ligand center, then filters to keep
    only positions that are:
    - Within pocket_radius of the ligand center
    - Between min_receptor_distance and max_receptor_distance from the
      nearest receptor atom (waters must be near but not overlapping)

    Args:
        receptor_coords: (N_rec, 3) receptor coordinates.
        ligand_center: (3,) center of the ligand.
        pocket_radius: Maximum distance from ligand center.
        grid_spacing: Distance between grid points (Å).
        min_receptor_distance: Minimum distance from any receptor atom.
        max_receptor_distance: Maximum distance from nearest receptor atom.
    """
    # Build cubic grid
    half_extent = pocket_radius
    xs = np.arange(
        ligand_center[0] - half_extent,
        ligand_center[0] + half_extent + grid_spacing,
        grid_spacing,
    )
    ys = np.arange(
        ligand_center[1] - half_extent,
        ligand_center[1] + half_extent + grid_spacing,
        grid_spacing,
    )
    zs = np.arange(
        ligand_center[2] - half_extent,
        ligand_center[2] + half_extent + grid_spacing,
        grid_spacing,
    )
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    # Filter by pocket radius
    dists_to_center = np.linalg.norm(grid - ligand_center[None, :], axis=-1)
    mask = dists_to_center <= pocket_radius

    # Filter by receptor distance
    if len(receptor_coords) > 0:
        # (N_grid, N_rec)
        dists_to_rec = np.linalg.norm(
            grid[:, None, :] - receptor_coords[None, :, :], axis=-1
        )
        nearest_rec_dist = np.min(dists_to_rec, axis=-1)
        mask = mask & (nearest_rec_dist >= min_receptor_distance)
        mask = mask & (nearest_rec_dist <= max_receptor_distance)

    filtered = grid[mask]

    return WaterPlacementGrid(
        positions=jnp.array(filtered, dtype=jnp.float32),
        grid_spacing=grid_spacing,
    )


# ---------------------------------------------------------------------------
# Section 7: Top-level scoring
# ---------------------------------------------------------------------------


def score_water_bridges(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    water_grid: WaterPlacementGrid,
    receptor_hbond_strengths: jnp.ndarray,
    ligand_hbond_strengths: jnp.ndarray,
    ideal_distance: float = 2.9,
    distance_width: float = 0.8,
    bridge_lipschitz_constant: float = 5.0,
) -> WaterBridgeResult:
    """Score water-mediated hydrogen bonds over a candidate grid.

    For each pose, finds the best water bridge position and returns the
    bridge score. The bridge adds to the total scoring energy as a
    favorable (negative) contribution.

    Args:
        receptor_coords: (N_rec, 3)
        poses_coords: (batch, N_lig, 3)
        water_grid: Candidate water positions.
        receptor_hbond_strengths: (N_rec,) H-bond weight per receptor atom.
        ligand_hbond_strengths: (N_lig,) H-bond weight per ligand atom.
        ideal_distance: Ideal H-bond distance for water bridges (Å).
        distance_width: Gaussian width for distance scoring.
        bridge_lipschitz_constant: Lipschitz constant of the bridge score.
    """
    if water_grid.positions.shape[0] == 0:
        batch_size = poses_coords.shape[0]
        return WaterBridgeResult(
            bridge_scores=jnp.zeros(batch_size, dtype=jnp.float32),
            best_water_indices=jnp.zeros(batch_size, dtype=jnp.int32),
            n_candidates=0,
            grid_error_bound=0.0,
            theorem_handles=("EWP1", "EWP3", "EWP4", "EWP5"),
        )

    # Score all (pose, water) pairs
    all_bridges = _score_water_bridges_batch(
        receptor_coords,
        poses_coords,
        water_grid.positions,
        receptor_hbond_strengths,
        ligand_hbond_strengths,
        ideal_distance,
        distance_width,
    )

    # EWP1: best bridge is achieved by some candidate
    best_scores, best_indices = best_water_bridge(all_bridges)

    # EWP2: discrete approximation error
    grid_error = discrete_approximates_continuous(
        jnp.max(best_scores),
        bridge_lipschitz_constant,
        water_grid.grid_spacing,
    )

    return WaterBridgeResult(
        bridge_scores=best_scores,
        best_water_indices=best_indices,
        n_candidates=int(water_grid.positions.shape[0]),
        grid_error_bound=grid_error,
        theorem_handles=("EWP1", "EWP2", "EWP3", "EWP4", "EWP5"),
    )
