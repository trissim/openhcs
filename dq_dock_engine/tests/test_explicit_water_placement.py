"""Tests for explicit water placement (EWP1-EWP5)."""

import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.explicit_water_placement import (
    WaterPlacementGrid,
    best_water_bridge,
    discrete_approximates_continuous,
    generate_water_grid,
    score_water_bridges,
    water_bridge_additive_error,
    water_bridge_score_bounded,
    water_bridge_score_nonneg,
)


def test_water_bridge_score_nonneg():
    """EWP5: bridge ≥ 0 when both components ≥ 0."""
    assert water_bridge_score_nonneg(0.5, 0.3) is True
    assert water_bridge_score_nonneg(0.0, 0.0) is True


def test_water_bridge_score_bounded():
    """EWP4: bridge ≤ 2 when both components in [0, 1]."""
    assert water_bridge_score_bounded(1.0, 1.0) is True
    assert water_bridge_score_bounded(0.5, 0.5) is True


def test_best_water_bridge_witness():
    """EWP1: best bridge is achieved by some candidate."""
    bridge_scores = jnp.array([
        [0.5, 0.8, 0.3],
        [0.9, 0.1, 0.7],
    ])
    best_scores, best_indices = best_water_bridge(bridge_scores)
    np.testing.assert_allclose(best_scores, jnp.array([0.8, 0.9]), atol=1e-6)
    np.testing.assert_array_equal(best_indices, jnp.array([1, 0]))


def test_discrete_approximation_error():
    """EWP2: error = L × h."""
    error = discrete_approximates_continuous(
        best_discrete_score=1.5,
        lipschitz_constant=5.0,
        grid_spacing=0.5,
    )
    np.testing.assert_allclose(error, 2.5, atol=1e-6)


def test_additive_composition():
    """EWP3: combined error = base + water."""
    combined = water_bridge_additive_error(base_error=0.5, water_bridge_error=0.0)
    np.testing.assert_allclose(combined, 0.5, atol=1e-6)

    combined = water_bridge_additive_error(base_error=0.5, water_bridge_error=2.5)
    np.testing.assert_allclose(combined, 3.0, atol=1e-6)


def test_generate_water_grid_filters_by_distance():
    """Grid points must be between min and max receptor distance."""
    receptor_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ligand_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    grid = generate_water_grid(
        receptor_coords=receptor_coords,
        ligand_center=ligand_center,
        pocket_radius=5.0,
        grid_spacing=1.0,
        min_receptor_distance=2.5,
        max_receptor_distance=4.5,
    )
    if grid.positions.shape[0] > 0:
        dists = jnp.linalg.norm(grid.positions, axis=-1)
        assert jnp.all(dists >= 2.5 - 0.01)
        assert jnp.all(dists <= 4.5 + 0.01)


def test_score_water_bridges_shape():
    """Smoke test: scoring produces correct output shapes."""
    receptor_coords = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ], dtype=jnp.float32)
    poses_coords = jnp.array([
        [[2.9, 0.0, 0.0]],
        [[6.0, 0.0, 0.0]],
    ], dtype=jnp.float32)
    water_grid = WaterPlacementGrid(
        positions=jnp.array([
            [1.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ], dtype=jnp.float32),
        grid_spacing=2.0,
    )

    result = score_water_bridges(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        water_grid=water_grid,
        receptor_hbond_strengths=jnp.array([1.0, 0.5], dtype=jnp.float32),
        ligand_hbond_strengths=jnp.array([1.0], dtype=jnp.float32),
    )
    assert result.bridge_scores.shape == (2,)
    assert result.best_water_indices.shape == (2,)
    assert result.n_candidates == 2
    assert result.grid_error_bound >= 0.0


def test_empty_grid_returns_zero():
    """No water candidates → zero bridge score."""
    result = score_water_bridges(
        receptor_coords=jnp.zeros((2, 3), dtype=jnp.float32),
        poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
        water_grid=WaterPlacementGrid(
            positions=jnp.zeros((0, 3), dtype=jnp.float32),
            grid_spacing=1.0,
        ),
        receptor_hbond_strengths=jnp.zeros(2, dtype=jnp.float32),
        ligand_hbond_strengths=jnp.zeros(1, dtype=jnp.float32),
    )
    assert result.bridge_scores.shape == (3,)
    np.testing.assert_allclose(result.bridge_scores, 0.0)
