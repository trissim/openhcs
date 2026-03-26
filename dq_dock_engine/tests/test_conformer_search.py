"""Tests for conformer_search module — CS1-CS9 Lean theorem translations."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from dq_dock_engine.docking.conformer_search import (
    BranchAndBoundConfig,
    EnergyLowerBound,
    RigidBodyKinematics,
    RotatableBond,
    ScoreUpperBound,
    TorsionCell,
    TorsionKinematics,
    branch_and_bound_search,
    build_torsion_kinematics,
    compose_channel_error_bounds,
    compose_channel_lower_bounds,
    compute_raw_lj_lipschitz,
    detect_rotatable_bonds,
    search_conformers,
    search_conformers_sequential_scan,
)
from dq_dock_engine.docking.chemistry_annotations import _infer_bond_adjacency
from dq_dock_engine.docking_config import DockingConfig, DockingMode


# ---------------------------------------------------------------------------
# CS1: compose_channel_error_bounds
# ---------------------------------------------------------------------------


class TestCS1ChannelErrorComposition:
    def test_additive(self):
        assert compose_channel_error_bounds(0.3, 0.5) == pytest.approx(0.8)

    def test_zero(self):
        assert compose_channel_error_bounds(0.0, 0.0) == 0.0

    def test_identity(self):
        assert compose_channel_error_bounds(1.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CS4: compose_channel_lower_bounds
# ---------------------------------------------------------------------------


class TestCS4LowerBoundComposition:
    def test_additive(self):
        assert compose_channel_lower_bounds(-10.0, -5.0) == pytest.approx(-15.0)

    def test_zero(self):
        assert compose_channel_lower_bounds(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# CS2: ScoreUpperBound (conformer_dominated)
# ---------------------------------------------------------------------------


class TestCS2ScoreUpperBound:
    def test_dominated_when_ub_below_witness(self):
        bound = ScoreUpperBound(
            center_score=3.0, lipschitz_constant=1.0, cell_radius=0.5
        )
        assert bound.bound_value() == pytest.approx(3.5)
        assert bound.is_dominated_by(4.0) is True

    def test_not_dominated_when_ub_above_witness(self):
        bound = ScoreUpperBound(
            center_score=3.0, lipschitz_constant=1.0, cell_radius=2.0
        )
        assert bound.bound_value() == pytest.approx(5.0)
        assert bound.is_dominated_by(4.0) is False


# ---------------------------------------------------------------------------
# CS5/CS8/CS9: EnergyLowerBound (energy_conformer_dominated + Lipschitz)
# ---------------------------------------------------------------------------


class TestCS5EnergyLowerBound:
    def test_bound_value_cs9(self):
        """CS9: energy(p0) - L * r <= energy(p) for all p in ball."""
        bound = EnergyLowerBound(
            center_energy=-5.0, lipschitz_constant=10.0, cell_radius=0.3
        )
        assert bound.bound_value() == pytest.approx(-8.0)

    def test_dominated_cs5(self):
        """CS5: if best_known < lb, cell is dominated."""
        bound = EnergyLowerBound(
            center_energy=-2.0, lipschitz_constant=1.0, cell_radius=0.5
        )
        assert bound.bound_value() == pytest.approx(-2.5)
        assert bound.is_dominated_by(-3.0) is True

    def test_not_dominated(self):
        bound = EnergyLowerBound(
            center_energy=-10.0, lipschitz_constant=1.0, cell_radius=0.5
        )
        assert bound.bound_value() == pytest.approx(-10.5)
        assert bound.is_dominated_by(-3.0) is False


# ---------------------------------------------------------------------------
# CS7: RigidBodyKinematics (isometric_kinematics_preserves_lipschitz)
# ---------------------------------------------------------------------------


class TestCS7RigidBodyKinematics:
    def test_lipschitz_is_one(self):
        kine = RigidBodyKinematics()
        assert kine.lipschitz_constant == 1.0

    def test_n_params(self):
        assert RigidBodyKinematics().n_params == 7


# ---------------------------------------------------------------------------
# CS6: TorsionKinematics (lipschitz_score_composition)
# ---------------------------------------------------------------------------


class TestCS6TorsionKinematics:
    def test_lipschitz_is_max_arm_length(self):
        bonds = (
            RotatableBond(0, 1, (2, 3), max_arm_length=3.0),
            RotatableBond(1, 2, (3,), max_arm_length=5.0),
        )
        coords = np.zeros((4, 3), dtype=np.float32)
        coords[1] = [1.0, 0.0, 0.0]
        coords[2] = [2.0, 0.0, 0.0]
        coords[3] = [3.0, 0.0, 0.0]
        kine = build_torsion_kinematics(bonds, coords, n_atoms=4)
        assert kine.lipschitz_constant == pytest.approx(5.0)

    def test_no_bonds_lipschitz_is_one(self):
        kine = build_torsion_kinematics((), np.zeros((4, 3)), n_atoms=4)
        assert kine.lipschitz_constant == 1.0


# ---------------------------------------------------------------------------
# TorsionCell
# ---------------------------------------------------------------------------


class TestTorsionCell:
    def test_center(self):
        cell = TorsionCell(
            lower=jnp.array([-1.0, -2.0]),
            upper=jnp.array([1.0, 2.0]),
        )
        center = cell.center()
        np.testing.assert_allclose(center, [0.0, 0.0], atol=1e-6)

    def test_radius(self):
        cell = TorsionCell(
            lower=jnp.array([-1.0, -1.0]),
            upper=jnp.array([1.0, 1.0]),
        )
        expected = math.sqrt(2.0)
        assert cell.radius() == pytest.approx(expected, rel=1e-5)

    def test_subdivide_bisects_longest(self):
        cell = TorsionCell(
            lower=jnp.array([-1.0, -3.0]),
            upper=jnp.array([1.0, 3.0]),
        )
        children = cell.subdivide()
        assert len(children) == 2
        # Should bisect dim 1 (width 6 > width 2)
        c0, c1 = children
        assert isinstance(c0, TorsionCell)
        assert isinstance(c1, TorsionCell)
        assert float(c0.upper[1]) == pytest.approx(0.0)
        assert float(c1.lower[1]) == pytest.approx(0.0)

    def test_subdivide_weighted_argmax_is_used_when_weights_provided(self):
        cell = TorsionCell(
            lower=jnp.array([-1.0, -1.0]),
            upper=jnp.array([1.0, 1.0]),
        )
        children = cell.subdivide(jnp.array([1.0, 5.0], dtype=jnp.float32))
        c0, c1 = children
        assert isinstance(c0, TorsionCell)
        assert isinstance(c1, TorsionCell)
        assert float(c0.upper[1]) == pytest.approx(0.0)
        assert float(c1.lower[1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rotatable bond detection
# ---------------------------------------------------------------------------


class TestRotatableBondDetection:
    def _linear_chain(self, n: int):
        """Build a linear chain of n carbon atoms spaced 1.5 A apart."""
        coords = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            coords[i, 0] = i * 1.5
        elements = tuple("C" for _ in range(n))
        adjacency = _infer_bond_adjacency(coords, elements)
        return coords, elements, adjacency

    def test_chain_of_four_has_one_rotatable(self):
        coords, elements, adjacency = self._linear_chain(4)
        bonds = detect_rotatable_bonds(adjacency, coords, elements)
        # C0-C1-C2-C3: bond C1-C2 is the only rotatable one
        # (C0-C1 and C2-C3 are terminal)
        assert len(bonds) == 1
        assert bonds[0].atom_i == 1
        assert bonds[0].atom_j == 2

    def test_chain_of_three_has_no_rotatable(self):
        coords, elements, adjacency = self._linear_chain(3)
        bonds = detect_rotatable_bonds(adjacency, coords, elements)
        assert len(bonds) == 0

    def test_chain_of_five_has_two_rotatable(self):
        coords, elements, adjacency = self._linear_chain(5)
        bonds = detect_rotatable_bonds(adjacency, coords, elements)
        assert len(bonds) == 2


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    def test_zero_angle_preserves_coords(self):
        coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        bonds = (RotatableBond(1, 2, (2, 3), max_arm_length=1.5),)
        kine = build_torsion_kinematics(bonds, np.asarray(coords), n_atoms=4)
        result = kine.forward(coords, jnp.array([0.0]))
        np.testing.assert_allclose(result, coords, atol=1e-5)

    def test_rotation_preserves_bond_lengths(self):
        coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        bonds = (RotatableBond(1, 2, (2, 3), max_arm_length=1.5),)
        kine = build_torsion_kinematics(bonds, np.asarray(coords), n_atoms=4)
        result = kine.forward(coords, jnp.array([jnp.pi / 4]))
        # Bond lengths should be preserved
        for i in range(3):
            original_dist = float(jnp.linalg.norm(coords[i + 1] - coords[i]))
            new_dist = float(jnp.linalg.norm(result[i + 1] - result[i]))
            assert new_dist == pytest.approx(original_dist, abs=1e-4)

    def test_pi_rotation_flips_atom(self):
        """Rotating atom at (3, 0, 0) by pi around x-axis at (1.5, 0, 0)
        should flip the y-component but keep x the same."""
        coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 1.0, 0.0]],
            dtype=jnp.float32,
        )
        bonds = (RotatableBond(0, 1, (2,), max_arm_length=1.8),)
        kine = build_torsion_kinematics(bonds, np.asarray(coords), n_atoms=3)
        result = kine.forward(coords, jnp.array([jnp.pi]))
        # Atom 2 should be rotated: y flipped, z flipped
        np.testing.assert_allclose(result[0], coords[0], atol=1e-5)
        np.testing.assert_allclose(result[1], coords[1], atol=1e-5)
        assert float(result[2, 1]) == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Branch-and-bound search
# ---------------------------------------------------------------------------


class TestBranchAndBound:
    def test_1d_quadratic(self):
        """Search over a 1D quadratic energy landscape: E(theta) = theta^2.
        Minimum at theta=0."""

        def score_fn(coords: jnp.ndarray) -> float:
            return float(jnp.sum(coords**2))

        base_coords = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
        bonds = (RotatableBond(0, 0, (0,), max_arm_length=1.0),)
        kine = build_torsion_kinematics(bonds, np.array([[1.0, 0.0, 0.0]]), n_atoms=1)

        initial_cell = TorsionCell(
            lower=jnp.array([-jnp.pi]),
            upper=jnp.array([jnp.pi]),
        )

        config = BranchAndBoundConfig(
            max_cells=500,
            min_cell_radius=0.1,
            score_lipschitz_constant=3.0,
            max_conformers=5,
        )

        result = branch_and_bound_search(
            kinematics=kine,
            base_coords=base_coords,
            score_fn=score_fn,
            initial_cell=initial_cell,
            config=config,
        )
        assert len(result.conformer_coords) >= 1
        assert all(isinstance(e, float) for e in result.conformer_energies)

    def test_pruning_incumbent_can_skip_dominated_root_cell(self):
        def score_fn(coords: jnp.ndarray) -> float:
            return float(jnp.sum(coords**2))

        base_coords = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
        bonds = (RotatableBond(0, 0, (0,), max_arm_length=1.0),)
        kine = build_torsion_kinematics(bonds, np.array([[1.0, 0.0, 0.0]]), n_atoms=1)

        initial_cell = TorsionCell(
            lower=jnp.array([-jnp.pi]),
            upper=jnp.array([jnp.pi]),
        )

        config = BranchAndBoundConfig(
            max_cells=500,
            min_cell_radius=0.1,
            score_lipschitz_constant=3.0,
            max_conformers=5,
        )

        result = branch_and_bound_search(
            kinematics=kine,
            base_coords=base_coords,
            score_fn=score_fn,
            initial_cell=initial_cell,
            config=config,
            pruning_incumbent_energy=-100.0,
        )
        assert len(result.conformer_coords) == 1
        assert result.conformer_energies == (-100.0,)

    def test_local_activity_mask_can_collapse_cell_dimensions(self):
        def score_fn(coords: jnp.ndarray) -> float:
            return float(jnp.sum(coords**2))

        base_coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        bonds = (
            RotatableBond(0, 1, (1, 2), max_arm_length=1.0),
            RotatableBond(1, 2, (2,), max_arm_length=1.0),
        )
        kine = build_torsion_kinematics(bonds, np.asarray(base_coords), n_atoms=3)
        initial_cell = TorsionCell(
            lower=jnp.array([-jnp.pi, -jnp.pi], dtype=jnp.float32),
            upper=jnp.array([jnp.pi, jnp.pi], dtype=jnp.float32),
        )
        config = BranchAndBoundConfig(
            max_cells=32,
            min_cell_radius=0.2,
            score_lipschitz_constant=3.0,
            max_conformers=5,
            per_bond_lipschitz=(3.0, 3.0),
        )

        result = branch_and_bound_search(
            kinematics=kine,
            base_coords=base_coords,
            score_fn=score_fn,
            initial_cell=initial_cell,
            config=config,
            local_activity_mask_fn=lambda cell, center: np.array(
                [True, False], dtype=bool
            ),
        )

        assert len(result.conformer_coords) >= 1

    def test_rigid_ligand_returns_single_conformer(self):
        """A ligand with no rotatable bonds returns its input as the only conformer."""
        coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        elements = ("C", "C")
        adjacency = _infer_bond_adjacency(np.asarray(coords), elements)

        result = search_conformers(
            base_coords=coords,
            adjacency=adjacency,
            elements=elements,
            score_fn=lambda c: float(jnp.sum(c**2)),
        )
        assert len(result.conformer_coords) == 1
        np.testing.assert_allclose(result.conformer_coords[0], coords, atol=1e-5)

    def test_reuse_initial_conformer_flag_is_explicit(self):
        assert (
            BranchAndBoundConfig(
                max_cells=3,
                min_cell_radius=0.1,
                score_lipschitz_constant=1.0,
                max_conformers=1,
            ).reuse_initial_conformer
            is False
        )
        assert (
            BranchAndBoundConfig(
                max_cells=3,
                min_cell_radius=0.1,
                score_lipschitz_constant=1.0,
                max_conformers=1,
                reuse_initial_conformer=True,
            ).reuse_initial_conformer
            is True
        )


# ---------------------------------------------------------------------------
# Integration: search_conformers
# ---------------------------------------------------------------------------


class TestSearchConformers:
    def test_chain_requires_explicit_theorem_derived_config(self):
        n = 5
        coords_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            coords_np[i, 0] = i * 1.5
        elements = tuple("C" for _ in range(n))
        adjacency = _infer_bond_adjacency(coords_np, elements)
        coords = jnp.array(coords_np, dtype=jnp.float32)

        with pytest.raises(ValueError, match="explicit theorem-derived"):
            search_conformers(
                base_coords=coords,
                adjacency=adjacency,
                elements=elements,
                score_fn=lambda c: float(jnp.sum(c**2)),
            )

    def test_chain_produces_conformers(self):
        """A 5-atom linear chain has 2 rotatable bonds → should produce conformers."""
        n = 5
        coords_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            coords_np[i, 0] = i * 1.5
        elements = tuple("C" for _ in range(n))
        adjacency = _infer_bond_adjacency(coords_np, elements)
        coords = jnp.array(coords_np, dtype=jnp.float32)

        result = search_conformers(
            base_coords=coords,
            adjacency=adjacency,
            elements=elements,
            score_fn=lambda c: float(jnp.sum(c**2)),
            config=BranchAndBoundConfig(
                max_cells=200,
                min_cell_radius=0.3,
                score_lipschitz_constant=3.0,
                max_conformers=5,
            ),
        )
        assert len(result.conformer_coords) >= 1
        for c in result.conformer_coords:
            assert c.shape == (n, 3)

    def test_chain_sequential_scan_produces_conformer(self):
        n = 5
        coords_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            coords_np[i, 0] = i * 1.5
        elements = tuple("C" for _ in range(n))
        adjacency = _infer_bond_adjacency(coords_np, elements)
        coords = jnp.array(coords_np, dtype=jnp.float32)

        result = search_conformers_sequential_scan(
            base_coords=coords,
            adjacency=adjacency,
            elements=elements,
            score_fn=lambda c: float(jnp.sum(c**2)),
        )
        assert len(result.conformer_coords) == 1
        assert result.conformer_coords[0].shape == (n, 3)


def test_compute_raw_lj_lipschitz_requires_physical_inputs() -> None:
    with pytest.raises(ValueError, match="min_pairwise_sigma"):
        compute_raw_lj_lipschitz(0.1, 0.0)

    with pytest.raises(ValueError, match="epsilon_lj"):
        compute_raw_lj_lipschitz(0.0, 3.5)


def test_docking_config_defaults_to_blind_conformer_search_seed() -> None:
    assert DockingConfig(mode=DockingMode.HEURISTIC).reuse_initial_conformer is False
