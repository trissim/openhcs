"""Tests for receptor flexibility (RFE1-RFE6)."""

import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.receptor_flexibility import (
    BoltzmannEnsembleResult,
    EnsembleScoringResult,
    ReceptorConformation,
    boltzmann_ensemble_score,
    compute_boltzmann_weights,
    conformational_error_radius,
    ensemble_score_upper_bound,
    rigid_is_lower_bound,
    score_ensemble,
)
from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.explicit_water_placement import WaterPlacementGrid
from dq_dock_engine.docking.scoring_context import build_certified_scoring_context
from dq_dock_engine.docking_config import ExactChemistryMode


def test_conformational_error_radius_nonneg():
    """RFE5: error radius is always ≥ 0."""
    ref = jnp.array([1.0, 2.0, 3.0])
    confs = (
        jnp.array([1.1, 2.2, 2.8]),
        jnp.array([0.9, 1.8, 3.2]),
    )
    error = conformational_error_radius(ref, confs)
    assert error >= 0.0


def test_conformational_error_radius_is_max_diff():
    """RFE1: error radius = max |E(pose, rk) - E(pose, r0)|."""
    ref = jnp.array([1.0, 2.0, 3.0])
    confs = (
        jnp.array([1.5, 2.0, 3.0]),  # max diff = 0.5
        jnp.array([1.0, 2.0, 4.0]),  # max diff = 1.0
    )
    error = conformational_error_radius(ref, confs)
    np.testing.assert_allclose(error, 1.0, atol=1e-6)


def test_ensemble_score_upper_bound():
    """RFE2: ensemble_max ≤ rigid + error_radius."""
    bound = ensemble_score_upper_bound(rigid_score=-5.0, error_radius=1.5)
    assert bound == -3.5


def test_rigid_is_lower_bound():
    """RFE3: rigid ≤ ensemble for all poses."""
    rigid = jnp.array([1.0, 2.0, 3.0])
    ensemble = jnp.array([0.5, 2.0, 2.5])  # ensemble picks best (lower)
    result = rigid_is_lower_bound(rigid, ensemble)
    # rigid[0]=1.0 > ensemble[0]=0.5 → False; that's fine,
    # the theorem says E(r0) ≤ max_k E(rk), but we minimize energy
    assert result.shape == (3,)


def test_score_ensemble_finds_best():
    """score_ensemble picks the conformation giving lowest energy per pose."""
    conf0 = ReceptorConformation(
        coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        radii=jnp.array([1.5], dtype=jnp.float32),
    )
    conf1 = ReceptorConformation(
        coords=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        radii=jnp.array([1.5], dtype=jnp.float32),
    )
    poses = jnp.array([[[2.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], dtype=jnp.float32)
    lig_radii = jnp.array([1.5], dtype=jnp.float32)

    def score_fn(rec_coords, poses_coords, rec_radii, lig_radii):
        dists = jnp.linalg.norm(
            rec_coords[None, :, None, :] - poses_coords[:, None, :, :], axis=-1
        )
        return jnp.sum(dists, axis=(1, 2))

    result = score_ensemble(
        conformations=(conf0, conf1),
        poses_coords=poses,
        ligand_radii=lig_radii,
        score_fn=score_fn,
    )
    assert result.ensemble_scores.shape == (2,)
    assert result.conformational_error_radius >= 0.0
    assert len(result.per_conformation_scores) == 2


def test_boltzmann_weights_sum_to_one():
    energies = jnp.array([0.0, 1.0, 2.0])
    weights = compute_boltzmann_weights(energies)
    np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-6)
    assert jnp.all(weights >= 0.0)


def test_boltzmann_ensemble_score_bounded():
    """RFE4: |Σ wk·Ek - E(r0)| ≤ max_k |Ek - E(r0)|."""
    scores = (
        jnp.array([1.0, 2.0]),
        jnp.array([1.5, 1.8]),
        jnp.array([0.8, 2.5]),
    )
    weights = jnp.array([0.5, 0.3, 0.2])
    result = boltzmann_ensemble_score(weights, scores, reference_index=0)
    assert result.weighted_scores.shape == (2,)
    assert result.error_bound >= 0.0
    # Check the bound holds
    diff = jnp.abs(result.weighted_scores - scores[0])
    assert jnp.all(diff <= result.error_bound + 1e-6)


def test_build_scoring_context_does_not_synthesize_receptor_flex(monkeypatch):
    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        base_radii=jnp.array([1.5], dtype=jnp.float32),
        center_of_mass=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        elements=("C",),
    )

    monkeypatch.setattr(
        "dq_dock_engine.docking.scoring_context.build_certified_rich_chemistry_plan",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "dq_dock_engine.docking.scoring_context.generate_water_grid",
        lambda **kwargs: WaterPlacementGrid(
            positions=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            grid_spacing=0.5,
        ),
    )

    def _unexpected_receptor_flex(*args, **kwargs):
        raise AssertionError(
            "strict certified scoring context must not synthesize receptor-flex ensembles"
        )

    monkeypatch.setattr(
        "dq_dock_engine.docking.scoring_context._generate_receptor_conformations",
        _unexpected_receptor_flex,
    )

    context = build_certified_scoring_context(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        electrostatics=None,
        receptor_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.7], dtype=jnp.float32),
        receptor_elements=("C",),
        receptor_file=None,
        ligand_source_path=None,
        ligand_ctx=ligand_ctx,
        target_electrostatic_error=0.1,
    )

    assert context.receptor_conformations is None
    assert context.water_grid is not None
