from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from dq_dock_engine.docking import pipeline as docking_pipeline
from dq_dock_engine.docking.core import DockingBox, LigandContext, PoseVector
from dq_dock_engine.docking.formal_pruning import certified_pruning_certificate
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    DockingConfig,
    DockingMode,
    ExactChemistryMode,
    OptimizerBackend,
)


def _make_request(*, target_error: float = 0.001):
    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        base_radii=jnp.array([1.7], dtype=jnp.float32),
        center_of_mass=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        elements=("C",),
    )
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([8.0, 8.0, 8.0], dtype=jnp.float32),
    )
    return docking_pipeline.PipelineDockingRequest(
        protein_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.7], dtype=jnp.float32),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses_override=3,
        key=jax.random.PRNGKey(0),
        config=DockingConfig(
            mode=DockingMode.CERTIFIED,
            optimizer_backend=OptimizerBackend.FORMAL,
            certified_scoring_family=CertifiedScoringFamily.LJ,
            target_error=target_error,
            exact_chemistry_mode=ExactChemistryMode.NONE,
        ),
    )


def test_certified_pruning_pass_uses_exact_certificate_for_top_k(monkeypatch) -> None:
    coarse_scores = jnp.array([0.0, 10.0, 0.1], dtype=jnp.float32)
    exact_scores = jnp.array([0.0, 0.05, 2.0], dtype=jnp.float32)
    request = _make_request(target_error=0.1)

    monkeypatch.setattr(
        docking_pipeline,
        "score_certified_softened_lj",
        lambda **kwargs: SimpleNamespace(scores=coarse_scores),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_exact_pose_batch",
        lambda request, *, poses_coords, electrostatics: exact_scores,
    )

    survivor_mask, returned_coarse_scores, delta = (
        docking_pipeline._certified_pruning_pass(
            request,
            poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
            electrostatics=None,
        )
    )

    expected = certified_pruning_certificate(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        k=2,
        delta=0.1,
    )
    assert jnp.array_equal(survivor_mask, expected.survivor_mask)
    assert jnp.array_equal(returned_coarse_scores, coarse_scores)
    assert delta == 0.1


def test_certified_pipeline_route_scores_only_valid_survivors(monkeypatch) -> None:
    request = _make_request()
    pose_vecs = PoseVector(
        translation=jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1)
        ),
    )
    batched_coords = jnp.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=jnp.float32
    )

    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics: (
            jnp.array([True, False, False]),
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
            0.001,
        ),
    )

    def fake_score_exact(request, *, poses_coords, electrostatics):
        del request, electrostatics
        assert poses_coords.shape[0] == 1
        return jnp.array([3.14], dtype=jnp.float32)

    monkeypatch.setattr(docking_pipeline, "_score_exact_pose_batch", fake_score_exact)

    initial_scores = docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
        request,
        batched_coords,
        pose_vecs,
    )

    assert jnp.allclose(
        initial_scores.final_scores,
        jnp.array([3.14, 1e6, 1e6], dtype=jnp.float32),
    )
    assert initial_scores.survivor_pose_vecs is not None
    assert initial_scores.survivor_pose_vecs.translation.shape[0] == 1
    assert initial_scores.survivor_exact_scores is not None
    assert jnp.allclose(
        initial_scores.survivor_exact_scores,
        jnp.array([3.14], dtype=jnp.float32),
    )


def test_certified_pipeline_route_optimizes_all_certified_survivors() -> None:
    request = _make_request()
    survivor_pose_vecs = PoseVector(
        translation=jnp.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1)
        ),
    )
    initial_scores = docking_pipeline.PipelineInitialScores(
        final_scores=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        survivor_pose_vecs=survivor_pose_vecs,
        survivor_exact_scores=jnp.array([0.4, 0.1, 0.2], dtype=jnp.float32),
        valid_survivor_mask=jnp.ones((3,), dtype=bool),
    )

    translations, quaternions, n_to_opt = (
        docking_pipeline.CertifiedPipelineRoute().optimization_inputs(
            request,
            pose_vecs=survivor_pose_vecs,
            initial_scores=initial_scores,
        )
    )

    assert n_to_opt == 3
    assert jnp.allclose(
        translations,
        jnp.array(
            [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
    )
    assert quaternions.shape == (3, 4)


def test_score_exact_pose_batch_scores_large_pose_sets_in_chunks(monkeypatch) -> None:
    request = _make_request()
    chunk_sizes: list[int] = []

    class FakeScoringContext:
        def score_exact_batch(self, **kwargs):
            chunk = kwargs["poses_coords"]
            chunk_sizes.append(int(chunk.shape[0]))
            return SimpleNamespace(scores=chunk[:, 0, 0])

    monkeypatch.setattr(
        docking_pipeline,
        "resolve_request_scoring_context",
        lambda request, *, engine=None: FakeScoringContext(),
    )

    poses_coords = jnp.arange(5, dtype=jnp.float32).reshape(5, 1, 1)
    exact_scores = docking_pipeline._score_exact_pose_batch(
        request,
        poses_coords=poses_coords,
        electrostatics=None,
        chunk_size=2,
    )

    assert chunk_sizes == [2, 2, 1]
    assert jnp.allclose(exact_scores, jnp.array([0, 1, 2, 3, 4], dtype=jnp.float32))


def test_certified_pipeline_route_fails_loud_when_survivor_capacity_is_exceeded(
    monkeypatch,
) -> None:
    request = _make_request()
    n_survivors = docking_pipeline.SURVIVOR_BATCH_SIZE + 1
    pose_vecs = PoseVector(
        translation=jnp.zeros((n_survivors, 3), dtype=jnp.float32),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            (n_survivors, 1),
        ),
    )
    batched_coords = jnp.zeros((n_survivors, 1, 3), dtype=jnp.float32)

    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics: (
            jnp.ones((n_survivors,), dtype=bool),
            jnp.zeros((n_survivors,), dtype=jnp.float32),
            0.001,
        ),
    )

    with pytest.raises(ValueError, match="exceeded fixed padded capacity"):
        docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
            request,
            batched_coords,
            pose_vecs,
        )
