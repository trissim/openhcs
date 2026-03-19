import jax
import jax.numpy as jnp

from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine
from dq_dock_engine.docking.formal_actions import create_certified_action_family
from dq_dock_engine.docking.formal_belief import (
    CertifiedPriorSpec,
    build_prior,
    update_posterior,
)
from dq_dock_engine.docking.formal_optimizer import refine_poses_certified
from dq_dock_engine.docking.formal_sampling import sample_certified_global_poses
from dq_dock_engine.docking.pipeline import run_docking_pipeline
from dq_dock_engine.docking_config import CERTIFIED_DOCKING


def test_certified_action_family_has_noop_first_and_stable_size():
    family = create_certified_action_family(
        translation_step=0.5,
        rotation_step_rad=float(jnp.pi / 12.0),
        stencil_level=0,
    )

    assert len(family.actions) == 13
    assert family.actions[0].is_noop is True
    assert family.actions[0].action_id == 0
    assert tuple(action.action_id for action in family.actions) == tuple(range(13))


def test_bayes_update_normalizes_survivor_support():
    prior = jnp.array([0.25, 0.25, 0.25, 0.25])
    survivor_mask = jnp.array([True, False, True, False])

    posterior = update_posterior(prior, survivor_mask)

    assert jnp.allclose(posterior, jnp.array([0.5, 0.0, 0.5, 0.0]))
    assert jnp.isclose(jnp.sum(posterior), 1.0)


def test_noop_biased_prior_is_explicit_and_normalized():
    prior = build_prior(CertifiedPriorSpec(kind="noop_biased", noop_mass=0.4), 5)

    assert jnp.isclose(jnp.sum(prior), 1.0)
    assert jnp.isclose(prior[0], 0.4)
    assert jnp.allclose(prior[1:], jnp.full((4,), 0.15))


def test_refine_poses_certified_uses_finite_action_family_search():
    receptor_coords = jnp.array([[3.0, 0.0, 0.0]])
    receptor_radii = jnp.array([1.0])
    ligand_radii = jnp.array([1.0])
    initial_coords = jnp.array([[[6.0, 0.0, 0.0]]])

    refined_coords, history = refine_poses_certified(
        coords_batch=initial_coords,
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        n_rounds=1,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
    )

    assert len(history) == 1
    belief = history[0][0].belief
    assert belief.selected_action != 0
    assert refined_coords.shape == initial_coords.shape
    assert float(refined_coords[0, 0, 0]) < float(initial_coords[0, 0, 0])


def test_certified_global_sampler_is_deterministic_and_nonrandom():
    box = DockingBox(center=jnp.array([6.0, 0.0, 0.0]), size=jnp.array([2.0, 2.0, 2.0]))
    pose_vec = sample_certified_global_poses(box, 8)

    assert pose_vec.translation.shape == (8, 3)
    assert pose_vec.quaternion.shape == (8, 4)
    assert not jnp.any(jnp.all(jnp.isclose(pose_vec.translation, box.center), axis=1))


def test_certified_pipeline_does_not_call_heuristic_sampler(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("heuristic sampler should not be used in certified mode")

    monkeypatch.setattr(
        "dq_dock_engine.docking.pocket_sampling.sample_intelligent_poses",
        fail_if_called,
    )

    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]]),
        base_radii=jnp.array([1.0]),
        center_of_mass=jnp.array([0.0, 0.0, 0.0]),
    )
    box = DockingBox(center=jnp.array([6.0, 0.0, 0.0]), size=jnp.array([0.2, 0.2, 0.2]))

    best_poses, _ = run_docking_pipeline(
        protein_coords=jnp.array([[3.0, 0.0, 0.0]]),
        receptor_radii=jnp.array([1.0]),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=1,
        engine=ScoringEngine.INTERNAL_LJ,
        key=jax.random.PRNGKey(0),
        config=CERTIFIED_DOCKING,
        top_k=1,
        optimize=False,
        use_pocket_guided=True,
    )

    assert len(best_poses) == 1
