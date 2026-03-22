import jax
import jax.numpy as jnp

from dq_dock_engine.docking.formal_belief import CertifiedBeliefState, CertifiedPriorSpec
from dq_dock_engine.docking.formal_optimizer import (
    CertifiedOptimizerState,
    refine_poses_certified,
)
from dq_dock_engine.docking.formal_actions import create_roundwise_certified_action_family


def _fake_state(selected_action: int, ambiguity_mask: jax.Array) -> CertifiedOptimizerState:
    prior = jnp.array([0.5, 0.5], dtype=jnp.float32)
    belief = CertifiedBeliefState(
        prior_spec=CertifiedPriorSpec(kind="uniform"),
        prior=prior,
        posterior=prior,
        posterior_rule="SURVIVOR_CONDITIONING",
        posterior_theorem="FLO9",
        coarse_scores=jnp.array([0.0, 1.0], dtype=jnp.float32),
        exact_scores=jnp.array([0.0, 1.0], dtype=jnp.float32),
        exact_error_bound=0.0,
        survivor_mask=jnp.array([True, True]),
        ambiguity_mask=ambiguity_mask,
        selected_action=selected_action,
        selected_action_rule="first_ambiguity_member",
        selected_action_theorem="FLO8",
        step_index=0,
    )
    return CertifiedOptimizerState(
        coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        action_family=None,
        belief=belief,
        retained_receptor_indices=jnp.array([0], dtype=jnp.int32),
        pruning_certificate=None,
    )


def test_roundwise_action_family_carries_forward_coarser_shells():
    family = create_roundwise_certified_action_family(
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
        round_index=2,
        support_expansion_level=2,
    )

    assert family.support_shell_levels == (2, 1, 0)
    assert family.stencil_level == 2
    assert family.translation_step == 0.5
    assert len(family.actions) == 37
    assert family.actions[0].is_noop is True
    assert tuple(action.action_id for action in family.actions) == tuple(range(37))


def test_refine_poses_certified_escalates_then_resets_support_expansion(monkeypatch):
    recorded_levels: list[int] = []

    def fake_refine_round(*, coords_batch, round_index, support_expansion_level, **kwargs):
        recorded_levels.append(support_expansion_level)
        if round_index == 0:
            return coords_batch, (_fake_state(1, jnp.array([True, True])),)
        if round_index == 1:
            return coords_batch, (_fake_state(1, jnp.array([True, False])),)
        return coords_batch, (_fake_state(0, jnp.array([True, False])),)

    monkeypatch.setattr(
        "dq_dock_engine.docking.formal_optimizer._refine_round",
        fake_refine_round,
    )

    refine_poses_certified(
        coords_batch=jnp.array([[[0.0, 0.0, 0.0]]], dtype=jnp.float32),
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        n_rounds=3,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
    )

    assert recorded_levels == [0, 1, 0]