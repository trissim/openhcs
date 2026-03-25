import jax
import jax.numpy as jnp
import pytest

from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    CertifiedPriorSpec,
)
from dq_dock_engine.docking.formal_optimizer import (
    CertifiedOptimizerState,
    refine_poses_certified,
)
from dq_dock_engine.docking.formal_actions import (
    compute_adaptive_translation_step,
    create_roundwise_certified_action_family,
    least_adequate_dyadic_round,
    least_positive_joint_adequate_dyadic_round,
    least_positive_adequate_dyadic_round,
)


def _fake_state(
    selected_action: int, ambiguity_mask: jax.Array
) -> CertifiedOptimizerState:
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
    assert {"SH1", "SH6", "SH10", "SH11", "SH12", "SH13"}.issubset(
        set(family.theorem_handles)
    )


def test_least_adequate_dyadic_round_matches_first_adequate_halving_level():
    assert least_adequate_dyadic_round(1.0, 0.25) == 2
    assert least_adequate_dyadic_round(1.0, 1.0) == 0
    assert least_adequate_dyadic_round(0.3, 0.05) == 3


def test_least_positive_adequate_dyadic_round_enforces_nonzero_round():
    assert least_positive_adequate_dyadic_round(1.0, 1.0) == 1
    assert least_positive_adequate_dyadic_round(1.0, 0.25) == 2


def test_least_positive_joint_adequate_dyadic_round_tracks_worst_channel():
    assert least_positive_joint_adequate_dyadic_round(1.0, 4.0, 0.5) == 3
    assert least_positive_joint_adequate_dyadic_round(0.25, 0.125, 0.25) == 1


def test_compute_adaptive_translation_step_uses_exact_lipschitz_ratio_when_valid():
    base_step = 0.25
    epsilon_lj = 0.1
    sigma = 3.5
    r_soft = sigma

    adapted = compute_adaptive_translation_step(base_step, epsilon_lj, sigma, r_soft)

    ratio = sigma / r_soft
    l_soft = abs(24.0 * epsilon_lj / r_soft * (2.0 * ratio**12 - ratio**6))
    l_raw = 762.0 * epsilon_lj / sigma
    assert adapted == pytest.approx(base_step * (l_raw / l_soft))


def test_compute_adaptive_translation_step_returns_base_step_when_preconditions_fail():
    assert compute_adaptive_translation_step(0.25, 0.1, 3.5, 10.0) == pytest.approx(
        0.25
    )


def test_refine_poses_certified_escalates_then_resets_support_expansion(monkeypatch):
    recorded_levels: list[int] = []

    def fake_refine_round(
        *, coords_batch, round_index, support_expansion_level, **kwargs
    ):
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
