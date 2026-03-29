from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import dq_dock_engine.docking.formal_optimizer as formal_optimizer_module

from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    CertifiedPriorSpec,
)
from dq_dock_engine.docking.formal_optimizer import (
    CertifiedOptimizerState,
    refine_poses_certified,
)


def _dummy_state(*, ambiguous: bool, step_index: int) -> CertifiedOptimizerState:
    ambiguity_mask = (
        jnp.array([True, True], dtype=bool)
        if ambiguous
        else jnp.array([True, False], dtype=bool)
    )
    belief = CertifiedBeliefState(
        prior_spec=CertifiedPriorSpec(kind="uniform"),
        prior=jnp.array([0.5, 0.5], dtype=jnp.float32),
        posterior=jnp.array([0.5, 0.5], dtype=jnp.float32),
        posterior_rule="survivor_conditioning",
        posterior_theorem="FLO9",
        coarse_scores=jnp.array([0.0, 1.0], dtype=jnp.float32),
        exact_scores=jnp.array([0.0, 1.0], dtype=jnp.float32),
        exact_error_bound=0.0,
        survivor_mask=jnp.array([True, False], dtype=bool),
        ambiguity_mask=ambiguity_mask,
        selected_action=0,
        selected_action_rule="ambiguity_band",
        selected_action_theorem="FLO8",
        step_index=step_index,
    )
    return CertifiedOptimizerState(
        coords=jnp.zeros((1, 3), dtype=jnp.float32),
        action_family=SimpleNamespace(stencil_level=step_index),
        belief=belief,
        retained_receptor_indices=jnp.array([0], dtype=jnp.int32),
        pruning_certificate=SimpleNamespace(),
    )


def test_refine_poses_certified_tracks_support_expansion_per_pose(monkeypatch) -> None:
    call_log: list[tuple[int, int, tuple[float, ...]]] = []

    def fake_refine_round(
        *,
        coords_batch,
        round_index,
        support_expansion_level,
        **kwargs,
    ):
        del kwargs
        pose_tags = tuple(
            float(coords_batch[i, 0, 0]) for i in range(coords_batch.shape[0])
        )
        call_log.append((int(round_index), int(support_expansion_level), pose_tags))
        states = []
        for pose_tag in pose_tags:
            ambiguous = round_index == 0 and pose_tag == 0.0
            states.append(_dummy_state(ambiguous=ambiguous, step_index=round_index))
        return coords_batch, tuple(states)

    monkeypatch.setattr(
        "dq_dock_engine.docking.formal_optimizer._refine_round",
        fake_refine_round,
    )

    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    refine_poses_certified(
        coords_batch=coords_batch,
        receptor_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_radii=jnp.ones((1,), dtype=jnp.float32),
        n_rounds=2,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
    )

    assert sorted(call_log) == [
        (0, 0, (0.0,)),
        (0, 0, (1.0,)),
        (1, 0, (1.0,)),
        (1, 1, (0.0,)),
    ]


def test_refine_round_masks_actions_outside_binding_site(monkeypatch) -> None:
    def fake_score_exact_and_coarse_round(**kwargs):
        del kwargs
        exact = jnp.array([[0.0, -10.0]], dtype=jnp.float32)
        coarse = jnp.array([[0.0, -10.0]], dtype=jnp.float32)
        return exact, coarse, 0.0, 0.0, None

    monkeypatch.setattr(
        formal_optimizer_module,
        "score_exact_and_coarse_round",
        fake_score_exact_and_coarse_round,
    )

    result = formal_optimizer_module._refine_round_jit_core(
        coords_batch=jnp.array([[[0.0, 0.0, 0.0]]], dtype=jnp.float32),
        receptor_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_radii=jnp.ones((1,), dtype=jnp.float32),
        target_error=0.001,
        coarse_target_error=0.001,
        translation_step=0.5,
        translation_deltas=jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        quaternion_deltas=jnp.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        prior_spec=CertifiedPriorSpec(kind="uniform"),
        retained_indices=jnp.array([0], dtype=jnp.int32),
        binding_site_center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        binding_site_radius=0.5,
        use_softened_coarse=False,
    )

    exact_scores = result["exact_scores_matrix"]
    coarse_scores = result["coarse_scores_matrix"]
    assert float(exact_scores[0, 0]) == 0.0
    assert float(coarse_scores[0, 0]) == 0.0
    assert float(exact_scores[0, 1]) > 1e20
    assert float(coarse_scores[0, 1]) > 1e20
