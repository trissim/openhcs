import pytest
import jax.numpy as jnp
import numpy as np

import dq_dock_engine.docking.scoring as scoring

from dq_dock_engine.docking.scoring import (
    CertifiedMetalCoordinationSpec,
    score_certified_metal_coordination_batch,
)

def test_metal_coordination_scoring() -> None:
    receptor_coords = jnp.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    ])
    poses_coords = jnp.array([
        [  # Pose 0
            [2.1, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        [  # Pose 1
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
    ])
    spec = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array([1.0, 0.9]),
        ligand_strengths=jnp.array([1.0, 0.5]),
        receptor_ideal_angles=jnp.array([0.0, 0.0]),
        ideal_distance=2.1,
        distance_width=0.3,
        cutoff=4.0,
    )

    res = score_certified_metal_coordination_batch(receptor_coords, poses_coords, spec)

    assert res.scores.shape == (2,)
    assert res.error_bound >= 0.0

    # Exact calculation for pose 0:
    # rec0(0,0,0) -> lig0(2.1,0,0) = 2.1. exp(0) = 1. strength = 1.0 * 1.0 = 1.0.
    # rec0(0,0,0) -> lig1(3.0,0,0) = 3.0. exp(-9) ~= 0. strength = 1.0 * 0.5 = 0.5.
    # Total exact pose 0 should be approx -1.0 (-0.00006...)
    assert -1.01 < res.scores[0] < -0.99

    # exact pose 1
    assert float(res.scores[1]) == 0.0


def test_inactive_metal_coordination_short_circuits_without_kernel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receptor_coords = jnp.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    poses_coords = jnp.array(
        [
            [[2.1, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    spec = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array([0.0, 0.0]),
        ligand_strengths=jnp.array([1.0, 0.5]),
        receptor_ideal_angles=jnp.array([0.0, 0.0]),
        ideal_distance=2.1,
        distance_width=0.3,
        cutoff=9.0,
    )

    assert not bool(spec.is_active)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("inactive metal spec should short-circuit")

    monkeypatch.setattr(scoring, "_score_directional_metal_exact_batch", _should_not_run)
    monkeypatch.setattr(scoring, "_score_directional_metal_cutoff_batch", _should_not_run)

    res = score_certified_metal_coordination_batch(receptor_coords, poses_coords, spec)

    np.testing.assert_allclose(np.asarray(res.scores), np.zeros(2))
    assert float(res.error_bound) == 0.0
    assert float(res.target_error) == 0.0
    assert float(res.cutoff_radius) == 0.0

if __name__ == "__main__":
    test_metal_coordination_scoring()
    print("Test passed!")
