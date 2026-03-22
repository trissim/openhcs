import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.formal_handles import (
    contact_surrogate_theorem_handles,
    directional_hbond_finite_theorem_handles,
    rich_chemistry_theorem_handles,
    screened_coulomb_theorem_handles,
)
from dq_dock_engine.docking.scoring import (
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedScreenedCoulombSpec,
    score_certified_contact_batch,
    score_certified_directional_hbond_batch,
    score_certified_lj_screened_coulomb_batch,
    score_certified_rich_chemistry_batch,
    score_certified_screened_coulomb_batch,
)
from dq_dock_engine.docking.rich_chemistry import build_directional_hbond_spec


def _toy_geometry():
    receptor_coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=jnp.float32)
    poses_coords = jnp.array(
        [
            [[2.8, 0.0, 0.0], [3.2, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [4.4, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    receptor_radii = jnp.array([1.7, 1.6], dtype=jnp.float32)
    ligand_radii = jnp.array([1.5, 1.4], dtype=jnp.float32)
    screened = CertifiedScreenedCoulombSpec(
        receptor_charges=jnp.array([0.4, -0.3], dtype=jnp.float32),
        ligand_charges=jnp.array([-0.5, 0.2], dtype=jnp.float32),
        kappa=0.7,
        cutoff=3.5,
        dielectric=4.0,
    )
    contact = CertifiedContactSurrogateSpec(
        receptor_weights=jnp.array([0.8, 0.6], dtype=jnp.float32),
        ligand_weights=jnp.array([0.9, 0.7], dtype=jnp.float32),
        beta=0.55,
        cutoff=4.0,
    )
    hbond = CertifiedDirectionalHBondSpec(
        receptor_directions=jnp.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        ligand_neighbor_indices=jnp.array(
            [[0, 1], [0, 1]], dtype=jnp.int32
        ),
        receptor_strengths=jnp.array([0.8, 0.6], dtype=jnp.float32),
        ligand_strengths=jnp.array([0.9, 0.7], dtype=jnp.float32),
        ideal_distance=2.8,
        distance_width=0.7,
        cutoff=4.0,
    )
    return (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        screened,
        contact,
        hbond,
    )


def test_new_chemistry_handle_helpers_surface_new_theorem_families() -> None:
    assert set(contact_surrogate_theorem_handles()) == {
        "CT1",
        "CT2",
        "CT3",
        "CT4",
        "CT5",
        "CT6",
    }
    assert {"SC1", "SC6"}.issubset(set(screened_coulomb_theorem_handles()))
    assert {"HB1", "HB9", "HB10", "HB11", "HB12"}.issubset(
        set(directional_hbond_finite_theorem_handles())
    )
    assert {"RC1", "RC10"}.issubset(set(rich_chemistry_theorem_handles()))


def test_new_chemistry_scoring_helpers_are_jit_safe() -> None:
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        screened,
        contact,
        hbond,
    ) = _toy_geometry()

    @jax.jit
    def run_all():
        screened_batch = score_certified_screened_coulomb_batch(
            receptor_coords, poses_coords, screened
        )
        contact_batch = score_certified_contact_batch(
            receptor_coords, poses_coords, contact
        )
        hbond_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, hbond
        )
        nonbonded_batch = score_certified_lj_screened_coulomb_batch(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            screened,
            target_error=0.001,
        )
        rich_batch = score_certified_rich_chemistry_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            screened_coulomb=screened,
            contact=contact,
            directional_hbond=hbond,
            target_error=0.001,
        )
        return (
            screened_batch.scores,
            screened_batch.error_bound,
            contact_batch.scores,
            contact_batch.error_bound,
            hbond_batch.scores,
            hbond_batch.error_bound,
            nonbonded_batch.scores,
            nonbonded_batch.error_bound,
            rich_batch.scores,
            rich_batch.error_bound,
        )

    outputs = run_all()
    for value in outputs:
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr))


def test_build_directional_hbond_spec_handles_small_receptor_neighbor_sets() -> None:
    spec = build_directional_hbond_spec(
        receptor_coords=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float32),
        receptor_elements=("O", "N"),
        ligand_coords=np.array([[0.5, 0.1, 0.0], [1.2, -0.1, 0.0]], dtype=np.float32),
        ligand_elements=("N", "O"),
    )

    receptor_directions = np.asarray(spec.receptor_directions)
    ligand_neighbor_indices = np.asarray(spec.ligand_neighbor_indices)

    assert receptor_directions.shape == (2, 3)
    assert ligand_neighbor_indices.shape == (2, 2)
    assert np.all(np.isfinite(receptor_directions))
