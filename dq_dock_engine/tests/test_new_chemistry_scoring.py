import jax
import jax.numpy as jnp
import numpy as np

import dq_dock_engine.docking.scoring as scoring

from dq_dock_engine.docking.formal_handles import (
    attractive_extended_chemistry_theorem_handles,
    attractive_directional_hbond_theorem_handles,
    attractive_halogen_bond_theorem_handles,
    attractive_pi_cation_theorem_handles,
    attractive_pi_stacking_theorem_handles,
    attractive_water_mediated_hbond_theorem_handles,
    contact_surrogate_theorem_handles,
    directional_hbond_finite_theorem_handles,
    extended_rich_chemistry_theorem_handles,
    negation_invariance_theorem_handles,
    rich_chemistry_theorem_handles,
    screened_coulomb_theorem_handles,
    support_expansion_theorem_handles,
    topk_bridge_theorem_handles,
)
from dq_dock_engine.docking.scoring import (
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedScreenedCoulombSpec,
    CertifiedMetalCoordinationSpec,
    CertifiedRichChemistryPlan,
    score_certified_attractive_chemistry_batch,
    score_certified_contact_batch,
    score_certified_directional_hbond_batch,
    score_certified_metal_coordination_batch,
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
        ligand_neighbor_indices=jnp.array([[0, 1], [0, 1]], dtype=jnp.int32),
        receptor_strengths=jnp.array([0.8, 0.6], dtype=jnp.float32),
        ligand_strengths=jnp.array([0.9, 0.7], dtype=jnp.float32),
        ideal_distance=2.8,
        distance_width=0.7,
        cutoff=4.0,
    )
    metal = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array([1.0, 0.0], dtype=jnp.float32),
        ligand_strengths=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    plan = CertifiedRichChemistryPlan(
        screened_coulomb=screened,
        contact=contact,
        directional_hbond=hbond,
        metal_coordination=metal,
    )
    return (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    )


def _toy_geometry_with_inactive_optional_terms():
    receptor_coords, poses_coords, receptor_radii, ligand_radii, plan = _toy_geometry()
    inactive_hbond = CertifiedDirectionalHBondSpec(
        receptor_directions=plan.directional_hbond.receptor_directions,
        ligand_neighbor_indices=plan.directional_hbond.ligand_neighbor_indices,
        receptor_strengths=jnp.zeros_like(plan.directional_hbond.receptor_strengths),
        ligand_strengths=plan.directional_hbond.ligand_strengths,
        ideal_distance=plan.directional_hbond.ideal_distance,
        distance_width=plan.directional_hbond.distance_width,
        cutoff=9.0,
    )
    inactive_metal = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.zeros_like(plan.metal_coordination.receptor_strengths),
        ligand_strengths=plan.metal_coordination.ligand_strengths,
        ideal_distance=plan.metal_coordination.ideal_distance,
        distance_width=plan.metal_coordination.distance_width,
        cutoff=11.0,
    )
    inactive_plan = CertifiedRichChemistryPlan(
        screened_coulomb=plan.screened_coulomb,
        contact=plan.contact,
        directional_hbond=inactive_hbond,
        metal_coordination=inactive_metal,
    )
    return receptor_coords, poses_coords, receptor_radii, ligand_radii, inactive_plan


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
    assert {"AH1", "AH8"}.issubset(set(attractive_directional_hbond_theorem_handles()))
    assert {"PP6", "PP10"}.issubset(set(attractive_pi_stacking_theorem_handles()))
    assert {"PC6", "PC10"}.issubset(set(attractive_pi_cation_theorem_handles()))
    assert {"XB6", "XB10"}.issubset(set(attractive_halogen_bond_theorem_handles()))
    assert {"WB6", "WB10"}.issubset(
        set(attractive_water_mediated_hbond_theorem_handles())
    )
    assert {"XR1", "XR5"}.issubset(set(attractive_extended_chemistry_theorem_handles()))
    assert {"XR6", "XR10"}.issubset(set(extended_rich_chemistry_theorem_handles()))
    assert {"NG1", "NG5"}.issubset(set(negation_invariance_theorem_handles()))
    assert {"AR1", "AR10"}.issubset(set(rich_chemistry_theorem_handles()))
    assert {"TK13", "TK15"}.issubset(set(topk_bridge_theorem_handles()))
    assert {"SH1", "SH6"}.issubset(set(support_expansion_theorem_handles()))


def test_new_chemistry_scoring_helpers_are_jit_safe() -> None:
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    @jax.jit
    def run_all():
        screened_batch = score_certified_screened_coulomb_batch(
            receptor_coords, poses_coords, plan.screened_coulomb
        )
        contact_batch = score_certified_contact_batch(
            receptor_coords, poses_coords, plan.contact
        )
        hbond_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.directional_hbond
        )
        metal_batch = score_certified_metal_coordination_batch(
            receptor_coords, poses_coords, plan.metal_coordination
        )
        nonbonded_batch = score_certified_lj_screened_coulomb_batch(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            plan.screened_coulomb,
            target_error=0.001,
        )
        rich_batch = score_certified_rich_chemistry_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            rich_chemistry_plan=plan,
            target_error=0.001,
        )
        return (
            screened_batch.scores,
            screened_batch.error_bound,
            contact_batch.scores,
            contact_batch.error_bound,
            hbond_batch.scores,
            hbond_batch.error_bound,
            metal_batch.scores,
            metal_batch.error_bound,
            nonbonded_batch.scores,
            nonbonded_batch.error_bound,
            rich_batch.scores,
            rich_batch.error_bound,
        )

    outputs = run_all()
    for value in outputs:
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr))


def test_attractive_chemistry_is_attractive_negative_energy() -> None:
    (
        receptor_coords,
        poses_coords,
        _receptor_radii,
        _ligand_radii,
        plan,
    ) = _toy_geometry()

    contact_batch = score_certified_contact_batch(
        receptor_coords, poses_coords, plan.contact
    )
    hbond_batch = score_certified_directional_hbond_batch(
        receptor_coords, poses_coords, plan.directional_hbond
    )
    metal_batch = score_certified_metal_coordination_batch(
        receptor_coords, poses_coords, plan.metal_coordination
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords, poses_coords, plan
    )

    np.testing.assert_allclose(
        np.asarray(attractive_batch.scores),
        -(np.asarray(contact_batch.scores) + np.asarray(hbond_batch.scores))
        + np.asarray(metal_batch.scores),
        atol=1e-5,
    )
    assert np.isclose(
        float(attractive_batch.error_bound),
        float(
            contact_batch.error_bound
            + hbond_batch.error_bound
            + metal_batch.error_bound
        ),
    )


def test_rich_chemistry_composes_nonbonded_with_attractive_chemistry() -> None:
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    nonbonded_batch = score_certified_lj_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan.screened_coulomb,
        target_error=0.001,
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords, poses_coords, plan
    )
    rich_batch = score_certified_rich_chemistry_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        rich_chemistry_plan=plan,
        target_error=0.001,
    )

    np.testing.assert_allclose(
        np.asarray(rich_batch.scores),
        np.asarray(nonbonded_batch.scores) + np.asarray(attractive_batch.scores),
    )


def test_inactive_directional_hbond_short_circuits_without_kernel_calls(
    monkeypatch,
) -> None:
    receptor_coords, poses_coords, _receptor_radii, _ligand_radii, plan = (
        _toy_geometry_with_inactive_optional_terms()
    )

    assert not bool(plan.directional_hbond.is_active)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("inactive directional hbond spec should short-circuit")

    monkeypatch.setattr(
        scoring, "_score_directional_hbond_exact_batch", _should_not_run
    )
    monkeypatch.setattr(
        scoring, "_score_directional_hbond_cutoff_batch", _should_not_run
    )

    batch = score_certified_directional_hbond_batch(
        receptor_coords, poses_coords, plan.directional_hbond
    )

    np.testing.assert_allclose(
        np.asarray(batch.scores), np.zeros(poses_coords.shape[0])
    )
    assert float(batch.error_bound) == 0.0
    assert float(batch.target_error) == 0.0
    assert float(batch.cutoff_radius) == 0.0


def test_inactive_rich_chemistry_terms_are_jit_safe_and_structurally_absent() -> None:
    receptor_coords, poses_coords, _receptor_radii, _ligand_radii, plan = (
        _toy_geometry_with_inactive_optional_terms()
    )
    contact_batch = score_certified_contact_batch(
        receptor_coords, poses_coords, plan.contact
    )

    assert not bool(plan.has_active_directional_hbond)
    assert not bool(plan.has_active_metal_coordination)

    @jax.jit
    def run_inactive():
        hbond_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.directional_hbond
        )
        metal_batch = score_certified_metal_coordination_batch(
            receptor_coords, poses_coords, plan.metal_coordination
        )
        attractive_batch = score_certified_attractive_chemistry_batch(
            receptor_coords, poses_coords, plan
        )
        return (
            hbond_batch.scores,
            hbond_batch.error_bound,
            hbond_batch.cutoff_radius,
            metal_batch.scores,
            metal_batch.error_bound,
            metal_batch.cutoff_radius,
            attractive_batch.scores,
            attractive_batch.error_bound,
            attractive_batch.cutoff_radius,
        )

    (
        hbond_scores,
        hbond_error_bound,
        hbond_cutoff_radius,
        metal_scores,
        metal_error_bound,
        metal_cutoff_radius,
        attractive_scores,
        attractive_error_bound,
        attractive_cutoff_radius,
    ) = run_inactive()

    np.testing.assert_allclose(
        np.asarray(hbond_scores), np.zeros(poses_coords.shape[0])
    )
    np.testing.assert_allclose(
        np.asarray(metal_scores), np.zeros(poses_coords.shape[0])
    )
    np.testing.assert_allclose(
        np.asarray(attractive_scores),
        -np.asarray(contact_batch.scores),
        atol=1e-5,
    )
    assert float(hbond_error_bound) == 0.0
    assert float(hbond_cutoff_radius) == 0.0
    assert float(metal_error_bound) == 0.0
    assert float(metal_cutoff_radius) == 0.0
    assert np.isclose(float(attractive_error_bound), float(contact_batch.error_bound))
    assert np.isclose(
        float(attractive_cutoff_radius), float(contact_batch.cutoff_radius)
    )


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
