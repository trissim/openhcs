"""Regression tests for JIT-safety of the rich-chemistry refinement path.

These tests live in a dedicated file so they are never skipped by the
stale-import guard in test_formal_optimizer.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking import pipeline as docking_pipeline
from dq_dock_engine.docking.chemistry_runtime import (
    AnchoredSiteArray,
    HalogenBondInteractionTerm,
    IndexedSiteArray,
    PiStackingInteractionTerm,
    SiteGeometry,
)
from dq_dock_engine.docking.formal_optimizer import refine_poses_certified
from dq_dock_engine.docking.scoring import (
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedExtendedInteractionBundle,
    CertifiedMetalCoordinationSpec,
    CertifiedRealSpaceEwaldSpec,
    CertifiedRichChemistryPlan,
    CertifiedScreenedCoulombSpec,
)
from dq_dock_engine.docking.scoring_context import CertifiedScoringContext
from dq_dock_engine.docking_config import ExactChemistryMode


def _directional_anchored_sites(
    *,
    positions: tuple[tuple[float, float, float], ...],
    vectors: tuple[tuple[float, float, float], ...],
    strengths: tuple[float, ...],
    anchors: tuple[int, ...],
) -> AnchoredSiteArray:
    return AnchoredSiteArray(
        geometry=SiteGeometry.DIRECTIONAL,
        positions=jnp.array(positions, dtype=jnp.float32),
        vectors=jnp.array(vectors, dtype=jnp.float32),
        strengths=jnp.array(strengths, dtype=jnp.float32),
        anchor_indices=jnp.array(anchors, dtype=jnp.int32),
    )


def _directional_indexed_sites(
    *,
    atoms: tuple[tuple[int, ...], ...],
    refs: tuple[tuple[int, ...], ...],
    strengths: tuple[float, ...],
) -> IndexedSiteArray:
    return IndexedSiteArray(
        geometry=SiteGeometry.DIRECTIONAL,
        atom_index_rows=jnp.array(atoms, dtype=jnp.int32),
        atom_index_mask=jnp.ones((len(atoms), len(atoms[0])), dtype=bool),
        reference_index_rows=jnp.array(refs, dtype=jnp.int32),
        reference_index_mask=jnp.ones((len(refs), len(refs[0])), dtype=bool),
        strengths=jnp.array(strengths, dtype=jnp.float32),
    )


def _make_halogen_context(n_receptor: int) -> CertifiedScoringContext:
    halogen = HalogenBondInteractionTerm(
        receptor_acceptors=_directional_anchored_sites(
            positions=((0.0, 0.0, 0.0),),
            vectors=((1.0, 0.0, 0.0),),
            strengths=(1.0,),
            anchors=(0,),
        ),
        receptor_donors=_directional_anchored_sites(
            positions=((0.0, 0.0, 0.0),),
            vectors=((1.0, 0.0, 0.0),),
            strengths=(0.0,),
            anchors=(0,),
        ),
        ligand_acceptors=_directional_indexed_sites(
            atoms=((0,),),
            refs=((1,),),
            strengths=(0.0,),
        ),
        ligand_donors=_directional_indexed_sites(
            atoms=((0,),),
            refs=((1,),),
            strengths=(1.0,),
        ),
    )
    rich_plan = CertifiedRichChemistryPlan(
        screened_coulomb=CertifiedScreenedCoulombSpec(
            receptor_charges=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_charges=jnp.zeros((2,), dtype=jnp.float32),
        ),
        contact=CertifiedContactSurrogateSpec(
            receptor_weights=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_weights=jnp.zeros((2,), dtype=jnp.float32),
        ),
        hbond_receptor_donor=CertifiedDirectionalHBondSpec(
            receptor_anchor_indices=jnp.zeros((n_receptor,), dtype=jnp.int32),
            receptor_directions=jnp.zeros((n_receptor, 3), dtype=jnp.float32),
            ligand_anchor_indices=jnp.zeros((2,), dtype=jnp.int32),
            ligand_local_directions=jnp.zeros((2, 3), dtype=jnp.float32),
            ligand_frame_coords=jnp.zeros((2, 3), dtype=jnp.float32),
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((2,), dtype=jnp.float32),
        ),
        hbond_ligand_donor=CertifiedDirectionalHBondSpec(
            receptor_anchor_indices=jnp.zeros((n_receptor,), dtype=jnp.int32),
            receptor_directions=jnp.zeros((n_receptor, 3), dtype=jnp.float32),
            ligand_anchor_indices=jnp.zeros((2,), dtype=jnp.int32),
            ligand_local_directions=jnp.zeros((2, 3), dtype=jnp.float32),
            ligand_frame_coords=jnp.zeros((2, 3), dtype=jnp.float32),
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((2,), dtype=jnp.float32),
            receptor_alignment_sign=-1.0,
            ligand_alignment_sign=1.0,
        ),
        metal_coordination=CertifiedMetalCoordinationSpec(
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((2,), dtype=jnp.float32),
            receptor_ideal_angles=jnp.zeros((n_receptor,), dtype=jnp.float32),
        ),
        pairwise_sigma=jnp.full((n_receptor, 2), 3.22, dtype=jnp.float32),
        extended_terms=CertifiedExtendedInteractionBundle(terms=(halogen,)),
    )
    return CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        rich_chemistry_plan=rich_plan,
    )


def _make_pi_stacking_context(n_receptor: int) -> CertifiedScoringContext:
    receptor_rings = AnchoredSiteArray(
        geometry=SiteGeometry.RING,
        positions=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        vectors=jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
        strengths=jnp.array([1.0], dtype=jnp.float32),
        anchor_indices=jnp.array([0], dtype=jnp.int32),
    )
    ligand_rings = IndexedSiteArray(
        geometry=SiteGeometry.RING,
        atom_index_rows=jnp.zeros((1, 1), dtype=jnp.int32),
        atom_index_mask=jnp.ones((1, 1), dtype=bool),
        reference_index_rows=jnp.zeros((1, 3), dtype=jnp.int32),
        reference_index_mask=jnp.ones((1, 3), dtype=bool),
        strengths=jnp.array([1.0], dtype=jnp.float32),
    )
    pi_stacking = PiStackingInteractionTerm(
        receptor_rings=receptor_rings,
        ligand_rings=ligand_rings,
    )
    rich_plan = CertifiedRichChemistryPlan(
        screened_coulomb=CertifiedScreenedCoulombSpec(
            receptor_charges=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_charges=jnp.zeros((1,), dtype=jnp.float32),
        ),
        contact=CertifiedContactSurrogateSpec(
            receptor_weights=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_weights=jnp.zeros((1,), dtype=jnp.float32),
        ),
        hbond_receptor_donor=CertifiedDirectionalHBondSpec(
            receptor_anchor_indices=jnp.zeros((n_receptor,), dtype=jnp.int32),
            receptor_directions=jnp.zeros((n_receptor, 3), dtype=jnp.float32),
            ligand_anchor_indices=jnp.zeros((1,), dtype=jnp.int32),
            ligand_local_directions=jnp.zeros((1, 3), dtype=jnp.float32),
            ligand_frame_coords=jnp.zeros((1, 3), dtype=jnp.float32),
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((1,), dtype=jnp.float32),
        ),
        hbond_ligand_donor=CertifiedDirectionalHBondSpec(
            receptor_anchor_indices=jnp.zeros((n_receptor,), dtype=jnp.int32),
            receptor_directions=jnp.zeros((n_receptor, 3), dtype=jnp.float32),
            ligand_anchor_indices=jnp.zeros((1,), dtype=jnp.int32),
            ligand_local_directions=jnp.zeros((1, 3), dtype=jnp.float32),
            ligand_frame_coords=jnp.zeros((1, 3), dtype=jnp.float32),
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((1,), dtype=jnp.float32),
            receptor_alignment_sign=-1.0,
            ligand_alignment_sign=1.0,
        ),
        metal_coordination=CertifiedMetalCoordinationSpec(
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((1,), dtype=jnp.float32),
            receptor_ideal_angles=jnp.zeros((n_receptor,), dtype=jnp.float32),
        ),
        pairwise_sigma=jnp.full((n_receptor, 1), 3.22, dtype=jnp.float32),
        extended_terms=CertifiedExtendedInteractionBundle(terms=(pi_stacking,)),
    )
    return CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        rich_chemistry_plan=rich_plan,
    )


def test_directional_hbond_receptor_subset_is_jit_safe_and_remaps_indices():
    spec = CertifiedDirectionalHBondSpec(
        receptor_anchor_indices=jnp.array([1, 3], dtype=jnp.int32),
        receptor_directions=jnp.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        ligand_anchor_indices=jnp.array([0], dtype=jnp.int32),
        ligand_local_directions=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        ligand_frame_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_strengths=jnp.array([0.8, 0.6], dtype=jnp.float32),
        ligand_strengths=jnp.array([1.0], dtype=jnp.float32),
    )

    @jax.jit
    def subset(indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        narrowed = spec.receptor_subset(indices)
        return narrowed.receptor_anchor_indices, narrowed.receptor_strengths

    remapped_indices, remapped_strengths = subset(jnp.array([4, 3], dtype=jnp.int32))

    np.testing.assert_array_equal(
        np.asarray(remapped_indices),
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(remapped_strengths),
        np.array([0.0, 0.6], dtype=np.float32),
    )


def test_refine_poses_certified_extended_rich_pi_stacking_no_tracer_breach():
    """Regression: receptor_subset must not be called inside JIT when rich
    chemistry is active.  Before the fix, AnchoredSiteArray.subset used
    np.isin / np.asarray on JAX tracers and raised TracerBoolConversionError."""
    n_receptor = 4
    receptor_coords = jnp.array(
        [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 1.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    receptor_radii = jnp.ones((n_receptor,), dtype=jnp.float32)
    ligand_radii = jnp.array([1.0], dtype=jnp.float32)
    initial_coords = jnp.array([[[10.0, 0.0, 0.0]]], dtype=jnp.float32)
    scoring_context = _make_pi_stacking_context(n_receptor)

    refined_coords, history = refine_poses_certified(
        coords_batch=initial_coords,
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        n_rounds=1,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
        scoring_context=scoring_context,
    )

    assert refined_coords.shape == initial_coords.shape
    assert len(history) == 1


def test_optimization_context_strips_extended_rich_terms_but_keeps_electrostatics():
    rich_only = _make_pi_stacking_context(4)
    electrostatics = CertifiedRealSpaceEwaldSpec(
        receptor_charges=jnp.zeros((4,), dtype=jnp.float32),
        ligand_charges=jnp.zeros((1,), dtype=jnp.float32),
    )
    context = CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        electrostatics=electrostatics,
        rich_chemistry_plan=rich_only.rich_chemistry_plan,
    )

    optimization_context = context.optimization_context()
    ranking_context = context.ranking_context()

    assert optimization_context.exact_chemistry_mode == ExactChemistryMode.NONE
    assert optimization_context.electrostatics is electrostatics
    assert optimization_context.rich_chemistry_plan is None
    assert optimization_context.water_grid is None
    assert optimization_context.receptor_conformations is None
    assert ranking_context.exact_chemistry_mode == ExactChemistryMode.NONE


def test_halogen_only_context_does_not_activate_orientation_disambiguation():
    context = _make_halogen_context(1)

    assert context.rich_orientation_disambiguation_active() is False
    assert context.ranking_context().exact_chemistry_mode == ExactChemistryMode.NONE


def test_pruning_context_strips_halogen_and_reenables_batch_pruning_delta():
    context = _make_halogen_context(1)

    pruning_context = context.pruning_context()

    assert pruning_context.rich_chemistry_plan is not None
    assert pruning_context.uses_batch_pruning_delta() is True
    assert bool(np.asarray(pruning_context.rich_chemistry_plan.has_active_extended_terms)) is False
    assert pruning_context.rich_chemistry_plan.extended_terms.terms == ()


def test_halogen_only_flip_disambiguation_matches_base_physics_score():
    context = _make_halogen_context(1)
    receptor_coords = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    poses_coords = jnp.array([[[3.2, 0.0, 0.0], [2.2, 0.0, 0.0]]], dtype=jnp.float32)
    receptor_radii = jnp.array([1.7], dtype=jnp.float32)
    ligand_radii = jnp.array([1.5, 1.5], dtype=jnp.float32)

    base_result = context.optimization_context().score_exact_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=0.001,
        epsilon=0.086,
    )
    disambiguation_result = context.score_flip_disambiguation_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=0.001,
        epsilon=0.086,
    )

    np.testing.assert_allclose(
        np.asarray(disambiguation_result.scores),
        np.asarray(base_result.scores),
    )
    np.testing.assert_allclose(
        np.asarray(disambiguation_result.posewise_error_bound),
        np.asarray(base_result.posewise_error_bound),
    )
    assert float(disambiguation_result.error_bound) == float(base_result.error_bound)


def test_orientation_margin_handles_ignore_halogen_terms():
    context = _make_halogen_context(1)

    assert docking_pipeline._orientation_margin_theorem_handles(context) == (
        "HB15",
        "AH11",
    )
