"""Regression tests for JIT-safety of the rich-chemistry refinement path.

These tests live in a dedicated file so they are never skipped by the
stale-import guard in test_formal_optimizer.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.chemistry_runtime import (
    AnchoredSiteArray,
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
    CertifiedRichChemistryPlan,
    CertifiedScreenedCoulombSpec,
)
from dq_dock_engine.docking.scoring_context import CertifiedScoringContext
from dq_dock_engine.docking_config import ExactChemistryMode


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

    remapped_indices, remapped_strengths = subset(
        jnp.array([4, 3], dtype=jnp.int32)
    )

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
