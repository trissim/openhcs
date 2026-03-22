"""Regression tests for JIT-safety of the rich-chemistry refinement path.

These tests live in a dedicated file so they are never skipped by the
stale-import guard in test_formal_optimizer.py.
"""

import jax.numpy as jnp

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
        directional_hbond=CertifiedDirectionalHBondSpec(
            receptor_directions=jnp.zeros((n_receptor, 3), dtype=jnp.float32),
            ligand_neighbor_indices=jnp.zeros((1, 1), dtype=jnp.int32),
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((1,), dtype=jnp.float32),
        ),
        metal_coordination=CertifiedMetalCoordinationSpec(
            receptor_strengths=jnp.zeros((n_receptor,), dtype=jnp.float32),
            ligand_strengths=jnp.zeros((1,), dtype=jnp.float32),
        ),
        extended_terms=CertifiedExtendedInteractionBundle(terms=(pi_stacking,)),
    )
    return CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        rich_chemistry_plan=rich_plan,
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
