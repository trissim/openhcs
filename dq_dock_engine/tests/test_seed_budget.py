from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dq_dock_engine.docking.core import DockingBox, LigandContext, PoseVector
from dq_dock_engine.docking.conformer_search import RotatableBond
from dq_dock_engine.docking.pipeline import (
    PipelineDockingRequest,
    PipelineInitialScores,
    PipelinePoseBatch,
    PipelineRoute,
    _conformer_local_improvement_bounds,
    _derive_target_error_from_rmsd,
    _derive_adaptive_torsion_support_spec,
    _derive_local_rotation_step_rad,
    _detect_rigid_equivalence_ambiguity,
    _ligand_radius,
    _probe_seed_budget_certificate,
    _seed_budget_torsion_count,
    _shared_certified_singleton_top1,
    derive_seed_budget,
    derive_seed_budget_plan,
)
from dq_dock_engine.docking.se3_refinement import (
    RefinementCertificate,
    SE3SpectralCertificate,
    _initial_probe_backtracking_round,
    _stabilized_probe_step_limits,
)
from dq_dock_engine.docking_config import (
    ExactChemistryMode,
    RefinementCertificationMode,
    SofteningPolicy,
    create_config,
)


def _dummy_ligand_context() -> LigandContext:
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32)
    return LigandContext(
        base_coords=coords,
        base_radii=jnp.array([1.7, 1.7], dtype=jnp.float32),
        center_of_mass=jnp.array([0.75, 0.0, 0.0], dtype=jnp.float32),
        elements=("C", "C"),
        adjacency=((1,), (0,)),
    )


def _flexible_ligand_context() -> LigandContext:
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    center_of_mass = jnp.mean(coords, axis=0)
    return LigandContext(
        base_coords=coords - center_of_mass,
        base_radii=jnp.full((5,), 1.7, dtype=jnp.float32),
        center_of_mass=center_of_mass,
        elements=("C", "C", "C", "C", "C"),
        adjacency=((1,), (0, 2), (1, 3), (2, 4), (3,)),
    )


def _dummy_certificate(*, q: float, n_steps: int) -> RefinementCertificate:
    return RefinementCertificate(
        spectral=SE3SpectralCertificate(
            lmin_param=2.0,
            lmax_param=6.0,
            sigma_min_sq=1.5,
            sigma_max_sq=2.0,
            mu_coord=1.0,
            M_coord=4.0,
        ),
        q=q,
        initial_gap=10.0,
        target_rmsd=0.5,
        n_steps=n_steps,
        mode=create_config("certified").refinement_certification,
    )


def test_derive_seed_budget_uses_formulaic_zero_step_capture() -> None:
    box_size = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    confidence = 0.9
    target_rmsd = 0.5
    ligand_radius = 2.0

    expected_trans = (4.0 / 3.0) * math.pi * target_rmsd**3
    rot_radius = target_rmsd / ligand_radius
    expected_rot = (4.0 / 3.0) * math.pi * rot_radius**3
    ratio = (expected_trans * expected_rot) / (1000.0 * (2.0 * math.pi**2))
    expected = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - ratio))

    assert (
        derive_seed_budget(
            confidence=confidence,
            box_size=box_size,
            target_rmsd=target_rmsd,
            ligand_radius=ligand_radius,
        )
        == expected
    )


def test_probe_certificate_expands_capture_radius_and_reduces_budget() -> None:
    box_size = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    base = derive_seed_budget(
        confidence=0.99,
        box_size=box_size,
        target_rmsd=0.5,
        ligand_radius=2.0,
    )
    probed = derive_seed_budget(
        confidence=0.99,
        box_size=box_size,
        target_rmsd=0.5,
        ligand_radius=2.0,
        probe_certificate=_dummy_certificate(q=0.25, n_steps=2),
    )

    assert probed < base


def test_rigid_seed_budget_ignores_conformer_torsions() -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_flexible_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    assert _seed_budget_torsion_count(request) == 0


def test_local_rotation_step_matches_translation_scale() -> None:
    assert _derive_local_rotation_step_rad(0.5, 10.0) == pytest.approx(0.05)
    assert _derive_local_rotation_step_rad(0.5, 0.0) == pytest.approx(math.pi / 2.0)


class _FakeRoute(PipelineRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        n = request.n_poses_override or 1
        pose_vecs = PoseVector(
            translation=jnp.zeros((n, 3), dtype=jnp.float32),
            quaternion=jnp.tile(
                jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (n, 1)
            ),
        )
        return PipelinePoseBatch(request=request, pose_vecs=pose_vecs)

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del request, batched_coords
        n = pose_vecs.translation.shape[0]
        return PipelineInitialScores(final_scores=jnp.arange(n, dtype=jnp.float32))


def test_probe_seed_budget_certificate_overrides_request_budget(monkeypatch) -> None:
    cert = _dummy_certificate(q=0.5, n_steps=4)

    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline._certified_refinement",
        lambda **kwargs: (
            kwargs["initial_translations"],
            kwargs["initial_quaternions"],
            [cert],
        ),
    )

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    updated_request, probe_cert = _probe_seed_budget_certificate(request, _FakeRoute())

    assert probe_cert == cert
    expected_budget = derive_seed_budget(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=_seed_budget_torsion_count(request),
        probe_certificate=cert,
    )
    assert updated_request.n_poses_override is None
    assert updated_request.seed_budget_plan is not None
    assert updated_request.seed_budget_plan.selected_budget == expected_budget
    assert updated_request.n_poses == expected_budget


def test_probe_seed_budget_certificate_falls_back_to_original_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline._certified_refinement",
        lambda **kwargs: (
            kwargs["initial_translations"],
            kwargs["initial_quaternions"],
            [None],
        ),
    )

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    updated_request, probe_cert = _probe_seed_budget_certificate(request, _FakeRoute())

    assert probe_cert is None
    assert updated_request.n_poses_override is None
    assert updated_request.seed_budget_plan is not None
    assert (
        updated_request.seed_budget_plan.selected_candidate.source
        == "baseline_zero_step_capture"
    )
    assert updated_request.n_poses == request.n_poses


def test_probe_seed_budget_certificate_uses_smallest_successful_budget(
    monkeypatch,
) -> None:
    cert_large = _dummy_certificate(q=0.5, n_steps=1)
    cert_small = _dummy_certificate(q=0.25, n_steps=2)
    calls = iter([cert_large, cert_small, None, None])

    class _RankedRoute(_FakeRoute):
        def score_pose_batch(
            self,
            request: PipelineDockingRequest,
            batched_coords: jnp.ndarray,
            pose_vecs: PoseVector,
        ) -> PipelineInitialScores:
            del request, batched_coords, pose_vecs
            return PipelineInitialScores(
                final_scores=jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
            )

    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline._certified_refinement",
        lambda **kwargs: (
            kwargs["initial_translations"],
            kwargs["initial_quaternions"],
            [next(calls)],
        ),
    )

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    updated_request, probe_cert = _probe_seed_budget_certificate(
        request, _RankedRoute()
    )

    budget_large = derive_seed_budget(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=0,
        probe_certificate=cert_large,
    )
    budget_small = derive_seed_budget(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=0,
        probe_certificate=cert_small,
    )

    assert probe_cert == cert_small
    assert updated_request.seed_budget_plan is not None
    assert updated_request.seed_budget_plan.selected_budget == min(
        budget_large,
        budget_small,
    )
    assert updated_request.n_poses == min(budget_large, budget_small)


def test_derive_seed_budget_plan_keeps_baseline_and_probe_candidates() -> None:
    cert = _dummy_certificate(q=0.25, n_steps=2)
    plan = derive_seed_budget_plan(
        confidence=0.99,
        box_size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        target_rmsd=0.5,
        ligand_radius=2.0,
        probe_candidates=((0, cert),),
    )

    assert len(plan.candidates) == 2
    assert plan.candidates[0].source == "baseline_zero_step_capture"
    assert plan.candidates[1].source == "observed_probe_capture"
    assert plan.selected_candidate.budget <= plan.candidates[0].budget
    assert plan.engineering_probe_pose_cap is not None
    assert plan.engineering_probe_top_k is not None


def test_probe_seed_budget_certificate_uses_observed_refinement_mode(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_refinement(**kwargs):
        captured["mode_override"] = kwargs["mode_override"]
        return kwargs["initial_translations"], kwargs["initial_quaternions"], [None]

    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline._certified_refinement",
        _fake_refinement,
    )

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    _probe_seed_budget_certificate(request, _FakeRoute())

    assert captured["mode_override"] == RefinementCertificationMode.OBSERVED


def test_detect_rigid_equivalence_ambiguity_flags_uncertified_flip_family() -> None:
    coords = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.array([0.0, 0.05, 1.0], dtype=jnp.float32)

    assert _detect_rigid_equivalence_ambiguity(
        coords,
        scores,
        error_bound=0.05,
        target_rmsd=1.0,
    )


def test_shared_certified_singleton_top1_requires_agreement_and_both_gaps() -> None:
    assert _shared_certified_singleton_top1(
        jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
        0.1,
        jnp.array([0.05, 1.5, 3.0], dtype=jnp.float32),
        0.1,
    )

    assert not _shared_certified_singleton_top1(
        jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
        0.1,
        jnp.array([0.0, 0.05, 2.0], dtype=jnp.float32),
        0.1,
    )

    assert not _shared_certified_singleton_top1(
        jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
        0.1,
        jnp.array([1.0, 0.0, 2.0], dtype=jnp.float32),
        0.1,
    )


def test_stabilized_probe_step_limits_split_rmsd_budget_evenly() -> None:
    translation_limit, rotation_limit = _stabilized_probe_step_limits(4.0, 2.0)

    assert translation_limit == 1.0
    assert rotation_limit == 0.25


def test_stabilized_probe_step_limits_allow_free_rotation_for_zero_radius() -> None:
    translation_limit, rotation_limit = _stabilized_probe_step_limits(0.0, 2.0)

    assert translation_limit == 1.0
    assert rotation_limit == math.pi


def test_initial_probe_backtracking_round_uses_joint_half_budget() -> None:
    assert _initial_probe_backtracking_round(4.0, 0.0, 3.0, 2.0) == 2
    assert _initial_probe_backtracking_round(0.0, 1.0, 4.0, 2.0) == 2


def test_adaptive_torsion_support_spec_uses_rmsd_capped_radius_and_convergence() -> (
    None
):
    max_cells, min_cell_radius, segments = _derive_adaptive_torsion_support_spec(
        per_bond_lipschitz=(1.0,),
        target_delta=100.0,
        target_rmsd=1.0,
        max_arm=1.0,
    )

    assert segments == (7,)
    assert max_cells == 7
    assert min_cell_radius == pytest.approx(math.nextafter(1.0, 0.0))


def test_adaptive_torsion_support_spec_derives_radius_from_target_error() -> None:
    max_cells, min_cell_radius, segments = _derive_adaptive_torsion_support_spec(
        per_bond_lipschitz=(4.0, 1.0),
        target_delta=2.0,
        target_rmsd=10.0,
        max_arm=1.0,
    )

    assert min_cell_radius == pytest.approx(math.nextafter(0.5, 0.0))
    assert segments == (101, 26)
    assert max_cells > 0


def test_conformer_local_improvement_bounds_prefers_physical_barriers(
    monkeypatch,
) -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )
    bonds = (
        RotatableBond(
            atom_i=0, atom_j=1, rotating_atom_indices=(1,), max_arm_length=1.0
        ),
    )

    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline.derive_uff_torsion_barrier_heights",
        lambda ligand_source_path, expected_elements, rotatable_bonds: jnp.array(
            [1.5], dtype=jnp.float32
        ),
    )

    bounds = _conformer_local_improvement_bounds(request, bonds, (2.0,))

    assert bounds is not None
    np.testing.assert_allclose(np.asarray(bounds), np.asarray([3.0], dtype=np.float32))


def test_derived_target_error_from_rmsd_uses_physical_lj_lipschitz() -> None:
    derived = _derive_target_error_from_rmsd(
        0.5,
        jnp.array([1.5, 2.0], dtype=jnp.float32),
        jnp.array([1.0, 1.2], dtype=jnp.float32),
        ExactChemistryMode.NONE,
        SofteningPolicy.NONE,
    )

    expected_sigma = 2.5
    expected_lipschitz = 762.0 * 0.086 / expected_sigma
    assert derived == pytest.approx(expected_lipschitz * 0.5)


def test_pipeline_request_derives_target_error_from_target_rmsd_when_unspecified() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.array([1.5], dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    expected = _derive_target_error_from_rmsd(
        0.5,
        request.receptor_radii,
        request.ligand_ctx.base_radii,
        ExactChemistryMode.EXTENDED_RICH,
        SofteningPolicy.CANONICAL_MAX_SIGMA,
    )
    assert request.target_error == pytest.approx(expected)


def test_derived_target_error_uses_canonical_softened_lipschitz_for_extended_rich() -> (
    None
):
    derived = _derive_target_error_from_rmsd(
        0.5,
        jnp.array([1.5], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        ExactChemistryMode.EXTENDED_RICH,
        SofteningPolicy.CANONICAL_MAX_SIGMA,
    )

    assert derived == pytest.approx((24.0 * 0.086 / 2.5) * 0.5)


def test_derived_target_error_uses_raw_lj_when_softening_policy_is_none() -> None:
    derived = _derive_target_error_from_rmsd(
        0.5,
        jnp.array([1.5], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        ExactChemistryMode.EXTENDED_RICH,
        SofteningPolicy.NONE,
    )

    assert derived == pytest.approx((762.0 * 0.086 / 2.5) * 0.5)
