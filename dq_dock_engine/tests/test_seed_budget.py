from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dq_dock_engine.docking import pipeline as docking_pipeline
from dq_dock_engine.docking.core import DockingBox, LigandContext, PoseVector
from dq_dock_engine.docking.certified_runtime_plans import (
    ActiveConformerEnergyGapWitness,
    ActiveConformerReturnedPoseWitness,
    ActiveRigidEnergyGapWitness,
    ActiveRigidReturnedPoseWitness,
    CertifiedConformerCoveragePlan,
    InactiveConformerReturnedPoseWitness,
    ReturnedPoseProofCase,
)
from dq_dock_engine.docking.formal_sampling import (
    derive_certified_rigid_seed_family_plan,
)
from dq_dock_engine.docking.conformer_search import (
    BranchAndBoundConfig,
    RotatableBond,
    TorsionCell,
)
from dq_dock_engine.docking.formal_handles import (
    auxiliary_patched_support_output_set_theorem_handles,
    conformer_coverage_theorem_handles,
    enriched_support_selection_transfer_theorem_handles,
    flat_landscape_output_member_theorem_handles,
    member_exact_gap_rmsd_theorem_handles,
    patched_support_coarse_margin_returned_pose_theorem_handles,
    patched_support_posewise_envelope_returned_pose_theorem_handles,
    patched_support_singleton_returned_pose_theorem_handles,
    returned_pose_energy_guarantee_theorem_handles,
)
from dq_dock_engine.docking.pipeline import (
    PipelineDockingRequest,
    PipelineInitialScores,
    PipelinePoseBatch,
    PipelineRoute,
    SEED_BUDGET_PROBE_POSES,
    _build_returned_pose_certification,
    _auxiliary_support_representative_choice,
    _certified_posewise_steric_dominance_singleton_choice,
    _certified_support_coarse_margin_singleton_choice,
    _flat_landscape_selector_enabled,
    _flat_landscape_structural_member_local_index,
    _derive_rigid_energy_gap_proof_plan,
    _derive_returned_pose_proof_plan,
    _has_conformer_ambiguity_set_certificate_chain,
    _has_rigid_returned_pose_certificate_chain,
    _conformer_local_improvement_bounds,
    _derive_conformer_coverage_plan,
    _derive_target_error_from_rmsd,
    _derive_adaptive_torsion_support_spec,
    _derive_local_rotation_step_rad,
    _detect_rigid_equivalence_ambiguity,
    _ligand_radius,
    _posewise_certified_top1_gap,
    _patched_support_singleton_score_slice,
    _resolved_patched_support_local_indices,
    _probe_seed_budget_certificate,
    _select_base_anchored_rigid_candidate,
    _run_conformer_search_for_pose,
    _recertify_conformer_updated_pose,
    _seed_budget_torsion_count,
    _shared_certified_singleton_top1,
    _winner_posewise_ambiguity_indices,
    _with_rigid_selection_gap,
    derive_seed_budget,
    derive_seed_budget_plan,
    run_docking_pipeline_request,
)
from dq_dock_engine.docking.core import ReturnedPoseContractDecision
from dq_dock_engine.docking.se3_refinement import (
    RefinementCertificate,
    SE3SpectralCertificate,
    _initial_probe_backtracking_round,
    _stabilized_probe_step_limits,
)
from dq_dock_engine.docking.charges import ChargeMethod
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


def _dummy_conformer_coverage_plan() -> CertifiedConformerCoveragePlan:
    return CertifiedConformerCoveragePlan(
        source="test_conformer_coverage",
        n_torsions=1,
        score_lipschitz_constant=2.0,
        per_bond_lipschitz=(1.0,),
        canonical_segments=(8,),
        support_size=8,
        max_cells=8,
        min_cell_radius=0.1,
        support_target_delta=0.1,
        target_delta=0.2,
        target_rmsd=0.5,
        max_arm=1.0,
        theorem_handles=conformer_coverage_theorem_handles(),
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


class _InspectProbeRoute(_FakeRoute):
    def __init__(self) -> None:
        self.probe_request: PipelineDockingRequest | None = None

    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        self.probe_request = request
        return super().generate_pose_batch(request)


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
    expected_plan = derive_seed_budget_plan(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=_seed_budget_torsion_count(request),
        probe_candidates=((0, cert),),
        rigid_seed_box=request.box,
    )
    expected_budget = expected_plan.selected_budget
    assert updated_request.n_poses_override is None
    assert updated_request.seed_budget_plan is not None
    assert updated_request.seed_budget_plan.selected_budget == expected_budget
    assert updated_request.rigid_seed_family_plan is not None
    assert updated_request.rigid_seed_family_plan.adequate_pose_count == expected_budget
    assert updated_request.n_poses == updated_request.rigid_seed_family_plan.pose_count


def test_probe_seed_budget_certificate_keeps_probe_family_small(monkeypatch) -> None:
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
    route = _InspectProbeRoute()

    _updated_request, _probe_cert = _probe_seed_budget_certificate(request, route)

    assert route.probe_request is not None
    assert route.probe_request.rigid_seed_family_plan is not None
    assert route.probe_request.rigid_seed_family_plan.adequate_pose_count == (
        SEED_BUDGET_PROBE_POSES
    )
    assert route.probe_request.rigid_seed_family_plan.pose_count == (
        SEED_BUDGET_PROBE_POSES
    )


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
    assert updated_request.rigid_seed_family_plan is not None
    assert (
        updated_request.seed_budget_plan.selected_candidate.source
        == "baseline_zero_step_capture"
    )
    assert (
        updated_request.seed_budget_plan.selected_budget
        == updated_request.rigid_seed_family_plan.adequate_pose_count
    )
    assert updated_request.n_poses == updated_request.rigid_seed_family_plan.pose_count


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
            [next(calls) for _ in range(int(kwargs["initial_translations"].shape[0]))],
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

    budget_large = derive_seed_budget_plan(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=0,
        probe_candidates=((0, cert_large),),
        rigid_seed_box=request.box,
    ).selected_budget
    budget_small = derive_seed_budget_plan(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=0,
        probe_candidates=((0, cert_small),),
        rigid_seed_box=request.box,
    ).selected_budget

    expected_plan = derive_seed_budget_plan(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=_ligand_radius(request.ligand_ctx),
        n_torsions=0,
        probe_candidates=((0, cert_large), (1, cert_small)),
        rigid_seed_box=request.box,
    )
    expected_probe_cert = (
        cert_large if expected_plan.selected_candidate.probe_rank == 0 else cert_small
    )

    assert probe_cert == expected_probe_cert
    assert updated_request.seed_budget_plan is not None
    assert updated_request.rigid_seed_family_plan is not None
    assert (
        updated_request.seed_budget_plan.selected_budget
        == expected_plan.selected_budget
    )
    assert (
        updated_request.rigid_seed_family_plan.adequate_pose_count
        == expected_plan.selected_budget
    )
    assert updated_request.n_poses == updated_request.rigid_seed_family_plan.pose_count


def test_derive_seed_budget_plan_keeps_baseline_and_probe_candidates() -> None:
    box = DockingBox(
        center=jnp.zeros((3,), dtype=jnp.float32),
        size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
    )
    cert = _dummy_certificate(q=0.25, n_steps=2)
    plan = derive_seed_budget_plan(
        confidence=0.99,
        box_size=box.size,
        target_rmsd=0.5,
        ligand_radius=2.0,
        probe_candidates=((0, cert),),
        rigid_seed_box=box,
    )

    assert len(plan.candidates) == 2
    assert plan.candidates[0].source == "baseline_zero_step_capture"
    assert plan.candidates[1].source == "observed_probe_capture"
    assert plan.selected_candidate.budget <= plan.candidates[0].budget
    assert plan.selected_family_plan is not None
    assert plan.selected_family_plan.adequate_pose_count == plan.selected_budget
    assert plan.selected_family_plan.pose_count >= plan.selected_budget
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


def test_returned_pose_certification_proves_rigid_target_rmsd_when_winner_and_refinement_are_certified() -> (
    None
):
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("TK16",),
        do_conf=False,
    )
    returned_cert = _build_returned_pose_certification(
        proof_plan=proof_plan,
    )

    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
    assert returned_cert.is_target_rmsd_certified


def test_returned_pose_certification_uses_rigid_energy_gap_witness_when_cover_chain_is_available() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[None, None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("TK16",),
        do_conf=False,
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON
    assert isinstance(proof_plan.conformer_witness, ActiveRigidEnergyGapWitness)
    assert proof_plan.conformer_witness_status == "active_rigid_energy_only"
    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_ENERGY_GAP
    assert returned_cert.is_energy_gap_certified
    assert not returned_cert.is_target_rmsd_certified


def test_returned_pose_certification_uses_patched_support_singleton_chain_for_rigid_target_rmsd() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=10.0),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([4.0, 0.0, 2.0], dtype=jnp.float32),
        final_error_bound=0.25,
        final_pose_coords=None,
        refinement_certificates=[None, _dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("TK16",),
        do_conf=False,
        patched_support_singleton_winner_index=1,
        patched_support_singleton_support_indices=(1, 2),
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON
    assert proof_plan.winner_index == 1
    assert proof_plan.support_indices == (1,)
    assert isinstance(proof_plan.conformer_witness, ActiveRigidReturnedPoseWitness)
    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
    assert set(patched_support_singleton_returned_pose_theorem_handles()).issubset(
        set(proof_plan.theorem_handles)
    )


def test_patched_support_singleton_score_slice_recomputes_when_scores_not_full_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )
    opt_coords = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
            [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    class _FakeScoringContext:
        def score_exact_batch(self, **kwargs):
            poses_coords = kwargs["poses_coords"]
            return SimpleNamespace(
                scores=jnp.asarray(poses_coords[:, 0, 0], dtype=jnp.float32),
                error_bound=jnp.asarray(0.5, dtype=jnp.float32),
            )

    monkeypatch.setattr(
        docking_pipeline,
        "_support_specific_cooperative_channel_abs_bounds",
        lambda *args, **kwargs: None,
    )

    support_scores, support_error_bound = _patched_support_singleton_score_slice(
        request=request,
        scoring_context=cast(Any, _FakeScoringContext()),
        opt_coords=opt_coords,
        support_local_indices=np.array([1, 3], dtype=np.int32),
        rich_scores=jnp.array([111.0, 222.0], dtype=jnp.float32),
        rich_error_bound=9.0,
    )

    np.testing.assert_allclose(
        np.asarray(jax.device_get(support_scores), dtype=np.float32),
        np.array([10.0, 30.0], dtype=np.float32),
    )
    assert support_error_bound == pytest.approx(0.5)


def test_resolved_patched_support_local_indices_unions_authoritative_and_fallback_support() -> (
    None
):
    execution_plan = SimpleNamespace(
        patched_support_indices=(11, 17, 23),
        final_survivor_indices=(5, 11, 13, 17, 19, 23),
    )

    support_local_indices = _resolved_patched_support_local_indices(
        cast(Any, execution_plan),
        np.array([0, 2], dtype=np.int32),
        total_count=6,
    )

    np.testing.assert_array_equal(
        support_local_indices,
        np.array([0, 1, 2, 3, 5], dtype=np.int32),
    )


def test_certified_support_coarse_margin_singleton_choice_certifies_exact_support_winner() -> (
    None
):
    winner_index, singleton, delta = _certified_support_coarse_margin_singleton_choice(
        exact_scores=jnp.array([0.0, 3.0, 4.0], dtype=jnp.float32),
        exact_error_bound=0.1,
        guide_scores=jnp.array([0.0, 3.0, 4.0], dtype=jnp.float32),
        guide_error_bound=0.1,
        support_indices=np.array([5, 7, 9], dtype=np.int32),
        omitted_posewise_bounds=np.array([0.2, 0.2, 0.2], dtype=np.float64),
        fallback_score=10.0,
    )

    assert singleton is True
    assert winner_index == 5
    assert delta == pytest.approx(0.4)


def test_auxiliary_support_representative_choice_prefers_disambiguation_plus_omission() -> (
    None
):
    winner_index = _auxiliary_support_representative_choice(
        support_indices=np.array([0, 2, 3], dtype=np.int32),
        base_scores=jnp.array([0.6, 0.3, 0.2], dtype=jnp.float32),
        rich_scores=jnp.array([0.6, 0.5, 0.4], dtype=jnp.float32),
        disambiguation_scores=jnp.array([-0.5, -0.2, -0.1], dtype=jnp.float32),
        omitted_posewise_bounds=np.array([4.0, 1.0, 0.2], dtype=np.float64),
    )

    assert winner_index == 3


def test_auxiliary_support_representative_choice_prefers_rich_dis_agreement() -> None:
    winner_index = _auxiliary_support_representative_choice(
        support_indices=np.array([0, 2, 3], dtype=np.int32),
        base_scores=jnp.array([0.1, 0.4, 0.5], dtype=jnp.float32),
        rich_scores=jnp.array([-1.0, -0.5, -0.4], dtype=jnp.float32),
        disambiguation_scores=jnp.array([-0.8, -0.1, 0.0], dtype=jnp.float32),
        omitted_posewise_bounds=np.array([8.0, 0.1, 0.1], dtype=np.float64),
    )

    assert winner_index == 0


def test_auxiliary_support_representative_choice_prefers_rich_argmin_when_scores_disagree() -> (
    None
):
    winner_index = _auxiliary_support_representative_choice(
        support_indices=np.array([0, 2, 3], dtype=np.int32),
        base_scores=jnp.array([0.6, 0.1, 0.8], dtype=jnp.float32),
        rich_scores=jnp.array([-1.0, -0.9, -0.4], dtype=jnp.float32),
        disambiguation_scores=jnp.array([-0.7, -0.8, 0.0], dtype=jnp.float32),
        omitted_posewise_bounds=np.array([4.0, 1.0, 0.1], dtype=np.float64),
    )

    assert winner_index == 0


def test_certified_posewise_steric_dominance_singleton_choice_certifies_strict_envelope_gap() -> (
    None
):
    winner_index, singleton = _certified_posewise_steric_dominance_singleton_choice(
        guide_scores=jnp.array([0.0, 3.0, 4.0], dtype=jnp.float32),
        support_indices=np.array([5, 7, 9], dtype=np.int32),
        omitted_posewise_bounds=np.array([0.1, 0.2, 0.2], dtype=np.float64),
        fallback_score=10.0,
    )

    assert singleton is True
    assert winner_index == 5


def test_patched_support_posewise_envelope_handles_are_exposed() -> None:
    assert set(patched_support_posewise_envelope_returned_pose_theorem_handles()) == {
        "BCRP5",
        "BCRP9",
        "BCRP11",
        "BCRP21",
        "BCRP22",
        "RPG11",
    }


def test_flat_landscape_selector_enabled_matches_small_ligand_large_pocket_rule() -> (
    None
):
    assert _flat_landscape_selector_enabled(ligand_atom_count=12, pocket_atom_count=203)
    assert _flat_landscape_selector_enabled(ligand_atom_count=5, pocket_atom_count=211)
    assert not _flat_landscape_selector_enabled(
        ligand_atom_count=15, pocket_atom_count=203
    )
    assert not _flat_landscape_selector_enabled(
        ligand_atom_count=12, pocket_atom_count=150
    )


def test_flat_landscape_structural_member_local_index_prefers_strongest_attraction() -> (
    None
):
    winner_index = _flat_landscape_structural_member_local_index(
        support_indices=np.array([0, 4, 7], dtype=np.int32),
        structural_scores=jnp.array([-0.05, -1.29, -0.61], dtype=jnp.float32),
    )

    assert winner_index == 4


def test_flat_landscape_output_member_handles_are_exposed() -> None:
    assert set(flat_landscape_output_member_theorem_handles()) == {"MCB1", "RPG14"}


def test_posewise_certified_top1_gap_detects_interval_separation() -> None:
    assert _posewise_certified_top1_gap(
        jnp.array([0.0, 1.0, 1.5], dtype=jnp.float32),
        np.array([0.1, 0.1, 0.1], dtype=np.float64),
    )
    assert not _posewise_certified_top1_gap(
        jnp.array([0.0, 0.1, 1.5], dtype=jnp.float32),
        np.array([0.1, 0.1, 0.1], dtype=np.float64),
    )


def test_winner_posewise_ambiguity_indices_uses_interval_overlap() -> None:
    assert _winner_posewise_ambiguity_indices(
        jnp.array([0.0, 0.2, 1.0], dtype=jnp.float32),
        np.array([0.05, 0.05, 0.01], dtype=np.float64),
    ) == (0,)
    assert _winner_posewise_ambiguity_indices(
        jnp.array([0.0, 0.2, 1.0], dtype=jnp.float32),
        np.array([0.15, 0.15, 0.01], dtype=np.float64),
    ) == (0, 1)


def test_returned_pose_proof_plan_uses_custom_patched_support_singleton_theorem_handles() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=10.0),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([4.0, 0.0, 2.0], dtype=jnp.float32),
        final_error_bound=0.25,
        final_pose_coords=None,
        refinement_certificates=[None, _dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("BCRC4",),
        do_conf=False,
        patched_support_singleton_theorem_handles=(
            patched_support_coarse_margin_returned_pose_theorem_handles() + ("BCRC4",)
        ),
        patched_support_singleton_winner_index=1,
        patched_support_singleton_support_indices=(1, 2),
    )

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON
    assert set(patched_support_coarse_margin_returned_pose_theorem_handles()).issubset(
        set(proof_plan.theorem_handles)
    )


def test_rigid_energy_ambiguity_plan_can_return_auxiliary_support_representative() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=10.0),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 0.05, 0.08], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[None, None, None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("BCRP5",),
        do_conf=False,
        auxiliary_output_set_theorem_handles=(
            auxiliary_patched_support_output_set_theorem_handles()
        ),
        auxiliary_output_set_winner_index=2,
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET
    assert proof_plan.winner_index == 2
    assert proof_plan.support_indices == (0, 1, 2)
    assert returned_cert is not None
    assert (
        returned_cert.decision
        == ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_ENERGY_GAP
    )
    assert set(auxiliary_patched_support_output_set_theorem_handles()).issubset(
        set(proof_plan.theorem_handles)
    )


def test_rigid_energy_output_member_can_upgrade_to_target_rmsd() -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=10.0),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 0.05, 0.08], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[None, None, _dummy_certificate(q=0.25, n_steps=2)],
        conformer_coverage_plan=None,
        winner_theorem_handles=("BCRP5",),
        do_conf=False,
        certified_energy_output_member_theorem_handles=(
            member_exact_gap_rmsd_theorem_handles()
        ),
        certified_energy_output_member_winner_index=2,
        certified_energy_output_member_gap_budget=1.0,
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON
    assert proof_plan.winner_index == 2
    assert proof_plan.support_indices == (2,)
    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
    assert _has_rigid_returned_pose_certificate_chain(proof_plan)
    assert set(member_exact_gap_rmsd_theorem_handles()).issubset(
        set(proof_plan.theorem_handles)
    )


def test_select_base_anchored_rigid_candidate_uses_combined20_base_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENHCS_QUATERNION_DICTIONARY", "20")
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            16,
            target_translation_cover_radius=1.0,
        ),
    )
    selected_index, selection_gap, debug = _select_base_anchored_rigid_candidate(
        request=request,
        survivor_scores=jnp.array([0.0, 0.25], dtype=jnp.float32),
        survivor_global_indices=np.array([12, 0], dtype=np.int32),
    )

    assert selected_index == 1
    assert selection_gap == pytest.approx(0.25)
    assert debug is not None
    assert debug["applied"] is True
    assert debug["base_winner_index"] == 1
    assert debug["exact_winner_index"] == 0


def test_rigid_energy_gap_certificate_chain_requires_selection_transfer_handles() -> (
    None
):
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )
    witness = ActiveRigidEnergyGapWitness(
        cover_rmsd_radius=0.1,
        cover_gap_budget=0.2,
        certified_energy_gap=0.5,
        theorem_handles=returned_pose_energy_guarantee_theorem_handles(),
        selection_gap_budget=0.3,
    )
    proof_plan = _derive_rigid_energy_gap_proof_plan(
        request=request,
        witness=witness,
        winner_theorem_handles=(),
        winner_refinement_failure_reason="test",
        note="test",
    )

    with pytest.raises(ValueError, match="complete rigid certificate chain"):
        _build_returned_pose_certification(proof_plan=proof_plan)

    certified_witness = _with_rigid_selection_gap(witness, selection_gap_budget=0.3)
    certified_proof_plan = _derive_rigid_energy_gap_proof_plan(
        request=request,
        witness=certified_witness,
        winner_theorem_handles=enriched_support_selection_transfer_theorem_handles(),
        winner_refinement_failure_reason="test",
        note="test",
    )
    returned_cert = _build_returned_pose_certification(proof_plan=certified_proof_plan)

    assert returned_cert is not None
    assert returned_cert.certified_energy_gap == pytest.approx(0.5)


def test_rigid_energy_gap_certificate_does_not_skip_formal_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dq_dock_engine.docking.formal_optimizer as formal_optimizer_module

    class _FakeExecutionPlan:
        def __init__(self) -> None:
            self.theorem_handles: tuple[str, ...] = ()
            self.final_survivor_indices: tuple[int, ...] | None = (0, 1)
            self.patched_support_indices: tuple[int, ...] | None = None
            self.refinement_budget = None

        @property
        def refinement_pose_indices(self) -> tuple[int, ...]:
            return (
                ()
                if self.refinement_budget is None
                else self.refinement_budget.pose_indices
            )

        def with_refinement_budget(self, refinement_budget, *, postfilter_cost_model):
            del postfilter_cost_model
            updated = _FakeExecutionPlan()
            updated.refinement_budget = refinement_budget
            return updated

        def debug_summary(self) -> dict[str, object]:
            return {}

    class _FakeRoute(docking_pipeline.PipelineRoute):
        def generate_pose_batch(self, request):
            pose_vecs = PoseVector(
                translation=jnp.zeros((2, 3), dtype=jnp.float32),
                quaternion=jnp.array(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    dtype=jnp.float32,
                ),
            )
            return PipelinePoseBatch(request=request, pose_vecs=pose_vecs)

        def score_pose_batch(self, request, batched_coords, pose_vecs):
            del request, batched_coords, pose_vecs
            return PipelineInitialScores(
                final_scores=jnp.array([0.0, 1.0], dtype=jnp.float32),
                execution_plan=cast(Any, _FakeExecutionPlan()),
            )

    def _raise_if_refinement_runs(**kwargs):
        del kwargs
        raise RuntimeError("formal refinement reached")

    class _FakeCertifiedContext:
        uses_extended_rich = False
        receptor_conformations = None
        electrostatics = None

        def optimization_context(self):
            return self

        def ranking_context(self):
            return self

        def pruning_context(self):
            return self

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
        n_poses_override=2,
        rigid_seed_family_plan=derive_certified_rigid_seed_family_plan(
            DockingBox(
                center=jnp.zeros((3,), dtype=jnp.float32),
                size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
            ),
            2,
            target_translation_cover_radius=1.0,
        ),
    )

    monkeypatch.setattr(
        docking_pipeline, "derive_pipeline_route", lambda request: _FakeRoute()
    )
    monkeypatch.setattr(
        docking_pipeline,
        "resolve_request_scoring_context",
        lambda *args, **kwargs: _FakeCertifiedContext(),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_certified_rigid_local_refinement_plan",
        lambda *args, **kwargs: SimpleNamespace(
            translation_cell_width=1.0,
            ligand_radius=1.0,
            base_translation_step=0.5,
            base_rotation_step_rad=0.5,
            n_search_rounds=1,
            local_improvement_bound=1.0,
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_exact_pose_batch",
        lambda *args, **kwargs: jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_posewise_rigid_local_improvement_bounds",
        lambda *args, **kwargs: (
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([True, True], dtype=bool),
            np.array([1.0, 1.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        formal_optimizer_module,
        "_run_exact_formal_refinement",
        _raise_if_refinement_runs,
    )
    monkeypatch.setattr(
        formal_optimizer_module,
        "_run_singleton_hybrid_formal_refinement",
        _raise_if_refinement_runs,
    )

    with pytest.raises(RuntimeError, match="formal refinement reached"):
        run_docking_pipeline_request(request)


def test_returned_pose_certification_promotes_conformer_ambiguity_set_when_certified() -> (
    None
):
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
        conformer_coverage_plan=CertifiedConformerCoveragePlan(
            source="test_coverage",
            n_torsions=1,
            score_lipschitz_constant=1.0,
            per_bond_lipschitz=(1.0,),
            canonical_segments=(2,),
            support_size=3,
            max_cells=5,
            min_cell_radius=0.1,
            support_target_delta=0.1,
            target_delta=0.1,
            target_rmsd=0.5,
            max_arm=1.0,
            theorem_handles=("CSC55",),
        ),
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 0.05], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=request.conformer_coverage_plan,
        winner_theorem_handles=("RPG5",),
        do_conf=True,
    )
    returned_cert = _build_returned_pose_certification(
        proof_plan=proof_plan,
    )

    assert _has_conformer_ambiguity_set_certificate_chain(proof_plan)
    assert returned_cert is not None
    assert (
        returned_cert.decision
        == ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_TARGET_RMSD
    )
    assert returned_cert.is_target_rmsd_certified


def test_returned_pose_certification_proves_conformer_target_rmsd_when_chain_is_complete() -> (
    None
):
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=_dummy_conformer_coverage_plan(),
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )
    returned_cert = _build_returned_pose_certification(
        proof_plan=proof_plan,
    )

    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
    assert returned_cert.is_target_rmsd_certified
    assert "RPG6" in returned_cert.theorem_handles


def test_returned_pose_certification_requires_conformer_coverage_witness() -> None:
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )
    returned_cert = _build_returned_pose_certification(
        proof_plan=proof_plan,
    )

    assert returned_cert is not None
    assert (
        returned_cert.decision
        == ReturnedPoseContractDecision.DOWNGRADED_NO_REFINEMENT_CERTIFICATE
    )
    assert not returned_cert.is_target_rmsd_certified
    assert isinstance(
        proof_plan.conformer_witness, InactiveConformerReturnedPoseWitness
    )


def test_returned_pose_proof_plan_derives_rigid_ambiguity_from_plan_state() -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=1.0),
    )
    final_pose_coords = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 0.05, 1.0], dtype=jnp.float32),
        final_error_bound=0.05,
        final_pose_coords=final_pose_coords,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None, None],
        conformer_coverage_plan=None,
        winner_theorem_handles=("TK16",),
        do_conf=False,
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.RIGID_AMBIGUITY
    assert returned_cert is not None
    assert (
        returned_cert.decision
        == ReturnedPoseContractDecision.DOWNGRADED_RIGID_AMBIGUITY
    )


def test_returned_pose_certification_uses_active_conformer_witness_variant() -> None:
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=_dummy_conformer_coverage_plan(),
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )

    assert isinstance(proof_plan.conformer_witness, ActiveConformerReturnedPoseWitness)
    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON


def test_returned_pose_certification_uses_energy_gap_witness_when_basin_missing() -> (
    None
):
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[None, None],
        conformer_coverage_plan=_dummy_conformer_coverage_plan(),
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )
    returned_cert = _build_returned_pose_certification(proof_plan=proof_plan)

    assert proof_plan.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON
    assert isinstance(proof_plan.conformer_witness, ActiveConformerEnergyGapWitness)
    assert returned_cert is not None
    assert returned_cert.decision == ReturnedPoseContractDecision.CERTIFIED_ENERGY_GAP
    assert not returned_cert.is_target_rmsd_certified
    assert returned_cert.is_energy_gap_certified
    assert returned_cert.certified_energy_gap == pytest.approx(0.2)


def test_returned_pose_proof_plan_debug_summary_exposes_case_and_witness_status() -> (
    None
):
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[_dummy_certificate(q=0.25, n_steps=2), None],
        conformer_coverage_plan=_dummy_conformer_coverage_plan(),
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )

    summary = proof_plan.debug_summary()

    assert summary["proof_case"] == ReturnedPoseProofCase.CERTIFIED_SINGLETON.value
    assert summary["conformer_witness_status"] == "active_complete"


def test_returned_pose_proof_plan_marks_conformer_improvement_when_rigid_cert_invalidates() -> (
    None
):
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

    proof_plan = _derive_returned_pose_proof_plan(
        request=request,
        final_scores=jnp.array([0.0, 3.0], dtype=jnp.float32),
        final_error_bound=0.1,
        final_pose_coords=None,
        refinement_certificates=[None, None],
        conformer_improved_mask=(True, False),
        conformer_coverage_plan=_dummy_conformer_coverage_plan(),
        winner_theorem_handles=("TK16",),
        do_conf=True,
    )

    summary = proof_plan.debug_summary()

    assert summary["winner_conformer_improved"] is True
    assert summary["winner_refinement_certificate_present"] is False
    assert (
        summary["proof_case"] == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON.value
    )
    assert summary["conformer_witness_status"] == "active_energy_only"
    assert "energy gap budget" in cast(str, summary["note"])


def test_recertify_conformer_updated_pose_threads_new_certificate(monkeypatch) -> None:
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
    pose_coords = request.ligand_ctx.base_coords

    monkeypatch.setattr(
        docking_pipeline,
        "_certified_refinement",
        lambda request, initial_translations, initial_quaternions, mode_override=None: (
            initial_translations,
            initial_quaternions,
            [_dummy_certificate(q=0.25, n_steps=2)],
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_exact_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context=None: (
            jnp.array([-1.25], dtype=jnp.float32)
        ),
    )

    refined_coords, refined_energy, certificate, cert_handles, basin_mu_coord = (
        _recertify_conformer_updated_pose(
            request,
            pose_coords,
            scoring_context=None,
            electrostatics=None,
        )
    )

    assert refined_coords.shape == pose_coords.shape
    assert refined_energy == pytest.approx(-1.25)
    assert cert_handles == ()
    assert basin_mu_coord is None or math.isnan(basin_mu_coord) or basin_mu_coord >= 0.0


def test_recertify_conformer_updated_pose_finds_spectral_certificate() -> None:
    protein_coords = jnp.array([[50.0, 0.0, 0.0]], dtype=jnp.float32)
    receptor_radii = jnp.array([1.5], dtype=jnp.float32)
    ligand_coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32)
    ligand_ctx = LigandContext(
        base_coords=ligand_coords,
        base_radii=jnp.array([1.5, 1.5], dtype=jnp.float32),
        center_of_mass=jnp.mean(ligand_coords, axis=0),
        elements=("C", "C"),
        charges=jnp.zeros((2,)),
        adjacency=((0, 1), (1, 0)),
    )
    request = PipelineDockingRequest(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_ctx=ligand_ctx,
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([100.0, 100.0, 100.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    captured_calls = []
    original_refinement = docking_pipeline._certified_refinement

    def mock_refinement(
        request, initial_translations, initial_quaternions, mode_override=None
    ):
        captured_calls.append((initial_translations, initial_quaternions))
        return original_refinement(
            request,
            initial_translations,
            initial_quaternions,
            mode_override=mode_override,
        )

    import unittest.mock as mock

    with mock.patch.object(docking_pipeline, "_certified_refinement", mock_refinement):
        try:
            _recertify_conformer_updated_pose(
                request,
                ligand_coords,
                scoring_context=None,
                electrostatics=None,
            )
        except Exception:
            pass

    assert len(captured_calls) == 1, "Should call _certified_refinement"
    init_trans, init_quat = captured_calls[0]
    pose_center = jnp.mean(ligand_coords, axis=0)
    expected_trans = pose_center[None, ...]
    assert jnp.allclose(init_trans, expected_trans), (
        f"Initial translation should be pose_center, got {init_trans}, expected {expected_trans}"
    )
    assert jnp.allclose(init_quat, jnp.array([[1.0, 0.0, 0.0, 0.0]])), (
        f"Initial quaternion should be identity, got {init_quat}"
    )


def test_run_conformer_search_restricts_config_to_active_bonds(monkeypatch) -> None:
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
    rotatable_bonds = (
        RotatableBond(
            atom_i=0,
            atom_j=1,
            rotating_atom_indices=(1, 2),
            max_arm_length=1.0,
        ),
        RotatableBond(
            atom_i=1,
            atom_j=2,
            rotating_atom_indices=(2, 3),
            max_arm_length=1.0,
        ),
        RotatableBond(
            atom_i=2,
            atom_j=3,
            rotating_atom_indices=(3, 4),
            max_arm_length=1.0,
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        docking_pipeline,
        "_build_conformer_score_fns",
        lambda *args, **kwargs: (lambda coords: 0.0, None),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_active_rotatable_bonds_for_pose",
        lambda *args, **kwargs: (
            (rotatable_bonds[0], rotatable_bonds[2]),
            np.array([True, False, True], dtype=bool),
        ),
    )

    def _fake_search_conformers(*, config, rotatable_bonds, **kwargs):
        captured["per_bond_lipschitz"] = config.per_bond_lipschitz
        captured["max_cells"] = config.max_cells
        captured["rotatable_bonds"] = rotatable_bonds
        return SimpleNamespace(
            conformer_coords=(),
            conformer_energies=(),
            theorem_handles=(),
        )

    def _fake_search_support_grid(*, config, rotatable_bonds, **kwargs):
        return _fake_search_conformers(
            config=config,
            rotatable_bonds=rotatable_bonds,
            **kwargs,
        )

    monkeypatch.setattr(docking_pipeline, "search_conformers", _fake_search_conformers)
    monkeypatch.setattr(
        docking_pipeline,
        "search_conformers_support_grid",
        _fake_search_support_grid,
    )

    _run_conformer_search_for_pose(
        request,
        quaternion=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        translation=jnp.zeros((3,), dtype=jnp.float32),
        scoring_context=None,
        electrostatics=None,
        rotatable_bonds=rotatable_bonds,
        conformer_config=BranchAndBoundConfig(
            max_cells=999999,
            min_cell_radius=0.1,
            score_lipschitz_constant=1.0,
            max_conformers=1,
            per_bond_lipschitz=(1.0, 2.0, 3.0),
        ),
    )

    assert captured["per_bond_lipschitz"] == (1.0, 3.0)
    assert cast(int, captured["max_cells"]) != 999999
    assert len(cast(tuple[RotatableBond, ...], captured["rotatable_bonds"])) == 2


def test_run_conformer_search_wires_cellwise_omission_hooks(monkeypatch) -> None:
    request = PipelineDockingRequest(
        protein_coords=jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        receptor_radii=jnp.ones((3,), dtype=jnp.float32),
        ligand_ctx=_flexible_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )
    rotatable_bonds = (
        RotatableBond(
            atom_i=0,
            atom_j=1,
            rotating_atom_indices=(1, 2),
            max_arm_length=1.0,
        ),
        RotatableBond(
            atom_i=1,
            atom_j=2,
            rotating_atom_indices=(2, 3),
            max_arm_length=1.0,
        ),
    )
    captured: dict[str, object] = {}

    class _FakeScoringContext:
        receptor_conformations = None

        def receptor_subset(self, indices):
            return self

    def _fake_build_conformer_score_fns(*args, receptor_coords=None, **kwargs):
        assert receptor_coords is not None
        return (lambda coords: float(receptor_coords.shape[0])), None

    monkeypatch.setattr(
        docking_pipeline,
        "_active_rotatable_bonds_for_pose",
        lambda *args, **kwargs: (rotatable_bonds, np.array([True, True], dtype=bool)),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_select_receptor_subset_for_conformer_family",
        lambda **kwargs: jnp.array([0, 1, 2], dtype=jnp.int32),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_per_receptor_atom_conformer_omission_bounds",
        lambda **kwargs: np.linspace(
            0.3,
            0.1,
            int(kwargs["receptor_coords"].shape[0]),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_select_receptor_subset_by_omission_budget",
        lambda omission_bounds, omission_budget: (
            (
                jnp.array([int(np.argmin(omission_bounds))], dtype=jnp.int32),
                float(np.sum(omission_bounds) - np.min(omission_bounds)),
            )
            if omission_bounds.shape[0] > 1
            else (jnp.array([0], dtype=jnp.int32), 0.0)
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_build_cell_local_activity_mask_fn",
        lambda **kwargs: lambda cell, center: np.array([True, False], dtype=bool),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_build_conformer_score_fns",
        _fake_build_conformer_score_fns,
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_build_conformer_coarse_score_fns",
        lambda *args, **kwargs: (lambda coords: 0.0, None),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_conformer_coverage_plan_from_lipschitz",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_restrict_conformer_search_config",
        lambda request, config, active_mask, bonds: config,
    )

    def _fake_search_conformers(**kwargs):
        cell_fn = kwargs["cell_bound_state_fn"]
        child_fn = kwargs["child_cell_bound_state_fn"]
        assert cell_fn is not None
        assert child_fn is not None
        root_cell = TorsionCell(
            lower=jnp.full((len(rotatable_bonds),), -jnp.pi, dtype=jnp.float32),
            upper=jnp.full((len(rotatable_bonds),), jnp.pi, dtype=jnp.float32),
        )
        root_state = cell_fn(root_cell, np.asarray(request.ligand_ctx.base_coords))
        child_state = child_fn(root_state, root_cell.subdivide()[0])
        captured["root_subset"] = int(root_state.payload.retained_indices.shape[0])
        captured["child_subset"] = int(child_state.payload.retained_indices.shape[0])
        captured["root_omitted"] = root_state.omitted_energy_bound
        captured["child_omitted"] = child_state.omitted_energy_bound
        captured["root_score_is_exact"] = root_state.score_is_exact
        captured["child_score_is_exact"] = child_state.score_is_exact
        captured["pruning_incumbent_energy"] = kwargs["pruning_incumbent_energy"]
        return SimpleNamespace(
            conformer_coords=(),
            conformer_energies=(),
            theorem_handles=(),
        )

    monkeypatch.setattr(docking_pipeline, "search_conformers", _fake_search_conformers)

    result = _run_conformer_search_for_pose(
        request,
        quaternion=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        translation=jnp.zeros((3,), dtype=jnp.float32),
        scoring_context=cast(Any, _FakeScoringContext()),
        electrostatics=None,
        rotatable_bonds=rotatable_bonds,
        conformer_config=BranchAndBoundConfig(
            max_cells=16,
            min_cell_radius=0.1,
            score_lipschitz_constant=1.0,
            max_conformers=1,
            per_bond_lipschitz=(1.0, 1.0),
        ),
        pruning_incumbent_energy=5.0,
        omission_budget=0.25,
    )

    assert result is None
    assert captured["root_subset"] == 1
    assert captured["child_subset"] == 1
    assert cast(float, captured["root_omitted"]) > 0.0
    assert cast(float, captured["child_omitted"]) >= cast(
        float, captured["root_omitted"]
    )
    assert captured["root_score_is_exact"] is False
    assert captured["child_score_is_exact"] is False
    assert captured["pruning_incumbent_energy"] == pytest.approx(5.0)


def test_certified_mode_rejects_non_optimized_outputs() -> None:
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
        optimize=False,
    )

    with pytest.raises(ValueError, match="requires optimize=True"):
        run_docking_pipeline_request(request)


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


def test_conformer_coverage_plan_owns_branch_and_bound_budget() -> None:
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
    bonds = (
        RotatableBond(
            atom_i=0,
            atom_j=1,
            rotating_atom_indices=(1,),
            max_arm_length=1.5,
        ),
    )

    coverage_plan = _derive_conformer_coverage_plan(request, rotatable_bonds=bonds)
    config = cast(
        BranchAndBoundConfig,
        coverage_plan.as_branch_and_bound_config(
            reuse_initial_conformer=False,
            max_conformers=1,
        ),
    )

    assert coverage_plan.n_torsions == 1
    assert coverage_plan.max_cells == config.max_cells
    assert coverage_plan.min_cell_radius == pytest.approx(config.min_cell_radius)
    assert coverage_plan.per_bond_lipschitz == config.per_bond_lipschitz
    assert coverage_plan.canonical_segments


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
    np.testing.assert_allclose(
        np.asarray(bounds),
        np.asarray([3.0 + 2.0 * np.pi * 2.0], dtype=np.float32),
    )


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
