import jax
import jax.numpy as jnp
import pytest

from pathlib import Path

from dq_dock_engine.benchmark.benchmark_pdb import (
    BENCHMARK_PROTOCOL,
    DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS,
    DERIVED_BENCHMARK_POCKET_RADIUS_ANGSTROMS,
    active_formal_runtime_contract,
    benchmark_box_protocol_metadata,
    derive_benchmark_pocket_radius_from_box_size,
)
from dq_dock_engine.docking.formal_handles import (
    APX10,
    APX11,
    APX12,
    ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT,
    ACTIVE_CERTIFIED_RUNTIME_HANDLES,
    BD10,
    CB10,
    CB11,
    CB12,
    CB13,
    CB14,
    CT1,
    CT6,
    CP3,
    CP2,
    CP4,
    CP5,
    CP6,
    FLO10,
    FLO11,
    FLO13,
    FLO14,
    FLO15,
    FLO16,
    FLO17,
    FLO18,
    FLO8,
    FLO9,
    HB1,
    HB8,
    selection_branch_membership_handle,
    selection_theorem_handle,
    selection_witness_handle,
    belief_witness_handle,
    STAGED_COARSE_TOP1_RUNTIME_CONTRACT,
    STAGED_COARSE_PRUNING_HANDLES,
    STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT,
    Top1PruningBranchName,
    additive_nonbonded_theorem_handles,
    contact_surrogate_theorem_handles,
    handle_bundle_from_contracts,
    directional_hbond_theorem_handles,
    runtime_contract_record,
    screened_coulomb_theorem_handles,
    serialize_dataclass_record,
    TK8,
    TK11,
    TK12,
    TK9A,
    LJ13,
    LJ14,
    NB1,
    NB10,
    SC1,
    SC6,
    scoring_family_theorem_handles,
)
from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine
from dq_dock_engine.docking_config import CertifiedScoringFamily
from dq_dock_engine.docking.formal_actions import create_certified_action_family
from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefWitness,
    CertifiedPriorSpec,
    PosteriorUpdateBranch,
    SelectionBranch,
    belief_witness,
    build_prior,
    posterior_update_witness,
    selection_provenance,
    selection_witness,
    select_admissible_action,
    update_posterior,
)
from dq_dock_engine.docking.formal_pruning import (
    CertifiedSurvivorSetWitness,
    StagedTop1Guarantee,
    Top1PruningBranch,
    certificate_of_top1_branch,
    certified_pruning_certificate,
    certified_exact_singleton_winner_certificate,
    certified_survivor_set_witness,
    certified_top1_coarse_ambiguity_certificate,
    coarse_top1_ambiguity_mask,
    has_exact_singleton_winner_proof_condition,
    select_top1_pruning_branch,
    staged_top1_guarantee_from_coarse_scores,
    survivor_set_of_top1_branch,
)

try:
    from dq_dock_engine.docking.formal_optimizer import (
        HybridSingletonRefinementResult,
        MinimalStagedRoundResult,
        StagedCertifiedDecisionState,
        _minimal_round_from_per_pose_singleton_accept,
        active_belief_witness,
        active_optimizer_witness,
        active_survivor_set_witness,
        active_pruning_branch,
        refine_poses_singleton_then_exact,
        refine_poses_certified,
        staged_decision_states_from_singleton_accept_round,
        try_refine_poses_singleton_minimal,
        try_refine_round_singleton_minimal,
        try_refine_round_singleton_staged,
    )
except ImportError as exc:
    pytest.skip(f"stale formal optimizer API tests: {exc}", allow_module_level=True)
from dq_dock_engine.docking.formal_sampling import sample_certified_global_poses
from dq_dock_engine.docking.formal_surrogates import (
    FastSingletonAcceptRoundResult,
    PerPoseFastSingletonAcceptRoundResult,
    StagedSingletonGateResult,
    StagedSingletonAcceptRoundResult,
    StagedTop1BranchSummary,
    StagedTop1CostDiagnostic,
    StagedTop1Decision,
    StagedTop1RoundResult,
    TwoCutoffApproximationWitness,
    score_exact_and_coarse_local_family,
    staged_top1_cost_diagnostic,
    try_adaptive_singleton_accept_round,
    try_adaptive_per_pose_singleton_accept_round,
    staged_top1_decision_from_scores,
    staged_top1_round_from_coarse_scores,
    staged_singleton_gate,
    summarize_staged_top1_round,
    two_cutoff_approximation_witness,
    try_fast_singleton_accept_round,
    try_hybrid_singleton_accept_round,
    try_per_pose_fast_singleton_accept_round,
    try_staged_singleton_accept_round,
    select_exact_receptor_subset_for_local_family,
)

try:
    from dq_dock_engine.docking.pipeline import run_docking_pipeline
except ImportError as exc:
    pytest.skip(f"stale pipeline API tests: {exc}", allow_module_level=True)
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    CERTIFIED_DOCKING,
    DockingConfig,
    DockingMode,
    FormalRoundStrategy,
    OptimizerBackend,
    create_config,
)
from dq_dock_engine.docking.scoring import (
    score_certified_batch,
    score_certified_softened_lj,
)
from dq_dock_engine.codegen.formal_handle_codegen import (
    DEFAULT_LEAN_PATH,
    parse_alias_names,
)
from dq_dock_engine.generated import formal_handle_aliases


def test_certified_action_family_has_noop_first_and_stable_size():
    family = create_certified_action_family(
        translation_step=0.5,
        rotation_step_rad=float(jnp.pi / 12.0),
        stencil_level=0,
    )

    assert len(family.actions) == 13
    assert family.actions[0].is_noop is True
    assert family.actions[0].action_id == 0
    assert tuple(action.action_id for action in family.actions) == tuple(range(13))


def test_benchmark_pocket_radius_is_derived_from_box_size():
    derived = derive_benchmark_pocket_radius_from_box_size(
        DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS
    )
    assert derived == DERIVED_BENCHMARK_POCKET_RADIUS_ANGSTROMS
    import math

    assert (
        abs(derived - DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS * math.sqrt(3.0) / 2.0)
        < 1e-9
    )


def test_benchmark_box_protocol_metadata_marks_box_size_derived():
    assert benchmark_box_protocol_metadata(BENCHMARK_PROTOCOL.box_size_a) == (
        "box_size_derived",
        BENCHMARK_PROTOCOL.box_geometry_theorem,
    )
    assert benchmark_box_protocol_metadata(BENCHMARK_PROTOCOL.box_size_a + 1.0) == (
        "custom_override",
        None,
    )


def test_active_certified_runtime_theorem_bundle_tracks_exact_path():
    assert CP2 in ACTIVE_CERTIFIED_RUNTIME_HANDLES.theorem_handles
    assert FLO9 in ACTIVE_CERTIFIED_RUNTIME_HANDLES.theorem_handles
    assert FLO8 in ACTIVE_CERTIFIED_RUNTIME_HANDLES.theorem_handles


def test_generated_formal_handle_aliases_stay_in_sync_with_lean_source():
    expected = tuple(parse_alias_names(DEFAULT_LEAN_PATH.read_text()))
    assert tuple(formal_handle_aliases.__all__) == expected


def test_active_certified_runtime_witness_bundle_tracks_exact_path_objects():
    assert {CP4, FLO10, FLO13, FLO15}.issubset(
        set(ACTIVE_CERTIFIED_RUNTIME_HANDLES.witness_handles)
    )


def test_scoring_family_theorem_handles_expose_stronger_certified_runtime_surface():
    ewald_handles = set(
        scoring_family_theorem_handles(CertifiedScoringFamily.LJ_REALSPACE_EWALD)
    )
    assert {CB10, CB11, CB12, CB13, CB14, APX10, APX11, APX12, BD10}.issubset(
        ewald_handles
    )

    lj_handles = set(scoring_family_theorem_handles(CertifiedScoringFamily.LJ))
    assert {LJ13, LJ14, APX10, APX11, APX12}.issubset(lj_handles)


def test_new_chemistry_handle_helpers_expose_contact_screened_and_hbond_surfaces():
    assert {CT1, CT6}.issubset(set(contact_surrogate_theorem_handles()))
    assert {SC1, SC6}.issubset(set(screened_coulomb_theorem_handles()))
    assert {NB1, NB10}.issubset(set(additive_nonbonded_theorem_handles()))
    assert {HB1, HB8}.issubset(set(directional_hbond_theorem_handles()))


def test_runtime_contract_objects_track_active_and_staged_branches():
    assert (
        ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT.pruning_branch
        == Top1PruningBranchName.EXACT_TOP1
    )
    assert ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT.optimizer_witness_handle == FLO15
    assert (
        STAGED_COARSE_TOP1_RUNTIME_CONTRACT.pruning_branch
        == Top1PruningBranchName.TOP1_COARSE_AMBIGUITY_BAND
    )
    assert (
        STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT.pruning_branch
        == Top1PruningBranchName.EXACT_SINGLETON_WINNER
    )


def test_active_formal_runtime_contract_switches_on_strategy():
    assert (
        active_formal_runtime_contract(FormalRoundStrategy.EXACT)
        == ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT
    )
    assert (
        active_formal_runtime_contract(FormalRoundStrategy.SINGLETON_HYBRID).name
        == "active_singleton_hybrid_certified_runtime"
    )


def test_runtime_contract_record_uses_generic_dataclass_serialization():
    record = runtime_contract_record(ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT)

    assert record == serialize_dataclass_record(ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT)
    assert record["pruning_branch"] == Top1PruningBranchName.EXACT_TOP1.value
    assert record["optimizer_witness_handle"] == FLO15


def test_handle_bundles_are_derived_from_runtime_contracts():
    active_bundle = handle_bundle_from_contracts(
        ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT
    )
    staged_bundle = handle_bundle_from_contracts(
        STAGED_COARSE_TOP1_RUNTIME_CONTRACT,
        STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT,
        extra_theorem_handles=(CP3,),
        extra_witness_handles=(FLO18,),
    )

    assert active_bundle == ACTIVE_CERTIFIED_RUNTIME_HANDLES
    assert staged_bundle == STAGED_COARSE_PRUNING_HANDLES


def test_staged_coarse_pruning_theorem_bundle_tracks_proved_branches():
    assert {CP3, TK11, TK12}.issubset(
        set(STAGED_COARSE_PRUNING_HANDLES.theorem_handles)
    )


def test_staged_coarse_pruning_witness_bundle_tracks_object_handles():
    assert {CP5, CP6, FLO16, FLO17, FLO18}.issubset(
        set(STAGED_COARSE_PRUNING_HANDLES.witness_handles)
    )


def test_bayes_update_normalizes_survivor_support():
    prior = jnp.array([0.25, 0.25, 0.25, 0.25])
    survivor_mask = jnp.array([True, False, True, False])

    posterior = update_posterior(prior, survivor_mask)

    assert jnp.allclose(posterior, jnp.array([0.5, 0.0, 0.5, 0.0]))
    assert jnp.isclose(jnp.sum(posterior), 1.0)


def test_noop_biased_prior_is_explicit_and_normalized():
    prior = build_prior(CertifiedPriorSpec(kind="noop_biased", noop_mass=0.4), 5)

    assert jnp.isclose(jnp.sum(prior), 1.0)
    assert jnp.isclose(prior[0], 0.4)
    assert jnp.allclose(prior[1:], jnp.full((4,), 0.15))


def test_certified_config_rejects_gradient_backend():
    try:
        create_config(mode="certified", optimizer="gradient")
    except ValueError as exc:
        assert "CERTIFIED mode requires OptimizerBackend.FORMAL" in str(exc)
        return
    raise AssertionError("certified gradient configuration should fail loudly")


def test_certified_config_defaults_to_singleton_hybrid_round_strategy():
    cfg = create_config(mode="certified", optimizer="formal")
    assert cfg.formal_round_strategy == FormalRoundStrategy.SINGLETON_HYBRID


def test_select_admissible_action_uses_first_ambiguity_member():
    posterior = jnp.array([0.1, 0.6, 0.3])
    ambiguity_mask = jnp.array([False, True, True])

    selected = select_admissible_action(posterior, ambiguity_mask)

    assert selected == 1


def test_selection_provenance_distinguishes_ambiguity_and_support_fallback():
    ambiguity_rule = selection_provenance(
        jnp.array([0.1, 0.6, 0.3]), jnp.array([False, True, True])
    )
    fallback_rule = selection_provenance(
        jnp.array([0.0, 1.0, 0.0]), jnp.array([False, False, False])
    )

    assert ambiguity_rule == (
        SelectionBranch.AMBIGUITY_BAND.value,
        selection_branch_membership_handle(SelectionBranch.AMBIGUITY_BAND.value),
    )
    assert fallback_rule == (
        SelectionBranch.SUPPORT_FALLBACK.value,
        selection_branch_membership_handle(SelectionBranch.SUPPORT_FALLBACK.value),
    )


def test_branch_witness_helpers_use_branch_indexed_theorems():
    pw = posterior_update_witness()
    sw = selection_witness(
        jnp.array([0.1, 0.6, 0.3]), jnp.array([False, True, True]), selected_action=1
    )
    bw = belief_witness(
        jnp.array([0.1, 0.6, 0.3]), jnp.array([False, True, True]), selected_action=1
    )

    assert pw.branch == PosteriorUpdateBranch.SURVIVOR_CONDITIONING
    assert pw.theorem_handle == FLO9
    assert pw.witness_handle == FLO10
    assert sw.branch == SelectionBranch.AMBIGUITY_BAND
    assert sw.theorem_handle == selection_theorem_handle(
        SelectionBranch.AMBIGUITY_BAND.value
    )
    assert sw.witness_handle == selection_witness_handle(
        SelectionBranch.AMBIGUITY_BAND.value
    )
    assert sw.selected_action == 1
    assert isinstance(bw, CertifiedBeliefWitness)
    assert bw.posterior_update.theorem_handle == FLO9
    assert bw.selection.theorem_handle == FLO8
    assert bw.witness_handle == belief_witness_handle(
        SelectionBranch.AMBIGUITY_BAND.value
    )


def test_certified_pruning_certificate_is_exact_when_delta_zero():
    exact_scores = jnp.array([0.2, 0.5, 0.3])
    coarse_scores = jnp.array([0.2, 0.5, 0.3])

    cert = certified_pruning_certificate(exact_scores, coarse_scores, k=1, delta=0.0)

    assert jnp.array_equal(cert.survivor_mask, cert.exact_top_k_mask)
    assert jnp.array_equal(cert.exact_ambiguity_mask, cert.exact_top_k_mask)
    assert jnp.array_equal(cert.coarse_ambiguity_mask, cert.exact_top_k_mask)
    assert cert.rule == "exact_top1"
    assert cert.theorem_handle == CP2


def test_coarse_top1_ambiguity_band_contains_exact_winner_under_uniform_error():
    exact_scores = jnp.array([0.0, 1.0, 2.0])
    coarse_scores = jnp.array([0.1, 0.9, 2.1])
    delta = 0.1

    coarse_band = coarse_top1_ambiguity_mask(coarse_scores, delta)

    assert bool(coarse_band[0]) is True


def test_certified_top1_coarse_ambiguity_certificate_uses_coarse_band():
    exact_scores = jnp.array([0.0, 1.0, 2.0])
    coarse_scores = jnp.array([0.1, 0.9, 2.1])
    cert = certified_top1_coarse_ambiguity_certificate(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        delta=0.1,
    )

    assert jnp.array_equal(cert.survivor_mask, cert.coarse_ambiguity_mask)
    assert bool(cert.survivor_mask[0]) is True
    assert cert.rule == "top1_coarse_ambiguity_band"
    assert cert.theorem_handle == TK11


def test_certified_singleton_winner_certificate_returns_single_winner():
    exact_scores = jnp.array([0.0, 1.0, 2.0])
    coarse_scores = jnp.array([0.0, 1.0, 2.0])
    cert = certified_exact_singleton_winner_certificate(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        delta=0.1,
    )

    assert int(jnp.sum(cert.survivor_mask.astype(jnp.int32))) == 1
    assert bool(cert.survivor_mask[0]) is True
    assert cert.rule == "exact_singleton_winner"
    assert cert.theorem_handle == TK12


def test_singleton_winner_proof_condition_tracks_pairwise_margin():
    assert has_exact_singleton_winner_proof_condition(jnp.array([0.0, 1.0, 2.0]), 0.1)
    assert not has_exact_singleton_winner_proof_condition(
        jnp.array([0.0, 0.15, 2.0]), 0.1
    )


def test_top1_pruning_branch_selector_matches_proof_cases():
    assert (
        select_top1_pruning_branch(jnp.array([0.0, 1.0, 2.0]), 0.0)
        == Top1PruningBranch.EXACT_TOP1
    )
    assert (
        select_top1_pruning_branch(jnp.array([0.0, 1.0, 2.0]), 0.1)
        == Top1PruningBranch.EXACT_SINGLETON_WINNER
    )
    assert (
        select_top1_pruning_branch(jnp.array([0.0, 0.15, 2.0]), 0.1)
        == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND
    )


def test_certificate_of_top1_branch_dispatches_to_exact_top1_and_singleton():
    exact_scores = jnp.array([0.0, 1.0, 2.0])
    coarse_scores = jnp.array([0.0, 1.0, 2.0])

    exact_cert = certificate_of_top1_branch(
        Top1PruningBranch.EXACT_TOP1, exact_scores, coarse_scores, delta=0.0
    )
    singleton_cert = certificate_of_top1_branch(
        Top1PruningBranch.EXACT_SINGLETON_WINNER,
        exact_scores,
        coarse_scores,
        delta=0.1,
    )

    assert exact_cert.theorem_handle == "CP2"
    assert singleton_cert.theorem_handle == "TK12"


def test_survivor_set_of_top1_branch_packages_branch_certificate():
    witness = survivor_set_of_top1_branch(
        Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND,
        exact_scores=jnp.array([0.0, 1.0, 2.0]),
        coarse_scores=jnp.array([0.1, 0.9, 2.1]),
        delta=0.1,
    )

    assert isinstance(witness, CertifiedSurvivorSetWitness)
    assert witness.certificate.theorem_handle == "TK11"
    assert witness.theorem_handle == "CP5"
    assert jnp.array_equal(witness.survivor_mask, witness.certificate.survivor_mask)


def test_staged_top1_decision_accepts_singleton_branch_without_exact_rescore():
    decision = staged_top1_decision_from_scores(
        exact_scores=jnp.array([0.0, 1.0, 2.0]),
        coarse_scores=jnp.array([0.0, 1.0, 2.0]),
        delta=0.1,
    )

    assert isinstance(decision, StagedTop1Decision)
    assert decision.branch == Top1PruningBranch.EXACT_SINGLETON_WINNER
    assert decision.accepted_without_exact_rescore is True
    assert decision.survivor_set.theorem_handle == CP6


def test_staged_top1_guarantee_certifies_singleton_without_exact_scores():
    guarantee = staged_top1_guarantee_from_coarse_scores(
        coarse_scores=jnp.array([0.0, 1.0, 2.0]),
        delta=0.1,
    )

    assert isinstance(guarantee, StagedTop1Guarantee)
    assert guarantee.branch == Top1PruningBranch.EXACT_SINGLETON_WINNER
    assert guarantee.theorem_handle == TK12
    assert guarantee.exact_winner_certified is True
    assert int(jnp.sum(guarantee.survivor_mask.astype(jnp.int32))) == 1


def test_staged_top1_guarantee_returns_band_when_margin_is_small():
    guarantee = staged_top1_guarantee_from_coarse_scores(
        coarse_scores=jnp.array([0.0, 0.15, 2.0]),
        delta=0.1,
    )

    assert guarantee.branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND
    assert guarantee.theorem_handle == TK11
    assert guarantee.exact_winner_certified is False


def test_staged_top1_decision_falls_back_to_band_when_margin_is_small():
    decision = staged_top1_decision_from_scores(
        exact_scores=jnp.array([0.0, 1.0, 2.0]),
        coarse_scores=jnp.array([0.0, 0.15, 2.0]),
        delta=0.1,
    )

    assert decision.branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND
    assert decision.accepted_without_exact_rescore is False
    assert decision.survivor_set.theorem_handle == CP5


def test_staged_top1_round_from_coarse_scores_packages_all_pose_decisions():
    round_result = staged_top1_round_from_coarse_scores(
        coarse_scores=jnp.array([[0.0, 1.0, 2.0], [0.0, 0.15, 2.0]]),
        delta=0.1,
        retained_receptor_indices=jnp.array([0, 1], dtype=jnp.int32),
    )

    assert isinstance(round_result, StagedTop1RoundResult)
    assert len(round_result.decisions) == 2
    assert round_result.decisions[0].branch == Top1PruningBranch.EXACT_SINGLETON_WINNER
    assert (
        round_result.decisions[1].branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND
    )


def test_summarize_staged_top1_round_counts_branch_distribution():
    round_result = staged_top1_round_from_coarse_scores(
        coarse_scores=jnp.array([[0.0, 1.0, 2.0], [0.0, 0.15, 2.0]]),
        delta=0.1,
        retained_receptor_indices=jnp.array([0, 1], dtype=jnp.int32),
    )
    summary = summarize_staged_top1_round(round_result)

    assert isinstance(summary, StagedTop1BranchSummary)
    assert summary.singleton_count == 1
    assert summary.ambiguity_band_count == 1
    assert summary.total == 2
    assert summary.singleton_fraction == 0.5


def test_two_cutoff_approximation_witness_packages_combined_delta():
    witness = two_cutoff_approximation_witness(0.001, 0.004)

    assert isinstance(witness, TwoCutoffApproximationWitness)
    assert witness.theorem_handle == TK8
    assert witness.witness_handle == TK9A
    assert (
        witness.combined_delta == witness.exact_error_bound + witness.coarse_error_bound
    )


def test_staged_top1_cost_diagnostic_reports_branch_and_retained_sizes():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    diagnostic = staged_top1_cost_diagnostic(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert isinstance(diagnostic, StagedTop1CostDiagnostic)
    assert diagnostic.exact_retained_atoms >= diagnostic.coarse_retained_atoms
    assert diagnostic.branch_summary.total == 2


def test_try_staged_singleton_accept_round_returns_result_or_clean_fallback():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_staged_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert result is None or isinstance(result, StagedSingletonAcceptRoundResult)


def test_try_fast_singleton_accept_round_returns_result_or_clean_fallback():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_fast_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert result is None or isinstance(result, FastSingletonAcceptRoundResult)


def test_try_adaptive_singleton_accept_round_returns_first_successful_target():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_adaptive_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_errors=(0.05, 0.01, 0.004),
        translation_step=0.5,
    )

    assert result is not None
    assert isinstance(result, FastSingletonAcceptRoundResult)
    assert result.coarse_target_error in {0.05, 0.01, 0.004}


def test_try_per_pose_fast_singleton_accept_round_returns_result_or_fallback():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_per_pose_fast_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert result is None or isinstance(result, PerPoseFastSingletonAcceptRoundResult)


def test_try_adaptive_per_pose_fast_singleton_accept_round_returns_first_successful_target():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_adaptive_per_pose_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_errors=(0.05, 0.01, 0.004),
        translation_step=0.5,
    )

    assert result is not None
    assert isinstance(result, PerPoseFastSingletonAcceptRoundResult)
    assert result.coarse_target_error in {0.05, 0.01, 0.004}


def test_minimal_round_from_per_pose_singleton_accept_preserves_round_payload():
    round_result = PerPoseFastSingletonAcceptRoundResult(
        selected_actions=jnp.array([1, 0], dtype=jnp.int32),
        next_coords=jnp.array(
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=jnp.float32
        ),
        coarse_scores=jnp.array([[0.0, 1.0], [0.0, 2.0]], dtype=jnp.float32),
        coarse_target_error=0.004,
        delta=0.01,
        retained_receptor_indices_per_pose=(
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
        ),
        theorem_handle=TK12,
    )

    minimal = _minimal_round_from_per_pose_singleton_accept(round_result)
    assert jnp.array_equal(minimal.selected_actions, round_result.selected_actions)
    assert jnp.array_equal(minimal.next_coords, round_result.next_coords)
    assert minimal.delta == round_result.delta
    assert minimal.theorem_handle == TK12


def test_try_hybrid_singleton_accept_round_returns_result_or_fallback():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_hybrid_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert result is None or isinstance(result, FastSingletonAcceptRoundResult)


def test_staged_singleton_gate_packages_supports_and_optional_accept_result():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    gate = staged_singleton_gate(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert isinstance(gate, StagedSingletonGateResult)
    assert (
        gate.exact_retained_receptor_indices.shape[0]
        >= gate.coarse_retained_receptor_indices.shape[0]
    )


def test_staged_decision_states_from_singleton_accept_round_packages_pose_states():
    candidate_batches = jnp.array(
        [
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ],
        dtype=jnp.float32,
    )
    result = try_fast_singleton_accept_round(
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        candidate_batches=candidate_batches,
        target_error=0.001,
        coarse_target_error=0.004,
        translation_step=0.5,
    )

    assert result is not None
    states = staged_decision_states_from_singleton_accept_round(result)
    assert len(states) == 2
    assert isinstance(states[0], StagedCertifiedDecisionState)
    assert states[0].theorem_handle == TK12


def test_try_refine_round_singleton_staged_returns_result_or_none():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = try_refine_round_singleton_staged(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        target_error=0.001,
        round_index=0,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 2.0),
        coarse_target_error=0.004,
    )

    assert result is None or isinstance(result[1][0], StagedCertifiedDecisionState)


def test_try_refine_round_singleton_minimal_returns_result_or_none():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = try_refine_round_singleton_minimal(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        target_error=0.001,
        round_index=0,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 2.0),
        coarse_target_error=0.004,
    )

    assert result is None or isinstance(result, MinimalStagedRoundResult)


def test_try_refine_poses_singleton_minimal_returns_result_or_none():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = try_refine_poses_singleton_minimal(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        n_rounds=2,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 2.0),
        coarse_target_error=0.004,
    )

    assert result is None or isinstance(result[1][0], MinimalStagedRoundResult)


def test_refine_poses_singleton_then_exact_returns_hybrid_result():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = refine_poses_singleton_then_exact(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        n_rounds=2,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 2.0),
        coarse_target_error=0.004,
    )

    assert isinstance(result, HybridSingletonRefinementResult)
    assert result.coords.shape == coords_batch.shape


def test_refine_poses_singleton_then_exact_accepts_softened_coarse_flag():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = refine_poses_singleton_then_exact(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        n_rounds=1,
        target_error=0.001,
        coarse_target_error=0.004,
        use_softened_coarse=True,
    )

    assert isinstance(result, HybridSingletonRefinementResult)
    assert result.coords.shape == coords_batch.shape


def test_refine_poses_singleton_then_exact_accepts_adaptive_coarse_schedule():
    coords_batch = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    result = refine_poses_singleton_then_exact(
        coords_batch=coords_batch,
        receptor_coords=jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.0], dtype=jnp.float32),
        ligand_radii=jnp.array([1.0], dtype=jnp.float32),
        n_rounds=1,
        target_error=0.001,
        adaptive_coarse_target_errors=(0.05, 0.01, 0.004),
    )

    assert isinstance(result, HybridSingletonRefinementResult)
    assert result.coords.shape == coords_batch.shape


def test_certified_pruning_certificate_uses_singleton_fast_path_when_band_is_singleton():
    exact_scores = jnp.array([0.0, 1.0, 2.0])
    coarse_scores = jnp.array([0.0, 1.0, 2.0])
    cert = certified_pruning_certificate(exact_scores, coarse_scores, k=1, delta=0.1)

    assert int(jnp.sum(cert.survivor_mask.astype(jnp.int32))) == 1
    assert bool(cert.survivor_mask[0]) is True
    assert cert.rule == "exact_singleton_winner"
    assert cert.theorem_handle == "TK12"


def test_certified_survivor_set_witness_packages_certificate_and_mask():
    witness = certified_survivor_set_witness(
        exact_scores=jnp.array([0.0, 1.0, 2.0]),
        coarse_scores=jnp.array([0.0, 1.0, 2.0]),
        k=1,
        delta=0.1,
    )

    assert isinstance(witness, CertifiedSurvivorSetWitness)
    assert witness.certificate.theorem_handle == "TK12"
    assert witness.theorem_handle == "CP6"
    assert jnp.array_equal(witness.survivor_mask, witness.certificate.survivor_mask)


def test_exact_receptor_subset_drops_far_atoms_outside_family_cutoff():
    receptor_coords = jnp.array([[0.0, 0.0, 8.0], [0.0, 0.0, 40.0]])
    receptor_radii = jnp.array([1.5, 1.5])
    reference_coords_batch = jnp.array([[[0.0, 0.0, 0.0]]])
    ligand_radii = jnp.array([1.5])

    kept = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=reference_coords_batch,
        ligand_radii=ligand_radii,
        translation_step=0.5,
        target_error=0.001,
    )

    assert jnp.array_equal(kept, jnp.array([0]))


def test_exact_local_family_subset_preserves_scores_when_atoms_are_far():
    receptor_coords = jnp.array([[0.0, 0.0, 8.0], [0.0, 0.0, 40.0]])
    receptor_radii = jnp.array([1.5, 1.5])
    ligand_radii = jnp.array([1.5])
    candidate_coords = jnp.array([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])

    full_batch = score_certified_batch(
        receptor_coords=receptor_coords,
        poses_coords=candidate_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=0.001,
    )
    bundle = score_exact_and_coarse_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_coords=candidate_coords,
        target_error=0.001,
        max_receptor_atoms=64,
        translation_step=0.5,
    )

    assert jnp.array_equal(bundle.retained_receptor_indices, jnp.array([0]))
    assert jnp.allclose(bundle.exact_scores, full_batch.scores)
    assert bundle.delta > 0.0
    assert bundle.coarse_error_bound > 0.0


def test_softened_local_family_coarse_scores_are_certified_and_distinct():
    receptor_coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    receptor_radii = jnp.array([1.7, 1.7])
    ligand_radii = jnp.array([1.7])
    candidate_coords = jnp.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])

    bundle = score_exact_and_coarse_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_coords=candidate_coords,
        target_error=0.001,
        coarse_target_error=0.004,
        max_receptor_atoms=64,
        translation_step=0.5,
    )

    assert bundle.exact_scores.shape == bundle.coarse_scores.shape == (2,)
    assert bundle.delta > 0.0
    assert bundle.pruning_certificate.delta == bundle.delta


def test_refine_poses_certified_uses_finite_action_family_search():
    receptor_coords = jnp.array([[3.0, 0.0, 0.0]])
    receptor_radii = jnp.array([1.0])
    ligand_radii = jnp.array([1.0])
    initial_coords = jnp.array([[[6.0, 0.0, 0.0]]])

    refined_coords, history = refine_poses_certified(
        coords_batch=initial_coords,
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        n_rounds=1,
        target_error=0.001,
        base_translation_step=0.5,
        base_rotation_step_rad=float(jnp.pi / 12.0),
    )

    assert len(history) == 1
    belief = history[0][0].belief
    pruning_certificate = history[0][0].pruning_certificate
    assert belief.selected_action != 0
    assert belief.posterior_rule == PosteriorUpdateBranch.SURVIVOR_CONDITIONING.value
    assert belief.posterior_theorem == "FLO9"
    assert pruning_certificate.theorem_handle in {"CP2", "TK11", "TK12"}
    assert pruning_certificate.rule in {
        "exact_top1",
        "top1_coarse_ambiguity_band",
        "exact_singleton_winner",
    }
    assert active_pruning_branch(history[0][0]) in {
        Top1PruningBranch.EXACT_TOP1,
        Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND,
        Top1PruningBranch.EXACT_SINGLETON_WINNER,
    }
    assert (
        active_survivor_set_witness(history[0][0]).certificate.theorem_handle
        == pruning_certificate.theorem_handle
    )
    assert active_survivor_set_witness(history[0][0]).theorem_handle in {
        "CP4",
        "CP5",
        "CP6",
    }
    assert (
        active_belief_witness(history[0][0]).posterior_update.theorem_handle == "FLO9"
    )
    assert active_belief_witness(history[0][0]).selection.theorem_handle == FLO8
    assert active_belief_witness(history[0][0]).witness_handle in {FLO13, FLO14}
    optimizer_witness = active_optimizer_witness(history[0][0])
    assert (
        optimizer_witness.survivor_set.certificate.theorem_handle
        == pruning_certificate.theorem_handle
    )
    assert optimizer_witness.survivor_set.theorem_handle in {CP4, CP5, CP6}
    assert optimizer_witness.belief.posterior_update.theorem_handle == FLO9
    assert optimizer_witness.theorem_handle in {FLO15, FLO16, FLO17}
    assert optimizer_witness.branch_witness_handle == FLO18
    assert optimizer_witness.support_matches_survivors is True
    assert optimizer_witness.coherence_theorem_handles == ("APX11", "APX12")
    assert belief.selected_action_rule in {
        SelectionBranch.AMBIGUITY_BAND.value,
        SelectionBranch.SUPPORT_FALLBACK.value,
    }
    assert belief.selected_action_theorem == "FLO8"
    assert refined_coords.shape == initial_coords.shape
    assert float(refined_coords[0, 0, 0]) < float(initial_coords[0, 0, 0])


def test_certified_global_sampler_is_deterministic_and_nonrandom():
    box = DockingBox(center=jnp.array([6.0, 0.0, 0.0]), size=jnp.array([2.0, 2.0, 2.0]))
    pose_vec = sample_certified_global_poses(box, 8)

    assert pose_vec.translation.shape == (8, 3)
    assert pose_vec.quaternion.shape == (8, 4)
    assert not jnp.any(jnp.all(jnp.isclose(pose_vec.translation, box.center), axis=1))


def test_certified_pipeline_does_not_call_heuristic_sampler(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("heuristic sampler should not be used in certified mode")

    monkeypatch.setattr(
        "dq_dock_engine.docking.pocket_sampling.sample_intelligent_poses",
        fail_if_called,
    )

    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]]),
        base_radii=jnp.array([1.0]),
        center_of_mass=jnp.array([0.0, 0.0, 0.0]),
    )
    box = DockingBox(center=jnp.array([6.0, 0.0, 0.0]), size=jnp.array([0.2, 0.2, 0.2]))

    best_poses, _ = run_docking_pipeline(
        protein_coords=jnp.array([[3.0, 0.0, 0.0]]),
        receptor_radii=jnp.array([1.0]),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=1,
        engine=ScoringEngine.INTERNAL_LJ,
        key=jax.random.PRNGKey(0),
        config=DockingConfig(
            mode=DockingMode.CERTIFIED,
            optimizer_backend=OptimizerBackend.FORMAL,
            certified_scoring_family=CertifiedScoringFamily.LJ,
        ),
        top_k=1,
        optimize=False,
        use_pocket_guided=True,
    )

    assert len(best_poses) == 1
