import Ssot.Basic
import Ssot.Bounds
import Ssot.ClaimClosure
import Ssot.Coherence
import Ssot.Completeness
import Ssot.CrossPaperDependencies  -- Bridge theorems linking Paper 2 → Paper 1
import Ssot.Foundations
import Ssot.Inconsistency
import Ssot.LangPython
import Ssot.ObserverModel
import Ssot.EntropyGeneral
import Ssot.Probabilistic
import Ssot.Requirements
import axis_framework

open Ssot

/-- Stable, semantic Lean-handle IDs for Paper 2 (IT).

    The build system parses `abbrev CODE := target` and uses these IDs in
    `lean_handle_ids_auto.tex`, which then powers `\LH{CODE}` links in the PDF. -/

noncomputable abbrev ENT1 := ClaimClosure.sideInfoBits
abbrev ENT2 := ObserverModel.binEntropy_half
abbrev ENT3 := @ObserverModel.klDiv_self_eq_zero
abbrev ENT4 := @ObserverModel.pmfEntropy_bernoulli
abbrev ENT5 := @ObserverModel.pmfEntropy_bernoulli_le_log_two
abbrev ENT6 := @ObserverModel.pmfEntropy_uniform_fin
abbrev ENT7 := @ObserverModel.pmf_eq_uniform_fin_of_klDiv_zero
abbrev ENT8 := @ObserverModel.pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin
abbrev ENT9 := @ObserverModel.pmfEntropy_bernoulli_ofReal
abbrev ENT10 := @ObserverModel.pmfEntropy_inducedPairPMF_eq_observationTagEntropy
abbrev ENT11 := @ObserverModel.pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy
abbrev ENT12 := @ObserverModel.pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin_nonempty
abbrev CIA1 := ClaimClosure.side_information_requirement
abbrev CIA2 := ClaimClosure.side_information_requirement
abbrev CIA3 := @ClaimClosure.finite_counting_converse
abbrev CIA4 := @ClaimClosure.info_dof_counting_contradiction

abbrev ORA1 := oracle_arbitrary
abbrev COH1 := dof_one_implies_coherent
abbrev COH2 := dof_gt_one_incoherence_possible
abbrev COH3 := determinate_truth_forces_ssot

abbrev BAS1 := Ssot.correctness_forcing
abbrev BAS2 := Ssot.dof_inconsistency_potential
abbrev BAS3 := Ssot.dof_gt_one_inconsistent

abbrev INS1 := Inconsistency.ssot_is_unique_optimum
abbrev INS2 := Inconsistency.ssot_required
abbrev INS3 := Inconsistency.ssot_unique_satisfier
abbrev INS4 := Inconsistency.resolution_requires_external_choice

abbrev CAP1 := ClaimClosure.cap_encoding_zero_error
abbrev CAP2 := ssot_guarantees_coherence
abbrev CAP3 := non_ssot_permits_incoherence
abbrev FLP1 := ClaimClosure.static_flp_core
abbrev RED1 := ClaimClosure.redundancy_incoherence_equiv
abbrev RAT1 := ClaimClosure.rate_incoherence_step
abbrev DES1 := ClaimClosure.design_necessity
abbrev SID1 := ClaimClosure.dof1_zero_side_information
abbrev SID2 := ClaimClosure.side_information_scales_with_redundancy
abbrev OBS1 := @ObserverModel.exact_recovery_implies_pair_injective
abbrev OBS2 := @ObserverModel.pair_injective_implies_exact_recovery
abbrev OBS3 := @ObserverModel.fiber_card_le_tag_alphabet
abbrev OBS4 := @ObserverModel.exact_recovery_global_count
abbrev OBS5 := @ObserverModel.clique_card_le_tag_alphabet
abbrev OBS6 := @ObserverModel.no_observation_only_exact_recovery_of_nontrivial_clique
abbrev OBS7 := @ObserverModel.architecture_support_global_bound
abbrev OBS8 := @ObserverModel.architecture_support_bound_from_clique
abbrev OBS9 := @ObserverModel.architecture_support_gt_one_of_nontrivial_clique
abbrev OBS10 := @ObserverModel.observation_only_architecture_impossible_on_nontrivial_clique
abbrev OBS11 := @ObserverModel.supportCount_eq_dof
abbrev OBS12 := @ObserverModel.nontrivial_clique_forces_dof_gt_one
abbrev OBS13 := @ObserverModel.exactOn_clique_card_le_tag_alphabet
abbrev PRB1 := @ObserverModel.successSet_card_le_budget
abbrev PRB2 := @ObserverModel.successProb_le_budget_mul_massBound
abbrev PRB3 := @ObserverModel.errorProb_ge_one_sub_budget_mul_massBound
abbrev PRB4 := ObserverModel.vanishingError
abbrev PRB5 := @ObserverModel.not_vanishingError_of_lower_bound
abbrev PRB6 := @ObserverModel.family_not_vanishing_of_budget_mass_gap
abbrev PRB7 := @ObserverModel.errorProb_uniform_ge_one_sub_budget_div_card
abbrev PRB8 := @ObserverModel.successProb_uniform_le_budget_div_card
abbrev PRB9 := @ObserverModel.weak_fano_uniform_budget_lower_bound
abbrev PRB10 := @ObserverModel.uniform_family_not_vanishing_of_budget_ratio_gap
abbrev PRB11 := @ObserverModel.successProb_uniform_eq_card_div_card
abbrev PRB12 := @ObserverModel.errorProb_uniform_eq_one_sub_card_div_card
abbrev PRB13 := @ObserverModel.weak_fano_uniform_successSet_lower_bound
abbrev PRB14 := @ObserverModel.weak_fano_uniform_via_successSet
abbrev PRB15 := @ObserverModel.uniform_success_failure_partition_entropy
abbrev PRB16 := @ObserverModel.fano_uniform_budgeted
abbrev PRB17 := @ObserverModel.fano_uniform_observation_only
abbrev PRB18 := @ObserverModel.observation_only_successProb_uniform_le_one_div_card
abbrev PRB19 := @ObserverModel.observation_only_errorProb_uniform_ge_one_sub_one_div_card
abbrev PRB20 := @ObserverModel.successProb_le_budget_mul_maxMass
abbrev PRB21 := @ObserverModel.errorProb_ge_one_sub_budget_mul_maxMass
abbrev PRB22 := @ObserverModel.weak_fano_maxMass_lower_bound
abbrev PRB23 := @ObserverModel.family_not_vanishing_of_budget_maxMass_gap
abbrev PRB24 := @ObserverModel.successSetCard_ge_successProb_div_maxMass
abbrev PRB25 := @ObserverModel.successSetCard_ge_one_sub_error_over_maxMass
abbrev PRB26 := @ObserverModel.observation_only_successProb_le_maxMass
abbrev PRB27 := @ObserverModel.observation_only_errorProb_ge_one_sub_maxMass
abbrev PRB28 := @ObserverModel.successProb_le_budget_times_exp_neg_minEntropy
abbrev PRB29 := @ObserverModel.errorProb_ge_one_sub_budget_times_exp_neg_minEntropy
abbrev PRB30 := @ObserverModel.sourceEntropy_uniformFiniteSource
abbrev PRB31 := @ObserverModel.minEntropy_le_sourceEntropy
abbrev PRB32 := @ObserverModel.successSetCard_ge_one_sub_error_times_exp_minEntropy
abbrev PRB33 := @ObserverModel.budget_ge_one_sub_error_times_exp_minEntropy
abbrev PRB34 := @ObserverModel.inv_card_le_maxMass
abbrev PRB35 := @ObserverModel.minEntropy_le_log_card
abbrev PRB36 := @ObserverModel.observation_only_successProb_le_exp_neg_minEntropy
abbrev PRB37 := @ObserverModel.observation_only_errorProb_ge_one_sub_exp_neg_minEntropy
abbrev PRB38 := @ObserverModel.sourceEntropy_nonneg
abbrev PRB39 := @ObserverModel.sourceEntropy_le_log_card
abbrev PRB40 := @ObserverModel.minEntropy_le_log_budget_sub_log_one_sub_error
abbrev PRB41 := @ObserverModel.observation_only_minEntropy_le_neg_log_one_sub_error
abbrev PRB42 := @ObserverModel.sourceEntropy_le_success_failure_partition
abbrev PRB43 := @ObserverModel.fano_arbitrary_budgeted
abbrev PRB44 := @ObserverModel.fano_arbitrary_observation_only
abbrev PRB45 := @ObserverModel.fano_arbitrary_conditional_style
abbrev PRB46 := @ObserverModel.fano_arbitrary_conditional_observation_only
abbrev PRB47 := @ObserverModel.sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair
abbrev PRB48 := @ObserverModel.conditionalEntropyGivenPair_le_sourceEntropy
abbrev PRB49 := @ObserverModel.mutualInfoSurrogate_eq_observationTagEntropy
abbrev PRB50 := @ObserverModel.conditionalEntropyGivenPair_le_fano_arbitrary
abbrev PRB51 := @ObserverModel.conditionalEntropyGivenPair_le_fano_observation_only
abbrev PRB52 := @ObserverModel.observationTagEntropy_le_log_budget
abbrev PRB53 := @ObserverModel.mutualInfoSurrogate_le_log_budget
abbrev PRB54 := @ObserverModel.exactOn_clique_subsetEntropy_le_log_tags
abbrev PRB55 := @ObserverModel.successSet_clique_entropy_le_log_tags
abbrev PRB56 := @ObserverModel.pmfEntropy_successFailurePMF
abbrev PRB57 := @ObserverModel.mutualInfoDeterministic_eq_observationTagEntropy
abbrev PRB58 := @ObserverModel.mutualInfoDeterministic_eq_source_minus_conditional
abbrev PRB59 := @ObserverModel.mutualInfoDeterministic_nonneg
abbrev PRB60 := @ObserverModel.mutualInfoDeterministic_le_log_budget
abbrev PRB61 := @ObserverModel.conditionalEntropyGivenPair_nonneg
abbrev PRB62 := @ObserverModel.observationTagEntropy_le_sourceEntropy
abbrev PRB63 := @ObserverModel.mutualInfoDeterministic_le_sourceEntropy
abbrev PRB64 := @ObserverModel.coarsenedObservationEntropy_le_observationTagEntropy
abbrev PRB65 := @ObserverModel.coarsenedMutualInfoDeterministic_eq_coarsenedObservationEntropy
abbrev PRB66 := @ObserverModel.coarsenedMutualInfoDeterministic_le_original
abbrev PRB67 := @ObserverModel.deterministicKernel_data_processing
abbrev PRB68 := @ObserverModel.deterministicKernel_entropy_data_processing
abbrev PRB69 := @ObserverModel.inducedPairPMFFin_eq_uniform_of_klDiv_zero
abbrev PRB70 := @ObserverModel.observationTagEntropy_eq_log_budget_of_klDiv_zero_uniform
abbrev PRB71 := @ObserverModel.mutualInfoDeterministic_eq_log_budget_of_klDiv_zero_uniform
abbrev PRB72 := @ObserverModel.nonuniform_inducedPairPMFFin_implies_klDiv_ne_zero
abbrev PRB73 := @ObserverModel.observationTagEntropy_gap_nonneg
abbrev PRB74 := @ObserverModel.observationTagEntropy_gap_eq_zero_of_klDiv_zero_uniform
abbrev PRB75 := @ObserverModel.observationTagEntropy_gap_pos_implies_klDiv_ne_zero
abbrev PRB76 := @ObserverModel.mutualInfoDeterministic_gap_pos_implies_klDiv_ne_zero
abbrev PRB77 := @ObserverModel.successFailureFiniteSource_entropy_eq_binEntropy
abbrev PRB78 := @ObserverModel.successFailureEntropy_le_sourceEntropy
abbrev PRB79 := @ObserverModel.rvEntropy_failureRVFin_eq_binEntropy
abbrev PRB80 := @ObserverModel.rvEntropy_failureRVFin_le_sourceEntropy
abbrev PRB81 := @ObserverModel.decodedOutputEntropy_le_mutualInfoDeterministic
abbrev PRB82 := @ObserverModel.decodedOutputEntropy_source_gap_nonneg
abbrev PRB83 := @ObserverModel.decodedOutputEntropy_log_gap_nonneg
abbrev PRB84 := @ObserverModel.decodedOutputEntropy_le_success_failure_partition
abbrev PRB85 := @ObserverModel.decodedOutputEntropy_fano_budgeted
abbrev PRB86 := @ObserverModel.observableEntropy_failureObservable_eq_binEntropy
abbrev PRB87 := @ObserverModel.decodedOutputEntropy_fano_observation_only
abbrev PRB88 := @ObserverModel.observableEntropy_failureObservable_le_sourceEntropy
abbrev PRB89 := @ObserverModel.decodedOutputObservable_eq_coarsenedObservationObservable
abbrev PRB90 := @ObserverModel.DeterministicObservable.entropy_coarsen_le_sourceEntropy
abbrev PRB91 := @ObserverModel.inducedDecodedOutputPMFFin_eq_uniform_of_klDiv_zero
abbrev PRB92 := @ObserverModel.decodedOutputEntropy_eq_log_outputAlphabet_of_klDiv_zero_uniform
abbrev PRB93 := @ObserverModel.decodedOutputEntropy_gap_eq_zero_of_klDiv_zero_uniform
abbrev PRB94 := @ObserverModel.decodedOutputEntropy_gap_pos_implies_klDiv_ne_zero
abbrev PRB95 := @ObserverModel.decodedOutputObservable_entropy_le_sourceEntropy
abbrev ENT13 := @ObserverModel.pmfEntropy_inducedDecodedOutputPMFFin_eq_decodedOutputEntropy

-- DER1: all_derived_from_source requires Location, DerivationSystem, source, and all_locations parameters
-- Use: `all_derived_from_source D source locations` where D : DerivationSystem Location
abbrev DER1 := @all_derived_from_source
abbrev DER2 := ClaimClosure.derivation_preserves_coherence_core
abbrev AXM1 := @complete_mono
abbrev AXM2 := @completeD_mono
abbrev AXM3 := @minimal_no_redundant_axes
abbrev AXM4 := @semantically_minimal_implies_independent
abbrev CPL1 := cost_ratio_eq_dof
abbrev REQ1 := structural_facts_fixed_at_definition
abbrev REQ2 := definition_hooks_necessary
abbrev REQ3 := introspection_necessary_for_verification
abbrev IND1 := both_requirements_independent
abbrev IND2 := both_requirements_independent'
abbrev SOT1 := ssot_iff
abbrev GEN1 := generated_file_is_second_encoding
abbrev LNG1 := ClaimClosure.language_realizability_criterion

abbrev BND1 := ssot_upper_bound
abbrev BND2 := non_ssot_lower_bound
abbrev BND3 := ssot_advantage_unbounded
abbrev BND4 := ClaimClosure.arbitrary_reduction_factor
abbrev REG1 := ClaimClosure.operating_regimes_partition
abbrev REG2 := ClaimClosure.pareto_optimality_p1
abbrev REG3 := ClaimClosure.no_tradeoff_at_p1
abbrev REG4 := ClaimClosure.amortized_complexity_core

abbrev PYH1 := Python.python_has_hooks
abbrev PYI1 := Python.python_has_introspection

/-! ## FIRST PRINCIPLES FORCING CHAIN
    These theorems establish that SSOT is FORCED by first principles.
    The chain: Derivation → Causality → Trichotomy → Only source_hooks works → SSOT unique

    If you accept "derivation" as a coherent concept, you MUST accept trichotomy.
    If you accept trichotomy, only source_hooks achieves SSOT.
    Therefore, SSOT is forced by first principles - cannot be honestly rejected.
-/

-- Trichotomy theorems (TRI*): Timing trichotomy and its necessity
abbrev TRI1 := timing_trichotomy_exhaustive      -- Trichotomy is exhaustive
abbrev TRI2 := trichotomy_necessary_for_causality -- Causality requires trichotomy
abbrev TRI3 := trichotomy_necessary_for_mechanism -- Mechanisms require trichotomy
abbrev TRI4 := no_mechanism_outside_trichotomy   -- No mechanism outside trichotomy

-- Model completeness theorems (MDL*): Mechanism enumeration and capability
abbrev MDL1 := mechanisms_cover_all_timings      -- Mechanisms cover all timing categories
abbrev MDL2 := mechanism_exhaustiveness          -- Every mechanism has valid timing
abbrev MDL3 := only_at_definition_works          -- Only at_definition timing works
abbrev MDL4 := mechanism_structural_capability   -- Source_hooks is uniquely capable
abbrev MDL5 := model_completeness                -- achieves_ssot ↔ source_hooks

-- Uniqueness theorems (UNQ*): SSOT mechanism is unique
abbrev UNQ1 := uniqueness                        -- Combined existence + uniqueness
abbrev UNQ2 := uniqueness_exists                 -- ∃ mechanism achieving SSOT
abbrev UNQ3 := uniqueness_unique                 -- All SSOT mechanisms are source_hooks

-- Adversary/impossibility theorems (ADV*): Why alternatives fail
abbrev ADV1 := adversary_construction            -- m ≠ source_hooks → fails
abbrev ADV2 := derivation_impossibility          -- No alternative works
abbrev ADV3 := compile_macros_insufficient       -- Compile macros fail
abbrev ADV4 := build_codegen_insufficient        -- Build codegen fails
abbrev ADV5 := runtime_reflection_too_late       -- Runtime reflection fails

-- Independence theorems (IND*): External tools don't count
abbrev IND3 := external_tools_not_derivation     -- External tools can be bypassed
abbrev IND4 := language_semantics_is_derivation  -- Language semantics is derivation
abbrev IND5 := ide_refactoring_not_derivation    -- IDE refactoring not derivation
