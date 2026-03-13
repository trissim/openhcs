import AbstractClassSystem.Core
import AbstractClassSystem.Bridge
import AbstractClassSystem.Extended
import AbstractClassSystem.Undecidability
import AbstractClassSystem.Neurosymbolic
import Paper1IT.BudgetSplit
import Paper1IT.FiniteRateDistortionConverse
import Paper1IT.FiniteRateDistortionBounds
import Paper1IT.FiberAllocation
import Paper1IT.FiberRateDistortion
import Paper1IT.GrowthCollisions
import Paper1IT.GrowthTagBudget
import Paper1IT.PMFEntropy
import Paper1IT.QueryBitBridge
import Paper1IT.RateDistortion
import Paper1IT.ZeroErrorConditionalEntropy
import Paper1IT.GraphEntropy
import Paper1IT.ComputabilityQuantization
import Paper1IT.GraphEntropyAsymptotic
import axis_framework
import lwd_converse
import discipline_migration

open AbstractClassSystem

/-- Stable, semantic Lean-handle IDs for Paper 1 (JSAIT).

    The build system parses `abbrev CODE := target` and uses these IDs in
    `lean_handle_ids_auto.tex`, which then powers `\LH{CODE}` links in the PDF.

    ## CORE CLAIM HANDLES
    These map to the 10 claims marked in the paper graph -/

abbrev CMP1 := complexity_gap_unbounded          -- Complexity gap is unbounded
abbrev NOM1 := nominal_centralized               -- Nominal typing centralizes
abbrev NOM2 := nominal_localization_constant_semantic  -- Nominal localization O(1)
abbrev DUC1 := duck_localization_linear          -- Duck localization O(n)
abbrev ENC1 := encoding_location_gap             -- Encoding location gap
abbrev INF1 := @AbstractClassSystem.Scope.observer_factors -- Observer factors through observable quotient
-- SHP1/SHP2/OBS1 require type parameters - use @shape_cannot_distinguish directly
abbrev PRV1 := @AbstractClassSystem.provenance_impossibility_universal -- Provenance not representation-computable
abbrev SHP2 := @shape_provenance_impossible       -- Shape provenance impossible
abbrev ADV1 := adversary_forces_n_minus_1_queries -- Adversary forces n-1 queries
abbrev MDL1 := model_completeness                -- Model completeness
abbrev FXI1 := fixed_axis_incompleteness         -- Fixed observable basis is incomplete
abbrev ZDT1 := @Ssot.Paper1IT.requiredTagBits_le_iff_le_pow -- Zero-error threshold by bit budget
abbrev PD1 := nominal_pareto_dominates_shape -- Nominal Pareto dominates structural
abbrev NOH1 := admissibility_no_hidden_state -- No hidden registries in valid traces

/-- Canonical graph-entropy handle IDs (shared downstream). -/
abbrev GPH2 := @Ssot.GraphEntropy.clique_card_le_colors
abbrev GPH3 := @Ssot.GraphEntropy.complete_graph_needs_all_colors
abbrev GPH4 := @Ssot.GraphEntropy.fiber_card_le_tag_alphabet
abbrev GPH5 := @Ssot.GraphEntropy.constant_observation_needs_all_tags
abbrev GPH6 := @Ssot.GraphEntropy.maxFiberCard_le_tag_alphabet
abbrev GPH7 := @Ssot.GraphEntropy.card_div_le_tag_alphabet
abbrev GPH8 := @Ssot.GraphEntropy.card_le_mul_tag_alphabet
abbrev GPH9 := @Ssot.GraphEntropy.tagFeasible_mono
abbrev GPH10 := @Ssot.GraphEntropy.maxFiberCard_le_of_tagFeasible
abbrev GPH11 := @Ssot.GraphEntropy.card_div_le_of_tagFeasible
abbrev GPH12 := @Ssot.GraphEntropy.tagFeasible_of_maxFiberCard_le
abbrev GPH13 := @Ssot.GraphEntropy.tagFeasible_iff_maxFiberCard_le
abbrev GPH14 := @Ssot.GraphEntropy.block_tagFeasible_of_tagFeasible
abbrev GPH15 := @Ssot.GraphEntropy.pow_maxFiberCard_le_of_block_tagFeasible
abbrev GPH16 := @Ssot.GraphEntropy.block_tagFeasible_iff_pow_maxFiberCard_le
abbrev GPH17 := @Ssot.GraphEntropy.maxFiberCard_blockObserve_eq_pow
abbrev GPH18 := @Ssot.GraphEntropy.blockTagRateBits_eq_mul_oneShot
abbrev GPH19 := @Ssot.GraphEntropy.blockTagRateBitsPerCoordinate_eq_oneShot
abbrev GPH20 := @Ssot.GraphEntropy.minBlockFeasibleAlphabet_eq_pow
abbrev GPH21 := @Ssot.GraphEntropy.minBlockFeasibleBits_eq_blockTagRateBits
abbrev GPH22 := @Ssot.GraphEntropy.tendsto_blockTagRateBitsPerCoordinate_succ
abbrev GPH23 := @Ssot.GraphEntropy.block_tagFeasible_pow_budget_iff
abbrev GPH24 := @Ssot.GraphEntropy.feasibleAtAlphabetBase_iff
abbrev GPH25 := @Ssot.GraphEntropy.maxFiberCard_le_one_of_injective
abbrev GPH26 := @Ssot.GraphEntropy.maxFiberCard_eq_one_iff_injective
abbrev GPH27 := @Ssot.GraphEntropy.tagFeasible_one_iff_injective
abbrev GPH28 := @Ssot.GraphEntropy.maxFiberCard_mono_of_factors_through
abbrev GPH29 := @Ssot.GraphEntropy.uniformFiberSuccessRate_le_tag_ratio
abbrev GPH30 := @Ssot.GraphEntropy.uniformFiberErrorRate_ge_one_sub_tag_ratio
abbrev GPH31 := @Ssot.GraphEntropy.inclusionMinimalCoordinateDistinguishing_not_equicardinal
abbrev GPH32 := @Ssot.GraphEntropy.optimalFiberBitLength_feasible
abbrev GPH33 := @Ssot.GraphEntropy.optimalExpectedAdaptiveBitLength_le
abbrev GPH34 := @Ssot.GraphEntropy.conditionalEntropyGiven_le_log2_mul_expectedAdaptiveBitLength
abbrev GPH35 := @Ssot.GraphEntropy.exists_conditionalCodes_expectedLength_le_entropy_bits_plus_one
abbrev GPH36 := @Ssot.GraphEntropy.fiberCard_mono_of_factors_through
abbrev GPH37 := @Ssot.GraphEntropy.optimalFiberBitLength_mono_of_factors_through
abbrev GPH38 := @Ssot.GraphEntropy.taskRecoverable_iff_maxTaskFiberCard_le_one
abbrev GPH39 := @Ssot.GraphEntropy.maxTaskFiberCard_mono_of_factors_through
abbrev GPH40 := @Ssot.GraphEntropy.taskRecoverable_pair_iff
abbrev GPH41 := @Ssot.GraphEntropy.maxTaskFiberCard_pair_le_left
abbrev GPH42 := @Ssot.GraphEntropy.observeFiber_prod_card
abbrev GPH43 := @Ssot.GraphEntropy.maxFiberCard_prod
abbrev GPH44 := @Ssot.GraphEntropy.maxFiberCard_eq_finiteSup
abbrev GPH45 := @Ssot.GraphEntropy.maxFiberCard_computable_by_fiber_enumeration
abbrev GPH46 := @Ssot.GraphEntropy.card_le_card_mul_maxFiberCard
abbrev GPH47 := @Ssot.GraphEntropy.ceilDiv_card_le_maxFiberCard
abbrev GPH48 := @Ssot.GraphEntropy.quantization_lower_bound_fin
abbrev GPH49 := @Ssot.GraphEntropy.card_le_card_mul_of_maxFiberCard_le
abbrev GPH50 := @Ssot.GraphEntropy.precision_requirement
abbrev GPH51 := @Ssot.GraphEntropy.precision_requirement_fin
abbrev GPH52 := @Ssot.GraphEntropy.fiber_is_clique
abbrev GPH53 := @Ssot.GraphEntropy.fiber_finset_is_clique
abbrev GPH54 := @Ssot.QueryBitBridge.card_distinguished_set_le_two_pow_card
abbrev GPH55 := @Ssot.QueryBitBridge.clog_card_le_query_count
abbrev GPH56 := @Ssot.QueryBitBridge.fiber_card_le_two_pow_query_count
abbrev GPH57 := @Ssot.QueryBitBridge.maxFiberCard_clog_le_query_count
abbrev GPH58 := @Ssot.QueryBitBridge.distinguishesOn_mono
abbrev GPH59 := @Ssot.QueryBitBridge.clog_card_le_query_count_mono
abbrev GPH60 := @Ssot.QueryBitBridge.distinguished_set_eq_full_transcript_capacity
abbrev GPH61 := @Ssot.QueryBitBridge.clog_query_floor_is_tight_of_full_capacity
abbrev GPH62 := @Ssot.QueryBitBridge.queryExpressivityGap_nonneg
abbrev GPH63 := @Ssot.QueryBitBridge.queryExpressivityGap_eq_zero_iff
abbrev GPH64 := @Ssot.QueryBitBridge.queryExpressivityGap_mono

/-- ## FIRST PRINCIPLES FORCING CHAIN
    These theorems establish that typing discipline choice is FORCED by first principles.
    The chain: Query Requirement → Axis Structure → Completeness → Nominal unique

    If you accept "type query answerable" as a requirement, you MUST accept axis structure.
    If you accept axis structure, nominal is the unique complete solution.
    Therefore, nominal typing is forced by first principles - cannot be honestly rejected. -/

-- Completeness/Exhaustiveness theorems (COMPL*)
abbrev COMPL1 := axes_complete                   -- Axes are complete
abbrev COMPL2 := BSH_completes_hierarchical_config  -- BSH completes hierarchical
abbrev COMPL3 := hierarchy_necessary             -- Hierarchy is necessary

-- Impossibility theorems (IMP*): Why alternatives fail
abbrev IMP1 := @shape_provenance_impossible       -- Shape cannot track provenance
abbrev AXI1 := @shape_cannot_distinguish         -- Same-namespace types are indistinguishable
abbrev EMB1 := semantic_non_embeddability        -- Nominal cannot embed into structural
abbrev EXT1 := @extension_impossibility          -- Namespace-only extensions stay shape-bound
abbrev ALT1 := protocol_is_concession            -- Protocol is a concession
abbrev ALT2 := protocol_not_alternative          -- Protocol is not an alternative
abbrev IMP2 := java_forced_to_composition        -- Java forced to composition pattern

-- Uniqueness theorems (UNIQ*)
abbrev UNIQ1 := unique_tree_encoding             -- Tree encoding is unique

-- Adversary/Forcing theorems (FORC*)
abbrev FORC1 := adversary_forces_n_minus_1_queries  -- Adversary forces queries
abbrev FORC2 := typing_choice_unavoidable        -- Typing choice unavoidable
abbrev FORC3 := capability_foreclosure_irreversible -- Capability foreclosure irreversible

-- Dominance theorems (DOM*): Strict ordering between disciplines
abbrev DOM1 := strict_dominance                  -- Strict dominance

/-- ## NEUROSYMBOLIC LINKING THEOREMS (NSL*)
    These theorems formalize the neurosymbolic pattern: neural encoder + symbolic handle.
    Key insight: the pair (encoder, handle) achieves zero-error identity recovery
    when the handle is injective, even if the encoder alone does not. -/
abbrev NSL1 := @AbstractClassSystem.Neurosymbolic.neurosymbolic_zero_error_identity
abbrev NSL2 := @AbstractClassSystem.Neurosymbolic.shape_alone_not_zero_error
abbrev NSL3 := @AbstractClassSystem.Neurosymbolic.neurosymbolic_necessary_for_zero_error
abbrev NSL4 := @AbstractClassSystem.Neurosymbolic.neurosymbolic_identity_recovery
abbrev NSL5 := @AbstractClassSystem.Neurosymbolic.neurosymbolic_injective_implies_recoverable
abbrev NSL6 := @AbstractClassSystem.Neurosymbolic.neurosymbolic_is_canonical_helper_view

/-- Deterministic finite fiberwise rate-distortion extensions (FRD*). -/
abbrev FRD1 := @ObserverModel.feasible_subsetMass_le_optimalFeasibleMass
abbrev FRD2 := @ObserverModel.optimalSubset_attains_optimalFeasibleMass

/-- Entropy-sensitive finite converse layer (RDC*). -/
abbrev RDC1 := @ObserverModel.finiteRateDistortionConverse
abbrev RDC2 := @ObserverModel.finiteConditionalRateDistortionConverse
abbrev RDC3 := @ObserverModel.finiteObservationOnlyRateDistortionConverse
abbrev RDC4 := @ObserverModel.finiteMinEntropyBudgetConverse
abbrev RDC5 := @ObserverModel.uniformFiniteRateDistortionConverse
abbrev RDC6 := @ObserverModel.finiteRateDistortionBound
abbrev RDC7 := @ObserverModel.logBudgetLowerBoundFromError
abbrev RDC8 := @ObserverModel.observationOnlyRateDistortionConverse
abbrev RDC9 := @ObserverModel.observationOnlyMinEntropyBound

/-- PMF entropy bridge layer (PMF*). -/
abbrev PMF1 := @ObserverModel.pmfEntropy_source_eq_sourceEntropy
abbrev PMF2 := @ObserverModel.conditionalEntropyGivenPair_eq_source_minus_pmfEntropy_pair
abbrev PMF3 := @ObserverModel.mutualInfoDeterministic_eq_pmfEntropy_pair

/-- Zero-error conditional-entropy sandwich (ZEC*). -/
abbrev ZEC1 := @Ssot.GraphEntropy.zeroErrorConditionalEntropySandwich

/-- Fiber-allocation monotonicity (FAL*). -/
abbrev FAL1 := @ObserverModel.allocatedRecoverableMass_mono
abbrev FAL2 := @ObserverModel.allocatedDistortion_anti

/-- Growth / collision and tag-budget handles. -/
abbrev GRC1 := @Ssot.Paper1IT.poissonCellCollisionProb_eq_one_sub_exp_mul_one_add
abbrev GRC2 := @Ssot.Paper1IT.poissonCollisionUnionBound_eq_sum_formula
abbrev GRC3 := @Ssot.Paper1IT.poissonCellCollisionProb_pos_iff
abbrev GTB1 := @Ssot.Paper1IT.requiredTagBits_positive_iff_two_le
abbrev GTB2 := @Ssot.Paper1IT.requiredTagBits_eq_zero_iff_le_one

/-- Abstract representation-vs-tag budget split handles. -/
abbrev BST1 := @Ssot.Paper1IT.exists_minimizing_split
abbrev BST2 := @Ssot.Paper1IT.exists_optimal_budget_split
abbrev BST3 := @Ssot.Paper1IT.exists_optimal_budget_split_le_reference

-- Coherence theorems (COH*): Why hedging/preference is incoherent
abbrev COH1 := preference_incoherent             -- Preference position incoherent
abbrev COH2 := AbstractClassSystem.hedging_incoherent  -- Hedging is incoherent

-- Bridge theorems (BRG*): Analysis shows positive EV
abbrev BRG1 := analysis_has_positive_ev          -- Analysis has positive EV
abbrev BRG2 := ignorant_choice_has_cost          -- Ignorant choice costs
abbrev BRG3 := retrofit_cost_dominates           -- Retrofit cost dominates

/-- Open-world robustness and undecidability handles. -/
abbrev ROB1 := @AbstractClassSystem.OpenWorld.extend_to_force_barrier
abbrev ROB2 := @AbstractClassSystem.OpenWorld.barrierFreedom_not_extension_stable
abbrev UND1 := @AbstractClassSystem.OpenWorld.hasBarrier_not_computable
abbrev UND2 := @AbstractClassSystem.OpenWorld.barrierFree_not_computable

/-- Legacy/stable IDs used in paper text and supplements. -/
abbrev ACS1 := adversary_forces_n_minus_1_queries
abbrev ACS2 := complexity_gap_unbounded
abbrev ACS3 := duck_localization_linear
abbrev ACS4 := encoding_location_gap
abbrev ACS5 := model_completeness
abbrev ACS6 := nominal_centralized
abbrev ACS7 := nominal_localization_constant_semantic
abbrev ACS8 := @shape_cannot_distinguish
abbrev ACS9 := @shape_provenance_impossible
abbrev PRIV1 := @AbstractClassSystem.nominal_identity_disclosure_constant
abbrev PRIV2 := @AbstractClassSystem.attribute_only_identity_disclosure_lower_bound
abbrev PRIV3 := @AbstractClassSystem.identity_disclosure_separation

abbrev L1 := @matroid_basis_equicardinality
abbrev L2 := fixed_axis_incompleteness
abbrev L3 := matroid_basis_equicardinality

theorem l4_exchange_wrapper {A : _root_.AxisSet}
    (horth : _root_.OrthogonalAxes A) : _root_.exchangeProperty A :=
  orthogonal_exchange horth

theorem l5_exchange_wrapper {A : _root_.AxisSet}
    (horth : _root_.OrthogonalAxes A) : _root_.exchangeProperty A :=
  orthogonal_implies_exchange horth

abbrev L4 := @l4_exchange_wrapper
abbrev L5 := @l5_exchange_wrapper
abbrev L6 := @nonorthogonal_complete_has_redundant_axis
abbrev L7 := @exists_semanticallyMinimal_subset
abbrev L8 := @exists_orthogonal_semanticallyMinimal_subset

abbrev LWDC1 := @LWDConverse.collision_block_requires_bits
abbrev LWDC2 := @LWDConverse.impossible_when_bits_too_small
abbrev LWDC3 := @LWDConverse.maximal_barrier_requires_bits
