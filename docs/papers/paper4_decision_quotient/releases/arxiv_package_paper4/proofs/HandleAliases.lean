import AbstractClassSystem.Core
import AbstractClassSystem.Bridge
import AbstractClassSystem.Extended
import AbstractClassSystem.Undecidability
import Paper1IT.GraphEntropy
import Paper1IT.GraphEntropyAsymptotic
import axis_framework
import lwd_converse

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
-- SHP1/SHP2/OBS1 require type parameters - use @shape_cannot_distinguish directly
abbrev SHP2 := @shape_provenance_impossible       -- Shape provenance impossible
abbrev ADV1 := adversary_forces_n_minus_1_queries -- Adversary forces n-1 queries
abbrev MDL1 := model_completeness                -- Model completeness

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
abbrev L7 := adversary_forces_n_minus_1_queries

abbrev LWDC1 := @LWDConverse.collision_block_requires_bits
abbrev LWDC2 := @LWDConverse.impossible_when_bits_too_small
abbrev LWDC3 := @LWDConverse.maximal_barrier_requires_bits
