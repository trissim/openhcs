import Paper4dFrontier.Classification
import Paper4dFrontier.Block2Progress
import Paper4dFrontier.BinaryPairwiseDichotomy
import Paper4dFrontier.DecisionRelevantPairwiseDichotomy
import Paper4dFrontier.Block6Obstruction
import Paper4dFrontier.MetaCharacterization
import Paper4dFrontier.AdmissibleCharacterization
import Paper4dFrontier.ObstructionPredicateCandidates
import Paper4dFrontier.CycleRankWitness
import Paper4dFrontier.CoreWitnesses
import Paper4dFrontier.FirstTwoIndicator
import Paper4dFrontier.DimensionalNoiseExtension
import Paper4dFrontier.EnumerationBounds
import Paper4dFrontier.LegacyTreeGap
import Paper4dFrontier.ListPolynomialEnumeration
import Paper4dFrontier.LiftReductions
import Paper4dFrontier.ParentTreewidth
import Paper4dFrontier.RealTreewidth
import Paper4dFrontier.BoundedActionsOnlyWitness2
import Paper4dFrontier.BoundedTreewidthOnlyWitness
import Paper4dFrontier.SeparableOnlyWitness
import Paper4dFrontier.SymmetryOnlyWitness
import Paper4dFrontier.TreeStructureHardnessBridge
import Paper4dFrontier.StructuralWitnesses
import Paper4dFrontier.SymmetryRankWitness
import Paper4dFrontier.LowRankOnlyWitness
import Paper4dFrontier.TreeDeletion
import Paper4dFrontier.TreeHellyEquivFinset
import Paper4dFrontier.TreeHellyFinset
import Paper4dFrontier.TreeHellyHelpers
import Paper4dFrontier.TreeHellyTheorem
import Paper4dFrontier.TreewidthClique
import Paper4dFrontier.Realizability
import Paper4dFrontier.FamilyAxioms
import Paper4dFrontier.ClosureLaws
import Paper4dFrontier.RealTreewidthWitnesses

open Paper4dFrontier

/-- Stable Lean-handle IDs for paper4d (frontier sequel).

The build system parses `abbrev CODE := target` and uses these IDs in
`lean_handle_ids_auto.tex`, which then powers `\LH{CODE}` links in the PDF. -/

abbrev FR1 := @currentFamilies_partition_by_role
abbrev FR2 := @currentFamilies_partition_counts
abbrev FR3 := @current_families_have_finite_basis
abbrev FR4 := @current_positive_interfaces_factor_through_role
abbrev FR5 := @lifted_families_reduce_to_core
abbrev FR6 := @degenerateFamilies_complete
abbrev FR25 := @currentFamilies_explicit
abbrev FR26 := @coreFamilies_explicit
abbrev FR27 := @liftedFamilies_explicit
abbrev FR28 := @degenerateFamilies_explicit
abbrev FR29 := @primitiveMechanisms_explicit

abbrev FR7 := @exists_decisionProblem_realizing_labeling
abbrev FR8 := @realizingProblem_decisionEquiv_iff
abbrev FR39 := @realizingProblemQuotientEquivLabelRange_apply_quotientMap
abbrev FR40 := @realizingProblem_quotientCard_eq_labelRangeCard
abbrev FR45 := @setoidRealizingProblem_decisionEquiv_iff
abbrev FR46 := @setoidRealizingProblemQuotientEquivSetoidQuotient_apply_quotientMap
abbrev FR47 := @exists_decisionProblem_realizing_setoid

abbrev FR9 := @decisionEquiv_iff_of_opt_eq
abbrev FR10 := @isSufficient_iff_of_opt_eq
abbrev FR11 := @isRelevant_iff_of_opt_eq
abbrev FR12 := @isIrrelevant_iff_of_opt_eq
abbrev FR30 := @isSufficient_iff_of_decisionEquiv_iff
abbrev FR31 := @isRelevant_iff_of_decisionEquiv_iff
abbrev FR32 := @isIrrelevant_iff_of_decisionEquiv_iff
abbrev FR33 := @isMinimalSufficient_iff_of_decisionEquiv_iff
abbrev FR34 := @sufficientSets_eq_of_decisionEquiv_iff
abbrev FR35 := @relevantSet_eq_of_decisionEquiv_iff
noncomputable abbrev FR36 := @quotientEquiv_of_decisionEquiv_iff
abbrev FR51 := @quotientEquiv_of_decisionEquiv_iff_apply_quotientMap
abbrev FR37 := @quotientCard_eq_of_decisionEquiv_iff
abbrev FR41 := @certificationStatistic_eq_of_decisionEquiv_iff
abbrev FR42 := @sufficientSetCount_eq_of_decisionEquiv_iff
abbrev FR43 := @minimalSufficientSetCount_eq_of_decisionEquiv_iff
abbrev FR44 := @relevantCoordCount_eq_of_decisionEquiv_iff
abbrev FR48 := @isSufficient_iff_agreementRel_le_decisionEquiv
abbrev FR49 := @isIrrelevant_iff_sufficient_erase
abbrev FR50 := @isRelevant_iff_not_sufficient_erase

abbrev FR13 := @isOptimal_positiveAffineTransform_iff
abbrev FR14 := @opt_eq_positiveAffineTransform
abbrev FR15 := @isSufficient_positiveAffineTransform_iff
abbrev FR16 := @isRelevant_positiveAffineTransform_iff

abbrev FR17 := @decisionEquiv_relabelActions_iff
abbrev FR18 := @decisionEquiv_relabelStates_iff
abbrev FR19 := @isSufficient_relabelActions_iff
abbrev FR20 := @isSufficient_relabelStates_iff

abbrev FR21 := @allProblems_relabelInvariant
abbrev FR22 := @allProblems_kernelUniversal
abbrev FR23 := @relabelInvariant_not_enough
abbrev FR24 := @optRangeCard_realizingIdentity
abbrev FR38 := @quotientCard_realizingIdentity

abbrev FR52 := @empty_sufficient_of_constant_opt
abbrev FR53 := @irrelevant_of_constant_opt
abbrev FR54 := @decisionEquiv_duplicateAction_iff
abbrev FR55 := @isSufficient_duplicateAction_iff
abbrev FR56 := @isRelevant_duplicateAction_iff
abbrev FR72 := @some_mem_opt_addDuplicateAction_iff
abbrev FR73 := @none_mem_opt_addDuplicateAction_iff
abbrev FR57 := @decisionEquiv_duplicateState_iff
abbrev FR58 := @isSufficient_duplicateState_iff
abbrev FR59 := @isRelevant_duplicateState_iff
abbrev FR60 := @duplicateStateQuotientEquivOriginal_apply_quotientMap
abbrev FR68 := @decisionEquiv_addDuplicateState_iff
abbrev FR69 := @isSufficient_addDuplicateState_iff
abbrev FR70 := @isRelevant_addDuplicateState_iff
abbrev FR71 := @addDuplicateStateQuotientEquivOriginal_apply_quotientMap
abbrev FR83 := @isSufficient_noiseExtended_iff
abbrev FR84 := @isRelevant_noiseExtended_iff
abbrev FR85 := @lastCoord_irrelevant_noiseExtended

abbrev FR61 := @bounded_actions_not_imply_separable_witness
abbrev FR62 := @separable_not_imply_bounded_actions_witness
abbrev FR63 := @low_rank_not_imply_coordinate_symmetry_witness
abbrev FR88 := @low_rank_not_imply_tree_structure_witness
abbrev FR66 := @booleanCube_card
abbrev FR67 := @exists_booleanCube_larger_than_monomial
abbrev FR81 := @exists_list_polynomial_lt_booleanCube
abbrev FR74 := @product_distribution_reduction_tractable
abbrev FR75 := @bounded_support_reduction_tractable
abbrev FR76 := @bounded_horizon_reduction_tractable
abbrev FR77 := @fully_observable_reduction_tractable
abbrev FR78 := @bounded_treewidth_not_imply_bounded_actions_witness
abbrev FR79 := @bounded_treewidth_not_imply_coordinate_symmetry_witness
abbrev FR80 := @tree_structure_strictly_inside_bounded_treewidth_witness
abbrev FR87 := @treewidth_le_zero_trivial
abbrev FR86 := @coordinate_symmetry_not_imply_low_rank_witness
abbrev FR109 := @firstTwoEq_not_symmetric
abbrev FR114 := @weak_tree_structured_not_width_one
abbrev FR89 := @realTreewidth_le_card_pred
abbrev FR90 := @connected_induce_inter_of_isTree
abbrev FR99 := @parentBagGraph_isTree
abbrev FR100 := @parentTree_realTreewidth_one
abbrev FR101 := @low_rank_only_witness
abbrev FR106 := @bounded_actions_only_witness
abbrev FR105 := @separable_only_witness
abbrev FR113 := @symmetry_only_witness
abbrev FR108 := @all_independent_core_mechanisms_nondegenerate
abbrev FR112 := @bounded_treewidth_only_witness
abbrev FR115 := @tree_structure_hardness_bridge
abbrev FR116 := @bounded_actions_hardness_bridge
abbrev FR117 := @nonseparable_hardness_bridge
abbrev FR118 := @pairCrossDifference_eq_binaryCrossDifference_of_lt
abbrev FR119 := @pairwise_zero_crossDifference_unaryDecomposition
abbrev FR120 := @binary_pairwise_symmetry_dichotomy
abbrev FR121 := @actionGapCrossDifference_eq_binaryCrossDifference_of_lt
abbrev FR122 := @decisionRelevant_zero_actionGap_implies_unaryReduction
abbrev FR123 := @binary_pairwise_symmetry_decision_relevant_dichotomy
abbrev FR124 := @block6_genuine_interaction_dichotomy_obstruction
abbrev FR125 := @decisionRelevantInteractionGraph_addActionOffset
abbrev FR126 := @action_offset_can_force_constant_optimizer
abbrev FR127 := @block6_action_offset_obstruction
abbrev FR128 := @offsetNormalizedDecisionRelevantInteractionGraph_wellDefined
abbrev FR129 := @block6_offset_normalized_obstruction
abbrev FR130 := @neverOptimalGhost_decisionRelevantGraph_eq_top
abbrev FR131 := @neverOptimalGhost_supportedGraph_eq_bot
abbrev FR132 := @support_filtering_removes_known_block6_obstructions
abbrev FR133 := @block6_ghost_action_obstruction
abbrev FR134 := @marginMasking_supportedGraph_eq_top
abbrev FR135 := @marginMasking_zero_sufficient
abbrev FR136 := @block6_optimizer_supported_obstruction
abbrev FR137 := @MarginBounded
abbrev FR138 := @dominantPair_marginBounded
abbrev FR139 := @dominantPair_supportedGraph_eq_top
abbrev FR140 := @block6_margin_bounded_obstruction
abbrev FR141 := @NormalizationPredicate
abbrev FR142 := @TractabilityCharacterization
abbrev FR143 := @Defeats
abbrev FR144 := @collapseLandscapeInfinity
abbrev FR145 := @not_collapseLandscapeInfinity_of_oracle_predicate
abbrev FR146 := @AdmissibleNormalizationPredicate
abbrev FR147 := @ClosureLawInvariant
abbrev FR148 := @StructuralExtractorOn
abbrev FR149 := @LocallyDefinable
abbrev FR150 := @admissibleCollapseLandscapeInfinity
abbrev FR151 := @DenseDecisionRelevantPredicate
abbrev FR152 := @MarginBoundedDenseDecisionRelevantPredicate
abbrev FR153 := @offsetCollapsedSlice_denseDecisionRelevant
abbrev FR154 := @neverOptimalGhostSlice_denseDecisionRelevant
abbrev FR155 := @marginMaskingSlice_denseDecisionRelevant
abbrev FR156 := @dominantPairSlice_marginBounded_denseDecisionRelevant
abbrev FR157 := @LocalPattern
abbrev FR158 := @BoundedPatternScheme
abbrev FR159 := @BoundedPatternDefinable
abbrev FR160 := @ClosureStep
abbrev FR161 := @ClosureHull
abbrev FR162 := @ClosureGeneratedByBoundedPatterns
abbrev FR163 := @OffsetCollapsedClosureSlice
abbrev FR164 := @offsetCollapsedClosureSlice_generatedByBoundedPatterns
abbrev FR165 := @offsetCollapsedClosureSlice_closureLawInvariant
abbrev FR166 := @hasOffsetCollapsedWitnessPattern_not_closureLawInvariant
abbrev FR167 := @rawOffsetWitness_boundedPattern_not_closureInvariant
abbrev FR168 := @offsetClosure_generated_and_invariant
abbrev FR169 := @offset_admissibility_refinement_gap
abbrev FR170 := @HasOffsetCollapsedScaledWitnessPattern
abbrev FR171 := @scaledOffsetChecker_polynomialTimeCheckable
abbrev FR172 := @scaledOffsetChecker_handles_uniform_scaling
abbrev FR173 := @LocalPattern.occursUpToPositiveScaleInSlice_globalScaleInvariant
abbrev FR174 := @scaledOffsetChecker_globalPositiveScaleInvariant
abbrev FR175 := @translatedOffsetCollapsed_additive_mismatch
abbrev FR110 := @parentTreeStructured_implies_treeStructured
abbrev FR111 := @treeStructured_and_unique_parent_iff_parentTreeStructured
abbrev FR102 := @completeGraph_not_realTreewidth_le_of_large
abbrev FR103 := @treeHelly_none_leaf_finset
abbrev FR91 := @treeHellyFinset_all
abbrev FR92 := @clique_has_common_bag
abbrev FR93 := @completeGraph_not_realTreewidth_le
abbrev FR94 := @bounded_actions_not_imply_real_treewidth_witness
abbrev FR95 := @coordinate_symmetry_not_imply_real_treewidth_witness
abbrev FR96 := @tree_structure_strictly_inside_real_treewidth_witness
abbrev FR97 := @real_treewidth_not_imply_bounded_actions_witness
abbrev FR98 := @real_treewidth_not_imply_coordinate_symmetry_witness
