import Paper4dFrontier.Classification
import Paper4dFrontier.Block2Progress
import Paper4dFrontier.BinaryPairwiseDichotomy
import Paper4dFrontier.DecisionRelevantPairwiseDichotomy
import Paper4dFrontier.Block6Obstruction
import Paper4dFrontier.MetaCharacterization
import Paper4dFrontier.AdmissibleCharacterization
import Paper4dFrontier.FullBinaryPairwiseDomain
import Paper4dFrontier.DistinctActionProfiles
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
import Paper4dFrontier.ApproximateAdmissibility
import Paper4dFrontier.ComputeCostInvariance
import Paper4dFrontier.ComputeCostExternalOutputs
import Paper4dFrontier.ComputeCostPayloads
import Paper4dFrontier.ComputeCostApplications
import Paper4dFrontier.StatisticalSemanticsApplications
import Paper4dFrontier.RealTreewidthWitnesses
import DecisionQuotient.Sufficiency
import DecisionQuotient.Quotient
import DecisionQuotient.Tractability.StructuralRank
import DecisionQuotient.Information
import DecisionQuotient.Information.RDSrank
import DecisionQuotient.Statistics.FisherInformation
import DecisionQuotient.Physics.WassersteinIntegrity

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
abbrev FR176 := @DecisionQuotient.DecisionProblem.quotient_is_coarsest
abbrev FR177 := @DecisionQuotient.DecisionProblem.quotient_has_unique_factorization
abbrev FR178 := @DecisionQuotient.DecisionProblem.srank_eq_relevant_card
abbrev FR179 := @DecisionQuotient.quotientEntropy_le_srank_binary
abbrev FR180 := @DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank
abbrev FR181 := @DecisionQuotient.Information.compression_below_srank_fails
abbrev FR182 := @DecisionQuotient.Information.srank_bits_sufficient
abbrev FR183 := @DecisionQuotient.Physics.single_future_zero_cost
abbrev FR184 := @DecisionQuotient.Physics.transportCost_pos_of_offDiag
abbrev FR185 := @no_closureInvariant_predicate_of_orbit_gap
abbrev FR186 := @no_admissibleNormalizationPredicate_decides_dominantPair
abbrev FR187 := @no_admissibleNormalizationPredicate_decides_marginBounded
abbrev FR188 := @no_admissibleNormalizationPredicate_decides_ghostActionTwoPairCrossOne
abbrev FR189 := @no_admissibleNormalizationPredicate_decides_offsetActionZeroPairCrossOne
abbrev FR190 := @admissibleCollapseLandscapeInfinity_full_paper
abbrev FR191 := @ClosureSoundPackage
abbrev FR192 := @ReasonableGuardrailPackage
abbrev FR193 := @no_closureSoundPackage_predicate_decides_four_obstruction_families
abbrev FR194 := @no_reasonableGuardrailPackage_predicate_decides_four_obstruction_families
abbrev FR195 := @ClosureClosedDomain
abbrev FR196 := @CorrectOnDomain
abbrev FR197 := @classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR198 := @no_correctOnDomain_classifier_of_orbit_gap
abbrev FR199 := @correct_classifier_inherits_closureLawInvariant
abbrev FR200 := @admissibleNormalizationPredicate_has_explicit_inhabitants
abbrev FR201 := @closureLawInvariant_of_iff_of_closureEquivalent
abbrev FR202 := @exists_orbit_gap_of_not_closureLawInvariant
abbrev FR203 := @closureLawInvariant_iff_no_orbit_gap
abbrev FR204 := @no_exact_closureLawInvariant_classifier_iff_exists_orbit_gap
noncomputable abbrev FR205 := @distinctActionCount
abbrev FR206 := @actionCount_profileCompressedSlice
abbrev FR207 := @decisionEquiv_profileCompressedSlice_iff
abbrev FR208 := @profileCompressedSlice_preserves_exactCertification
abbrev FR209 := @profileCompressedSlice_bounded_actions
abbrev FR210 := @OrbitGapOn
abbrev FR211 := @no_orbitGapOn_of_exact_classifiable_by_closureLawInvariant_onDomain
abbrev FR212 := @exact_classifiable_by_closureLawInvariant_onDomain_iff_no_orbitGapOn
abbrev FR213 := @no_exact_closureLawInvariant_classifier_onDomain_iff_orbitGapOn
abbrev FR347 := @closure_sound_package_no_go_uses_only_closureLawInvariant
abbrev FR348 := @fullBinaryPairwiseDomain_closure_closed
abbrev FR349 := @fullBinaryPairwiseDomain_contains_four_obstruction_families
abbrev FR350 := @no_correct_tractability_classifier_on_fullBinaryPairwiseDomain
abbrev FR214 := @boundedPatternScheme_holds_largeActionCount_iff
abbrev FR215 := @boundedPatternDefinable_eventually_constant_in_actionCount
abbrev FR216 := @boundedPatternDefinable_largeActionCount_agrees
abbrev FR217 := @payloadSufficient_iff_realizingProblem_isSufficient
abbrev FR218 := @payloadRelevant_iff_realizingProblem_isRelevant
abbrev FR219 := @payloadIrrelevant_iff_realizingProblem_isIrrelevant
abbrev FR220 := @payloadMinimalSufficient_iff_realizingProblem_isMinimalSufficient
abbrev FR221 := @payloadSufficientSets_eq_realizingProblem_sufficientSets
abbrev FR222 := @payloadRelevantSet_eq_realizingProblem_relevantSet
abbrev FR223 := @feasiblePayloadSufficient_iff_lawDecisionProblem_isSufficient
abbrev FR224 := @feasiblePayloadRelevant_iff_lawDecisionProblem_isRelevant
abbrev FR225 := @feasiblePayloadIrrelevant_iff_lawDecisionProblem_isIrrelevant
abbrev FR226 := @setValuedPayloadSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR227 := @setValuedPayloadRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR228 := @setValuedPayloadIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR229 := @outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR230 := @outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR231 := @outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR232 := @approximationSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR233 := @approximationSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR234 := @approximationSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR235 := @outputSemanticsSufficientSets_eq_of_outputSemanticsEquivalent
abbrev FR236 := @outputSemanticsRelevantSet_eq_of_outputSemanticsEquivalent
abbrev FR237 := @quotientMap_realizes_setoid_asOutputSemantics
abbrev FR238 := @quotientOutputSemanticsSufficient_iff_relationSufficient
abbrev FR239 := @quotientOutputSemanticsRelevant_iff_relationRelevant
abbrev FR240 := @exists_outputSemantics_realizing_setoid
abbrev FR241 := @outputSemanticsSetoid
abbrev FR242 := @SemanticallyExtensionalMap
abbrev FR243 := @semanticallyExtensionalMap_factors_through_outputSemanticsQuotient
abbrev FR244 := @semanticallyExtensionalClaim_factors_through_outputSemanticsQuotient
abbrev FR245 := @optimizerComputation_polytime_closureLawInvariant
abbrev FR246 := @optimizerSetPayload_polytime_closureLawInvariant
abbrev FR247 := @optimizerSetSearch_polytime_closureLawInvariant
abbrev FR248 := @optimizerComputation_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR249 := @optimizerComputation_polytime_classifier_agrees_on_closureEquivalent
abbrev FR250 := @no_correct_optimizerComputation_polytime_classifier_decides_dominantPair
abbrev FR251 := @no_admissibleNormalizationPredicate_optimizerComputation_polytime_and_dominantPair
abbrev FR252 := @CorrectnessForcesOrbitAgreementOnDomain
abbrev FR253 := @no_orbitGapOn_of_correct_classifier_onDomain_of_forcedOrbitAgreement
abbrev FR254 := @correct_classifier_onDomain_iff_no_orbitGapOn_of_forcedOrbitAgreement
abbrev FR255 := @no_correct_classifier_onDomain_iff_orbitGapOn_of_forcedOrbitAgreement
abbrev FR256 := @optimizerComputation_polytime_correct_classifier_onDomain_iff_no_orbitGapOn
abbrev FR257 := @optimizerSetPayloadPolytime
abbrev FR258 := @optimizerSetSearchPolytime
abbrev FR259 := @optimizerSetPayload_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR260 := @optimizerSetPayload_polytime_classifier_agrees_on_closureEquivalent
abbrev FR261 := @optimizerSetSearch_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR262 := @optimizerSetSearch_polytime_classifier_agrees_on_closureEquivalent
abbrev FR263 := @relevant_of_uniformApprox_of_strict_gap_witness
abbrev FR264 := @relevance_can_flip_under_arbitrarily_small_uniform_perturbation
abbrev FR265 := @not_decisionEquiv_of_uniformApprox_of_strict_gap_witness
abbrev FR266 := @not_sufficient_of_uniformApprox_of_strict_gap_witness
abbrev FR267 := @sufficiency_can_flip_under_arbitrarily_small_uniform_perturbation
abbrev FR268 := @pacGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR269 := @pacGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR270 := @pacGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR271 := @regretGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR272 := @regretGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR273 := @regretGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR274 := @statisticalRiskSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR275 := @statisticalRiskSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR276 := @statisticalRiskSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR277 := @anytimeGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR278 := @anytimeGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR279 := @anytimeGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR280 := @finiteHorizonGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR281 := @finiteHorizonGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR282 := @finiteHorizonGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR283 := @statisticalGuaranteeSemanticsSufficientSets_eq_of_outputSemanticsEquivalent
abbrev FR284 := @statisticalGuaranteeSemanticsRelevantSet_eq_of_outputSemanticsEquivalent
abbrev FR285 := @statisticalGuarantee_classifier_agrees_on_closureEquivalent_of_correctOnDomain_of_transfer
abbrev FR286 := @no_correctOnDomain_statisticalGuarantee_classifier_of_orbit_gap_of_transfer
abbrev FR287 := @statisticalGuarantee_correct_classifier_onDomain_iff_no_orbitGapOn_of_transfer
abbrev FR288 := @opt_eq_of_uniformApprox_of_uniformStrictGapCover
abbrev FR289 := @sufficientSets_eq_of_uniformApprox_of_uniformStrictGapCover
abbrev FR290 := @relevantSet_eq_of_uniformApprox_of_uniformStrictGapCover
abbrev FR291 := @randomizedGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
abbrev FR292 := @randomizedGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
abbrev FR293 := @randomizedGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
abbrev FR294 := @randomizedGuarantee_classifier_agrees_on_closureEquivalent_of_correctOnDomain_of_transfer
abbrev FR295 := @no_correctOnDomain_randomizedGuarantee_classifier_of_orbit_gap_of_transfer
abbrev FR296 := @randomizedGuarantee_correct_classifier_onDomain_iff_no_orbitGapOn_of_transfer
abbrev FR297 := @transferredSemantics_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR298 := @no_correctOnDomain_transferredSemantics_classifier_of_orbit_gap
abbrev FR299 := @transferredSemantics_correct_classifier_onDomain_iff_no_orbitGapOn
abbrev FR300 := @decisionEquiv_iff_of_uniformApprox_of_uniformStrictGapCover
abbrev FR301 := @isMinimalSufficient_iff_of_uniformApprox_of_uniformStrictGapCover
abbrev FR302 := @outputSemanticsExactRelevanceProfile_eq_of_outputSemanticsEquivalent
abbrev FR303 := @outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
abbrev FR304 := @agreeOn_univ_iff_eq_of_finiteIndicatorCoordinateSpace
abbrev FR305 := @finite_outputSemantics_allCoordinatesSufficient
abbrev FR306 := @finite_outputSemantics_realized_by_exactCertification
abbrev FR307 := @agreeOn_univ_iff_eq_of_singletonIdentityCoordinateSpace
abbrev FR308 := @outputSemantics_admits_coordinatePresentation
abbrev FR351 := @ExactCorrectnessSpecification.universal_scope_over_rigorously_specified_problems
abbrev FR352 := @DecisionQuotient.DecisionProblem.minimalSufficient_eq_relevant'
abbrev FR353 := @DecisionQuotient.DecisionProblem.sufficientSets_principal'
abbrev FR354 := @DecisionQuotient.srank_le_sufficient_card
abbrev FR355 := @closureLawInvariant_iff_closureHull_eq_self
abbrev FR356 := @no_orbitGapOn_iff_closureHull_disjoint
abbrev FR357 := @exact_classifiable_by_closureLawInvariant_onDomain_iff_closureHull_disjoint
abbrev FR358 := @closureHull_least_exact_classifier_onDomain_of_no_orbitGapOn
abbrev FR309 := @outputSemantics_realized_by_singletonMeasurableCoordinatePresentation
abbrev FR310 := @outputSemantics_realized_by_singletonContinuousCoordinatePresentation
abbrev FR311 := @encodable_stateSpace_admits_countableBooleanPresentation
abbrev FR312 := @encodable_discreteMeasurable_stateSpace_admits_measurable_countableBooleanPresentation
abbrev FR313 := @encodable_discreteTopological_stateSpace_admits_continuous_countableBooleanPresentation
abbrev FR314 := @agreeOn_univ_iff_eq_of_finiteBinaryCoordinateSpace
abbrev FR315 := @finite_exactSpecification_admits_lowDimBooleanPresentation
abbrev FR316 := @IdentityOutputClosureSpec.classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR317 := @IdentityOutputClosureSpec.no_correctOnDomain_classifier_of_orbit_gap
abbrev FR318 := @hypothesisOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR319 := @estimatorOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR320 := @policyOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR321 := @randomizedProcedure_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR322 := @no_correctOnDomain_hypothesisOutput_classifier_of_orbit_gap
abbrev FR323 := @no_correctOnDomain_estimatorOutput_classifier_of_orbit_gap
abbrev FR324 := @no_correctOnDomain_policyOutput_classifier_of_orbit_gap
abbrev FR325 := @no_correctOnDomain_randomizedProcedure_classifier_of_orbit_gap
abbrev FR326 := @TransportedOutputClosureSpec.classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR327 := @TransportedOutputClosureSpec.no_correctOnDomain_classifier_of_orbit_gap
abbrev FR328 := @IdentityOutputClosureSpec.correctOnDomain_iff_correctOnDomain_transport
abbrev FR329 := @representationRelativeHypothesis_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR330 := @representationRelativeEstimator_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR331 := @representationRelativePolicy_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR332 := @representationRelativeRandomizedProcedure_classifier_agrees_on_closureEquivalent_of_correctOnDomain
abbrev FR333 := @no_correctOnDomain_representationRelativeHypothesis_classifier_of_orbit_gap
abbrev FR334 := @no_correctOnDomain_representationRelativeEstimator_classifier_of_orbit_gap
abbrev FR335 := @no_correctOnDomain_representationRelativePolicy_classifier_of_orbit_gap
abbrev FR336 := @no_correctOnDomain_representationRelativeRandomizedProcedure_classifier_of_orbit_gap
abbrev FR337 := @ExactSemanticsSetoid
abbrev FR338 := @ExactSemanticsQuotient
noncomputable abbrev FR339 := @exactSemanticsDecisionProblem
abbrev FR340 := @booleanPayloadTransfer
abbrev FR341 := @predicateTransfer
abbrev FR342 := @sufficiency_is_relation_refinement
abbrev FR343 := @relevance_is_erased_failure_of_refinement
abbrev FR344 := @outputSemanticsExactRelevanceProfile_eq_exactSemanticsDecisionProblem
abbrev FR345 := @exactSemanticsQuotient_universal_characterization
abbrev FR346 := @exactnessMeansExactAgreementWithValidity
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
