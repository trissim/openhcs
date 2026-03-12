/-
  Compact Lean-handle aliases used in paper prose/tables.
  Implemented as namespace-level exports to preserve exact theorem constants.

  ## Triviality Level
  TRIVIAL: This is purely a re-export file. No proofs whatsoever.

  ## Dependencies
  - Chain: Depends on all main modules but exports them under simpler names
  - The nontrivial work is in the imported modules (ClaimClosure, Sigma2PHardness, etc.)
-/

import DecisionQuotient.Basic
import DecisionQuotient.ClaimClosure
import DecisionQuotient.IntegrityCompetence
import DecisionQuotient.HardnessDistribution
import DecisionQuotient.ToolCollapse
import DecisionQuotient.Hardness.Sigma2PHardness
import DecisionQuotient.PhysicalBudgetCrossover
import DecisionQuotient.Hardness.ConfigReduction
import DecisionQuotient.InteriorVerification
import DecisionQuotient.DecisionGeometry
import DecisionQuotient.UniverseObjective
import DecisionQuotient.Physics.Instantiation
import DecisionQuotient.Physics.AccessRegime
import DecisionQuotient.Physics.PhysicalHardness
import DecisionQuotient.Physics.DecisionTime
import DecisionQuotient.Physics.Conversation
import DecisionQuotient.Physics.ConstraintForcing
import DecisionQuotient.Physics.MeasureNecessity
import DecisionQuotient.Physics.AssumptionNecessity
import DecisionQuotient.Physics.ObserverRelativeState
import DecisionQuotient.Physics.AnchorChecks
import DecisionQuotient.Physics.PhysicalIncompleteness
import DecisionQuotient.Physics.ClaimTransport
import DecisionQuotient.Physics.Uncertainty
import DecisionQuotient.Physics.HeisenbergStrong
import DecisionQuotient.Physics.BoundedAcquisition
import DecisionQuotient.Physics.WolpertMismatch
import DecisionQuotient.Physics.WolpertConstraints
import DecisionQuotient.Physics.WolpertDecomposition
import DecisionQuotient.Physics.TUR
import DecisionQuotient.Physics.WassersteinIntegrity
import DecisionQuotient.Physics.TransportCost
import DecisionQuotient.Physics.InvariantAgreement
import DecisionQuotient.Physics.LocalityPhysics
import DecisionQuotient.GraphNontriviality
import DecisionQuotient.WitnessCheckingDuality
import DecisionQuotient.Summary
import DecisionQuotient.Dichotomy
import DecisionQuotient.Tractability.StructuralRank
import DecisionQuotient.Tractability.BoundedActions
import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Dimensional
import DecisionQuotient.Tractability.Tightness
import DecisionQuotient.Information
import DecisionQuotient.Information.RateDistortion
import DecisionQuotient.Information.RDSrank
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.InflationEntropyBridge
import DecisionQuotient.InflationEntropyMinimality
import DecisionQuotient.Statistics.FisherInformation
import DecisionQuotient.StochasticSequential.ClaimClosure
import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Quotient
import DecisionQuotient.StochasticSequential.SetValued
import DecisionQuotient.StochasticSequential.Hardness
import DecisionQuotient.StochasticSequential.Hierarchy
import DecisionQuotient.StochasticSequential.PolynomialReduction
import DecisionQuotient.StochasticSequential.ThermodynamicLift
import DecisionQuotient.BayesianBridge
import DecisionQuotient.BayesFromDQ
import DecisionQuotient.BayesFoundations
import DecisionQuotient.BayesOptimalityProof
import DecisionQuotient.FunctionalInformation
import DecisionQuotient.Quotient
import DecisionQuotient.AbstractionCollapse
import DecisionQuotient.Bridges

namespace DecisionQuotient

namespace CC
export DecisionQuotient.ClaimClosure (
  RegimeSimulation
  adq_ordering
  system_transfer_licensed_iff_snapshot
  anchor_sigma2p_complete_conditional
  anchor_sigma2p_reduction_core
  anchor_query_relation_false_iff
  anchor_query_relation_true_iff
  boundaryCharacterized_iff_exists_sufficient_subset
  bounded_actions_detectable
  bridge_boundary_represented_family
  bridge_failure_witness_non_one_step
  bridge_transfer_iff_one_step_class
  certified_total_bits_split_core
  cost_asymmetry_eth_conditional
  declaredBudgetSlice
  declaredRegimeFamily_complete
  declared_physics_no_universal_exact_certifier_core
  dichotomy_conditional
  epsilon_admissible_iff_raw_lt_certified_total_core
  exact_admissible_iff_raw_lt_certified_total_core
  exact_certainty_inflation_under_hardness_core
  exact_raw_eq_certified_iff_certainty_inflation_core
  exact_raw_only_of_no_exact_admissible_core
  explicit_assumptions_required_of_not_excused_core
  explicit_state_upper_core
  hard_family_all_coords_core
  horizonTwoWitness_immediate_empty_sufficient
  horizon_gt_one_bridge_can_fail_on_sufficiency
  information_barrier_opt_oracle_core
  information_barrier_state_batch_core
  information_barrier_value_entry_core
  integrity_resource_bound_for_sufficiency
  integrity_universal_applicability_core
  meta_coordinate_irrelevant_of_invariance_on_declared_slice
  meta_coordinate_not_relevant_on_declared_slice
  minsuff_collapse_core
  minsuff_collapse_to_conp_conditional
  minsuff_conp_complete_conditional
  no_auto_minimize_of_p_neq_conp
  no_exact_claim_admissible_under_hardness_core
  no_exact_claim_under_declared_assumptions_unless_excused_core
  no_exact_identifier_implies_not_boundary_characterized
  no_uncertified_exact_claim_core
  one_step_bridge
  oracle_lattice_transfer_as_regime_simulation
  physical_crossover_above_cap_core
  physical_crossover_core
  physical_crossover_hardness_core
  physical_crossover_policy_core
  process_bridge_failure_witness
  poseAnchorQuery
  pose_returns_anchor_query_object
  posed_anchor_checked_true_implies_truth
  posed_anchor_exact_claim_admissible_iff_competent
  posed_anchor_exact_claim_requires_evidence
  posed_anchor_no_competence_no_exact_claim
  posed_anchor_query_truth_iff_exists_anchor
  posed_anchor_query_truth_iff_exists_forall
  posed_anchor_signal_positive_certified_implies_admissible
  query_obstruction_boolean_corollary
  query_obstruction_finite_state_core
  regime_core_claim_proved
  regime_simulation_transfers_hardness
  reusable_heuristic_of_detectable
  selectorSufficient_not_implies_setSufficient
  separable_detectable
  snapshot_vs_process_typed_boundary
  standard_assumption_ledger_unpack
  stochastic_objective_bridge_can_fail_on_sufficiency
  subproblem_hardness_lifts_to_full
  subproblem_transfer_as_regime_simulation
  sufficiency_conp_complete_conditional
  sufficiency_conp_reduction_core
  sufficiency_iff_dq_ratio
  sufficiency_iff_projectedOptCover_eq_opt
  thermo_conservation_additive_core
  thermo_energy_carbon_lift_core
  thermo_eventual_lift_core
  thermo_hardness_bundle_core
  thermo_mandatory_cost_core
  tractable_bounded_core
  tractable_separable_core
  tractable_subcases_conditional
  tractable_tree_core
  transition_coupled_bridge_can_fail_on_sufficiency
  tree_structure_detectable
  typed_claim_admissibility_core
  typed_static_class_completeness
  universal_solver_framing_core
)
end CC

namespace IC
export DecisionQuotient.IntegrityCompetence (
  CertaintyInflation
  CompletionFractionDefined
  EvidenceForReport
  ExactCertaintyInflation
  Percent
  RLFFWeights
  ReportSignal
  ReportBitModel
  SignalConsistent
  admissible_irrational_strictly_more_than_rational
  admissible_matrix_counts
  abstain_signal_exists_with_guess_self
  certaintyInflation_iff_not_admissible
  certificationOverheadBits
  certificationOverheadBits_of_evidence
  certificationOverheadBits_of_no_evidence
  certifiedTotalBits
  certifiedTotalBits_ge_raw
  certifiedTotalBits_gt_raw_of_evidence
  certifiedTotalBits_of_evidence
  certifiedTotalBits_of_no_evidence
  claim_admissible_of_evidence
  competence_implies_integrity
  completion_fraction_defined_of_declared_bound
  epsilon_competence_implies_integrity
  evidence_nonempty_iff_claim_admissible
  evidence_of_claim_admissible
  exact_claim_admissible_iff_exact_evidence_nonempty
  exact_claim_requires_evidence
  exactCertaintyInflation_iff_no_exact_competence
  exact_raw_only_of_no_exact_admissible
  integrity_forces_abstention
  integrity_not_competent_of_nonempty_scope
  integrity_resource_bound
  no_completion_fraction_without_declared_bound
  overModelVerdict_rational_iff
  percentZero
  rlffBaseReward
  rlffReward
  rlff_abstain_strictly_prefers_no_certificates
  rlff_maximizer_has_evidence
  rlff_maximizer_is_admissible
  self_reflected_confidence_not_certification
  signal_certified_positive_implies_admissible
  signal_consistent_of_claim_admissible
  signal_no_evidence_forces_zero_certified
  signal_exact_no_competence_forces_zero_certified
  steps_run_scalar_always_defined
  steps_run_scalar_falsifiable
  zero_epsilon_competence_iff_exact
)
end IC

namespace HD
export DecisionQuotient.HardnessDistribution (
  centralization_dominance_bundle
  centralization_step_saves_n_minus_one
  centralized_higher_leverage
  complete_model_dominates_after_threshold
  gap_conservation_card
  generalizedTotal_with_saturation_eventually_constant
  generalized_dominance_can_fail_without_right_boundedness
  generalized_dominance_can_fail_without_wrong_growth
  generalized_right_dominates_wrong_of_bounded_vs_identity_lower
  generalized_right_eventually_dominates_wrong
  hardnessEfficiency_eq_central_share
  isRightHardness
  isWrongHardness
  linear_lt_exponential_plus_constant_eventually
  native_dominates_manual
  no_positive_slope_linear_represents_saturating
  requiredWork
  requiredWork_eq_affine_in_sites
  right_dominates_wrong
  saturatingSiteCost_eventually_constant
  simplicityTax_grows
  hardnessLowerBound
  hardness_is_irreducible_required_work
  totalDOF_eventually_constant_iff_zero_distributed
  totalDOF_ge_intrinsic
  totalExternalWork_eq_n_mul_gapCard
  workGrowthDegree
  workGrowthDegree_zero_iff_eventually_constant
)
end HD

namespace DP
export DecisionQuotient.DecisionProblem (
  minimalSufficient_iff_relevant
  relevantSet_is_minimal
  sufficient_implies_selectorSufficient
)
export DecisionQuotient.ClaimClosure.DecisionProblem (
  epsOpt_zero_eq_opt
  sufficient_iff_zeroEpsilonSufficient
)
end DP

namespace S2P
export DecisionQuotient.Sigma2PHardness (
  exactlyIdentifiesRelevant_iff_sufficient_and_subset_relevantFinset
  min_representationGap_zero_iff_relevant_card
  min_sufficient_set_iff_relevant_card
  representationGap
  representationGap_eq_waste_plus_missing
  representationGap_eq_zero_iff
  representationGap_missing_eq_gapCard
  representationGap_zero_iff_minimalSufficient
  sufficient_iff_relevant_subset
)
end S2P

namespace PBC
export DecisionQuotient.PhysicalBudgetCrossover (
  CrossoverAt
  SuccinctInfeasible
  SuccinctUnbounded
  explicit_infeasible_succinct_feasible_of_crossover
  exists_least_crossover_point
  has_crossover_of_bounded_succinct_unbounded_explicit
  explicit_eventual_infeasibility_of_monotone_and_witness
  crossover_eventually_of_eventual_split
  payoff_threshold_explicit_vs_succinct
  no_universal_survivor_without_succinct_bound
  policy_closure_at_divergence
  policy_closure_beyond_divergence
)
end PBC

namespace CR
export DecisionQuotient.ConfigReduction (
  config_sufficiency_iff_behavior_preserving
)
end CR

namespace IV
export DecisionQuotient.InteriorVerification (
  GoalClass
  GoalClass.classMonotoneOn
  GoalClass.isNonNegativelyTautologicalCoord
  GoalClass.isTautologicalCoord
  GoalClass.tautologicalSet
  InteriorDominanceVerifiable
  TautologicalSetIdentifiable
  agreeOnSet
  interiorParetoDominates
  interior_certificate_implies_non_rejection
  interior_dominance_implies_universal_non_rejection
  interior_dominance_not_full_sufficiency
  interior_verification_tractable_certificate
  not_sufficientOnSet_of_counterexample
  singleton_coordinate_interior_certificate
)
end IV

namespace DG
export DecisionQuotient (
  Outside
  anchoredSlice
  anchoredSliceEquivOutside
  card_outside_eq_sub
  card_anchoredSlice
  card_anchoredSlice_eq_pow_sub
  card_anchoredSlice_eq_uniform
  anchoredSlice_mul_fixed_eq_full
  constantBoolDP
  firstCoordDP
  constantBoolDP_opt
  firstCoordDP_opt
  constantBoolDP_empty_sufficient
  firstCoordDP_empty_not_sufficient
  boolHypercube_node_count
  node_count_does_not_determine_edge_geometry
)
export DecisionQuotient.DecisionProblem (
  edgeOnComplement
  edgeOnComplement_iff_not_sufficient
)
end DG

namespace DT
export DecisionQuotient.Physics.DecisionTime (
  TimedState
  DecisionProcess
  tick
  DecisionEvent
  TimeUnitStep
  time_is_discrete
  time_coordinate_falsifiable
  tick_increments_time
  tick_decision_witness
  tick_is_decision_event
  decision_event_implies_time_unit
  decision_taking_place_is_unit_of_time
  decision_event_iff_eq_tick
  run
  run_time_exact
  run_elapsed_time_eq_ticks
  decisionTrace
  decisionTrace_length_eq_ticks
  decision_count_equals_elapsed_time
  SubstrateKind
  SubstrateModel
  substrate_step_realizes_decision_event
  substrate_step_is_time_unit
  time_unit_law_substrate_invariant
)
end DT

namespace CV
export DecisionQuotient.Physics.Conversation (
  RecurrentCircuit
  CoupledConversation
  JointState
  tick
  tick_uses_shared_node
  tick_shared_is_merged_emissions
  BoundedChannel
  channelObserver
  channel_projection_eq_iff_quantized_eq
  clampBit
  clampChannel
  clamp_output_is_discrete
  clamp_projection_eq_iff_same_clamped_bit
  clampDecisionEvent
  clampDecisionBitOps
  clampDecisionBitOps_pos_of_event
  clampDecisionEvent_iff_bitOps_pos
  clamp_event_implies_positive_energy
  BinaryAnswer
  ConversationReport
  explanationGap
  explanationGap_add_explanationBits
  toClaimReport
  abstain_iff_no_answer
  yes_no_iff_exact_claim
  toClaimReport_ne_epsilon
  toReportSignal
  toReportSignal_declares_bound
  toReportSignal_completion_defined_of_budget
  toReportSignal_signal_consistent_zero_certified
  abstain_report_can_carry_explanation
)
end CV

namespace PI
export DecisionQuotient.Physics.PhysicalIncompleteness (
  UniverseModel
  PhysicallyInstantiated
  no_surjective_instantiation_of_card_gap
  physical_incompleteness_of_card_gap
  physical_incompleteness_of_bounds
  under_resolution_implies_collision
  under_resolution_implies_decision_collision
)
end PI

namespace CT
export DecisionQuotient.Physics.ClaimTransport (
  PhysicalEncoding
  physical_claim_lifts_from_core
  physical_claim_lifts_from_core_conditional
  physical_counterexample_yields_core_counterexample
  physical_counterexample_invalidates_core_rule
  no_physical_counterexample_of_core_theorem
  LawGapInstance
  lawGapEncoding
  lawGapPhysicalClaim
  law_gap_physical_claim_holds
  no_law_gap_counterexample
  physical_bridge_bundle
)
end CT

namespace IN
export DecisionQuotient.Physics.Instantiation (
  Geometry
  Dynamics
  Circuit
  geometry_plus_dynamics_is_circuit
  DecisionInterpretation
  DecisionCircuit
  Molecule
  Reaction
  ReactionOutcome
  MoleculeGeometry
  MoleculeDynamics
  MoleculeCircuit
  MoleculeAsCircuit
  MoleculeAsDecisionCircuit
  molecule_decision_preserves_geometry
  molecule_decision_preserves_dynamics
  asDecisionCircuit
  asDecisionCircuit_preserves_circuit
  Configuration
  EnergyLandscape
  k_Boltzmann
  LandauerBound
  law_objective_schema
  law_opt_eq_feasible_of_gap
  law_opt_singleton_of_deterministic
)
end IN

namespace ARG
export PhysicalComplexity.AccessRegime (
  PhysicalDevice
  AccessRegime
  RegimeEval
  RegimeSample
  RegimeProof
  RegimeWithCertificate
  RegimeEvalOn
  RegimeSampleOn
  RegimeProofOn
  RegimeWithCertificateOn
  HardUnderEval
  AuditableWithCertificate
  certificate_upgrades_regime
  certificate_upgrades_regime_on
  physical_succinct_certification_hard
  certificate_amortizes_hardness
  regime_upgrade_with_certificate
  regime_upgrade_with_certificate_on
  AccessChannelLaw
  FiveWayMeet
)
end ARG

namespace PH
export PhysicalComplexity (
  k_Boltzmann
  PhysicalComputer
  bit_energy_cost
  Landauer_bound
  InstanceSize
  ComputationalRequirement
  coNP_requirement
  coNP_physically_impossible
  coNP_not_in_P_physically
  sufficiency_physically_impossible
  PhysicalCollapseAtRequirement
  no_physical_collapse_at_requirement
  canonical_physical_collapse_impossible
  p_eq_np_physically_impossible_of_collapse_map
  p_eq_np_physically_impossible_canonical
  CoherentDQRejectionAtRequirement
  coherent_dq_rejection_impossible_at_requirement
  coherent_dq_rejection_impossible_canonical
)
end PH

namespace TC
export DecisionQuotient.ToolCollapse (
  WorkProfile
  WorkModel
  ToolModel
  EventualStrictImprovement
  EffectiveCollapse
  tool_never_worse
  strict_improvement_has_witness
  effective_collapse_of_eventual_strict
  expBaseline
  linearTool
  linear_tool_eventual_strict
  linear_tool_effective_collapse
)
end TC

namespace UQ
export DecisionQuotient.Physics.Uncertainty (
  binaryIdentityProblem
  binaryIdentityProblem_opt_true
  binaryIdentityProblem_opt_false
  exists_decision_problem_with_nontrivial_opt
  PhysicalNontrivialOptAssumption
  exists_decision_problem_with_nontrivial_opt_of_physical
)
end UQ

namespace HS
export DecisionQuotient.Physics.HeisenbergStrong (
  NoisyPhysicalEncoding
  HeisenbergStrongBinding
  strong_binding_implies_core_nontrivial
  strong_binding_yields_core_encoding_witness
  strong_binding_implies_physical_nontrivial_opt_assumption
  strong_binding_implies_nontrivial_opt_via_uncertainty
)
end HS

namespace WD
export DecisionQuotient (
  witnessBudgetEmpty
  checkingBudgetPairs
  checking_witnessing_duality_budget
  no_sound_checker_below_witness_budget
  checking_time_ge_witness_budget
)
end WD

namespace UO
export DecisionQuotient (
  UniverseDynamics
  feasibleActions
  lawDecisionProblem
  lawUtility
  logicallyDeterministic
  universe_sets_objective_schema
  lawUtility_eq_of_allowed_iff
  opt_eq_feasible_of_gap
  infeasible_not_optimal_of_gap
  opt_singleton_of_logicallyDeterministic
  opt_eq_of_allowed_iff
)
end UO

-- NOTE: Do NOT use "DQ" as a namespace - it conflicts with auto-generated DQ### IDs.
namespace CCC
export DecisionQuotient.CC (
  anchor_sigma2p_complete_conditional
  cost_asymmetry_eth_conditional
  dichotomy_conditional
  minsuff_collapse_to_conp_conditional
  minsuff_conp_complete_conditional
  sufficiency_conp_complete_conditional
  tractable_subcases_conditional
)
end CCC

/-! ## Stochastic/Sequential Hierarchy (DC prefix) -/

-- DC: Dichotomy/Complexity claims from StochasticSequential
abbrev DC1 := StochasticSequential.static_stochastic_strict_separation
abbrev DC2 := StochasticSequential.stochastic_sequential_strict_separation
abbrev DC3 := StochasticSequential.complexity_dichotomy_hierarchy
abbrev DC4 := StochasticSequential.regime_hierarchy
abbrev DC5 := StochasticSequential.coNP_subset_PP
abbrev DC6 := StochasticSequential.PP_subset_PSPACE
abbrev DC7 := StochasticSequential.coNP_subset_PSPACE
abbrev DC8 := StochasticSequential.static_to_coNP
abbrev DC9 := StochasticSequential.stochastic_to_PP
abbrev DC10 := StochasticSequential.sequential_to_PSPACE
abbrev DC11 := StochasticSequential.ClaimClosure.claim_six_subcases
abbrev DC12 := StochasticSequential.ClaimClosure.claim_hierarchy
abbrev DC13 := StochasticSequential.ClaimClosure.claim_tractable_subcases_to_P
abbrev DC14 := StochasticSequential.stochastic_dichotomy
abbrev DC15 := StochasticSequential.above_threshold_hard
abbrev DC16 := @StochasticSequential.StochasticAnchorSufficient
abbrev DC17 := @StochasticSequential.StochasticAnchorSufficiencyCheck
abbrev DC18 := @StochasticSequential.stochastic_anchor_check_iff
abbrev DC19 := @StochasticSequential.stochastic_anchor_sufficient_of_stochastic_sufficient
abbrev DC20 := @StochasticSequential.SequentialAnchorSufficient
abbrev DC21 := @StochasticSequential.SequentialAnchorSufficiencyCheck
abbrev DC22 := @StochasticSequential.sequential_anchor_check_iff
abbrev DC23 := @StochasticSequential.sequential_anchor_sufficient_of_sequential_sufficient
abbrev DC24 := @StochasticSequential.StochasticAnchorCheckInstance
abbrev DC25 := @StochasticSequential.reduceMAJSAT_correct_anchor_strict
abbrev DC26 := @StochasticSequential.reduceMAJSAT_to_stochastic_anchor_reduction
abbrev DC27 := @StochasticSequential.SequentialAnchorCheckInstance
abbrev DC28 := @StochasticSequential.reduceTQBF_correct_anchor
abbrev DC29 := @StochasticSequential.reduceTQBF_to_sequential_anchor_reduction
abbrev DC30 := @StochasticSequential.StatePotential
abbrev DC31 := @StochasticSequential.utilityFromPotentialDrop_le_iff_nextPotential_ge
abbrev DC32 := @StochasticSequential.utility_from_action_state_potential
abbrev DC33 := @StochasticSequential.stochasticExpectedUtility_eq_neg_expectedActionPotential
abbrev DC34 := @StochasticSequential.stochasticExpectedUtility_le_iff_expectedActionPotential_ge
abbrev DC35 := @StochasticSequential.landauerEnergyFloor_nonneg
abbrev DC36 := @StochasticSequential.landauerEnergyFloor_mono_bits
abbrev DC37 := @StochasticSequential.thermodynamicCost_eq_landauerEnergyFloorRoom_states

-- SS: Stochastic/Sequential completeness theorems (polymorphic - instantiated in ClaimClosure)
-- SS1-SS5 are polymorphic theorems, referenced via DC handles for paper mapping

/-! ## Conversation Physics (CV) handles -/

abbrev CV1 := @Physics.Conversation.RecurrentCircuit
abbrev CV2 := @Physics.Conversation.CoupledConversation
abbrev CV3 := @Physics.Conversation.JointState
abbrev CV4 := @Physics.Conversation.tick_uses_shared_node
abbrev CV5 := @Physics.Conversation.tick_shared_is_merged_emissions
abbrev CV6 := @Physics.Conversation.channel_projection_eq_iff_quantized_eq
abbrev CV7 := @Physics.Conversation.clamp_projection_eq_iff_same_clamped_bit
abbrev CV8 := @Physics.Conversation.clampDecisionEvent_iff_bitOps_pos
abbrev CV9 := @Physics.Conversation.clamp_event_implies_positive_energy
abbrev CV10 := @Physics.Conversation.explanationGap_add_explanationBits
abbrev CV11 := @Physics.Conversation.toClaimReport
abbrev CV12 := @Physics.Conversation.abstain_iff_no_answer
abbrev CV13 := @Physics.Conversation.yes_no_iff_exact_claim
abbrev CV14 := @Physics.Conversation.toReportSignal_completion_defined_of_budget
abbrev CV15 := @Physics.Conversation.toReportSignal_signal_consistent_zero_certified
abbrev CV16 := @Physics.Conversation.abstain_report_can_carry_explanation

/-! ## Physics Claims Handle Abbreviations -/

-- Decision Equivalence (DE) handles
abbrev DE1 := ClaimClosure.DE1
abbrev DE2 := ClaimClosure.DE2
abbrev DE3 := ClaimClosure.DE3
abbrev DE4 := ClaimClosure.DE4

-- Molecular Integrity (MI) handles
abbrev MI1 := ClaimClosure.MI1
abbrev MI2 := ClaimClosure.MI2
abbrev MI3 := ClaimClosure.MI3
abbrev MI4 := ClaimClosure.MI4
abbrev MI5 := ClaimClosure.MI5

-- Self-Reference (SR) handles
abbrev SR1 := ClaimClosure.SR1
abbrev SR2 := ClaimClosure.SR2
abbrev SR3 := ClaimClosure.SR3
abbrev SR4 := ClaimClosure.SR4
abbrev SR5 := ClaimClosure.SR5

-- Information Access (IA) handles
abbrev IA1 := ClaimClosure.IA1
abbrev IA2 := ClaimClosure.IA2
abbrev IA3 := ClaimClosure.IA3
abbrev IA4 := ClaimClosure.IA4
abbrev IA5 := ClaimClosure.IA5
abbrev IA6 := ClaimClosure.IA6
abbrev IA7 := ClaimClosure.IA7
abbrev IA8 := ClaimClosure.IA8
abbrev IA9 := ClaimClosure.IA9
abbrev IA10 := ClaimClosure.IA10
abbrev IA11 := ClaimClosure.IA11
abbrev IA12 := ClaimClosure.IA12
abbrev IA13 := ClaimClosure.IA13
abbrev IA14 := @Physics.InvariantAgreement.IA14_observer_disagreement_impossible
abbrev IA15 := @Physics.InvariantAgreement.sameUniverse
abbrev IA16 := @Physics.InvariantAgreement.IA16_no_invariant_undefined_membership
abbrev IA17 := @Physics.InvariantAgreement.IA17_ego_trap
abbrev IA18 := @Physics.InvariantAgreement.IA18_escalation_complete

-- Gap Energy (GE) handles
abbrev GE1 := ClaimClosure.GE1
abbrev GE2 := ClaimClosure.GE2
abbrev GE3 := ClaimClosure.GE3
abbrev GE4 := ClaimClosure.GE4
abbrev GE5 := ClaimClosure.GE5
abbrev GE6 := ClaimClosure.GE6
abbrev GE7 := ClaimClosure.GE7
abbrev GE8 := ClaimClosure.GE8
abbrev GE9 := ClaimClosure.GE9

-- Integrity Equilibrium (IE) handles
abbrev IE1 := ClaimClosure.IE1
abbrev IE2 := ClaimClosure.IE2
abbrev IE3 := ClaimClosure.IE3
abbrev IE4 := ClaimClosure.IE4
abbrev IE5 := ClaimClosure.IE5
abbrev IE6 := ClaimClosure.IE6
abbrev IE7 := ClaimClosure.IE7
abbrev IE8 := ClaimClosure.IE8
abbrev IE9 := ClaimClosure.IE9
abbrev IE10 := ClaimClosure.IE10
abbrev IE11 := ClaimClosure.IE11
abbrev IE12 := ClaimClosure.IE12
abbrev IE13 := ClaimClosure.IE13
abbrev IE14 := ClaimClosure.IE14
abbrev IE15 := ClaimClosure.IE15
abbrev IE16 := ClaimClosure.IE16
abbrev IE17 := ClaimClosure.IE17

-- Substrate Energy (SE) handles
abbrev SE1 := ClaimClosure.SE1
abbrev SE2 := ClaimClosure.SE2
abbrev SE3 := ClaimClosure.SE3
abbrev SE4 := ClaimClosure.SE4
abbrev SE5 := ClaimClosure.SE5
abbrev SE6 := ClaimClosure.SE6

-- Channel (CH) handles
abbrev CH1 := ClaimClosure.CH1
abbrev CH2 := ClaimClosure.CH2
abbrev CH3 := ClaimClosure.CH3
abbrev CH5 := ClaimClosure.CH5
abbrev CH6 := ClaimClosure.CH6

-- Atomic/Orbital (AC) handles
abbrev AC1 := ClaimClosure.AtomicCircuitExports.AC1
abbrev AC3 := ClaimClosure.AtomicCircuitExports.AC3
abbrev AC4 := ClaimClosure.AtomicCircuitExports.AC4
abbrev AC5 := ClaimClosure.AtomicCircuitExports.AC5
abbrev AC6 := ClaimClosure.AtomicCircuitExports.AC6
abbrev AC8 := ClaimClosure.AtomicCircuitExports.AC8
abbrev AC9 := ClaimClosure.AtomicCircuitExports.AC9
abbrev AC11 := ClaimClosure.AtomicCircuitExports.AC11

-- Discrete State (DS) handles
abbrev DS1 := ClaimClosure.DS1
abbrev DS2 := ClaimClosure.DS2
abbrev DS3 := ClaimClosure.DS3
abbrev DS4 := ClaimClosure.DS4
abbrev DS5 := ClaimClosure.DS5
abbrev DS6 := ClaimClosure.DS6

-- Decision Quotient (DQ) handles from IntegrityEquilibrium
abbrev DQ1 := ClaimClosure.DQ1  -- Mutual information
abbrev DQ2 := ClaimClosure.DQ2  -- DecisionQuotient structure
abbrev DQ3 := ClaimClosure.DQ3  -- DQ from gap
abbrev DQ4 := ClaimClosure.DQ4  -- Zero gap = DQ 1
abbrev DQ5 := ClaimClosure.DQ5  -- Max gap = DQ 0
abbrev DQ6 := ClaimClosure.DQ6  -- DQ + Gap = Total
abbrev DQ7 := ClaimClosure.DQ7  -- DQ monotonic
abbrev DQ8 := ClaimClosure.DQ8  -- DQ thermodynamic interpretation

-- Decision Problem (DP) additional handles
abbrev DP6 := ClaimClosure.DP6  -- Empty-set sufficiency iff constant
abbrev DP7 := ClaimClosure.DP7  -- Non-sufficiency iff counterexample
abbrev DP8 := ClaimClosure.DP8  -- Empty-set non-sufficiency iff distinct opt

-- Query Complexity (QC) handles - polymorphic theorems referenced via CC handles for paper mapping
-- QC1-QC7 are polymorphic theorems, paper uses CC49, CC50 for query obstruction

-- Bayesian Bridge (BB) handles - connecting Bayes to Decision Quotient
abbrev BB1 := BayesianDQ
abbrev BB2 := BayesianDQ.certaintyGain
abbrev BB3 := dq_is_bayesian_certainty_fraction
abbrev BB4 := bayesian_dq_matches_physics_dq
abbrev BB5 := dq_derived_from_bayes

-- Physical no-go extension (PH) handles - P=NP transfer schema
abbrev PH11 := @PhysicalComplexity.PhysicalCollapseAtRequirement
abbrev PH12 := @PhysicalComplexity.no_physical_collapse_at_requirement
abbrev PH13 := @PhysicalComplexity.canonical_physical_collapse_impossible
abbrev PH14 := @PhysicalComplexity.p_eq_np_physically_impossible_of_collapse_map
abbrev PH15 := @PhysicalComplexity.p_eq_np_physically_impossible_canonical
abbrev PH16 := @PhysicalComplexity.P_eq_NP_via_SAT
abbrev PH17 := @PhysicalComplexity.SAT3ReductionBridge
abbrev PH18 := @PhysicalComplexity.sat_reduction_transfers_energy_lower_bound
abbrev PH19 := @PhysicalComplexity.physical_collapse_of_polytime_sat_realization
abbrev PH20 := @PhysicalComplexity.p_eq_np_physically_impossible_via_sat_bridge
abbrev PH21 := @PhysicalComplexity.SAT3HardFamily
abbrev PH22 := @PhysicalComplexity.p_eq_np_physically_impossible_via_sat_hard_family
abbrev PH23 := @PhysicalComplexity.collapse_possible_without_positive_bit_cost
abbrev PH24 := @PhysicalComplexity.collapse_possible_without_exponential_lower_bound
abbrev PH25 := @PhysicalComplexity.no_go_transfer_requires_collapse_map
abbrev PH26 := @PhysicalComplexity.no_collapse_of_bounded_budget_pos_cost_exp_lb
abbrev PH27 := @PhysicalComplexity.collapse_implies_assumption_failure_disjunction
abbrev PH28 := @PhysicalComplexity.deterministic_no_physical_collapse
abbrev PH29 := @PhysicalComplexity.probabilistic_no_physical_collapse
abbrev PH30 := @PhysicalComplexity.sequential_no_physical_collapse
abbrev PH31 := @PhysicalComplexity.collapse_possible_with_unbounded_budget_profile
abbrev PH32 := @PhysicalComplexity.exp_budget_profile_unbounded
abbrev PH33 := @PhysicalComplexity.finite_budget_assumption_is_necessary
abbrev PH34 := @PhysicalComplexity.CoherentDQRejectionAtRequirement
abbrev PH35 := @PhysicalComplexity.coherent_dq_rejection_impossible_at_requirement
abbrev PH36 := @PhysicalComplexity.coherent_dq_rejection_impossible_canonical

-- Bayes-from-DQ (BF) handles - nondegenerate-prior forcing before Bayes uniqueness
abbrev BF1 := @certainty_of_not_nondegenerateBelief
abbrev BF2 := @nondegenerateBelief_of_uncertaintyForced
abbrev BF3 := @forced_action_under_uncertainty
abbrev BF4 := @bayes_update_exists_of_nondegenerateBelief

-- Constraint-forcing (CF) handles - logic/time scaffold -> law/decision/thermo forcing
abbrev CF1 := @Physics.ConstraintForcing.laws_not_determined_of_parameter_separation
abbrev CF2 := @Physics.ConstraintForcing.logic_time_not_sufficient_for_unique_law
abbrev CF3 := @Physics.ConstraintForcing.laws_determined_implies_objective_determined
abbrev CF4 := @Physics.ConstraintForcing.objective_not_determined_of_parameter_separation
abbrev CF5 := @Physics.ConstraintForcing.forcedDecisionBits_pos_of_deadline
abbrev CF6 := @Physics.ConstraintForcing.actionForced_of_deadline
abbrev CF7 := @Physics.ConstraintForcing.nondegenerateBelief_of_deadline_and_uncertainty
abbrev CF8 := @Physics.ConstraintForcing.forced_decision_implies_positive_landauer_cost
abbrev CF9 := @Physics.ConstraintForcing.forced_decision_implies_positive_nv_work

-- Measure-necessity (MN) handles - quantitative vs stochastic typing requirements
abbrev MN1 := @Physics.MeasureNecessity.quantitative_claim_has_measure
abbrev MN2 := @Physics.MeasureNecessity.stochastic_claim_has_probability_measure
abbrev MN3 := @Physics.MeasureNecessity.stochastic_claim_has_measure
abbrev MN4 := @Physics.MeasureNecessity.count_univ_bool
abbrev MN5 := @Physics.MeasureNecessity.counting_measure_not_probability_on_bool
abbrev MN6 := @Physics.MeasureNecessity.deterministic_dirac_is_probability
abbrev MN7 := @Physics.MeasureNecessity.quantitative_value_depends_on_measure
abbrev MN8 := @Physics.MeasureNecessity.deterministic_models_still_measure_based
abbrev MN9 := @Physics.MeasureNecessity.measure_does_not_imply_probability
abbrev MN10 := @Physics.MeasureNecessity.quantitative_measure_is_logical_prerequisite
abbrev MN11 := @Physics.MeasureNecessity.stochastic_probability_is_logical_prerequisite

-- Assumption-necessity meta (AN) handles - universal sound-system schema
abbrev AN1 := @Physics.AssumptionNecessity.no_assumption_free_proof_of_refutable_claim
abbrev AN2 := @Physics.AssumptionNecessity.countermodel_violates_some_assumption
abbrev AN3 := @Physics.AssumptionNecessity.physical_claim_requires_physical_assumption
abbrev AN4 := @Physics.AssumptionNecessity.physical_claim_requires_empirically_justified_physical_assumption
abbrev AN5 := @Physics.AssumptionNecessity.strong_physical_no_go_meta

-- Physical-state universality (PS) handles - generic physical-state semantics
abbrev PS1 := @Physics.ClaimTransport.PhysicalStateSemantics
abbrev PS2 := @Physics.ClaimTransport.physical_state_has_witness
abbrev PS3 := @Physics.ClaimTransport.physical_state_claim_of_instance_claim
abbrev PS4 := @Physics.ClaimTransport.physical_state_claim_of_universal_core

-- Observer-relative physical state-space (OR) handles
abbrev OR1 := @Physics.ObserverRelativeState.ObserverClass
abbrev OR2 := @Physics.ObserverRelativeState.obsEquiv
abbrev OR3 := @Physics.ObserverRelativeState.EffectiveStateSpace
abbrev OR4 := @Physics.ObserverRelativeState.project_eq_iff
abbrev OR5 := @Physics.ObserverRelativeState.observer_relative_equivalence_witness
abbrev OR6 := @Physics.ObserverRelativeState.PhysicalObserverClass
abbrev OR7 := @Physics.ObserverRelativeState.PhysicalStateSpace
abbrev OR8 := @Physics.ObserverRelativeState.physical_state_space_has_instance_witness
abbrev OR9 := @Physics.ObserverRelativeState.physical_observer_relative_effective_space

-- Physical anchor-check (PA) handles - observer collapse to anchor-check implications
abbrev PA1 := @Physics.AnchorChecks.obsEquiv_all_of_effective_subsingleton
abbrev PA2 := @Physics.AnchorChecks.stochasticAnchorSufficient_iff_exists_anchor_singleton
abbrev PA3 := @Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton
abbrev PA4 := @Physics.AnchorChecks.stochastic_sufficient_of_observer_collapse_and_seed
abbrev PA5 := @Physics.AnchorChecks.stochastic_anchor_check_of_observer_collapse_and_seed
abbrev PA6 := @Physics.AnchorChecks.sequential_sufficient_of_observer_collapse
abbrev PA7 := @Physics.AnchorChecks.sequential_anchor_check_of_observer_collapse
abbrev PA8 := @Physics.AnchorChecks.physical_observer_collapse_implies_obsEquiv_all
abbrev PA9 := @Physics.AnchorChecks.physical_stochastic_anchor_check_of_observer_collapse_and_seed

-- Graph nontriviality (GN) handles - cycles, observer surprisal, Bayes/DQ/physics bridge
abbrev GN1 := @LogicGraph.isCycle
abbrev GN2 := @LogicGraph.cycleWitnessBits_pos
abbrev GN3 := @LogicGraph.pathProb_nonneg
abbrev GN4 := @LogicGraph.pathSurprisal_nonneg_of_positive_mass
abbrev GN5 := @LogicGraph.nontrivialityScore_unknown
abbrev GN6 := @LogicGraph.observerEntropy_nonneg
abbrev GN7 := @LogicGraph.dqFromEntropy_in_unit_interval
abbrev GN8 := @LogicGraph.path_belief_forced_under_uncertainty
abbrev GN9 := @LogicGraph.bayes_update_exists_for_observer_paths
abbrev GN10 := @LogicGraph.cycle_witness_implies_positive_landauer
abbrev GN11 := @LogicGraph.cycle_witness_implies_positive_nv_work
abbrev GN12 := @LogicGraph.dna_erasure_implies_positive_landauer
abbrev GN13 := @LogicGraph.dna_room_temp_environmental_stability

-- Foundations (FN) handles - machine-checked Bayes optimality chain
-- (BayesOptimalityProof: zero axioms, zero sorry)
abbrev FN7 := @BayesOptimalityProof.KL_nonneg
abbrev FN8 := @BayesOptimalityProof.entropy_le_crossEntropy
abbrev FN12 := @BayesOptimalityProof.crossEntropy_eq_entropy_add_KL
abbrev FN14 := @BayesOptimalityProof.bayes_is_optimal
abbrev FN15 := @BayesOptimalityProof.KL_eq_sum_klFun
abbrev FN16 := @BayesOptimalityProof.KL_eq_zero_iff_eq

-- Functional-information (FI) handles - counting and thermodynamic derivations
noncomputable abbrev FI1 := @FunctionalInformation.functionalFraction
noncomputable abbrev FI2 := @FunctionalInformation.functionalInformationBits
abbrev FI3 := @FunctionalInformation.functional_information_from_counting
noncomputable abbrev FI4 := @FunctionalInformation.minimumFunctionalEnergy
noncomputable abbrev FI5 := @FunctionalInformation.functionalInformationBitsFromEnergy
abbrev FI6 := @FunctionalInformation.functional_information_from_thermodynamics
abbrev FI7 := @FunctionalInformation.first_principles_thermo_coincide

-- Thermodynamic-lift (TL) handles
-- TL1-TL4 expose the main floor-based statements used by the artifact.
-- TL5-TL6 retain the exact-calibration equalities as explicit idealized specializations.
noncomputable abbrev TL1 := @ThermodynamicLift.landauerJoulesPerBit
abbrev TL2 := @ThermodynamicLift.landauerJoulesPerBit_pos
abbrev TL3 := @ThermodynamicLift.joulesPerBit_pos_of_landauer_floor
abbrev TL4 := @ThermodynamicLift.energy_lower_mandatory_of_landauer_floor
abbrev TL5 := @ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration
abbrev TL6 := @ThermodynamicLift.energy_lower_mandatory_of_landauer_calibration

-- Wolpert-constraint (WC) handles - explicit floor-plus-overhead interface
abbrev WC1 := @Physics.WolpertConstraints.landauer_floor_plus_overhead_lower_bound
abbrev WC2 := @Physics.WolpertConstraints.effective_model_dominates_landauer_floor
abbrev WC3 := @Physics.WolpertConstraints.effective_model_strictly_exceeds_landauer_of_strict_overhead
abbrev WC4 := @Physics.WolpertConstraints.energy_lower_bound_mono_under_overhead
abbrev WC5 := @Physics.WolpertConstraints.physical_grounding_bundle_with_wolpert_overhead

-- Tool-collapse (TC) handles - model-relative leverage collapse
abbrev TC1 := @ToolCollapse.WorkProfile
abbrev TC2 := @ToolCollapse.WorkModel
abbrev TC3 := @ToolCollapse.ToolModel
abbrev TC4 := @ToolCollapse.EventualStrictImprovement
abbrev TC5 := @ToolCollapse.EffectiveCollapse
abbrev TC6 := @ToolCollapse.tool_never_worse
abbrev TC7 := @ToolCollapse.strict_improvement_has_witness
abbrev TC8 := @ToolCollapse.effective_collapse_of_eventual_strict
abbrev TC9 := @ToolCollapse.expBaseline
abbrev TC10 := @ToolCollapse.linearTool
abbrev TC11 := @ToolCollapse.linear_tool_eventual_strict
abbrev TC12 := @ToolCollapse.linear_tool_effective_collapse

-- Anchor-query node (AQ) handles
abbrev AQ1 := @ClaimClosure.AQ1
abbrev AQ2 := @ClaimClosure.AQ2
abbrev AQ3 := @ClaimClosure.AQ3
abbrev AQ4 := @ClaimClosure.AQ4
abbrev AQ5 := @ClaimClosure.AQ5
abbrev AQ6 := @ClaimClosure.AQ6
abbrev AQ7 := @ClaimClosure.AQ7
abbrev AQ8 := @ClaimClosure.AQ8

/-! ## First-Principles Introduction Theorems (IT, EI, BA, FS)

These handles support the 8 core first-principles theorems in the paper introduction.
The derivation chain is:
  1. Counting Gap (BA10) - pure math first principle
  2. Bounded Acquisition (BA1-BA9) - SR + QM bridges
  3. Entropy-Rank Inequality (IT1-IT3) - information theory
  4. Energy-Information Duality (EI1-EI2) - Landauer lift
  5. Fisher-Rank Identity (FS1-FS3) - statistical geometry

TD (Thermodynamics) is NOT a first principle - it supplies the empirical value
ε = k_B T ln 2 per Landauer, which calibrates the Counting Gap.
-/

-- BA: Bounded Acquisition (Physics/BoundedAcquisition.lean)
-- BA10 is the Counting Gap Theorem - the primary first principle
abbrev BA1 := @Physics.BoundedAcquisition.BoundedRegion
abbrev BA2 := @Physics.BoundedAcquisition.acquisition_rate_bound
abbrev BA3 := @Physics.BoundedAcquisition.acquisitions_are_transitions
abbrev BA4 := @Physics.BoundedAcquisition.one_bit_per_transition
abbrev BA5 := @Physics.BoundedAcquisition.resolution_reads_sufficient
abbrev BA6 := @Physics.BoundedAcquisition.srank_le_resolution_bits
abbrev BA7 := @Physics.BoundedAcquisition.energy_ge_srank_cost
abbrev BA8 := @Physics.BoundedAcquisition.srank_one_energy_minimum
abbrev BA9 := @Physics.BoundedAcquisition.physical_grounding_bundle
abbrev BA10 := @Physics.BoundedAcquisition.counting_gap_theorem

-- IT: Information Theory (Information.lean)
-- IT3 is the Entropy-Rank Inequality: H(D) ≤ srank
abbrev IT1 := @DecisionProblem.numOptClasses
noncomputable abbrev IT2 := @DecisionProblem.quotientEntropy
abbrev IT3 := @quotientEntropy_le_srank_binary
abbrev IT4 := @numOptClasses_le_pow_srank_binary
abbrev IT5 := @nontrivial_bounds_binary
abbrev IT6 := @nontrivial_implies_srank_pos

-- EI: Energy-Information Duality (ThermodynamicLift.lean)
-- EI1 is the main Energy-Information theorem: E ≥ k_B T · H_nats
abbrev EI1 := @ThermodynamicLift.energy_ge_kbt_nat_entropy
abbrev EI2 := @ThermodynamicLift.energy_ground_state_tracks_entropy
noncomputable abbrev EI3 := @ThermodynamicLift.landauerJoulesPerBit
abbrev EI4 := @ThermodynamicLift.landauerJoulesPerBit_pos
abbrev EI5 := @ThermodynamicLift.neukart_vinokur_duality

-- FS: Fisher Statistics (Statistics/FisherInformation.lean)
-- FS1 is the Fisher-Rank Identity: sum of Fisher scores = srank
abbrev FS1 := @Statistics.sum_fisherScore_eq_srank
abbrev FS2 := @Statistics.fisherMatrix_rank_eq_srank
abbrev FS3 := @Statistics.fisherMatrix_trace_eq_srank
abbrev FS4 := @Statistics.fisherScore_relevant
abbrev FS5 := @Statistics.fisherScore_irrelevant

/-! ## First-Principles Introduction Additions

Additional novel first-principles theorems for the introduction:
- QT: Quotient Theory (Universal Property / Theorem A)
- AB: Abstraction Boundary (collapse beyond the quotient)
- WD: Witness-Checking Duality (certificate lower bounds)
- BC: Bayes from Counting (probability axioms from counting measure)
-/

-- QT: Quotient Theory (Quotient.lean)
-- QT1 is Theorem A: the Decision Quotient is the coarsest abstraction preserving Opt
abbrev QT1 := @DecisionProblem.quotient_is_coarsest
abbrev QT2 := @DecisionProblem.quotientMap_preservesOpt
abbrev QT3 := @DecisionProblem.quotient_represents_opt_equiv
abbrev QT4 := @DecisionProblem.factors_implies_respects
noncomputable abbrev QT5 := @DecisionProblem.quotientEquivOptRange
abbrev QT6 := @DecisionProblem.quotientEquivOptRange_apply_quotientMap
abbrev QT7 := @DecisionProblem.quotient_has_unique_factorization

-- AB: Abstraction Boundary (AbstractionCollapse.lean)
abbrev AB1 := @DecisionProblem.not_preservesOpt_iff_erasesDecisionRelevantDistinction
abbrev AB2 := @DecisionProblem.surjective_abstraction_factors_or_erases
abbrev AB3 := @DecisionProblem.collapseBeyondQuotient_physically_impossible
abbrev AB4 := @DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors

-- WM: Wolpert Mismatch (Physics/WolpertMismatch.lean)
abbrev WM1 := @Physics.WolpertMismatch.mismatchKL_nonneg
abbrev WM2 := @Physics.WolpertMismatch.mismatchKL_eq_zero_iff_eq
abbrev WM3 := @Physics.WolpertMismatch.mismatchKL_pos_of_exists_ne
abbrev WM4 := @Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne
abbrev WM5 := @Physics.WolpertDecomposition.periodic_modular_mismatch_of_distribution_mismatch
abbrev WM6 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch

-- WR: Wolpert Residual (Physics/WolpertResidual.lean + decomposition lift)
abbrev WR1 := @Physics.WolpertResidual.pairwiseResidualKL_nonneg
abbrev WR2 := @Physics.WolpertResidual.pairwiseResidualKL_pos_of_asymmetry
abbrev WR3 := @Physics.WolpertResidual.residualNatLowerBound_pos_of_asymmetry
abbrev WR4 := @Physics.WolpertDecomposition.stopping_time_residual_of_pairwise_flow_asymmetry
abbrev WR5 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry
abbrev WR6 := @Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway
abbrev WR7 := @Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split
abbrev WR8 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_discrete_edge_split
abbrev WR9 := @Physics.WolpertDecomposition.stopping_time_residual_of_finite_discrete_witness
abbrev WR10 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness

-- WP: Wolpert Physics (Physics/WolpertDecomposition.lean)
abbrev WP1 := @Physics.WolpertDecomposition.DecomposedProcessModel.totalOverheadPerBit_eq_sum
abbrev WP2 := @Physics.WolpertDecomposition.landauer_floor_plus_decomposition_lower_bound
abbrev WP3 := @Physics.WolpertDecomposition.effective_model_dominates_landauer_floor_decomposition
abbrev WP4 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_periodic_modular_mismatch
abbrev WP5 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_stopping_time_residual
abbrev WP6 := @Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component
abbrev WP7 := @Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound
abbrev WP8 := @Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource
abbrev WP9 := @Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition

-- WD: Witness-Checking Duality (WitnessCheckingDuality.lean)
-- WD1 is the main duality: any sound checker needs ≥ 2^(n-1) witness pairs
abbrev WD1 := @checking_witnessing_duality_budget
abbrev WD2 := @no_sound_checker_below_witness_budget
abbrev WD3 := @checking_time_ge_witness_budget
abbrev WD4 := @witnessBudgetEmpty
abbrev WD5 := @checkingBudgetPairs

-- BC: Bayes from Counting (BayesFoundations.lean)
-- BC1-BC3 are the Kolmogorov axioms derived from counting measure
abbrev BC1 := @Foundations.counting_nonneg
abbrev BC2 := @Foundations.counting_total
abbrev BC3 := @Foundations.counting_additive
abbrev BC4 := @Foundations.bayes_from_conditional
abbrev BC5 := @Foundations.entropy_contraction

/-! ## Thermodynamic Uncertainty Relations (TUR) handles
    Physics/TUR.lean - Independent bridge: precision costs entropy production
    TUR: Var(J)/⟨J⟩² ≥ 2/σ_Σ
-/

-- TUR1: Transition probabilities are non-negative
abbrev TUR1 := @Physics.transitionProb_nonneg
-- TUR2: Transition probabilities sum to 1
abbrev TUR2 := @Physics.transitionProb_sum_one
-- TUR3: Entropy production is non-negative (Second Law) [axiom]
-- Note: entropyProduction_nonneg is an axiom, handle separately if needed
-- TUR4: The TUR bound itself [axiom]
-- Note: tur_bound is an axiom, handle separately if needed
-- TUR5: The TUR bridge theorem
abbrev TUR5 := @Physics.tur_bridge
-- TUR6: Multiple futures imply positive entropy production
abbrev TUR6 := @Physics.multiple_futures_entropy_production

/-! ## Wasserstein/Optimal Transport (W) handles
    Physics/WassersteinIntegrity.lean - Independent bridge: state change costs transport
-/

-- W1: Diagonal coupling has zero transport cost
abbrev W1 := @Physics.single_future_zero_cost
-- W2: Off-diagonal mass implies positive transport cost
abbrev W2 := @Physics.transportCost_pos_of_offDiag
-- W3: Integrity is the centroid (minimal transport)
abbrev W3 := @Physics.integrity_is_centroid
-- W4: The Wasserstein bridge statement
abbrev W4 := @Physics.wasserstein_bridge

/-! ## Transport Cost (XC) handles
    Physics/TransportCost.lean - DOF > 1 implies higher transport
    (Using XC to avoid conflict with existing TC prefix for ToolCollapse)
-/

-- XC1: srank determines state space size
abbrev XC1 := @Physics.srank_determines_states
-- XC2: More states means more transport cost
abbrev XC2 := @Physics.more_states_more_transport
-- XC3: Transport cost lower bound by srank
abbrev XC3 := @Physics.transport_lower_bound
-- XC4: Transport independent of energy (Landauer)
abbrev XC4 := @Physics.transport_independent_of_energy
-- XC5: Transport independent of precision (TUR)
abbrev XC5 := @Physics.transport_independent_of_precision
-- XC6: srank is the unified complexity measure
abbrev XC6 := @Physics.srank_unified_complexity
-- XC7: Complete bridge set (five independent derivations)
abbrev XC7 := @Physics.complete_bridge_set

/-! ## Rate-Distortion (RD) handles
    Information/RateDistortion.lean - Independent bridge: compression requires bits
-/

-- RD1: Shannon entropy is non-negative
abbrev RD1 := @Information.shannonEntropy_nonneg
-- RD2: Rate at zero distortion equals entropy
abbrev RD2 := @Information.rate_zero_distortion
-- RD3: Rate is monotonically decreasing in distortion
abbrev RD3 := @Information.rate_monotone

/-! ## Rate-Distortion/srank Bridge (RS) handles
    Information/RDSrank.lean - Bridge: R(0) = srank
-/

-- RS1: Decision equivalence preserves decision outcome
abbrev RS1 := @Information.equiv_preserves_decision
-- RS2: Rate equals srank theorem
abbrev RS2 := @Information.rate_equals_srank
-- RS3: Compression below srank fails
abbrev RS3 := @Information.compression_below_srank_fails
-- RS4: srank bits are sufficient for decision
abbrev RS4 := @Information.srank_bits_sufficient
-- RS5: The rate-distortion bridge statement
abbrev RS5 := @Information.rate_distortion_bridge

-- ════════════════════════════════════════════════════════════════════════════
-- Locality Physics (LP) handles — P ≠ NP as necessary for physics
-- ════════════════════════════════════════════════════════════════════════════

/-! ### Part 0: Premises

EP1-EP3: Empirical axioms (physics)
FP1-FP7: First principles (pure math) — THE NONTRIVIALITY PROOF
EP4: DERIVED (not an axiom) — follows from FP1-FP7 + observation
-/

-- EP1: Landauer principle (empirical axiom)
abbrev EP1 := @Physics.LocalityPhysics.landauer_principle
-- EP2: Finite regional energy (empirical axiom)
abbrev EP2 := @Physics.LocalityPhysics.finite_regional_energy
-- EP3: Finite signal speed (empirical axiom)
abbrev EP3 := @Physics.LocalityPhysics.finite_signal_speed

-- FP1-FP7: First principles — THE NONTRIVIALITY PROOF (no physics)
-- FP1: Trivial states → all equal (cardinality)
abbrev FP1 := @Physics.LocalityPhysics.trivial_states_all_equal
-- FP2: All equal → constant function (logic)
abbrev FP2 := @Physics.LocalityPhysics.equal_states_constant_function
-- FP3: Constant function → singleton image (set theory)
abbrev FP3 := @Physics.LocalityPhysics.constant_function_singleton_image
-- FP4: Singleton → zero entropy (log 1 = 0)
abbrev FP4 := @Physics.LocalityPhysics.singleton_image_zero_entropy
-- FP5: Zero entropy → no information (definition)
abbrev FP5 := @Physics.LocalityPhysics.zero_entropy_no_information
-- FP6: MASTER THEOREM — Triviality → no information
abbrev FP6 := @Physics.LocalityPhysics.triviality_implies_no_information
-- FP7: CONTRAPOSITIVE — Information → nontriviality
abbrev FP7 := @Physics.LocalityPhysics.information_requires_nontriviality

-- EP4: Nontrivial physics (DERIVED, not axiom — follows from FP7 + observation)
abbrev EP4 := @Physics.LocalityPhysics.nontrivial_physics

-- FP8-FP15: THE SECOND LAW FROM COUNTING (first principles)
-- FP8: Atypical states are rare (counting)
abbrev FP8 := @Physics.LocalityPhysics.atypical_states_rare
-- FP9: Random selection misses target (probability)
abbrev FP9 := @Physics.LocalityPhysics.random_misses_target
-- FP10: Errors accumulate without checking (multiplication)
abbrev FP10 := @Physics.LocalityPhysics.errors_accumulate
-- FP11: Wrong paths dominate (exponentiation)
abbrev FP11 := @Physics.LocalityPhysics.wrong_paths_dominate
-- FP12: SECOND LAW STRUCTURE — maintaining order requires information
abbrev FP12 := @Physics.LocalityPhysics.second_law_from_counting
-- FP13: Verification is information (definition)
abbrev FP13 := @Physics.LocalityPhysics.verification_is_information
-- FP14: Entropy = information (definition)
abbrev FP14 := @Physics.LocalityPhysics.entropy_is_information
-- FP15: LANDAUER STRUCTURE — erasure increases environment entropy
abbrev FP15 := @Physics.LocalityPhysics.landauer_structure
-- EP1': Landauer with calibration (empirical part is only kT)
abbrev EP1' := @Physics.LocalityPhysics.landauer_with_calibration

-- FPT1-FPT10: TIME POSITIVITY FROM NON-CONTRADICTION (first principles)
-- "A computational step must take >0 time, otherwise we could count through all proofs."
-- This is pure logic: law of non-contradiction + definition of function.
-- FPT1: Timeline — assigns unique state to each moment (definition)
abbrev FPT1 := @Physics.LocalityPhysics.Timeline
-- FPT2: Functions are deterministic (reflexivity)
abbrev FPT2 := @Physics.LocalityPhysics.FPT2_function_deterministic
-- FPT3: Different outputs imply different inputs (contrapositive)
abbrev FPT3 := @Physics.LocalityPhysics.FPT3_outputs_differ_inputs_differ
-- FPT3': Computational step structure (state change definition)
abbrev FPT3' := @Physics.LocalityPhysics.ComputationalStep
-- FPT4: THE CORE THEOREM — A step requires distinct moments (non-contradiction)
abbrev FPT4 := @Physics.LocalityPhysics.FPT4_step_requires_distinct_moments
-- FPT5: Distinct moments with ordering → positive duration (arithmetic)
abbrev FPT5 := @Physics.LocalityPhysics.FPT5_distinct_moments_positive_duration
-- FPT6: Every computational step takes positive time (FPT4 + FPT5)
abbrev FPT6 := @Physics.LocalityPhysics.FPT6_step_takes_positive_time
-- FPT7: No instantaneous steps exist (restatement of FPT4)
abbrev FPT7 := @Physics.LocalityPhysics.FPT7_no_instantaneous_steps
-- FPT8: Propagation takes time (steps × step_time > 0)
abbrev FPT8 := @Physics.LocalityPhysics.FPT8_propagation_takes_time
-- FPT9: Speed bounded by positive time (distance/time ≤ distance)
abbrev FPT9 := @Physics.LocalityPhysics.FPT9_speed_bounded_by_positive_time
-- FPT10: EC3 IS LOGICAL — finite propagation speed follows from t > 0
abbrev FPT10 := @Physics.LocalityPhysics.FPT10_ec3_is_logical

-- IP1-IP7: INDEPENDENCE PROOFS — The empirical claims are irreducible
-- IP1: EC1 (temperature) can be true
abbrev IP1 := @Physics.LocalityPhysics.ec1_can_be_true
-- IP2: EC1 independent of EC2, EC3
abbrev IP2 := @Physics.LocalityPhysics.ec1_independent_of_ec2_ec3
-- IP3: EC2 (finite resources) can be true
abbrev IP3 := @Physics.LocalityPhysics.ec2_can_be_true
-- IP4: EC2 independent of EC1, EC3
abbrev IP4 := @Physics.LocalityPhysics.ec2_independent_of_ec1_ec3
-- IP5: EC3 (local causality) can be true
abbrev IP5 := @Physics.LocalityPhysics.ec3_can_be_true
-- IP6: EC3 independent of EC1, EC2
abbrev IP6 := @Physics.LocalityPhysics.ec3_independent_of_ec1_ec2
-- IP7: MASTER INDEPENDENCE — All three claims are mutually independent
abbrev IP7 := @Physics.LocalityPhysics.empirical_claims_mutually_independent

-- LP1: Spacetime point structure
abbrev LP1 := @Physics.LocalityPhysics.SpacetimePoint
-- LP2: Light cone definition
abbrev LP2 := @Physics.LocalityPhysics.lightCone
-- LP3: Causal past definition
abbrev LP3 := @Physics.LocalityPhysics.causalPast
-- LP4: Self in light cone
abbrev LP4 := @Physics.LocalityPhysics.self_in_lightCone
-- LP5: Self in causal past
abbrev LP5 := @Physics.LocalityPhysics.self_in_causalPast
-- LP6: Local region structure
abbrev LP6 := @Physics.LocalityPhysics.LocalRegion
-- LP7: canObserve predicate
abbrev LP7 := @Physics.LocalityPhysics.canObserve
-- LP8: Spacelike separated definition
abbrev LP8 := @Physics.LocalityPhysics.spacelikeSeparated
-- LP9: Spacelike disjoint observation
abbrev LP9 := @Physics.LocalityPhysics.spacelike_disjoint_observation
-- LP10a: Bounded region has spacelike complement
abbrev LP10a := @Physics.LocalityPhysics.bounded_region_has_spacelike_complement
-- LP10b: Distant regions are spacelike separated
abbrev LP10b := @Physics.LocalityPhysics.distant_regions_spacelike_separated
-- LP10c: Spacelike separated regions exist (constructive)
abbrev LP10c := @Physics.LocalityPhysics.spacelike_separated_regions_exist
-- LP11: Local configuration structure
abbrev LP11 := @Physics.LocalityPhysics.LocalConfiguration
-- LP12: Locally valid predicate
abbrev LP12 := @Physics.LocalityPhysics.isLocallyValid
-- LP13: Board merge result type
abbrev LP13 := @Physics.LocalityPhysics.MergeResult
-- LP14: Board merge operation
abbrev LP14 := @Physics.LocalityPhysics.boardMerge
-- LP15: Independent configs can disagree
abbrev LP15 := @Physics.LocalityPhysics.independent_configs_can_disagree
-- LP16: Merge compatible iff
abbrev LP16 := @Physics.LocalityPhysics.merge_compatible_iff
-- LP17: Merge contradiction iff
abbrev LP17 := @Physics.LocalityPhysics.merge_contradiction_iff
-- LP18: Locality implies possible contradiction
abbrev LP18 := @Physics.LocalityPhysics.locality_implies_possible_contradiction
-- LP19: Superposition definition
abbrev LP19 := @Physics.LocalityPhysics.Superposition
-- LP20: Superposition can contain contradictions
abbrev LP20 := @Physics.LocalityPhysics.superposition_can_contain_contradictions
-- LP21: Superposition requires separation
abbrev LP21 := @Physics.LocalityPhysics.superposition_requires_separation
-- LP22: Bell separation is real
abbrev LP22 := @Physics.LocalityPhysics.bell_separation_is_real
-- LP23: Measurement is merge contradiction
abbrev LP23 := @Physics.LocalityPhysics.measurement_is_merge_contradiction
-- LP24: Entanglement is shared origin
abbrev LP24 := @Physics.LocalityPhysics.entanglement_is_shared_origin
-- LP31: Complete knowledge requires all queries
abbrev LP31 := @Physics.LocalityPhysics.complete_knowledge_requires_all_queries
-- LP32: Finite energy constraint
abbrev LP32 := @Physics.LocalityPhysics.finite_energy_constraint
-- LP33: Self knowledge impossible if insufficient energy
abbrev LP33 := @Physics.LocalityPhysics.self_knowledge_impossible_if_insufficient_energy
-- LP34: Bounded energy forces locality
abbrev LP34 := @Physics.LocalityPhysics.bounded_energy_forces_locality
-- LP35: Locality implies independent regions
abbrev LP35 := @Physics.LocalityPhysics.locality_implies_independent_regions
-- LP36: Independent regions imply possible contradiction
abbrev LP36 := @Physics.LocalityPhysics.independent_regions_imply_possible_contradiction
-- LP38: THE GRAND THEOREM — P ≠ NP necessary for physics
abbrev LP38 := @Physics.LocalityPhysics.pne_np_necessary_for_physics
-- LP39: Matter exists because P ≠ NP
abbrev LP39 := @Physics.LocalityPhysics.matter_exists_because_pne_np
-- LP40: Physics is the game
abbrev LP40 := @Physics.LocalityPhysics.physics_is_the_game
-- LP41: Without positive query cost, no bound
abbrev LP41 := @Physics.LocalityPhysics.without_positive_query_cost_no_bound
-- LP42: Without nontrivial states, no disagreement
abbrev LP42 := @Physics.LocalityPhysics.without_nontrivial_states_no_disagreement
-- LP43: Without separation, no independence
abbrev LP43 := @Physics.LocalityPhysics.without_separation_no_independence
-- LP44: Without finite capacity, no gap
abbrev LP44 := @Physics.LocalityPhysics.without_finite_capacity_no_gap
-- LP45: All premises used
abbrev LP45 := @Physics.LocalityPhysics.all_premises_used
-- LP46: Premises necessary and sufficient
abbrev LP46 := @Physics.LocalityPhysics.premises_necessary_and_sufficient

/-! ### Part VI: Value Requires Intractability (LP50-LP56) -/

-- LP50: Shannon — Information value IS intractability
abbrev LP50 := @Physics.LocalityPhysics.shannon_value_is_intractability
-- LP51: Economics — Value requires scarcity (intractable supply)
abbrev LP51 := @Physics.LocalityPhysics.economic_value_requires_scarcity
-- LP52: Thermodynamics — Work requires gradients (intractable equilibrium)
abbrev LP52 := @Physics.LocalityPhysics.thermodynamic_value_requires_gradient
-- LP53: Decision theory — VOI requires uncertainty (intractable knowledge)
abbrev LP53 := @Physics.LocalityPhysics.voi_requires_uncertainty
-- LP54: Physics — Matter requires locality (intractable self-knowledge)
abbrev LP54 := @Physics.LocalityPhysics.physics_requires_intractability
-- LP55: The value theorem — Value is the intractable part
abbrev LP55 := @Physics.LocalityPhysics.value_is_intractability
-- LP56: Observers value what they cannot fully observe
abbrev LP56 := @Physics.LocalityPhysics.observers_value_the_intractable

/-! ### Part VII: The Gap Is Time (LP57-LP61) -/

-- LP57: Finite steps cover finite states
abbrev LP57 := @Physics.LocalityPhysics.finite_steps_finite_coverage
-- LP58: The counting gap
abbrev LP58 := @Physics.LocalityPhysics.counting_gap
-- LP59: Time is steps times step duration
abbrev LP59 := @Physics.LocalityPhysics.time_is_counting
-- LP60: Gap equivalence (measure = counting = time = energy)
abbrev LP60 := @Physics.LocalityPhysics.gap_equivalence
-- LP60': Simplified gap equivalence
abbrev LP60' := @Physics.LocalityPhysics.gap_equivalence_simple
-- LP61: Light cone is the time gap in spacetime
abbrev LP61 := @Physics.LocalityPhysics.light_cone_is_time_gap

/-! ## Bridge Composition Theorems (BR) handles
    Bridges.lean: Non-trivial cross-cluster compositions that connect
    isolated proof clusters to the main claim graph.
-/

-- BR1: ETH + structural rank = exponential hardness
abbrev BR1 := @Bridges.eth_structural_rank_exponential_hardness
-- BR2: Fisher rank bounds minimal sufficient set size
abbrev BR2 := @Bridges.fisher_rank_lower_bounds_sufficient_set
-- BR3: FPT + srank = parameterized dichotomy
abbrev BR3 := @Bridges.fpt_srank_parameterized_dichotomy
-- BR4: TUR + srank = thermodynamic cost of decision precision
abbrev BR4 := @Bridges.tur_srank_thermodynamic_cost
-- BR5: Dichotomy + ETH = complete complexity classification
abbrev BR5 := @Bridges.dichotomy_eth_complete_classification
-- BR6: Reduction + ETH = coNP-completeness + exponential hardness
abbrev BR6 := @Bridges.reduction_eth_conp_exponential
-- BR7: Geometry + covering = certificate complexity
abbrev BR7 := @Bridges.geometry_covering_certificate_complexity
-- BR8: Rate-distortion + Fisher information bridge
abbrev BR8 := @Bridges.rate_distortion_fisher_information_bridge
-- BR9: Counting complexity + #P-hardness
abbrev BR9 := @Bridges.counting_complexity_sharp_p_hardness
-- BR10: Approximation + counting hardness bridge
abbrev BR10 := @Bridges.approximation_counting_hardness_bridge

/-! ## Structural-rank primitive handles (SK)
    StructuralRank.lean: direct familiar-construction facts about srank as the
    cardinality of the relevant-coordinate support.
-/

-- SK1: srank is the cardinality of the relevant-coordinate support
abbrev SK1 := @DecisionProblem.srank_eq_relevant_card
-- SK2: srank is bounded by the number of coordinates
abbrev SK2 := @DecisionProblem.srank_le_n
-- SK3: srank = 0 iff the decision boundary is coordinate-constant
abbrev SK3 := @DecisionProblem.srank_zero_iff_constant

/-! ## Inflation-entropy bridge handles (IEB)
    InflationEntropyBridge.lean + InflationEntropyMinimality.lean
-/

-- IEB1: Monotone optimizer-class growth under embedding compatibility
abbrev IEB1 := @InflationEntropyBridge.classes_monotone
-- IEB2: Monotone quotient entropy under embedding compatibility
abbrev IEB2 := @InflationEntropyBridge.entropy_monotone
-- IEB3: Strict optimizer-class growth from new-class witness
abbrev IEB3 := @InflationEntropyBridge.classes_strict_increase
-- IEB4: Strict entropy growth from new-class witness
abbrev IEB4 := @InflationEntropyBridge.entropy_strict_increase
-- IEB5: Utility compatibility implies optimizer compatibility
abbrev IEB5 := @InflationEntropyBridge.optCompat_of_utilityCompat
-- IEB6: Thermal floor monotone under class monotonicity
abbrev IEB6 := @InflationEntropyBridge.thermal_floor_monotone_of_classes
-- IEB7: Thermal floor strict growth under new-class witness
abbrev IEB7 := @InflationEntropyBridge.thermal_floor_strict_of_new_class
-- IEB8: Later floor implies earlier floor
abbrev IEB8 := @InflationEntropyBridge.later_energy_floor_implies_earlier_floor
-- IEB9: A2 irreducibility witness (compatibility needed)
abbrev IEB9 := @InflationEntropyMinimality.not_redundant_A2_for_mono_classes
-- IEB10: A3 irreducibility witness (strictness needs new class)
abbrev IEB10 := @InflationEntropyMinimality.not_redundant_A3_for_strict_entropy
-- IEB11: P1 irreducibility witness (kB positivity)
abbrev IEB11 := @InflationEntropyMinimality.not_redundant_P1_for_positive_floor
-- IEB12: P2 irreducibility witness (temperature positivity)
abbrev IEB12 := @InflationEntropyMinimality.not_redundant_P2_for_positive_floor
-- IEB13: A1 irreducibility witness in weak (no-embedding) family
abbrev IEB13 := @InflationEntropyMinimality.not_redundant_A1_for_mono_classes_weak
-- IEB16: F1 irreducibility witness (finite counting requirement)
abbrev IEB16 := @InflationEntropyMinimality.not_redundant_F1_for_finite_counting_requirement
-- IEB14: F2 irreducibility witness (nonempty needed for positivity branch)
abbrev IEB14 := @InflationEntropyMinimality.not_redundant_F2_for_numOptClasses_pos
-- IEB15: P3 irreducibility witness (energy floor premise needed)
abbrev IEB15 := @InflationEntropyMinimality.not_redundant_P3_for_energy_from_entropy_bridge

/-! ## Stochastic set-valued bridge handles (SSV)
    StochasticSequential/SetValued.lean
-/

-- SSV1: agreeOn-equivalent anchors induce equal fiber expected utility partitions
abbrev SSV1 := @StochasticSequential.fiberExpectedUtility_eq_of_agreeOn
-- SSV2: agreeOn-equivalent anchors induce equal fiber optimal sets
abbrev SSV2 := @StochasticSequential.fiberOpt_eq_of_agreeOn
-- SSV3: set-valued stochastic sufficiency holds (ties allowed)
abbrev SSV3 := @StochasticSequential.stochasticSetSufficient_universal

end DecisionQuotient
