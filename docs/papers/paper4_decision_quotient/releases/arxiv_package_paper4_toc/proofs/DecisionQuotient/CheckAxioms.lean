import DecisionQuotient.Summary
import DecisionQuotient.BayesOptimalityProof
import DecisionQuotient.Hardness
import DecisionQuotient.WitnessCheckingDuality
import DecisionQuotient.Quotient
import DecisionQuotient.AbstractionCollapse
import DecisionQuotient.Information
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.Physics.WolpertMismatch
import DecisionQuotient.Physics.WolpertConstraints
import DecisionQuotient.Physics.WolpertDecomposition
import DecisionQuotient.PhysicalBudgetCrossover
import DecisionQuotient.Dichotomy
import DecisionQuotient.FunctionalInformation
import DecisionQuotient.BayesFromDQ
import DecisionQuotient.IntegrityCompetence
import DecisionQuotient.Physics.PhysicalHardness
import DecisionQuotient.InflationEntropyBridge
import DecisionQuotient.InflationEntropyMinimality
import DecisionQuotient.StochasticSequential.SetValued
import DecisionQuotient.Hardness.ApproximationHardness
import DecisionQuotient.Hardness.SetCoverReduction
import DecisionQuotient.Hardness.MinSufficientApproximation

-- Axiom audit for the paper-4 theorem stack and its immediate upstream support.
-- Expected: only Lean 4 kernel axioms (propext, Classical.choice, Quot.sound).
-- Any extra axioms would indicate hidden assumptions in the proof kernel.

-- ── PAPER 4: COMPLEXITY / APPROXIMATION / RELIABILITY ───────────────────────
#print axioms DecisionQuotient.sufficiency_check_coNP_hard
#print axioms DecisionQuotient.min_sufficient_set_coNP_hard
#print axioms DecisionQuotient.anchor_sufficiency_sigma2p
#print axioms DecisionQuotient.tractable_small_state_space
#print axioms DecisionQuotient.checking_witnessing_duality_summary
#print axioms DecisionQuotient.min_sufficient_set_inapprox_statement
#print axioms DecisionQuotient.min_sufficient_inapproximability_conditional
#print axioms DecisionQuotient.min_sufficient_ratio_inapprox_of_set_cover_ratio_inapprox
#print axioms DecisionQuotient.counted_min_sufficient_inapproximability_conditional
#print axioms DecisionQuotient.counted_min_sufficient_ratio_inapprox_of_counted_set_cover_ratio_inapprox
#print axioms DecisionQuotient.countedGapTautologyRun_correct
#print axioms DecisionQuotient.countedGapTautologyRun_polytime
#print axioms DecisionQuotient.SetCoverInstance.sufficiency_iff_cover
#print axioms DecisionQuotient.SetCoverInstance.min_sufficient_iff_set_cover

-- ── UPSTREAM SUPPORT: BAYES / INFORMATION ────────────────────────────────────
#print axioms DecisionQuotient.BayesOptimalityProof.bayes_is_optimal
#print axioms DecisionQuotient.BayesOptimalityProof.KL_nonneg
#print axioms DecisionQuotient.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL

-- ── UPSTREAM SUPPORT: QUOTIENT / UNIVERSAL PROPERTY ──────────────────────────
#print axioms DecisionQuotient.DecisionProblem.opt_factors_through_quotient
#print axioms DecisionQuotient.DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors

-- ── UPSTREAM SUPPORT: SUFFICIENCY / INFORMATION BRIDGES ─────────────────────
#print axioms DecisionQuotient.relevantSet_isSufficient
#print axioms DecisionQuotient.numOptClasses_le_pow_srank_binary

-- ── DOWNSTREAM SUPPORT: THERMODYNAMIC / PHYSICAL LIBRARY ────────────────────
-- Landauer positivity: k_B, T > 0 → k_BT ln2 > 0
#print axioms DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit_pos
-- Energy-information duality: E ≥ k_BT × H_nats(D)
#print axioms DecisionQuotient.ThermodynamicLift.energy_ge_kbt_nat_entropy
-- Hardness + thermo bundle (conditional on ¬P=coNP)
#print axioms DecisionQuotient.ThermodynamicLift.hardness_thermo_bundle_conditional
-- Neukart-Vinokur duality: dU = λ·dC
#print axioms DecisionQuotient.ThermodynamicLift.neukart_vinokur_bundle
-- Mandatory cost from a Landauer floor hypothesis
#print axioms DecisionQuotient.ThermodynamicLift.energy_lower_mandatory_of_landauer_floor
-- Exact-calibration specialization of the floor theorem
#print axioms DecisionQuotient.ThermodynamicLift.energy_lower_mandatory_of_landauer_calibration
-- Derived KL mismatch branch: strict mismatch positivity is theorem-level
#print axioms DecisionQuotient.Physics.WolpertMismatch.mismatchKL_pos_of_exists_ne
-- Derived KL mismatch branch lifted into the Wolpert decomposition interface
#print axioms DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch
-- Derived finite residual-asymmetry branch from bidirectional edge-flow KL
#print axioms DecisionQuotient.Physics.WolpertResidual.pairwiseResidualKL_pos_of_asymmetry
-- Derived finite residual branch lifted into the Wolpert decomposition interface
#print axioms DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry
-- Exhaustive finite discrete residual branch via reverse-edge case split
#print axioms DecisionQuotient.Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway
-- Unified finite discrete residual branch lifted into the Wolpert decomposition interface
#print axioms DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_discrete_edge_split
-- Process-level finite discrete residual witness branch
#print axioms DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness
-- Wolpert-style constrained-process interface: floor plus explicit overhead
#print axioms DecisionQuotient.Physics.WolpertConstraints.physical_grounding_bundle_with_wolpert_overhead
-- Wolpert decomposition interface: mismatch + residual refinement
#print axioms DecisionQuotient.Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition

-- ── DOWNSTREAM SUPPORT: PHYSICAL HARDNESS / P≠NP NO-GO ──────────────────────
-- Core: exponential ops exceed any finite budget
#print axioms PhysicalComplexity.coNP_physically_impossible
-- No-go: P=NP → physical collapse → contradiction
#print axioms PhysicalComplexity.p_eq_np_physically_impossible_of_collapse_map
-- Canonical specialization for the coNP_requirement profile
#print axioms PhysicalComplexity.p_eq_np_physically_impossible_canonical
-- Kernel minimality: all three premises are independently necessary
#print axioms PhysicalComplexity.kernel_assumption_set_minimal
-- Sharp assumption-failure disjunction
#print axioms PhysicalComplexity.collapse_implies_assumption_failure_disjunction

-- ── PAPER 4 / 4C INTERFACE: INTEGRITY / COMPETENCE ─────────────────────────
-- Logical core: universal competence → complexity collapse → ¬competence
#print axioms DecisionQuotient.IntegrityCompetence.integrity_resource_bound
-- Structural consequence: integrity forces abstention or budget violation
#print axioms DecisionQuotient.IntegrityCompetence.integrity_forces_abstention
-- Exact certainty inflation ↔ no exact competence
#print axioms DecisionQuotient.IntegrityCompetence.exactCertaintyInflation_iff_no_exact_competence
-- RLFF: any global reward-maximizer is evidence-backed when abstain > penalty
#print axioms DecisionQuotient.IntegrityCompetence.rlff_maximizer_has_evidence

-- ── DOWNSTREAM SUPPORT: INFLATION / ENTROPY BRIDGE ──────────────────────────
#print axioms DecisionQuotient.InflationEntropyBridge.classes_monotone
#print axioms DecisionQuotient.InflationEntropyBridge.entropy_monotone
#print axioms DecisionQuotient.InflationEntropyBridge.classes_strict_increase
#print axioms DecisionQuotient.InflationEntropyBridge.entropy_strict_increase
#print axioms DecisionQuotient.InflationEntropyBridge.optCompat_of_utilityCompat
#print axioms DecisionQuotient.InflationEntropyBridge.classes_monotone_of_utilityCompat
#print axioms DecisionQuotient.InflationEntropyBridge.entropy_monotone_of_utilityCompat
#print axioms DecisionQuotient.InflationEntropyBridge.thermal_floor_monotone_of_classes
#print axioms DecisionQuotient.InflationEntropyBridge.thermal_floor_strict_of_new_class
#print axioms DecisionQuotient.InflationEntropyBridge.later_energy_floor_implies_earlier_floor
#print axioms DecisionQuotient.InflationEntropyBridge.Temporal.temporal_classes_monotone_of_utilityCompat
#print axioms DecisionQuotient.InflationEntropyBridge.Temporal.temporal_entropy_monotone_of_utilityCompat

-- ── DOWNSTREAM SUPPORT: INFLATION MINIMALITY WITNESSES ──────────────────────
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_A2_for_mono_classes
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_A3_for_strict_entropy
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_P1_for_positive_floor
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_P2_for_positive_floor
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_A1_for_mono_classes_weak
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_F1_for_finite_counting_requirement
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_F2_for_numOptClasses_pos
#print axioms DecisionQuotient.InflationEntropyMinimality.not_redundant_P3_for_energy_from_entropy_bridge

-- ── PAPER 4: STOCHASTIC / SEQUENTIAL BRIDGES ────────────────────────────────
#print axioms DecisionQuotient.StochasticSequential.stochasticSetSufficient_universal
#print axioms DecisionQuotient.StochasticSequential.stochasticSufficient_implies_setSufficient
