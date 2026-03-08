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

-- Axiom audit for paper 4 key theorems
-- Expected: only Lean 4 kernel axioms (propext, Classical.choice, Quot.sound)
-- Any extra axioms = hidden assumptions baked invisibly into the proof kernel

-- ── COMPLEXITY HARDNESS ─────────────────────────────────────────────────────
#print axioms DecisionQuotient.sufficiency_check_coNP_hard
#print axioms DecisionQuotient.min_sufficient_set_coNP_hard
#print axioms DecisionQuotient.anchor_sufficiency_sigma2p
#print axioms DecisionQuotient.tractable_small_state_space
#print axioms DecisionQuotient.checking_witnessing_duality_summary

-- ── BAYES / INFORMATION ─────────────────────────────────────────────────────
#print axioms DecisionQuotient.BayesOptimalityProof.bayes_is_optimal
#print axioms DecisionQuotient.BayesOptimalityProof.KL_nonneg
#print axioms DecisionQuotient.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL

-- ── DECISION QUOTIENT UNIVERSAL PROPERTY ────────────────────────────────────
#print axioms DecisionQuotient.DecisionProblem.opt_factors_through_quotient
#print axioms DecisionQuotient.DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors

-- ── INFORMATION / SUFFICIENCY ───────────────────────────────────────────────
#print axioms DecisionQuotient.relevantSet_isSufficient
#print axioms DecisionQuotient.numOptClasses_le_pow_srank_binary

-- ── THERMODYNAMIC LIFT ──────────────────────────────────────────────────────
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

-- ── PHYSICAL HARDNESS / P≠NP NO-GO ─────────────────────────────────────────
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

-- ── INTEGRITY / COMPETENCE ──────────────────────────────────────────────────
-- Logical core: universal competence → complexity collapse → ¬competence
#print axioms DecisionQuotient.IntegrityCompetence.integrity_resource_bound
-- Structural consequence: integrity forces abstention or budget violation
#print axioms DecisionQuotient.IntegrityCompetence.integrity_forces_abstention
-- Exact certainty inflation ↔ no exact competence
#print axioms DecisionQuotient.IntegrityCompetence.exactCertaintyInflation_iff_no_exact_competence
-- RLFF: any global reward-maximizer is evidence-backed when abstain > penalty
#print axioms DecisionQuotient.IntegrityCompetence.rlff_maximizer_has_evidence
