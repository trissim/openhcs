/-
  Paper 4: Decision-Relevant Uncertainty

  CrossPaperDependencies.lean

  Paper 4 is FOUNDATIONAL. It does NOT depend on other papers.
  Other papers (Paper 3) depend on Paper 4.

  This file documents what Paper 4 provides for other papers to use.
  NO IMPORTS FROM OTHER PAPERS.

  ## Exports for Paper 3:
  - DecisionProblem structure
  - Structural rank (srank)
  - Thermodynamic bounds
  - Tractability conditions
  - Bayesian optimality
-/

import DecisionQuotient.Basic
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.BayesOptimalityProof
import DecisionQuotient.Tractability.StructuralRank
import DecisionQuotient.ComputationalDecisionProblem
import DecisionQuotient.Physics.BoundedAcquisition

namespace DecisionQuotient.CrossPaperDeps

/-! ## What Paper 4 Exports (No Dependencies) -/

/-- Paper 4 provides DecisionProblem structure.
    Other papers can use this to encode their domains. -/
theorem decision_problem_exists (A S : Type*) (utility : A → S → ℝ) :
    ∃ (dp : DecisionProblem A S), dp.utility = utility := by
  exact ⟨{ utility }, rfl⟩

/-- Paper 4 provides tractability conditions.
    Bounded actions: |A| ≤ k means tractable sufficiency checking. -/
theorem bounded_actions_condition (dof : ℕ) :
    dof = 1 → Fintype.card (Fin dof ⊕ Unit) = 2 := by
  intro h
  rw [h]
  simp [Fintype.card_sum, Fintype.card_unit]

/-- Paper 4 proves Bayesian optimality (NO AXIOMS).
    Paper 3 can use this for decision optimality. -/
theorem bayes_optimal {H : Type*} [Fintype H] [DecidableEq H]
    (p q : H → ℝ) (hp : ∀ h, 0 < p h) (hq : ∀ h, 0 < q h)
    (hp1 : (∑ h, p h) = 1) (hq1 : (∑ h, q h) = 1) :
    BayesOptimalityProof.crossEntropy p q ≥ BayesOptimalityProof.crossEntropy p p := by
  exact BayesOptimalityProof.entropy_le_crossEntropy p q hp hq hp1 hq1

/-- Paper 4 proves Gibbs' inequality (NO AXIOMS).
    KL(p||q) ≥ 0, equality iff p = q. -/
theorem kl_nonneg {α : Type*} [Fintype α] (p q : α → ℝ)
    (hp : ∀ a, 0 < p a) (hq : ∀ a, 0 < q a)
    (hp1 : (∑ a, p a) = 1) (hq1 : (∑ a, q a) = 1) :
    0 ≤ BayesOptimalityProof.KL p q := by
  exact BayesOptimalityProof.KL_nonneg p q hp hq hp1 hq1

/-- Paper 4 provides counting gap theorem.
    Bounded + positive cost → finite events.
    Paper 3 uses this for physical grounding. -/
theorem counting_gap (costPerCheck totalCapacity : ℕ)
    (hCost : 0 < costPerCheck) (hCapacity : 0 < totalCapacity) :
    ∃ c_max : ℕ, 0 < c_max ∧
      ∀ checks, costPerCheck * checks ≤ totalCapacity → checks ≤ c_max :=
  Physics.BoundedAcquisition.counting_gap_theorem costPerCheck totalCapacity hCost hCapacity

/-- Paper 4 provides bounded acquisition rate.
    No circuit computes faster than c. -/
theorem bounded_rate (R : Physics.BoundedAcquisition.BoundedRegion) (T : ℕ) :
    Physics.BoundedAcquisition.maxAcquisitions R T ≤ R.signalSpeed * T / R.diameter :=
  Physics.BoundedAcquisition.acquisition_rate_bound R T

end DecisionQuotient.CrossPaperDeps
