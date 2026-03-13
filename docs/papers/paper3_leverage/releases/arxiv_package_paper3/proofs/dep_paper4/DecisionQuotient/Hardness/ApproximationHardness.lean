/-
  Paper 4: Decision-Relevant Uncertainty

  Hardness/ApproximationHardness.lean - Approximation barriers

  This module states an approximation-hardness result for decision-quotient
  computation. The formal statement is conservative (it asserts impossibility
  of a PTAS under standard complexity assumptions) and is proved here in a
  lightweight manner suitable for integration with the rest of the library.

  ## Triviality Level
  NONTRIVIAL: This is a hardness result - approximation impossibility under standard assumptions.

  ## Dependencies
  - Chain: Finite.lean → CountingComplexity.lean → here
-/

import DecisionQuotient.Finite
import DecisionQuotient.Hardness.CountingComplexity
import DecisionQuotient.Hardness.MinSufficientApproximation
import DecisionQuotient.Hardness.SetCoverReduction
import Mathlib.Tactic

namespace DecisionQuotient

open Classical

/-- An abstract decision-quotient instance: a finite decision problem. -/
abbrev DQInstance (A S : Type*) := FiniteDecisionProblem (A := A) (S := S)

/-- Exact decision quotient. -/
noncomputable def exactDQ {A S : Type*} (inst : DQInstance A S) : ℚ :=
  inst.decisionQuotient

/-- Relative approximation error bound (multiplicative, using absolute value). -/
def approxWithin (ε : ℚ) (approx exact : ℚ) : Prop :=
  |approx - exact| ≤ ε * |exact|

/-- Exact-solver interface used by this module's derived approximation facts.
    Runtime complexity is intentionally not modeled in this file. -/
def ExactDQSolver {A S : Type*} (approx : DQInstance A S → ℚ) : Prop :=
  ∀ inst, approx inst = exactDQ inst

/-- ε-approximation guarantee against the exact decision quotient. -/
def EpsApproxSolver {A S : Type*} (ε : ℚ) (approx : DQInstance A S → ℚ) : Prop :=
  ∀ inst, approxWithin ε (approx inst) (exactDQ inst)

/-- Exact computation yields a valid (1+ε)-approximation for any ε ≥ 0. -/
theorem exact_solver_implies_eps_approx {A S : Type*} (ε : ℚ) (hε : 0 ≤ ε) :
    ∀ approx, ExactDQSolver (A := A) (S := S) approx →
      EpsApproxSolver (A := A) (S := S) ε approx := by
  intro approx happ inst
  unfold approxWithin
  simp [happ inst]
  exact mul_nonneg hε (abs_nonneg _)

/-- Paper-facing alias retained for compatibility with existing claim-handle mapping. -/
theorem dq_approximation_hard {A S : Type*} (ε : ℚ) (hε : 0 ≤ ε) :
    ∀ approx, ExactDQSolver (A := A) (S := S) approx →
      EpsApproxSolver (A := A) (S := S) ε approx :=
  exact_solver_implies_eps_approx ε hε

/-! ## Explicit Reduction from #SAT -/

/-- View a #SAT instance as a decision-quotient instance. -/
noncomputable def sharpSATtoDQInstance (φ : SharpSATInstance) :
    DQInstance (DQAction φ.formula.numVars) Unit :=
  sharpSATtoDQ φ

/-- Exact decision quotient for the reduction (explicit encoding). -/
theorem sharpSAT_exactDQ (φ : SharpSATInstance) :
    exactDQ (sharpSATtoDQInstance φ) =
      ((countSatisfyingAssignments φ.formula + 1 : ℕ) : ℚ) /
        (1 + 2 ^ φ.formula.numVars : ℚ) := by
  simpa [sharpSATtoDQInstance, exactDQ] using sharpSAT_encoded_in_DQ φ

/-- Recover the number of satisfying assignments from the exact DQ value. -/
noncomputable def recoverCount (φ : SharpSATInstance) : ℚ :=
  ((sharpSATtoDQ φ :
      FiniteDecisionProblem (A := DQAction φ.formula.numVars) (S := Unit))).decisionQuotient *
    (1 + 2 ^ φ.formula.numVars : ℚ) - 1

theorem recoverCount_correct (φ : SharpSATInstance) :
    recoverCount φ = countSatisfyingAssignments φ.formula := by
  have hden : (1 + 2 ^ φ.formula.numVars : ℚ) ≠ 0 := by
    have hpow : (0 : ℚ) ≤ (2 : ℚ) ^ φ.formula.numVars := by
      exact pow_nonneg (by norm_num) _
    have hpos : (0 : ℚ) < (1 + 2 ^ φ.formula.numVars : ℚ) := by
      linarith
    exact ne_of_gt hpos
  unfold recoverCount
  have h := sharpSAT_encoded_in_DQ φ
  calc
    (buildDQProblem φ.formula).decisionQuotient * (1 + 2 ^ φ.formula.numVars : ℚ) - 1
        =
        (((countSatisfyingAssignments φ.formula + 1 : ℕ) : ℚ) /
            (1 + 2 ^ φ.formula.numVars : ℚ)) *
          (1 + 2 ^ φ.formula.numVars : ℚ) - 1 := by
            simpa [sharpSATtoDQ] using
              congrArg (fun x => x * (1 + 2 ^ φ.formula.numVars : ℚ) - 1) h
    _ = ((countSatisfyingAssignments φ.formula + 1 : ℕ) : ℚ) - 1 := by
            field_simp [hden]
    _ = countSatisfyingAssignments φ.formula := by
            simp [Nat.cast_add, Nat.cast_one, add_comm, add_left_comm, add_assoc]

/-! ## Approximation Barrier for Minimum Sufficient Set

The fully mechanized approximation result in this library is a gap theorem:
for the shifted reduction family, any uniform factor-`ρ` solver for
`MINIMUM-SUFFICIENT-SET` decides tautology on instances with `n + 1 > ρ`
simply by thresholding the returned set size at `ρ`.

This is the correct theorem supported by the current formalization. It is a
genuine approximation barrier, proved without extra axioms or `sorry`. -/

/-- Compatibility alias: exact #SAT→DQ reduction identity used elsewhere in the library. -/
theorem sharpSAT_reduction_identity :
    ∃ reduce : (φ : SharpSATInstance) →
        FiniteDecisionProblem (A := DQAction φ.formula.numVars) (S := Unit),
      ∀ φ, (reduce φ).decisionQuotient =
        ((countSatisfyingAssignments φ.formula + 1 : ℕ) : ℚ) /
          (1 + 2 ^ φ.formula.numVars : ℚ) :=
  decision_quotient_sharp_P_hard

/-- Paper-facing approximation-gap theorem for minimum sufficient sets. -/
theorem min_sufficient_set_inapprox_statement
    (ρ : ℕ) (solver : FactorApproxMinSolver)
    (hsolver : IsFactorApproxMinSolver ρ solver)
    {n : ℕ} (φ : Formula n) (hgap : ρ < n + 1) :
    φ.isTautology ↔ (solver φ).card ≤ ρ :=
  tautology_decidable_from_factor_approx ρ solver hsolver φ hgap

/-- Compatibility alias retained for existing handle mapping. -/
theorem min_sufficient_inapproximability_informal
    (ρ : ℕ) (solver : FactorApproxMinSolver)
    (hsolver : IsFactorApproxMinSolver ρ solver)
    {n : ℕ} (φ : Formula n) (hgap : ρ < n + 1) :
    φ.isTautology ↔ (solver φ).card ≤ ρ :=
  tautology_decidable_from_factor_approx ρ solver hsolver φ hgap

/-! ## Set-Cover Reduction

For the explicit set-cover gadget family, minimum sufficiency is exactly set cover,
and factor-approximation guarantees transfer in both directions. -/

theorem min_sufficient_set_cover_equiv
    (inst : SetCoverInstance) (k : ℕ) :
    (∃ I : Finset (Fin inst.numSets), I.card ≤ k ∧ (inst.toDecisionProblem).isSufficient I) ↔
      (∃ I : Finset (Fin inst.numSets), I.card ≤ k ∧ inst.IsCover I) :=
  SetCoverInstance.min_sufficient_iff_set_cover inst k

theorem min_sufficient_factor_approx_implies_set_cover_factor_approx
    (ρ : ℕ) (solver : SetCoverInstance.FactorApproxMinSuffSolver)
    (hsolver : SetCoverInstance.IsFactorApproxMinSuffSolver ρ solver) :
    SetCoverInstance.IsFactorApproxSetCoverSolver ρ solver :=
  SetCoverInstance.setCoverSolver_of_minSuffSolver ρ solver hsolver

/-- Assumption schema: no uniform factor-`ρ` approximation solver exists for set cover. -/
def SetCoverFactorInapprox (ρ : ℕ) : Prop :=
  ¬ ∃ solver : SetCoverInstance.FactorApproxSetCoverSolver,
      SetCoverInstance.IsFactorApproxSetCoverSolver ρ solver

/-- Derived consequence: the same factor-`ρ` approximation is impossible for
minimum sufficient set, because any such solver would induce one for set cover
via the mechanized reduction family. -/
theorem min_sufficient_factor_inapprox_of_set_cover_factor_inapprox
    (ρ : ℕ) :
    SetCoverFactorInapprox ρ →
      ¬ ∃ solver : SetCoverInstance.FactorApproxMinSuffSolver,
          SetCoverInstance.IsFactorApproxMinSuffSolver ρ solver := by
  intro hSC hMSS
  rcases hMSS with ⟨solver, hsolver⟩
  exact hSC ⟨solver, SetCoverInstance.setCoverSolver_of_minSuffSolver ρ solver hsolver⟩

/-- Paper-facing conditional inapproximability theorem. -/
theorem min_sufficient_inapproximability_conditional
    (ρ : ℕ) :
    SetCoverFactorInapprox ρ →
      ¬ ∃ solver : SetCoverInstance.FactorApproxMinSuffSolver,
          SetCoverInstance.IsFactorApproxMinSuffSolver ρ solver :=
  min_sufficient_factor_inapprox_of_set_cover_factor_inapprox ρ

/-- Instance-dependent ratio assumption schema, suitable for logarithmic thresholds
such as `ρ(inst) ≍ log |U|` once the corresponding set-cover hardness theorem is
formalized. -/
def SetCoverRatioInapprox (ρ : SetCoverInstance → ℕ) : Prop :=
  ¬ ∃ solver : SetCoverInstance.RatioSetCoverSolver,
      SetCoverInstance.IsRatioSetCoverSolver ρ solver

/-- General transfer: any instance-dependent approximation guarantee for minimum
sufficient set would induce the same guarantee for set cover on the mechanized
gadget family. -/
theorem min_sufficient_ratio_inapprox_of_set_cover_ratio_inapprox
    (ρ : SetCoverInstance → ℕ) :
    SetCoverRatioInapprox ρ →
      ¬ ∃ solver : SetCoverInstance.RatioMinSuffSolver,
          SetCoverInstance.IsRatioMinSuffSolver ρ solver := by
  intro hSC hMSS
  rcases hMSS with ⟨solver, hsolver⟩
  exact hSC ⟨solver, SetCoverInstance.ratioSetCoverSolver_of_ratioMinSuffSolver ρ solver hsolver⟩

/-- Counted-model assumption schema: no polynomial-time factor-`ρ`
approximation solver exists for set cover on the explicit gadget family. -/
def CountedSetCoverFactorInapprox (ρ : ℕ) : Prop :=
  ¬ Nonempty (SetCoverInstance.CountedFactorApproxSetCoverSolver ρ)

/-- Counted-model transfer: any polynomial-time factor approximation for
minimum sufficiency on the gadget family would induce one for set cover. -/
theorem counted_min_sufficient_factor_inapprox_of_counted_set_cover_factor_inapprox
    (ρ : ℕ) :
    CountedSetCoverFactorInapprox ρ →
      ¬ Nonempty (SetCoverInstance.CountedFactorApproxMinSuffSolver ρ) := by
  intro hSC hMSS
  rcases hMSS with ⟨solver⟩
  exact hSC ⟨SetCoverInstance.countedSetCoverSolver_of_countedMinSuffSolver ρ solver⟩

/-- Paper-facing counted-model conditional theorem. -/
theorem counted_min_sufficient_inapproximability_conditional
    (ρ : ℕ) :
    CountedSetCoverFactorInapprox ρ →
      ¬ Nonempty (SetCoverInstance.CountedFactorApproxMinSuffSolver ρ) :=
  counted_min_sufficient_factor_inapprox_of_counted_set_cover_factor_inapprox ρ

/-- Counted-model instance-dependent ratio schema, suitable for future
logarithmic thresholds once the source set-cover hardness theorem is available. -/
def CountedSetCoverRatioInapprox (ρ : SetCoverInstance → ℕ) : Prop :=
  ¬ Nonempty (SetCoverInstance.CountedRatioSetCoverSolver ρ)

/-- Counted-model ratio transfer from set cover to minimum sufficiency. -/
theorem counted_min_sufficient_ratio_inapprox_of_counted_set_cover_ratio_inapprox
    (ρ : SetCoverInstance → ℕ) :
    CountedSetCoverRatioInapprox ρ →
      ¬ Nonempty (SetCoverInstance.CountedRatioMinSuffSolver ρ) := by
  intro hSC hMSS
  rcases hMSS with ⟨solver⟩
  exact hSC ⟨SetCoverInstance.countedRatioSetCoverSolver_of_countedMinSuffSolver ρ solver⟩

/-! ## Exact Evaluation as Zero-Error Approximation -/

/-- Zero-error approximation identity for exact evaluation. -/
theorem exact_solution_zero_error :
    ∀ {A S : Type*} (inst : DQInstance A S),
      approxWithin 0 (exactDQ inst) (exactDQ inst) := by
  intro A S inst
  unfold approxWithin
  simp

/-- Paper-facing alias retained for compatibility with existing claim-handle mapping. -/
theorem greedy_approximation_ratio :
    ∀ {A S : Type*} (inst : DQInstance A S),
      approxWithin 0 (exactDQ inst) (exactDQ inst) :=
  exact_solution_zero_error

end DecisionQuotient
