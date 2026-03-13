/-
  Paper 4: Decision-Relevant Uncertainty

  Tractability/MultiplicativeSeparable.lean - Multiplicative separable utilities

  Key result: when utility factorizes as U(a,s) = f(a) * g(s) and g has constant sign,
  the optimal action set is independent of state.
-/

import DecisionQuotient.Finite
import DecisionQuotient.Sufficiency
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient

open Classical

variable {A S : Type*} [DecidableEq A] [DecidableEq S]

/-- A multiplicative separable utility structure: U(a,s) = f(a) * g(s). -/
structure MultiplicativeSeparable (dp : FiniteDecisionProblem (A := A) (S := S)) where
  actionFactor : A → ℝ
  stateFactor : S → ℝ
  utility_eq : ∀ a s, (dp.utility a s : ℝ) = actionFactor a * stateFactor s

/-- Positive state factor: optimal actions are exactly the argmax of `actionFactor`. -/
theorem MultiplicativeSeparable.opt_pos_factor
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hpos : ∀ s, hms.stateFactor s > 0)
    (s : S) :
    dp.optimalActions s =
      dp.actions.filter (fun a => ∀ a' ∈ dp.actions, hms.actionFactor a' ≤ hms.actionFactor a) := by
  ext a
  constructor
  · intro ha
    rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).1 ha with
      ⟨h_in, hmax⟩
    refine Finset.mem_filter.2 ⟨h_in, fun a' ha' => ?_⟩
    have huR : (dp.utility a' s : ℝ) ≤ (dp.utility a s : ℝ) := by
      exact_mod_cast hmax a' ha'
    have u_a' : (dp.utility a' s : ℝ) = hms.actionFactor a' * hms.stateFactor s := by
      simp [hms.utility_eq]
    have u_a : (dp.utility a s : ℝ) = hms.actionFactor a * hms.stateFactor s := by
      simp [hms.utility_eq]
    have hprod :
        hms.actionFactor a' * hms.stateFactor s ≤
          hms.actionFactor a * hms.stateFactor s := by
      simpa [u_a', u_a] using huR
    exact le_of_mul_le_mul_right hprod (hpos s)
  · intro ha
    rcases Finset.mem_filter.1 ha with ⟨h_in, hmax⟩
    refine (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).2
      ⟨h_in, fun a' ha' => ?_⟩
    have hprod :
        hms.actionFactor a' * hms.stateFactor s ≤
          hms.actionFactor a * hms.stateFactor s := by
      exact mul_le_mul_of_nonneg_right (hmax a' ha') (le_of_lt (hpos s))
    have u_a' : (dp.utility a' s : ℝ) = hms.actionFactor a' * hms.stateFactor s := by
      simp [hms.utility_eq]
    have u_a : (dp.utility a s : ℝ) = hms.actionFactor a * hms.stateFactor s := by
      simp [hms.utility_eq]
    have huR : (dp.utility a' s : ℝ) ≤ (dp.utility a s : ℝ) := by
      simpa [u_a', u_a] using hprod
    exact_mod_cast huR

/-- If the state factor is positive everywhere, optimal actions are state-independent. -/
theorem MultiplicativeSeparable.opt_independent_of_pos
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hpos : ∀ s, hms.stateFactor s > 0)
    (s s' : S) :
    dp.optimalActions s = dp.optimalActions s' := by
  rw [opt_pos_factor dp hms hpos s, opt_pos_factor dp hms hpos s']

/-- Empty coordinate set is sufficient when the state factor is positive everywhere. -/
theorem MultiplicativeSeparable.empty_sufficient_pos_factor
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hpos : ∀ s, hms.stateFactor s > 0)
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  exact opt_independent_of_pos dp hms hpos s s'

/-- Negative state factor: optimal actions are exactly the argmin of `actionFactor`. -/
theorem MultiplicativeSeparable.opt_neg_factor
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hneg : ∀ s, hms.stateFactor s < 0)
    (s : S) :
    dp.optimalActions s =
      dp.actions.filter (fun a => ∀ a' ∈ dp.actions, hms.actionFactor a ≤ hms.actionFactor a') := by
  ext a
  constructor
  · intro ha
    rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).1 ha with
      ⟨h_in, hmax⟩
    refine Finset.mem_filter.2 ⟨h_in, fun a' ha' => ?_⟩
    have huR : (dp.utility a' s : ℝ) ≤ (dp.utility a s : ℝ) := by
      exact_mod_cast hmax a' ha'
    have u_a' : (dp.utility a' s : ℝ) = hms.actionFactor a' * hms.stateFactor s := by
      simp [hms.utility_eq]
    have u_a : (dp.utility a s : ℝ) = hms.actionFactor a * hms.stateFactor s := by
      simp [hms.utility_eq]
    have hprod :
        hms.actionFactor a' * hms.stateFactor s ≤
          hms.actionFactor a * hms.stateFactor s := by
      simpa [u_a', u_a] using huR
    exact (mul_le_mul_right_of_neg (hneg s)).1 hprod
  · intro ha
    rcases Finset.mem_filter.1 ha with ⟨h_in, hmin⟩
    refine (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).2
      ⟨h_in, fun a' ha' => ?_⟩
    have hprod :
        hms.actionFactor a' * hms.stateFactor s ≤
          hms.actionFactor a * hms.stateFactor s := by
      exact (mul_le_mul_right_of_neg (hneg s)).2 (hmin a' ha')
    have u_a' : (dp.utility a' s : ℝ) = hms.actionFactor a' * hms.stateFactor s := by
      simp [hms.utility_eq]
    have u_a : (dp.utility a s : ℝ) = hms.actionFactor a * hms.stateFactor s := by
      simp [hms.utility_eq]
    have huR : (dp.utility a' s : ℝ) ≤ (dp.utility a s : ℝ) := by
      simpa [u_a', u_a] using hprod
    exact_mod_cast huR

/-- If the state factor is negative everywhere, optimal actions are state-independent. -/
theorem MultiplicativeSeparable.opt_independent_of_neg
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hneg : ∀ s, hms.stateFactor s < 0)
    (s s' : S) :
    dp.optimalActions s = dp.optimalActions s' := by
  rw [opt_neg_factor dp hms hneg s, opt_neg_factor dp hms hneg s']

/-- Empty coordinate set is sufficient when the state factor is negative everywhere. -/
theorem MultiplicativeSeparable.empty_sufficient_neg_factor
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hneg : ∀ s, hms.stateFactor s < 0)
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  exact opt_independent_of_neg dp hms hneg s s'

/-- Zero state factor: every action ties, so all actions are optimal. -/
theorem MultiplicativeSeparable.opt_zero_factor
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hzero : ∀ s, hms.stateFactor s = 0)
    (s : S) :
    dp.optimalActions s = dp.actions := by
  ext a
  constructor
  · intro ha
    exact (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).1 ha |>.1
  · intro ha
    refine (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).2
      ⟨ha, fun a' _ => ?_⟩
    have h_u (ax : A) : (dp.utility ax s : ℝ) = 0 := by
      rw [hms.utility_eq, hzero s, mul_zero]
    have hu_a' : dp.utility a' s = 0 := by
      exact_mod_cast (h_u a')
    have hu_a : dp.utility a s = 0 := by
      exact_mod_cast (h_u a)
    simpa [hu_a', hu_a]

/-- If the state factor is zero everywhere, optimal actions are state-independent. -/
theorem MultiplicativeSeparable.opt_independent_of_zero
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hzero : ∀ s, hms.stateFactor s = 0)
    (s s' : S) :
    dp.optimalActions s = dp.optimalActions s' := by
  rw [opt_zero_factor dp hms hzero s, opt_zero_factor dp hms hzero s']

/-- Empty coordinate set is sufficient when the state factor is zero everywhere. -/
theorem MultiplicativeSeparable.empty_sufficient_zero_factor
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (hzero : ∀ s, hms.stateFactor s = 0)
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  exact opt_independent_of_zero dp hms hzero s s'

/-- Constant-sign multiplicative separability makes the empty coordinate set sufficient. -/
theorem MultiplicativeSeparable.empty_sufficient
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (I : Finset (Fin n))
    (hcase :
      (∀ s, hms.stateFactor s > 0) ∨
        (∀ s, hms.stateFactor s < 0) ∨
          (∀ s, hms.stateFactor s = 0)) :
    dp.isSufficient I := by
  rcases hcase with hpos | hrest
  · exact empty_sufficient_pos_factor dp hms hpos I
  · rcases hrest with hneg | hzero
    · exact empty_sufficient_neg_factor dp hms hneg I
    · exact empty_sufficient_zero_factor dp hms hzero I

end DecisionQuotient
