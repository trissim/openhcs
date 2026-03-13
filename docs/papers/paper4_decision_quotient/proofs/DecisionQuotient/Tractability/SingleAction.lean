/-
  Paper 4: Decision-Relevant Uncertainty

  Tractability/SingleAction.lean - Trivial Case: Single Action

  Key Result: If |A| = 1, any coordinate set is sufficient.
  
  With only one action, that action is always optimal regardless of state.

  ## Dependencies
  - Chain: Sufficiency.lean → here (degenerate case)
-/

import DecisionQuotient.Finite
import Mathlib.Data.Finset.Card

namespace DecisionQuotient

variable {A S : Type*} [DecidableEq A] [DecidableEq S]


/-! ## Single Action -/

/- With only one action, any coordinate set is sufficient. -/
theorem single_action_all_sufficient
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hcard : dp.actions.card = 1)
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  -- sufficiency follows because optimalActions does not depend on state; prove equality to dp.actions for each state
  intro s hs s' hs' _
  -- pick the unique action as a term so it's available in all nested tactic blocks
  let uniqueA := Classical.choose (Finset.card_eq_one.mp hcard)
  have ha_unique := Classical.choose_spec (Finset.card_eq_one.mp hcard)
  have eq_s : dp.optimalActions s = dp.actions := by
    ext x
    constructor
    · intro hx
      rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := x)).1 hx with
        ⟨hmem, _⟩
      exact hmem
    · intro hx
      -- rewrite actions to the singleton and use the unique witness
      rw [ha_unique] at hx
      have h_eq : x = uniqueA := Finset.mem_singleton.1 hx
      subst h_eq
      apply (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := uniqueA)).mpr
      constructor
      · exact ha_unique.symm ▸ Finset.mem_singleton_self uniqueA
      · intro a' ha'
        rw [ha_unique] at ha'
        have h_eq2 : a' = uniqueA := Finset.mem_singleton.1 ha'
        subst h_eq2
        exact le_refl (dp.utility uniqueA s)
  have eq_s' : dp.optimalActions s' = dp.actions := by
    ext x
    constructor
    · intro hx
      rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s') (a := x)).1 hx with
        ⟨hmem, _⟩
      exact hmem
    · intro hx
      -- rewrite actions to the singleton and use the unique witness
      rw [ha_unique] at hx
      have h_eq : x = uniqueA := Finset.mem_singleton.1 hx
      subst h_eq
      apply (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s') (a := uniqueA)).mpr
      constructor
      · exact ha_unique.symm ▸ Finset.mem_singleton_self uniqueA
      · intro a' ha'
        rw [ha_unique] at ha'
        have h_eq2 : a' = uniqueA := Finset.mem_singleton.1 ha'
        subst h_eq2
        exact le_refl (dp.utility uniqueA s')
  exact (eq_s).trans (eq_s').symm

end DecisionQuotient
