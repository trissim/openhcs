/-
  Paper 4: Decision-Relevant Uncertainty

  Summary.lean - mechanically backed summary aliases

  This module exposes compact theorem names that alias concrete results proved
  in the underlying modules. It does not introduce placeholder statements.
-/

import DecisionQuotient.Hardness
import DecisionQuotient.Tractability.BoundedActions
import DecisionQuotient.Tractability.BoundedStateSpace
import DecisionQuotient.Tractability.SingleAction
import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.MultiplicativeSeparable
import DecisionQuotient.Tractability.Dominance
import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Tightness
import DecisionQuotient.Tractability.FPT
import DecisionQuotient.Dichotomy
import DecisionQuotient.ComplexityMain

namespace DecisionQuotient.Summary

/-- coNP-hardness reduction core for SUFFICIENCY-CHECK. -/
theorem conp_completeness {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  (DecisionQuotient.tautology_iff_sufficient φ).symm

/-- Bounded-actions tractability alias. -/
theorem bounded_actions_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [CoordinateSpace S n]
    [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
    (k : ℕ) (cdp : ComputableDecisionProblem A S)
    (hcard : Fintype.card A ≤ k) :
    ∃ (decide : Finset (Fin n) → Bool),
      ∀ I, decide I = true ↔ cdp.toAbstract.isSufficient I :=
  DecisionQuotient.sufficiency_poly_bounded_actions (k := k) (cdp := cdp) hcard

/-- Separable-utility tractability alias. -/
theorem separable_utility_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hsep : SeparableUtility (dp := dp)) :
    ∃ algo : Finset (Fin n) → Bool,
      ∀ I, algo I = true ↔ dp.isSufficient I :=
  DecisionQuotient.sufficiency_poly_separable (dp := dp) hsep

/-- Tree-structured tractability alias. -/
theorem tree_structure_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [CoordinateSpace S n]
    [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
    (cdp : ComputableDecisionProblem A S)
    (deps : Fin n → Finset (Fin n)) (htree : TreeStructured deps) :
    ∃ algo : Finset (Fin n) → Bool,
      ∀ I, algo I = true ↔ cdp.toAbstract.isSufficient I :=
  DecisionQuotient.sufficiency_poly_tree_structured (cdp := cdp) deps htree

/-- Multiplicative-separable tractability alias. -/
theorem multiplicative_separable_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (I : Finset (Fin n))
    (hcase :
      (∀ s, hms.stateFactor s > 0) ∨
        (∀ s, hms.stateFactor s < 0) ∨
          (∀ s, hms.stateFactor s = 0)) :
    dp.isSufficient I :=
  DecisionQuotient.MultiplicativeSeparable.empty_sufficient
    (dp := dp) (hms := hms) (I := I) hcase

/-- Strict-dominance tractability alias. -/
theorem strict_global_dominance_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : StrictGlobalDominance (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.StrictGlobalDominance.empty_sufficient (dp := dp) (hdom := hdom) I

/-- Constant-optimal-set tractability alias. -/
theorem constant_optimal_set_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hconst : ConstantOptimalSet (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.ConstantOptimalSet.empty_sufficient (dp := dp) (hconst := hconst) I

/-- Single-action tractability alias. -/
theorem single_action_all_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hcard : dp.actions.card = 1)
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.single_action_all_sufficient (dp := dp) hcard I

/-- Bounded-state-space brute-force tractability alias. -/
theorem bounded_state_space_bruteforce
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [Fintype S]
    (k : ℕ) (hk : Fintype.card S ≤ k)
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (I : Finset (Fin n)) :
    dp.isSufficient I ↔
      ∀ (s₁ : S) (hs₁ : s₁ ∈ dp.states) (s₂ : S) (hs₂ : s₂ ∈ dp.states),
        (∀ i ∈ I, CoordinateSpace.proj s₁ i = CoordinateSpace.proj s₂ i) →
          dp.optimalActions s₁ = dp.optimalActions s₂ :=
  DecisionQuotient.bounded_state_sufficiency_trivial k hk dp I

/-- Tightness alias. -/
theorem tractability_tightness :
    (∀ (n : ℕ) (φ : Formula n),
      (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology) ∧
    (∀ (n : ℕ) (φ : Formula n), (∃ a, φ.eval a = false) →
      ¬∃ (av : ReductionAction → ℝ) (sv : ReductionState n → ℝ),
        ∀ a s, reductionUtility φ a s = av a + sv s) ∧
    (∀ {A S : Type*} [Unique A] (dp : DecisionProblem A S)
      {n : ℕ} [CoordinateSpace S n] (I : Finset (Fin n)), dp.isSufficient I) :=
  DecisionQuotient.tractability_conditions_tight

/-- Parameterized-results alias. -/
theorem parameterized_results :
    (∀ {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
        {n : ℕ} [CoordinateSpace S n]
        [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
        (cdp : ComputableDecisionProblem A S),
        ∃ f : ℕ → ℕ, ∃ algo : Finset (Fin n) → Bool,
          (∀ I, algo I = true ↔ cdp.toAbstract.isSufficient I) ∧
          (∀ k, 1 ≤ f k)) ∧
    (∀ {n : ℕ} (φ : Formula n),
      (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology) :=
  DecisionQuotient.parameterized_complexity_summary

/-- Dichotomy alias in terms of minimal-sufficient-set size relative to state
cardinality. -/
theorem complexity_dichotomy
    {S : Type*} {n : ℕ}
    [Fintype S] [Nonempty S]
    (I : Finset (Fin n)) :
    (I.card ≤ Nat.log 2 (Fintype.card S)) ∨
    (I.card > Nat.log 2 (Fintype.card S)) := by
  by_cases h : I.card ≤ Nat.log 2 (Fintype.card S)
  · exact Or.inl h
  · exact Or.inr (lt_of_not_ge h)

end DecisionQuotient.Summary
