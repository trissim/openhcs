/-
  Paper 4: Decision-Relevant Uncertainty

  Tractability/Dominance.lean - dominance-based tractable cases.
-/

import DecisionQuotient.Finite
import DecisionQuotient.Sufficiency

namespace DecisionQuotient

open Classical

variable {A S : Type*} [DecidableEq A] [DecidableEq S]

/-! ## Strict Global Dominance -/

/-- A decision problem has a strictly dominant action if one action strictly dominates
all other available actions at every state. -/
structure StrictGlobalDominance (dp : FiniteDecisionProblem (A := A) (S := S)) where
  dominant : A
  dominant_in_actions : dominant ∈ dp.actions
  prove_strict :
    ∀ (s : S) (a : A), a ≠ dominant → a ∈ dp.actions → dp.utility dominant s > dp.utility a s

/-- If an action strictly dominates all others everywhere, it is uniquely optimal. -/
theorem StrictGlobalDominance.opt_singleton
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : StrictGlobalDominance (dp := dp))
    (s : S) :
    dp.optimalActions s = {hdom.dominant} := by
  ext a
  constructor
  · intro ha
    rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s) (a := a)).1 ha with
      ⟨haA, hmax⟩
    by_cases heq : a = hdom.dominant
    · simpa [heq]
    · have hle := hmax hdom.dominant hdom.dominant_in_actions
      have hstrict := hdom.prove_strict s a heq haA
      exact False.elim ((not_le_of_gt hstrict) hle)
  · intro ha
    have heq : a = hdom.dominant := by
      simpa using ha
    subst heq
    refine (FiniteDecisionProblem.mem_optimalActions_iff
      (dp := dp) (s := s) (a := hdom.dominant)).2 ?_
    refine ⟨hdom.dominant_in_actions, fun a' ha' => ?_⟩
    by_cases heq' : a' = hdom.dominant
    · simpa [heq']
    · exact le_of_lt (hdom.prove_strict s a' heq' ha')

/-- With strict global dominance, the empty coordinate set is sufficient. -/
theorem StrictGlobalDominance.empty_sufficient
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : StrictGlobalDominance (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  rw [StrictGlobalDominance.opt_singleton (dp := dp) (hdom := hdom) s,
    StrictGlobalDominance.opt_singleton (dp := dp) (hdom := hdom) s']

/-! ## Constant Optimal Set -/

/-- A decision problem has constant optimal set if the same finite set is optimal
at every state. -/
structure ConstantOptimalSet (dp : FiniteDecisionProblem (A := A) (S := S)) where
  constantOpt : Finset A
  prove_const : ∀ s : S, dp.optimalActions s = constantOpt

/-- If the optimal set is constant, the empty coordinate set is sufficient. -/
theorem ConstantOptimalSet.empty_sufficient
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hconst : ConstantOptimalSet (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  rw [hconst.prove_const s, hconst.prove_const s']

/-! ## Weak Global Dominance -/

/-- A weaker dominance condition: one action is at least as good as all others
everywhere, and strictly better than all others at some state. -/
structure WeakGlobalDominance (dp : FiniteDecisionProblem (A := A) (S := S)) where
  dominant : A
  dominant_in_actions : dominant ∈ dp.actions
  prove_weak : ∀ (s : S) (a : A), dp.utility dominant s ≥ dp.utility a s
  prove_strict_at_some : ∃ s₀, ∀ a, a ≠ dominant → dp.utility dominant s₀ > dp.utility a s₀

/-- Under weak global dominance, the dominant action is always optimal. -/
theorem WeakGlobalDominance.opt_contains
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : WeakGlobalDominance (dp := dp))
    (s : S) :
    hdom.dominant ∈ dp.optimalActions s := by
  refine (FiniteDecisionProblem.mem_optimalActions_iff
    (dp := dp) (s := s) (a := hdom.dominant)).2 ?_
  exact ⟨hdom.dominant_in_actions, fun a' _ => hdom.prove_weak s a'⟩

/-- At a state where strict dominance holds, the dominant action is uniquely optimal. -/
theorem WeakGlobalDominance.opt_singleton_at_strict
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : WeakGlobalDominance (dp := dp)) :
    dp.optimalActions (Classical.choose hdom.prove_strict_at_some) = {hdom.dominant} := by
  let s0 : S := Classical.choose hdom.prove_strict_at_some
  have hstrict : ∀ a, a ≠ hdom.dominant → dp.utility hdom.dominant s0 > dp.utility a s0 := by
    simpa [s0] using Classical.choose_spec hdom.prove_strict_at_some
  ext a
  constructor
  · intro ha
    by_cases heq : a = hdom.dominant
    · simpa [heq]
    · rcases (FiniteDecisionProblem.mem_optimalActions_iff (dp := dp) (s := s0) (a := a)).1 ha with
        ⟨haA, hmax⟩
      have hle := hmax hdom.dominant hdom.dominant_in_actions
      have hgt := hstrict a heq
      exact False.elim ((not_le_of_gt hgt) hle)
  · intro ha
    have heq : a = hdom.dominant := by
      simpa using ha
    subst heq
    simpa [s0] using WeakGlobalDominance.opt_contains (dp := dp) (hdom := hdom) s0

/-! ## Dominance Detection -/

/-- Check whether the head of the action list strictly dominates all later actions. -/
noncomputable def FiniteDecisionProblem.hasStrictDominant
    (dp : FiniteDecisionProblem (A := A) (S := S)) : Option A := by
  classical
  let actions := dp.actions.toList
  exact match actions with
  | [] => none
  | a :: as =>
      if ∀ a' ∈ as, ∀ s, dp.utility a s > dp.utility a' s then some a else none

noncomputable def FiniteDecisionProblem.hasStrictDominant_implies_StrictGlobalDominance
    {dp : FiniteDecisionProblem (A := A) (S := S)} {a : A}
    (h : dp.hasStrictDominant = some a) :
    StrictGlobalDominance (dp := dp) := by
  classical
  cases hlist : dp.actions.toList with
  | nil =>
      simp [FiniteDecisionProblem.hasStrictDominant, hlist] at h
  | cons b bs =>
      by_cases hcond : ∀ a' ∈ bs, ∀ s, dp.utility b s > dp.utility a' s
      · have hinfo : (∀ a' ∈ bs, ∀ s, dp.utility b s > dp.utility a' s) ∧ b = a := by
          simpa [FiniteDecisionProblem.hasStrictDominant, hlist, hcond] using h
        have hba : b = a := hinfo.2
        subst hba
        have hdom_in : b ∈ dp.actions := by
          rw [← Finset.mem_toList]
          simpa [hlist]
        refine
          { dominant := b
            dominant_in_actions := hdom_in
            prove_strict := fun s a' hne ha' => ?_ }
        have hmem_list : a' ∈ b :: bs := by
          simpa [hlist] using Finset.mem_toList.mpr ha'
        rcases List.mem_cons.1 hmem_list with rfl | hmem_bs
        · contradiction
        · exact hcond a' hmem_bs s
      · simp [FiniteDecisionProblem.hasStrictDominant, hlist, hcond] at h

end DecisionQuotient
