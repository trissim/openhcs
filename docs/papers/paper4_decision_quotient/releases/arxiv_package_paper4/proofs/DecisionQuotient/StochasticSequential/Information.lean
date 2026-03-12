/-
  Paper 4b: Stochastic and Sequential Regimes

  Information.lean - Information-theoretic quantities for stochastic/sequential

  Key result: for binary-action stochastic problems, stochastic sufficiency
  (unique prior-optimal action) is equivalent to a strictly positive decision gap.

  The decision gap = |EU(accept) - EU(reject)| measures how resolved the
  optimal action is under the prior. Zero gap = tie = insufficient. Positive
  gap = unique optimum = sufficient.

  For the MAJSAT reduction:
    gap(stochProblem φ) = |satCount(φ)/2^n - 1/2|
  which is zero exactly when satCount = 2^(n-1), the PP-hard threshold.
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace DecisionQuotient.StochasticSequential

open Classical
open StochAction

/-! ## Shannon Entropy (kept for reference) -/

/-- Shannon entropy of a unit-action stochastic problem's state distribution -/
noncomputable def entropy {S : Type*} [Fintype S] (P : StochasticDecisionProblem Unit S) : ℝ :=
  ∑ s, if P.distribution s > 0 then -P.distribution s * Real.log (P.distribution s) else 0

/-! ## Decision Gap

The decision gap is the absolute margin between the two actions' expected utilities.
It is zero iff the prior leaves the controller indifferent (a tie), and positive iff one
action strictly dominates — i.e., the optimal action is uniquely determined.
-/

/-- Decision gap: |EU(accept) - EU(reject)|.
    Positive iff there is a unique optimal action under the prior. -/
noncomputable def decisionGap {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) : ℝ :=
  |stochasticExpectedUtility P StochAction.accept -
   stochasticExpectedUtility P StochAction.reject|

/-- Gap is always nonneg -/
theorem decisionGap_nonneg {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) :
    0 ≤ decisionGap P := abs_nonneg _

/-- Gap is zero iff the two expected utilities are equal (tie) -/
theorem decisionGap_zero_iff {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) :
    decisionGap P = 0 ↔
    stochasticExpectedUtility P StochAction.accept =
    stochasticExpectedUtility P StochAction.reject := by
  unfold decisionGap
  rw [abs_eq_zero, sub_eq_zero]

/-! ## Singleton Optimal Set Characterizations -/

/-- stochasticOpt = {accept} ↔ EU(reject) < EU(accept) -/
theorem stochasticOpt_eq_accept_iff {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) :
    P.stochasticOpt = {StochAction.accept} ↔
    stochasticExpectedUtility P StochAction.reject <
    stochasticExpectedUtility P StochAction.accept := by
  constructor
  · intro h
    -- reject ∉ {accept}, so reject ∉ stochasticOpt
    have hrej : StochAction.reject ∉ P.stochasticOpt := by rw [h]; simp
    unfold StochasticDecisionProblem.stochasticOpt at hrej
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hrej
    obtain ⟨a', ha'⟩ := hrej
    cases a' with
    | reject => linarith
    | accept => exact ha'
  · intro hlt
    rw [Set.eq_singleton_iff_unique_mem]
    refine ⟨?_, ?_⟩
    · -- accept ∈ stochasticOpt
      unfold StochasticDecisionProblem.stochasticOpt
      simp only [Set.mem_setOf_eq]
      intro a'; cases a' with
      | accept => exact le_refl _
      | reject => exact le_of_lt hlt
    · -- only accept satisfies ∀ a', eu a' ≤ eu x
      intro x hx
      unfold StochasticDecisionProblem.stochasticOpt at hx
      simp only [Set.mem_setOf_eq] at hx
      cases x with
      | accept => rfl
      | reject =>
        -- hx: ∀ a', eu a' ≤ eu reject, so eu accept ≤ eu reject, contradicts hlt
        have := hx StochAction.accept
        linarith

/-- stochasticOpt = {reject} ↔ EU(accept) < EU(reject) -/
theorem stochasticOpt_eq_reject_iff {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) :
    P.stochasticOpt = {StochAction.reject} ↔
    stochasticExpectedUtility P StochAction.accept <
    stochasticExpectedUtility P StochAction.reject := by
  constructor
  · intro h
    have hacc : StochAction.accept ∉ P.stochasticOpt := by rw [h]; simp
    unfold StochasticDecisionProblem.stochasticOpt at hacc
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hacc
    obtain ⟨a', ha'⟩ := hacc
    cases a' with
    | accept => linarith
    | reject => exact ha'
  · intro hlt
    rw [Set.eq_singleton_iff_unique_mem]
    refine ⟨?_, ?_⟩
    · unfold StochasticDecisionProblem.stochasticOpt
      simp only [Set.mem_setOf_eq]
      intro a'; cases a' with
      | reject => exact le_refl _
      | accept => exact le_of_lt hlt
    · intro x hx
      unfold StochasticDecisionProblem.stochasticOpt at hx
      simp only [Set.mem_setOf_eq] at hx
      cases x with
      | reject => rfl
      | accept =>
        have := hx StochAction.reject
        linarith

/-! ## Central Theorem: Sufficiency ↔ Positive Gap -/

/-- Stochastic sufficiency (∅) is equivalent to a strictly positive decision gap.
    This is the information-theoretic characterization: the prior resolves the
    decision iff the expected utility gap is nonzero.

    Proof sketch:
    - Sufficient (∃ unique optimal action) → EU(accept) ≠ EU(reject) → gap > 0
    - gap > 0 → EU(accept) ≠ EU(reject) → one strictly dominates → unique optimal -/
theorem stochastic_sufficient_iff_gap_pos {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) :
    StochasticSufficient P ∅ ↔ 0 < decisionGap P := by
  rw [stochasticSufficient_empty_iff]
  unfold decisionGap
  rw [abs_pos, sub_ne_zero]
  constructor
  · -- Sufficient → EU(accept) ≠ EU(reject)
    rintro ⟨a, ha⟩
    cases a with
    | accept =>
      intro heq
      have hlt := (stochasticOpt_eq_accept_iff P).mp ha
      linarith
    | reject =>
      intro heq
      have hlt := (stochasticOpt_eq_reject_iff P).mp ha
      linarith
  · -- EU(accept) ≠ EU(reject) → Sufficient
    intro hne
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · exact ⟨StochAction.reject, (stochasticOpt_eq_reject_iff P).mpr hlt⟩
    · exact ⟨StochAction.accept, (stochasticOpt_eq_accept_iff P).mpr hgt⟩

/-! ## MAJSAT Gap Computation -/

/-- The gap of the MAJSAT reduction = |satCount/2^n - 1/2|.
    This follows directly from EU(accept) = satCount/2^n and EU(reject) = 1/2. -/
theorem stochProblem_gap_eq (φ : Formula n) :
    decisionGap (stochProblem φ) =
    |(φ.satCount : ℝ) / (2 : ℝ)^n - 1 / 2| := by
  unfold decisionGap
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject
  norm_num

/-- The MAJSAT problem is stochastically sufficient iff satCount ≠ 2^n/2
    (i.e., the formula is not exactly at the PP-hard tie threshold).

    Proof: purely in ℕ arithmetic, using the three cases from Basic.lean:
    - satCount > 2^n/2 → opt = {accept} → sufficient
    - satCount < 2^n/2 → opt = {reject} → sufficient
    - satCount = 2^n/2 → opt = {accept,reject} (not singleton) → not sufficient -/
theorem majsat_sufficient_iff_not_tie (φ : Formula n) (hn : n ≥ 1) :
    StochasticSufficient (stochProblem φ) ∅ ↔ φ.satCount ≠ 2^n / 2 := by
  rw [stochasticSufficient_empty_iff]
  constructor
  · -- Sufficient → satCount ≠ 2^n/2
    intro ⟨a, ha⟩ htie
    -- If satCount = 2^n/2, both actions are optimal (not a singleton)
    have hboth := exact_half_both_optimal φ hn htie
    -- ha says opt = {a}, hboth says opt = {accept, reject}: contradiction
    rw [ha] at hboth
    -- {a} = {accept, reject} is impossible since |{accept,reject}| = 2
    have : StochAction.reject ∈ ({a} : Set StochAction) := by
      rw [hboth]; simp
    simp only [Set.mem_singleton_iff] at this
    -- this: reject = a, so {a} = {reject}, but hboth says {reject} = {accept, reject}
    subst this
    have : StochAction.accept ∈ ({StochAction.reject} : Set StochAction) := by
      rw [hboth]; simp
    simp at this
  · -- satCount ≠ 2^n/2 → Sufficient
    intro hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
    · exact ⟨StochAction.reject, strict_not_majsat_reject_unique φ hlt⟩
    · exact ⟨StochAction.accept, strict_majsat_accept_unique φ hgt⟩

/-! ## Information Sufficiency

Information sufficiency is the decision-theoretic notion: coordinate set I is
informationally sufficient when knowing the I-fiber uniquely resolves the optimal
action for every fiber. This is exactly `StochasticSufficient`, reframed as a
named interface for the information-theoretic interpretation.
-/

/-- Information sufficiency: I contains enough signal to resolve the decision.
    For every fiber (value of I-coordinates), there is a unique optimal action.
    Equivalent to StochasticSufficient by definition. -/
def informationSufficient {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  StochasticSufficient P I

/-- I-sufficiency for binary actions and ∅ is equivalent to a positive decision gap.
    The gap is the information-theoretic signal: zero gap = no information = observe. -/
theorem informationSufficient_empty_iff_gap_pos {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) :
    informationSufficient P (∅ : Finset (Fin n)) ↔ 0 < decisionGap P :=
  stochastic_sufficient_iff_gap_pos P

/-- I-sufficiency implies J-sufficiency for any subset J ⊆ I when
    the finer conditioning of I resolves all J-fibers.
    (Special case: ∅-sufficiency follows from gap > 0 for binary actions.) -/
theorem informationSufficient_empty_of_gap_pos {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S)
    (h : 0 < decisionGap P) :
    informationSufficient P (∅ : Finset (Fin n)) :=
  (informationSufficient_empty_iff_gap_pos P).mpr h

end DecisionQuotient.StochasticSequential
