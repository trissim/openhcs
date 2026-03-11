/-
  Paper 4b: Stochastic and Sequential Regimes

  Computation.lean - Exact finite deciders for regime-typed queries

  This file does not claim standard PP/PSPACE membership via explicit TM
  witnesses. Instead, it internalizes exact boolean deciders for the finite
  query predicates that appear in the stochastic and sequential regimes, and
  proves those deciders correct.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.AlgorithmComplexity
import DecisionQuotient.PolynomialReduction

namespace DecisionQuotient.StochasticSequential

open Classical
open DecisionQuotient

/-- A boolean exactly decides a proposition. -/
def HasExactBoolDecider (Q : Prop) : Prop :=
  ∃ b : Bool, b = true ↔ Q

/-- A counted computation exactly decides a proposition within an explicit step
    bound. -/
def HasCountedSearchWitness (Q : Prop) (c : Counted Bool) (B : ℕ) : Prop :=
  (c.result = true ↔ Q) ∧ c.steps ≤ B

/-- Counted search through a finite list for an element satisfying a boolean
    predicate. Each predicate call costs one abstract step. -/
def countedAnyList {α : Type*} (xs : List α) (p : α → Bool) : Counted Bool :=
  match xs with
  | [] => Counted.pure false
  | x :: xs' =>
      Counted.bind (Counted.tick (p x)) fun bx =>
        if bx then Counted.pure true else countedAnyList xs' p

theorem countedAnyList_result_true_iff {α : Type*} (xs : List α) (p : α → Bool) :
    (countedAnyList xs p).result = true ↔ ∃ x ∈ xs, p x = true := by
  induction xs with
  | nil =>
      simp [countedAnyList, Counted.result, Counted.pure]
  | cons x xs ih =>
      by_cases hx : p x = true
      · simp [countedAnyList, Counted.bind, Counted.tick, Counted.pure, Counted.result, hx]
      · simpa [countedAnyList, Counted.bind, Counted.tick, Counted.result, hx] using ih

theorem countedAnyList_steps_le_length {α : Type*} (xs : List α) (p : α → Bool) :
    (countedAnyList xs p).steps ≤ xs.length := by
  induction xs with
  | nil =>
      simp [countedAnyList, Counted.steps, Counted.pure]
  | cons x xs ih =>
      by_cases hx : p x = true
      · simp [countedAnyList, Counted.bind, Counted.tick, Counted.pure, Counted.steps, hx]
      · simp [countedAnyList, Counted.bind, Counted.tick, Counted.steps, hx]
        cases hrec : countedAnyList xs p with
        | mk m b =>
            simp [countedAnyList, Counted.bind, Counted.tick, Counted.steps, hx, hrec] at ih ⊢
            omega

theorem countedAnyList_result_false_iff {α : Type*} (xs : List α) (p : α → Bool) :
    (countedAnyList xs p).result = false ↔ ∀ x ∈ xs, p x = false := by
  induction xs with
  | nil =>
      simp [countedAnyList, Counted.result, Counted.pure]
  | cons x xs ih =>
      by_cases hx : p x = true
      · simp [countedAnyList, Counted.bind, Counted.tick, Counted.pure, Counted.result, hx]
      · have hx' : p x = false := by cases hpx : p x <;> simp_all
        simpa [countedAnyList, Counted.bind, Counted.tick, Counted.result, hx, hx'] using ih

theorem decide_not_exists_true_iff {α : Type*} [DecidableEq α] (P : α → Prop) [DecidablePred P] :
    decide (¬ ∃ a, P a) = true ↔ ¬ ∃ a, P a := by
  simp

theorem decide_not_exists_false_iff {α : Type*} [DecidableEq α] (P : α → Prop) [DecidablePred P] :
    decide (¬ ∃ a, P a) = false ↔ ∃ a, P a := by
  simp

theorem decide_imp_true_iff (P Q : Prop) [Decidable P] [Decidable Q] :
    decide (P → Q) = true ↔ (P → Q) := by
  constructor <;> intro h
  · by_cases hP : P
    · by_cases hQ : Q
      · intro _; exact hQ
      · simp [hP, hQ] at h
    · intro _
      exact False.elim (hP ‹P›)
  · by_cases hP : P
    · by_cases hQ : Q
      · simp [hP, hQ]
      · have := h hP
        contradiction
    · simp [hP]

/-! ## Stochastic Query Deciders -/

/-- Exact boolean decider for stochastic preservation sufficiency. -/
noncomputable def stochasticPreservationBool
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (StochasticPreservationSufficient P I)

theorem stochasticPreservationBool_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    stochasticPreservationBool P I = true ↔ StochasticPreservationSufficient P I := by
  classical
  simp [stochasticPreservationBool]

/-- Exact boolean decider for stochastic sufficiency. -/
noncomputable def stochasticSufficientBool
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (StochasticSufficient P I)

theorem stochasticSufficientBool_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    stochasticSufficientBool P I = true ↔ StochasticSufficient P I := by
  classical
  simp [stochasticSufficientBool]

/-- Exact boolean decider for stochastic anchor sufficiency. -/
noncomputable def stochasticAnchorSufficientBool
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (StochasticAnchorSufficiencyCheck P I)

theorem stochasticAnchorSufficientBool_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    stochasticAnchorSufficientBool P I = true ↔ StochasticAnchorSufficiencyCheck P I := by
  classical
  simp [stochasticAnchorSufficientBool]

/-- Exact boolean decider for stochastic minimum sufficiency. -/
noncomputable def stochasticMinimumSufficiencyBool
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) : Bool := by
  classical
  exact decide (StochasticMinimumSufficiencyCheck P k)

theorem stochasticMinimumSufficiencyBool_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    stochasticMinimumSufficiencyBool P k = true ↔ StochasticMinimumSufficiencyCheck P k := by
  classical
  simp [stochasticMinimumSufficiencyBool]

theorem stochastic_sufficient_has_exact_bool_decider
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasExactBoolDecider (StochasticSufficient P I) := by
  exact ⟨stochasticSufficientBool P I, stochasticSufficientBool_spec P I⟩

theorem stochastic_anchor_has_exact_bool_decider
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasExactBoolDecider (StochasticAnchorSufficiencyCheck P I) := by
  exact ⟨stochasticAnchorSufficientBool P I, stochasticAnchorSufficientBool_spec P I⟩

theorem stochastic_minimum_has_exact_bool_decider
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    HasExactBoolDecider (StochasticMinimumSufficiencyCheck P k) := by
  exact ⟨stochasticMinimumSufficiencyBool P k, stochasticMinimumSufficiencyBool_spec P k⟩

theorem stochastic_preservation_has_exact_bool_decider
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasExactBoolDecider (StochasticPreservationSufficient P I) := by
  exact ⟨stochasticPreservationBool P I, stochasticPreservationBool_spec P I⟩

/-- Explicit exhaustive-search decider for the stochastic minimum query over all
    coordinate subsets. -/
noncomputable def countedStochasticMinimumSearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) : Counted Bool :=
  countedAnyList (Finset.powerset (Finset.univ : Finset (Fin n))).toList
    (fun I => decide (I.card ≤ k) && stochasticSufficientBool P I)

theorem countedStochasticMinimumSearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    (countedStochasticMinimumSearch P k).result = true ↔ StochasticMinimumSufficiencyCheck P k := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨I, hI, hPred⟩
    have hPred' : decide (I.card ≤ k) = true ∧ stochasticSufficientBool P I = true := by
      simpa using hPred
    exact ⟨I, by
      constructor
      · simpa using hPred'.1
      · exact (stochasticSufficientBool_spec P I).mp hPred'.2⟩
  · intro h
    rcases h with ⟨I, hCard, hSuff⟩
    have hmemPow : I ∈ Finset.powerset (Finset.univ : Finset (Fin n)) := by
      apply Finset.mem_powerset.mpr
      exact Finset.subset_univ I
    have hmemList : I ∈ (Finset.powerset (Finset.univ : Finset (Fin n))).toList := by
      simpa using hmemPow
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨I, hmemList, ?_⟩
    simp [hCard, stochasticSufficientBool_spec P I, hSuff]

theorem countedStochasticMinimumSearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    (countedStochasticMinimumSearch P k).steps ≤ 2 ^ n := by
  have hlen : (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length = 2 ^ n := by
    simp
  calc
    (countedStochasticMinimumSearch P k).steps
        ≤ (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length :=
          countedAnyList_steps_le_length _ _
    _ = 2 ^ n := hlen

/-- Explicit counted violation scan for stochastic sufficiency on a fixed
    information set. It searches for a state whose fiber optimum is not a
    singleton. -/
noncomputable def countedStochasticSufficiencySearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  let c := countedAnyList (Finset.univ.toList : List S)
    (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))
  (c.steps, !c.result)

theorem countedStochasticSufficiencySearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticSufficiencySearch P I).result = true ↔ StochasticSufficient P I := by
  constructor
  · intro h
    have hInnerFalse : (countedAnyList
        (Finset.univ.toList : List S)
        (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))).result = false := by
      unfold countedStochasticSufficiencySearch at h
      simpa [Counted.result] using h
    have hfalse := (countedAnyList_result_false_iff
      (Finset.univ.toList : List S)
      (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))).mp hInnerFalse
    intro s
    have hsMem : s ∈ (Finset.univ.toList : List S) := by
      simpa using (Finset.mem_toList.mpr (Finset.mem_univ s))
    have hsFalse : decide (¬ ∃ a : A, fiberOpt P I s = {a}) = false := hfalse s hsMem
    by_cases hsEx : ∃ a : A, fiberOpt P I s = {a}
    · exact hsEx
    · have hsTrue : decide (¬ ∃ a : A, fiberOpt P I s = {a}) = true := by simp [hsEx]
      rw [hsTrue] at hsFalse
      cases hsFalse
  · intro h
    have hInnerFalse : (countedAnyList
        (Finset.univ.toList : List S)
        (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))).result = false :=
      (countedAnyList_result_false_iff
      (Finset.univ.toList : List S)
      (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))).mpr (by
        intro s hsMem
        by_cases hsEx : ∃ a : A, fiberOpt P I s = {a}
        · simp [hsEx]
        · exact False.elim (hsEx (h s)))
    unfold countedStochasticSufficiencySearch
    simpa [Counted.result] using hInnerFalse

theorem countedStochasticSufficiencySearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticSufficiencySearch P I).steps ≤ Fintype.card S := by
  unfold countedStochasticSufficiencySearch
  simpa [Counted.steps] using countedAnyList_steps_le_length
    (Finset.univ.toList : List S)
    (fun s => decide (¬ ∃ a : A, fiberOpt P I s = {a}))

/-- Explicit counted violation scan for stochastic preservation on a fixed
    information set. It searches for a state where the conditional fiber optimum
    differs from the full-information optimizer. -/
noncomputable def countedStochasticPreservationSearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  let c := countedAnyList (Finset.univ.toList : List S)
    (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))
  (c.steps, !c.result)

theorem countedStochasticPreservationSearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticPreservationSearch P I).result = true ↔ StochasticPreservationSufficient P I := by
  constructor
  · intro h s
    have hInnerFalse : (countedAnyList
        (Finset.univ.toList : List S)
        (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))).result = false := by
      unfold countedStochasticPreservationSearch at h
      simpa [Counted.result] using h
    have hfalse := (countedAnyList_result_false_iff
      (Finset.univ.toList : List S)
      (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))).mp hInnerFalse
    have hsMem : s ∈ (Finset.univ.toList : List S) := by
      simpa using (Finset.mem_toList.mpr (Finset.mem_univ s))
    have hsFalse : decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s) = false := hfalse s hsMem
    by_cases hsNe : fiberOpt P I s ≠ P.toDecisionProblem.Opt s
    · have hsTrue : decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s) = true := by
        simp [hsNe]
      rw [hsTrue] at hsFalse
      cases hsFalse
    · exact by_contra fun hEq => hsNe hEq
  · intro h
    have hInnerFalse : (countedAnyList
        (Finset.univ.toList : List S)
        (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))).result = false :=
      (countedAnyList_result_false_iff
        (Finset.univ.toList : List S)
        (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))).mpr (by
          intro s hsMem
          by_cases hbad : fiberOpt P I s ≠ P.toDecisionProblem.Opt s
          · exact False.elim (hbad (h s))
          · simp [hbad])
    unfold countedStochasticPreservationSearch
    simpa [Counted.result] using hInnerFalse

theorem countedStochasticPreservationSearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticPreservationSearch P I).steps ≤ Fintype.card S := by
  unfold countedStochasticPreservationSearch
  simpa [Counted.steps] using countedAnyList_steps_le_length
    (Finset.univ.toList : List S)
    (fun s => decide (fiberOpt P I s ≠ P.toDecisionProblem.Opt s))

theorem stochasticPreservation_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (StochasticPreservationSufficient P I)
      (countedStochasticPreservationSearch P I)
      (Fintype.card S) := by
  constructor
  · exact countedStochasticPreservationSearch_spec P I
  · exact countedStochasticPreservationSearch_steps P I

/-- Explicit counted witness search for stochastic anchor sufficiency. -/
noncomputable def countedStochasticAnchorSearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  countedAnyList ((Finset.univ.product (Finset.univ : Finset A)).toList)
    (fun wa => decide
      (fiberOpt P I wa.1 = ({wa.2} : Set A) ∧
       ∀ s : S, agreeOn s wa.1 I → fiberOpt P I s = {wa.2}))

theorem countedStochasticAnchorSearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticAnchorSearch P I).result = true ↔ StochasticAnchorSufficiencyCheck P I := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨wa, _, hwa⟩
    exact ⟨wa.1, wa.2, by simpa [countedStochasticAnchorSearch] using hwa⟩
  · rintro ⟨s0, a, hs0, hrest⟩
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨(s0, a), ?_, ?_⟩
    · simpa using (Finset.mem_toList.mpr (Finset.mem_product.mpr ⟨Finset.mem_univ s0, Finset.mem_univ a⟩))
    · simpa [countedStochasticAnchorSearch, hs0, hrest]

theorem countedStochasticAnchorSearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticAnchorSearch P I).steps ≤ Fintype.card S * Fintype.card A := by
  calc
    (countedStochasticAnchorSearch P I).steps
        ≤ ((Finset.univ.product (Finset.univ : Finset A)).toList).length :=
          countedAnyList_steps_le_length _ _
    _ = Fintype.card S * Fintype.card A := by simp

theorem stochasticSufficiency_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (StochasticSufficient P I)
      (countedStochasticSufficiencySearch P I)
      (Fintype.card S) := by
  constructor
  · exact countedStochasticSufficiencySearch_spec P I
  · exact countedStochasticSufficiencySearch_steps P I

theorem stochasticAnchor_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (StochasticAnchorSufficiencyCheck P I)
      (countedStochasticAnchorSearch P I)
      (Fintype.card S * Fintype.card A) := by
  constructor
  · exact countedStochasticAnchorSearch_spec P I
  · exact countedStochasticAnchorSearch_steps P I

theorem stochasticMinimum_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    HasCountedSearchWitness
      (StochasticMinimumSufficiencyCheck P k)
      (countedStochasticMinimumSearch P k)
      (2 ^ n) := by
  constructor
  · exact countedStochasticMinimumSearch_spec P k
  · exact countedStochasticMinimumSearch_steps P k

/-! ## Sequential Query Deciders -/

/-- Exact boolean decider for sequential sufficiency. -/
noncomputable def sequentialSufficientBool
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (SequentialSufficient P I)

theorem sequentialSufficientBool_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    sequentialSufficientBool P I = true ↔ SequentialSufficient P I := by
  classical
  simp [sequentialSufficientBool]

/-- Exact boolean decider for sequential anchor sufficiency. -/
noncomputable def sequentialAnchorSufficientBool
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (SequentialAnchorSufficiencyCheck P I)

theorem sequentialAnchorSufficientBool_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    sequentialAnchorSufficientBool P I = true ↔ SequentialAnchorSufficiencyCheck P I := by
  classical
  simp [sequentialAnchorSufficientBool]

/-- Exact boolean decider for sequential minimum sufficiency. -/
noncomputable def sequentialMinimumSufficiencyBool
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) : Bool := by
  classical
  exact decide (SequentialMinimumSufficiencyCheck P k)

theorem sequentialMinimumSufficiencyBool_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    sequentialMinimumSufficiencyBool P k = true ↔ SequentialMinimumSufficiencyCheck P k := by
  classical
  simp [sequentialMinimumSufficiencyBool]

theorem sequential_sufficient_has_exact_bool_decider
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    HasExactBoolDecider (SequentialSufficient P I) := by
  exact ⟨sequentialSufficientBool P I, sequentialSufficientBool_spec P I⟩

theorem sequential_anchor_has_exact_bool_decider
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    HasExactBoolDecider (SequentialAnchorSufficiencyCheck P I) := by
  exact ⟨sequentialAnchorSufficientBool P I, sequentialAnchorSufficientBool_spec P I⟩

theorem sequential_minimum_has_exact_bool_decider
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    HasExactBoolDecider (SequentialMinimumSufficiencyCheck P k) := by
  exact ⟨sequentialMinimumSufficiencyBool P k, sequentialMinimumSufficiencyBool_spec P k⟩

/-- Explicit exhaustive-search decider for the sequential minimum query over
    all coordinate subsets. -/
noncomputable def countedSequentialMinimumSearch
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) : Counted Bool :=
  countedAnyList (Finset.powerset (Finset.univ : Finset (Fin n))).toList
    (fun I => decide (I.card ≤ k) && sequentialSufficientBool P I)

theorem countedSequentialMinimumSearch_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    (countedSequentialMinimumSearch P k).result = true ↔ SequentialMinimumSufficiencyCheck P k := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨I, hI, hPred⟩
    have hPred' : decide (I.card ≤ k) = true ∧ sequentialSufficientBool P I = true := by
      simpa using hPred
    exact ⟨I, by
      constructor
      · simpa using hPred'.1
      · exact (sequentialSufficientBool_spec P I).mp hPred'.2⟩
  · intro h
    rcases h with ⟨I, hCard, hSuff⟩
    have hmemPow : I ∈ Finset.powerset (Finset.univ : Finset (Fin n)) := by
      apply Finset.mem_powerset.mpr
      exact Finset.subset_univ I
    have hmemList : I ∈ (Finset.powerset (Finset.univ : Finset (Fin n))).toList := by
      simpa using hmemPow
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨I, hmemList, ?_⟩
    simp [hCard, sequentialSufficientBool_spec P I, hSuff]

theorem countedSequentialMinimumSearch_steps
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    (countedSequentialMinimumSearch P k).steps ≤ 2 ^ n := by
  have hlen : (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length = 2 ^ n := by
    simp
  calc
    (countedSequentialMinimumSearch P k).steps
        ≤ (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length :=
          countedAnyList_steps_le_length _ _
    _ = 2 ^ n := hlen

/-- Explicit counted violation scan for sequential sufficiency on a fixed
    information set. It searches for a pair of agreeing states with different
    optimal sets. -/
noncomputable def countedSequentialSufficiencySearch
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Counted Bool :=
  let c := countedAnyList ((Finset.univ.product (Finset.univ : Finset S)).toList)
    (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))
  (c.steps, !c.result)

theorem countedSequentialSufficiencySearch_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    (countedSequentialSufficiencySearch P I).result = true ↔ SequentialSufficient P I := by
  constructor
  · intro h s s' hagree
    have hInnerFalse : (countedAnyList
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))).result = false := by
      unfold countedSequentialSufficiencySearch at h
      simpa [Counted.result] using h
    have hfalse := (countedAnyList_result_false_iff
      ((Finset.univ.product (Finset.univ : Finset S)).toList)
      (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))).mp hInnerFalse
    have hmem : (s, s') ∈ ((Finset.univ.product (Finset.univ : Finset S)).toList) := by
      simpa using (Finset.mem_toList.mpr (Finset.mem_product.mpr ⟨Finset.mem_univ s, Finset.mem_univ s'⟩))
    have hsFalse : decide (agreeOn s s' I ∧ P.toDecisionProblem.Opt s ≠ P.toDecisionProblem.Opt s') = false :=
      hfalse (s, s') hmem
    by_cases hneq : P.toDecisionProblem.Opt s ≠ P.toDecisionProblem.Opt s'
    · have hsTrue : decide (agreeOn s s' I ∧ P.toDecisionProblem.Opt s ≠ P.toDecisionProblem.Opt s') = true := by
        simp [hagree, hneq]
      rw [hsTrue] at hsFalse
      cases hsFalse
    · exact by_contra fun hEq => hneq hEq
  · intro h
    have hInnerFalse : (countedAnyList
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))).result = false :=
      (countedAnyList_result_false_iff
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))).mpr (by
          intro ss hss
          by_cases hbad : agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2
          · exact False.elim (hbad.2 (h ss.1 ss.2 hbad.1))
          · simp [hbad])
    unfold countedSequentialSufficiencySearch
    simpa [Counted.result] using hInnerFalse

theorem countedSequentialSufficiencySearch_steps
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    (countedSequentialSufficiencySearch P I).steps ≤ Fintype.card S * Fintype.card S := by
  unfold countedSequentialSufficiencySearch
  simpa [Counted.steps] using countedAnyList_steps_le_length
    ((Finset.univ.product (Finset.univ : Finset S)).toList)
    (fun ss => decide (agreeOn ss.1 ss.2 I ∧ P.toDecisionProblem.Opt ss.1 ≠ P.toDecisionProblem.Opt ss.2))

/-- Explicit counted witness search for sequential anchor sufficiency. -/
noncomputable def countedSequentialAnchorSearch
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Counted Bool :=
  countedAnyList (Finset.univ.toList : List S)
    (fun s0 => decide (∀ s : S, agreeOn s s0 I → P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s0))

theorem countedSequentialAnchorSearch_spec
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    (countedSequentialAnchorSearch P I).result = true ↔ SequentialAnchorSufficiencyCheck P I := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨s0, _, hs0⟩
    exact ⟨s0, by simpa [countedSequentialAnchorSearch] using hs0⟩
  · rintro ⟨s0, hs0⟩
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨s0, ?_, ?_⟩
    · simpa using (Finset.mem_toList.mpr (Finset.mem_univ s0))
    · simpa [countedSequentialAnchorSearch] using hs0

theorem countedSequentialAnchorSearch_steps
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    (countedSequentialAnchorSearch P I).steps ≤ Fintype.card S := by
  calc
    (countedSequentialAnchorSearch P I).steps ≤ (Finset.univ.toList : List S).length :=
      countedAnyList_steps_le_length _ _
    _ = Fintype.card S := by simp

theorem sequentialSufficiency_counted_search_witness
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (SequentialSufficient P I)
      (countedSequentialSufficiencySearch P I)
      (Fintype.card S * Fintype.card S) := by
  constructor
  · exact countedSequentialSufficiencySearch_spec P I
  · exact countedSequentialSufficiencySearch_steps P I

theorem sequentialAnchor_counted_search_witness
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (SequentialAnchorSufficiencyCheck P I)
      (countedSequentialAnchorSearch P I)
      (Fintype.card S) := by
  constructor
  · exact countedSequentialAnchorSearch_spec P I
  · exact countedSequentialAnchorSearch_steps P I

theorem sequentialMinimum_counted_search_witness
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    HasCountedSearchWitness
      (SequentialMinimumSufficiencyCheck P k)
      (countedSequentialMinimumSearch P k)
      (2 ^ n) := by
  constructor
  · exact countedSequentialMinimumSearch_spec P k
  · exact countedSequentialMinimumSearch_steps P k

theorem stochastic_query_search_matrix
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (k : ℕ) :
    HasCountedSearchWitness
      (StochasticSufficient P I)
      (countedStochasticSufficiencySearch P I)
      (Fintype.card S) ∧
    HasCountedSearchWitness
      (StochasticAnchorSufficiencyCheck P I)
      (countedStochasticAnchorSearch P I)
      (Fintype.card S * Fintype.card A) ∧
    HasCountedSearchWitness
      (StochasticMinimumSufficiencyCheck P k)
      (countedStochasticMinimumSearch P k)
      (2 ^ n) := by
  exact ⟨stochasticSufficiency_counted_search_witness P I,
    stochasticAnchor_counted_search_witness P I,
    stochasticMinimum_counted_search_witness P k⟩

theorem sequential_query_search_matrix
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) (k : ℕ) :
    HasCountedSearchWitness
      (SequentialSufficient P I)
      (countedSequentialSufficiencySearch P I)
      (Fintype.card S * Fintype.card S) ∧
    HasCountedSearchWitness
      (SequentialAnchorSufficiencyCheck P I)
      (countedSequentialAnchorSearch P I)
      (Fintype.card S) ∧
    HasCountedSearchWitness
      (SequentialMinimumSufficiencyCheck P k)
      (countedSequentialMinimumSearch P k)
      (2 ^ n) := by
  exact ⟨sequentialSufficiency_counted_search_witness P I,
    sequentialAnchor_counted_search_witness P I,
    sequentialMinimum_counted_search_witness P k⟩

/-! ## Explicit-State P Membership Wrappers -/

/-- Explicit-state input wrapper for stochastic sufficiency/anchor queries. The
    budgets are part of the input size accounting. -/
structure StochasticExplicitInput
    (A S : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] where
  problem : StochasticDecisionProblem A S
  infoSet : Finset (Fin n)
  stateBudget : ℕ
  actionBudget : ℕ
  state_bound : Fintype.card S ≤ stateBudget
  action_bound : Fintype.card A ≤ actionBudget

noncomputable instance instSizeOfStochasticExplicitInput
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    SizeOf (StochasticExplicitInput A S n) where
  sizeOf q := q.stateBudget + q.actionBudget + sizeOf q.infoSet + 1

theorem stochasticExplicit_size_ge_state
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticExplicitInput A S n) :
    q.stateBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf, instSizeOfStochasticExplicitInput]
  omega

theorem stochasticExplicit_size_ge_action
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticExplicitInput A S n) :
    q.actionBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf, instSizeOfStochasticExplicitInput]
  omega

theorem stochastic_sufficiency_inP_explicit
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticExplicitInput A S n => StochasticSufficient q.problem q.infoSet) := by
  use (fun q => countedStochasticSufficiencySearch q.problem q.infoSet), 1, 1
  constructor
  · intro q
    calc
      (countedStochasticSufficiencySearch q.problem q.infoSet).steps ≤ Fintype.card S :=
        countedStochasticSufficiencySearch_steps _ _
      _ ≤ q.stateBudget := q.state_bound
      _ ≤ sizeOf q := stochasticExplicit_size_ge_state q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedStochasticSufficiencySearch_spec _ _

theorem stochastic_preservation_inP_explicit
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticExplicitInput A S n => StochasticPreservationSufficient q.problem q.infoSet) := by
  use (fun q => countedStochasticPreservationSearch q.problem q.infoSet), 1, 1
  constructor
  · intro q
    calc
      (countedStochasticPreservationSearch q.problem q.infoSet).steps ≤ Fintype.card S :=
        countedStochasticPreservationSearch_steps _ _
      _ ≤ q.stateBudget := q.state_bound
      _ ≤ sizeOf q := stochasticExplicit_size_ge_state q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedStochasticPreservationSearch_spec _ _

theorem stochastic_anchor_inP_explicit
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticExplicitInput A S n => StochasticAnchorSufficiencyCheck q.problem q.infoSet) := by
  use (fun q => countedStochasticAnchorSearch q.problem q.infoSet), 1, 2
  constructor
  · intro q
    calc
      (countedStochasticAnchorSearch q.problem q.infoSet).steps ≤ Fintype.card S * Fintype.card A :=
        countedStochasticAnchorSearch_steps _ _
      _ ≤ q.stateBudget * q.actionBudget := Nat.mul_le_mul q.state_bound q.action_bound
      _ ≤ (sizeOf q) * (sizeOf q) := Nat.mul_le_mul (stochasticExplicit_size_ge_state q) (stochasticExplicit_size_ge_action q)
      _ = (sizeOf q) ^ 2 := by simp [pow_two]
      _ ≤ 1 * (sizeOf q) ^ 2 + 1 := by omega
  · intro q
    exact countedStochasticAnchorSearch_spec _ _

/-- Explicit-state input wrapper for sequential sufficiency/anchor queries. -/
structure SequentialExplicitInput
    (A S O : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] where
  problem : SequentialDecisionProblem A S O
  infoSet : Finset (Fin n)
  stateBudget : ℕ
  state_bound : Fintype.card S ≤ stateBudget

noncomputable instance instSizeOfSequentialExplicitInput
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    SizeOf (SequentialExplicitInput A S O n) where
  sizeOf q := q.stateBudget + sizeOf q.infoSet + 1

theorem sequentialExplicit_size_ge_state
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n]
    (q : SequentialExplicitInput A S O n) :
    q.stateBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf, instSizeOfSequentialExplicitInput]
  omega

theorem sequential_sufficiency_inP_explicit
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : SequentialExplicitInput A S O n => SequentialSufficient q.problem q.infoSet) := by
  use (fun q => countedSequentialSufficiencySearch q.problem q.infoSet), 1, 2
  constructor
  · intro q
    calc
      (countedSequentialSufficiencySearch q.problem q.infoSet).steps ≤ Fintype.card S * Fintype.card S :=
        countedSequentialSufficiencySearch_steps _ _
      _ ≤ q.stateBudget * q.stateBudget := Nat.mul_le_mul q.state_bound q.state_bound
      _ ≤ (sizeOf q) * (sizeOf q) := Nat.mul_le_mul (sequentialExplicit_size_ge_state q) (sequentialExplicit_size_ge_state q)
      _ = (sizeOf q) ^ 2 := by simp [pow_two]
      _ ≤ 1 * (sizeOf q) ^ 2 + 1 := by omega
  · intro q
    exact countedSequentialSufficiencySearch_spec _ _

theorem sequential_anchor_inP_explicit
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : SequentialExplicitInput A S O n => SequentialAnchorSufficiencyCheck q.problem q.infoSet) := by
  use (fun q => countedSequentialAnchorSearch q.problem q.infoSet), 1, 1
  constructor
  · intro q
    calc
      (countedSequentialAnchorSearch q.problem q.infoSet).steps ≤ Fintype.card S :=
        countedSequentialAnchorSearch_steps _ _
      _ ≤ q.stateBudget := q.state_bound
      _ ≤ sizeOf q := sequentialExplicit_size_ge_state q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedSequentialAnchorSearch_spec _ _

/-- Paper-facing summary wrapper for the explicit-state sequential sufficiency
upper bound. -/
theorem sequential_sufficiency_upper_bound_summary
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : SequentialExplicitInput A S O n => SequentialSufficient q.problem q.infoSet) :=
  sequential_sufficiency_inP_explicit

/-- Paper-facing summary wrapper for the explicit-state sequential anchor upper
bound. -/
theorem sequential_anchor_upper_bound_summary
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : SequentialExplicitInput A S O n => SequentialAnchorSufficiencyCheck q.problem q.infoSet) :=
  sequential_anchor_inP_explicit

/-- Paper-facing summary wrapper for the explicit counted-search package for
sequential minimum queries. -/
theorem sequential_minimum_upper_bound_summary
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (k : ℕ) :
    HasCountedSearchWitness
      (SequentialMinimumSufficiencyCheck P k)
      (countedSequentialMinimumSearch P k)
      (2 ^ n) :=
  sequentialMinimum_counted_search_witness P k

end DecisionQuotient.StochasticSequential
