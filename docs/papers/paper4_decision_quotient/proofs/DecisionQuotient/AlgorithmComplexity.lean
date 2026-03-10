/-
  Paper 4: Decision-Relevant Uncertainty

  AlgorithmComplexity.lean - Step-Counting and Time Complexity

  This file defines a step-counting computation monad and proves time complexity
  bounds for the sufficiency-checking algorithm.

  Key results:
  - Counted monad for tracking computational steps
  - O(|S|²) bound for checkSufficiency algorithm
  - Polynomial-time characterization

  ## Triviality Level
  TRIVIAL: This is algorithmic analysis - proving polynomial bounds is standard.
  The nontrivial work is in proving the hardness results.

  ## Dependencies
  - Chain: Sufficiency.lean → Computation.lean → here
-/

import DecisionQuotient.Sufficiency
import DecisionQuotient.Hardness.Sigma2PExhaustive.AnchorSufficiency
import Mathlib.Algebra.Polynomial.Eval.Defs

namespace DecisionQuotient

/-! ## Step-Counting Monad

A computation that tracks the number of steps taken. -/

/-- A computation with step counting -/
def Counted (α : Type*) := ℕ × α

/-- Extract the step count from a counted computation -/
def Counted.steps {α : Type*} (c : Counted α) : ℕ := c.1

/-- Extract the result from a counted computation -/
def Counted.result {α : Type*} (c : Counted α) : α := c.2

/-- Pure computation (no steps) -/
def Counted.pure {α : Type*} (a : α) : Counted α := (0, a)

/-- Sequence two counted computations -/
def Counted.bind {α β : Type*} (ca : Counted α) (f : α → Counted β) : Counted β :=
  let (n, a) := ca
  let (m, b) := f a
  (n + m, b)

/-- One step costs 1 -/
def Counted.tick {α : Type*} (a : α) : Counted α := (1, a)

instance : Monad Counted where
  pure := Counted.pure
  bind := Counted.bind

/-- Steps add up properly -/
theorem Counted.bind_steps {α β : Type*} (ca : Counted α) (f : α → Counted β) :
    (Counted.bind ca f).steps = ca.steps + (f ca.result).steps := by
  obtain ⟨n, a⟩ := ca
  obtain ⟨m, b⟩ := f a
  rfl

/-! ## Sufficiency Check with Step Counting

We define a step-counted version of the sufficiency check algorithm. -/

/-- Count one comparison as one step -/
def countedCompare {A : Type*} [DecidableEq (Set A)] (s1 s2 : Set A) : Counted Bool :=
  Counted.tick (decide (s1 = s2))

/-- Check if two states have the same Opt set (with step counting) -/
def countedOptEqual {A S : Type*} [DecidableEq (Set A)]
    (dp : DecisionProblem A S) (s s' : S) : Counted Bool :=
  countedCompare (dp.Opt s) (dp.Opt s')

/-- Check all pairs in a list (with step counting) -/
def countedCheckPairs {A S : Type*} [DecidableEq S] [DecidableEq (Set A)]
    (dp : DecisionProblem A S)
    (equiv : S → S → Prop) [DecidableRel equiv]
    (pairs : List (S × S)) : Counted Bool :=
  match pairs with
  | [] => Counted.pure true
  | (s, s') :: rest =>
    Counted.bind (countedOptEqual dp s s') fun eq1 =>
      if ¬equiv s s' then Counted.pure false  -- not equivalent states
      else if ¬eq1 then Counted.pure false    -- Opt differs
      else countedCheckPairs dp equiv rest

/-- Counted existential search through a finite list for a boolean predicate. -/
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

/-- Helper: steps of bind with tick -/
@[simp]
theorem Counted.tick_bind_steps {α β : Type*} (a : α) (f : α → Counted β) :
    (Counted.bind (Counted.tick a) f).steps = 1 + (f a).steps := by
  unfold Counted.bind Counted.tick Counted.steps
  split
  next m b heq =>
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl⟩ := heq
    split
    next m' b' heq' =>
      simp only [Prod.fst, heq']

/-- Steps of pure is 0 -/
@[simp]
theorem Counted.pure_steps {α : Type*} (a : α) : (Counted.pure a).steps = 0 := rfl

/-- Steps equals fst -/
@[simp]
theorem Counted.steps_eq_fst {α : Type*} (c : Counted α) : c.steps = c.1 := rfl

/-- Number of steps for checking pairs is bounded by list length -/
theorem countedCheckPairs_steps_bound {A S : Type*} [DecidableEq S] [DecidableEq (Set A)]
    (dp : DecisionProblem A S)
    (equiv : S → S → Prop) [DecidableRel equiv]
    (pairs : List (S × S)) :
    (countedCheckPairs dp equiv pairs).steps ≤ pairs.length := by
  induction pairs with
  | nil =>
    simp only [countedCheckPairs, Counted.pure, Counted.steps, List.length_nil, le_refl]
  | cons p rest ih =>
    obtain ⟨s, s'⟩ := p
    simp only [countedCheckPairs, countedOptEqual, countedCompare, List.length_cons]
    rw [Counted.tick_bind_steps]
    split_ifs
    all_goals
      simp only [Counted.pure, Counted.steps, Prod.fst]
      first
      | omega
      | (have hconv : (countedCheckPairs dp equiv rest).1 =
            (countedCheckPairs dp equiv rest).steps := rfl
         rw [hconv]; omega)

/-! ## Polynomial Time Complexity

Definition of polynomial-time computability for our algorithms. -/

/-- A counted function is polynomial-time if steps ≤ polynomial in input size -/
def IsPolynomialTime {α β : Type*} [SizeOf α] (f : α → Counted β) : Prop :=
  ∃ (c k : ℕ), ∀ a : α, (f a).steps ≤ c * (sizeOf a) ^ k + c

/-- Constant-time computations are polynomial -/
theorem IsPolynomialTime.const {α β : Type*} [SizeOf α] (b : β) :
    IsPolynomialTime (fun _ : α => Counted.pure b) := by
  use 0, 0
  intro _
  simp [Counted.pure, Counted.steps]

/-- A function taking at most n steps is polynomial-time -/
theorem IsPolynomialTime.of_steps_le {α β : Type*} [SizeOf α]
    (f : α → Counted β) (n : ℕ) (hbound : ∀ a, (f a).steps ≤ n) :
    IsPolynomialTime f := by
  use n, 0
  intro a
  have := hbound a
  simp only [pow_zero, mul_one]
  omega

/-! ## Main Complexity Result

The sufficiency-checking algorithm runs in polynomial time. -/

/-- For finite state spaces, checking all pairs takes O(|S|²) comparisons -/
theorem sufficiency_check_polynomial {A S : Type*} [Fintype S] [DecidableEq S] [DecidableEq (Set A)]
    (dp : DecisionProblem A S)
    (equiv : S → S → Prop) [DecidableRel equiv] :
    ∃ (c : ℕ), c = Fintype.card S * Fintype.card S ∧
    ∀ (pairs : List (S × S)), pairs.length ≤ c →
    (countedCheckPairs dp equiv pairs).steps ≤ c := by
  use Fintype.card S * Fintype.card S
  constructor
  · rfl
  · intro pairs hlen
    calc (countedCheckPairs dp equiv pairs).steps
        ≤ pairs.length := countedCheckPairs_steps_bound dp equiv pairs
      _ ≤ Fintype.card S * Fintype.card S := hlen

/-! ## Explicit Finite Searches for Anchor/Minimum Queries -/

variable {n : ℕ}

/-- Exact boolean decider for static minimum sufficiency. -/
noncomputable def minimumSufficientBool {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) : Bool := by
  classical
  exact decide (∃ I : Finset (Fin n), I.card ≤ k ∧ dp.isSufficient I)

theorem minimumSufficientBool_spec {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) :
    minimumSufficientBool (n := n) dp k = true ↔
      ∃ I : Finset (Fin n), I.card ≤ k ∧ dp.isSufficient I := by
  classical
  simp [minimumSufficientBool]

/-- Exact boolean decider for static anchor sufficiency. -/
noncomputable def anchorSufficientBool {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) : Bool := by
  classical
  exact decide (dp.anchorSufficient I)

theorem anchorSufficientBool_spec {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    anchorSufficientBool (n := n) dp I = true ↔ dp.anchorSufficient I := by
  classical
  simp [anchorSufficientBool]

/-- Explicit exhaustive-search decider for static minimum sufficiency. -/
noncomputable def countedMinimumSufficientSearch {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) : Counted Bool :=
  countedAnyList (Finset.powerset (Finset.univ : Finset (Fin n))).toList
    (fun I => by
      classical
      exact decide (I.card ≤ k ∧ dp.isSufficient I))

theorem countedMinimumSufficientSearch_spec {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) :
    (countedMinimumSufficientSearch (n := n) dp k).result = true ↔
      ∃ I : Finset (Fin n), I.card ≤ k ∧ dp.isSufficient I := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨I, _, hPred⟩
    exact ⟨I, by simpa using hPred⟩
  · intro h
    rcases h with ⟨I, hCard, hSuff⟩
    have hmemPow : I ∈ Finset.powerset (Finset.univ : Finset (Fin n)) := by
      apply Finset.mem_powerset.mpr
      exact Finset.subset_univ I
    have hmemList : I ∈ (Finset.powerset (Finset.univ : Finset (Fin n))).toList := by
      simpa using hmemPow
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨I, hmemList, ?_⟩
    simpa using (show I.card ≤ k ∧ dp.isSufficient I from ⟨hCard, hSuff⟩)

theorem countedMinimumSufficientSearch_steps {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) :
    (countedMinimumSufficientSearch (n := n) dp k).steps ≤ 2 ^ n := by
  have hlen : (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length = 2 ^ n := by
    simp
  calc
    (countedMinimumSufficientSearch (n := n) dp k).steps
        ≤ (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length :=
          countedAnyList_steps_le_length _ _
    _ = 2 ^ n := hlen

/-- Explicit exhaustive-search decider for static anchor sufficiency. -/
noncomputable def countedAnchorSufficientSearch {A S : Type*} [Fintype S] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  countedAnyList (Finset.univ.toList : List S)
    (fun s0 => by
      classical
      exact decide (dp.isSufficientAt I s0))

theorem countedAnchorSufficientSearch_spec {A S : Type*} [Fintype S] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (countedAnchorSufficientSearch (n := n) dp I).result = true ↔ dp.anchorSufficient I := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨s0, _, hs0⟩
    exact ⟨s0, by simpa [DecisionProblem.isSufficientAt] using hs0⟩
  · rintro ⟨s0, hs0⟩
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨s0, ?_, ?_⟩
    · simpa using (Finset.mem_toList.mpr (Finset.mem_univ s0))
    · simpa [DecisionProblem.isSufficientAt] using hs0

theorem countedAnchorSufficientSearch_steps {A S : Type*} [Fintype S] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (countedAnchorSufficientSearch (n := n) dp I).steps ≤ Fintype.card S := by
  calc
    (countedAnchorSufficientSearch (n := n) dp I).steps ≤ (Finset.univ.toList : List S).length :=
      countedAnyList_steps_le_length _ _
    _ = Fintype.card S := by simp

theorem staticMinimum_counted_search_witness {A S : Type*} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (k : ℕ) :
    ((countedMinimumSufficientSearch (n := n) dp k).result = true ↔
      ∃ I : Finset (Fin n), I.card ≤ k ∧ dp.isSufficient I) ∧
    (countedMinimumSufficientSearch (n := n) dp k).steps ≤ 2 ^ n := by
  exact ⟨countedMinimumSufficientSearch_spec (n := n) dp k,
    countedMinimumSufficientSearch_steps (n := n) dp k⟩

theorem staticAnchor_counted_search_witness {A S : Type*} [Fintype S] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    ((countedAnchorSufficientSearch (n := n) dp I).result = true ↔
      dp.anchorSufficient I) ∧
    (countedAnchorSufficientSearch (n := n) dp I).steps ≤ Fintype.card S := by
  exact ⟨countedAnchorSufficientSearch_spec (n := n) dp I,
    countedAnchorSufficientSearch_steps (n := n) dp I⟩

theorem static_query_search_matrix_legacy {A S : Type*} [Fintype S] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) (k : ℕ) :
    ((countedAnchorSufficientSearch (n := n) dp I).result = true ↔ dp.anchorSufficient I) ∧
    (countedAnchorSufficientSearch (n := n) dp I).steps ≤ Fintype.card S ∧
    ((countedMinimumSufficientSearch (n := n) dp k).result = true ↔
      ∃ J : Finset (Fin n), J.card ≤ k ∧ dp.isSufficient J) ∧
    (countedMinimumSufficientSearch (n := n) dp k).steps ≤ 2 ^ n := by
  exact ⟨countedAnchorSufficientSearch_spec (n := n) dp I,
    countedAnchorSufficientSearch_steps (n := n) dp I,
    countedMinimumSufficientSearch_spec (n := n) dp k,
    countedMinimumSufficientSearch_steps (n := n) dp k⟩

/-- Explicit counted violation scan for static sufficiency on a fixed
    information set. It searches for an agreeing pair with different optimal
    sets. -/
noncomputable def countedStaticSufficiencySearch {A S : Type*}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  let c := countedAnyList ((Finset.univ.product (Finset.univ : Finset S)).toList)
    (fun ss => by
      classical
      exact decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))
  (c.steps, !c.result)

theorem countedStaticSufficiencySearch_spec {A S : Type*}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (countedStaticSufficiencySearch (n := n) dp I).result = true ↔ dp.isSufficient I := by
  classical
  constructor
  · intro h s s' hagree
    have hInnerFalse : (countedAnyList
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => by
          classical
          exact decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))).result = false := by
      unfold countedStaticSufficiencySearch at h
      simpa [Counted.result] using h
    have hfalse := (countedAnyList_result_false_iff
      ((Finset.univ.product (Finset.univ : Finset S)).toList)
      (fun ss => by
        classical
        exact decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))).mp hInnerFalse
    have hmem : (s, s') ∈ ((Finset.univ.product (Finset.univ : Finset S)).toList) := by
      simpa using (Finset.mem_toList.mpr (Finset.mem_product.mpr ⟨Finset.mem_univ s, Finset.mem_univ s'⟩))
    have hsFalse : decide (agreeOn s s' I ∧ dp.Opt s ≠ dp.Opt s') = false := hfalse (s, s') hmem
    by_cases hneq : dp.Opt s ≠ dp.Opt s'
    · have hsTrue : decide (agreeOn s s' I ∧ dp.Opt s ≠ dp.Opt s') = true := by
        simp [hagree, hneq]
      rw [hsTrue] at hsFalse
      cases hsFalse
    · exact not_ne_iff.mp hneq
  · intro h
    have hInnerFalse : (countedAnyList
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))).result = false :=
      (countedAnyList_result_false_iff
        ((Finset.univ.product (Finset.univ : Finset S)).toList)
        (fun ss => by
          classical
          exact decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))).mpr (by
          intro ss hss
          by_cases hbad : agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2
          · exact False.elim (hbad.2 (h ss.1 ss.2 hbad.1))
          · have : decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2) = false := by
              simp [hbad]
            exact this)
    unfold countedStaticSufficiencySearch
    simpa [Counted.result] using hInnerFalse

theorem countedStaticSufficiencySearch_steps {A S : Type*}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (countedStaticSufficiencySearch (n := n) dp I).steps ≤ Fintype.card S * Fintype.card S := by
  unfold countedStaticSufficiencySearch
  simpa [Counted.steps] using countedAnyList_steps_le_length
    ((Finset.univ.product (Finset.univ : Finset S)).toList)
    (fun ss => by
      classical
      exact decide (agreeOn ss.1 ss.2 I ∧ dp.Opt ss.1 ≠ dp.Opt ss.2))

theorem staticSufficiency_counted_search_witness {A S : Type*}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    ((countedStaticSufficiencySearch (n := n) dp I).result = true ↔ dp.isSufficient I) ∧
    (countedStaticSufficiencySearch (n := n) dp I).steps ≤ Fintype.card S * Fintype.card S := by
  exact ⟨countedStaticSufficiencySearch_spec (n := n) dp I,
    countedStaticSufficiencySearch_steps (n := n) dp I⟩

theorem static_query_search_matrix {A S : Type*}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) (k : ℕ) :
    ((countedStaticSufficiencySearch (n := n) dp I).result = true ↔ dp.isSufficient I) ∧
    (countedStaticSufficiencySearch (n := n) dp I).steps ≤ Fintype.card S * Fintype.card S ∧
    ((countedAnchorSufficientSearch (n := n) dp I).result = true ↔ dp.anchorSufficient I) ∧
    (countedAnchorSufficientSearch (n := n) dp I).steps ≤ Fintype.card S ∧
    ((countedMinimumSufficientSearch (n := n) dp k).result = true ↔
      ∃ J : Finset (Fin n), J.card ≤ k ∧ dp.isSufficient J) ∧
    (countedMinimumSufficientSearch (n := n) dp k).steps ≤ 2 ^ n := by
  exact ⟨countedStaticSufficiencySearch_spec (n := n) dp I,
    countedStaticSufficiencySearch_steps (n := n) dp I,
    countedAnchorSufficientSearch_spec (n := n) dp I,
    countedAnchorSufficientSearch_steps (n := n) dp I,
    countedMinimumSufficientSearch_spec (n := n) dp k,
    countedMinimumSufficientSearch_steps (n := n) dp k⟩

end DecisionQuotient
