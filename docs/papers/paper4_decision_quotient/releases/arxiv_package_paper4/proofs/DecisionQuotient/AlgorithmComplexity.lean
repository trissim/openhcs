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

end DecisionQuotient

