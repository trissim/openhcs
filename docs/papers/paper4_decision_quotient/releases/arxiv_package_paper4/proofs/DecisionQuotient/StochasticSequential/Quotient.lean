/-
  Paper 4b: Stochastic and Sequential Regimes

  Quotient.lean - Decision quotient for stochastic/sequential problems

  The decision quotient identifies states that are indistinguishable for decision purposes:
  two states are equivalent when they yield the same locally optimal action set
  under the fiber defined by coordinate set I.

  The quotient collapses exactly the decision-irrelevant distinctions:
  - I-equivalent states require the same action (for any optimal strategy)
  - The quotient type is S modulo this relation
  - I is sufficient iff the quotient is "flat": every equivalence class has a singleton opt

  Key theorems:
  - stochasticDecisionEquiv is an equivalence relation
  - Sufficient I induces a flat quotient (each class has a unique optimal action)
  - Full coordinate set (univ) induces the finest quotient
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.Setoid.Basic

namespace DecisionQuotient.StochasticSequential

open Classical

/-! ## Stochastic Decision Equivalence

Two states are I-equivalent when they lie in the same fiber of the coordinate
projection and thus have the same locally optimal action set.
-/

/-- Two states are I-equivalent iff observing I cannot distinguish them
    for decision purposes: they have the same fiber-optimal action set. -/
def stochasticDecisionEquiv {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s s' : S) : Prop :=
  fiberOpt P I s = fiberOpt P I s'

/-- I-equivalence is reflexive -/
theorem stochasticDecisionEquiv_refl {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s : S) :
    stochasticDecisionEquiv P I s s := rfl

/-- I-equivalence is symmetric -/
theorem stochasticDecisionEquiv_symm {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s s' : S) :
    stochasticDecisionEquiv P I s s' → stochasticDecisionEquiv P I s' s :=
  Eq.symm

/-- I-equivalence is transitive -/
theorem stochasticDecisionEquiv_trans {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s s' s'' : S) :
    stochasticDecisionEquiv P I s s' →
    stochasticDecisionEquiv P I s' s'' →
    stochasticDecisionEquiv P I s s'' :=
  Eq.trans

/-- I-equivalence as a Setoid -/
def stochasticEquivSetoid {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Setoid S where
  r := stochasticDecisionEquiv P I
  iseqv := ⟨
    stochasticDecisionEquiv_refl P I,
    stochasticDecisionEquiv_symm P I _ _,
    stochasticDecisionEquiv_trans P I _ _ _
  ⟩

/-! ## Flatness and Sufficiency

The quotient is "flat" (every equivalence class has a unique optimal action)
exactly when I is sufficient.
-/

/-- The quotient is flat: each equivalence class has a unique optimal action.
    This is exactly the condition that I is sufficient. -/
def flatQuotient {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  StochasticSufficient P I

/-- Flat quotient characterization: each state's fiber-optimal set is a singleton -/
theorem flatQuotient_iff {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    flatQuotient P I ↔ ∀ s : S, ∃ a : A, fiberOpt P I s = {a} := by
  unfold flatQuotient StochasticSufficient
  rfl

/-! ## Quotient Refinement

A finer observation set (I ⊆ J) yields a finer equivalence relation:
every J-class is contained in some I-class.

Key insight: agreeOn is antimonotone in the coordinate set. If two states agree
on J-coordinates and I ⊆ J, they automatically agree on I-coordinates.
-/

/-- agreeOn is antimonotone in the coordinate set: I ⊆ J → agreeOn s s' J → agreeOn s s' I.
    Observing more coordinates is a stronger condition. -/
lemma agreeOn_mono {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (I J : Finset (Fin n)) (hIJ : I ⊆ J) (s s' : S) :
    agreeOn s s' J → agreeOn s s' I := by
  intro hJ i hi
  exact hJ i (hIJ hi)

/-- If s and s' agree on I, then agreeOn t s I ↔ agreeOn t s' I for any t -/
lemma agreeOn_iff_of_agreeOn {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (s s' : S) (I : Finset (Fin n)) (hagree : agreeOn s s' I) (t : S) :
    agreeOn t s I ↔ agreeOn t s' I := by
  unfold agreeOn
  constructor
  · intro h i hi
    rw [h i hi, hagree i hi]
  · intro h i hi
    rw [h i hi, ← hagree i hi]

/-- States agreeing on I-coordinates have the same I-fiber expected utility.
    The sum over the I-fiber is identical when centered at agreeing states. -/
lemma fiberExpectedUtility_of_agreeOn {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s s' : S)
    (hagree : agreeOn s s' I) (a : A) :
    fiberExpectedUtility P I s a = fiberExpectedUtility P I s' a := by
  unfold fiberExpectedUtility
  congr 1
  ext t
  -- The conditional is the same since agreeOn t s I ↔ agreeOn t s' I
  simp only [agreeOn_iff_of_agreeOn s s' I hagree t]

/-- States agreeing on I-coordinates have the same I-fiber optimal set. -/
theorem fiberOpt_of_agreeOn {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s s' : S)
    (hagree : agreeOn s s' I) :
    fiberOpt P I s = fiberOpt P I s' := by
  unfold fiberOpt
  ext a
  simp only [Set.mem_setOf_eq]
  constructor <;> intro h a'
  · rw [← fiberExpectedUtility_of_agreeOn P I s s' hagree a,
        ← fiberExpectedUtility_of_agreeOn P I s s' hagree a']
    exact h a'
  · rw [fiberExpectedUtility_of_agreeOn P I s s' hagree a,
        fiberExpectedUtility_of_agreeOn P I s s' hagree a']
    exact h a'

theorem stochastic_anchor_sufficient_of_stochastic_sufficient
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty S]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    StochasticSufficient P I → StochasticAnchorSufficient P I := by
  intro hSuff
  let s₀ : S := Classical.arbitrary S
  rcases hSuff s₀ with ⟨a, ha⟩
  refine ⟨s₀, a, ha, ?_⟩
  intro s hs
  have hEq : fiberOpt P I s = fiberOpt P I s₀ := fiberOpt_of_agreeOn P I s s₀ hs
  simpa [ha] using hEq

/-- Observing more coordinates (J ⊇ I) and agreeing on those yields I-equivalence.
    This is the correct form of quotient refinement: agreeOn J → stochasticDecisionEquiv I.

    The original claim "J-equiv → I-equiv" is FALSE in general because stochasticDecisionEquiv
    compares fiber-optimal SETS, not coordinate agreement. Two states can have the same
    J-fiber optimal set without being in the same J-fiber. -/
theorem quotient_refines {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I J : Finset (Fin n)) (hIJ : I ⊆ J) :
    ∀ s s' : S, agreeOn s s' J → stochasticDecisionEquiv P I s s' := by
  intro s s' hagreeJ
  unfold stochasticDecisionEquiv
  -- States agreeing on J agree on I (subset monotonicity)
  have hagreeI : agreeOn s s' I := agreeOn_mono I J hIJ s s' hagreeJ
  exact fiberOpt_of_agreeOn P I s s' hagreeI

/-! ## ∅-Quotient: Global Sufficiency -/

/-- The ∅-quotient identifies all states (fiberOpt P ∅ s = stochasticOpt for all s) -/
theorem empty_quotient_all_equiv {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (s s' : S) :
    stochasticDecisionEquiv P ∅ s s' := by
  unfold stochasticDecisionEquiv
  simp [fiberOpt_empty]

/-- ∅-quotient flatness ↔ prior sufficiency -/
theorem empty_flat_iff_prior_sufficient {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A] [Nonempty S]
    (P : StochasticDecisionProblem A S) :
    flatQuotient P (∅ : Finset (Fin n)) ↔ ∃ a : A, P.stochasticOpt = {a} :=
  stochasticSufficient_empty_iff P

end DecisionQuotient.StochasticSequential
