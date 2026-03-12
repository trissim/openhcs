import DecisionQuotient.StochasticSequential.Basic

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient
open Classical

theorem agreeOn_anchor_iff
    {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (I : Finset (Fin n)) {s s' x : S}
    (hss' : agreeOn s s' I) :
    agreeOn x s I ↔ agreeOn x s' I := by
  constructor
  · intro hxs i hi
    exact (hxs i hi).trans (hss' i hi)
  · intro hxs i hi
    exact (hxs i hi).trans ((hss' i hi).symm)

theorem fiberExpectedUtility_eq_of_agreeOn
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S)
    (I : Finset (Fin n)) {s s' : S}
    (hss' : agreeOn s s' I) (a : A) :
    fiberExpectedUtility P I s a = fiberExpectedUtility P I s' a := by
  unfold fiberExpectedUtility
  apply Finset.sum_congr rfl
  intro x _
  by_cases hxs : agreeOn x s I
  · have hxs' : agreeOn x s' I := (agreeOn_anchor_iff (I := I) hss').1 hxs
    simp [hxs, hxs']
  · have hxs' : ¬ agreeOn x s' I := by
      intro hcontra
      exact hxs ((agreeOn_anchor_iff (I := I) hss').2 hcontra)
    simp [hxs, hxs']

theorem fiberOpt_eq_of_agreeOn
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S)
    (I : Finset (Fin n)) {s s' : S}
    (hss' : agreeOn s s' I) :
    fiberOpt P I s = fiberOpt P I s' := by
  ext a
  constructor
  · intro ha
    intro a'
    calc
      fiberExpectedUtility P I s' a'
          = fiberExpectedUtility P I s a' := by
              symm
              exact fiberExpectedUtility_eq_of_agreeOn (P := P) (I := I) hss' a'
      _ ≤ fiberExpectedUtility P I s a := ha a'
      _ = fiberExpectedUtility P I s' a :=
            fiberExpectedUtility_eq_of_agreeOn (P := P) (I := I) hss' a
  · intro ha
    intro a'
    calc
      fiberExpectedUtility P I s a'
          = fiberExpectedUtility P I s' a' :=
            fiberExpectedUtility_eq_of_agreeOn (P := P) (I := I) hss' a'
      _ ≤ fiberExpectedUtility P I s' a := ha a'
      _ = fiberExpectedUtility P I s a := by
            symm
            exact fiberExpectedUtility_eq_of_agreeOn (P := P) (I := I) hss' a

/-- Set-valued stochastic sufficiency: observed-coordinate agreement implies equal
conditional optimal-action sets (ties allowed). -/
def StochasticSetSufficient
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∀ s s' : S, agreeOn s s' I → fiberOpt P I s = fiberOpt P I s'

theorem stochasticSetSufficient_universal
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    StochasticSetSufficient P I := by
  intro s s' hss'
  exact fiberOpt_eq_of_agreeOn (P := P) (I := I) hss'

theorem stochasticSufficient_implies_setSufficient
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (_h : StochasticSufficient P I) :
    StochasticSetSufficient P I :=
  stochasticSetSufficient_universal (P := P) (I := I)

end DecisionQuotient.StochasticSequential
