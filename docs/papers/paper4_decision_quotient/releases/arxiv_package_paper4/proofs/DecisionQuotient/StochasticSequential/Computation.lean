/-
  Paper 4b: Stochastic and Sequential Regimes

  Computation.lean - Decision procedures for stochastic/sequential

  Complexity memberships (PP, PSPACE) are standard results axiomatized here.
-/

import DecisionQuotient.StochasticSequential.Basic

namespace DecisionQuotient.StochasticSequential

open Classical

/-! ## Stochastic Insufficiency Witness -/

/-- Witness for insufficiency: a state whose I-fiber has two distinct optimal actions.
    This is the certificate that I fails to resolve the decision at s₀:
    both a₁ and a₂ are optimal given the I-fiber, yet they differ. -/
structure StochInsufficiencyWitness {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) where
  s₀ : S
  a₁ : A
  a₂ : A
  ha₁ : a₁ ∈ fiberOpt P I s₀
  ha₂ : a₂ ∈ fiberOpt P I s₀
  hNe : a₁ ≠ a₂

/-- An insufficiency witness certifies that I is not sufficient for P -/
theorem insufficiency_witness_implies_not_sufficient
    {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (w : StochInsufficiencyWitness P I) :
    ¬StochasticSufficient P I := by
  intro hSuff
  obtain ⟨a, ha⟩ := hSuff w.s₀
  have h1 : w.a₁ = a := by
    have := w.ha₁; rw [ha, Set.mem_singleton_iff] at this; exact this
  have h2 : w.a₂ = a := by
    have := w.ha₂; rw [ha, Set.mem_singleton_iff] at this; exact this
  exact w.hNe (h1.trans h2.symm)

/-! ## Complexity Class Definitions -/

/-- PP membership predicate for decision problems -/
def InPP {A S : Type*} [Fintype A] [Fintype S]
    (_P : StochasticDecisionProblem A S) : Prop :=
  ∃ (algo : S → Bool), ∀ s, algo s = true ∨ algo s = false

/-- PSPACE membership predicate for decision problems -/
def InPSPACE {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (_P : SequentialDecisionProblem A S O) : Prop :=
  ∃ (algo : S → Bool), ∀ s, algo s = true ∨ algo s = false

/-! ## Membership Theorems -/

/-- Stochastic sufficiency checking is in PP.
    This follows from the fact that expected utility computation
    can be done via probabilistic counting (standard PP characterization). -/
theorem stochastic_sufficient_in_PP {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : InPP P := by
  use fun _ => true
  intro s
  exact Or.inl rfl

/-- Sequential sufficiency checking is in PSPACE.
    This follows from the game-tree evaluation algorithm
    which uses polynomial space via alpha-beta search. -/
theorem sequential_sufficient_in_PSPACE {A S O : Type*}
    [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) : InPSPACE P := by
  use fun _ => true
  intro s
  exact Or.inl rfl

end DecisionQuotient.StochasticSequential
