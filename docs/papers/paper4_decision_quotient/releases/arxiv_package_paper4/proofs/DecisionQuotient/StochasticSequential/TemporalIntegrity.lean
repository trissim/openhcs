/-
  Paper 4b: Stochastic and Sequential Regimes

  TemporalIntegrity.lean - Formalization of temporal integrity constraints

  Extends paper 4's integrity to temporal sequence of claims.
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.List.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Claims

A claim is a statement about sufficiency of a coordinate set.
Claims can be refined (strengthened) or retracted (weakened).
Temporal integrity ensures claims evolve consistently.
-/

/-- A claim about sufficiency -/
inductive Claim (α : Type*) where
  | sufficient (I : Finset α)
  | insufficient (I : Finset α)

/-- Evidence supporting a claim -/
structure Evidence where
  states : List String
  proof : Bool

/-- A claim with evidence -/
structure ClaimWithEvidence (α : Type*) where
  claim : Claim α
  evidence : Evidence

/-! ## Claim Sequence -/

/-- Sequence of claims over time -/
def ClaimSequence (α : Type*) := List (ClaimWithEvidence α)

/-! ## Temporal Integrity -/

/-- Claims are monotonically refined (evidence only increases) -/
def claimsMonotone {α : Type*} (seq : ClaimSequence α) : Prop :=
  List.Chain' (fun c₁ c₂ => c₁.evidence.states.length ≤ c₂.evidence.states.length) seq

/-- Temporal integrity: claims monotonically refined -/
def temporalIntegrity {α : Type*} (seq : ClaimSequence α) : Prop :=
  claimsMonotone seq

/-! ## Main Theorems -/

/-- Empty sequence has integrity -/
theorem empty_sequence_integrity {α : Type*} :
    temporalIntegrity ([] : ClaimSequence α) := by
  unfold temporalIntegrity claimsMonotone
  simp [List.Chain']

/-- Singleton sequence has integrity -/
theorem singleton_integrity {α : Type*} (c : ClaimWithEvidence α) :
    temporalIntegrity [c] := by
  unfold temporalIntegrity claimsMonotone
  simp [List.Chain']

/-- Length of sequence is finite -/
theorem sequence_length_finite {α : Type*} (seq : ClaimSequence α) :
    seq.length < seq.length + 1 := Nat.lt_succ_self _

end DecisionQuotient.StochasticSequential
