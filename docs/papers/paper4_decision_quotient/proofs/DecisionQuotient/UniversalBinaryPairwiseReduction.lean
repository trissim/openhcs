/-
  Paper 4: Decision-Relevant Uncertainty

  UniversalBinaryPairwiseReduction.lean

  This file records the precise limitation of the binary-pairwise frontier:
  an exact utility-preserving reduction from arbitrary binary-coordinate
  utilities to pairwise form on the same coordinates is false.

  The universal semantics claims in the paper are handled elsewhere by
  `Paper4dFrontier.Realizability`; this file isolates the stronger claim that
  every binary utility itself is pairwise. The counterexample is the cubic
  all-ones indicator on three coordinates.
-/

import DecisionQuotient.Tractability.TreeStructure
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

namespace DecisionQuotient

open scoped BigOperators

/-- The eight binary states of arity three, written by coordinates. -/
def cubeState (x y z : Fin 2) : Fin 3 → Fin 2
  | 0 => x
  | 1 => y
  | _ => z

/-- The third mixed difference on the three binary coordinates. Pairwise utilities
have vanishing third mixed difference. -/
def tripleCrossDifference {A : Type*}
    (u : A → (Fin 3 → Fin 2) → ℤ) (a : A) : ℤ :=
  u a (cubeState 1 1 1) - u a (cubeState 1 1 0) -
    u a (cubeState 1 0 1) - u a (cubeState 0 1 1) +
    u a (cubeState 1 0 0) + u a (cubeState 0 1 0) +
    u a (cubeState 0 0 1) - u a (cubeState 0 0 0)

/-- A pairwise utility has zero third mixed difference. -/
theorem tripleCrossDifference_eq_zero_of_pairwise
    {A : Type*} {u : A → (Fin 3 → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (a : A) :
    tripleCrossDifference u a = 0 := by
  unfold tripleCrossDifference
  rw [pw.decomp a (cubeState 1 1 1)]
  rw [pw.decomp a (cubeState 1 1 0)]
  rw [pw.decomp a (cubeState 1 0 1)]
  rw [pw.decomp a (cubeState 0 1 1)]
  rw [pw.decomp a (cubeState 1 0 0)]
  rw [pw.decomp a (cubeState 0 1 0)]
  rw [pw.decomp a (cubeState 0 0 1)]
  rw [pw.decomp a (cubeState 0 0 0)]
  simp_rw [Fin.sum_univ_three]
  simp [cubeState]
  ring

/-- The cubic all-ones indicator. This is the standard non-pairwise witness. -/
def tripleIndicator : Unit → (Fin 3 → Fin 2) → ℤ :=
  fun _ s => if s 0 = 1 ∧ s 1 = 1 ∧ s 2 = 1 then 1 else 0

theorem tripleIndicator_tripleCrossDifference :
    tripleCrossDifference tripleIndicator () = 1 := by
  unfold tripleCrossDifference tripleIndicator
  simp [cubeState]

/-- The cubic all-ones indicator does not admit a pairwise presentation on the
same three binary coordinates. -/
theorem tripleIndicator_not_pairwise :
    ¬ Nonempty (PairwiseUtility tripleIndicator) := by
  intro h
  rcases h with ⟨pw⟩
  have hzero : tripleCrossDifference tripleIndicator () = 0 :=
    tripleCrossDifference_eq_zero_of_pairwise pw ()
  have hone : tripleCrossDifference tripleIndicator () = 1 :=
    tripleIndicator_tripleCrossDifference
  omega

/-- There exists a binary-coordinate utility that is not pairwise on the same
coordinate set. -/
theorem exists_binary_utility_not_pairwise :
    ∃ u : Unit → (Fin 3 → Fin 2) → ℤ, ¬ Nonempty (PairwiseUtility u) := by
  exact ⟨tripleIndicator, tripleIndicator_not_pairwise⟩

/-- Consequently, there is no universal exact utility-preserving reduction from
arbitrary binary-coordinate utilities to pairwise form on the same coordinates. -/
theorem not_universal_same_coordinate_pairwise_presentation :
    ¬ ∀ u : Unit → (Fin 3 → Fin 2) → ℤ, Nonempty (PairwiseUtility u) := by
  intro h
  exact tripleIndicator_not_pairwise (h tripleIndicator)

end DecisionQuotient
