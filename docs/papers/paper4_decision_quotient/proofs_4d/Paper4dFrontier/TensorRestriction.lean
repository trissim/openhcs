import DecisionQuotient.Tractability.SeparableUtility
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

def extendFirstTwoWithZero {k m : ℕ} [NeZero k] (s : Fin 2 → Fin k) : Fin (2 + m) → Fin k :=
  Fin.append s (fun _ => 0)

def extendFirstTwoWithTail {k m : ℕ} (tail : Fin m → Fin k) (s : Fin 2 → Fin k) : Fin (2 + m) → Fin k :=
  Fin.append s tail

def restrictFirstTwo {A : Type*} {k m : ℕ} [NeZero k]
    (u : A → (Fin (2 + m) → Fin k) → ℤ) : A → (Fin 2 → Fin k) → ℤ :=
  fun a s => u a (extendFirstTwoWithZero s)

def restrictFirstTwoWithTail {A : Type*} {k m : ℕ}
    (u : A → (Fin (2 + m) → Fin k) → ℤ) (tail : Fin m → Fin k) : A → (Fin 2 → Fin k) → ℤ :=
  fun a s => u a (extendFirstTwoWithTail tail s)

def sliceAction {A : Type*} {n : ℕ} {Coord : Fin n → Type*}
    (u : A → ((i : Fin n) → Coord i) → ℤ) (a0 : A) : Unit → ((i : Fin n) → Coord i) → ℤ :=
  fun _ s => u a0 s

noncomputable def sliceActionTensorRank {A : Type*} {n R : ℕ} {Coord : Fin n → Type*}
    [∀ i, Fintype (Coord i)] {u : A → ((i : Fin n) → Coord i) → ℤ} (a0 : A)
    (decomp : TensorRankDecomposition u R) :
    TensorRankDecomposition (sliceAction u a0) R where
  weight := fun r => decomp.weight r * decomp.actionFactor r a0
  actionFactor := fun _ _ => 1
  coordFactor := decomp.coordFactor
  decomp := by
    intro _ s
    unfold sliceAction
    rw [decomp.decomp]
    refine Finset.sum_congr rfl ?_
    intro r hr
    ring_nf

noncomputable def restrictTensorRankFirstTwo {A : Type*} {k m R : ℕ} [NeZero k]
    {u : A → (Fin (2 + m) → Fin k) → ℤ}
    (decomp : TensorRankDecomposition u R) :
    TensorRankDecomposition (restrictFirstTwo u) R where
  weight := fun r => decomp.weight r * ∏ j : Fin m, decomp.coordFactor r (Fin.natAdd 2 j) 0
  actionFactor := decomp.actionFactor
  coordFactor := fun r i => decomp.coordFactor r (Fin.castAdd m i)
  decomp := by
    intro a s
    unfold restrictFirstTwo extendFirstTwoWithZero
    rw [decomp.decomp]
    refine Finset.sum_congr rfl ?_
    intro r hr
    rw [Fin.prod_univ_add]
    simp [Fin.append]
    ring_nf

noncomputable def restrictTensorRankFirstTwoWithTail {A : Type*} {k m R : ℕ}
    {u : A → (Fin (2 + m) → Fin k) → ℤ} (tail : Fin m → Fin k)
    (decomp : TensorRankDecomposition u R) :
    TensorRankDecomposition (restrictFirstTwoWithTail u tail) R where
  weight := fun r => decomp.weight r * ∏ j : Fin m, decomp.coordFactor r (Fin.natAdd 2 j) (tail j)
  actionFactor := decomp.actionFactor
  coordFactor := fun r i => decomp.coordFactor r (Fin.castAdd m i)
  decomp := by
    intro a s
    unfold restrictFirstTwoWithTail extendFirstTwoWithTail
    rw [decomp.decomp]
    refine Finset.sum_congr rfl ?_
    intro r hr
    rw [Fin.prod_univ_add]
    simp [Fin.append]
    ring_nf

end Paper4dFrontier
