import Paper4dFrontier.TensorRestriction
import Paper4dFrontier.RealTreewidthWitnesses
import DecisionQuotient.Finite

namespace Paper4dFrontier

open DecisionQuotient
open Classical

def unaryZeroIndicator (x : Fin 2) : ℤ := if x = 0 then 1 else 0

def rankOneOnlyUtility (aBound w : ℕ) : Fin (aBound + 2) → (Fin (2 + w) → Fin 2) → ℤ :=
  fun a s => a.1 * unaryZeroIndicator (s 0)

def rankOneOnlyPairwise (aBound w : ℕ) : PairwiseUtility (rankOneOnlyUtility aBound w) where
  unary i a x := if i = 0 then a.1 * unaryZeroIndicator x else 0
  binary _ _ _ _ _ := 0
  interacts := @completeInteracts (2 + w)
  interacts_symm := @completeInteracts_symm (2 + w)
  decomp := by
    intro a s
    simp [rankOneOnlyUtility, completeInteracts, unaryZeroIndicator]

def rankOneOnlyTensor (aBound w : ℕ) : TensorRankDecomposition (rankOneOnlyUtility aBound w) 1 where
  weight := fun _ => 1
  actionFactor := fun _ a => a.1
  coordFactor := fun _ i x => if i = 0 then unaryZeroIndicator x else 1
  decomp := by
    intro a s
    simp [rankOneOnlyUtility, unaryZeroIndicator]

def rankOneOnlyDimensional (aBound w : ℕ) : Fin (aBound + 2) → DimensionalStateSpace 2 (2 + w) → ℤ :=
  fun a s => rankOneOnlyUtility aBound w a s.state

theorem rankOneOnly_not_symmetric (aBound w : ℕ) :
    ¬ SymmetricUtility (rankOneOnlyDimensional aBound w) := by
  intro hsym
  let σ : CoordinatePermutation (2 + w) := Equiv.swap 0 1
  let s : DimensionalStateSpace 2 (2 + w) := ⟨fun i => if i = 0 then 0 else 1⟩
  have hs : rankOneOnlyDimensional aBound w (1 : Fin (aBound + 2)) s = 1 := by
    simp [rankOneOnlyDimensional, rankOneOnlyUtility, unaryZeroIndicator, s]
  have hsp : rankOneOnlyDimensional aBound w (1 : Fin (aBound + 2)) (s.permute σ) = 0 := by
    have hdim : 2 + w ≠ 1 := by omega
    simp [rankOneOnlyDimensional, rankOneOnlyUtility, unaryZeroIndicator, s, σ, hdim,
      DimensionalStateSpace.permute, Equiv.swap_apply_def]
  have hEq := hsym σ (1 : Fin (aBound + 2)) s
  rw [hs, hsp] at hEq
  omega

def rankOneOnlyDP (w : ℕ) : FiniteDecisionProblem (A := Fin 2) (S := (Fin (2 + w) → Fin 2)) where
  actions := Finset.univ
  states := Finset.univ
  utility := rankOneOnlyUtility 0 w

theorem rankOneOnly_not_separable (w : ℕ) :
    ¬ Nonempty (SeparableUtility (dp := rankOneOnlyDP w)) := by
  intro h
  rcases h with ⟨hsep⟩
  let s0 : Fin (2 + w) → Fin 2 := fun i => if i = 0 then 0 else 1
  let s1 : Fin (2 + w) → Fin 2 := fun _ => 1
  have h10 : 1 = hsep.actionValue 1 + hsep.stateValue s0 := by
    exact hsep.utility_eq 1 s0
  have h00 : 0 = hsep.actionValue 0 + hsep.stateValue s0 := by
    simpa using hsep.utility_eq 0 s0
  have h11 : 0 = hsep.actionValue 1 + hsep.stateValue s1 := by
    simpa using hsep.utility_eq 1 s1
  have h01 : 0 = hsep.actionValue 0 + hsep.stateValue s1 := by
    simpa using hsep.utility_eq 0 s1
  linarith

theorem low_rank_only_witness (aBound w : ℕ) :
    ∃ u : Fin (aBound + 2) → (Fin (2 + w) → Fin 2) → ℤ,
      Nonempty (TensorRankDecomposition u 1) ∧
      Nonempty (PairwiseUtility u) ∧
      ¬ Nonempty (SeparableUtility (dp := rankOneOnlyDP w)) ∧
      ¬ SymmetricUtility (rankOneOnlyDimensional aBound w) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + w)) (@completeInteracts_symm (2 + w))) w ∧
      aBound < Fintype.card (Fin (aBound + 2)) := by
  refine ⟨rankOneOnlyUtility aBound w, ⟨rankOneOnlyTensor aBound w⟩, ⟨rankOneOnlyPairwise aBound w⟩,
    rankOneOnly_not_separable w, rankOneOnly_not_symmetric aBound w, ?_, by simp⟩
  simpa [interactionGraph_completeInteracts_eq_top, Nat.add_comm] using
    completeGraph_not_realTreewidth_le_of_large (n := 2 + w) (w := w) (by omega)

end Paper4dFrontier
