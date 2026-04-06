import Paper4dFrontier.TensorRestriction
import Paper4dFrontier.RealTreewidth
import Paper4dFrontier.StructuralWitnesses
import Paper4dFrontier.SymmetryRankWitness
import DecisionQuotient.Finite

namespace Paper4dFrontier

open DecisionQuotient
open Classical

abbrev BTWState (R : ℕ) := Fin (2 + 1) → Fin (R + 2)

def boundedTreewidthOnlyUtility (aBound R : ℕ) : Fin (aBound + 2) → BTWState R → ℤ :=
  fun a s => a.1 * (if s 0 = s 1 then 1 else 0)

def boundedTreewidthOnlyDP (aBound R : ℕ) : FiniteDecisionProblem (A := Fin (aBound + 2)) (S := BTWState R) where
  actions := Finset.univ
  states := Finset.univ
  utility := boundedTreewidthOnlyUtility aBound R

theorem boundedTreewidthOnly_not_separable (aBound R : ℕ) :
    ¬ Nonempty (SeparableUtility (dp := boundedTreewidthOnlyDP aBound R)) := by
  intro h
  rcases h with ⟨hsep⟩
  let sEq : BTWState R := fun _ => 0
  let sNe : BTWState R := fun i => if i = 0 then 0 else 1
  have h10 : 1 = hsep.actionValue 1 + hsep.stateValue sEq := by
    exact hsep.utility_eq 1 sEq
  have h00 : 0 = hsep.actionValue 0 + hsep.stateValue sEq := by
    simpa using hsep.utility_eq 0 sEq
  have h11 : 0 = hsep.actionValue 1 + hsep.stateValue sNe := by
    simpa [boundedTreewidthOnlyDP, boundedTreewidthOnlyUtility, sNe] using hsep.utility_eq 1 sNe
  have h01 : 0 = hsep.actionValue 0 + hsep.stateValue sNe := by
    simpa using hsep.utility_eq 0 sNe
  linarith

def boundedTreewidthOnlyDimensional (aBound R : ℕ) : Fin (aBound + 2) → DimensionalStateSpace (R + 2) (2 + 1) → ℤ :=
  fun a s => boundedTreewidthOnlyUtility aBound R a s.state

theorem boundedTreewidthOnly_not_symmetric (aBound R : ℕ) :
    ¬ SymmetricUtility (boundedTreewidthOnlyDimensional aBound R) := by
  intro hsym
  let σ : CoordinatePermutation (2 + 1) := Equiv.swap 0 2
  let s : DimensionalStateSpace (R + 2) (2 + 1) := ⟨fun i => if i = 0 then 0 else if i = 1 then 0 else 1⟩
  have hs : boundedTreewidthOnlyDimensional aBound R 1 s = 1 := by
    simp [boundedTreewidthOnlyDimensional, boundedTreewidthOnlyUtility, s]
  have hsp : boundedTreewidthOnlyDimensional aBound R 1 (s.permute σ) = 0 := by
    simp [boundedTreewidthOnlyDimensional, boundedTreewidthOnlyUtility, s, σ, DimensionalStateSpace.permute, Equiv.swap_apply_def]
  have hEq := hsym σ 1 s
  rw [hs, hsp] at hEq
  omega

theorem boundedTreewidthOnly_not_low_rank (aBound R : ℕ) :
    ¬ Nonempty (TensorRankDecomposition (boundedTreewidthOnlyUtility aBound R) R) := by
  intro h
  rcases h with ⟨decomp⟩
  let a1 : Fin (aBound + 2) := 1
  let sliced := sliceActionTensorRank a1 decomp
  let restricted := restrictTensorRankFirstTwo (u := sliceAction (boundedTreewidthOnlyUtility aBound R) a1) (k := R + 2) (m := 1) sliced
  have hEq : restrictFirstTwo (u := sliceAction (boundedTreewidthOnlyUtility aBound R) a1) = diagIndicatorUtility (R + 2) := by
    funext u
    cases u
    funext s
    have h0idx : (0 : Fin (2 + 1)) = Fin.castAdd 1 (0 : Fin 2) := by
      apply Fin.ext
      simp
    have h1idx : (1 : Fin (2 + 1)) = Fin.castAdd 1 (1 : Fin 2) := by
      apply Fin.ext
      simp
    have h0 : extendFirstTwoWithZero s (0 : Fin (2 + 1)) = s 0 := by
      rw [h0idx]
      show Fin.addCases s (fun _ => 0) (Fin.castAdd 1 (0 : Fin 2)) = s 0
      rfl
    have h1 : extendFirstTwoWithZero s (1 : Fin (2 + 1)) = s 1 := by
      rw [h1idx]
      show Fin.addCases s (fun _ => 0) (Fin.castAdd 1 (1 : Fin 2)) = s 1
      rfl
    unfold restrictFirstTwo sliceAction boundedTreewidthOnlyUtility diagIndicatorUtility
    rw [h0, h1]
    simp [a1]
  have hrest : Nonempty (TensorRankDecomposition (diagIndicatorUtility (R + 2)) R) := by
    have hrest' : Nonempty (TensorRankDecomposition (restrictFirstTwo (u := sliceAction (boundedTreewidthOnlyUtility aBound R) a1)) R) := ⟨restricted⟩
    simpa [hEq] using hrest'
  exact diagIndicator_no_rank_of_lt (R + 2) R (by omega) hrest

theorem interactionGraph_cycleInteracts_eq_top :
    InteractionGraph cycleInteracts cycleInteracts_symm = (⊤ : SimpleGraph (Fin 3)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [InteractionGraph, cycleInteracts]

def boundedTreewidthOnlyPairwise (aBound R : ℕ) : PairwiseUtility (boundedTreewidthOnlyUtility aBound R) where
  unary _ _ _ := 0
  binary i j a x y := if i = 0 ∧ j = 1 then a.1 * (if x = y then 1 else 0) else 0
  interacts := cycleInteracts
  interacts_symm := cycleInteracts_symm
  decomp := by
    intro a s
    rw [Fin.sum_univ_three, Fin.sum_univ_three, Fin.sum_univ_three, Fin.sum_univ_three]
    simp [boundedTreewidthOnlyUtility, cycleInteracts]

theorem bounded_treewidth_only_witness (aBound R : ℕ) :
    ∃ u : Fin (aBound + 2) → BTWState R → ℤ,
      Nonempty (PairwiseUtility u) ∧
      realTreewidth_le (InteractionGraph cycleInteracts cycleInteracts_symm) 2 ∧
      ¬ TreeStructured cycleDeps ∧
      ¬ Nonempty (SeparableUtility (dp := boundedTreewidthOnlyDP aBound R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (boundedTreewidthOnlyDimensional aBound R) ∧
      aBound < Fintype.card (Fin (aBound + 2)) := by
  refine ⟨boundedTreewidthOnlyUtility aBound R, ⟨boundedTreewidthOnlyPairwise aBound R⟩, ?_, cycleDeps_not_treeStructured,
    boundedTreewidthOnly_not_separable aBound R, boundedTreewidthOnly_not_low_rank aBound R, boundedTreewidthOnly_not_symmetric aBound R, by simp⟩
  simpa [interactionGraph_cycleInteracts_eq_top] using realTreewidth_le_card_pred (⊤ : SimpleGraph (Fin 3))

end Paper4dFrontier
