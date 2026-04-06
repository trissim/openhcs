import Paper4dFrontier.TensorRestriction
import Paper4dFrontier.RealTreewidthWitnesses
import Paper4dFrontier.SymmetryRankWitness
import Paper4dFrontier.FirstTwoIndicator
import DecisionQuotient.Finite

namespace Paper4dFrontier

open DecisionQuotient
open Classical

def separableOnlyUtility (aBound m R : ℕ) : Fin (aBound + 2) → (Fin (2 + (m + 1)) → Fin (R + 2)) → ℤ :=
  fun a s => a.1 + if s 0 = s 1 then 1 else 0

def separableOnlyDP (aBound m R : ℕ) : FiniteDecisionProblem (A := Fin (aBound + 2)) (S := (Fin (2 + (m + 1)) → Fin (R + 2))) where
  actions := Finset.univ
  states := Finset.univ
  utility := separableOnlyUtility aBound m R

def separableOnlySeparable (aBound m R : ℕ) : SeparableUtility (dp := separableOnlyDP aBound m R) where
  actionValue := fun a => a.1
  stateValue := fun s => if s 0 = s 1 then 1 else 0
  utility_eq := by intro a s; simp [separableOnlyDP, separableOnlyUtility]

def separableOnlyDimensional (aBound m R : ℕ) : Fin (aBound + 2) → DimensionalStateSpace (R + 2) (2 + (m + 1)) → ℤ :=
  fun a s => separableOnlyUtility aBound m R a s.state

theorem separableOnly_not_symmetric (aBound m R : ℕ) :
    ¬ SymmetricUtility (separableOnlyDimensional aBound m R) := by
  intro hsym
  apply firstTwoEq_not_symmetric m R
  intro σ _ s
  simpa [firstTwoEqDimensional, separableOnlyDimensional, separableOnlyUtility] using hsym σ (0 : Fin (aBound + 2)) s

theorem separableOnly_not_low_rank (aBound R w : ℕ) :
    ¬ Nonempty (TensorRankDecomposition (separableOnlyUtility aBound w R) R) := by
  intro h
  rcases h with ⟨decomp⟩
  let sliced := sliceActionTensorRank (0 : Fin (aBound + 2)) decomp
  let restricted := restrictTensorRankFirstTwo
    (u := sliceAction (separableOnlyUtility aBound w R) (0 : Fin (aBound + 2)))
    (k := R + 2) (m := w + 1) sliced
  have hEq : restrictFirstTwo (u := sliceAction (separableOnlyUtility aBound w R) (0 : Fin (aBound + 2)))
      = diagIndicatorUtility (R + 2) := by
    funext u
    cases u
    funext s
    have h0idx : (0 : Fin (2 + (w + 1))) = Fin.castAdd (w + 1) (0 : Fin 2) := by
      apply Fin.ext
      have hlt : 0 < 2 + (w + 1) := by omega
      simp [Nat.mod_eq_of_lt hlt]
    have h1idx : (1 : Fin (2 + (w + 1))) = Fin.castAdd (w + 1) (1 : Fin 2) := by
      apply Fin.ext
      have hlt : 1 < 2 + (w + 1) := by omega
      simp [Nat.mod_eq_of_lt hlt]
    have h0 : extendFirstTwoWithZero s (0 : Fin (2 + (w + 1))) = s 0 := by
      rw [h0idx]
      simpa [extendFirstTwoWithZero, Fin.append] using (Fin.append_left s (fun _ => 0) (0 : Fin 2))
    have h1 : extendFirstTwoWithZero s (1 : Fin (2 + (w + 1))) = s 1 := by
      rw [h1idx]
      simpa [extendFirstTwoWithZero, Fin.append] using (Fin.append_left s (fun _ => 0) (1 : Fin 2))
    unfold restrictFirstTwo sliceAction separableOnlyUtility diagIndicatorUtility
    rw [h0, h1]
    simp
  have hrest : Nonempty (TensorRankDecomposition (diagIndicatorUtility (R + 2)) R) := by
    have hrest' : Nonempty (TensorRankDecomposition
      (restrictFirstTwo (u := sliceAction (separableOnlyUtility aBound w R) (0 : Fin (aBound + 2)))) R) := ⟨restricted⟩
    simpa [hEq] using hrest'
  exact diagIndicator_no_rank_of_lt (R + 2) R (by omega) hrest

theorem separable_only_witness (aBound R w : ℕ) :
    ∃ u : Fin (aBound + 2) → (Fin (2 + (w + 1)) → Fin (R + 2)) → ℤ,
      Nonempty (SeparableUtility (dp := separableOnlyDP aBound w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (separableOnlyDimensional aBound w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w ∧
      aBound < Fintype.card (Fin (aBound + 2)) := by
  refine ⟨separableOnlyUtility aBound w R, ⟨separableOnlySeparable aBound w R⟩,
    separableOnly_not_low_rank aBound R w, separableOnly_not_symmetric aBound w R, ?_, by simp⟩
  simpa [interactionGraph_completeInteracts_eq_top] using
    completeGraph_not_realTreewidth_le_of_large (n := 2 + (w + 1)) (w := w) (by omega)

end Paper4dFrontier
