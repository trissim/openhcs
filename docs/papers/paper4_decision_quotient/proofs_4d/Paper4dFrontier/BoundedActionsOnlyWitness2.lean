import Paper4dFrontier.TensorRestriction
import Paper4dFrontier.RealTreewidthWitnesses
import Paper4dFrontier.FirstTwoIndicator
import Paper4dFrontier.SymmetryRankWitness
import DecisionQuotient.Finite

namespace Paper4dFrontier

open DecisionQuotient
open Classical

def boundedActionsOnlyUtility (w R : ℕ) : Bool → (Fin (2 + (w + 1)) → Fin (R + 2)) → ℤ :=
  fun a s => if a then (if s 0 = s 1 then 1 else 0) else 0

def boundedActionsOnlyDP (w R : ℕ) : FiniteDecisionProblem (A := Bool) (S := (Fin (2 + (w + 1)) → Fin (R + 2))) where
  actions := Finset.univ
  states := Finset.univ
  utility := boundedActionsOnlyUtility w R

def boundedActionsOnlyDimensional (w R : ℕ) : Bool → DimensionalStateSpace (R + 2) (2 + (w + 1)) → ℤ :=
  fun a s => boundedActionsOnlyUtility w R a s.state

theorem boundedActionsOnly_not_separable (w R : ℕ) :
    ¬ Nonempty (SeparableUtility (dp := boundedActionsOnlyDP w R)) := by
  intro h
  rcases h with ⟨hsep⟩
  let sEq : Fin (2 + (w + 1)) → Fin (R + 2) := fun _ => 0
  let sNe : Fin (2 + (w + 1)) → Fin (R + 2) := fun i => if i = 0 then 0 else 1
  have hdim : (2 + (w + 1)) ≠ 1 := by omega
  have h11 : 1 = hsep.actionValue true + hsep.stateValue sEq := by
    exact hsep.utility_eq true sEq
  have h01 : 0 = hsep.actionValue false + hsep.stateValue sEq := by
    simpa using hsep.utility_eq false sEq
  have h10 : 0 = hsep.actionValue true + hsep.stateValue sNe := by
    simpa [boundedActionsOnlyDP, boundedActionsOnlyUtility, sNe, hdim] using hsep.utility_eq true sNe
  have h00 : 0 = hsep.actionValue false + hsep.stateValue sNe := by
    simpa using hsep.utility_eq false sNe
  linarith

theorem boundedActionsOnly_not_symmetric (w R : ℕ) :
    ¬ SymmetricUtility (boundedActionsOnlyDimensional w R) := by
  intro hsym
  apply firstTwoEq_not_symmetric w R
  intro σ _ s
  simpa [firstTwoEqDimensional, boundedActionsOnlyDimensional, boundedActionsOnlyUtility] using hsym σ true s

theorem boundedActionsOnly_not_low_rank (w R : ℕ) :
    ¬ Nonempty (TensorRankDecomposition (boundedActionsOnlyUtility w R) R) := by
  intro h
  rcases h with ⟨decomp⟩
  let sliced := sliceActionTensorRank true decomp
  let restricted := restrictTensorRankFirstTwo
    (u := sliceAction (boundedActionsOnlyUtility w R) true)
    (k := R + 2) (m := w + 1) sliced
  have hEq : restrictFirstTwo (u := sliceAction (boundedActionsOnlyUtility w R) true)
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
    unfold restrictFirstTwo sliceAction boundedActionsOnlyUtility diagIndicatorUtility
    rw [h0, h1]
    simp
  have hrest : Nonempty (TensorRankDecomposition (diagIndicatorUtility (R + 2)) R) := by
    have hrest' : Nonempty (TensorRankDecomposition
      (restrictFirstTwo (u := sliceAction (boundedActionsOnlyUtility w R) true)) R) := ⟨restricted⟩
    simpa [hEq] using hrest'
  exact diagIndicator_no_rank_of_lt (R + 2) R (by omega) hrest

theorem bounded_actions_only_witness (R w : ℕ) :
    ∃ u : Bool → (Fin (2 + (w + 1)) → Fin (R + 2)) → ℤ,
      ¬ Nonempty (SeparableUtility (dp := boundedActionsOnlyDP w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (boundedActionsOnlyDimensional w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w := by
  refine ⟨boundedActionsOnlyUtility w R,
    boundedActionsOnly_not_separable w R,
    boundedActionsOnly_not_low_rank w R,
    boundedActionsOnly_not_symmetric w R, ?_⟩
  simpa [interactionGraph_completeInteracts_eq_top] using
    completeGraph_not_realTreewidth_le_of_large (n := 2 + (w + 1)) (w := w) (by omega)

end Paper4dFrontier
