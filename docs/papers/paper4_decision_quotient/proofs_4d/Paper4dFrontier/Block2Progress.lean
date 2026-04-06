import Paper4dFrontier.LowRankOnlyWitness
import Paper4dFrontier.SeparableOnlyWitness
import Paper4dFrontier.BoundedActionsOnlyWitness2
import Paper4dFrontier.BoundedTreewidthOnlyWitness
import Paper4dFrontier.SymmetryOnlyWitness
import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Dimensional

namespace Paper4dFrontier

open DecisionQuotient

theorem all_independent_core_mechanisms_nondegenerate (aBound R w : ℕ) :
    (∃ u : Fin (aBound + 2) → (Fin (2 + w) → Fin 2) → ℤ,
      Nonempty (TensorRankDecomposition u 1) ∧
      Nonempty (PairwiseUtility u) ∧
      ¬ Nonempty (SeparableUtility (dp := rankOneOnlyDP w)) ∧
      ¬ SymmetricUtility (rankOneOnlyDimensional aBound w) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + w)) (@completeInteracts_symm (2 + w))) w ∧
      aBound < Fintype.card (Fin (aBound + 2))) ∧
    (∃ u : Fin (aBound + 2) → (Fin (2 + (w + 1)) → Fin (R + 2)) → ℤ,
      Nonempty (SeparableUtility (dp := separableOnlyDP aBound w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (separableOnlyDimensional aBound w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w ∧
      aBound < Fintype.card (Fin (aBound + 2))) ∧
    (∃ u : Bool → (Fin (2 + (w + 1)) → Fin (R + 2)) → ℤ,
      ¬ Nonempty (SeparableUtility (dp := boundedActionsOnlyDP w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (boundedActionsOnlyDimensional w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w) ∧
    (∃ u : Fin (aBound + 2) → BTWState R → ℤ,
      Nonempty (PairwiseUtility u) ∧
      realTreewidth_le (InteractionGraph cycleInteracts cycleInteracts_symm) 2 ∧
      ¬ TreeStructured cycleDeps ∧
      ¬ Nonempty (SeparableUtility (dp := boundedTreewidthOnlyDP aBound R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      ¬ SymmetricUtility (boundedTreewidthOnlyDimensional aBound R) ∧
      aBound < Fintype.card (Fin (aBound + 2))) ∧
    (∃ u : Fin (aBound + 2) → (Fin (2 + (w + 1)) → Fin (SymK w R)) → ℤ,
      ¬ Nonempty (SeparableUtility (dp := symmetryOnlyDP aBound w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      SymmetricUtility (symmetryOnlyDimensional aBound w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w ∧
      aBound < Fintype.card (Fin (aBound + 2))) := by
  exact ⟨low_rank_only_witness aBound w, separable_only_witness aBound R w,
    bounded_actions_only_witness R w, bounded_treewidth_only_witness aBound R,
    symmetry_only_witness aBound w R⟩

end Paper4dFrontier
