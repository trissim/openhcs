import Paper4dFrontier.TreewidthClique
import DecisionQuotient.Tractability.Dimensional
import DecisionQuotient.Tractability.TreeStructure
import Paper4dFrontier.StructuralWitnesses

namespace Paper4dFrontier

open DecisionQuotient

def completeInteracts {n : ℕ} : Fin n → Fin n → Prop := fun i j => i ≠ j

theorem completeInteracts_symm {n : ℕ} : ∀ (i j : Fin n), completeInteracts i j → completeInteracts j i := by
  intro i j hij
  exact hij.symm

theorem interactionGraph_completeInteracts_eq_top (n : ℕ) :
    InteractionGraph (@completeInteracts n) (@completeInteracts_symm n) = ⊤ := by
  ext i j
  simp [InteractionGraph, completeInteracts]

theorem interactionGraph_noInteracts_eq_bot (n : ℕ) :
    InteractionGraph (@noInteracts n) (@noInteracts_symm n) = ⊥ := by
  ext i j
  simp [InteractionGraph, noInteracts]

def boundedActionCompleteUtility (n : ℕ) : Bool → (Fin n → Fin 2) → ℤ :=
  fun a s => ∑ i, ((if a then 1 else 0) + if s i = 1 then 1 else 0)

def boundedActionCompletePairwise (n : ℕ) : PairwiseUtility (boundedActionCompleteUtility n) where
  unary i a x := (if a then 1 else 0) + (if x = 1 then 1 else 0)
  binary _ _ _ _ _ := 0
  interacts := completeInteracts
  interacts_symm := completeInteracts_symm
  decomp := by
    intro a s
    simp [boundedActionCompleteUtility, completeInteracts]

theorem bounded_actions_not_imply_real_treewidth_witness (w : ℕ) :
    ∃ u : Bool → (Fin (w + 2) → Fin 2) → ℤ,
      Nonempty (PairwiseUtility u) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (w + 2)) (@completeInteracts_symm (w + 2))) w := by
  refine ⟨boundedActionCompleteUtility (w + 2), ⟨boundedActionCompletePairwise (w + 2)⟩, ?_⟩
  simpa [interactionGraph_completeInteracts_eq_top] using completeGraph_not_realTreewidth_le w

def manyActionWidthZeroUtility (m : ℕ) : Fin m → (Fin 1 → Fin 2) → ℤ :=
  fun a _ => a.1

def manyActionWidthZeroPairwise (m : ℕ) : PairwiseUtility (manyActionWidthZeroUtility m) where
  unary _ a _ := a.1
  binary _ _ _ _ _ := 0
  interacts := @completeInteracts 1
  interacts_symm := @completeInteracts_symm 1
  decomp := by
    intro a s
    simp [manyActionWidthZeroUtility, completeInteracts]

theorem real_treewidth_not_imply_bounded_actions_witness (k : ℕ) :
    ∃ u : Fin (k + 1) → (Fin 1 → Fin 2) → ℤ,
      Nonempty (PairwiseUtility u) ∧
      realTreewidth_le (InteractionGraph (@completeInteracts 1) (@completeInteracts_symm 1)) 0 ∧
      k < Fintype.card (Fin (k + 1)) := by
  refine ⟨manyActionWidthZeroUtility (k + 1), ⟨manyActionWidthZeroPairwise (k + 1)⟩, ?_, by simp⟩
  simpa [interactionGraph_completeInteracts_eq_top] using realTreewidth_le_card_pred (⊤ : SimpleGraph (Fin 1))

def symmetricCompleteUtility (n : ℕ) : Unit → (Fin n → Fin 2) → ℤ :=
  fun _ _ => 0

def symmetricCompleteDimensional (n : ℕ) : Unit → DimensionalStateSpace 2 n → ℤ :=
  fun _ _ => 0

theorem symmetricCompleteDimensional_symmetric (n : ℕ) :
    SymmetricUtility (symmetricCompleteDimensional n) := by
  intro σ a s
  simp [symmetricCompleteDimensional]

def symmetricCompletePairwise (n : ℕ) : PairwiseUtility (symmetricCompleteUtility n) where
  unary _ _ _ := 0
  binary _ _ _ _ _ := 0
  interacts := completeInteracts
  interacts_symm := completeInteracts_symm
  decomp := by
    intro a s
    simp [symmetricCompleteUtility, completeInteracts]

theorem coordinate_symmetry_not_imply_real_treewidth_witness (w : ℕ) :
    ∃ u : Unit → (Fin (w + 2) → Fin 2) → ℤ,
      Nonempty (PairwiseUtility u) ∧
      SymmetricUtility (symmetricCompleteDimensional (w + 2)) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (w + 2)) (@completeInteracts_symm (w + 2))) w := by
  refine ⟨symmetricCompleteUtility (w + 2), ⟨symmetricCompletePairwise (w + 2)⟩,
    symmetricCompleteDimensional_symmetric (w + 2), ?_⟩
  simpa [interactionGraph_completeInteracts_eq_top] using completeGraph_not_realTreewidth_le w

theorem real_treewidth_not_imply_coordinate_symmetry_witness :
    Nonempty (PairwiseUtility nonSymmetricTreewidthUtility) ∧
    realTreewidth_le (InteractionGraph (@noInteracts 2) (@noInteracts_symm 2)) 1 ∧
    ¬ SymmetricUtility nonSymmetricTreewidthDimensional := by
  refine ⟨⟨nonSymmetricTreewidthPairwise⟩, ?_, nonSymmetricTreewidth_not_symmetric⟩
  simpa [interactionGraph_noInteracts_eq_bot] using realTreewidth_le_card_pred (⊥ : SimpleGraph (Fin 2))

theorem tree_structure_strictly_inside_real_treewidth_witness :
    ¬ TreeStructured cycleDeps ∧ realTreewidth_le (⊤ : SimpleGraph (Fin 3)) 2 := by
  refine ⟨cycleDeps_not_treeStructured, ?_⟩
  simpa using realTreewidth_le_card_pred (⊤ : SimpleGraph (Fin 3))

end Paper4dFrontier
