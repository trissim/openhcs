import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Dimensional
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

def finBitVal (x : Fin 2) : ℤ := if x = 1 then 1 else 0

/-- Empty interaction relation. -/
def noInteracts {n : ℕ} : Fin n → Fin n → Prop := fun _ _ => False

theorem noInteracts_symm {n : ℕ} : ∀ (i j : Fin n), noInteracts i j → noInteracts j i := by
  intro i j h
  cases h

/-- A bounded-treewidth witness family with arbitrarily many actions. -/
def manyActionTreewidthUtility (m : ℕ) : Fin m → (Fin 1 → Fin 2) → ℤ :=
  fun a s => a.1 + finBitVal (s 0)

def manyActionTreewidthPairwise (m : ℕ) : PairwiseUtility (manyActionTreewidthUtility m) where
  unary i a x := a.1 + finBitVal x
  binary _ _ _ _ _ := 0
  interacts := noInteracts
  interacts_symm := noInteracts_symm
  decomp := by
    intro a s
    simp [manyActionTreewidthUtility, finBitVal, noInteracts]

theorem emptyGraph1_treewidth_zero :
    treewidth_le (InteractionGraph (@noInteracts 1) (@noInteracts_symm 1)) 0 := by
  refine ⟨fun _ => {0}, ?_⟩
  intro i
  simp

theorem emptyGraph2_treewidth_zero :
    treewidth_le (InteractionGraph (@noInteracts 2) (@noInteracts_symm 2)) 0 := by
  refine ⟨fun _ => {0}, ?_⟩
  intro i
  simp

theorem bounded_treewidth_not_imply_bounded_actions_witness (k : ℕ) :
    ∃ u : Fin (k + 1) → (Fin 1 → Fin 2) → ℤ,
      Nonempty (PairwiseUtility u) ∧
      treewidth_le (InteractionGraph (@noInteracts 1) (@noInteracts_symm 1)) 0 ∧
      k < Fintype.card (Fin (k + 1)) := by
  refine ⟨manyActionTreewidthUtility (k + 1), ⟨manyActionTreewidthPairwise (k + 1)⟩,
    emptyGraph1_treewidth_zero, ?_⟩
  simp

/-- A treewidth witness that is not coordinate-symmetric. -/
def nonSymmetricTreewidthUtility : Unit → (Fin 2 → Fin 2) → ℤ
  | (), s => finBitVal (s 0)

def nonSymmetricTreewidthPairwise : PairwiseUtility nonSymmetricTreewidthUtility where
  unary i _ x := if i = 0 then finBitVal x else 0
  binary _ _ _ _ _ := 0
  interacts := noInteracts
  interacts_symm := noInteracts_symm
  decomp := by
    intro a s
    simp [nonSymmetricTreewidthUtility, finBitVal, noInteracts]

def nonSymmetricTreewidthDimensional : Unit → DimensionalStateSpace 2 2 → ℤ
  | (), s => finBitVal (s.state 0)

theorem nonSymmetricTreewidth_not_symmetric :
    ¬ SymmetricUtility nonSymmetricTreewidthDimensional := by
  intro hsym
  let σ : CoordinatePermutation 2 := Equiv.swap 0 1
  let s : DimensionalStateSpace 2 2 := ⟨fun i => if i = 0 then 1 else 0⟩
  have hs : nonSymmetricTreewidthDimensional () s = 1 := by
    simp [nonSymmetricTreewidthDimensional, s, finBitVal]
  have hsp : nonSymmetricTreewidthDimensional () (s.permute σ) = 0 := by
    simp [nonSymmetricTreewidthDimensional, s, σ, finBitVal,
      DimensionalStateSpace.permute, Equiv.swap_apply_def]
  have hEq := hsym σ () s
  rw [hs, hsp] at hEq
  omega

theorem bounded_treewidth_not_imply_coordinate_symmetry_witness :
    Nonempty (PairwiseUtility nonSymmetricTreewidthUtility) ∧
    treewidth_le (InteractionGraph (@noInteracts 2) (@noInteracts_symm 2)) 0 ∧
    ¬ SymmetricUtility nonSymmetricTreewidthDimensional := by
  exact ⟨⟨nonSymmetricTreewidthPairwise⟩, emptyGraph2_treewidth_zero,
    nonSymmetricTreewidth_not_symmetric⟩

/-- 3-cycle dependency witness for strictness of tree structure inside bounded treewidth. -/
def cycleDeps : Fin 3 → Finset (Fin 3)
  | 0 => {2}
  | 1 => {0}
  | 2 => {1}

theorem cycleDeps_not_treeStructured : ¬ TreeStructured cycleDeps := by
  intro htree
  have hmem : 2 ∈ cycleDeps 0 := by simp [cycleDeps]
  have hlt := htree 0 2 hmem
  norm_num at hlt

def cycleInteracts : Fin 3 → Fin 3 → Prop
  | 0, 1 => True
  | 1, 0 => True
  | 1, 2 => True
  | 2, 1 => True
  | 2, 0 => True
  | 0, 2 => True
  | _, _ => False

theorem cycleInteracts_symm : ∀ i j, cycleInteracts i j → cycleInteracts j i := by
  intro i j hij
  fin_cases i <;> fin_cases j <;> simp [cycleInteracts] at hij ⊢

theorem cycleGraph_treewidth_two :
    treewidth_le (InteractionGraph cycleInteracts cycleInteracts_symm) 2 := by
  refine ⟨fun _ => {0, 1, 2}, ?_⟩
  intro i
  simp

theorem tree_structure_strictly_inside_bounded_treewidth_witness :
    ¬ TreeStructured cycleDeps ∧
    treewidth_le (InteractionGraph cycleInteracts cycleInteracts_symm) 2 := by
  exact ⟨cycleDeps_not_treeStructured, cycleGraph_treewidth_two⟩

end Paper4dFrontier
