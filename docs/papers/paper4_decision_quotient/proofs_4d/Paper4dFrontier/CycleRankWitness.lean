import Paper4dFrontier.StructuralWitnesses
import DecisionQuotient.Tractability.SeparableUtility
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

def bitIndicator (b x : Fin 2) : ℤ := if x = b then 1 else 0

def edgeEquality (x y : Fin 2) : ℤ := if x = y then 1 else 0

/-- Sum of three pairwise equality checks around a 3-cycle. -/
def cycleEqualityUtility : Unit → (Fin 3 → Fin 2) → ℤ
  | (), s => edgeEquality (s 0) (s 1) + edgeEquality (s 1) (s 2) + edgeEquality (s 2) (s 0)

def cycleEqualityRank6 : TensorRankDecomposition cycleEqualityUtility 6 where
  weight := fun _ => 1
  actionFactor := fun _ _ => 1
  coordFactor := fun r i x =>
    match r.1, i.1 with
    | 0, 0 => bitIndicator 0 x
    | 0, 1 => bitIndicator 0 x
    | 0, _ => 1
    | 1, 0 => bitIndicator 1 x
    | 1, 1 => bitIndicator 1 x
    | 1, _ => 1
    | 2, 1 => bitIndicator 0 x
    | 2, 2 => bitIndicator 0 x
    | 2, _ => 1
    | 3, 1 => bitIndicator 1 x
    | 3, 2 => bitIndicator 1 x
    | 3, _ => 1
    | 4, 2 => bitIndicator 0 x
    | 4, 0 => bitIndicator 0 x
    | 4, _ => 1
    | 5, 2 => bitIndicator 1 x
    | 5, 0 => bitIndicator 1 x
    | 5, _ => 1
    | _, _ => 1
  decomp := by
    intro a s
    have h0 : s 0 = 0 ∨ s 0 = 1 := by
      rcases Fin.eq_zero_or_eq_succ (s 0) with h0 | ⟨j0, h0⟩
      · exact Or.inl h0
      · fin_cases j0
        exact Or.inr h0
    have h1 : s 1 = 0 ∨ s 1 = 1 := by
      rcases Fin.eq_zero_or_eq_succ (s 1) with h1 | ⟨j1, h1⟩
      · exact Or.inl h1
      · fin_cases j1
        exact Or.inr h1
    have h2 : s 2 = 0 ∨ s 2 = 1 := by
      rcases Fin.eq_zero_or_eq_succ (s 2) with h2 | ⟨j2, h2⟩
      · exact Or.inl h2
      · fin_cases j2
        exact Or.inr h2
    rcases h0 with h0 | h0 <;> rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2 <;>
      simp [cycleEqualityUtility, edgeEquality, bitIndicator, h0, h1, h2,
        Fin.sum_univ_six, Fin.prod_univ_three]

theorem low_rank_not_imply_tree_structure_witness :
    ∃ u : Unit → (Fin 3 → Fin 2) → ℤ,
      Nonempty (TensorRankDecomposition u 6) ∧ ¬ TreeStructured cycleDeps := by
  exact ⟨cycleEqualityUtility, ⟨cycleEqualityRank6⟩, cycleDeps_not_treeStructured⟩

end Paper4dFrontier
