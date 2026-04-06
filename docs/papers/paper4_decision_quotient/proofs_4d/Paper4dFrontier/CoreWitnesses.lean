import DecisionQuotient.Finite
import DecisionQuotient.Instances
import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.Dimensional
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Classical

/-- Two-action witness with state-dependent optimizer, hence non-separable. -/
def twoActionStateDependent : FiniteDecisionProblem (A := Bool) (S := Bool) where
  actions := Finset.univ
  states := Finset.univ
  utility a s := if a && s then 1 else 0

theorem twoActionStateDependent_actions_card : twoActionStateDependent.actions.card = 2 := by
  simp [twoActionStateDependent]

theorem twoActionStateDependent_not_separable :
    ¬ Nonempty (SeparableUtility (dp := twoActionStateDependent)) := by
  rintro ⟨hsep⟩
  have htt : 1 = hsep.actionValue true + hsep.stateValue true := by
    simpa [twoActionStateDependent, Bool.and_eq_true] using hsep.utility_eq true true
  have htf : 0 = hsep.actionValue true + hsep.stateValue false := by
    simpa [twoActionStateDependent] using hsep.utility_eq true false
  have hft : 0 = hsep.actionValue false + hsep.stateValue true := by
    simpa [twoActionStateDependent] using hsep.utility_eq false true
  have hff : 0 = hsep.actionValue false + hsep.stateValue false := by
    simpa [twoActionStateDependent] using hsep.utility_eq false false
  linarith

theorem bounded_actions_not_imply_separable_witness :
    ∃ dp : FiniteDecisionProblem (A := Bool) (S := Bool),
      dp.actions.card = 2 ∧ ¬ Nonempty (SeparableUtility (dp := dp)) := by
  exact ⟨twoActionStateDependent, twoActionStateDependent_actions_card,
    twoActionStateDependent_not_separable⟩

/-- Separable witness family with arbitrarily many actions. -/
def manyActionSeparable (m : ℕ) : FiniteDecisionProblem (A := Fin m) (S := Unit) where
  actions := Finset.univ
  states := Finset.univ
  utility a _ := a.1

def manyActionSeparable_sep (m : ℕ) :
    SeparableUtility (dp := manyActionSeparable m) where
  actionValue := fun a => a.1
  stateValue := fun _ => 0
  utility_eq := by
    intro a s
    simp [manyActionSeparable]

theorem manyActionSeparable_actions_card (m : ℕ) :
    (manyActionSeparable m).actions.card = m := by
  simp [manyActionSeparable]

theorem separable_not_imply_bounded_actions_witness (k : ℕ) :
    ∃ dp : FiniteDecisionProblem (A := Fin (k + 1)) (S := Unit),
      Nonempty (SeparableUtility (dp := dp)) ∧ k < dp.actions.card := by
  refine ⟨manyActionSeparable (k + 1), ⟨manyActionSeparable_sep (k + 1)⟩, ?_⟩
  simpa [manyActionSeparable_actions_card]

/-- Rank-1 utility witness on two binary coordinates. -/
def rankOneNonSymmetricUtility : Unit → (Fin 2 → Fin 2) → ℤ
  | (), s =>
      (if s 0 = 0 then 1 else 2) *
      (if s 1 = 0 then 1 else 3)

def rankOneNonSymmetricUtility_rank1 :
    TensorRankDecomposition rankOneNonSymmetricUtility 1 where
  weight := fun _ => 1
  actionFactor := fun _ _ => 1
  coordFactor := fun _ i x =>
    if i = 0 then
      if x = 0 then 1 else 2
    else
      if x = 0 then 1 else 3
  decomp := by
    intro a s
    simp [rankOneNonSymmetricUtility]

/-- The same rank-1 witness, viewed on dimensional states. -/
def rankOneNonSymmetricDimensional : Unit → DimensionalStateSpace 2 2 → ℤ
  | (), s => rankOneNonSymmetricUtility () s.state

theorem rankOneNonSymmetricDimensional_not_symmetric :
    ¬ SymmetricUtility rankOneNonSymmetricDimensional := by
  intro hsym
  let σ : CoordinatePermutation 2 := Equiv.swap 0 1
  let s : DimensionalStateSpace 2 2 :=
    ⟨fun i => if i = 0 then 1 else 0⟩
  have hs : rankOneNonSymmetricDimensional () s = 2 := by
    simp [rankOneNonSymmetricDimensional, rankOneNonSymmetricUtility, s]
  have hsp : rankOneNonSymmetricDimensional () (s.permute σ) = 3 := by
    simp [rankOneNonSymmetricDimensional, rankOneNonSymmetricUtility, s, σ,
      DimensionalStateSpace.permute, Equiv.swap_apply_def]
  have hEq := hsym σ () s
  rw [hs, hsp] at hEq
  omega

theorem low_rank_not_imply_coordinate_symmetry_witness :
    ∃ u : Unit → (Fin 2 → Fin 2) → ℤ,
      Nonempty (TensorRankDecomposition u 1) ∧
      ¬ SymmetricUtility (fun a s => u a s.state) := by
  exact ⟨rankOneNonSymmetricUtility,
    ⟨rankOneNonSymmetricUtility_rank1⟩,
    rankOneNonSymmetricDimensional_not_symmetric⟩

end Paper4dFrontier
