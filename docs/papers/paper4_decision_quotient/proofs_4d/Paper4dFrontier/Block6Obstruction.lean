import Paper4dFrontier.RealTreewidthWitnesses
import Paper4dFrontier.BinaryPairwiseDichotomy
import DecisionQuotient.Tractability.Dimensional
import DecisionQuotient.Tractability.Tightness
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Classical

/-- A unary utility depending only on the first coordinate, but presented with a
complete interaction relation. This shows that non-symmetry together with large
declared interaction graph does not force hardness. -/
def firstCoordOnlyCompleteUtility (w : ℕ) : Unit → (Fin (w + 2) → Fin 2) → ℤ
  | (), s => if s 0 = 1 then 1 else 0

def firstCoordOnlyCompletePairwise (w : ℕ) : PairwiseUtility (firstCoordOnlyCompleteUtility w) where
  unary i _ x := if i = 0 then if x = 1 then 1 else 0 else 0
  binary _ _ _ _ _ := 0
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    simp [firstCoordOnlyCompleteUtility, completeInteracts]

def firstCoordOnlyCompleteDimensional (w : ℕ) : Unit → DimensionalStateSpace 2 (w + 2) → ℤ
  | (), s => if s.state 0 = 1 then 1 else 0

def firstCoordOnlyCompleteProblemUtility (w : ℕ) : Unit → DimensionalStateSpace 2 (w + 2) → ℝ
  | (), s => if s.state 0 = 1 then 1 else 0

def firstCoordOnlyCompleteProblem (w : ℕ) : DecisionProblem Unit (DimensionalStateSpace 2 (w + 2)) where
  utility := firstCoordOnlyCompleteProblemUtility w

theorem firstCoordOnlyComplete_not_symmetric (w : ℕ) :
    ¬ SymmetricUtility (firstCoordOnlyCompleteDimensional w) := by
  intro hsym
  let σ : CoordinatePermutation (w + 2) := Equiv.swap 0 1
  let s : DimensionalStateSpace 2 (w + 2) :=
    ⟨fun i => if i = 0 then 1 else 0⟩
  have hs : firstCoordOnlyCompleteDimensional w () s = 1 := by
    simp [firstCoordOnlyCompleteDimensional, s]
  have hsp : firstCoordOnlyCompleteDimensional w () (s.permute σ) = 0 := by
    simp [firstCoordOnlyCompleteDimensional, s, σ, DimensionalStateSpace.permute,
      Equiv.swap_apply_def]
  have hEq := hsym σ () s
  rw [hs, hsp] at hEq
  omega

theorem firstCoordOnlyComplete_all_sufficient (w : ℕ) (I : Finset (Fin (w + 2))) :
    (firstCoordOnlyCompleteProblem w).isSufficient I := by
  simpa [firstCoordOnlyCompleteProblem] using
    (single_action_always_sufficient (dp := firstCoordOnlyCompleteProblem w) I)

/-- Obstruction to the naive full Block 6 dichotomy: there is an unbounded-treewidth,
non-symmetric binary pairwise family whose sufficiency problem is trivial because
it has a single action. -/
theorem block6_full_dichotomy_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (firstCoordOnlyCompleteUtility w)) ∧
    ¬ SymmetricUtility (firstCoordOnlyCompleteDimensional w) ∧
    ¬ realTreewidth_le
      (InteractionGraph (@completeInteracts (w + 2)) (@completeInteracts_symm (w + 2))) w ∧
    ∀ I : Finset (Fin (w + 2)), (firstCoordOnlyCompleteProblem w).isSufficient I := by
  refine ⟨⟨firstCoordOnlyCompletePairwise w⟩, firstCoordOnlyComplete_not_symmetric w, ?_, ?_⟩
  · simpa [interactionGraph_completeInteracts_eq_top] using completeGraph_not_realTreewidth_le w
  · intro I
    exact firstCoordOnlyComplete_all_sufficient w I

def completePairIndicator (x y : Fin 2) : ℤ :=
  if x = 1 ∧ y = 1 then 1 else 0

theorem completePairIndicator_cross : binaryCrossDifference completePairIndicator = 1 := by
  simp [binaryCrossDifference, completePairIndicator]

noncomputable def constantGapCompleteUtility (w : ℕ) : Bool → (Fin (w + 2) → Fin 2) → ℤ := by
  classical
  exact fun a s =>
    (∑ i : Fin (w + 2), if i = 0 then (if a then 0 else 1) + (if s i = 1 then 1 else 0) else 0) +
      ∑ i : Fin (w + 2),
        ∑ j : Fin (w + 2),
          if completeInteracts i j ∧ i < j then (completePairIndicator (s i) (s j)) else 0

def constantGapCompletePairwise (w : ℕ) : PairwiseUtility (constantGapCompleteUtility w) where
  unary i a x := if i = 0 then (if a then 0 else 1) + (if x = 1 then 1 else 0) else 0
  binary _ _ _ x y := completePairIndicator x y
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    simp [constantGapCompleteUtility]

noncomputable def constantGapCompleteDimensional (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℤ := by
  classical
  exact fun a s =>
    (∑ i : Fin (w + 2), if i = 0 then (if a then 0 else 1) + (if s.state i = 1 then 1 else 0) else 0) +
      ∑ i : Fin (w + 2),
        ∑ j : Fin (w + 2),
          if completeInteracts i j ∧ i < j then (completePairIndicator (s.state i) (s.state j)) else 0

def completePairIndicatorReal (x y : Fin 2) : ℝ :=
  if x = 1 ∧ y = 1 then 1 else 0

noncomputable def constantGapCompleteProblemUtility (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℝ := by
  classical
  exact fun a s =>
    (∑ i : Fin (w + 2), if i = 0 then (if a then 0 else 1) + (if s.state i = 1 then 1 else 0) else 0) +
      ∑ i : Fin (w + 2),
        ∑ j : Fin (w + 2),
          if completeInteracts i j ∧ i < j then (completePairIndicatorReal (s.state i) (s.state j)) else 0

noncomputable def constantGapCompleteProblem (w : ℕ) : DecisionProblem Bool (DimensionalStateSpace 2 (w + 2)) where
  utility := constantGapCompleteProblemUtility w

theorem completePairSum_single_one (n : ℕ) (i0 : Fin n) :
    (∑ i : Fin n,
      ∑ j : Fin n,
        if i ≠ j ∧ i < j then (completePairIndicator (if i = i0 then 1 else 0) (if j = i0 then 1 else 0)) else 0) = 0 := by
  refine Finset.sum_eq_zero ?_
  intro i hi
  refine Finset.sum_eq_zero ?_
  intro j hj
  by_cases hi0 : i = i0
  · by_cases hj0 : j = i0
    · have hnotlt : ¬ i < j := by simpa [hi0, hj0]
      simp [hi0, hj0, hnotlt, completePairIndicator]
    · simp [hi0, hj0, completePairIndicator]
  · simp [hi0, completePairIndicator]

theorem constantGapComplete_not_symmetric (w : ℕ) :
    ¬ SymmetricUtility (constantGapCompleteDimensional w) := by
  intro hsym
  let n := w + 2
  let i0 : Fin n := 0
  let i1 : Fin n := 1
  let σ : CoordinatePermutation n := Equiv.swap i0 i1
  let s : DimensionalStateSpace 2 n :=
    ⟨fun i => if i = i0 then 1 else 0⟩
  have hsum :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then (completePairIndicator (s.state i) (s.state j)) else 0) = 0 := by
    simpa [s, i0, completeInteracts] using completePairSum_single_one n i0
  have hperm : (s.permute σ).state = fun i => if i = i1 then 1 else 0 := by
    funext i
    by_cases hi1 : i = i1
    · subst hi1
      change s.state (σ.symm i1) = 1
      have hswap : σ.symm i1 = i0 := by simpa [σ] using (Equiv.swap_apply_right i0 i1)
      simp [s, hswap, i0]
    · by_cases hi0 : i = i0
      · subst hi0
        change s.state (σ.symm i0) = 0
        have hswap : σ.symm i0 = i1 := by simpa [σ] using (Equiv.swap_apply_left i0 i1)
        simp [s, hswap, i0, i1]
      · have hswap : σ.symm i = i := by simpa [σ] using (Equiv.swap_apply_of_ne_of_ne hi0 hi1)
        simp [DimensionalStateSpace.permute, s, hswap, hi0, hi1]
  have hsumPerm :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then (completePairIndicator ((s.permute σ).state i) ((s.permute σ).state j)) else 0) = 0 := by
    rw [hperm]
    simpa [i1, completeInteracts] using completePairSum_single_one n i1
  have hs : constantGapCompleteDimensional w false s = 2 := by
    unfold constantGapCompleteDimensional
    rw [hsum]
    simp [s, i0, i1, n]
  have hsp : constantGapCompleteDimensional w false (s.permute σ) = 1 := by
    unfold constantGapCompleteDimensional
    rw [hsumPerm]
    rw [hperm]
    simp [i0, i1, n]
  have hEq := hsym σ false s
  rw [hs, hsp] at hEq
  omega

theorem constantGapComplete_hasBinaryPairInteraction (w : ℕ) {i j : Fin (w + 2)}
    (hij : i ≠ j) :
    HasBinaryPairInteraction (constantGapCompleteUtility w) i j := by
  obtain hijlt | hjilt := lt_or_gt_of_ne hij
  · refine ⟨false, ?_⟩
    have h := pairCrossDifference_eq_binaryCrossDifference_of_lt
      (pw := constantGapCompletePairwise w) (a := false) hijlt
    have h' : pairCrossDifference (constantGapCompleteUtility w) false i j = 1 := by
      simpa [constantGapCompletePairwise, completeInteracts, completePairIndicator_cross, hijlt.ne]
        using h
    rw [h']
    norm_num
  · have hrev : HasBinaryPairInteraction (constantGapCompleteUtility w) j i := by
      refine ⟨false, ?_⟩
      have h := pairCrossDifference_eq_binaryCrossDifference_of_lt
        (pw := constantGapCompletePairwise w) (a := false) hjilt
      have h' : pairCrossDifference (constantGapCompleteUtility w) false j i = 1 := by
        simpa [constantGapCompletePairwise, completeInteracts, completePairIndicator_cross, hjilt.ne]
          using h
      rw [h']
      norm_num
    exact HasBinaryPairInteraction_symm (u := constantGapCompleteUtility w) j i hrev

theorem constantGapComplete_genuineInteractionGraph_eq_top (w : ℕ) :
    genuineInteractionGraph (constantGapCompleteUtility w) = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [genuineInteractionGraph, InteractionGraph]
  · simp [genuineInteractionGraph, InteractionGraph, h, constantGapComplete_hasBinaryPairInteraction w h]

theorem constantGapComplete_false_optimal (w : ℕ) (s : DimensionalStateSpace 2 (w + 2)) :
    (constantGapCompleteProblem w).isOptimal false s := by
  intro a'
  cases a' <;> simp [DecisionProblem.isOptimal, constantGapCompleteProblem,
    constantGapCompleteProblemUtility] <;> linarith

theorem constantGapComplete_true_not_optimal (w : ℕ) (s : DimensionalStateSpace 2 (w + 2)) :
    true ∉ (constantGapCompleteProblem w).Opt s := by
  intro htrue
  have h := htrue false
  simp [DecisionProblem.Opt, DecisionProblem.isOptimal, constantGapCompleteProblem,
    constantGapCompleteProblemUtility] at h
  linarith

theorem constantGapComplete_opt (w : ℕ) (s : DimensionalStateSpace 2 (w + 2)) :
    (constantGapCompleteProblem w).Opt s = {false} := by
  ext a
  cases a
  · simp [DecisionProblem.Opt, constantGapComplete_false_optimal]
  · simpa [DecisionProblem.Opt] using constantGapComplete_true_not_optimal w s

theorem constantGapComplete_all_sufficient (w : ℕ) (I : Finset (Fin (w + 2))) :
    (constantGapCompleteProblem w).isSufficient I := by
  intro s s' _
  rw [constantGapComplete_opt, constantGapComplete_opt]

/-- Even after replacing the declared interaction graph by the genuine interaction
graph, the naive Block 6 dichotomy still fails: this family has complete genuine
pair interaction and is not coordinate-symmetric, but the optimizer is constant. -/
theorem block6_genuine_interaction_dichotomy_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (constantGapCompleteUtility w)) ∧
    ¬ SymmetricUtility (constantGapCompleteDimensional w) ∧
    ¬ realTreewidth_le (genuineInteractionGraph (constantGapCompleteUtility w)) w ∧
    ∀ I : Finset (Fin (w + 2)), (constantGapCompleteProblem w).isSufficient I := by
  refine ⟨⟨constantGapCompletePairwise w⟩, constantGapComplete_not_symmetric w, ?_, ?_⟩
  · simpa [constantGapComplete_genuineInteractionGraph_eq_top] using completeGraph_not_realTreewidth_le w
  · intro I
    exact constantGapComplete_all_sufficient w I

end Paper4dFrontier
