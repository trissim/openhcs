import Paper4dFrontier.RealTreewidthWitnesses
import Paper4dFrontier.BinaryPairwiseDichotomy
import Paper4dFrontier.DecisionRelevantPairwiseDichotomy
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

noncomputable def offsetBaseAsymmetricPairUtility (w : ℕ) : Bool → (Fin (w + 2) → Fin 2) → ℤ
  | false, s =>
      (if s 0 = 1 then 1 else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicator (s i) (s j) else 0
  | true, _ => 0

noncomputable def offsetBaseAsymmetricPairPairwise (w : ℕ) : PairwiseUtility (offsetBaseAsymmetricPairUtility w) where
  unary i a x := if a = false ∧ i = 0 then if x = 1 then 1 else 0 else 0
  binary _ _ a x y := if a = false then completePairIndicator x y else 0
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    cases a <;> simp [offsetBaseAsymmetricPairUtility, completeInteracts] <;> ring

noncomputable def offsetBaseAsymmetricPairDimensional (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℤ
  | false, s =>
      (if s.state 0 = 1 then 1 else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0
  | true, _ => 0

noncomputable def offsetBaseAsymmetricPairProblemUtility (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℝ
  | false, s =>
      (if s.state 0 = 1 then 1 else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0
  | true, _ => 0

noncomputable def offsetBaseAsymmetricPairProblem (w : ℕ) :
    DecisionProblem Bool (DimensionalStateSpace 2 (w + 2)) where
  utility := offsetBaseAsymmetricPairProblemUtility w

noncomputable def offsetCollapsedAsymmetricPairUtility (w : ℕ) : Bool → (Fin (w + 2) → Fin 2) → ℤ :=
  addActionOffset (offsetBaseAsymmetricPairUtility w) (fun a => if a then 0 else 1)

noncomputable def offsetCollapsedAsymmetricPairPairwise (w : ℕ) : PairwiseUtility (offsetCollapsedAsymmetricPairUtility w) where
  unary i a x := if i = 0 then if a then 0 else 1 + (if x = 1 then 1 else 0) else 0
  binary _ _ a x y := if a = false then completePairIndicator x y else 0
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    cases a <;> simp [offsetCollapsedAsymmetricPairUtility, addActionOffset,
      offsetBaseAsymmetricPairUtility, completeInteracts] <;> ring

noncomputable def offsetCollapsedAsymmetricPairDimensional (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℤ
  | false, s =>
      1 + (if s.state 0 = 1 then 1 else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0
  | true, _ => 0

noncomputable def offsetCollapsedAsymmetricPairProblemUtility (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℝ
  | false, s =>
      1 + (if s.state 0 = 1 then 1 else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0
  | true, _ => 0

noncomputable def offsetCollapsedAsymmetricPairProblem (w : ℕ) :
    DecisionProblem Bool (DimensionalStateSpace 2 (w + 2)) where
  utility := offsetCollapsedAsymmetricPairProblemUtility w

theorem completePairSum_single_one_real (n : ℕ) (i0 : Fin n) :
    (∑ i : Fin n,
      ∑ j : Fin n,
        if i ≠ j ∧ i < j then
          completePairIndicatorReal (if i = i0 then 1 else 0) (if j = i0 then 1 else 0)
        else 0) = 0 := by
  refine Finset.sum_eq_zero ?_
  intro i hi
  refine Finset.sum_eq_zero ?_
  intro j hj
  by_cases hi0 : i = i0
  · by_cases hj0 : j = i0
    · have hnotlt : ¬ i < j := by simpa [hi0, hj0]
      simp [hi0, hj0, hnotlt, completePairIndicatorReal]
    · simp [hi0, hj0, completePairIndicatorReal]
  · simp [hi0, completePairIndicatorReal]

theorem offsetBaseAsymmetricPair_hasDecisionRelevantInteraction (w : ℕ) {i j : Fin (w + 2)}
    (hij : i ≠ j) :
    HasDecisionRelevantBinaryPairInteraction (offsetBaseAsymmetricPairUtility w) i j := by
  obtain hijlt | hjilt := lt_or_gt_of_ne hij
  · refine ⟨false, true, ?_⟩
    have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
      (pw := offsetBaseAsymmetricPairPairwise w) (a := false) (b := true) hijlt
    have h' : actionGapCrossDifference (offsetBaseAsymmetricPairUtility w) false true i j = 1 := by
      simpa [offsetBaseAsymmetricPairPairwise, completeInteracts, completePairIndicator_cross, hijlt.ne]
        using h
    rw [h']
    norm_num
  · have hrev : HasDecisionRelevantBinaryPairInteraction (offsetBaseAsymmetricPairUtility w) j i := by
      refine ⟨false, true, ?_⟩
      have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
        (pw := offsetBaseAsymmetricPairPairwise w) (a := false) (b := true) hjilt
      have h' : actionGapCrossDifference (offsetBaseAsymmetricPairUtility w) false true j i = 1 := by
        simpa [offsetBaseAsymmetricPairPairwise, completeInteracts, completePairIndicator_cross, hjilt.ne]
          using h
      rw [h']
      norm_num
    exact HasDecisionRelevantBinaryPairInteraction_symm (u := offsetBaseAsymmetricPairUtility w) j i hrev

theorem offsetBaseAsymmetricPair_decisionRelevantGraph_eq_top (w : ℕ) :
    decisionRelevantInteractionGraph (offsetBaseAsymmetricPairUtility w) = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [decisionRelevantInteractionGraph, InteractionGraph]
  · simp [decisionRelevantInteractionGraph, InteractionGraph, h,
      offsetBaseAsymmetricPair_hasDecisionRelevantInteraction w h]

theorem offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top (w : ℕ) :
    decisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w) = ⊤ := by
  simpa [offsetCollapsedAsymmetricPairUtility] using
    (decisionRelevantInteractionGraph_addActionOffset
      (u := offsetBaseAsymmetricPairUtility w) (c := fun a => if a then 0 else 1)).trans
      (offsetBaseAsymmetricPair_decisionRelevantGraph_eq_top w)

theorem offsetCollapsedAsymmetricPair_not_symmetric (w : ℕ) :
    ¬ SymmetricUtility (offsetCollapsedAsymmetricPairDimensional w) := by
  intro hsym
  let n := w + 2
  let i0 : Fin n := 0
  let i1 : Fin n := 1
  let σ : CoordinatePermutation n := Equiv.swap i0 i1
  let s : DimensionalStateSpace 2 n := ⟨fun i => if i = i0 then 1 else 0⟩
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
          if completeInteracts i j ∧ i < j then
            completePairIndicator ((s.permute σ).state i) ((s.permute σ).state j)
          else 0) = 0 := by
    rw [hperm]
    simpa [i1, completeInteracts] using completePairSum_single_one n i1
  have hs : offsetCollapsedAsymmetricPairDimensional w false s = 2 := by
    unfold offsetCollapsedAsymmetricPairDimensional
    have hsum :
        (∑ i : Fin n,
          ∑ j : Fin n,
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0) = 0 := by
      simpa [s, i0, completeInteracts] using completePairSum_single_one n i0
    simpa [offsetCollapsedAsymmetricPairDimensional, s, i0, i1, n, hsum]
  have hsp : offsetCollapsedAsymmetricPairDimensional w false (s.permute σ) = 1 := by
    have hsumPerm' :
        (∑ i : Fin n,
          ∑ j : Fin n,
            if completeInteracts i j ∧ i < j then
              completePairIndicator (if i = i1 then 1 else 0) (if j = i1 then 1 else 0)
            else 0) = 0 := by
      simpa [hperm] using hsumPerm
    simpa [offsetCollapsedAsymmetricPairDimensional, hperm, i0, i1, n, hsumPerm']
  have hEq := hsym σ false s
  rw [hs, hsp] at hEq
  omega

theorem offsetBaseAsymmetricPair_optimizer_not_constant (w : ℕ) :
    ∃ s t : DimensionalStateSpace 2 (w + 2),
      (offsetBaseAsymmetricPairProblem w).Opt s ≠ (offsetBaseAsymmetricPairProblem w).Opt t := by
  let s0 : DimensionalStateSpace 2 (w + 2) := ⟨fun _ => 0⟩
  let s1 : DimensionalStateSpace 2 (w + 2) := ⟨fun i => if i = 0 then 1 else 0⟩
  refine ⟨s0, s1, ?_⟩
  have h0 : true ∈ (offsetBaseAsymmetricPairProblem w).Opt s0 := by
    simp [DecisionProblem.Opt, DecisionProblem.isOptimal, offsetBaseAsymmetricPairProblem,
      offsetBaseAsymmetricPairProblemUtility, s0, completeInteracts, completePairIndicatorReal]
  have h1 : true ∉ (offsetBaseAsymmetricPairProblem w).Opt s1 := by
    intro htrue
    have h := htrue false
    have hsum :
        (∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s1.state i) (s1.state j) else 0) = 0 := by
      simpa [s1, completeInteracts] using completePairSum_single_one_real (w + 2) (0 : Fin (w + 2))
    simp [DecisionProblem.Opt, DecisionProblem.isOptimal, offsetBaseAsymmetricPairProblem,
      offsetBaseAsymmetricPairProblemUtility, s1, hsum] at h
    linarith
  intro hEq
  exact h1 (hEq ▸ h0)

theorem offsetCollapsedAsymmetricPair_false_optimal (w : ℕ)
    (s : DimensionalStateSpace 2 (w + 2)) :
    (offsetCollapsedAsymmetricPairProblem w).isOptimal false s := by
  have hsum : (0 : ℝ) ≤ ∑ i : Fin (w + 2),
      ∑ j : Fin (w + 2),
        if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 := by
    refine Finset.sum_nonneg ?_
    intro i hi
    refine Finset.sum_nonneg ?_
    intro j hj
    by_cases h : completeInteracts i j ∧ i < j
    · by_cases hs : s.state i = 1 ∧ s.state j = 1
      · simp [h, hs, completePairIndicatorReal]
      · simp [h, hs, completePairIndicatorReal]
    · simp [h]
  have hunary : (0 : ℝ) ≤ if s.state 0 = 1 then 1 else 0 := by positivity
  have hpos : (0 : ℝ) < offsetCollapsedAsymmetricPairProblemUtility w false s := by
    simp [offsetCollapsedAsymmetricPairProblemUtility]
    linarith
  intro a'
  cases a'
  · simp [DecisionProblem.isOptimal]
  · simp [DecisionProblem.isOptimal, offsetCollapsedAsymmetricPairProblem,
      offsetCollapsedAsymmetricPairProblemUtility]
    linarith

theorem offsetCollapsedAsymmetricPair_true_not_optimal (w : ℕ)
    (s : DimensionalStateSpace 2 (w + 2)) :
    true ∉ (offsetCollapsedAsymmetricPairProblem w).Opt s := by
  have hsum : (0 : ℝ) ≤ ∑ i : Fin (w + 2),
      ∑ j : Fin (w + 2),
        if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 := by
    refine Finset.sum_nonneg ?_
    intro i hi
    refine Finset.sum_nonneg ?_
    intro j hj
    by_cases h : completeInteracts i j ∧ i < j
    · by_cases hs : s.state i = 1 ∧ s.state j = 1
      · simp [h, hs, completePairIndicatorReal]
      · simp [h, hs, completePairIndicatorReal]
    · simp [h]
  have hunary : (0 : ℝ) ≤ if s.state 0 = 1 then 1 else 0 := by positivity
  have hpos : (0 : ℝ) < offsetCollapsedAsymmetricPairProblemUtility w false s := by
    simp [offsetCollapsedAsymmetricPairProblemUtility]
    linarith
  intro htrue
  have h := htrue false
  simp [DecisionProblem.Opt, DecisionProblem.isOptimal, offsetCollapsedAsymmetricPairProblem,
    offsetCollapsedAsymmetricPairProblemUtility] at h
  linarith

theorem offsetCollapsedAsymmetricPair_opt (w : ℕ) (s : DimensionalStateSpace 2 (w + 2)) :
    (offsetCollapsedAsymmetricPairProblem w).Opt s = {false} := by
  ext a
  cases a
  · simp [DecisionProblem.Opt, offsetCollapsedAsymmetricPair_false_optimal]
  · simpa [DecisionProblem.Opt] using offsetCollapsedAsymmetricPair_true_not_optimal w s

theorem offsetCollapsedAsymmetricPair_all_sufficient (w : ℕ) (I : Finset (Fin (w + 2))) :
    (offsetCollapsedAsymmetricPairProblem w).isSufficient I := by
  intro s s' _
  rw [offsetCollapsedAsymmetricPair_opt, offsetCollapsedAsymmetricPair_opt]

theorem action_offset_can_force_constant_optimizer (w : ℕ) :
    decisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w) =
      decisionRelevantInteractionGraph (offsetBaseAsymmetricPairUtility w) ∧
    (∃ s t : DimensionalStateSpace 2 (w + 2),
      (offsetBaseAsymmetricPairProblem w).Opt s ≠ (offsetBaseAsymmetricPairProblem w).Opt t) ∧
    (∀ s : DimensionalStateSpace 2 (w + 2),
      (offsetCollapsedAsymmetricPairProblem w).Opt s = {false}) := by
  refine ⟨decisionRelevantInteractionGraph_addActionOffset
    (u := offsetBaseAsymmetricPairUtility w) (c := fun a => if a then 0 else 1),
    offsetBaseAsymmetricPair_optimizer_not_constant w, ?_⟩
  intro s
  exact offsetCollapsedAsymmetricPair_opt w s

theorem offsetBaseCollapsed_actionOffsetEquivalent (w : ℕ) :
    ActionOffsetEquivalent (offsetBaseAsymmetricPairUtility w) (offsetCollapsedAsymmetricPairUtility w) := by
  refine ⟨fun a => if a then 0 else 1, ?_⟩
  rfl

theorem block6_action_offset_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (offsetCollapsedAsymmetricPairUtility w)) ∧
    ¬ SymmetricUtility (offsetCollapsedAsymmetricPairDimensional w) ∧
    decisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w) = ⊤ ∧
    ¬ realTreewidth_le (decisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w)) w ∧
    ∀ I : Finset (Fin (w + 2)), (offsetCollapsedAsymmetricPairProblem w).isSufficient I := by
  refine ⟨⟨offsetCollapsedAsymmetricPairPairwise w⟩,
    offsetCollapsedAsymmetricPair_not_symmetric w, ?_, ?_, ?_⟩
  · exact offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top w
  · simpa [offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top w] using
      completeGraph_not_realTreewidth_le w
  · intro I
    exact offsetCollapsedAsymmetricPair_all_sufficient w I

theorem block6_offset_normalized_obstruction (w : ℕ) :
    ActionOffsetEquivalent (offsetBaseAsymmetricPairUtility w) (offsetCollapsedAsymmetricPairUtility w) ∧
    offsetNormalizedDecisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w) = ⊤ ∧
    ¬ SymmetricUtility (offsetCollapsedAsymmetricPairDimensional w) ∧
    ¬ realTreewidth_le
      (offsetNormalizedDecisionRelevantInteractionGraph (offsetCollapsedAsymmetricPairUtility w)) w ∧
    ∀ I : Finset (Fin (w + 2)), (offsetCollapsedAsymmetricPairProblem w).isSufficient I := by
  refine ⟨offsetBaseCollapsed_actionOffsetEquivalent w, ?_,
    offsetCollapsedAsymmetricPair_not_symmetric w, ?_, ?_⟩
  · simpa [offsetNormalizedDecisionRelevantInteractionGraph,
      offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top w]
  · simpa [offsetNormalizedDecisionRelevantInteractionGraph,
      offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top w] using
      completeGraph_not_realTreewidth_le w
  · intro I
    exact offsetCollapsedAsymmetricPair_all_sufficient w I

noncomputable def neverOptimalGhostUtility (w : ℕ) : Fin 3 → (Fin (w + 2) → Fin 2) → ℤ :=
  fun a s =>
    (∑ i : Fin (w + 2),
      if a = 0 ∧ i = 0 then
        if s i = 1 then 1 else 0
      else if a = 1 ∧ i = 0 then
        if s i = 0 then 1 else 0
      else if a = 2 ∧ i = 0 then
        -1
      else 0) +
    ∑ i : Fin (w + 2),
      ∑ j : Fin (w + 2),
        if completeInteracts i j ∧ i < j then
          if a = 2 then -completePairIndicator (s i) (s j) else 0
        else 0

noncomputable def neverOptimalGhostPairwise (w : ℕ) : PairwiseUtility (neverOptimalGhostUtility w) where
  unary i a x :=
    if a = 0 ∧ i = 0 then
      if x = 1 then 1 else 0
    else if a = 1 ∧ i = 0 then
      if x = 0 then 1 else 0
    else if a = 2 ∧ i = 0 then
      -1
    else 0
  binary _ _ a x y := if a = 2 then -completePairIndicator x y else 0
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    rfl

noncomputable def neverOptimalGhostDimensional (w : ℕ) : Fin 3 → DimensionalStateSpace 2 (w + 2) → ℤ :=
  fun a s =>
    if a = 0 then
      if s.state 0 = 1 then 1 else 0
    else if a = 1 then
      if s.state 0 = 0 then 1 else 0
    else
      -1 -
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0

noncomputable def neverOptimalGhostProblemUtility (w : ℕ) :
    Fin 3 → DimensionalStateSpace 2 (w + 2) → ℝ :=
  fun a s =>
    if a = 0 then
      if s.state 0 = 1 then 1 else 0
    else if a = 1 then
      if s.state 0 = 0 then 1 else 0
    else
      -1 -
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0

noncomputable def neverOptimalGhostProblem (w : ℕ) :
    DecisionProblem (Fin 3) (DimensionalStateSpace 2 (w + 2)) where
  utility := neverOptimalGhostProblemUtility w

theorem neg_completePairIndicator_cross :
    binaryCrossDifference (fun x y => -completePairIndicator x y) = -1 := by
  unfold binaryCrossDifference completePairIndicator
  norm_num

theorem neverOptimalGhost_hasDecisionRelevantInteraction (w : ℕ) {i j : Fin (w + 2)}
    (hij : i ≠ j) :
    HasDecisionRelevantBinaryPairInteraction (neverOptimalGhostUtility w) i j := by
  obtain hijlt | hjilt := lt_or_gt_of_ne hij
  · refine ⟨2, 0, ?_⟩
    have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
      (pw := neverOptimalGhostPairwise w) (a := (2 : Fin 3)) (b := (0 : Fin 3)) hijlt
    have h' : actionGapCrossDifference (neverOptimalGhostUtility w) 2 0 i j = -1 := by
      simpa [neverOptimalGhostPairwise, completeInteracts, neg_completePairIndicator_cross, hijlt.ne]
        using h
    rw [h']
    norm_num
  · have hrev : HasDecisionRelevantBinaryPairInteraction (neverOptimalGhostUtility w) j i := by
      refine ⟨2, 0, ?_⟩
      have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
        (pw := neverOptimalGhostPairwise w) (a := (2 : Fin 3)) (b := (0 : Fin 3)) hjilt
      have h' : actionGapCrossDifference (neverOptimalGhostUtility w) 2 0 j i = -1 := by
        simpa [neverOptimalGhostPairwise, completeInteracts, neg_completePairIndicator_cross, hjilt.ne]
          using h
      rw [h']
      norm_num
    exact HasDecisionRelevantBinaryPairInteraction_symm (u := neverOptimalGhostUtility w) j i hrev

theorem neverOptimalGhost_decisionRelevantGraph_eq_top (w : ℕ) :
    decisionRelevantInteractionGraph (neverOptimalGhostUtility w) = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [decisionRelevantInteractionGraph, InteractionGraph]
  · simp [decisionRelevantInteractionGraph, InteractionGraph, h,
      neverOptimalGhost_hasDecisionRelevantInteraction w h]

theorem neverOptimalGhost_gap01_zero (w : ℕ) (i j : Fin (w + 2)) :
    actionGapCrossDifference (neverOptimalGhostUtility w) 0 1 i j = 0 := by
  unfold actionGapCrossDifference pairCrossDifference
  simpa [neverOptimalGhostUtility] using
    (pairState_cross_unary_zero (i := i) (j := j) (k := (0 : Fin (w + 2)))
      (f := fun x => (if x = 1 then 1 else 0) - (if x = 0 then 1 else 0)))

theorem completePairIndicatorReal_sum_nonneg {n : ℕ}
    (s : DimensionalStateSpace 2 n) :
    (0 : ℝ) ≤ ∑ i : Fin n,
      ∑ j : Fin n,
        if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 := by
  refine Finset.sum_nonneg ?_
  intro i hi
  refine Finset.sum_nonneg ?_
  intro j hj
  by_cases h : completeInteracts i j ∧ i < j
  · by_cases hs : s.state i = 1 ∧ s.state j = 1
    · simp [h, hs, completePairIndicatorReal]
    · simp [h, hs, completePairIndicatorReal]
  · simp [h]

theorem neverOptimalGhost_opt_if_one (w : ℕ) (s : DimensionalStateSpace 2 (w + 2))
    (hs : s.state 0 = 1) :
    (neverOptimalGhostProblem w).Opt s = {0} := by
  ext a
  fin_cases a
  · simp [DecisionProblem.Opt, DecisionProblem.isOptimal, neverOptimalGhostProblem,
      neverOptimalGhostProblemUtility, hs]
    intro a'
    fin_cases a' <;> simp [neverOptimalGhostProblemUtility, hs]
    have hsum := completePairIndicatorReal_sum_nonneg s
    linarith
  · constructor
    · intro h
      have hlt := h 0
      simp [DecisionProblem.isOptimal, neverOptimalGhostProblem, neverOptimalGhostProblemUtility,
        hs] at hlt
      exfalso
      linarith
    · intro h
      cases h
  · constructor
    · intro h
      have hlt := h 0
      have hsum := completePairIndicatorReal_sum_nonneg s
      simp [DecisionProblem.isOptimal, neverOptimalGhostProblem, neverOptimalGhostProblemUtility,
        hs] at hlt
      exfalso
      linarith
    · intro h
      cases h

theorem neverOptimalGhost_opt_if_zero (w : ℕ) (s : DimensionalStateSpace 2 (w + 2))
    (hs : s.state 0 = 0) :
    (neverOptimalGhostProblem w).Opt s = {1} := by
  ext a
  fin_cases a
  · constructor
    · intro h
      have hlt := h 1
      simp [DecisionProblem.Opt, DecisionProblem.isOptimal, neverOptimalGhostProblem,
        neverOptimalGhostProblemUtility, hs] at hlt
      exfalso
      linarith
    · intro h
      cases h
  · simp [DecisionProblem.Opt, DecisionProblem.isOptimal, neverOptimalGhostProblem,
      neverOptimalGhostProblemUtility, hs]
    intro a'
    fin_cases a' <;> simp [neverOptimalGhostProblemUtility, hs]
    have hsum := completePairIndicatorReal_sum_nonneg s
    linarith
  · constructor
    · intro h
      have hlt := h 1
      have hsum := completePairIndicatorReal_sum_nonneg s
      simp [DecisionProblem.isOptimal, neverOptimalGhostProblem, neverOptimalGhostProblemUtility,
        hs] at hlt
      exfalso
      linarith
    · intro h
      cases h

theorem neverOptimalGhost_supported_zero (w : ℕ) :
    OptimizerSupported (neverOptimalGhostProblem w) (0 : Fin 3) := by
  refine ⟨⟨fun i => if i = 0 then 1 else 0⟩, ?_⟩
  rw [neverOptimalGhost_opt_if_one (w := w) (s := ⟨fun i => if i = 0 then 1 else 0⟩)]
  · simp
  · simp

theorem neverOptimalGhost_supported_one (w : ℕ) :
    OptimizerSupported (neverOptimalGhostProblem w) (1 : Fin 3) := by
  refine ⟨⟨fun _ => 0⟩, ?_⟩
  rw [neverOptimalGhost_opt_if_zero (w := w) (s := ⟨fun _ => 0⟩)]
  · simp
  · simp

theorem neverOptimalGhost_not_supported_two (w : ℕ) :
    ¬ OptimizerSupported (neverOptimalGhostProblem w) (2 : Fin 3) := by
  intro h
  rcases h with ⟨s, hsupp⟩
  by_cases hs : s.state 0 = 1
  · rw [neverOptimalGhost_opt_if_one (w := w) (s := s) hs] at hsupp
    simp at hsupp
  · have hs0 : s.state 0 = 0 := by
      apply Fin.ext
      have hne : (s.state 0).1 ≠ 1 := by
        intro h1
        apply hs
        exact Fin.ext h1
      omega
    rw [neverOptimalGhost_opt_if_zero (w := w) (s := s) hs0] at hsupp
    simp at hsupp

theorem neverOptimalGhost_supported_iff (w : ℕ) (a : Fin 3) :
    OptimizerSupported (neverOptimalGhostProblem w) a ↔ a = 0 ∨ a = 1 := by
  fin_cases a
  · constructor
    · intro _
      simp
    · intro _
      exact neverOptimalGhost_supported_zero w
  · constructor
    · intro _
      simp
    · intro _
      exact neverOptimalGhost_supported_one w
  · constructor
    · intro h
      exact False.elim (neverOptimalGhost_not_supported_two w h)
    · intro h
      simp at h

theorem neverOptimalGhost_supportedGraph_eq_bot (w : ℕ) :
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (neverOptimalGhostProblem w)) (neverOptimalGhostUtility w) = ⊥ := by
  have hnone : ∀ i j : Fin (w + 2),
      ¬ SupportedDecisionRelevantBinaryPairInteraction
        (OptimizerSupported (neverOptimalGhostProblem w)) (neverOptimalGhostUtility w) i j := by
    intro i j hed
    rcases hed with ⟨a, b, ha, hb, hcross⟩
    have ha' := (neverOptimalGhost_supported_iff w a).1 ha
    have hb' := (neverOptimalGhost_supported_iff w b).1 hb
    rcases ha' with rfl | rfl <;> rcases hb' with rfl | rfl
    · exact hcross (actionGapCrossDifference_self_action (neverOptimalGhostUtility w) 0 i j)
    · exact hcross (neverOptimalGhost_gap01_zero w i j)
    · rw [actionGapCrossDifference_swap_actions] at hcross
      simp [neverOptimalGhost_gap01_zero w i j] at hcross
    · exact hcross (actionGapCrossDifference_self_action (neverOptimalGhostUtility w) 1 i j)
  ext i j
  by_cases h : i = j
  · subst h
    simp [supportedDecisionRelevantInteractionGraph, InteractionGraph]
  · simp [supportedDecisionRelevantInteractionGraph, InteractionGraph, h, hnone i j]

theorem offsetCollapsed_supported_iff (w : ℕ) (a : Bool) :
    OptimizerSupported (offsetCollapsedAsymmetricPairProblem w) a ↔ a = false := by
  cases a
  · constructor
    · intro _
      rfl
    · intro _
      refine ⟨⟨fun _ => 0⟩, ?_⟩
      simp [DecisionProblem.Opt, offsetCollapsedAsymmetricPair_false_optimal]
  · constructor
    · intro h
      rcases h with ⟨s, hs⟩
      exact False.elim (offsetCollapsedAsymmetricPair_true_not_optimal w s hs)
    · intro h
      cases h

theorem offsetCollapsed_supportedGraph_eq_bot (w : ℕ) :
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (offsetCollapsedAsymmetricPairProblem w))
      (offsetCollapsedAsymmetricPairUtility w) = ⊥ := by
  have hnone : ∀ i j : Fin (w + 2),
      ¬ SupportedDecisionRelevantBinaryPairInteraction
        (OptimizerSupported (offsetCollapsedAsymmetricPairProblem w))
        (offsetCollapsedAsymmetricPairUtility w) i j := by
    intro i j hed
    rcases hed with ⟨a, b, ha, hb, hcross⟩
    have ha' := (offsetCollapsed_supported_iff w a).1 ha
    have hb' := (offsetCollapsed_supported_iff w b).1 hb
    subst a
    subst b
    exact hcross (actionGapCrossDifference_self_action (offsetCollapsedAsymmetricPairUtility w) false i j)
  ext i j
  by_cases h : i = j
  · subst h
    simp [supportedDecisionRelevantInteractionGraph, InteractionGraph]
  · simp [supportedDecisionRelevantInteractionGraph, InteractionGraph, h, hnone i j]

theorem support_filtering_removes_known_block6_obstructions (w : ℕ) :
    supportedDecisionRelevantInteractionGraph
        (OptimizerSupported (offsetCollapsedAsymmetricPairProblem w))
        (offsetCollapsedAsymmetricPairUtility w) = ⊥ ∧
      supportedDecisionRelevantInteractionGraph
        (OptimizerSupported (neverOptimalGhostProblem w))
        (neverOptimalGhostUtility w) = ⊥ := by
  exact ⟨offsetCollapsed_supportedGraph_eq_bot w, neverOptimalGhost_supportedGraph_eq_bot w⟩

theorem neverOptimalGhost_zero_sufficient (w : ℕ) :
    (neverOptimalGhostProblem w).isSufficient ({0} : Finset (Fin (w + 2))) := by
  intro s s' hagree
  have hcoord : s.state 0 = s'.state 0 := by simpa using hagree 0 (by simp)
  by_cases hs : s.state 0 = 1
  · have hs' : s'.state 0 = 1 := by rw [← hcoord]; exact hs
    rw [neverOptimalGhost_opt_if_one (w := w) (s := s) hs,
      neverOptimalGhost_opt_if_one (w := w) (s := s') hs']
  · have hs0 : s.state 0 = 0 := by
      apply Fin.ext
      have hne : (s.state 0).1 ≠ 1 := by
        intro h1
        apply hs
        exact Fin.ext h1
      omega
    have hs' : s'.state 0 = 0 := by
      calc
        s'.state 0 = s.state 0 := by simpa using hcoord.symm
        _ = 0 := hs0
    rw [neverOptimalGhost_opt_if_zero (w := w) (s := s) hs0,
      neverOptimalGhost_opt_if_zero (w := w) (s := s') hs']

theorem block6_ghost_action_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (neverOptimalGhostUtility w)) ∧
    decisionRelevantInteractionGraph (neverOptimalGhostUtility w) = ⊤ ∧
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (neverOptimalGhostProblem w)) (neverOptimalGhostUtility w) = ⊥ ∧
    (neverOptimalGhostProblem w).isSufficient ({0} : Finset (Fin (w + 2))) := by
  refine ⟨⟨neverOptimalGhostPairwise w⟩,
    neverOptimalGhost_decisionRelevantGraph_eq_top w,
    neverOptimalGhost_supportedGraph_eq_bot w,
    neverOptimalGhost_zero_sufficient w⟩

def marginMaskingConstant (w : ℕ) : ℤ := ((w + 2) : ℤ) * ((w + 2) : ℤ) + 1

def marginMaskingConstantReal (w : ℕ) : ℝ := ((w + 2 : ℕ) : ℝ) * ((w + 2 : ℕ) : ℝ) + 1

noncomputable def marginMaskingUtility (w : ℕ) : Bool → (Fin (w + 2) → Fin 2) → ℤ :=
  fun a s =>
    (∑ i : Fin (w + 2),
      if a = false ∧ i = 0 then
        if s i = 0 then marginMaskingConstant w else 0
      else if a = true ∧ i = 0 then
        if s i = 1 then marginMaskingConstant w else 0
      else 0) +
    ∑ i : Fin (w + 2),
      ∑ j : Fin (w + 2),
        if completeInteracts i j ∧ i < j then
          if a = false then completePairIndicator (s i) (s j) else 0
        else 0

noncomputable def marginMaskingPairwise (w : ℕ) : PairwiseUtility (marginMaskingUtility w) where
  unary i a x :=
    if a = false ∧ i = 0 then
      if x = 0 then marginMaskingConstant w else 0
    else if a = true ∧ i = 0 then
      if x = 1 then marginMaskingConstant w else 0
    else 0
  binary _ _ a x y := if a = false then completePairIndicator x y else 0
  interacts := @completeInteracts (w + 2)
  interacts_symm := @completeInteracts_symm (w + 2)
  decomp := by
    intro a s
    rfl

noncomputable def marginMaskingDimensional (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 2) → ℤ :=
  fun a s =>
    if a = false then
      (if s.state 0 = 0 then marginMaskingConstant w else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0
    else
      if s.state 0 = 1 then marginMaskingConstant w else 0

noncomputable def marginMaskingProblemUtility (w : ℕ) :
    Bool → DimensionalStateSpace 2 (w + 2) → ℝ :=
  fun a s =>
    if a = false then
      (if s.state 0 = 0 then marginMaskingConstantReal w else 0) +
        ∑ i : Fin (w + 2),
          ∑ j : Fin (w + 2),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0
    else
      if s.state 0 = 1 then marginMaskingConstantReal w else 0

noncomputable def marginMaskingProblem (w : ℕ) :
    DecisionProblem Bool (DimensionalStateSpace 2 (w + 2)) where
  utility := marginMaskingProblemUtility w

theorem completePairIndicatorReal_sum_le_square {n : ℕ}
    (s : DimensionalStateSpace 2 n) :
    (∑ i : Fin n,
      ∑ j : Fin n,
        if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0)
      ≤ (n : ℝ) * (n : ℝ) := by
  calc
    (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0)
      ≤ ∑ i : Fin n, ∑ j : Fin n, (1 : ℝ) := by
          refine Finset.sum_le_sum ?_
          intro i hi
          refine Finset.sum_le_sum ?_
          intro j hj
          by_cases h : completeInteracts i j ∧ i < j
          · by_cases hs : s.state i = 1 ∧ s.state j = 1
            · simp [h, hs, completePairIndicatorReal]
            · simp [h, hs, completePairIndicatorReal]
          · simp [h]
    _ = (n : ℝ) * (n : ℝ) := by simp

theorem marginMasking_opt_if_zero (w : ℕ) (s : DimensionalStateSpace 2 (w + 2))
    (hs : s.state 0 = 0) :
    (marginMaskingProblem w).Opt s = {false} := by
  have hsum := completePairIndicatorReal_sum_nonneg s
  have hub := completePairIndicatorReal_sum_le_square s
  have hMnonneg : 0 ≤ marginMaskingConstantReal w := by
    unfold marginMaskingConstantReal
    positivity
  ext a
  cases a
  · constructor
    · intro _
      simp
    · intro _
      intro a'
      cases a'
      · simp [DecisionProblem.isOptimal, marginMaskingProblem, marginMaskingProblemUtility, hs]
      · simp [DecisionProblem.isOptimal, marginMaskingProblem, marginMaskingProblemUtility, hs,
          marginMaskingConstantReal]
        have hcomp : 0 ≤ marginMaskingConstantReal w +
            ∑ i : Fin (w + 2),
              ∑ j : Fin (w + 2),
                if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 :=
          add_nonneg hMnonneg hsum
        simpa [marginMaskingConstantReal] using hcomp
  · constructor
    · intro h
      have hlt := h false
      simp [DecisionProblem.Opt, DecisionProblem.isOptimal, marginMaskingProblem,
        marginMaskingProblemUtility, hs, marginMaskingConstantReal] at hlt
      nlinarith
    · intro h
      cases h

theorem marginMasking_opt_if_one (w : ℕ) (s : DimensionalStateSpace 2 (w + 2))
    (hs : s.state 0 = 1) :
    (marginMaskingProblem w).Opt s = {true} := by
  have hsum := completePairIndicatorReal_sum_nonneg s
  have hub := completePairIndicatorReal_sum_le_square s
  have hMbound : ((w + 2 : ℕ) : ℝ) * ((w + 2 : ℕ) : ℝ) < marginMaskingConstantReal w := by
    unfold marginMaskingConstantReal
    nlinarith
  ext a
  cases a
  · constructor
    · intro h
      have hlt := h true
      simp [DecisionProblem.Opt, DecisionProblem.isOptimal, marginMaskingProblem,
        marginMaskingProblemUtility, hs, marginMaskingConstantReal] at hlt
      have hle : marginMaskingConstantReal w ≤
          ∑ i : Fin (w + 2),
            ∑ j : Fin (w + 2),
              if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 := by
        simpa [marginMaskingConstantReal] using hlt
      have hcontr : marginMaskingConstantReal w ≤ ((w + 2 : ℕ) : ℝ) * ((w + 2 : ℕ) : ℝ) :=
        le_trans hle hub
      nlinarith
    · intro h
      cases h
  · constructor
    · intro _
      simp
    · intro _
      intro a'
      cases a'
      · have hcomp :
          (∑ i : Fin (w + 2),
            ∑ j : Fin (w + 2),
              if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0)
            ≤ marginMaskingConstantReal w :=
          le_trans hub (le_of_lt hMbound)
        simpa [DecisionProblem.isOptimal, marginMaskingProblem, marginMaskingProblemUtility, hs,
          marginMaskingConstantReal] using hcomp
      · simp [DecisionProblem.isOptimal, marginMaskingProblem, marginMaskingProblemUtility, hs]

theorem marginMasking_supported_false (w : ℕ) :
    OptimizerSupported (marginMaskingProblem w) false := by
  refine ⟨⟨fun _ => 0⟩, ?_⟩
  rw [marginMasking_opt_if_zero (w := w) (s := ⟨fun _ => 0⟩)]
  · simp
  · simp

theorem marginMasking_supported_true (w : ℕ) :
    OptimizerSupported (marginMaskingProblem w) true := by
  refine ⟨⟨fun i => if i = 0 then 1 else 0⟩, ?_⟩
  rw [marginMasking_opt_if_one (w := w) (s := ⟨fun i => if i = 0 then 1 else 0⟩)]
  · simp
  · simp

theorem marginMasking_hasSupportedInteraction (w : ℕ) {i j : Fin (w + 2)}
    (hij : i ≠ j) :
    SupportedDecisionRelevantBinaryPairInteraction
      (OptimizerSupported (marginMaskingProblem w)) (marginMaskingUtility w) i j := by
  obtain hijlt | hjilt := lt_or_gt_of_ne hij
  · refine ⟨false, true, marginMasking_supported_false w, marginMasking_supported_true w, ?_⟩
    have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
      (pw := marginMaskingPairwise w) (a := false) (b := true) hijlt
    have h' : actionGapCrossDifference (marginMaskingUtility w) false true i j = 1 := by
      simpa [marginMaskingPairwise, completeInteracts, completePairIndicator_cross, hijlt.ne] using h
    rw [h']
    norm_num
  · have hrev : SupportedDecisionRelevantBinaryPairInteraction
        (OptimizerSupported (marginMaskingProblem w)) (marginMaskingUtility w) j i := by
      refine ⟨false, true, marginMasking_supported_false w, marginMasking_supported_true w, ?_⟩
      have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
        (pw := marginMaskingPairwise w) (a := false) (b := true) hjilt
      have h' : actionGapCrossDifference (marginMaskingUtility w) false true j i = 1 := by
        simpa [marginMaskingPairwise, completeInteracts, completePairIndicator_cross, hjilt.ne] using h
      rw [h']
      norm_num
    exact SupportedDecisionRelevantBinaryPairInteraction_symm
      (OptimizerSupported (marginMaskingProblem w)) j i hrev

theorem marginMasking_supportedGraph_eq_top (w : ℕ) :
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (marginMaskingProblem w)) (marginMaskingUtility w) = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [supportedDecisionRelevantInteractionGraph, InteractionGraph]
  · simp [supportedDecisionRelevantInteractionGraph, InteractionGraph, h,
      marginMasking_hasSupportedInteraction w h]

theorem marginMasking_not_symmetric (w : ℕ) :
    ¬ SymmetricUtility (marginMaskingDimensional w) := by
  intro hsym
  let n := w + 2
  let i0 : Fin n := 0
  let i1 : Fin n := 1
  let σ : CoordinatePermutation n := Equiv.swap i0 i1
  let s : DimensionalStateSpace 2 n := ⟨fun i => if i = i1 then 1 else 0⟩
  have hperm : (s.permute σ).state = fun i => if i = i0 then 1 else 0 := by
    funext i
    by_cases hi0 : i = i0
    · subst hi0
      change s.state (σ.symm i0) = 1
      have hswap : σ.symm i0 = i1 := by simpa [σ] using (Equiv.swap_apply_left i0 i1)
      simp [s, hswap, i0, i1]
    · by_cases hi1 : i = i1
      · subst hi1
        change s.state (σ.symm i1) = 0
        have hswap : σ.symm i1 = i0 := by simpa [σ] using (Equiv.swap_apply_right i0 i1)
        simp [s, hswap, i0, i1]
      · have hswap : σ.symm i = i := by simpa [σ] using (Equiv.swap_apply_of_ne_of_ne hi0 hi1)
        simp [DimensionalStateSpace.permute, s, hswap, hi0, hi1]
  have hsum :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0) = 0 := by
    simpa [s, i1, completeInteracts] using completePairSum_single_one n i1
  have hsum' :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then
            completePairIndicator ((s.permute σ).state i) ((s.permute σ).state j) else 0) = 0 := by
    rw [hperm]
    simpa [i0, completeInteracts] using completePairSum_single_one n i0
  have hs0 : s.state 0 = 0 := by
    simp [s, i1]
  have hperm0 : (s.permute σ).state 0 = 1 := by
    simpa [hperm, i0]
  have hs : marginMaskingDimensional w false s = marginMaskingConstant w := by
    unfold marginMaskingDimensional
    rw [if_pos rfl, hs0, if_pos rfl, hsum]
    ring
  have hsp : marginMaskingDimensional w false (s.permute σ) = 0 := by
    unfold marginMaskingDimensional
    rw [if_pos rfl, hperm0, if_neg (by decide), hsum']
    ring
  have hEq := hsym σ false s
  rw [hs, hsp] at hEq
  have hpos : 0 < marginMaskingConstant w := by
    unfold marginMaskingConstant
    positivity
  exact ne_of_gt hpos hEq

theorem marginMasking_zero_sufficient (w : ℕ) :
    (marginMaskingProblem w).isSufficient ({0} : Finset (Fin (w + 2))) := by
  intro s s' hagree
  have hcoord : s.state 0 = s'.state 0 := by simpa using hagree 0 (by simp)
  by_cases hs : s.state 0 = 0
  · have hs' : s'.state 0 = 0 := by rw [← hcoord]; exact hs
    rw [marginMasking_opt_if_zero (w := w) (s := s) hs,
      marginMasking_opt_if_zero (w := w) (s := s') hs']
  · have hs1 : s.state 0 = 1 := by
      apply Fin.ext
      have hne : (s.state 0).1 ≠ 0 := by
        intro h0
        apply hs
        exact Fin.ext h0
      omega
    have hs' : s'.state 0 = 1 := by
      calc
        s'.state 0 = s.state 0 := by simpa using hcoord.symm
        _ = 1 := hs1
    rw [marginMasking_opt_if_one (w := w) (s := s) hs1,
      marginMasking_opt_if_one (w := w) (s := s') hs']

theorem block6_optimizer_supported_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (marginMaskingUtility w)) ∧
    ¬ SymmetricUtility (marginMaskingDimensional w) ∧
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (marginMaskingProblem w)) (marginMaskingUtility w) = ⊤ ∧
    ¬ realTreewidth_le
      (supportedDecisionRelevantInteractionGraph
        (OptimizerSupported (marginMaskingProblem w)) (marginMaskingUtility w)) w ∧
    (marginMaskingProblem w).isSufficient ({0} : Finset (Fin (w + 2))) := by
  refine ⟨⟨marginMaskingPairwise w⟩, marginMasking_not_symmetric w,
    marginMasking_supportedGraph_eq_top w, ?_, marginMasking_zero_sufficient w⟩
  simpa [marginMasking_supportedGraph_eq_top w] using completeGraph_not_realTreewidth_le w

def dominantPairConstant (w : ℕ) : ℤ := ((w + 3) : ℤ) * ((w + 3) : ℤ) + 1

def dominantPairConstantReal (w : ℕ) : ℝ := ((w + 3 : ℕ) : ℝ) * ((w + 3 : ℕ) : ℝ) + 1

def signedEqualityIndicator (x y : Fin 2) : ℤ := if x = y then 1 else -1

def signedEqualityIndicatorReal (x y : Fin 2) : ℝ := if x = y then 1 else -1

theorem signedEqualityIndicator_cross : binaryCrossDifference signedEqualityIndicator = 4 := by
  simp [binaryCrossDifference, signedEqualityIndicator]

def dominantPairBinary (w : ℕ) (i j : Fin (w + 3)) (x y : Fin 2) : ℤ :=
  if i = 0 ∧ j = 1 then
    dominantPairConstant w * signedEqualityIndicator x y + completePairIndicator x y
  else
    completePairIndicator x y

theorem dominantPairBinary_special_cross (w : ℕ) :
    binaryCrossDifference
        (fun x y => dominantPairConstant w * signedEqualityIndicator x y + completePairIndicator x y) =
      4 * dominantPairConstant w + 1 := by
  simp [binaryCrossDifference, signedEqualityIndicator, completePairIndicator]
  ring

noncomputable def dominantPairUtility (w : ℕ) : Bool → (Fin (w + 3) → Fin 2) → ℤ
  | false, s =>
      ∑ i : Fin (w + 3),
        ∑ j : Fin (w + 3),
          if completeInteracts i j ∧ i < j then dominantPairBinary w i j (s i) (s j) else 0
  | true, _ => 0

noncomputable def dominantPairPairwise (w : ℕ) : PairwiseUtility (dominantPairUtility w) where
  unary _ _ _ := 0
  binary i j a x y := if a = false then dominantPairBinary w i j x y else 0
  interacts := @completeInteracts (w + 3)
  interacts_symm := @completeInteracts_symm (w + 3)
  decomp := by
    intro a s
    cases a <;> simp [dominantPairUtility, dominantPairBinary]

noncomputable def dominantPairDimensional (w : ℕ) : Bool → DimensionalStateSpace 2 (w + 3) → ℤ :=
  fun a s =>
    if a = false then
      dominantPairConstant w * signedEqualityIndicator (s.state 0) (s.state 1) +
        ∑ i : Fin (w + 3),
          ∑ j : Fin (w + 3),
            if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0
    else
      0

noncomputable def dominantPairProblemUtility (w : ℕ) :
    Bool → DimensionalStateSpace 2 (w + 3) → ℝ :=
  fun a s =>
    if a = false then
      dominantPairConstantReal w * signedEqualityIndicatorReal (s.state 0) (s.state 1) +
        ∑ i : Fin (w + 3),
          ∑ j : Fin (w + 3),
            if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0
    else
      0

noncomputable def dominantPairProblem (w : ℕ) :
    DecisionProblem Bool (DimensionalStateSpace 2 (w + 3)) where
  utility := dominantPairProblemUtility w

theorem dominantPair_marginBounded (w : ℕ) :
    MarginBounded (dominantPairPairwise w) := by
  apply marginBounded_of_unary_zero (pw := dominantPairPairwise w)
  intro i a x
  simp [dominantPairPairwise]

theorem dominantPair_opt_if_eq (w : ℕ) (s : DimensionalStateSpace 2 (w + 3))
    (hs : s.state 0 = s.state 1) :
    (dominantPairProblem w).Opt s = {false} := by
  have hsum := completePairIndicatorReal_sum_nonneg s
  have hsSign : signedEqualityIndicatorReal (s.state 0) (s.state 1) = 1 := by
    simp [signedEqualityIndicatorReal, hs]
  have hpos : (0 : ℝ) < dominantPairConstantReal w := by
    unfold dominantPairConstantReal
    positivity
  ext a
  cases a
  · constructor
    · intro _
      simp
    · intro _
      intro a'
      cases a'
      · simp [DecisionProblem.isOptimal]
      · have hcomp : (0 : ℝ) ≤ dominantPairConstantReal w +
            ∑ i : Fin (w + 3),
              ∑ j : Fin (w + 3),
                if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0 := by
          linarith
        simpa [DecisionProblem.isOptimal, dominantPairProblem, dominantPairProblemUtility, hsSign]
          using hcomp
  · constructor
    · intro h
      have hlt := h false
      simp [DecisionProblem.Opt, DecisionProblem.isOptimal, dominantPairProblem,
        dominantPairProblemUtility, hsSign] at hlt
      linarith
    · intro h
      cases h

theorem dominantPair_opt_if_ne (w : ℕ) (s : DimensionalStateSpace 2 (w + 3))
    (hs : s.state 0 ≠ s.state 1) :
    (dominantPairProblem w).Opt s = {true} := by
  have hub := completePairIndicatorReal_sum_le_square s
  have hsSign : signedEqualityIndicatorReal (s.state 0) (s.state 1) = -1 := by
    simp [signedEqualityIndicatorReal, hs]
  have hMbound : ((w + 3 : ℕ) : ℝ) * ((w + 3 : ℕ) : ℝ) < dominantPairConstantReal w := by
    unfold dominantPairConstantReal
    nlinarith
  ext a
  cases a
  · constructor
    · intro h
      have hlt := h true
      simp [DecisionProblem.Opt, DecisionProblem.isOptimal, dominantPairProblem,
        dominantPairProblemUtility, hsSign] at hlt
      have hcontr : dominantPairConstantReal w ≤ ((w + 3 : ℕ) : ℝ) * ((w + 3 : ℕ) : ℝ) :=
        le_trans hlt hub
      nlinarith
    · intro h
      cases h
  · constructor
    · intro _
      simp
    · intro _
      intro a'
      cases a'
      · have hcomp :
            (∑ i : Fin (w + 3),
              ∑ j : Fin (w + 3),
                if completeInteracts i j ∧ i < j then completePairIndicatorReal (s.state i) (s.state j) else 0)
              ≤ dominantPairConstantReal w :=
          le_trans hub (le_of_lt hMbound)
        simpa [DecisionProblem.isOptimal, dominantPairProblem, dominantPairProblemUtility, hsSign]
          using hcomp
      · simp [DecisionProblem.isOptimal, dominantPairProblem, dominantPairProblemUtility]

theorem dominantPair_supported_false (w : ℕ) :
    OptimizerSupported (dominantPairProblem w) false := by
  refine ⟨⟨fun _ => 0⟩, ?_⟩
  rw [dominantPair_opt_if_eq (w := w) (s := ⟨fun _ => 0⟩)]
  · simp
  · simp

theorem dominantPair_supported_true (w : ℕ) :
    OptimizerSupported (dominantPairProblem w) true := by
  refine ⟨⟨fun i => if i = 1 then 1 else 0⟩, ?_⟩
  rw [dominantPair_opt_if_ne (w := w) (s := ⟨fun i => if i = 1 then 1 else 0⟩)]
  · simp
  · simp

theorem dominantPair_hasSupportedInteraction (w : ℕ) {i j : Fin (w + 3)}
    (hij : i ≠ j) :
    SupportedDecisionRelevantBinaryPairInteraction
      (OptimizerSupported (dominantPairProblem w)) (dominantPairUtility w) i j := by
  obtain hijlt | hjilt := lt_or_gt_of_ne hij
  · refine ⟨false, true, dominantPair_supported_false w, dominantPair_supported_true w, ?_⟩
    have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
      (pw := dominantPairPairwise w) (a := false) (b := true) hijlt
    by_cases h01 : i = 0 ∧ j = 1
    · have h' : actionGapCrossDifference (dominantPairUtility w) false true i j =
          4 * dominantPairConstant w + 1 := by
        simpa [dominantPairPairwise, dominantPairBinary, completeInteracts, h01,
          dominantPairBinary_special_cross, hijlt.ne] using h
      rw [h']
      unfold dominantPairConstant
      omega
    · have h' : actionGapCrossDifference (dominantPairUtility w) false true i j = 1 := by
        simpa [dominantPairPairwise, dominantPairBinary, completeInteracts, h01,
          completePairIndicator_cross, hijlt.ne] using h
      rw [h']
      norm_num
  · have hrev : SupportedDecisionRelevantBinaryPairInteraction
        (OptimizerSupported (dominantPairProblem w)) (dominantPairUtility w) j i := by
      refine ⟨false, true, dominantPair_supported_false w, dominantPair_supported_true w, ?_⟩
      have h := actionGapCrossDifference_eq_binaryCrossDifference_of_lt
        (pw := dominantPairPairwise w) (a := false) (b := true) hjilt
      by_cases h10 : j = 0 ∧ i = 1
      · have h' : actionGapCrossDifference (dominantPairUtility w) false true j i =
            4 * dominantPairConstant w + 1 := by
          simpa [dominantPairPairwise, dominantPairBinary, completeInteracts, h10,
            dominantPairBinary_special_cross, hjilt.ne] using h
        rw [h']
        unfold dominantPairConstant
        omega
      · have h' : actionGapCrossDifference (dominantPairUtility w) false true j i = 1 := by
          simpa [dominantPairPairwise, dominantPairBinary, completeInteracts, h10,
            completePairIndicator_cross, hjilt.ne] using h
        rw [h']
        norm_num
    exact SupportedDecisionRelevantBinaryPairInteraction_symm
      (OptimizerSupported (dominantPairProblem w)) j i hrev

theorem dominantPair_supportedGraph_eq_top (w : ℕ) :
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (dominantPairProblem w)) (dominantPairUtility w) = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [supportedDecisionRelevantInteractionGraph, InteractionGraph]
  · simp [supportedDecisionRelevantInteractionGraph, InteractionGraph, h,
      dominantPair_hasSupportedInteraction w h]

theorem dominantPair_not_symmetric (w : ℕ) :
    ¬ SymmetricUtility (dominantPairDimensional w) := by
  intro hsym
  let n := w + 3
  let i0 : Fin n := 0
  let i1 : Fin n := 1
  let i2 : Fin n := 2
  have hn1 : 1 < n := by
    omega
  have hn2 : 2 < n := by
    omega
  have hi12 : i1 ≠ i2 := by
    intro h
    have hval := congrArg Fin.val h
    have hi1val : (i1 : ℕ) = 1 := by
      simp [i1, Nat.mod_eq_of_lt hn1]
    have hi2val : (i2 : ℕ) = 2 := by
      simp [i2, Nat.mod_eq_of_lt hn2]
    rw [hi1val, hi2val] at hval
    omega
  let σ : CoordinatePermutation n := Equiv.swap i1 i2
  let s : DimensionalStateSpace 2 n := ⟨fun i => if i = i2 then 1 else 0⟩
  have hperm : (s.permute σ).state = fun i => if i = i1 then 1 else 0 := by
    funext i
    by_cases hi1 : i = i1
    · subst hi1
      change s.state (σ.symm i1) = 1
      have hswap : σ.symm i1 = i2 := by simpa [σ] using (Equiv.swap_apply_left i1 i2)
      simp [s, hswap, i1, i2, hi12]
    · by_cases hi2 : i = i2
      · subst hi2
        change s.state (σ.symm i2) = 0
        have hswap : σ.symm i2 = i1 := by simpa [σ] using (Equiv.swap_apply_right i1 i2)
        simp [s, hswap, i1, i2, hi12]
      · have hswap : σ.symm i = i := by simpa [σ] using (Equiv.swap_apply_of_ne_of_ne hi1 hi2)
        simp [DimensionalStateSpace.permute, s, hswap, hi1, hi2]
  have hsum :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then completePairIndicator (s.state i) (s.state j) else 0) = 0 := by
    simpa [s, i2, completeInteracts] using completePairSum_single_one n i2
  have hsum' :
      (∑ i : Fin n,
        ∑ j : Fin n,
          if completeInteracts i j ∧ i < j then
            completePairIndicator ((s.permute σ).state i) ((s.permute σ).state j)
          else 0) = 0 := by
    rw [hperm]
    simpa [i1, completeInteracts] using completePairSum_single_one n i1
  have hneq02 : (0 : Fin n) ≠ i2 := by
    intro h
    have hval := congrArg Fin.val h
    have hi2val : (i2 : ℕ) = 2 := by
      simp [i2, Nat.mod_eq_of_lt hn2]
    rw [hi2val] at hval
    norm_num at hval
  have hneq01 : (0 : Fin n) ≠ i1 := by
    intro h
    have hval := congrArg Fin.val h
    have hi1val : (i1 : ℕ) = 1 := by
      simp [i1, Nat.mod_eq_of_lt hn1]
    rw [hi1val] at hval
    norm_num at hval
  have hneq12 : (1 : Fin n) ≠ i2 := by
    simpa [i1] using hi12
  have hs0 : s.state 0 = 0 := by
    simp [s, hneq02]
  have hs1 : s.state 1 = 0 := by
    simp [s, hneq12]
  have hp0 : (s.permute σ).state 0 = 0 := by
    rw [hperm]
    simp [hneq01]
  have hp1 : (s.permute σ).state 1 = 1 := by
    rw [hperm]
    simp [i1]
  have hsSign : signedEqualityIndicator (s.state 0) (s.state 1) = 1 := by
    rw [hs0, hs1]
    simp [signedEqualityIndicator]
  have hpSign : signedEqualityIndicator ((s.permute σ).state 0) ((s.permute σ).state 1) = -1 := by
    rw [hp0, hp1]
    simp [signedEqualityIndicator]
  have hs : dominantPairDimensional w false s = dominantPairConstant w := by
    unfold dominantPairDimensional
    rw [if_pos rfl, hsSign, hsum]
    ring
  have hsp : dominantPairDimensional w false (s.permute σ) = - dominantPairConstant w := by
    unfold dominantPairDimensional
    rw [if_pos rfl, hpSign, hsum']
    ring
  have hEq := hsym σ false s
  rw [hs, hsp] at hEq
  have hpos : 0 < dominantPairConstant w := by
    unfold dominantPairConstant
    positivity
  linarith

theorem dominantPair_zeroOne_sufficient (w : ℕ) :
    (dominantPairProblem w).isSufficient (insert 1 ({0} : Finset (Fin (w + 3)))) := by
  intro s s' hagree
  have h0 : s.state 0 = s'.state 0 := by
    simpa using hagree 0 (by simp)
  have h1 : s.state 1 = s'.state 1 := by
    simpa using hagree 1 (by simp)
  by_cases hs : s.state 0 = s.state 1
  · have hs' : s'.state 0 = s'.state 1 := by
      calc
        s'.state 0 = s.state 0 := by simpa using h0.symm
        _ = s.state 1 := hs
        _ = s'.state 1 := by simpa using h1
    rw [dominantPair_opt_if_eq (w := w) (s := s) hs,
      dominantPair_opt_if_eq (w := w) (s := s') hs']
  · have hs' : s'.state 0 ≠ s'.state 1 := by
      intro hEq
      apply hs
      calc
        s.state 0 = s'.state 0 := by simpa using h0
        _ = s'.state 1 := hEq
        _ = s.state 1 := by simpa using h1.symm
    rw [dominantPair_opt_if_ne (w := w) (s := s) hs,
      dominantPair_opt_if_ne (w := w) (s := s') hs']

/-- A strict unary-vs-pair margin bound still does not rescue the optimizer-supported
decision-relevant graph: dense tiny interactions can coexist with a single dominant
supported pair that controls the optimizer on a two-coordinate sufficient set. -/
theorem block6_margin_bounded_obstruction (w : ℕ) :
    Nonempty (PairwiseUtility (dominantPairUtility w)) ∧
    MarginBounded (dominantPairPairwise w) ∧
    ¬ SymmetricUtility (dominantPairDimensional w) ∧
    supportedDecisionRelevantInteractionGraph
      (OptimizerSupported (dominantPairProblem w)) (dominantPairUtility w) = ⊤ ∧
    ¬ realTreewidth_le
      (supportedDecisionRelevantInteractionGraph
        (OptimizerSupported (dominantPairProblem w)) (dominantPairUtility w)) w ∧
    (dominantPairProblem w).isSufficient (insert 1 ({0} : Finset (Fin (w + 3)))) := by
  refine ⟨⟨dominantPairPairwise w⟩, dominantPair_marginBounded w,
    dominantPair_not_symmetric w, dominantPair_supportedGraph_eq_top w, ?_,
    dominantPair_zeroOne_sufficient w⟩
  simpa [dominantPair_supportedGraph_eq_top w] using
    (completeGraph_not_realTreewidth_le_of_large (n := w + 3) (w := w) (by omega))

end Paper4dFrontier
