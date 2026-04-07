import Paper4dFrontier.Block6Obstruction
import Paper4dFrontier.AdmissibleCharacterization

namespace Paper4dFrontier

open Classical
open DecisionQuotient

def DenseDecisionRelevantSlice (U : BinaryPairwiseSlice) : Prop :=
  decisionRelevantInteractionGraph U.utility = ⊤

def DenseDecisionRelevantPredicate : SliceNormalizationPredicate where
  holdsOnSlice := DenseDecisionRelevantSlice
  graphOnSlice U _ := decisionRelevantInteractionGraph U.utility

def MarginBoundedSlice (U : BinaryPairwiseSlice) : Prop :=
  MarginBounded U.pairwise

def MarginBoundedDenseDecisionRelevantSlice (U : BinaryPairwiseSlice) : Prop :=
  MarginBoundedSlice U ∧ DenseDecisionRelevantSlice U

def MarginBoundedDenseDecisionRelevantPredicate : SliceNormalizationPredicate where
  holdsOnSlice := MarginBoundedDenseDecisionRelevantSlice
  graphOnSlice U _ := decisionRelevantInteractionGraph U.utility

noncomputable def offsetCollapsedSlice (w : ℕ) : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := w + 2
  utility := offsetCollapsedAsymmetricPairUtility w
  pairwise := offsetCollapsedAsymmetricPairPairwise w

noncomputable def neverOptimalGhostSlice (w : ℕ) : BinaryPairwiseSlice where
  Action := Fin 3
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := w + 2
  utility := neverOptimalGhostUtility w
  pairwise := neverOptimalGhostPairwise w

noncomputable def marginMaskingSlice (w : ℕ) : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := w + 2
  utility := marginMaskingUtility w
  pairwise := marginMaskingPairwise w

noncomputable def dominantPairSlice (w : ℕ) : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := w + 3
  utility := dominantPairUtility w
  pairwise := dominantPairPairwise w

noncomputable def offsetCollapsedWitnessPattern : LocalPattern where
  radius := 1
  signature :=
    { actionCount := 2
      vertexCount := 2
      root := 0
      unary := (offsetCollapsedSlice 0).syntax.unary
      binary := (offsetCollapsedSlice 0).syntax.binary
      interacts := (offsetCollapsedSlice 0).syntax.interacts
      interacts_symm := (offsetCollapsedSlice 0).syntax.interacts_symm }

def HasOffsetCollapsedWitnessPattern (U : BinaryPairwiseSlice) : Prop :=
  offsetCollapsedWitnessPattern.OccursInSlice U

def HasOffsetCollapsedScaledWitnessPattern (U : BinaryPairwiseSlice) : Prop :=
  offsetCollapsedWitnessPattern.OccursUpToPositiveScaleInSlice U

theorem offsetCollapsedWitnessPattern_bounded :
    offsetCollapsedWitnessPattern.WithinBounds
      offsetCollapsedWitnessPattern.radius
      offsetCollapsedWitnessPattern.signature.vertexCount
      offsetCollapsedWitnessPattern.signature.actionCount
      offsetCollapsedWitnessPattern.selfMagnitudeBound :=
  LocalPattern.withinSelfBounds _

noncomputable def translatedOffsetCollapsedUtility : Bool → (Fin 2 → Fin 2) → ℤ :=
  fun a s => offsetCollapsedAsymmetricPairUtility 0 a s + if s ⟨1, by decide⟩ = 1 then 1 else 0

noncomputable def translatedOffsetCollapsedPairwise : PairwiseUtility translatedOffsetCollapsedUtility where
  unary i a x :=
    (offsetCollapsedAsymmetricPairPairwise 0).unary i a x +
      if i = ⟨1, by decide⟩ then if x = 1 then 1 else 0 else 0
  binary i j a x y := (offsetCollapsedAsymmetricPairPairwise 0).binary i j a x y
  interacts := (offsetCollapsedAsymmetricPairPairwise 0).interacts
  interacts_symm := (offsetCollapsedAsymmetricPairPairwise 0).interacts_symm
  decomp := by
    intro a s
    cases a <;> simp [translatedOffsetCollapsedUtility,
      offsetCollapsedAsymmetricPairUtility, addActionOffset,
      offsetBaseAsymmetricPairUtility, offsetCollapsedAsymmetricPairPairwise,
      completeInteracts] <;> ring

noncomputable def translatedOffsetCollapsedSlice : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := 2
  utility := translatedOffsetCollapsedUtility
  pairwise := translatedOffsetCollapsedPairwise

def translatedOffsetCollapsedPositiveAffineWitness :
    PositiveAffineWitness (offsetCollapsedSlice 0) translatedOffsetCollapsedSlice where
  hArity := rfl
  relabel := Equiv.refl Bool
  alpha := fun s => if s ⟨1, by decide⟩ = 1 then 1 else 0
  beta := fun _ => 1
  beta_pos := by intro _; decide
  utility_eq := by
    intro a s
    simp [translatedOffsetCollapsedSlice, translatedOffsetCollapsedUtility,
      offsetCollapsedSlice, castState]
    ring

theorem offsetCollapsedSlice_syntax_actionCount (w : ℕ) :
    (offsetCollapsedSlice w).syntax.actionCount = 2 := by
  simp [offsetCollapsedSlice, BinaryPairwiseSlice.syntax, BinaryPairwiseSlice.actionCount]

theorem offsetCollapsedSlice_hasWitnessPattern (w : ℕ) :
    HasOffsetCollapsedWitnessPattern (offsetCollapsedSlice w) := by
  have hverts : offsetCollapsedWitnessPattern.signature.vertexCount ≤ (offsetCollapsedSlice w).arity := by
    simp [offsetCollapsedWitnessPattern, offsetCollapsedSlice]
  let i0 : Fin offsetCollapsedWitnessPattern.signature.vertexCount := ⟨0, by decide⟩
  let i1 : Fin offsetCollapsedWitnessPattern.signature.vertexCount := ⟨1, by decide⟩
  let f : Fin offsetCollapsedWitnessPattern.signature.vertexCount → Fin (offsetCollapsedSlice w).arity :=
    fun i => ⟨i.1, Nat.lt_of_lt_of_le i.2 hverts⟩
  have hf : Function.Injective f := by
    intro i j hij
    exact Fin.ext (by simpa [f] using congrArg Fin.val hij)
  let σ : Fin offsetCollapsedWitnessPattern.signature.actionCount ≃
      Fin (offsetCollapsedSlice w).syntax.actionCount :=
    finCongr (offsetCollapsedSlice_syntax_actionCount w).symm
  refine ⟨f, hf, σ, ?_, ?_, ?_, ?_⟩
  · intro v
    fin_cases v
    · refine ⟨SimpleGraph.Walk.nil, by simp [offsetCollapsedWitnessPattern, f]⟩
    · have hadj : (offsetCollapsedSlice w).syntax.interactionGraph.Adj (f i0) (f i1) := by
        have hne : f i0 ≠ f i1 := by
          intro hEq
          have hval : (0 : ℕ) = 1 := by
            simpa [f, i0, i1] using congrArg Fin.val hEq
          omega
        refine (show f i0 ≠ f i1 ∧ (offsetCollapsedSlice w).syntax.interacts (f i0) (f i1) from ?_)
        exact ⟨hne, by simpa [offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
          offsetCollapsedAsymmetricPairPairwise, completeInteracts] using hne⟩
      refine ⟨SimpleGraph.Walk.cons hadj SimpleGraph.Walk.nil, by simp [offsetCollapsedWitnessPattern]⟩
  · intro i a x
    fin_cases i <;> fin_cases a <;> fin_cases x
    all_goals
      simp [offsetCollapsedWitnessPattern, offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
        offsetCollapsedAsymmetricPairPairwise, f, σ, i0, i1]
    all_goals
      rfl
  · intro i j hij
    fin_cases i <;> fin_cases j <;>
      simpa [offsetCollapsedWitnessPattern, offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
        offsetCollapsedAsymmetricPairPairwise, f, i0, i1, completeInteracts] using hij
  · intro i j a x y
    fin_cases i <;> fin_cases j <;> fin_cases a <;> fin_cases x <;> fin_cases y
    all_goals
      simp [offsetCollapsedWitnessPattern, offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
        offsetCollapsedAsymmetricPairPairwise, f, σ, i0, i1]
    all_goals
      rfl

theorem offsetCollapsedWitnessPattern_boundedPatternDefinable :
    BoundedPatternDefinable HasOffsetCollapsedWitnessPattern := by
  simpa [HasOffsetCollapsedWitnessPattern] using
    boundedPatternDefinable_of_singleton_witness
      offsetCollapsedWitnessPattern
      offsetCollapsedWitnessPattern.radius
      offsetCollapsedWitnessPattern.signature.vertexCount
      offsetCollapsedWitnessPattern.signature.actionCount
      offsetCollapsedWitnessPattern.selfMagnitudeBound
      offsetCollapsedWitnessPattern_bounded

theorem offsetCollapsedSlice_hasScaledWitnessPattern (w : ℕ) :
    HasOffsetCollapsedScaledWitnessPattern (offsetCollapsedSlice w) :=
  offsetCollapsedWitnessPattern.occursUpToPositiveScaleInSlice_of_occurs
    (offsetCollapsedSlice_hasWitnessPattern w)

theorem scaledOffsetChecker_polynomialTimeCheckable :
    PolynomialTimeCheckable HasOffsetCollapsedScaledWitnessPattern := by
  classical
  exact polynomialTimeCheckable_of_decidable HasOffsetCollapsedScaledWitnessPattern

theorem scaledOffsetChecker_handles_uniform_scaling :
    HasOffsetCollapsedScaledWitnessPattern (offsetCollapsedSlice 0) ∧
      HasOffsetCollapsedScaledWitnessPattern (scaleSlice (2 : ℤ) (offsetCollapsedSlice 0)) := by
  exact ⟨offsetCollapsedSlice_hasScaledWitnessPattern 0,
    offsetCollapsedWitnessPattern.occursUpToPositiveScaleInSlice_of_occurs_scale 2 (by decide)
      (offsetCollapsedSlice_hasWitnessPattern 0)⟩

theorem scaledOffsetChecker_globalPositiveScaleInvariant
    {U : BinaryPairwiseSlice} (k : ℕ) (hk : 0 < k) :
    HasOffsetCollapsedScaledWitnessPattern U →
      HasOffsetCollapsedScaledWitnessPattern (scaleSlice (k : ℤ) U) := by
  exact offsetCollapsedWitnessPattern.occursUpToPositiveScaleInSlice_globalScaleInvariant k hk

theorem offsetWitness_second_vertex_actionSum_x1_zero :
    offsetCollapsedWitnessPattern.signature.unary ⟨1, by decide⟩ (Fintype.equivFin Bool false) 1 +
      offsetCollapsedWitnessPattern.signature.unary ⟨1, by decide⟩ (Fintype.equivFin Bool true) 1 = 0 := by
  simp [offsetCollapsedWitnessPattern, offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
    offsetCollapsedAsymmetricPairPairwise]

theorem translatedOffsetCollapsedPairwise_actionSum_x1_coord0 :
    translatedOffsetCollapsedPairwise.unary ⟨0, by decide⟩ false 1 +
      translatedOffsetCollapsedPairwise.unary ⟨0, by decide⟩ true 1 = 2 := by
  simp [translatedOffsetCollapsedPairwise, offsetCollapsedAsymmetricPairPairwise]

theorem translatedOffsetCollapsedPairwise_actionSum_x1_coord1 :
    translatedOffsetCollapsedPairwise.unary ⟨1, by decide⟩ false 1 +
      translatedOffsetCollapsedPairwise.unary ⟨1, by decide⟩ true 1 = 2 := by
  simp [translatedOffsetCollapsedPairwise, offsetCollapsedAsymmetricPairPairwise]

theorem translatedOffsetCollapsed_additive_mismatch :
    (offsetCollapsedWitnessPattern.signature.unary ⟨1, by decide⟩ (Fintype.equivFin Bool false) 1 +
        offsetCollapsedWitnessPattern.signature.unary ⟨1, by decide⟩ (Fintype.equivFin Bool true) 1 = 0) ∧
      (translatedOffsetCollapsedPairwise.unary ⟨0, by decide⟩ false 1 +
        translatedOffsetCollapsedPairwise.unary ⟨0, by decide⟩ true 1 = 2) ∧
      (translatedOffsetCollapsedPairwise.unary ⟨1, by decide⟩ false 1 +
        translatedOffsetCollapsedPairwise.unary ⟨1, by decide⟩ true 1 = 2) := by
  exact ⟨offsetWitness_second_vertex_actionSum_x1_zero,
    translatedOffsetCollapsedPairwise_actionSum_x1_coord0,
    translatedOffsetCollapsedPairwise_actionSum_x1_coord1⟩

def OffsetCollapsedClosureSlice (U : BinaryPairwiseSlice) : Prop :=
  ClosureHull HasOffsetCollapsedWitnessPattern U

def OffsetCollapsedClosurePredicate : SliceNormalizationPredicate where
  holdsOnSlice := OffsetCollapsedClosureSlice
  graphOnSlice U _ := ⊤

theorem offsetCollapsedClosureSlice_generatedByBoundedPatterns :
    ClosureGeneratedByBoundedPatterns OffsetCollapsedClosureSlice := by
  refine ⟨HasOffsetCollapsedWitnessPattern, offsetCollapsedWitnessPattern_boundedPatternDefinable, ?_⟩
  intro U
  rfl

theorem offsetCollapsedClosureSlice_closureLawInvariant :
    ClosureLawInvariant OffsetCollapsedClosureSlice :=
  closureHull_closureLawInvariant _

theorem offsetCollapsedSlice_in_offsetCollapsedClosure (w : ℕ) :
    OffsetCollapsedClosureSlice (offsetCollapsedSlice w) := by
  exact closureHull_intro (offsetCollapsedSlice_hasWitnessPattern w)

noncomputable def doubledOffsetCollapsedSlice : BinaryPairwiseSlice :=
  scaleSlice (2 : ℤ) (offsetCollapsedSlice 0)

theorem doubledOffsetCollapsedSlice_not_hasWitnessPattern :
    ¬ HasOffsetCollapsedWitnessPattern doubledOffsetCollapsedSlice := by
  intro hOcc
  let i0 : Fin offsetCollapsedWitnessPattern.signature.vertexCount := ⟨0, by decide⟩
  let aFalse : Fin offsetCollapsedWitnessPattern.signature.actionCount := Fintype.equivFin Bool false
  rcases hOcc with ⟨f, hf, σ, _, hunary, _, _⟩
  have hpat : offsetCollapsedWitnessPattern.signature.unary i0 aFalse 0 = 1 := by
    simp [offsetCollapsedWitnessPattern, offsetCollapsedSlice, BinaryPairwiseSlice.syntax,
      offsetCollapsedAsymmetricPairPairwise, aFalse, i0]
  have hun := hunary i0 aFalse 0
  have hscaled : doubledOffsetCollapsedSlice.syntax.unary (f i0) (σ aFalse) 0 =
      2 * (offsetCollapsedSlice 0).syntax.unary (f i0) (σ aFalse) 0 := by
    simpa [doubledOffsetCollapsedSlice] using
      (scaleSlice_syntax_unary (k := (2 : ℤ)) (U := offsetCollapsedSlice 0)
        (i := f i0) (a := σ aFalse) (x := 0))
  rw [hpat, hscaled] at hun
  omega

theorem doubledOffsetCollapsedSlice_in_offsetCollapsedClosure :
    OffsetCollapsedClosureSlice doubledOffsetCollapsedSlice := by
  refine ⟨offsetCollapsedSlice 0, ?_, offsetCollapsedSlice_hasWitnessPattern 0⟩
  exact Relation.EqvGen.rel _ _ (ClosureStep.positiveAffine
    (scaleSlice_positiveAffineWitness 2 (by decide) (offsetCollapsedSlice 0)))

theorem hasOffsetCollapsedWitnessPattern_not_closureLawInvariant :
    ¬ ClosureLawInvariant HasOffsetCollapsedWitnessPattern := by
  intro hInv
  have hiff := hInv.positive_affine
    (scaleSlice_positiveAffineWitness 2 (by decide) (offsetCollapsedSlice 0))
  exact doubledOffsetCollapsedSlice_not_hasWitnessPattern
    ((hiff).mp (offsetCollapsedSlice_hasWitnessPattern 0))

theorem rawOffsetWitness_boundedPattern_not_closureInvariant :
    BoundedPatternDefinable HasOffsetCollapsedWitnessPattern ∧
      ¬ ClosureLawInvariant HasOffsetCollapsedWitnessPattern := by
  exact ⟨offsetCollapsedWitnessPattern_boundedPatternDefinable,
    hasOffsetCollapsedWitnessPattern_not_closureLawInvariant⟩

theorem offsetClosure_generated_and_invariant :
    ClosureGeneratedByBoundedPatterns OffsetCollapsedClosureSlice ∧
      ClosureLawInvariant OffsetCollapsedClosureSlice := by
  exact ⟨offsetCollapsedClosureSlice_generatedByBoundedPatterns,
    offsetCollapsedClosureSlice_closureLawInvariant⟩

theorem offset_admissibility_refinement_gap :
    (BoundedPatternDefinable HasOffsetCollapsedWitnessPattern ∧
        ¬ ClosureLawInvariant HasOffsetCollapsedWitnessPattern) ∧
      (ClosureGeneratedByBoundedPatterns OffsetCollapsedClosureSlice ∧
        ClosureLawInvariant OffsetCollapsedClosureSlice) := by
  exact ⟨rawOffsetWitness_boundedPattern_not_closureInvariant,
    offsetClosure_generated_and_invariant⟩

theorem supportedInteraction_implies_decisionRelevant
    {A : Type*} {n : ℕ} (support : A → Prop)
    {u : A → (Fin n → Fin 2) → ℤ} {i j : Fin n} :
    SupportedDecisionRelevantBinaryPairInteraction support u i j →
      HasDecisionRelevantBinaryPairInteraction u i j := by
  rintro ⟨a, b, _, _, hab⟩
  exact ⟨a, b, hab⟩

theorem decisionRelevantGraph_eq_top_of_supported_eq_top
    {A : Type*} {n : ℕ} (support : A → Prop)
    {u : A → (Fin n → Fin 2) → ℤ}
    (hTop : supportedDecisionRelevantInteractionGraph support u = ⊤) :
    decisionRelevantInteractionGraph u = ⊤ := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [decisionRelevantInteractionGraph, InteractionGraph]
  · have hsupp : (supportedDecisionRelevantInteractionGraph support u).Adj i j := by
      simpa [hTop, h] using (show (⊤ : SimpleGraph (Fin n)).Adj i j from by simp [h])
    have hsuppRel : SupportedDecisionRelevantBinaryPairInteraction support u i j := by
      simpa [supportedDecisionRelevantInteractionGraph, InteractionGraph, h] using hsupp
    have hrel : HasDecisionRelevantBinaryPairInteraction u i j := by
      exact supportedInteraction_implies_decisionRelevant support hsuppRel
    simp [decisionRelevantInteractionGraph, InteractionGraph, h, hrel]

theorem offsetCollapsedSlice_denseDecisionRelevant (w : ℕ) :
    DenseDecisionRelevantSlice (offsetCollapsedSlice w) := by
  simpa [DenseDecisionRelevantSlice, offsetCollapsedSlice] using
    offsetCollapsedAsymmetricPair_decisionRelevantGraph_eq_top w

theorem neverOptimalGhostSlice_denseDecisionRelevant (w : ℕ) :
    DenseDecisionRelevantSlice (neverOptimalGhostSlice w) := by
  simpa [DenseDecisionRelevantSlice, neverOptimalGhostSlice] using
    neverOptimalGhost_decisionRelevantGraph_eq_top w

theorem marginMaskingSlice_denseDecisionRelevant (w : ℕ) :
    DenseDecisionRelevantSlice (marginMaskingSlice w) := by
  have hTop := decisionRelevantGraph_eq_top_of_supported_eq_top
    (support := OptimizerSupported (marginMaskingProblem w))
    (u := marginMaskingUtility w)
    (marginMasking_supportedGraph_eq_top w)
  simpa [DenseDecisionRelevantSlice, marginMaskingSlice] using hTop

theorem dominantPairSlice_denseDecisionRelevant (w : ℕ) :
    DenseDecisionRelevantSlice (dominantPairSlice w) := by
  have hTop := decisionRelevantGraph_eq_top_of_supported_eq_top
    (support := OptimizerSupported (dominantPairProblem w))
    (u := dominantPairUtility w)
    (dominantPair_supportedGraph_eq_top w)
  simpa [DenseDecisionRelevantSlice, dominantPairSlice] using hTop

theorem dominantPairSlice_marginBounded (w : ℕ) :
    MarginBoundedSlice (dominantPairSlice w) := by
  simpa [MarginBoundedSlice, dominantPairSlice] using dominantPair_marginBounded w

theorem dominantPairSlice_marginBounded_denseDecisionRelevant (w : ℕ) :
    MarginBoundedDenseDecisionRelevantSlice (dominantPairSlice w) := by
  exact ⟨dominantPairSlice_marginBounded w, dominantPairSlice_denseDecisionRelevant w⟩

theorem maxPairInteractionMagnitude_le_of_forall {A : Type*} [Fintype A] {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} (pw : PairwiseUtility u) (M : Nat)
    (h : ∀ i j a, Int.natAbs (binaryCrossDifference (pw.binary i j a)) ≤ M) :
    maxPairInteractionMagnitude pw ≤ M := by
  unfold maxPairInteractionMagnitude
  refine Finset.sup_le ?_
  intro i hi
  refine Finset.sup_le ?_
  intro j hj
  refine Finset.sup_le ?_
  intro a ha
  exact h i j a

def fin2_0 : Fin 2 := ⟨0, by decide⟩

def fin2_1 : Fin 2 := ⟨1, by decide⟩

def marginMaskingBoostBinary (x y : Fin 2) : ℤ :=
  2 * signedEqualityIndicator x y

noncomputable def marginMaskingBoostStateTerm (s : Fin 2 → Fin 2) : ℤ :=
  ∑ i : Fin 2,
    ∑ j : Fin 2,
      if completeInteracts i j ∧ i < j then
        if i = fin2_0 ∧ j = fin2_1 then marginMaskingBoostBinary (s i) (s j) else 0
      else 0

noncomputable def translatedMarginMaskingUtility : Bool → (Fin 2 → Fin 2) → ℤ :=
  fun a s => marginMaskingUtility 0 a s + marginMaskingBoostStateTerm s

noncomputable def translatedMarginMaskingPairwise : PairwiseUtility translatedMarginMaskingUtility where
  unary i a x := (marginMaskingPairwise 0).unary i a x
  binary i j a x y :=
    (marginMaskingPairwise 0).binary i j a x y +
      if i = fin2_0 ∧ j = fin2_1 then marginMaskingBoostBinary x y else 0
  interacts := @completeInteracts 2
  interacts_symm := @completeInteracts_symm 2
  decomp := by
    intro a s
    simp [translatedMarginMaskingUtility, marginMaskingBoostStateTerm,
      marginMaskingPairwise, marginMaskingUtility,
      completeInteracts]
    ring

noncomputable def translatedMarginMaskingSlice : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := 2
  utility := translatedMarginMaskingUtility
  pairwise := translatedMarginMaskingPairwise

noncomputable def translatedMarginMaskingPositiveAffineWitness :
    PositiveAffineWitness (marginMaskingSlice 0) translatedMarginMaskingSlice where
  hArity := rfl
  relabel := Equiv.refl Bool
  alpha := marginMaskingBoostStateTerm
  beta := fun _ => 1
  beta_pos := by intro _; decide
  utility_eq := by
    intro a s
    simp [translatedMarginMaskingSlice, translatedMarginMaskingUtility,
      marginMaskingSlice, castState]
    ring

theorem translatedMarginMaskingSlice_in_orbit :
    ClosureEquivalent (marginMaskingSlice 0) translatedMarginMaskingSlice := by
  exact closureEquivalent_of_positiveAffineWitness translatedMarginMaskingPositiveAffineWitness

theorem marginMaskingSlice0_maxPairInteractionMagnitude_eq_one :
    maxPairInteractionMagnitude (marginMaskingSlice 0).pairwise = 1 := by
  have hsup :
      (Finset.univ.sup fun i : Fin 2 => Finset.univ.sup fun j : Fin 2 => (1 : Nat)) = 1 := by
    decide
  simp [maxPairInteractionMagnitude, marginMaskingSlice, marginMaskingPairwise,
    binaryCrossDifference, completePairIndicator, hsup]

theorem marginMaskingSlice0_not_marginBounded :
    ¬ MarginBoundedSlice (marginMaskingSlice 0) := by
  intro hBounded
  have hAt := hBounded fin2_0 false 0
  have hUnary :
      Int.natAbs ((marginMaskingSlice 0).pairwise.unary fin2_0 false 0) = 5 := by
    simp [marginMaskingSlice, marginMaskingPairwise, marginMaskingConstant, fin2_0]
  rw [marginMaskingSlice0_maxPairInteractionMagnitude_eq_one, hUnary] at hAt
  have hNot : ¬ (5 : Nat) ≤ 2 := by decide
  exact hNot hAt

theorem translatedMarginMasking_maxPairInteractionMagnitude_ge_eight :
    8 ≤ maxPairInteractionMagnitude translatedMarginMaskingPairwise := by
  simp [maxPairInteractionMagnitude, translatedMarginMaskingPairwise,
    marginMaskingPairwise, marginMaskingBoostBinary, binaryCrossDifference,
    signedEqualityIndicator, fin2_0, fin2_1]

theorem translatedMarginMaskingSlice_marginBounded :
    MarginBoundedSlice translatedMarginMaskingSlice := by
  have hMaxGe8 : 8 ≤ maxPairInteractionMagnitude translatedMarginMaskingSlice.pairwise := by
    simpa [translatedMarginMaskingSlice] using translatedMarginMasking_maxPairInteractionMagnitude_ge_eight
  have hMarginLe : (5 : Nat) ≤ 2 * maxPairInteractionMagnitude translatedMarginMaskingSlice.pairwise := by
    have hBase : (5 : Nat) ≤ 2 * 8 := by decide
    exact le_trans hBase (Nat.mul_le_mul_left 2 hMaxGe8)
  intro i a x
  have hUnaryLe :
      Int.natAbs (translatedMarginMaskingSlice.pairwise.unary i a x) ≤ 5 := by
    fin_cases i <;> cases a <;> fin_cases x
    all_goals
      simp [translatedMarginMaskingSlice, translatedMarginMaskingPairwise,
        marginMaskingPairwise, marginMaskingConstant]
  exact le_trans hUnaryLe hMarginLe

def marginMaskingClosureOrbitGap : Prop :=
  ClosureEquivalent (marginMaskingSlice 0) translatedMarginMaskingSlice ∧
    ¬ MarginBoundedSlice (marginMaskingSlice 0) ∧
    MarginBoundedSlice translatedMarginMaskingSlice

theorem marginMaskingClosureOrbitGap_exists : marginMaskingClosureOrbitGap := by
  exact ⟨translatedMarginMaskingSlice_in_orbit,
    marginMaskingSlice0_not_marginBounded,
    translatedMarginMaskingSlice_marginBounded⟩

def pairCrossMagnitude (U : BinaryPairwiseSlice) (i j : Fin U.arity) (a : U.Action) : Nat :=
  Int.natAbs (binaryCrossDifference (U.pairwise.binary i j a))

def UniqueDominantPairAt (U : BinaryPairwiseSlice)
    (i : Fin U.arity) (j : Fin U.arity) (a : U.Action) : Prop :=
  (∀ i' j' a', pairCrossMagnitude U i' j' a' ≤ pairCrossMagnitude U i j a) ∧
    (∀ i' j' a', pairCrossMagnitude U i' j' a' = pairCrossMagnitude U i j a →
      i' = i ∧ j' = j ∧ a' = a)

def fin3_0 : Fin 3 := ⟨0, by decide⟩

def fin3_1 : Fin 3 := ⟨1, by decide⟩

def fin3_2 : Fin 3 := ⟨2, by decide⟩

def shiftedDominantCommonBinary (x y : Fin 2) : ℤ :=
  dominantPairConstant 0 * signedEqualityIndicator x y + 2 * completePairIndicator x y

noncomputable def translatedDominantPairOrbitUtility : Bool → (Fin 3 → Fin 2) → ℤ
  | a, s =>
      ∑ i : Fin 3,
        ∑ j : Fin 3,
          if completeInteracts i j ∧ i < j then
            ((if a = false then dominantPairBinary 0 i j (s i) (s j) else 0) +
              (if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0))
          else 0

noncomputable def translatedDominantPairOrbitPairwise : PairwiseUtility translatedDominantPairOrbitUtility where
  unary _ _ _ := 0
  binary i j a x y :=
    (if a = false then dominantPairBinary 0 i j x y else 0) +
      (if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary x y else 0)
  interacts := @completeInteracts 3
  interacts_symm := @completeInteracts_symm 3
  decomp := by
    intro a s
    cases a <;> simp [translatedDominantPairOrbitUtility, completeInteracts] <;> ring

noncomputable def translatedDominantPairOrbitSlice : BinaryPairwiseSlice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := 3
  utility := translatedDominantPairOrbitUtility
  pairwise := translatedDominantPairOrbitPairwise

noncomputable def translatedDominantPairOrbitPositiveAffineWitness :
    PositiveAffineWitness (dominantPairSlice 0) translatedDominantPairOrbitSlice where
  hArity := rfl
  relabel := Equiv.refl Bool
  alpha := fun s =>
    ∑ i : Fin 3,
      ∑ j : Fin 3,
        if completeInteracts i j ∧ i < j then
          if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
        else 0
  beta := fun _ => 1
  beta_pos := by intro _; decide
  utility_eq := by
    intro a s
    cases a with
    | false =>
        have hsplit :
            translatedDominantPairOrbitUtility false s =
              (∑ i : Fin 3,
                ∑ j : Fin 3,
                  if completeInteracts i j ∧ i < j then
                    if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                  else 0) +
              dominantPairUtility 0 false s := by
          unfold translatedDominantPairOrbitUtility dominantPairUtility
          calc
            (∑ i : Fin 3,
              ∑ j : Fin 3,
                if completeInteracts i j ∧ i < j then
                  dominantPairBinary 0 i j (s i) (s j) +
                    if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                else 0)
                = ∑ i : Fin 3,
                    ∑ j : Fin 3,
                      ((if completeInteracts i j ∧ i < j then dominantPairBinary 0 i j (s i) (s j) else 0) +
                        (if completeInteracts i j ∧ i < j then
                          if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                        else 0)) := by
                    refine Finset.sum_congr rfl ?_
                    intro i hi
                    refine Finset.sum_congr rfl ?_
                    intro j hj
                    by_cases h : completeInteracts i j ∧ i < j <;> simp [h]
            _ = ∑ i : Fin 3,
                  ((∑ j : Fin 3,
                      if completeInteracts i j ∧ i < j then dominantPairBinary 0 i j (s i) (s j) else 0) +
                    (∑ j : Fin 3,
                      if completeInteracts i j ∧ i < j then
                        if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                      else 0)) := by
                  refine Finset.sum_congr rfl ?_
                  intro i hi
                  rw [Finset.sum_add_distrib]
            _ = (∑ i : Fin 3,
                  ∑ j : Fin 3,
                    if completeInteracts i j ∧ i < j then dominantPairBinary 0 i j (s i) (s j) else 0) +
                ∑ i : Fin 3,
                  ∑ j : Fin 3,
                    if completeInteracts i j ∧ i < j then
                      if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                    else 0 := by
                  rw [Finset.sum_add_distrib]
            _ =
                (∑ i : Fin 3,
                  ∑ j : Fin 3,
                    if completeInteracts i j ∧ i < j then
                      if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                    else 0) +
                dominantPairUtility 0 false s := by
                  rw [add_comm]
                  rfl
        simpa [translatedDominantPairOrbitSlice, dominantPairSlice, castState] using hsplit
    | true =>
        have htrue :
            translatedDominantPairOrbitUtility true s =
              ∑ i : Fin 3,
                ∑ j : Fin 3,
                  if completeInteracts i j ∧ i < j then
                    if i = fin3_1 ∧ j = fin3_2 then shiftedDominantCommonBinary (s i) (s j) else 0
                  else 0 := by
          unfold translatedDominantPairOrbitUtility
          refine Finset.sum_congr rfl ?_
          intro i hi
          refine Finset.sum_congr rfl ?_
          intro j hj
          by_cases h : completeInteracts i j ∧ i < j <;> simp [h]
        simpa [translatedDominantPairOrbitSlice, dominantPairSlice, dominantPairUtility, castState]
          using htrue

theorem dominantPairSlice0_uniqueDominantPair01 :
    UniqueDominantPairAt (dominantPairSlice 0) fin3_0 fin3_1 false := by
  constructor
  · intro i j a
    fin_cases i <;> fin_cases j <;> cases a <;>
      simp [pairCrossMagnitude, dominantPairSlice, dominantPairPairwise,
        dominantPairBinary, binaryCrossDifference, signedEqualityIndicator,
        completePairIndicator, dominantPairConstant, fin3_0, fin3_1]
  · intro i j a h
    fin_cases i <;> fin_cases j <;> cases a <;>
      simp [pairCrossMagnitude, dominantPairSlice, dominantPairPairwise,
        dominantPairBinary, binaryCrossDifference, signedEqualityIndicator,
        completePairIndicator, dominantPairConstant, fin3_0, fin3_1] at h ⊢

theorem translatedDominantPairOrbit_uniqueDominantPair12 :
    UniqueDominantPairAt translatedDominantPairOrbitSlice fin3_1 fin3_2 false := by
  constructor
  · intro i j a
    fin_cases i <;> fin_cases j <;> cases a <;>
      simp [pairCrossMagnitude, translatedDominantPairOrbitSlice,
        translatedDominantPairOrbitPairwise, dominantPairBinary,
        shiftedDominantCommonBinary, binaryCrossDifference, signedEqualityIndicator,
        completePairIndicator, dominantPairConstant, fin3_1, fin3_2]
  · intro i j a h
    fin_cases i <;> fin_cases j <;> cases a <;>
      simp [pairCrossMagnitude, translatedDominantPairOrbitSlice,
        translatedDominantPairOrbitPairwise, dominantPairBinary,
        shiftedDominantCommonBinary, binaryCrossDifference, signedEqualityIndicator,
        completePairIndicator, dominantPairConstant, fin3_1, fin3_2] at h ⊢

def dominantPairClosureOrbitGap : Prop :=
  ClosureEquivalent (dominantPairSlice 0) translatedDominantPairOrbitSlice ∧
    UniqueDominantPairAt (dominantPairSlice 0) fin3_0 fin3_1 false ∧
    UniqueDominantPairAt translatedDominantPairOrbitSlice fin3_1 fin3_2 false

theorem dominantPairClosureOrbitGap_exists : dominantPairClosureOrbitGap := by
  exact ⟨Relation.EqvGen.rel _ _ (ClosureStep.positiveAffine
      translatedDominantPairOrbitPositiveAffineWitness),
    dominantPairSlice0_uniqueDominantPair01,
    translatedDominantPairOrbit_uniqueDominantPair12⟩

abbrev Slice := BinaryPairwiseSlice

def firstFin (n : ℕ) (h : 2 ≤ n) : Fin n := ⟨0, by omega⟩

def secondFin (n : ℕ) (h : 2 ≤ n) : Fin n := ⟨1, by omega⟩

def HasUniqueDominantPair (S : Slice) : Prop :=
  ∃ hArity : 2 ≤ S.arity, ∃ a : S.Action,
    UniqueDominantPairAt S (firstFin S.arity hArity) (secondFin S.arity hArity) a

theorem closureLawInvariant_iff_of_closureEquivalent
    {P : Slice → Prop} (hInv : ClosureLawInvariant P)
    {S1 S2 : Slice} (hEqv : ClosureEquivalent S1 S2) :
    P S1 ↔ P S2 := by
  induction hEqv with
  | rel _ _ hStep =>
      cases hStep with
      | actionRelabel h => exact hInv.action_relabel h
      | coordinateRelabel h => exact hInv.coordinate_relabel h
      | positiveAffine h => exact hInv.positive_affine h
      | duplicateAction h => exact hInv.duplicate_action h
      | duplicateState h => exact hInv.duplicate_state h
      | irrelevantCoordinate h => exact hInv.irrelevant_coordinate h
  | refl _ => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ ih12 ih23 => exact Iff.trans ih12 ih23

theorem no_closureInvariant_predicate_of_orbit_gap
    {P Q : Slice → Prop} (hInv : ClosureLawInvariant P)
    {S1 S2 : Slice} (hEqv : ClosureEquivalent S1 S2)
    (hQ1 : Q S1) (hQ2 : ¬ Q S2) :
    ¬ (∀ S, P S ↔ Q S) := by
  intro hDecides
  have hPOrbit : P S1 ↔ P S2 :=
    closureLawInvariant_iff_of_closureEquivalent hInv hEqv
  have hP1 : P S1 := (hDecides S1).2 hQ1
  have hP2 : P S2 := hPOrbit.mp hP1
  exact hQ2 ((hDecides S2).1 hP2)

theorem dominantPairSlice0_hasUniqueDominantPair :
    HasUniqueDominantPair (dominantPairSlice 0) := by
  refine ⟨by decide, false, ?_⟩
  simpa [firstFin, secondFin, fin3_0, fin3_1] using
    dominantPairSlice0_uniqueDominantPair01

theorem translatedDominantPairOrbit_not_hasUniqueDominantPair :
    ¬ HasUniqueDominantPair translatedDominantPairOrbitSlice := by
  intro h
  rcases h with ⟨hArity, a, hDom01⟩
  have hDom12 := translatedDominantPairOrbit_uniqueDominantPair12
  have h12le01 :
      pairCrossMagnitude translatedDominantPairOrbitSlice fin3_1 fin3_2 false ≤
        pairCrossMagnitude translatedDominantPairOrbitSlice
          (firstFin translatedDominantPairOrbitSlice.arity hArity)
          (secondFin translatedDominantPairOrbitSlice.arity hArity) a := by
    exact hDom01.1 fin3_1 fin3_2 false
  have h01le12 :
      pairCrossMagnitude translatedDominantPairOrbitSlice
          (firstFin translatedDominantPairOrbitSlice.arity hArity)
          (secondFin translatedDominantPairOrbitSlice.arity hArity) a ≤
        pairCrossMagnitude translatedDominantPairOrbitSlice fin3_1 fin3_2 false := by
    exact hDom12.1 _ _ _
  have hEq :
      pairCrossMagnitude translatedDominantPairOrbitSlice
          (firstFin translatedDominantPairOrbitSlice.arity hArity)
          (secondFin translatedDominantPairOrbitSlice.arity hArity) a =
        pairCrossMagnitude translatedDominantPairOrbitSlice fin3_1 fin3_2 false :=
    Nat.le_antisymm h01le12 h12le01
  rcases hDom12.2 _ _ _ hEq with ⟨hFirst, _, _⟩
  have hValEq :
      (firstFin translatedDominantPairOrbitSlice.arity hArity).val = fin3_1.val :=
    congrArg Fin.val hFirst
  have hZero : (firstFin translatedDominantPairOrbitSlice.arity hArity).val = 0 := by
    simp [firstFin]
  have hOne : fin3_1.val = 1 := by
    simp [fin3_1]
  omega

theorem no_closureInvariant_predicate_decides_dominantPair :
    ∀ (P : Slice → Prop), ClosureLawInvariant P →
      ¬ (∀ S, P S ↔ HasUniqueDominantPair S) := by
  intro P hInv hDecides
  rcases dominantPairClosureOrbitGap_exists with ⟨hOrbit, hDom01, _hDom12⟩
  have hPOrbit : P (dominantPairSlice 0) ↔ P translatedDominantPairOrbitSlice :=
    closureLawInvariant_iff_of_closureEquivalent hInv hOrbit
  have hP0 : P (dominantPairSlice 0) :=
    (hDecides (dominantPairSlice 0)).2
      (by
        refine ⟨by decide, false, ?_⟩
        simpa [firstFin, secondFin, fin3_0, fin3_1] using hDom01)
  have hPt : P translatedDominantPairOrbitSlice := hPOrbit.mp hP0
  have hHt : HasUniqueDominantPair translatedDominantPairOrbitSlice :=
    (hDecides translatedDominantPairOrbitSlice).1 hPt
  exact translatedDominantPairOrbit_not_hasUniqueDominantPair hHt

theorem no_admissibleNormalizationPredicate_decides_dominantPair :
    ∀ P : AdmissibleNormalizationPredicate,
      ¬ (∀ S : Slice, P.holdsOnSlice S ↔ HasUniqueDominantPair S) := by
  intro P
  exact no_closureInvariant_predicate_decides_dominantPair
    P.holdsOnSlice P.closureLawInvariant

def admissibleCollapseLandscapeInfinityOnDominantPairFamily : Prop :=
  ∀ P : AdmissibleNormalizationPredicate,
    ¬ (∀ S : Slice, P.holdsOnSlice S ↔ HasUniqueDominantPair S)

theorem admissibleCollapseLandscapeInfinityOnDominantPairFamily_holds :
    admissibleCollapseLandscapeInfinityOnDominantPairFamily := by
  exact no_admissibleNormalizationPredicate_decides_dominantPair

theorem no_closureInvariant_predicate_decides_marginBounded :
    ∀ (P : Slice → Prop), ClosureLawInvariant P →
      ¬ (∀ S, P S ↔ MarginBoundedSlice S) := by
  intro P hInv hDecides
  rcases marginMaskingClosureOrbitGap_exists with ⟨hOrbit, hNotBounded, hBounded⟩
  have hPOrbit : P (marginMaskingSlice 0) ↔ P translatedMarginMaskingSlice :=
    closureLawInvariant_iff_of_closureEquivalent hInv hOrbit
  have hPT : P translatedMarginMaskingSlice :=
    (hDecides translatedMarginMaskingSlice).2 hBounded
  have hP0 : P (marginMaskingSlice 0) := hPOrbit.mpr hPT
  have hB0 : MarginBoundedSlice (marginMaskingSlice 0) :=
    (hDecides (marginMaskingSlice 0)).1 hP0
  exact hNotBounded hB0

theorem no_admissibleNormalizationPredicate_decides_marginBounded :
    ∀ P : AdmissibleNormalizationPredicate,
      ¬ (∀ S : Slice, P.holdsOnSlice S ↔ MarginBoundedSlice S) := by
  intro P
  exact no_closureInvariant_predicate_decides_marginBounded
    P.holdsOnSlice P.closureLawInvariant

def admissibleCollapseLandscapeInfinityOnMarginBounded : Prop :=
  ∀ P : AdmissibleNormalizationPredicate,
    ¬ (∀ S : Slice, P.holdsOnSlice S ↔ MarginBoundedSlice S)

theorem admissibleCollapseLandscapeInfinityOnMarginBounded_holds :
    admissibleCollapseLandscapeInfinityOnMarginBounded := by
  exact no_admissibleNormalizationPredicate_decides_marginBounded

def zeroActionFin (n : ℕ) (h : 1 ≤ n) : Fin n := ⟨0, by omega⟩

def twoActionFin (n : ℕ) (h : 3 ≤ n) : Fin n := ⟨2, by omega⟩

def GhostActionTwoPairCrossOneSlice (U : Slice) : Prop :=
  ∃ hAr : 2 ≤ U.arity, ∃ a : U.Action,
    U.pairwise.unary (firstFin U.arity hAr) a 0 = -1 ∧
      U.pairwise.unary (firstFin U.arity hAr) a 1 = -1 ∧
      pairCrossMagnitude U (firstFin U.arity hAr) (secondFin U.arity hAr) a = 1

def OffsetActionZeroPairCrossOneSlice (U : Slice) : Prop :=
  ∃ hAr : 2 ≤ U.arity, ∃ a b : U.Action,
    a ≠ b ∧
      pairCrossMagnitude U (firstFin U.arity hAr) (secondFin U.arity hAr) a = 1 ∧
      pairCrossMagnitude U (firstFin U.arity hAr) (secondFin U.arity hAr) b = 0

def ghostPairBoostBinary (x y : Fin 2) : ℤ :=
  2 * signedEqualityIndicator x y

noncomputable def ghostPairBoostStateTerm (s : Fin 2 → Fin 2) : ℤ :=
  ∑ i : Fin 2,
    ∑ j : Fin 2,
      if completeInteracts i j ∧ i < j then
        if i = fin2_0 ∧ j = fin2_1 then ghostPairBoostBinary (s i) (s j) else 0
      else 0

noncomputable def translatedNeverOptimalGhostUtility : Fin 3 → (Fin 2 → Fin 2) → ℤ :=
  fun a s => neverOptimalGhostUtility 0 a s + ghostPairBoostStateTerm s

noncomputable def translatedNeverOptimalGhostPairwise :
    PairwiseUtility translatedNeverOptimalGhostUtility where
  unary i a x := (neverOptimalGhostPairwise 0).unary i a x
  binary i j a x y :=
    (neverOptimalGhostPairwise 0).binary i j a x y +
      if i = fin2_0 ∧ j = fin2_1 then ghostPairBoostBinary x y else 0
  interacts := @completeInteracts 2
  interacts_symm := @completeInteracts_symm 2
  decomp := by
    intro a s
    simp [translatedNeverOptimalGhostUtility, ghostPairBoostStateTerm,
      neverOptimalGhostPairwise, neverOptimalGhostUtility, completeInteracts]
    ring

noncomputable def translatedNeverOptimalGhostSlice : Slice where
  Action := Fin 3
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := 2
  utility := translatedNeverOptimalGhostUtility
  pairwise := translatedNeverOptimalGhostPairwise

noncomputable def translatedNeverOptimalGhostPositiveAffineWitness :
    PositiveAffineWitness (neverOptimalGhostSlice 0) translatedNeverOptimalGhostSlice where
  hArity := rfl
  relabel := Equiv.refl (Fin 3)
  alpha := ghostPairBoostStateTerm
  beta := fun _ => 1
  beta_pos := by intro _; decide
  utility_eq := by
    intro a s
    simp [translatedNeverOptimalGhostSlice, translatedNeverOptimalGhostUtility,
      neverOptimalGhostSlice, castState]
    ring

theorem translatedNeverOptimalGhostSlice_in_orbit :
    ClosureEquivalent (neverOptimalGhostSlice 0) translatedNeverOptimalGhostSlice := by
  exact closureEquivalent_of_positiveAffineWitness translatedNeverOptimalGhostPositiveAffineWitness

theorem neverOptimalGhostSlice0_hasGhostActionTwoPairCrossOne :
    GhostActionTwoPairCrossOneSlice (neverOptimalGhostSlice 0) := by
  refine ⟨by decide, (2 : Fin 3), ?_, ?_, ?_⟩
  · simp [neverOptimalGhostSlice, neverOptimalGhostPairwise, firstFin]
  · simp [neverOptimalGhostSlice, neverOptimalGhostPairwise, firstFin]
  simp [GhostActionTwoPairCrossOneSlice, pairCrossMagnitude,
    neverOptimalGhostSlice, neverOptimalGhostPairwise, firstFin, secondFin,
    twoActionFin, binaryCrossDifference, completePairIndicator, fin2_0, fin2_1]

theorem translatedNeverOptimalGhost_not_hasGhostActionTwoPairCrossOne :
    ¬ GhostActionTwoPairCrossOneSlice translatedNeverOptimalGhostSlice := by
  intro h
  rcases h with ⟨hAr, a, hu0, hu1, hCross⟩
  have hFirst : firstFin translatedNeverOptimalGhostSlice.arity hAr = fin2_0 := by
    apply Fin.ext
    simp [firstFin, translatedNeverOptimalGhostSlice, fin2_0]
  have hSecond : secondFin translatedNeverOptimalGhostSlice.arity hAr = fin2_1 := by
    apply Fin.ext
    simp [secondFin, translatedNeverOptimalGhostSlice, fin2_1]
  have hu0' : translatedNeverOptimalGhostSlice.pairwise.unary fin2_0 a 0 = -1 := by
    simpa [hFirst] using hu0
  have hu1' : translatedNeverOptimalGhostSlice.pairwise.unary fin2_0 a 1 = -1 := by
    simpa [hFirst] using hu1
  have hCross' :
      pairCrossMagnitude translatedNeverOptimalGhostSlice fin2_0 fin2_1 a = 1 := by
    simpa [hFirst, hSecond] using hCross
  fin_cases a <;>
    simp [translatedNeverOptimalGhostSlice, translatedNeverOptimalGhostPairwise,
      neverOptimalGhostPairwise, pairCrossMagnitude, ghostPairBoostBinary,
      binaryCrossDifference, signedEqualityIndicator, completePairIndicator,
      fin2_0, fin2_1] at hu0' hu1' hCross'

def ghostActionClosureOrbitGap : Prop :=
  ClosureEquivalent (neverOptimalGhostSlice 0) translatedNeverOptimalGhostSlice ∧
    GhostActionTwoPairCrossOneSlice (neverOptimalGhostSlice 0) ∧
    ¬ GhostActionTwoPairCrossOneSlice translatedNeverOptimalGhostSlice

theorem ghostActionClosureOrbitGap_exists : ghostActionClosureOrbitGap := by
  exact ⟨translatedNeverOptimalGhostSlice_in_orbit,
    neverOptimalGhostSlice0_hasGhostActionTwoPairCrossOne,
    translatedNeverOptimalGhost_not_hasGhostActionTwoPairCrossOne⟩

def offsetPairBoostBinary (x y : Fin 2) : ℤ :=
  2 * signedEqualityIndicator x y

noncomputable def offsetPairBoostStateTerm (s : Fin 2 → Fin 2) : ℤ :=
  ∑ i : Fin 2,
    ∑ j : Fin 2,
      if completeInteracts i j ∧ i < j then
        if i = fin2_0 ∧ j = fin2_1 then offsetPairBoostBinary (s i) (s j) else 0
      else 0

noncomputable def translatedOffsetPairBoostUtility : Bool → (Fin 2 → Fin 2) → ℤ :=
  fun a s => offsetCollapsedAsymmetricPairUtility 0 a s + offsetPairBoostStateTerm s

noncomputable def translatedOffsetPairBoostPairwise : PairwiseUtility translatedOffsetPairBoostUtility where
  unary i a x := (offsetCollapsedAsymmetricPairPairwise 0).unary i a x
  binary i j a x y :=
    (offsetCollapsedAsymmetricPairPairwise 0).binary i j a x y +
      if i = fin2_0 ∧ j = fin2_1 then offsetPairBoostBinary x y else 0
  interacts := @completeInteracts 2
  interacts_symm := @completeInteracts_symm 2
  decomp := by
    intro a s
    cases a <;>
      simp [translatedOffsetPairBoostUtility, offsetPairBoostStateTerm,
        offsetCollapsedAsymmetricPairPairwise, offsetCollapsedAsymmetricPairUtility,
        addActionOffset, offsetBaseAsymmetricPairUtility, completeInteracts] <;>
      ring

noncomputable def translatedOffsetPairBoostSlice : Slice where
  Action := Bool
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := 2
  utility := translatedOffsetPairBoostUtility
  pairwise := translatedOffsetPairBoostPairwise

noncomputable def translatedOffsetPairBoostPositiveAffineWitness :
    PositiveAffineWitness (offsetCollapsedSlice 0) translatedOffsetPairBoostSlice where
  hArity := rfl
  relabel := Equiv.refl Bool
  alpha := offsetPairBoostStateTerm
  beta := fun _ => 1
  beta_pos := by intro _; decide
  utility_eq := by
    intro a s
    simp [translatedOffsetPairBoostSlice, translatedOffsetPairBoostUtility,
      offsetCollapsedSlice, castState]
    ring

theorem translatedOffsetPairBoostSlice_in_orbit :
    ClosureEquivalent (offsetCollapsedSlice 0) translatedOffsetPairBoostSlice := by
  exact closureEquivalent_of_positiveAffineWitness translatedOffsetPairBoostPositiveAffineWitness

theorem offsetCollapsedSlice0_hasOffsetActionZeroPairCrossOne :
    OffsetActionZeroPairCrossOneSlice (offsetCollapsedSlice 0) := by
  refine ⟨by decide, false, true, by simp, ?_, ?_⟩
  · simp [OffsetActionZeroPairCrossOneSlice, pairCrossMagnitude,
      offsetCollapsedSlice, offsetCollapsedAsymmetricPairPairwise,
      firstFin, secondFin, binaryCrossDifference, completePairIndicator,
      fin2_0, fin2_1]
  · simp [OffsetActionZeroPairCrossOneSlice, pairCrossMagnitude,
      offsetCollapsedSlice, offsetCollapsedAsymmetricPairPairwise,
      firstFin, secondFin, binaryCrossDifference, completePairIndicator,
      fin2_0, fin2_1]

theorem translatedOffsetPairBoost_not_hasOffsetActionZeroPairCrossOne :
    ¬ OffsetActionZeroPairCrossOneSlice translatedOffsetPairBoostSlice := by
  intro h
  rcases h with ⟨hAr, a, b, hab, hCrossA, hCrossB⟩
  have hFirst : firstFin translatedOffsetPairBoostSlice.arity hAr = fin2_0 := by
    apply Fin.ext
    simp [firstFin, translatedOffsetPairBoostSlice, fin2_0]
  have hSecond : secondFin translatedOffsetPairBoostSlice.arity hAr = fin2_1 := by
    apply Fin.ext
    simp [secondFin, translatedOffsetPairBoostSlice, fin2_1]
  have hCrossA' : pairCrossMagnitude translatedOffsetPairBoostSlice fin2_0 fin2_1 a = 1 := by
    simpa [hFirst, hSecond] using hCrossA
  have hCrossB' : pairCrossMagnitude translatedOffsetPairBoostSlice fin2_0 fin2_1 b = 0 := by
    simpa [hFirst, hSecond] using hCrossB
  cases a <;> cases b <;> simp at hab
  · simp [pairCrossMagnitude, translatedOffsetPairBoostSlice,
      translatedOffsetPairBoostPairwise, offsetCollapsedAsymmetricPairPairwise,
      offsetPairBoostBinary, binaryCrossDifference, signedEqualityIndicator,
      completePairIndicator, fin2_0, fin2_1] at hCrossA'
  · simp [pairCrossMagnitude, translatedOffsetPairBoostSlice,
      translatedOffsetPairBoostPairwise, offsetCollapsedAsymmetricPairPairwise,
      offsetPairBoostBinary, binaryCrossDifference, signedEqualityIndicator,
      completePairIndicator, fin2_0, fin2_1] at hCrossA'

def offsetActionClosureOrbitGap : Prop :=
  ClosureEquivalent (offsetCollapsedSlice 0) translatedOffsetPairBoostSlice ∧
    OffsetActionZeroPairCrossOneSlice (offsetCollapsedSlice 0) ∧
    ¬ OffsetActionZeroPairCrossOneSlice translatedOffsetPairBoostSlice

theorem offsetActionClosureOrbitGap_exists : offsetActionClosureOrbitGap := by
  exact ⟨translatedOffsetPairBoostSlice_in_orbit,
    offsetCollapsedSlice0_hasOffsetActionZeroPairCrossOne,
    translatedOffsetPairBoost_not_hasOffsetActionZeroPairCrossOne⟩

theorem no_closureInvariant_predicate_decides_ghostActionTwoPairCrossOne :
    ∀ (P : Slice → Prop), ClosureLawInvariant P →
      ¬ (∀ S, P S ↔ GhostActionTwoPairCrossOneSlice S) := by
  intro P hInv hDecides
  rcases ghostActionClosureOrbitGap_exists with ⟨hOrbit, hBase, hTransNot⟩
  have hPOrbit : P (neverOptimalGhostSlice 0) ↔ P translatedNeverOptimalGhostSlice :=
    closureLawInvariant_iff_of_closureEquivalent hInv hOrbit
  have hPBase : P (neverOptimalGhostSlice 0) :=
    (hDecides (neverOptimalGhostSlice 0)).2 hBase
  have hPTrans : P translatedNeverOptimalGhostSlice := hPOrbit.mp hPBase
  have hTarget : GhostActionTwoPairCrossOneSlice translatedNeverOptimalGhostSlice :=
    (hDecides translatedNeverOptimalGhostSlice).1 hPTrans
  exact hTransNot hTarget

theorem no_closureInvariant_predicate_decides_offsetActionZeroPairCrossOne :
    ∀ (P : Slice → Prop), ClosureLawInvariant P →
      ¬ (∀ S, P S ↔ OffsetActionZeroPairCrossOneSlice S) := by
  intro P hInv hDecides
  rcases offsetActionClosureOrbitGap_exists with ⟨hOrbit, hBase, hTransNot⟩
  have hPOrbit : P (offsetCollapsedSlice 0) ↔ P translatedOffsetPairBoostSlice :=
    closureLawInvariant_iff_of_closureEquivalent hInv hOrbit
  have hPBase : P (offsetCollapsedSlice 0) :=
    (hDecides (offsetCollapsedSlice 0)).2 hBase
  have hPTrans : P translatedOffsetPairBoostSlice := hPOrbit.mp hPBase
  have hTarget : OffsetActionZeroPairCrossOneSlice translatedOffsetPairBoostSlice :=
    (hDecides translatedOffsetPairBoostSlice).1 hPTrans
  exact hTransNot hTarget

theorem no_admissibleNormalizationPredicate_decides_ghostActionTwoPairCrossOne :
    ∀ P : AdmissibleNormalizationPredicate,
      ¬ (∀ S : Slice, P.holdsOnSlice S ↔ GhostActionTwoPairCrossOneSlice S) := by
  intro P
  exact no_closureInvariant_predicate_decides_ghostActionTwoPairCrossOne
    P.holdsOnSlice P.closureLawInvariant

theorem no_admissibleNormalizationPredicate_decides_offsetActionZeroPairCrossOne :
    ∀ P : AdmissibleNormalizationPredicate,
      ¬ (∀ S : Slice, P.holdsOnSlice S ↔ OffsetActionZeroPairCrossOneSlice S) := by
  intro P
  exact no_closureInvariant_predicate_decides_offsetActionZeroPairCrossOne
    P.holdsOnSlice P.closureLawInvariant

def admissibleCollapseLandscapeInfinityOnGhostActionPairCrossOne : Prop :=
  ∀ P : AdmissibleNormalizationPredicate,
    ¬ (∀ S : Slice, P.holdsOnSlice S ↔ GhostActionTwoPairCrossOneSlice S)

def admissibleCollapseLandscapeInfinityOnOffsetActionPairCrossOne : Prop :=
  ∀ P : AdmissibleNormalizationPredicate,
    ¬ (∀ S : Slice, P.holdsOnSlice S ↔ OffsetActionZeroPairCrossOneSlice S)

theorem admissibleCollapseLandscapeInfinityOnGhostActionPairCrossOne_holds :
    admissibleCollapseLandscapeInfinityOnGhostActionPairCrossOne := by
  exact no_admissibleNormalizationPredicate_decides_ghostActionTwoPairCrossOne

theorem admissibleCollapseLandscapeInfinityOnOffsetActionPairCrossOne_holds :
    admissibleCollapseLandscapeInfinityOnOffsetActionPairCrossOne := by
  exact no_admissibleNormalizationPredicate_decides_offsetActionZeroPairCrossOne

theorem admissibleCollapseLandscapeInfinity_full :
    ∀ P : AdmissibleNormalizationPredicate,
      ¬ ((∀ S : Slice, P.holdsOnSlice S ↔ HasUniqueDominantPair S) ∧
        (∀ S : Slice, P.holdsOnSlice S ↔ MarginBoundedSlice S) ∧
        (∀ S : Slice, P.holdsOnSlice S ↔ GhostActionTwoPairCrossOneSlice S) ∧
        (∀ S : Slice, P.holdsOnSlice S ↔ OffsetActionZeroPairCrossOneSlice S)) := by
  intro P hAll
  have hDom : False :=
    (no_admissibleNormalizationPredicate_decides_dominantPair P) hAll.1
  have hMargin : False :=
    (no_admissibleNormalizationPredicate_decides_marginBounded P) hAll.2.1
  have hGhost : False :=
    (no_admissibleNormalizationPredicate_decides_ghostActionTwoPairCrossOne P) hAll.2.2.1
  have hOffset : False :=
    (no_admissibleNormalizationPredicate_decides_offsetActionZeroPairCrossOne P) hAll.2.2.2
  exact hDom

def AdmissibleNormalizationPredicateBridge (P : NormalizationPredicate) : Prop :=
  ∃ A : AdmissibleNormalizationPredicate,
    A.toNormalizationPredicate = P

def TractabilityCharacterizationBridge (P : NormalizationPredicate) : Prop :=
  ∃ A : AdmissibleNormalizationPredicate,
    A.toNormalizationPredicate = P ∧
      ((∀ S : Slice, A.holdsOnSlice S ↔ HasUniqueDominantPair S) ∧
        (∀ S : Slice, A.holdsOnSlice S ↔ MarginBoundedSlice S) ∧
        (∀ S : Slice, A.holdsOnSlice S ↔ GhostActionTwoPairCrossOneSlice S) ∧
        (∀ S : Slice, A.holdsOnSlice S ↔ OffsetActionZeroPairCrossOneSlice S))

local notation "AdmissibleNormalizationPredicate" => AdmissibleNormalizationPredicateBridge
local notation "TractabilityCharacterization" => TractabilityCharacterizationBridge

theorem admissibleCollapseLandscapeInfinity_full_paper :
    ∀ (P : NormalizationPredicate), AdmissibleNormalizationPredicate P →
      ¬ TractabilityCharacterization P := by
  intro P hAdm hTract
  rcases hAdm with ⟨_, _⟩
  rcases hTract with ⟨A, _hAeq, hAll⟩
  exact (admissibleCollapseLandscapeInfinity_full A) hAll

theorem denseDecisionRelevantPredicate_polynomialTimeCheckable :
    PolynomialTimeCheckable DenseDecisionRelevantSlice := by
  classical
  exact polynomialTimeCheckable_of_decidable DenseDecisionRelevantSlice

theorem marginBoundedDenseDecisionRelevantPredicate_polynomialTimeCheckable :
    PolynomialTimeCheckable MarginBoundedDenseDecisionRelevantSlice := by
  classical
  exact polynomialTimeCheckable_of_decidable MarginBoundedDenseDecisionRelevantSlice

end Paper4dFrontier
