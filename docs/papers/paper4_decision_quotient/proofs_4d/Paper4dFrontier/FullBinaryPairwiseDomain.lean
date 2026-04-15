import Paper4dFrontier.AdmissibleCharacterization
import Paper4dFrontier.ObstructionPredicateCandidates

namespace Paper4dFrontier

def fullBinaryPairwiseDomain : BinaryPairwiseSlice → Prop := fun _ => True

theorem closure_sound_package_no_go_uses_only_closureLawInvariant
    {Γ : (BinaryPairwiseSlice → Prop) → Prop}
    (hΓ : ∀ {P : BinaryPairwiseSlice → Prop}, Γ P → ClosureLawInvariant P) :
    ∀ P : BinaryPairwiseSlice → Prop,
      Γ P →
        ¬ ((∀ S : BinaryPairwiseSlice, P S ↔ HasUniqueDominantPair S) ∧
          (∀ S : BinaryPairwiseSlice, P S ↔ MarginBoundedSlice S) ∧
          (∀ S : BinaryPairwiseSlice, P S ↔ GhostActionTwoPairCrossOneSlice S) ∧
          (∀ S : BinaryPairwiseSlice, P S ↔ OffsetActionZeroPairCrossOneSlice S)) := by
  intro P hP
  have hΓ' : ClosureSoundPackage Γ := by
    intro P hP
    exact hΓ hP
  exact no_closureSoundPackage_predicate_decides_four_obstruction_families hΓ' P hP

theorem fullBinaryPairwiseDomain_closure_closed :
    ClosureClosedDomain fullBinaryPairwiseDomain := by
  intro U V hU hEqv
  trivial

theorem fullBinaryPairwiseDomain_contains_four_obstruction_families :
    fullBinaryPairwiseDomain (dominantPairSlice 0) ∧
    fullBinaryPairwiseDomain (marginMaskingSlice 0) ∧
    fullBinaryPairwiseDomain (neverOptimalGhostSlice 0) ∧
    fullBinaryPairwiseDomain (offsetCollapsedSlice 0) := by
  simp [fullBinaryPairwiseDomain]

theorem no_correct_tractability_classifier_on_fullBinaryPairwiseDomain
    {T C : BinaryPairwiseSlice → Prop}
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain fullBinaryPairwiseDomain T C) :
    ¬ ((∀ S : BinaryPairwiseSlice, C S ↔ HasUniqueDominantPair S) ∧
      (∀ S : BinaryPairwiseSlice, C S ↔ MarginBoundedSlice S) ∧
      (∀ S : BinaryPairwiseSlice, C S ↔ GhostActionTwoPairCrossOneSlice S) ∧
      (∀ S : BinaryPairwiseSlice, C S ↔ OffsetActionZeroPairCrossOneSlice S)) := by
  rcases dominantPairClosureOrbitGap_exists with ⟨hOrbit, hDom01, _hDom12⟩
  have hBase : HasUniqueDominantPair (dominantPairSlice 0) := by
    exact dominantPairSlice0_hasUniqueDominantPair
  have hNotTrans : ¬ HasUniqueDominantPair translatedDominantPairOrbitSlice :=
    translatedDominantPairOrbit_not_hasUniqueDominantPair
  intro hDecides
  have hDecides' : ∀ S : BinaryPairwiseSlice, fullBinaryPairwiseDomain S → (C S ↔ HasUniqueDominantPair S) := by
    intro S _
    exact (hDecides.1 S)
  exact no_correctOnDomain_classifier_of_orbit_gap
    fullBinaryPairwiseDomain_closure_closed hT hCorrect trivial hOrbit hBase hNotTrans hDecides'

end Paper4dFrontier
