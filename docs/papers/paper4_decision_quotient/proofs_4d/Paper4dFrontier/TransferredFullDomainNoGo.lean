import Paper4dFrontier.FullBinaryPairwiseDomain
import Paper4dFrontier.StatisticalSemanticsApplications

namespace Paper4dFrontier

/-- A classifier decides the four canonical obstruction families when it agrees
with each family predicate on every binary pairwise slice. -/
def DecidesFourObstructionFamilies (C : BinaryPairwiseSlice → Prop) : Prop :=
  (∀ S : BinaryPairwiseSlice, C S ↔ HasUniqueDominantPair S) ∧
    (∀ S : BinaryPairwiseSlice, C S ↔ MarginBoundedSlice S) ∧
    (∀ S : BinaryPairwiseSlice, C S ↔ GhostActionTwoPairCrossOneSlice S) ∧
    (∀ S : BinaryPairwiseSlice, C S ↔ OffsetActionZeroPairCrossOneSlice S)

theorem no_correct_transferredSemantics_classifier_on_fullBinaryPairwiseDomain
    {G T C : BinaryPairwiseSlice → Prop}
    (hT : ClosureLawInvariant T)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hCorrect : CorrectOnDomain fullBinaryPairwiseDomain G C) :
    ¬ DecidesFourObstructionFamilies C := by
  have hCorrectT : CorrectOnDomain fullBinaryPairwiseDomain T C := by
    intro U hU
    exact (hCorrect hU).trans (hTransfer U)
  exact no_correct_tractability_classifier_on_fullBinaryPairwiseDomain hT hCorrectT

theorem no_transferredSemantics_characterization_on_fullBinaryPairwiseDomain
    {G T : BinaryPairwiseSlice → Prop}
    (hT : ClosureLawInvariant T)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U) :
    ¬ ∃ C : BinaryPairwiseSlice → Prop,
      CorrectOnDomain fullBinaryPairwiseDomain G C ∧ DecidesFourObstructionFamilies C := by
  intro h
  rcases h with ⟨C, hCorrect, hDecides⟩
  exact (no_correct_transferredSemantics_classifier_on_fullBinaryPairwiseDomain
    hT hTransfer hCorrect) hDecides

end Paper4dFrontier
