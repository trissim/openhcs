import Paper4dFrontier.ComputeCostApplications
import Paper4dFrontier.TransferredFullDomainNoGo

namespace Paper4dFrontier

/-- Canonical exact correctness specification attached to a binary pairwise slice:
the valid outputs at each state are exactly the optimizer sets of the slice. -/
def optimizerSetExactSpecification (U : BinaryPairwiseSlice) :
    ExactCorrectnessSpecification (SliceState U) where
  Output := optimizerSetPayloadType U
  valid := optimizerSetOutputRelation U

/-- The optimizer-set exact specification agrees with its induced decision problem
at the exact-relevance-profile level. This is the semantics-first form of the
canonical optimizer-set presentation. -/
theorem optimizerSetExactSpecification_exactness_means_exact_agreement
    (U : BinaryPairwiseSlice) :
    outputSemanticsExactRelevanceProfile
        (S := SliceState U) (A := optimizerSetPayloadType U) (n := U.arity)
        (optimizerSetExactSpecification U).valid =
      decisionProblemExactRelevanceProfile (n := U.arity)
        ((optimizerSetExactSpecification U).decisionProblem) := by
  simpa [optimizerSetExactSpecification] using
    (ExactCorrectnessSpecification.exactness_means_exact_agreement
      (S := SliceState U) (n := U.arity) (spec := optimizerSetExactSpecification U))

/-- A semantics-level classifier on exact correctness specifications. -/
abbrev ExactSpecificationClassifier : Type 1 :=
  ∀ {S : Type}, ExactCorrectnessSpecification S → Prop

/-- Restrict a semantics-level classifier to the canonical optimizer-set exact
specifications coming from binary pairwise slices. -/
def onCanonicalOptimizerSetExactSpecifications
    (G : ExactSpecificationClassifier) : BinaryPairwiseSlice → Prop :=
  fun U => G (optimizerSetExactSpecification U)

/-- Semantics-first no-go: once a predicate on exact correctness specifications
agrees on the canonical optimizer-set presentations with a closure-law-invariant
pairwise predicate, the full binary-pairwise no-go transfers to that semantic
predicate. The pairwise slices are only the witness class. -/
theorem no_correct_exactSpecification_classifier_on_fullBinaryPairwiseDomain_of_transfer
    {G : ExactSpecificationClassifier} {T C : BinaryPairwiseSlice → Prop}
    (hT : ClosureLawInvariant T)
    (hTransfer : ∀ U : BinaryPairwiseSlice,
      onCanonicalOptimizerSetExactSpecifications G U ↔ T U)
    (hCorrect : CorrectOnDomain fullBinaryPairwiseDomain
      (onCanonicalOptimizerSetExactSpecifications G) C) :
    ¬ DecidesFourObstructionFamilies C := by
  exact no_correct_transferredSemantics_classifier_on_fullBinaryPairwiseDomain
    hT hTransfer hCorrect

theorem no_exactSpecification_characterization_on_fullBinaryPairwiseDomain_of_transfer
    {G : ExactSpecificationClassifier} {T : BinaryPairwiseSlice → Prop}
    (hT : ClosureLawInvariant T)
    (hTransfer : ∀ U : BinaryPairwiseSlice,
      onCanonicalOptimizerSetExactSpecifications G U ↔ T U) :
    ¬ ∃ C : BinaryPairwiseSlice → Prop,
      CorrectOnDomain fullBinaryPairwiseDomain
        (onCanonicalOptimizerSetExactSpecifications G) C ∧
        DecidesFourObstructionFamilies C := by
  exact no_transferredSemantics_characterization_on_fullBinaryPairwiseDomain
    hT hTransfer

/-- Specialization to exact optimizer-set search semantics. If a semantics-level
classifier on exact specifications agrees with the canonical optimizer-set search
predicate on binary pairwise slices, the full-domain obstruction applies. -/
theorem no_correct_optimizerSetSearch_exactSpecification_classifier_on_fullBinaryPairwiseDomain
    {G : ExactSpecificationClassifier} {C : BinaryPairwiseSlice → Prop}
    (hTransfer : ∀ U : BinaryPairwiseSlice,
      onCanonicalOptimizerSetExactSpecifications G U ↔ optimizerSetSearchPolytime U)
    (hCorrect : CorrectOnDomain fullBinaryPairwiseDomain
      (onCanonicalOptimizerSetExactSpecifications G) C) :
    ¬ DecidesFourObstructionFamilies C := by
  exact no_correct_exactSpecification_classifier_on_fullBinaryPairwiseDomain_of_transfer
    optimizerSetSearch_polytime_closureLawInvariant hTransfer hCorrect

theorem no_optimizerSetSearch_exactSpecification_characterization_on_fullBinaryPairwiseDomain
    {G : ExactSpecificationClassifier}
    (hTransfer : ∀ U : BinaryPairwiseSlice,
      onCanonicalOptimizerSetExactSpecifications G U ↔ optimizerSetSearchPolytime U) :
    ¬ ∃ C : BinaryPairwiseSlice → Prop,
      CorrectOnDomain fullBinaryPairwiseDomain
        (onCanonicalOptimizerSetExactSpecifications G) C ∧
        DecidesFourObstructionFamilies C := by
  exact no_exactSpecification_characterization_on_fullBinaryPairwiseDomain_of_transfer
    optimizerSetSearch_polytime_closureLawInvariant hTransfer

end Paper4dFrontier
