import Paper4dFrontier.ComputeCostPayloads
import Paper4dFrontier.ObstructionPredicateCandidates

namespace Paper4dFrontier

open DecisionQuotient

def optimizerComputationPolytime (U : BinaryPairwiseSlice) : Prop :=
  optimizerClosureTransportFamily.family.PolytimePredicate U

theorem optimizerComputation_polytime_closureLawInvariant :
    ClosureLawInvariant optimizerComputationPolytime := by
  simpa [optimizerComputationPolytime] using
    optimizerClosureTransportFamily.polytimePredicate_closureLawInvariant

theorem optimizerSetPayload_polytime_closureLawInvariant :
    ClosureLawInvariant optimizerSetPayloadClosureTransportFamily.family.PolytimePredicate := by
  exact optimizerSetPayloadClosureTransportFamily.polytimePredicate_closureLawInvariant

theorem optimizerSetSearch_polytime_closureLawInvariant :
    ClosureLawInvariant optimizerSetSearchClosureTransportFamily.family.PolytimePredicate := by
  exact optimizerSetSearchClosureTransportFamily.polytimePredicate_closureLawInvariant

theorem optimizerComputation_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D optimizerComputationPolytime C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  exact optimizerClosureTransportFamily.compute_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hCorrect hDU hEqv

theorem optimizerComputation_polytime_correctnessForcesOrbitAgreementOnDomain
    {D : BinaryPairwiseSlice → Prop} (hClosed : ClosureClosedDomain D) :
    CorrectnessForcesOrbitAgreementOnDomain D optimizerComputationPolytime := by
  intro C hCorrect U V hDU hEqv
  exact optimizerComputation_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hCorrect hDU hEqv

theorem optimizerComputation_polytime_correct_classifier_onDomain_iff_no_orbitGapOn
    {D : BinaryPairwiseSlice → Prop} (hClosed : ClosureClosedDomain D) :
    (∃ C : BinaryPairwiseSlice → Prop, CorrectOnDomain D optimizerComputationPolytime C) ↔
      ¬ OrbitGapOn D optimizerComputationPolytime := by
  exact correct_classifier_onDomain_iff_no_orbitGapOn_of_forcedOrbitAgreement
    hClosed (optimizerComputation_polytime_correctnessForcesOrbitAgreementOnDomain hClosed)

theorem no_correct_optimizerComputation_polytime_classifier_onDomain_iff_orbitGapOn
    {D : BinaryPairwiseSlice → Prop} (hClosed : ClosureClosedDomain D) :
    (¬ ∃ C : BinaryPairwiseSlice → Prop, CorrectOnDomain D optimizerComputationPolytime C) ↔
      OrbitGapOn D optimizerComputationPolytime := by
  exact no_correct_classifier_onDomain_iff_orbitGapOn_of_forcedOrbitAgreement
    hClosed (optimizerComputation_polytime_correctnessForcesOrbitAgreementOnDomain hClosed)

theorem optimizerComputation_polytime_classifier_agrees_on_closureEquivalent
    {C : BinaryPairwiseSlice → Prop}
    (hCorrect : ∀ U : BinaryPairwiseSlice, C U ↔ optimizerComputationPolytime U)
    {U V : BinaryPairwiseSlice} (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  exact optimizerComputation_polytime_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (D := fun _ => True) (C := C)
    (by intro U V _ _; trivial) (fun _ _ => hCorrect _) trivial hEqv

theorem no_correct_optimizerComputation_polytime_classifier_decides_dominantPair
    {C : BinaryPairwiseSlice → Prop}
    (hCorrect : ∀ U : BinaryPairwiseSlice, C U ↔ optimizerComputationPolytime U) :
    ¬ (∀ S : BinaryPairwiseSlice, C S ↔ HasUniqueDominantPair S) := by
  have hClosed : ClosureClosedDomain (fun _ : BinaryPairwiseSlice => True) := by
    intro U V _ _
    trivial
  rcases dominantPairClosureOrbitGap_exists with ⟨hEqv, hDom01, _hDom12⟩
  have hQ0 : HasUniqueDominantPair (dominantPairSlice 0) := by
    exact dominantPairSlice0_hasUniqueDominantPair
  have hQ1 : ¬ HasUniqueDominantPair translatedDominantPairOrbitSlice :=
    translatedDominantPairOrbit_not_hasUniqueDominantPair
  intro hDecides
  have hDecides' : ∀ S : BinaryPairwiseSlice, True → (C S ↔ HasUniqueDominantPair S) := by
    intro S _
    exact hDecides S
  exact optimizerClosureTransportFamily.no_correctOnDomain_compute_classifier_of_orbit_gap
    (D := fun _ : BinaryPairwiseSlice => True)
    (C := C)
    (Q := HasUniqueDominantPair)
    hClosed
    (fun _ _ => hCorrect _)
    trivial
    hEqv
    hQ0
    hQ1
    hDecides'

theorem no_admissibleNormalizationPredicate_optimizerComputation_polytime_and_dominantPair
    (P : AdmissibleNormalizationPredicate) :
    ¬ ((∀ S : BinaryPairwiseSlice, P.holdsOnSlice S ↔ optimizerComputationPolytime S) ∧
      (∀ S : BinaryPairwiseSlice, P.holdsOnSlice S ↔ HasUniqueDominantPair S)) := by
  rintro ⟨hCompute, hDom⟩
  exact (no_correct_optimizerComputation_polytime_classifier_decides_dominantPair hCompute) hDom

end Paper4dFrontier
