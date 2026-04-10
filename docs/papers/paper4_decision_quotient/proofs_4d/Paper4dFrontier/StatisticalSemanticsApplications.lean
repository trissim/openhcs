import Paper4dFrontier.Realizability
import Paper4dFrontier.AdmissibleCharacterization

namespace Paper4dFrontier

theorem transferredSemantics_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    {D G T C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  have hCorrectT : CorrectOnDomain D T C := by
    intro W hDW
    exact (hCorrect hDW).trans (hTransfer W)
  exact classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hT hCorrectT hDU hEqv

theorem transferredSemantics_correctnessForcesOrbitAgreementOnDomain
    {D G T : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T) :
    CorrectnessForcesOrbitAgreementOnDomain D G := by
  intro C hCorrect U V hDU hEqv
  exact transferredSemantics_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hTransfer hT hCorrect hDU hEqv

theorem no_correctOnDomain_transferredSemantics_classifier_of_orbit_gap
    {D G T C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) := by
  have hCorrectT : CorrectOnDomain D T C := by
    intro W hDW
    exact (hCorrect hDW).trans (hTransfer W)
  exact no_correctOnDomain_classifier_of_orbit_gap
    hClosed hT hCorrectT hDU hEqv hQU hQV

theorem transferredSemantics_correct_classifier_onDomain_iff_no_orbitGapOn
    {D G T : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T) :
    (∃ C : BinaryPairwiseSlice → Prop, CorrectOnDomain D G C) ↔
      ¬ OrbitGapOn D G := by
  exact correct_classifier_onDomain_iff_no_orbitGapOn_of_forcedOrbitAgreement
    hClosed (transferredSemantics_correctnessForcesOrbitAgreementOnDomain
      hClosed hTransfer hT)

theorem statisticalGuarantee_classifier_agrees_on_closureEquivalent_of_correctOnDomain_of_transfer
    {D G T C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  exact transferredSemantics_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hTransfer hT hCorrect hDU hEqv

theorem statisticalGuarantee_correctnessForcesOrbitAgreementOnDomain_of_transfer
    {D G T : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T) :
    CorrectnessForcesOrbitAgreementOnDomain D G := by
  exact transferredSemantics_correctnessForcesOrbitAgreementOnDomain
    hClosed hTransfer hT

theorem no_correctOnDomain_statisticalGuarantee_classifier_of_orbit_gap_of_transfer
    {D G T C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) := by
  exact no_correctOnDomain_transferredSemantics_classifier_of_orbit_gap
    hClosed hTransfer hT hCorrect hDU hEqv hQU hQV

theorem statisticalGuarantee_correct_classifier_onDomain_iff_no_orbitGapOn_of_transfer
    {D G T : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T) :
    (∃ C : BinaryPairwiseSlice → Prop, CorrectOnDomain D G C) ↔
      ¬ OrbitGapOn D G := by
  exact transferredSemantics_correct_classifier_onDomain_iff_no_orbitGapOn
    hClosed hTransfer hT

theorem randomizedGuarantee_classifier_agrees_on_closureEquivalent_of_correctOnDomain_of_transfer
    {D G T C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  exact statisticalGuarantee_classifier_agrees_on_closureEquivalent_of_correctOnDomain_of_transfer
    hClosed hTransfer hT hCorrect hDU hEqv

theorem no_correctOnDomain_randomizedGuarantee_classifier_of_orbit_gap_of_transfer
    {D G T C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T)
    (hCorrect : CorrectOnDomain D G C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) := by
  exact no_correctOnDomain_statisticalGuarantee_classifier_of_orbit_gap_of_transfer
    hClosed hTransfer hT hCorrect hDU hEqv hQU hQV

theorem randomizedGuarantee_correct_classifier_onDomain_iff_no_orbitGapOn_of_transfer
    {D G T : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hTransfer : ∀ U : BinaryPairwiseSlice, G U ↔ T U)
    (hT : ClosureLawInvariant T) :
    (∃ C : BinaryPairwiseSlice → Prop, CorrectOnDomain D G C) ↔
      ¬ OrbitGapOn D G := by
  exact transferredSemantics_correct_classifier_onDomain_iff_no_orbitGapOn
    hClosed hTransfer hT

end Paper4dFrontier
