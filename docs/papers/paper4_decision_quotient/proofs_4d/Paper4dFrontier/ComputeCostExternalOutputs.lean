import Paper4dFrontier.ComputeCostInvariance

namespace Paper4dFrontier

open Classical
open DecisionQuotient

/-- A closure-respecting external-output semantics whose output object is unchanged
across representation changes; only the input state is transported. -/
structure IdentityOutputClosureSpec where
  Output : Type
  admissible : ∀ U : BinaryPairwiseSlice, SliceState U → Output → Prop
  action_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (castState h.hArity.symm s) x ↔ admissible V s x
  coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (coordinateRelabelPullState h s) x ↔ admissible V s x
  positive_affine : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (castState h.hArity.symm s) x ↔ admissible V s x
  duplicate_action : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (castState h.hArity.symm s) x ↔ admissible V s x
  duplicate_state : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (h.projectState s) x ↔ admissible V s x
  irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState V, ∀ x : Output,
      admissible U (h.projectState s) x ↔ admissible V s x

namespace IdentityOutputClosureSpec

def family (F : IdentityOutputClosureSpec) : SliceComputeFamily :=
  admissibleOutputSearchFamily (fun _ => F.Output) (fun U => F.admissible U)

def polytimePredicate (F : IdentityOutputClosureSpec) : BinaryPairwiseSlice → Prop :=
  F.family.PolytimePredicate

def actionRelabelEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.action_relabel h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          castState_symm_castState] using
          (F.action_relabel h (castState h.hArity s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

def coordinateRelabelEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.coordinate_relabel h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          coordinateRelabelPullState, coordinateRelabelPushState] using
          (F.coordinate_relabel h (coordinateRelabelPushState h s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

def positiveAffineEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.positive_affine h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          castState_symm_castState] using
          (F.positive_affine h (castState h.hArity s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

def duplicateActionEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.duplicate_action h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          castState_symm_castState] using
          (F.duplicate_action h (castState h.hArity s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

noncomputable def duplicateStateEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.duplicate_state h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          h.project_sectionState] using
          (F.duplicate_state h (h.sectionState s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

noncomputable def irrelevantCoordinateEquiv (F : IdentityOutputClosureSpec) {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) :
    SliceProblemEquiv (F.family.problem U) (F.family.problem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
          (F.irrelevant_coordinate h s x).mp hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun x => Counted.tick x
      sound := by
        intro s x hx
        simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem,
          h.project_section] using
          (F.irrelevant_coordinate h (h.sectionState s) x).mpr hx
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro x; simp [Counted.tick]⟩ }

noncomputable def toClosureTransportFamily (F : IdentityOutputClosureSpec) : ClosureTransportFamily where
  family := F.family
  action_relabel := F.actionRelabelEquiv
  coordinate_relabel := F.coordinateRelabelEquiv
  positive_affine := F.positiveAffineEquiv
  duplicate_action := F.duplicateActionEquiv
  duplicate_state := F.duplicateStateEquiv
  irrelevant_coordinate := F.irrelevantCoordinateEquiv

theorem polytimePredicate_closureLawInvariant (F : IdentityOutputClosureSpec) :
    ClosureLawInvariant F.polytimePredicate := by
  simpa [polytimePredicate] using (F.toClosureTransportFamily).polytimePredicate_closureLawInvariant

theorem classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (F : IdentityOutputClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  (F.toClosureTransportFamily).compute_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hCorrect hDU hEqv

theorem no_correctOnDomain_classifier_of_orbit_gap
    (F : IdentityOutputClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  (F.toClosureTransportFamily).no_correctOnDomain_compute_classifier_of_orbit_gap
    hClosed hCorrect hDU hEqv hQU hQV

end IdentityOutputClosureSpec

abbrev HypothesisClosureSpec := IdentityOutputClosureSpec
abbrev EstimatorClosureSpec := IdentityOutputClosureSpec
abbrev PolicyClosureSpec := IdentityOutputClosureSpec
abbrev RandomizedProcedureClosureSpec := IdentityOutputClosureSpec

def hypothesisOutputPolytime (H : HypothesisClosureSpec) : BinaryPairwiseSlice → Prop :=
  H.polytimePredicate

def estimatorOutputPolytime (E : EstimatorClosureSpec) : BinaryPairwiseSlice → Prop :=
  E.polytimePredicate

def policyOutputPolytime (P : PolicyClosureSpec) : BinaryPairwiseSlice → Prop :=
  P.polytimePredicate

def randomizedProcedurePolytime (R : RandomizedProcedureClosureSpec) : BinaryPairwiseSlice → Prop :=
  R.polytimePredicate

theorem hypothesisOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (H : HypothesisClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (hypothesisOutputPolytime H) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  H.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem estimatorOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (E : EstimatorClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (estimatorOutputPolytime E) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  E.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem policyOutput_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (P : PolicyClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (policyOutputPolytime P) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  P.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem randomizedProcedure_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (R : RandomizedProcedureClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (randomizedProcedurePolytime R) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  R.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem no_correctOnDomain_hypothesisOutput_classifier_of_orbit_gap
    (H : HypothesisClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (hypothesisOutputPolytime H) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  H.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_estimatorOutput_classifier_of_orbit_gap
    (E : EstimatorClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (estimatorOutputPolytime E) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  E.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_policyOutput_classifier_of_orbit_gap
    (P : PolicyClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (policyOutputPolytime P) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  P.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_randomizedProcedure_classifier_of_orbit_gap
    (R : RandomizedProcedureClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (randomizedProcedurePolytime R) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  R.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

end Paper4dFrontier
