import Paper4dFrontier.ComputeCostInvariance
import DecisionQuotient.StochasticSequential.Basic

namespace Paper4dFrontier

open Classical
open DecisionQuotient
open DecisionQuotient.StochasticSequential

namespace Distribution

noncomputable def map {A B : Type*} [Fintype A] [Fintype B]
    (f : A → B) (d : Distribution A) : Distribution B where
  pmf := fun b => ∑ a, (if f a = b then d.pmf a else 0 : ℝ)
  sum_eq_one := by
    classical
    calc
      ∑ b : B, ∑ a : A, (if f a = b then d.pmf a else 0 : ℝ)
          = ∑ a : A, ∑ b : B, (if f a = b then d.pmf a else 0 : ℝ) := by
              exact Finset.sum_comm
      _ = ∑ a : A, d.pmf a := by
            refine Finset.sum_congr rfl ?_
            intro a ha
            rw [Finset.sum_eq_single (f a)]
            · simp
            · intro b _ hne
              by_cases hEq : f a = b
              · exact False.elim (hne hEq.symm)
              · simp [hEq]
            · simpa using (Finset.mem_univ (f a))
      _ = 1 := d.sum_eq_one
  nonneg := by
    intro b
    refine Finset.sum_nonneg ?_
    intro a ha
    split_ifs <;> simp [d.nonneg a]

@[simp] theorem map_apply {A B : Type*} [Fintype A] [Fintype B]
    (f : A → B) (d : Distribution A) (b : B) :
    (map f d).pmf b = ∑ a, (if f a = b then d.pmf a else 0 : ℝ) := rfl

end Distribution

/-- A closure-respecting external-output semantics whose output type may vary with
the representation and is transported explicitly by each closure witness. -/
structure TransportedOutputClosureSpec where
  Output : BinaryPairwiseSlice → Type _
  admissible : ∀ U : BinaryPairwiseSlice, SliceState U → Output U → Prop
  action_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))
  coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))
  positive_affine : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))
  duplicate_action : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))
  duplicate_state : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))
  irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → SliceProblemEquiv
      (admissibleOutputSearchProblem (Output U) (admissible U))
      (admissibleOutputSearchProblem (Output V) (admissible V))

namespace TransportedOutputClosureSpec

def family (F : TransportedOutputClosureSpec) : SliceComputeFamily :=
  admissibleOutputSearchFamily F.Output F.admissible

def polytimePredicate (F : TransportedOutputClosureSpec) : BinaryPairwiseSlice → Prop :=
  F.family.PolytimePredicate

def toClosureTransportFamily (F : TransportedOutputClosureSpec) : ClosureTransportFamily where
  family := F.family
  action_relabel := F.action_relabel
  coordinate_relabel := F.coordinate_relabel
  positive_affine := F.positive_affine
  duplicate_action := F.duplicate_action
  duplicate_state := F.duplicate_state
  irrelevant_coordinate := F.irrelevant_coordinate

theorem polytimePredicate_closureLawInvariant (F : TransportedOutputClosureSpec) :
    ClosureLawInvariant F.polytimePredicate := by
  simpa [polytimePredicate] using (F.toClosureTransportFamily).polytimePredicate_closureLawInvariant

theorem classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (F : TransportedOutputClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  (F.toClosureTransportFamily).compute_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hCorrect hDU hEqv

theorem no_correctOnDomain_classifier_of_orbit_gap
    (F : TransportedOutputClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  (F.toClosureTransportFamily).no_correctOnDomain_compute_classifier_of_orbit_gap
    hClosed hCorrect hDU hEqv hQU hQV

end TransportedOutputClosureSpec

section StateFunctionOutputs

variable {X : Type}

def actionRelabelStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (castState h.hArity.symm s)

def coordinateRelabelStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (coordinateRelabelPullState h s)

def positiveAffineStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (castState h.hArity.symm s)

def duplicateActionStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (castState h.hArity.symm s)

def duplicateStateStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (h.projectState s)

def irrelevantCoordinateStateFunctionPush {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (f : SliceState U → X) : SliceState V → X :=
  fun s => f (h.projectState s)

structure StateFunctionOutputAdmissibilitySpec (X : Type) where
  admissible : ∀ U : BinaryPairwiseSlice, SliceState U → (SliceState U → X) → Prop
  action_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (castState h.hArity.symm s) f ↔
        admissible V s (actionRelabelStateFunctionPush h f)
  coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (coordinateRelabelPullState h s) f ↔
        admissible V s (coordinateRelabelStateFunctionPush h f)
  positive_affine : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (castState h.hArity.symm s) f ↔
        admissible V s (positiveAffineStateFunctionPush h f)
  duplicate_action : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (castState h.hArity.symm s) f ↔
        admissible V s (duplicateActionStateFunctionPush h f)
  duplicate_state : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (h.projectState s) f ↔
        admissible V s (duplicateStateStateFunctionPush h f)
  duplicate_state_factorable : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState V, ∀ g : SliceState V → X,
      admissible V s g →
        ∀ t₁ t₂ : SliceState V, h.projectState t₁ = h.projectState t₂ → g t₁ = g t₂
  irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState V, ∀ f : SliceState U → X,
      admissible U (h.projectState s) f ↔
        admissible V s (irrelevantCoordinateStateFunctionPush h f)
  irrelevant_coordinate_factorable : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState V, ∀ g : SliceState V → X,
      admissible V s g →
        ∀ t₁ t₂ : SliceState V, h.projectState t₁ = h.projectState t₂ → g t₁ = g t₂

namespace StateFunctionOutputAdmissibilitySpec

noncomputable def toTransportedOutputClosureSpec (F : StateFunctionOutputAdmissibilitySpec X) :
    TransportedOutputClosureSpec where
  Output := fun U => SliceState U → X
  admissible := F.admissible
  action_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun f => Counted.tick (actionRelabelStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.action_relabel h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun g => Counted.tick (fun s => g (castState h.hArity s))
            sound := by
              intro s g hg
              have hpush : actionRelabelStateFunctionPush h (fun t => g (castState h.hArity t)) = g := by
                funext t
                simp [actionRelabelStateFunctionPush, castState_symm_castState]
              have hg' : F.admissible V (castState h.hArity s)
                  (actionRelabelStateFunctionPush h (fun t => g (castState h.hArity t))) := by
                simpa [hpush] using hg
              simpa using
                (F.action_relabel h (castState h.hArity s) (fun t => g (castState h.hArity t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }
  coordinate_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
            pushOutput := fun f => Counted.tick (coordinateRelabelStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.coordinate_relabel h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
            pushOutput := fun g => Counted.tick (fun s => g (coordinateRelabelPushState h s))
            sound := by
              intro s g hg
              have hstate : coordinateRelabelPullState h (coordinateRelabelPushState h s) = s := by
                simp [coordinateRelabelPullState, coordinateRelabelPushState]
              have hpush : coordinateRelabelStateFunctionPush h (fun t => g (coordinateRelabelPushState h t)) = g := by
                funext t
                simp [coordinateRelabelStateFunctionPush, coordinateRelabelPullState, coordinateRelabelPushState]
              have hg' : F.admissible V (coordinateRelabelPushState h s)
                  (coordinateRelabelStateFunctionPush h (fun t => g (coordinateRelabelPushState h t))) := by
                simpa [hpush] using hg
              simpa [hstate] using
                (F.coordinate_relabel h (coordinateRelabelPushState h s)
                  (fun t => g (coordinateRelabelPushState h t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }
  positive_affine := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun f => Counted.tick (positiveAffineStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.positive_affine h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun g => Counted.tick (fun s => g (castState h.hArity s))
            sound := by
              intro s g hg
              have hpush : positiveAffineStateFunctionPush h (fun t => g (castState h.hArity t)) = g := by
                funext t
                simp [positiveAffineStateFunctionPush, castState_symm_castState]
              have hg' : F.admissible V (castState h.hArity s)
                  (positiveAffineStateFunctionPush h (fun t => g (castState h.hArity t))) := by
                simpa [hpush] using hg
              simpa using
                (F.positive_affine h (castState h.hArity s) (fun t => g (castState h.hArity t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }
  duplicate_action := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun f => Counted.tick (duplicateActionStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.duplicate_action h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun g => Counted.tick (fun s => g (castState h.hArity s))
            sound := by
              intro s g hg
              have hpush : duplicateActionStateFunctionPush h (fun t => g (castState h.hArity t)) = g := by
                funext t
                simp [duplicateActionStateFunctionPush, castState_symm_castState]
              have hg' : F.admissible V (castState h.hArity s)
                  (duplicateActionStateFunctionPush h (fun t => g (castState h.hArity t))) := by
                simpa [hpush] using hg
              simpa using
                (F.duplicate_action h (castState h.hArity s) (fun t => g (castState h.hArity t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }
  duplicate_state := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun f => Counted.tick (duplicateStateStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.duplicate_state h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun g => Counted.tick (fun s => g (h.sectionState s))
            sound := by
              intro s g hg
              have hpush : duplicateStateStateFunctionPush h (fun t => g (h.sectionState t)) = g := by
                funext t
                have hEq : h.projectState (h.sectionState (h.projectState t)) = h.projectState t := by
                  simp [h.project_sectionState]
                simpa [duplicateStateStateFunctionPush] using
                  (F.duplicate_state_factorable h (h.sectionState s) g hg _ _ hEq)
              have hg' : F.admissible V (h.sectionState s)
                  (duplicateStateStateFunctionPush h (fun t => g (h.sectionState t))) := by
                simpa [hpush] using hg
              simpa [h.project_sectionState] using
                (F.duplicate_state h (h.sectionState s) (fun t => g (h.sectionState t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }
  irrelevant_coordinate := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun f => Counted.tick (irrelevantCoordinateStateFunctionPush h f)
            sound := by
              intro s f hf
              exact (F.irrelevant_coordinate h s f).mp hf
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro f; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun g => Counted.tick (fun s => g (h.sectionState s))
            sound := by
              intro s g hg
              have hpush : irrelevantCoordinateStateFunctionPush h (fun t => g (h.sectionState t)) = g := by
                funext t
                have hEq : h.projectState (h.sectionState (h.projectState t)) = h.projectState t := by
                  simp [h.project_section]
                simpa [irrelevantCoordinateStateFunctionPush] using
                  (F.irrelevant_coordinate_factorable h (h.sectionState s) g hg _ _ hEq)
              have hg' : F.admissible V (h.sectionState s)
                  (irrelevantCoordinateStateFunctionPush h (fun t => g (h.sectionState t))) := by
                simpa [hpush] using hg
              simpa [h.project_section] using
                (F.irrelevant_coordinate h (h.sectionState s) (fun t => g (h.sectionState t))).mpr hg'
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro g; simp [Counted.tick]⟩ } }

end StateFunctionOutputAdmissibilitySpec

abbrev RepresentationRelativeHypothesisFunctionClosureSpec (X : Type) :=
  StateFunctionOutputAdmissibilitySpec X

abbrev RepresentationRelativeEstimatorFunctionClosureSpec (X : Type) :=
  StateFunctionOutputAdmissibilitySpec X

end StateFunctionOutputs

section PolicyOutputs

def actionRelabelPolicyPush {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.relabel (π (castState h.hArity.symm s))

def actionRelabelPolicyPull {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.relabel.symm (π (castState h.hArity s))

def coordinateRelabelPolicyPush {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.relabel (π (coordinateRelabelPullState h s))

def coordinateRelabelPolicyPull {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.relabel.symm (π (coordinateRelabelPushState h s))

def positiveAffinePolicyPush {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.relabel (π (castState h.hArity.symm s))

def positiveAffinePolicyPull {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.relabel.symm (π (castState h.hArity s))

noncomputable def duplicateActionPolicyPush {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.liftAction (π (castState h.hArity.symm s))

def duplicateActionPolicyPull {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.projectAction (π (castState h.hArity s))

def duplicateStatePolicyPush {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.relabel (π (h.projectState s))

noncomputable def duplicateStatePolicyPull {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.relabel.symm (π (h.sectionState s))

def irrelevantCoordinatePolicyPush {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (π : SliceState U → U.Action) : SliceState V → V.Action :=
  fun s => h.relabel (π (h.projectState s))

noncomputable def irrelevantCoordinatePolicyPull {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (π : SliceState V → V.Action) : SliceState U → U.Action :=
  fun s => h.relabel.symm (π (h.sectionState s))

structure PolicyOutputAdmissibilitySpec where
  admissible : ∀ U : BinaryPairwiseSlice, SliceState U → (SliceState U → U.Action) → Prop
  action_relabel_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (actionRelabelPolicyPush h π)
  action_relabel_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (actionRelabelPolicyPull h π)
  coordinate_relabel_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (coordinateRelabelPullState h s) π →
        admissible V s (coordinateRelabelPolicyPush h π)
  coordinate_relabel_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (coordinateRelabelPushState h s) π →
        admissible U s (coordinateRelabelPolicyPull h π)
  positive_affine_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (positiveAffinePolicyPush h π)
  positive_affine_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (positiveAffinePolicyPull h π)
  duplicate_action_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (duplicateActionPolicyPush h π)
  duplicate_action_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (duplicateActionPolicyPull h π)
  duplicate_state_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (h.projectState s) π →
        admissible V s (duplicateStatePolicyPush h π)
  duplicate_state_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (h.sectionState s) π →
        admissible U s (duplicateStatePolicyPull h π)
  irrelevant_coordinate_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState V, ∀ π : SliceState U → U.Action,
      admissible U (h.projectState s) π →
        admissible V s (irrelevantCoordinatePolicyPush h π)
  irrelevant_coordinate_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState U, ∀ π : SliceState V → V.Action,
      admissible V (h.sectionState s) π →
        admissible U s (irrelevantCoordinatePolicyPull h π)

namespace PolicyOutputAdmissibilitySpec

noncomputable def toTransportedOutputClosureSpec (F : PolicyOutputAdmissibilitySpec) :
    TransportedOutputClosureSpec where
  Output := fun U => SliceState U → U.Action
  admissible := F.admissible
  action_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (actionRelabelPolicyPush h π)
            sound := by intro s π hπ; exact F.action_relabel_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (actionRelabelPolicyPull h π)
            sound := by intro s π hπ; exact F.action_relabel_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  coordinate_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
            pushOutput := fun π => Counted.tick (coordinateRelabelPolicyPush h π)
            sound := by intro s π hπ; exact F.coordinate_relabel_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
            pushOutput := fun π => Counted.tick (coordinateRelabelPolicyPull h π)
            sound := by intro s π hπ; exact F.coordinate_relabel_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  positive_affine := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (positiveAffinePolicyPush h π)
            sound := by intro s π hπ; exact F.positive_affine_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (positiveAffinePolicyPull h π)
            sound := by intro s π hπ; exact F.positive_affine_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  duplicate_action := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (duplicateActionPolicyPush h π)
            sound := by intro s π hπ; exact F.duplicate_action_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (duplicateActionPolicyPull h π)
            sound := by intro s π hπ; exact F.duplicate_action_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  duplicate_state := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun π => Counted.tick (duplicateStatePolicyPush h π)
            sound := by intro s π hπ; exact F.duplicate_state_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun π => Counted.tick (duplicateStatePolicyPull h π)
            sound := by intro s π hπ; exact F.duplicate_state_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  irrelevant_coordinate := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun π => Counted.tick (irrelevantCoordinatePolicyPush h π)
            sound := by intro s π hπ; exact F.irrelevant_coordinate_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun π => Counted.tick (irrelevantCoordinatePolicyPull h π)
            sound := by intro s π hπ; exact F.irrelevant_coordinate_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }

end PolicyOutputAdmissibilitySpec

abbrev ConcretePolicyClosureSpec := PolicyOutputAdmissibilitySpec

end PolicyOutputs

section RandomizedProcedureOutputs

noncomputable def actionRelabelRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.relabel (π (castState h.hArity.symm s))

noncomputable def actionRelabelRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.relabel.symm (π (castState h.hArity s))

noncomputable def coordinateRelabelRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.relabel (π (coordinateRelabelPullState h s))

noncomputable def coordinateRelabelRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.relabel.symm (π (coordinateRelabelPushState h s))

noncomputable def positiveAffineRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.relabel (π (castState h.hArity.symm s))

noncomputable def positiveAffineRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.relabel.symm (π (castState h.hArity s))

noncomputable def duplicateActionRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.liftAction (π (castState h.hArity.symm s))

noncomputable def duplicateActionRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.projectAction (π (castState h.hArity s))

noncomputable def duplicateStateRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.relabel (π (h.projectState s))

noncomputable def duplicateStateRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.relabel.symm (π (h.sectionState s))

noncomputable def irrelevantCoordinateRandomizedProcedurePush {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V)
    (π : SliceState U → Distribution U.Action) : SliceState V → Distribution V.Action :=
  fun s => Distribution.map h.relabel (π (h.projectState s))

noncomputable def irrelevantCoordinateRandomizedProcedurePull {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V)
    (π : SliceState V → Distribution V.Action) : SliceState U → Distribution U.Action :=
  fun s => Distribution.map h.relabel.symm (π (h.sectionState s))

structure RandomizedProcedureOutputAdmissibilitySpec where
  admissible : ∀ U : BinaryPairwiseSlice,
    SliceState U → (SliceState U → Distribution U.Action) → Prop
  action_relabel_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (actionRelabelRandomizedProcedurePush h π)
  action_relabel_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : ActionRelabelWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (actionRelabelRandomizedProcedurePull h π)
  coordinate_relabel_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (coordinateRelabelPullState h s) π →
        admissible V s (coordinateRelabelRandomizedProcedurePush h π)
  coordinate_relabel_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : CoordinateRelabelWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (coordinateRelabelPushState h s) π →
        admissible U s (coordinateRelabelRandomizedProcedurePull h π)
  positive_affine_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (positiveAffineRandomizedProcedurePush h π)
  positive_affine_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : PositiveAffineWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (positiveAffineRandomizedProcedurePull h π)
  duplicate_action_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (castState h.hArity.symm s) π →
        admissible V s (duplicateActionRandomizedProcedurePush h π)
  duplicate_action_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateActionWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (castState h.hArity s) π →
        admissible U s (duplicateActionRandomizedProcedurePull h π)
  duplicate_state_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (h.projectState s) π →
        admissible V s (duplicateStateRandomizedProcedurePush h π)
  duplicate_state_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : DuplicateStateWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (h.sectionState s) π →
        admissible U s (duplicateStateRandomizedProcedurePull h π)
  irrelevant_coordinate_forward : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState V,
    ∀ π : SliceState U → Distribution U.Action,
      admissible U (h.projectState s) π →
        admissible V s (irrelevantCoordinateRandomizedProcedurePush h π)
  irrelevant_coordinate_backward : ∀ {U V : BinaryPairwiseSlice},
    (h : IrrelevantCoordinateWitness U V) → ∀ s : SliceState U,
    ∀ π : SliceState V → Distribution V.Action,
      admissible V (h.sectionState s) π →
        admissible U s (irrelevantCoordinateRandomizedProcedurePull h π)

namespace RandomizedProcedureOutputAdmissibilitySpec

noncomputable def toTransportedOutputClosureSpec (F : RandomizedProcedureOutputAdmissibilitySpec) :
    TransportedOutputClosureSpec where
  Output := fun U => SliceState U → Distribution U.Action
  admissible := F.admissible
  action_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (actionRelabelRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.action_relabel_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (actionRelabelRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.action_relabel_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  coordinate_relabel := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
            pushOutput := fun π => Counted.tick (coordinateRelabelRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.coordinate_relabel_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
            pushOutput := fun π => Counted.tick (coordinateRelabelRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.coordinate_relabel_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  positive_affine := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (positiveAffineRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.positive_affine_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (positiveAffineRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.positive_affine_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  duplicate_action := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (castState h.hArity.symm s)
            pushOutput := fun π => Counted.tick (duplicateActionRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.duplicate_action_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (castState h.hArity s)
            pushOutput := fun π => Counted.tick (duplicateActionRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.duplicate_action_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  duplicate_state := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun π => Counted.tick (duplicateStateRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.duplicate_state_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun π => Counted.tick (duplicateStateRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.duplicate_state_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }
  irrelevant_coordinate := by
    intro U V h
    refine
      { forward :=
          { pullState := fun s => Counted.tick (h.projectState s)
            pushOutput := fun π => Counted.tick (irrelevantCoordinateRandomizedProcedurePush h π)
            sound := by intro s π hπ; exact F.irrelevant_coordinate_forward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ }
        backward :=
          { pullState := fun s => Counted.tick (h.sectionState s)
            pushOutput := fun π => Counted.tick (irrelevantCoordinateRandomizedProcedurePull h π)
            sound := by intro s π hπ; exact F.irrelevant_coordinate_backward h s π hπ
            pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
            pushOutput_poly := ⟨1, 0, by intro π; simp [Counted.tick]⟩ } }

end RandomizedProcedureOutputAdmissibilitySpec

abbrev ConcreteRandomizedProcedureClosureSpec := RandomizedProcedureOutputAdmissibilitySpec

end RandomizedProcedureOutputs

abbrev RepresentationRelativeHypothesisClosureSpec := TransportedOutputClosureSpec
abbrev RepresentationRelativeEstimatorClosureSpec := TransportedOutputClosureSpec
abbrev RepresentationRelativePolicyClosureSpec := TransportedOutputClosureSpec
abbrev RepresentationRelativeRandomizedProcedureClosureSpec := TransportedOutputClosureSpec

def representationRelativeHypothesisOutputPolytime
    (H : RepresentationRelativeHypothesisClosureSpec) : BinaryPairwiseSlice → Prop :=
  H.polytimePredicate

def representationRelativeEstimatorOutputPolytime
    (E : RepresentationRelativeEstimatorClosureSpec) : BinaryPairwiseSlice → Prop :=
  E.polytimePredicate

def representationRelativePolicyOutputPolytime
    (P : RepresentationRelativePolicyClosureSpec) : BinaryPairwiseSlice → Prop :=
  P.polytimePredicate

def representationRelativeRandomizedProcedurePolytime
    (R : RepresentationRelativeRandomizedProcedureClosureSpec) : BinaryPairwiseSlice → Prop :=
  R.polytimePredicate

theorem representationRelativeHypothesis_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (H : RepresentationRelativeHypothesisClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeHypothesisOutputPolytime H) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  H.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem representationRelativeEstimator_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (E : RepresentationRelativeEstimatorClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeEstimatorOutputPolytime E) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  E.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem representationRelativePolicy_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (P : RepresentationRelativePolicyClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativePolicyOutputPolytime P) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  P.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem representationRelativeRandomizedProcedure_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (R : RepresentationRelativeRandomizedProcedureClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeRandomizedProcedurePolytime R) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  R.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed hCorrect hDU hEqv

theorem no_correctOnDomain_representationRelativeHypothesis_classifier_of_orbit_gap
    (H : RepresentationRelativeHypothesisClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeHypothesisOutputPolytime H) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  H.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_representationRelativeEstimator_classifier_of_orbit_gap
    (E : RepresentationRelativeEstimatorClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeEstimatorOutputPolytime E) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  E.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_representationRelativePolicy_classifier_of_orbit_gap
    (P : RepresentationRelativePolicyClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativePolicyOutputPolytime P) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  P.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

theorem no_correctOnDomain_representationRelativeRandomizedProcedure_classifier_of_orbit_gap
    (R : RepresentationRelativeRandomizedProcedureClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D (representationRelativeRandomizedProcedurePolytime R) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  R.no_correctOnDomain_classifier_of_orbit_gap hClosed hCorrect hDU hEqv hQU hQV

/-- A closure-respecting external-output semantics whose output object is unchanged
across representation changes; only the input state is transported. -/
structure IdentityOutputClosureSpec where
  Output : Type _
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

noncomputable def toTransportedOutputClosureSpec (F : IdentityOutputClosureSpec) :
    TransportedOutputClosureSpec where
  Output := fun _ => F.Output
  admissible := F.admissible
  action_relabel := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.actionRelabelEquiv h
  coordinate_relabel := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.coordinateRelabelEquiv h
  positive_affine := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.positiveAffineEquiv h
  duplicate_action := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.duplicateActionEquiv h
  duplicate_state := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.duplicateStateEquiv h
  irrelevant_coordinate := by
    intro U V h
    simpa [family, admissibleOutputSearchFamily, admissibleOutputSearchProblem] using
      F.irrelevantCoordinateEquiv h

theorem polytimePredicate_eq_transport (F : IdentityOutputClosureSpec) :
    F.polytimePredicate = (F.toTransportedOutputClosureSpec).polytimePredicate := by
  funext U
  rfl

theorem correctOnDomain_iff_correctOnDomain_transport
    (F : IdentityOutputClosureSpec) {D C : BinaryPairwiseSlice → Prop} :
    CorrectOnDomain D F.polytimePredicate C ↔
      CorrectOnDomain D ((F.toTransportedOutputClosureSpec).polytimePredicate) C := by
  rw [polytimePredicate_eq_transport]

theorem classifier_agrees_on_closureEquivalent_of_correctOnDomain_via_transport
    (F : IdentityOutputClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D ((F.toTransportedOutputClosureSpec).polytimePredicate) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  (F.toTransportedOutputClosureSpec).classifier_agrees_on_closureEquivalent_of_correctOnDomain
    hClosed hCorrect hDU hEqv

theorem no_correctOnDomain_classifier_of_orbit_gap_via_transport
    (F : IdentityOutputClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D ((F.toTransportedOutputClosureSpec).polytimePredicate) C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  (F.toTransportedOutputClosureSpec).no_correctOnDomain_classifier_of_orbit_gap
    hClosed hCorrect hDU hEqv hQU hQV

theorem classifier_agrees_on_closureEquivalent_of_correctOnDomain_eq_transport
    (F : IdentityOutputClosureSpec)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V := by
  exact classifier_agrees_on_closureEquivalent_of_correctOnDomain_via_transport
    F hClosed ((correctOnDomain_iff_correctOnDomain_transport F).mp hCorrect) hDU hEqv

theorem no_correctOnDomain_classifier_of_orbit_gap_eq_transport
    (F : IdentityOutputClosureSpec)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.polytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) := by
  exact no_correctOnDomain_classifier_of_orbit_gap_via_transport
    F hClosed ((correctOnDomain_iff_correctOnDomain_transport F).mp hCorrect)
    hDU hEqv hQU hQV

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
