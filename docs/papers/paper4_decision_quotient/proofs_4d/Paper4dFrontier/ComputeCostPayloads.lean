import Paper4dFrontier.ComputeCostInvariance

namespace Paper4dFrontier

open Classical
open DecisionQuotient

/-- Canonical exact payload: return the optimizer set itself. -/
def optimizerSetPayloadType (U : BinaryPairwiseSlice) : Type :=
  Set U.Action

def optimizerSetPayload (U : BinaryPairwiseSlice) (s : SliceState U) : optimizerSetPayloadType U :=
  (U.toDecisionProblem).Opt s

def optimizerSetOutputRelation (U : BinaryPairwiseSlice)
    (s : SliceState U) (X : optimizerSetPayloadType U) : Prop :=
  X = optimizerSetPayload U s

def optimizerSetPayloadFamily : SliceComputeFamily :=
  deterministicPayloadFamily optimizerSetPayloadType optimizerSetPayload

def optimizerSetSearchFamily : SliceComputeFamily :=
  admissibleOutputSearchFamily optimizerSetPayloadType optimizerSetOutputRelation

def relabelOutputSet {A B : Type*} (e : A ≃ B) (X : Set A) : Set B :=
  e '' X

def duplicateActionForwardOutputSet {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (X : Set U.Action) : Set V.Action :=
  { b : V.Action | h.projectAction b ∈ X }

def duplicateActionBackwardOutputSet {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (Y : Set V.Action) : Set U.Action :=
  h.projectAction '' Y

theorem actionRelabelWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (s : SliceState V) :
    relabelOutputSet h.relabel ((U.toDecisionProblem).Opt (castState h.hArity.symm s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact (actionRelabelWitness_isOptimal_iff h a s).2 ha
  · intro hb
    refine ⟨h.relabel.symm b, ?_, by simp⟩
    exact (actionRelabelWitness_isOptimal_iff h (h.relabel.symm b) s).1 (by simpa using hb)

theorem actionRelabelWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (s : SliceState U) :
    relabelOutputSet h.relabel.symm ((V.toDecisionProblem).Opt (castState h.hArity s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, hbEq⟩
    have hbEq' : b = h.relabel a := by
      simpa using congrArg h.relabel hbEq
    have hb' : (V.toDecisionProblem).isOptimal (h.relabel a) (castState h.hArity s) := by
      simpa [DecisionProblem.Opt, hbEq'] using hb
    have hBack := (actionRelabelWitness_isOptimal_iff h a (castState h.hArity s)).1 hb'
    simpa [castState_symm_castState] using hBack
  · intro ha
    refine ⟨h.relabel a, ?_, by simp⟩
    have ha' : (U.toDecisionProblem).isOptimal a (castState h.hArity.symm (castState h.hArity s)) := by
      simpa [castState_symm_castState] using ha
    exact (actionRelabelWitness_isOptimal_iff h a (castState h.hArity s)).2 ha'

theorem coordinateRelabelWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (s : SliceState V) :
    relabelOutputSet h.relabel ((U.toDecisionProblem).Opt (coordinateRelabelPullState h s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact (coordinateRelabelWitness_isOptimal_iff h a s).2 ha
  · intro hb
    refine ⟨h.relabel.symm b, ?_, by simp⟩
    exact (coordinateRelabelWitness_isOptimal_iff h (h.relabel.symm b) s).1 (by simpa using hb)

theorem coordinateRelabelWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (s : SliceState U) :
    relabelOutputSet h.relabel.symm ((V.toDecisionProblem).Opt (coordinateRelabelPushState h s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, hbEq⟩
    have hbEq' : b = h.relabel a := by
      simpa using congrArg h.relabel hbEq
    have hb' : (V.toDecisionProblem).isOptimal (h.relabel a) (coordinateRelabelPushState h s) := by
      simpa [DecisionProblem.Opt, hbEq'] using hb
    have hBack :
        (U.toDecisionProblem).isOptimal (h.relabel.symm (h.relabel a))
          (coordinateRelabelPullState h (coordinateRelabelPushState h s)) :=
      (coordinateRelabelWitness_isOptimal_iff h (h.relabel.symm (h.relabel a))
        (coordinateRelabelPushState h s)).1 (by simpa using hb')
    simpa [coordinateRelabelPullState, coordinateRelabelPushState] using hBack
  · intro ha
    refine ⟨h.relabel a, ?_, by simp⟩
    exact (coordinateRelabelWitness_isOptimal_iff h a (coordinateRelabelPushState h s)).2
      (by simpa [coordinateRelabelPullState, coordinateRelabelPushState] using ha)

theorem positiveAffineWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (s : SliceState V) :
    relabelOutputSet h.relabel ((U.toDecisionProblem).Opt (castState h.hArity.symm s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact (positiveAffineWitness_isOptimal_iff h a s).2 ha
  · intro hb
    refine ⟨h.relabel.symm b, ?_, by simp⟩
    exact (positiveAffineWitness_isOptimal_iff h (h.relabel.symm b) s).1 (by simpa using hb)

theorem positiveAffineWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (s : SliceState U) :
    relabelOutputSet h.relabel.symm ((V.toDecisionProblem).Opt (castState h.hArity s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, hbEq⟩
    have hbEq' : b = h.relabel a := by
      simpa using congrArg h.relabel hbEq
    have hb' : (V.toDecisionProblem).isOptimal (h.relabel a) (castState h.hArity s) := by
      simpa [DecisionProblem.Opt, hbEq'] using hb
    have hBack := (positiveAffineWitness_isOptimal_iff h a (castState h.hArity s)).1 hb'
    simpa [castState_symm_castState] using hBack
  · intro ha
    refine ⟨h.relabel a, ?_, by simp⟩
    have ha' : (U.toDecisionProblem).isOptimal a (castState h.hArity.symm (castState h.hArity s)) := by
      simpa [castState_symm_castState] using ha
    exact (positiveAffineWitness_isOptimal_iff h a (castState h.hArity s)).2 ha'

theorem duplicateActionWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (b : V.Action) (s : SliceState U) :
    (V.toDecisionProblem).isOptimal b (castState h.hArity s) ↔
      (U.toDecisionProblem).isOptimal (h.projectAction b) s := by
  constructor
  · exact duplicateActionWitness_project_isOptimal h b s
  · intro hOpt b'
    calc
      (V.toDecisionProblem).utility b' (castState h.hArity s)
          = (U.toDecisionProblem).utility (h.projectAction b') s := by
              change ((V.utility b' (castState h.hArity s) : ℤ) : ℝ) =
                ((U.utility (h.projectAction b') s : ℤ) : ℝ)
              have hEq : V.utility b' (castState h.hArity s) = U.utility (h.projectAction b') s := by
                simpa using h.utility_eq b' s
              exact_mod_cast hEq
      _ ≤ (U.toDecisionProblem).utility (h.projectAction b) s := hOpt (h.projectAction b')
      _ = (V.toDecisionProblem).utility b (castState h.hArity s) := by
            change ((U.utility (h.projectAction b) s : ℤ) : ℝ) =
              ((V.utility b (castState h.hArity s) : ℤ) : ℝ)
            have hEq : V.utility b (castState h.hArity s) = U.utility (h.projectAction b) s := by
              simpa using h.utility_eq b s
            exact_mod_cast hEq.symm

theorem duplicateActionWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (s : SliceState V) :
    duplicateActionForwardOutputSet h ((U.toDecisionProblem).Opt (castState h.hArity.symm s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · intro hb
    have hb' := (duplicateActionWitness_isOptimal_iff h b (castState h.hArity.symm s)).2 hb
    simpa [DecisionProblem.Opt, castState_castState_symm] using hb'
  · intro hb
    have hb' : (V.toDecisionProblem).isOptimal b
        (castState h.hArity (castState h.hArity.symm s)) := by
      simpa [DecisionProblem.Opt, castState_castState_symm] using hb
    simpa [duplicateActionForwardOutputSet, DecisionProblem.Opt, castState_castState_symm] using
      (duplicateActionWitness_isOptimal_iff h b (castState h.hArity.symm s)).1 hb'

theorem duplicateActionWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (s : SliceState U) :
    duplicateActionBackwardOutputSet h ((V.toDecisionProblem).Opt (castState h.hArity s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, rfl⟩
    exact (duplicateActionWitness_isOptimal_iff h b s).1 hb
  · intro ha
    refine ⟨h.liftAction a, ?_, h.project_liftAction a⟩
    exact duplicateActionWitness_lift_isOptimal h a s ha

theorem duplicateStateWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).isOptimal (h.relabel a) s ↔
      (U.toDecisionProblem).isOptimal a (h.projectState s) := by
  constructor
  · intro hOpt a'
    calc
      (U.toDecisionProblem).utility a' (h.projectState s)
          = (V.toDecisionProblem).utility (h.relabel a') s := by
              change ((U.utility a' (h.projectState s) : ℤ) : ℝ) =
                ((V.utility (h.relabel a') s : ℤ) : ℝ)
              have hEq : V.utility (h.relabel a') s = U.utility a' (h.projectState s) := by
                simpa using h.utility_eq a' s
              exact_mod_cast hEq.symm
      _ ≤ (V.toDecisionProblem).utility (h.relabel a) s := hOpt (h.relabel a')
      _ = (U.toDecisionProblem).utility a (h.projectState s) := by
            change ((V.utility (h.relabel a) s : ℤ) : ℝ) =
              ((U.utility a (h.projectState s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
              simpa using h.utility_eq a s
            exact_mod_cast hEq
  · intro hOpt b'
    calc
      (V.toDecisionProblem).utility b' s
          = (U.toDecisionProblem).utility (h.relabel.symm b') (h.projectState s) := by
              change ((V.utility b' s : ℤ) : ℝ) =
                ((U.utility (h.relabel.symm b') (h.projectState s) : ℤ) : ℝ)
              have hEq : V.utility b' s = U.utility (h.relabel.symm b') (h.projectState s) := by
                simpa using h.utility_eq (h.relabel.symm b') s
              exact_mod_cast hEq
      _ ≤ (U.toDecisionProblem).utility a (h.projectState s) := hOpt (h.relabel.symm b')
      _ = (V.toDecisionProblem).utility (h.relabel a) s := by
            change ((U.utility a (h.projectState s) : ℤ) : ℝ) =
              ((V.utility (h.relabel a) s : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
              simpa using h.utility_eq a s
            exact_mod_cast hEq.symm

theorem duplicateStateWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (s : SliceState V) :
    relabelOutputSet h.relabel ((U.toDecisionProblem).Opt (h.projectState s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact (duplicateStateWitness_isOptimal_iff h a s).2 ha
  · intro hb
    refine ⟨h.relabel.symm b, ?_, by simp⟩
    exact (duplicateStateWitness_isOptimal_iff h (h.relabel.symm b) s).1 (by simpa using hb)

theorem duplicateStateWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (s : SliceState U) :
    relabelOutputSet h.relabel.symm ((V.toDecisionProblem).Opt (h.sectionState s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, hbEq⟩
    have hbEq' : b = h.relabel a := by
      simpa using congrArg h.relabel hbEq
    have hb' : (V.toDecisionProblem).isOptimal (h.relabel a) (h.sectionState s) := by
      simpa [DecisionProblem.Opt, hbEq'] using hb
    have hBack := (duplicateStateWitness_isOptimal_iff h a (h.sectionState s)).1 hb'
    simpa [h.project_sectionState] using hBack
  · intro ha
    refine ⟨h.relabel a, ?_, by simp⟩
    exact (duplicateStateWitness_isOptimal_iff h a (h.sectionState s)).2
      (by simpa [h.project_sectionState] using ha)

theorem irrelevantCoordinateWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).isOptimal (h.relabel a) s ↔
      (U.toDecisionProblem).isOptimal a (h.projectState s) := by
  constructor
  · intro hOpt a'
    calc
      (U.toDecisionProblem).utility a' (h.projectState s)
          = (V.toDecisionProblem).utility (h.relabel a') s := by
              change ((U.utility a' (h.projectState s) : ℤ) : ℝ) =
                ((V.utility (h.relabel a') s : ℤ) : ℝ)
              have hEq : V.utility (h.relabel a') s = U.utility a' (h.projectState s) := by
                simpa using h.utility_eq a' s
              exact_mod_cast hEq.symm
      _ ≤ (V.toDecisionProblem).utility (h.relabel a) s := hOpt (h.relabel a')
      _ = (U.toDecisionProblem).utility a (h.projectState s) := by
            change ((V.utility (h.relabel a) s : ℤ) : ℝ) =
              ((U.utility a (h.projectState s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
              simpa using h.utility_eq a s
            exact_mod_cast hEq
  · intro hOpt b'
    calc
      (V.toDecisionProblem).utility b' s
          = (U.toDecisionProblem).utility (h.relabel.symm b') (h.projectState s) := by
              change ((V.utility b' s : ℤ) : ℝ) =
                ((U.utility (h.relabel.symm b') (h.projectState s) : ℤ) : ℝ)
              have hEq : V.utility b' s = U.utility (h.relabel.symm b') (h.projectState s) := by
                simpa using h.utility_eq (h.relabel.symm b') s
              exact_mod_cast hEq
      _ ≤ (U.toDecisionProblem).utility a (h.projectState s) := hOpt (h.relabel.symm b')
      _ = (V.toDecisionProblem).utility (h.relabel a) s := by
            change ((U.utility a (h.projectState s) : ℤ) : ℝ) =
              ((V.utility (h.relabel a) s : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
              simpa using h.utility_eq a s
            exact_mod_cast hEq.symm

theorem irrelevantCoordinateWitness_optSet_eq {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (s : SliceState V) :
    relabelOutputSet h.relabel ((U.toDecisionProblem).Opt (h.projectState s)) =
      (V.toDecisionProblem).Opt s := by
  ext b
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact (irrelevantCoordinateWitness_isOptimal_iff h a s).2 ha
  · intro hb
    refine ⟨h.relabel.symm b, ?_, by simp⟩
    exact (irrelevantCoordinateWitness_isOptimal_iff h (h.relabel.symm b) s).1 (by simpa using hb)

theorem irrelevantCoordinateWitness_optSet_eq_symm {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (s : SliceState U) :
    relabelOutputSet h.relabel.symm ((V.toDecisionProblem).Opt (h.sectionState s)) =
      (U.toDecisionProblem).Opt s := by
  ext a
  constructor
  · rintro ⟨b, hb, hbEq⟩
    have hbEq' : b = h.relabel a := by
      simpa using congrArg h.relabel hbEq
    have hb' : (V.toDecisionProblem).isOptimal (h.relabel a) (h.sectionState s) := by
      simpa [DecisionProblem.Opt, hbEq'] using hb
    have hBack := (irrelevantCoordinateWitness_isOptimal_iff h a (h.sectionState s)).1 hb'
    simpa [h.project_section] using hBack
  · intro ha
    refine ⟨h.relabel a, ?_, by simp⟩
    exact (irrelevantCoordinateWitness_isOptimal_iff h a (h.sectionState s)).2
      (by simpa [h.project_section] using ha)

def optimizerSetPayloadActionRelabelEquiv {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun X => Counted.tick (relabelOutputSet h.relabel X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact actionRelabelWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun Y => Counted.tick (relabelOutputSet h.relabel.symm Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact actionRelabelWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

def optimizerSetPayloadCoordinateRelabelEquiv {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
      pushOutput := fun X => Counted.tick (relabelOutputSet h.relabel X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact coordinateRelabelWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
      pushOutput := fun Y => Counted.tick (relabelOutputSet h.relabel.symm Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact coordinateRelabelWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

def optimizerSetPayloadPositiveAffineEquiv {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun X => Counted.tick (relabelOutputSet h.relabel X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact positiveAffineWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun Y => Counted.tick (relabelOutputSet h.relabel.symm Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact positiveAffineWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

noncomputable def optimizerSetPayloadDuplicateActionEquiv {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun X => Counted.tick (duplicateActionForwardOutputSet h X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact duplicateActionWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun Y => Counted.tick (duplicateActionBackwardOutputSet h Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact duplicateActionWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

noncomputable def optimizerSetPayloadDuplicateStateEquiv {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun X => Counted.tick (relabelOutputSet h.relabel X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact duplicateStateWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun Y => Counted.tick (relabelOutputSet h.relabel.symm Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact duplicateStateWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

def optimizerSetPayloadIrrelevantCoordinateEquiv {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) :
    SliceProblemEquiv ((optimizerSetPayloadFamily).problem U) ((optimizerSetPayloadFamily).problem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun X => Counted.tick (relabelOutputSet h.relabel X)
      sound := by
        intro s X hX
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hX ⊢
        rw [hX]
        exact irrelevantCoordinateWitness_optSet_eq h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro X; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun Y => Counted.tick (relabelOutputSet h.relabel.symm Y)
      sound := by
        intro s Y hY
        simp [optimizerSetPayloadFamily, deterministicPayloadFamily,
          deterministicPayloadProblem, optimizerSetPayload] at hY ⊢
        rw [hY]
        exact irrelevantCoordinateWitness_optSet_eq_symm h s
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro Y; simp [Counted.tick]⟩ }

noncomputable def optimizerSetPayloadClosureTransportFamily : ClosureTransportFamily :=
  mkDeterministicPayloadClosureTransportFamily optimizerSetPayloadType optimizerSetPayload
    optimizerSetPayloadActionRelabelEquiv
    optimizerSetPayloadCoordinateRelabelEquiv
    optimizerSetPayloadPositiveAffineEquiv
    optimizerSetPayloadDuplicateActionEquiv
    optimizerSetPayloadDuplicateStateEquiv
    optimizerSetPayloadIrrelevantCoordinateEquiv

noncomputable def optimizerSetSearchClosureTransportFamily : ClosureTransportFamily :=
  mkAdmissibleOutputSearchClosureTransportFamily optimizerSetPayloadType optimizerSetOutputRelation
    optimizerSetPayloadActionRelabelEquiv
    optimizerSetPayloadCoordinateRelabelEquiv
    optimizerSetPayloadPositiveAffineEquiv
    optimizerSetPayloadDuplicateActionEquiv
    optimizerSetPayloadDuplicateStateEquiv
    optimizerSetPayloadIrrelevantCoordinateEquiv

end Paper4dFrontier
