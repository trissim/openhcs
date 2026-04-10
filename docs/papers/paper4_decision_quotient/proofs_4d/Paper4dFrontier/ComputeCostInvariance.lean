import Paper4dFrontier.DistinctActionProfiles
import Mathlib.Tactic

namespace Paper4dFrontier

open Classical
open DecisionQuotient

/-- A compute problem over the state space of a fixed slice. -/
structure SliceComputeProblem (U : BinaryPairwiseSlice) where
  Output : Type
  admissible : SliceState U → Output → Prop

/-- A counted solver for a fixed slice-level compute problem. -/
abbrev SliceSolver {U : BinaryPairwiseSlice} (P : SliceComputeProblem U) : Type :=
  SliceState U → Counted P.Output

def SliceComputeProblem.Solves {U : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) (solve : SliceSolver P) : Prop :=
  ∀ s : SliceState U, P.admissible s (solve s).result

def SliceComputeProblem.Polytime {U : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) (solve : SliceSolver P) : Prop :=
  ∃ c k : ℕ,
    ∀ s : SliceState U,
      (solve s).steps ≤ c * (U.encodingSize + 1) ^ k + c

def SliceComputeProblem.PolytimeSolvable {U : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) : Prop :=
  ∃ solve : SliceSolver P, P.Solves solve ∧ P.Polytime solve

theorem SliceComputeProblem.polytime_of_steps_le_const {U : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) (solve : SliceSolver P) (n : ℕ)
    (hbound : ∀ s : SliceState U, (solve s).steps ≤ n) :
    P.Polytime solve := by
  refine ⟨n, 0, ?_⟩
  intro s
  have hs := hbound s
  simp only [pow_zero, mul_one]
  omega

/-- A reduction transports target states back to source states and then transports
source outputs forward to target outputs. -/
structure SliceProblemReduction {U V : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) (Q : SliceComputeProblem V) where
  pullState : SliceState V → Counted (SliceState U)
  pushOutput : P.Output → Counted Q.Output
  sound : ∀ s : SliceState V, ∀ a : P.Output,
    P.admissible (pullState s).result a →
      Q.admissible s (pushOutput a).result
  pullState_poly : ∃ c k : ℕ,
    ∀ s : SliceState V,
      (pullState s).steps ≤ c * (V.encodingSize + 1) ^ k + c
  pushOutput_poly : ∃ c k : ℕ,
    ∀ a : P.Output,
      (pushOutput a).steps ≤ c * (V.encodingSize + 1) ^ k + c

def SliceProblemReduction.transportSolver {U V : BinaryPairwiseSlice}
    {P : SliceComputeProblem U} {Q : SliceComputeProblem V}
    (r : SliceProblemReduction P Q) (solve : SliceSolver P) : SliceSolver Q :=
  fun s => do
    let s' ← r.pullState s
    let a ← solve s'
    r.pushOutput a

theorem SliceProblemReduction.transportSolver_solves {U V : BinaryPairwiseSlice}
    {P : SliceComputeProblem U} {Q : SliceComputeProblem V}
    (r : SliceProblemReduction P Q) (solve : SliceSolver P)
    (hsolve : P.Solves solve) :
    Q.Solves (r.transportSolver solve) := by
  intro s
  exact r.sound s ((solve (r.pullState s).result).result) (hsolve (r.pullState s).result)

theorem SliceProblemReduction.transportSolver_polytime {U V : BinaryPairwiseSlice}
    {P : SliceComputeProblem U} {Q : SliceComputeProblem V}
    (r : SliceProblemReduction P Q) (solve : SliceSolver P)
    (hsolve : P.Polytime solve) :
    Q.Polytime (r.transportSolver solve) := by
  rcases r.pullState_poly with ⟨cpull, kpull, hpull⟩
  rcases r.pushOutput_poly with ⟨cpush, kpush, hpush⟩
  rcases hsolve with ⟨csolve, ksolve, hsolve_time⟩
  let npull : ℕ := cpull * (V.encodingSize + 1) ^ kpull + cpull
  let npush : ℕ := cpush * (V.encodingSize + 1) ^ kpush + cpush
  let nsolve : ℕ := csolve * (U.encodingSize + 1) ^ ksolve + csolve
  refine SliceComputeProblem.polytime_of_steps_le_const Q (r.transportSolver solve)
    (npull + nsolve + npush) ?_
  intro s
  have hpull' : (r.pullState s).steps ≤ npull := by
    simpa [npull] using hpull s
  have hsolve' : (solve (r.pullState s).result).steps ≤ nsolve := by
    simpa [nsolve] using hsolve_time (r.pullState s).result
  have hpush' : (r.pushOutput ((solve (r.pullState s).result).result)).steps ≤ npush := by
    simpa [npush] using hpush ((solve (r.pullState s).result).result)
  have hsteps1 :
      (r.transportSolver solve s).steps =
        (r.pullState s).steps +
          (Counted.bind (solve (r.pullState s).result) fun a => r.pushOutput a).steps := by
    unfold SliceProblemReduction.transportSolver
    simpa using Counted.bind_steps (r.pullState s)
      (fun s' => Counted.bind (solve s') fun a => r.pushOutput a)
  have hsteps2 :
      (Counted.bind (solve (r.pullState s).result) fun a => r.pushOutput a).steps =
        (solve (r.pullState s).result).steps +
          (r.pushOutput ((solve (r.pullState s).result).result)).steps := by
    simpa using Counted.bind_steps (solve (r.pullState s).result)
      (fun a => r.pushOutput a)
  rw [hsteps1, hsteps2]
  omega

/-- Bidirectional transport data for a pair of slice-level compute problems. -/
structure SliceProblemEquiv {U V : BinaryPairwiseSlice}
    (P : SliceComputeProblem U) (Q : SliceComputeProblem V) where
  forward : SliceProblemReduction P Q
  backward : SliceProblemReduction Q P

theorem SliceProblemEquiv.polytimeSolvable_iff {U V : BinaryPairwiseSlice}
    {P : SliceComputeProblem U} {Q : SliceComputeProblem V}
    (e : SliceProblemEquiv P Q) :
    P.PolytimeSolvable ↔ Q.PolytimeSolvable := by
  constructor
  · rintro ⟨solve, hsolve, hpoly⟩
    exact ⟨e.forward.transportSolver solve,
      e.forward.transportSolver_solves solve hsolve,
      e.forward.transportSolver_polytime solve hpoly⟩
  · rintro ⟨solve, hsolve, hpoly⟩
    exact ⟨e.backward.transportSolver solve,
      e.backward.transportSolver_solves solve hsolve,
      e.backward.transportSolver_polytime solve hpoly⟩

/-- A uniform family of compute problems indexed by slices. -/
structure SliceComputeFamily where
  problem : ∀ U : BinaryPairwiseSlice, SliceComputeProblem U

def SliceComputeFamily.PolytimePredicate (F : SliceComputeFamily) : BinaryPairwiseSlice → Prop :=
  fun U => (F.problem U).PolytimeSolvable

/-- Witness-by-witness transport package for a uniform compute family. -/
structure ClosureTransportFamily where
  family : SliceComputeFamily
  action_relabel : ∀ {U V : BinaryPairwiseSlice},
    ActionRelabelWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)
  coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
    CoordinateRelabelWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)
  positive_affine : ∀ {U V : BinaryPairwiseSlice},
    PositiveAffineWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)
  duplicate_action : ∀ {U V : BinaryPairwiseSlice},
    DuplicateActionWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)
  duplicate_state : ∀ {U V : BinaryPairwiseSlice},
    DuplicateStateWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)
  irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
    IrrelevantCoordinateWitness U V → SliceProblemEquiv (family.problem U) (family.problem V)

theorem ClosureTransportFamily.polytimePredicate_closureLawInvariant
    (F : ClosureTransportFamily) :
    ClosureLawInvariant F.family.PolytimePredicate := by
  refine
    { action_relabel := ?_
      coordinate_relabel := ?_
      positive_affine := ?_
      duplicate_action := ?_
      duplicate_state := ?_
      irrelevant_coordinate := ?_ }
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.action_relabel h)
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.coordinate_relabel h)
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.positive_affine h)
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.duplicate_action h)
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.duplicate_state h)
  · intro U V h
    exact SliceProblemEquiv.polytimeSolvable_iff (F.irrelevant_coordinate h)

theorem ClosureTransportFamily.compute_classifier_agrees_on_closureEquivalent_of_correctOnDomain
    (F : ClosureTransportFamily)
    {D C : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.family.PolytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V) :
    C U ↔ C V :=
  Paper4dFrontier.classifier_agrees_on_closureEquivalent_of_correctOnDomain hClosed
    (F.polytimePredicate_closureLawInvariant) hCorrect hDU hEqv

theorem ClosureTransportFamily.no_correctOnDomain_compute_classifier_of_orbit_gap
    (F : ClosureTransportFamily)
    {D C Q : BinaryPairwiseSlice → Prop}
    (hClosed : ClosureClosedDomain D)
    (hCorrect : CorrectOnDomain D F.family.PolytimePredicate C)
    {U V : BinaryPairwiseSlice} (hDU : D U) (hEqv : ClosureEquivalent U V)
    (hQU : Q U) (hQV : ¬ Q V) :
    ¬ (∀ S, D S → (C S ↔ Q S)) :=
  Paper4dFrontier.no_correctOnDomain_classifier_of_orbit_gap hClosed
    (F.polytimePredicate_closureLawInvariant) hCorrect hDU hEqv hQU hQV

/-- Deterministic payload computation. -/
def deterministicPayloadProblem {U : BinaryPairwiseSlice}
    (T : Type) (φ : SliceState U → T) : SliceComputeProblem U where
  Output := T
  admissible s t := t = φ s

/-- Admissible-output search for an arbitrary admissible-output relation. -/
def admissibleOutputSearchProblem {U : BinaryPairwiseSlice}
    (T : Type) (R : SliceState U → T → Prop) : SliceComputeProblem U where
  Output := T
  admissible := R

/-- Optimizer computation for a slice. -/
def optimizerComputationProblem (U : BinaryPairwiseSlice) : SliceComputeProblem U where
  Output := U.Action
  admissible s a := (U.toDecisionProblem).isOptimal a s

def deterministicPayloadFamily (T : BinaryPairwiseSlice → Type)
    (φ : ∀ U : BinaryPairwiseSlice, SliceState U → T U) : SliceComputeFamily where
  problem U := deterministicPayloadProblem (T U) (φ U)

def admissibleOutputSearchFamily (T : BinaryPairwiseSlice → Type)
    (R : ∀ U : BinaryPairwiseSlice, SliceState U → T U → Prop) : SliceComputeFamily where
  problem U := admissibleOutputSearchProblem (T U) (R U)

def optimizerComputationFamily : SliceComputeFamily where
  problem := optimizerComputationProblem

def mkDeterministicPayloadClosureTransportFamily
    (T : BinaryPairwiseSlice → Type)
    (φ : ∀ U : BinaryPairwiseSlice, SliceState U → T U)
    (action_relabel : ∀ {U V : BinaryPairwiseSlice},
      ActionRelabelWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V))
    (coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
      CoordinateRelabelWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V))
    (positive_affine : ∀ {U V : BinaryPairwiseSlice},
      PositiveAffineWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V))
    (duplicate_action : ∀ {U V : BinaryPairwiseSlice},
      DuplicateActionWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V))
    (duplicate_state : ∀ {U V : BinaryPairwiseSlice},
      DuplicateStateWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V))
    (irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
      IrrelevantCoordinateWitness U V →
        SliceProblemEquiv
          ((deterministicPayloadFamily T φ).problem U)
          ((deterministicPayloadFamily T φ).problem V)) :
    ClosureTransportFamily where
  family := deterministicPayloadFamily T φ
  action_relabel := action_relabel
  coordinate_relabel := coordinate_relabel
  positive_affine := positive_affine
  duplicate_action := duplicate_action
  duplicate_state := duplicate_state
  irrelevant_coordinate := irrelevant_coordinate

def mkAdmissibleOutputSearchClosureTransportFamily
    (T : BinaryPairwiseSlice → Type)
    (R : ∀ U : BinaryPairwiseSlice, SliceState U → T U → Prop)
    (action_relabel : ∀ {U V : BinaryPairwiseSlice},
      ActionRelabelWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V))
    (coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
      CoordinateRelabelWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V))
    (positive_affine : ∀ {U V : BinaryPairwiseSlice},
      PositiveAffineWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V))
    (duplicate_action : ∀ {U V : BinaryPairwiseSlice},
      DuplicateActionWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V))
    (duplicate_state : ∀ {U V : BinaryPairwiseSlice},
      DuplicateStateWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V))
    (irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
      IrrelevantCoordinateWitness U V →
        SliceProblemEquiv
          ((admissibleOutputSearchFamily T R).problem U)
          ((admissibleOutputSearchFamily T R).problem V)) :
    ClosureTransportFamily where
  family := admissibleOutputSearchFamily T R
  action_relabel := action_relabel
  coordinate_relabel := coordinate_relabel
  positive_affine := positive_affine
  duplicate_action := duplicate_action
  duplicate_state := duplicate_state
  irrelevant_coordinate := irrelevant_coordinate

@[simp] theorem castState_symm_castState {m n : ℕ} (h : m = n)
    (s : Fin m → Fin 2) :
    castState h.symm (castState h s) = s := by
  subst h
  rfl

@[simp] theorem castState_castState_symm {m n : ℕ} (h : m = n)
    (s : Fin n → Fin 2) :
    castState h (castState h.symm s) = s := by
  subst h
  rfl

@[simp] theorem permuteState_symm_apply {n : ℕ}
    (σ : Equiv.Perm (Fin n)) (s : Fin n → Fin 2) :
    permuteState σ.symm (permuteState σ s) = s := by
  funext i
  simp [permuteState]

@[simp] theorem permuteState_apply_symm {n : ℕ}
    (σ : Equiv.Perm (Fin n)) (s : Fin n → Fin 2) :
    permuteState σ (permuteState σ.symm s) = s := by
  funext i
  simp [permuteState]

theorem actionRelabelWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).isOptimal (h.relabel a) s ↔
      (U.toDecisionProblem).isOptimal a (castState h.hArity.symm s) := by
  constructor
  · intro hOpt a'
    have hh := hOpt (h.relabel a')
    calc
      (U.toDecisionProblem).utility a' (castState h.hArity.symm s)
          = (V.toDecisionProblem).utility (h.relabel a') s := by
              change ((U.utility a' (castState h.hArity.symm s) : ℤ) : ℝ) =
                ((V.utility (h.relabel a') s : ℤ) : ℝ)
              have hEq : V.utility (h.relabel a') s = U.utility a' (castState h.hArity.symm s) := by
                simpa using h.utility_eq a' (castState h.hArity.symm s)
              exact_mod_cast hEq.symm
      _ ≤ (V.toDecisionProblem).utility (h.relabel a) s := hh
      _ = (U.toDecisionProblem).utility a (castState h.hArity.symm s) := by
            change ((V.utility (h.relabel a) s : ℤ) : ℝ) =
              ((U.utility a (castState h.hArity.symm s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (castState h.hArity.symm s) := by
              simpa using h.utility_eq a (castState h.hArity.symm s)
            exact_mod_cast hEq
  · intro hOpt b'
    have hh := hOpt (h.relabel.symm b')
    calc
      (V.toDecisionProblem).utility b' s
          = (U.toDecisionProblem).utility (h.relabel.symm b') (castState h.hArity.symm s) := by
              change ((V.utility b' s : ℤ) : ℝ) =
                ((U.utility (h.relabel.symm b') (castState h.hArity.symm s) : ℤ) : ℝ)
              have hEq : V.utility b' s = U.utility (h.relabel.symm b') (castState h.hArity.symm s) := by
                simpa using h.utility_eq (h.relabel.symm b') (castState h.hArity.symm s)
              exact_mod_cast hEq
      _ ≤ (U.toDecisionProblem).utility a (castState h.hArity.symm s) := hh
      _ = (V.toDecisionProblem).utility (h.relabel a) s := by
            change ((U.utility a (castState h.hArity.symm s) : ℤ) : ℝ) =
              ((V.utility (h.relabel a) s : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (castState h.hArity.symm s) := by
              simpa using h.utility_eq a (castState h.hArity.symm s)
            exact_mod_cast hEq.symm

def actionRelabelOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : ActionRelabelWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun a => Counted.tick (h.relabel a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using
          (actionRelabelWitness_isOptimal_iff h a s).2 ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun b => Counted.tick (h.relabel.symm b)
      sound := by
        intro s b hb
        simpa [optimizerComputationProblem] using
          (actionRelabelWitness_isOptimal_iff h (h.relabel.symm b) (castState h.hArity s)).1
            (by simpa using hb)
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

def coordinateRelabelPullState {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) : SliceState V → SliceState U :=
  fun s => permuteState h.perm.symm (castState h.hArity.symm s)

def coordinateRelabelPushState {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) : SliceState U → SliceState V :=
  fun s => castState h.hArity (permuteState h.perm s)

theorem coordinateRelabelWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).isOptimal (h.relabel a) s ↔
      (U.toDecisionProblem).isOptimal a (coordinateRelabelPullState h s) := by
  constructor
  · intro hOpt a'
    have hh := hOpt (h.relabel a')
    calc
      (U.toDecisionProblem).utility a' (coordinateRelabelPullState h s)
          = (V.toDecisionProblem).utility (h.relabel a') s := by
              change ((U.utility a' (coordinateRelabelPullState h s) : ℤ) : ℝ) =
                ((V.utility (h.relabel a') s : ℤ) : ℝ)
              have hEq : V.utility (h.relabel a') s = U.utility a' (coordinateRelabelPullState h s) := by
                simpa [coordinateRelabelPullState] using h.utility_eq a' (coordinateRelabelPullState h s)
              exact_mod_cast hEq.symm
      _ ≤ (V.toDecisionProblem).utility (h.relabel a) s := hh
      _ = (U.toDecisionProblem).utility a (coordinateRelabelPullState h s) := by
            change ((V.utility (h.relabel a) s : ℤ) : ℝ) =
              ((U.utility a (coordinateRelabelPullState h s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (coordinateRelabelPullState h s) := by
              simpa [coordinateRelabelPullState] using h.utility_eq a (coordinateRelabelPullState h s)
            exact_mod_cast hEq
  · intro hOpt b'
    have hh := hOpt (h.relabel.symm b')
    calc
      (V.toDecisionProblem).utility b' s
          = (U.toDecisionProblem).utility (h.relabel.symm b') (coordinateRelabelPullState h s) := by
              change ((V.utility b' s : ℤ) : ℝ) =
                ((U.utility (h.relabel.symm b') (coordinateRelabelPullState h s) : ℤ) : ℝ)
              have hEq : V.utility b' s = U.utility (h.relabel.symm b') (coordinateRelabelPullState h s) := by
                simpa [coordinateRelabelPullState] using
                  h.utility_eq (h.relabel.symm b') (coordinateRelabelPullState h s)
              exact_mod_cast hEq
      _ ≤ (U.toDecisionProblem).utility a (coordinateRelabelPullState h s) := hh
      _ = (V.toDecisionProblem).utility (h.relabel a) s := by
            change ((U.utility a (coordinateRelabelPullState h s) : ℤ) : ℝ) =
              ((V.utility (h.relabel a) s : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a) s = U.utility a (coordinateRelabelPullState h s) := by
              simpa [coordinateRelabelPullState] using h.utility_eq a (coordinateRelabelPullState h s)
            exact_mod_cast hEq.symm

def coordinateRelabelOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : CoordinateRelabelWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPullState h s)
      pushOutput := fun a => Counted.tick (h.relabel a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using
          (coordinateRelabelWitness_isOptimal_iff h a s).2 ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (coordinateRelabelPushState h s)
      pushOutput := fun b => Counted.tick (h.relabel.symm b)
      sound := by
        intro s b hb
        have hEq : coordinateRelabelPullState h (coordinateRelabelPushState h s) = s := by
          simp [coordinateRelabelPullState, coordinateRelabelPushState]
        have hOpt : (V.toDecisionProblem).isOptimal b (coordinateRelabelPushState h s) := by
          simpa [hEq] using hb
        have hOpt' :
            (V.toDecisionProblem).isOptimal (h.relabel (h.relabel.symm b))
              (coordinateRelabelPushState h s) := by
          simpa using hOpt
        have hBack :
            (U.toDecisionProblem).isOptimal (h.relabel.symm b)
              (coordinateRelabelPullState h (coordinateRelabelPushState h s)) := by
          simpa [Equiv.apply_symm_apply] using
            (coordinateRelabelWitness_isOptimal_iff h (h.relabel.symm b)
              (coordinateRelabelPushState h s)).1 hOpt'
        have hBack' : (U.toDecisionProblem).isOptimal (h.relabel.symm b) s := by
          simpa [hEq] using hBack
        simpa [optimizerComputationProblem] using
          hBack'
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

theorem positiveAffineWitness_utility_eq {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).utility (h.relabel a) s =
      (h.alpha (castState h.hArity.symm s) : ℝ) +
        (h.beta (castState h.hArity.symm s) : ℝ) *
          (U.toDecisionProblem).utility a (castState h.hArity.symm s) := by
  have hEq : V.utility (h.relabel a) s =
      h.alpha (castState h.hArity.symm s) +
        (h.beta (castState h.hArity.symm s) : ℤ) * U.utility a (castState h.hArity.symm s) := by
    simpa using h.utility_eq a (castState h.hArity.symm s)
  have hEqR : ((V.utility (h.relabel a) s : ℤ) : ℝ) =
      (((h.alpha (castState h.hArity.symm s) +
        (h.beta (castState h.hArity.symm s) : ℤ) *
          U.utility a (castState h.hArity.symm s) : ℤ)) : ℝ) := by
    exact_mod_cast hEq
  simpa [BinaryPairwiseSlice.toDecisionProblem] using hEqR

theorem positiveAffineWitness_isOptimal_iff {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) (a : U.Action) (s : SliceState V) :
    (V.toDecisionProblem).isOptimal (h.relabel a) s ↔
      (U.toDecisionProblem).isOptimal a (castState h.hArity.symm s) := by
  constructor
  · intro hOpt a'
    have hh := hOpt (h.relabel a')
    have hh' :
        (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility a' (castState h.hArity.symm s)
          ≤
        (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility a (castState h.hArity.symm s) := by
      calc
        (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility a' (castState h.hArity.symm s)
            = (V.toDecisionProblem).utility (h.relabel a') s := by
                symm
                exact positiveAffineWitness_utility_eq h a' s
        _ ≤ (V.toDecisionProblem).utility (h.relabel a) s := hh
        _ = (h.alpha (castState h.hArity.symm s) : ℝ) +
              (h.beta (castState h.hArity.symm s) : ℝ) *
                (U.toDecisionProblem).utility a (castState h.hArity.symm s) :=
              positiveAffineWitness_utility_eq h a s
    have hβ : (0 : ℝ) < (h.beta (castState h.hArity.symm s) : ℝ) := by
      exact_mod_cast h.beta_pos (castState h.hArity.symm s)
    nlinarith
  · intro hOpt b'
    have hh := hOpt (h.relabel.symm b')
    have hh' :
        (U.toDecisionProblem).utility (h.relabel.symm b') (castState h.hArity.symm s)
          ≤ (U.toDecisionProblem).utility a (castState h.hArity.symm s) := by
      simpa [BinaryPairwiseSlice.toDecisionProblem, DecisionProblem.isOptimal] using hh
    have hβ : (0 : ℝ) < (h.beta (castState h.hArity.symm s) : ℝ) := by
      exact_mod_cast h.beta_pos (castState h.hArity.symm s)
    have hh'' :
        (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility (h.relabel.symm b') (castState h.hArity.symm s)
          ≤
        (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility a (castState h.hArity.symm s) := by
      nlinarith
    calc
      (V.toDecisionProblem).utility b' s
          = (h.alpha (castState h.hArity.symm s) : ℝ) +
              (h.beta (castState h.hArity.symm s) : ℝ) *
                (U.toDecisionProblem).utility (h.relabel.symm b') (castState h.hArity.symm s) := by
              simpa [Equiv.apply_symm_apply] using positiveAffineWitness_utility_eq h (h.relabel.symm b') s
      _ ≤ (h.alpha (castState h.hArity.symm s) : ℝ) +
            (h.beta (castState h.hArity.symm s) : ℝ) *
              (U.toDecisionProblem).utility a (castState h.hArity.symm s) := hh''
      _ = (V.toDecisionProblem).utility (h.relabel a) s := by
            symm
            exact positiveAffineWitness_utility_eq h a s

def positiveAffineOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun a => Counted.tick (h.relabel a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using
          (positiveAffineWitness_isOptimal_iff h a s).2 ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun b => Counted.tick (h.relabel.symm b)
      sound := by
        intro s b hb
        simpa [optimizerComputationProblem] using
          (positiveAffineWitness_isOptimal_iff h (h.relabel.symm b) (castState h.hArity s)).1
            (by simpa using hb)
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

noncomputable def DuplicateActionWitness.liftAction {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) : U.Action → V.Action :=
  fun a => Classical.choose (h.surjective_projectAction a)

theorem DuplicateActionWitness.project_liftAction {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (a : U.Action) :
    h.projectAction (h.liftAction a) = a :=
  Classical.choose_spec (h.surjective_projectAction a)

theorem duplicateActionWitness_project_isOptimal {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (b : V.Action) (s : SliceState U) :
    (V.toDecisionProblem).isOptimal b (castState h.hArity s) →
      (U.toDecisionProblem).isOptimal (h.projectAction b) s := by
  intro hOpt a'
  rcases h.surjective_projectAction a' with ⟨b', hb'⟩
  have hh := hOpt b'
  calc
    (U.toDecisionProblem).utility a' s
        = (V.toDecisionProblem).utility b' (castState h.hArity s) := by
            change ((U.utility a' s : ℤ) : ℝ) = ((V.utility b' (castState h.hArity s) : ℤ) : ℝ)
            have hEq : V.utility b' (castState h.hArity s) = U.utility a' s := by
              simpa [hb'] using h.utility_eq b' s
            exact_mod_cast hEq.symm
    _ ≤ (V.toDecisionProblem).utility b (castState h.hArity s) := hh
    _ = (U.toDecisionProblem).utility (h.projectAction b) s := by
          change ((V.utility b (castState h.hArity s) : ℤ) : ℝ) =
            ((U.utility (h.projectAction b) s : ℤ) : ℝ)
          have hEq : V.utility b (castState h.hArity s) = U.utility (h.projectAction b) s := by
            simpa using h.utility_eq b s
          exact_mod_cast hEq

theorem duplicateActionWitness_lift_isOptimal {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) (a : U.Action) (s : SliceState U) :
    (U.toDecisionProblem).isOptimal a s →
      (V.toDecisionProblem).isOptimal (h.liftAction a) (castState h.hArity s) := by
  intro hOpt b'
  have hh := hOpt (h.projectAction b')
  calc
    (V.toDecisionProblem).utility b' (castState h.hArity s)
        = (U.toDecisionProblem).utility (h.projectAction b') s := by
            change ((V.utility b' (castState h.hArity s) : ℤ) : ℝ) =
              ((U.utility (h.projectAction b') s : ℤ) : ℝ)
            have hEq : V.utility b' (castState h.hArity s) = U.utility (h.projectAction b') s := by
              simpa using h.utility_eq b' s
            exact_mod_cast hEq
    _ ≤ (U.toDecisionProblem).utility a s := hh
    _ = (V.toDecisionProblem).utility (h.liftAction a) (castState h.hArity s) := by
          change ((U.utility a s : ℤ) : ℝ) =
            ((V.utility (h.liftAction a) (castState h.hArity s) : ℤ) : ℝ)
          have hEq : V.utility (h.liftAction a) (castState h.hArity s) = U.utility a s := by
            simpa [h.project_liftAction] using h.utility_eq (h.liftAction a) s
          exact_mod_cast hEq.symm

noncomputable def duplicateActionOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : DuplicateActionWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (castState h.hArity.symm s)
      pushOutput := fun a => Counted.tick (h.liftAction a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using
          duplicateActionWitness_lift_isOptimal h a (castState h.hArity.symm s) ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (castState h.hArity s)
      pushOutput := fun b => Counted.tick (h.projectAction b)
      sound := by
        intro s b hb
        simpa [optimizerComputationProblem] using
          duplicateActionWitness_project_isOptimal h b s hb
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

noncomputable def DuplicateStateWitness.sectionState {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) : SliceState U → SliceState V :=
  fun s => Classical.choose (h.surjective_projectState s)

theorem DuplicateStateWitness.project_sectionState {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (s : SliceState U) :
    h.projectState (h.sectionState s) = s :=
  Classical.choose_spec (h.surjective_projectState s)

theorem duplicateStateWitness_relabel_isOptimal {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (a : U.Action) (s : SliceState V) :
    (U.toDecisionProblem).isOptimal a (h.projectState s) →
      (V.toDecisionProblem).isOptimal (h.relabel a) s := by
  intro hOpt b'
  have hh := hOpt (h.relabel.symm b')
  calc
    (V.toDecisionProblem).utility b' s
        = (U.toDecisionProblem).utility (h.relabel.symm b') (h.projectState s) := by
            change ((V.utility b' s : ℤ) : ℝ) =
              ((U.utility (h.relabel.symm b') (h.projectState s) : ℤ) : ℝ)
            have hEq : V.utility b' s = U.utility (h.relabel.symm b') (h.projectState s) := by
              simpa using h.utility_eq (h.relabel.symm b') s
            exact_mod_cast hEq
    _ ≤ (U.toDecisionProblem).utility a (h.projectState s) := hh
    _ = (V.toDecisionProblem).utility (h.relabel a) s := by
          change ((U.utility a (h.projectState s) : ℤ) : ℝ) =
            ((V.utility (h.relabel a) s : ℤ) : ℝ)
          have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
            simpa using h.utility_eq a s
          exact_mod_cast hEq.symm

theorem duplicateStateWitness_symm_isOptimal {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) (b : V.Action) (s : SliceState U) :
    (V.toDecisionProblem).isOptimal b (h.sectionState s) →
      (U.toDecisionProblem).isOptimal (h.relabel.symm b) s := by
  intro hOpt a'
  have hh := hOpt (h.relabel a')
  calc
    (U.toDecisionProblem).utility a' s
        = (V.toDecisionProblem).utility (h.relabel a') (h.sectionState s) := by
            change ((U.utility a' s : ℤ) : ℝ) =
              ((V.utility (h.relabel a') (h.sectionState s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a') (h.sectionState s) = U.utility a' s := by
              simpa [h.project_sectionState] using h.utility_eq a' (h.sectionState s)
            exact_mod_cast hEq.symm
    _ ≤ (V.toDecisionProblem).utility b (h.sectionState s) := hh
    _ = (U.toDecisionProblem).utility (h.relabel.symm b) s := by
          change ((V.utility b (h.sectionState s) : ℤ) : ℝ) =
            ((U.utility (h.relabel.symm b) s : ℤ) : ℝ)
          have hEq : V.utility b (h.sectionState s) = U.utility (h.relabel.symm b) s := by
            simpa [h.project_sectionState] using h.utility_eq (h.relabel.symm b) (h.sectionState s)
          exact_mod_cast hEq

noncomputable def duplicateStateOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : DuplicateStateWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun a => Counted.tick (h.relabel a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using duplicateStateWitness_relabel_isOptimal h a s ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun b => Counted.tick (h.relabel.symm b)
      sound := by
        intro s b hb
        simpa [optimizerComputationProblem] using duplicateStateWitness_symm_isOptimal h b s hb
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

theorem irrelevantCoordinateWitness_relabel_isOptimal {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (a : U.Action) (s : SliceState V) :
    (U.toDecisionProblem).isOptimal a (h.projectState s) →
      (V.toDecisionProblem).isOptimal (h.relabel a) s := by
  intro hOpt b'
  have hh := hOpt (h.relabel.symm b')
  calc
    (V.toDecisionProblem).utility b' s
        = (U.toDecisionProblem).utility (h.relabel.symm b') (h.projectState s) := by
            change ((V.utility b' s : ℤ) : ℝ) =
              ((U.utility (h.relabel.symm b') (h.projectState s) : ℤ) : ℝ)
            have hEq : V.utility b' s = U.utility (h.relabel.symm b') (h.projectState s) := by
              simpa using h.utility_eq (h.relabel.symm b') s
            exact_mod_cast hEq
    _ ≤ (U.toDecisionProblem).utility a (h.projectState s) := hh
    _ = (V.toDecisionProblem).utility (h.relabel a) s := by
          change ((U.utility a (h.projectState s) : ℤ) : ℝ) =
            ((V.utility (h.relabel a) s : ℤ) : ℝ)
          have hEq : V.utility (h.relabel a) s = U.utility a (h.projectState s) := by
            simpa using h.utility_eq a s
          exact_mod_cast hEq.symm

theorem irrelevantCoordinateWitness_symm_isOptimal {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) (b : V.Action) (s : SliceState U) :
    (V.toDecisionProblem).isOptimal b (h.sectionState s) →
      (U.toDecisionProblem).isOptimal (h.relabel.symm b) s := by
  intro hOpt a'
  have hh := hOpt (h.relabel a')
  calc
    (U.toDecisionProblem).utility a' s
        = (V.toDecisionProblem).utility (h.relabel a') (h.sectionState s) := by
            change ((U.utility a' s : ℤ) : ℝ) =
              ((V.utility (h.relabel a') (h.sectionState s) : ℤ) : ℝ)
            have hEq : V.utility (h.relabel a') (h.sectionState s) = U.utility a' s := by
              simpa [h.project_section] using h.utility_eq a' (h.sectionState s)
            exact_mod_cast hEq.symm
    _ ≤ (V.toDecisionProblem).utility b (h.sectionState s) := hh
    _ = (U.toDecisionProblem).utility (h.relabel.symm b) s := by
          change ((V.utility b (h.sectionState s) : ℤ) : ℝ) =
            ((U.utility (h.relabel.symm b) s : ℤ) : ℝ)
          have hEq : V.utility b (h.sectionState s) = U.utility (h.relabel.symm b) s := by
            simpa [h.project_section] using h.utility_eq (h.relabel.symm b) (h.sectionState s)
          exact_mod_cast hEq

def irrelevantCoordinateOptimizerEquiv {U V : BinaryPairwiseSlice}
    (h : IrrelevantCoordinateWitness U V) :
    SliceProblemEquiv (optimizerComputationProblem U) (optimizerComputationProblem V) where
  forward :=
    { pullState := fun s => Counted.tick (h.projectState s)
      pushOutput := fun a => Counted.tick (h.relabel a)
      sound := by
        intro s a ha
        simpa [optimizerComputationProblem] using
          irrelevantCoordinateWitness_relabel_isOptimal h a s ha
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro a; simp [Counted.tick]⟩ }
  backward :=
    { pullState := fun s => Counted.tick (h.sectionState s)
      pushOutput := fun b => Counted.tick (h.relabel.symm b)
      sound := by
        intro s b hb
        simpa [optimizerComputationProblem] using
          irrelevantCoordinateWitness_symm_isOptimal h b s hb
      pullState_poly := ⟨1, 0, by intro s; simp [Counted.tick]⟩
      pushOutput_poly := ⟨1, 0, by intro b; simp [Counted.tick]⟩ }

noncomputable def optimizerClosureTransportFamily : ClosureTransportFamily where
  family := optimizerComputationFamily
  action_relabel := actionRelabelOptimizerEquiv
  coordinate_relabel := coordinateRelabelOptimizerEquiv
  positive_affine := positiveAffineOptimizerEquiv
  duplicate_action := duplicateActionOptimizerEquiv
  duplicate_state := duplicateStateOptimizerEquiv
  irrelevant_coordinate := irrelevantCoordinateOptimizerEquiv

end Paper4dFrontier
