import Paper4dFrontier.MetaCharacterization
import DecisionQuotient.AlgorithmComplexity

namespace Paper4dFrontier

open Classical
open DecisionQuotient

/-- A slice-level normalization predicate acts on a single binary pairwise utility. -/
structure SliceNormalizationPredicate where
  holdsOnSlice : BinaryPairwiseSlice → Prop
  graphOnSlice : ∀ U : BinaryPairwiseSlice, holdsOnSlice U → SimpleGraph (Fin U.arity)

def SliceNormalizationPredicate.toNormalizationPredicate
    (P : SliceNormalizationPredicate) : NormalizationPredicate where
  holds := fun U => ∀ t : ℕ, P.holdsOnSlice (U t)
  graph := fun U hU t => P.graphOnSlice (U t) (hU t)

def BinaryPairwiseSlice.actionCount (U : BinaryPairwiseSlice) : Nat :=
  Fintype.card U.Action

def scaleUtility {A : Type*} {n : ℕ}
    (k : ℤ) (u : A → (Fin n → Fin 2) → ℤ) : A → (Fin n → Fin 2) → ℤ :=
  fun a s => k * u a s

noncomputable def scalePairwise {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} (k : ℤ) (pw : PairwiseUtility u) :
    PairwiseUtility (scaleUtility k u) where
  unary i a x := k * pw.unary i a x
  binary i j a x y := k * pw.binary i j a x y
  interacts := pw.interacts
  interacts_symm := pw.interacts_symm
  decomp := by
    intro a s
    calc
      scaleUtility k u a s = k * ((∑ i : Fin n, pw.unary i a (s i)) +
          (∑ i : Fin n, ∑ j : Fin n,
            if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0)) := by
            simp [scaleUtility, pw.decomp]
      _ = k * (∑ i : Fin n, pw.unary i a (s i)) +
            k * (∑ i : Fin n, ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0) := by
            ring
      _ = (∑ i : Fin n, k * pw.unary i a (s i)) +
            k * (∑ i : Fin n, ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0) := by
            rw [Finset.mul_sum]
      _ = (∑ i : Fin n, k * pw.unary i a (s i)) +
            ∑ i : Fin n, k * ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0 := by
            rw [Finset.mul_sum]
      _ = (∑ i : Fin n, k * pw.unary i a (s i)) +
            ∑ i : Fin n, ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then k * pw.binary i j a (s i) (s j) else 0 := by
            refine congrArg (fun z => (∑ i : Fin n, k * pw.unary i a (s i)) + z) ?_
            refine Finset.sum_congr rfl ?_
            intro i hi
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl ?_
            intro j hj
            by_cases h : pw.interacts i j ∧ i < j
            · simp [h]
            · simp [h]

noncomputable def scaleSlice (k : ℤ) (U : BinaryPairwiseSlice) : BinaryPairwiseSlice where
  Action := U.Action
  instFintypeAction := U.instFintypeAction
  instDecidableEqAction := U.instDecidableEqAction
  arity := U.arity
  utility := scaleUtility k U.utility
  pairwise := scalePairwise k U.pairwise

/-- Canonical finite presentation of the pairwise data of a slice. Relabeling the
action type to `Fin actionCount` keeps the representation uniform. -/
structure BinaryPairwiseSyntax where
  actionCount : ℕ
  arity : ℕ
  unary : (i : Fin arity) → Fin actionCount → Fin 2 → ℤ
  binary : (i j : Fin arity) → Fin actionCount → Fin 2 → Fin 2 → ℤ
  interacts : Fin arity → Fin arity → Prop
  interacts_symm : ∀ i j, interacts i j → interacts j i

def BinaryPairwiseSyntax.interactionGraph (X : BinaryPairwiseSyntax) :
    SimpleGraph (Fin X.arity) :=
  InteractionGraph X.interacts X.interacts_symm

noncomputable def BinaryPairwiseSlice.syntax (U : BinaryPairwiseSlice) : BinaryPairwiseSyntax where
  actionCount := U.actionCount
  arity := U.arity
  unary i a x := U.pairwise.unary i ((Fintype.equivFin U.Action).symm a) x
  binary i j a x y := U.pairwise.binary i j ((Fintype.equivFin U.Action).symm a) x y
  interacts := U.pairwise.interacts
  interacts_symm := U.pairwise.interacts_symm

noncomputable def BinaryPairwiseSyntax.maxUnaryMagnitude (X : BinaryPairwiseSyntax) : Nat :=
  Finset.univ.sup fun i : Fin X.arity =>
    Finset.univ.sup fun a : Fin X.actionCount =>
      Finset.univ.sup fun x : Fin 2 =>
        Int.natAbs (X.unary i a x)

noncomputable def BinaryPairwiseSyntax.maxBinaryMagnitude (X : BinaryPairwiseSyntax) : Nat :=
  Finset.univ.sup fun i : Fin X.arity =>
    Finset.univ.sup fun j : Fin X.arity =>
      Finset.univ.sup fun a : Fin X.actionCount =>
        Finset.univ.sup fun x : Fin 2 =>
          Finset.univ.sup fun y : Fin 2 =>
            Int.natAbs (X.binary i j a x y)

/-- A coarse syntactic size measure for the pairwise presentation. This is the
ambient size parameter used by the admissibility interface. -/
noncomputable def BinaryPairwiseSyntax.encodingSize (X : BinaryPairwiseSyntax) : Nat :=
  X.actionCount * X.arity * 2 +
    X.actionCount * X.arity * X.arity * 4 +
    X.maxUnaryMagnitude + X.maxBinaryMagnitude + 1

noncomputable def BinaryPairwiseSlice.encodingSize (U : BinaryPairwiseSlice) : Nat :=
  U.syntax.encodingSize

def PolynomialTimeCheckable (Q : BinaryPairwiseSlice → Prop) : Prop :=
  ∃ check : BinaryPairwiseSlice → Counted Bool,
    ∃ c k : ℕ,
      ∀ U : BinaryPairwiseSlice,
        ((check U).result = true ↔ Q U) ∧
          (check U).steps ≤ c * (U.encodingSize + 1) ^ k + c

noncomputable def countedDecidePredicate (Q : BinaryPairwiseSlice → Prop)
    [DecidablePred Q] : BinaryPairwiseSlice → Counted Bool :=
  fun U => Counted.tick (decide (Q U))

theorem polynomialTimeCheckable_of_decidable (Q : BinaryPairwiseSlice → Prop)
    [DecidablePred Q] : PolynomialTimeCheckable Q := by
  refine ⟨countedDecidePredicate Q, 1, 0, ?_⟩
  intro U
  constructor
  · change decide (Q U) = true ↔ Q U
    simp
  · change 1 ≤ 1 * (U.encodingSize + 1) ^ 0 + 1
    simp

def castState {m n : ℕ} (h : m = n) : (Fin m → Fin 2) → (Fin n → Fin 2) := by
  subst h
  exact id

def castFin {m n : ℕ} (h : m = n) : Fin m → Fin n := by
  subst h
  exact id

theorem scaleSlice_syntax_unary (k : ℤ) (U : BinaryPairwiseSlice)
    (i : Fin U.arity) (a : Fin U.actionCount) (x : Fin 2) :
    (scaleSlice k U).syntax.unary i a x = k * U.syntax.unary i a x := by
  simp [BinaryPairwiseSlice.syntax, scaleSlice, scalePairwise]

theorem scaleSlice_syntax_actionCount (k : ℤ) (U : BinaryPairwiseSlice) :
    (scaleSlice k U).syntax.actionCount = U.actionCount := by
  simp [BinaryPairwiseSlice.syntax, scaleSlice, BinaryPairwiseSlice.actionCount]

theorem scaleSlice_syntax_unary_cast (k : ℤ) (U : BinaryPairwiseSlice)
    (i : Fin U.arity) (a : Fin ((scaleSlice k U).syntax.actionCount)) (x : Fin 2) :
    (scaleSlice k U).syntax.unary i a x =
      k * U.syntax.unary i (castFin (scaleSlice_syntax_actionCount k U) a) x := by
  simpa [scaleSlice_syntax_actionCount] using
    (scaleSlice_syntax_unary (k := k) (U := U) (i := i)
      (a := castFin (scaleSlice_syntax_actionCount k U) a) (x := x))

def permuteState {n : ℕ} (σ : Equiv.Perm (Fin n)) : (Fin n → Fin 2) → (Fin n → Fin 2) :=
  fun s i => s (σ.symm i)

structure ActionRelabelWitness (U V : BinaryPairwiseSlice) where
  hArity : U.arity = V.arity
  relabel : U.Action ≃ V.Action
  utility_eq : ∀ a : U.Action, ∀ s : Fin U.arity → Fin 2,
    V.utility (relabel a) (castState hArity s) = U.utility a s

structure CoordinateRelabelWitness (U V : BinaryPairwiseSlice) where
  hArity : U.arity = V.arity
  relabel : U.Action ≃ V.Action
  perm : Equiv.Perm (Fin U.arity)
  utility_eq : ∀ a : U.Action, ∀ s : Fin U.arity → Fin 2,
    V.utility (relabel a) (castState hArity (permuteState perm s)) = U.utility a s

structure PositiveAffineWitness (U V : BinaryPairwiseSlice) where
  hArity : U.arity = V.arity
  relabel : U.Action ≃ V.Action
  alpha : (Fin U.arity → Fin 2) → ℤ
  beta : (Fin U.arity → Fin 2) → ℕ
  beta_pos : ∀ s : Fin U.arity → Fin 2, 0 < beta s
  utility_eq : ∀ a : U.Action, ∀ s : Fin U.arity → Fin 2,
    V.utility (relabel a) (castState hArity s) = alpha s + (beta s : ℤ) * U.utility a s

def scaleSlice_positiveAffineWitness (k : ℕ) (hk : 0 < k)
    (U : BinaryPairwiseSlice) :
    PositiveAffineWitness U (scaleSlice (k : ℤ) U) where
  hArity := rfl
  relabel := Equiv.refl U.Action
  alpha := fun _ => 0
  beta := fun _ => k
  beta_pos := fun _ => hk
  utility_eq := by
    intro a s
    simp [scaleSlice, scaleUtility, castState, hk.ne']

structure DuplicateActionWitness (U V : BinaryPairwiseSlice) where
  hArity : U.arity = V.arity
  projectAction : V.Action → U.Action
  utility_eq : ∀ a : V.Action, ∀ s : Fin U.arity → Fin 2,
    V.utility a (castState hArity s) = U.utility (projectAction a) s
  surjective_projectAction : Function.Surjective projectAction
  hasDuplicate : ∃ a0 : U.Action, ∃ b0 b1 : V.Action,
    b0 ≠ b1 ∧ projectAction b0 = a0 ∧ projectAction b1 = a0

structure DuplicateStateWitness (U V : BinaryPairwiseSlice) where
  relabel : U.Action ≃ V.Action
  projectState : (Fin V.arity → Fin 2) → (Fin U.arity → Fin 2)
  utility_eq : ∀ a : U.Action, ∀ s : Fin V.arity → Fin 2,
    V.utility (relabel a) s = U.utility a (projectState s)
  surjective_projectState : Function.Surjective projectState
  noninjective_projectState : ¬ Function.Injective projectState

structure IrrelevantCoordinateWitness (U V : BinaryPairwiseSlice) where
  relabel : U.Action ≃ V.Action
  projectState : (Fin V.arity → Fin 2) → (Fin U.arity → Fin 2)
  sectionState : (Fin U.arity → Fin 2) → (Fin V.arity → Fin 2)
  utility_eq : ∀ a : U.Action, ∀ s : Fin V.arity → Fin 2,
    V.utility (relabel a) s = U.utility a (projectState s)
  project_section : ∀ s : Fin U.arity → Fin 2, projectState (sectionState s) = s

structure ClosureLawInvariant (Q : BinaryPairwiseSlice → Prop) : Prop where
  action_relabel : ∀ {U V : BinaryPairwiseSlice},
    ActionRelabelWitness U V → (Q U ↔ Q V)
  coordinate_relabel : ∀ {U V : BinaryPairwiseSlice},
    CoordinateRelabelWitness U V → (Q U ↔ Q V)
  positive_affine : ∀ {U V : BinaryPairwiseSlice},
    PositiveAffineWitness U V → (Q U ↔ Q V)
  duplicate_action : ∀ {U V : BinaryPairwiseSlice},
    DuplicateActionWitness U V → (Q U ↔ Q V)
  duplicate_state : ∀ {U V : BinaryPairwiseSlice},
    DuplicateStateWitness U V → (Q U ↔ Q V)
  irrelevant_coordinate : ∀ {U V : BinaryPairwiseSlice},
    IrrelevantCoordinateWitness U V → (Q U ↔ Q V)

def StructuralExtractorOn
    (Q : BinaryPairwiseSlice → Prop)
    (graph : ∀ U : BinaryPairwiseSlice, Q U → SimpleGraph (Fin U.arity)) : Prop :=
  ∃ extract : ∀ X : BinaryPairwiseSyntax, SimpleGraph (Fin X.arity),
    ∀ U : BinaryPairwiseSlice, ∀ hU : Q U, graph U hU = extract U.syntax

/-- Rooted radius-`r` neighborhood signature of a coordinate inside the interaction
graph together with the restricted unary and pairwise tables. -/
structure RootedNeighborhoodSignature where
  actionCount : ℕ
  vertexCount : ℕ
  root : Fin vertexCount
  unary : (i : Fin vertexCount) → Fin actionCount → Fin 2 → ℤ
  binary : (i j : Fin vertexCount) → Fin actionCount → Fin 2 → Fin 2 → ℤ
  interacts : Fin vertexCount → Fin vertexCount → Prop
  interacts_symm : ∀ i j, interacts i j → interacts j i

def RootedNeighborhoodSignature.interactionGraph (X : RootedNeighborhoodSignature) :
    SimpleGraph (Fin X.vertexCount) :=
  InteractionGraph X.interacts X.interacts_symm

noncomputable def RootedNeighborhoodSignature.maxUnaryMagnitude
    (X : RootedNeighborhoodSignature) : Nat :=
  Finset.univ.sup fun i : Fin X.vertexCount =>
    Finset.univ.sup fun a : Fin X.actionCount =>
      Finset.univ.sup fun x : Fin 2 =>
        Int.natAbs (X.unary i a x)

noncomputable def RootedNeighborhoodSignature.maxBinaryMagnitude
    (X : RootedNeighborhoodSignature) : Nat :=
  Finset.univ.sup fun i : Fin X.vertexCount =>
    Finset.univ.sup fun j : Fin X.vertexCount =>
      Finset.univ.sup fun a : Fin X.actionCount =>
        Finset.univ.sup fun x : Fin 2 =>
          Finset.univ.sup fun y : Fin 2 =>
            Int.natAbs (X.binary i j a x y)

structure LocalPattern where
  radius : ℕ
  signature : RootedNeighborhoodSignature

def LocalPattern.WithinBounds (P : LocalPattern)
    (radiusBound vertexBound actionBound magnitudeBound : ℕ) : Prop :=
  P.radius ≤ radiusBound ∧
    P.signature.vertexCount ≤ vertexBound ∧
    P.signature.actionCount ≤ actionBound ∧
    P.signature.maxUnaryMagnitude ≤ magnitudeBound ∧
    P.signature.maxBinaryMagnitude ≤ magnitudeBound

noncomputable def LocalPattern.selfMagnitudeBound (P : LocalPattern) : ℕ :=
  max P.signature.maxUnaryMagnitude P.signature.maxBinaryMagnitude

theorem LocalPattern.withinSelfBounds (P : LocalPattern) :
    P.WithinBounds P.radius P.signature.vertexCount P.signature.actionCount P.selfMagnitudeBound := by
  refine ⟨le_rfl, le_rfl, le_rfl, ?_, ?_⟩
  · exact le_max_left _ _
  · exact le_max_right _ _

def LocalPattern.OccursInSyntax (P : LocalPattern) (X : BinaryPairwiseSyntax) : Prop :=
  ∃ f : Fin P.signature.vertexCount → Fin X.arity,
    Function.Injective f ∧
    ∃ σ : Fin P.signature.actionCount ≃ Fin X.actionCount,
      (∀ v : Fin P.signature.vertexCount,
        ∃ p : (X.interactionGraph).Walk (f P.signature.root) (f v), p.length ≤ P.radius) ∧
      (∀ i : Fin P.signature.vertexCount,
        ∀ a : Fin P.signature.actionCount,
        ∀ x : Fin 2,
          P.signature.unary i a x = X.unary (f i) (σ a) x) ∧
      (∀ i j : Fin P.signature.vertexCount,
        P.signature.interacts i j → X.interacts (f i) (f j)) ∧
      (∀ i j : Fin P.signature.vertexCount,
        ∀ a : Fin P.signature.actionCount,
        ∀ x y : Fin 2,
          P.signature.binary i j a x y = X.binary (f i) (f j) (σ a) x y)

def LocalPattern.OccursInSlice (P : LocalPattern) (U : BinaryPairwiseSlice) : Prop :=
  P.OccursInSyntax U.syntax

def LocalPattern.OccursUpToPositiveScaleInSyntax (P : LocalPattern) (X : BinaryPairwiseSyntax) : Prop :=
  ∃ m : ℕ,
    0 < m ∧
    ∃ f : Fin P.signature.vertexCount → Fin X.arity,
      Function.Injective f ∧
      ∃ σ : Fin P.signature.actionCount ≃ Fin X.actionCount,
        (∀ v : Fin P.signature.vertexCount,
          ∃ p : (X.interactionGraph).Walk (f P.signature.root) (f v), p.length ≤ P.radius) ∧
        (∀ i : Fin P.signature.vertexCount,
          ∀ a : Fin P.signature.actionCount,
          ∀ x : Fin 2,
            X.unary (f i) (σ a) x = (m : ℤ) * P.signature.unary i a x) ∧
        (∀ i j : Fin P.signature.vertexCount,
          P.signature.interacts i j → X.interacts (f i) (f j)) ∧
        (∀ i j : Fin P.signature.vertexCount,
          ∀ a : Fin P.signature.actionCount,
          ∀ x y : Fin 2,
            X.binary (f i) (f j) (σ a) x y = (m : ℤ) * P.signature.binary i j a x y)

def LocalPattern.OccursUpToPositiveScaleInSlice (P : LocalPattern) (U : BinaryPairwiseSlice) : Prop :=
  P.OccursUpToPositiveScaleInSyntax U.syntax

theorem LocalPattern.occursUpToPositiveScaleInSyntax_of_occurs
    (P : LocalPattern) {X : BinaryPairwiseSyntax} :
    P.OccursInSyntax X → P.OccursUpToPositiveScaleInSyntax X := by
  rintro ⟨f, hf, σ, hreach, hunary, hinter, hbinary⟩
  refine ⟨1, by decide, f, hf, σ, hreach, ?_, hinter, ?_⟩
  · intro i a x
    rw [← hunary i a x]
    ring
  · intro i j a x y
    rw [← hbinary i j a x y]
    ring

theorem LocalPattern.occursUpToPositiveScaleInSlice_of_occurs
    (P : LocalPattern) {U : BinaryPairwiseSlice} :
    P.OccursInSlice U → P.OccursUpToPositiveScaleInSlice U :=
  P.occursUpToPositiveScaleInSyntax_of_occurs

theorem LocalPattern.occursUpToPositiveScaleInSlice_of_occurs_scale
    (P : LocalPattern) {U : BinaryPairwiseSlice} (k : ℕ) (hk : 0 < k) :
    P.OccursInSlice U → P.OccursUpToPositiveScaleInSlice (scaleSlice (k : ℤ) U) := by
  rintro ⟨f, hf, σ, hreach, hunary, hinter, hbinary⟩
  let τ : Fin P.signature.actionCount ≃ Fin (scaleSlice (k : ℤ) U).syntax.actionCount := by
    simpa [BinaryPairwiseSlice.syntax, scaleSlice, BinaryPairwiseSlice.actionCount,
      scaleSlice_syntax_actionCount] using σ
  refine ⟨k, hk, f, hf, τ, hreach, ?_, ?_, ?_⟩
  · intro i a x
    calc
      (scaleSlice (k : ℤ) U).syntax.unary (f i) (τ a) x
          = (k : ℤ) * U.syntax.unary (f i) (σ a) x := by
              simpa [τ] using
                (scaleSlice_syntax_unary (k := (k : ℤ)) (U := U) (i := f i) (a := σ a) (x := x))
      _ = (k : ℤ) * P.signature.unary i a x := by rw [← hunary i a x]
  · intro i j hij
    exact hinter i j hij
  · intro i j a x y
    calc
      (scaleSlice (k : ℤ) U).syntax.binary (f i) (f j) (τ a) x y
          = (k : ℤ) * U.syntax.binary (f i) (f j) (σ a) x y := by
              simp [BinaryPairwiseSlice.syntax, scaleSlice, scalePairwise, τ]
              left
              rfl
      _ = (k : ℤ) * P.signature.binary i j a x y := by rw [← hbinary i j a x y]

theorem LocalPattern.occursUpToPositiveScaleInSlice_globalScaleInvariant
    (P : LocalPattern) {U : BinaryPairwiseSlice} (k : ℕ) (hk : 0 < k) :
    P.OccursUpToPositiveScaleInSlice U → P.OccursUpToPositiveScaleInSlice (scaleSlice (k : ℤ) U) := by
  rintro ⟨m, hm, f, hf, σ, hreach, hunary, hinter, hbinary⟩
  let τ : Fin P.signature.actionCount ≃ Fin (scaleSlice (k : ℤ) U).syntax.actionCount := by
    simpa [BinaryPairwiseSlice.syntax, scaleSlice, BinaryPairwiseSlice.actionCount,
      scaleSlice_syntax_actionCount] using σ
  refine ⟨k * m, Nat.mul_pos hk hm, f, hf, τ, hreach, ?_, ?_, ?_⟩
  · intro i a x
    calc
      (scaleSlice (k : ℤ) U).syntax.unary (f i) (τ a) x
          = (k : ℤ) * U.syntax.unary (f i) (σ a) x := by
              simpa [τ] using
                (scaleSlice_syntax_unary (k := (k : ℤ)) (U := U) (i := f i) (a := σ a) (x := x))
      _ = (k : ℤ) * ((m : ℤ) * P.signature.unary i a x) := by rw [hunary i a x]
      _ = ((k * m : ℕ) : ℤ) * P.signature.unary i a x := by
        simp [Nat.cast_mul, mul_assoc, mul_left_comm, mul_comm]
  · intro i j hij
    exact hinter i j hij
  · intro i j a x y
    calc
      (scaleSlice (k : ℤ) U).syntax.binary (f i) (f j) (τ a) x y
          = (k : ℤ) * U.syntax.binary (f i) (f j) (σ a) x y := by
              simp [BinaryPairwiseSlice.syntax, scaleSlice, scalePairwise, τ]
              left
              rfl
      _ = (k : ℤ) * ((m : ℤ) * P.signature.binary i j a x y) := by rw [hbinary i j a x y]
      _ = ((k * m : ℕ) : ℤ) * P.signature.binary i j a x y := by
        simp [Nat.cast_mul, mul_assoc, mul_left_comm, mul_comm]

structure BoundedPatternScheme where
  radiusBound : ℕ
  vertexBound : ℕ
  actionBound : ℕ
  magnitudeBound : ℕ
  witnesses : List LocalPattern
  forbidden : List LocalPattern
  witness_bounded : ∀ p ∈ witnesses,
    p.WithinBounds radiusBound vertexBound actionBound magnitudeBound
  forbidden_bounded : ∀ p ∈ forbidden,
    p.WithinBounds radiusBound vertexBound actionBound magnitudeBound

def BoundedPatternScheme.Holds (S : BoundedPatternScheme) (U : BinaryPairwiseSlice) : Prop :=
  (S.witnesses ≠ [] ∧ ∃ p ∈ S.witnesses, p.OccursInSlice U) ∨
    (S.forbidden ≠ [] ∧ ∀ p ∈ S.forbidden, ¬ p.OccursInSlice U)

def BoundedPatternDefinable (Q : BinaryPairwiseSlice → Prop) : Prop :=
  ∃ S : BoundedPatternScheme, ∀ U : BinaryPairwiseSlice, Q U ↔ S.Holds U

def singletonWitnessScheme (P : LocalPattern)
    (radiusBound vertexBound actionBound magnitudeBound : ℕ)
    (hP : P.WithinBounds radiusBound vertexBound actionBound magnitudeBound) :
    BoundedPatternScheme where
  radiusBound := radiusBound
  vertexBound := vertexBound
  actionBound := actionBound
  magnitudeBound := magnitudeBound
  witnesses := [P]
  forbidden := []
  witness_bounded := by
    intro p hp
    simpa using List.mem_singleton.mp hp ▸ hP
  forbidden_bounded := by
    intro p hp
    cases hp

theorem singletonWitnessScheme_holds_iff
    (P : LocalPattern) (radiusBound vertexBound actionBound magnitudeBound : ℕ)
    (hP : P.WithinBounds radiusBound vertexBound actionBound magnitudeBound)
    (U : BinaryPairwiseSlice) :
    (singletonWitnessScheme P radiusBound vertexBound actionBound magnitudeBound hP).Holds U ↔
      P.OccursInSlice U := by
  constructor
  · intro h
    rcases h with ⟨hneq, p, hp, hpOcc⟩ | hforbid
    · simpa using (List.mem_singleton.mp hp ▸ hpOcc)
    · rcases hforbid with ⟨hneq, _⟩
      simpa [singletonWitnessScheme] using hneq
  · intro hOcc
    left
    refine ⟨?_, P, ?_, hOcc⟩
    show [P] ≠ []
    intro hnil
    cases hnil
    show P ∈ [P]
    simp

theorem boundedPatternDefinable_of_singleton_witness
    (P : LocalPattern) (radiusBound vertexBound actionBound magnitudeBound : ℕ)
    (hP : P.WithinBounds radiusBound vertexBound actionBound magnitudeBound) :
    BoundedPatternDefinable (fun U : BinaryPairwiseSlice => P.OccursInSlice U) := by
  refine ⟨singletonWitnessScheme P radiusBound vertexBound actionBound magnitudeBound hP, ?_⟩
  intro U
  exact (singletonWitnessScheme_holds_iff P radiusBound vertexBound actionBound magnitudeBound hP U).symm

inductive ClosureStep : BinaryPairwiseSlice → BinaryPairwiseSlice → Prop where
  | actionRelabel {U V : BinaryPairwiseSlice} : ActionRelabelWitness U V → ClosureStep U V
  | coordinateRelabel {U V : BinaryPairwiseSlice} : CoordinateRelabelWitness U V → ClosureStep U V
  | positiveAffine {U V : BinaryPairwiseSlice} : PositiveAffineWitness U V → ClosureStep U V
  | duplicateAction {U V : BinaryPairwiseSlice} : DuplicateActionWitness U V → ClosureStep U V
  | duplicateState {U V : BinaryPairwiseSlice} : DuplicateStateWitness U V → ClosureStep U V
  | irrelevantCoordinate {U V : BinaryPairwiseSlice} : IrrelevantCoordinateWitness U V → ClosureStep U V

abbrev ClosureEquivalent : BinaryPairwiseSlice → BinaryPairwiseSlice → Prop :=
  Relation.EqvGen ClosureStep

theorem closureStep_of_positiveAffineWitness {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) : ClosureStep U V :=
  ClosureStep.positiveAffine h

theorem closureEquivalent_of_positiveAffineWitness {U V : BinaryPairwiseSlice}
    (h : PositiveAffineWitness U V) : ClosureEquivalent U V :=
  Relation.EqvGen.rel _ _ (closureStep_of_positiveAffineWitness h)

def ClosureHull (Q : BinaryPairwiseSlice → Prop) (U : BinaryPairwiseSlice) : Prop :=
  ∃ V : BinaryPairwiseSlice, ClosureEquivalent V U ∧ Q V

theorem closureHull_intro {Q : BinaryPairwiseSlice → Prop} {U : BinaryPairwiseSlice}
    (hQ : Q U) : ClosureHull Q U := by
  exact ⟨U, Relation.EqvGen.refl U, hQ⟩

theorem closureHull_iff_of_closureEquivalent
    {Q : BinaryPairwiseSlice → Prop} {U V : BinaryPairwiseSlice}
    (hUV : ClosureEquivalent U V) : ClosureHull Q U ↔ ClosureHull Q V := by
  constructor
  · rintro ⟨W, hWU, hQW⟩
    exact ⟨W, Relation.EqvGen.trans _ _ _ hWU hUV, hQW⟩
  · rintro ⟨W, hWV, hQW⟩
    exact ⟨W, Relation.EqvGen.trans _ _ _ hWV (Relation.EqvGen.symm _ _ hUV), hQW⟩

theorem closureHull_closureLawInvariant (Q : BinaryPairwiseSlice → Prop) :
    ClosureLawInvariant (ClosureHull Q) := by
  refine
    { action_relabel := ?_
      coordinate_relabel := ?_
      positive_affine := ?_
      duplicate_action := ?_
      duplicate_state := ?_
      irrelevant_coordinate := ?_ }
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.actionRelabel h))
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.coordinateRelabel h))
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.positiveAffine h))
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.duplicateAction h))
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.duplicateState h))
  · intro U V h
    exact closureHull_iff_of_closureEquivalent (Relation.EqvGen.rel _ _ (ClosureStep.irrelevantCoordinate h))

def ClosureGeneratedByBoundedPatterns (P : BinaryPairwiseSlice → Prop) : Prop :=
  ∃ Q : BinaryPairwiseSlice → Prop,
    BoundedPatternDefinable Q ∧
      ∀ U : BinaryPairwiseSlice, P U ↔ ClosureHull Q U

def BinaryPairwiseSyntax.inBall (X : BinaryPairwiseSyntax) (r : ℕ) (center : Fin X.arity) : Set (Fin X.arity) :=
  {v | ∃ p : (X.interactionGraph).Walk center v, p.length ≤ r}

noncomputable def BinaryPairwiseSyntax.neighborhood
    (X : BinaryPairwiseSyntax) (r : ℕ) (center : Fin X.arity) : RootedNeighborhoodSignature := by
  classical
  let ball : Set (Fin X.arity) := X.inBall r center
  let verts := {v : Fin X.arity // v ∈ ball}
  let e : Fin (Fintype.card verts) ≃ verts := (Fintype.equivFin verts).symm
  have hcenter : center ∈ ball := by
    refine ⟨SimpleGraph.Walk.nil, ?_⟩
    simp
  let c : verts := ⟨center, hcenter⟩
  refine
    { actionCount := X.actionCount
      vertexCount := Fintype.card verts
      root := Fintype.equivFin verts c
      unary := fun i a x => X.unary (e i).1 a x
      binary := fun i j a x y => X.binary (e i).1 (e j).1 a x y
      interacts := fun i j => X.interacts (e i).1 (e j).1
      interacts_symm := ?_ }
  intro i j hij
  exact X.interacts_symm (e i).1 (e j).1 hij

def LocallyDefinable (Q : BinaryPairwiseSlice → Prop) : Prop :=
  ∃ r : ℕ, ∃ test : RootedNeighborhoodSignature → Bool,
    ∀ U : BinaryPairwiseSlice,
      Q U ↔ ∀ i : Fin U.arity, test (U.syntax.neighborhood r i) = true

/-- Restricted normalization predicates: slice-local, polynomial-time decidable,
closure-law invariant, structurally extracted, and definable by a bounded witness/
forbidden local-pattern scheme. -/
structure AdmissibleNormalizationPredicate extends SliceNormalizationPredicate where
  polynomialTimeCheckable : PolynomialTimeCheckable holdsOnSlice
  closureLawInvariant : ClosureLawInvariant holdsOnSlice
  structuralExtractor : StructuralExtractorOn holdsOnSlice graphOnSlice
  boundedPatternDefinable : BoundedPatternDefinable holdsOnSlice

def AdmissibleNormalizationPredicate.toNormalizationPredicate
    (P : AdmissibleNormalizationPredicate) : NormalizationPredicate :=
  P.toSliceNormalizationPredicate.toNormalizationPredicate

structure AdmissibleTractabilityCharacterization where
  predicates : List AdmissibleNormalizationPredicate

def AdmissibleTractabilityCharacterization.toTractabilityCharacterization
    (C : AdmissibleTractabilityCharacterization) : TractabilityCharacterization where
  predicates := C.predicates.map AdmissibleNormalizationPredicate.toNormalizationPredicate

def admissibleCollapseLandscapeInfinity (M : ExactRelevanceComplexityModel) : Prop :=
  ∀ C : AdmissibleTractabilityCharacterization,
    ∃ U : BinaryPairwiseLandscape,
      Defeats M U C.toTractabilityCharacterization

end Paper4dFrontier
