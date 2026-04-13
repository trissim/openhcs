import DecisionQuotient.Quotient
import DecisionQuotient.Sufficiency
import DecisionQuotient.UniverseObjective
import Mathlib.MeasureTheory.MeasurableSpace.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Nat.Size
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

universe u v

/-- An exact correctness specification consists of an output type together with
the validity relation that declares which outputs are correct at each state. -/
structure ExactCorrectnessSpecification (S : Type u) where
  Output : Type v
  valid : S → Output → Prop

section Realizability

variable {S T : Type*} [DecidableEq T]

/-- A labeling `φ : S → T` can be realized as a singleton optimizer map by
making the designated label the unique action of utility `1` and all others `0`.
-/
def realizingProblem (φ : S → T) : DecisionProblem T S where
  utility a s := if a = φ s then 1 else 0

theorem realizingProblem_isOptimal_iff (φ : S → T) (s : S) (a : T) :
    (realizingProblem φ).isOptimal a s ↔ a = φ s := by
  constructor
  · intro h
    by_contra hne
    have hcontra := h (φ s)
    have : (1 : ℝ) ≤ 0 := by
      simpa [realizingProblem, hne] using hcontra
    linarith
  · intro ha
    subst ha
    intro a'
    by_cases h' : a' = φ s <;> simp [realizingProblem, h']

theorem realizingProblem_opt_eq_singleton (φ : S → T) (s : S) :
    (realizingProblem φ).Opt s = ({φ s} : Set T) := by
  ext a
  simp [DecisionProblem.Opt, realizingProblem_isOptimal_iff]

theorem realizingProblem_decisionEquiv_iff (φ : S → T) (s s' : S) :
    (realizingProblem φ).DecisionEquiv s s' ↔ φ s = φ s' := by
  constructor
  · intro h
    have hs : φ s ∈ (realizingProblem φ).Opt s := by
      rw [realizingProblem_opt_eq_singleton]
      simp
    rw [DecisionProblem.DecisionEquiv] at h
    have hs' : φ s ∈ (realizingProblem φ).Opt s' := by simpa [h] using hs
    rw [realizingProblem_opt_eq_singleton] at hs'
    simpa using hs'
  · intro h
    rw [DecisionProblem.DecisionEquiv, realizingProblem_opt_eq_singleton, realizingProblem_opt_eq_singleton, h]

section PayloadTransfer

variable {S : Type*} {T : Type*} [DecidableEq T] {n : ℕ} [CoordinateSpace S n]

/-- Coordinate sufficiency for an arbitrary deterministic payload map. -/
def payloadSufficient (φ : S → T) (I : Finset (Fin n)) : Prop :=
  ∀ s s' : S, agreeOn s s' I → φ s = φ s'

/-- Coordinate relevance for an arbitrary deterministic payload map. -/
def payloadRelevant (φ : S → T) (i : Fin n) : Prop :=
  ∃ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) ∧
      φ s ≠ φ s'

/-- Coordinate irrelevance for an arbitrary deterministic payload map. -/
def payloadIrrelevant (φ : S → T) (i : Fin n) : Prop :=
  ∀ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) →
      φ s = φ s'

/-- Minimal coordinate sufficiency for an arbitrary deterministic payload map. -/
def payloadMinimalSufficient (φ : S → T) (I : Finset (Fin n)) : Prop :=
  payloadSufficient φ I ∧ ∀ J : Finset (Fin n), J ⊂ I → ¬ payloadSufficient φ J

/-- The family of sufficient coordinate sets for a payload map. -/
def payloadSufficientSets (φ : S → T) : Set (Finset (Fin n)) :=
  { I | payloadSufficient φ I }

/-- The set of relevant coordinates for a payload map. -/
def payloadRelevantSet (φ : S → T) : Set (Fin n) :=
  { i | payloadRelevant φ i }

theorem payloadSufficient_iff_realizingProblem_isSufficient
    (φ : S → T) (I : Finset (Fin n)) :
    payloadSufficient φ I ↔ (realizingProblem φ).isSufficient I := by
  unfold payloadSufficient DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    simpa [DecisionProblem.DecisionEquiv] using
      (realizingProblem_decisionEquiv_iff φ s s').2 (h s s' hagree)
  · intro h s s' hagree
    exact (realizingProblem_decisionEquiv_iff φ s s').1 (by
      simpa [DecisionProblem.DecisionEquiv] using h s s' hagree)

theorem payloadRelevant_iff_realizingProblem_isRelevant
    (φ : S → T) (i : Fin n) :
    payloadRelevant φ i ↔ (realizingProblem φ).isRelevant i := by
  unfold payloadRelevant DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq ((realizingProblem_decisionEquiv_iff φ s s').1 (by
      simpa [DecisionProblem.DecisionEquiv] using hEq))
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq (by
      simpa [DecisionProblem.DecisionEquiv] using
        (realizingProblem_decisionEquiv_iff φ s s').2 hEq)

theorem payloadIrrelevant_iff_realizingProblem_isIrrelevant
    (φ : S → T) (i : Fin n) :
    payloadIrrelevant φ i ↔ (realizingProblem φ).isIrrelevant i := by
  unfold payloadIrrelevant DecisionProblem.isIrrelevant
  constructor
  · intro h s s' hcoords
    simpa [DecisionProblem.DecisionEquiv] using
      (realizingProblem_decisionEquiv_iff φ s s').2 (h s s' hcoords)
  · intro h s s' hcoords
    exact (realizingProblem_decisionEquiv_iff φ s s').1 (by
      simpa [DecisionProblem.DecisionEquiv] using h s s' hcoords)

theorem payloadMinimalSufficient_iff_realizingProblem_isMinimalSufficient
    (φ : S → T) (I : Finset (Fin n)) :
    payloadMinimalSufficient φ I ↔ (realizingProblem φ).isMinimalSufficient I := by
  constructor
  · intro h
    refine ⟨(payloadSufficient_iff_realizingProblem_isSufficient φ I).1 h.1, ?_⟩
    intro J hJI hSuff
    exact h.2 J hJI ((payloadSufficient_iff_realizingProblem_isSufficient φ J).2 hSuff)
  · intro h
    refine ⟨(payloadSufficient_iff_realizingProblem_isSufficient φ I).2 h.1, ?_⟩
    intro J hJI hSuff
    exact h.2 J hJI ((payloadSufficient_iff_realizingProblem_isSufficient φ J).1 hSuff)

theorem payloadSufficientSets_eq_realizingProblem_sufficientSets
    (φ : S → T) :
    payloadSufficientSets φ = (realizingProblem φ).sufficientSets := by
  ext I
  exact payloadSufficient_iff_realizingProblem_isSufficient φ I

theorem payloadRelevantSet_eq_realizingProblem_relevantSet
    (φ : S → T) :
    payloadRelevantSet φ = (realizingProblem φ).relevantSet := by
  ext i
  exact payloadRelevant_iff_realizingProblem_isRelevant φ i

/-- Boolean payloads are the two-label special case of deterministic payloads. -/
theorem booleanPayloadTransfer
    (φ : S → Bool) :
    (∀ I : Finset (Fin n),
      payloadSufficient φ I ↔ (realizingProblem φ).isSufficient I) ∧
    (∀ i : Fin n,
      payloadRelevant φ i ↔ (realizingProblem φ).isRelevant i) := by
  refine ⟨?_, ?_⟩
  · intro I
    exact payloadSufficient_iff_realizingProblem_isSufficient φ I
  · intro i
    exact payloadRelevant_iff_realizingProblem_isRelevant φ i

/-- Reify a yes/no correctness predicate as a Boolean payload. -/
def predicatePayload (P : S → Prop) [DecidablePred P] : S → Bool :=
  fun s => decide (P s)

/-- Exact yes/no predicates reduce definitionally to exact relevance certification
for their induced Boolean decision problem. -/
theorem predicateTransfer (P : S → Prop) [DecidablePred P] :
    (∀ I : Finset (Fin n),
      payloadSufficient (predicatePayload P) I ↔
        (realizingProblem (predicatePayload P)).isSufficient I) ∧
    (∀ i : Fin n,
      payloadRelevant (predicatePayload P) i ↔
        (realizingProblem (predicatePayload P)).isRelevant i) := by
  simpa [predicatePayload] using
    (booleanPayloadTransfer (S := S) (n := n) (φ := fun s => decide (P s)))

end PayloadTransfer

section SetValuedPayloadTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Coordinate sufficiency for a nonempty set-valued payload. -/
def feasiblePayloadSufficient (D : UniverseDynamics S A) (I : Finset (Fin n)) : Prop :=
  ∀ s s' : S, agreeOn s s' I → feasibleActions D s = feasibleActions D s'

/-- Coordinate relevance for a nonempty set-valued payload. -/
def feasiblePayloadRelevant (D : UniverseDynamics S A) (i : Fin n) : Prop :=
  ∃ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) ∧
      feasibleActions D s ≠ feasibleActions D s'

/-- Coordinate irrelevance for a nonempty set-valued payload. -/
def feasiblePayloadIrrelevant (D : UniverseDynamics S A) (i : Fin n) : Prop :=
  ∀ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) →
      feasibleActions D s = feasibleActions D s'

theorem feasiblePayloadSufficient_iff_lawDecisionProblem_isSufficient
    (D : UniverseDynamics S A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (hExists : ∀ s : S, ∃ a : A, D.allowed s a) (I : Finset (Fin n)) :
    feasiblePayloadSufficient D I ↔ (lawDecisionProblem D uAllowed uBlocked).isSufficient I := by
  unfold feasiblePayloadSufficient DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    calc
      (lawDecisionProblem D uAllowed uBlocked).Opt s = feasibleActions D s :=
        opt_eq_feasible_of_gap D hGap (hExists s)
      _ = feasibleActions D s' := h s s' hagree
      _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := by
        symm
        exact opt_eq_feasible_of_gap D hGap (hExists s')
  · intro h s s' hagree
    calc
      feasibleActions D s = (lawDecisionProblem D uAllowed uBlocked).Opt s := by
        symm
        exact opt_eq_feasible_of_gap D hGap (hExists s)
      _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := h s s' hagree
      _ = feasibleActions D s' :=
        opt_eq_feasible_of_gap D hGap (hExists s')

theorem feasiblePayloadRelevant_iff_lawDecisionProblem_isRelevant
    (D : UniverseDynamics S A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (hExists : ∀ s : S, ∃ a : A, D.allowed s a) (i : Fin n) :
    feasiblePayloadRelevant D i ↔ (lawDecisionProblem D uAllowed uBlocked).isRelevant i := by
  unfold feasiblePayloadRelevant DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq (by
      calc
        feasibleActions D s = (lawDecisionProblem D uAllowed uBlocked).Opt s := by
          symm
          exact opt_eq_feasible_of_gap D hGap (hExists s)
        _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := hEq
        _ = feasibleActions D s' := opt_eq_feasible_of_gap D hGap (hExists s'))
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq (by
      calc
        (lawDecisionProblem D uAllowed uBlocked).Opt s = feasibleActions D s :=
          opt_eq_feasible_of_gap D hGap (hExists s)
        _ = feasibleActions D s' := hEq
        _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := by
          symm
          exact opt_eq_feasible_of_gap D hGap (hExists s'))

theorem feasiblePayloadIrrelevant_iff_lawDecisionProblem_isIrrelevant
    (D : UniverseDynamics S A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (hExists : ∀ s : S, ∃ a : A, D.allowed s a) (i : Fin n) :
    feasiblePayloadIrrelevant D i ↔ (lawDecisionProblem D uAllowed uBlocked).isIrrelevant i := by
  unfold feasiblePayloadIrrelevant DecisionProblem.isIrrelevant
  constructor
  · intro h s s' hcoords
    calc
      (lawDecisionProblem D uAllowed uBlocked).Opt s = feasibleActions D s :=
        opt_eq_feasible_of_gap D hGap (hExists s)
      _ = feasibleActions D s' := h s s' hcoords
      _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := by
        symm
        exact opt_eq_feasible_of_gap D hGap (hExists s')
  · intro h s s' hcoords
    calc
      feasibleActions D s = (lawDecisionProblem D uAllowed uBlocked).Opt s := by
        symm
        exact opt_eq_feasible_of_gap D hGap (hExists s)
      _ = (lawDecisionProblem D uAllowed uBlocked).Opt s' := h s s' hcoords
      _ = feasibleActions D s' := opt_eq_feasible_of_gap D hGap (hExists s')

end SetValuedPayloadTransfer

section TotalizedSetValuedPayloadTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Coordinate sufficiency for an arbitrary set-valued payload. -/
def setValuedPayloadSufficient (F : S → Set A) (I : Finset (Fin n)) : Prop :=
  ∀ s s' : S, agreeOn s s' I → F s = F s'

/-- Coordinate relevance for an arbitrary set-valued payload. -/
def setValuedPayloadRelevant (F : S → Set A) (i : Fin n) : Prop :=
  ∃ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) ∧
      F s ≠ F s'

/-- Coordinate irrelevance for an arbitrary set-valued payload. -/
def setValuedPayloadIrrelevant (F : S → Set A) (i : Fin n) : Prop :=
  ∀ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) →
      F s = F s'

/-- Totalize a possibly empty set-valued payload by adjoining a distinguished
failure token `none`. -/
def totalizedPayload (F : S → Set A) (s : S) : Set (Option A) :=
  {oa : Option A | match oa with
    | some a => a ∈ F s
    | none => F s = ∅}

def totalizedPayloadDynamics (F : S → Set A) : UniverseDynamics S (Option A) where
  allowed s oa := oa ∈ totalizedPayload F s

theorem totalizedPayload_eq_iff {F : S → Set A} {s s' : S} :
    totalizedPayload F s = totalizedPayload F s' ↔ F s = F s' := by
  constructor
  · intro h
    ext a
    have hs : some a ∈ totalizedPayload F s ↔ some a ∈ totalizedPayload F s' := by
      simpa [h]
    simpa [totalizedPayload] using hs
  · intro h
    ext oa
    cases oa with
    | none => simp [totalizedPayload, h]
    | some a => simp [totalizedPayload, h]

theorem totalizedPayload_nonempty (F : S → Set A) (s : S) :
    ∃ oa : Option A, (totalizedPayloadDynamics F).allowed s oa := by
  classical
  by_cases hEmpty : F s = ∅
  · exact ⟨none, hEmpty⟩
  · obtain ⟨a, ha⟩ : ∃ a : A, a ∈ F s := by
      by_contra hNo
      apply hEmpty
      ext a
      constructor
      · intro ha
        exact False.elim (hNo ⟨a, ha⟩)
      · intro hFalse
        cases hFalse
    exact ⟨some a, ha⟩

theorem setValuedPayloadSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (F : S → Set A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient F I ↔
      (lawDecisionProblem (totalizedPayloadDynamics F) uAllowed uBlocked).isSufficient I := by
  have hExists : ∀ s : S, ∃ oa : Option A, (totalizedPayloadDynamics F).allowed s oa :=
    totalizedPayload_nonempty F
  rw [← feasiblePayloadSufficient_iff_lawDecisionProblem_isSufficient
      (D := totalizedPayloadDynamics F) hGap hExists I]
  unfold setValuedPayloadSufficient feasiblePayloadSufficient
  constructor
  · intro h s s' hagree
    exact (totalizedPayload_eq_iff).2 (h s s' hagree)
  · intro h s s' hagree
    exact (totalizedPayload_eq_iff).1 (h s s' hagree)

theorem setValuedPayloadRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (F : S → Set A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant F i ↔
      (lawDecisionProblem (totalizedPayloadDynamics F) uAllowed uBlocked).isRelevant i := by
  have hExists : ∀ s : S, ∃ oa : Option A, (totalizedPayloadDynamics F).allowed s oa :=
    totalizedPayload_nonempty F
  rw [← feasiblePayloadRelevant_iff_lawDecisionProblem_isRelevant
      (D := totalizedPayloadDynamics F) hGap hExists i]
  unfold setValuedPayloadRelevant feasiblePayloadRelevant
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩
    exact ⟨s, s', hcoords, fun hEq => hneq ((totalizedPayload_eq_iff).1 hEq)⟩
  · rintro ⟨s, s', hcoords, hneq⟩
    exact ⟨s, s', hcoords, fun hEq => hneq ((totalizedPayload_eq_iff).2 hEq)⟩

theorem setValuedPayloadIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (F : S → Set A) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant F i ↔
      (lawDecisionProblem (totalizedPayloadDynamics F) uAllowed uBlocked).isIrrelevant i := by
  have hExists : ∀ s : S, ∃ oa : Option A, (totalizedPayloadDynamics F).allowed s oa :=
    totalizedPayload_nonempty F
  rw [← feasiblePayloadIrrelevant_iff_lawDecisionProblem_isIrrelevant
      (D := totalizedPayloadDynamics F) hGap hExists i]
  unfold setValuedPayloadIrrelevant feasiblePayloadIrrelevant
  constructor
  · intro h s s' hcoords
    exact (totalizedPayload_eq_iff).2 (h s s' hcoords)
  · intro h s s' hcoords
    exact (totalizedPayload_eq_iff).1 (h s s' hcoords)

end TotalizedSetValuedPayloadTransfer

section RelationalSemanticsTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Arbitrary exact output semantics presented as a state-indexed admissible-output
relation. -/
def outputSemantics (R : S → A → Prop) : S → Set A :=
  fun s => { a : A | R s a }

/-- The canonical state equivalence induced by an exact admissible-output semantics. -/
def exactSemanticsSetoid (R : S → A → Prop) : Setoid S where
  r s s' := outputSemantics R s = outputSemantics R s'
  iseqv := by
    refine ⟨?_, ?_, ?_⟩
    · intro s
      rfl
    · intro s s' h
      exact h.symm
    · intro s s' s'' h₁ h₂
      exact h₁.trans h₂

abbrev ExactSemanticsSetoid (R : S → A → Prop) : Setoid S :=
  exactSemanticsSetoid (S := S) (A := A) R

/-- The quotient of states by equality of admissible-output classes. -/
abbrev ExactSemanticsQuotient (R : S → A → Prop) : Type _ :=
  Quotient (ExactSemanticsSetoid R)

/-- A canonical exact-certification decision problem realizing an exact output
semantics via a strict allowed-versus-blocked utility gap. -/
noncomputable def exactSemanticsDecisionProblem (R : S → A → Prop) : DecisionProblem (Option A) S :=
  lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R)) 1 0

theorem outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (outputSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
        uAllowed uBlocked).isSufficient I :=
  setValuedPayloadSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (F := outputSemantics R) hGap I

theorem outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (outputSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
        uAllowed uBlocked).isRelevant i :=
  setValuedPayloadRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (F := outputSemantics R) hGap i

theorem outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (outputSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  setValuedPayloadIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (F := outputSemantics R) hGap i

/-- Two exact output semantics are equivalent when they induce the same equality
relation on admissible-output sets. -/
def outputSemanticsEquivalent (R : S → A → Prop) (R' : S → A → Prop) : Prop :=
  ∀ s s' : S, outputSemantics R s = outputSemantics R s' ↔ outputSemantics R' s = outputSemantics R' s'

theorem outputSemanticsEquivalent_refl (R : S → A → Prop) :
    outputSemanticsEquivalent R R := by
  intro s s'
  rfl

theorem outputSemanticsEquivalent_symm {R R' : S → A → Prop}
    (h : outputSemanticsEquivalent R R') :
    outputSemanticsEquivalent R' R := by
  intro s s'
  exact (h s s').symm

theorem outputSemanticsEquivalent_trans {R R' R'' : S → A → Prop}
    (h₁ : outputSemanticsEquivalent R R')
    (h₂ : outputSemanticsEquivalent R' R'') :
    outputSemanticsEquivalent R R'' := by
  intro s s'
  exact (h₁ s s').trans (h₂ s s')

def outputSemanticsSetoid : Setoid (S → A → Prop) where
  r := outputSemanticsEquivalent
  iseqv :=
    ⟨outputSemanticsEquivalent_refl,
      outputSemanticsEquivalent_symm,
      outputSemanticsEquivalent_trans⟩

def SemanticallyExtensionalMap {X : Type*} (F : (S → A → Prop) → X) : Prop :=
  ∀ {R R' : S → A → Prop}, outputSemanticsEquivalent R R' → F R = F R'

def SemanticallyExtensionalClaim (C : (S → A → Prop) → Prop) : Prop :=
  SemanticallyExtensionalMap C

theorem semanticallyExtensionalMap_factors_through_outputSemanticsQuotient
    {X : Type*} {F : (S → A → Prop) → X}
    (hF : SemanticallyExtensionalMap (S := S) (A := A) F) :
    ∃ G : Quotient (outputSemanticsSetoid (S := S) (A := A)) → X,
      ∀ R : S → A → Prop, G (Quotient.mk _ R) = F R := by
  refine ⟨Quotient.lift F (fun R R' h => hF h), ?_⟩
  intro R
  rfl

theorem semanticallyExtensionalClaim_factors_through_outputSemanticsQuotient
    {C : (S → A → Prop) → Prop}
    (hC : SemanticallyExtensionalClaim (S := S) (A := A) C) :
    ∃ D : Quotient (outputSemanticsSetoid (S := S) (A := A)) → Prop,
      ∀ R : S → A → Prop, D (Quotient.mk _ R) ↔ C R := by
  rcases semanticallyExtensionalMap_factors_through_outputSemanticsQuotient
      (S := S) (A := A) (X := Prop) hC with ⟨D, hD⟩
  refine ⟨D, ?_⟩
  intro R
  rw [hD R]

theorem setValuedPayloadSufficient_congr_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R')
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (outputSemantics R) I ↔
      setValuedPayloadSufficient (outputSemantics R') I := by
  unfold setValuedPayloadSufficient outputSemanticsEquivalent at *
  constructor
  · intro h s s' hagree
    exact (hEqv s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (hEqv s s').2 (h s s' hagree)

theorem setValuedPayloadRelevant_congr_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R')
    (i : Fin n) :
    setValuedPayloadRelevant (outputSemantics R) i ↔
      setValuedPayloadRelevant (outputSemantics R') i := by
  unfold setValuedPayloadRelevant outputSemanticsEquivalent at *
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq ((hEqv s s').2 hEq)
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hneq ((hEqv s s').1 hEq)

theorem setValuedPayloadIrrelevant_congr_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R')
    (i : Fin n) :
    setValuedPayloadIrrelevant (outputSemantics R) i ↔
      setValuedPayloadIrrelevant (outputSemantics R') i := by
  unfold setValuedPayloadIrrelevant outputSemanticsEquivalent at *
  constructor
  · intro h s s' hcoords
    exact (hEqv s s').1 (h s s' hcoords)
  · intro h s s' hcoords
    exact (hEqv s s').2 (h s s' hcoords)

def outputSemanticsSufficientSets (R : S → A → Prop) : Set (Finset (Fin n)) :=
  { I | setValuedPayloadSufficient (outputSemantics R) I }

def outputSemanticsRelevantSet (R : S → A → Prop) : Set (Fin n) :=
  { i | setValuedPayloadRelevant (outputSemantics R) i }

structure ExactRelevanceProfile (n : ℕ) where
  sufficientSets : Set (Finset (Fin n))
  relevantSet : Set (Fin n)

def outputSemanticsExactRelevanceProfile (R : S → A → Prop) : ExactRelevanceProfile n where
  sufficientSets := outputSemanticsSufficientSets R
  relevantSet := outputSemanticsRelevantSet R

def decisionProblemExactRelevanceProfile {B : Type*}
    (D : DecisionProblem B S) : ExactRelevanceProfile n where
  sufficientSets := D.sufficientSets
  relevantSet := D.relevantSet

theorem outputSemanticsSufficientSets_eq_of_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R') :
    outputSemanticsSufficientSets R = outputSemanticsSufficientSets R' := by
  ext I
  exact setValuedPayloadSufficient_congr_outputSemanticsEquivalent hEqv I

theorem outputSemanticsRelevantSet_eq_of_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R') :
    outputSemanticsRelevantSet R = outputSemanticsRelevantSet R' := by
  ext i
  exact setValuedPayloadRelevant_congr_outputSemanticsEquivalent hEqv i

theorem outputSemanticsExactRelevanceProfile_eq_of_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R') :
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) R =
      outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) R' := by
  unfold outputSemanticsExactRelevanceProfile
  rw [outputSemanticsSufficientSets_eq_of_outputSemanticsEquivalent hEqv,
    outputSemanticsRelevantSet_eq_of_outputSemanticsEquivalent hEqv]

theorem outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) R =
      decisionProblemExactRelevanceProfile (n := n)
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R)) uAllowed uBlocked) := by
  unfold outputSemanticsExactRelevanceProfile decisionProblemExactRelevanceProfile
  have hSuff : outputSemanticsSufficientSets (S := S) (A := A) (n := n) R =
      (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R)) uAllowed uBlocked).sufficientSets := by
    ext I
    exact outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
      (R := R) hGap I
  have hRel : outputSemanticsRelevantSet (S := S) (A := A) (n := n) R =
      (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R)) uAllowed uBlocked).relevantSet := by
    ext i
    exact outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
      (R := R) hGap i
  rw [hSuff, hRel]

theorem outputSemanticsExactRelevanceProfile_eq_exactSemanticsDecisionProblem
    (R : S → A → Prop) :
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) R =
      decisionProblemExactRelevanceProfile (n := n)
        (exactSemanticsDecisionProblem (S := S) (A := A) R) := by
  simpa [exactSemanticsDecisionProblem] using
    (outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
      (S := S) (A := A) (n := n) R (show (0 : ℝ) < 1 by norm_num))

section FiniteCoordinatePresentation

open Classical

noncomputable def finiteIndicatorCoordinateSpace (S : Type*) [Fintype S] :
    CoordinateSpace S (Fintype.card S) where
  Coord := fun _ => Bool
  proj := fun s i => (Fintype.equivFin S s = i)

theorem agreeOn_univ_iff_eq_of_finiteIndicatorCoordinateSpace
    (S : Type*) [Fintype S] [DecidableEq S] {s s' : S} :
    letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
    agreeOn s s' (Finset.univ : Finset (Fin (Fintype.card S))) ↔ s = s' := by
  classical
  letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
  constructor
  · intro h
    let i : Fin (Fintype.card S) := Fintype.equivFin S s
    have hi : i ∈ (Finset.univ : Finset (Fin (Fintype.card S))) := by simp
    have hproj := h i hi
    have hbool : (Fintype.equivFin S s = i) = (Fintype.equivFin S s' = i) := by
      simpa [finiteIndicatorCoordinateSpace] using hproj
    have hs : Fintype.equivFin S s = i := by simp [i]
    have hs' : Fintype.equivFin S s' = i := by simpa [← hbool] using hs
    have hEqv : Fintype.equivFin S s' = Fintype.equivFin S s := hs'.trans hs.symm
    exact ((Fintype.equivFin S).injective hEqv).symm
  · intro h i hi
    simpa [h]

theorem finite_outputSemantics_allCoordinatesSufficient
    (S : Type*) [Fintype S] [DecidableEq S]
    {A : Type*} (R : S → A → Prop) :
    letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
    setValuedPayloadSufficient (outputSemantics (S := S) (A := A) R)
      (Finset.univ : Finset (Fin (Fintype.card S))) := by
  classical
  letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
  intro s s' hAgree
  have hEq : s = s' :=
    (agreeOn_univ_iff_eq_of_finiteIndicatorCoordinateSpace S).1 hAgree
  simpa [hEq]

theorem finite_outputSemantics_realized_by_exactCertification
    (S : Type*) [Fintype S] [DecidableEq S]
    {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := Fintype.card S) R =
      decisionProblemExactRelevanceProfile (n := Fintype.card S)
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) := by
  classical
  letI : CoordinateSpace S (Fintype.card S) := finiteIndicatorCoordinateSpace S
  exact outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
    (S := S) (A := A) (n := Fintype.card S) R hGap

end FiniteCoordinatePresentation

section SingletonCoordinatePresentation

noncomputable def singletonIdentityCoordinateSpace (S : Type*) : CoordinateSpace S 1 where
  Coord := fun _ => S
  proj := fun s _ => s

theorem agreeOn_univ_iff_eq_of_singletonIdentityCoordinateSpace
    (S : Type*) {s s' : S} :
    letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
    agreeOn s s' (Finset.univ : Finset (Fin 1)) ↔ s = s' := by
  letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
  constructor
  · intro h
    have h0 := h 0 (by simp)
    simpa [singletonIdentityCoordinateSpace] using h0
  · intro h i hi
    fin_cases i
    simpa [singletonIdentityCoordinateSpace, h]

theorem outputSemantics_admits_coordinatePresentation
    {S : Type*} {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := 1) R =
      decisionProblemExactRelevanceProfile (n := 1)
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) := by
  letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
  exact outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
    (S := S) (A := A) (n := 1) R hGap

structure MeasurableCoordinatePresentation (S : Type*) [MeasurableSpace S] where
  Coord : Type*
  measurableCoord : MeasurableSpace Coord
  proj : S → Coord
  measurable_proj : Measurable proj

attribute [instance] MeasurableCoordinatePresentation.measurableCoord

structure ContinuousCoordinatePresentation (S : Type*) [TopologicalSpace S] where
  Coord : Type*
  topologicalCoord : TopologicalSpace Coord
  proj : S → Coord
  continuous_proj : Continuous proj

attribute [instance] ContinuousCoordinatePresentation.topologicalCoord

def singletonMeasurableCoordinatePresentation (S : Type*) [MeasurableSpace S] :
    MeasurableCoordinatePresentation S where
  Coord := S
  measurableCoord := inferInstance
  proj := id
  measurable_proj := measurable_id

def singletonContinuousCoordinatePresentation (S : Type*) [TopologicalSpace S] :
    ContinuousCoordinatePresentation S where
  Coord := S
  topologicalCoord := inferInstance
  proj := id
  continuous_proj := continuous_id

theorem outputSemantics_realized_by_singletonMeasurableCoordinatePresentation
    {S : Type*} [MeasurableSpace S] {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    let P := singletonMeasurableCoordinatePresentation S
    letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := 1) R =
      decisionProblemExactRelevanceProfile (n := 1)
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) := by
  letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
  exact outputSemantics_admits_coordinatePresentation (S := S) (A := A) R hGap

theorem outputSemantics_realized_by_singletonContinuousCoordinatePresentation
    {S : Type*} [TopologicalSpace S] {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    let P := singletonContinuousCoordinatePresentation S
    letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := 1) R =
      decisionProblemExactRelevanceProfile (n := 1)
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) := by
  letI : CoordinateSpace S 1 := singletonIdentityCoordinateSpace S
  exact outputSemantics_admits_coordinatePresentation (S := S) (A := A) R hGap

end SingletonCoordinatePresentation

section BooleanPresentation

structure CountableBooleanPresentation (S : Type*) where
  bit : S → ℕ → Bool
  separates : ∀ s s' : S, (∀ i : ℕ, bit s i = bit s' i) ↔ s = s'

structure MeasurableCountableBooleanPresentation (S : Type*) [MeasurableSpace S]
    extends CountableBooleanPresentation S where
  measurable_bit : ∀ i : ℕ, Measurable (fun s => bit s i)

structure ContinuousCountableBooleanPresentation (S : Type*) [TopologicalSpace S]
    extends CountableBooleanPresentation S where
  continuous_bit : ∀ i : ℕ, Continuous (fun s => bit s i)

def encodableBooleanPresentation (S : Type*) [Encodable S] : CountableBooleanPresentation S where
  bit := fun s i => Nat.testBit (Encodable.encode s) i
  separates := by
    intro s s'
    constructor
    · intro h
      apply Encodable.encode_injective
      exact Nat.eq_of_testBit_eq (fun i => h i)
    · intro h i
      simpa [h]

theorem encodable_stateSpace_admits_countableBooleanPresentation
    (S : Type*) [Encodable S] :
    ∃ P : CountableBooleanPresentation S, True :=
  ⟨encodableBooleanPresentation S, trivial⟩

def encodableDiscreteMeasurableBooleanPresentation
    (S : Type*) [Encodable S] [MeasurableSpace S] [DiscreteMeasurableSpace S] :
    MeasurableCountableBooleanPresentation S where
  toCountableBooleanPresentation := encodableBooleanPresentation S
  measurable_bit := by
    intro i
    exact Measurable.of_discrete

theorem encodable_discreteMeasurable_stateSpace_admits_measurable_countableBooleanPresentation
    (S : Type*) [Encodable S] [MeasurableSpace S] [DiscreteMeasurableSpace S] :
    ∃ P : MeasurableCountableBooleanPresentation S, True :=
  ⟨encodableDiscreteMeasurableBooleanPresentation S, trivial⟩

def encodableDiscreteContinuousBooleanPresentation
    (S : Type*) [Encodable S] [TopologicalSpace S] [DiscreteTopology S] :
    ContinuousCountableBooleanPresentation S where
  toCountableBooleanPresentation := encodableBooleanPresentation S
  continuous_bit := by
    intro i
    exact continuous_of_discreteTopology

theorem encodable_discreteTopological_stateSpace_admits_continuous_countableBooleanPresentation
    (S : Type*) [Encodable S] [TopologicalSpace S] [DiscreteTopology S] :
    ∃ P : ContinuousCountableBooleanPresentation S, True :=
  ⟨encodableDiscreteContinuousBooleanPresentation S, trivial⟩

noncomputable def finiteBinaryCoordinateSpace (S : Type*) [Fintype S] [DecidableEq S] :
    CoordinateSpace S (Nat.size (Fintype.card S)) where
  Coord := fun _ => Bool
  proj := fun s i => Nat.testBit (Fintype.equivFin S s) i

theorem agreeOn_univ_iff_eq_of_finiteBinaryCoordinateSpace
    (S : Type*) [Fintype S] [DecidableEq S] {s s' : S} :
    letI : CoordinateSpace S (Nat.size (Fintype.card S)) := finiteBinaryCoordinateSpace S
    agreeOn s s' (Finset.univ : Finset (Fin (Nat.size (Fintype.card S)))) ↔ s = s' := by
  classical
  letI : CoordinateSpace S (Nat.size (Fintype.card S)) := finiteBinaryCoordinateSpace S
  constructor
  · intro h
    apply (Fintype.equivFin S).injective
    apply Fin.ext
    exact Nat.eq_of_testBit_eq (fun i => by
      by_cases hi : i < Nat.size (Fintype.card S)
      · let j : Fin (Nat.size (Fintype.card S)) := ⟨i, hi⟩
        have hj : j ∈ (Finset.univ : Finset (Fin (Nat.size (Fintype.card S)))) := by simp
        simpa [finiteBinaryCoordinateSpace, j] using h j hj
      · have hslt : (Fintype.equivFin S s : ℕ) < 2 ^ Nat.size (Fintype.card S) := by
          exact lt_of_lt_of_le (show (Fintype.equivFin S s : ℕ) < Fintype.card S from
            (Fintype.equivFin S s).is_lt) (Nat.lt_size_self _).le
        have hs'lt : (Fintype.equivFin S s' : ℕ) < 2 ^ Nat.size (Fintype.card S) := by
          exact lt_of_lt_of_le (show (Fintype.equivFin S s' : ℕ) < Fintype.card S from
            (Fintype.equivFin S s').is_lt) (Nat.lt_size_self _).le
        have hsizele : Nat.size (Fintype.card S) ≤ i := Nat.le_of_not_gt hi
        have hfalse : Nat.testBit (Fintype.equivFin S s) i = false :=
          Nat.testBit_eq_false_of_lt <|
            lt_of_lt_of_le hslt (Nat.pow_le_pow_right (by decide) hsizele)
        have hfalse' : Nat.testBit (Fintype.equivFin S s') i = false :=
          Nat.testBit_eq_false_of_lt <|
            lt_of_lt_of_le hs'lt (Nat.pow_le_pow_right (by decide) hsizele)
        rw [hfalse, hfalse'])
  · intro h i hi
    simpa [h]

theorem finite_outputSemantics_realized_by_lowDimExactCertification
    (S : Type*) [Fintype S] [DecidableEq S] {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    letI : CoordinateSpace S (Nat.size (Fintype.card S)) := finiteBinaryCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := Nat.size (Fintype.card S)) R =
      decisionProblemExactRelevanceProfile (n := Nat.size (Fintype.card S))
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) := by
  letI : CoordinateSpace S (Nat.size (Fintype.card S)) := finiteBinaryCoordinateSpace S
  exact outputSemanticsExactRelevanceProfile_eq_totalizedLawDecisionProblem
    (S := S) (A := A) (n := Nat.size (Fintype.card S)) R hGap

theorem finite_exactSpecification_admits_lowDimBooleanPresentation
    (S : Type*) [Fintype S] [DecidableEq S] {A : Type*} (R : S → A → Prop)
    {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed) :
    letI : CoordinateSpace S (Nat.size (Fintype.card S)) := finiteBinaryCoordinateSpace S
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := Nat.size (Fintype.card S)) R =
      decisionProblemExactRelevanceProfile (n := Nat.size (Fintype.card S))
        (lawDecisionProblem (totalizedPayloadDynamics (outputSemantics R))
          uAllowed uBlocked) :=
  finite_outputSemantics_realized_by_lowDimExactCertification (S := S) (A := A) R hGap

end BooleanPresentation

section DeterministicRelationRealizability

variable {T : Type*}

def singletonOutputRelation (φ : S → T) : S → T → Prop :=
  fun s a => a = φ s

theorem singletonOutputSemantics_eq_iff (φ : S → T) (s s' : S) :
    outputSemantics (singletonOutputRelation φ) s =
      outputSemantics (singletonOutputRelation φ) s' ↔
        φ s = φ s' := by
  constructor
  · intro h
    have hs : φ s ∈ outputSemantics (singletonOutputRelation φ) s := by
      simp [outputSemantics, singletonOutputRelation]
    have hs' : φ s ∈ outputSemantics (singletonOutputRelation φ) s' := by
      simpa [h] using hs
    simpa [outputSemantics, singletonOutputRelation] using hs'
  · intro h
    ext a
    simp [outputSemantics, singletonOutputRelation, h]

theorem quotientMap_realizes_setoid_asOutputSemantics
    (r : Setoid S) (s s' : S) :
    outputSemantics (singletonOutputRelation (fun x : S => Quotient.mk r x)) s =
      outputSemantics (singletonOutputRelation (fun x : S => Quotient.mk r x)) s' ↔
        r.r s s' := by
  rw [singletonOutputSemantics_eq_iff]
  constructor
  · intro h
    exact Quotient.exact h
  · intro h
    exact Quotient.sound h

def relationSufficient (r : Setoid S) (I : Finset (Fin n)) : Prop :=
  ∀ s s' : S, agreeOn s s' I → r.r s s'

def relationRelevant (r : Setoid S) (i : Fin n) : Prop :=
  ∃ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) ∧
      ¬ r.r s s'

def relationIrrelevant (r : Setoid S) (i : Fin n) : Prop :=
  ∀ s s' : S,
    (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) →
      r.r s s'

theorem quotientOutputSemanticsSufficient_iff_relationSufficient
    (r : Setoid S) (I : Finset (Fin n)) :
    setValuedPayloadSufficient
        (outputSemantics (singletonOutputRelation (fun x : S => Quotient.mk r x))) I ↔
      relationSufficient r I := by
  unfold setValuedPayloadSufficient relationSufficient
  constructor
  · intro h s s' hagree
    exact (quotientMap_realizes_setoid_asOutputSemantics r s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (quotientMap_realizes_setoid_asOutputSemantics r s s').2 (h s s' hagree)

theorem quotientOutputSemanticsRelevant_iff_relationRelevant
    (r : Setoid S) (i : Fin n) :
    setValuedPayloadRelevant
        (outputSemantics (singletonOutputRelation (fun x : S => Quotient.mk r x))) i ↔
      relationRelevant r i := by
  unfold setValuedPayloadRelevant relationRelevant
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hrel
    exact hneq ((quotientMap_realizes_setoid_asOutputSemantics r s s').2 hrel)
  · rintro ⟨s, s', hcoords, hrel⟩
    refine ⟨s, s', hcoords, ?_⟩
    intro hEq
    exact hrel ((quotientMap_realizes_setoid_asOutputSemantics r s s').1 hEq)

theorem quotientOutputSemanticsIrrelevant_iff_relationIrrelevant
    (r : Setoid S) (i : Fin n) :
    setValuedPayloadIrrelevant
        (outputSemantics (singletonOutputRelation (fun x : S => Quotient.mk r x))) i ↔
      relationIrrelevant r i := by
  unfold setValuedPayloadIrrelevant relationIrrelevant
  constructor
  · intro h s s' hcoords
    exact (quotientMap_realizes_setoid_asOutputSemantics r s s').1 (h s s' hcoords)
  · intro h s s' hcoords
    exact (quotientMap_realizes_setoid_asOutputSemantics r s s').2 (h s s' hcoords)

theorem exists_outputSemantics_realizing_setoid (r : Setoid S) :
    ∃ R : S → Quotient r → Prop,
      ∀ s s' : S, outputSemantics R s = outputSemantics R s' ↔ r.r s s' := by
  refine ⟨singletonOutputRelation (fun x : S => Quotient.mk r x), ?_⟩
  intro s s'
  exact quotientMap_realizes_setoid_asOutputSemantics r s s'

theorem sufficiency_is_relation_refinement
    (R : S → A → Prop) (I : Finset (Fin n)) :
    setValuedPayloadSufficient (outputSemantics R) I ↔
      relationSufficient (ExactSemanticsSetoid R) I := by
  rfl

theorem relevance_is_erased_failure_of_refinement
    (R : S → A → Prop) (i : Fin n) :
    setValuedPayloadRelevant (outputSemantics R) i ↔
      ¬ relationSufficient (ExactSemanticsSetoid R)
        (Finset.univ.erase i) := by
  constructor
  · rintro ⟨s, s', hcoords, hneq⟩ hRefine
    exact hneq (hRefine s s' (by
      intro j hj
      exact hcoords j (Finset.mem_erase.mp hj).1))
  · intro hNotSuff
    by_contra hNotRel
    apply hNotSuff
    intro s s' hagree
    by_contra hneq
    apply hNotRel
    refine ⟨s, s', ?_, hneq⟩
    intro j hj
    exact hagree j (by simp [hj])

theorem exactSemanticsQuotient_universal_characterization
    (R : S → A → Prop) :
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) R =
        decisionProblemExactRelevanceProfile (n := n)
          (exactSemanticsDecisionProblem (S := S) (A := A) R) ∧
      (∀ I : Finset (Fin n),
        setValuedPayloadSufficient (outputSemantics R) I ↔
          relationSufficient (ExactSemanticsSetoid R) I) ∧
      (∀ i : Fin n,
        setValuedPayloadRelevant (outputSemantics R) i ↔
          ¬ relationSufficient (ExactSemanticsSetoid R)
            (Finset.univ.erase i)) := by
  refine ⟨?_, ?_⟩
  · exact outputSemanticsExactRelevanceProfile_eq_exactSemanticsDecisionProblem
      (S := S) (A := A) (n := n) R
  · refine ⟨?_, ?_⟩
    · intro I
      exact sufficiency_is_relation_refinement (S := S) (A := A) (n := n) R I
    · intro i
      exact relevance_is_erased_failure_of_refinement (S := S) (A := A) (n := n) R i

/-- Here `exact` means exact agreement with the validity relation itself. The
relation may encode approximation thresholds, randomized outputs, statistical
guarantees, or failure states. -/
theorem exactnessMeansExactAgreementWithValidity
    (Valid : S → A → Prop) :
    outputSemanticsExactRelevanceProfile (S := S) (A := A) (n := n) Valid =
      decisionProblemExactRelevanceProfile (n := n)
        (exactSemanticsDecisionProblem (S := S) (A := A) Valid) := by
  exact outputSemanticsExactRelevanceProfile_eq_exactSemanticsDecisionProblem
    (S := S) (A := A) (n := n) Valid

namespace ExactCorrectnessSpecification

abbrev semanticsSetoid (spec : ExactCorrectnessSpecification S) : Setoid S :=
  ExactSemanticsSetoid (S := S) (A := spec.Output) spec.valid

abbrev semanticsQuotient (spec : ExactCorrectnessSpecification S) : Type _ :=
  ExactSemanticsQuotient (S := S) (A := spec.Output) spec.valid

noncomputable def decisionProblem
    (spec : ExactCorrectnessSpecification S) : DecisionProblem (Option spec.Output) S :=
  exactSemanticsDecisionProblem (S := S) (A := spec.Output) spec.valid

theorem exactness_means_exact_agreement
    (spec : ExactCorrectnessSpecification S) :
    outputSemanticsExactRelevanceProfile (S := S) (A := spec.Output) (n := n) spec.valid =
      decisionProblemExactRelevanceProfile (n := n) (decisionProblem (S := S) spec) := by
  exact exactnessMeansExactAgreementWithValidity
    (S := S) (A := spec.Output) (n := n) spec.valid

theorem universal_characterization
    (spec : ExactCorrectnessSpecification S) :
    outputSemanticsExactRelevanceProfile (S := S) (A := spec.Output) (n := n) spec.valid =
        decisionProblemExactRelevanceProfile (n := n) (decisionProblem (S := S) spec) ∧
      (∀ I : Finset (Fin n),
        setValuedPayloadSufficient (outputSemantics spec.valid) I ↔
          relationSufficient (semanticsSetoid (S := S) spec) I) ∧
      (∀ i : Fin n,
        setValuedPayloadRelevant (outputSemantics spec.valid) i ↔
          ¬ relationSufficient (semanticsSetoid (S := S) spec)
            (Finset.univ.erase i)) := by
  exact exactSemanticsQuotient_universal_characterization
    (S := S) (A := spec.Output) (n := n) spec.valid

end ExactCorrectnessSpecification

end DeterministicRelationRealizability

end RelationalSemanticsTransfer

section ApproximationSemanticsTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Exact approximation semantics presented by a state-indexed admissible-output
relation. -/
def approximationSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem approximationSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (approximationSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (approximationSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem approximationSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (approximationSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (approximationSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem approximationSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (approximationSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (approximationSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

end ApproximationSemanticsTransfer

section StatisticalGuaranteeSemanticsTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- PAC-style guarantee semantics presented by a state-indexed admissible-output
relation on hypotheses, learners, estimators, or policies. -/
def pacGuaranteeSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem pacGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (pacGuaranteeSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (pacGuaranteeSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem pacGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (pacGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (pacGuaranteeSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem pacGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (pacGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (pacGuaranteeSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

/-- Regret-guarantee semantics presented by a state-indexed admissible-output
relation on hypotheses, learners, estimators, or policies. -/
def regretGuaranteeSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem regretGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (regretGuaranteeSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (regretGuaranteeSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem regretGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (regretGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (regretGuaranteeSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem regretGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (regretGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (regretGuaranteeSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

/-- Statistical-risk semantics presented by a state-indexed admissible-output
relation on hypotheses, learners, estimators, or policies. -/
def statisticalRiskSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem statisticalRiskSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (statisticalRiskSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (statisticalRiskSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem statisticalRiskSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (statisticalRiskSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (statisticalRiskSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem statisticalRiskSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (statisticalRiskSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (statisticalRiskSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

/-- Anytime-guarantee semantics presented by a state-indexed admissible-output
relation on hypotheses, learners, estimators, or policies. -/
def anytimeGuaranteeSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem anytimeGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (anytimeGuaranteeSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (anytimeGuaranteeSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem anytimeGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (anytimeGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (anytimeGuaranteeSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem anytimeGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (anytimeGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (anytimeGuaranteeSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

/-- Finite-horizon guarantee semantics presented by a state-indexed admissible-output
relation on hypotheses, learners, estimators, or policies. -/
def finiteHorizonGuaranteeSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem finiteHorizonGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (finiteHorizonGuaranteeSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (finiteHorizonGuaranteeSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem finiteHorizonGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (finiteHorizonGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (finiteHorizonGuaranteeSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem finiteHorizonGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (finiteHorizonGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (finiteHorizonGuaranteeSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

end StatisticalGuaranteeSemanticsTransfer

section RandomizedGuaranteeSemanticsTransfer

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Randomized-output guarantee semantics presented by a state-indexed admissible-output
relation on distributions, kernels, randomized estimators, or randomized policies. -/
def randomizedGuaranteeSemantics (R : S → A → Prop) : S → Set A :=
  outputSemantics R

theorem randomizedGuaranteeSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (I : Finset (Fin n)) :
    setValuedPayloadSufficient (randomizedGuaranteeSemantics R) I ↔
      (lawDecisionProblem (totalizedPayloadDynamics (randomizedGuaranteeSemantics R))
        uAllowed uBlocked).isSufficient I :=
  outputSemanticsSufficient_iff_totalizedLawDecisionProblem_isSufficient
    (R := R) hGap I

theorem randomizedGuaranteeSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadRelevant (randomizedGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (randomizedGuaranteeSemantics R))
        uAllowed uBlocked).isRelevant i :=
  outputSemanticsRelevant_iff_totalizedLawDecisionProblem_isRelevant
    (R := R) hGap i

theorem randomizedGuaranteeSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R : S → A → Prop) {uAllowed uBlocked : ℝ} (hGap : uBlocked < uAllowed)
    (i : Fin n) :
    setValuedPayloadIrrelevant (randomizedGuaranteeSemantics R) i ↔
      (lawDecisionProblem (totalizedPayloadDynamics (randomizedGuaranteeSemantics R))
        uAllowed uBlocked).isIrrelevant i :=
  outputSemanticsIrrelevant_iff_totalizedLawDecisionProblem_isIrrelevant
    (R := R) hGap i

end RandomizedGuaranteeSemanticsTransfer

section StatisticalGuaranteeSemanticsQuotientConsequences

variable {S : Type*} {A : Type*} {n : ℕ} [CoordinateSpace S n]

theorem statisticalGuaranteeSemanticsSufficientSets_eq_of_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R') :
    outputSemanticsSufficientSets R = outputSemanticsSufficientSets R' :=
  outputSemanticsSufficientSets_eq_of_outputSemanticsEquivalent hEqv

theorem statisticalGuaranteeSemanticsRelevantSet_eq_of_outputSemanticsEquivalent
    {R R' : S → A → Prop} (hEqv : outputSemanticsEquivalent R R') :
    outputSemanticsRelevantSet R = outputSemanticsRelevantSet R' :=
  outputSemanticsRelevantSet_eq_of_outputSemanticsEquivalent hEqv

end StatisticalGuaranteeSemanticsQuotientConsequences

theorem realizingProblem_quotientMap_eq_iff (φ : S → T) (s s' : S) :
    (realizingProblem φ).quotientMap s = (realizingProblem φ).quotientMap s' ↔ φ s = φ s' := by
  rw [(realizingProblem φ).quotient_represents_opt_equiv]
  simpa [DecisionProblem.DecisionEquiv] using (realizingProblem_decisionEquiv_iff φ s s')

noncomputable def realizingProblemQuotientToLabelRange (φ : S → T) :
    (realizingProblem φ).DecisionQuotientType → ↥(Set.range φ) :=
  Quotient.lift
    (fun s => ⟨φ s, ⟨s, rfl⟩⟩)
    (fun s s' h => Subtype.ext ((realizingProblem_decisionEquiv_iff φ s s').1 h))

theorem realizingProblemQuotientToLabelRange_surjective (φ : S → T) :
    Function.Surjective (realizingProblemQuotientToLabelRange φ) := by
  intro y
  rcases y.property with ⟨s, hs⟩
  refine ⟨(realizingProblem φ).quotientMap s, ?_⟩
  apply Subtype.ext
  simpa [realizingProblemQuotientToLabelRange] using hs

theorem realizingProblemQuotientToLabelRange_injective (φ : S → T) :
    Function.Injective (realizingProblemQuotientToLabelRange φ) := by
  intro q q' h
  refine Quotient.inductionOn₂ q q' ?_ h
  intro s s' hEq
  apply Quotient.sound
  exact (realizingProblem_decisionEquiv_iff φ s s').2 (congrArg Subtype.val hEq)

noncomputable def realizingProblemQuotientEquivLabelRange (φ : S → T) :
    (realizingProblem φ).DecisionQuotientType ≃ ↥(Set.range φ) :=
  Equiv.ofBijective (realizingProblemQuotientToLabelRange φ)
    ⟨realizingProblemQuotientToLabelRange_injective φ,
      realizingProblemQuotientToLabelRange_surjective φ⟩

theorem realizingProblemQuotientEquivLabelRange_apply_quotientMap
    (φ : S → T) (s : S) :
    realizingProblemQuotientEquivLabelRange φ ((realizingProblem φ).quotientMap s) = ⟨φ s, ⟨s, rfl⟩⟩ := rfl

theorem realizingProblem_quotientCard_eq_labelRangeCard (φ : S → T)
    [Fintype (realizingProblem φ).DecisionQuotientType] [Fintype ↥(Set.range φ)] :
    Fintype.card (realizingProblem φ).DecisionQuotientType = Fintype.card ↥(Set.range φ) := by
  exact Fintype.card_congr (realizingProblemQuotientEquivLabelRange φ)

noncomputable def rangeEquivCodomainOfSurjective {X Y : Type*} (f : X → Y)
    (hf : Function.Surjective f) : ↥(Set.range f) ≃ Y where
  toFun x := x.1
  invFun y := ⟨y, hf y⟩
  left_inv := by intro x; apply Subtype.ext; rfl
  right_inv := by intro y; rfl

section SetoidRealization

variable {S : Type*}

/-- Every setoid on the state space can be realized as a decision quotient. -/
noncomputable def setoidRealizingProblem (σ : Setoid S) : DecisionProblem (Quotient σ) S := by
  classical
  exact realizingProblem (fun s => Quotient.mk σ s)

theorem setoidRealizingProblem_decisionEquiv_iff (σ : Setoid S) (s s' : S) :
    (setoidRealizingProblem σ).DecisionEquiv s s' ↔ σ.r s s' := by
  classical
  constructor
  · intro h
    exact Quotient.exact ((realizingProblem_decisionEquiv_iff (fun s => Quotient.mk σ s) s s').1 h)
  · intro h
    exact (realizingProblem_decisionEquiv_iff (fun s => Quotient.mk σ s) s s').2 (Quotient.sound h)

noncomputable def setoidRealizingProblemQuotientEquivSetoidQuotient (σ : Setoid S) :
    (setoidRealizingProblem σ).DecisionQuotientType ≃ Quotient σ := by
  classical
  exact (realizingProblemQuotientEquivLabelRange (fun s => Quotient.mk σ s)).trans
    (rangeEquivCodomainOfSurjective (fun s => Quotient.mk σ s) Quotient.mk_surjective)

theorem setoidRealizingProblemQuotientEquivSetoidQuotient_apply_quotientMap
    (σ : Setoid S) (s : S) :
    setoidRealizingProblemQuotientEquivSetoidQuotient σ
      ((setoidRealizingProblem σ).quotientMap s) = Quotient.mk σ s := by
  classical
  rfl

theorem exists_decisionProblem_realizing_setoid (σ : Setoid S) :
    ∃ dp : DecisionProblem (Quotient σ) S,
      ∀ s s' : S, dp.DecisionEquiv s s' ↔ σ.r s s' := by
  classical
  refine ⟨setoidRealizingProblem σ, ?_⟩
  intro s s'
  exact setoidRealizingProblem_decisionEquiv_iff σ s s'

end SetoidRealization

theorem exists_decisionProblem_realizing_labeling (φ : S → T) :
    ∃ dp : DecisionProblem T S,
      (∀ s : S, dp.Opt s = ({φ s} : Set T)) ∧
      (∀ s s' : S, dp.DecisionEquiv s s' ↔ φ s = φ s') := by
  refine ⟨realizingProblem φ, ?_, ?_⟩
  · intro s
    exact realizingProblem_opt_eq_singleton φ s
  · intro s s'
    exact realizingProblem_decisionEquiv_iff φ s s'

end Realizability

end Paper4dFrontier
