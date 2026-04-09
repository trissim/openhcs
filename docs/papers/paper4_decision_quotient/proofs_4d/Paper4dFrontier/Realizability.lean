import DecisionQuotient.Quotient
import DecisionQuotient.Sufficiency
import DecisionQuotient.UniverseObjective
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

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
