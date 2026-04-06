import Paper4dFrontier.Realizability
import DecisionQuotient.Sufficiency
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

universe u v

/-- A representation-level family of decision problems. -/
def ProblemFamily : Type (max (u + 1) (v + 1)) :=
  ∀ {A : Type u} {S : Type v}, DecisionProblem A S → Prop

@[ext] theorem DecisionProblem.ext {A S : Type*} {dp₁ dp₂ : DecisionProblem A S}
    (h : ∀ a s, dp₁.utility a s = dp₂.utility a s) : dp₁ = dp₂ := by
  cases dp₁ with
  | mk u₁ =>
      cases dp₂ with
      | mk u₂ =>
          have hu : u₁ = u₂ := by
            funext a s
            exact h a s
          cases hu
          rfl

/-- Transport a decision problem along an equivalence of action labels. -/
def relabelActions {A B S : Type*} (e : A ≃ B) (dp : DecisionProblem A S) : DecisionProblem B S where
  utility b s := dp.utility (e.symm b) s

/-- Transport a decision problem along an equivalence of state labels. -/
def relabelStates {A S T : Type*} (e : S ≃ T) (dp : DecisionProblem A S) : DecisionProblem A T where
  utility a t := dp.utility a (e.symm t)

/-- Statewise positive affine reparameterization of utility. -/
def positiveAffineTransform {A S : Type*}
    (α β : S → ℝ) (dp : DecisionProblem A S) : DecisionProblem A S where
  utility a s := α s + β s * dp.utility a s

/-- The state relation induced by agreement on a coordinate set. -/
def agreementRel {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (I : Finset (Fin n)) : S → S → Prop :=
  fun s s' => agreeOn s s' I

theorem isOptimal_relabelActions_iff {A B S : Type*}
    (e : A ≃ B) (dp : DecisionProblem A S) (b : B) (s : S) :
    (relabelActions e dp).isOptimal b s ↔ dp.isOptimal (e.symm b) s := by
  constructor
  · intro h a'
    simpa [DecisionProblem.isOptimal, relabelActions] using h (e a')
  · intro h b'
    simpa [DecisionProblem.isOptimal, relabelActions] using h (e.symm b')

theorem decisionEquiv_relabelActions_iff {A B S : Type*}
    (e : A ≃ B) (dp : DecisionProblem A S) (s s' : S) :
    (relabelActions e dp).DecisionEquiv s s' ↔ dp.DecisionEquiv s s' := by
  unfold DecisionProblem.DecisionEquiv
  constructor
  · intro h
    ext a
    have hh := congrArg (fun t : Set B => e a ∈ t) h
    simpa [DecisionProblem.Opt, isOptimal_relabelActions_iff] using hh
  · intro h
    ext b
    have hh := congrArg (fun t : Set A => e.symm b ∈ t) h
    simpa [DecisionProblem.Opt, isOptimal_relabelActions_iff] using hh

def transportedCoordinateSpace {S T : Type*} {n : ℕ} [CoordinateSpace S n]
    (e : S ≃ T) : CoordinateSpace T n where
  Coord := fun i => CoordinateSpace.Coord (S := S) i
  proj := fun t i => CoordinateSpace.proj (S := S) (e.symm t) i

theorem agreeOn_transported_iff {S T : Type*} {n : ℕ} [CoordinateSpace S n]
    (e : S ≃ T) (t t' : T) (I : Finset (Fin n)) :
    @agreeOn T n (transportedCoordinateSpace e) t t' I ↔ agreeOn (e.symm t) (e.symm t') I := by
  rfl

theorem isOptimal_relabelStates_iff {A S T : Type*}
    (e : S ≃ T) (dp : DecisionProblem A S) (a : A) (t : T) :
    (relabelStates e dp).isOptimal a t ↔ dp.isOptimal a (e.symm t) := by
  rfl

theorem decisionEquiv_relabelStates_iff {A S T : Type*}
    (e : S ≃ T) (dp : DecisionProblem A S) (t t' : T) :
    (relabelStates e dp).DecisionEquiv t t' ↔ dp.DecisionEquiv (e.symm t) (e.symm t') := by
  simp [DecisionProblem.DecisionEquiv, DecisionProblem.Opt, isOptimal_relabelStates_iff]

theorem isOptimal_positiveAffineTransform_iff {A S : Type*}
    (α β : S → ℝ) (hβ : ∀ s : S, 0 < β s) (dp : DecisionProblem A S) (a : A) (s : S) :
    (positiveAffineTransform α β dp).isOptimal a s ↔ dp.isOptimal a s := by
  constructor
  · intro h a'
    have hh := h a'
    dsimp [positiveAffineTransform, DecisionProblem.isOptimal] at hh ⊢
    have hb : 0 < β s := hβ s
    nlinarith
  · intro h a'
    have hh := h a'
    dsimp [positiveAffineTransform, DecisionProblem.isOptimal] at hh ⊢
    have hb : 0 < β s := hβ s
    nlinarith

theorem opt_eq_positiveAffineTransform {A S : Type*}
    (α β : S → ℝ) (hβ : ∀ s : S, 0 < β s) (dp : DecisionProblem A S) (s : S) :
    (positiveAffineTransform α β dp).Opt s = dp.Opt s := by
  ext a
  simpa [DecisionProblem.Opt] using
    (isOptimal_positiveAffineTransform_iff α β hβ dp a s)

theorem decisionEquiv_iff_of_opt_eq {A S : Type*}
    {dp₁ dp₂ : DecisionProblem A S} (hOpt : ∀ s : S, dp₁.Opt s = dp₂.Opt s)
    (s s' : S) :
    dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s' := by
  unfold DecisionProblem.DecisionEquiv
  rw [hOpt s, hOpt s']

theorem isSufficient_iff_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    (I : Finset (Fin n)) :
    dp₁.isSufficient I ↔ dp₂.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact (hDec s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (hDec s s').2 (h s s' hagree)

theorem isSufficient_iff_agreementRel_le_decisionEquiv {A S : Type*} {n : ℕ}
    [CoordinateSpace S n] (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    dp.isSufficient I ↔ ∀ s s' : S, agreementRel I s s' → dp.DecisionEquiv s s' := by
  unfold agreementRel DecisionProblem.isSufficient DecisionProblem.DecisionEquiv
  rfl

theorem agreeOn_univ_erase_iff {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (s s' : S) (i : Fin n) :
    agreeOn s s' (Finset.univ.erase i) ↔
      ∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j := by
  constructor
  · intro h j hj
    exact h j (by simp [hj])
  · intro h j hj
    exact h j (Finset.mem_erase.mp hj).1

theorem isIrrelevant_iff_sufficient_erase {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    dp.isIrrelevant i ↔ dp.isSufficient (Finset.univ.erase i) := by
  unfold DecisionProblem.isIrrelevant DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact h s s' ((agreeOn_univ_erase_iff s s' i).1 hagree)
  · intro h s s' hcoord
    exact h s s' ((agreeOn_univ_erase_iff s s' i).2 hcoord)

theorem isRelevant_iff_not_sufficient_erase {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    dp.isRelevant i ↔ ¬ dp.isSufficient (Finset.univ.erase i) := by
  constructor
  · intro hRel hSuff
    have hirr : dp.isIrrelevant i := (isIrrelevant_iff_sufficient_erase dp i).2 hSuff
    exact ((dp.irrelevant_iff_not_relevant i).1 hirr) hRel
  · intro hNotSuff
    by_contra hNotRel
    have hirr : dp.isIrrelevant i := (dp.irrelevant_iff_not_relevant i).2 hNotRel
    have hSuff : dp.isSufficient (Finset.univ.erase i) := (isIrrelevant_iff_sufficient_erase dp i).1 hirr
    exact hNotSuff hSuff

theorem isSufficient_iff_of_opt_eq {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S} (hOpt : ∀ s : S, dp₁.Opt s = dp₂.Opt s)
    (I : Finset (Fin n)) :
    dp₁.isSufficient I ↔ dp₂.isSufficient I := by
  exact isSufficient_iff_of_decisionEquiv_iff (decisionEquiv_iff_of_opt_eq hOpt) I

theorem isRelevant_iff_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    (i : Fin n) :
    dp₁.isRelevant i ↔ dp₂.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor <;>
    rintro ⟨s, s', hcoords, hneq⟩ <;>
    refine ⟨s, s', hcoords, ?_⟩
  · intro hEq
    exact hneq ((hDec s s').2 hEq)
  · intro hEq
    exact hneq ((hDec s s').1 hEq)

theorem isRelevant_iff_of_opt_eq {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S} (hOpt : ∀ s : S, dp₁.Opt s = dp₂.Opt s)
    (i : Fin n) :
    dp₁.isRelevant i ↔ dp₂.isRelevant i := by
  exact isRelevant_iff_of_decisionEquiv_iff (decisionEquiv_iff_of_opt_eq hOpt) i

theorem isIrrelevant_iff_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    (i : Fin n) :
    dp₁.isIrrelevant i ↔ dp₂.isIrrelevant i := by
  rw [dp₁.irrelevant_iff_not_relevant, dp₂.irrelevant_iff_not_relevant,
    isRelevant_iff_of_decisionEquiv_iff hDec i]

theorem isIrrelevant_iff_of_opt_eq {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S} (hOpt : ∀ s : S, dp₁.Opt s = dp₂.Opt s)
    (i : Fin n) :
    dp₁.isIrrelevant i ↔ dp₂.isIrrelevant i := by
  exact isIrrelevant_iff_of_decisionEquiv_iff (decisionEquiv_iff_of_opt_eq hOpt) i

theorem isMinimalSufficient_iff_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    (I : Finset (Fin n)) :
    dp₁.isMinimalSufficient I ↔ dp₂.isMinimalSufficient I := by
  unfold DecisionProblem.isMinimalSufficient
  constructor
  · intro h
    refine ⟨(isSufficient_iff_of_decisionEquiv_iff hDec I).1 h.1, ?_⟩
    intro J hJ hSuff
    exact h.2 J hJ ((isSufficient_iff_of_decisionEquiv_iff hDec J).2 hSuff)
  · intro h
    refine ⟨(isSufficient_iff_of_decisionEquiv_iff hDec I).2 h.1, ?_⟩
    intro J hJ hSuff
    exact h.2 J hJ ((isSufficient_iff_of_decisionEquiv_iff hDec J).1 hSuff)

theorem sufficientSets_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    dp₁.sufficientSets = dp₂.sufficientSets := by
  ext I
  exact isSufficient_iff_of_decisionEquiv_iff hDec I

theorem relevantSet_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    dp₁.relevantSet = dp₂.relevantSet := by
  ext i
  exact isRelevant_iff_of_decisionEquiv_iff hDec i

theorem certificationStatistic_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    {X : Type*} (f : Set (Finset (Fin n)) → Set (Fin n) → X) :
    f dp₁.sufficientSets dp₁.relevantSet = f dp₂.sufficientSets dp₂.relevantSet := by
  rw [sufficientSets_eq_of_decisionEquiv_iff hDec, relevantSet_eq_of_decisionEquiv_iff hDec]

noncomputable def sufficientSetCount {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : Nat := by
  classical
  exact Fintype.card {I : Finset (Fin n) // dp.isSufficient I}

noncomputable def minimalSufficientSetCount {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : Nat := by
  classical
  exact Fintype.card {I : Finset (Fin n) // dp.isMinimalSufficient I}

noncomputable def relevantCoordCount {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : Nat := by
  classical
  exact Fintype.card {i : Fin n // dp.isRelevant i}

noncomputable def sufficientSetEquiv_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    {I : Finset (Fin n) // dp₁.isSufficient I} ≃ {I : Finset (Fin n) // dp₂.isSufficient I} where
  toFun x := ⟨x.1, (isSufficient_iff_of_decisionEquiv_iff hDec x.1).1 x.2⟩
  invFun x := ⟨x.1, (isSufficient_iff_of_decisionEquiv_iff hDec x.1).2 x.2⟩
  left_inv := by intro x; cases x; rfl
  right_inv := by intro x; cases x; rfl

noncomputable def minimalSufficientSetEquiv_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    {I : Finset (Fin n) // dp₁.isMinimalSufficient I} ≃ {I : Finset (Fin n) // dp₂.isMinimalSufficient I} where
  toFun x := ⟨x.1, (isMinimalSufficient_iff_of_decisionEquiv_iff hDec x.1).1 x.2⟩
  invFun x := ⟨x.1, (isMinimalSufficient_iff_of_decisionEquiv_iff hDec x.1).2 x.2⟩
  left_inv := by intro x; cases x; rfl
  right_inv := by intro x; cases x; rfl

noncomputable def relevantCoordEquiv_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    {i : Fin n // dp₁.isRelevant i} ≃ {i : Fin n // dp₂.isRelevant i} where
  toFun x := ⟨x.1, (isRelevant_iff_of_decisionEquiv_iff hDec x.1).1 x.2⟩
  invFun x := ⟨x.1, (isRelevant_iff_of_decisionEquiv_iff hDec x.1).2 x.2⟩
  left_inv := by intro x; cases x; rfl
  right_inv := by intro x; cases x; rfl

theorem sufficientSetCount_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    sufficientSetCount dp₁ = sufficientSetCount dp₂ := by
  classical
  exact Fintype.card_congr (sufficientSetEquiv_of_decisionEquiv_iff hDec)

theorem minimalSufficientSetCount_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    minimalSufficientSetCount dp₁ = minimalSufficientSetCount dp₂ := by
  classical
  exact Fintype.card_congr (minimalSufficientSetEquiv_of_decisionEquiv_iff hDec)

theorem relevantCoordCount_eq_of_decisionEquiv_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    relevantCoordCount dp₁ = relevantCoordCount dp₂ := by
  classical
  exact Fintype.card_congr (relevantCoordEquiv_of_decisionEquiv_iff hDec)

noncomputable def quotientEquiv_of_decisionEquiv_iff {A S : Type*}
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    dp₁.DecisionQuotientType ≃ dp₂.DecisionQuotientType :=
  Quotient.congrRight fun s s' => by simpa [DecisionProblem.decisionSetoid] using hDec s s'

theorem quotientEquiv_of_decisionEquiv_iff_apply_quotientMap {A S : Type*}
    {dp₁ dp₂ : DecisionProblem A S}
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s')
    (s : S) :
    quotientEquiv_of_decisionEquiv_iff hDec (dp₁.quotientMap s) = dp₂.quotientMap s := by
  rfl

theorem quotientCard_eq_of_decisionEquiv_iff {A S : Type*}
    {dp₁ dp₂ : DecisionProblem A S}
    [Fintype dp₁.DecisionQuotientType] [Fintype dp₂.DecisionQuotientType]
    (hDec : ∀ s s' : S, dp₁.DecisionEquiv s s' ↔ dp₂.DecisionEquiv s s') :
    Fintype.card dp₁.DecisionQuotientType = Fintype.card dp₂.DecisionQuotientType := by
  exact Fintype.card_congr (quotientEquiv_of_decisionEquiv_iff hDec)

theorem isSufficient_positiveAffineTransform_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (α β : S → ℝ) (hβ : ∀ s : S, 0 < β s) (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (positiveAffineTransform α β dp).isSufficient I ↔ dp.isSufficient I := by
  exact isSufficient_iff_of_opt_eq (fun s => opt_eq_positiveAffineTransform α β hβ dp s) I

theorem isRelevant_positiveAffineTransform_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (α β : S → ℝ) (hβ : ∀ s : S, 0 < β s) (dp : DecisionProblem A S) (i : Fin n) :
    (positiveAffineTransform α β dp).isRelevant i ↔ dp.isRelevant i := by
  exact isRelevant_iff_of_opt_eq (fun s => opt_eq_positiveAffineTransform α β hβ dp s) i

/-- Candidate closure axiom: the family ignores mere relabeling of actions. -/
def ClosedUnderActionRelabel (F : ProblemFamily) : Prop :=
  ∀ {A B : Type u} {S : Type v} (e : A ≃ B) (dp : DecisionProblem A S),
    F (A := A) (S := S) dp → F (A := B) (S := S) (relabelActions e dp)

/-- Candidate closure axiom: the family ignores mere relabeling of states. -/
def ClosedUnderStateRelabel (F : ProblemFamily) : Prop :=
  ∀ {A : Type u} {S T : Type v} (e : S ≃ T) (dp : DecisionProblem A S),
    F (A := A) (S := S) dp → F (A := A) (S := T) (relabelStates e dp)

/-- Expressivity axiom saying the family contains a realizer for every labeling kernel. -/
def ContainsAllRealizers (F : ProblemFamily) : Prop :=
  ∀ {S : Type v} {T : Type u} [DecidableEq T] (φ : S → T),
    F (A := T) (S := S) (realizingProblem φ)

/-- Every labeling kernel can be witnessed inside the family. -/
def KernelUniversal (F : ProblemFamily) : Prop :=
  ∀ {S : Type v} {T : Type u} [DecidableEq T] (φ : S → T),
    ∃ dp : DecisionProblem T S,
      F (A := T) (S := S) dp ∧ ∀ s : S, dp.Opt s = ({φ s} : Set T)

/-- Minimal semantic invariance required of any structural frontier class. -/
def RelabelInvariant (F : ProblemFamily) : Prop :=
  ClosedUnderActionRelabel F ∧ ClosedUnderStateRelabel F

/-- The universal class of all decision problems. -/
def allProblems : ProblemFamily := fun {_ _} _ => True

theorem relabelStates_realizingProblem {S S' T : Type*} [DecidableEq T]
    (e : S ≃ S') (φ : S → T) :
    relabelStates e (realizingProblem φ) = realizingProblem (fun s' => φ (e.symm s')) := by
  rfl

theorem relabelActions_realizingProblem {S T U : Type*} [DecidableEq T] [DecidableEq U]
    (e : T ≃ U) (φ : S → T) :
    relabelActions e (realizingProblem φ) = realizingProblem (fun s => e (φ s)) := by
  ext u s
  by_cases h : u = e (φ s)
  · have hs : e.symm u = φ s := by simp [h] at *
    simp [relabelActions, realizingProblem, h]
  · have hs : e.symm u ≠ φ s := by
      intro hsu
      apply h
      simpa using congrArg e hsu
    simp [relabelActions, realizingProblem, h, hs]

theorem allProblems_closedUnderActionRelabel : ClosedUnderActionRelabel allProblems := by
  unfold ClosedUnderActionRelabel allProblems
  intro A B S e dp h
  trivial

theorem allProblems_closedUnderStateRelabel : ClosedUnderStateRelabel allProblems := by
  unfold ClosedUnderStateRelabel allProblems
  intro A S T e dp h
  trivial

theorem allProblems_containsAllRealizers : ContainsAllRealizers allProblems := by
  intro S T inst φ
  trivial

theorem isSufficient_relabelActions_iff {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (e : A ≃ B) (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (relabelActions e dp).isSufficient I ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact (decisionEquiv_relabelActions_iff e dp s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (decisionEquiv_relabelActions_iff e dp s s').2 (h s s' hagree)

theorem isSufficient_relabelStates_iff {A S T : Type*} {n : ℕ} [CoordinateSpace S n]
    (e : S ≃ T) (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    @DecisionProblem.isSufficient A T n (transportedCoordinateSpace e) (relabelStates e dp) I
      ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    have hagree' : @agreeOn T n (transportedCoordinateSpace e) (e s) (e s') I := by
      rw [agreeOn_transported_iff]
      simpa using hagree
    have hEq := h (e s) (e s') hagree'
    simpa using (decisionEquiv_relabelStates_iff e dp (e s) (e s')).1 hEq
  · intro h t t' hagree
    have hagree' : agreeOn (e.symm t) (e.symm t') I :=
      (agreeOn_transported_iff e t t' I).1 hagree
    have hEq := h (e.symm t) (e.symm t') hagree'
    exact (decisionEquiv_relabelStates_iff e dp t t').2 hEq

theorem kernelUniversal_of_containsAllRealizers (F : ProblemFamily)
    (hF : ContainsAllRealizers F) : KernelUniversal F := by
  intro S T inst φ
  refine ⟨realizingProblem φ, hF (S := S) (T := T) φ, ?_⟩
  intro s
  exact realizingProblem_opt_eq_singleton φ s

theorem allProblems_kernelUniversal : KernelUniversal allProblems := by
  exact kernelUniversal_of_containsAllRealizers allProblems allProblems_containsAllRealizers

theorem allProblems_relabelInvariant : RelabelInvariant allProblems := by
  exact ⟨allProblems_closedUnderActionRelabel, allProblems_closedUnderStateRelabel⟩

theorem relabelInvariant_not_enough :
    RelabelInvariant allProblems ∧ KernelUniversal allProblems := by
  exact ⟨allProblems_relabelInvariant, allProblems_kernelUniversal⟩

theorem optRangeCard_realizingIdentity (k : Nat) :
    let dp : DecisionProblem (Fin (k + 1)) (Fin (k + 1)) :=
      realizingProblem (fun i : Fin (k + 1) => i)
    letI : Fintype ↥(Set.range dp.Opt) := (Set.finite_range dp.Opt).fintype
    Fintype.card ↥(Set.range dp.Opt) = k + 1 := by
  let dp : DecisionProblem (Fin (k + 1)) (Fin (k + 1)) :=
    realizingProblem (fun i : Fin (k + 1) => i)
  letI : Fintype ↥(Set.range dp.Opt) := (Set.finite_range dp.Opt).fintype
  let f : Fin (k + 1) → ↥(Set.range dp.Opt) :=
    fun t => ⟨dp.Opt t, ⟨t, rfl⟩⟩
  have hInj : Function.Injective f := by
    intro s s' hEq
    have hset : dp.Opt s = dp.Opt s' := congrArg Subtype.val hEq
    have hsing : ({s} : Set (Fin (k + 1))) = ({s'} : Set (Fin (k + 1))) := by
      simpa [dp, realizingProblem_opt_eq_singleton] using hset
    have hmem : s ∈ ({s'} : Set (Fin (k + 1))) := by
      simpa [hsing] using (show s ∈ ({s} : Set (Fin (k + 1))) by simp)
    simpa using hmem
  have hSurj : Function.Surjective f := by
    intro y
    rcases y.property with ⟨s, hs⟩
    refine ⟨s, ?_⟩
    apply Subtype.ext
    simpa [dp, f, realizingProblem_opt_eq_singleton] using hs
  have hLower : k + 1 ≤ Fintype.card ↥(Set.range dp.Opt) :=
    by simpa using Fintype.card_le_of_injective f hInj
  have hUpper : Fintype.card ↥(Set.range dp.Opt) ≤ k + 1 :=
    by simpa using Fintype.card_le_of_surjective f hSurj
  exact Nat.le_antisymm hUpper hLower

theorem quotientCard_realizingIdentity (k : Nat) :
    let dp : DecisionProblem (Fin (k + 1)) (Fin (k + 1)) :=
      realizingProblem (fun i : Fin (k + 1) => i)
    letI : Fintype ↥(Set.range dp.Opt) := (Set.finite_range dp.Opt).fintype
    letI : Fintype dp.DecisionQuotientType := Fintype.ofEquiv _ dp.quotientEquivOptRange.symm
    Fintype.card dp.DecisionQuotientType = k + 1 := by
  let dp : DecisionProblem (Fin (k + 1)) (Fin (k + 1)) :=
    realizingProblem (fun i : Fin (k + 1) => i)
  letI : Fintype ↥(Set.range dp.Opt) := (Set.finite_range dp.Opt).fintype
  letI : Fintype dp.DecisionQuotientType := Fintype.ofEquiv _ dp.quotientEquivOptRange.symm
  calc
    Fintype.card dp.DecisionQuotientType
      = Fintype.card ↥(Set.range dp.Opt) := Fintype.card_congr dp.quotientEquivOptRange
    _ = k + 1 := optRangeCard_realizingIdentity k

end Paper4dFrontier
