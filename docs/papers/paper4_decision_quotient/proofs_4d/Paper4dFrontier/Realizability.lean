import DecisionQuotient.Quotient
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
