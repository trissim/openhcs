import DecisionQuotient.Information
import DecisionQuotient.Physics.TemporalCountingGap

open Classical

namespace DecisionQuotient
namespace InflationEntropyBridge

structure DynamicDecisionFamily (A : Type*) where
  State : ℕ → Type*
  instFintype : ∀ t : ℕ, Fintype (State t)
  instNonempty : ∀ t : ℕ, Nonempty (State t)
  dp : ∀ t : ℕ, DecisionProblem A (State t)
  emb : ∀ {t₁ t₂ : ℕ}, t₁ ≤ t₂ → State t₁ → State t₂
  optCompat : ∀ {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) (s : State t₁),
      (dp t₂).Opt (emb h s) = (dp t₁).Opt s

attribute [instance] DynamicDecisionFamily.instFintype
attribute [instance] DynamicDecisionFamily.instNonempty

variable {A : Type*}

theorem optCompat_of_utilityCompat
    {S₁ S₂ : Type*}
    (dp₁ : DecisionProblem A S₁)
    (dp₂ : DecisionProblem A S₂)
    (emb : S₁ → S₂)
    (hU : ∀ (s : S₁) (a : A), dp₂.utility a (emb s) = dp₁.utility a s) :
    ∀ s : S₁, dp₂.Opt (emb s) = dp₁.Opt s := by
  intro s
  ext a
  unfold DecisionProblem.Opt DecisionProblem.isOptimal
  constructor
  · intro ha a'
    simpa [hU s a, hU s a'] using ha a'
  · intro ha a'
    simpa [hU s a, hU s a'] using ha a'

structure DynamicUtilityFamily (A : Type*) where
  State : ℕ → Type*
  instFintype : ∀ t : ℕ, Fintype (State t)
  instNonempty : ∀ t : ℕ, Nonempty (State t)
  dp : ∀ t : ℕ, DecisionProblem A (State t)
  emb : ∀ {t₁ t₂ : ℕ}, t₁ ≤ t₂ → State t₁ → State t₂
  utilityCompat : ∀ {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) (s : State t₁) (a : A),
      (dp t₂).utility a (emb h s) = (dp t₁).utility a s

attribute [instance] DynamicUtilityFamily.instFintype
attribute [instance] DynamicUtilityFamily.instNonempty

def DynamicUtilityFamily.toDynamicDecisionFamily
    (M : DynamicUtilityFamily A) : DynamicDecisionFamily A where
  State := M.State
  instFintype := M.instFintype
  instNonempty := M.instNonempty
  dp := M.dp
  emb := M.emb
  optCompat := by
    intro t₁ t₂ h s
    exact optCompat_of_utilityCompat (dp₁ := M.dp t₁) (dp₂ := M.dp t₂)
      (emb := M.emb h) (hU := fun s a => M.utilityCompat h s a) s

def numOptClassesAt [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) (t : ℕ) : ℕ :=
  (M.dp t).numOptClasses

noncomputable def quotientEntropyAt [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) (t : ℕ) : ℝ :=
  (M.dp t).quotientEntropy

theorem classes_monotone [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    numOptClassesAt M t₁ ≤ numOptClassesAt M t₂ := by
  classical
  unfold numOptClassesAt DecisionProblem.numOptClasses
  have hsubset :
      (Finset.univ.image (M.dp t₁).Opt) ⊆ (Finset.univ.image (M.dp t₂).Opt) := by
    intro O hO
    rcases Finset.mem_image.mp hO with ⟨s, hs, hEq⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨M.emb (t₁ := t₁) (t₂ := t₂) h s, Finset.mem_univ _, ?_⟩
    exact (M.optCompat (t₁ := t₁) (t₂ := t₂) h s).trans hEq
  exact Finset.card_le_card hsubset

theorem entropy_monotone [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    quotientEntropyAt M t₁ ≤ quotientEntropyAt M t₂ := by
  have hClasses : numOptClassesAt M t₁ ≤ numOptClassesAt M t₂ :=
    classes_monotone M h
  have hPos1 : 0 < (M.dp t₁).numOptClasses := (M.dp t₁).numOptClasses_pos
  have hLog :
      Real.log ((M.dp t₁).numOptClasses : ℝ) ≤ Real.log ((M.dp t₂).numOptClasses : ℝ) :=
    Real.log_le_log (by exact_mod_cast hPos1) (by exact_mod_cast hClasses)
  have hInvNonneg : 0 ≤ (Real.log 2)⁻¹ := by positivity
  unfold quotientEntropyAt DecisionProblem.quotientEntropy
  simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_right hLog hInvNonneg

theorem classes_monotone_of_utilityCompat [DecidableEq (Set A)]
    (M : DynamicUtilityFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    numOptClassesAt (M.toDynamicDecisionFamily) t₁ ≤
      numOptClassesAt (M.toDynamicDecisionFamily) t₂ :=
  classes_monotone (M.toDynamicDecisionFamily) h

theorem entropy_monotone_of_utilityCompat [DecidableEq (Set A)]
    (M : DynamicUtilityFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    quotientEntropyAt (M.toDynamicDecisionFamily) t₁ ≤
      quotientEntropyAt (M.toDynamicDecisionFamily) t₂ :=
  entropy_monotone (M.toDynamicDecisionFamily) h

theorem classes_strict_increase [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂)
    (hNew : ∃ O : Set A,
      O ∈ (Finset.univ.image (M.dp t₂).Opt) ∧
      O ∉ (Finset.univ.image (M.dp t₁).Opt)) :
    numOptClassesAt M t₁ < numOptClassesAt M t₂ := by
  classical
  let old : Finset (Set A) := Finset.univ.image (M.dp t₁).Opt
  let new : Finset (Set A) := Finset.univ.image (M.dp t₂).Opt
  have hsubset : old ⊆ new := by
    intro O hO
    rcases Finset.mem_image.mp hO with ⟨s, hs, hEq⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨M.emb (t₁ := t₁) (t₂ := t₂) h s, Finset.mem_univ _, ?_⟩
    exact (M.optCompat (t₁ := t₁) (t₂ := t₂) h s).trans hEq
  rcases hNew with ⟨O, hOnew, hOnotOld⟩
  have hneq : old ≠ new := by
    intro hEq
    have hOold : O ∈ old := hEq.symm ▸ hOnew
    exact hOnotOld hOold
  have hssub : old ⊂ new := by
    refine Finset.ssubset_iff_subset_ne.mpr ?_
    exact ⟨hsubset, hneq⟩
  have hcard : old.card < new.card := Finset.card_lt_card hssub
  unfold numOptClassesAt DecisionProblem.numOptClasses
  simpa [old, new] using hcard

theorem entropy_strict_increase [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂)
    (hNew : ∃ O : Set A,
      O ∈ (Finset.univ.image (M.dp t₂).Opt) ∧
      O ∉ (Finset.univ.image (M.dp t₁).Opt)) :
    quotientEntropyAt M t₁ < quotientEntropyAt M t₂ := by
  have hClasses : numOptClassesAt M t₁ < numOptClassesAt M t₂ :=
    classes_strict_increase M h hNew
  have hPos1 : 0 < (M.dp t₁).numOptClasses := (M.dp t₁).numOptClasses_pos
  have hLog :
      Real.log ((M.dp t₁).numOptClasses : ℝ) < Real.log ((M.dp t₂).numOptClasses : ℝ) :=
    Real.log_lt_log (by exact_mod_cast hPos1) (by exact_mod_cast hClasses)
  have hInvPos : 0 < (Real.log 2)⁻¹ := by positivity
  unfold quotientEntropyAt DecisionProblem.quotientEntropy
  simpa [div_eq_mul_inv] using mul_lt_mul_of_pos_right hLog hInvPos

theorem thermal_floor_monotone_of_classes [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂)
    (kB T : ℝ) (hkB : 0 < kB) (hT : 0 < T) :
    kB * T * Real.log (numOptClassesAt M t₁ : ℝ) ≤
      kB * T * Real.log (numOptClassesAt M t₂ : ℝ) := by
  have hClasses : numOptClassesAt M t₁ ≤ numOptClassesAt M t₂ :=
    classes_monotone M h
  have hPos1 : 0 < numOptClassesAt M t₁ := by
    unfold numOptClassesAt
    exact (M.dp t₁).numOptClasses_pos
  have hLog :
      Real.log (numOptClassesAt M t₁ : ℝ) ≤ Real.log (numOptClassesAt M t₂ : ℝ) :=
    Real.log_le_log (by exact_mod_cast hPos1) (by exact_mod_cast hClasses)
  have hScale : 0 ≤ kB * T := mul_nonneg hkB.le hT.le
  exact mul_le_mul_of_nonneg_left hLog hScale

theorem thermal_floor_strict_of_new_class [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂)
    (hNew : ∃ O : Set A,
      O ∈ (Finset.univ.image (M.dp t₂).Opt) ∧
      O ∉ (Finset.univ.image (M.dp t₁).Opt))
    (kB T : ℝ) (hkB : 0 < kB) (hT : 0 < T) :
    kB * T * Real.log (numOptClassesAt M t₁ : ℝ) <
      kB * T * Real.log (numOptClassesAt M t₂ : ℝ) := by
  have hClasses : numOptClassesAt M t₁ < numOptClassesAt M t₂ :=
    classes_strict_increase M h hNew
  have hPos1 : 0 < numOptClassesAt M t₁ := by
    unfold numOptClassesAt
    exact (M.dp t₁).numOptClasses_pos
  have hLog :
      Real.log (numOptClassesAt M t₁ : ℝ) < Real.log (numOptClassesAt M t₂ : ℝ) :=
    Real.log_lt_log (by exact_mod_cast hPos1) (by exact_mod_cast hClasses)
  have hScale : 0 < kB * T := mul_pos hkB hT
  exact mul_lt_mul_of_pos_left hLog hScale

theorem later_energy_floor_implies_earlier_floor [DecidableEq (Set A)]
    (M : DynamicDecisionFamily A) {t₁ t₂ : ℕ} (h : t₁ ≤ t₂)
    (kB T : ℝ) (hkB : 0 < kB) (hT : 0 < T)
    (E₂ : ℝ)
    (hE₂ : E₂ ≥ kB * T * Real.log (numOptClassesAt M t₂ : ℝ)) :
    E₂ ≥ kB * T * Real.log (numOptClassesAt M t₁ : ℝ) := by
  have hFloor := thermal_floor_monotone_of_classes M h kB T hkB hT
  linarith

namespace Temporal

open Physics.TemporalCountingGap

def StateAt (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) : Type :=
  Fin (StateSpaceCardinality psf.a ρ t)

instance instFintypeStateAt (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) :
    Fintype (StateAt psf ρ t) := by
  unfold StateAt
  infer_instance

theorem state_cardinality_pos
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    (t : ℕ) :
    0 < StateSpaceCardinality psf.a ρ t := by
  unfold StateSpaceCardinality
  have hpow : 0 < (psf.a t) ^ 3 := Nat.pow_pos (psf.h_pos t)
  exact Nat.mul_pos hρ hpow

instance instNonemptyStateAt
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ) (t : ℕ) :
    Nonempty (StateAt psf ρ t) := by
  refine ⟨⟨0, ?_⟩⟩
  exact state_cardinality_pos psf ρ hρ t

def embedState
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    StateAt psf ρ t₁ → StateAt psf ρ t₂ := by
  intro x
  refine ⟨x.1, ?_⟩
  exact lt_of_lt_of_le x.2 (states_nondecreasing psf ρ hρ t₁ t₂ h)

theorem embedState_injective
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    Function.Injective (embedState psf ρ hρ h) := by
  intro x y hxy
  exact Fin.ext (by simpa [embedState] using congrArg Fin.val hxy)

theorem state_cardinality_strict_growth
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t₁ t₂ : ℕ} (h : t₁ < t₂) :
    Fintype.card (StateAt psf ρ t₁) < Fintype.card (StateAt psf ρ t₂) := by
  simpa [StateAt] using states_increase_with_time psf ρ hρ t₁ t₂ h

structure TemporalUtilityFamily
    (A : Type*)
    (psf : PhysicalScaleFactor)
    (ρ : ℕ)
    (hρ : 0 < ρ) where
  dpAt : ∀ t : ℕ, DecisionProblem A (StateAt psf ρ t)
  utilityCompat : ∀ {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) (s : StateAt psf ρ t₁) (a : A),
      (dpAt t₂).utility a (embedState psf ρ hρ h s) = (dpAt t₁).utility a s

def TemporalUtilityFamily.toDynamicUtilityFamily
    {A : Type*}
    {psf : PhysicalScaleFactor}
    {ρ : ℕ}
    {hρ : 0 < ρ}
    (M : TemporalUtilityFamily A psf ρ hρ) :
    DynamicUtilityFamily A where
  State := StateAt psf ρ
  instFintype := instFintypeStateAt psf ρ
  instNonempty := instNonemptyStateAt psf ρ hρ
  dp := M.dpAt
  emb := fun {_ _} h => embedState psf ρ hρ h
  utilityCompat := by
    intro t₁ t₂ h s a
    exact M.utilityCompat h s a

theorem temporal_classes_monotone_of_utilityCompat
    {A : Type*}
    {psf : PhysicalScaleFactor}
    {ρ : ℕ}
    {hρ : 0 < ρ}
    [DecidableEq (Set A)]
    (M : TemporalUtilityFamily A psf ρ hρ)
    {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    numOptClassesAt (M.toDynamicUtilityFamily.toDynamicDecisionFamily) t₁ ≤
      numOptClassesAt (M.toDynamicUtilityFamily.toDynamicDecisionFamily) t₂ := by
  exact classes_monotone_of_utilityCompat (M := M.toDynamicUtilityFamily) h

theorem temporal_entropy_monotone_of_utilityCompat
    {A : Type*}
    {psf : PhysicalScaleFactor}
    {ρ : ℕ}
    {hρ : 0 < ρ}
    [DecidableEq (Set A)]
    (M : TemporalUtilityFamily A psf ρ hρ)
    {t₁ t₂ : ℕ} (h : t₁ ≤ t₂) :
    quotientEntropyAt (M.toDynamicUtilityFamily.toDynamicDecisionFamily) t₁ ≤
      quotientEntropyAt (M.toDynamicUtilityFamily.toDynamicDecisionFamily) t₂ := by
  exact entropy_monotone_of_utilityCompat (M := M.toDynamicUtilityFamily) h

end Temporal

end InflationEntropyBridge
end DecisionQuotient
