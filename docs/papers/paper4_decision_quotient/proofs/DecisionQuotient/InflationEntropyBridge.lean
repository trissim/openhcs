import DecisionQuotient.Information
import DecisionQuotient.BayesFromDQ
import DecisionQuotient.Physics.ClaimTransport
import DecisionQuotient.Physics.ConstraintForcing
import DecisionQuotient.Physics.MeasureNecessity
import DecisionQuotient.Physics.TemporalCountingGap

open Classical MeasureTheory

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

instance instMeasurableSpaceStateAt
    (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) :
    MeasurableSpace (StateAt psf ρ t) := ⊤

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

/-- A support-complete semantics for valid answers at time `t` is any carrier
    whose image covers the full valid successor-state slice `StateAt psf ρ t`. -/
structure SupportCompleteValidAnswerSemantics
    (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) where
  Answer : Type*
  toState : Answer → StateAt psf ρ t
  support_complete : Function.Surjective toState

/-- Canonical support-complete semantics: the valid-answer carrier is the
    expanded successor slice itself. -/
def canonicalValidAnswerSemantics
    (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) :
    SupportCompleteValidAnswerSemantics psf ρ t where
  Answer := StateAt psf ρ t
  toState := id
  support_complete := by
    intro s
    exact ⟨s, rfl⟩

theorem canonicalValidAnswerSemantics_identifies_StateAt
    (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) :
    ∀ s : StateAt psf ρ t,
      ∃ a : (canonicalValidAnswerSemantics psf ρ t).Answer,
        (canonicalValidAnswerSemantics psf ρ t).toState a = s := by
  intro s
  exact ⟨s, rfl⟩

/-- Uniform counting-normalized prior on the temporally valid state space. -/
noncomputable def uniformPrior
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ) (t : ℕ) :
    ProbDist (StateAt psf ρ t) where
  prob := fun _ => 1 / (Fintype.card (StateAt psf ρ t) : ℝ)
  nonneg := by
    intro _
    positivity
  sum_one := by
    have hCardPos : 0 < Fintype.card (StateAt psf ρ t) := by
      simpa [StateAt] using state_cardinality_pos psf ρ hρ t
    have hCardNe : (Fintype.card (StateAt psf ρ t) : ℝ) ≠ 0 := by
      exact_mod_cast Nat.ne_of_gt hCardPos
    calc
      Finset.univ.sum (fun _ : StateAt psf ρ t =>
          1 / (Fintype.card (StateAt psf ρ t) : ℝ)) =
          (Fintype.card (StateAt psf ρ t) : ℝ) *
            (1 / (Fintype.card (StateAt psf ρ t) : ℝ)) := by
            simp
      _ = 1 := by
        field_simp [hCardNe]

@[simp] theorem uniformPrior_prob
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ) (t : ℕ)
    (s : StateAt psf ρ t) :
    (uniformPrior psf ρ hρ t).prob s =
      1 / (Fintype.card (StateAt psf ρ t) : ℝ) := by
  rfl

/-- Positive time already forces more than one valid physical state under the
    cosmological-expansion model. -/
theorem state_cardinality_gt_one_of_positive_time
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    1 < Fintype.card (StateAt psf ρ t) := by
  have hGrowth : StateSpaceCardinality psf.a ρ 0 < StateSpaceCardinality psf.a ρ t :=
    states_increase_with_time psf ρ hρ 0 t ht
  have hOrigin : StateSpaceCardinality psf.a ρ 0 = ρ := state_space_at_origin psf ρ
  have hOneLeOrigin : 1 ≤ StateSpaceCardinality psf.a ρ 0 := by
    simpa [hOrigin] using Nat.succ_le_of_lt hρ
  have hCard : 1 < StateSpaceCardinality psf.a ρ t :=
    lt_of_le_of_lt hOneLeOrigin hGrowth
  simpa [StateAt] using hCard

/-- On any temporally expanded state space with more than one valid state, the
    normalized counting prior cannot collapse to certainty on one hypothesis. -/
theorem uniformPrior_uncertainty_of_card_gt_one
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ) (t : ℕ)
    (hCard : 1 < Fintype.card (StateAt psf ρ t)) :
    UncertaintyForced (uniformPrior psf ρ hρ t) := by
  intro hCertain
  rcases hCertain with ⟨s, hs⟩
  have hCardPos : 0 < Fintype.card (StateAt psf ρ t) := by
    exact lt_trans Nat.zero_lt_one hCard
  have hCardNe : (Fintype.card (StateAt psf ρ t) : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt hCardPos
  have hs' : 1 / (Fintype.card (StateAt psf ρ t) : ℝ) = 1 := by
    simpa [uniformPrior] using hs
  have hEq : (Fintype.card (StateAt psf ρ t) : ℝ) = 1 := by
    field_simp [hCardNe] at hs'
    linarith
  have hEqNat : Fintype.card (StateAt psf ρ t) = 1 := by
    exact_mod_cast hEq
  exact (Nat.ne_of_gt hCard) hEqNat

/-- Expansion-induced multiplicity of valid states yields nondegenerate belief
    under the normalized counting prior. -/
theorem uniformPrior_nondegenerate_of_card_gt_one
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ) (t : ℕ)
    (hCard : 1 < Fintype.card (StateAt psf ρ t)) :
    NondegenerateBelief (uniformPrior psf ρ hρ t) := by
  exact nondegenerateBelief_of_uncertaintyForced _
    (uniformPrior_uncertainty_of_card_gt_one psf ρ hρ t hCard)

/-- Positive time gives expansion-induced uncertainty for the uniform prior on
    valid successor states. -/
theorem uniformPrior_uncertainty_of_positive_time
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    UncertaintyForced (uniformPrior psf ρ hρ t) := by
  exact uniformPrior_uncertainty_of_card_gt_one psf ρ hρ t
    (state_cardinality_gt_one_of_positive_time psf ρ hρ ht)

/-- Positive time gives a genuinely nondegenerate belief state on the expanded
    valid successor space. -/
theorem uniformPrior_nondegenerate_of_positive_time
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    NondegenerateBelief (uniformPrior psf ρ hρ t) := by
  exact uniformPrior_nondegenerate_of_card_gt_one psf ρ hρ t
    (state_cardinality_gt_one_of_positive_time psf ρ hρ ht)

/-- Raw counting on the expanded state space is not probability-normalized once
    multiple valid states exist; stochastic reasoning therefore requires a
    probability normalization layer. -/
theorem counting_measure_not_probability_on_stateAt_of_positive_time
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    ¬ IsProbabilityMeasure (Measure.count : Measure (StateAt psf ρ t)) := by
  exact Physics.MeasureNecessity.counting_measure_not_probability_of_card_gt_one
    (state_cardinality_gt_one_of_positive_time psf ρ hρ ht)

/-- Main bridge: cosmological expansion yields a support-complete probabilistic
    prior on valid successor states, and raw counting alone is insufficient
    because it is not probability-normalized once the space has expanded. -/
theorem cosmological_expansion_forces_probabilistic_reasoning
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    ∃ prior : ProbDist (StateAt psf ρ t),
      UncertaintyForced prior ∧
      NondegenerateBelief prior ∧
      ¬ IsProbabilityMeasure (Measure.count : Measure (StateAt psf ρ t)) := by
  refine ⟨uniformPrior psf ρ hρ t, ?_⟩
  constructor
  · exact uniformPrior_uncertainty_of_positive_time psf ρ hρ ht
  constructor
  · exact uniformPrior_nondegenerate_of_positive_time psf ρ hρ ht
  · exact counting_measure_not_probability_on_stateAt_of_positive_time psf ρ hρ ht

/-- Strengthened bridge: the expanded valid-answer space admits a named
    support-complete semantics, and on that full support one is forced to pass
    from raw counting to a normalized probability distribution. -/
theorem cosmological_expansion_forces_support_complete_probabilistic_reasoning
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    Function.Surjective (canonicalValidAnswerSemantics psf ρ t).toState ∧
    UncertaintyForced (uniformPrior psf ρ hρ t) ∧
    NondegenerateBelief (uniformPrior psf ρ hρ t) ∧
    ¬ IsProbabilityMeasure (Measure.count : Measure (StateAt psf ρ t)) := by
  constructor
  · exact (canonicalValidAnswerSemantics psf ρ t).support_complete
  constructor
  · exact uniformPrior_uncertainty_of_positive_time psf ρ hρ ht
  constructor
  · exact uniformPrior_nondegenerate_of_positive_time psf ρ hρ ht
  · exact counting_measure_not_probability_on_stateAt_of_positive_time psf ρ hρ ht

/-- Chosen physical-decision encoding whose instance carrier is the expanded
    valid successor slice itself. Every instance is identified with its
    corresponding `StateAt` element, and the encoded decision problem is fixed
    on that slice. -/
def stateIndexedPhysicalEncoding
    (dp : DecisionProblem A (StateAt psf ρ t)) :
    Physics.ClaimTransport.PhysicalEncoding (StateAt psf ρ t) A (StateAt psf ρ t) where
  encode := fun _ => dp

/-- Physical-state semantics attached to the chosen state-indexed encoding. -/
def stateIndexedPhysicalStateSemantics
    (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) :
    Physics.ClaimTransport.PhysicalStateSemantics (StateAt psf ρ t) (StateAt psf ρ t) where
  observe := id
  isPhysical := fun _ => True
  realizable := by
    intro s _
    exact ⟨s, rfl⟩

/-- Explicit identification theorem: under the chosen physical-decision
    encoding, every physically valid answer is exactly a point of `StateAt`. -/
theorem stateIndexedPhysicalEncoding_identifies_StateAt
    (dp : DecisionProblem A (StateAt psf ρ t)) :
    ∀ s : StateAt psf ρ t,
      ∃ p : StateAt psf ρ t,
        (stateIndexedPhysicalStateSemantics psf ρ t).observe p = s ∧
        (stateIndexedPhysicalEncoding (psf := psf) (ρ := ρ) (t := t) dp).encode p = dp := by
  intro s
  exact ⟨s, rfl, rfl⟩

/-- If a physical decision is deadline-forced while cosmological expansion has
    already produced multiple valid successor states, then the decision cannot
    be represented by a degenerate belief state. -/
theorem forced_decision_requires_probabilistic_reasoning_of_positive_time
    {S A Θ : Type*}
    (F : Physics.ConstraintForcing.LogicTimeScaffold S)
    (L : Physics.ConstraintForcing.TimedLawFamily Θ S A)
    (θ : Θ)
    (hState : ∃ s, F.consistent s)
    (hDeadline : Physics.ConstraintForcing.deadlineForcesAction F L θ)
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t : ℕ} (ht : 0 < t) :
    ∃ _ : A, NondegenerateBelief (uniformPrior psf ρ hρ t) := by
  have hAction : ActionForced A :=
    Physics.ConstraintForcing.actionForced_of_deadline F L θ hState hDeadline
  exact forced_action_under_uncertainty hAction _
    (uniformPrior_uncertainty_of_positive_time psf ρ hρ ht)

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
