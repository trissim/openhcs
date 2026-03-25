/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SupportExpansion.lean

  Finite support-shell semantics for the local certified action family. This
  models the round-wise union of shells used by the runtime support-expansion
  strategy.
-/
import Mathlib.Data.Finset.Max
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace SupportExpansion

open Classical

/-- Number of coarser shells added to the active round. -/
def coarserShellCount (roundIndex supportExpansionLevel : ℕ) : ℕ :=
  min roundIndex supportExpansionLevel

/-- Shell levels carried by the runtime support-expansion family. -/
noncomputable def supportShellLevels (roundIndex supportExpansionLevel : ℕ) : Finset ℕ :=
  (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1)).image
    (fun offset => roundIndex - offset)

/-- The merged runtime family keeps one noop plus 12 local actions per shell. -/
noncomputable def mergedActionCount (roundIndex supportExpansionLevel : ℕ) : ℕ :=
  1 + 12 * (supportShellLevels roundIndex supportExpansionLevel).card

/-- Translation step used by shell `roundIndex` under dyadic refinement. -/
noncomputable def dyadicTranslationStep (baseStep : ℝ) (roundIndex : ℕ) : ℝ :=
  baseStep / (2 : ℝ) ^ roundIndex

/-- A dyadic refinement round is adequate once its translation step is at most
    the requested target resolution. -/
def AdequateDyadicRound (baseStep target : ℝ) (roundIndex : ℕ) : Prop :=
  dyadicTranslationStep baseStep roundIndex ≤ target

/-- Joint adequacy for two dyadic step channels (e.g. translation and the
    rotation-induced displacement radius). -/
def AdequateJointDyadicRound (baseStep₁ baseStep₂ target : ℝ) (roundIndex : ℕ) : Prop :=
  AdequateDyadicRound baseStep₁ target roundIndex ∧
    AdequateDyadicRound baseStep₂ target roundIndex

theorem roundIndex_mem_supportShellLevels (roundIndex supportExpansionLevel : ℕ) :
    roundIndex ∈ supportShellLevels roundIndex supportExpansionLevel := by
  unfold supportShellLevels
  refine Finset.mem_image.mpr ?_
  refine ⟨0, by simp [coarserShellCount], by simp⟩

theorem coarsestShell_mem_supportShellLevels (roundIndex supportExpansionLevel : ℕ) :
    roundIndex - coarserShellCount roundIndex supportExpansionLevel
      ∈ supportShellLevels roundIndex supportExpansionLevel := by
  unfold supportShellLevels
  refine Finset.mem_image.mpr ?_
  refine ⟨coarserShellCount roundIndex supportExpansionLevel, by simp, rfl⟩

theorem supportShellLevels_card (roundIndex supportExpansionLevel : ℕ) :
    (supportShellLevels roundIndex supportExpansionLevel).card =
      coarserShellCount roundIndex supportExpansionLevel + 1 := by
  classical
  unfold supportShellLevels
  calc
    (Finset.image (fun offset => roundIndex - offset)
        (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1))).card
        = (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1)).card := by
            refine Finset.card_image_of_injOn ?_
            intro a ha
            intro b hb
            intro hEq
            have haRange : a < coarserShellCount roundIndex supportExpansionLevel + 1 := by
              simpa using ha
            have hbRange : b < coarserShellCount roundIndex supportExpansionLevel + 1 := by
              simpa using hb
            have ha' : a ≤ roundIndex := by
              have hMin : coarserShellCount roundIndex supportExpansionLevel ≤ roundIndex := by
                unfold coarserShellCount
                exact min_le_left _ _
              omega
            have hb' : b ≤ roundIndex := by
              have hMin : coarserShellCount roundIndex supportExpansionLevel ≤ roundIndex := by
                unfold coarserShellCount
                exact min_le_left _ _
              omega
            have hAdd : roundIndex = (roundIndex - b) + a := by
              simpa [Nat.sub_add_cancel ha'] using congrArg (fun t => t + a) hEq
            have hRound : roundIndex = (roundIndex - b) + b := by
              exact (Nat.sub_add_cancel hb').symm
            have hSame : (roundIndex - b) + a = (roundIndex - b) + b := by
              calc
                (roundIndex - b) + a = roundIndex := by exact hAdd.symm
                _ = (roundIndex - b) + b := by exact hRound
            exact Nat.add_left_cancel hSame
    _ = coarserShellCount roundIndex supportExpansionLevel + 1 := by
          simp

theorem supportShellLevels_monotone {roundIndex e1 e2 : ℕ}
    (hExp : e1 ≤ e2) :
    supportShellLevels roundIndex e1 ⊆ supportShellLevels roundIndex e2 := by
  intro shell hs
  unfold supportShellLevels at hs ⊢
  rcases Finset.mem_image.mp hs with ⟨offset, hOffset, rfl⟩
  refine Finset.mem_image.mpr ?_
  refine ⟨offset, ?_, rfl⟩
  simp only [Finset.mem_range] at hOffset ⊢
  unfold coarserShellCount at hOffset ⊢
  omega

theorem supportShellLevels_max' (roundIndex supportExpansionLevel : ℕ) :
    (supportShellLevels roundIndex supportExpansionLevel).max'
      ⟨roundIndex, roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel⟩ = roundIndex := by
  refine le_antisymm ?_ ?_
  · have hMaxMem :
        (supportShellLevels roundIndex supportExpansionLevel).max'
          ⟨roundIndex, roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel⟩
          ∈ supportShellLevels roundIndex supportExpansionLevel :=
      Finset.max'_mem _ _
    rcases Finset.mem_image.mp hMaxMem with ⟨offset, hOffset, hEq⟩
    have hLe : roundIndex - offset ≤ roundIndex := by
      omega
    simpa [hEq] using hLe
  · exact Finset.le_max' _ _ (roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel)

theorem mergedActionCount_eq (roundIndex supportExpansionLevel : ℕ) :
    mergedActionCount roundIndex supportExpansionLevel =
      1 + 12 * (coarserShellCount roundIndex supportExpansionLevel + 1) := by
  unfold mergedActionCount
  rw [supportShellLevels_card]

lemma natCast_le_twoPow (n : ℕ) :
    (n : ℝ) ≤ (2 : ℝ) ^ n := by
  induction n with
  | zero => norm_num
  | succ n ih =>
      have hpow_one : (1 : ℝ) ≤ (2 : ℝ) ^ n := by
        have hpow_nat : (1 : ℕ) ≤ (2 : ℕ) ^ n := by
          exact Nat.succ_le_of_lt (Nat.pow_pos (show 0 < (2 : ℕ) by decide))
        exact_mod_cast hpow_nat
      calc
        ((n + 1 : ℕ) : ℝ) = (n : ℝ) + 1 := by norm_num
        _ ≤ (2 : ℝ) ^ n + (2 : ℝ) ^ n := by
          gcongr
        _ = (2 : ℝ) ^ (n + 1) := by
          rw [pow_succ]
          ring

theorem dyadicTranslationStep_antitone
    (baseStep : ℝ)
    (hBase : 0 ≤ baseStep)
    {r₁ r₂ : ℕ}
    (hRound : r₁ ≤ r₂) :
    dyadicTranslationStep baseStep r₂ ≤ dyadicTranslationStep baseStep r₁ := by
  unfold dyadicTranslationStep
  have hpowNat : (2 : ℕ) ^ r₁ ≤ (2 : ℕ) ^ r₂ :=
    Nat.pow_le_pow_right (by norm_num) hRound
  have hpow : (2 : ℝ) ^ r₁ ≤ (2 : ℝ) ^ r₂ := by
    exact_mod_cast hpowNat
  have hpos₁ : 0 < (2 : ℝ) ^ r₁ := by positivity
  have hpos₂ : 0 < (2 : ℝ) ^ r₂ := by positivity
  exact div_le_div_of_nonneg_left hBase hpos₁ hpow

theorem adequateDyadicRound_mono
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    {r₁ r₂ : ℕ}
    (hRound : r₁ ≤ r₂)
    (hAdeq : AdequateDyadicRound baseStep target r₁) :
    AdequateDyadicRound baseStep target r₂ := by
  unfold AdequateDyadicRound at hAdeq ⊢
  exact le_trans (dyadicTranslationStep_antitone baseStep hBase hRound) hAdeq

theorem adequateJointDyadicRound_mono
    (baseStep₁ baseStep₂ target : ℝ)
    (hBase₁ : 0 ≤ baseStep₁)
    (hBase₂ : 0 ≤ baseStep₂)
    {r₁ r₂ : ℕ}
    (hRound : r₁ ≤ r₂)
    (hAdeq : AdequateJointDyadicRound baseStep₁ baseStep₂ target r₁) :
    AdequateJointDyadicRound baseStep₁ baseStep₂ target r₂ := by
  rcases hAdeq with ⟨h₁, h₂⟩
  exact ⟨adequateDyadicRound_mono baseStep₁ target hBase₁ hRound h₁,
    adequateDyadicRound_mono baseStep₂ target hBase₂ hRound h₂⟩

/-- Exact finite geometric-series identity for dyadic translation steps. -/
theorem dyadicTranslationStep_sum_with_tail_eq
    (baseStep : ℝ)
    (roundCount : ℕ) :
    (Finset.range roundCount).sum (fun r => dyadicTranslationStep baseStep r) +
      2 * dyadicTranslationStep baseStep roundCount = 2 * baseStep := by
  induction roundCount with
  | zero =>
      simp [dyadicTranslationStep]
  | succ roundCount ih =>
      rw [Finset.sum_range_succ]
      have htail :
          dyadicTranslationStep baseStep roundCount +
            2 * dyadicTranslationStep baseStep (roundCount + 1) =
            2 * dyadicTranslationStep baseStep roundCount := by
        unfold dyadicTranslationStep
        rw [pow_succ]
        have hpow : (2 : ℝ) ^ roundCount ≠ 0 := by positivity
        field_simp [hpow]
        ring
      calc
        (Finset.range roundCount).sum (fun r => dyadicTranslationStep baseStep r) +
            dyadicTranslationStep baseStep roundCount +
            2 * dyadicTranslationStep baseStep (roundCount + 1)
            = (Finset.range roundCount).sum (fun r => dyadicTranslationStep baseStep r) +
                (dyadicTranslationStep baseStep roundCount +
                  2 * dyadicTranslationStep baseStep (roundCount + 1)) := by ring
        _ = (Finset.range roundCount).sum (fun r => dyadicTranslationStep baseStep r) +
              2 * dyadicTranslationStep baseStep roundCount := by rw [htail]
        _ = 2 * baseStep := ih

/-- The cumulative dyadic translation path is always bounded by twice the root step. -/
theorem dyadicTranslationStep_sum_le_two_mul
    (baseStep : ℝ)
    (hBase : 0 ≤ baseStep)
    (roundCount : ℕ) :
    (Finset.range roundCount).sum (fun r => dyadicTranslationStep baseStep r) ≤ 2 * baseStep := by
  have hEq := dyadicTranslationStep_sum_with_tail_eq baseStep roundCount
  have hTailNonneg : 0 ≤ 2 * dyadicTranslationStep baseStep roundCount := by
    unfold dyadicTranslationStep
    have hpow_nonneg : 0 ≤ (2 : ℝ) ^ roundCount := by positivity
    exact mul_nonneg (by positivity) (div_nonneg hBase hpow_nonneg)
  linarith

/-- If each round chooses one of two dyadic step channels, the cumulative path
    displacement is bounded by twice the larger root step. This matches the
    runtime's rigid local optimizer, which takes either a translation step or a
    rotation-induced displacement step at each round, but not both at once. -/
theorem dyadicMaxStep_sum_le_two_mul_max
    (baseStep₁ baseStep₂ : ℝ)
    (hBase₁ : 0 ≤ baseStep₁)
    (hBase₂ : 0 ≤ baseStep₂)
    (roundCount : ℕ) :
    (Finset.range roundCount).sum
        (fun r => max (dyadicTranslationStep baseStep₁ r) (dyadicTranslationStep baseStep₂ r))
      ≤ 2 * max baseStep₁ baseStep₂ := by
  have hPointwise : ∀ r,
      max (dyadicTranslationStep baseStep₁ r) (dyadicTranslationStep baseStep₂ r)
        ≤ dyadicTranslationStep (max baseStep₁ baseStep₂) r := by
    intro r
    unfold dyadicTranslationStep
    have hpow_pos : 0 < (2 : ℝ) ^ r := by positivity
    refine max_le_iff.mpr ?_
    constructor
    · exact (_root_.div_le_div_of_nonneg_right (le_max_left _ _) (le_of_lt hpow_pos))
    · exact (_root_.div_le_div_of_nonneg_right (le_max_right _ _) (le_of_lt hpow_pos))
  calc
    (Finset.range roundCount).sum
        (fun r => max (dyadicTranslationStep baseStep₁ r) (dyadicTranslationStep baseStep₂ r))
        ≤ (Finset.range roundCount).sum (fun r => dyadicTranslationStep (max baseStep₁ baseStep₂) r) := by
          refine Finset.sum_le_sum ?_
          intro r _
          exact hPointwise r
    _ ≤ 2 * max baseStep₁ baseStep₂ := by
          have hMaxBase : 0 ≤ max baseStep₁ baseStep₂ := by
            exact le_trans hBase₁ (le_max_left _ _)
          exact dyadicTranslationStep_sum_le_two_mul (max baseStep₁ baseStep₂)
            hMaxBase roundCount

theorem exists_adequateDyadicRound
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target) :
    ∃ roundIndex, AdequateDyadicRound baseStep target roundIndex := by
  refine ⟨Nat.ceil (baseStep / target), ?_⟩
  unfold AdequateDyadicRound dyadicTranslationStep
  have hratio_nonneg : 0 ≤ baseStep / target := by
    exact div_nonneg hBase (le_of_lt hTarget)
  have hceil : baseStep / target ≤ (Nat.ceil (baseStep / target) : ℝ) :=
    Nat.le_ceil _
  have hpow : (Nat.ceil (baseStep / target) : ℝ) ≤
      (2 : ℝ) ^ Nat.ceil (baseStep / target) :=
    natCast_le_twoPow _
  have hratio_le_pow : baseStep / target ≤ (2 : ℝ) ^ Nat.ceil (baseStep / target) := by
    exact le_trans hceil hpow
  have hscaled : (baseStep / target) * target ≤
      ((2 : ℝ) ^ Nat.ceil (baseStep / target)) * target := by
    exact mul_le_mul_of_nonneg_right hratio_le_pow (le_of_lt hTarget)
  have hleft : (baseStep / target) * target = baseStep := by
    field_simp [hTarget.ne']
  have hpow_pos : 0 < (2 : ℝ) ^ Nat.ceil (baseStep / target) := by positivity
  rw [hleft] at hscaled
  exact (_root_.div_le_iff₀ hpow_pos).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hscaled)

/-- Canonical least dyadic round whose translation step meets the target. This
    matches the runtime loop that repeatedly halves the step until it is small
    enough, rather than using an implementation-specific closed form. -/
noncomputable def leastAdequateDyadicRound
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target) : ℕ :=
  Nat.find (exists_adequateDyadicRound baseStep target hBase hTarget)

theorem leastAdequateDyadicRound_spec
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target) :
    AdequateDyadicRound baseStep target
      (leastAdequateDyadicRound baseStep target hBase hTarget) := by
  exact Nat.find_spec (exists_adequateDyadicRound baseStep target hBase hTarget)

theorem leastAdequateDyadicRound_minimal
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target)
    {roundIndex : ℕ}
    (hAdeq : AdequateDyadicRound baseStep target roundIndex) :
    leastAdequateDyadicRound baseStep target hBase hTarget ≤ roundIndex := by
  exact Nat.find_min' (exists_adequateDyadicRound baseStep target hBase hTarget) hAdeq

/-- Runtime variant of `leastAdequateDyadicRound` that insists on executing at
    least one refinement round. This matches the local optimizer loop, which
    always performs a nonempty certified action-family pass. -/
noncomputable def leastPositiveAdequateDyadicRound
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target) : ℕ :=
  max 1 (leastAdequateDyadicRound baseStep target hBase hTarget)

theorem leastPositiveAdequateDyadicRound_spec
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target) :
    0 < leastPositiveAdequateDyadicRound baseStep target hBase hTarget ∧
    AdequateDyadicRound baseStep target
      (leastPositiveAdequateDyadicRound baseStep target hBase hTarget) := by
  constructor
  · unfold leastPositiveAdequateDyadicRound
    positivity
  · unfold leastPositiveAdequateDyadicRound
    exact adequateDyadicRound_mono baseStep target hBase (Nat.le_max_right _ _)
      (leastAdequateDyadicRound_spec baseStep target hBase hTarget)

theorem leastPositiveAdequateDyadicRound_minimal
    (baseStep target : ℝ)
    (hBase : 0 ≤ baseStep)
    (hTarget : 0 < target)
    {roundIndex : ℕ}
    (hPos : 0 < roundIndex)
    (hAdeq : AdequateDyadicRound baseStep target roundIndex) :
    leastPositiveAdequateDyadicRound baseStep target hBase hTarget ≤ roundIndex := by
  unfold leastPositiveAdequateDyadicRound
  have hLeast : leastAdequateDyadicRound baseStep target hBase hTarget ≤ roundIndex :=
    leastAdequateDyadicRound_minimal baseStep target hBase hTarget hAdeq
  omega

/-- Canonical least positive round that simultaneously satisfies two dyadic step
    constraints. This is the runtime-facing quantity for rigid refinement where
    both translation and rotation-induced displacement must fall below the same
    RMSD target scale. -/
noncomputable def leastPositiveJointAdequateDyadicRound
    (baseStep₁ baseStep₂ target : ℝ)
    (hBase₁ : 0 ≤ baseStep₁)
    (hBase₂ : 0 ≤ baseStep₂)
    (hTarget : 0 < target) : ℕ :=
  max
    (leastPositiveAdequateDyadicRound baseStep₁ target hBase₁ hTarget)
    (leastPositiveAdequateDyadicRound baseStep₂ target hBase₂ hTarget)

theorem leastPositiveJointAdequateDyadicRound_spec
    (baseStep₁ baseStep₂ target : ℝ)
    (hBase₁ : 0 ≤ baseStep₁)
    (hBase₂ : 0 ≤ baseStep₂)
    (hTarget : 0 < target) :
    0 < leastPositiveJointAdequateDyadicRound baseStep₁ baseStep₂ target hBase₁ hBase₂ hTarget ∧
    AdequateJointDyadicRound baseStep₁ baseStep₂ target
      (leastPositiveJointAdequateDyadicRound baseStep₁ baseStep₂ target hBase₁ hBase₂ hTarget) := by
  constructor
  · unfold leastPositiveJointAdequateDyadicRound
    have hpos₁ := (leastPositiveAdequateDyadicRound_spec baseStep₁ target hBase₁ hTarget).1
    exact lt_of_lt_of_le hpos₁ (Nat.le_max_left _ _)
  · unfold leastPositiveJointAdequateDyadicRound AdequateJointDyadicRound
    constructor
    · exact adequateDyadicRound_mono baseStep₁ target hBase₁ (Nat.le_max_left _ _)
        (leastPositiveAdequateDyadicRound_spec baseStep₁ target hBase₁ hTarget).2
    · exact adequateDyadicRound_mono baseStep₂ target hBase₂ (Nat.le_max_right _ _)
        (leastPositiveAdequateDyadicRound_spec baseStep₂ target hBase₂ hTarget).2

theorem leastPositiveJointAdequateDyadicRound_minimal
    (baseStep₁ baseStep₂ target : ℝ)
    (hBase₁ : 0 ≤ baseStep₁)
    (hBase₂ : 0 ≤ baseStep₂)
    (hTarget : 0 < target)
    {roundIndex : ℕ}
    (hPos : 0 < roundIndex)
    (hAdeq : AdequateJointDyadicRound baseStep₁ baseStep₂ target roundIndex) :
    leastPositiveJointAdequateDyadicRound baseStep₁ baseStep₂ target hBase₁ hBase₂ hTarget ≤ roundIndex := by
  rcases (show AdequateDyadicRound baseStep₁ target roundIndex ∧ AdequateDyadicRound baseStep₂ target roundIndex by
      simpa [AdequateJointDyadicRound] using hAdeq) with ⟨hAdeq₁, hAdeq₂⟩
  have h₁ : leastPositiveAdequateDyadicRound baseStep₁ target hBase₁ hTarget ≤ roundIndex :=
    leastPositiveAdequateDyadicRound_minimal baseStep₁ target hBase₁ hTarget hPos hAdeq₁
  have h₂ : leastPositiveAdequateDyadicRound baseStep₂ target hBase₂ hTarget ≤ roundIndex :=
    leastPositiveAdequateDyadicRound_minimal baseStep₂ target hBase₂ hTarget hPos hAdeq₂
  unfold leastPositiveJointAdequateDyadicRound
  omega

end SupportExpansion
end Tractability
end DecisionQuotient
