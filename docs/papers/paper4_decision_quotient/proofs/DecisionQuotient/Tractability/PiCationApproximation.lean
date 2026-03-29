/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PiCationApproximation.lean

  Pi-cation is modeled as a bounded three-factor multiplicative surrogate and
  reuses the generic finite exact-vs-coarse machinery for three-factor families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.AromaticRingGeometry
import DecisionQuotient.Tractability.GaussianDecayBounds

namespace DecisionQuotient
namespace Tractability
namespace PiCationApproximation

open DirectionalHBondApproximation
open GridConvergence
open GaussianDecayBounds

universe u v

noncomputable abbrev piCationScore := directionalHBondScore

noncomputable abbrev piCationDecisionProblem {A : Type u} {S : Type v}
    (radial plane cation : A → S → ℝ) : DecisionProblem A S :=
  directionalHBondDecisionProblem radial plane cation

noncomputable abbrev attractivePiCationDecisionProblem {A : Type u} {S : Type v}
    (radial plane cation : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial plane cation

noncomputable abbrev finitePiCationErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ) : ℝ :=
  finiteDirectionalHBondErrorRadius
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse

abbrev finitePiCationErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse

abbrev finitePiCationErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_nonneg
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse

abbrev exact_vs_coarse_piCation_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse

noncomputable abbrev exact_vs_coarse_piCation_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

abbrev exact_vs_coarse_piCation_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1_sound
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

noncomputable abbrev exact_vs_coarse_piCation_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

noncomputable abbrev exact_vs_coarse_piCation_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_optimizer_witness
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

abbrev exact_vs_coarse_attractivePiCation_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ) :=
  exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse

noncomputable abbrev exact_vs_coarse_attractivePiCation_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

abbrev exact_vs_coarse_attractivePiCation_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

noncomputable abbrev exact_vs_coarse_attractivePiCation_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

noncomputable abbrev exact_vs_coarse_attractivePiCation_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact planeExact cationExact : A → S → ℝ)
    (radialCoarse planeCoarse cationCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    radialExact planeExact cationExact radialCoarse planeCoarse cationCoarse s

/-
  Strengths absorption theorems.

  The Python runtime `PiCationInteractionTerm._scores` computes:

    pair_scores = -(strengths * radial * alignment * offset_factor)

  where `strengths` depends only on the ring-pair index (state `s`), not the
  pose (action `a`). The four-factor product is therefore equal to the
  canonical three-factor `directionalHBondScore` with strengths absorbed:

    strengths * radial * alignment * offset
    = directionalHBondScore (strengths * radial) alignment offset
-/

/-- The four-factor pi-cation score equals the canonical three-factor
    surrogate with ring-pair strengths absorbed into the radial component. -/
theorem piCationScore_strengths_eq (strengths radial alignment offset : ℝ) :
    strengths * radial * alignment * offset =
      piCationScore (strengths * radial) alignment offset := by
  unfold piCationScore directionalHBondScore
  ring

/-- Ring-pair strengths in [0,1] absorbed into the radial factor yield a valid
    UnitIntervalFactor, enabling the three-factor certificate. -/
theorem scaledPiCationRadial_unitIntervalFactor {A : Type u} {S : Type v}
    (strengths : S → ℝ) (radial : A → S → ℝ)
    (hStr : ∀ s, 0 ≤ strengths s ∧ strengths s ≤ 1)
    (hRad : UnitIntervalFactor radial) :
    UnitIntervalFactor (fun a s => strengths s * radial a s) :=
  scaledByStateWeight_unitIntervalFactor strengths radial hStr hRad

/-- Ring-pair strengths in [0,1] absorbed into the radial factor do not worsen
    the Lipschitz constant Lr, so exact_vs_coarse error bounds are unchanged. -/
theorem scaledPiCationRadial_lipschitz {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (strCont : Scont → ℝ) (radCont : A → Scont → ℝ) (radGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (stateError : Sgrid → ℝ) (Lr : ℝ)
    (hStr : ∀ s, 0 ≤ strCont s ∧ strCont s ≤ 1)
    (hLip : LipschitzUtilityApprox radCont radGrid lift stateError Lr) :
    LipschitzUtilityApprox
      (fun a s => strCont s * radCont a s)
      (fun a sGrid => strCont (lift sGrid) * radGrid a sGrid)
      lift stateError Lr :=
  scaledByStateWeight_lipschitz strCont radCont radGrid lift stateError Lr hStr hLip

/-- A nonnegative external weight times a pi-cation score is bounded by the same
    weight times its radial factor when alignment and offset factors lie in
    `[0,1]`. -/
theorem weighted_piCation_le_weighted_radial
    (w radial alignment offset : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hRadial_nonneg : 0 ≤ radial)
    (hAlign_nonneg : 0 ≤ alignment)
    (hAlign_le_one : alignment ≤ 1)
    (hOffset_nonneg : 0 ≤ offset)
    (hOffset_le_one : offset ≤ 1) :
    w * piCationScore radial alignment offset ≤ w * radial := by
  simpa [piCationScore] using
    weighted_directionalHBond_le_weighted_radial
      w radial alignment offset hw_nonneg hRadial_nonneg
      hAlign_nonneg hAlign_le_one hOffset_nonneg hOffset_le_one

/-- Tail bound for the runtime-shaped pi-cation score once the radial distance is
    beyond cutoff and alignment/offset stay in `[0,1]`. -/
theorem weighted_piCation_tail_bound
    (w ideal width cutoff r alignment offset : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hwidth : 0 < width)
    (hcut : ideal ≤ cutoff)
    (hr : cutoff ≤ r)
    (hAlign_nonneg : 0 ≤ alignment)
    (hAlign_le_one : alignment ≤ 1)
    (hOffset_nonneg : 0 ≤ offset)
    (hOffset_le_one : offset ≤ 1) :
    w * piCationScore (Real.exp (-(((r - ideal) / width) ^ 2))) alignment offset ≤
      w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
  have hRadial_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ 2)) :=
    (Real.exp_pos _).le
  have hUpper :=
    weighted_piCation_le_weighted_radial
      w (Real.exp (-(((r - ideal) / width) ^ 2))) alignment offset
      hw_nonneg hRadial_nonneg hAlign_nonneg hAlign_le_one hOffset_nonneg hOffset_le_one
  have hTail :
      Real.exp (-(((r - ideal) / width) ^ 2)) ≤ Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
    have hBound :=
      GaussianDecayBounds.shiftedGaussian_tail_bound 1 ideal width cutoff r
        hwidth hcut hr
    simpa [shiftedGaussianScore] using hBound
  have hScaledTail :
      w * Real.exp (-(((r - ideal) / width) ^ 2)) ≤ w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
    exact mul_le_mul_of_nonneg_left hTail hw_nonneg
  exact le_trans hUpper hScaledTail

end PiCationApproximation
end Tractability
end DecisionQuotient
