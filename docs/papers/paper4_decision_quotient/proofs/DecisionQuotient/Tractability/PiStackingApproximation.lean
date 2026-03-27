/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PiStackingApproximation.lean

  Pi-stacking is modeled as a bounded three-factor multiplicative surrogate.
  We intentionally reuse the fully generic finite exact-vs-coarse machinery
  already proved for directional three-factor surrogates.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.GaussianDecayBounds

namespace DecisionQuotient
namespace Tractability
namespace PiStackingApproximation

open DirectionalHBondApproximation
open GridConvergence
open GaussianDecayBounds

universe u v

noncomputable abbrev piStackingScore := directionalHBondScore

noncomputable abbrev piStackingDecisionProblem {A : Type u} {S : Type v}
    (radial face offset : A → S → ℝ) : DecisionProblem A S :=
  directionalHBondDecisionProblem radial face offset

noncomputable abbrev attractivePiStackingDecisionProblem {A : Type u} {S : Type v}
    (radial face offset : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial face offset

noncomputable abbrev coarseAttractivePiStackingDecisionProblem {A : Type u} {S : Type v}
    (radial face offset : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial face offset

noncomputable abbrev finitePiStackingErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ) : ℝ :=
  finiteDirectionalHBondErrorRadius
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse

abbrev finitePiStackingErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse

abbrev finitePiStackingErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_nonneg
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse

abbrev exact_vs_coarse_piStacking_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse

noncomputable abbrev exact_vs_coarse_piStacking_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

abbrev exact_vs_coarse_piStacking_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1_sound
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

noncomputable abbrev exact_vs_coarse_piStacking_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

noncomputable abbrev exact_vs_coarse_piStacking_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_optimizer_witness
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

abbrev exact_vs_coarse_attractivePiStacking_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ) :=
  exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse

noncomputable abbrev exact_vs_coarse_attractivePiStacking_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

abbrev exact_vs_coarse_attractivePiStacking_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

noncomputable abbrev exact_vs_coarse_attractivePiStacking_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

noncomputable abbrev exact_vs_coarse_attractivePiStacking_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact faceExact offsetExact : A → S → ℝ)
    (radialCoarse faceCoarse offsetCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    radialExact faceExact offsetExact radialCoarse faceCoarse offsetCoarse s

/-
  Strengths absorption theorems.

  The Python runtime `PiStackingInteractionTerm._pair_scores` computes:

    strengths = receptor_rings.strengths[None,:,None] * ligand_rings.strengths[None,None,:]
    pair_scores = -(strengths * radial * face_alignment * offset_factor)

  where `strengths` depends only on the ring-pair index (state `s`), not the
  pose (action `a`). The four-factor product is therefore equal to the
  canonical three-factor `directionalHBondScore` with strengths absorbed:

    strengths * radial * face * offset
    = directionalHBondScore (strengths * radial) face offset

  The two theorems below prove:
    (1) the absorbed `scaledRadial = strengths * radial` is still a UnitIntervalFactor
    (2) its Lipschitz constant is no larger than the original radial constant Lr

  Together they show the existing `exact_vs_coarse_attractivePiStacking_*` family
  applies directly once `scaledRadial` is passed as the `radial` argument,
  closing the formal gap flagged by the `CONDITIONALLY_CERTIFIED` annotation on
  `score_certified_pi_stacking_batch` (assumption 1: all factors in [0,1]).
-/

/-- The four-factor pi-stacking score equals the canonical three-factor
    surrogate with ring-pair strengths absorbed into the radial component. -/
theorem piStackingScore_strengths_eq (strengths radial face offset : ℝ) :
    strengths * radial * face * offset =
      piStackingScore (strengths * radial) face offset := by
  unfold piStackingScore directionalHBondScore
  ring

/-- Ring-pair strengths in [0,1] absorbed into the radial factor yield a valid
    UnitIntervalFactor, enabling the three-factor certificate. -/
theorem scaledPiStackingRadial_unitIntervalFactor {A : Type u} {S : Type v}
    (strengths : S → ℝ) (radial : A → S → ℝ)
    (hStr : ∀ s, 0 ≤ strengths s ∧ strengths s ≤ 1)
    (hRad : UnitIntervalFactor radial) :
    UnitIntervalFactor (fun a s => strengths s * radial a s) :=
  scaledByStateWeight_unitIntervalFactor strengths radial hStr hRad

/-- Ring-pair strengths in [0,1] absorbed into the radial factor do not worsen
    the Lipschitz constant Lr, so exact_vs_coarse error bounds are unchanged. -/
theorem scaledPiStackingRadial_lipschitz {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (strCont : Scont → ℝ) (radCont : A → Scont → ℝ) (radGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (stateError : Sgrid → ℝ) (Lr : ℝ)
    (hStr : ∀ s, 0 ≤ strCont s ∧ strCont s ≤ 1)
    (hLip : LipschitzUtilityApprox radCont radGrid lift stateError Lr) :
    LipschitzUtilityApprox
      (fun a s => strCont s * radCont a s)
      (fun a sGrid => strCont (lift sGrid) * radGrid a sGrid)
      lift stateError Lr :=
  scaledByStateWeight_lipschitz strCont radCont radGrid lift stateError Lr hStr hLip

/-- A nonnegative external weight times a pi-stacking score is bounded by the same
    weight times its radial factor when face-alignment and offset factors lie in
    `[0,1]`. -/
theorem weighted_piStacking_le_weighted_radial
    (w radial face offset : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hRadial_nonneg : 0 ≤ radial)
    (hFace_nonneg : 0 ≤ face)
    (hFace_le_one : face ≤ 1)
    (hOffset_nonneg : 0 ≤ offset)
    (hOffset_le_one : offset ≤ 1) :
    w * piStackingScore radial face offset ≤ w * radial := by
  simpa [piStackingScore] using
    weighted_directionalHBond_le_weighted_radial
      w radial face offset hw_nonneg hRadial_nonneg
      hFace_nonneg hFace_le_one hOffset_nonneg hOffset_le_one

/-- Tail bound for weighted pi-stacking once the radial distance is beyond the
    cutoff and the non-radial factors stay in `[0,1]`. -/
theorem weighted_piStacking_tail_bound
    (w ideal width cutoff r face offset : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hwidth : 0 < width)
    (hcut : ideal ≤ cutoff)
    (hr : cutoff ≤ r)
    (hFace_nonneg : 0 ≤ face)
    (hFace_le_one : face ≤ 1)
    (hOffset_nonneg : 0 ≤ offset)
    (hOffset_le_one : offset ≤ 1) :
    w * piStackingScore (Real.exp (-(((r - ideal) / width) ^ 2))) face offset ≤
      w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
  have hRadial_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ 2)) :=
    (Real.exp_pos _).le
  have hUpper :=
    weighted_piStacking_le_weighted_radial
      w (Real.exp (-(((r - ideal) / width) ^ 2))) face offset
      hw_nonneg hRadial_nonneg hFace_nonneg hFace_le_one hOffset_nonneg hOffset_le_one
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

end PiStackingApproximation
end Tractability
end DecisionQuotient
