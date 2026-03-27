/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PiCationApproximation.lean

  Pi-cation is modeled as a bounded three-factor multiplicative surrogate and
  reuses the generic finite exact-vs-coarse machinery for three-factor families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.GaussianDecayBounds

namespace DecisionQuotient
namespace Tractability
namespace PiCationApproximation

open DirectionalHBondApproximation
open GaussianDecayBounds

universe u v

noncomputable abbrev piCationScore := directionalHBondScore

/-- Runtime-shaped two-factor pi-cation score: radial times alignment. -/
noncomputable def piCationTwoFactorScore (radial alignment : ℝ) : ℝ :=
  radial * alignment

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

/-- A nonnegative external weight times the runtime-shaped two-factor pi-cation
    score is bounded by the same weight times the radial factor when the
    alignment factor lies in `[0,1]`. -/
theorem weighted_piCationTwoFactor_le_weighted_radial
    (w radial alignment : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hRadial_nonneg : 0 ≤ radial)
    (hAlign_nonneg : 0 ≤ alignment)
    (hAlign_le_one : alignment ≤ 1) :
    w * piCationTwoFactorScore radial alignment ≤ w * radial := by
  unfold piCationTwoFactorScore
  have hInner : radial * alignment ≤ radial := by
    have htmp : radial * alignment ≤ radial * 1 := by
      exact mul_le_mul_of_nonneg_left hAlign_le_one hRadial_nonneg
    simpa using htmp
  exact mul_le_mul_of_nonneg_left hInner hw_nonneg

/-- Tail bound for the runtime-shaped pi-cation score once the radial distance is
    beyond cutoff and alignment stays in `[0,1]`. -/
theorem weighted_piCationTwoFactor_tail_bound
    (w ideal width cutoff r alignment : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hwidth : 0 < width)
    (hcut : ideal ≤ cutoff)
    (hr : cutoff ≤ r)
    (hAlign_nonneg : 0 ≤ alignment)
    (hAlign_le_one : alignment ≤ 1) :
    w * piCationTwoFactorScore (Real.exp (-(((r - ideal) / width) ^ 2))) alignment ≤
      w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
  have hRadial_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ 2)) :=
    (Real.exp_pos _).le
  have hUpper :=
    weighted_piCationTwoFactor_le_weighted_radial
      w (Real.exp (-(((r - ideal) / width) ^ 2))) alignment
      hw_nonneg hRadial_nonneg hAlign_nonneg hAlign_le_one
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
