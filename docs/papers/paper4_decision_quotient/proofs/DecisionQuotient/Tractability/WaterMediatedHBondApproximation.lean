/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/WaterMediatedHBondApproximation.lean

  Water-mediated hydrogen bonding is modeled as a bounded three-factor
  multiplicative surrogate and reuses the generic finite exact-vs-coarse
  machinery for three-factor families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.GaussianDecayBounds

namespace DecisionQuotient
namespace Tractability
namespace WaterMediatedHBondApproximation

open DirectionalHBondApproximation
open GaussianDecayBounds

universe u v

noncomputable abbrev waterMediatedHBondScore := directionalHBondScore

noncomputable abbrev waterMediatedHBondDecisionProblem {A : Type u} {S : Type v}
    (radial water ligand : A → S → ℝ) : DecisionProblem A S :=
  directionalHBondDecisionProblem radial water ligand

noncomputable abbrev attractiveWaterMediatedHBondDecisionProblem {A : Type u} {S : Type v}
    (radial water ligand : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial water ligand

noncomputable abbrev finiteWaterMediatedHBondErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ) : ℝ :=
  finiteDirectionalHBondErrorRadius
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse

abbrev finiteWaterMediatedHBondErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse

abbrev finiteWaterMediatedHBondErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_nonneg
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse

abbrev exact_vs_coarse_waterMediatedHBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse

noncomputable abbrev exact_vs_coarse_waterMediatedHBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

abbrev exact_vs_coarse_waterMediatedHBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1_sound
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

noncomputable abbrev exact_vs_coarse_waterMediatedHBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

noncomputable abbrev exact_vs_coarse_waterMediatedHBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_optimizer_witness
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

abbrev exact_vs_coarse_attractiveWaterMediatedHBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ) :=
  exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse

noncomputable abbrev exact_vs_coarse_attractiveWaterMediatedHBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

abbrev exact_vs_coarse_attractiveWaterMediatedHBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

noncomputable abbrev exact_vs_coarse_attractiveWaterMediatedHBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

noncomputable abbrev exact_vs_coarse_attractiveWaterMediatedHBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact waterExact ligandExact : A → S → ℝ)
    (radialCoarse waterCoarse ligandCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    radialExact waterExact ligandExact radialCoarse waterCoarse ligandCoarse s

/-- A nonnegative external weight times a water-mediated H-bond score is bounded
    by the same weight times its radial factor when both angular factors lie in
    `[0,1]`. -/
theorem weighted_waterMediatedHBond_le_weighted_radial
    (w radial water ligand : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hRadial_nonneg : 0 ≤ radial)
    (hWater_nonneg : 0 ≤ water)
    (hWater_le_one : water ≤ 1)
    (hLig_nonneg : 0 ≤ ligand)
    (hLig_le_one : ligand ≤ 1) :
    w * waterMediatedHBondScore radial water ligand ≤ w * radial := by
  simpa [waterMediatedHBondScore] using
    weighted_directionalHBond_le_weighted_radial
      w radial water ligand hw_nonneg hRadial_nonneg
      hWater_nonneg hWater_le_one hLig_nonneg hLig_le_one

/-- Tail bound for weighted water-mediated H-bond scores once the radial distance
    is beyond cutoff and both angular factors stay in `[0,1]`. -/
theorem weighted_waterMediatedHBond_tail_bound
    (w ideal width cutoff r water ligand : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hwidth : 0 < width)
    (hcut : ideal ≤ cutoff)
    (hr : cutoff ≤ r)
    (hWater_nonneg : 0 ≤ water)
    (hWater_le_one : water ≤ 1)
    (hLig_nonneg : 0 ≤ ligand)
    (hLig_le_one : ligand ≤ 1) :
    w * waterMediatedHBondScore (Real.exp (-(((r - ideal) / width) ^ 2))) water ligand ≤
      w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
  have hRadial_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ 2)) :=
    (Real.exp_pos _).le
  have hUpper :=
    weighted_waterMediatedHBond_le_weighted_radial
      w (Real.exp (-(((r - ideal) / width) ^ 2))) water ligand
      hw_nonneg hRadial_nonneg hWater_nonneg hWater_le_one hLig_nonneg hLig_le_one
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

end WaterMediatedHBondApproximation
end Tractability
end DecisionQuotient
