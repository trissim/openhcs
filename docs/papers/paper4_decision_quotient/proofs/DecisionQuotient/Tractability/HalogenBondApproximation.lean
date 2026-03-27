/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/HalogenBondApproximation.lean

  Halogen-bonding is modeled as a bounded three-factor multiplicative surrogate
  and reuses the generic finite exact-vs-coarse machinery for three-factor
  families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.GaussianDecayBounds

namespace DecisionQuotient
namespace Tractability
namespace HalogenBondApproximation

open DirectionalHBondApproximation
open GaussianDecayBounds

universe u v

noncomputable abbrev halogenBondScore := directionalHBondScore

noncomputable abbrev halogenBondDecisionProblem {A : Type u} {S : Type v}
    (radial donor acceptor : A → S → ℝ) : DecisionProblem A S :=
  directionalHBondDecisionProblem radial donor acceptor

noncomputable abbrev attractiveHalogenBondDecisionProblem {A : Type u} {S : Type v}
    (radial donor acceptor : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial donor acceptor

noncomputable abbrev finiteHalogenBondErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : ℝ :=
  finiteDirectionalHBondErrorRadius
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev finiteHalogenBondErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev finiteHalogenBondErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_nonneg
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev exact_vs_coarse_halogenBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

noncomputable abbrev exact_vs_coarse_halogenBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_halogenBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1_sound
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_halogenBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_halogenBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_attractiveHalogenBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_attractiveHalogenBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev attractiveHalogenBond_native_unique_top1_of_certified_background_margin
    {A : Type u}
    (background : A → ℝ)
    (radial donor acceptor : A → ℝ)
    (native : A)
    (Δ : ℝ) :=
  attractiveDirectionalHBond_native_unique_top1_of_certified_background_margin
    background radial donor acceptor native Δ

abbrev attractiveHalogenBond_native_unique_top1_of_equal_background_and_active
    {bg_nat bg_flip r_nat d_nat a_nat r_flip a_flip : ℝ}
    (hbg : bg_nat = bg_flip)
    (hr : 0 < r_nat) (hd : 0 < d_nat) (ha : 0 < a_nat) :=
  attractiveDirectionalHBond_native_unique_top1_of_equal_background_and_active
    (bg_nat := bg_nat)
    (bg_flip := bg_flip)
    (r_nat := r_nat)
    (d_nat := d_nat)
    (a_nat := a_nat)
    (r_flip := r_flip)
    (a_flip := a_flip)
    hbg hr hd ha

/-- A nonnegative external weight times a halogen-bond score is bounded by the
    same weight times its radial factor when both angular factors lie in `[0,1]`. -/
theorem weighted_halogenBond_le_weighted_radial
    (w radial donor acceptor : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hRadial_nonneg : 0 ≤ radial)
    (hDon_nonneg : 0 ≤ donor)
    (hDon_le_one : donor ≤ 1)
    (hAcc_nonneg : 0 ≤ acceptor)
    (hAcc_le_one : acceptor ≤ 1) :
    w * halogenBondScore radial donor acceptor ≤ w * radial := by
  simpa [halogenBondScore] using
    weighted_directionalHBond_le_weighted_radial
      w radial donor acceptor hw_nonneg hRadial_nonneg
      hDon_nonneg hDon_le_one hAcc_nonneg hAcc_le_one

/-- Tail bound for weighted halogen-bond scores once the radial distance is beyond
    cutoff and both angular factors stay in `[0,1]`. -/
theorem weighted_halogenBond_tail_bound
    (w ideal width cutoff r donor acceptor : ℝ)
    (hw_nonneg : 0 ≤ w)
    (hwidth : 0 < width)
    (hcut : ideal ≤ cutoff)
    (hr : cutoff ≤ r)
    (hDon_nonneg : 0 ≤ donor)
    (hDon_le_one : donor ≤ 1)
    (hAcc_nonneg : 0 ≤ acceptor)
    (hAcc_le_one : acceptor ≤ 1) :
    w * halogenBondScore (Real.exp (-(((r - ideal) / width) ^ 2))) donor acceptor ≤
      w * Real.exp (-(((cutoff - ideal) / width) ^ 2)) := by
  have hRadial_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ 2)) :=
    (Real.exp_pos _).le
  have hUpper :=
    weighted_halogenBond_le_weighted_radial
      w (Real.exp (-(((r - ideal) / width) ^ 2))) donor acceptor
      hw_nonneg hRadial_nonneg hDon_nonneg hDon_le_one hAcc_nonneg hAcc_le_one
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

end HalogenBondApproximation
end Tractability
end DecisionQuotient
