/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/WaterMediatedHBondApproximation.lean

  Water-mediated hydrogen bonding is modeled as a bounded three-factor
  multiplicative surrogate and reuses the generic finite exact-vs-coarse
  machinery for three-factor families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation

namespace DecisionQuotient
namespace Tractability
namespace WaterMediatedHBondApproximation

open DirectionalHBondApproximation

abbrev waterMediatedHBondScore := directionalHBondScore

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

end WaterMediatedHBondApproximation
end Tractability
end DecisionQuotient
