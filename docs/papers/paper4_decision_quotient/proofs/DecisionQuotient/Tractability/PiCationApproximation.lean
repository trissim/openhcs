/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PiCationApproximation.lean

  Pi-cation is modeled as a bounded three-factor multiplicative surrogate and
  reuses the generic finite exact-vs-coarse machinery for three-factor families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation

namespace DecisionQuotient
namespace Tractability
namespace PiCationApproximation

open DirectionalHBondApproximation

abbrev piCationScore := directionalHBondScore

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

end PiCationApproximation
end Tractability
end DecisionQuotient
