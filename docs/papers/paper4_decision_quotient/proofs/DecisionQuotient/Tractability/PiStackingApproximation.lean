/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PiStackingApproximation.lean

  Pi-stacking is modeled as a bounded three-factor multiplicative surrogate.
  We intentionally reuse the fully generic finite exact-vs-coarse machinery
  already proved for directional three-factor surrogates.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation

namespace DecisionQuotient
namespace Tractability
namespace PiStackingApproximation

open DirectionalHBondApproximation

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

end PiStackingApproximation
end Tractability
end DecisionQuotient
