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
open GridConvergence

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

end PiStackingApproximation
end Tractability
end DecisionQuotient
