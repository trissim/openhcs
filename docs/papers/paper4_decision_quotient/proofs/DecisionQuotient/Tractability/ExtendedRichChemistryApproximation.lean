/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ExtendedRichChemistryApproximation.lean

  Additive composition theorems for the extended rich-chemistry family built
  from theorem-backed nonbonded terms plus attractive chemistry contributions:

  - attractive polar surrogate
  - attractive metal coordination
  - attractive pi-stacking
  - attractive pi-cation
  - attractive halogen bonding
  - attractive water-mediated hydrogen bonding
-/
import DecisionQuotient.Tractability.NonbondedApproximation
import DecisionQuotient.Tractability.RichChemistryApproximation
import DecisionQuotient.Tractability.MetalCoordinationApproximation
import DecisionQuotient.Tractability.PiStackingApproximation
import DecisionQuotient.Tractability.PiCationApproximation
import DecisionQuotient.Tractability.HalogenBondApproximation
import DecisionQuotient.Tractability.WaterMediatedHBondApproximation

namespace DecisionQuotient
namespace Tractability
namespace ExtendedRichChemistryApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open NonbondedApproximation
open RichChemistryApproximation
open MetalCoordinationApproximation
open PiStackingApproximation
open PiCationApproximation
open HalogenBondApproximation
open WaterMediatedHBondApproximation

universe u v

/-- Attractive chemistry bundle without the nonbonded base term. -/
noncomputable def exactAttractiveExtendedChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (sumDecisionProblems
      (sumDecisionProblems
        (sumDecisionProblems
          (sumDecisionProblems
            (exactAttractivePolarSurrogateDecisionProblem
              distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
            (exactAttractiveMetalCoordinationDecisionProblem
              metalW metalIdeal metalWidth metalDistance))
          (attractivePiStackingDecisionProblem
            ppRadialExact ppFaceExact ppOffsetExact))
        (attractivePiCationDecisionProblem
          pcRadialExact pcPlaneExact pcCationExact))
      (attractiveHalogenBondDecisionProblem
        xbRadialExact xbDonorExact xbAcceptorExact))
    (attractiveWaterMediatedHBondDecisionProblem
      wbRadialExact wbWaterExact wbLigandExact)

noncomputable def coarseAttractiveExtendedChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (sumDecisionProblems
      (sumDecisionProblems
        (sumDecisionProblems
          (sumDecisionProblems
            (coarseAttractivePolarSurrogateDecisionProblem
              distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
            (cutoffAttractiveMetalCoordinationDecisionProblem
              metalW metalIdeal metalWidth metalRc metalDistance))
          (attractivePiStackingDecisionProblem
            ppRadialCoarse ppFaceCoarse ppOffsetCoarse))
        (attractivePiCationDecisionProblem
          pcRadialCoarse pcPlaneCoarse pcCationCoarse))
      (attractiveHalogenBondDecisionProblem
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse))
    (attractiveWaterMediatedHBondDecisionProblem
      wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def attractiveExtendedChemistryErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) : ℝ :=
  attractivePolarSurrogateErrorRadius
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
    + metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance
    + finitePiStackingErrorRadius
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
    + finitePiCationErrorRadius
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
    + finiteHalogenBondErrorRadius
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
    + finiteWaterMediatedHBondErrorRadius
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse

theorem exact_vs_coarse_attractiveExtendedChemistry_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (exactAttractiveExtendedChemistryDecisionProblem
        distance w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact)
      (coarseAttractiveExtendedChemistryDecisionProblem
        distance w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (attractiveExtendedChemistryErrorRadius
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact
        wbRadialCoarse wbWaterCoarse wbLigandCoarse) := by
  unfold exactAttractiveExtendedChemistryDecisionProblem
    coarseAttractiveExtendedChemistryDecisionProblem attractiveExtendedChemistryErrorRadius
  have hPolarMetal :
      UniformUtilityApprox
        (sumDecisionProblems
          (exactAttractivePolarSurrogateDecisionProblem
            distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
          (exactAttractiveMetalCoordinationDecisionProblem
            metalW metalIdeal metalWidth metalDistance))
        (sumDecisionProblems
          (coarseAttractivePolarSurrogateDecisionProblem
            distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
          (cutoffAttractiveMetalCoordinationDecisionProblem
            metalW metalIdeal metalWidth metalRc metalDistance))
        (attractivePolarSurrogateErrorRadius
            distance w β rcCT
            polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
            polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
          + metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance) := by
    exact sum_uniformApprox
      (exactAttractivePolarSurrogateDecisionProblem
        distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
      (coarseAttractivePolarSurrogateDecisionProblem
        distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
      (exactAttractiveMetalCoordinationDecisionProblem metalW metalIdeal metalWidth metalDistance)
      (cutoffAttractiveMetalCoordinationDecisionProblem metalW metalIdeal metalWidth metalRc metalDistance)
      (attractivePolarSurrogateErrorRadius
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
      (metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance)
      (exact_vs_coarse_attractivePolarSurrogate_uniformApprox
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
      (exact_vs_cutoff_attractiveMetalCoordination_uniformApprox
        metalW metalIdeal metalWidth metalRc metalDistance)
  have hPolarMetalPi :
      UniformUtilityApprox
        (sumDecisionProblems
          (sumDecisionProblems
            (exactAttractivePolarSurrogateDecisionProblem
              distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
            (exactAttractiveMetalCoordinationDecisionProblem
              metalW metalIdeal metalWidth metalDistance))
          (attractivePiStackingDecisionProblem
            ppRadialExact ppFaceExact ppOffsetExact))
        (sumDecisionProblems
          (sumDecisionProblems
            (coarseAttractivePolarSurrogateDecisionProblem
              distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
            (cutoffAttractiveMetalCoordinationDecisionProblem
              metalW metalIdeal metalWidth metalRc metalDistance))
          (attractivePiStackingDecisionProblem
            ppRadialCoarse ppFaceCoarse ppOffsetCoarse))
        ((attractivePolarSurrogateErrorRadius
            distance w β rcCT
            polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
            polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
          + metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance)
          + finitePiStackingErrorRadius
              ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse) := by
    exact sum_uniformApprox
      _ _
      (attractivePiStackingDecisionProblem ppRadialExact ppFaceExact ppOffsetExact)
      (attractivePiStackingDecisionProblem ppRadialCoarse ppFaceCoarse ppOffsetCoarse)
      _
      (finitePiStackingErrorRadius
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse)
      hPolarMetal
      (exact_vs_coarse_attractivePiStacking_uniformApprox
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse)
  have hPolarMetalPiPC :
      UniformUtilityApprox
        (sumDecisionProblems
          (sumDecisionProblems
            (sumDecisionProblems
              (exactAttractivePolarSurrogateDecisionProblem
                distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
              (exactAttractiveMetalCoordinationDecisionProblem
                metalW metalIdeal metalWidth metalDistance))
            (attractivePiStackingDecisionProblem ppRadialExact ppFaceExact ppOffsetExact))
          (attractivePiCationDecisionProblem pcRadialExact pcPlaneExact pcCationExact))
        (sumDecisionProblems
          (sumDecisionProblems
            (sumDecisionProblems
              (coarseAttractivePolarSurrogateDecisionProblem
                distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
              (cutoffAttractiveMetalCoordinationDecisionProblem
                metalW metalIdeal metalWidth metalRc metalDistance))
            (attractivePiStackingDecisionProblem ppRadialCoarse ppFaceCoarse ppOffsetCoarse))
          (attractivePiCationDecisionProblem pcRadialCoarse pcPlaneCoarse pcCationCoarse))
        (((attractivePolarSurrogateErrorRadius
            distance w β rcCT
            polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
            polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
          + metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance)
          + finitePiStackingErrorRadius
              ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse)
          + finitePiCationErrorRadius
              pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse) := by
    exact sum_uniformApprox
      _ _
      (attractivePiCationDecisionProblem pcRadialExact pcPlaneExact pcCationExact)
      (attractivePiCationDecisionProblem pcRadialCoarse pcPlaneCoarse pcCationCoarse)
      _
      (finitePiCationErrorRadius
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse)
      hPolarMetalPi
      (exact_vs_coarse_attractivePiCation_uniformApprox
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse)
  have hPolarMetalPiPCXB :
      UniformUtilityApprox
        (sumDecisionProblems
          (sumDecisionProblems
            (sumDecisionProblems
              (sumDecisionProblems
                (exactAttractivePolarSurrogateDecisionProblem
                  distance w β polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor)
                (exactAttractiveMetalCoordinationDecisionProblem
                  metalW metalIdeal metalWidth metalDistance))
              (attractivePiStackingDecisionProblem ppRadialExact ppFaceExact ppOffsetExact))
            (attractivePiCationDecisionProblem pcRadialExact pcPlaneExact pcCationExact))
          (attractiveHalogenBondDecisionProblem xbRadialExact xbDonorExact xbAcceptorExact))
        (sumDecisionProblems
          (sumDecisionProblems
            (sumDecisionProblems
              (sumDecisionProblems
                (coarseAttractivePolarSurrogateDecisionProblem
                  distance w β rcCT polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor)
                (cutoffAttractiveMetalCoordinationDecisionProblem
                  metalW metalIdeal metalWidth metalRc metalDistance))
              (attractivePiStackingDecisionProblem ppRadialCoarse ppFaceCoarse ppOffsetCoarse))
            (attractivePiCationDecisionProblem pcRadialCoarse pcPlaneCoarse pcCationCoarse))
          (attractiveHalogenBondDecisionProblem xbRadialCoarse xbDonorCoarse xbAcceptorCoarse))
        ((((attractivePolarSurrogateErrorRadius
            distance w β rcCT
            polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
            polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
          + metalCoordinationCutoffErrorRadius metalW metalIdeal metalWidth metalRc metalDistance)
          + finitePiStackingErrorRadius
              ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse)
          + finitePiCationErrorRadius
              pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse)
          + finiteHalogenBondErrorRadius
              xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse) := by
    exact sum_uniformApprox
      _ _
      (attractiveHalogenBondDecisionProblem xbRadialExact xbDonorExact xbAcceptorExact)
      (attractiveHalogenBondDecisionProblem xbRadialCoarse xbDonorCoarse xbAcceptorCoarse)
      _
      (finiteHalogenBondErrorRadius
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse)
      hPolarMetalPiPC
      (exact_vs_coarse_attractiveHalogenBond_uniformApprox
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse)
  exact sum_uniformApprox
    _ _
    (attractiveWaterMediatedHBondDecisionProblem wbRadialExact wbWaterExact wbLigandExact)
    (attractiveWaterMediatedHBondDecisionProblem wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    _
    (finiteWaterMediatedHBondErrorRadius
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    hPolarMetalPiPCXB
    (exact_vs_coarse_attractiveWaterMediatedHBond_uniformApprox
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

theorem attractiveExtendedChemistryErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) :
    0 ≤ attractiveExtendedChemistryErrorRadius
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse := by
  have hPolar := attractivePolarSurrogateErrorRadius_nonneg
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
  have hMetal := metalCoordinationCutoffErrorRadius_nonneg
      metalW metalIdeal metalWidth metalRc metalDistance
  have hPP := finitePiStackingErrorRadius_nonneg
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
  have hPC := finitePiCationErrorRadius_nonneg
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
  have hXB := finiteHalogenBondErrorRadius_nonneg
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
  have hWB := finiteWaterMediatedHBondErrorRadius_nonneg
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse
  unfold attractiveExtendedChemistryErrorRadius
  nlinarith

noncomputable def exact_vs_coarse_attractiveExtendedChemistry_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactAttractiveExtendedChemistryDecisionProblem
      distance w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
    (fun a => coarseAttractiveExtendedChemistryDecisionProblem
      distance w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
    (attractiveExtendedChemistryErrorRadius
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (fun a => exact_vs_coarse_attractiveExtendedChemistry_uniformApprox
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
    (attractiveExtendedChemistryErrorRadius_nonneg
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

theorem exact_vs_coarse_attractiveExtendedChemistry_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactAttractiveExtendedChemistryDecisionProblem
        distance w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
      (fun a => coarseAttractiveExtendedChemistryDecisionProblem
        distance w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
      (attractiveExtendedChemistryErrorRadius
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (fun a => exact_vs_coarse_attractiveExtendedChemistry_uniformApprox
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
      (attractiveExtendedChemistryErrorRadius_nonneg
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)).exactTopK
      ⊆ (exact_vs_coarse_attractiveExtendedChemistry_certified_top1
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s).survivors := by
  simpa [exact_vs_coarse_attractiveExtendedChemistry_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactAttractiveExtendedChemistryDecisionProblem
        distance w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
      (fun a => coarseAttractiveExtendedChemistryDecisionProblem
        distance w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
      (attractiveExtendedChemistryErrorRadius
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (fun a => exact_vs_coarse_attractiveExtendedChemistry_uniformApprox
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
      (attractiveExtendedChemistryErrorRadius_nonneg
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def exact_vs_coarse_attractiveExtendedChemistry_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactAttractiveExtendedChemistryDecisionProblem
      distance w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
    (fun a => coarseAttractiveExtendedChemistryDecisionProblem
      distance w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
    (attractiveExtendedChemistryErrorRadius
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (fun a => exact_vs_coarse_attractiveExtendedChemistry_uniformApprox
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
    (attractiveExtendedChemistryErrorRadius_nonneg
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def exact_vs_coarse_attractiveExtendedChemistry_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_attractiveExtendedChemistry_coherent_optimizer_witness
    distance w β rcCT
    polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
    polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
    metalW metalIdeal metalWidth metalRc metalDistance
    ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
    pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
    xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
    wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s).toOptimizerWitness

/-- Full extended rich chemistry family: nonbonded base plus attractive extended chemistry. -/
noncomputable def exactExtendedRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ q_i q_j κ : ℝ)
    (w β : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (exactAttractiveExtendedChemistryDecisionProblem
      distance w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact)

noncomputable def coarseExtendedRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (coarseAttractiveExtendedChemistryDecisionProblem
      distance w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def extendedRichChemistryErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) : ℝ :=
  ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC
    + attractiveExtendedChemistryErrorRadius
        distance w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse

theorem exact_vs_coarse_extendedRichChemistry_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (exactExtendedRichChemistryDecisionProblem
        distance ε σ q_i q_j κ
        w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact)
      (coarseExtendedRichChemistryDecisionProblem
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (extendedRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact
        wbRadialCoarse wbWaterCoarse wbLigandCoarse) := by
  unfold exactExtendedRichChemistryDecisionProblem coarseExtendedRichChemistryDecisionProblem extendedRichChemistryErrorRadius
  exact sum_uniformApprox
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (exactAttractiveExtendedChemistryDecisionProblem
      distance w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact)
    (coarseAttractiveExtendedChemistryDecisionProblem
      distance w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (attractiveExtendedChemistryErrorRadius
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC)
    (exact_vs_coarse_attractiveExtendedChemistry_uniformApprox
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

theorem extendedRichChemistryErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ) :
    0 ≤ extendedRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse := by
  unfold extendedRichChemistryErrorRadius
  exact add_nonneg
    (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)
    (attractiveExtendedChemistryErrorRadius_nonneg
      distance w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def exact_vs_coarse_extendedRichChemistry_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactExtendedRichChemistryDecisionProblem
      distance ε σ q_i q_j κ
      w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
    (fun a => coarseExtendedRichChemistryDecisionProblem
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
    (extendedRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (fun a => exact_vs_coarse_extendedRichChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
    (extendedRichChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

theorem exact_vs_coarse_extendedRichChemistry_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactExtendedRichChemistryDecisionProblem
        distance ε σ q_i q_j κ
        w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
      (fun a => coarseExtendedRichChemistryDecisionProblem
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
      (extendedRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (fun a => exact_vs_coarse_extendedRichChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
      (extendedRichChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)).exactTopK
      ⊆ (exact_vs_coarse_extendedRichChemistry_certified_top1
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s).survivors := by
  simpa [exact_vs_coarse_extendedRichChemistry_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactExtendedRichChemistryDecisionProblem
        distance ε σ q_i q_j κ
        w β
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        metalW metalIdeal metalWidth metalDistance
        ppRadialExact ppFaceExact ppOffsetExact
        pcRadialExact pcPlaneExact pcCationExact
        xbRadialExact xbDonorExact xbAcceptorExact
        wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
      (fun a => coarseExtendedRichChemistryDecisionProblem
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
      (extendedRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
      (fun a => exact_vs_coarse_extendedRichChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
      (extendedRichChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC
        w β rcCT
        polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
        polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
        metalW metalIdeal metalWidth metalRc metalDistance
        ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
        pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
        xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
        wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def exact_vs_coarse_extendedRichChemistry_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactExtendedRichChemistryDecisionProblem
      distance ε σ q_i q_j κ
      w β
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      metalW metalIdeal metalWidth metalDistance
      ppRadialExact ppFaceExact ppOffsetExact
      pcRadialExact pcPlaneExact pcCationExact
      xbRadialExact xbDonorExact xbAcceptorExact
      wbRadialExact wbWaterExact wbLigandExact |>.utility a s)
    (fun a => coarseExtendedRichChemistryDecisionProblem
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialCoarse wbWaterCoarse wbLigandCoarse |>.utility a s)
    (extendedRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)
    (fun a => exact_vs_coarse_extendedRichChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse a s)
    (extendedRichChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC
      w β rcCT
      polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
      polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
      metalW metalIdeal metalWidth metalRc metalDistance
      ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
      pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
      xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
      wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse)

noncomputable def exact_vs_coarse_extendedRichChemistry_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_extendedRichChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC
    w β rcCT
    polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
    polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
    metalW metalIdeal metalWidth metalRc metalDistance
    ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
    pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
    xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
    wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s).toOptimizerWitness

end ExtendedRichChemistryApproximation
end Tractability
end DecisionQuotient
