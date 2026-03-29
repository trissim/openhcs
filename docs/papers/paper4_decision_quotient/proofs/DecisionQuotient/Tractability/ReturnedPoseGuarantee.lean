/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ReturnedPoseGuarantee.lean

  End-to-end theorem-facing contracts for returned certified docking outputs.

  This file packages the last proof-side compositions needed by the runtime:

  * ambiguity-band outputs can be treated as certified output sets when every
    band member inherits the requested RMSD target, and
  * a unique exact singleton winner over a finite conformer family inherits the
    same target when support coverage plus energy control certify the family.
-/

import DecisionQuotient.Tractability.ConformerSupportCoverage
import DecisionQuotient.Tractability.BlindConformerPipelineRefinements
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Computation.ArrayDSL

namespace DecisionQuotient
namespace Tractability
namespace ReturnedPoseGuarantee

open ConformerSupportCoverage
open BlindConformerPipelineOptimality
open BlindConformerPipelineRefinements
open EnergyRMSDConvergence
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Computation.ArrayDSL

universe u

variable {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]

/-- Any member of the exact top-1 set realizes the `k = 1` utility threshold. -/
theorem ambiguityBand_member_energy_gap_le_of_top1
    (energy : A → ℝ)
    {winner a : A}
    (epsBand : ℝ)
    (hWinnerTop1 : winner ∈ topKSet (fun x => -energy x) 1)
    (haBand : a ∈ ambiguityBand (fun x => -energy x) 1 (by omega) epsBand) :
    energy a - energy winner ≤ epsBand := by
  rw [mem_ambiguityBand_iff] at haBand
  have hkthEq : kthUtility (fun x => -energy x) 1 (by omega) = -energy winner := by
    exact kthUtility_eq_of_mem_top1 (u := fun x => -energy x) hWinnerTop1
  rw [hkthEq] at haBand
  linarith

/-- If an exact top-1 witness already has certified energy gap `gapWinner` to the
    model optimum and `a` lies inside an ambiguity band of width `epsBand` around
    that exact top-1 boundary, then `a` inherits an RMSD target once the combined
    gap budget fits inside the certified quadratic basin. -/
theorem ambiguityBand_member_yields_rmsd_target_of_top1_gap
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    {winner a : A}
    (epsBand gapWinner epsTarget : ℝ)
    (hEnergy : ∀ b, runtimeEnergy b = coordEnergy (coords b))
    (hWinnerTop1 : winner ∈ topKSet (fun x => -runtimeEnergy x) 1)
    (haBand : a ∈ ambiguityBand (fun x => -runtimeEnergy x) 1 (by omega) epsBand)
    (hWinnerGap : coordEnergy (coords winner) - coordEnergy xStar ≤ gapWinner)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hBudget : gapWinner + epsBand ≤ targetEnergyGap basin.μ n epsTarget) :
    rmsd (coords a) xStar ≤ epsTarget := by
  have hBandGap : runtimeEnergy a - runtimeEnergy winner ≤ epsBand := by
    exact ambiguityBand_member_energy_gap_le_of_top1 runtimeEnergy epsBand hWinnerTop1 haBand
  have hCoordGap : coordEnergy (coords a) - coordEnergy xStar ≤ gapWinner + epsBand := by
    have hEa : runtimeEnergy a = coordEnergy (coords a) := hEnergy a
    have hEw : runtimeEnergy winner = coordEnergy (coords winner) := hEnergy winner
    linarith
  apply rmsd_le_of_energyGap_le_target coordEnergy xStar (coords a) basin hn epsTarget hεTarget
  exact le_trans hCoordGap hBudget

/-- Precise ambiguity-set contract: if every member of the exact/coarse ambiguity
    band inherits the requested RMSD target via the winner-gap budget, then the
    whole ambiguity band is a certified output set for that target. -/
theorem ambiguityBand_support_yields_certified_output_set
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (winner : A)
    (epsBand gapWinner epsTarget : ℝ)
    (hEnergy : ∀ b, runtimeEnergy b = coordEnergy (coords b))
    (hWinnerTop1 : winner ∈ topKSet (fun x => -runtimeEnergy x) 1)
    (hWinnerGap : coordEnergy (coords winner) - coordEnergy xStar ≤ gapWinner)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hBudget : gapWinner + epsBand ≤ targetEnergyGap basin.μ n epsTarget) :
    CertifiedOutputSet
      (ambiguityBand (fun x => -runtimeEnergy x) 1 (by omega) epsBand)
      (fun a => rmsd (coords a) xStar ≤ epsTarget) := by
  intro a ha
  exact ambiguityBand_member_yields_rmsd_target_of_top1_gap
    coords runtimeEnergy coordEnergy xStar epsBand gapWinner epsTarget
    hEnergy hWinnerTop1 ha hWinnerGap basin hn hεTarget hBudget

/-- End-to-end ambiguity-set theorem on a covered finite family: if the exact
    rescored winner of the family is used as the ambiguity-band anchor, the family
    RMSD cover induces winner energy gap `L * εCover`, and the ambiguity width
    `epsBand` still fits inside the same quadratic-basin target-energy threshold,
    then every member of the ambiguity band satisfies the requested RMSD target. -/
theorem ambiguityBand_support_of_cover_yields_certified_output_set
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (winner : A)
    (L εCover epsBand epsTarget : ℝ)
    (hEnergy : ∀ b, runtimeEnergy b = coordEnergy (coords b))
    (hWinnerTop1 : winner ∈ topKSet (fun x => -runtimeEnergy x) 1)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (hWinnerBest : ∀ a, runtimeEnergy winner ≤ runtimeEnergy a)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hBudget : L * εCover + epsBand ≤ targetEnergyGap basin.μ n epsTarget) :
    CertifiedOutputSet
      (ambiguityBand (fun x => -runtimeEnergy x) 1 (by omega) epsBand)
      (fun a => rmsd (coords a) xStar ≤ epsTarget) := by
  have hWinnerMem : coords winner ∈ ((Finset.univ : Finset A).image coords) := by
    exact Finset.mem_image.mpr ⟨winner, Finset.mem_univ _, rfl⟩
  have hWinnerBestCoord :
      ∀ z, z ∈ ((Finset.univ : Finset A).image coords) → coordEnergy (coords winner) ≤ coordEnergy z := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨a, _, rfl⟩
    have hBest := hWinnerBest a
    have hEw := hEnergy winner
    have hEa := hEnergy a
    linarith
  have hWinnerGap : coordEnergy (coords winner) - coordEnergy xStar ≤ L * εCover := by
    exact exact_library_winner_energy_gap_of_cover_le
      coordEnergy ((Finset.univ : Finset A).image coords)
      L εCover hL hCover hεCover hOpt hLip hWinnerMem hWinnerBestCoord
  exact ambiguityBand_support_yields_certified_output_set
    coords runtimeEnergy coordEnergy xStar winner epsBand (L * εCover) epsTarget
    hEnergy hWinnerTop1 hWinnerGap basin hn hεTarget hBudget

/-- Weaker ambiguity-set contract for non-convex settings: if the exact rescored
    winner of a covered finite family is used as the ambiguity-band anchor, then
    every ambiguity-band member has exact energy at most `L * εCover + epsBand`
    above the optimal conformer. This requires no quadratic basin. -/
theorem ambiguityBand_support_of_cover_yields_certified_energy_output_set
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (winner : A)
    (L εCover epsBand : ℝ)
    (hEnergy : ∀ b, runtimeEnergy b = coordEnergy (coords b))
    (hWinnerTop1 : winner ∈ topKSet (fun x => -runtimeEnergy x) 1)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (hWinnerBest : ∀ a, runtimeEnergy winner ≤ runtimeEnergy a) :
    CertifiedOutputSet
      (ambiguityBand (fun x => -runtimeEnergy x) 1 (by omega) epsBand)
      (fun a => coordEnergy (coords a) - coordEnergy xStar ≤ L * εCover + epsBand) := by
  intro a ha
  have hWinnerMem : coords winner ∈ ((Finset.univ : Finset A).image coords) := by
    exact Finset.mem_image.mpr ⟨winner, Finset.mem_univ _, rfl⟩
  have hWinnerBestCoord :
      ∀ z, z ∈ ((Finset.univ : Finset A).image coords) → coordEnergy (coords winner) ≤ coordEnergy z := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨b, _, rfl⟩
    have hBest := hWinnerBest b
    have hEw := hEnergy winner
    have hEb := hEnergy b
    linarith
  have hWinnerGap : coordEnergy (coords winner) - coordEnergy xStar ≤ L * εCover := by
    exact exact_library_winner_energy_gap_of_cover_le
      coordEnergy ((Finset.univ : Finset A).image coords)
      L εCover hL hCover hεCover hOpt hLip hWinnerMem hWinnerBestCoord
  have hBandGap : runtimeEnergy a - runtimeEnergy winner ≤ epsBand := by
    exact ambiguityBand_member_energy_gap_le_of_top1 runtimeEnergy epsBand hWinnerTop1 ha
  have hEa : runtimeEnergy a = coordEnergy (coords a) := hEnergy a
  have hEw : runtimeEnergy winner = coordEnergy (coords winner) := hEnergy winner
  linarith

/-- Once a support is certified as an output set for some property, any concrete
    member chosen from that support also satisfies the property. -/
theorem returned_choice_of_member_of_certified_output_set
    (support : Finset A)
    (prop : A → Prop)
    (winner : A)
    (hCertified : CertifiedOutputSet support prop)
    (hWinnerMem : winner ∈ support) :
    prop winner := by
  exact hCertified winner hWinnerMem

/-- If a support set is already certified to satisfy a uniform exact-energy-gap
    budget above `xStar`, then any chosen member whose shared budget fits inside a
    certified quadratic basin inherits the requested RMSD target. -/
theorem returned_choice_of_member_of_certified_energy_output_set_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (support : Finset A)
    (winner : A)
    (gapBudget epsTarget : ℝ)
    (hCertified : CertifiedOutputSet support
      (fun a => coordEnergy (coords a) - coordEnergy xStar ≤ gapBudget))
    (hWinnerMem : winner ∈ support)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hBudget : gapBudget ≤ targetEnergyGap basin.μ n epsTarget) :
    rmsd (coords winner) xStar ≤ epsTarget := by
  have hGap : coordEnergy (coords winner) - coordEnergy xStar ≤ gapBudget :=
    hCertified winner hWinnerMem
  apply rmsd_le_of_energyGap_le_target coordEnergy xStar (coords winner) basin hn epsTarget hεTarget
  exact le_trans hGap hBudget

/-- A selected member whose exact runtime energy sits within `gapMember` of the
    exact top-1 winner inherits the RMSD target once the winner's certified cover
    gap plus that exact member gap fits inside the member's certified basin. -/
theorem member_with_energy_gap_to_top1_yields_rmsd_target_of_top1_gap
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    {winner a : A}
    (gapMember gapWinner epsTarget : ℝ)
    (hEnergy : ∀ b, runtimeEnergy b = coordEnergy (coords b))
    (hWinnerTop1 : winner ∈ topKSet (fun x => -runtimeEnergy x) 1)
    (hMemberGap : runtimeEnergy a - runtimeEnergy winner ≤ gapMember)
    (hWinnerGap : coordEnergy (coords winner) - coordEnergy xStar ≤ gapWinner)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hBudget : gapWinner + gapMember ≤ targetEnergyGap basin.μ n epsTarget) :
    rmsd (coords a) xStar ≤ epsTarget := by
  have hCoordGap : coordEnergy (coords a) - coordEnergy xStar ≤ gapWinner + gapMember := by
    have hEa : runtimeEnergy a = coordEnergy (coords a) := hEnergy a
    have hEw : runtimeEnergy winner = coordEnergy (coords winner) := hEnergy winner
    linarith
  apply rmsd_le_of_energyGap_le_target coordEnergy xStar (coords a) basin hn epsTarget hεTarget
  exact le_trans hCoordGap hBudget

/-- Any strict singleton winner of an auxiliary patched-support score family may be
    returned safely once that support set is already certified as an output set.

    This theorem separates certification of the *set* from deterministic choice of
    a single representative inside the set: the auxiliary score only needs to stay
    inside the certified support and be singleton there after patching. -/
theorem returned_choice_of_auxiliary_patched_support_singleton_of_certified_output_set
    (support : Finset A)
    (prop : A → Prop)
    (auxEnergy : A → ℝ)
    (winner : A)
    (fallback : ℝ)
    (hCertified : CertifiedOutputSet support prop)
    (hWinnerMem : winner ∈ support)
    (hWinnerStrict : ∀ z, z ∈ support → z ≠ winner → auxEnergy winner < auxEnergy z)
    (hFallback : auxEnergy winner < fallback) :
    let runtimeEnergy := patchedSupportEnergy support auxEnergy fallback
    let hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A) := by
      have hBase :=
        patchedSupportEnergy_singleton_of_strict_support_argmin_without_top1_subset
          auxEnergy support winner fallback hWinnerMem hWinnerStrict hFallback
      ext p
      rw [Finset.mem_singleton]
      have hpBase := Finset.ext_iff.mp hBase p
      simp only [energyTopK, Finset.mem_filter, Finset.mem_univ, true_and,
        Finset.mem_singleton] at hpBase
      have hEqCounts :
          strictLowerCount runtimeEnergy p = strictBetterCount (fun q => -runtimeEnergy q) p := by
        unfold strictLowerCount strictBetterCount
        congr 1
        ext q
        simp
      have hpTop : p ∈ topKSet (fun a => -runtimeEnergy a) 1 ↔
          strictBetterCount (fun q => -runtimeEnergy q) p < 1 :=
        mem_topKSet_iff (fun a => -runtimeEnergy a) 1 p
      rw [hpTop, <- hEqCounts]
      exact hpBase
    prop ((coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -runtimeEnergy a) winner hSingleton).belief.selection.choice) := by
  let runtimeEnergy := patchedSupportEnergy support auxEnergy fallback
  let hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A) := by
    have hBase :=
      patchedSupportEnergy_singleton_of_strict_support_argmin_without_top1_subset
        auxEnergy support winner fallback hWinnerMem hWinnerStrict hFallback
    ext p
    rw [Finset.mem_singleton]
    have hpBase := Finset.ext_iff.mp hBase p
    simp only [energyTopK, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_singleton] at hpBase
    have hEqCounts :
        strictLowerCount runtimeEnergy p = strictBetterCount (fun q => -runtimeEnergy q) p := by
      unfold strictLowerCount strictBetterCount
      congr 1
      ext q
      simp
    have hpTop : p ∈ topKSet (fun a => -runtimeEnergy a) 1 ↔
        strictBetterCount (fun q => -runtimeEnergy q) p < 1 :=
      mem_topKSet_iff (fun a => -runtimeEnergy a) 1 p
    rw [hpTop, <- hEqCounts]
    exact hpBase
  have hChoiceEq :
      (coherentOptimizerWitness_of_exact_singleton_top1
        (fun a => -runtimeEnergy a) winner hSingleton).belief.selection.choice = winner := by
    exact (coherentOptimizerWitness_of_exact_singleton_top1_choice
      (fun a => -runtimeEnergy a) winner hSingleton).1
  have hWinnerProp : prop winner := hCertified winner hWinnerMem
  simpa [hChoiceEq] using hWinnerProp

/-- Coverage is necessary for the singleton returned-pose contract: if every
    runtime action lies outside the requested RMSD target ball, then the
    singleton winner returned by the optimizer also lies outside that ball. This
    theorem makes the missing support-coverage hypothesis explicit. -/
theorem returned_choice_of_exact_singleton_winner_misses_rmsd_target_without_cover
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (xStar : CoordSet n)
    (winner : A)
    (epsTarget : ℝ)
    (hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A))
    (hNoCover : ∀ a, epsTarget < rmsd (coords a) xStar) :
    let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -runtimeEnergy a) winner hSingleton
    epsTarget < rmsd (coords w.belief.selection.choice) xStar := by
  let w := coherentOptimizerWitness_of_exact_singleton_top1
    (fun a => -runtimeEnergy a) winner hSingleton
  have hChoiceEq : w.belief.selection.choice = winner := by
    exact (coherentOptimizerWitness_of_exact_singleton_top1_choice
      (fun a => -runtimeEnergy a) winner hSingleton).1
  simpa [w, hChoiceEq] using hNoCover winner

/-- End-to-end returned-pose theorem for the unique-winner path: if the finite
    runtime action family covers conformer space densely enough, the coordinate
    energy is RMSD-Lipschitz on that space, the exact rescored runtime winner is
    the best element of the finite family, and the runtime has certified singleton
    support `{winner}`, then the returned selected action satisfies the requested
    RMSD target. -/
theorem returned_choice_of_exact_singleton_winner_of_cover_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (runtimeEnergy : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (winner : A)
    (L εCover epsTarget : ℝ)
    (hEnergy : ∀ a, runtimeEnergy a = coordEnergy (coords a))
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (hWinnerBest : ∀ a, runtimeEnergy winner ≤ runtimeEnergy a)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n epsTarget)
    (hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A)) :
    let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -runtimeEnergy a) winner hSingleton
    rmsd (coords w.belief.selection.choice) xStar ≤ epsTarget := by
  have hWinnerMem : coords winner ∈ ((Finset.univ : Finset A).image coords) := by
    exact Finset.mem_image.mpr ⟨winner, Finset.mem_univ _, rfl⟩
  have hWinnerBestCoord :
      ∀ z, z ∈ ((Finset.univ : Finset A).image coords) → coordEnergy (coords winner) ≤ coordEnergy z := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨a, _, rfl⟩
    have hBest := hWinnerBest a
    have hEw := hEnergy winner
    have hEa := hEnergy a
    linarith
  have hWinnerRMSD : rmsd (coords winner) xStar ≤ epsTarget := by
    exact exact_library_winner_of_cover_yields_rmsd_target
      coordEnergy ((Finset.univ : Finset A).image coords)
      L εCover epsTarget hL hCover hεCover hOpt hLip hWinnerMem hWinnerBestCoord
      basin hn hεTarget hGapBudget
  let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -runtimeEnergy a) winner hSingleton
  have hChoiceEq : w.belief.selection.choice = winner := by
    exact (coherentOptimizerWitness_of_exact_singleton_top1_choice
      (fun a => -runtimeEnergy a) winner hSingleton).1
  simpa [w, hChoiceEq] using hWinnerRMSD

/-- Winner-only fast-path contract for patched support-only exact rescoring.

    If the runtime patches every pose outside a certified omitted-attractive support
    set to the base witness threshold, the exact argmin on that support is strict,
    and the support was built from theorem-backed omitted-channel lower bounds, then
    the coherent optimizer witness over the patched runtime family returns a pose
    satisfying the same RMSD target as the exact library winner. -/
theorem returned_choice_of_patched_omitted_support_singleton_winner_of_cover_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (coordEnergy : CoordSet n → ℝ)
    (baseScore exactEnergy omittedBound : A → ℝ)
    (xStar : CoordSet n)
    (winner witness : A)
    (fallback L εCover epsTarget : ℝ)
    (hEnergy : ∀ a, exactEnergy a = coordEnergy (coords a))
    (hLower : ∀ a, baseScore a - omittedBound a ≤ exactEnergy a)
    (hWitness : exactEnergy witness ≤ baseScore witness)
    (hWinnerMem : winner ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore witness))
    (hWinnerStrict : ∀ z,
      z ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore witness) →
      z ≠ winner → exactEnergy winner < exactEnergy z)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : @IsOptimal (CoordSet n) (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n epsTarget)
    (hFallback : exactEnergy winner < fallback) :
    let support := canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore witness)
    let runtimeEnergy := patchedSupportEnergy support exactEnergy fallback
    let hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A) := by
      have hBase := patchedSupportEnergy_singleton_of_strict_support_argmin
        exactEnergy support winner fallback
        hWinnerMem hWinnerStrict
        (top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_baseWitness
          baseScore exactEnergy omittedBound witness hLower hWitness)
        hFallback
      ext p
      rw [Finset.mem_singleton]
      have hpBase := Finset.ext_iff.mp hBase p
      simp only [energyTopK, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton] at hpBase
      have hEqCounts : strictLowerCount runtimeEnergy p = strictBetterCount (fun q => -runtimeEnergy q) p := by
        unfold strictLowerCount strictBetterCount
        congr 1
        ext q
        simp
      have hpTop : p ∈ topKSet (fun a => -runtimeEnergy a) 1 ↔ strictBetterCount (fun q => -runtimeEnergy q) p < 1 :=
        mem_topKSet_iff (fun a => -runtimeEnergy a) 1 p
      rw [hpTop, <-hEqCounts]
      exact hpBase
    rmsd ((coherentOptimizerWitness_of_exact_singleton_top1 (fun a => -runtimeEnergy a) winner hSingleton).belief.selection.choice |> coords) xStar ≤ epsTarget := by
  let support := canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore witness)
  let runtimeEnergy := patchedSupportEnergy support exactEnergy fallback
  have hWinnerBestExact : IsOptimal (fun a => -exactEnergy a) winner :=
    canonicalRetain_argmin_is_global_of_omittedAttractiveLowerBound_and_baseWitness_subset
      baseScore exactEnergy omittedBound hWinnerMem
      (fun z hz => by
        by_cases hEq : z = winner
        · simpa [hEq]
        · exact le_of_lt (hWinnerStrict z hz hEq))
      hLower hWitness
  have hWinnerMemImage : coords winner ∈ ((Finset.univ : Finset A).image coords) := by
    exact Finset.mem_image.mpr ⟨winner, Finset.mem_univ _, rfl⟩
  have hWinnerBestCoord :
      ∀ z, z ∈ ((Finset.univ : Finset A).image coords) → coordEnergy (coords winner) ≤ coordEnergy z := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨a, _, rfl⟩
    have hBest := hWinnerBestExact a
    have hEw := hEnergy winner
    have hEa := hEnergy a
    linarith
  have hWinnerRMSD : rmsd (coords winner) xStar ≤ epsTarget := by
    exact exact_library_winner_of_cover_yields_rmsd_target
      coordEnergy ((Finset.univ : Finset A).image coords)
      L εCover epsTarget hL hCover hεCover hOpt hLip hWinnerMemImage hWinnerBestCoord
      basin hn hεTarget hGapBudget
  have hSingleton : topKSet (fun a => -runtimeEnergy a) 1 = ({winner} : Finset A) := by
    have hBase := patchedSupportEnergy_singleton_of_strict_support_argmin
      exactEnergy support winner fallback
      hWinnerMem hWinnerStrict
      (top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_baseWitness
        baseScore exactEnergy omittedBound witness hLower hWitness)
      hFallback
    ext p
    rw [Finset.mem_singleton]
    have hpBase := Finset.ext_iff.mp hBase p
    simp only [energyTopK, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton] at hpBase
    have hEqCounts : strictLowerCount runtimeEnergy p = strictBetterCount (fun q => -runtimeEnergy q) p := by
      unfold strictLowerCount strictBetterCount
      congr 1
      ext q
      simp
    have hpTop : p ∈ topKSet (fun a => -runtimeEnergy a) 1 ↔ strictBetterCount (fun q => -runtimeEnergy q) p < 1 :=
      mem_topKSet_iff (fun a => -runtimeEnergy a) 1 p
    rw [hpTop, <-hEqCounts]
    exact hpBase
  let w := coherentOptimizerWitness_of_exact_singleton_top1 (fun a => -runtimeEnergy a) winner hSingleton
  have hChoiceEq : w.belief.selection.choice = winner := by
    exact (coherentOptimizerWitness_of_exact_singleton_top1_choice
      (fun a => -runtimeEnergy a) winner hSingleton).1
  simpa [w, hChoiceEq] using hWinnerRMSD

/-- Runtime-facing singleton path theorem: if an exact/coarse energy gap margin
    certifies singleton top-1 identity `aStar`, and the finite runtime family also
    satisfies the conformer coverage assumptions, then the returned optimizer
    choice satisfies the requested RMSD target. This theorem is the direct bridge
    the runtime needs once it has both a final singleton-winner witness and a
    conformer-family coverage certificate. -/
theorem exact_energy_gap_certified_choice_of_cover_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (eExact eCoarse : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (aStar : A)
    (δ L εCover epsTarget : ℝ)
    (hEnergy : ∀ a, eExact a = coordEnergy (coords a))
    (hApprox : ∀ x, |eExact x - eCoarse x| ≤ δ)
    (hStrict : ∀ b, b ≠ aStar → eCoarse aStar + 2 * δ < eCoarse b)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n epsTarget) :
    let hSingleton := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      eExact eCoarse aStar δ hApprox hStrict
    let w := coherentOptimizerWitness_of_exact_singleton_top1 (fun a => -eExact a) aStar hSingleton
    rmsd (coords w.belief.selection.choice) xStar ≤ epsTarget := by
  let hSingleton := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
    eExact eCoarse aStar δ hApprox hStrict
  have hBest : ∀ a, eExact aStar ≤ eExact a := by
    intro a
    have hTop1 : aStar ∈ topKSet (fun x => -eExact x) 1 := by
      rw [hSingleton]
      simp
    have haTop1Count : strictBetterCount (fun x => -eExact x) aStar < 1 := by
      exact (mem_topKSet_iff (fun x => -eExact x) 1 aStar).mp hTop1
    have hNotStrict : ¬ eExact a < eExact aStar := by
      intro hLt
      have hmem : a ∈ (Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x) := by
        simp [hLt]
      unfold strictBetterCount at haTop1Count
      have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card :=
        Finset.card_pos.mpr ⟨a, hmem⟩
      have hCardLtOne : ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card < 1 := by
        simpa using haTop1Count
      have hCardZero : ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card = 0 := by
        omega
      rw [hCardZero] at hCardPos
      omega
    exact le_of_not_gt hNotStrict
  exact returned_choice_of_exact_singleton_winner_of_cover_yields_rmsd_target
    coords eExact coordEnergy xStar aStar L εCover epsTarget
    hEnergy hL hCover hεCover hOpt hLip hBest basin hn hεTarget hGapBudget hSingleton

/-- Rigid-docking specialization of the singleton returned-pose contract.

    Interpret a finite sampled pose family `F` as a deterministic SE(3) support,
    map each supported pose to coordinates via `coords`, and assume those
    coordinates epsilon-cover the relevant rigid-pose library in RMSD. If the
    restricted exact/coarse scorer certifies a singleton top-1 supported pose,
    then the returned rigid pose satisfies the requested RMSD target.

    This is the runtime-facing bridge needed to justify a finite SE(3) epsilon-net
    of sampled rigid placements. -/
theorem sampledActionFamily_exact_energy_gap_certified_rigid_choice_of_cover_yields_rmsd_target
    {A : Type u} {S : Type v} [DecidableEq A] [LinearOrder A]
    (exactDP coarseDP : DecisionProblem A S)
    (F : SampledDocking.SampledActionFamily A)
    (s : S)
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : SampledDocking.SupportedAction F → CoordSet n)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (aStar : SampledDocking.SupportedAction F)
    (δ L εCover epsTarget : ℝ)
    (hEnergy : ∀ a : SampledDocking.SupportedAction F,
      (SampledDocking.restrictedDecisionProblem exactDP F).utility a s = coordEnergy (coords a))
    (hApprox : ∀ x : SampledDocking.SupportedAction F,
      |(SampledDocking.restrictedDecisionProblem exactDP F).utility x s -
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility x s| ≤ δ)
    (hStrict : ∀ b : SampledDocking.SupportedAction F, b ≠ aStar →
      (SampledDocking.restrictedDecisionProblem coarseDP F).utility aStar s + 2 * δ <
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility b s)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers
      ((Finset.univ : Finset (SampledDocking.SupportedAction F)).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L)
    (basin : CertifiedQuadraticBasin coordEnergy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ epsTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n epsTarget) :
    let hSingleton := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      (fun a : SampledDocking.SupportedAction F =>
        (SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
      (fun a : SampledDocking.SupportedAction F =>
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility a s)
      aStar δ hApprox hStrict
    let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a : SampledDocking.SupportedAction F =>
        -(SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
      aStar hSingleton
    rmsd (coords w.belief.selection.choice) xStar ≤ epsTarget := by
  exact exact_energy_gap_certified_choice_of_cover_yields_rmsd_target
    (coords := coords)
    (eExact := fun a : SampledDocking.SupportedAction F =>
      (SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
    (eCoarse := fun a : SampledDocking.SupportedAction F =>
      (SampledDocking.restrictedDecisionProblem coarseDP F).utility a s)
    (coordEnergy := coordEnergy)
    (xStar := xStar)
    (aStar := aStar)
    (δ := δ)
    (L := L)
    (εCover := εCover)
    (epsTarget := epsTarget)
    hEnergy hApprox hStrict hL hCover hεCover hOpt hLip basin hn hεTarget hGapBudget

/-- Runtime-facing singleton energy certificate for non-convex settings: if an
    exact/coarse energy gap margin certifies singleton top-1 identity `aStar`, and
    the finite runtime family satisfies the conformer coverage assumptions, then
    the returned optimizer choice has exact energy at most `L * εCover` above the
    optimal conformer. This requires no quadratic basin. -/
theorem exact_energy_gap_certified_choice_of_cover_has_energy_gap_le
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : A → CoordSet n)
    (eExact eCoarse : A → ℝ)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (aStar : A)
    (δ L εCover : ℝ)
    (hEnergy : ∀ a, eExact a = coordEnergy (coords a))
    (hApprox : ∀ x, |eExact x - eCoarse x| ≤ δ)
    (hStrict : ∀ b, b ≠ aStar → eCoarse aStar + 2 * δ < eCoarse b)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers ((Finset.univ : Finset A).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L) :
    let hSingleton := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      eExact eCoarse aStar δ hApprox hStrict
    let w := coherentOptimizerWitness_of_exact_singleton_top1 (fun a => -eExact a) aStar hSingleton
    coordEnergy (coords w.belief.selection.choice) - coordEnergy xStar ≤ L * εCover := by
  have hBest : ∀ a, eExact aStar ≤ eExact a := by
    intro a
    have hTop1 : aStar ∈ topKSet (fun x => -eExact x) 1 := by
      let hSingleton :=
        RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
          eExact eCoarse aStar δ hApprox hStrict
      rw [hSingleton]
      simp
    have haTop1Count : strictBetterCount (fun x => -eExact x) aStar < 1 := by
      exact (mem_topKSet_iff (fun x => -eExact x) 1 aStar).mp hTop1
    have hNotStrict : ¬ eExact a < eExact aStar := by
      intro hLt
      have hmem : a ∈ (Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x) := by
        simp [hLt]
      unfold strictBetterCount at haTop1Count
      have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card :=
        Finset.card_pos.mpr ⟨a, hmem⟩
      have hCardLtOne : ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card < 1 := by
        simpa using haTop1Count
      have hCardZero : ((Finset.univ : Finset A).filter (fun x => -eExact aStar < -eExact x)).card = 0 := by
        omega
      rw [hCardZero] at hCardPos
      omega
    exact le_of_not_gt hNotStrict
  have hWinnerMem : coords aStar ∈ ((Finset.univ : Finset A).image coords) := by
    exact Finset.mem_image.mpr ⟨aStar, Finset.mem_univ _, rfl⟩
  have hBestCoord :
      ∀ z, z ∈ ((Finset.univ : Finset A).image coords) → coordEnergy (coords aStar) ≤ coordEnergy z := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨a, _, rfl⟩
    have hBest := hBest a
    have hEaStar := hEnergy aStar
    have hEa := hEnergy a
    linarith
  have hGap : coordEnergy (coords aStar) - coordEnergy xStar ≤ L * εCover := by
    exact exact_library_winner_energy_gap_of_cover_le
      coordEnergy ((Finset.univ : Finset A).image coords)
      L εCover hL hCover hεCover hOpt hLip hWinnerMem hBestCoord
  exact exact_energy_gap_certified_choice_inherits_property
    (P := fun a => coordEnergy (coords a) - coordEnergy xStar ≤ L * εCover)
    eExact eCoarse aStar δ hApprox hStrict hGap

/-- Rigid-docking specialization of the non-convex singleton energy-gap
    contract. If a sampled rigid support RMSD-covers the ambient rigid library and
    the runtime certifies singleton top-1 identity via an exact/coarse gap margin,
    then the returned rigid choice has exact energy at most `L * εCover` above the
    optimal covered rigid pose. This avoids any quadratic-basin hypothesis. -/
theorem sampledActionFamily_exact_energy_gap_certified_rigid_choice_of_cover_has_energy_gap_le
    {A : Type u} {S : Type v} [DecidableEq A] [LinearOrder A]
    (exactDP coarseDP : DecisionProblem A S)
    (F : SampledDocking.SampledActionFamily A)
    (s : S)
    {n : ℕ} [DecidableEq (CoordSet n)]
    (coords : SampledDocking.SupportedAction F → CoordSet n)
    (coordEnergy : CoordSet n → ℝ)
    (xStar : CoordSet n)
    (aStar : SampledDocking.SupportedAction F)
    (δ L εCover : ℝ)
    (hEnergy : ∀ a : SampledDocking.SupportedAction F,
      (SampledDocking.restrictedDecisionProblem exactDP F).utility a s = coordEnergy (coords a))
    (hApprox : ∀ x : SampledDocking.SupportedAction F,
      |(SampledDocking.restrictedDecisionProblem exactDP F).utility x s -
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility x s| ≤ δ)
    (hStrict : ∀ b : SampledDocking.SupportedAction F, b ≠ aStar →
      (SampledDocking.restrictedDecisionProblem coarseDP F).utility aStar s + 2 * δ <
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility b s)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers
      ((Finset.univ : Finset (SampledDocking.SupportedAction F)).image coords) εCover)
    (hεCover : 0 ≤ εCover)
    (hOpt : IsOptimal (fun x => -coordEnergy x) xStar)
    (hLip : RMSDLipschitzEnergy coordEnergy L) :
    let hSingleton := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      (fun a : SampledDocking.SupportedAction F =>
        (SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
      (fun a : SampledDocking.SupportedAction F =>
        (SampledDocking.restrictedDecisionProblem coarseDP F).utility a s)
      aStar δ hApprox hStrict
    let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a : SampledDocking.SupportedAction F =>
        -(SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
      aStar hSingleton
    coordEnergy (coords w.belief.selection.choice) - coordEnergy xStar ≤ L * εCover := by
  exact exact_energy_gap_certified_choice_of_cover_has_energy_gap_le
    (coords := coords)
    (eExact := fun a : SampledDocking.SupportedAction F =>
      (SampledDocking.restrictedDecisionProblem exactDP F).utility a s)
    (eCoarse := fun a : SampledDocking.SupportedAction F =>
      (SampledDocking.restrictedDecisionProblem coarseDP F).utility a s)
    (coordEnergy := coordEnergy)
    (xStar := xStar)
    (aStar := aStar)
    (δ := δ)
    (L := L)
    (εCover := εCover)
    hEnergy hApprox hStrict hL hCover hεCover hOpt hLip

end ReturnedPoseGuarantee
end Tractability
end DecisionQuotient
