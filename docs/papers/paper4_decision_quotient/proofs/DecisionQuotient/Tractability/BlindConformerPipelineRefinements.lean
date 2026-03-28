/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/BlindConformerPipelineRefinements.lean

  Refinements of the abstract blind-conformer pipeline optimality model:

  * top-k preservation through certified lower-bound thresholding,
  * two-stage cost dominance over all-exact scoring when the coarse pass is
    paid for by the exact work it avoids,
  * seed-budget monotonicity and minimal-adequate-budget optimality,
  * receptor-flexibility correction as a necessary pre-pruning adjustment.
-/

import DecisionQuotient.Tractability.BlindConformerPipelineOptimality
import DecisionQuotient.Tractability.ConformerSupportCoverage
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace BlindConformerPipelineRefinements

open BlindConformerPipelineOptimality
open ConformerSupportCoverage

variable {P : Type*} [Fintype P] [DecidableEq P]

/-- Number of poses with strictly lower energy than `p`. -/
noncomputable def strictLowerCount (energy : P → ℝ) (p : P) : Nat :=
  ((Finset.univ : Finset P).filter fun q => energy q < energy p).card

/-- Energy-side top-k set: poses with fewer than `k` strictly better rivals. -/
noncomputable def energyTopK (energy : P → ℝ) (k : Nat) : Finset P :=
  (Finset.univ : Finset P).filter fun p => strictLowerCount energy p < k

/-- Threshold certificate saying the exact top-k all lie below frontier `τ`. -/
def TopKCoveredByThreshold (energy : P → ℝ) (k : Nat) (τ : ℝ) : Prop :=
  ∀ ⦃p : P⦄, p ∈ energyTopK energy k → energy p ≤ τ

/-- If `lowerBound` is a certified lower bound on exact energy and the exact top-k
    all lie below threshold `τ`, then the canonical lower-bound retain set keeps
    every exact top-k pose. -/
theorem energyTopK_subset_canonicalRetain
    (lowerBound exactEnergy : P → ℝ)
    (k : Nat) (τ : ℝ)
    (hLower : ∀ p, lowerBound p ≤ exactEnergy p)
    (hCover : TopKCoveredByThreshold exactEnergy k τ) :
    energyTopK exactEnergy k ⊆ canonicalRetain lowerBound τ := by
  intro p hp
  have hExact : exactEnergy p ≤ τ := hCover hp
  have hLowerP : lowerBound p ≤ exactEnergy p := hLower p
  simp [canonicalRetain]
  exact le_trans hLowerP hExact

/-- Any exact top-1 pose has energy no greater than the energy of an arbitrary
    witness pose. -/
theorem top1_energy_le_witness
    (exactEnergy : P → ℝ)
    (w : P)
    {p : P}
    (hp : p ∈ energyTopK exactEnergy 1) :
    exactEnergy p ≤ exactEnergy w := by
  by_contra hgt
  have hwLower : exactEnergy w < exactEnergy p := lt_of_not_ge hgt
  have hmem : w ∈ (Finset.univ : Finset P).filter fun q => exactEnergy q < exactEnergy p := by
    simp [hwLower]
  have hcardPos :
      0 < ((Finset.univ : Finset P).filter fun q => exactEnergy q < exactEnergy p).card := by
    exact Finset.card_pos.mpr ⟨w, hmem⟩
  have hpTop : strictLowerCount exactEnergy p < 1 := by
    simpa [energyTopK] using hp
  have hZero : strictLowerCount exactEnergy p = 0 := by omega
  have hStrictPos : 0 < strictLowerCount exactEnergy p := by
    simpa [strictLowerCount] using hcardPos
  omega

/-- If `lowerBound` lower-bounds exact energy and `w` is any witness pose, then
    every exact top-1 pose lies in the canonical retain set at threshold
    `exactEnergy w`. -/
theorem top1_subset_canonicalRetain_of_lowerBound_and_witness
    (lowerBound exactEnergy : P → ℝ)
    (w : P)
    (hLower : ∀ p, lowerBound p ≤ exactEnergy p) :
    energyTopK exactEnergy 1 ⊆ canonicalRetain lowerBound (exactEnergy w) := by
  apply energyTopK_subset_canonicalRetain lowerBound exactEnergy 1 (exactEnergy w) hLower
  intro p hp
  exact top1_energy_le_witness exactEnergy w hp

/-- Specialized support theorem for omitted attractive channels.

    If `baseScore p - omittedBound p` lower-bounds the final exact energy, then
    every exact top-1 pose lies in the canonical retain set defined by that
    posewise omitted-channel lower bound at the witness threshold `exactEnergy w`. -/
theorem top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_witness
    (baseScore exactEnergy omittedBound : P → ℝ)
    (w : P)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p) :
    energyTopK exactEnergy 1 ⊆
      canonicalRetain (fun p => baseScore p - omittedBound p) (exactEnergy w) := by
  exact top1_subset_canonicalRetain_of_lowerBound_and_witness
    (fun p => baseScore p - omittedBound p) exactEnergy w hLower

/-- If a witness exact score is itself upper-bounded by its base score, then the
    base witness threshold is also a certified-safe top-1 support threshold for
    omitted-attractive lower bounds. -/
theorem top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_baseWitness
    (baseScore exactEnergy omittedBound : P → ℝ)
    (w : P)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p)
    (hWitness : exactEnergy w ≤ baseScore w) :
    energyTopK exactEnergy 1 ⊆
      canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w) := by
  intro p hp
  have hMem :
      p ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (exactEnergy w) :=
    top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_witness
      baseScore exactEnergy omittedBound w hLower hp
  have hLe : baseScore p - omittedBound p ≤ exactEnergy w := by
    simpa [canonicalRetain] using hMem
  have hWitnessLe : exactEnergy w ≤ baseScore w := hWitness
  simpa [canonicalRetain] using (le_trans hLe hWitnessLe)

/-- If a support set contains a top-1 witness and `a` is an exact argmin on that
    support, then `a` is globally exact-optimal. -/
theorem support_argmin_is_global_of_top1_member
    (exactEnergy : P → ℝ)
    (support : Finset P)
    {a w : P}
    (ha : a ∈ support)
    (hArgmin : ∀ b ∈ support, exactEnergy a ≤ exactEnergy b)
    (hwTop : w ∈ energyTopK exactEnergy 1)
    (hwSupport : w ∈ support) :
    IsOptimal (fun x => -exactEnergy x) a := by
  intro b
  have hA : exactEnergy a ≤ exactEnergy w := hArgmin w hwSupport
  have hW : exactEnergy w ≤ exactEnergy b :=
    top1_energy_le_witness exactEnergy b hwTop
  linarith

/-- If a support set contains the entire exact top-1 set and `a` is an exact
    argmin on that support, then `a` is globally exact-optimal. -/
theorem support_argmin_is_global_of_top1_subset
    [Nonempty P]
    (exactEnergy : P → ℝ)
    (support : Finset P)
    {a : P}
    (ha : a ∈ support)
    (hArgmin : ∀ b ∈ support, exactEnergy a ≤ exactEnergy b)
    (hTopSubset : energyTopK exactEnergy 1 ⊆ support) :
    IsOptimal (fun x => -exactEnergy x) a := by
  have hTopNonempty : (energyTopK exactEnergy 1).Nonempty := by
    classical
    have hFin := FiniteTopK.topKSet_nonempty (u := fun x => - exactEnergy x) (k := 1) (by omega)
    rw [FiniteTopK.topKSet, FiniteTopK.topKWithTies] at hFin
    have heq : ∀ x, strictLowerCount exactEnergy x = FiniteTopK.strictBetterCount (fun q => -exactEnergy q) x := by
      intro x
      unfold strictLowerCount FiniteTopK.strictBetterCount
      apply congrArg
      apply Finset.filter_congr
      intro y _
      exact ⟨fun h => by linarith, fun h => by linarith⟩
    simp only [energyTopK, heq]
    exact hFin
  rcases hTopNonempty with ⟨w, hwTop⟩
  have hwSupport : w ∈ support := hTopSubset hwTop
  exact support_argmin_is_global_of_top1_member exactEnergy support ha hArgmin hwTop hwSupport

/-- If a support set contains the exact top-1 set and `winner` is a strict argmin
    on that support, then the exact top-1 set is the singleton `{winner}`. -/
theorem support_strict_argmin_is_exact_singleton_of_top1_subset
    [Nonempty P]
    (exactEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (hWinnerMem : winner ∈ support)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → exactEnergy winner < exactEnergy z)
    (hTopSubset : energyTopK exactEnergy 1 ⊆ support) :
    energyTopK exactEnergy 1 = ({winner} : Finset P) := by
  apply Finset.ext
  intro p
  constructor
  · intro hp
    have hpSupport : p ∈ support := hTopSubset hp
    by_cases hEq : p = winner
    · simpa [hEq]
    · have hStrictPW : exactEnergy winner < exactEnergy p := hStrict p hpSupport hEq
      have hTopLe : exactEnergy p ≤ exactEnergy winner := top1_energy_le_witness exactEnergy winner hp
      exact False.elim ((not_lt_of_ge hTopLe) hStrictPW)
  · intro hp
    have hWinnerGlobal : IsOptimal (fun x => -exactEnergy x) winner :=
      support_argmin_is_global_of_top1_subset exactEnergy support hWinnerMem
        (fun z hz => by
          by_cases hEq : z = winner
          · simpa [hEq]
          · exact le_of_lt (hStrict z hz hEq))
        hTopSubset
    have hWinnerTop : winner ∈ energyTopK exactEnergy 1 := by
      classical
      have hFinTop : winner ∈ FiniteTopK.topKSet (fun x => - exactEnergy x) 1 :=
        optimal_mem_topKSet_one (u := fun x => - exactEnergy x) hWinnerGlobal
      rw [FiniteTopK.mem_topKSet_iff] at hFinTop
      simp only [energyTopK]
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have heq : FiniteTopK.strictBetterCount (fun q => -exactEnergy q) winner =
                 ((Finset.univ : Finset P).filter fun q => exactEnergy q < exactEnergy winner).card := by
        unfold FiniteTopK.strictBetterCount
        apply congrArg
        apply Finset.filter_congr
        intro y _
        exact ⟨fun h => by linarith, fun h => by linarith⟩
      rwa [heq] at hFinTop
    have hEq : p = winner := Finset.mem_singleton.mp hp
    subst hEq
    exact hWinnerTop

/-- Support-restricted energy-gap singleton bridge.

    If a coarse guide score is uniformly within `δ` of the exact energy on a
    certified support set that contains the exact top-1 poses, and `winner`
    enjoys a strict `2δ` coarse margin over every other support member, then the
    exact top-1 set is already the singleton `{winner}`. -/
theorem support_strict_argmin_is_exact_singleton_of_support_coarse_energy_gap_margin
    [Nonempty P]
    (exactEnergy coarseEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (δ : ℝ)
    (hWinnerMem : winner ∈ support)
    (hApprox : ∀ z, z ∈ support → |exactEnergy z - coarseEnergy z| ≤ δ)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → coarseEnergy winner + 2 * δ < coarseEnergy z)
    (hTopSubset : energyTopK exactEnergy 1 ⊆ support) :
    energyTopK exactEnergy 1 = ({winner} : Finset P) := by
  apply support_strict_argmin_is_exact_singleton_of_top1_subset
  · exact hWinnerMem
  · intro z hz hne
    have hWinnerApprox := hApprox winner hWinnerMem
    have hZApprox := hApprox z hz
    have hWinnerUpper : exactEnergy winner ≤ coarseEnergy winner + δ := by
      have hRight := (abs_le.mp hWinnerApprox).2
      linarith
    have hZLower : coarseEnergy z - δ ≤ exactEnergy z := by
      have hLeft := (abs_le.mp hZApprox).1
      linarith
    have hGuideGap : coarseEnergy winner + 2 * δ < coarseEnergy z := hStrict z hz hne
    linarith
  · exact hTopSubset

/-- If a pose is the exact argmin over the canonical omitted-attractive support set
    built from a base-score witness, then it is globally exact-optimal. -/
theorem canonicalRetain_argmin_is_global_of_omittedAttractiveLowerBound_and_baseWitness
    (baseScore exactEnergy omittedBound : P → ℝ)
    {a w : P}
    (ha : a ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (hArgmin : ∀ b ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w),
      exactEnergy a ≤ exactEnergy b)
    (hwTop : w ∈ energyTopK exactEnergy 1)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p)
    (hWitness : exactEnergy w ≤ baseScore w) :
    IsOptimal (fun x => -exactEnergy x) a := by
  have hTopSubset :=
    top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_baseWitness
      baseScore exactEnergy omittedBound w hLower hWitness
  have hwSupport : w ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w) :=
    hTopSubset hwTop
  exact support_argmin_is_global_of_top1_member exactEnergy
    (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    ha hArgmin hwTop hwSupport

/-- Runtime-facing omitted-attractive support theorem: if the canonical retain set
    is built from a base-score witness threshold and `a` is the exact argmin on
    that support, then `a` is globally exact-optimal. -/
theorem canonicalRetain_argmin_is_global_of_omittedAttractiveLowerBound_and_baseWitness_subset
    [Nonempty P]
    (baseScore exactEnergy omittedBound : P → ℝ)
    {a w : P}
    (ha : a ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (hArgmin : ∀ b ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w),
      exactEnergy a ≤ exactEnergy b)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p)
    (hWitness : exactEnergy w ≤ baseScore w) :
    IsOptimal (fun x => -exactEnergy x) a := by
  have hTopSubset :=
    top1_subset_canonicalRetain_of_omittedAttractiveLowerBound_and_baseWitness
      baseScore exactEnergy omittedBound w hLower hWitness
  exact support_argmin_is_global_of_top1_subset exactEnergy
    (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    ha hArgmin hTopSubset

/-- Patch an exact-energy family by keeping exact scores on `support` and using a
    fallback score outside the support. This models runtime winner-only rescoring
    on a certified support subset. -/
def patchedSupportEnergy
    (support : Finset P) (exactEnergy : P → ℝ) (fallback : ℝ) (p : P) : ℝ :=
  if p ∈ support then exactEnergy p else fallback

/-- If `winner` is the exact argmin over `support` and the fallback score is no
    better than `winner`, then `winner` is also the global argmin of the patched
    support-only score family. -/
theorem support_argmin_is_patched_global_argmin
    (exactEnergy : P → ℝ)
    (support : Finset P)
    (fallback : ℝ)
    {winner : P}
    (hWinnerMem : winner ∈ support)
    (hWinnerBest : ∀ z, z ∈ support → exactEnergy winner ≤ exactEnergy z)
    (hFallback : exactEnergy winner ≤ fallback) :
    ∀ z, patchedSupportEnergy support exactEnergy fallback winner ≤
      patchedSupportEnergy support exactEnergy fallback z := by
  intro z
  by_cases hz : z ∈ support
  · simp [patchedSupportEnergy, hWinnerMem, hz]
    exact hWinnerBest z hz
  · simp [patchedSupportEnergy, hWinnerMem, hz]
    exact hFallback

/-- Specialized patched-score theorem for omitted-attractive support sets.

    If `winner` is the exact argmin on the canonical omitted-attractive support
    built from the base witness threshold `baseScore w`, then patching every pose
    outside that support to `baseScore w` keeps `winner` as the global argmin of
    the patched runtime score vector. -/
theorem omittedAttractive_support_argmin_is_patched_global_argmin_of_baseWitness
    (baseScore exactEnergy omittedBound : P → ℝ)
    {winner w : P}
    (hWinnerMem : winner ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (hWinnerBest : ∀ z, z ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w) →
      exactEnergy winner ≤ exactEnergy z)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p)
    (hWitness : exactEnergy w ≤ baseScore w) :
    ∀ z,
      patchedSupportEnergy
        (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
        exactEnergy (baseScore w) winner
        ≤
      patchedSupportEnergy
        (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
        exactEnergy (baseScore w) z := by
  exact support_argmin_is_patched_global_argmin
    exactEnergy
    (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (baseScore w)
    hWinnerMem hWinnerBest (by
      have hwSupport : w ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w) := by
        simp [canonicalRetain]
        linarith [hLower w, hWitness]
      have h1 : exactEnergy winner ≤ exactEnergy w := hWinnerBest w hwSupport
      linarith
    )

/-- Runtime-facing patched-support theorem: if `winner` is the exact argmin on the
    omitted-attractive canonical support built from a base witness threshold, then
    patching every pose outside that support to the base witness threshold keeps
    `winner` as the global argmin of the patched runtime score family. -/
theorem omittedAttractive_support_argmin_is_patched_global_argmin_of_baseWitness_subset
    [Nonempty P]
    (baseScore exactEnergy omittedBound : P → ℝ)
    {winner w : P}
    (hWinnerMem : winner ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (hWinnerBest : ∀ z, z ∈ canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w) →
      exactEnergy winner ≤ exactEnergy z)
    (hLower : ∀ p, baseScore p - omittedBound p ≤ exactEnergy p)
    (hWitness : exactEnergy w ≤ baseScore w) :
    ∀ z,
      patchedSupportEnergy
        (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
        exactEnergy (baseScore w) winner
        ≤
      patchedSupportEnergy
        (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
        exactEnergy (baseScore w) z := by
  have hGlobal : IsOptimal (fun x => -exactEnergy x) winner :=
    canonicalRetain_argmin_is_global_of_omittedAttractiveLowerBound_and_baseWitness_subset
      baseScore exactEnergy omittedBound hWinnerMem hWinnerBest hLower hWitness
  have hFallback : exactEnergy winner ≤ baseScore w := by
    have h : -exactEnergy w ≤ -exactEnergy winner := hGlobal w
    linarith
  exact support_argmin_is_patched_global_argmin
    exactEnergy
    (canonicalRetain (fun p => baseScore p - omittedBound p) (baseScore w))
    (baseScore w)
    hWinnerMem hWinnerBest hFallback

/-- If a support set contains the exact top-1 set and `winner` is a strict argmin
    on that support with fallback strictly above `winner`, then the patched
    support-only score family has singleton top-1 `{winner}`. -/
theorem patchedSupportEnergy_singleton_of_strict_support_argmin
    [Nonempty P]
    (exactEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (fallback : ℝ)
    (hWinnerMem : winner ∈ support)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → exactEnergy winner < exactEnergy z)
    (hTopSubset : energyTopK exactEnergy 1 ⊆ support)
    (hFallback : exactEnergy winner < fallback) :
    energyTopK (patchedSupportEnergy support exactEnergy fallback) 1 = ({winner} : Finset P) := by
  apply support_strict_argmin_is_exact_singleton_of_top1_subset
  · exact hWinnerMem
  · intro z hz hne
    by_cases hzSupport : z ∈ support
    · simp [patchedSupportEnergy, hWinnerMem, hzSupport]
      exact hStrict z hzSupport hne
    · simp [patchedSupportEnergy, hWinnerMem, hzSupport]
      exact hFallback
  · intro p hp
    simp only [energyTopK] at hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
    have hpCount : strictLowerCount (patchedSupportEnergy support exactEnergy fallback) p < 1 := hp
    by_cases hpSupport : p ∈ support
    · exact hpSupport
    · have hmem : winner ∈ (Finset.univ : Finset P).filter fun q => patchedSupportEnergy support exactEnergy fallback q < patchedSupportEnergy support exactEnergy fallback p := by
        simp [patchedSupportEnergy, hWinnerMem, hpSupport]
        exact hFallback
      have hcardPos : 0 < ((Finset.univ : Finset P).filter fun q => patchedSupportEnergy support exactEnergy fallback q < patchedSupportEnergy support exactEnergy fallback p).card := by
        exact Finset.card_pos.mpr ⟨winner, hmem⟩
      have : ¬ strictLowerCount (patchedSupportEnergy support exactEnergy fallback) p < 1 := by
        unfold strictLowerCount
        omega
      exact False.elim (this hpCount)

/-- A strict support argmin plus a fallback above the winner already forces the
    patched support-only energy family to have singleton top-1, without any
    separate global-top1-subset hypothesis. -/
theorem patchedSupportEnergy_singleton_of_strict_support_argmin_without_top1_subset
    [Nonempty P]
    (exactEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (fallback : ℝ)
    (hWinnerMem : winner ∈ support)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → exactEnergy winner < exactEnergy z)
    (hFallback : exactEnergy winner < fallback) :
    energyTopK (patchedSupportEnergy support exactEnergy fallback) 1 = ({winner} : Finset P) := by
  have hPatchedTopSubset : energyTopK (patchedSupportEnergy support exactEnergy fallback) 1 ⊆ support := by
    intro p hp
    simp only [energyTopK] at hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
    have hpCount : strictLowerCount (patchedSupportEnergy support exactEnergy fallback) p < 1 := hp
    by_cases hpSupport : p ∈ support
    · exact hpSupport
    · have hmem : winner ∈ (Finset.univ : Finset P).filter fun q =>
          patchedSupportEnergy support exactEnergy fallback q <
            patchedSupportEnergy support exactEnergy fallback p := by
        simp [patchedSupportEnergy, hWinnerMem, hpSupport]
        exact hFallback
      have hcardPos : 0 < ((Finset.univ : Finset P).filter fun q =>
          patchedSupportEnergy support exactEnergy fallback q <
            patchedSupportEnergy support exactEnergy fallback p).card := by
        exact Finset.card_pos.mpr ⟨winner, hmem⟩
      have : ¬ strictLowerCount (patchedSupportEnergy support exactEnergy fallback) p < 1 := by
        unfold strictLowerCount
        omega
      exact False.elim (this hpCount)
  apply support_strict_argmin_is_exact_singleton_of_top1_subset
  · exact hWinnerMem
  · intro z hz hne
    simp [patchedSupportEnergy, hWinnerMem, hz]
    exact hStrict z hz hne
  · exact hPatchedTopSubset

/-- Patched-support singleton bridge from a support-restricted coarse margin.

    If a coarse guide score is uniformly within `δ` of the exact energy on a
    support set containing the exact top-1 poses, `winner` has a strict `2δ`
    guide-score margin on that support, and the fallback score sits strictly
    above the exact winner energy, then the patched support-only energy family
    has singleton top-1 `{winner}`. -/
theorem patchedSupportEnergy_singleton_of_support_coarse_energy_gap_margin
    [Nonempty P]
    (exactEnergy coarseEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (fallback δ : ℝ)
    (hWinnerMem : winner ∈ support)
    (hApprox : ∀ z, z ∈ support → |exactEnergy z - coarseEnergy z| ≤ δ)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → coarseEnergy winner + 2 * δ < coarseEnergy z)
    (hTopSubset : energyTopK exactEnergy 1 ⊆ support)
    (hFallback : exactEnergy winner < fallback) :
    energyTopK (patchedSupportEnergy support exactEnergy fallback) 1 = ({winner} : Finset P) := by
  apply patchedSupportEnergy_singleton_of_strict_support_argmin
  · exact hWinnerMem
  · intro z hz hne
    have hWinnerApprox := hApprox winner hWinnerMem
    have hZApprox := hApprox z hz
    have hWinnerUpper : exactEnergy winner ≤ coarseEnergy winner + δ := by
      have hRight := (abs_le.mp hWinnerApprox).2
      linarith
    have hZLower : coarseEnergy z - δ ≤ exactEnergy z := by
      have hLeft := (abs_le.mp hZApprox).1
      linarith
    have hGuideGap : coarseEnergy winner + 2 * δ < coarseEnergy z := hStrict z hz hne
    linarith
  · exact hTopSubset
  · exact hFallback

/-- Support-restricted coarse-margin bridge without a separate global-top1-subset
    premise. The fallback excludes everything outside `support`, so a strict
    support witness suffices to make the patched family singleton. -/
theorem patchedSupportEnergy_singleton_of_support_coarse_energy_gap_margin_without_top1_subset
    [Nonempty P]
    (exactEnergy coarseEnergy : P → ℝ)
    (support : Finset P)
    (winner : P)
    (fallback δ : ℝ)
    (hWinnerMem : winner ∈ support)
    (hApprox : ∀ z, z ∈ support → |exactEnergy z - coarseEnergy z| ≤ δ)
    (hStrict : ∀ z, z ∈ support → z ≠ winner → coarseEnergy winner + 2 * δ < coarseEnergy z)
    (hFallback : exactEnergy winner < fallback) :
    energyTopK (patchedSupportEnergy support exactEnergy fallback) 1 = ({winner} : Finset P) := by
  apply patchedSupportEnergy_singleton_of_strict_support_argmin_without_top1_subset
  · exact hWinnerMem
  · intro z hz hne
    have hWinnerApprox := hApprox winner hWinnerMem
    have hZApprox := hApprox z hz
    have hWinnerUpper : exactEnergy winner ≤ coarseEnergy winner + δ := by
      have hRight := (abs_le.mp hWinnerApprox).2
      linarith
    have hZLower : coarseEnergy z - δ ≤ exactEnergy z := by
      have hLeft := (abs_le.mp hZApprox).1
      linarith
    have hGuideGap : coarseEnergy winner + 2 * δ < coarseEnergy z := hStrict z hz hne
    linarith
  · exact hFallback

/-- If a support set contains the exact top-k and the fallback score is no better
    than the exact top-k threshold, then every exact top-k pose remains in the
    top-k set of the patched support-only score family. -/
theorem energyTopK_subset_patchedSupportEnergy_topK
    (exactEnergy : P → ℝ)
    (support : Finset P)
    (k : Nat)
    (τ fallback : ℝ)
    (hSupport : energyTopK exactEnergy k ⊆ support)
    (hCover : TopKCoveredByThreshold exactEnergy k τ)
    (hFallback : τ ≤ fallback) :
    energyTopK exactEnergy k ⊆ energyTopK (patchedSupportEnergy support exactEnergy fallback) k := by
  intro p hp
  have hpSupport : p ∈ support := hSupport hp
  have hpTau : exactEnergy p ≤ τ := hCover hp
  simp only [energyTopK] at hp ⊢
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
  have hCount : strictLowerCount exactEnergy p < k := hp
  have hCountMono :
      strictLowerCount (patchedSupportEnergy support exactEnergy fallback) p ≤ strictLowerCount exactEnergy p := by
    unfold strictLowerCount
    refine Finset.card_le_card ?_
    intro q hq
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hq ⊢
    by_cases hqSupport : q ∈ support
    · simp [patchedSupportEnergy, hpSupport, hqSupport] at hq ⊢
      exact hq
    · have hPatchedQ : patchedSupportEnergy support exactEnergy fallback q = fallback := by
        simp [patchedSupportEnergy, hqSupport]
      have hPatchedP : patchedSupportEnergy support exactEnergy fallback p = exactEnergy p := by
        simp [patchedSupportEnergy, hpSupport]
      rw [hPatchedQ, hPatchedP] at hq
      have : ¬ fallback < exactEnergy p := by
        exact not_lt_of_ge (le_trans hpTau hFallback)
      exact False.elim (this hq)
  exact lt_of_le_of_lt hCountMono hCount

/-- Any certified-safe retain set for frontier `τ` must contain every exact top-k
    pose whose exact energies are covered by that frontier. -/
theorem energyTopK_subset_of_certifiedSafeRetain
    (lowerBound exactEnergy : P → ℝ)
    (k : Nat) (τ : ℝ)
    {retain : Finset P}
    (hLower : ∀ p, lowerBound p ≤ exactEnergy p)
    (hCover : TopKCoveredByThreshold exactEnergy k τ)
    (hSafe : CertifiedSafeForThreshold lowerBound retain τ) :
    energyTopK exactEnergy k ⊆ retain := by
  have hCanon : energyTopK exactEnergy k ⊆ canonicalRetain lowerBound τ :=
    energyTopK_subset_canonicalRetain lowerBound exactEnergy k τ hLower hCover
  exact Finset.Subset.trans hCanon (canonicalRetain_subset_of_certifiedSafe lowerBound τ hSafe)

/-- Exact all-pose evaluation cost. -/
def allExactCost (exactCost : P → ℝ) : ℝ :=
  retainedCost (Finset.univ : Finset P) exactCost

/-- If the coarse/prefilter stage costs no more than the exact work avoided on the
    poses it prunes, then the canonical two-stage pipeline costs no more than
    evaluating exact scoring on all poses. -/
theorem canonical_twoStage_le_allExact
    (lowerBound : P → ℝ)
    (τ prefilterCost : ℝ)
    (exactCost : P → ℝ)
    (hPrefilterCovered :
      prefilterCost ≤ retainedCost ((Finset.univ : Finset P) \ canonicalRetain lowerBound τ) exactCost)
    :
    pipelineCost prefilterCost (canonicalRetain lowerBound τ) exactCost ≤
      allExactCost exactCost := by
  unfold pipelineCost allExactCost retainedCost
  unfold retainedCost at hPrefilterCovered
  have hSplit := Finset.sum_sdiff
    (s₁ := canonicalRetain lowerBound τ)
    (s₂ := (Finset.univ : Finset P))
    (f := exactCost)
    (by
      intro p hp
      exact Finset.mem_univ p)
  have hStep0 :=
    add_le_add_left hPrefilterCovered (Finset.sum (canonicalRetain lowerBound τ) exactCost)
  have hStep :
      prefilterCost + Finset.sum (canonicalRetain lowerBound τ) exactCost ≤
        Finset.sum ((Finset.univ : Finset P) \ canonicalRetain lowerBound τ) exactCost +
          Finset.sum (canonicalRetain lowerBound τ) exactCost := by
    simpa [add_comm, add_left_comm, add_assoc] using hStep0
  have hEq :
      Finset.sum ((Finset.univ : Finset P) \ canonicalRetain lowerBound τ) exactCost +
          Finset.sum (canonicalRetain lowerBound τ) exactCost =
        Finset.sum (Finset.univ : Finset P) exactCost := by
    simpa [add_comm, add_left_comm, add_assoc] using hSplit
  exact le_trans hStep (le_of_eq hEq)

/-- Total cost of a seed budget `n`: seed overhead plus the canonical retain-set
    cost induced by the corresponding lower-bound family. -/
noncomputable def canonicalSeedBudgetCost
    (lowerBoundFamily : ℕ → P → ℝ)
    (τ : ℝ)
    (seedOverhead : ℕ → ℝ)
    (postFilterCost : P → ℝ)
    (n : ℕ) : ℝ :=
  pipelineCost (seedOverhead n) (canonicalRetain (lowerBoundFamily n) τ) postFilterCost

/-- More seed budgets can only lower certified lower bounds. -/
def LowerBoundFamilyMonotone
    (lowerBoundFamily : ℕ → P → ℝ) : Prop :=
  ∀ ⦃m n : ℕ⦄, m ≤ n → ∀ p, lowerBoundFamily n p ≤ lowerBoundFamily m p

/-- Under monotone lower-bound families, the canonical retain set grows with the
    number of seed budgets. -/
theorem canonicalRetain_mono_of_lowerBoundFamilyMonotone
    (lowerBoundFamily : ℕ → P → ℝ)
    (τ : ℝ)
    (hMono : LowerBoundFamilyMonotone lowerBoundFamily)
    {m n : ℕ}
    (hmn : m ≤ n) :
    canonicalRetain (lowerBoundFamily m) τ ⊆ canonicalRetain (lowerBoundFamily n) τ := by
  intro p hp
  have hMem : lowerBoundFamily m p ≤ τ := by
    simpa [canonicalRetain] using hp
  have hLower : lowerBoundFamily n p ≤ lowerBoundFamily m p := hMono hmn p
  simp [canonicalRetain]
  exact le_trans hLower hMem

/-- Canonical seed-budget cost is monotone once seed overhead is monotone and
    post-filter costs are nonnegative. -/
theorem canonicalSeedBudgetCost_mono
    (lowerBoundFamily : ℕ → P → ℝ)
    (τ : ℝ)
    (seedOverhead : ℕ → ℝ)
    (postFilterCost : P → ℝ)
    (hMono : LowerBoundFamilyMonotone lowerBoundFamily)
    (hSeedMono : Monotone seedOverhead)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p)
    {m n : ℕ}
    (hmn : m ≤ n) :
    canonicalSeedBudgetCost lowerBoundFamily τ seedOverhead postFilterCost m ≤
      canonicalSeedBudgetCost lowerBoundFamily τ seedOverhead postFilterCost n := by
  unfold canonicalSeedBudgetCost
  have hRetain :
      retainedCost (canonicalRetain (lowerBoundFamily m) τ) postFilterCost ≤
        retainedCost (canonicalRetain (lowerBoundFamily n) τ) postFilterCost :=
    retainedCost_mono
      (canonicalRetain_mono_of_lowerBoundFamilyMonotone lowerBoundFamily τ hMono hmn)
      postFilterCost hNonneg
  have hSeed : seedOverhead m ≤ seedOverhead n := hSeedMono hmn
  exact add_le_add hSeed hRetain

/-- A seed budget is adequate for exact top-k if its canonical retain set keeps
    every exact top-k pose below threshold `τ`. -/
def AdequateSeedBudget
    (lowerBoundFamily : ℕ → P → ℝ)
    (exactEnergy : P → ℝ)
    (k : Nat)
    (τ : ℝ)
    (n : ℕ) : Prop :=
  energyTopK exactEnergy k ⊆ canonicalRetain (lowerBoundFamily n) τ

/-- Adequacy is monotone in the number of seeds whenever the lower-bound family is
    monotone. -/
theorem adequateSeedBudget_mono
    (lowerBoundFamily : ℕ → P → ℝ)
    (exactEnergy : P → ℝ)
    (k : Nat)
    (τ : ℝ)
    (hMono : LowerBoundFamilyMonotone lowerBoundFamily)
    {m n : ℕ}
    (hmn : m ≤ n)
    (hAdeq : AdequateSeedBudget lowerBoundFamily exactEnergy k τ m) :
    AdequateSeedBudget lowerBoundFamily exactEnergy k τ n := by
  exact Finset.Subset.trans hAdeq
    (canonicalRetain_mono_of_lowerBoundFamilyMonotone lowerBoundFamily τ hMono hmn)

/-- A minimal adequate seed budget is optimal among all adequate seed budgets once
    seed-budget cost is monotone. -/
theorem minimal_adequate_seedBudget_optimal
    (lowerBoundFamily : ℕ → P → ℝ)
    (exactEnergy : P → ℝ)
    (k : Nat)
    (τ : ℝ)
    (seedOverhead : ℕ → ℝ)
    (postFilterCost : P → ℝ)
    (hMono : LowerBoundFamilyMonotone lowerBoundFamily)
    (hSeedMono : Monotone seedOverhead)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p)
    {n₀ n : ℕ}
    (hMinAdeq : AdequateSeedBudget lowerBoundFamily exactEnergy k τ n₀)
    (hMinimal : ∀ m < n₀, ¬ AdequateSeedBudget lowerBoundFamily exactEnergy k τ m)
    (hAdeqN : AdequateSeedBudget lowerBoundFamily exactEnergy k τ n) :
    canonicalSeedBudgetCost lowerBoundFamily τ seedOverhead postFilterCost n₀ ≤
      canonicalSeedBudgetCost lowerBoundFamily τ seedOverhead postFilterCost n := by
  have hn₀le : n₀ ≤ n := by
    by_contra hNot
    exact hMinimal n (Nat.lt_of_not_ge hNot) hAdeqN
  exact canonicalSeedBudgetCost_mono
    lowerBoundFamily τ seedOverhead postFilterCost hMono hSeedMono hNonneg hn₀le

/-- Receptor-flexibility correction lowers the certified conformer lower bound by
    the receptor mismatch allowance. -/
def flexCorrectedLowerBound
    (rigidScore improvementBound receptorFlexError : P → ℝ) (p : P) : ℝ :=
  rigidScore p - improvementBound p - receptorFlexError p

/-- If both the reference and alternative receptor-conformation scores are each
    approximated within `δ`, then the exact receptor-flexibility gap is bounded by
    the coarse gap plus `2δ`. -/
theorem exact_flex_gap_le_coarse_gap_plus_two_delta
    (exactRef exactConf coarseRef coarseConf δ : ℝ)
    (hRef : |exactRef - coarseRef| ≤ δ)
    (hConf : |exactConf - coarseConf| ≤ δ) :
    |exactConf - exactRef| ≤ |coarseConf - coarseRef| + 2 * δ := by
  have hRef' : |coarseRef - exactRef| ≤ δ := by
    simpa [abs_sub_comm] using hRef
  have hConf' : |exactConf - coarseConf| ≤ δ := hConf
  have hTriangle :
      |(exactConf - coarseConf) + (coarseConf - coarseRef) + (coarseRef - exactRef)| ≤
        |exactConf - coarseConf| + |coarseConf - coarseRef| + |coarseRef - exactRef| := by
    calc
      |(exactConf - coarseConf) + (coarseConf - coarseRef) + (coarseRef - exactRef)|
        = |((exactConf - coarseConf) + (coarseConf - coarseRef)) + (coarseRef - exactRef)| := by ring_nf
      _ ≤ |(exactConf - coarseConf) + (coarseConf - coarseRef)| + |coarseRef - exactRef| := by
        exact abs_add_le _ _
      _ ≤ |exactConf - coarseConf| + |coarseConf - coarseRef| + |coarseRef - exactRef| := by
        gcongr
        exact abs_add_le _ _
  calc
    |exactConf - exactRef|
      = |(exactConf - coarseConf) + (coarseConf - coarseRef) + (coarseRef - exactRef)| := by ring_nf
    _ ≤ |exactConf - coarseConf| + |coarseConf - coarseRef| + |coarseRef - exactRef| := hTriangle
    _ ≤ δ + |coarseConf - coarseRef| + δ := by gcongr
    _ = |coarseConf - coarseRef| + 2 * δ := by ring

/-- Certified blind-conformer lower bound from a coarse rigid score, a conformer
    improvement allowance, and a coarse receptor-flexibility allowance.

    This packages the runtime formula

      coarseRef - confImprove - coarseFlexGap - 3δ

    as a sound lower bound on the final exact energy whenever the exact rigid
    score is approximated within `δ` and the exact receptor-flexibility gap is
    bounded using `exact_flex_gap_le_coarse_gap_plus_two_delta`. -/
theorem coarse_rigid_with_flex_and_conformer_lower_bound
    (exactRigid coarseRigid exactFinal confImprove coarseFlexGap δ : ℝ)
    (hApprox : |exactRigid - coarseRigid| ≤ δ)
    (hFinal : exactRigid - confImprove - (coarseFlexGap + 2 * δ) ≤ exactFinal) :
    coarseRigid - confImprove - coarseFlexGap - 3 * δ ≤ exactFinal := by
  have hLower : coarseRigid - δ ≤ exactRigid := by
    linarith [(abs_le.mp hApprox).left]
  linarith

/-- Certified lower bound for the flexible objective after adding receptor-error
    allowance. -/
theorem flexCorrected_is_certified_lowerBound
    (rigidScore bestFlexibleEnergy improvementBound receptorFlexError : P → ℝ)
    (hBound : ∀ p,
      bestFlexibleEnergy p ≥ flexCorrectedLowerBound rigidScore improvementBound receptorFlexError p) :
    ∀ p,
      flexCorrectedLowerBound rigidScore improvementBound receptorFlexError p ≤
        bestFlexibleEnergy p := by
  intro p
  exact hBound p

/-- If receptor-flexibility correction is nonnegative, the flex-corrected retain
    set contains the rigid-only retain set. -/
theorem rigidRetain_subset_flexCorrectedRetain
    (rigidScore improvementBound receptorFlexError : P → ℝ)
    (τ : ℝ)
    (hFlexNonneg : ∀ p, 0 ≤ receptorFlexError p) :
    canonicalRetain (conformerLowerBound rigidScore improvementBound) τ ⊆
      canonicalRetain (flexCorrectedLowerBound rigidScore improvementBound receptorFlexError) τ := by
  intro p hp
  have hRigid : conformerLowerBound rigidScore improvementBound p ≤ τ := by
    simpa [canonicalRetain] using hp
  have hErr : 0 ≤ receptorFlexError p := hFlexNonneg p
  have hFlexLeRigid :
      flexCorrectedLowerBound rigidScore improvementBound receptorFlexError p ≤
        conformerLowerBound rigidScore improvementBound p := by
    simp [flexCorrectedLowerBound, conformerLowerBound]
    linarith
  have hFlex : flexCorrectedLowerBound rigidScore improvementBound receptorFlexError p ≤ τ :=
    le_trans hFlexLeRigid hRigid
  simpa [canonicalRetain] using hFlex

/-- Witness theorem: if a pose is pruned by the rigid-only lower bound but kept by
    the flex-corrected lower bound, then receptor-flexibility correction must be
    applied before pruning to remain certified-safe for the flexible objective. -/
theorem flexCorrection_can_change_pruning_decision
    (rigidScore improvementBound receptorFlexError : P → ℝ)
    (τ : ℝ)
    (p : P)
    (hRigidPruned : τ < conformerLowerBound rigidScore improvementBound p)
    (hFlexKept : flexCorrectedLowerBound rigidScore improvementBound receptorFlexError p ≤ τ) :
    p ∉ canonicalRetain (conformerLowerBound rigidScore improvementBound) τ ∧
      p ∈ canonicalRetain (flexCorrectedLowerBound rigidScore improvementBound receptorFlexError) τ := by
  constructor
  · simp [canonicalRetain, not_le_of_gt hRigidPruned]
  · simp [canonicalRetain, hFlexKept]

end BlindConformerPipelineRefinements
end Tractability
end DecisionQuotient
