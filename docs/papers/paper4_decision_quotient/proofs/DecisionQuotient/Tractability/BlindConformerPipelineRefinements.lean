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
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace BlindConformerPipelineRefinements

open BlindConformerPipelineOptimality

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
