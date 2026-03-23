/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/BlindConformerPipelineOptimality.lean

  This file formalizes an abstract pipeline-optimality result for site-known,
  blind-conformer docking.

  The key runtime question is not whether a lower-bound-guided pipeline is
  empirically best on every benchmark, but whether it is optimal among all
  *certified-safe* pipelines under an explicit cost model.

  We model:

  * `rigidScore p`               : cheap rigid score for pose `p`
  * `improvementBound p`         : certified upper bound on how much conformer
                                   adaptation can improve pose `p`
  * `conformerLowerBound p`      : certified lower bound on the best achievable
                                   conformer-aware energy for pose `p`
  * `τ`                          : incumbent / frontier energy threshold
  * `retain`                     : poses kept for expensive conformer search

  A pipeline is *certified-safe* if every excluded pose has lower bound strictly
  above the threshold. Under nonnegative post-filter costs, the canonical set

      { p | conformerLowerBound p ≤ τ }

  is the unique minimal certified-safe retain set, and therefore minimizes total
  runtime among all certified-safe pipelines sharing the same prefilter stage.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace BlindConformerPipelineOptimality

open scoped BigOperators

variable {P : Type*} [Fintype P] [DecidableEq P]

/-- Certified lower bound on the best conformer-aware energy reachable from a
    rigid pose. -/
def conformerLowerBound
    (rigidScore improvementBound : P → ℝ) (p : P) : ℝ :=
  rigidScore p - improvementBound p

/-- Canonical retain set: keep exactly those poses whose certified conformer
    lower bound does not already exceed the current threshold. -/
noncomputable def canonicalRetain
    (lowerBound : P → ℝ) (τ : ℝ) : Finset P :=
  Finset.univ.filter fun p => lowerBound p ≤ τ

/-- A retain set is certified-safe for threshold `τ` when every excluded pose is
    formally ruled out by its lower bound. -/
def CertifiedSafeForThreshold
    (lowerBound : P → ℝ) (retain : Finset P) (τ : ℝ) : Prop :=
  ∀ ⦃p : P⦄, p ∉ retain → τ < lowerBound p

/-- Total post-filter work for a retain set under a pose-wise nonnegative cost
    model. -/
def retainedCost
    (retain : Finset P) (postFilterCost : P → ℝ) : ℝ :=
  retain.sum postFilterCost

/-- Full pipeline cost = fixed prefilter cost + post-filter retained cost. -/
def pipelineCost
    (prefilterCost : ℝ) (retain : Finset P) (postFilterCost : P → ℝ) : ℝ :=
  prefilterCost + retainedCost retain postFilterCost

/-- The canonical retain set is certified-safe. -/
theorem canonicalRetain_certifiedSafe
    (lowerBound : P → ℝ) (τ : ℝ) :
    CertifiedSafeForThreshold lowerBound (canonicalRetain lowerBound τ) τ := by
  intro p hp
  have hmem : ¬ lowerBound p ≤ τ := by
    intro hle
    exact hp (by
      simp [canonicalRetain, hle])
  exact lt_of_not_ge hmem

/-- Any certified-safe retain set must contain the canonical one. -/
theorem canonicalRetain_subset_of_certifiedSafe
    (lowerBound : P → ℝ) (τ : ℝ)
    {retain : Finset P}
    (hSafe : CertifiedSafeForThreshold lowerBound retain τ) :
    canonicalRetain lowerBound τ ⊆ retain := by
  intro p hp
  by_contra hnot
  have hlt := hSafe hnot
  have hle : lowerBound p ≤ τ := by
    simpa [canonicalRetain] using hp
  exact (not_lt_of_ge hle) hlt

/-- Retained cost is monotone under subset inclusion when per-pose costs are
    nonnegative. -/
theorem retainedCost_mono
    {retain₁ retain₂ : Finset P}
    (hSubset : retain₁ ⊆ retain₂)
    (postFilterCost : P → ℝ)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p) :
    retainedCost retain₁ postFilterCost ≤ retainedCost retain₂ postFilterCost := by
  unfold retainedCost
  have hExtraNonneg : 0 ≤ Finset.sum (retain₂ \ retain₁) postFilterCost := by
    refine Finset.sum_nonneg ?_
    intro x _
    exact hNonneg x
  have hExpand := Finset.sum_sdiff hSubset (f := postFilterCost)
  have : Finset.sum retain₁ postFilterCost ≤
      Finset.sum (retain₂ \ retain₁) postFilterCost + Finset.sum retain₁ postFilterCost := by
    exact le_add_of_nonneg_left hExtraNonneg
  calc
    Finset.sum retain₁ postFilterCost
      ≤ Finset.sum (retain₂ \ retain₁) postFilterCost + Finset.sum retain₁ postFilterCost := this
    _ = Finset.sum retain₂ postFilterCost := by
      simpa [add_comm, add_left_comm, add_assoc] using hExpand

/-- Among all certified-safe retain sets, the canonical retain set minimizes the
    post-filter cost. -/
theorem canonicalRetain_minimizes_retainedCost
    (lowerBound : P → ℝ) (τ : ℝ)
    {retain : Finset P}
    (hSafe : CertifiedSafeForThreshold lowerBound retain τ)
    (postFilterCost : P → ℝ)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p) :
    retainedCost (canonicalRetain lowerBound τ) postFilterCost ≤
      retainedCost retain postFilterCost := by
  apply retainedCost_mono
  · exact canonicalRetain_subset_of_certifiedSafe lowerBound τ hSafe
  · exact hNonneg

/-- Adding a fixed prefilter cost preserves the optimality of the canonical
    retain set among all certified-safe pipelines. -/
theorem canonicalRetain_minimizes_pipelineCost
    (lowerBound : P → ℝ) (τ prefilterCost : ℝ)
    {retain : Finset P}
    (hSafe : CertifiedSafeForThreshold lowerBound retain τ)
    (postFilterCost : P → ℝ)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p) :
    pipelineCost prefilterCost (canonicalRetain lowerBound τ) postFilterCost ≤
      pipelineCost prefilterCost retain postFilterCost := by
  unfold pipelineCost
  have h :=
    canonicalRetain_minimizes_retainedCost lowerBound τ hSafe postFilterCost hNonneg
  simpa [add_comm, add_left_comm, add_assoc] using add_le_add_left h prefilterCost

/-- If `improvementBound p` upper-bounds the best possible conformer-driven
    energy decrease from pose `p`, then `rigidScore p - improvementBound p` is a
    certified lower bound on the best conformer-aware energy reachable from `p`.

    This theorem does not yet choose a pipeline; it just packages the runtime
    lower bound derived from rigid score and conformer improvement certificates. -/
theorem rigid_minus_improvement_is_certified_lowerBound
    (rigidScore bestConformerEnergy improvementBound : P → ℝ)
    (hBound : ∀ p, bestConformerEnergy p ≥ rigidScore p - improvementBound p) :
    ∀ p, conformerLowerBound rigidScore improvementBound p ≤ bestConformerEnergy p := by
  intro p
  exact hBound p

/-- Main optimal-pipeline theorem for blind-conformer docking.

    Under an explicit cost model with nonnegative post-filter costs, and under
    the certified lower-bound model induced by rigid score minus conformer
    improvement bound, the canonical retain set minimizes total runtime among all
    certified-safe pipelines that share the same prefilter stage. -/
theorem rigid_plus_bound_pipeline_optimal
    (rigidScore improvementBound : P → ℝ)
    (τ prefilterCost : ℝ)
    {retain : Finset P}
    (hSafe : CertifiedSafeForThreshold
      (conformerLowerBound rigidScore improvementBound) retain τ)
    (postFilterCost : P → ℝ)
    (hNonneg : ∀ p, 0 ≤ postFilterCost p) :
    pipelineCost prefilterCost
      (canonicalRetain (conformerLowerBound rigidScore improvementBound) τ)
      postFilterCost
      ≤
    pipelineCost prefilterCost retain postFilterCost := by
  exact canonicalRetain_minimizes_pipelineCost
    (conformerLowerBound rigidScore improvementBound) τ prefilterCost hSafe postFilterCost hNonneg

end BlindConformerPipelineOptimality
end Tractability
end DecisionQuotient
