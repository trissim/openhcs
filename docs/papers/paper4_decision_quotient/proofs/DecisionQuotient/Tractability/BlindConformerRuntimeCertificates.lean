/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/BlindConformerRuntimeCertificates.lean

  Implementation-facing certificates that bridge the abstract blind-conformer
  pipeline optimality results to concrete runtime formulas.

  This file packages four families of results needed by the Python runtime:

  1. Uniform approximation gives concrete lower/upper bounds.
  2. Bounded omitted attractive channels can be replaced by zero in coarse
     pruning, with an additive certified error budget.
  3. Pose-specific torsion subsets induce tighter certified conformer-improvement
     bounds than the ligand-global bound.
  4. Canonical pruning plus canonical RMSD-certified iteration budgets are jointly
     optimal under the additive runtime model.
-/

import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.BlindConformerPipelineOptimality
import DecisionQuotient.Tractability.BlindConformerPipelineRefinements
import DecisionQuotient.Tractability.EnergyRMSDConvergence
import DecisionQuotient.Computation.ArrayDSL
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace BlindConformerRuntimeCertificates

open CoarseApproximation
open BlindConformerPipelineOptimality
open BlindConformerPipelineRefinements
open EnergyRMSDConvergence
open Computation.ArrayDSL
open scoped BigOperators

universe u v

/-- Uniform approximation immediately gives a concrete lower bound. -/
theorem exact_ge_coarse_minus_delta
    {A : Type u} {S : Type v}
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (a : A) (s : S) :
    coarse.utility a s - δ ≤ exact.utility a s := by
  have h := hApprox a s
  linarith [abs_le.mp h]

/-- Uniform approximation immediately gives a concrete upper bound. -/
theorem exact_le_coarse_plus_delta
    {A : Type u} {S : Type v}
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (a : A) (s : S) :
    exact.utility a s ≤ coarse.utility a s + δ := by
  have h := hApprox a s
  linarith [abs_le.mp h]

/-- Decision problem for a bounded omitted channel. -/
noncomputable def omittedChannelDecisionProblem
    {A : Type u} {S : Type v}
    (channel : A → S → ℝ) : DecisionProblem A S where
  utility := channel

/-- Zero decision problem used when a bounded omitted channel is skipped in the
    coarse stage. -/
noncomputable def zeroDecisionProblem {A : Type u} {S : Type v} : DecisionProblem A S where
  utility := fun _ _ => 0

/-- A uniformly bounded omitted channel is a uniform approximation of zero with
    radius equal to its bound. -/
theorem bounded_channel_uniformApprox_zero
    {A : Type u} {S : Type v}
    (channel : A → S → ℝ)
    (B : ℝ)
    (hBound : ∀ a s, |channel a s| ≤ B) :
    UniformUtilityApprox (omittedChannelDecisionProblem channel) (zeroDecisionProblem) B := by
  intro a s
  simpa [omittedChannelDecisionProblem, zeroDecisionProblem] using hBound a s

/-- Sum of bounded omitted channels. -/
noncomputable def omittedChannelSumDecisionProblem
    {I : Type*} [Fintype I]
    {A : Type u} {S : Type v}
    (channels : I → A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => ∑ i, channels i a s

/-- The sum of bounded omitted channels is a uniform approximation of zero with
    radius equal to the sum of the per-channel bounds. -/
theorem bounded_channel_sum_uniformApprox_zero
    {I : Type*} [Fintype I]
    {A : Type u} {S : Type v}
    (channels : I → A → S → ℝ)
    (bounds : I → ℝ)
    (hBound : ∀ i a s, |channels i a s| ≤ bounds i)
    (hNonneg : ∀ i, 0 ≤ bounds i) :
    UniformUtilityApprox
      (omittedChannelSumDecisionProblem channels)
      (zeroDecisionProblem)
      (∑ i, bounds i) := by
  intro a s
  show |(∑ i, channels i a s) - 0| ≤ ∑ i, bounds i
  rw [sub_zero]
  calc
    |∑ i, channels i a s| ≤ ∑ i, |channels i a s| := by
      simpa using (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset I)) (f := fun i => channels i a s))
    _ ≤ ∑ i, bounds i := by
      refine Finset.sum_le_sum ?_
      intro i _
      exact hBound i a s

/-- Base exact/coarse approximation composes with omitted bounded channels. -/
theorem base_plus_omitted_uniformApprox
    {I : Type*} [Fintype I]
    {A : Type u} {S : Type v}
    (exactBase coarseBase : DecisionProblem A S)
    (channels : I → A → S → ℝ)
    (δ : ℝ)
    (bounds : I → ℝ)
    (hApprox : UniformUtilityApprox exactBase coarseBase δ)
    (hBound : ∀ i a s, |channels i a s| ≤ bounds i)
    (hNonneg : ∀ i, 0 ≤ bounds i) :
    UniformUtilityApprox
      (sumDecisionProblems exactBase (omittedChannelSumDecisionProblem channels))
      coarseBase
      (δ + ∑ i, bounds i) := by
  have hOmitted :
      UniformUtilityApprox
        (omittedChannelSumDecisionProblem channels)
        (zeroDecisionProblem)
        (∑ i, bounds i) :=
    bounded_channel_sum_uniformApprox_zero channels bounds hBound hNonneg
  have hSum :=
    sum_uniformApprox exactBase coarseBase
      (omittedChannelSumDecisionProblem channels)
      (zeroDecisionProblem)
      δ (∑ i, bounds i) hApprox hOmitted
  simpa [sumDecisionProblems, zeroDecisionProblem] using hSum

/-- Runtime-facing lower bound when coarse pruning omits bounded attractive terms. -/
theorem exact_with_omitted_ge_coarse_minus_totalError
    {I : Type*} [Fintype I]
    {A : Type u} {S : Type v}
    (exactBase coarseBase : DecisionProblem A S)
    (channels : I → A → S → ℝ)
    (δ : ℝ)
    (bounds : I → ℝ)
    (hApprox : UniformUtilityApprox exactBase coarseBase δ)
    (hBound : ∀ i a s, |channels i a s| ≤ bounds i)
    (hNonneg : ∀ i, 0 ≤ bounds i)
    (a : A) (s : S) :
    coarseBase.utility a s - (δ + ∑ i, bounds i)
      ≤ (sumDecisionProblems exactBase (omittedChannelSumDecisionProblem channels)).utility a s := by
  exact exact_ge_coarse_minus_delta
    (sumDecisionProblems exactBase (omittedChannelSumDecisionProblem channels))
    coarseBase
    (δ + ∑ i, bounds i)
    (base_plus_omitted_uniformApprox exactBase coarseBase channels δ bounds hApprox hBound hNonneg)
    a s

/-- Pose-specific improvement bound from a subset of active torsions. -/
noncomputable def torsionImprovementBudget
    {I : Type*} [Fintype I] [DecidableEq I]
    (localBounds : I → ℝ)
    (active : Finset I) : ℝ :=
  Finset.sum active localBounds

/-- Tighter active-torsion subsets give no larger improvement budgets. -/
theorem torsionImprovementBudget_mono
    {I : Type*} [Fintype I] [DecidableEq I]
    (localBounds : I → ℝ)
    (hNonneg : ∀ i, 0 ≤ localBounds i)
    {active₁ active₂ : Finset I}
    (hSubset : active₁ ⊆ active₂) :
    torsionImprovementBudget localBounds active₁ ≤
      torsionImprovementBudget localBounds active₂ := by
  unfold torsionImprovementBudget
  exact BlindConformerPipelineOptimality.retainedCost_mono hSubset localBounds hNonneg

/-- If a pose-specific conformer improvement is bounded by the active torsion set,
    then it is also bounded by every certified superset. -/
theorem pose_specific_improvement_bound_of_active_subset
    {I : Type*} [Fintype I] [DecidableEq I]
    {improvement : ℝ}
    (localBounds : I → ℝ)
    (hNonneg : ∀ i, 0 ≤ localBounds i)
    {active certified : Finset I}
    (hSubset : active ⊆ certified)
    (hImprove : improvement ≤ torsionImprovementBudget localBounds active) :
    improvement ≤ torsionImprovementBudget localBounds certified := by
  exact le_trans hImprove (torsionImprovementBudget_mono localBounds hNonneg hSubset)

/-- Total runtime model: pruning/orchestration cost plus certified refinement cost. -/
noncomputable def endToEndBlindConformerCost
    {P : Type*} [Fintype P] [DecidableEq P]
    (prefilterCost : ℝ)
    (retain : Finset P)
    (postFilterCost : P → ℝ)
    (setupCost stepCost : ℝ)
    (t : ℕ) : ℝ :=
  pipelineCost prefilterCost retain postFilterCost + refinementCost setupCost stepCost t

/-- Canonical pruning plus canonical RMSD-certified refinement budget is jointly
    optimal under the additive runtime model. -/
theorem canonical_pruning_and_budget_optimal
    {P : Type*} [Fintype P] [DecidableEq P]
    {n : ℕ}
    (lowerBound : P → ℝ)
    (τ prefilterCost : ℝ)
    {retain : Finset P}
    (hSafe : CertifiedSafeForThreshold lowerBound retain τ)
    (postFilterCost : P → ℝ)
    (hPostNonneg : ∀ p, 0 ≤ postFilterCost p)
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (setupCost stepCost : ℝ)
    (hstep : 0 ≤ stepCost)
    {t : ℕ}
    (hAdeqT : AdequateIterationBudget energy center poseAt basin conv eps t) :
    endToEndBlindConformerCost prefilterCost (canonicalRetain lowerBound τ) postFilterCost
      setupCost stepCost
      (canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps)
      ≤
    endToEndBlindConformerCost prefilterCost retain postFilterCost setupCost stepCost t := by
  unfold endToEndBlindConformerCost
  have hPrune := canonicalRetain_minimizes_pipelineCost lowerBound τ prefilterCost hSafe postFilterCost hPostNonneg
  have hBudget := canonicalAdequateIterationBudget_optimal energy center poseAt basin conv hn eps heps setupCost stepCost hstep hAdeqT
  exact add_le_add hPrune hBudget

/-- The canonical refinement budget immediately certifies the requested RMSD target. -/
theorem rmsd_target_of_canonical_runtime_budget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    EnergyRMSDConvergence.rmsd
      (poseAt (canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps))
      center ≤ eps :=
  rmsd_target_of_canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps

/-- If a channel is omitted (coarse = 0), the uniform approximation bound B
    must be the supremum of the channel's absolute VALUE across all (a, s).

    This is NOT the cutoff tail error — it is the maximum energy the channel
    can ever produce. If you truly omit a channel, B must bound |channel(a,s)|,
    not some approximation error at a cutoff radius.

    Numerically: for screened Coulomb, this would be the max electrostatic
    energy at minimum distance — potentially enormous. So channels with large
    peak values should NOT be omitted; they should use cutoff approximations
    (which have small tail errors) instead. -/
theorem omitted_channel_is_bounded_by_supremum
    {A : Type u} {S : Type v}
    (channel : A → S → ℝ)
    (B : ℝ)
    (hBound : ∀ a s, |channel a s| ≤ B) :
    UniformUtilityApprox
      (omittedChannelDecisionProblem channel)
      zeroDecisionProblem
      B := by
  intro a s
  simpa [omittedChannelDecisionProblem, zeroDecisionProblem] using hBound a s

end BlindConformerRuntimeCertificates
end Tractability
end DecisionQuotient
