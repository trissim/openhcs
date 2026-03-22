/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SignInvariance.lean

  Generic sign-transform lemmas for score/utility families. These let us derive
  attractive energy families from positive compatibility surrogates while
  preserving the same uniform discrepancy radius.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.GridConvergence

namespace DecisionQuotient
namespace Tractability
namespace SignInvariance

open CoarseApproximation
open CertifiedPruning
open FormalLocalOptimizer
open GridConvergence

universe u v

/-- Pointwise utility negation. -/
def negUtility {A : Type u} (u : A → ℝ) : A → ℝ :=
  fun a => -u a

/-- Negate the utility of a decision problem. -/
def negDecisionProblem {A : Type u} {S : Type v}
    (dp : DecisionProblem A S) : DecisionProblem A S where
  utility := fun a s => -(dp.utility a s)

theorem negUtility_abs_diff_eq {A : Type u}
    (uExact uCoarse : A → ℝ) (a : A) :
    |negUtility uExact a - negUtility uCoarse a| = |uExact a - uCoarse a| := by
  have hNeg : negUtility uExact a - negUtility uCoarse a = -(uExact a - uCoarse a) := by
    ring_nf
    simp [negUtility]
  rw [hNeg, abs_neg]

theorem negDecisionProblem_abs_diff_eq {A : Type u} {S : Type v}
    (exactDP coarseDP : DecisionProblem A S) (a : A) (s : S) :
    |(negDecisionProblem exactDP).utility a s - (negDecisionProblem coarseDP).utility a s|
      = |exactDP.utility a s - coarseDP.utility a s| := by
  have hNeg :
      (negDecisionProblem exactDP).utility a s - (negDecisionProblem coarseDP).utility a s
        = -(exactDP.utility a s - coarseDP.utility a s) := by
    ring_nf
    simp [negDecisionProblem]
  rw [hNeg, abs_neg]

theorem neg_utility_uniformApprox {A : Type u}
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta) :
    ∀ a, |negUtility uExact a - negUtility uCoarse a| ≤ delta := by
  intro a
  rw [negUtility_abs_diff_eq]
  exact hApprox a

theorem neg_uniformApprox {A : Type u} {S : Type v}
    (exactDP coarseDP : DecisionProblem A S)
    (delta : ℝ)
    (hApprox : UniformUtilityApprox exactDP coarseDP delta) :
    UniformUtilityApprox (negDecisionProblem exactDP) (negDecisionProblem coarseDP) delta := by
  intro a s
  rw [negDecisionProblem_abs_diff_eq]
  exact hApprox a s

noncomputable def certified_top1_survivor_set_of_negated_uniformApprox
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (negUtility uExact)
    (negUtility uCoarse)
    delta
    (neg_utility_uniformApprox uExact uCoarse delta hApprox)
    hDelta

theorem certified_top1_survivor_set_of_negated_uniformApprox_sound
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (certificate_of_top1_coarse_ambiguityBand
      (negUtility uExact)
      (negUtility uCoarse)
      delta
      (neg_utility_uniformApprox uExact uCoarse delta hApprox)
      hDelta).exactTopK
      ⊆ (certified_top1_survivor_set_of_negated_uniformApprox uExact uCoarse delta hApprox hDelta).survivors := by
  simpa [certified_top1_survivor_set_of_negated_uniformApprox]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (negUtility uExact)
      (negUtility uCoarse)
      delta
      (neg_utility_uniformApprox uExact uCoarse delta hApprox)
      hDelta

noncomputable def coherent_optimizer_witness_of_negated_uniformApprox_top1
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (negUtility uExact)
    (negUtility uCoarse)
    delta
    (neg_utility_uniformApprox uExact uCoarse delta hApprox)
    hDelta

noncomputable def optimizer_witness_of_negated_uniformApprox_top1
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    OptimizerWitness A :=
  (coherent_optimizer_witness_of_negated_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).toOptimizerWitness

theorem neg_resolutionControlledApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res) :
    ResolutionControlledApprox
      (fun a s => -uCont a s)
      (fun a sGrid => -uGrid a sGrid)
      lift eps res := by
  intro a sGrid
  have hDiff :
      |(fun a s => -uCont a s) a (lift sGrid) - (fun a sGrid => -uGrid a sGrid) a sGrid|
        = |uCont a (lift sGrid) - uGrid a sGrid| := by
    have hNeg : -uCont a (lift sGrid) - -uGrid a sGrid = -(uCont a (lift sGrid) - uGrid a sGrid) := by
      ring_nf
    rw [hNeg, abs_neg]
  rw [hDiff]
  exact hApprox a sGrid

theorem neg_lipschitzUtilityApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (L : ℝ)
    (hApprox : LipschitzUtilityApprox uCont uGrid lift stateError L) :
    LipschitzUtilityApprox
      (fun a s => -uCont a s)
      (fun a sGrid => -uGrid a sGrid)
      lift stateError L := by
  intro a sGrid
  have hDiff :
      |(fun a s => -uCont a s) a (lift sGrid) - (fun a sGrid => -uGrid a sGrid) a sGrid|
        = |uCont a (lift sGrid) - uGrid a sGrid| := by
    have hNeg : -uCont a (lift sGrid) - -uGrid a sGrid = -(uCont a (lift sGrid) - uGrid a sGrid) := by
      ring_nf
    rw [hNeg, abs_neg]
  rw [hDiff]
  exact hApprox a sGrid

end SignInvariance
end Tractability
end DecisionQuotient
