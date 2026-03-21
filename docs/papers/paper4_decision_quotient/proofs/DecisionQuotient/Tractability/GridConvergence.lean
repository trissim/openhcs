/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/GridConvergence.lean

  Abstract continuous-to-discrete convergence bridge. This file does not yet
  prove that a specific docking score is Lipschitz; instead it isolates the
  exact theorem shape needed to turn a resolution bound into a
  `UniformUtilityApprox` theorem.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.FormalLocalOptimizer
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace GridConvergence

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer

universe u v

/-- Pointwise utility perturbation controlled by a resolution-dependent error
    envelope. -/
def ResolutionControlledApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ) : Prop :=
  ∀ a sGrid, |uCont a (lift sGrid) - uGrid a sGrid| ≤ eps res

/-- Turning a resolution-controlled approximation into a `UniformUtilityApprox`
    statement on the lifted grid state space. -/
theorem resolutionControlledApprox_implies_uniformApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res) :
    UniformUtilityApprox
      { utility := fun a sGrid => uCont a (lift sGrid) }
      { utility := uGrid }
      (eps res) := by
  intro a sGrid
  exact hApprox a sGrid

/-- A standard Lipschitz-style hypothesis for utility transport from continuous
    states to their grid representatives. -/
def LipschitzUtilityApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (L : ℝ) : Prop :=
  ∀ a sGrid, |uCont a (lift sGrid) - uGrid a sGrid| ≤ L * stateError sGrid

/-- If the state discretization error is uniformly bounded by `res`, then a
    Lipschitz utility bound yields a uniform approximation radius `L * res`. -/
theorem lipschitzUtilityApprox_implies_resolutionControlled
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (L res : ℝ)
    (hLip : LipschitzUtilityApprox uCont uGrid lift stateError L)
    (hState : ∀ sGrid, stateError sGrid ≤ res)
    (hL : 0 ≤ L) :
    ResolutionControlledApprox uCont uGrid lift (fun r => L * r) res := by
  intro a sGrid
  have h1 := hLip a sGrid
  have h2 : L * stateError sGrid ≤ L * res := by
    exact mul_le_mul_of_nonneg_left (hState sGrid) hL
  exact h1.trans h2

/--
  Resolution-controlled approximation yields a theorem-backed certified top-1
  survivor set on the grid state space.
-/
noncomputable def resolutionControlledApprox_certified_top1
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res)
    (hEps : 0 ≤ eps res) :
    CertifiedSurvivorSet A :=
  certifiedSurvivorSet_of_top1_coarse_ambiguityBand
    (fun a => uCont a (lift sGrid))
    (fun a => uGrid a sGrid)
    (eps res)
    (fun a => hApprox a sGrid)
    hEps

/-- Soundness of the resolution-controlled certified top-1 survivor set. -/
theorem resolutionControlledApprox_certified_top1_sound
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res)
    (hEps : 0 ≤ eps res) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => uCont a (lift sGrid))
      (fun a => uGrid a sGrid)
      (eps res)
      (fun a => hApprox a sGrid)
      hEps).exactTopK
      ⊆ (resolutionControlledApprox_certified_top1 uCont uGrid lift eps res sGrid hApprox hEps).survivors := by
  simpa [resolutionControlledApprox_certified_top1]
    using certificate_top1_coarse_ambiguityBand_sound
      (fun a => uCont a (lift sGrid))
      (fun a => uGrid a sGrid)
      (eps res)
      (fun a => hApprox a sGrid)
      hEps

/--
  Resolution-controlled approximation also yields a runtime-facing optimizer
  witness on the grid state space.
-/
noncomputable def resolutionControlledApprox_coherent_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res)
    (hEps : 0 ≤ eps res) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => uCont a (lift sGrid))
    (fun a => uGrid a sGrid)
    (eps res)
    (fun a => hApprox a sGrid)
    hEps

noncomputable def resolutionControlledApprox_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox uCont uGrid lift eps res)
    (hEps : 0 ≤ eps res) :
    OptimizerWitness A :=
  (resolutionControlledApprox_coherent_optimizer_witness uCont uGrid lift eps res sGrid hApprox hEps).toOptimizerWitness

end GridConvergence
end Tractability
end DecisionQuotient
