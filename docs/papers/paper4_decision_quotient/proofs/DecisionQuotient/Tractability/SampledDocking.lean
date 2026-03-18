/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SampledDocking.lean

  Finite sampled docking wrappers built on discretized states and actions.
  This file turns "sample poses then rank them" into an explicit finite
  decision-problem object.
-/
import DecisionQuotient.Basic
import DecisionQuotient.Tractability.DiscretizedAction
import DecisionQuotient.Tractability.GridMDInstances

namespace DecisionQuotient
namespace Tractability
namespace SampledDocking

open DiscretizedAction
open DiscretizedState

universe u v

/-- A finite support family of sampled actions. -/
structure SampledActionFamily (A : Type u) [DecidableEq A] where
  support : Finset A
  nonempty : support.Nonempty

/-- The subtype of actions retained by a sampled action family. -/
abbrev SupportedAction {A : Type u} [DecidableEq A]
    (F : SampledActionFamily A) : Type u := { a : A // a ∈ F.support }

/-- Restrict a decision problem to a finite sampled action support. -/
def restrictedDecisionProblem {A : Type u} {S : Type v} [DecidableEq A]
    (dp : DecisionProblem A S) (F : SampledActionFamily A) :
    DecisionProblem (SupportedAction F) S where
  utility := fun a s => dp.utility a.1 s

/-- Any globally optimal action that lies in the sampled support remains optimal
    in the restricted sampled problem. -/
theorem supported_action_opt_of_ambient_opt
    {A : Type u} {S : Type v} [DecidableEq A]
    (dp : DecisionProblem A S) (F : SampledActionFamily A) (s : S)
    (a : SupportedAction F)
    (hOpt : a.1 ∈ dp.Opt s) :
    a ∈ (restrictedDecisionProblem dp F).Opt s := by
  intro a'
  exact hOpt a'.1

/-- If the sampled support contains at least one globally optimal action, then
    the restricted optimum equals the ambient optimum intersected with the
    sampled support. This is the precise statement needed for sampled docking:
    restricting to the sampled domain is exact provided the sample set captures
    at least one global optimum. -/
theorem restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt
    {A : Type u} {S : Type v} [DecidableEq A]
    (dp : DecisionProblem A S) (F : SampledActionFamily A) (s : S)
    (hWitness : ∃ a : SupportedAction F, a.1 ∈ dp.Opt s) :
    (restrictedDecisionProblem dp F).Opt s =
      { a : SupportedAction F | a.1 ∈ dp.Opt s } := by
  ext a
  constructor
  · intro hRestricted
    rcases hWitness with ⟨aStar, hStarOpt⟩
    intro a'
    calc
      dp.utility a' s ≤ dp.utility aStar.1 s := hStarOpt a'
      _ ≤ dp.utility a.1 s := hRestricted aStar
  · intro hAmbientSlice
    exact supported_action_opt_of_ambient_opt dp F s a hAmbientSlice

/-- Finite sampled docking package with exact and coarse utility families over
    the same sampled action support and discretized state space. -/
structure SampledDockingProblem (NP NL N : Nat) where
  samples : SampledActionFamily (GridMDAction NL N)
  exactUtility : GridMDAction NL N → GridMDState NP NL N → ℝ
  coarseUtility : GridMDAction NL N → GridMDState NP NL N → ℝ

/-- Ambient exact sampled docking decision problem. -/
def SampledDockingProblem.exactAmbientDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (GridMDAction NL N) (GridMDState NP NL N) where
  utility := prob.exactUtility

/-- Ambient coarse sampled docking decision problem. -/
def SampledDockingProblem.coarseAmbientDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (GridMDAction NL N) (GridMDState NP NL N) where
  utility := prob.coarseUtility

/-- Exact decision problem restricted to the retained sampled actions. -/
def SampledDockingProblem.exactDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (SupportedAction prob.samples) (GridMDState NP NL N) :=
  restrictedDecisionProblem prob.exactAmbientDecisionProblem prob.samples

/-- Coarse decision problem restricted to the retained sampled actions. -/
def SampledDockingProblem.coarseDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (SupportedAction prob.samples) (GridMDState NP NL N) :=
  restrictedDecisionProblem prob.coarseAmbientDecisionProblem prob.samples

end SampledDocking
end Tractability
end DecisionQuotient
