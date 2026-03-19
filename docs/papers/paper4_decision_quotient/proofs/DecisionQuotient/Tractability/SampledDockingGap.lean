/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SampledDockingGap.lean

  Wrapper theorems instantiating epsilon-margin invariance for sampled docking
  exact-vs-coarse score families.
-/
import DecisionQuotient.Tractability.EpsilonUtilityGap
import DecisionQuotient.Tractability.SampledDocking

namespace DecisionQuotient
namespace Tractability
namespace SampledDockingGap

open SampledDocking
open DiscretizedState

universe u v

/-- Score-family tag used to lift exact/coarse scoring to a single decision
    problem over an expanded state space. -/
inductive ScoreMode where
  | exact
  | coarse
  deriving DecidableEq

/-- Lift exact and coarse decision problems into one problem over tagged states. -/
def liftedDecisionProblem {A : Type u} {S : Type v}
    (exactDP coarseDP : DecisionProblem A S) :
    DecisionProblem A (ScoreMode × S) where
  utility := fun a taggedState =>
    match taggedState.1 with
    | .exact => exactDP.utility a taggedState.2
    | .coarse => coarseDP.utility a taggedState.2

/-- On the exact tag, the lifted strict utility gap reduces to the original
    exact decision problem's strict utility gap. -/
theorem strictUtilityGap_lifted_exact {A : Type u} {S : Type v} [Fintype A]
    (exactDP coarseDP : DecisionProblem A S) (aStar : A) (s : S) :
    StrictUtilityGap (liftedDecisionProblem exactDP coarseDP) aStar (ScoreMode.exact, s) =
      StrictUtilityGap exactDP aStar s := by
  unfold StrictUtilityGap liftedDecisionProblem
  simp

/-- Exact-vs-coarse winner preservation for a fixed sampled state. -/
theorem sampled_epsilon_margin_invariance {A : Type u} {S : Type v} [Fintype A]
    (exactDP coarseDP : DecisionProblem A S)
    (s : S) (aStar : A)
    (delta : ℝ)
    (hDelta : 0 ≤ delta)
    (hStrict : StrictOpt exactDP aStar s)
    (hPerturb : ∀ a, |exactDP.utility a s - coarseDP.utility a s| ≤ delta)
    (hBound : delta < StrictUtilityGap exactDP aStar s / 2) :
    exactDP.Opt s = coarseDP.Opt s := by
  let liftedDP := liftedDecisionProblem exactDP coarseDP
  have hStrictLifted : StrictOpt liftedDP aStar (ScoreMode.exact, s) := by
    intro a ha
    simpa [liftedDP, liftedDecisionProblem] using hStrict a ha
  have hPerturbLifted :
      ∀ a, |liftedDP.utility a (ScoreMode.exact, s) - liftedDP.utility a (ScoreMode.coarse, s)| ≤ delta := by
    intro a
    simpa [liftedDP, liftedDecisionProblem] using hPerturb a
  have hBoundLifted : delta < StrictUtilityGap liftedDP aStar (ScoreMode.exact, s) / 2 := by
    rw [strictUtilityGap_lifted_exact exactDP coarseDP aStar s]
    exact hBound
  simpa [liftedDP, liftedDecisionProblem] using
    epsilon_margin_invariance liftedDP (ScoreMode.exact, s) (ScoreMode.coarse, s)
      aStar delta hDelta hStrictLifted hPerturbLifted hBoundLifted

/-- Sampled docking specialization of exact-vs-coarse winner preservation. -/
theorem SampledDockingProblem.exact_coarse_opt_agree_of_gap
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (s : GridMDState NP NL N)
    (aStar : SupportedAction prob.samples)
    (delta : ℝ)
    (hDelta : 0 ≤ delta)
    (hStrict : StrictOpt prob.exactDecisionProblem aStar s)
    (hPerturb : ∀ a,
      |prob.exactDecisionProblem.utility a s - prob.coarseDecisionProblem.utility a s| ≤ delta)
    (hBound : delta < StrictUtilityGap prob.exactDecisionProblem aStar s / 2) :
    prob.exactDecisionProblem.Opt s = prob.coarseDecisionProblem.Opt s :=
  sampled_epsilon_margin_invariance
    prob.exactDecisionProblem prob.coarseDecisionProblem s aStar delta
    hDelta hStrict hPerturb hBound

end SampledDockingGap
end Tractability
end DecisionQuotient
