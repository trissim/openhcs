/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoarseApproximation.lean

  Abstract uniform-approximation interface connecting exact/coarse scores to
  winner preservation and pruning certificates.
-/
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.SampledDockingGap

namespace DecisionQuotient
namespace Tractability
namespace CoarseApproximation

open SampledDockingGap
open CertifiedPruning
open FiniteTopK

universe u v

/-- Uniform score approximation between an exact and a coarse decision problem. -/
def UniformUtilityApprox {A : Type u} {S : Type v}
    (exactDP coarseDP : DecisionProblem A S) (delta : ℝ) : Prop :=
  ∀ a s, |exactDP.utility a s - coarseDP.utility a s| ≤ delta

/-- Uniform approximation plus a strict utility gap implies winner preservation. -/
theorem uniform_approx_implies_opt_invariance
    {A : Type u} {S : Type v} [Fintype A]
    (exactDP coarseDP : DecisionProblem A S)
    (delta : ℝ)
    (hApprox : UniformUtilityApprox exactDP coarseDP delta)
    (s : S) (aStar : A)
    (hDelta : 0 ≤ delta)
    (hStrict : StrictOpt exactDP aStar s)
    (hBound : delta < StrictUtilityGap exactDP aStar s / 2) :
    exactDP.Opt s = coarseDP.Opt s :=
  sampled_epsilon_margin_invariance exactDP coarseDP s aStar delta hDelta hStrict
    (fun a => hApprox a s) hBound

/-- Uniform approximation plus a threshold margin yields a theorem-backed
    pruning certificate for a fixed state. -/
noncomputable def uniform_approx_pruning_certificate
    {A : Type u} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a, a ∈ topKWithTies uExact k → tau + delta ≤ uExact a) :
    PruningCertificate A :=
  certificate_of_topK_margin uExact uCoarse k tau delta hApprox hMargin

end CoarseApproximation
end Tractability
end DecisionQuotient
