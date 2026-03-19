/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoarseApproximation.lean

  Abstract uniform-approximation interface connecting exact/coarse scores to
  winner preservation and pruning certificates.
-/
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.DiscretizedState
import DecisionQuotient.Tractability.SampledDockingGap

namespace DecisionQuotient
namespace Tractability
namespace CoarseApproximation

open SampledDockingGap
open CertifiedPruning
open FiniteTopK
open Classical

universe u v

/-- Uniform score approximation between an exact and a coarse decision problem. -/
def UniformUtilityApprox {A : Type u} {S : Type v}
    (exactDP coarseDP : DecisionProblem A S) (delta : ℝ) : Prop :=
  ∀ a s, |exactDP.utility a s - coarseDP.utility a s| ≤ delta

/-- Exact finite-domain worst-case score discrepancy. This is not yet a physical
    bound, but it is a rigorous finite-domain radius that always witnesses a
    uniform approximation statement when both the action and state spaces are
    finite and nonempty. -/
noncomputable def scoreDiffs
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S]
    (exactDP coarseDP : DecisionProblem A S) : Finset ℝ :=
  (Finset.univ : Finset (A × S)).image
    (fun p => |exactDP.utility p.1 p.2 - coarseDP.utility p.1 p.2|)

theorem scoreDiffs_nonempty
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (exactDP coarseDP : DecisionProblem A S) :
    (scoreDiffs exactDP coarseDP).Nonempty := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  refine ⟨|exactDP.utility a s - coarseDP.utility a s|, Finset.mem_image.mpr ?_⟩
  exact ⟨(a, s), by simp, by simp⟩

noncomputable def finiteUniformErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (exactDP coarseDP : DecisionProblem A S) : ℝ :=
  (scoreDiffs exactDP coarseDP).max' (scoreDiffs_nonempty exactDP coarseDP)

theorem abs_diff_le_finiteUniformErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (exactDP coarseDP : DecisionProblem A S)
    (a : A) (s : S) :
    |exactDP.utility a s - coarseDP.utility a s| ≤
      finiteUniformErrorRadius exactDP coarseDP := by
  classical
  let diffs : Finset ℝ := scoreDiffs exactDP coarseDP
  have hmem : |exactDP.utility a s - coarseDP.utility a s| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, by simp⟩
  rw [finiteUniformErrorRadius]
  exact Finset.le_max' diffs _ hmem

theorem finiteUniformErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (exactDP coarseDP : DecisionProblem A S) :
    UniformUtilityApprox exactDP coarseDP (finiteUniformErrorRadius exactDP coarseDP) := by
  intro a s
  exact abs_diff_le_finiteUniformErrorRadius exactDP coarseDP a s

/-- Sampled docking specialization: every finite sampled docking problem admits
    a canonical exact finite-domain discrepancy radius witnessing uniform
    approximation between its exact and coarse score families. -/
theorem SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses
    {NP NL N : Nat} (prob : SampledDocking.SampledDockingProblem NP NL N) :
    UniformUtilityApprox prob.exactDecisionProblem prob.coarseDecisionProblem
      (finiteUniformErrorRadius prob.exactDecisionProblem prob.coarseDecisionProblem) := by
  exact finiteUniformErrorRadius_witnesses_uniformApprox
    prob.exactDecisionProblem prob.coarseDecisionProblem

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
