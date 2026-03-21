/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoarseApproximation.lean

  Abstract uniform-approximation interface connecting exact/coarse scores to
  winner preservation and pruning certificates.
-/
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.DiscretizedState
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.SampledDockingGap

namespace DecisionQuotient
namespace Tractability
namespace CoarseApproximation

open SampledDockingGap
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
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

/-- Pointwise utility sum of two decision problems on the same action/state space. -/
def sumDecisionProblems {A : Type u} {S : Type v}
    (dp1 dp2 : DecisionProblem A S) : DecisionProblem A S where
  utility := fun a s => dp1.utility a s + dp2.utility a s

/-- Uniform approximations compose under pointwise score addition. -/
theorem sum_uniformApprox
    {A : Type u} {S : Type v}
    (exact1 coarse1 exact2 coarse2 : DecisionProblem A S)
    (delta1 delta2 : ℝ)
    (h1 : UniformUtilityApprox exact1 coarse1 delta1)
    (h2 : UniformUtilityApprox exact2 coarse2 delta2) :
    UniformUtilityApprox
      (sumDecisionProblems exact1 exact2)
      (sumDecisionProblems coarse1 coarse2)
      (delta1 + delta2) := by
  intro a s
  simp [sumDecisionProblems]
  rw [abs_le]
  have h1' := abs_le.mp (h1 a s)
  have h2' := abs_le.mp (h2 a s)
  constructor <;> linarith [h1'.1, h1'.2, h2'.1, h2'.2]

/-- If two score families both approximate a shared reference utility, then they
    approximate each other within the sum of their radii. -/
theorem shared_reference_uniformApprox_of_two_sided_bounds
    {A : Type u} {S : Type v}
    (uRef exactDP coarseDP : DecisionProblem A S)
    (deltaExact deltaCoarse : ℝ)
    (hExact : UniformUtilityApprox uRef exactDP deltaExact)
    (hCoarse : UniformUtilityApprox uRef coarseDP deltaCoarse) :
    UniformUtilityApprox exactDP coarseDP (deltaExact + deltaCoarse) := by
  intro a s
  rw [abs_le]
  have hE := abs_le.mp (hExact a s)
  have hC := abs_le.mp (hCoarse a s)
  constructor <;> linarith

/-- Object-level witness that an exact and coarse scorer share a common
    reference approximation, yielding a combined uniform discrepancy bound. -/
structure SharedReferenceApproxWitness
    {A : Type u} {S : Type v}
    (uRef exactDP coarseDP : DecisionProblem A S) where
  deltaExact : ℝ
  deltaCoarse : ℝ
  deltaCombined : ℝ
  exactApproxRef : UniformUtilityApprox uRef exactDP deltaExact
  coarseApproxRef : UniformUtilityApprox uRef coarseDP deltaCoarse
  combinedApprox : UniformUtilityApprox exactDP coarseDP deltaCombined

/-- Construct the canonical shared-reference witness with the summed radius. -/
def sharedReferenceApproxWitness
    {A : Type u} {S : Type v}
    (uRef exactDP coarseDP : DecisionProblem A S)
    (deltaExact deltaCoarse : ℝ)
    (hExact : UniformUtilityApprox uRef exactDP deltaExact)
    (hCoarse : UniformUtilityApprox uRef coarseDP deltaCoarse) :
    SharedReferenceApproxWitness uRef exactDP coarseDP :=
  { deltaExact := deltaExact
    deltaCoarse := deltaCoarse
    deltaCombined := deltaExact + deltaCoarse
    exactApproxRef := hExact
    coarseApproxRef := hCoarse
    combinedApprox :=
      shared_reference_uniformApprox_of_two_sided_bounds
        uRef exactDP coarseDP deltaExact deltaCoarse hExact hCoarse }

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

/-- Generic top-1 certified survivor set induced by a uniform approximation radius. -/
noncomputable def certified_top1_survivor_set_of_uniformApprox
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CertifiedSurvivorSet A :=
  certifiedSurvivorSet_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta

/-- Soundness of the generic top-1 certified survivor set. -/
theorem certified_top1_survivor_set_of_uniformApprox_sound
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).exactTopK ⊆
      (certified_top1_survivor_set_of_uniformApprox uExact uCoarse delta hApprox hDelta).survivors := by
  simpa [certified_top1_survivor_set_of_uniformApprox]
    using certificate_top1_coarse_ambiguityBand_sound uExact uCoarse delta hApprox hDelta

/--
  Generic optimizer witness induced by a uniform approximation radius.
  The witness uses the ambiguity-band selection branch on the coarse score.
-/
noncomputable def coherent_optimizer_witness_of_uniformApprox_top1
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CoherentOptimizerWitness A := by
  have hBand : (ambiguityBand uCoarse 1 (by omega) (2 * delta)).Nonempty := by
    rcases topKSet_nonempty uExact (k := 1) (by omega) with ⟨a0, ha0⟩
    refine ⟨a0, ?_⟩
    exact exact_top1_subset_coarse_ambiguityBand_of_uniform_error uExact uCoarse delta hApprox hDelta ha0
  exact coherentOptimizerWitness_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta hBand

/--
  Generic optimizer witness induced by a uniform approximation radius.
  The witness uses the ambiguity-band selection branch on the coarse score.
-/
noncomputable def optimizer_witness_of_uniformApprox_top1
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    OptimizerWitness A :=
  (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).toOptimizerWitness

/-- In the coherent uniform-approximation witness, every exact top-1 action survives in runtime support. -/
theorem coherent_uniformApprox_exactTop1_subset_support
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).survivorSet.certificate.exactTopK
      ⊆ (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).belief.selection.support :=
  (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).exactTopK_subset_support

/-- In the coherent uniform-approximation witness, the chosen action lies in the certified survivor set. -/
theorem coherent_uniformApprox_choice_mem_survivors
    {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).belief.selection.choice
      ∈ (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).survivorSet.survivors :=
  (coherent_optimizer_witness_of_uniformApprox_top1 uExact uCoarse delta hApprox hDelta).choice_mem_survivors

end CoarseApproximation
end Tractability
end DecisionQuotient
