/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SoftLJApproximation.lean

  Finite-domain exact/coarse approximation for exact LJ versus softened LJ.
-/
import DecisionQuotient.Tractability.LJApproximation
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace SoftLJApproximation

open LJApproximation
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Classical

universe u v

/-- Softened LJ score using a lower-bounded effective radius. -/
noncomputable def softenedLJScore (ε σ rSoft r : ℝ) : ℝ :=
  exactLJScore ε σ (max r rSoft)

noncomputable def softenedLJDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) : DecisionProblem A S where
  utility := fun a s => softenedLJScore ε σ rSoft (distance a s)

noncomputable def ljSofteningErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - softenedLJScore ε σ rSoft r|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactLJScore ε σ (distance a s) - softenedLJScore ε σ rSoft (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨distance a s, ?_, rfl⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem ljSofteningErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ)
    (a : A) (s : S) :
    |exactLJScore ε σ (distance a s) - softenedLJScore ε σ rSoft (distance a s)| ≤
      ljSofteningErrorRadius distance ε σ rSoft := by
  classical
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - softenedLJScore ε σ rSoft r|)
  have hDistMem : distance a s ∈ sampledDistances distance := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  have hMem : |exactLJScore ε σ (distance a s) - softenedLJScore ε σ rSoft (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨distance a s, hDistMem, rfl⟩
  unfold ljSofteningErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_softened_lj_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) :
    UniformUtilityApprox
      (exactLJDecisionProblem distance ε σ)
      (softenedLJDecisionProblem distance ε σ rSoft)
      (ljSofteningErrorRadius distance ε σ rSoft) := by
  intro a s
  simpa [exactLJDecisionProblem, softenedLJDecisionProblem] using
    ljSofteningErrorRadius_spec distance ε σ rSoft a s

theorem ljSofteningErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) :
    0 ≤ ljSofteningErrorRadius distance ε σ rSoft := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _) (ljSofteningErrorRadius_spec distance ε σ rSoft a s)

/-- Exact-vs-softened LJ induces a theorem-backed certified top-1 survivor set. -/
noncomputable def exact_vs_softened_lj_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
    (fun a => softenedLJDecisionProblem distance ε σ rSoft |>.utility a s)
    (ljSofteningErrorRadius distance ε σ rSoft)
    (fun a => exact_vs_softened_lj_uniformApprox distance ε σ rSoft a s)
    (ljSofteningErrorRadius_nonneg distance ε σ rSoft)

/-- Soundness of the exact-vs-softened LJ certified top-1 survivor set. -/
theorem exact_vs_softened_lj_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
      (fun a => softenedLJDecisionProblem distance ε σ rSoft |>.utility a s)
      (ljSofteningErrorRadius distance ε σ rSoft)
      (fun a => exact_vs_softened_lj_uniformApprox distance ε σ rSoft a s)
      (ljSofteningErrorRadius_nonneg distance ε σ rSoft)).exactTopK
      ⊆ (exact_vs_softened_lj_certified_top1 distance ε σ rSoft s).survivors := by
  simpa [exact_vs_softened_lj_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
      (fun a => softenedLJDecisionProblem distance ε σ rSoft |>.utility a s)
      (ljSofteningErrorRadius distance ε σ rSoft)
      (fun a => exact_vs_softened_lj_uniformApprox distance ε σ rSoft a s)
      (ljSofteningErrorRadius_nonneg distance ε σ rSoft)

/-- Exact-vs-softened LJ yields a runtime-facing optimizer witness. -/
noncomputable def exact_vs_softened_lj_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
    (fun a => softenedLJDecisionProblem distance ε σ rSoft |>.utility a s)
    (ljSofteningErrorRadius distance ε σ rSoft)
    (fun a => exact_vs_softened_lj_uniformApprox distance ε σ rSoft a s)
    (ljSofteningErrorRadius_nonneg distance ε σ rSoft)

noncomputable def exact_vs_softened_lj_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_softened_lj_coherent_optimizer_witness distance ε σ rSoft s).toOptimizerWitness

end SoftLJApproximation
end Tractability
end DecisionQuotient
