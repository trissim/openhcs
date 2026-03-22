/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/MetalCoordinationApproximation.lean

  Finite-domain exact/coarse approximation for a bounded short-range
  metal coordination (e.g., zinc) surrogate.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.SignInvariance
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace MetalCoordinationApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open SignInvariance
open Classical

universe u v

/-- Bounded Gaussian-like radial metal coordination surrogate. -/
noncomputable def exactMetalCoordinationScore (w ideal width r : ℝ) : ℝ :=
  w * Real.exp (-(((r - ideal) / width) ^ (2 : ℕ)))

/-- Hard-cutoff coarse metal coordination surrogate. -/
noncomputable def cutoffMetalCoordinationScore (w ideal width rc r : ℝ) : ℝ :=
  if r < rc then exactMetalCoordinationScore w ideal width r else 0

noncomputable def exactMetalCoordinationDecisionProblem {A : Type u} {S : Type v}
    (w ideal width : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactMetalCoordinationScore w ideal width (distance a s)

noncomputable def cutoffMetalCoordinationDecisionProblem {A : Type u} {S : Type v}
    (w ideal width rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffMetalCoordinationScore w ideal width rc (distance a s)

noncomputable def metalCoordinationCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactMetalCoordinationScore w ideal width (distance p.1 p.2) - cutoffMetalCoordinationScore w ideal width rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactMetalCoordinationScore w ideal width (distance a s) - cutoffMetalCoordinationScore w ideal width rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem metalCoordinationCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactMetalCoordinationScore w ideal width (distance a s) - cutoffMetalCoordinationScore w ideal width rc (distance a s)| ≤
      metalCoordinationCutoffErrorRadius w ideal width rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactMetalCoordinationScore w ideal width (distance p.1 p.2) - cutoffMetalCoordinationScore w ideal width rc (distance p.1 p.2)|)
  have hMem : |exactMetalCoordinationScore w ideal width (distance a s) - cutoffMetalCoordinationScore w ideal width rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold metalCoordinationCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_metalCoordination_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactMetalCoordinationDecisionProblem w ideal width distance)
      (cutoffMetalCoordinationDecisionProblem w ideal width rc distance)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance) := by
  intro a s
  simpa [exactMetalCoordinationDecisionProblem, cutoffMetalCoordinationDecisionProblem] using
    metalCoordinationCutoffErrorRadius_spec w ideal width rc distance a s

theorem metalCoordinationCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) :
    0 ≤ metalCoordinationCutoffErrorRadius w ideal width rc distance := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _) (metalCoordinationCutoffErrorRadius_spec w ideal width rc distance a s)

/-- Exact-vs-cutoff metal coordination surrogate induces a theorem-backed certified top-1 survivor set. -/
noncomputable def exact_vs_cutoff_metalCoordination_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
    (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
    (metalCoordinationCutoffErrorRadius w ideal width rc distance)
    (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
    (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

/-- Soundness of the exact-vs-cutoff metal coordination certified top-1 survivor set. -/
theorem exact_vs_cutoff_metalCoordination_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
      (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance)
      (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
      (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)).exactTopK
      ⊆ (exact_vs_cutoff_metalCoordination_certified_top1 w ideal width rc distance s).survivors := by
  simpa [exact_vs_cutoff_metalCoordination_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
      (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance)
      (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
      (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

/-- Exact-vs-cutoff metal coordination surrogate yields a runtime-facing optimizer witness. -/
noncomputable def exact_vs_cutoff_metalCoordination_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
    (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
    (metalCoordinationCutoffErrorRadius w ideal width rc distance)
    (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
    (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

noncomputable def exact_vs_cutoff_metalCoordination_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_metalCoordination_coherent_optimizer_witness w ideal width rc distance s).toOptimizerWitness

/-- Attractive metal coordination energy family: the negative of the bounded metal coordination surrogate. -/
noncomputable def exactAttractiveMetalCoordinationDecisionProblem {A : Type u} {S : Type v}
    (w ideal width : ℝ) (distance : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <| exactMetalCoordinationDecisionProblem w ideal width distance

noncomputable def cutoffAttractiveMetalCoordinationDecisionProblem {A : Type u} {S : Type v}
    (w ideal width rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <| cutoffMetalCoordinationDecisionProblem w ideal width rc distance

theorem exact_vs_cutoff_attractiveMetalCoordination_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactAttractiveMetalCoordinationDecisionProblem w ideal width distance)
      (cutoffAttractiveMetalCoordinationDecisionProblem w ideal width rc distance)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance) := by
  unfold exactAttractiveMetalCoordinationDecisionProblem cutoffAttractiveMetalCoordinationDecisionProblem
  exact neg_uniformApprox
    (exactMetalCoordinationDecisionProblem w ideal width distance)
    (cutoffMetalCoordinationDecisionProblem w ideal width rc distance)
    (metalCoordinationCutoffErrorRadius w ideal width rc distance)
    (exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance)

noncomputable def exact_vs_cutoff_attractiveMetalCoordination_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_negated_uniformApprox
    (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
    (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
    (metalCoordinationCutoffErrorRadius w ideal width rc distance)
    (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
    (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

theorem exact_vs_cutoff_attractiveMetalCoordination_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (negUtility <| fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
      (negUtility <| fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance)
      (neg_utility_uniformApprox
        (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
        (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
        (metalCoordinationCutoffErrorRadius w ideal width rc distance)
        (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s))
      (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)).exactTopK
      ⊆ (exact_vs_cutoff_attractiveMetalCoordination_certified_top1 w ideal width rc distance s).survivors := by
  simpa [exact_vs_cutoff_attractiveMetalCoordination_certified_top1]
    using certified_top1_survivor_set_of_negated_uniformApprox_sound
      (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
      (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
      (metalCoordinationCutoffErrorRadius w ideal width rc distance)
      (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
      (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

noncomputable def exact_vs_cutoff_attractiveMetalCoordination_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_negated_uniformApprox_top1
    (fun a => exactMetalCoordinationDecisionProblem w ideal width distance |>.utility a s)
    (fun a => cutoffMetalCoordinationDecisionProblem w ideal width rc distance |>.utility a s)
    (metalCoordinationCutoffErrorRadius w ideal width rc distance)
    (fun a => exact_vs_cutoff_metalCoordination_uniformApprox w ideal width rc distance a s)
    (metalCoordinationCutoffErrorRadius_nonneg w ideal width rc distance)

noncomputable def exact_vs_cutoff_attractiveMetalCoordination_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_attractiveMetalCoordination_coherent_optimizer_witness w ideal width rc distance s).toOptimizerWitness

end MetalCoordinationApproximation
end Tractability
end DecisionQuotient
