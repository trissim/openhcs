/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ScreenedCoulombApproximation.lean

  Finite-domain exact/coarse approximation for screened Coulomb-style scoring.
-/
import DecisionQuotient.Tractability.CoulombApproximation
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace ScreenedCoulombApproximation

open CoulombApproximation
open Ewald
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Classical

universe u v

/-- Exact single-pair screened Coulomb score with inverse screening length `κ`. -/
noncomputable def exactScreenedCoulombScore (q_i q_j κ r : ℝ) : ℝ :=
  coulombPotential q_i q_j r * Real.exp (-κ * r)

/-- Hard-cutoff screened Coulomb score. -/
noncomputable def cutoffScreenedCoulombScore (q_i q_j κ rc r : ℝ) : ℝ :=
  if r < rc then exactScreenedCoulombScore q_i q_j κ r else 0

noncomputable def exactScreenedCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j κ : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactScreenedCoulombScore q_i q_j κ (distance a s)

noncomputable def cutoffScreenedCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffScreenedCoulombScore q_i q_j κ rc (distance a s)

noncomputable def screenedCoulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p =>
        |exactScreenedCoulombScore q_i q_j κ (distance p.1 p.2) -
          cutoffScreenedCoulombScore q_i q_j κ rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactScreenedCoulombScore q_i q_j κ (distance a s) -
      cutoffScreenedCoulombScore q_i q_j κ rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem screenedCoulombCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactScreenedCoulombScore q_i q_j κ (distance a s) -
      cutoffScreenedCoulombScore q_i q_j κ rc (distance a s)| ≤
      screenedCoulombCutoffErrorRadius q_i q_j κ rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p =>
        |exactScreenedCoulombScore q_i q_j κ (distance p.1 p.2) -
          cutoffScreenedCoulombScore q_i q_j κ rc (distance p.1 p.2)|)
  have hMem :
      |exactScreenedCoulombScore q_i q_j κ (distance a s) -
        cutoffScreenedCoulombScore q_i q_j κ rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold screenedCoulombCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_screened_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactScreenedCoulombDecisionProblem q_i q_j κ distance)
      (cutoffScreenedCoulombDecisionProblem q_i q_j κ rc distance)
      (screenedCoulombCutoffErrorRadius q_i q_j κ rc distance) := by
  intro a s
  simpa [exactScreenedCoulombDecisionProblem, cutoffScreenedCoulombDecisionProblem] using
    screenedCoulombCutoffErrorRadius_spec q_i q_j κ rc distance a s

theorem screenedCoulombCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) :
    0 ≤ screenedCoulombCutoffErrorRadius q_i q_j κ rc distance := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _)
    (screenedCoulombCutoffErrorRadius_spec q_i q_j κ rc distance a s)

/-- Exact-vs-cutoff screened Coulomb induces a theorem-backed certified top-1 survivor set. -/
noncomputable def exact_vs_cutoff_screened_coulomb_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactScreenedCoulombDecisionProblem q_i q_j κ distance |>.utility a s)
    (fun a => cutoffScreenedCoulombDecisionProblem q_i q_j κ rc distance |>.utility a s)
    (screenedCoulombCutoffErrorRadius q_i q_j κ rc distance)
    (fun a => exact_vs_cutoff_screened_coulomb_uniformApprox q_i q_j κ rc distance a s)
    (screenedCoulombCutoffErrorRadius_nonneg q_i q_j κ rc distance)

/-- Soundness of the exact-vs-cutoff screened Coulomb certified top-1 survivor set. -/
theorem exact_vs_cutoff_screened_coulomb_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactScreenedCoulombDecisionProblem q_i q_j κ distance |>.utility a s)
      (fun a => cutoffScreenedCoulombDecisionProblem q_i q_j κ rc distance |>.utility a s)
      (screenedCoulombCutoffErrorRadius q_i q_j κ rc distance)
      (fun a => exact_vs_cutoff_screened_coulomb_uniformApprox q_i q_j κ rc distance a s)
      (screenedCoulombCutoffErrorRadius_nonneg q_i q_j κ rc distance)).exactTopK
      ⊆ (exact_vs_cutoff_screened_coulomb_certified_top1 q_i q_j κ rc distance s).survivors := by
  simpa [exact_vs_cutoff_screened_coulomb_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactScreenedCoulombDecisionProblem q_i q_j κ distance |>.utility a s)
      (fun a => cutoffScreenedCoulombDecisionProblem q_i q_j κ rc distance |>.utility a s)
      (screenedCoulombCutoffErrorRadius q_i q_j κ rc distance)
      (fun a => exact_vs_cutoff_screened_coulomb_uniformApprox q_i q_j κ rc distance a s)
      (screenedCoulombCutoffErrorRadius_nonneg q_i q_j κ rc distance)

/-- Exact-vs-cutoff screened Coulomb yields a runtime-facing optimizer witness. -/
noncomputable def exact_vs_cutoff_screened_coulomb_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactScreenedCoulombDecisionProblem q_i q_j κ distance |>.utility a s)
    (fun a => cutoffScreenedCoulombDecisionProblem q_i q_j κ rc distance |>.utility a s)
    (screenedCoulombCutoffErrorRadius q_i q_j κ rc distance)
    (fun a => exact_vs_cutoff_screened_coulomb_uniformApprox q_i q_j κ rc distance a s)
    (screenedCoulombCutoffErrorRadius_nonneg q_i q_j κ rc distance)

noncomputable def exact_vs_cutoff_screened_coulomb_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_screened_coulomb_coherent_optimizer_witness q_i q_j κ rc distance s).toOptimizerWitness

end ScreenedCoulombApproximation
end Tractability
end DecisionQuotient
