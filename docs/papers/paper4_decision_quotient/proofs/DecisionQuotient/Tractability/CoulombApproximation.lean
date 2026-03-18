/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoulombApproximation.lean

  Finite-domain exact/coarse approximation for Coulomb-style scoring.
-/
import DecisionQuotient.Tractability.EwaldSummation
import DecisionQuotient.Tractability.CoarseApproximation
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace CoulombApproximation

open Ewald
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open Classical

universe u v

/-- Exact single-pair Coulomb score. -/
noncomputable def exactCoulombScore (q_i q_j r : ℝ) : ℝ :=
  coulombPotential q_i q_j r

/-- Hard-cutoff Coulomb score. -/
noncomputable def cutoffCoulombScore (q_i q_j rc r : ℝ) : ℝ :=
  if r < rc then coulombPotential q_i q_j r else 0

noncomputable def exactCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactCoulombScore q_i q_j (distance a s)

noncomputable def cutoffCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffCoulombScore q_i q_j rc (distance a s)

noncomputable def coulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem coulombCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ≤
      coulombCutoffErrorRadius q_i q_j rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  have hMem : |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold coulombCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactCoulombDecisionProblem q_i q_j distance)
      (cutoffCoulombDecisionProblem q_i q_j rc distance)
      (coulombCutoffErrorRadius q_i q_j rc distance) := by
  intro a s
  simpa [exactCoulombDecisionProblem, cutoffCoulombDecisionProblem] using
    coulombCutoffErrorRadius_spec q_i q_j rc distance a s

end CoulombApproximation
end Tractability
end DecisionQuotient
