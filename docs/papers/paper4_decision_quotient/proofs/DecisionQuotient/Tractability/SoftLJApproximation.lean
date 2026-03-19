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

end SoftLJApproximation
end Tractability
end DecisionQuotient
