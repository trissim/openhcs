/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LJApproximation.lean

  First concrete exact/coarse scorer pair: exact Lennard-Jones versus cutoff
  Lennard-Jones on a finite sampled domain.
-/
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.Tractability.CoarseApproximation
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace LJApproximation

open Computation.ArrayDSL
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open Classical

universe u v

/-- Exact single-pair Lennard-Jones score. -/
noncomputable def exactLJScore (ε σ : ℝ) (r : ℝ) : ℝ :=
  lennardJones ε σ r

/-- Hard-cutoff coarse single-pair Lennard-Jones score. -/
noncomputable def cutoffLJScore (ε σ rc : ℝ) (r : ℝ) : ℝ :=
  if r < rc then lennardJones ε σ r else 0

/-- Decision problem induced by an exact LJ score over a sampled distance map. -/
noncomputable def exactLJDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ : ℝ) : DecisionProblem A S where
  utility := fun a s => exactLJScore ε σ (distance a s)

/-- Decision problem induced by a cutoff LJ score over a sampled distance map. -/
noncomputable def cutoffLJDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ rc : ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffLJScore ε σ rc (distance a s)

/-- The finite set of sampled distances realized by the action/state domain. -/
noncomputable def sampledDistances {A : Type u} {S : Type v}
    [Fintype A] [Fintype S]
    (distance : A → S → ℝ) : Finset ℝ :=
  (Finset.univ : Finset (A × S)).image (fun p => distance p.1 p.2)

/-- Max exact-vs-cutoff discrepancy over the sampled distance set. -/
noncomputable def ljCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - cutoffLJScore ε σ rc r|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨distance a s, ?_, rfl⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem ljCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (a : A) (s : S) :
    |exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)| ≤
      ljCutoffErrorRadius distance ε σ rc := by
  classical
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - cutoffLJScore ε σ rc r|)
  have hDistMem : distance a s ∈ sampledDistances distance := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  have hMem : |exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨distance a s, hDistMem, rfl⟩
  unfold ljCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

/-- Concrete uniform-approximation theorem for exact LJ versus cutoff LJ on a
    finite sampled domain. -/
theorem exact_vs_cutoff_lj_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) :
    UniformUtilityApprox
      (exactLJDecisionProblem distance ε σ)
      (cutoffLJDecisionProblem distance ε σ rc)
      (ljCutoffErrorRadius distance ε σ rc) := by
  intro a s
  simpa [exactLJDecisionProblem, cutoffLJDecisionProblem] using
    ljCutoffErrorRadius_spec distance ε σ rc a s

theorem exact_vs_cutoff_lj_opt_invariance {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (s : S) (aStar : A)
    (hStrict : StrictOpt (exactLJDecisionProblem distance ε σ) aStar s)
    (hBound :
      ljCutoffErrorRadius distance ε σ rc <
        StrictUtilityGap (exactLJDecisionProblem distance ε σ) aStar s / 2) :
    (exactLJDecisionProblem distance ε σ).Opt s =
      (cutoffLJDecisionProblem distance ε σ rc).Opt s :=
  by
    have hDelta : 0 ≤ ljCutoffErrorRadius distance ε σ rc := by
      rcases ‹Nonempty A› with ⟨a⟩
      rcases ‹Nonempty S› with ⟨s0⟩
      exact le_trans (abs_nonneg _) (ljCutoffErrorRadius_spec distance ε σ rc a s0)
    exact uniform_approx_implies_opt_invariance
      (exactLJDecisionProblem distance ε σ)
      (cutoffLJDecisionProblem distance ε σ rc)
      (ljCutoffErrorRadius distance ε σ rc)
      (exact_vs_cutoff_lj_uniformApprox distance ε σ rc)
      s aStar hDelta hStrict hBound

noncomputable def exact_vs_cutoff_lj_pruning_certificate {A : Type u}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uDistance : A → ℝ) (ε σ rc : ℝ)
    (k : Nat)
    (tau : ℝ)
    (hMargin : ∀ a,
      a ∈ topKWithTies (fun x => exactLJScore ε σ (uDistance x)) k →
      tau + ljCutoffErrorRadius (fun (a : A) (_ : Unit) => uDistance a) ε σ rc ≤ exactLJScore ε σ (uDistance a)) :
    PruningCertificate A :=
  uniform_approx_pruning_certificate
    (fun a => exactLJScore ε σ (uDistance a))
    (fun a => cutoffLJScore ε σ rc (uDistance a))
    k tau (ljCutoffErrorRadius (fun (a : A) (_ : Unit) => uDistance a) ε σ rc)
    (by
      intro a
      simpa using ljCutoffErrorRadius_spec (fun (a : A) (_ : Unit) => uDistance a) ε σ rc a ())
    hMargin

end LJApproximation
end Tractability
end DecisionQuotient
