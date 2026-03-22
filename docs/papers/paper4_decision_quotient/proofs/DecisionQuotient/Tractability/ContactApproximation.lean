/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ContactApproximation.lean

  Finite-domain exact/coarse approximation for a bounded pairwise contact /
  desolvation surrogate.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace ContactApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Classical

universe u v

/-- Bounded Gaussian-like pairwise contact/desolvation surrogate. -/
noncomputable def exactContactScore (w β r : ℝ) : ℝ :=
  w * Real.exp (-((β * r) ^ (2 : ℕ)))

/-- Hard-cutoff coarse contact surrogate. -/
noncomputable def cutoffContactScore (w β rc r : ℝ) : ℝ :=
  if r < rc then exactContactScore w β r else 0

noncomputable def exactContactDecisionProblem {A : Type u} {S : Type v}
    (w β : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactContactScore w β (distance a s)

noncomputable def cutoffContactDecisionProblem {A : Type u} {S : Type v}
    (w β rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffContactScore w β rc (distance a s)

noncomputable def contactCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactContactScore w β (distance p.1 p.2) - cutoffContactScore w β rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactContactScore w β (distance a s) - cutoffContactScore w β rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem contactCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactContactScore w β (distance a s) - cutoffContactScore w β rc (distance a s)| ≤
      contactCutoffErrorRadius w β rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactContactScore w β (distance p.1 p.2) - cutoffContactScore w β rc (distance p.1 p.2)|)
  have hMem : |exactContactScore w β (distance a s) - cutoffContactScore w β rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold contactCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_contact_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactContactDecisionProblem w β distance)
      (cutoffContactDecisionProblem w β rc distance)
      (contactCutoffErrorRadius w β rc distance) := by
  intro a s
  simpa [exactContactDecisionProblem, cutoffContactDecisionProblem] using
    contactCutoffErrorRadius_spec w β rc distance a s

theorem contactCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) :
    0 ≤ contactCutoffErrorRadius w β rc distance := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _) (contactCutoffErrorRadius_spec w β rc distance a s)

/-- Exact-vs-cutoff contact surrogate induces a theorem-backed certified top-1 survivor set. -/
noncomputable def exact_vs_cutoff_contact_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactContactDecisionProblem w β distance |>.utility a s)
    (fun a => cutoffContactDecisionProblem w β rc distance |>.utility a s)
    (contactCutoffErrorRadius w β rc distance)
    (fun a => exact_vs_cutoff_contact_uniformApprox w β rc distance a s)
    (contactCutoffErrorRadius_nonneg w β rc distance)

/-- Soundness of the exact-vs-cutoff contact certified top-1 survivor set. -/
theorem exact_vs_cutoff_contact_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactContactDecisionProblem w β distance |>.utility a s)
      (fun a => cutoffContactDecisionProblem w β rc distance |>.utility a s)
      (contactCutoffErrorRadius w β rc distance)
      (fun a => exact_vs_cutoff_contact_uniformApprox w β rc distance a s)
      (contactCutoffErrorRadius_nonneg w β rc distance)).exactTopK
      ⊆ (exact_vs_cutoff_contact_certified_top1 w β rc distance s).survivors := by
  simpa [exact_vs_cutoff_contact_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactContactDecisionProblem w β distance |>.utility a s)
      (fun a => cutoffContactDecisionProblem w β rc distance |>.utility a s)
      (contactCutoffErrorRadius w β rc distance)
      (fun a => exact_vs_cutoff_contact_uniformApprox w β rc distance a s)
      (contactCutoffErrorRadius_nonneg w β rc distance)

/-- Exact-vs-cutoff contact surrogate yields a runtime-facing optimizer witness. -/
noncomputable def exact_vs_cutoff_contact_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactContactDecisionProblem w β distance |>.utility a s)
    (fun a => cutoffContactDecisionProblem w β rc distance |>.utility a s)
    (contactCutoffErrorRadius w β rc distance)
    (fun a => exact_vs_cutoff_contact_uniformApprox w β rc distance a s)
    (contactCutoffErrorRadius_nonneg w β rc distance)

noncomputable def exact_vs_cutoff_contact_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_contact_coherent_optimizer_witness w β rc distance s).toOptimizerWitness

end ContactApproximation
end Tractability
end DecisionQuotient
