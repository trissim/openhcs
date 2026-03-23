/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/MetalCoordinationApproximation.lean

  Finite-domain exact/coarse approximation for a bounded short-range
  metal coordination (e.g., zinc) surrogate.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.GaussianDecayBounds
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

/-! ### Physically Derived Cutoff Bounds

The finite-domain error radius `metalCoordinationCutoffErrorRadius` is exact but
opaque — it tells you the worst-case error but not how it depends on the physical
parameters (w, ideal, width, rc).

The theorems below derive a **closed-form error bound** from the Gaussian decay
structure of the surrogate, connecting to `GaussianDecayBounds.lean`. This gives
a physically meaningful cutoff formula:

  rc ≥ ideal + width · √(ln(|w|/ε))

guarantees error ≤ ε for the metal coordination surrogate.
-/

open GaussianDecayBounds

/-- The metal coordination score beyond the ideal distance decays as a Gaussian
    in (r - ideal) with rate β = 1/width. For r ≥ rc ≥ ideal, the pointwise
    cutoff error |exact(r) - cutoff(r)| = |exact(r)| ≤ |w| · exp(-((rc-ideal)/width)²).

    This is the physical content: the error decays Gaussian-fast away from ideal. -/
theorem metalCoordination_tail_bound
    (w ideal width r rc : ℝ)
    (hwidth_pos : 0 < width) (hrc_ge_ideal : ideal ≤ rc) (hr_ge_rc : rc ≤ r) :
    |exactMetalCoordinationScore w ideal width r| ≤
      |w| * Real.exp (-(((rc - ideal) / width) ^ 2)) := by
  unfold exactMetalCoordinationScore
  rw [abs_mul]
  have h_exp_nonneg : 0 ≤ Real.exp (-(((r - ideal) / width) ^ (2 : ℕ))) :=
    Real.exp_pos _ |>.le
  rw [abs_of_nonneg h_exp_nonneg]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg w)
  apply Real.exp_le_exp_of_le
  -- Need: ((rc - ideal)/width)² ≤ ((r - ideal)/width)²
  have hr_ge_ideal : ideal ≤ r := le_trans hrc_ge_ideal hr_ge_rc
  have h_num_nonneg_rc : 0 ≤ rc - ideal := sub_nonneg.mpr hrc_ge_ideal
  have h_num_nonneg_r : 0 ≤ r - ideal := sub_nonneg.mpr hr_ge_ideal
  have h_denom_pos : (0 : ℝ) < width := hwidth_pos
  have h_div_nonneg_rc : 0 ≤ (rc - ideal) / width := div_nonneg h_num_nonneg_rc (le_of_lt h_denom_pos)
  have h_div_nonneg_r : 0 ≤ (r - ideal) / width := div_nonneg h_num_nonneg_r (le_of_lt h_denom_pos)
  have h_div_le : (rc - ideal) / width ≤ (r - ideal) / width := by
    apply div_le_div_of_nonneg_right _ hwidth_pos.le
    linarith
  have h_sq_le : ((rc - ideal) / width) ^ 2 ≤ ((r - ideal) / width) ^ 2 :=
    sq_le_sq' (by linarith) h_div_le
  -- The definition uses `^ (2 : ℕ)` which is definitionally `^ 2`
  linarith

/-- Physically derived uniform error bound: when the cutoff distance satisfies
    rc ≥ ideal + width · √(ln(|w|/ε)), the worst-case error is at most ε.

    This replaces the opaque finite-domain max with a constructive formula
    that the runtime can evaluate to choose optimal cutoff radii. -/
theorem metalCoordination_cutoff_sufficient
    (w ideal width rc ε : ℝ)
    (hw_pos : 0 < |w|) (hε_pos : 0 < ε) (hwidth_pos : 0 < width)
    (hrc_bound : ideal + width * Real.sqrt (Real.log (|w| / ε)) ≤ rc) :
    |w| * Real.exp (-(((rc - ideal) / width) ^ 2)) ≤ ε := by
  -- Apply gaussian_exp_bound with W = |w|, β = 1/width, R = rc - ideal
  -- Then β * R = (rc - ideal) / width, which is exactly our exponent argument.
  have hrc_shifted : width * Real.sqrt (Real.log (|w| / ε)) ≤ rc - ideal := by linarith
  have hβ_pos : (0 : ℝ) < 1 / width := by positivity
  have hR_ge : Real.sqrt (Real.log (|w| / ε)) / (1 / width) ≤ rc - ideal := by
    have h_simp : Real.sqrt (Real.log (|w| / ε)) / (1 / width) =
        width * Real.sqrt (Real.log (|w| / ε)) := by
      field_simp
    linarith [h_simp]
  have h := GaussianDecayBounds.gaussian_exp_bound |w| ε (1 / width) (rc - ideal)
    hw_pos hε_pos hβ_pos hR_ge
  -- h : |w| * exp(-((1/width) * (rc - ideal))^2) ≤ ε
  -- (1/width) * (rc - ideal) = (rc - ideal) / width
  have heq : 1 / width * (rc - ideal) = (rc - ideal) / width := by ring
  rw [heq] at h
  exact h

/-- Optimal cutoff distance for metal coordination to achieve error ε.
    rc_min = ideal + width · √(ln(|w|/ε)) -/
noncomputable def metalCoordinationMinCutoff (w ideal width ε : ℝ) : ℝ :=
  ideal + width * Real.sqrt (Real.log (|w| / ε))

theorem metalCoordinationMinCutoff_sufficient
    (w ideal width ε : ℝ)
    (hw_pos : 0 < |w|) (hε_pos : 0 < ε) (hwidth_pos : 0 < width) :
    |w| * Real.exp (-(((metalCoordinationMinCutoff w ideal width ε - ideal) / width) ^ 2)) ≤ ε := by
  exact metalCoordination_cutoff_sufficient w ideal width
    (metalCoordinationMinCutoff w ideal width ε) ε
    hw_pos hε_pos hwidth_pos (le_refl _)

end MetalCoordinationApproximation
end Tractability
end DecisionQuotient
