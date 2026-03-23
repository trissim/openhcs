/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/DirectionalMetalCoordinationApproximation.lean

  Geometry-specific metal coordination with angular terms.

  Physics: Metal coordination geometry depends on both distance AND angle.
  Zinc prefers tetrahedral (109.5°), iron prefers octahedral (90°/180°).
  The purely radial model in MetalCoordinationApproximation.lean ignores
  this angular dependence.

  This file extends the radial model with a normalized angular factor:
    directionalMetalScore = strength · radial(r) · geometry(θ)
  where:
    - strength ∈ ℝ (coupling weight)
    - radial ∈ [0, 1] (Gaussian distance decay, normalized)
    - geometry ∈ [0, 1] (angular preference factor)

  Key results:

  1. `directionalMetalScore_sub_le_component_sum`
     Two-factor Lipschitz bound: |f₁g₁ - f₂g₂| ≤ Lf·err + Lg·err
     when factors are in [0, 1] and each is Lipschitz.

  2. `angular_factor_tightens_tail`
     The angular factor ∈ [0, 1] can only reduce the tail error, so the
     radial-only cutoff bound remains valid for the directional score.

  3. `directional_metal_cutoff_uniformApprox`
     Finite-domain uniform approximation for the directional model with
     hard distance cutoff.

  4. Full certified top-1 survivor set and optimizer witness chain.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.SignInvariance
import DecisionQuotient.Tractability.MetalCoordinationApproximation

namespace DecisionQuotient
namespace Tractability
namespace DirectionalMetalCoordinationApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open SignInvariance
open Classical

universe u v

-- ---------------------------------------------------------------------------
-- Section 1: Directional metal coordination score
-- ---------------------------------------------------------------------------

/-- Directional metal coordination score: strength × radial × geometry.
    Both radial and geometry are assumed to be normalized to [0, 1]. -/
noncomputable def directionalMetalScore (radial geometry : ℝ) : ℝ :=
  radial * geometry

/-- Pointwise unit-interval constraint for [0, 1]-valued factors. -/
def MetalUnitIntervalFactor {A : Type u} {S : Type v} (f : A → S → ℝ) : Prop :=
  ∀ a s, 0 ≤ f a s ∧ f a s ≤ 1

-- ---------------------------------------------------------------------------
-- Section 2: Two-factor Lipschitz bound
-- ---------------------------------------------------------------------------

/-- Two-factor product Lipschitz bound.
    If f, g are in [0, 1] and each factor has approximation error bounded by
    Lf·err and Lg·err respectively, then |f₁g₁ - f₂g₂| ≤ (Lf + Lg)·err.

    Proof via telescope: f₁g₁ - f₂g₂ = (f₁-f₂)g₁ + f₂(g₁-g₂). -/
theorem directionalMetalScore_sub_le_component_sum
    {f1 f2 g1 g2 Lf Lg err : ℝ}
    (hf1 : 0 ≤ f1) (hf1b : f1 ≤ 1)
    (hf2 : 0 ≤ f2) (hf2b : f2 ≤ 1)
    (hg1 : 0 ≤ g1) (hg1b : g1 ≤ 1)
    (hg2 : 0 ≤ g2) (hg2b : g2 ≤ 1)
    (hLf : 0 ≤ Lf) (hLg : 0 ≤ Lg)
    (herr : 0 ≤ err)
    (hRadial : |f1 - f2| ≤ Lf * err)
    (hGeometry : |g1 - g2| ≤ Lg * err) :
    |directionalMetalScore f1 g1 - directionalMetalScore f2 g2| ≤
      (Lf + Lg) * err := by
  unfold directionalMetalScore
  -- Telescope: f₁g₁ - f₂g₂ = (f₁-f₂)·g₁ + f₂·(g₁-g₂)
  have hTelescope : f1 * g1 - f2 * g2 = (f1 - f2) * g1 + f2 * (g1 - g2) := by ring
  rw [hTelescope]
  calc |((f1 - f2) * g1 + f2 * (g1 - g2))|
      ≤ |(f1 - f2) * g1| + |f2 * (g1 - g2)| := abs_add_le _ _
    _ = |f1 - f2| * |g1| + |f2| * |g1 - g2| := by
        rw [abs_mul, abs_mul]
    _ ≤ |f1 - f2| * 1 + 1 * |g1 - g2| := by
        gcongr
        · rwa [abs_of_nonneg hg1]
        · rwa [abs_of_nonneg hf2]
    _ = |f1 - f2| + |g1 - g2| := by ring
    _ ≤ Lf * err + Lg * err := by linarith [hRadial, hGeometry]
    _ = (Lf + Lg) * err := by ring

-- ---------------------------------------------------------------------------
-- Section 3: Angular factor tightens the tail bound
-- ---------------------------------------------------------------------------

/-- The angular factor ∈ [0, 1] can only reduce the tail beyond the cutoff.
    If |w · radial(r)| ≤ B for r ≥ rc, then |w · radial(r) · angular(θ)| ≤ B.

    This means the radial-only cutoff error bound from
    MetalCoordinationApproximation.lean remains valid after adding angular
    terms. The geometry factor is "free" — it refines the score without
    increasing the certified error. -/
theorem angular_factor_tightens_tail
    (w radial angular : ℝ)
    (h_angular_nonneg : 0 ≤ angular)
    (h_angular_le_one : angular ≤ 1)
    (h_radial_bound : |w * radial| ≤ B) :
    |w * radial * angular| ≤ B := by
  rw [abs_mul]
  have h_abs_angular : |angular| = angular := abs_of_nonneg h_angular_nonneg
  rw [h_abs_angular]
  calc |w * radial| * angular ≤ |w * radial| * 1 := by
        gcongr
    _ = |w * radial| := mul_one _
    _ ≤ B := h_radial_bound

-- ---------------------------------------------------------------------------
-- Section 4: Finite-domain exact/coarse pair
-- ---------------------------------------------------------------------------

/-- Exact directional metal coordination: distance × angular factors. -/
noncomputable def exactDirectionalMetalScore (w ideal width r geometry : ℝ) : ℝ :=
  MetalCoordinationApproximation.exactMetalCoordinationScore w ideal width r * geometry

/-- Cutoff directional metal coordination: zero beyond cutoff radius. -/
noncomputable def cutoffDirectionalMetalScore (w ideal width rc r geometry : ℝ) : ℝ :=
  if r < rc then exactDirectionalMetalScore w ideal width r geometry else 0

noncomputable def exactDirectionalMetalDecisionProblem {A : Type u} {S : Type v}
    (w ideal width : ℝ) (distance geometry : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactDirectionalMetalScore w ideal width (distance a s) (geometry a s)

noncomputable def cutoffDirectionalMetalDecisionProblem {A : Type u} {S : Type v}
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffDirectionalMetalScore w ideal width rc (distance a s) (geometry a s)

noncomputable def directionalMetalCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactDirectionalMetalScore w ideal width (distance p.1 p.2) (geometry p.1 p.2)
        - cutoffDirectionalMetalScore w ideal width rc (distance p.1 p.2) (geometry p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    exact ⟨_, Finset.mem_image.mpr ⟨(a, s), by simp, rfl⟩⟩

theorem directionalMetalCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ)
    (a : A) (s : S) :
    |exactDirectionalMetalScore w ideal width (distance a s) (geometry a s)
      - cutoffDirectionalMetalScore w ideal width rc (distance a s) (geometry a s)| ≤
      directionalMetalCutoffErrorRadius w ideal width rc distance geometry := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactDirectionalMetalScore w ideal width (distance p.1 p.2) (geometry p.1 p.2)
        - cutoffDirectionalMetalScore w ideal width rc (distance p.1 p.2) (geometry p.1 p.2)|)
  have hMem : |exactDirectionalMetalScore w ideal width (distance a s) (geometry a s)
      - cutoffDirectionalMetalScore w ideal width rc (distance a s) (geometry a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold directionalMetalCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem directional_metal_cutoff_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) :
    UniformUtilityApprox
      (exactDirectionalMetalDecisionProblem w ideal width distance geometry)
      (cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry)
      (directionalMetalCutoffErrorRadius w ideal width rc distance geometry) := by
  intro a s
  simpa [exactDirectionalMetalDecisionProblem, cutoffDirectionalMetalDecisionProblem] using
    directionalMetalCutoffErrorRadius_spec w ideal width rc distance geometry a s

theorem directionalMetalCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) :
    0 ≤ directionalMetalCutoffErrorRadius w ideal width rc distance geometry := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _)
    (directionalMetalCutoffErrorRadius_spec w ideal width rc distance geometry a s)

-- ---------------------------------------------------------------------------
-- Section 5: Certified top-1 survivor set
-- ---------------------------------------------------------------------------

noncomputable def directional_metal_cutoff_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactDirectionalMetalDecisionProblem w ideal width distance geometry |>.utility a s)
    (fun a => cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry |>.utility a s)
    (directionalMetalCutoffErrorRadius w ideal width rc distance geometry)
    (fun a => directional_metal_cutoff_uniformApprox w ideal width rc distance geometry a s)
    (directionalMetalCutoffErrorRadius_nonneg w ideal width rc distance geometry)

theorem directional_metal_cutoff_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactDirectionalMetalDecisionProblem w ideal width distance geometry |>.utility a s)
      (fun a => cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry |>.utility a s)
      (directionalMetalCutoffErrorRadius w ideal width rc distance geometry)
      (fun a => directional_metal_cutoff_uniformApprox w ideal width rc distance geometry a s)
      (directionalMetalCutoffErrorRadius_nonneg w ideal width rc distance geometry)).exactTopK
      ⊆ (directional_metal_cutoff_certified_top1 w ideal width rc distance geometry s).survivors := by
  simpa [directional_metal_cutoff_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactDirectionalMetalDecisionProblem w ideal width distance geometry |>.utility a s)
      (fun a => cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry |>.utility a s)
      (directionalMetalCutoffErrorRadius w ideal width rc distance geometry)
      (fun a => directional_metal_cutoff_uniformApprox w ideal width rc distance geometry a s)
      (directionalMetalCutoffErrorRadius_nonneg w ideal width rc distance geometry)

-- ---------------------------------------------------------------------------
-- Section 6: Attractive variant via sign invariance
-- ---------------------------------------------------------------------------

noncomputable def exactAttractiveDirectionalMetalDecisionProblem {A : Type u} {S : Type v}
    (w ideal width : ℝ) (distance geometry : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <| exactDirectionalMetalDecisionProblem w ideal width distance geometry

noncomputable def cutoffAttractiveDirectionalMetalDecisionProblem {A : Type u} {S : Type v}
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <| cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry

theorem attractive_directional_metal_cutoff_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) :
    UniformUtilityApprox
      (exactAttractiveDirectionalMetalDecisionProblem w ideal width distance geometry)
      (cutoffAttractiveDirectionalMetalDecisionProblem w ideal width rc distance geometry)
      (directionalMetalCutoffErrorRadius w ideal width rc distance geometry) := by
  unfold exactAttractiveDirectionalMetalDecisionProblem cutoffAttractiveDirectionalMetalDecisionProblem
  exact neg_uniformApprox
    (exactDirectionalMetalDecisionProblem w ideal width distance geometry)
    (cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry)
    (directionalMetalCutoffErrorRadius w ideal width rc distance geometry)
    (directional_metal_cutoff_uniformApprox w ideal width rc distance geometry)

noncomputable def attractive_directional_metal_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactDirectionalMetalDecisionProblem w ideal width distance geometry |>.utility a s)
    (fun a => cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry |>.utility a s)
    (directionalMetalCutoffErrorRadius w ideal width rc distance geometry)
    (fun a => directional_metal_cutoff_uniformApprox w ideal width rc distance geometry a s)
    (directionalMetalCutoffErrorRadius_nonneg w ideal width rc distance geometry)

end DirectionalMetalCoordinationApproximation
end Tractability
end DecisionQuotient
