/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/EnergyRMSDConvergence.lean

  Implementation-facing bridge from certified energy-space bounds to RMSD-space
  guarantees for pose refinement.

  This file avoids any new axioms. Instead it isolates the exact theorem shape
  the runtime needs:

  * a quadratic-growth certificate in a certified local basin, and
  * a per-iteration energy convergence certificate for the chosen optimizer.

  From those two ingredients we derive explicit RMSD bounds and a target-energy
  threshold sufficient for a requested RMSD tolerance.
-/

import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.Computation.ArrayDSL

namespace DecisionQuotient
namespace Tractability
namespace EnergyRMSDConvergence

open scoped BigOperators
open Computation.ArrayDSL

/-- Sum of squared atomic displacements between two poses. -/
noncomputable def squaredDisplacement {n : ℕ} (x y : CoordSet n) : ℝ :=
  ∑ i, ‖x i - y i‖ ^ 2

/-- Heavy-atom RMSD over a coordinate set of size `n`. -/
noncomputable def rmsd {n : ℕ} (x y : CoordSet n) : ℝ :=
  Real.sqrt (squaredDisplacement x y / (n : ℝ))

/-- Energy gap sufficient for RMSD tolerance `eps` under quadratic growth `μ`. -/
noncomputable def targetEnergyGap (μ : ℝ) (n : ℕ) (eps : ℝ) : ℝ :=
  μ * (n : ℝ) * eps ^ 2 / 2

/-- Restrict an energy function to the line segment from `center` to `x`. -/
noncomputable def segmentEnergy {n : ℕ}
    (energy : CoordSet n → ℝ) (center x : CoordSet n) : ℝ → ℝ :=
  fun t => energy (center + t • (x - center))

/-- Certified quadratic-growth basin around `center`. -/
structure CertifiedQuadraticBasin {n : ℕ}
    (energy : CoordSet n → ℝ) (center : CoordSet n) where
  μ : ℝ
  μ_pos : 0 < μ
  quadratic_growth : ∀ x,
    (μ / 2) * squaredDisplacement x center ≤ energy x - energy center

/-- Stronger local certificate that sandwiches the energy gap between lower and
    upper quadratic models. This is the natural runtime object if Hessian-based
    curvature bounds are available on a certified basin. -/
structure CertifiedQuadraticWindow {n : ℕ}
    (energy : CoordSet n → ℝ) (center : CoordSet n) where
  μ : ℝ
  M : ℝ
  μ_pos : 0 < μ
  M_nonneg : 0 ≤ M
  lower_quadratic : ∀ x,
    (μ / 2) * squaredDisplacement x center ≤ energy x - energy center
  upper_quadratic : ∀ x,
    energy x - energy center ≤ (M / 2) * squaredDisplacement x center

/-- Certified linear energy convergence for an optimizer trajectory. -/
structure CertifiedLinearEnergyConvergence
    (energyGapAt : ℕ → ℝ) where
  q : ℝ
  q_nonneg : 0 ≤ q
  q_lt_one : q < 1
  initialGap_nonneg : 0 ≤ energyGapAt 0
  rate : ∀ t : ℕ, energyGapAt t ≤ q ^ t * energyGapAt 0

/-- One-step contraction certificate for an optimizer's energy gap dynamics. -/
structure CertifiedOneStepEnergyContraction
    (energyGapAt : ℕ → ℝ) where
  q : ℝ
  q_nonneg : 0 ≤ q
  q_lt_one : q < 1
  initialGap_nonneg : 0 ≤ energyGapAt 0
  step_contract : ∀ t : ℕ, energyGapAt (t + 1) ≤ q * energyGapAt t

/-- Certified step-size parameters from a local smooth/strongly-convex model.
    This does not itself prove gradient-descent convergence, but packages the
    algebraic constraints needed to derive a valid contraction factor once the
    runtime has certified local curvature bounds. -/
structure CertifiedGradientStepParameters where
  μ : ℝ
  M : ℝ
  α : ℝ
  μ_pos : 0 < μ
  μ_le_M : μ ≤ M
  α_pos : 0 < α
  α_le_invM : α ≤ 1 / M

/-- Local curvature certified along every line segment from `center`. This is the
    exact bridge needed to turn runtime Hessian/spectral certificates into the
    quadratic-window object consumed by the RMSD theorems. -/
structure CertifiedSegmentCurvature {n : ℕ}
    (energy : CoordSet n → ℝ) (center : CoordSet n) where
  μ : ℝ
  M : ℝ
  μ_pos : 0 < μ
  M_nonneg : 0 ≤ M
  ray_contDiff : ∀ x,
    ContDiffOn ℝ 2 (segmentEnergy energy center x) (Set.Icc 0 1)
  ray_stationary : ∀ x,
    derivWithin (segmentEnergy energy center x) (Set.Icc 0 1) 0 = 0
  ray_lower : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    μ * squaredDisplacement x center ≤ iteratedDeriv 2 (segmentEnergy energy center x) t
  ray_upper : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    iteratedDeriv 2 (segmentEnergy energy center x) t ≤ M * squaredDisplacement x center

/-- A more explicit Hessian-facing local certificate. The runtime may certify a
    scalar curvature oracle `hessAlong x t` along each segment from `center` to `x`
    and then prove it coincides with the second derivative of the restricted energy.
    This is the most implementation-facing bridge from Hessian-vector products or
    spectral enclosures to the abstract curvature machinery used below. -/
structure CertifiedRayleighHessianBounds {n : ℕ}
    (energy : CoordSet n → ℝ) (center : CoordSet n) where
  μ : ℝ
  M : ℝ
  hessAlong : CoordSet n → ℝ → ℝ
  μ_pos : 0 < μ
  M_nonneg : 0 ≤ M
  ray_contDiff : ∀ x,
    ContDiffOn ℝ 2 (segmentEnergy energy center x) (Set.Icc 0 1)
  ray_stationary : ∀ x,
    derivWithin (segmentEnergy energy center x) (Set.Icc 0 1) 0 = 0
  hessAlong_eq : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    hessAlong x t = iteratedDeriv 2 (segmentEnergy energy center x) t
  rayleigh_lower : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    μ * squaredDisplacement x center ≤ hessAlong x t
  rayleigh_upper : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    hessAlong x t ≤ M * squaredDisplacement x center

/-- A still more runtime-facing certificate: instead of talking directly about the
    Hessian along a segment, package the spectral enclosure that the runtime would
    naturally compute. The scalar `rayleighEval x t` is intended to come from a
    Hessian-vector product or enclosing eigensolver, while `lmin` and `lmax`
    represent certified lower/upper eigenvalue bounds on the local basin. -/
structure CertifiedLocalSpectralEnclosure {n : ℕ}
    (energy : CoordSet n → ℝ) (center : CoordSet n) where
  lmin : ℝ
  lmax : ℝ
  rayleighEval : CoordSet n → ℝ → ℝ
  lmin_pos : 0 < lmin
  lmax_nonneg : 0 ≤ lmax
  ray_contDiff : ∀ x,
    ContDiffOn ℝ 2 (segmentEnergy energy center x) (Set.Icc 0 1)
  ray_stationary : ∀ x,
    derivWithin (segmentEnergy energy center x) (Set.Icc 0 1) 0 = 0
  rayleigh_eq_secondDeriv : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    rayleighEval x t = iteratedDeriv 2 (segmentEnergy energy center x) t
  eigen_lower : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    lmin * squaredDisplacement x center ≤ rayleighEval x t
  eigen_upper : ∀ x t, t ∈ Set.Icc (0 : ℝ) 1 →
    rayleighEval x t ≤ lmax * squaredDisplacement x center

/-- Implementation-facing contraction factor for a linear-rate optimizer. -/
noncomputable def contractionFactor (α μ : ℝ) : ℝ :=
  1 - α * μ

noncomputable def CertifiedGradientStepParameters.q
    (params : CertifiedGradientStepParameters) : ℝ :=
  contractionFactor params.α params.μ

/-- Simple affine runtime model for local refinement: fixed setup plus per-step cost. -/
noncomputable def refinementCost (setupCost stepCost : ℝ) (t : ℕ) : ℝ :=
  setupCost + (t : ℝ) * stepCost

/-- Closed-form sufficient iteration budget obtained by solving the certified linear
    rate inequality `q^t * gap0 ≤ target` in logarithmic form. -/
noncomputable def logarithmicIterationBound (q gap0 target : ℝ) : ℕ :=
  Nat.ceil (Real.log (gap0 / target) / Real.log (1 / q))

/-- A step budget is adequate for RMSD target `eps` once its certified energy-rate
    upper bound drops below the target energy gap. -/
def AdequateIterationBudget {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (t : ℕ) : Prop :=
  conv.q ^ t * (energy (poseAt 0) - energy center) ≤ targetEnergyGap basin.μ n eps

lemma squaredDisplacement_nonneg {n : ℕ} (x y : CoordSet n) :
    0 ≤ squaredDisplacement x y := by
  unfold squaredDisplacement
  exact Finset.sum_nonneg (fun i _ => sq_nonneg ‖x i - y i‖)

lemma rmsd_nonneg {n : ℕ} (x y : CoordSet n) : 0 ≤ rmsd x y := by
  unfold rmsd
  exact Real.sqrt_nonneg _

lemma rmsd_sq_eq {n : ℕ} (x y : CoordSet n) (hn : 0 < n) :
    rmsd x y ^ 2 = squaredDisplacement x y / (n : ℝ) := by
  unfold rmsd
  rw [Real.sq_sqrt]
  refine div_nonneg (squaredDisplacement_nonneg x y) ?_
  exact_mod_cast (le_of_lt hn)

lemma squaredDisplacement_le_of_pointwiseDist_le {n : ℕ}
    (x y : CoordSet n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hPoint : ∀ i, dist (x i) (y i) ≤ eps) :
    squaredDisplacement x y ≤ (n : ℝ) * eps ^ 2 := by
  unfold squaredDisplacement
  calc
    ∑ i, ‖x i - y i‖ ^ 2
        ≤ ∑ i, eps ^ 2 := by
          apply Finset.sum_le_sum
          intro i _
          have hi : ‖x i - y i‖ ≤ eps := by
            simpa [dist_eq_norm] using hPoint i
          nlinarith [norm_nonneg (x i - y i), heps, hi]
    _ = (n : ℝ) * eps ^ 2 := by
          rw [Finset.sum_const, Finset.card_univ]
          simp [nsmul_eq_mul]

theorem rmsd_le_of_pointwiseDist_le {n : ℕ}
    (x y : CoordSet n)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hPoint : ∀ i, dist (x i) (y i) ≤ eps) :
    rmsd x y ≤ eps := by
  have hdisp : squaredDisplacement x y ≤ (n : ℝ) * eps ^ 2 :=
    squaredDisplacement_le_of_pointwiseDist_le x y eps heps hPoint
  have hdiv : squaredDisplacement x y / (n : ℝ) ≤ eps ^ 2 := by
    have hn_pos : 0 < (n : ℝ) := by exact_mod_cast hn
    rw [_root_.div_le_iff₀ hn_pos]
    simpa [mul_comm, mul_left_comm, mul_assoc] using hdisp
  have hsqrt : Real.sqrt (squaredDisplacement x y / (n : ℝ)) ≤ Real.sqrt (eps ^ 2) := by
    exact Real.sqrt_le_sqrt hdiv
  have hroot : Real.sqrt (eps ^ 2) = eps := by
    rw [Real.sqrt_sq_eq_abs]
    simpa [abs_of_nonneg heps]
  simpa [rmsd, hroot] using hsqrt

theorem rmsd_le_of_pointwiseSplitDist_le {n : ℕ}
    (x mid y : CoordSet n)
    (hn : 0 < n)
    (tBound rBound eps : ℝ)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ eps)
    (hTrans : ∀ i, dist (x i) (mid i) ≤ tBound)
    (hRot : ∀ i, dist (mid i) (y i) ≤ rBound) :
    rmsd x y ≤ eps := by
  have heps : 0 ≤ eps := le_trans (add_nonneg ht hr) hBudget
  apply le_trans (rmsd_le_of_pointwiseDist_le x y hn (tBound + rBound) (add_nonneg ht hr) ?_) hBudget
  intro i
  calc
    dist (x i) (y i) ≤ dist (x i) (mid i) + dist (mid i) (y i) := dist_triangle _ _ _
    _ ≤ tBound + rBound := add_le_add (hTrans i) (hRot i)

lemma targetEnergyGap_nonneg (μ : ℝ) (n : ℕ) (eps : ℝ)
    (hμ : 0 ≤ μ) : 0 ≤ targetEnergyGap μ n eps := by
  unfold targetEnergyGap
  positivity

lemma targetEnergyGap_mono_eps
    (μ : ℝ) (n : ℕ) {eps₁ eps₂ : ℝ}
    (hμ : 0 ≤ μ)
    (heps : 0 ≤ eps₁)
    (hmono : eps₁ ≤ eps₂) :
    targetEnergyGap μ n eps₁ ≤ targetEnergyGap μ n eps₂ := by
  unfold targetEnergyGap
  have hsquare : eps₁ ^ 2 ≤ eps₂ ^ 2 := by
    nlinarith
  have hscale : μ * (n : ℝ) / 2 ≥ 0 := by
    positivity
  nlinarith

lemma targetEnergyGap_mono_mu
    {μ₁ μ₂ : ℝ} (n : ℕ) (eps : ℝ)
    (hmono : μ₁ ≤ μ₂) :
    targetEnergyGap μ₁ n eps ≤ targetEnergyGap μ₂ n eps := by
  unfold targetEnergyGap
  have hsquare : 0 ≤ eps ^ 2 := by positivity
  have hscale : 0 ≤ (n : ℝ) * eps ^ 2 / 2 := by positivity
  nlinarith

lemma targetEnergyGap_pos
    (μ : ℝ) (n : ℕ) (eps : ℝ)
    (hμ : 0 < μ)
    (hn : 0 < n)
    (heps : 0 < eps) :
    0 < targetEnergyGap μ n eps := by
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have heps2 : 0 < eps ^ 2 := by positivity
  have hmid : 0 < (n : ℝ) * eps ^ 2 := mul_pos hnR heps2
  have hprod : 0 < μ * ((n : ℝ) * eps ^ 2) := mul_pos hμ hmid
  have htwo : (0 : ℝ) < 2 := by norm_num
  unfold targetEnergyGap
  nlinarith

lemma segmentEnergy_zero {n : ℕ}
    (energy : CoordSet n → ℝ) (center x : CoordSet n) :
    segmentEnergy energy center x 0 = energy center := by
  unfold segmentEnergy
  simp

lemma segmentEnergy_one {n : ℕ}
    (energy : CoordSet n → ℝ) (center x : CoordSet n) :
    segmentEnergy energy center x 1 = energy x := by
  unfold segmentEnergy
  simp

def CertifiedQuadraticWindow.toCertifiedQuadraticBasin
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (window : CertifiedQuadraticWindow energy center) :
    CertifiedQuadraticBasin energy center :=
  { μ := window.μ
    μ_pos := window.μ_pos
    quadratic_growth := window.lower_quadratic }

def CertifiedRayleighHessianBounds.toCertifiedSegmentCurvature
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedRayleighHessianBounds energy center) :
    CertifiedSegmentCurvature energy center :=
  { μ := cert.μ
    M := cert.M
    μ_pos := cert.μ_pos
    M_nonneg := cert.M_nonneg
    ray_contDiff := cert.ray_contDiff
    ray_stationary := cert.ray_stationary
    ray_lower := by
      intro x t ht
      calc
        cert.μ * squaredDisplacement x center ≤ cert.hessAlong x t := cert.rayleigh_lower x t ht
        _ = iteratedDeriv 2 (segmentEnergy energy center x) t := cert.hessAlong_eq x t ht
    ray_upper := by
      intro x t ht
      calc
        iteratedDeriv 2 (segmentEnergy energy center x) t = cert.hessAlong x t := by
          symm
          exact cert.hessAlong_eq x t ht
        _ ≤ cert.M * squaredDisplacement x center := cert.rayleigh_upper x t ht }

lemma contractionFactor_nonneg
    (α μ : ℝ)
    (h_le : α * μ ≤ 1) :
    0 ≤ contractionFactor α μ := by
  unfold contractionFactor
  linarith

lemma contractionFactor_lt_one
    (α μ : ℝ)
    (hα : 0 < α)
    (hμ : 0 < μ) :
    contractionFactor α μ < 1 := by
  unfold contractionFactor
  have hprod : 0 < α * μ := mul_pos hα hμ
  linarith

lemma contractionFactor_nonneg_of_step_le_inv
    (α μ : ℝ)
    (hμ : 0 ≤ μ)
    (hstep : α ≤ 1 / μ)
    (hμ_pos : 0 < μ) :
    0 ≤ contractionFactor α μ := by
  apply contractionFactor_nonneg
  have hmul := mul_le_mul_of_nonneg_right hstep hμ
  simpa [hμ_pos.ne'] using hmul

lemma CertifiedGradientStepParameters.M_pos
    (params : CertifiedGradientStepParameters) : 0 < params.M := by
  exact lt_of_lt_of_le params.μ_pos params.μ_le_M

lemma CertifiedGradientStepParameters.q_nonneg
    (params : CertifiedGradientStepParameters) : 0 ≤ params.q := by
  unfold CertifiedGradientStepParameters.q
  apply contractionFactor_nonneg
  have hα_nonneg : 0 ≤ params.α := le_of_lt params.α_pos
  have hαμ_le_αM : params.α * params.μ ≤ params.α * params.M := by
    exact mul_le_mul_of_nonneg_left params.μ_le_M hα_nonneg
  have hαM_le_one : params.α * params.M ≤ 1 := by
    have hmul := mul_le_mul_of_nonneg_right params.α_le_invM (le_of_lt params.M_pos)
    simpa [params.M_pos.ne', mul_comm, mul_left_comm, mul_assoc] using hmul
  exact le_trans hαμ_le_αM hαM_le_one

lemma CertifiedGradientStepParameters.q_lt_one
    (params : CertifiedGradientStepParameters) : params.q < 1 := by
  unfold CertifiedGradientStepParameters.q
  exact contractionFactor_lt_one params.α params.μ params.α_pos params.μ_pos

lemma CertifiedGradientStepParameters.q_mem_Ico
    (params : CertifiedGradientStepParameters) :
    params.q ∈ Set.Ico 0 1 := by
  constructor
  · exact params.q_nonneg
  · exact params.q_lt_one

/-- Runtime-facing certificate for a concrete gradient-descent refinement trace.
    The contraction factor is not provided ad hoc; it is derived from certified
    step parameters `(α, μ, M)`, while the only algorithm-specific obligation is
    the one-step gap contraction itself. -/
structure CertifiedGradientDescentDynamics
    (energyGapAt : ℕ → ℝ) where
  params : CertifiedGradientStepParameters
  initialGap_nonneg : 0 ≤ energyGapAt 0
  step_contract : ∀ t : ℕ, energyGapAt (t + 1) ≤ params.q * energyGapAt t

/-- Taylor-Lagrange witness for the energy gap along a segment from `center` to `x`.
    When the line restriction has zero first derivative at the center, the energy
    gap is exactly one half of a second derivative value along the segment. -/
lemma exists_segmentEnergy_gap_eq_secondDeriv_half
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (hcont : ContDiffOn ℝ 2 (segmentEnergy energy center x) (Set.Icc 0 1))
    (hstat : derivWithin (segmentEnergy energy center x) (Set.Icc 0 1) 0 = 0) :
    ∃ u ∈ Set.Ioo (0 : ℝ) 1,
      energy x - energy center = iteratedDeriv 2 (segmentEnergy energy center x) u / 2 := by
  have h01 : (0 : ℝ) < 1 := by norm_num
  obtain ⟨u, hu, hTaylor⟩ := taylor_mean_remainder_lagrange_iteratedDeriv h01 hcont
  refine ⟨u, hu, ?_⟩
  have hTaylor' :
      segmentEnergy energy center x 1 -
          taylorWithinEval (segmentEnergy energy center x) 1 (Set.Icc 0 1) 0 1 =
        iteratedDeriv 2 (segmentEnergy energy center x) u / 2 := by
    simpa using hTaylor
  have hTaylor'' := hTaylor'
  simp [taylorWithinEval_succ, taylor_within_zero_eval, iteratedDerivWithin_one,
    hstat, segmentEnergy_one, segmentEnergy_zero] at hTaylor''
  exact hTaylor''

/-- Certified segment-curvature bounds imply a quadratic energy window. -/
def CertifiedSegmentCurvature.toCertifiedQuadraticWindow
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (curv : CertifiedSegmentCurvature energy center) :
    CertifiedQuadraticWindow energy center := by
  refine
    { μ := curv.μ
      M := curv.M
      μ_pos := curv.μ_pos
      M_nonneg := curv.M_nonneg
      lower_quadratic := ?_
      upper_quadratic := ?_ }
  · intro x
    obtain ⟨u, hu, hgap⟩ := exists_segmentEnergy_gap_eq_secondDeriv_half
      energy center x (curv.ray_contDiff x) (curv.ray_stationary x)
    have hu_mem : u ∈ Set.Icc (0 : ℝ) 1 := ⟨le_of_lt hu.1, le_of_lt hu.2⟩
    have hlow := curv.ray_lower x u hu_mem
    have hscale := mul_le_mul_of_nonneg_right hlow (by norm_num : (0 : ℝ) ≤ 1 / 2)
    have hscale' : curv.μ / 2 * squaredDisplacement x center ≤
        iteratedDeriv 2 (segmentEnergy energy center x) u / 2 := by
      nlinarith
    simpa [hgap] using hscale'
  · intro x
    obtain ⟨u, hu, hgap⟩ := exists_segmentEnergy_gap_eq_secondDeriv_half
      energy center x (curv.ray_contDiff x) (curv.ray_stationary x)
    have hu_mem : u ∈ Set.Icc (0 : ℝ) 1 := ⟨le_of_lt hu.1, le_of_lt hu.2⟩
    have hupp := curv.ray_upper x u hu_mem
    have hscale := mul_le_mul_of_nonneg_right hupp (by norm_num : (0 : ℝ) ≤ 1 / 2)
    have hscale' : iteratedDeriv 2 (segmentEnergy energy center x) u / 2 ≤
        curv.M / 2 * squaredDisplacement x center := by
      nlinarith
    simpa [hgap] using hscale'

def CertifiedRayleighHessianBounds.toCertifiedQuadraticWindow
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedRayleighHessianBounds energy center) :
    CertifiedQuadraticWindow energy center :=
  cert.toCertifiedSegmentCurvature.toCertifiedQuadraticWindow

def CertifiedRayleighHessianBounds.toCertifiedQuadraticBasin
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedRayleighHessianBounds energy center) :
    CertifiedQuadraticBasin energy center :=
  cert.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin

def CertifiedLocalSpectralEnclosure.toCertifiedRayleighHessianBounds
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedLocalSpectralEnclosure energy center) :
    CertifiedRayleighHessianBounds energy center :=
  { μ := cert.lmin
    M := cert.lmax
    hessAlong := cert.rayleighEval
    μ_pos := cert.lmin_pos
    M_nonneg := cert.lmax_nonneg
    ray_contDiff := cert.ray_contDiff
    ray_stationary := cert.ray_stationary
    hessAlong_eq := cert.rayleigh_eq_secondDeriv
    rayleigh_lower := cert.eigen_lower
    rayleigh_upper := cert.eigen_upper }

def CertifiedLocalSpectralEnclosure.toCertifiedSegmentCurvature
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedLocalSpectralEnclosure energy center) :
    CertifiedSegmentCurvature energy center :=
  cert.toCertifiedRayleighHessianBounds.toCertifiedSegmentCurvature

def CertifiedLocalSpectralEnclosure.toCertifiedQuadraticWindow
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedLocalSpectralEnclosure energy center) :
    CertifiedQuadraticWindow energy center :=
  cert.toCertifiedSegmentCurvature.toCertifiedQuadraticWindow

def CertifiedLocalSpectralEnclosure.toCertifiedQuadraticBasin
    {n : ℕ}
    {energy : CoordSet n → ℝ}
    {center : CoordSet n}
    (cert : CertifiedLocalSpectralEnclosure energy center) :
    CertifiedQuadraticBasin energy center :=
  cert.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin

/-- A one-step contraction certificate automatically upgrades to a full certified
    linear energy convergence object by induction. -/
def CertifiedOneStepEnergyContraction.toCertifiedLinearEnergyConvergence
    {energyGapAt : ℕ → ℝ}
    (hstep : CertifiedOneStepEnergyContraction energyGapAt) :
    CertifiedLinearEnergyConvergence energyGapAt := by
  refine
    { q := hstep.q
      q_nonneg := hstep.q_nonneg
      q_lt_one := hstep.q_lt_one
      initialGap_nonneg := hstep.initialGap_nonneg
      rate := ?_ }
  intro t
  induction t with
  | zero =>
      simp
  | succ t ih =>
      calc
        energyGapAt (t + 1) ≤ hstep.q * energyGapAt t := hstep.step_contract t
        _ ≤ hstep.q * (hstep.q ^ t * energyGapAt 0) := by
          exact mul_le_mul_of_nonneg_left ih hstep.q_nonneg
        _ = hstep.q ^ (t + 1) * energyGapAt 0 := by ring

noncomputable def CertifiedGradientDescentDynamics.toCertifiedOneStepEnergyContraction
    {energyGapAt : ℕ → ℝ}
    (dyn : CertifiedGradientDescentDynamics energyGapAt) :
    CertifiedOneStepEnergyContraction energyGapAt :=
  { q := dyn.params.q
    q_nonneg := dyn.params.q_nonneg
    q_lt_one := dyn.params.q_lt_one
    initialGap_nonneg := dyn.initialGap_nonneg
    step_contract := dyn.step_contract }

noncomputable def CertifiedGradientDescentDynamics.toCertifiedLinearEnergyConvergence
    {energyGapAt : ℕ → ℝ}
    (dyn : CertifiedGradientDescentDynamics energyGapAt) :
    CertifiedLinearEnergyConvergence energyGapAt :=
  dyn.toCertifiedOneStepEnergyContraction.toCertifiedLinearEnergyConvergence

lemma energyGap_nonneg_of_quadratic_growth
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
  (basin : CertifiedQuadraticBasin energy center) :
    0 ≤ energy x - energy center := by
  have hleft : 0 ≤ (basin.μ / 2) * squaredDisplacement x center := by
    have hμ : 0 ≤ basin.μ / 2 := by nlinarith [basin.μ_pos]
    exact mul_nonneg hμ (squaredDisplacement_nonneg x center)
  exact le_trans hleft (basin.quadratic_growth x)

/-- Quadratic growth implies an explicit RMSD upper bound from an energy gap. -/
theorem rmsd_le_of_quadratic_growth
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (hn : 0 < n) :
    rmsd x center ≤
      Real.sqrt (2 * (energy x - energy center) / (basin.μ * (n : ℝ))) := by
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hμR : 0 < basin.μ := basin.μ_pos
  have hgap_nonneg : 0 ≤ energy x - energy center :=
    energyGap_nonneg_of_quadratic_growth energy center x basin
  have hdisp_le : squaredDisplacement x center ≤
      2 * (energy x - energy center) / basin.μ := by
    have hscaled :=
      mul_le_mul_of_nonneg_left (basin.quadratic_growth x) (by positivity : 0 ≤ 2 / basin.μ)
    have hidentity : (2 / basin.μ) * ((basin.μ / 2) * squaredDisplacement x center)
        = squaredDisplacement x center := by
      field_simp [hμR.ne']
    have hright : (2 / basin.μ) * (energy x - energy center)
        = 2 * (energy x - energy center) / basin.μ := by ring
    rw [hidentity, hright] at hscaled
    exact hscaled
  have hdiv : squaredDisplacement x center / (n : ℝ) ≤
      2 * (energy x - energy center) / (basin.μ * (n : ℝ)) := by
    have := div_le_div_of_nonneg_right hdisp_le (le_of_lt hnR)
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using this
  have hsqrt := Real.sqrt_le_sqrt hdiv
  simpa [rmsd] using hsqrt

/-- If the energy gap is below the target threshold, the pose is within the
    requested RMSD tolerance. -/
theorem rmsd_le_of_energyGap_le_target
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hgap : energy x - energy center ≤ targetEnergyGap basin.μ n eps) :
    rmsd x center ≤ eps := by
  have hbase := rmsd_le_of_quadratic_growth energy center x basin hn
  have hμR : 0 < basin.μ := basin.μ_pos
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hinside :
      2 * (energy x - energy center) / (basin.μ * (n : ℝ)) ≤ eps ^ 2 := by
    have hmul := mul_le_mul_of_nonneg_left hgap (by positivity : 0 ≤ 2 / (basin.μ * (n : ℝ)))
    unfold targetEnergyGap at hmul
    have htarget :
        (2 / (basin.μ * (n : ℝ))) * (basin.μ * (n : ℝ) * eps ^ 2 / 2) = eps ^ 2 := by
      field_simp [hμR.ne', hnR.ne']
    have hleft :
        (2 / (basin.μ * (n : ℝ))) * (energy x - energy center) =
          2 * (energy x - energy center) / (basin.μ * (n : ℝ)) := by ring
    rw [hleft, htarget] at hmul
    exact hmul
  have hgap_nonneg := energyGap_nonneg_of_quadratic_growth energy center x basin
  have hinside_nonneg : 0 ≤ 2 * (energy x - energy center) / (basin.μ * (n : ℝ)) := by
    positivity
  have hsqrt_bound :
      Real.sqrt (2 * (energy x - energy center) / (basin.μ * (n : ℝ))) ≤ Real.sqrt (eps ^ 2) :=
    Real.sqrt_le_sqrt hinside
  have hsqrt_eps : Real.sqrt (eps ^ 2) = eps := by
    rw [Real.sqrt_sq_eq_abs]
    exact abs_of_nonneg heps
  exact le_trans hbase (by simpa [hsqrt_eps] using hsqrt_bound)

/-- Upper quadratic growth converts an RMSD certificate back into an energy-gap
    certificate. This is useful for choosing a target optimization budget from a
    geometric tolerance. -/
theorem energyGap_le_of_rmsd_le
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (window : CertifiedQuadraticWindow energy center)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hrmsd : rmsd x center ≤ eps) :
    energy x - energy center ≤ targetEnergyGap window.M n eps := by
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hrmsd_nonneg : 0 ≤ rmsd x center := rmsd_nonneg x center
  have hsq : rmsd x center ^ 2 ≤ eps ^ 2 := by
    nlinarith
  have hdisp_div : squaredDisplacement x center / (n : ℝ) ≤ eps ^ 2 := by
    simpa [rmsd_sq_eq x center hn] using hsq
  have hdisp : squaredDisplacement x center ≤ (n : ℝ) * eps ^ 2 := by
    have hmul := mul_le_mul_of_nonneg_right hdisp_div (le_of_lt hnR)
    field_simp [hnR.ne'] at hmul
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul
  have hquad := window.upper_quadratic x
  have hbound : (window.M / 2) * squaredDisplacement x center ≤ targetEnergyGap window.M n eps := by
    unfold targetEnergyGap
    have hMhalf : 0 ≤ window.M / 2 := by
      nlinarith [window.M_nonneg]
    have hscale := mul_le_mul_of_nonneg_left hdisp hMhalf
    have hrewrite : (window.M / 2) * ((n : ℝ) * eps ^ 2) = window.M * (n : ℝ) * eps ^ 2 / 2 := by
      ring
    rw [hrewrite] at hscale
    exact hscale
  exact le_trans hquad hbound

/-- A parameter-space quadratic window transfers to coordinate space once the
    runtime has certified squared-displacement comparison bounds in both
    directions. This is the algebraic core behind using Jacobian singular-value
    bounds to convert `(lmin_param, lmax_param)` into coordinate-space
    `(μ_coord, M_coord)`. -/
theorem parameter_window_transfers_to_coordinate_window_sq
    (μparam Mparam σminSq σmaxSq dParamSq dCoordSq gap : ℝ)
    (hμ : 0 ≤ μparam)
    (hM : 0 ≤ Mparam)
    (hσmin : 0 < σminSq)
    (hσmax : 0 < σmaxSq)
    (hLowerParam : (μparam / 2) * dParamSq ≤ gap)
    (hUpperParam : gap ≤ (Mparam / 2) * dParamSq)
    (hCoordUpper : dCoordSq ≤ σmaxSq * dParamSq)
    (hCoordLower : σminSq * dParamSq ≤ dCoordSq) :
    ((μparam / σmaxSq) / 2) * dCoordSq ≤ gap ∧
      gap ≤ ((Mparam / σminSq) / 2) * dCoordSq := by
  constructor
  · have hscaled : ((μparam / σmaxSq) / 2) * dCoordSq ≤
        ((μparam / σmaxSq) / 2) * (σmaxSq * dParamSq) := by
      exact mul_le_mul_of_nonneg_left hCoordUpper (by positivity)
    have hrewrite : ((μparam / σmaxSq) / 2) * (σmaxSq * dParamSq) =
        (μparam / 2) * dParamSq := by
      field_simp [hσmax.ne']
    rw [hrewrite] at hscaled
    exact le_trans hscaled hLowerParam
  · have hscaled : ((Mparam / σminSq) / 2) * (σminSq * dParamSq) ≤
        ((Mparam / σminSq) / 2) * dCoordSq := by
      exact mul_le_mul_of_nonneg_left hCoordLower (by positivity)
    have hrewrite : ((Mparam / σminSq) / 2) * (σminSq * dParamSq) =
        (Mparam / 2) * dParamSq := by
      field_simp [hσmin.ne']
    rw [hrewrite] at hscaled
    exact le_trans hUpperParam hscaled

/-- Upper quadratic control at the initial pose and linear energy convergence
    together give a direct RMSD target certificate after `t` refinement steps.

    This is the runtime-facing bridge needed for theorem-honest two-phase seed
    budgets: if the initial RMSD lies inside a radius `eps0` whose induced
    energy gap contracts enough after `t` steps, then the `t`-th iterate is
    certified to lie within the requested RMSD tolerance `eps`. -/
theorem rmsd_target_of_initial_rmsd_and_linear_energy_convergence
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (window : CertifiedQuadraticWindow energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps eps0 : ℝ)
    (heps : 0 ≤ eps)
    (heps0 : 0 ≤ eps0)
    (t : ℕ)
    (hinit : rmsd (poseAt 0) center ≤ eps0)
    (hcontract : conv.q ^ t * window.M * eps0 ^ 2 ≤ window.μ * eps ^ 2) :
    rmsd (poseAt t) center ≤ eps := by
  have hgap0 : energy (poseAt 0) - energy center ≤ targetEnergyGap window.M n eps0 := by
    exact energyGap_le_of_rmsd_le energy center (poseAt 0) window hn eps0 heps0 hinit
  have hq_nonneg : 0 ≤ conv.q ^ t := by
    exact pow_nonneg conv.q_nonneg t
  have hscaled0 :
      conv.q ^ t * (energy (poseAt 0) - energy center) ≤
        conv.q ^ t * targetEnergyGap window.M n eps0 := by
    exact mul_le_mul_of_nonneg_left hgap0 hq_nonneg
  have htarget :
      conv.q ^ t * targetEnergyGap window.M n eps0 ≤ targetEnergyGap window.μ n eps := by
    unfold targetEnergyGap
    have hfactor_nonneg : 0 ≤ (n : ℝ) / 2 := by positivity
    have hmul := mul_le_mul_of_nonneg_left hcontract hfactor_nonneg
    have hleft :
        ((n : ℝ) / 2) * (conv.q ^ t * window.M * eps0 ^ 2) =
          conv.q ^ t * (window.M * (n : ℝ) * eps0 ^ 2 / 2) := by
      ring
    have hright :
        ((n : ℝ) / 2) * (window.μ * eps ^ 2) =
          window.μ * (n : ℝ) * eps ^ 2 / 2 := by
      ring
    rw [hleft, hright] at hmul
    exact hmul
  have hbudget :
      conv.q ^ t * (energy (poseAt 0) - energy center) ≤ targetEnergyGap window.μ n eps := by
    exact le_trans hscaled0 htarget
  apply rmsd_le_of_energyGap_le_target
    energy center (poseAt t) window.toCertifiedQuadraticBasin hn eps heps
  exact le_trans (conv.rate t) hbudget

/-- Selecting the largest certified prefix from a finite probe trajectory is
    sound: once a runtime has identified a nonempty finite set of prefix lengths
    that satisfy the contraction inequality from
    `rmsd_target_of_initial_rmsd_and_linear_energy_convergence`, the maximal
    such prefix still certifies the requested RMSD target. This matches the
    runtime policy of scanning an observed probe trajectory backwards and
    returning the latest certifiable prefix. -/
theorem rmsd_target_of_maxCertifiedPrefix
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (window : CertifiedQuadraticWindow energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps eps0 : ℝ)
    (heps : 0 ≤ eps)
    (heps0 : 0 ≤ eps0)
    (candidates : Finset ℕ)
    (hinit : rmsd (poseAt 0) center ≤ eps0)
    (hcertNonempty :
      (candidates.filter (fun t => conv.q ^ t * window.M * eps0 ^ 2 ≤ window.μ * eps ^ 2)).Nonempty) :
    let certified := candidates.filter (fun t => conv.q ^ t * window.M * eps0 ^ 2 ≤ window.μ * eps ^ 2)
    let tStar := certified.max' hcertNonempty
    rmsd (poseAt tStar) center ≤ eps := by
  let certified := candidates.filter (fun t => conv.q ^ t * window.M * eps0 ^ 2 ≤ window.μ * eps ^ 2)
  let tStar := certified.max' hcertNonempty
  have ht_mem : tStar ∈ certified := by
    exact Finset.max'_mem certified hcertNonempty
  have hcontract : conv.q ^ tStar * window.M * eps0 ^ 2 ≤ window.μ * eps ^ 2 := by
    exact (Finset.mem_filter.mp ht_mem).2
  exact rmsd_target_of_initial_rmsd_and_linear_energy_convergence
    energy center poseAt window conv hn eps eps0 heps heps0 tStar hinit hcontract

/-- Any certified linear energy-rate immediately yields an RMSD bound after `t`
    refinement steps once combined with a certified quadratic basin. -/
theorem rmsd_le_of_linear_energy_convergence
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (t : ℕ) :
    rmsd (poseAt t) center ≤
      Real.sqrt (2 * (conv.q ^ t * (energy (poseAt 0) - energy center)) /
        (basin.μ * (n : ℝ))) := by
  have hbase := rmsd_le_of_quadratic_growth energy center (poseAt t) basin hn
  have hμR : 0 < basin.μ := basin.μ_pos
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hrate := conv.rate t
  have hscaled := mul_le_mul_of_nonneg_left hrate (by positivity : 0 ≤ 2 / (basin.μ * (n : ℝ)))
  have hleft :
      (2 / (basin.μ * (n : ℝ))) * (energy (poseAt t) - energy center) =
        2 * (energy (poseAt t) - energy center) / (basin.μ * (n : ℝ)) := by ring
  have hright :
      (2 / (basin.μ * (n : ℝ))) * (conv.q ^ t * (energy (poseAt 0) - energy center)) =
        2 * (conv.q ^ t * (energy (poseAt 0) - energy center)) / (basin.μ * (n : ℝ)) := by ring
  rw [hleft, hright] at hscaled
  have hsqrt := Real.sqrt_le_sqrt hscaled
  exact le_trans hbase hsqrt

/-- Implementation-facing stopping rule: once the certified energy-rate bound drops
    below the target-energy threshold, the current pose is certified to satisfy the
    requested RMSD tolerance. -/
theorem rmsd_target_of_linear_energy_convergence
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (t : ℕ)
    (hbudget : conv.q ^ t * (energy (poseAt 0) - energy center) ≤ targetEnergyGap basin.μ n eps) :
    rmsd (poseAt t) center ≤ eps := by
  apply rmsd_le_of_energyGap_le_target energy center (poseAt t) basin hn eps heps
  exact le_trans (conv.rate t) hbudget

/-- Adequacy is monotone in the iteration budget whenever the certified rate has
    contraction factor in `[0,1)`. Once enough steps are taken, any larger budget
    is also enough. -/
theorem adequateIterationBudget_mono
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    {t u : ℕ}
    (htu : t ≤ u)
    (hAdeq : AdequateIterationBudget energy center poseAt basin conv eps t) :
    AdequateIterationBudget energy center poseAt basin conv eps u := by
  unfold AdequateIterationBudget at *
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le htu
  have hq_le_one : conv.q ≤ 1 := le_of_lt conv.q_lt_one
  have hpowk : conv.q ^ k ≤ 1 := by
    simpa using (pow_le_one₀ (n := k) conv.q_nonneg hq_le_one)
  have hpowt_nonneg : 0 ≤ conv.q ^ t := by
    exact pow_nonneg conv.q_nonneg t
  have hpow_mono : conv.q ^ (t + k) ≤ conv.q ^ t := by
    rw [pow_add]
    have hmul : conv.q ^ t * conv.q ^ k ≤ conv.q ^ t * 1 :=
      mul_le_mul_of_nonneg_left hpowk hpowt_nonneg
    simpa using hmul
  have hscaled :=
    mul_le_mul_of_nonneg_right hpow_mono conv.initialGap_nonneg
  exact le_trans hscaled hAdeq

/-- Larger RMSD tolerances are easier to certify: any budget adequate for a tighter
    tolerance remains adequate for a looser one. -/
theorem adequateIterationBudget_mono_eps
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    {eps₁ eps₂ : ℝ}
    (heps₁ : 0 ≤ eps₁)
    (hepsmono : eps₁ ≤ eps₂)
    (t : ℕ)
    (hAdeq : AdequateIterationBudget energy center poseAt basin conv eps₁ t) :
    AdequateIterationBudget energy center poseAt basin conv eps₂ t := by
  unfold AdequateIterationBudget at *
  exact le_trans hAdeq (targetEnergyGap_mono_eps basin.μ n (le_of_lt basin.μ_pos) heps₁ hepsmono)

/-- Refinement cost is monotone in the iteration budget when per-step work is
    nonnegative. -/
theorem refinementCost_mono
    (setupCost stepCost : ℝ)
    (hstep : 0 ≤ stepCost)
    {t u : ℕ}
    (htu : t ≤ u) :
    refinementCost setupCost stepCost t ≤ refinementCost setupCost stepCost u := by
  unfold refinementCost
  have hcast : (t : ℝ) ≤ (u : ℝ) := by exact_mod_cast htu
  nlinarith

/-- The smallest adequate iteration budget is optimal under a monotone runtime
    model. This is the refinement analogue of minimal adequate seed budgets in the
    blind-conformer pipeline proofs. -/
theorem minimal_adequate_iterationBudget_optimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (setupCost stepCost : ℝ)
    (hstep : 0 ≤ stepCost)
    {t₀ t : ℕ}
    (_hMinAdeq : AdequateIterationBudget energy center poseAt basin conv eps t₀)
    (hMinimal : ∀ m < t₀, ¬ AdequateIterationBudget energy center poseAt basin conv eps m)
    (hAdeqT : AdequateIterationBudget energy center poseAt basin conv eps t) :
    refinementCost setupCost stepCost t₀ ≤ refinementCost setupCost stepCost t := by
  have ht₀le : t₀ ≤ t := by
    by_contra hnot
    exact hMinimal t (Nat.lt_of_not_ge hnot) hAdeqT
  exact refinementCost_mono setupCost stepCost hstep ht₀le

/-- Once a budget is certified adequate, the RMSD target follows immediately. -/
theorem rmsd_target_of_adequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (t : ℕ)
    (hAdeq : AdequateIterationBudget energy center poseAt basin conv eps t) :
    rmsd (poseAt t) center ≤ eps := by
  exact rmsd_target_of_linear_energy_convergence
    energy center poseAt basin conv hn eps heps t hAdeq

theorem adequateIterationBudget_zero_of_initialGap_le_target
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hgap : energy (poseAt 0) - energy center ≤ targetEnergyGap basin.μ n eps) :
    AdequateIterationBudget energy center poseAt basin conv eps 0 := by
  unfold AdequateIterationBudget
  simpa using hgap

theorem adequateIterationBudget_of_logarithmicIterationBound
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hq_pos : 0 < conv.q)
    (htarget_pos : 0 < targetEnergyGap basin.μ n eps) :
    AdequateIterationBudget energy center poseAt basin conv eps
      (logarithmicIterationBound conv.q (energy (poseAt 0) - energy center) (targetEnergyGap basin.μ n eps)) := by
  let gap0 := energy (poseAt 0) - energy center
  let target := targetEnergyGap basin.μ n eps
  let t := logarithmicIterationBound conv.q gap0 target
  have hq_lt_one : conv.q < 1 := conv.q_lt_one
  have hq_nonneg : 0 ≤ conv.q := conv.q_nonneg
  have hq_ne : conv.q ≠ 0 := hq_pos.ne'
  have hbase_pos : 0 < 1 / conv.q := by positivity
  have hbase_log_pos : 0 < Real.log (1 / conv.q) := by
    have hbase_gt_one : 1 < 1 / conv.q := by
      field_simp [hq_ne]
      nlinarith
    exact Real.log_pos hbase_gt_one
  have hceil : Real.log (gap0 / target) / Real.log (1 / conv.q) ≤ t := Nat.le_ceil _
  have hlog : Real.log (gap0 / target) ≤ t * Real.log (1 / conv.q) := by
    rw [_root_.div_le_iff₀ hbase_log_pos] at hceil
    simpa [mul_comm, mul_left_comm, mul_assoc] using hceil
  have hratio_le : gap0 / target ≤ (1 / conv.q) ^ t := by
    exact Real.pow_le_of_le_log hbase_pos hlog
  have hscaled : gap0 ≤ target * (1 / conv.q) ^ t := by
    rw [_root_.div_le_iff₀ htarget_pos] at hratio_le
    simpa [mul_comm, mul_left_comm, mul_assoc] using hratio_le
  have hpow_pos : 0 < conv.q ^ t := pow_pos hq_pos _
  have hfinal : conv.q ^ t * gap0 ≤ target := by
    have hmul := mul_le_mul_of_nonneg_left hscaled (le_of_lt hpow_pos)
    have hcancel : conv.q ^ t * (target * (1 / conv.q) ^ t) = target := by
      calc
        conv.q ^ t * (target * (1 / conv.q) ^ t)
            = target * (conv.q ^ t * (1 / conv.q) ^ t) := by ring
        _ = target * 1 := by
          congr 1
          simp [one_div, inv_pow, hq_ne]
        _ = target := by ring
    exact le_trans hmul (le_of_eq hcancel)
  unfold AdequateIterationBudget
  simpa [gap0, target, t] using hfinal

theorem rmsd_target_of_zeroIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hgap : energy (poseAt 0) - energy center ≤ targetEnergyGap basin.μ n eps) :
    rmsd (poseAt 0) center ≤ eps := by
  apply rmsd_target_of_adequateIterationBudget energy center poseAt basin conv hn eps heps 0
  exact adequateIterationBudget_zero_of_initialGap_le_target energy center poseAt basin conv eps hgap

/-- Any certified linear rate with contraction factor in `[0,1)` eventually reaches
    every strictly positive RMSD tolerance. This gives a theorem-backed existence
    result for a sufficient refinement budget. -/
theorem exists_adequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t := by
  let gap0 := energy (poseAt 0) - energy center
  have htarget : 0 < targetEnergyGap basin.μ n eps := by
    have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
    have heps2 : 0 < eps ^ 2 := by positivity
    have hmid : 0 < (n : ℝ) * eps ^ 2 := mul_pos hnR heps2
    have hprod : 0 < basin.μ * ((n : ℝ) * eps ^ 2) := mul_pos basin.μ_pos hmid
    have htwo : (0 : ℝ) < 2 := by norm_num
    unfold targetEnergyGap
    nlinarith
  have hqabs : |conv.q| < 1 := by
    rw [abs_of_nonneg conv.q_nonneg]
    exact conv.q_lt_one
  have htendsto : Filter.Tendsto (fun t : ℕ => conv.q ^ t * gap0) Filter.atTop (nhds 0) := by
    simpa [gap0] using (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hqabs).mul_const gap0
  have hevent : ∀ᶠ t : ℕ in Filter.atTop, conv.q ^ t * gap0 < targetEnergyGap basin.μ n eps := by
    exact htendsto (Iio_mem_nhds htarget)
  rcases Filter.eventually_atTop.1 hevent with ⟨t, ht⟩
  refine ⟨t, ?_⟩
  unfold AdequateIterationBudget
  exact le_of_lt (ht t le_rfl)

/-- The least certified adequate iteration budget, packaged from any existence proof. -/
noncomputable def leastAdequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t) : ℕ :=
  by
    classical
    exact Nat.find hexists

theorem leastAdequateIterationBudget_spec
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t) :
    AdequateIterationBudget energy center poseAt basin conv eps
      (leastAdequateIterationBudget energy center poseAt basin conv eps hexists) := by
  unfold leastAdequateIterationBudget
  classical
  exact Nat.find_spec hexists

theorem leastAdequateIterationBudget_minimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t)
    {m : ℕ}
    (hm : AdequateIterationBudget energy center poseAt basin conv eps m) :
    leastAdequateIterationBudget energy center poseAt basin conv eps hexists ≤ m := by
  unfold leastAdequateIterationBudget
  classical
  exact Nat.find_min' hexists hm

/-- The least adequate iteration budget is cost-optimal for any nonnegative
    per-step refinement cost. This is the directly implementation-facing version:
    search for the smallest theorem-certified budget and stop there. -/
theorem leastAdequateIterationBudget_optimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (setupCost stepCost : ℝ)
    (hstep : 0 ≤ stepCost)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t)
    {t : ℕ}
    (hAdeqT : AdequateIterationBudget energy center poseAt basin conv eps t) :
    refinementCost setupCost stepCost
      (leastAdequateIterationBudget energy center poseAt basin conv eps hexists) ≤
    refinementCost setupCost stepCost t := by
  apply refinementCost_mono
  · exact hstep
  · exact leastAdequateIterationBudget_minimal energy center poseAt basin conv eps hexists hAdeqT

/-- The least adequate budget also immediately certifies the requested RMSD
    tolerance. -/
theorem rmsd_target_of_leastAdequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t) :
    rmsd
      (poseAt (leastAdequateIterationBudget energy center poseAt basin conv eps hexists))
      center ≤ eps := by
  apply rmsd_target_of_adequateIterationBudget energy center poseAt basin conv hn eps heps
  exact leastAdequateIterationBudget_spec energy center poseAt basin conv eps hexists

theorem leastAdequateIterationBudget_eq_zero_of_zero_adequate
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (eps : ℝ)
    (hexists : ∃ t, AdequateIterationBudget energy center poseAt basin conv eps t)
    (hAdeq0 : AdequateIterationBudget energy center poseAt basin conv eps 0) :
    leastAdequateIterationBudget energy center poseAt basin conv eps hexists = 0 := by
  apply Nat.le_antisymm
  · exact leastAdequateIterationBudget_minimal energy center poseAt basin conv eps hexists hAdeq0
  · exact Nat.zero_le _

/-- Canonical theorem-backed iteration budget for a positive RMSD tolerance. This
    packages existence and minimization into a single runtime-facing quantity. -/
noncomputable def canonicalAdequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) : ℕ :=
  leastAdequateIterationBudget energy center poseAt basin conv eps
    (exists_adequateIterationBudget energy center poseAt basin conv hn eps heps)

theorem canonicalAdequateIterationBudget_spec
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    AdequateIterationBudget energy center poseAt basin conv eps
      (canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps) := by
  unfold canonicalAdequateIterationBudget
  exact leastAdequateIterationBudget_spec energy center poseAt basin conv eps
    (exists_adequateIterationBudget energy center poseAt basin conv hn eps heps)

theorem canonicalAdequateIterationBudget_optimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (setupCost stepCost : ℝ)
    (hstep : 0 ≤ stepCost)
    {t : ℕ}
    (hAdeqT : AdequateIterationBudget energy center poseAt basin conv eps t) :
    refinementCost setupCost stepCost
      (canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps) ≤
    refinementCost setupCost stepCost t := by
  unfold canonicalAdequateIterationBudget
  exact leastAdequateIterationBudget_optimal energy center poseAt basin conv eps
    setupCost stepCost hstep
    (exists_adequateIterationBudget energy center poseAt basin conv hn eps heps) hAdeqT

theorem canonicalAdequateIterationBudget_eq_zero_of_initialGap_le_target
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (hgap : energy (poseAt 0) - energy center ≤ targetEnergyGap basin.μ n eps) :
    canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps = 0 := by
  unfold canonicalAdequateIterationBudget
  apply leastAdequateIterationBudget_eq_zero_of_zero_adequate
  exact adequateIterationBudget_zero_of_initialGap_le_target energy center poseAt basin conv eps hgap

theorem rmsd_target_of_canonicalAdequateIterationBudget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    rmsd
      (poseAt (canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps))
      center ≤ eps := by
  have heps_nonneg : 0 ≤ eps := le_of_lt heps
  unfold canonicalAdequateIterationBudget
  exact rmsd_target_of_leastAdequateIterationBudget energy center poseAt basin conv hn eps
    heps_nonneg (exists_adequateIterationBudget energy center poseAt basin conv hn eps heps)

theorem canonicalAdequateIterationBudget_le_logarithmicIterationBound
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (hq_pos : 0 < conv.q) :
    canonicalAdequateIterationBudget energy center poseAt basin conv hn eps heps ≤
      logarithmicIterationBound conv.q (energy (poseAt 0) - energy center) (targetEnergyGap basin.μ n eps) := by
  unfold canonicalAdequateIterationBudget
  apply leastAdequateIterationBudget_minimal
  exact adequateIterationBudget_of_logarithmicIterationBound energy center poseAt basin conv eps hq_pos
    (targetEnergyGap_pos basin.μ n eps basin.μ_pos hn heps)

theorem rmsd_target_of_logarithmicIterationBound
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (conv : CertifiedLinearEnergyConvergence (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (hq_pos : 0 < conv.q) :
    rmsd
      (poseAt (logarithmicIterationBound conv.q (energy (poseAt 0) - energy center)
        (targetEnergyGap basin.μ n eps))) center ≤ eps := by
  have hAdeq := adequateIterationBudget_of_logarithmicIterationBound
    energy center poseAt basin conv eps hq_pos
    (targetEnergyGap_pos basin.μ n eps basin.μ_pos hn heps)
  apply rmsd_target_of_adequateIterationBudget energy center poseAt basin conv hn eps (le_of_lt heps)
  exact hAdeq

/-- Fully derived canonical iteration budget obtained from local segment-curvature
    and one-step contraction certificates. This is the main implementation-facing
    quantity: once these two local certificates are available, the refinement budget
    is mechanically determined. -/
noncomputable def canonicalIterationBudgetFromLocalCertificates
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (step : CertifiedOneStepEnergyContraction (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) : ℕ :=
  canonicalAdequateIterationBudget
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    step.toCertifiedLinearEnergyConvergence hn eps heps

theorem canonicalIterationBudgetFromLocalCertificates_spec
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (step : CertifiedOneStepEnergyContraction (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    AdequateIterationBudget
      energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
      step.toCertifiedLinearEnergyConvergence eps
      (canonicalIterationBudgetFromLocalCertificates energy center poseAt curv step hn eps heps) := by
  unfold canonicalIterationBudgetFromLocalCertificates
  exact canonicalAdequateIterationBudget_spec
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    step.toCertifiedLinearEnergyConvergence hn eps heps

theorem canonicalIterationBudgetFromLocalCertificates_optimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (step : CertifiedOneStepEnergyContraction (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (setupCost stepCost : ℝ)
    (hstepCost : 0 ≤ stepCost)
    {t : ℕ}
    (hAdeqT : AdequateIterationBudget
      energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
      step.toCertifiedLinearEnergyConvergence eps t) :
    refinementCost setupCost stepCost
      (canonicalIterationBudgetFromLocalCertificates energy center poseAt curv step hn eps heps) ≤
    refinementCost setupCost stepCost t := by
  unfold canonicalIterationBudgetFromLocalCertificates
  exact canonicalAdequateIterationBudget_optimal
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    step.toCertifiedLinearEnergyConvergence hn eps heps setupCost stepCost hstepCost hAdeqT

theorem rmsd_target_of_canonicalIterationBudgetFromLocalCertificates
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (step : CertifiedOneStepEnergyContraction (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    rmsd
      (poseAt (canonicalIterationBudgetFromLocalCertificates energy center poseAt curv step hn eps heps))
      center ≤ eps := by
  unfold canonicalIterationBudgetFromLocalCertificates
  exact rmsd_target_of_canonicalAdequateIterationBudget
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    step.toCertifiedLinearEnergyConvergence hn eps heps

/-- The same canonical budget, packaged directly from a concrete certified
    gradient-descent dynamics certificate instead of the lower-level one-step
    contraction certificate. -/
noncomputable def canonicalIterationBudgetFromGradientDescentDynamics
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) : ℕ :=
  canonicalAdequateIterationBudget
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence hn eps heps

theorem canonicalIterationBudgetFromGradientDescentDynamics_spec
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    AdequateIterationBudget
      energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
      dyn.toCertifiedLinearEnergyConvergence eps
      (canonicalIterationBudgetFromGradientDescentDynamics energy center poseAt curv dyn hn eps heps) := by
  unfold canonicalIterationBudgetFromGradientDescentDynamics
  exact canonicalAdequateIterationBudget_spec
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence hn eps heps

theorem canonicalIterationBudgetFromGradientDescentDynamics_optimal
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps)
    (setupCost stepCost : ℝ)
    (hstepCost : 0 ≤ stepCost)
    {t : ℕ}
    (hAdeqT : AdequateIterationBudget
      energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
      dyn.toCertifiedLinearEnergyConvergence eps t) :
    refinementCost setupCost stepCost
      (canonicalIterationBudgetFromGradientDescentDynamics energy center poseAt curv dyn hn eps heps) ≤
    refinementCost setupCost stepCost t := by
  unfold canonicalIterationBudgetFromGradientDescentDynamics
  exact canonicalAdequateIterationBudget_optimal
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence hn eps heps setupCost stepCost hstepCost hAdeqT

theorem rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 < eps) :
    rmsd
      (poseAt (canonicalIterationBudgetFromGradientDescentDynamics energy center poseAt curv dyn hn eps heps))
      center ≤ eps := by
  unfold canonicalIterationBudgetFromGradientDescentDynamics
  exact rmsd_target_of_canonicalAdequateIterationBudget
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence hn eps heps

noncomputable def logarithmicIterationBoundFromGradientDescentDynamics
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (eps : ℝ) : ℕ :=
  logarithmicIterationBound dyn.params.q (energy (poseAt 0) - energy center)
    (targetEnergyGap curv.μ n eps)

theorem adequateIterationBudget_of_logarithmicIterationBoundFromGradientDescentDynamics
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (hq_pos : 0 < dyn.params.q)
    (heps : 0 < eps) :
    AdequateIterationBudget
      energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
      dyn.toCertifiedLinearEnergyConvergence eps
      (logarithmicIterationBoundFromGradientDescentDynamics energy center poseAt curv dyn eps) := by
  unfold logarithmicIterationBoundFromGradientDescentDynamics
  exact adequateIterationBudget_of_logarithmicIterationBound
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence eps hq_pos
    (targetEnergyGap_pos curv.μ n eps curv.μ_pos hn heps)

theorem rmsd_target_of_logarithmicIterationBoundFromGradientDescentDynamics
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center : CoordSet n)
    (poseAt : ℕ → CoordSet n)
    (curv : CertifiedSegmentCurvature energy center)
    (dyn : CertifiedGradientDescentDynamics (fun t => energy (poseAt t) - energy center))
    (hn : 0 < n)
    (eps : ℝ)
    (hq_pos : 0 < dyn.params.q)
    (heps : 0 < eps) :
    rmsd
      (poseAt (logarithmicIterationBoundFromGradientDescentDynamics energy center poseAt curv dyn eps))
      center ≤ eps := by
  unfold logarithmicIterationBoundFromGradientDescentDynamics
  exact rmsd_target_of_logarithmicIterationBound
    energy center poseAt curv.toCertifiedQuadraticWindow.toCertifiedQuadraticBasin
    dyn.toCertifiedLinearEnergyConvergence hn eps heps hq_pos

end EnergyRMSDConvergence
end Tractability
end DecisionQuotient
