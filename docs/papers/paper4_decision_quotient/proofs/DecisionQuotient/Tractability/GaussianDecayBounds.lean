/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/GaussianDecayBounds.lean

  Optimal cutoff derivation for Gaussian-decay potentials (contact surrogate).
  
  The contact surrogate uses exp(-(βr)²) decay, which falls off faster than
  exponential. This gives the cutoff formula: R = √(ln(W/ε)) / β
  
  For β=0.6 Å⁻¹ and target error ε=0.001 with unit weights (W=1):
    R = √(ln(1000)) / 0.6 ≈ √6.9 / 0.6 ≈ 4.4 Å
  
  The default 6.0 Å cutoff provides ~36% safety margin.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace DecisionQuotient
namespace Tractability
namespace GaussianDecayBounds

open Real

/-! ### Core Gaussian Decay Functions -/

/-- Gaussian decay potential: w × exp(-(βr)²) -/
noncomputable def gaussianScore (w β r : ℝ) : ℝ :=
  w * exp (-(β * r) ^ 2)

/-- Hard-cutoff Gaussian score -/
noncomputable def cutoffGaussianScore (w β R r : ℝ) : ℝ :=
  if r < R then gaussianScore w β r else 0

/-! ### Error Bounds -/

/-- The tail error from cutting off at R is bounded by |w| × exp(-(βR)²) -/
theorem gaussian_tail_bound
    (w β R r : ℝ) (hβ_pos : 0 < β) (hR_pos : 0 < R) (hr_ge_R : R ≤ r) :
    |gaussianScore w β r| ≤ |w| * exp (-(β * R) ^ 2) := by
  unfold gaussianScore
  rw [abs_mul]
  have h_exp_nonneg : 0 ≤ exp (-(β * r) ^ 2) := exp_pos _ |>.le
  rw [abs_of_nonneg h_exp_nonneg]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg w)
  apply exp_le_exp_of_le
  have h1 : (β * R) ^ 2 ≤ (β * r) ^ 2 := by
    have hr_nonneg : 0 ≤ r := le_trans (le_of_lt hR_pos) hr_ge_R
    have hβR_nonneg : 0 ≤ β * R := mul_nonneg (le_of_lt hβ_pos) (le_of_lt hR_pos)
    have hβr_nonneg : 0 ≤ β * r := mul_nonneg (le_of_lt hβ_pos) hr_nonneg
    have hβr_ge_βR : β * R ≤ β * r := mul_le_mul_of_nonneg_left hr_ge_R (le_of_lt hβ_pos)
    exact sq_le_sq' (by linarith) hβr_ge_βR
  linarith

/-- Error bound: W × exp(-(βR)²) ≤ ε when R ≥ √(ln(W/ε)) / β -/
theorem gaussian_exp_bound
    (W ε β R : ℝ) (hW_pos : 0 < W) (hε_pos : 0 < ε) (hβ_pos : 0 < β)
    (hR_bound : sqrt (log (W / ε)) / β ≤ R) :
    W * exp (-(β * R) ^ 2) ≤ ε := by
  have h_arg_pos : 0 < W / ε := by positivity
  -- Need: (β * R)² ≥ ln(W/ε)
  have h_sq_bound : log (W / ε) ≤ (β * R) ^ 2 := by
    have h1 : sqrt (log (W / ε)) ≤ β * R := by
      calc sqrt (log (W / ε)) = sqrt (log (W / ε)) / β * β := by field_simp
           _ ≤ R * β := by exact mul_le_mul_of_nonneg_right hR_bound (le_of_lt hβ_pos)
           _ = β * R := by ring
    by_cases h_log_nonneg : 0 ≤ log (W / ε)
    · have h2 : (sqrt (log (W / ε))) ^ 2 = log (W / ε) := sq_sqrt h_log_nonneg
      have hR_nonneg : 0 ≤ R := by
        have hsqrt_nonneg : 0 ≤ sqrt (log (W / ε)) / β := by positivity
        linarith
      have hβR_nonneg : 0 ≤ β * R := mul_nonneg (le_of_lt hβ_pos) hR_nonneg
      calc log (W / ε) = (sqrt (log (W / ε))) ^ 2 := h2.symm
           _ ≤ (β * R) ^ 2 := by
               apply sq_le_sq' _ h1
               have hsqrt_nonneg : 0 ≤ sqrt (log (W / ε)) := sqrt_nonneg _
               linarith
    · push_neg at h_log_nonneg
      have : (β * R) ^ 2 ≥ 0 := sq_nonneg _
      linarith
  -- exp(-(βR)²) ≤ exp(-ln(W/ε)) = ε/W
  have h_exp_bound : exp (-(β * R) ^ 2) ≤ ε / W := by
    calc exp (-(β * R) ^ 2)
         ≤ exp (-log (W / ε)) := by apply exp_le_exp_of_le; linarith
       _ = (exp (log (W / ε)))⁻¹ := exp_neg _
       _ = (W / ε)⁻¹ := by rw [exp_log h_arg_pos]
       _ = ε / W := by field_simp
  calc W * exp (-(β * R) ^ 2)
       ≤ W * (ε / W) := mul_le_mul_of_nonneg_left h_exp_bound (le_of_lt hW_pos)
     _ = ε := by field_simp

/-! ### Optimal Cutoff Formula -/

/-- Minimum cutoff for Gaussian decay to achieve error ε.
    R_min = √(ln(W/ε)) / β -/
noncomputable def gaussianMinCutoff (W ε β : ℝ) : ℝ :=
  sqrt (log (W / ε)) / β

theorem gaussianMinCutoff_sufficient
    (W ε β : ℝ) (hW_pos : 0 < W) (hε_pos : 0 < ε) (hβ_pos : 0 < β) :
    W * exp (-(β * gaussianMinCutoff W ε β) ^ 2) ≤ ε := by
  exact gaussian_exp_bound W ε β (gaussianMinCutoff W ε β)
    hW_pos hε_pos hβ_pos (le_refl _)

/-- At exactly R = √(ln(W/ε))/β, the error equals ε (tight bound). -/
theorem gaussianMinCutoff_tight
    (W ε β : ℝ) (hW_pos : 0 < W) (hε_pos : 0 < ε) (hβ_pos : 0 < β)
    (h_log_nonneg : 0 ≤ log (W / ε)) :
    W * exp (-(β * gaussianMinCutoff W ε β) ^ 2) = ε := by
  unfold gaussianMinCutoff
  have h_arg_pos : 0 < W / ε := by positivity
  have h_sq : (β * (sqrt (log (W / ε)) / β)) ^ 2 = log (W / ε) := by
    have h1 : β * (sqrt (log (W / ε)) / β) = sqrt (log (W / ε)) := by field_simp
    rw [h1, sq_sqrt h_log_nonneg]
  calc W * exp (-(β * (sqrt (log (W / ε)) / β)) ^ 2)
       = W * exp (-log (W / ε)) := by rw [h_sq]
     _ = W * (W / ε)⁻¹ := by rw [exp_neg, exp_log h_arg_pos]
     _ = W * (ε / W) := by field_simp
     _ = ε := by field_simp

/-- OPTIMALITY: The derived cutoff is the MINIMUM R achieving error ≤ ε.
    Any 0 ≤ R < √(ln(W/ε))/β will have error > ε.
    Requires W > ε (i.e., log(W/ε) > 0) for the cutoff to be positive.
    The R ≥ 0 constraint is physically meaningful: distances are non-negative. -/
theorem gaussianMinCutoff_optimal
    (W ε β R : ℝ) (hW_pos : 0 < W) (hε_pos : 0 < ε) (hβ_pos : 0 < β)
    (hR_nonneg : 0 ≤ R)
    (h_log_pos : 0 < log (W / ε))
    (hR_lt : R < gaussianMinCutoff W ε β) :
    ε < W * exp (-(β * R) ^ 2) := by
  unfold gaussianMinCutoff at hR_lt
  have h_arg_pos : 0 < W / ε := by positivity
  have h_log_nonneg : 0 ≤ log (W / ε) := le_of_lt h_log_pos
  have h_sq_lt : (β * R) ^ 2 < log (W / ε) := by
    have hsqrt_pos : 0 < sqrt (log (W / ε)) := sqrt_pos.mpr h_log_pos
    have h1 : β * R < sqrt (log (W / ε)) := by
      calc β * R = R * β := by ring
           _ < sqrt (log (W / ε)) / β * β := by
               apply mul_lt_mul_of_pos_right hR_lt hβ_pos
           _ = sqrt (log (W / ε)) := by field_simp
    have hβR_nonneg : 0 ≤ β * R := mul_nonneg (le_of_lt hβ_pos) hR_nonneg
    have h2 : (β * R) ^ 2 < (sqrt (log (W / ε))) ^ 2 := by
      apply sq_lt_sq' _ h1
      have hsqrt_nonneg : 0 ≤ sqrt (log (W / ε)) := sqrt_nonneg _
      linarith
    rw [sq_sqrt h_log_nonneg] at h2
    exact h2
  have h_exp_gt : ε / W < exp (-(β * R) ^ 2) := by
    calc ε / W = (W / ε)⁻¹ := by field_simp
         _ = (exp (log (W / ε)))⁻¹ := by rw [exp_log h_arg_pos]
         _ = exp (-log (W / ε)) := (exp_neg _).symm
         _ < exp (-(β * R) ^ 2) := by apply exp_strictMono; linarith
  calc ε = W * (ε / W) := by field_simp
       _ < W * exp (-(β * R) ^ 2) := mul_lt_mul_of_pos_left h_exp_gt hW_pos

end GaussianDecayBounds
end Tractability
end DecisionQuotient

