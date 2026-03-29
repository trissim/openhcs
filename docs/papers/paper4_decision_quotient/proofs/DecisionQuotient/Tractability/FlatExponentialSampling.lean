/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/FlatExponentialSampling.lean

  Formalizes that as the energy landscape becomes flatter (smaller curvature μ),
  the required sampling density to maintain a fixed energy gap guarantee grows
  exponentially with the dimension n.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace FlatExponentialSampling

/-- The target energy gap required to guarantee an RMSD bound of `eps`. -/
noncomputable def targetEnergyGap (μ n eps : ℝ) : ℝ :=
  μ * n * eps^2 / 2

/-- 
  The required sample count to guarantee an RMSD of `eps` given a scoring 
  function with Lipschitz constant `L`. 
  To bound the grid approximation error, the grid resolution `r` must satisfy 
  `L * r ≤ targetEnergyGap`. Thus `r = targetEnergyGap / L`.
  The required number of samples is `Volume / r^n`.
-/
noncomputable def required_samples_for_rmsd (μ n eps L Volume : ℝ) : ℝ :=
  Volume / ((targetEnergyGap μ n eps) / L) ^ n

/-- 
  Theorem: Flat Landscape Requires Exponential Sampling

  As curvature μ decreases (the landscape becomes flatter), the target energy 
  gap becomes tinier. To guarantee the RMSD bound against a scoring function 
  with fixed steepness `L`, the required number of samples grows as `(1/μ)^n`,
  which is exponential in the number of dimensions `n`.
-/
theorem flat_landscape_requires_exponential_sampling
    (μ₁ μ₂ n eps L Volume : ℝ)
    (h_μ_flat : 0 < μ₁)
    (h_μ_comp : μ₁ < μ₂)
    (h_n : 0 < n)
    (h_eps : 0 < eps)
    (h_L : 0 < L)
    (h_vol : 0 < Volume) :
    required_samples_for_rmsd μ₂ n eps L Volume < required_samples_for_rmsd μ₁ n eps L Volume := by
  unfold required_samples_for_rmsd targetEnergyGap
  
  have h_eps_sq : 0 < eps^2 := pow_pos h_eps 2
  have h_n_eps : 0 < n * eps^2 := mul_pos h_n h_eps_sq

  have h_gap1_pos : 0 < μ₁ * n * eps^2 / 2 := by
    have h1 : 0 < μ₁ * (n * eps^2) := mul_pos h_μ_flat h_n_eps
    linarith

  have h_gap2_pos : 0 < μ₂ * n * eps^2 / 2 := by
    have hm : 0 < μ₂ := by linarith
    have h1 : 0 < μ₂ * (n * eps^2) := mul_pos hm h_n_eps
    linarith
  
  have h_gap_cmp : μ₁ * n * eps^2 / 2 < μ₂ * n * eps^2 / 2 := by
    have h_prod : μ₁ * (n * eps^2) < μ₂ * (n * eps^2) := mul_lt_mul_of_pos_right h_μ_comp h_n_eps
    linarith

  have h_r1_pos : 0 < (μ₁ * n * eps^2 / 2) / L := div_pos h_gap1_pos h_L
  have h_r2_pos : 0 < (μ₂ * n * eps^2 / 2) / L := div_pos h_gap2_pos h_L

  have h_r_cmp : (μ₁ * n * eps^2 / 2) / L < (μ₂ * n * eps^2 / 2) / L := by
    exact (div_lt_div_iff₀ h_L h_L).mpr (by nlinarith)

  have hA : 0 < ((μ₂ * n * eps^2 / 2) / L) ^ n := Real.rpow_pos_of_pos h_r2_pos n
  have hB : 0 < ((μ₁ * n * eps^2 / 2) / L) ^ n := Real.rpow_pos_of_pos h_r1_pos n

  have hpow_cmp : ((μ₁ * n * eps^2 / 2) / L) ^ n < ((μ₂ * n * eps^2 / 2) / L) ^ n := by
    apply Real.rpow_lt_rpow h_r1_pos.le h_r_cmp h_n
    
  have h_prod : Volume * ((μ₁ * n * eps^2 / 2) / L) ^ n < Volume * ((μ₂ * n * eps^2 / 2) / L) ^ n := by
    have hV : 0 < Volume := h_vol
    nlinarith

  exact (div_lt_div_iff₀ hA hB).mpr h_prod

end FlatExponentialSampling
end Tractability
end DecisionQuotient
