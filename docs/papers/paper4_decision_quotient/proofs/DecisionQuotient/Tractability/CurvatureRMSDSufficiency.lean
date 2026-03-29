/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CurvatureRMSDSufficiency.lean

  Formalizes the insight that flat landscapes (small μ) yield weak certificates.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.Tractability.CertificateConfidenceClassification

namespace DecisionQuotient
namespace Tractability
namespace CurvatureRMSDSufficiency

open CertificateConfidenceClassification
open EnergyRMSDConvergence

/-- 
  For flat landscapes (where curvature μ is bounded by some tiny μ_min), 
  a typical observed energy gap will be classified as INSUFFICIENT,
  showing that certificates are inherently weak on flat landscapes.
-/
theorem flat_landscape_implies_certificate_insufficiency
    (μ μ_min eps gap : ℝ)
    (n : ℕ)
    (hFlat : μ ≤ μ_min)
    (h_mu_min : μ_min < 2 * gap / (100 * (n : ℝ) * eps^2))
    (hn : 0 < (n : ℝ))
    (heps : 0 < eps)
    (hgap : 0 < gap) :
    classifyConfidence gap (targetEnergyGap μ n eps) = ConfidenceTier.INSUFFICIENT := by
  unfold classifyConfidence
  have h_eps_sq : 0 < eps^2 := sq_pos_of_ne_zero (ne_of_gt heps)
  have h_ne_zero : 100 * (n : ℝ) * eps^2 > 0 := by
    apply mul_pos
    · exact mul_pos (by norm_num) hn
    · exact h_eps_sq
  
  have h_mu_bound : μ_min * (100 * (n : ℝ) * eps^2) < 2 * gap := by
    exact (lt_div_iff₀ h_ne_zero).mp h_mu_min

  have h_target_strict : 100 * targetEnergyGap μ n eps < gap := by
    unfold targetEnergyGap
    calc
      100 * (μ * (n : ℝ) * eps^2 / 2)
        = 50 * μ * (n : ℝ) * eps^2 := by ring
      _ ≤ 50 * μ_min * (n : ℝ) * eps^2 := by
        have h_pos : 0 ≤ 50 * (n : ℝ) * eps^2 := by positivity
        nlinarith
      _ = μ_min * (100 * (n : ℝ) * eps^2) / 2 := by ring
      _ < (2 * gap) / 2 := by linarith
      _ = gap := by ring

  split_ifs with h1 h2 h3
  · linarith [h_target_strict, h1]
  · linarith [h_target_strict, h2]
  · linarith [h_target_strict, h3]
  · rfl

end CurvatureRMSDSufficiency
end Tractability
end DecisionQuotient
