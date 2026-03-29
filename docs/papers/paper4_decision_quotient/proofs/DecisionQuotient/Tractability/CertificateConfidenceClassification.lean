/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CertificateConfidenceClassification.lean

  Certificate confidence classification for energy-gap certificates.

  Given an observed energy gap and the target energy gap derived from
  landscape curvature μ and coordinate dimension n, this file defines
  confidence tiers (HIGH, MEDIUM, LOW, INSUFFICIENT) and proves the
  runtime-facing classification theorems.

  The key insight is that `targetEnergyGap μ n ε = μ·n·ε²/2` depends
  on the curvature μ: flat landscapes (small μ) yield tiny target gaps,
  making even "small" observed gaps relatively large. Conversely, a
  large observed gap on a flat landscape is meaningless if it still
  exceeds the target gap by a large factor — the RMSD guarantee holds
  but the landscape provides little discriminative power.

  These theorems let the runtime classify certificate quality and make
  informed decisions about whether to skip refinement, extend sampling,
  or flag the result as insufficiently constrained.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.Tractability.EnergyRMSDConvergence

namespace DecisionQuotient
namespace Tractability
namespace CertificateConfidenceClassification

open EnergyRMSDConvergence
open Computation.ArrayDSL

-- ═══════════════════════════════════════════════════════════════════
-- §1. Confidence tier definitions
-- ═══════════════════════════════════════════════════════════════════

/-- Confidence tier for a certificate's energy gap relative to a target. -/
inductive ConfidenceTier where
  | HIGH         -- gap ≤ targetGap: RMSD is certified within tolerance
  | MEDIUM       -- targetGap < gap ≤ 10 × targetGap: RMSD is bounded but loose
  | LOW          -- 10 × targetGap < gap ≤ 100 × targetGap: weak certificate
  | INSUFFICIENT -- gap > 100 × targetGap: certificate is practically useless
  deriving DecidableEq, Repr

/-- Classify the ratio of observed gap to target gap into a confidence tier. -/
noncomputable def classifyConfidence (gap targetGap : ℝ) : ConfidenceTier :=
  if gap ≤ targetGap then ConfidenceTier.HIGH
  else if gap ≤ 10 * targetGap then ConfidenceTier.MEDIUM
  else if gap ≤ 100 * targetGap then ConfidenceTier.LOW
  else ConfidenceTier.INSUFFICIENT

-- ═══════════════════════════════════════════════════════════════════
-- §2. Core classification theorems
-- ═══════════════════════════════════════════════════════════════════

/-- CCQ1: A HIGH-confidence certificate implies the RMSD guarantee holds.
    This is a direct consequence of `rmsd_le_of_energyGap_le_target`. -/
theorem high_confidence_implies_rmsd_certified
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (gap : ℝ)
    (hgap_eq : gap = energy x - energy center)
    (hclass : classifyConfidence gap (targetEnergyGap basin.μ n eps) = ConfidenceTier.HIGH) :
    rmsd x center ≤ eps := by
  unfold classifyConfidence at hclass
  split_ifs at hclass with h
  · exact rmsd_le_of_energyGap_le_target energy center x basin hn eps heps (hgap_eq ▸ h)
  all_goals contradiction

/-- CCQ2: A MEDIUM-confidence certificate implies RMSD ≤ √10 · ε.
    The observed gap is at most 10× the target, so RMSD scales by √10. -/
theorem medium_confidence_rmsd_bound
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (gap : ℝ)
    (hgap_eq : gap = energy x - energy center)
    (hclass : classifyConfidence gap (targetEnergyGap basin.μ n eps) = ConfidenceTier.MEDIUM) :
    rmsd x center ≤ Real.sqrt 10 * eps := by
  unfold classifyConfidence at hclass
  split_ifs at hclass with h1 h2
  have hscaled : 10 * targetEnergyGap basin.μ n eps =
      targetEnergyGap basin.μ n (Real.sqrt 10 * eps) := by
    unfold targetEnergyGap
    have hsq : (Real.sqrt 10 * eps) ^ 2 = 10 * eps ^ 2 := by
      rw [mul_pow, Real.sq_sqrt (by norm_num : (10 : ℝ) ≥ 0)]
    rw [hsq]; ring
  have heps' : 0 ≤ Real.sqrt 10 * eps := mul_nonneg (Real.sqrt_nonneg _) heps
  have hgap_le : energy x - energy center ≤ targetEnergyGap basin.μ n (Real.sqrt 10 * eps) := by
    rw [← hscaled, ← hgap_eq]; exact h2
  exact rmsd_le_of_energyGap_le_target energy center x basin hn _ heps' hgap_le

/-- CCQ3: A LOW-confidence certificate implies RMSD ≤ 10 · ε.
    The observed gap is at most 100× the target. -/
theorem low_confidence_rmsd_bound
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (center x : CoordSet n)
    (basin : CertifiedQuadraticBasin energy center)
    (hn : 0 < n)
    (eps : ℝ)
    (heps : 0 ≤ eps)
    (gap : ℝ)
    (hgap_eq : gap = energy x - energy center)
    (hclass : classifyConfidence gap (targetEnergyGap basin.μ n eps) = ConfidenceTier.LOW) :
    rmsd x center ≤ 10 * eps := by
  unfold classifyConfidence at hclass
  split_ifs at hclass with h1 h2 h3
  have hscaled : 100 * targetEnergyGap basin.μ n eps =
      targetEnergyGap basin.μ n (10 * eps) := by
    unfold targetEnergyGap
    have hsq : (10 * eps) ^ 2 = 100 * eps ^ 2 := by ring
    rw [hsq]; ring
  have heps' : 0 ≤ 10 * eps := by linarith
  have hgap_le : energy x - energy center ≤ targetEnergyGap basin.μ n (10 * eps) := by
    rw [← hscaled, ← hgap_eq]; exact h3
  exact rmsd_le_of_energyGap_le_target energy center x basin hn _ heps' hgap_le

/-- CCQ4: Target energy gap is monotone in curvature — flatter landscapes
    (smaller μ) yield smaller target gaps, making the classification
    thresholds tighter. -/
theorem targetEnergyGap_monotone_in_mu
    (n : ℕ)
    (eps : ℝ)
    (_hn : 0 < n)
    (_heps : 0 ≤ eps)
    (μ₁ μ₂ : ℝ)
    (_hμ₁ : 0 < μ₁)
    (hle : μ₁ ≤ μ₂) :
    targetEnergyGap μ₁ n eps ≤ targetEnergyGap μ₂ n eps := by
  unfold targetEnergyGap
  have hn_pos : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
  have h1 : μ₁ * (n : ℝ) ≤ μ₂ * (n : ℝ) := mul_le_mul_of_nonneg_right hle hn_pos
  have h2 : μ₁ * (n : ℝ) * eps ^ 2 ≤ μ₂ * (n : ℝ) * eps ^ 2 := mul_le_mul_of_nonneg_right h1 (sq_nonneg eps)
  exact div_le_div_of_nonneg_right h2 (by norm_num)

/-- CCQ5: Target energy gap scales quadratically in ε — tightening the
    RMSD tolerance by factor k requires k² tighter energy gap. -/
theorem targetEnergyGap_quadratic_in_eps
    (μ : ℝ)
    (n : ℕ)
    (eps k : ℝ)
    (hμ : 0 < μ)
    (hn : 0 < n)
    (hk : 0 < k) :
    targetEnergyGap μ n (eps / k) = targetEnergyGap μ n eps / k ^ 2 := by
  unfold targetEnergyGap
  have hk_ne_zero : k ≠ 0 := ne_of_gt hk
  field_simp

end CertificateConfidenceClassification
end Tractability
end DecisionQuotient
