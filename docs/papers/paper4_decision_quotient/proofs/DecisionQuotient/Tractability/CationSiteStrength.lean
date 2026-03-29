/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CationSiteStrength.lean

  Residue-level cation surrogate strengths should be tied to formal charge, not
  ad hoc multipliers. For the monocationic lysinium and guanidinium motifs used
  by the runtime, the canonical site strength is `|+1| = 1`.
-/
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace CationSiteStrength

/-- Canonical cation-site strength induced by formal charge magnitude. -/
noncomputable def siteStrengthOfFormalCharge (formalCharge : ℝ) : ℝ :=
  |formalCharge|

/-- Formal charge carried by lysine terminal ammonium. -/
noncomputable def lysiniumFormalCharge : ℝ := 1

/-- Formal charge carried by arginine guanidinium. -/
noncomputable def guanidiniumFormalCharge : ℝ := 1

theorem siteStrengthOfFormalCharge_nonneg (formalCharge : ℝ) :
    0 ≤ siteStrengthOfFormalCharge formalCharge := by
  simp [siteStrengthOfFormalCharge]

theorem siteStrengthOfFormalCharge_unitMonocation :
    siteStrengthOfFormalCharge 1 = 1 := by
  simp [siteStrengthOfFormalCharge]

theorem lysinium_siteStrength_eq_one :
    siteStrengthOfFormalCharge lysiniumFormalCharge = 1 := by
  simpa [lysiniumFormalCharge] using siteStrengthOfFormalCharge_unitMonocation

theorem guanidinium_siteStrength_eq_one :
    siteStrengthOfFormalCharge guanidiniumFormalCharge = 1 := by
  simpa [guanidiniumFormalCharge] using siteStrengthOfFormalCharge_unitMonocation

end CationSiteStrength
end Tractability
end DecisionQuotient
