/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LigandFlexibilityRequirement.lean

  Formalizes the requirement that small ligands without internal flexibility
  must either have flat binding landscapes or require conformer search to
  achieve high-curvature tight binding.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace LigandFlexibilityRequirement

/-- The maximum physically possible curvature per atom for a rigid interaction. -/
def MAX_CURVATURE_PER_ATOM : ℝ := 1.0

/-- The curvature threshold below which a landscape is practically 'flat'. -/
def CURVATURE_THRESHOLD : ℝ := 15.0

/-- Conformer search is evaluated to be required if rotatable bonds must be sampled. -/
def requires_conformer_search (n_rotatable : ℕ) : Prop :=
  n_rotatable > 0

/-- The physical constraint that the total rigid-body curvature is bounded
    by the number of atoms times the maximum per-atom curvature. 
    Any curvature beyond this must come from internal rotatable constraints. -/
def rigid_curvature_bound (n_atoms n_rotatable : ℕ) (μ : ℝ) : Prop :=
  n_rotatable = 0 → μ ≤ (n_atoms : ℝ) * MAX_CURVATURE_PER_ATOM

/-- 
  Theorem: Small Ligand Flexibility Requirement

  For a small ligand (atoms < 15), if it is modeled as rigid, either the resulting 
  energy landscape is flat (μ < CURVATURE_THRESHOLD) or internal rotatable bonds 
  must actually be sampled (requires_conformer_search). 
-/
theorem small_ligand_requires_flexibility
    (n_atoms n_rotatable : ℕ)
    (μ : ℝ)
    (hSmall : n_atoms < 15)
    (hRigidBound : rigid_curvature_bound n_atoms n_rotatable μ) :
    μ < CURVATURE_THRESHOLD ∨ requires_conformer_search n_rotatable := by
  by_cases hRot : n_rotatable > 0
  · right
    exact hRot
  · left
    have hZero : n_rotatable = 0 := by omega
    have hBound := hRigidBound hZero
    have hAtomBound : (n_atoms : ℝ) ≤ 14 := by exact_mod_cast (Nat.le_of_lt_succ hSmall)
    have hMaxCurv : (n_atoms : ℝ) * MAX_CURVATURE_PER_ATOM ≤ 14 * MAX_CURVATURE_PER_ATOM := by
      unfold MAX_CURVATURE_PER_ATOM
      linarith
    unfold CURVATURE_THRESHOLD MAX_CURVATURE_PER_ATOM at *
    linarith

end LigandFlexibilityRequirement
end Tractability
end DecisionQuotient
