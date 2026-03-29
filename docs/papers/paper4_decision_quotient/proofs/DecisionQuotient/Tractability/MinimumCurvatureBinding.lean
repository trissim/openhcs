/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/MinimumCurvatureBinding.lean

  Formalizes that small ligands in large pockets have inherently flat energy 
  landscapes, making them ill-conditioned for rigid certificate methods.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace MinimumCurvatureBinding

/-- The maximum curvature any single ligand-pocket atom pair can contribute. -/
def MAX_PAIRWISE_CURVATURE : ℝ := 0.5 

/-- The maximum curvature achievable by a ligand in a pocket is bounded by the 
    number of ligand interactions. For a generic pocket, a small ligand can only 
    interact with a coordination shell of pocket atoms, bounded purely by its geometry. -/
def max_achievable_curvature (n_ligand_atoms : ℕ) : ℝ :=
  (n_ligand_atoms : ℝ) * 12.0 * MAX_PAIRWISE_CURVATURE

/-- If the required minimum curvature for a tractable binding certificate (μ_min) 
    exceeds the maximum achievable curvature for a small ligand, the docking 
    problem is inherently ill-conditioned (flat by construction). -/
def is_ill_conditioned (μ_min : ℝ) (n_ligand_atoms : ℕ) : Prop :=
  max_achievable_curvature n_ligand_atoms < μ_min

/-- 
  Theorem: Minimum Curvature for Binding 

  A small ligand (e.g. <15 atoms) structurally cannot provide enough 
  curvature (e.g. μ_min = 100) to safely constrain rigid docking, ensuring
  the problem is ill-conditioned.
-/
theorem minimum_curvature_of_ligand_pocket
    (n_ligand_atoms n_pocket_atoms : ℕ)
    (μ_min μ_actual : ℝ)
    (hSmallLigand : n_ligand_atoms < 15)
    (_hLargePocket : n_pocket_atoms > 150)
    (_hPhysicalBound : μ_actual ≤ max_achievable_curvature n_ligand_atoms)
    (hReq : μ_min = 100.0) :
    μ_actual < μ_min ∨ is_ill_conditioned μ_min n_ligand_atoms := by
  have hMax : max_achievable_curvature n_ligand_atoms ≤ 14 * 12 * MAX_PAIRWISE_CURVATURE := by
    unfold max_achievable_curvature MAX_PAIRWISE_CURVATURE
    have h1 : (n_ligand_atoms : ℝ) ≤ 14 := by exact_mod_cast (Nat.le_of_lt_succ hSmallLigand)
    linarith
  
  have hMaxVal : 14 * 12 * MAX_PAIRWISE_CURVATURE = 84 := by
    unfold MAX_PAIRWISE_CURVATURE; norm_num
  
  have hBound : max_achievable_curvature n_ligand_atoms < μ_min := by
    rw [hReq]
    linarith

  right
  unfold is_ill_conditioned
  exact hBound

end MinimumCurvatureBinding
end Tractability
end DecisionQuotient
