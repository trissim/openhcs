/-
  Paper 2: Single Source of Truth

  CrossPaperDependencies.lean

  Paper 2 depends on Paper 1 (Axis, minimality, independence).
  Paper 3 depends on Paper 2 (SSOT, cost_ratio, coherence).

  This file documents:
  - What Paper 2 imports from Paper 1 (with theorem invocations)
  - What Paper 2 exports for Paper 3
  
  NO SORRIES. ALL PROOFS RIGOROUS.
-/

-- Paper 1 imports
import axis_framework

-- Paper 2 internal
import Ssot.SSOT
import Ssot.Bounds
import Ssot.Coherence

namespace Paper2.CrossPaperDeps

open Ssot

/-! ## Paper 2 REQUIRES Paper 1: Axis Theory (RIGOROUS) -/

/-- Paper 2 uses Paper 1's independent axes.
    
    RIGOR: Invokes Paper 1's `independence_of_minimal`.
    In Paper 2 terms: SSOT has one independent axis. -/
theorem ssot_has_independent_axis (A : AxisSet) (D : Domain α)
    (a : Axis) (ha : a ∈ A) (hmin : minimal A D) :
    independent a A D :=
  independence_of_minimal ha hmin

/-- Paper 2 uses Paper 1's minimal sets imply complete.
    
    RIGOR: Uses Paper 1's `minimal` definition which includes completeness. -/
theorem ssot_minimal_implies_complete (A : AxisSet) (D : Domain α)
    (hmin : minimal A D) :
    complete A D := hmin.1

/-! ## What Paper 2 Exports for Paper 3 -/

/-- Paper 2 provides SSOT definition.
    Paper 3 uses this for architecture characterization. -/
theorem ssot_def : satisfies_SSOT dof ↔ dof = 1 := Iff.rfl

/-- Paper 2 proves SSOT implies DOF = 1.
    Paper 3 uses this for leverage bounds. -/
theorem ssot_optimality_export (dof : Nat) (h : satisfies_SSOT dof) :
    dof = 1 := ssot_optimality dof h

/-- Paper 2 provides cost ratio.
    Paper 3 uses this for modification cost analysis. -/
theorem cost_ratio_export (dof : Nat) :
    ssot_cost_ratio dof = dof := cost_ratio_eq_dof dof

/-- Paper 2 proves non-SSOT inconsistency.
    Paper 3 uses this for coherence theorems.
    
    RIGOR: Derives dof ≠ 1 from dof = 0 ∨ dof > 1. -/
theorem non_ssot_inconsistent_export (dof : Nat) (h : ¬satisfies_SSOT dof) :
    dof ≠ 1 := by
  have hdisj := non_ssot_inconsistency dof h
  omega

/-- Paper 2 proves cost ratio grows with DOF.
    Paper 3 uses this for leverage hierarchy. -/
theorem cost_grows_with_dof_export (dof1 dof2 : Nat) (h : dof2 > dof1) :
    ssot_cost_ratio dof2 > ssot_cost_ratio dof1 := by
  rw [cost_ratio_eq_dof, cost_ratio_eq_dof]
  omega

/-- Paper 2 proves SSOT has minimal modification cost.
    Paper 3 uses this for SSOT advantage. -/
theorem ssot_minimal_cost : ssot_modification_cost = 1 := rfl

/-- Paper 2 proves non-SSOT cost scales with DOF.
    Paper 3 uses this for leverage comparison. -/
theorem non_ssot_cost_scales (dof : Nat) :
    non_ssot_modification_cost dof = dof := rfl

end Paper2.CrossPaperDeps
