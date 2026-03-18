/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/DiscretizedAction.lean

  Finite sampled action objects for discretized docking.
  This is the action-side companion to `DiscretizedState.lean`.
-/
import DecisionQuotient.Tractability.DiscretizedState
import Mathlib.Data.Fintype.Basic

namespace DecisionQuotient
namespace Tractability
namespace DiscretizedAction

open MolecularSrank
open DiscretizedState

/--
  A discretized ligand action where every ligand atom lies on the bounded grid.
  This is the finite action-side analogue of `GridMDState`.
-/
structure GridMDAction (NL N : Nat) where
  ligandAtoms : (Fin NL) → BoundedGridAtom N NL

/-- Equivalence used to derive `Fintype` for `GridMDAction`. -/
def gridMDActionEquiv (NL N : Nat) :
    GridMDAction NL N ≃ ((Fin NL) → BoundedGridAtom N NL) where
  toFun a := a.ligandAtoms
  invFun f := { ligandAtoms := f }
  left_inv _ := rfl
  right_inv _ := rfl

noncomputable instance (NL N : Nat) : DecidableEq (GridMDAction NL N) := by
  classical
  infer_instance

instance (NL N : Nat) : Fintype (GridMDAction NL N) :=
  Fintype.ofEquiv _ (gridMDActionEquiv NL N).symm

/-- Lift a discretized action back into the molecular docking action space. -/
noncomputable def liftGridAction {NL N : Nat}
    (res : ℝ) (ga : GridMDAction NL N) : MDAction :=
  { ligand := (List.finRange NL).map (fun i => liftGridAtom res (ga.ligandAtoms i)) }

/-- The lifted action has exactly one ligand atom per discretized ligand index. -/
theorem liftGridAction_ligand_length {NL N : Nat}
    (res : ℝ) (ga : GridMDAction NL N) :
    (liftGridAction res ga).ligand.length = NL := by
  simp [liftGridAction]

end DiscretizedAction
end Tractability
end DecisionQuotient
