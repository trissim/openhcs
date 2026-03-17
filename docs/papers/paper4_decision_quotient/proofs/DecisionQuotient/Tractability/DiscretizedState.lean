/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/DiscretizedState.lean
  
  Rigorous formalization of discretized thermodynamic microstates.
-/
import DecisionQuotient.Basic
import DecisionQuotient.Tractability.MolecularSrank
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fin.Basic

namespace DecisionQuotient
namespace Tractability
namespace DiscretizedState

open Tractability.MolecularSrank

/-- 
  A bounded integer coordinate on a grid of size 2*N + 1
  representing roughly the span [-N * resolution, N * resolution].
-/
@[reducible] def BoundedGrid (N : Nat) := Fin (2 * N + 1)

/--
  A discretized atomic position mapped to a 3D numerical grid.
-/
structure BoundedGridAtom (N N_Atoms : Nat) where
  index : Fin N_Atoms
  posGrid : BoundedGrid N × BoundedGrid N × BoundedGrid N

/-- Map internal Equiv for Fintype derivation -/
def boundedGridAtomEquiv (N N_Atoms : Nat) : 
    BoundedGridAtom N N_Atoms ≃ (Fin N_Atoms × (BoundedGrid N × BoundedGrid N × BoundedGrid N)) where
  toFun a := (a.index, a.posGrid)
  invFun p := { index := p.1, posGrid := p.2 }
  left_inv _ := rfl
  right_inv _ := rfl

instance (N N_Atoms : Nat) : Fintype (BoundedGridAtom N N_Atoms) := 
  Fintype.ofEquiv _ (boundedGridAtomEquiv N N_Atoms).symm

/--
  A completely discretized MDState where all atomic coordinates lie on a finite grid of size N.
-/
structure GridMDState (NP NL N : Nat) where
  proteinAtoms : (Fin NP) → BoundedGridAtom N NP
  ligandAtoms : (Fin NL) → BoundedGridAtom N NL

instance (NP NL N : Nat) : Fintype (GridMDState NP NL N) := by
  let E : GridMDState NP NL N ≃ (((Fin NP) → BoundedGridAtom N NP) × ((Fin NL) → BoundedGridAtom N NL)) :=
    { toFun := fun s => (s.proteinAtoms, s.ligandAtoms),
      invFun := fun p => { proteinAtoms := p.1, ligandAtoms := p.2 },
      left_inv := fun _ => rfl,
      right_inv := fun _ => rfl }
  exact Fintype.ofEquiv _ E.symm

/-- Convert a 1D grid coordinate to continuous space using `res` as physical spacing (e.g. 0.1 Å). -/
noncomputable def gridToContinuous (N : Nat) (res : ℝ) (gPos : BoundedGrid N) : ℝ :=
  (gPos.val : ℝ) * res - (N : ℝ) * res

noncomputable def liftGridAtom {N NA : Nat} (res : ℝ) (ga : BoundedGridAtom N NA) : Atom :=
  { index := ga.index.val,
    position := (gridToContinuous N res ga.posGrid.1, 
                 gridToContinuous N res ga.posGrid.2.1, 
                 gridToContinuous N res ga.posGrid.2.2),
    charge := 0, -- Static assumed for structural proofs
    mass := 12 } -- Carbon mass assumed for simplicity

noncomputable def liftGridState {NP NL N : Nat} (res : ℝ) (gs : GridMDState NP NL N) : MDState :=
  { protein := (List.finRange NP).map (fun i => liftGridAtom res (gs.proteinAtoms i)),
    ligand := (List.finRange NL).map (fun i => liftGridAtom res (gs.ligandAtoms i)) }

end DiscretizedState
end Tractability
end DecisionQuotient
