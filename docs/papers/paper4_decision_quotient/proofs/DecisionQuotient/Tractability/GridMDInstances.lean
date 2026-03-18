/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/GridMDInstances.lean

  CoordinateSpace/ProductSpace instances for discretized MD states.
  The construction reuses the existing function-space product instance by
  flattening grid states to finite coordinate functions.
-/
import DecisionQuotient.Instances
import DecisionQuotient.Tractability.DiscretizedState
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace GridMDInstances

open DiscretizedState

/-- Structured coordinates for a discretized MD state. -/
inductive GridCoord (NP NL : Nat)
  | protein (atom : Fin NP) (axis : Fin 3)
  | ligand (atom : Fin NL) (axis : Fin 3)
  deriving DecidableEq, Fintype

/-- Total coordinate count induced by `GridCoord`. -/
@[reducible] def GridCoordCount (NP NL : Nat) : Nat :=
  Fintype.card (GridCoord NP NL)

/-- Project one axis from a discretized 3D grid position. -/
def tripleProj {N : Nat}
    (triple : BoundedGrid N × BoundedGrid N × BoundedGrid N)
    (axis : Fin 3) : BoundedGrid N :=
  if _h0 : axis = 0 then
    triple.1
  else if _h1 : axis = 1 then
    triple.2.1
  else
    triple.2.2

/-- Coordinate projection by structured grid coordinate. -/
def gridCoordProj {NP NL N : Nat}
    (s : GridMDState NP NL N) : GridCoord NP NL → BoundedGrid N
  | .protein atom axis => tripleProj (s.proteinAtoms atom).posGrid axis
  | .ligand atom axis => tripleProj (s.ligandAtoms atom).posGrid axis

/-- Reconstruct a discretized MD state from a structured coordinate function. -/
def coordFunToGridMDState {NP NL N : Nat}
    (g : GridCoord NP NL → BoundedGrid N) : GridMDState NP NL N :=
  { proteinAtoms := fun atom =>
      { index := atom
        posGrid :=
          ( g (.protein atom 0)
          , g (.protein atom 1)
          , g (.protein atom 2) ) }
    ligandAtoms := fun atom =>
      { index := atom
        posGrid :=
          ( g (.ligand atom 0)
          , g (.ligand atom 1)
          , g (.ligand atom 2) ) } }

/-- Reconstruction agrees with structured coordinate projection. -/
theorem gridCoordProj_coordFunToGridMDState {NP NL N : Nat}
    (g : GridCoord NP NL → BoundedGrid N) :
    gridCoordProj (coordFunToGridMDState g) = g := by
  funext coord
  cases coord with
  | protein atom axis =>
      fin_cases axis <;> simp [gridCoordProj, coordFunToGridMDState, tripleProj]
  | ligand atom axis =>
      fin_cases axis <;> simp [gridCoordProj, coordFunToGridMDState, tripleProj]

/-- Flatten a discretized MD state to a finite coordinate function. -/
noncomputable def toFlat {NP NL N : Nat}
    (s : GridMDState NP NL N) : Fin (GridCoordCount NP NL) → BoundedGrid N :=
  fun i => gridCoordProj s ((Fintype.equivFin (GridCoord NP NL)).symm i)

/-- Reconstruct a discretized MD state from a flat coordinate function. -/
noncomputable def fromFlat {NP NL N : Nat}
    (f : Fin (GridCoordCount NP NL) → BoundedGrid N) : GridMDState NP NL N :=
  coordFunToGridMDState (fun coord => f ((Fintype.equivFin (GridCoord NP NL)) coord))

/-- Flattening a reconstructed state recovers the original flat function. -/
theorem toFlat_fromFlat {NP NL N : Nat}
    (f : Fin (GridCoordCount NP NL) → BoundedGrid N) :
    toFlat (fromFlat f) = f := by
  funext i
  unfold toFlat fromFlat
  simpa using congrFun
    (gridCoordProj_coordFunToGridMDState
      (NP := NP) (NL := NL) (N := N)
      (fun coord => f ((Fintype.equivFin (GridCoord NP NL)) coord)))
    ((Fintype.equivFin (GridCoord NP NL)).symm i)

/-- Reconstructing a discretized state from its flat coordinate function
    recovers the original discretized state. This establishes that the
    `toFlat`/`fromFlat` pair are inverses on the canonical representation
    used elsewhere in the proofs. -/
theorem fromFlat_toFlat {NP NL N : Nat}
    (s : GridMDState NP NL N) :
    fromFlat (toFlat s) = s := by
  -- Destructure the state into its component maps for a direct, fieldwise
  -- extensionality proof.
  cases s with
  | mk protein ligand =>
    -- Prove equality of the two function fields produced by `fromFlat` and
    -- the original state by `funext` and then finishing with `GridMDState.mk`'s
    -- injectivity lemma.
    have hprot :
      (fromFlat (toFlat { proteinAtoms := protein, ligandAtoms := ligand })).proteinAtoms =
        protein := by
      funext atom
      -- Compare individual coordinate axes; `fin_cases` exposes the three axes
      -- so `simp` can reduce the definitions to reflexivity.
      funext axis
      fin_cases axis
      simp [fromFlat, toFlat, coordFunToGridMDState, gridCoordProj, tripleProj]

    have hlig :
      (fromFlat (toFlat { proteinAtoms := protein, ligandAtoms := ligand })).ligandAtoms =
        ligand := by
      funext atom
      funext axis
      fin_cases axis
      simp [fromFlat, toFlat, coordFunToGridMDState, gridCoordProj, tripleProj]

    -- Use the generated `mk.injEq` lemma to assemble the field equalities into
    -- the full state equality.
    apply GridMDState.mk.injEq.mpr
    constructor
    · exact hprot
    · exact hlig

theorem toFlat_injective {NP NL N : Nat} :
    Function.Injective (@toFlat NP NL N) := by
  intro s t h
  have := congrArg fromFlat h
  simpa [fromFlat_toFlat] using this

noncomputable instance {NP NL N : Nat} : CoordinateSpace (GridMDState NP NL N) (GridCoordCount NP NL) where
  Coord := fun _ => BoundedGrid N
  proj := fun s i => toFlat s i

noncomputable instance {NP NL N : Nat} : ProductSpace (GridMDState NP NL N) (GridCoordCount NP NL) where
  Coord := fun _ => BoundedGrid N
  proj := fun s i => toFlat s i
  replace := fun s i s' =>
    fromFlat ((functionProductSpace (BoundedGrid N) (GridCoordCount NP NL)).replace
      (toFlat s) i (toFlat s'))
  replace_proj_eq := by
    intro s s' i
    rw [toFlat_fromFlat]
    exact (functionProductSpace (BoundedGrid N) (GridCoordCount NP NL)).replace_proj_eq
      (toFlat s) (toFlat s') i
  replace_proj_ne := by
    intro s s' i j hne
    rw [toFlat_fromFlat]
    exact (functionProductSpace (BoundedGrid N) (GridCoordCount NP NL)).replace_proj_ne
      (toFlat s) (toFlat s') i j hne

/-- The generic irrelevance-erasure theorem applies directly to discretized MD
    states through the `ProductSpace` instance above. -/
theorem gridMDState_sufficient_erase_irrelevant
    {NP NL N : Nat} {A : Type*}
    [DecidableEq (Fin (GridCoordCount NP NL))]
    (dp : DecisionProblem A (GridMDState NP NL N))
    (I : Finset (Fin (GridCoordCount NP NL)))
    (i : Fin (GridCoordCount NP NL))
    (hI : dp.isSufficient I)
    (hirr : dp.isIrrelevant i) :
    dp.isSufficient (I.erase i) :=
  dp.sufficient_erase_irrelevant' I i hI hirr

end GridMDInstances
end Tractability
end DecisionQuotient
