/-
  Paper 4: Decision-Relevant Uncertainty

  Computation/ImplicitSurface.lean - Implicit Protein Surface via Distance Functions
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Topology.Connected.Basic
import DecisionQuotient.Computation.Geometry3D

namespace DecisionQuotient
namespace Computation
namespace ImplicitSurface

open Classical

/-- Shorthand for Geometry3D.Point3 -/
def Point3 := Geometry3D.Point3

instance : TopologicalSpace Point3 := by
  unfold Point3; infer_instance

/-! ## 1. Atomic Coordinates -/

structure Atom where
  position : Point3
  radius : ℝ

abbrev Molecule := List Atom

/-! ## 2. Distance Function -/

/-- Van der Waals distance from molecule to point x.
    Uses the mathematical Infimum to avoid foldl accumulator bugs.
    d(x) = 0 is the true VDW surface.
-/
noncomputable def vdwDistance (mol : Molecule) (x : Point3) : ℝ :=
  sInf { d | ∃ a ∈ mol, d = Geometry3D.distance x a.position - a.radius }

/-! ## 3. Surface Point -/

structure SurfacePoint (mol : Molecule) where
  position : Point3
  isOnSurface : vdwDistance mol position = 0
  definingAtom : Atom
  definingAtomInMol : definingAtom ∈ mol
  definingAtomOnSurface : Geometry3D.distance position definingAtom.position = definingAtom.radius

/-- Outward unit normal at the surface point, derived from its defining atom. -/
noncomputable def outwardNormal {mol : Molecule} (p : SurfacePoint mol) : Point3 :=
  let v := Geometry3D.sub p.position p.definingAtom.position
  Geometry3D.smul (1 / p.definingAtom.radius) v

/-! ## 4. Algebraic Concavity via Rolling Probe -/

/-- Number of atoms intersected by a probe sphere.
    Uses List.filter length, which is fully computable and theorem-friendly.
-/
noncomputable def intersectingAtomCount
    (mol : Molecule)
    (probeCenter : Point3)
    (r_probe : ℝ) : ℕ :=
  (mol.filter (fun atom => Geometry3D.distance probeCenter atom.position < atom.radius + r_probe)).length

/-- A surface point is locally concave if a probe sphere offset along the normal touches ≥ 2 atoms. -/
noncomputable def isLocallyConcave
    {mol : Molecule}
    (p : SurfacePoint mol)
    (r_probe : ℝ) : Prop :=
  ∃ ε : ℝ, ε > 0 ∧
    let probeCenter := Geometry3D.add p.position (Geometry3D.smul ε (outwardNormal p))
    intersectingAtomCount mol probeCenter r_probe ≥ 2

/-! ## 5. Concave Region -/

structure ConcaveRegion (mol : Molecule) (r_probe : ℝ) where
  points : Set (SurfacePoint mol)
  concave : ∀ p ∈ points, isLocallyConcave p r_probe
  /-- Map surface points to R^3 so Mathlib can apply Euclidean topological connectedness -/
  connected : IsConnected (SurfacePoint.position '' points)
  bounded : ∃ (R : ℝ) (c : Point3), ∀ p ∈ points, Geometry3D.distance p.position c ≤ R

/-! ## 6. Key Theorems -/

/--
  AXIOM 3 (REWRITE NEEDED): Bounded concave regions have positive volume.
  This currently says "volume > 0" but a surface has zero volume.
  Should be restated as: "bounded concave regions have nonempty interior"
  or "∃ probe placements that fit in the concave region".
-/
axiom concave_region_has_volume
    {mol : Molecule} {r_probe : ℝ}
    (region : ConcaveRegion mol r_probe) :
    ∃ (vol : ℝ), vol > 0

/-- AXIOM 4: Concavity equivalence - PROVEN (definitional rfl) -/
theorem concavity_equivalence
    {mol : Molecule}
    (p : SurfacePoint mol)
    (r_probe : ℝ) :
    isLocallyConcave p r_probe ↔
    ∃ ε > 0,
      let probeCenter := Geometry3D.add p.position (Geometry3D.smul ε (outwardNormal p))
      intersectingAtomCount mol probeCenter r_probe ≥ 2 := by
  rfl

/--
  AXIOM 5 (REWRITE NEEDED): Ligand accessibility.
  Current statement goes the wrong direction (inside sphere → vdw ≤ 0).
  Should be: "concave regions are accessible to probes of radius ≤ r_probe".
-/
axiom concave_region_is_accessible
    {mol : Molecule} {r_probe : ℝ}
    (region : ConcaveRegion mol r_probe)
    (ligandRadius : ℝ)
    (hLigand : ligandRadius ≤ r_probe) :
    ∀ p ∈ region.points, ∀ q : Point3,
      Geometry3D.distance q p.position ≤ r_probe - ligandRadius →
      vdwDistance mol q ≤ 0

end ImplicitSurface
end Computation
end DecisionQuotient
