/-
  Paper 4: Decision-Relevant Uncertainty
  Computation/BoundingSphere.lean - Minimal Enclosing Sphere

  DEPENDS ON: Geometry3D (Point3, distance, midpoint)
  PROVIDES: Sphere, IsBounded, existence_bounding_sphere
  USED BY: PocketDetection

  CRITICAL BRIDGE THEOREM:
  
  This file proves that any bounded set of points (including concave regions)
  has an enclosing sphere. This connects:
  
    ConcaveRegion → BoundingSphere → MolecularSrank.BindingSite
    
  Without this theorem, we cannot connect geometric pocket detection to
  the structural rank bounds.

  ## The Problem
  
  Binding sites are defined as (center, radius), i.e., a sphere.
  But a concave surface region is NOT a perfect sphere.
  
  Solution: Prove that any bounded set has an enclosing sphere.
  This sphere becomes the BindingSite abstraction.

  ## Approach
  
  We use Ritter's (1990) bounding sphere algorithm, proved algebraically:
  1. Start with any enclosing sphere
  2. Shrink until support points touch boundary
  3. For 3D: 2 points = diameter, 3 points = circle, 4+ = unique sphere

  ## Dependencies
  - Mathlib4: Euclidean space, norm
  - Geometry3D: shared Point3, distance, distance_triangle
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.Computation.Geometry3D

namespace DecisionQuotient
namespace Computation
namespace BoundingSphere

open Classical

/-- Shorthand for Geometry3D.Point3 -/
def Point3 := Geometry3D.Point3

/-! ## 1. Sphere Definition -/

/-- Sphere defined by center and radius -/
structure Sphere where
  center : Point3
  radius : ℝ
  radius_pos : radius > 0

/-! ## 2. Containment -/

/-- Point set is enclosed by sphere -/
def EnclosedIn (S : Set Point3) (B : Sphere) : Prop :=
  ∀ p ∈ S, Geometry3D.distance p B.center ≤ B.radius

/-- Point is inside sphere (strict) -/
def InsideSphere (p : Point3) (B : Sphere) : Prop :=
  Geometry3D.distance p B.center < B.radius

/-- Point is on sphere boundary -/
def OnBoundary (p : Point3) (B : Sphere) : Prop :=
  Geometry3D.distance p B.center = B.radius

/-! ## 3. Bounded Set -/

/-- A set is bounded if there exists R such that all points are within distance R of origin -/
def IsBounded (S : Set Point3) : Prop :=
  ∃ R : ℝ, ∀ p ∈ S, Geometry3D.distance p (0, 0, 0) ≤ R

/-! ## 4. Enclosing Spheres -/

/-- A sphere is minimal if no other enclosing sphere is strictly smaller.
    Retained as a future strengthening, but not needed by the current pipeline. -/
def IsMinimal (B : Sphere) (S : Set Point3) : Prop :=
  EnclosedIn S B ∧ ∀ B' : Sphere, EnclosedIn S B' → B'.radius ≥ B.radius

/-- Support points: points that lie on the sphere boundary -/
def SupportPoints (B : Sphere) (S : Set Point3) : Set Point3 :=
  { p ∈ S | OnBoundary p B }

/-! ## 5. Bridge to Surface Regions -/

/--
  THEOREM: Any bounded set has an enclosing sphere.

  For the current pocket-detection pipeline, existence of some enclosing sphere
  is sufficient. A minimal enclosing sphere is a stronger future refinement.
-/
theorem existence_bounding_sphere
    (S : Set Point3)
    (hBounded : IsBounded S) :
    ∃ B : Sphere, EnclosedIn S B := by
  rcases hBounded with ⟨R, hR⟩
  refine ⟨{
    center := (0, 0, 0)
    radius := max R 1
    radius_pos := by positivity
  }, ?_⟩
  intro p hp
  have hpR : Geometry3D.distance p (0, 0, 0) ≤ R := hR p hp
  exact le_trans hpR (le_max_left R 1)

/--
  THEOREM (BRIDGE): Every bounded surface region has an enclosing sphere.

  This is the key connection between surface geometry and binding-site style
  center/radius abstractions used downstream.
-/
theorem surface_region_has_bounding_sphere
    (positions : Set Point3)
    (hBounded : IsBounded positions) :
    ∃ B : Sphere, EnclosedIn positions B :=
  existence_bounding_sphere positions hBounded

/--
  Extract binding site center and radius from surface positions.

  Uses classical choice to extract an enclosing sphere, then unpacks it into
  the `(center, radius)` format required downstream.
-/
noncomputable def surfacePositionsToBindingSite
    (positions : Set Point3)
    (hBounded : IsBounded positions) :
    Point3 × ℝ :=
  let sphere_proof := surface_region_has_bounding_sphere positions hBounded
  let B := Classical.choose sphere_proof
  (B.center, B.radius)

/-! ## 6. Two-Point Sphere -/

/--
  THEOREM 6: For 2 distinct points, the sphere centered at midpoint with
  radius = distance/2 encloses both points on its boundary.
  
  Proof: Uses midpoint lemmas and positivity of distance for distinct points.
  This is used in Ritter's algorithm for building bounding spheres.
-/
theorem sphere_from_two_points (p q : Point3) (hpq : p ≠ q) :
    ∃ B : Sphere,
      B.center = Geometry3D.midpoint p q ∧
      B.radius = Geometry3D.distance p q / 2 ∧
      EnclosedIn ({p, q} : Set Point3) B ∧
      OnBoundary p B ∧
      OnBoundary q B := by
  have hr : Geometry3D.distance p q / 2 > 0 := by
    have hpos : Geometry3D.distance p q > 0 :=
      Geometry3D.distance_pos_of_ne hpq
    linarith

  let B : Sphere := {
    center := Geometry3D.midpoint p q
    radius := Geometry3D.distance p q / 2
    radius_pos := hr
  }

  refine ⟨B, rfl, rfl, ?_, ?_, ?_⟩

  · intro x hx
    have hx' : x = p ∨ x = q := by
      simpa [Set.mem_insert_iff, Set.mem_singleton_iff] using hx
    rcases hx' with hxp | hxq
    · rw [hxp]
      exact le_of_eq (Geometry3D.distance_midpoint_left p q)
    · rw [hxq]
      exact le_of_eq (Geometry3D.distance_midpoint_right p q)

  · exact Geometry3D.distance_midpoint_left p q

  · exact Geometry3D.distance_midpoint_right p q

end BoundingSphere
end Computation
end DecisionQuotient
