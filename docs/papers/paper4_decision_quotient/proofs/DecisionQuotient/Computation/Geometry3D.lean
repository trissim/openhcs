/-
  Paper 4: Decision-Relevant Uncertainty
  Computation/Geometry3D.lean - Shared 3D Euclidean Geometry

  SHARED FOUNDATION: All geometry files depend on this.
  - Geometry3D.lean: shared Point3, distance, vector ops + axioms
  - ImplicitSurface.lean: defines concavity and surface regions
  - BoundingSphere.lean: enclosing spheres over shared Point3
  - PocketDetection.lean: bridges concave regions to bounded search regions

  Centralizes Point3 and distance without depending on ArrayDSL
  (which uses Fin n → ℝ). Future: can switch representation later.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient.Computation.Geometry3D

/-- A 3D point in Euclidean space -/
@[reducible]
def Point3 := ℝ × ℝ × ℝ

/-- Euclidean distance in ℝ³ -/
noncomputable def distance (p q : Point3) : ℝ :=
  Real.sqrt (
    (p.1 - q.1)^2 +
    (p.2.1 - q.2.1)^2 +
    (p.2.2 - q.2.2)^2
  )

/-- Vector addition -/
def add (p q : Point3) : Point3 :=
  (p.1 + q.1,
   p.2.1 + q.2.1,
   p.2.2 + q.2.2)

/-- Vector subtraction -/
def sub (p q : Point3) : Point3 :=
  (p.1 - q.1,
   p.2.1 - q.2.1,
   p.2.2 - q.2.2)

/-- Scalar multiplication -/
def smul (c : ℝ) (p : Point3) : Point3 :=
  (c * p.1,
   c * p.2.1,
   c * p.2.2)

/-- Midpoint of two points -/
noncomputable def midpoint (p q : Point3) : Point3 :=
  smul (1/2) (add p q)

------------------------------------------------------------
-- GEOMETRIC FACTS (proved or axiomatized)
------------------------------------------------------------

/-- Distance zero iff points equal -/
axiom distance_eq_zero_iff
  (p q : Point3) :
  distance p q = 0 ↔ p = q

/-- Distance is nonnegative -/
axiom distance_nonneg
  (p q : Point3) :
  0 ≤ distance p q

/-- Positive distance for distinct points -/
lemma distance_pos_of_ne {p q : Point3} (h : p ≠ q) :
    distance p q > 0 := by
  have hnonneg : 0 ≤ distance p q := distance_nonneg p q
  have hne : distance p q ≠ 0 := by
    intro hz
    apply h
    exact (distance_eq_zero_iff p q).mp hz
  have h0ne : 0 ≠ distance p q := by
    simpa [eq_comm] using hne
  exact lt_of_le_of_ne hnonneg h0ne

/-- Triangle inequality for Euclidean distance -/
axiom distance_triangle
  (p q r : Point3) :
  distance p r ≤ distance p q + distance q r

/-- Midpoint distance lemma (left): distance from p to midpoint(p,q) -/
axiom distance_midpoint_left (p q : Point3) :
    distance p (midpoint p q) = distance p q / 2

/-- Midpoint distance lemma (right): distance from q to midpoint(p,q) -/
axiom distance_midpoint_right (p q : Point3) :
    distance q (midpoint p q) = distance p q / 2

end Geometry3D
end Computation
end DecisionQuotient
