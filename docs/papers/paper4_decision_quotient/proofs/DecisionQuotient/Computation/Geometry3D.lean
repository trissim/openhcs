/-
  Paper 4: Decision-Relevant Uncertainty
  Computation/Geometry3D.lean - Shared 3D Euclidean Geometry

  Minimal stable foundation: centralizes Point3 and distance
  without depending on ArrayDSL (which uses Fin n → ℝ).

  Future: can switch to Fin 3 → ℝ representation later
  once all downstream files are aligned.
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
-- TEMPORARY GEOMETRIC FACTS
-- (replace later with real proofs if desired)
------------------------------------------------------------

/-- Triangle inequality for Euclidean distance -/
axiom distance_triangle
  (p q r : Point3) :
  distance p r ≤ distance p q + distance q r

/-- Distance is nonnegative -/
axiom distance_nonneg
  (p q : Point3) :
  0 ≤ distance p q

/-- Distance zero iff points equal -/
axiom distance_eq_zero_iff
  (p q : Point3) :
  distance p q = 0 ↔ p = q

end Geometry3D
end Computation
end DecisionQuotient
