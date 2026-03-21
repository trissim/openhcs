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
import Mathlib.Analysis.SpecialFunctions.Sqrt
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
-- GEOMETRIC FACTS (proved from Real.sqrt properties)
------------------------------------------------------------

/-- Distance is nonnegative -/
lemma distance_nonneg (p q : Point3) : 0 ≤ distance p q := by
  unfold distance
  exact Real.sqrt_nonneg _

/-- Distance is symmetric -/
lemma distance_sym (p q : Point3) : distance p q = distance q p := by
  unfold distance
  congr 1 <;> ring

/-- Distance zero iff points equal -/
lemma distance_eq_zero_iff (p q : Point3) : distance p q = 0 ↔ p = q := by
  rcases p with ⟨px, py, pz⟩
  rcases q with ⟨qx, qy, qz⟩
  constructor
  · intro h
    unfold distance at h
    have hle : (px - qx)^2 + (py - qy)^2 + (pz - qz)^2 ≤ 0 := by
      have hsqrt := (Real.sqrt_eq_zero').mp h
      exact hsqrt
    have hge : 0 ≤ (px - qx)^2 + (py - qy)^2 + (pz - qz)^2 := by
      nlinarith [sq_nonneg (px - qx), sq_nonneg (py - qy), sq_nonneg (pz - qz)]
    have hsum : (px - qx)^2 + (py - qy)^2 + (pz - qz)^2 = 0 := by
      linarith
    have hx0 : (px - qx)^2 = 0 := by nlinarith
    have hy0 : (py - qy)^2 = 0 := by nlinarith
    have hz0 : (pz - qz)^2 = 0 := by nlinarith
    have hx : px = qx := by nlinarith
    have hy : py = qy := by nlinarith
    have hz : pz = qz := by nlinarith
    simp [hx, hy, hz]
  · intro h
    cases h
    unfold distance
    norm_num

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
