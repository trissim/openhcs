/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/AromaticRingGeometry.lean

  Geometric derivation of the lateral offset width used for aromatic-face
  interactions. We model a six-membered aromatic ring as a regular hexagon with
  empirical C-C bond length 1.39 A; the canonical lateral scale is then the
  hexagon inradius a * sqrt(3) / 2.
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace DecisionQuotient
namespace Tractability
namespace AromaticRingGeometry

open Real

/-- Inradius of a regular hexagon with side length `sideLength`. -/
noncomputable def regularHexagonInradius (sideLength : ℝ) : ℝ :=
  sideLength * sqrt 3 / 2

/-- Empirical aromatic C-C bond length used for benzene-like ring faces. -/
noncomputable def aromaticCarbonBondLength : ℝ := 1.39

/-- Canonical aromatic-face lateral scale used by pi-face interactions. -/
noncomputable def aromaticFaceOffsetWidth : ℝ :=
  regularHexagonInradius aromaticCarbonBondLength

/-- The aromatic-face offset width is the regular-hexagon inradius induced by
    the empirical aromatic C-C bond length. -/
theorem aromaticFaceOffsetWidth_eq_hexagonInradius :
    aromaticFaceOffsetWidth = aromaticCarbonBondLength * sqrt 3 / 2 := by
  simp [aromaticFaceOffsetWidth, regularHexagonInradius]

theorem aromaticCarbonBondLength_pos : 0 < aromaticCarbonBondLength := by
  norm_num [aromaticCarbonBondLength]

theorem aromaticFaceOffsetWidth_pos : 0 < aromaticFaceOffsetWidth := by
  unfold aromaticFaceOffsetWidth regularHexagonInradius aromaticCarbonBondLength
  positivity

end AromaticRingGeometry
end Tractability
end DecisionQuotient
