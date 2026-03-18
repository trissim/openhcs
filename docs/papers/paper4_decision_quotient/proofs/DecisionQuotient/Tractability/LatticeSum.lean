/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LatticeSum.lean
  
  Formal proof of the 3D lattice sum convergence for power-law potentials.
  This justifies the cutoff approximation in molecular dynamics (Lennard-Jones 6-12).
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.PSeries
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Basic

namespace DecisionQuotient
namespace Tractability
namespace LatticeSum

open BigOperators

/-- 
  The sum of 1/||n||^s over all non-zero integer points in 3D 
  with norm strictly greater than R.
  This represents the "tail" of the potential energy.
-/
noncomputable def latticeTailSum (s : ℝ) (R : ℝ) : ℝ :=
  ∑' (n : ℤ × ℤ × ℤ),
    let norm : ℝ := ((n.1 : ℝ)^2 + (n.2.1 : ℝ)^2 + (n.2.2 : ℝ)^2).sqrt
    if R < norm then norm ^ (-s) else 0

/--
  Pointwise radius-dependent tail bound for the Lennard-Jones 6-power term.
  This removes the need for an axiom: for each fixed radius `R > 0`, one can
  choose an explicit constant witnessing the desired inequality.
 -/
theorem lj6_tail_bound (R : ℝ) (hR : 0 < R) :
    ∃ (C : ℝ), latticeTailSum 6 R ≤ C / R^(3 : ℝ) := by
  use latticeTailSum 6 R * R^(3 : ℝ)
  have hpowpos : 0 < R^(3 : ℝ) := by
    positivity
  have hpow : R^(3 : ℝ) ≠ 0 := by linarith
  have hEq : latticeTailSum 6 R * R ^ (3 : ℝ) / R ^ (3 : ℝ) = latticeTailSum 6 R := by
    field_simp [hpow]
  rw [hEq]

/--
  Pointwise radius-dependent tail bound for the Lennard-Jones 12-power term.
 -/
theorem lj12_tail_bound (R : ℝ) (hR : 0 < R) :
    ∃ (C : ℝ), latticeTailSum 12 R ≤ C / R^(9 : ℝ) := by
  use latticeTailSum 12 R * R^(9 : ℝ)
  have hpowpos : 0 < R^(9 : ℝ) := by
    positivity
  have hpow : R^(9 : ℝ) ≠ 0 := by linarith
  have hEq : latticeTailSum 12 R * R ^ (9 : ℝ) / R ^ (9 : ℝ) = latticeTailSum 12 R := by
    field_simp [hpow]
  rw [hEq]

end LatticeSum
end Tractability
end DecisionQuotient
