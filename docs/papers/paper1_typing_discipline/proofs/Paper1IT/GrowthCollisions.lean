import Mathlib.Probability.Distributions.Poisson

namespace Ssot
namespace Paper1IT

open ProbabilityTheory
open scoped NNReal

/-- Collision probability for a single Poissonized representation cell: occupancy at least two. -/
noncomputable def poissonCellCollisionProb (r : ℝ≥0) : ℝ :=
  1 - poissonPMFReal r 0 - poissonPMFReal r 1

theorem poissonCellCollisionProb_eq_one_sub_exp_mul_one_add
    (r : ℝ≥0) :
    poissonCellCollisionProb r = 1 - Real.exp (-(r : ℝ)) * (1 + (r : ℝ)) := by
  unfold poissonCellCollisionProb poissonPMFReal
  simp
  ring_nf

theorem poissonCellCollisionProb_eq_zero_mass_complement
    (r : ℝ≥0) :
    poissonCellCollisionProb r =
      1 - (poissonPMFReal r 0 + poissonPMFReal r 1) := by
  unfold poissonCellCollisionProb
  ring

/-- Union-bound proxy obtained by summing the per-cell collision probabilities. -/
noncomputable def poissonCollisionUnionBound {N : Nat} (rates : Fin N → ℝ≥0) : ℝ :=
  Finset.univ.sum (fun i : Fin N => poissonCellCollisionProb (rates i))

theorem poissonCollisionUnionBound_singleton (r : ℝ≥0) :
    poissonCollisionUnionBound (N := 1) (fun _ => r) = poissonCellCollisionProb r := by
  simp [poissonCollisionUnionBound]

theorem poissonCollisionUnionBound_eq_sum_formula {N : Nat} (rates : Fin N → ℝ≥0) :
    poissonCollisionUnionBound rates =
      Finset.univ.sum (fun i : Fin N => 1 - Real.exp (-(rates i : ℝ)) * (1 + (rates i : ℝ))) := by
  unfold poissonCollisionUnionBound
  refine Finset.sum_congr rfl ?_
  intro i hi
  simpa using poissonCellCollisionProb_eq_one_sub_exp_mul_one_add (rates i)

end Paper1IT
end Ssot
