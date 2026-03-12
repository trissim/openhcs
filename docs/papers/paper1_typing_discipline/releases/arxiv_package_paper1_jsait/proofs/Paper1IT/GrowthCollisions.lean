import Mathlib.Analysis.Complex.Exponential
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

@[simp] theorem poissonCellCollisionProb_zero : poissonCellCollisionProb 0 = 0 := by
  rw [poissonCellCollisionProb_eq_one_sub_exp_mul_one_add]
  norm_num

theorem poissonCellCollisionProb_nonneg (r : ℝ≥0) : 0 ≤ poissonCellCollisionProb r := by
  rw [poissonCellCollisionProb_eq_one_sub_exp_mul_one_add]
  have hle : (1 : ℝ) + (r : ℝ) ≤ Real.exp (r : ℝ) := by
    simpa [add_comm] using Real.add_one_le_exp (r : ℝ)
  have hexp_pos : 0 < Real.exp (r : ℝ) := Real.exp_pos _
  have hdiv : ((1 : ℝ) + (r : ℝ)) / Real.exp (r : ℝ) ≤ 1 := by
    have hquot := div_le_div_of_nonneg_right hle hexp_pos.le
    simpa using hquot
  have hmul : Real.exp (-(r : ℝ)) * ((1 : ℝ) + (r : ℝ)) ≤ 1 := by
    rw [Real.exp_neg]
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hdiv
  linarith

theorem poissonCellCollisionProb_pos_of_pos {r : ℝ≥0} (hr : 0 < r) :
    0 < poissonCellCollisionProb r := by
  rw [poissonCellCollisionProb_eq_one_sub_exp_mul_one_add]
  have hlt : (1 : ℝ) + (r : ℝ) < Real.exp (r : ℝ) := by
    have hrne : (r : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt hr)
    simpa [add_comm] using Real.add_one_lt_exp hrne
  have hexp_pos : 0 < Real.exp (r : ℝ) := Real.exp_pos _
  have hdiv : ((1 : ℝ) + (r : ℝ)) / Real.exp (r : ℝ) < 1 := by
    have hquot := div_lt_div_of_pos_right hlt hexp_pos
    simpa using hquot
  have hmul : Real.exp (-(r : ℝ)) * ((1 : ℝ) + (r : ℝ)) < 1 := by
    rw [Real.exp_neg]
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hdiv
  linarith

theorem poissonCellCollisionProb_pos_iff {r : ℝ≥0} :
    0 < poissonCellCollisionProb r ↔ 0 < r := by
  constructor
  · intro h
    by_contra hr
    have hzero : r = 0 := by exact le_antisymm (le_of_not_gt hr) (show 0 ≤ r by exact r.2)
    subst hzero
    simpa using h
  · intro hr
    exact poissonCellCollisionProb_pos_of_pos hr

/-- Union-bound proxy obtained by summing the per-cell collision probabilities. -/
noncomputable def poissonCollisionUnionBound {N : Nat} (rates : Fin N → ℝ≥0) : ℝ :=
  Finset.univ.sum (fun i : Fin N => poissonCellCollisionProb (rates i))

theorem poissonCollisionUnionBound_singleton (r : ℝ≥0) :
    poissonCollisionUnionBound (N := 1) (fun _ => r) = poissonCellCollisionProb r := by
  simp [poissonCollisionUnionBound]

theorem poissonCollisionUnionBound_nonneg {N : Nat} (rates : Fin N → ℝ≥0) :
    0 ≤ poissonCollisionUnionBound rates := by
  unfold poissonCollisionUnionBound
  exact Finset.sum_nonneg (by intro i hi; exact poissonCellCollisionProb_nonneg (rates i))

theorem poissonCollisionUnionBound_pos_of_exists_pos {N : Nat} (rates : Fin N → ℝ≥0)
    (hpos : ∃ i, 0 < rates i) : 0 < poissonCollisionUnionBound rates := by
  rcases hpos with ⟨i, hi⟩
  unfold poissonCollisionUnionBound
  have hiPos : 0 < poissonCellCollisionProb (rates i) := poissonCellCollisionProb_pos_of_pos hi
  have hiLe : poissonCellCollisionProb (rates i) ≤ ∑ j : Fin N, poissonCellCollisionProb (rates j) := by
    exact Finset.single_le_sum (s := Finset.univ) (a := i)
      (by intro j hj; exact poissonCellCollisionProb_nonneg (rates j)) (by simp)
  exact lt_of_lt_of_le hiPos hiLe

theorem poissonCollisionUnionBound_eq_sum_formula {N : Nat} (rates : Fin N → ℝ≥0) :
    poissonCollisionUnionBound rates =
      Finset.univ.sum (fun i : Fin N => 1 - Real.exp (-(rates i : ℝ)) * (1 + (rates i : ℝ))) := by
  unfold poissonCollisionUnionBound
  refine Finset.sum_congr rfl ?_
  intro i hi
  simpa using poissonCellCollisionProb_eq_one_sub_exp_mul_one_add (rates i)

end Paper1IT
end Ssot
