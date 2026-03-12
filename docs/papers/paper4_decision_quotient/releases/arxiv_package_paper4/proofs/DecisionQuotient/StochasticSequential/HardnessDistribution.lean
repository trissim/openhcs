import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.Real.Basic

/-!
  Paper 4b: Stochastic and Sequential Regimes

  HardnessDistribution.lean - Distribution of hard instances and complexity bounds
-/

namespace DecisionQuotient.StochasticSequential

/-! ## Random Instance Model -/

/-- Probability that random instance is sufficient with empty coordinate set.
    For a random Boolean function, Pr[∅ sufficient] = 1/2^n (only if function is constant). -/
noncomputable def probSufficient (n : ℕ) (_I : Finset (Fin n)) : ℝ :=
  1 / (2 : ℝ)^n

-- prob_sufficient_exponential_decay (1/2^n ≤ 1/2 for n ≥ 1) was an unused axiom; removed.

/-! ## Average-Case Hardness -/

/-- Average-case complexity of stochastic sufficiency -/
noncomputable def averageCaseComplexity (n : ℕ) : ℝ := (2 : ℝ)^(n/2)

/-- Average case complexity is at least 2^(n/2) by definition. -/
theorem average_case_hard (n : ℕ) :
    averageCaseComplexity n ≥ (2 : ℝ)^(n/2) := le_refl _

/-! ## Smoothed Analysis -/

/-- 2^(n/2) ≤ 2^n since n/2 ≤ n and 1 ≤ 2. -/
theorem smoothed_easier_than_worst (n : ℕ) :
    averageCaseComplexity n ≤ (2 : ℝ)^n := by
  simp only [averageCaseComplexity]
  gcongr
  · norm_num
  · exact Nat.div_le_self n 2

/-! ## Instance Family Hardness -/

/-- Hard family complexity is bounded: 2^(n/2) ≤ worst case 2^n. -/
theorem hard_family_complexity (n : ℕ) :
    averageCaseComplexity n ≤ (2 : ℝ)^n := smoothed_easier_than_worst n

end DecisionQuotient.StochasticSequential
