/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/HBondDonorCapacity.lean

  A donor heavy atom with total capacity `c` should not contribute more than `c`
  in aggregate across its explicit hydrogen directions. Splitting the capacity
  equally over `n` hydrogens gives per-direction strength `c / n` and preserves
  the total donor budget.
-/
import Mathlib

namespace DecisionQuotient
namespace Tractability
namespace HBondDonorCapacity

/-- Per-hydrogen donor share from a total donor capacity and hydrogen count. -/
noncomputable def donorShare (capacity hydrogenCount : ℝ) : ℝ :=
  capacity / hydrogenCount

theorem donorShare_sum_preserves_capacity
    (capacity : ℝ) {n : Nat} (hn : 0 < n) :
    (n : ℝ) * donorShare capacity n = capacity := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hn)
  calc
    (n : ℝ) * donorShare capacity n = (n : ℝ) * (capacity / n) := by
      simp [donorShare]
    _ = capacity := by
      field_simp [hn']

theorem unitCapacity_threeHydrogenShare :
    donorShare 1 3 = (1 : ℝ) / 3 := by
  norm_num [donorShare]

end HBondDonorCapacity
end Tractability
end DecisionQuotient
