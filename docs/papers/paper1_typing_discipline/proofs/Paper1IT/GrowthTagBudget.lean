import Mathlib.Data.Nat.Log
import Mathlib.Tactic

namespace Ssot
namespace Paper1IT

/-- Required exact tag budget for a cell of occupancy `n`. -/
def requiredTagBits (n : Nat) : Nat :=
  Nat.clog 2 n

@[simp] theorem requiredTagBits_zero : requiredTagBits 0 = 0 := by
  simp [requiredTagBits]

@[simp] theorem requiredTagBits_one : requiredTagBits 1 = 0 := by
  simp [requiredTagBits]

theorem requiredTagBits_eq_zero_of_le_one {n : Nat} (hn : n ≤ 1) :
    requiredTagBits n = 0 := by
  cases n with
  | zero => simp [requiredTagBits]
  | succ n =>
      have : n = 0 := by omega
      subst this
      simp [requiredTagBits]

theorem requiredTagBits_monotone {m n : Nat} (hmn : m ≤ n) :
    requiredTagBits m ≤ requiredTagBits n := by
  unfold requiredTagBits
  exact Nat.clog_mono_right 2 hmn

theorem requiredTagBits_le_of_le_pow {n L : Nat} (hn : n ≤ 2 ^ L) :
    requiredTagBits n ≤ L := by
  unfold requiredTagBits
  exact Nat.clog_le_of_le_pow hn

theorem lt_requiredTagBits_of_pow_lt {n L : Nat} (hL : 2 ^ L < n) :
    L < requiredTagBits n := by
  unfold requiredTagBits
  simpa using (Nat.lt_clog_iff_pow_lt (by norm_num : 1 < 2)).2 hL

theorem one_le_requiredTagBits_of_two_le {n : Nat} (hn : 2 ≤ n) :
    1 ≤ requiredTagBits n := by
  have hpow : 2 ^ 0 < n := by simpa using hn
  simpa using lt_requiredTagBits_of_pow_lt hpow

theorem requiredTagBits_positive_iff_two_le {n : Nat} :
    0 < requiredTagBits n ↔ 2 ≤ n := by
  constructor
  · intro h
    by_contra hn
    have hle : n ≤ 1 := by omega
    rw [requiredTagBits_eq_zero_of_le_one hle] at h
    omega
  · intro hn
    exact one_le_requiredTagBits_of_two_le hn

end Paper1IT
end Ssot
