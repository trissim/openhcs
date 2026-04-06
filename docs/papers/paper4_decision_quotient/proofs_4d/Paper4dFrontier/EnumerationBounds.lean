import DecisionQuotient.Instances
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Nat.Cast.Order.Basic
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

theorem booleanCube_card (n : ℕ) : Fintype.card (Fin n → Bool) = 2 ^ n := by
  simp [Fintype.card_bool]

/-- A concrete exponential-over-monomial witness. -/
theorem exists_booleanCube_larger_than_monomial (k : ℕ) :
    ∃ n : ℕ, n ^ k < Fintype.card (Fin n → Bool) := by
  let n := 2 ^ (2 * k)
  refine ⟨n, ?_⟩
  have hn : n = 2 ^ (2 * k) := rfl
  rw [booleanCube_card, hn, ← Nat.pow_mul]
  change 2 ^ (2 * k * k) < 2 ^ (2 ^ (2 * k))
  have hexp : 2 * k * k < 2 ^ (2 * k) := by
    have haux : 2 * k ^ 2 + 1 ≤ 2 ^ (2 * k) := Nat.two_mul_sq_add_one_le_two_pow_two_mul k
    have hk2 : 2 * k * k = 2 * k ^ 2 := by
      rw [pow_two, Nat.mul_assoc]
    rw [hk2]
    have hlt : 2 * k ^ 2 < 2 * k ^ 2 + 1 := Nat.lt_succ_self _
    exact lt_of_lt_of_le hlt haux
  exact Nat.pow_lt_pow_right Nat.one_lt_two hexp

theorem exists_scaled_monomial_lt_booleanCube (c k : ℕ) :
    ∃ n : ℕ, 1 ≤ n ∧ c * n ^ k < 2 ^ n := by
  by_cases hc : c = 0
  · refine ⟨1, by decide, ?_⟩
    simp [hc]
  · let m := c + k + 1
    let n := 2 ^ (4 * m)
    have hm : 1 ≤ m := by
      dsimp [m]
      omega
    have hn : 1 ≤ n := by
      dsimp [n]
      exact Nat.one_le_two_pow
    refine ⟨n, hn, ?_⟩
    have hc2 : c ≤ 2 ^ c := Nat.le_of_lt c.lt_two_pow_self
    have hnPow : n ^ k = 2 ^ (4 * m * k) := by
      dsimp [n]
      rw [← Nat.pow_mul]
    calc
      c * n ^ k ≤ 2 ^ c * n ^ k := Nat.mul_le_mul_right _ hc2
      _ = 2 ^ (c + 4 * m * k) := by rw [hnPow, ← Nat.pow_add]
      _ < 2 ^ n := by
        have hc_le : c ≤ m := by
          dsimp [m]
          omega
        have hk_le : k ≤ m := by
          dsimp [m]
          omega
        have hmk : 4 * m * k ≤ 4 * m * m := Nat.mul_le_mul_left (4 * m) hk_le
        have hmk' : 4 * m * k ≤ 4 * m ^ 2 := by
          simpa [pow_two, Nat.mul_assoc] using hmk
        have hmquad : m ≤ 4 * m ^ 2 := by
          have h41 : 1 ≤ 4 * m := by
            omega
          calc
            m = m * 1 := by simp
            _ ≤ m * (4 * m) := Nat.mul_le_mul_left _ h41
            _ = 4 * m ^ 2 := by
              rw [pow_two]
              ring_nf
        have h1 : c + 4 * m * k ≤ 8 * m ^ 2 := by
          have hm4 : m + 4 * m ^ 2 ≤ 8 * m ^ 2 := by
            omega
          exact le_trans (Nat.add_le_add hc_le hmk') hm4
        have h2 : 8 * m ^ 2 < 8 * m ^ 2 + 1 := Nat.lt_succ_self _
        have h3 : 8 * m ^ 2 + 1 ≤ 2 ^ (4 * m) := by
          simpa [pow_two, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using
            (Nat.two_mul_sq_add_one_le_two_pow_two_mul (2 * m))
        have hexp : c + 4 * m * k < n := by
          dsimp [n]
          exact lt_of_le_of_lt h1 (lt_of_lt_of_le h2 h3)
        exact Nat.pow_lt_pow_right Nat.one_lt_two hexp

end Paper4dFrontier
