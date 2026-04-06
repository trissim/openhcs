import DecisionQuotient.Instances
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Nat.Cast.Order.Basic

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

end Paper4dFrontier
