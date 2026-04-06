import Paper4dFrontier.EnumerationBounds
import Mathlib.Tactic

namespace Paper4dFrontier

/-- Polynomial evaluation from a list of coefficients in ascending order,
implemented by Horner recursion. -/
def polyEvalList : List ℕ → ℕ → ℕ
  | [], _ => 0
  | c :: cs, n => c + n * polyEvalList cs n

def polyCoeffSumList : List ℕ → ℕ
  | [] => 0
  | c :: cs => c + polyCoeffSumList cs

theorem polyEvalList_le_monomial_bound (cs : List ℕ) {n : ℕ} (hn : 1 ≤ n) :
    polyEvalList cs n ≤ polyCoeffSumList cs * n ^ cs.length := by
  induction cs with
  | nil =>
      simp [polyEvalList, polyCoeffSumList]
  | cons c cs ih =>
      simp [polyEvalList, polyCoeffSumList, List.length]
      have ih' := ih hn
      have hpow : n ^ cs.length ≤ n ^ (cs.length + 1) := by
        simpa [pow_succ] using Nat.le_mul_of_pos_left (Nat.pow_pos (Nat.zero_lt_of_lt hn) _)
      have hmain : n * polyEvalList cs n ≤ polyCoeffSumList cs * n ^ (cs.length + 1) := by
        calc
          n * polyEvalList cs n ≤ n * (polyCoeffSumList cs * n ^ cs.length) := Nat.mul_le_mul_left _ ih'
          _ = polyCoeffSumList cs * n ^ (cs.length + 1) := by ring_nf; rw [Nat.mul_assoc, pow_succ]
      have hc : c ≤ c * n ^ (cs.length + 1) := by
        have hone : 1 ≤ n ^ (cs.length + 1) := Nat.one_le_pow_of_one_le' hn _
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left c hone
      calc
        c + n * polyEvalList cs n ≤ c * n ^ (cs.length + 1) + polyCoeffSumList cs * n ^ (cs.length + 1) :=
          Nat.add_le_add hc hmain
        _ = (c + polyCoeffSumList cs) * n ^ (cs.length + 1) := by rw [Nat.add_mul]

theorem exists_list_polynomial_lt_booleanCube (cs : List ℕ) :
    ∃ n : ℕ, polyEvalList cs n < 2 ^ n := by
  obtain ⟨n, hn, hmono⟩ := exists_scaled_monomial_lt_booleanCube (polyCoeffSumList cs) cs.length
  refine ⟨n, lt_of_le_of_lt ?_ hmono⟩
  exact polyEvalList_le_monomial_bound cs hn

end Paper4dFrontier
