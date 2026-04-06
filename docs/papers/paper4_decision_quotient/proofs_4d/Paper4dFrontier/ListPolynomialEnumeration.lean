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
  induction cs generalizing n with
  | nil =>
      simp [polyEvalList, polyCoeffSumList]
  | cons c cs ih =>
      simp [polyEvalList, polyCoeffSumList, List.length]
      have ih' := ih hn
      have hpow : n ^ cs.length ≤ n ^ (cs.length + 1) := by
        have hpowpos : 0 < n ^ cs.length := by
          rw [Nat.pow_pos_iff]
          exact Or.inl (Nat.zero_lt_of_lt hn)
        simpa [pow_succ, Nat.mul_comm] using
          (Nat.le_mul_of_pos_left (n := n) (m := n ^ cs.length) (Nat.zero_lt_of_lt hn))
      have hmain : n * polyEvalList cs n ≤ polyCoeffSumList cs * n ^ (cs.length + 1) := by
        calc
          n * polyEvalList cs n ≤ n * (polyCoeffSumList cs * n ^ cs.length) := Nat.mul_le_mul_left _ ih'
          _ = polyCoeffSumList cs * n ^ (cs.length + 1) := by ring_nf
      have hc : c ≤ c * n ^ (cs.length + 1) := by
        have hone : 1 ≤ n ^ (cs.length + 1) := by
          have hpowpos : 0 < n ^ (cs.length + 1) := by
            rw [Nat.pow_pos_iff]
            exact Or.inl (Nat.zero_lt_of_lt hn)
          exact Nat.succ_le_of_lt hpowpos
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
