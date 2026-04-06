import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.Dimensional
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.RowCol
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Matrix

/-- Diagonal indicator utility on two coordinates. -/
def diagIndicatorUtility (k : ℕ) : Unit → (Fin 2 → Fin k) → ℤ
  | (), s => if s 0 = s 1 then 1 else 0

def diagIndicatorDimensional (k : ℕ) : Unit → DimensionalStateSpace k 2 → ℤ
  | (), s => if s.state 0 = s.state 1 then 1 else 0

theorem diagIndicatorDimensional_symmetric (k : ℕ) :
    SymmetricUtility (diagIndicatorDimensional k) := by
  intro σ a s
  dsimp [diagIndicatorDimensional, DimensionalStateSpace.permute]
  rcases Fin.eq_zero_or_eq_succ (σ.symm 0) with h0 | ⟨j, h0⟩
  · have h1 : σ.symm 1 = 1 := by
      rcases Fin.eq_zero_or_eq_succ (σ.symm 1) with h1z | ⟨j, h1⟩
      · exfalso
        have hEq : (0 : Fin 2) = 1 := σ.symm.injective (by simpa [h0, h1z])
        exact Fin.zero_ne_one hEq
      · fin_cases j
        simpa using h1
    simp [h0, h1]
  · fin_cases j
    have h1 : σ.symm 1 = 0 := by
      rcases Fin.eq_zero_or_eq_succ (σ.symm 1) with h1z | ⟨j', h1⟩
      · simpa using h1z
      · fin_cases j'
        exfalso
        have hEq : (1 : Fin 2) = 0 := σ.symm.injective (by simpa [h0, h1])
        exact Fin.zero_ne_one hEq.symm
    by_cases hs : s.state 0 = s.state 1
    · simp [h0, h1, hs, eq_comm]
    · have hs' : ¬ s.state 1 = s.state 0 := by simpa [eq_comm] using hs
      simp [h0, h1, hs, hs', eq_comm]

def diagIndicatorMatrix (k : ℕ) : Matrix (Fin k) (Fin k) ℚ :=
  1

def diagLeftMatrix {k R : ℕ}
    (decomp : TensorRankDecomposition (diagIndicatorUtility k) R) :
    Matrix (Fin k) (Fin R) ℚ :=
  fun x r => ((decomp.weight r * decomp.actionFactor r () * decomp.coordFactor r 0 x : ℤ) : ℚ)

def diagRightMatrix {k R : ℕ}
    (decomp : TensorRankDecomposition (diagIndicatorUtility k) R) :
    Matrix (Fin R) (Fin k) ℚ :=
  fun r y => (decomp.coordFactor r 1 y : ℚ)

theorem diagIndicatorMatrix_factorization {k R : ℕ}
    (decomp : TensorRankDecomposition (diagIndicatorUtility k) R) :
    diagIndicatorMatrix k = diagLeftMatrix decomp * diagRightMatrix decomp := by
  ext x y
  have h := decomp.decomp () ![x, y]
  simpa [diagIndicatorMatrix, diagLeftMatrix, diagRightMatrix, diagIndicatorUtility,
    Matrix.one_apply, Matrix.mul_apply, Fin.prod_univ_two, mul_assoc] using
    congrArg (fun z : ℤ => (z : ℚ)) h

theorem diagIndicatorMatrix_rank (k : ℕ) : (diagIndicatorMatrix k).rank = k := by
  simpa [diagIndicatorMatrix] using (Matrix.rank_one : Matrix.rank (1 : Matrix (Fin k) (Fin k) ℚ) = k)

theorem coordinate_symmetry_not_imply_low_rank_witness (R : ℕ) :
    ∃ u : Unit → (Fin 2 → Fin (R + 1)) → ℤ,
      SymmetricUtility (diagIndicatorDimensional (R + 1)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) := by
  refine ⟨diagIndicatorUtility (R + 1), diagIndicatorDimensional_symmetric (R + 1), ?_⟩
  intro h
  rcases h with ⟨decomp⟩
  let A := diagLeftMatrix decomp
  let B := diagRightMatrix decomp
  have hfac : diagIndicatorMatrix (R + 1) = A * B := diagIndicatorMatrix_factorization decomp
  have hUpper : (diagIndicatorMatrix (R + 1)).rank ≤ R := by
    rw [hfac]
    exact le_trans (Matrix.rank_mul_le_right A B) (Matrix.rank_le_height B)
  have hLower : R + 1 ≤ (diagIndicatorMatrix (R + 1)).rank := by
    rw [diagIndicatorMatrix_rank]
  exact Nat.not_succ_le_self R (le_trans hLower hUpper)

end Paper4dFrontier
