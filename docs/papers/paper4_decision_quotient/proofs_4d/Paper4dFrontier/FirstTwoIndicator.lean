import DecisionQuotient.Tractability.Dimensional
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient

def firstTwoEqDimensional (k m : ℕ) : Unit → DimensionalStateSpace k (2 + (m + 1)) → ℤ :=
  fun _ s => if s.state 0 = s.state 1 then 1 else 0

theorem firstTwoEq_not_symmetric (m R : ℕ) :
    ¬ SymmetricUtility (firstTwoEqDimensional (R + 2) m) := by
  intro hsym
  let n := 2 + (m + 1)
  let i0 : Fin n := ⟨0, by omega⟩
  let i1 : Fin n := ⟨1, by omega⟩
  let i2 : Fin n := ⟨2, by omega⟩
  have hi10 : i1 ≠ i0 := by
    intro h
    have h' := congrArg Fin.val h
    norm_num at h'
  have hi12 : i1 ≠ i2 := by
    intro h
    have h' := congrArg Fin.val h
    norm_num at h'
  let σ : CoordinatePermutation n := Equiv.swap i0 i2
  let s : DimensionalStateSpace (R + 2) n :=
    ⟨fun i => if i = i0 then 0 else if i = i1 then 0 else if i = i2 then 1 else 0⟩
  have h0i : (0 : Fin n) = i0 := by
    change (0 : Fin n) = ⟨0, by omega⟩
    rfl
  have h1i : (1 : Fin n) = i1 := by
    apply Fin.ext
    have hlt : 1 < n := by
      dsimp [n]
      omega
    simp [i1, n, Nat.mod_eq_of_lt hlt]
  have hs_i0 : s.state i0 = (0 : Fin (R + 2)) := by
    unfold s
    simp [i0]
  have hs_i1 : s.state i1 = (0 : Fin (R + 2)) := by
    unfold s
    simp [i0, i1]
  have hs_i2 : s.state i2 = (1 : Fin (R + 2)) := by
    unfold s
    simp [i0, i1, i2]
  have hs : firstTwoEqDimensional (R + 2) m () s = 1 := by
    unfold firstTwoEqDimensional
    rw [h0i, h1i, hs_i0, hs_i1]
    norm_num
  have hs0 : (s.permute σ).state i0 = (1 : Fin (R + 2)) := by
    change s.state (σ.symm i0) = (1 : Fin (R + 2))
    have hσ0 : σ.symm i0 = i2 := by simpa [σ] using (Equiv.swap_apply_left i0 i2)
    rw [hσ0]
    exact hs_i2
  have hs1 : (s.permute σ).state i1 = (0 : Fin (R + 2)) := by
    change s.state (σ.symm i1) = (0 : Fin (R + 2))
    have hσ1 : σ.symm i1 = i1 := by simpa [σ] using (Equiv.swap_apply_of_ne_of_ne hi10 hi12)
    rw [hσ1]
    exact hs_i1
  have hsp0 : firstTwoEqDimensional (R + 2) m () (s.permute σ) = 0 := by
    unfold firstTwoEqDimensional
    have : ¬ (s.permute σ).state 0 = (s.permute σ).state 1 := by
      intro h
      rw [h0i, h1i, hs0, hs1] at h
      exact Fin.zero_ne_one h.symm
    split_ifs with hEq
    · exact False.elim (this hEq)
    · rfl
  have hEq := hsym σ () s
  rw [hs, hsp0] at hEq
  omega

end Paper4dFrontier
