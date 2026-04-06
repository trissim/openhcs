import Paper4dFrontier.TensorRestriction
import Paper4dFrontier.SymmetryRankWitness
import Paper4dFrontier.RealTreewidthWitnesses
import Paper4dFrontier.FirstTwoIndicator
import DecisionQuotient.Finite
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Classical
open Matrix

abbrev SymK (w R : ℕ) := (w + 1) + (R + 2)

def symTailEmbed (w R : ℕ) : Fin (w + 1) → Fin (SymK w R) := Fin.castAdd (R + 2)
def symFreeEmbed (w R : ℕ) : Fin (R + 2) → Fin (SymK w R) := Fin.natAdd (w + 1)

def symStateEq (w R : ℕ) : Fin (2 + (w + 1)) → Fin (SymK w R) :=
  Fin.append (fun _ : Fin 2 => symFreeEmbed w R 0) (symTailEmbed w R)

def symStateNe (w R : ℕ) : Fin (2 + (w + 1)) → Fin (SymK w R) :=
  Fin.append (fun i : Fin 2 => if i = 0 then symFreeEmbed w R 0 else symFreeEmbed w R 1) (symTailEmbed w R)

def symmetryOnlyUtility (aBound w R : ℕ) : Fin (aBound + 2) → (Fin (2 + (w + 1)) → Fin (SymK w R)) → ℤ :=
  fun a s => if Function.Injective s then 0 else a.1

def symmetryOnlyDP (aBound w R : ℕ) : FiniteDecisionProblem (A := Fin (aBound + 2)) (S := (Fin (2 + (w + 1)) → Fin (SymK w R))) where
  actions := Finset.univ
  states := Finset.univ
  utility := symmetryOnlyUtility aBound w R

theorem symStateEq_not_injective (w R : ℕ) : ¬ Function.Injective (symStateEq w R) := by
  intro hinj
  let i0 : Fin (2 + (w + 1)) := Fin.castAdd (w + 1) (0 : Fin 2)
  let i1 : Fin (2 + (w + 1)) := Fin.castAdd (w + 1) (1 : Fin 2)
  have heq : symStateEq w R i0 = symStateEq w R i1 := by
    simp [symStateEq, i0, i1, symFreeEmbed, Fin.append]
  have h : i0 = i1 := hinj heq
  have h' : (0 : Fin 2) = 1 := by
    exact (Fin.castAdd_inj (n := w + 1)).1 (by simpa [i0, i1] using h)
  exact Fin.zero_ne_one h'

theorem symStateNe_injective (w R : ℕ) : Function.Injective (symStateNe w R) := by
  change Function.Injective (Fin.append (fun i : Fin 2 => if i = 0 then symFreeEmbed w R 0 else symFreeEmbed w R 1) (symTailEmbed w R))
  rw [Fin.append_injective_iff]
  refine ⟨?_, ?_, ?_⟩
  · intro i j h
    fin_cases i <;> fin_cases j
    · rfl
    · exfalso
      have h01 : (0 : Fin (R + 2)) = 1 := by simpa [symFreeEmbed] using h
      exact Fin.zero_ne_one h01
    · exfalso
      have h10 : (1 : Fin (R + 2)) = 0 := by simpa [symFreeEmbed] using h
      exact Fin.zero_ne_one h10.symm
    · rfl
  · exact Fin.castAdd_injective _ _
  · intro i j h
    fin_cases i <;>
      have hval := congrArg Fin.val h <;>
      simp [symTailEmbed, symFreeEmbed] at hval <;>
      have hj : (j : ℕ) < w + 1 := j.2 <;>
      omega

theorem symmetryOnly_not_separable (aBound w R : ℕ) :
    ¬ Nonempty (SeparableUtility (dp := symmetryOnlyDP aBound w R)) := by
  intro h
  rcases h with ⟨hsep⟩
  have h11 : 1 = hsep.actionValue 1 + hsep.stateValue (symStateEq w R) := by
    calc
      1 = (symmetryOnlyDP aBound w R).utility 1 (symStateEq w R) := by
        simp [symmetryOnlyDP, symmetryOnlyUtility, symStateEq_not_injective]
      _ = hsep.actionValue 1 + hsep.stateValue (symStateEq w R) := by
        simpa [symmetryOnlyDP] using hsep.utility_eq 1 (symStateEq w R)
  have h01 : 0 = hsep.actionValue 0 + hsep.stateValue (symStateEq w R) := by
    calc
      0 = (symmetryOnlyDP aBound w R).utility 0 (symStateEq w R) := by
        simp [symmetryOnlyDP, symmetryOnlyUtility, symStateEq_not_injective]
      _ = hsep.actionValue 0 + hsep.stateValue (symStateEq w R) := by
        simpa [symmetryOnlyDP] using hsep.utility_eq 0 (symStateEq w R)
  have h10 : 0 = hsep.actionValue 1 + hsep.stateValue (symStateNe w R) := by
    calc
      0 = (symmetryOnlyDP aBound w R).utility 1 (symStateNe w R) := by
        simp [symmetryOnlyDP, symmetryOnlyUtility, symStateNe_injective]
      _ = hsep.actionValue 1 + hsep.stateValue (symStateNe w R) := by
        simpa [symmetryOnlyDP] using hsep.utility_eq 1 (symStateNe w R)
  have h00 : 0 = hsep.actionValue 0 + hsep.stateValue (symStateNe w R) := by
    calc
      0 = (symmetryOnlyDP aBound w R).utility 0 (symStateNe w R) := by
        simp [symmetryOnlyDP, symmetryOnlyUtility, symStateNe_injective]
      _ = hsep.actionValue 0 + hsep.stateValue (symStateNe w R) := by
        simpa [symmetryOnlyDP] using hsep.utility_eq 0 (symStateNe w R)
  linarith

def symmetryOnlyDimensional (aBound w R : ℕ) : Fin (aBound + 2) → DimensionalStateSpace (SymK w R) (2 + (w + 1)) → ℤ :=
  fun a s => symmetryOnlyUtility aBound w R a s.state

theorem symmetryInjective_permute_iff {k n : ℕ} (σ : CoordinatePermutation n)
    (s : DimensionalStateSpace k n) :
    Function.Injective (s.permute σ).state ↔ Function.Injective s.state := by
  constructor
  · intro h x y hxy
    have hcomp : (s.permute σ).state (σ x) = (s.permute σ).state (σ y) := by
      simpa [DimensionalStateSpace.permute] using hxy
    exact σ.injective (h hcomp)
  · intro h x y hxy
    have hs : s.state (σ.symm x) = s.state (σ.symm y) := by
      simpa [DimensionalStateSpace.permute] using hxy
    have hs' : σ.symm x = σ.symm y := h hs
    exact σ.injective (by simpa using hs')

theorem symmetryOnly_symmetric (aBound w R : ℕ) : SymmetricUtility (symmetryOnlyDimensional aBound w R) := by
  intro σ a s
  unfold symmetryOnlyDimensional symmetryOnlyUtility
  by_cases hs : Function.Injective s.state
  · have hs' : Function.Injective (s.permute σ).state := by
      exact (symmetryInjective_permute_iff σ s).2 hs
    simp [hs, hs']
  · have hs' : ¬ Function.Injective (s.permute σ).state := by
      intro hperm
      exact hs ((symmetryInjective_permute_iff σ s).1 hperm)
    simp [hs, hs']

def symmetryRestrictedWithTail (aBound w R : ℕ) : Unit → (Fin 2 → Fin (SymK w R)) → ℤ :=
  restrictFirstTwoWithTail (u := sliceAction (symmetryOnlyUtility aBound w R) (1 : Fin (aBound + 2))) (symTailEmbed w R)

theorem symmetryRestrictedWithTail_diagonal (aBound w R : ℕ) (x y : Fin (R + 2)) :
    symmetryRestrictedWithTail aBound w R () ![symFreeEmbed w R x, symFreeEmbed w R y] = if x = y then 1 else 0 := by
  unfold symmetryRestrictedWithTail restrictFirstTwoWithTail sliceAction symmetryOnlyUtility extendFirstTwoWithTail
  by_cases hxy : x = y
  · subst hxy
    have hni : ¬ Function.Injective (Fin.append ![symFreeEmbed w R x, symFreeEmbed w R x] (symTailEmbed w R)) := by
      intro hinj
      let i0 : Fin (2 + (w + 1)) := Fin.castAdd (w + 1) (0 : Fin 2)
      let i1 : Fin (2 + (w + 1)) := Fin.castAdd (w + 1) (1 : Fin 2)
      have heq : Fin.append ![symFreeEmbed w R x, symFreeEmbed w R x] (symTailEmbed w R) i0 =
          Fin.append ![symFreeEmbed w R x, symFreeEmbed w R x] (symTailEmbed w R) i1 := by
        simp [i0, i1, Fin.append]
      have h : i0 = i1 := hinj heq
      have h' : (0 : Fin 2) = 1 := by
        exact (Fin.castAdd_inj (n := w + 1)).1 (by simpa [i0, i1] using h)
      exact Fin.zero_ne_one h'
    simp [hni]
  · have hinj : Function.Injective (Fin.append ![symFreeEmbed w R x, symFreeEmbed w R y] (symTailEmbed w R)) := by
      rw [Fin.append_injective_iff]
      refine ⟨?_, Fin.castAdd_injective _ _, ?_⟩
      · intro i j hij
        fin_cases i <;> fin_cases j
        · rfl
        · exfalso; exact hxy (by simpa [symFreeEmbed] using hij)
        · exfalso; exact hxy (by simpa [symFreeEmbed] using hij.symm)
        · rfl
      · intro i j h
        have hj : (j : ℕ) < w + 1 := j.2
        fin_cases i <;>
          have hval := congrArg Fin.val h <;>
          simp [symTailEmbed, symFreeEmbed] at hval <;>
          omega
    simp [hxy, hinj]

noncomputable def symmetryMsmall (aBound w R : ℕ) : Matrix (Fin (R + 2)) (Fin (R + 2)) ℚ :=
  fun x y => ((symmetryRestrictedWithTail aBound w R () ![symFreeEmbed w R x, symFreeEmbed w R y] : ℤ) : ℚ)

noncomputable def symmetryAsmall (aBound w R : ℕ) (decomp : TensorRankDecomposition (symmetryOnlyUtility aBound w R) R) : Matrix (Fin (R + 2)) (Fin R) ℚ :=
  let r := restrictTensorRankFirstTwoWithTail (u := sliceAction (symmetryOnlyUtility aBound w R) (1 : Fin (aBound + 2))) (symTailEmbed w R)
    (sliceActionTensorRank (1 : Fin (aBound + 2)) decomp)
  fun x i => ((r.weight i * r.actionFactor i () * r.coordFactor i 0 (symFreeEmbed w R x) : ℤ) : ℚ)

noncomputable def symmetryBsmall (aBound w R : ℕ) (decomp : TensorRankDecomposition (symmetryOnlyUtility aBound w R) R) : Matrix (Fin R) (Fin (R + 2)) ℚ :=
  let r := restrictTensorRankFirstTwoWithTail (u := sliceAction (symmetryOnlyUtility aBound w R) (1 : Fin (aBound + 2))) (symTailEmbed w R)
    (sliceActionTensorRank (1 : Fin (aBound + 2)) decomp)
  fun i y => ((r.coordFactor i 1 (symFreeEmbed w R y) : ℤ) : ℚ)

theorem symmetry_rank_upper (aBound w R : ℕ) (decomp : TensorRankDecomposition (symmetryOnlyUtility aBound w R) R) :
    (symmetryMsmall aBound w R).rank ≤ R := by
  let r := restrictTensorRankFirstTwoWithTail (u := sliceAction (symmetryOnlyUtility aBound w R) (1 : Fin (aBound + 2))) (symTailEmbed w R)
    (sliceActionTensorRank (1 : Fin (aBound + 2)) decomp)
  let A := symmetryAsmall aBound w R decomp
  let B := symmetryBsmall aBound w R decomp
  have hfac : symmetryMsmall aBound w R = A * B := by
    ext x y
    dsimp [symmetryMsmall, A, B, symmetryAsmall, symmetryBsmall, symmetryRestrictedWithTail]
    rw [Matrix.mul_apply]
    simpa [mul_assoc] using congrArg (fun z : ℤ => (z : ℚ)) (r.decomp () ![symFreeEmbed w R x, symFreeEmbed w R y])
  rw [hfac]
  exact le_trans (Matrix.rank_mul_le_right A B) (Matrix.rank_le_height B)

theorem symmetryOnly_not_low_rank (aBound w R : ℕ) :
    ¬ Nonempty (TensorRankDecomposition (symmetryOnlyUtility aBound w R) R) := by
  intro h
  rcases h with ⟨decomp⟩
  have hRankUpper := symmetry_rank_upper aBound w R decomp
  have hEqDiag : symmetryMsmall aBound w R = diagIndicatorMatrix (R + 2) := by
    ext x y
    simp [symmetryMsmall, symmetryRestrictedWithTail_diagonal, diagIndicatorMatrix, Matrix.one_apply]
  rw [hEqDiag, diagIndicatorMatrix_rank] at hRankUpper
  exact Nat.not_lt_of_ge hRankUpper (by omega)

theorem symmetry_only_witness (aBound w R : ℕ) :
    ∃ u : Fin (aBound + 2) → (Fin (2 + (w + 1)) → Fin (SymK w R)) → ℤ,
      ¬ Nonempty (SeparableUtility (dp := symmetryOnlyDP aBound w R)) ∧
      ¬ Nonempty (TensorRankDecomposition u R) ∧
      SymmetricUtility (symmetryOnlyDimensional aBound w R) ∧
      ¬ realTreewidth_le (InteractionGraph (@completeInteracts (2 + (w + 1))) (@completeInteracts_symm (2 + (w + 1)))) w ∧
      aBound < Fintype.card (Fin (aBound + 2)) := by
  refine ⟨symmetryOnlyUtility aBound w R,
    symmetryOnly_not_separable aBound w R,
    symmetryOnly_not_low_rank aBound w R,
    symmetryOnly_symmetric aBound w R,
    ?_, by simp⟩
  simpa [interactionGraph_completeInteracts_eq_top] using
    completeGraph_not_realTreewidth_le_of_large (n := 2 + (w + 1)) (w := w) (by omega)

end Paper4dFrontier
