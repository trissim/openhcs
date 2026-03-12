/-
  SSOT Formalization - Information Theory: Entropy Foundations
  Paper 2: Formal Foundations for the Single Source of Truth Principle

  This file formalizes Shannon entropy for finite discrete distributions.
  It provides the foundation for information-theoretic arguments in paper2_it.

  Design principle: Handle edge cases explicitly (0 * log 0 = 0 convention)
  and use standard Mathlib definitions for real analysis.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

/-!
## Finite Probability Distributions

A finite probability distribution over n outcomes is a function from Fin n to ℝ
assigning non-negative probabilities that sum to 1.
-/

/-- Finite probability distribution over n outcomes -/
structure FiniteDist (n : ℕ) where
  prob : Fin n → ℝ
  nonneg : ∀ i, 0 ≤ prob i
  sum_one : ∑ i, prob i = 1

namespace FiniteDist

/-- Support of a distribution: indices with positive probability -/
noncomputable def support {n : ℕ} (p : FiniteDist n) : Finset (Fin n) :=
  {i : Fin n | 0 < p.prob i}

/-- Cardinality of the support -/
noncomputable def supportSize {n : ℕ} (p : FiniteDist n) : ℕ :=
  p.support.card

/-- A distribution is deterministic if it has exactly one outcome with probability 1 -/
def isDeterministic {n : ℕ} (p : FiniteDist n) : Prop :=
  ∃ i₀ : Fin n, p.prob i₀ = 1 ∧ ∀ i ≠ i₀, p.prob i = 0

/-- Uniform distribution over n outcomes -/
noncomputable def uniform (n : ℕ) (hn : n > 0) : FiniteDist n where
  prob i := (1 : ℝ) / n
  nonneg i := by
    have h : 0 < (1 : ℝ) / n := by
      apply div_pos
      · norm_num
      · exact Nat.cast_pos.mpr hn
    exact le_of_lt h
  sum_one := by
    rw [Finset.sum_const, Finset.card_fin]
    have hn0 : (n : ℝ) ≠ 0 := by
      exact_mod_cast (Nat.ne_of_gt hn)
    have hmul : (n : ℝ) * ((1 : ℝ) / n) = 1 := by
      field_simp [hn0]
    simpa [nsmul_eq_mul] using hmul

end FiniteDist

/-!
## Shannon Entropy

Shannon entropy H(X) = -∑ p_i * log(p_i), where we use the convention
that 0 * log(0) = 0 (limit as p → 0⁺).
-/

/-- The term p * log(p) with the convention 0 * log(0) = 0 -/
noncomputable def entropyTerm (p : ℝ) : ℝ :=
  if p = 0 then 0 else p * Real.log p

/-- Shannon entropy of a finite distribution -/
noncomputable def entropy {n : ℕ} (p : FiniteDist n) : ℝ :=
  -∑ i, entropyTerm (p.prob i)

/-!
## Basic Properties of Entropy
-/

namespace Entropy

/--
Classical entropy facts used as external assumptions.

These correspond to standard information-theoretic results (Jensen/Fano toolchain)
that are used conditionally by the paper-level closure theorems.
-/
class ClassicalEntropyAssumptions : Prop where
  entropy_nonneg_ax : ∀ {n : ℕ} (p : FiniteDist n), 0 ≤ entropy p
  entropy_le_log_card_ax : ∀ {n : ℕ} (p : FiniteDist n),
      entropy p ≤ Real.log (p.supportSize : ℝ)
  entropy_uniform_ax : ∀ {n : ℕ} (hn : n > 0),
      entropy (FiniteDist.uniform n hn) = Real.log n
  entropy_zero_iff_ax : ∀ {n : ℕ} (p : FiniteDist n),
      entropy p = 0 ↔ p.isDeterministic

variable [ClassicalEntropyAssumptions]

/-- Entropy is non-negative (conditional on `ClassicalEntropyAssumptions`). -/
theorem entropy_nonneg {n : ℕ} (p : FiniteDist n) : 0 ≤ entropy p := by
  exact ClassicalEntropyAssumptions.entropy_nonneg_ax p

/-- Entropy is bounded by log of support size (conditional). -/
theorem entropy_le_log_card {n : ℕ} (p : FiniteDist n) :
    entropy p ≤ Real.log (p.supportSize : ℝ) := by
  exact ClassicalEntropyAssumptions.entropy_le_log_card_ax p

/-- Entropy of the uniform distribution (conditional). -/
theorem entropy_uniform {n : ℕ} (hn : n > 0) :
    entropy (FiniteDist.uniform n hn) = Real.log n := by
  exact ClassicalEntropyAssumptions.entropy_uniform_ax hn

/--
Helper: for nonnegative probabilities, `entropyTerm p = 0` iff `p = 0` or `p = 1`.

Without the nonnegativity premise, `Real.log` also vanishes at `p = -1`.
-/
theorem entropyTerm_zero_iff {p : ℝ} (hp : 0 ≤ p) :
    entropyTerm p = 0 ↔ p = 0 ∨ p = 1 := by
  by_cases h0 : p = 0
  · simp [entropyTerm, h0]
  · have hp_pos : 0 < p := lt_of_le_of_ne hp (Ne.symm h0)
    constructor
    · intro h
      have hlog : Real.log p = 0 := by
        exact (mul_eq_zero.mp (by simpa [entropyTerm, h0] using h)).resolve_left h0
      right
      exact Real.eq_one_of_pos_of_log_eq_zero hp_pos hlog
    · intro h
      rcases h with h' | h'
      · contradiction
      · subst h'
        simp [entropyTerm]

/-- Entropy is zero iff distribution is deterministic (conditional). -/
theorem entropy_zero_iff {n : ℕ} (p : FiniteDist n) :
    entropy p = 0 ↔ p.isDeterministic := by
  exact ClassicalEntropyAssumptions.entropy_zero_iff_ax p

end Entropy
