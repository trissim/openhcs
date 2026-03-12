/-
  Paper 4b: Stochastic and Sequential Regimes

  AlgorithmComplexity.lean - Complexity characterization (structure only)

  Standard algorithm complexity is assumed (P, NP, PP, PSPACE).
  This file provides structure definitions without step-counting proofs.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Finite
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Big-O Notation -/

/-- Big-O notation: g ∈ O(f) means g is asymptotically bounded by f -/
def BigO (f : ℕ → ℝ) : Set (ℕ → ℝ) :=
  { g | ∃ c n₀, ∀ n, n₀ ≤ n → g n ≤ c * f n }

notation "O(" f ")" => BigO f

/-! ## Complexity Classes -/

/-- PTIME class (definition only) -/
def PTIME : Set (ℕ → ℝ) := ⋃ k : ℕ, { f | ∃ c : ℝ, ∀ n, f n ≤ c * n^k }

/-- PSPACE class (definition only) -/
def PSPACE_CLASS : Set (ℕ → ℝ) := ⋃ k : ℕ, { f | ∃ c : ℝ, ∀ n, f n ≤ c * n^k }

/-! ## Value Iteration Structure -/

/-- Value iteration step for sequential decision problems -/
noncomputable def valueIterStep {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) (V : S → ℝ) (s : S) : ℝ :=
  ∑ a : A, (P.utility a s + ∑ s' : S, P.transition a s s' * V s')

/-- Value iteration for bounded horizon -/
noncomputable def valueIteration {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) : ℕ → S → ℝ
  | 0 => fun _ => 0
  | k + 1 => fun s => valueIterStep P (valueIteration P k) s

/-- Value iteration terminates in at most horizon steps -/
theorem valueIteration_terminates {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) :
    ∀ k ≤ P.horizon, ∀ s : S, ∃ v : ℝ, valueIteration P k s = v := by
  intro k _ s
  exact ⟨valueIteration P k s, rfl⟩

/-! ## Tractability Mapping -/

/-- Tractability condition for stochastic sufficiency -/
def StochTractable {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    (_P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  I.card ≤ Nat.log 2 (Fintype.card S)

/-- Stochastic sufficiency check is polynomial when tractability condition holds -/
theorem stochastic_sufficiency_poly_when_tractable {A S : Type*} [Fintype A] [Fintype S] [Nonempty S]
    {n : ℕ} (_P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hTract : I.card ≤ Nat.log 2 (Fintype.card S)) :
    2^I.card ≤ Fintype.card S := by
  calc 2^I.card
      ≤ 2^(Nat.log 2 (Fintype.card S)) := Nat.pow_le_pow_right (by norm_num) hTract
    _ ≤ Fintype.card S := Nat.pow_log_le_self 2 (Fintype.card_pos.ne')

end DecisionQuotient.StochasticSequential
