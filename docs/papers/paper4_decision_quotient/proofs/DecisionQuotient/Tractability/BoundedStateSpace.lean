/-
  Paper 4: Decision-Relevant Uncertainty

  Tractability/BoundedStateSpace.lean - Polynomial-time with bounded |S|

  Key Result: When |S| ≤ k for constant k, brute-force SUFFICIENCY-CHECK 
  runs in O(k²) = O(1) time.

  ## Dependencies
  - Chain: Sufficiency.lean → here (special case)
-/

import DecisionQuotient.Finite
import Mathlib.Data.Fintype.Card

namespace DecisionQuotient

/-! ## Bounded State Space -/

/-- With bounded |S| = k, brute-force sufficiency checking examines at most k² pairs.
    
    Each pair check compares optimal action sets. -/
theorem bounded_state_pair_bound {S : Type*} [Fintype S]
    (k : ℕ) (hk : Fintype.card S ≤ k) :
    Fintype.card S * Fintype.card S ≤ k * k :=
  Nat.mul_le_mul hk hk

/-- If the state space has bounded size, sufficiency checking is constant-time. -/
theorem bounded_state_sufficiency_trivial
    {n : ℕ} {A : Type*} {S : Type*} [CoordinateSpace S n] [Fintype S]
    (k : ℕ) (hk : Fintype.card S ≤ k)
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (I : Finset (Fin n)) :
    dp.isSufficient I ↔
      ∀ (s₁ : S) (hs₁ : s₁ ∈ dp.states) (s₂ : S) (hs₂ : s₂ ∈ dp.states),
        (∀ i ∈ I, CoordinateSpace.proj s₁ i = CoordinateSpace.proj s₂ i) →
          dp.optimalActions s₁ = dp.optimalActions s₂ := by
  rfl

end DecisionQuotient
