/-
  Paper 4b: Stochastic and Sequential Regimes

  Instances.lean - Concrete instantiations of stochastic/sequential problems

  Provides example problems for each regime.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Tractability
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Static Instance -/

/-- Classic umbrella decision -/
def umbrellaStatic : DecisionProblem Bool (Fin 2) :=
  { utility := fun a s =>
    match a, s with
    | true, 0 => 2    -- carry, rain
    | true, 1 => -1   -- carry, no rain
    | false, 0 => -2  -- don't carry, rain
    | false, 1 => 0 }  -- don't carry, no rain

/-! ## Stochastic Instance -/

/-- Umbrella with weather forecast distribution -/
noncomputable def umbrellaStochastic : StochasticDecisionProblem Bool (Fin 2) :=
  { toDecisionProblem := umbrellaStatic
    distribution := fun s =>
      match s with
      | 0 => 0.3  -- rain probability
      | 1 => 0.7 } -- no rain probability

/-- Expected utility of carrying umbrella is well-defined -/
theorem umbrella_expected_carry :
    ∃ v : ℝ, v = umbrellaStochastic.distribution 0 * umbrellaStatic.utility true 0 +
             umbrellaStochastic.distribution 1 * umbrellaStatic.utility true 1 := by
  use 0.3 * 2 + 0.7 * (-1)
  rfl

/-! ## Sequential Instance -/

/-- Multi-stage gambler problem -/
noncomputable def gamblerSequential (goal : ℕ) [NeZero goal] :
    SequentialDecisionProblem Bool (Fin (goal + 1)) (Fin 2) :=
  { toDecisionProblem :=
    { utility := fun a s =>
        if a ∧ s.val + 1 = goal then 1 else 0 }
    transition := fun _ _ _ => 1 / (goal + 1)  -- Uniform
    observationModel := fun _ _ => 1 / 2  -- Binary observation
    horizon := 10 }

/-! ## Product Distribution Instance -/

/-- Independent coin flips -/
noncomputable def coinFlips (n : ℕ) [NeZero (2^n)] : StochasticDecisionProblem Unit (Fin (2^n)) :=
  { toDecisionProblem :=
    { utility := fun _ _ => 1 }
    distribution := fun _ => 1 / (2^n : ℝ) }

/-- Coin flips have product distribution: uniform over Fin (2^n) is a product
    of n independent uniform coin flips. The uniformity check (all probabilities equal)
    captures this since uniform = ∏ᵢ uniform_marginal_i for independent coordinates. -/
theorem coinFlips_product (n : ℕ) [NeZero (2^n)] :
    isProductDistribution (coinFlips n).distribution := by
  unfold isProductDistribution coinFlips
  intro s s'
  -- Both s and s' have probability 1 / (2^n : ℝ)
  rfl

/-! ## Fully Observable MDP Instance -/

/-- Simple inventory MDP -/
noncomputable def inventoryMDP (capacity : ℕ) [NeZero capacity] :
    SequentialDecisionProblem (Fin 2) (Fin capacity) Unit :=
  { toDecisionProblem :=
    { utility := fun a s =>
        if a = 0 then -(s.val : ℝ) else (s.val : ℝ) }
    transition := fun _ _ _ => 1 / capacity  -- Uniform
    observationModel := fun _ _ => 1  -- Deterministic unit observation
    horizon := 5 }

/-- Inventory MDP observation model is deterministic (fully observable) -/
theorem inventoryMDP_fully_observable (capacity : ℕ) [NeZero capacity] :
    ∀ s : Fin capacity, (inventoryMDP capacity).observationModel s () = 1 := by
  intro _
  rfl

end DecisionQuotient.StochasticSequential
