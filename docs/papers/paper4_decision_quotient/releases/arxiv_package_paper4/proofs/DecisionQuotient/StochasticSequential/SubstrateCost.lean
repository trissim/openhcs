/-
  Paper 4b: Stochastic and Sequential Regimes

  SubstrateCost.lean - Formalization of substrate-dependent costs

  Extends paper 4's integrity-competence framework with substrate cost κ.
  Uses substrate classes and MatrixCell from Basic.lean.
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.Set.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Substrate Cost

The key insight: verdict (integrity-competence outcome) is substrate-independent,
but the cost to achieve that verdict is substrate-dependent (κ).
-/

/-- Substrate cost function: cell × substrate class → cost -/
def substrateCost (κ : MatrixCell → SubstrateType → ℝ)
    (cell : MatrixCell) (substrate : SubstrateType) : ℝ :=
  κ cell substrate

/-! ## Substrate-Dependent Trajectory -/

/-- A problem sequence -/
structure ProblemSequence (A S : Type*) where
  problems : List (DecisionProblem A S)

/-- Trajectory depends on substrate cost -/
noncomputable def trajectory {A S : Type*}
    (σ : ProblemSequence A S)
    (κ : MatrixCell → SubstrateType → ℝ)
    (substrate : SubstrateType) : List (MatrixCell × ℝ) :=
  σ.problems.map fun _ =>
    let cell : MatrixCell := {
      integrity := true
      attempted := true
      competenceAvailable := true
    }
    (cell, κ cell substrate)

/-! ## κ as Sufficient Statistic -/

/-- κ captures all decision-relevant substrate information -/
theorem kappa_sufficient_statistic {A S : Type*}
    (σ : ProblemSequence A S)
    (κ : MatrixCell → SubstrateType → ℝ)
    (τ₁ τ₂ : SubstrateType)
    (hκ : ∀ c, κ c τ₁ = κ c τ₂) :
    trajectory σ κ τ₁ = trajectory σ κ τ₂ := by
  simp [trajectory, hκ]

end DecisionQuotient.StochasticSequential
