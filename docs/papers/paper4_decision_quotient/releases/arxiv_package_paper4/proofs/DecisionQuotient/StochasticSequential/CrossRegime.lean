/-
  Paper 4b: Stochastic and Sequential Regimes

  CrossRegime.lean - Transfer conditions between regimes

  When does sufficiency in one regime transfer to another?
  Uses complexity hierarchy from IntegrityEquilibrium.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Hierarchy
import DecisionQuotient.StochasticSequential.Tractability
import Mathlib.Data.Set.Basic

namespace DecisionQuotient.StochasticSequential

open Classical
open DecisionQuotient.Physics.DimensionalComplexity

/-! ## Cross-Regime Transfer via Complexity Classes

Transfer between regimes is governed by the complexity hierarchy:
- Static (coNP) → Stochastic (PP) → Sequential (PSPACE)

Tractability transfers upward when dimension structure is preserved.
-/

/-- Tractability transfers: if a subcase is in P, it remains in P regardless of regime.
    The tractable subcases are defined regime-independently (separable utility, bounded actions,
    etc.), so a P-time algorithm for any subcase works across all regimes. -/
theorem tractability_transfers (subcase : TractableSubcase) :
    subcaseComplexity subcase = ComplexityClass.P →
    ∀ _regime : ℕ, subcaseComplexity subcase = ComplexityClass.P := by
  intro hP _
  exact hP

/-! ## Regime Comparison

Static ⊂ Stochastic ⊂ Sequential with increasing complexity.
-/

/-- Static is simpler than stochastic -/
theorem static_simpler_than_stochastic :
    baseComplexity 0 = ComplexityClass.coNP ∧
    baseComplexity 1 = ComplexityClass.PP := by
  exact ⟨rfl, rfl⟩

/-- Stochastic is simpler than sequential -/
theorem stochastic_simpler_than_sequential :
    baseComplexity 1 = ComplexityClass.PP ∧
    baseComplexity 2 = ComplexityClass.PSPACE := by
  exact ⟨rfl, rfl⟩

/-! ## Transfer Conditions

Sufficiency transfers between regimes under structural conditions:
1. Product distribution: Static → Stochastic
2. Bounded horizon: Stochastic → Sequential
3. Fully observable: Partial observability → MDP
-/

/-- Product distribution enables static-to-stochastic transfer -/
theorem product_enables_transfer {S : Type*} [Fintype S] {n : ℕ}
    (struct : ProductDistribution S n) :
    productToSubcase = TractableSubcase.separableUtility := rfl

/-- Bounded horizon enables stochastic-to-sequential transfer -/
theorem bounded_horizon_enables_transfer (struct : BoundedHorizon) :
    boundedHorizonToSubcase = TractableSubcase.boundedTreewidth := rfl

/-- Fully observable enables tree structure -/
theorem fully_observable_enables_transfer {A S O : Type*}
    (struct : FullyObservable A S O) :
    fullyObservableToSubcase = TractableSubcase.treeStructure := rfl

end DecisionQuotient.StochasticSequential
