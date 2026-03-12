/-
  Paper 4b: Stochastic and Sequential Regimes

  Tractability.lean - Polynomial-time solvable special cases

  Maps stochastic/sequential tractability to the 6 tractable subcases from
  IntegrityEquilibrium.lean. Polynomial-time algorithms are standard; we only
  mechanize the structural mappings.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Finite
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Card

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient
open DecisionQuotient.Physics.DimensionalComplexity

/-! ## Tractability Structures

These structures characterize when stochastic/sequential problems become tractable.
Each maps to one of the 6 tractable subcases. -/

/-- A product distribution factors as product of marginals.
    Maps to TractableSubcase.separableUtility -/
structure ProductDistribution (S : Type*) [Fintype S] (n : ℕ) where
  marginals : Fin n → Fin 2 → ℝ
  nonneg : ∀ i b, marginals i b ≥ 0
  normalized : ∀ i, ∑ b : Fin 2, marginals i b = 1

/-- Bounded support distribution.
    Maps to TractableSubcase.boundedActions (via enumeration) -/
structure BoundedSupport (S : Type*) [Fintype S] where
  bound : ℕ
  support : Finset S
  card_le : support.card ≤ bound

/-- Bounded horizon sequential problem.
    Maps to TractableSubcase.boundedTreewidth -/
structure BoundedHorizon where
  horizon : ℕ
  bound : ℕ
  horizon_le : horizon ≤ bound

/-- Fully observable MDP (no partial observability).
    Maps to TractableSubcase.treeStructure.
    The observation function is injective: different states produce different observations,
    so the current state can be uniquely recovered from the observation. -/
structure FullyObservable (A S O : Type*) where
  obs : S → O
  obs_injective : Function.Injective obs

/-! ## Mapping to 6 Tractable Subcases -/

/-- Product distribution → separable utility -/
def productToSubcase : TractableSubcase := TractableSubcase.separableUtility

/-- Bounded support → bounded actions (enumerate support) -/
def boundedSupportToSubcase : TractableSubcase := TractableSubcase.boundedActions

/-- Bounded horizon → bounded treewidth -/
def boundedHorizonToSubcase : TractableSubcase := TractableSubcase.boundedTreewidth

/-- Fully observable MDP → tree structure -/
def fullyObservableToSubcase : TractableSubcase := TractableSubcase.treeStructure

/-! ## Tractability Theorems

These theorems state that each structural condition implies polynomial-time
solvability. The algorithms are standard (dynamic programming, enumeration);
we mechanize only the structural mapping. -/

/-- Product distribution implies tractability via separable utility -/
theorem product_distribution_tractable {S : Type*} [Fintype S] {n : ℕ}
    (struct : ProductDistribution S n) :
    subcaseComplexity productToSubcase = ComplexityClass.P := by
  rfl

/-- Bounded support implies tractability via enumeration -/
theorem bounded_support_tractable {S : Type*} [Fintype S]
    (struct : BoundedSupport S) :
    subcaseComplexity boundedSupportToSubcase = ComplexityClass.P := by
  rfl

/-- Bounded horizon implies tractability via bounded treewidth -/
theorem bounded_horizon_tractable (struct : BoundedHorizon) :
    subcaseComplexity boundedHorizonToSubcase = ComplexityClass.P := by
  rfl

/-- Fully observable MDP implies tractability via tree structure -/
theorem fully_observable_tractable {A S O : Type*}
    (struct : FullyObservable A S O) :
    subcaseComplexity fullyObservableToSubcase = ComplexityClass.P := by
  rfl

/-! ## Support Utilities -/

/-- Support of a distribution -/
noncomputable def distributionSupport {S : Type*} [Fintype S] (dist : S → ℝ) : Finset S :=
  Finset.univ.filter (fun s => dist s > 0)

end DecisionQuotient.StochasticSequential
