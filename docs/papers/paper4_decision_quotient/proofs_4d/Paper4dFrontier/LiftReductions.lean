import DecisionQuotient.StochasticSequential.Tractability

namespace Paper4dFrontier

open DecisionQuotient
open DecisionQuotient.DimensionalComplexity
open DecisionQuotient.StochasticSequential

theorem product_distribution_reduces_to_separable :
    productToSubcase = TractableSubcase.separableUtility := rfl

theorem bounded_support_reduces_to_bounded_actions :
    boundedSupportToSubcase = TractableSubcase.boundedActions := rfl

theorem bounded_horizon_reduces_to_bounded_treewidth :
    boundedHorizonToSubcase = TractableSubcase.boundedTreewidth := rfl

theorem fully_observable_reduces_to_tree_structure :
    fullyObservableToSubcase = TractableSubcase.treeStructure := rfl

theorem product_distribution_reduction_tractable {S : Type*} [Fintype S] {n : ℕ}
    (h : ProductDistribution S n) :
    productToSubcase = TractableSubcase.separableUtility ∧
    subcaseComplexity productToSubcase = ComplexityClass.P := by
  exact ⟨product_distribution_reduces_to_separable,
    product_distribution_tractable h⟩

theorem bounded_support_reduction_tractable {S : Type*} [Fintype S]
    (h : BoundedSupport S) :
    boundedSupportToSubcase = TractableSubcase.boundedActions ∧
    subcaseComplexity boundedSupportToSubcase = ComplexityClass.P := by
  exact ⟨bounded_support_reduces_to_bounded_actions,
    bounded_support_tractable h⟩

theorem bounded_horizon_reduction_tractable (h : BoundedHorizon) :
    boundedHorizonToSubcase = TractableSubcase.boundedTreewidth ∧
    subcaseComplexity boundedHorizonToSubcase = ComplexityClass.P := by
  exact ⟨bounded_horizon_reduces_to_bounded_treewidth,
    bounded_horizon_tractable h⟩

theorem fully_observable_reduction_tractable {A S O : Type*}
    (h : FullyObservable A S O) :
    fullyObservableToSubcase = TractableSubcase.treeStructure ∧
    subcaseComplexity fullyObservableToSubcase = ComplexityClass.P := by
  exact ⟨fully_observable_reduces_to_tree_structure,
    fully_observable_tractable h⟩

end Paper4dFrontier
