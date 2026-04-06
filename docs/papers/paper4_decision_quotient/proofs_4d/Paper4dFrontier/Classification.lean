import Paper4dFrontier.Basic

namespace Paper4dFrontier

open DecisionQuotient
open DecisionQuotient.DimensionalComplexity
open DecisionQuotient.StochasticSequential

/-- Degenerate mechanisms that yield tractability without introducing a new
structural certification pattern. -/
inductive DegenerateMechanism where
  | constantOptimizer
  | finiteEnumeration
  deriving DecidableEq

/-- Primitive tractability mechanisms currently visible in the mechanized frontier. -/
inductive PrimitiveMechanism where
  | core (s : CoreSubcase)
  | constantOptimizer
  | finiteEnumeration
  deriving DecidableEq

/-- Coarser role played by a currently named tractable family. -/
inductive FamilyRole where
  | core (s : CoreSubcase)
  | lifted (s : CoreSubcase)
  | degenerate (m : DegenerateMechanism)
  deriving DecidableEq

/-- Named tractable families currently formalized across the Paper 4 ecosystem. -/
inductive CurrentFormalizedFamily where
  | boundedActions
  | separableUtility
  | lowTensorRank
  | treeStructure
  | boundedTreewidth
  | coordinateSymmetry
  | productDistribution
  | boundedSupport
  | boundedHorizon
  | fullyObservable
  | singleAction
  | boundedStateSpace
  | strictGlobalDominance
  | constantOptimalSet
  | multiplicativeSeparableConstantSign
  deriving DecidableEq

/-- The six core structural subcases from Paper 4. -/
def coreSubcases : List CoreSubcase := [
  .boundedActions,
  .separableUtility,
  .lowTensorRank,
  .treeStructure,
  .boundedTreewidth,
  .coordinateSymmetry
]

/-- Explicit primitive basis used in the current draft. -/
def primitiveMechanisms : List PrimitiveMechanism := [
  .core .boundedActions,
  .core .separableUtility,
  .core .lowTensorRank,
  .core .treeStructure,
  .core .boundedTreewidth,
  .core .coordinateSymmetry,
  .constantOptimizer,
  .finiteEnumeration
]

/-- Inventory of the currently named mechanized tractable families. -/
def currentFamilies : List CurrentFormalizedFamily := [
  .boundedActions,
  .separableUtility,
  .lowTensorRank,
  .treeStructure,
  .boundedTreewidth,
  .coordinateSymmetry,
  .productDistribution,
  .boundedSupport,
  .boundedHorizon,
  .fullyObservable,
  .singleAction,
  .boundedStateSpace,
  .strictGlobalDominance,
  .constantOptimalSet,
  .multiplicativeSeparableConstantSign
]

/-- The six primitive nondegenerate families already isolated in Paper 4. -/
def coreFamilies : List CurrentFormalizedFamily := [
  .boundedActions,
  .separableUtility,
  .lowTensorRank,
  .treeStructure,
  .boundedTreewidth,
  .coordinateSymmetry
]

/-- Families that are tractable by reduction to an existing core subcase. -/
def liftedFamilies : List CurrentFormalizedFamily := [
  .productDistribution,
  .boundedSupport,
  .boundedHorizon,
  .fullyObservable
]

/-- Edge cases that become tractable by optimizer collapse or finite enumeration. -/
def degenerateFamilies : List CurrentFormalizedFamily := [
  .singleAction,
  .boundedStateSpace,
  .strictGlobalDominance,
  .constantOptimalSet,
  .multiplicativeSeparableConstantSign
]

/-- Forget the role-specific presentation and retain only the primitive mechanism. -/
def primitiveOfRole : FamilyRole → PrimitiveMechanism
  | .core s => .core s
  | .lifted s => .core s
  | .degenerate .constantOptimizer => .constantOptimizer
  | .degenerate .finiteEnumeration => .finiteEnumeration

/-- The subcase-facing part of the current positive theorem interface. -/
def roleSubcaseInterface : FamilyRole → Option CoreSubcase
  | .core s => some s
  | .lifted s => some s
  | .degenerate _ => none

/-- The degenerate-facing part of the current positive theorem interface. -/
def roleDegenerateInterface : FamilyRole → Option DegenerateMechanism
  | .core _ => none
  | .lifted _ => none
  | .degenerate m => some m

/-- Complexity-facing summary exported by the current positive theorem interface. -/
def roleComplexityInterface : FamilyRole → ComplexityClass
  | .core s => subcaseComplexity s
  | .lifted s => subcaseComplexity s
  | .degenerate _ => ComplexityClass.P

/-- Role classification for the currently named mechanized tractable families. -/
def classifyCurrentFamilyRole : CurrentFormalizedFamily → FamilyRole
  | .boundedActions => .core .boundedActions
  | .separableUtility => .core .separableUtility
  | .lowTensorRank => .core .lowTensorRank
  | .treeStructure => .core .treeStructure
  | .boundedTreewidth => .core .boundedTreewidth
  | .coordinateSymmetry => .core .coordinateSymmetry
  | .productDistribution => .lifted productToSubcase
  | .boundedSupport => .lifted boundedSupportToSubcase
  | .boundedHorizon => .lifted boundedHorizonToSubcase
  | .fullyObservable => .lifted fullyObservableToSubcase
  | .singleAction => .degenerate .constantOptimizer
  | .boundedStateSpace => .degenerate .finiteEnumeration
  | .strictGlobalDominance => .degenerate .constantOptimizer
  | .constantOptimalSet => .degenerate .constantOptimizer
  | .multiplicativeSeparableConstantSign => .degenerate .constantOptimizer

/-- Classification map from currently named families to the smaller primitive basis. -/
def classifyCurrentFamily : CurrentFormalizedFamily → PrimitiveMechanism
  | f => primitiveOfRole (classifyCurrentFamilyRole f)

/-- Family-indexed view of the subcase-facing theorem interface. -/
def familySubcaseInterface : CurrentFormalizedFamily → Option CoreSubcase
  | .boundedActions => some .boundedActions
  | .separableUtility => some .separableUtility
  | .lowTensorRank => some .lowTensorRank
  | .treeStructure => some .treeStructure
  | .boundedTreewidth => some .boundedTreewidth
  | .coordinateSymmetry => some .coordinateSymmetry
  | .productDistribution => some productToSubcase
  | .boundedSupport => some boundedSupportToSubcase
  | .boundedHorizon => some boundedHorizonToSubcase
  | .fullyObservable => some fullyObservableToSubcase
  | .singleAction => none
  | .boundedStateSpace => none
  | .strictGlobalDominance => none
  | .constantOptimalSet => none
  | .multiplicativeSeparableConstantSign => none

/-- Family-indexed view of the degenerate theorem interface. -/
def familyDegenerateInterface : CurrentFormalizedFamily → Option DegenerateMechanism
  | .boundedActions => none
  | .separableUtility => none
  | .lowTensorRank => none
  | .treeStructure => none
  | .boundedTreewidth => none
  | .coordinateSymmetry => none
  | .productDistribution => none
  | .boundedSupport => none
  | .boundedHorizon => none
  | .fullyObservable => none
  | .singleAction => some .constantOptimizer
  | .boundedStateSpace => some .finiteEnumeration
  | .strictGlobalDominance => some .constantOptimizer
  | .constantOptimalSet => some .constantOptimizer
  | .multiplicativeSeparableConstantSign => some .constantOptimizer

/-- Family-indexed complexity summary exported by the current positive theorem interface. -/
def familyComplexityInterface : CurrentFormalizedFamily → ComplexityClass
  | .boundedActions => subcaseComplexity .boundedActions
  | .separableUtility => subcaseComplexity .separableUtility
  | .lowTensorRank => subcaseComplexity .lowTensorRank
  | .treeStructure => subcaseComplexity .treeStructure
  | .boundedTreewidth => subcaseComplexity .boundedTreewidth
  | .coordinateSymmetry => subcaseComplexity .coordinateSymmetry
  | .productDistribution => subcaseComplexity productToSubcase
  | .boundedSupport => subcaseComplexity boundedSupportToSubcase
  | .boundedHorizon => subcaseComplexity boundedHorizonToSubcase
  | .fullyObservable => subcaseComplexity fullyObservableToSubcase
  | .singleAction => ComplexityClass.P
  | .boundedStateSpace => ComplexityClass.P
  | .strictGlobalDominance => ComplexityClass.P
  | .constantOptimalSet => ComplexityClass.P
  | .multiplicativeSeparableConstantSign => ComplexityClass.P

theorem coreSubcases_nodup : coreSubcases.Nodup := by
  decide

theorem coreSubcases_complete (s : CoreSubcase) : s ∈ coreSubcases := by
  cases s <;> simp [coreSubcases]

theorem primitiveMechanisms_nodup : primitiveMechanisms.Nodup := by
  decide

theorem primitiveMechanisms_length : primitiveMechanisms.length = 8 := by
  rfl

theorem primitiveMechanisms_explicit :
    primitiveMechanisms = [
      .core .boundedActions,
      .core .separableUtility,
      .core .lowTensorRank,
      .core .treeStructure,
      .core .boundedTreewidth,
      .core .coordinateSymmetry,
      .constantOptimizer,
      .finiteEnumeration
    ] := by
  rfl

theorem currentFamilies_nodup : currentFamilies.Nodup := by
  decide

theorem currentFamilies_length : currentFamilies.length = 15 := by
  rfl

theorem currentFamilies_explicit :
    currentFamilies = [
      .boundedActions,
      .separableUtility,
      .lowTensorRank,
      .treeStructure,
      .boundedTreewidth,
      .coordinateSymmetry,
      .productDistribution,
      .boundedSupport,
      .boundedHorizon,
      .fullyObservable,
      .singleAction,
      .boundedStateSpace,
      .strictGlobalDominance,
      .constantOptimalSet,
      .multiplicativeSeparableConstantSign
    ] := by
  rfl

theorem coreFamilies_nodup : coreFamilies.Nodup := by
  decide

theorem coreFamilies_length : coreFamilies.length = 6 := by
  rfl

theorem coreFamilies_explicit :
    coreFamilies = [
      .boundedActions,
      .separableUtility,
      .lowTensorRank,
      .treeStructure,
      .boundedTreewidth,
      .coordinateSymmetry
    ] := by
  rfl

theorem liftedFamilies_nodup : liftedFamilies.Nodup := by
  decide

theorem liftedFamilies_length : liftedFamilies.length = 4 := by
  rfl

theorem liftedFamilies_explicit :
    liftedFamilies = [
      .productDistribution,
      .boundedSupport,
      .boundedHorizon,
      .fullyObservable
    ] := by
  rfl

theorem degenerateFamilies_nodup : degenerateFamilies.Nodup := by
  decide

theorem degenerateFamilies_length : degenerateFamilies.length = 5 := by
  rfl

theorem degenerateFamilies_explicit :
    degenerateFamilies = [
      .singleAction,
      .boundedStateSpace,
      .strictGlobalDominance,
      .constantOptimalSet,
      .multiplicativeSeparableConstantSign
    ] := by
  rfl

theorem currentFamilies_complete (f : CurrentFormalizedFamily) : f ∈ currentFamilies := by
  cases f <;> simp [currentFamilies]

theorem coreFamilies_complete (f : CurrentFormalizedFamily) :
    f ∈ coreFamilies ↔ ∃ s, classifyCurrentFamilyRole f = .core s := by
  cases f <;> simp [coreFamilies, classifyCurrentFamilyRole]

theorem liftedFamilies_complete (f : CurrentFormalizedFamily) :
    f ∈ liftedFamilies ↔ ∃ s, classifyCurrentFamilyRole f = .lifted s := by
  cases f <;>
    simp [
      liftedFamilies,
      classifyCurrentFamilyRole,
      productToSubcase,
      boundedSupportToSubcase,
      boundedHorizonToSubcase,
      fullyObservableToSubcase,
    ]

theorem degenerateFamilies_complete (f : CurrentFormalizedFamily) :
    f ∈ degenerateFamilies ↔ ∃ m, classifyCurrentFamilyRole f = .degenerate m := by
  cases f <;> simp [degenerateFamilies, classifyCurrentFamilyRole]

theorem classifyCurrentFamily_eq_primitiveOfRole (f : CurrentFormalizedFamily) :
    classifyCurrentFamily f = primitiveOfRole (classifyCurrentFamilyRole f) := by
  rfl

theorem roleComplexityInterface_is_P (r : FamilyRole) :
    roleComplexityInterface r = ComplexityClass.P := by
  cases r <;> simp [roleComplexityInterface, subcaseComplexity]

theorem familySubcaseInterface_factors_through_role (f : CurrentFormalizedFamily) :
    familySubcaseInterface f = roleSubcaseInterface (classifyCurrentFamilyRole f) := by
  cases f <;>
    simp [
      familySubcaseInterface,
      roleSubcaseInterface,
      classifyCurrentFamilyRole,
      productToSubcase,
      boundedSupportToSubcase,
      boundedHorizonToSubcase,
      fullyObservableToSubcase,
    ]

theorem familyDegenerateInterface_factors_through_role (f : CurrentFormalizedFamily) :
    familyDegenerateInterface f = roleDegenerateInterface (classifyCurrentFamilyRole f) := by
  cases f <;> simp [familyDegenerateInterface, roleDegenerateInterface, classifyCurrentFamilyRole]

theorem familyComplexityInterface_factors_through_role (f : CurrentFormalizedFamily) :
    familyComplexityInterface f = roleComplexityInterface (classifyCurrentFamilyRole f) := by
  cases f <;>
    simp [
      familyComplexityInterface,
      roleComplexityInterface,
      classifyCurrentFamilyRole,
      subcaseComplexity,
      productToSubcase,
      boundedSupportToSubcase,
      boundedHorizonToSubcase,
      fullyObservableToSubcase,
    ]

theorem current_positive_interfaces_factor_through_role (f : CurrentFormalizedFamily) :
    familySubcaseInterface f = roleSubcaseInterface (classifyCurrentFamilyRole f) ∧
    familyDegenerateInterface f = roleDegenerateInterface (classifyCurrentFamilyRole f) ∧
    familyComplexityInterface f = roleComplexityInterface (classifyCurrentFamilyRole f) := by
  exact ⟨
    familySubcaseInterface_factors_through_role f,
    familyDegenerateInterface_factors_through_role f,
    familyComplexityInterface_factors_through_role f,
  ⟩

theorem currentFamilies_partition_by_role (f : CurrentFormalizedFamily) :
    ((f ∈ coreFamilies) ∧ f ∉ liftedFamilies ∧ f ∉ degenerateFamilies) ∨
    ((f ∈ liftedFamilies) ∧ f ∉ coreFamilies ∧ f ∉ degenerateFamilies) ∨
    ((f ∈ degenerateFamilies) ∧ f ∉ coreFamilies ∧ f ∉ liftedFamilies) := by
  cases f <;> simp [coreFamilies, liftedFamilies, degenerateFamilies]

theorem currentFamilies_partition_counts :
    coreFamilies.length = 6 ∧
    liftedFamilies.length = 4 ∧
    degenerateFamilies.length = 5 := by
  simp [coreFamilies_length, liftedFamilies_length, degenerateFamilies_length]

theorem currentFamilies_partition_total :
    coreFamilies.length + liftedFamilies.length + degenerateFamilies.length =
      currentFamilies.length := by
  decide

theorem lifted_families_reduce_to_core :
    classifyCurrentFamily .productDistribution = .core .separableUtility ∧
    classifyCurrentFamily .boundedSupport = .core .boundedActions ∧
    classifyCurrentFamily .boundedHorizon = .core .boundedTreewidth ∧
    classifyCurrentFamily .fullyObservable = .core .treeStructure := by
  simp [
    classifyCurrentFamily,
    primitiveOfRole,
    classifyCurrentFamilyRole,
    productToSubcase,
    boundedSupportToSubcase,
    boundedHorizonToSubcase,
    fullyObservableToSubcase,
  ]

theorem current_families_have_finite_basis (f : CurrentFormalizedFamily) :
    classifyCurrentFamily f ∈ primitiveMechanisms := by
  cases f <;>
    simp [
      classifyCurrentFamily,
      primitiveOfRole,
      classifyCurrentFamilyRole,
      primitiveMechanisms,
      productToSubcase,
      boundedSupportToSubcase,
      boundedHorizonToSubcase,
      fullyObservableToSubcase,
    ]

end Paper4dFrontier
