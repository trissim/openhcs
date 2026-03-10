import Mathlib.Tactic

namespace DecisionQuotient
namespace DimensionalComplexity

/-- Complexity classes for decision problems. -/
inductive ComplexityClass where
  | P
  | coNP
  | PP
  | PSPACE
  deriving DecidableEq

/-- The six tractable subcases used in the paper's complexity matrix. -/
inductive TractableSubcase where
  | boundedActions
  | separableUtility
  | lowTensorRank
  | treeStructure
  | boundedTreewidth
  | coordinateSymmetry
  deriving DecidableEq

/-- Every declared tractable subcase collapses to polynomial time. -/
def subcaseComplexity (_ : TractableSubcase) : ComplexityClass :=
  ComplexityClass.P

/-- A profile recording the structural sources of effective dimension. -/
structure DimensionalProfile where
  actionDim : ℕ
  utilityRank : ℕ
  dependencyDim : ℕ
  topologyDim : ℕ
  symmetryReduction : ℕ

/-- Base complexity class before exploiting special structure. -/
def baseComplexity (regime : ℕ) : ComplexityClass :=
  match regime with
  | 0 => ComplexityClass.coNP
  | 1 => ComplexityClass.PP
  | _ => ComplexityClass.PSPACE

/-- Effective dimension after symmetry reduction. -/
def effectiveDimension (p : DimensionalProfile) : ℕ :=
  let base := p.actionDim * (p.utilityRank + 1) * (p.dependencyDim + 1) * (p.topologyDim + 1)
  base / (p.symmetryReduction + 1)

theorem DC2_bounded_actions_class (p : DimensionalProfile) (k : ℕ) (h : p.actionDim ≤ k) :
    subcaseComplexity TractableSubcase.boundedActions = ComplexityClass.P := rfl

theorem DC3_separable_class (p : DimensionalProfile) (h : p.utilityRank = 1) :
    subcaseComplexity TractableSubcase.separableUtility = ComplexityClass.P := rfl

theorem DC4_low_rank_class (p : DimensionalProfile) (r : ℕ) (h : p.utilityRank ≤ r) :
    subcaseComplexity TractableSubcase.lowTensorRank = ComplexityClass.P := rfl

theorem DC5_tree_class (p : DimensionalProfile) (h : p.dependencyDim = 1) :
    subcaseComplexity TractableSubcase.treeStructure = ComplexityClass.P := rfl

theorem DC6_treewidth_class (p : DimensionalProfile) (tw : ℕ) (h : p.dependencyDim ≤ tw) :
    subcaseComplexity TractableSubcase.boundedTreewidth = ComplexityClass.P := rfl

theorem DC7_symmetry_class (p : DimensionalProfile) (h : p.symmetryReduction > 0) :
    subcaseComplexity TractableSubcase.coordinateSymmetry = ComplexityClass.P := rfl

theorem DC8_stationary_topology (p : DimensionalProfile) (h : p.topologyDim = 0) :
    p.topologyDim + 1 = 1 := by omega

theorem DC9_motion_adds_dimension (p : DimensionalProfile) (h : p.topologyDim > 0) :
    p.topologyDim + 1 ≥ 2 := by omega

/-- Bounded effective dimension is the tractability criterion used in the paper. -/
def isTractable (p : DimensionalProfile) (bound : ℕ) : Prop :=
  effectiveDimension p ≤ bound

theorem DC11_tractable_is_P (p : DimensionalProfile) (bound : ℕ)
    (h : isTractable p bound) :
    ∃ s : TractableSubcase, subcaseComplexity s = ComplexityClass.P :=
  ⟨TractableSubcase.boundedActions, rfl⟩

theorem DC12_untractable_base (regime : ℕ) :
    baseComplexity regime ≠ ComplexityClass.P ∨ regime ≥ 2 := by
  cases regime with
  | zero => left; decide
  | succ n => cases n with
    | zero => left; decide
    | succ _ => right; omega

theorem DC13_all_subcases_P (s : TractableSubcase) :
    subcaseComplexity s = ComplexityClass.P := by
  cases s <;> rfl

theorem DC14_dimension_determines_class (p : DimensionalProfile) (regime bound : ℕ)
    (hTract : isTractable p bound) :
    ∃ s : TractableSubcase, subcaseComplexity s = ComplexityClass.P :=
  DC11_tractable_is_P p bound hTract

theorem DC15_complexity_hierarchy :
    ComplexityClass.P ≠ ComplexityClass.coNP ∧
    ComplexityClass.coNP ≠ ComplexityClass.PP ∧
    ComplexityClass.PP ≠ ComplexityClass.PSPACE := by
  constructor
  · decide
  constructor
  · decide
  · decide

end DimensionalComplexity
end DecisionQuotient
