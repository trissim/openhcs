import Mathlib.Data.Nat.Basic

/-!
# Tool-Relative Effective Complexity Collapse

This module formalizes model-relative statements of tool leverage:
- A baseline work model `M`
- A tool-augmented model `T` over `M` that is pointwise no worse
- Eventual strict improvement and the induced effective-collapse predicate

No universal class-collapse claim is made; every statement is relative to the
declared model and tool.
-/

namespace DecisionQuotient
namespace ToolCollapse

/-- Size-indexed discrete work profile. -/
abbrev WorkProfile := ℕ → ℕ

/-- Baseline computational work model. -/
structure WorkModel where
  work : WorkProfile

/-- Tool-augmented model over a fixed baseline.
`no_worse` is the monotonic leverage condition. -/
structure ToolModel (M : WorkModel) where
  work : WorkProfile
  no_worse : ∀ n : ℕ, work n ≤ M.work n

/-- Eventual strict improvement: from some threshold onward, the tool requires
strictly less work than the baseline. -/
def EventualStrictImprovement (M : WorkModel) (T : ToolModel M) : Prop :=
  ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ → T.work n < M.work n

/-- Effective-collapse relation used in the paper: never worse and eventually
strictly better. -/
def EffectiveCollapse (M : WorkModel) (T : ToolModel M) : Prop :=
  (∀ n : ℕ, T.work n ≤ M.work n) ∧ EventualStrictImprovement M T

/-- A tool augmentation is pointwise no worse by definition. -/
theorem tool_never_worse (M : WorkModel) (T : ToolModel M) :
    ∀ n : ℕ, T.work n ≤ M.work n :=
  T.no_worse

/-- Eventual strict improvement immediately yields an explicit witness size. -/
theorem strict_improvement_has_witness
    {M : WorkModel} {T : ToolModel M}
    (hStrict : EventualStrictImprovement M T) :
    ∃ n : ℕ, T.work n < M.work n := by
  rcases hStrict with ⟨n₀, hAfter⟩
  exact ⟨n₀, hAfter n₀ (le_rfl)⟩

/-- No-worse + eventual strict improvement gives effective collapse. -/
theorem effective_collapse_of_eventual_strict
    (M : WorkModel) (T : ToolModel M)
    (hStrict : EventualStrictImprovement M T) :
    EffectiveCollapse M T :=
  ⟨T.no_worse, hStrict⟩

/-- Canonical exponential baseline profile. -/
def expBaseline : WorkModel where
  work := fun n => 2 ^ n

/-- Canonical linear tool profile over the exponential baseline. -/
def linearTool : ToolModel expBaseline where
  work := fun n => n
  no_worse := by
    intro n
    exact Nat.le_of_lt n.lt_two_pow_self

/-- In the exponential-vs-linear witness family, strict improvement holds from
size 1 onward. -/
theorem linear_tool_eventual_strict :
    EventualStrictImprovement expBaseline linearTool := by
  refine ⟨1, ?_⟩
  intro n _
  exact n.lt_two_pow_self

/-- Therefore the witness family satisfies effective collapse. -/
theorem linear_tool_effective_collapse :
    EffectiveCollapse expBaseline linearTool :=
  effective_collapse_of_eventual_strict expBaseline linearTool linear_tool_eventual_strict

end ToolCollapse
end DecisionQuotient

