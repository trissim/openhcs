/-
  ConcreteExamples.lean

  Small concrete instances corresponding to the paper's toy POMDP
  numeric worked example. These are explicit finite examples that should
  type-check immediately and serve as links between the LaTeX worked
  instance and the mechanization predicates.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.Examples.PreservationExamples

namespace DecisionQuotient.Examples

open DecisionQuotient.StochasticSequential

section ConcretePOMDP

/- finite types for the toy instance -/
inductive S | s1 | s2 deriving DecidableEq, Fintype
inductive A | a | b deriving DecidableEq, Fintype
inductive O | o1 | o2 deriving DecidableEq, Fintype

noncomputable def util : A → S → ℝ
| A.a, S.s1 => 2
| A.b, S.s1 => 1
| A.a, S.s2 => 0
| A.b, S.s2 => 3

noncomputable def dist : S → ℝ
| S.s1 => 1/2
| S.s2 => 1/2

noncomputable def toySDP : StochasticDecisionProblem A S :=
  { utility := fun a s => util a s,
    distribution := dist }

def coarseO : S → Unit := fun _ => ()

theorem toy_full_opt_s1_contains_a : A.a ∈ toySDP.toDecisionProblem.Opt S.s1 := by
  intro a'
  cases a' <;> norm_num [DecisionProblem.Opt, DecisionProblem.isOptimal, toySDP, util]

end ConcretePOMDP

end DecisionQuotient.Examples
