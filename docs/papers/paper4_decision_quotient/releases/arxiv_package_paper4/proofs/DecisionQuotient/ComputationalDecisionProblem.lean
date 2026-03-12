/-
  Paper 4: Decision-Relevant Uncertainty

  ComputationalDecisionProblem.lean

  Adds computational layer: decision problems with solvers.
  Connects mathematical decision problems to computational solving.

  ## Purpose
  - Extend DecisionProblem with a `solve` function (algorithm)
  - Define solver correctness (soundness/completeness)
  - Enable Paper 3's five-way equivalence theorem
 -/

import DecisionQuotient.Basic
import DecisionQuotient.Sufficiency
import DecisionQuotient.Tractability.StructuralRank

namespace DecisionQuotient

/-! ## Computational Decision Problem -/

/-- A computational decision problem extends the mathematical decision problem
    with an actual solver algorithm.
    
    The solver takes a state and returns:
    - `none`: solver abstains (cannot determine optimal action)
    - `some a`: solver claims action `a` is optimal -/
structure ComputationalDecisionProblem (A S : Type*) extends DecisionProblem A S where
  solve : S → Option A

/-! ## Solver Correctness -/

/-- A solver is sound: any returned action is actually optimal -/
def ComputationalDecisionProblem.sound {A S : Type*} 
    (cdp : ComputationalDecisionProblem A S) : Prop :=
  ∀ (s : S) (a : A), 
    cdp.solve s = some a → a ∈ cdp.Opt s

/-- A solver is complete: it returns something for every state -/
def ComputationalDecisionProblem.complete {A S : Type*} 
    (cdp : ComputationalDecisionProblem A S) : Prop :=
  ∀ (s : S), ∃ (a : A), cdp.solve s = some a

/-- A solver is correct: sound and complete -/
def ComputationalDecisionProblem.correct {A S : Type*} 
    (cdp : ComputationalDecisionProblem A S) : Prop :=
  cdp.sound ∧ cdp.complete

/-! ## Complexity Interface -/

/-- Solver runtime cost function -/
def SolverRuntime (S : Type*) := S → ℕ

/-- Tractable: correct solver with bounded runtime.
    This is the interface that Paper 3's five-way equivalence uses. -/
structure TractableSolver (A S : Type*) [Fintype S] where
  cdp : ComputationalDecisionProblem A S
  cost : SolverRuntime S
  correct : cdp.correct
  bounded : ∃ (k : ℕ), ∀ (s : S), cost s ≤ (Fintype.card S) ^ k + 1

/-- The five-way equivalence theorem needs this predicate -/
def IsTractable (A S : Type*) [Fintype S] : Prop :=
  Nonempty (TractableSolver A S)

end DecisionQuotient
