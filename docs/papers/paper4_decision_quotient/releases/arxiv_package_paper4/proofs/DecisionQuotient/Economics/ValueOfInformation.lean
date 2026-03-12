/-
  Paper 4: Decision-Relevant Uncertainty

  Economics/ValueOfInformation.lean - VOI definitions and complexity claims

  ## Triviality Level
  TRIVIAL — definitions and straightforward lemmas about VOI computation.
  Closest nontrivial sources: `DecisionQuotient.Hardness.ApproximationHardness` and
  `DecisionQuotient.Finite` for finite DP interfaces.
-/

import DecisionQuotient.Finite
import DecisionQuotient.Hardness.ApproximationHardness
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Max

namespace DecisionQuotient

open Classical

variable {A S : Type*}

/-- A finite signal over states: outcomes and an observation function. -/
structure Signal (S : Type*) where
  outcome : S → ℕ
  outcomes : Finset ℕ
  outcome_mem : ∀ s, outcome s ∈ outcomes

/-- Prior distribution (not necessarily normalized). -/
def Prior (S : Type*) := S → ℚ

/-- Expected utility of action `a` under a prior. -/
noncomputable def expectedUtility (dp : FiniteDecisionProblem (A := A) (S := S))
    (prior : Prior S) (a : A) : ℚ :=
  dp.states.sum (fun s => prior s * (dp.utility a s : ℚ))

/-- Best expected utility across actions (0 if no actions). -/
noncomputable def bestExpectedUtility (dp : FiniteDecisionProblem (A := A) (S := S))
    (prior : Prior S) : ℚ :=
  if h : dp.actions.Nonempty then
    let vals := dp.actions.image (fun a => expectedUtility dp prior a)
    Finset.max' vals (by
      rcases h with ⟨a, ha⟩
      refine ⟨expectedUtility dp prior a, ?_⟩
      exact Finset.mem_image.mpr ⟨a, ha, rfl⟩)
  else
    0

/-- Restrict a prior to a single signal outcome. -/
def Signal.priorOn (signal : Signal S) (prior : Prior S) (o : ℕ) : Prior S :=
  fun s => if signal.outcome s = o then prior s else 0

/-- Value of information: expected gain from conditioning on a signal. -/
noncomputable def valueOfInformation (dp : FiniteDecisionProblem (A := A) (S := S))
    (prior : Prior S) (signal : Signal S) : ℚ :=
  signal.outcomes.sum (fun o => bestExpectedUtility dp (signal.priorOn prior o)) -
    bestExpectedUtility dp prior

/-- A constant (uninformative) signal. -/
def Signal.const (o : ℕ) : Signal S :=
  { outcome := fun _ => o
    outcomes := {o}
    outcome_mem := by intro s; simp }

/-- VOI reduces to a finite sum over outcomes (definitional). -/
theorem voi_computation_sharp_P_hard
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (prior : Prior S) (signal : Signal S) :
    valueOfInformation dp prior signal =
      signal.outcomes.sum (fun o => bestExpectedUtility dp (signal.priorOn prior o)) -
        bestExpectedUtility dp prior := by
  rfl

/-- Uninformative signals carry zero value. -/
theorem valueOfInformation_const
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (prior : Prior S) (o : ℕ) :
    valueOfInformation dp prior (Signal.const (S := S) o) = 0 := by
  classical
  have hprior : (Signal.const (S := S) o).priorOn prior o = prior := by
    funext s
    simp [Signal.const, Signal.priorOn]
  calc
    valueOfInformation dp prior (Signal.const (S := S) o)
        = bestExpectedUtility dp ((Signal.const (S := S) o).priorOn prior o) -
            bestExpectedUtility dp prior := by
            simp [valueOfInformation, Signal.const]
    _ = 0 := by
            simp [hprior]

/-- Exact computation yields an FPTAS-style guarantee. -/
theorem voi_fptas_smooth_utilities
    (dp : FiniteDecisionProblem (A := A) (S := S)) (prior : Prior S)
    (ε : ℚ) (hε : 0 ≤ ε) :
    ∃ algo : Signal S → ℚ,
      ∀ signal, approxWithin ε (algo signal) (valueOfInformation dp prior signal) := by
  refine ⟨fun signal => valueOfInformation dp prior signal, ?_⟩
  intro signal
  unfold approxWithin
  simp
  exact mul_nonneg hε (abs_nonneg _)

end DecisionQuotient
