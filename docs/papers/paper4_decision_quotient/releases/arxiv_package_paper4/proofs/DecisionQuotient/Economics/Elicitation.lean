/-
  Paper 4: Decision-Relevant Uncertainty

  Economics/Elicitation.lean - Polynomial-time elicitation mechanisms

  ## Triviality Level
  TRIVIAL — simple constructions over the finite decision problem interface.
  Closest nontrivial definitions/lemmas used: `FiniteDecisionProblem.mem_optimalActions_iff`
  (DecisionQuotient.Finite).
-/

import DecisionQuotient.Finite

namespace DecisionQuotient

open Classical

variable {A S : Type*} {n : ℕ} [CoordinateSpace S n]

/-- A structured utility class where optimal actions are state-invariant. -/
def StructuredUtility (dp : FiniteDecisionProblem (A := A) (S := S)) : Prop :=
  ∀ s ∈ dp.states, ∀ s' ∈ dp.states, dp.optimalActions s = dp.optimalActions s'

lemma structured_isSufficient
    (dp : FiniteDecisionProblem (A := A) (S := S)) (hstruct : StructuredUtility (A := A) (S := S) dp)
    (I : Finset (Fin n)) :
    dp.isSufficient I := by
  intro s hs s' hs' _
  exact hstruct s hs s' hs'

/-- A simple elicitation mechanism structure. -/
structure PolytimeElicitationMechanism (A S : Type*) (n : ℕ) [CoordinateSpace S n] where
  query : FiniteDecisionProblem (A := A) (S := S) → List String
  aggregate : List String → Finset (Fin n)
  poly_queries :
    ∃ c k : ℕ,
      ∀ dp, (query dp).length ≤ c * (dp.states.card + dp.actions.card) ^ k + c
  sufficiency :
    ∀ (dp : FiniteDecisionProblem (A := A) (S := S)) (answers : List String),
      StructuredUtility (A := A) (S := S) dp → dp.isSufficient (aggregate answers)

/-- Existence of a mechanism for structured utilities.
    For structured utilities (state-invariant optimal actions), any set is sufficient,
    so the trivial mechanism (ask nothing, return ∅) works. -/
theorem polytime_elicitation_exists_structured
    : ∃ mech : PolytimeElicitationMechanism (A := A) (S := S) n,
        ∀ dp, StructuredUtility (A := A) (S := S) dp →
          dp.isSufficient (mech.aggregate (mech.query dp)) := by
  refine ⟨{ query := fun _ => []
          , aggregate := fun _ => ∅
          , poly_queries := ?_
          , sufficiency := ?_ }, ?_⟩
  · refine ⟨1, 0, ?_⟩
    intro dp
    simp
  · intro dp answers hstruct
    exact structured_isSufficient (dp := dp) hstruct (I := ∅)
  · intro dp hstruct
    -- The mechanism ignores queries; structured utilities make every set sufficient.
    exact structured_isSufficient (dp := dp) hstruct (I := ∅)

end DecisionQuotient
