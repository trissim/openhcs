/-
  Paper 4: Decision-Relevant Uncertainty

  AbstractionCollapse.lean

  Structural bridge theorem:
  - state-level abstraction is many-to-one collapse
  - the decision quotient is the coarsest lossless collapse
  - any extra collapse erases a decision-relevant distinction
  - if one maps that extra erasure to a physically feasible collapse, the
    existing no-collapse theorems force contradiction

  This file does not add new physics. It packages the already-mechanized
  quotient and physical-hardness layers into one explicit bridge.
-/

import DecisionQuotient.Quotient
import DecisionQuotient.Physics.PhysicalHardness

namespace DecisionQuotient

open Classical

variable {A S T : Type*}

/-- A state abstraction collapses a decision-relevant distinction when it
identifies two states that do not share the same optimal-action set. -/
def DecisionProblem.erasesDecisionRelevantDistinction
    (dp : DecisionProblem A S) (φ : S → T) : Prop :=
  ∃ s s' : S, φ s = φ s' ∧ dp.Opt s ≠ dp.Opt s'

/-- Failing to preserve `Opt` is exactly the same as collapsing a
decision-relevant distinction. -/
theorem DecisionProblem.not_preservesOpt_iff_erasesDecisionRelevantDistinction
    (dp : DecisionProblem A S) (φ : S → T) :
    ¬ dp.preservesOpt φ ↔ dp.erasesDecisionRelevantDistinction φ := by
  constructor
  · intro h
    unfold DecisionProblem.preservesOpt at h
    unfold DecisionProblem.erasesDecisionRelevantDistinction
    push_neg at h
    exact h
  · intro hErase
    intro hPres
    rcases hErase with ⟨s, s', hEq, hNe⟩
    exact hNe (hPres s s' hEq)

/-- For a surjective abstraction, there are only two structural possibilities:
either it factors through the decision quotient, or it erases a
decision-relevant distinction. -/
theorem DecisionProblem.surjective_abstraction_factors_or_erases
    (dp : DecisionProblem A S) (φ : S → T)
    (hSurj : Function.Surjective φ) :
    (∃ ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s)) ∨
      dp.erasesDecisionRelevantDistinction φ := by
  by_cases hPres : dp.preservesOpt φ
  · exact Or.inl (dp.quotient_is_coarsest φ hPres hSurj)
  · exact Or.inr ((dp.not_preservesOpt_iff_erasesDecisionRelevantDistinction φ).mp hPres)

/-- If a collapsed decision-relevant distinction were mapped to a physically
feasible complexity collapse at the canonical requirement profile, the existing
physical no-collapse theorems force contradiction. -/
theorem DecisionProblem.collapseBeyondQuotient_physically_impossible
    (dp : DecisionProblem A S) (φ : S → T)
    (c : PhysicalComplexity.PhysicalComputer)
    (hMap :
      dp.erasesDecisionRelevantDistinction φ →
      PhysicalComplexity.PhysicalCollapseAtRequirement c PhysicalComplexity.coNP_requirement) :
    ¬ dp.erasesDecisionRelevantDistinction φ := by
  intro hErase
  have hCoherent :
      PhysicalComplexity.CoherentDQRejectionAtRequirement
        c PhysicalComplexity.coNP_requirement
        (dp.erasesDecisionRelevantDistinction φ) := ⟨hErase, hMap⟩
  exact PhysicalComplexity.coherent_dq_rejection_impossible_canonical c hCoherent

/-- Bridge theorem: if a surjective abstraction is equipped with a map sending
any extra erasure beyond the decision quotient to a physically feasible
collapse, then the physical no-collapse layer forces that abstraction to factor
through the decision quotient after all. -/
theorem DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors
    (dp : DecisionProblem A S) (φ : S → T)
    (hSurj : Function.Surjective φ)
    (c : PhysicalComplexity.PhysicalComputer)
    (hMap :
      dp.erasesDecisionRelevantDistinction φ →
      PhysicalComplexity.PhysicalCollapseAtRequirement c PhysicalComplexity.coNP_requirement) :
    ∃ ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s) := by
  have hNoErase :
      ¬ dp.erasesDecisionRelevantDistinction φ :=
    dp.collapseBeyondQuotient_physically_impossible φ c hMap
  have hPres : dp.preservesOpt φ := by
    intro s s' hEq
    by_contra hNe
    exact hNoErase ⟨s, s', hEq, hNe⟩
  exact dp.quotient_is_coarsest φ hPres hSurj

end DecisionQuotient
