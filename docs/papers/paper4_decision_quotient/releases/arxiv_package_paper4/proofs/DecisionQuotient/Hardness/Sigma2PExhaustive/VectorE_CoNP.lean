/-
  Paper 4: Decision-Relevant Uncertainty

  Hardness/Sigma2PExhaustive/VectorE_CoNP.lean

  This file packages the structural collapse for MSS on Boolean product spaces:
  MSS (∃ I, |I| ≤ k ∧ sufficient) is equivalent to a cardinality bound on the
  relevant coordinate set. This gives coNP-style certificates (witnesses of
  relevance) and blocks Σ₂ᴾ-hardness for this model unless PH collapses.

  ## Triviality Level
  NONTRIVIAL: This is a core hardness result - shows coNP structure for Boolean cubes.

  ## Dependencies
  - Chain: Sigma2PHardness.lean → here (collapse proof)
-/

import DecisionQuotient.Hardness.Sigma2PHardness

namespace DecisionQuotient

open Sigma2PHardness

/-! ### Vector E: coNP characterization on Boolean cubes

This file packages the structural collapse for MSS on Boolean product spaces:
MSS (∃ I, |I| ≤ k ∧ sufficient) is equivalent to a cardinality bound on the
relevant coordinate set. This gives coNP-style certificates (witnesses of
relevance) and blocks Σ₂ᴾ-hardness for this model unless PH collapses.
-/

theorem mss_equiv_relevant_card {A : Type*} {n : ℕ}
    (dp : DecisionProblem A (Fin n → Bool)) (k : ℕ) :
    (∃ I : Finset (Fin n), I.card ≤ k ∧ dp.isSufficient I) ↔
      (relevantFinset dp).card ≤ k :=
  min_sufficient_set_iff_relevant_card (dp := dp) k

end DecisionQuotient
