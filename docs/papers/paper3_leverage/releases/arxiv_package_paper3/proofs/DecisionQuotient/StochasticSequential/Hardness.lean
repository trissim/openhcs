/-
  Paper 4b: Stochastic and Sequential Regimes

  Hardness.lean - Honest hardness packaging for stochastic/sequential problems

  This file intentionally packages only what is fully mechanized in the current
  development:

  - reduction correctness
  - polynomial output-size bounds (`SizeBoundedReduction`)

  It does NOT claim a full machine-checked PP/PSPACE membership proof. Those
  would require explicit TM witnesses for the relevant decision procedures.
  The corresponding standard-complexity membership arguments remain paper-level
  claims rather than end-to-end TM formalizations in this repository.
-/

import DecisionQuotient.StochasticSequential.PolynomialReduction

namespace DecisionQuotient.StochasticSequential

/-- Fully mechanized stochastic hardness package: MAJSAT reduces, with
    polynomial output-size bounds, to empty-set stochastic sufficiency via the
    pure three-action gadget. -/
abbrev HonestStochasticSufficiencyPPHard := StochasticSufficiencyPPHard

/-- Fully mechanized stochastic anchor hardness package: MAJSAT reduces, with
    polynomial output-size bounds, to empty-set stochastic anchor sufficiency. -/
abbrev HonestStochasticAnchorPPHard := PureStochasticAnchorPPHard

/-- Fully mechanized stochastic minimum-sufficiency hardness package: MAJSAT
    reduces, with polynomial output-size bounds, to the `k = 0` slice of the
    stochastic minimum-sufficiency query. -/
abbrev HonestStochasticMinimumPPHard := StochasticMinimumSufficiencyPPHard

/-- Fully mechanized sequential hardness package: TQBF reduces, with polynomial
    output-size bounds, to empty-set sequential sufficiency. -/
abbrev HonestSequentialSufficiencyPSPACEHard := SequentialSufficiencyPSPACEHard

/-- Fully mechanized sequential minimum-sufficiency hardness package: TQBF
    reduces, with polynomial output-size bounds, to the `k = 0` slice of the
    sequential minimum-sufficiency query. -/
abbrev HonestSequentialMinimumPSPACEHard := SequentialMinimumSufficiencyPSPACEHard

/-- Fully mechanized sequential anchor hardness package: TQBF reduces, with
    polynomial output-size bounds, to empty-set sequential anchor sufficiency. -/
abbrev HonestSequentialAnchorPSPACEHard := SequentialAnchorPSPACEHard

theorem stochastic_sufficiency_pp_hard_honest (hn : n ≥ 1) :
    HonestStochasticSufficiencyPPHard n :=
  stochastic_sufficiency_pp_hard hn

theorem stochastic_anchor_check_pp_hard_honest (hn : n ≥ 1) :
    HonestStochasticAnchorPPHard n :=
  stochastic_anchor_check_pp_hard hn

theorem stochastic_minimum_sufficiency_pp_hard_honest (hn : n ≥ 1) :
    HonestStochasticMinimumPPHard n :=
  stochastic_minimum_sufficiency_pp_hard hn

theorem sequential_sufficiency_pspace_hard_honest (n : ℕ) :
    HonestSequentialSufficiencyPSPACEHard n :=
  sequential_sufficiency_pspace_hard n

theorem sequential_minimum_sufficiency_pspace_hard_honest (n : ℕ) :
    HonestSequentialMinimumPSPACEHard n :=
  sequential_minimum_sufficiency_pspace_hard n

theorem sequential_anchor_check_pspace_hard_honest (n : ℕ) :
    HonestSequentialAnchorPSPACEHard n :=
  sequential_anchor_check_pspace_hard n

end DecisionQuotient.StochasticSequential
