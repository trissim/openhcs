/-
  Paper 4b: Stochastic and Sequential Regimes

  Summary.lean - Summary of main results

  Collects all main theorems for easy reference.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Hierarchy
import DecisionQuotient.StochasticSequential.Tractability
import DecisionQuotient.StochasticSequential.SubstrateCost
import DecisionQuotient.StochasticSequential.CrossRegime
import DecisionQuotient.StochasticSequential.PreservationVariants

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient.DimensionalComplexity

/-! ## Main Theorems -/

-- Complexity hierarchy from Hierarchy.lean
#check static_simpler_than_stochastic
#check stochastic_simpler_than_sequential

-- Substrate independence
#check substrate_independence_verdict

-- Tractability via subcases
#check product_distribution_tractable
#check bounded_horizon_tractable

-- Preservation variants
#check StochasticMinimumPreservation
#check StochasticAnchorPreservation
#check stochastic_minimum_preservation_iff_static_of_full_support
#check stochastic_anchor_preservation_implies_static_anchor
#check static_anchor_implies_stochastic_anchor_preservation_of_full_support
#check stochastic_anchor_preservation_iff_static_anchor_of_full_support
#check stochasticMinimumPreservation_counted_search_witness
#check stochasticAnchorPreservation_counted_search_witness

/-! ## Complexity Summary

Summary of complexity results and mechanization status:

| Regime       | Problem                              | Complexity |
|--------------|--------------------------------------|------------|
| Static       | SUFFICIENCY / MINIMUM / ANCHOR       | coNP-c / coNP-c / Sigma2P-c |
| Stochastic   | preservation base / decisiveness / decisiveness min / decisiveness anchor | P(explicit) / PP-hard / PP-hard / PP-hard |
| Sequential   | SUFFICIENCY / MINIMUM / ANCHOR       | PSPACE-c / PSPACE-hard / PSPACE-hard |

Transfer conditions:
- Static → Stochastic: product distributions
- Static → Sequential: horizon = 1, deterministic
- Stochastic → Sequential: memoryless transitions

The artifact now also internalizes exact finite boolean deciders for the query
predicates. The hardness side is fully mechanized via size-bounded reductions
from standard complete problems (MAJSAT, TQBF). Full TM-witness membership
proofs for PP/PSPACE are not yet packaged end-to-end in this repository.
-/

end DecisionQuotient.StochasticSequential
