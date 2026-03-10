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

/-! ## Complexity Summary

Summary of complexity results and mechanization status:

| Regime       | Problem                              | Complexity |
|--------------|--------------------------------------|------------|
| Static       | SUFFICIENCY / MINIMUM / ANCHOR       | coNP-c / coNP-c / Sigma2P-c |
| Stochastic   | SUFFICIENCY / MINIMUM / ANCHOR       | PP-c / PP-hard / PP-hard |
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
