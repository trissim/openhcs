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

open DecisionQuotient.Physics.DimensionalComplexity

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

Summary of complexity results:

| Regime       | Problem                    | Complexity |
|--------------|----------------------------|------------|
| Static       | SUFFICIENCY-CHECK         | coNP-complete (Paper 4) |
| Stochastic   | STOCHASTIC-SUFFICIENCY    | PP-complete |
| Sequential   | SEQUENTIAL-SUFFICIENCY    | PSPACE-complete |

Transfer conditions:
- Static → Stochastic: product distributions
- Static → Sequential: horizon = 1, deterministic
- Stochastic → Sequential: memoryless transitions

All complexity results follow from:
1. Reduction from standard complete problems (MAJSAT, TQBF)
2. 6 tractable subcases machinery from IntegrityEquilibrium
-/

end DecisionQuotient.StochasticSequential
