/-
# Leverage-Driven Software Architecture

Main entry point for all leverage framework proofs.

This formalization supports Paper 3: "Leverage-Driven Software Architecture:
A Probabilistic Framework for Design Decisions"

Key modules:
- Leverage/Foundations: Core definitions (Architecture, DOF, Capabilities, Leverage)
- Leverage/Probability: Error probability model and EXACT ordering theorems
- Leverage/Theorems: Main results (leverage-error tradeoff, optimality criteria)
- Leverage/SSOT: Instance showing Paper 2 fits the framework
- Leverage/Typing: Instance showing Paper 1 fits the framework
- Leverage/Examples: Concrete calculations for microservices, APIs, configuration, etc.
- Leverage/WeightedLeverage: Generalized leverage with value functions
- LambdaDR: λ_DR calculus - the core PL theory contribution (biconditional SSOT characterization)

Statistics (2025-01-08):
- Total lines: 2615+
- Definitions/Theorems/Lemmas: 240+
- Sorries: 0 (all proofs complete)

Proof Architecture:
1. All leverage arithmetic is machine-verified (cross-multiplication, transitivity)
2. Axis theory from Paper 1 is imported and used (orthogonality → independence)
3. Error ordering is EXACT (not approximate): linear and series models give identical orderings
4. The semantic bridge (orthogonality = statistical independence) is the standard
   information-theoretic interpretation - this is not a gap but foundational

All proofs are definitional or use decidable tactics (native_decide, omega, simp).
No axioms except standard Lean/Mathlib axioms.

Author: Formalized for Paper 3
Date: 2025-01-08
-/

import Leverage.Foundations
import Leverage.Probability
import Leverage.Theorems
import Leverage.Physical
import Leverage.SSOT
import Leverage.Typing
import Leverage.Examples
import Leverage.WeightedLeverage
import Leverage.FiveWayEquivalence
import Leverage.ColumnComplexityBridge
import Leverage.DockingTheoryBridge
import Leverage.CrossPaperDependencies
import LambdaDR

/-!
## Summary of Key Results

### Foundations (Leverage/Foundations.lean)
- `Architecture`: Core structure with DOF and capabilities
- `ssot_dof_eq_one`: SSOT has DOF = 1
- `lower_dof_higher_leverage`: Same caps, lower DOF → higher leverage
- `ssot_max_leverage`: SSOT maximizes leverage for fixed capabilities
- `leverage_antimonotone_dof`: Leverage is anti-monotone in DOF

### Probability (Leverage/Probability.lean)
- `ErrorRate`: Discrete probability model
- `lower_dof_lower_errors`: Lower DOF → fewer expected errors
- `ssot_minimal_errors`: SSOT minimizes expected errors
- `compose_increases_error`: Composition increases error (DOF adds)

### Main Theorems (Leverage/Theorems.lean)
- `leverage_error_tradeoff`: Max leverage ⟹ min error (Theorem 3.1)
- `metaprogramming_unbounded_leverage`: Metaprog achieves unbounded leverage (Theorem 3.2)
- `optimal_minimizes_error`: Optimal architecture minimizes error (Theorem 3.4)
- `ssot_dominates_non_ssot`: SSOT dominates non-SSOT (same caps)
- `ssot_pareto_optimal`: SSOT is Pareto-optimal

### Physical Grounding (Leverage/Physical.lean)
- `energyLowerBound_eq_joules_times_dof`: energy floor proportional to DOF
- `lower_dof_lower_energy`: lower DOF implies strictly lower energy floor
- `higher_leverage_same_caps_implies_lower_energy`: higher leverage lowers energy floor in fixed capability class

### SSOT Instance (Leverage/SSOT.lean)
- `ssot_leverage_dominance`: SSOT dominates non-SSOT by factor n
- `ssot_modification_complexity`: Modification ratio = n
- `ssot_unbounded_advantage`: Advantage grows unbounded
- `only_definition_hooks_achieve_ssot`: Python uniqueness (cites Paper 2)

### Typing Instance (Leverage/Typing.lean)
- `nominal_dominates_duck`: Nominal > duck leverage (Theorem 4.2.6)
- `capability_gap`: Gap = 4 B-dependent capabilities
- `nominal_strict_dominance`: Nominal wins on all metrics
- `complexity_gap_unbounded`: Type checking gap unbounded

### Examples (Leverage/Examples.lean)
- Microservices vs Monolith calculations
- REST API design (generic vs specific)
- Configuration (convention over configuration)
- Database normalization
- Error probability scaling

### Weighted Leverage (Leverage/WeightedLeverage.lean)
- ValueFunction structure (assigns values to capabilities)
- weighted_leverage definition (generalized leverage metric)
- Performance/reliability/security as value functions
- `cache_optimal_for_performance`: Cache wins under performance value
- `direct_optimal_for_maintainability`: Direct wins under uniform value
- `pareto_optimal_iff_value_optimal`: Every Pareto-optimal architecture is optimal for SOME value function
- Universal principle: All architectural "tradeoffs" are weighted leverage maximization

### Finite Compression Bridge (Leverage/ColumnComplexityBridge.lean)
- finite fiber/cardinality scaffold reused from Paper 1
- finite tie-broken argmin encoder induced by a compression Hamiltonian
- exact shared-codeword collision moment theorem in compression language
- finite zero-identity-debt transfer from raw argmin fibers to the tie-broken encoder

### Docking Theory Bridge (Leverage/DockingTheoryBridge.lean)
- local paper3 exposure of exact-sufficiency hardness core
- local cutoff-local structural-rank bounds for molecular docking
- docking-language bridge from general hardness to molecular low-rank regimes

### λ_DR Calculus (LambdaDR.lean)
- Core PL theory contribution: biconditional characterization of SSOT-capable languages
- `ssot_iff_both_capabilities`: SSOT achievable ↔ (defHook ∧ introspection)
- `capabilities_independent`: Neither capability alone suffices
- `complexity_gap_unbounded`: O(1) vs O(n) modification complexity
- `python_unique_mainstream`: Python is unique SSOT language in TIOBE top-10
- `four_ssot_viable_languages`: Python, CLOS, Smalltalk, Ruby are SSOT-complete
- `fragment_partition`: All languages map to one of 4 λ_DR fragments
- 25+ theorems, 0 sorry placeholders
-/
