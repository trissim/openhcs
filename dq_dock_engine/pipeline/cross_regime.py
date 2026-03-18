from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

"""
Cross-Regime Transfer conditions.
Direct translation of StochasticSequential/CrossRegime.lean.

Key theorems:
- Tractability transfers across regimes (P stays P)
- Product distribution: Static → Stochastic (→ separable utility)
- Bounded horizon: Stochastic → Sequential (→ bounded treewidth)
- Full observability → tree structure
"""

class Regime(Enum):
    """Computational regime from complexity hierarchy."""
    STATIC = 0       # coNP
    STOCHASTIC = 1   # PP
    SEQUENTIAL = 2   # PSPACE

class ComplexityClass(Enum):
    """Complexity class for each regime."""
    P = auto()
    coNP = auto()
    PP = auto()
    PSPACE = auto()

# From CrossRegime.lean::baseComplexity
_BASE_COMPLEXITY = {
    Regime.STATIC: ComplexityClass.coNP,
    Regime.STOCHASTIC: ComplexityClass.PP,
    Regime.SEQUENTIAL: ComplexityClass.PSPACE,
}

def base_complexity(regime: Regime) -> ComplexityClass:
    """Base complexity of each regime."""
    return _BASE_COMPLEXITY[regime]

# --- Transfer Conditions ---

@dataclass(frozen=True)
class TransferCondition:
    """Condition for tractability to transfer between regimes."""
    from_regime: Regime
    to_regime: Regime
    condition_name: str
    is_satisfied: bool

def tractability_transfers(subcase_in_P: bool) -> bool:
    """
    From tractability_transfers:
    If a subcase is in P, it remains in P regardless of regime.
    Tractable subcases are regime-independent.
    """
    return subcase_in_P

def static_to_stochastic_transfer(has_product_distribution: bool) -> TransferCondition:
    """
    From product_enables_transfer:
    Product distribution → separable utility → static sufficiency transfers.
    """
    return TransferCondition(
        Regime.STATIC, Regime.STOCHASTIC,
        "product_distribution → separable_utility",
        has_product_distribution
    )

def stochastic_to_sequential_transfer(has_bounded_horizon: bool) -> TransferCondition:
    """
    From bounded_horizon_enables_transfer:
    Bounded horizon → bounded treewidth → stochastic sufficiency transfers.
    """
    return TransferCondition(
        Regime.STOCHASTIC, Regime.SEQUENTIAL,
        "bounded_horizon → bounded_treewidth",
        has_bounded_horizon
    )

def full_observability_transfer(is_fully_observable: bool) -> TransferCondition:
    """
    From fully_observable_enables_transfer:
    Full observability → tree structure → tractable.
    """
    return TransferCondition(
        Regime.STOCHASTIC, Regime.SEQUENTIAL,
        "fully_observable → tree_structure",
        is_fully_observable
    )

def check_all_transfers(
    has_product_dist: bool = False,
    has_bounded_horizon: bool = False,
    is_fully_observable: bool = False
) -> list:
    """Check all transfer conditions and return satisfied ones."""
    conditions = [
        static_to_stochastic_transfer(has_product_dist),
        stochastic_to_sequential_transfer(has_bounded_horizon),
        full_observability_transfer(is_fully_observable),
    ]
    return [c for c in conditions if c.is_satisfied]
