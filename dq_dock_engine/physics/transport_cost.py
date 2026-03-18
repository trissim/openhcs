import jax.numpy as jnp
from dataclasses import dataclass

"""
Transport Cost — Wasserstein distance and srank.
Direct translation of Physics/TransportCost.lean.

Key theorems:
- TC1: relevant_state_count = 2^srank
- TC2: more states → higher transport cost
- TC3: transport ≥ 1 when srank > 0
- TC6: srank = unified complexity measure (5 independent derivations)
"""

def relevant_state_count(srank: int) -> int:
    """
    TC1: 2^srank distinguishable states.
    From TransportCost.lean::relevantStateCount.
    """
    return 2 ** srank

def more_states_more_transport(srank1: int, srank2: int) -> bool:
    """
    TC2: srank1 < srank2 → relevantStateCount(dp1) < relevantStateCount(dp2).
    From more_states_more_transport.
    """
    return srank1 < srank2 and relevant_state_count(srank1) < relevant_state_count(srank2)

def transport_lower_bound(srank: int) -> int:
    """
    TC3: Minimum transport ≥ 1 when srank > 0.
    From transport_lower_bound.
    """
    if srank <= 0:
        return 0
    return relevant_state_count(srank)

@dataclass(frozen=True)
class UnifiedComplexity:
    """
    TC6: srank as unified complexity measure.
    Five independent derivations:
    1. Computational: P vs coNP threshold
    2. Information: R(0) = srank bits (rate-distortion)
    3. Energy: srank × kT ln 2 joules (Landauer)
    4. Precision: TUR scales with entropy ~ srank
    5. Transport: Wasserstein scales with 2^srank states
    """
    srank: int
    
    @property
    def computational(self) -> str:
        return "P" if self.srank == 0 else "coNP-hard"
    
    @property
    def information_bits(self) -> int:
        """R(0) = srank."""
        return self.srank
    
    @property
    def energy_cost(self) -> float:
        """srank × kT ln 2 at 300K."""
        kT = 1.380649e-23 * 300.0
        return self.srank * kT * float(jnp.log(2.0))
    
    @property
    def transport_states(self) -> int:
        """2^srank distinguishable states."""
        return relevant_state_count(self.srank)
    
    @property
    def tur_entropy_scale(self) -> float:
        """Entropy production scales with srank."""
        return float(self.srank) * float(jnp.log(2.0))
