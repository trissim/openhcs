import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Bounded Acquisition — BA1 through BA10.
Direct translation of Physics/BoundedAcquisition.lean.

BA10: Counting Gap Theorem (pure math, no physics)
BA1:  BoundedRegion model (diameter d, signal speed c)
BA2:  Bounded acquisition rate: ≤ c×T/d events in time T
BA3:  Acquisitions are discrete (from QM)
BA4:  One transition = one bit
BA5:  Resolution reads sufficient coordinates
BA6:  Bit operations ≥ srank
BA7:  Energy ≥ srank × joulesPerBit
BA8:  srank = 1 is the energy ground state
BA9:  Physical grounding bundle: info + time + energy = srank
"""

# --- BA10: Counting Gap Theorem ---

def counting_gap_theorem(cost_per_check: int, total_capacity: int) -> int:
    """
    BA10: Any bounded system with positive per-check cost and finite
    capacity has a finite maximum check count.
    
    From counting_gap_theorem:
      ∃ c_max, 0 < c_max ∧ ∀ checks, cost × checks ≤ capacity → checks ≤ c_max
    
    Returns c_max = total_capacity (the bound).
    """
    if cost_per_check <= 0:
        raise ValueError("cost_per_check must be positive (Landauer: checks cost energy)")
    if total_capacity <= 0:
        raise ValueError("total_capacity must be positive")
    return total_capacity  # c_max = capacity, since checks ≤ cost*checks ≤ capacity

# --- BA1: Bounded Region ---

@dataclass(frozen=True)
class BoundedRegion:
    """
    BA1: A bounded region of spacetime.
    diameter: d (positive)
    signal_speed: c (positive, from SR)
    
    From BoundedAcquisition.lean::BoundedRegion.
    """
    diameter: float
    signal_speed: float = 2.998e8  # c, from SR
    
    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")
        if self.signal_speed <= 0:
            raise ValueError("signal_speed must be positive")

    @property
    def max_acquisition_rate(self) -> float:
        """BA1: Maximum rate = c/d events per unit time."""
        return self.signal_speed / self.diameter

# --- BA2: Bounded Acquisition Rate ---

def max_acquisitions(region: BoundedRegion, time: float) -> float:
    """
    BA2: Maximum acquisition events in time T.
    
    From maxAcquisitions: signalSpeed × T / diameter.
    """
    return region.signal_speed * time / region.diameter

# --- BA3-BA4: Discrete Transitions = Bits ---

@dataclass(frozen=True)
class DiscreteSystem:
    """
    BA3: A discrete system with finite state space.
    Transitions are discrete events (from QM: eigenstate jumps).
    BA4: Each transition = 1 bit of information.
    """
    n_states: int
    step_fn: Callable[[int], int]  # state → next state
    
    def bit_operations(self, initial_state: int, n_steps: int) -> int:
        """Count transitions (bit operations) over n_steps."""
        state = initial_state
        bits = 0
        for _ in range(n_steps):
            next_state = self.step_fn(state)
            if next_state != state:
                bits += 1  # One transition = one bit (BA4)
            state = next_state
        return bits

# --- BA5-BA6: Resolution Requires ≥ srank Bits ---

def srank_le_resolution_bits(srank: int, sufficient_set_size: int) -> bool:
    """
    BA6: srank ≤ |I| for any sufficient coordinate set I.
    
    From srank_le_resolution_bits:
      dp.srank ≤ I.card
    
    Physical meaning: any correct resolver must read ≥ srank coordinates,
    each read = 1 bit operation (BA4).
    """
    return srank <= sufficient_set_size

# --- BA7: Energy ≥ srank × joulesPerBit ---

def energy_lower_bound(srank: int, joules_per_bit: float) -> float:
    """
    BA7: Energy cost ≥ srank × joulesPerBit.
    
    From energy_ge_srank_cost:
      joulesPerBit × srank ≤ energyLowerBound M |I|
    
    Derived from TD (Landauer) applied to BA6.
    """
    return srank * joules_per_bit

def srank_one_is_ground_state(joules_per_bit: float) -> float:
    """
    BA8: srank = 1 is the energy ground state.
    Minimum energy per decision = 1 × joulesPerBit.
    """
    return joules_per_bit

# --- BA9: Physical Grounding Bundle ---

@dataclass(frozen=True)
class PhysicalGroundingBundle:
    """
    BA9: The three costs unify at srank.
    
    From physical_grounding_bundle:
    1. Information: srank ≤ |I| (min bits to resolve)
    2. Energy: srank × joulesPerBit ≤ total energy
    3. Mandatory: energy > 0 when |I| > 0
    """
    srank: int
    sufficient_set_size: int
    joules_per_bit: float
    
    @property
    def information_cost(self) -> int:
        """Minimum bits to resolve: srank."""
        return self.srank
    
    @property
    def energy_cost(self) -> float:
        """Minimum energy: srank × joulesPerBit."""
        return self.srank * self.joules_per_bit
    
    @property
    def time_cost(self, region: BoundedRegion = None) -> float:
        """Minimum time: srank × d/c (from BA2 + BA6)."""
        if region is None:
            return float(self.srank)  # In natural units
        return self.srank * region.diameter / region.signal_speed
    
    def validate(self) -> bool:
        """Verify BA9 conditions hold."""
        return (
            self.srank <= self.sufficient_set_size and  # BA6
            self.energy_cost > 0 and                     # BA7 + BA8
            self.sufficient_set_size > 0                 # non-trivial
        )
