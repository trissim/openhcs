import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable
from enum import Enum, auto

"""
Physical Core theorems.
Direct translation of Physics/PhysicalCore.lean.

Key results:
1. Finite budget blocks exact certification beyond threshold
2. Substrate transport soundness
3. Certified bit accounting → physical work lower bound
4. Evidence-gated abstain optimality (RLFF)
"""

# --- Physical Constants (from PhysicalCore + BoundedAcquisition) ---

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Universal physical constants from the Lean formalizations.
    Frozen: no mutation, explicit injection.
    """
    kB: float = 1.380649e-23       # Boltzmann constant (J/K)
    T: float = 300.0               # Temperature (K)
    c: float = 2.998e8             # Speed of light (m/s)
    hbar: float = 1.054571817e-34  # Reduced Planck constant (J·s)
    
    @property
    def landauer_floor(self) -> float:
        """k_B × T × ln(2) — minimum energy per bit erasure."""
        return self.kB * self.T * jnp.log(2.0)

CONSTANTS = PhysicalConstants()

# --- Finite Budget Impossibility (PhysicalCore §1) ---

def bit_energy_cost(joulesPerBit: float) -> float:
    """Energy cost per bit operation. From PhysicalCore::bit_energy_cost."""
    return joulesPerBit

def feasible_work(energy_max: float, bits: int, joulesPerBit: float) -> bool:
    """
    Physical feasibility: bits × joulesPerBit ≤ E_max.
    From PhysicalCore::feasibleWork.
    """
    return bits * joulesPerBit <= energy_max

def budget_exceeds_threshold(
    energy_max: float,
    required_ops: Callable[[int], int],
    joulesPerBit: float
) -> int:
    """
    Find n₀ where required_ops(n) × joulesPerBit exceeds E_max.
    
    From finite_budget_blocks_exact_certification_beyond_threshold:
      ∃ n₀, ∀ n ≥ n₀, E_max < required_ops(n) × bit_energy_cost
    
    Returns n₀ (the threshold size).
    """
    n = 1
    while required_ops(n) * joulesPerBit <= energy_max:
        n += 1
        if n > 10000:  # Safety bound
            raise ValueError("Required ops do not grow fast enough to exceed budget")
    return n

# --- Certified Bit Accounting (PhysicalCore §4) ---

@dataclass(frozen=True)
class ReportBitModel:
    """
    Bit-accounting model for claim reports.
    From PhysicalCore::ReportBitModel.
    """
    raw_bits: int        # Bits for the raw answer
    cert_bits: int       # Additional bits for certification
    
    @property
    def total_certified_bits(self) -> int:
        """raw + certification overhead."""
        return self.raw_bits + self.cert_bits

def certified_work_lower_bound(
    bit_model: ReportBitModel,
    joulesPerBit: float
) -> float:
    """
    Physical work ≥ total_certified_bits × joulesPerBit.
    
    From certified_work_lower_bound_raw:
      rawBits × cost ≤ certifiedTotalBits × cost
    """
    return bit_model.total_certified_bits * joulesPerBit

def raw_work_lower_bound(
    bit_model: ReportBitModel,
    joulesPerBit: float
) -> float:
    """Raw work (without certification)."""
    return bit_model.raw_bits * joulesPerBit

# --- Evidence-Gated Abstain (PhysicalCore §3) ---

class ClaimReport(Enum):
    """Report types for RLFF."""
    EXACT = auto()
    EPSILON = auto()
    ABSTAIN = auto()

@dataclass(frozen=True)
class RLFFWeights:
    """Reward weights for RLFF decision."""
    abstain_reward: float = 0.0
    inadmissible_penalty: float = 1.0

def rlff_reward(
    report: ClaimReport,
    has_certificate: bool,
    weights: RLFFWeights
) -> float:
    """
    RLFF reward function.
    
    From rlff_abstain_strict_global_maximizer_of_no_certificates:
    If no certificates exist and abstain_reward > -inadmissible_penalty,
    then abstain is the strict global maximizer.
    """
    if report == ClaimReport.ABSTAIN:
        return weights.abstain_reward
    elif has_certificate:
        return 1.0  # Full reward for certified claim
    else:
        return -weights.inadmissible_penalty

def abstain_is_optimal(weights: RLFFWeights, has_any_certificate: bool) -> bool:
    """
    From rlff_abstain_strict_global_maximizer_of_no_certificates:
    Abstain is optimal when no certificates exist and abstain beats penalty.
    """
    if has_any_certificate:
        return False
    return -weights.inadmissible_penalty < weights.abstain_reward
