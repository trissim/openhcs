import jax.numpy as jnp
from jax import jit

"""
Thermodynamic bounds implementation.
Aligns with ThermodynamicLift.lean: landauerConstant * card(S).
"""

# Physical Constants (SI Units)
KB = 1.380649e-23  # Boltzmann constant (J/K)
T_ROOM = 300.0      # Room temperature (K)
LN2 = jnp.log(2.0)

@jit
def landauer_constant(temperature: float = T_ROOM) -> float:
    """Fundamental energy limit to erase 1 bit of information: kT ln 2."""
    return KB * temperature * LN2

@jit
def thermodynamic_cost(srank: int, n_states: int, temperature: float = T_ROOM) -> float:
    """
    Fundamental energy cost of a binding decision.
    Cost = srank * log2(n_states) * kT * ln 2
    """
    # Bits = srank * log2(n_states)
    bits = srank * jnp.log2(float(n_states))
    return bits * landauer_constant(temperature)

@jit
def time_energy_tradeoff(energy_budget: float, srank: int, temperature: float = T_ROOM) -> float:
    """
    Minimal time required given an energy budget.
    t ≥ energy_budget / (kT * ln 2 * srank)
    """
    return energy_budget / (landauer_constant(temperature) * srank)

@jit
def bits_of_complexity(srank: int, precision_bits: int) -> float:
    """Estimated bits of architectural complexity for a given srank."""
    return float(srank * precision_bits)
