import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Born-Oppenheimer approximation.
Direct translation of Computation/BornOppenheimer.lean.

Key axiom (EP4): In the adiabatic limit (m_e/m_N → 0), nuclear dynamics
evolve on a PES equal to the electronic ground state energy.

U(q) = E_0(q) where E_0 is the ground-state eigenvalue of H_el(q).
"""

@dataclass(frozen=True)
class FastHamiltonian:
    """
    Abstract electronic Hamiltonian parameterized by nuclear positions.
    From BornOppenheimer.lean::FastHamiltonian.
    
    ground_state_energy: q → E₀(q) — parametric ground-state energy.
    """
    ground_state_energy: Callable[[jnp.ndarray], float]

def born_oppenheimer_pes(H_electronic: FastHamiltonian) -> Callable[[jnp.ndarray], float]:
    """
    Born-Oppenheimer axiom (EP4):
    The classical PES U(q) = H_electronic.ground_state_energy(q).
    
    From born_oppenheimer in Lean:
      ∃ U : DiffFunctionN n, ∀ q, U.fn q = H_electronic.groundStateEnergy q
    
    This is the foundational axiom connecting QM to classical MD:
    nuclear dynamics on U(q) are valid when electronic relaxation is fast
    relative to nuclear motion (adiabatic separation).
    """
    return H_electronic.ground_state_energy

def validate_adiabatic_separation(
    pes_fn: Callable[[jnp.ndarray], float],
    fast_hamiltonian: FastHamiltonian,
    test_positions: jnp.ndarray
) -> float:
    """
    Validate that PES matches the fast Hamiltonian ground state energy.
    Returns max |U(q) - E₀(q)| over test positions.
    """
    errors = []
    for q in test_positions:
        u_val = pes_fn(q)
        e0_val = fast_hamiltonian.ground_state_energy(q)
        errors.append(abs(u_val - e0_val))
    return max(errors)
