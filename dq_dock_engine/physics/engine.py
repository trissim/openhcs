import jax
import jax.numpy as jnp
from jax import jit, grad
from dataclasses import dataclass
from typing import Callable
from abc import ABC, abstractmethod

# --- Integrator ABC (Consistent Interface Design) ---

class Integrator(ABC):
    """ABC contract for all integrators: Verlet, Langevin, etc."""
    @abstractmethod
    def step(self, state: 'MDState', force_fn: Callable, dt: float) -> 'MDState':
        """Advance state by one timestep. Pure function."""

@dataclass(frozen=True)
class MDState:
    """
    Immutable MD state. PRNG key is explicitly injected, never defaulted.
    Consistent with OpenHCS frozen principles and JAX JIT compatibility.
    """
    positions: jnp.ndarray   # (N, 3)
    velocities: jnp.ndarray  # (N, 3)
    masses: jnp.ndarray      # (N,)
    time: float
    key: jnp.ndarray         # JAX PRNG key — explicitly injected, no default

# --- Velocity-Verlet (from SymplecticIntegrator.lean) ---

class VelocityVerlet(Integrator):
    """
    Direct translation of SymplecticIntegrator.lean::velocityVerletStep.
    Uses DSL primitives: elemBinaryAdd, map(mulConstDiff), computeForces.
    """
    def step(self, state: MDState, force_fn: Callable, dt: float) -> MDState:
        return _velocity_verlet_step(state, force_fn, dt)

def _velocity_verlet_step(state: MDState, force_fn: Callable, dt: float) -> MDState:
    q, v, m = state.positions, state.velocities, state.masses[:, None]

    f_t = force_fn(q)                          # computeForces U q
    v_half = v + (dt / 2.0) * (f_t / m)       # elemBinaryAdd p (map (mulConstDiff (dt/2)) F)
    q_next = q + dt * v_half                   # elemBinaryAdd q (map (mulConstDiff dt) v_half)
    f_next = force_fn(q_next)                  # computeForces U q_next
    v_next = v_half + (dt / 2.0) * (f_next / m)  # elemBinaryAdd p_half (map (mulConstDiff (dt/2)) F_next)

    return MDState(q_next, v_next, state.masses, state.time + dt, state.key)

# --- Langevin (from LangevinIntegrator.lean) ---

class Langevin(Integrator):
    """
    Direct translation of LangevinIntegrator.lean::langevinMomentumStep.
    dp = F(q)dt - γp dt + σ dW
    σ = sqrt(2γmkBT) (fluctuation-dissipation theorem).
    """
    def __init__(self, gamma: float, temperature: float, kb: float = 1.380649e-23):
        self.gamma = gamma
        self.temperature = temperature
        self.kb = kb

    def step(self, state: MDState, force_fn: Callable, dt: float) -> MDState:
        return _langevin_step(
            state, force_fn, dt, self.gamma, self.temperature, self.kb
        )

def _langevin_step(
    state: MDState, force_fn: Callable, dt: float,
    gamma: float, temperature: float, kb: float
) -> MDState:
    q, v, m = state.positions, state.velocities, state.masses[:, None]

    # σ = sqrt(2γmkBT) — fluctuationDissipationTheorem
    sigma = jnp.sqrt(2 * gamma * m * kb * temperature)

    # F = -∇U — computeForces
    F = force_fn(q)

    # impulse = F * dt — map (mulConstDiff dt) F
    impulse = F * dt / m

    # friction = -γ v dt — map (mulConstDiff (-γ*dt)) v
    friction = -gamma * v * dt

    # thermal = σ W / m — map (mulConstDiff σ) W
    key, subkey = jax.random.split(state.key)
    W = jax.random.normal(subkey, v.shape) * jnp.sqrt(dt)
    thermal = sigma * W / m

    # v_next = v + impulse + friction + thermal — three elemBinaryAdd
    v_next = v + impulse + friction + thermal

    # Position update (BAOAB-like splitting)
    q_next = q + dt * v_next

    return MDState(q_next, v_next, state.masses, state.time + dt, key)

# --- Force computation (from ArrayDSL.lean::computeForces) ---

def compute_forces(potential_fn: Callable, positions: jnp.ndarray) -> jnp.ndarray:
    """F = -∇U. Direct translation of ArrayDSL.lean::computeForces."""
    return -grad(potential_fn)(positions)

# --- Hamiltonian (from SymplecticIntegrator.lean::hamiltonian) ---

def hamiltonian(potential_fn: Callable, state: MDState) -> float:
    """H(q,p) = KE + PE. Direct translation of SymplecticIntegrator.lean::hamiltonian."""
    kinetic = 0.5 * jnp.sum(state.velocities ** 2 * state.masses[:, None])
    potential = potential_fn(state.positions)
    return kinetic + potential
