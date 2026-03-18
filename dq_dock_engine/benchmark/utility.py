import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, List

"""
Utility Matrix construction for srank computation.
Builds U[a,s] from molecular binding problem for tractability analysis.
"""

@dataclass(frozen=True)
class BindingState:
    """Discretized molecular state for utility matrix."""
    position_hash: int
    energy: float

@dataclass(frozen=True)
class BindingAction:
    """Docking action (ligand pose)."""
    pose_index: int
    rmsd_to_native: float = 0.0

def build_utility_matrix(
    scoring_fn: Callable,
    states: List[BindingState],
    actions: List[BindingAction]
) -> jnp.ndarray:
    """
    Build U[a, s] = scoring_fn(action, state).
    
    For molecular docking:
    - actions = ligand poses
    - states = protein conformations
    - U(a, s) = -binding_energy(pose_a, conformation_s)
    """
    n_actions = len(actions)
    n_states = len(states)
    U = jnp.zeros((n_actions, n_states))
    
    for i, action in enumerate(actions):
        for j, state in enumerate(states):
            U = U.at[i, j].set(scoring_fn(action, state))
    
    return U

def build_from_energy_matrix(
    energies: jnp.ndarray,
    temperature: float = 300.0
) -> jnp.ndarray:
    """
    Convert energy matrix E[a,s] to utility matrix U[a,s] = -E[a,s].
    Negative because we want to maximize utility (minimize energy).
    """
    return -energies

def srank_from_utility(utility_matrix: jnp.ndarray) -> int:
    """
    Compute srank from utility matrix.
    Uses the structural router's separability and rank detection.
    """
    from dq_dock_engine.physics.tractability.separable import detect_separability, detect_low_tensor_rank
    
    if detect_separability(utility_matrix):
        return 0  # Separable: srank = 0
    
    rank = detect_low_tensor_rank(utility_matrix, max_rank=utility_matrix.shape[1])
    return rank

def utility_summary(utility_matrix: jnp.ndarray) -> dict:
    """Summary statistics of utility matrix."""
    return {
        "shape": utility_matrix.shape,
        "n_actions": utility_matrix.shape[0],
        "n_states": utility_matrix.shape[1],
        "srank": srank_from_utility(utility_matrix),
        "mean": float(jnp.mean(utility_matrix)),
        "std": float(jnp.std(utility_matrix)),
        "range": float(jnp.max(utility_matrix) - jnp.min(utility_matrix)),
    }
