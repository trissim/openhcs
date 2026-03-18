import jax
import jax.numpy as jnp
from jax import vmap, jit
from typing import Callable, Any

"""
JAX implementation of verified Array DSL primitives from ArrayDSL.lean.
These kernels form the 'common language' between Lean proofs and JAX execution.
"""

@jit
def elementwise_binary_add(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """DSL: elemBinaryAdd"""
    return a + b

@jit
def elementwise_binary_sub(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """DSL: elemBinarySub"""
    return a - b

@jit
def norm(v: jnp.ndarray) -> jnp.ndarray:
    """DSL: norm"""
    return jnp.linalg.norm(v, axis=-1)

@jit
def distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """DSL: distance"""
    return jnp.linalg.norm(q1 - q2, axis=-1)

@jit
def pairwise_distances(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """DSL: pairwiseDistances. Shape: (N, M)"""
    # Using broadcasting for efficient pairwise computation
    return jnp.linalg.norm(q1[:, None, :] - q2[None, :, :], axis=-1)

@jit
def apply_cutoff(distances: jnp.ndarray, cutoff: float) -> jnp.ndarray:
    """DSL: applyCutoff. Zero out distances beyond cutoff."""
    return jnp.where(distances < cutoff, distances, 0.0)

@jit
def lennard_jones_potential(r: jnp.ndarray, epsilon: float, sigma: float) -> jnp.ndarray:
    """
    DSL: lennardJones.
    U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    Includes singularity guard for r close to 0.
    """
    # Guard against division by zero
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    inv_r6 = (sigma / r_safe) ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4 * epsilon * (inv_r12 - inv_r6)
    # If r was below threshold, we return a very large value (repulsion)
    return jnp.where(r > 1e-10, potential, 1e12)

@jit
def sum_pair_potentials(
    q1: jnp.ndarray, 
    q2: jnp.ndarray, 
    potential_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> float:
    """DSL: sumPairPotentials. Sums potential_fn(dist) for all pairs."""
    dists = pairwise_distances(q1, q2)
    return jnp.sum(potential_fn(dists))

def map_op(func: Callable[[Any], Any], operand: jnp.ndarray) -> jnp.ndarray:
    """DSL: map. Implemented via jax.vmap."""
    return vmap(func)(operand)

def reduce_sum(operand: jnp.ndarray) -> float:
    """DSL: reduce. Implemented via jnp.sum."""
    return jnp.sum(operand)
