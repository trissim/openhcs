import jax
import jax.numpy as jnp
from jax import vmap, jit
from typing import Callable, Any, cast

from dq_dock_engine.arraydsl import (
    applyCutoff as dsl_apply_cutoff,
    coulombCutoff as dsl_coulomb_cutoff,
    elemBinaryAdd as dsl_elem_binary_add,
    elemBinarySub as dsl_elem_binary_sub,
    ewaldRealSpaceKernel as dsl_ewald_real_space_kernel,
    lennardJones as dsl_lennard_jones,
    minimumImagePairwiseDistances as dsl_minimum_image_pairwise_distances,
    pairwiseDistances3D as dsl_pairwise_distances_3d,
    rigidTransform3D as dsl_rigid_transform_3d,
    rowWiseDistance as dsl_row_wise_distance,
    rowWiseNorm as dsl_row_wise_norm,
    typedLennardJonesCutoff as dsl_typed_lj_cutoff,
    typedLennardJonesMatrix as dsl_typed_lj_matrix,
    upperTriangleMaskedSum as dsl_upper_triangle_masked_sum,
)

"""
JAX physics kernels routed through the generated Lean ArrayDSL bridge.

These wrappers preserve the historical engine API while delegating the core
array operations to generated JAX implementations derived from the Lean export.
"""


@jit
def elementwise_binary_add(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """DSL: elemBinaryAdd"""
    return dsl_elem_binary_add(a, b)


@jit
def elementwise_binary_sub(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """DSL: elemBinarySub"""
    return dsl_elem_binary_sub(a, b)


@jit
def norm(v: jnp.ndarray) -> jnp.ndarray:
    """DSL: norm"""
    return dsl_row_wise_norm(v)


@jit
def distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """DSL: distance"""
    return dsl_row_wise_distance(q1, q2)


@jit
def pairwise_distances(q1: jnp.ndarray, q2: jnp.ndarray) -> jax.Array:
    """DSL: pairwiseDistances. Shape: (N, M)"""
    return cast(jax.Array, dsl_pairwise_distances_3d(q1, q2))


@jit
def rigid_transform_3d(
    coords: jnp.ndarray, quaternion: jnp.ndarray, translation: jnp.ndarray
) -> jax.Array:
    """DSL: rigidTransform3D."""
    return cast(jax.Array, dsl_rigid_transform_3d(coords, quaternion, translation))


@jit
def minimum_image_pairwise_distances(
    coords1: jnp.ndarray, coords2: jnp.ndarray, box_size: jnp.ndarray
) -> jax.Array:
    """DSL: minimumImagePairwiseDistances."""
    return cast(
        jax.Array,
        dsl_minimum_image_pairwise_distances(coords1, coords2, box_size),
    )


@jit
def apply_cutoff(distances: jnp.ndarray, cutoff: float) -> jax.Array:
    """DSL: applyCutoff. Zero out distances beyond cutoff."""
    return cast(jax.Array, dsl_apply_cutoff(distances, cutoff))


@jit
def lennard_jones_potential(r: jnp.ndarray, epsilon: float, sigma: float) -> jax.Array:
    """
    DSL: lennardJones.
    U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    Includes singularity guard for r close to 0.
    """
    return cast(jax.Array, dsl_lennard_jones(epsilon, sigma, r))


@jit
def typed_lennard_jones_matrix(
    distances: jnp.ndarray, epsilons: jnp.ndarray, sigmas: jnp.ndarray
) -> jax.Array:
    """DSL: typedLennardJonesMatrix."""
    return cast(jax.Array, dsl_typed_lj_matrix(distances, epsilons, sigmas))


@jit
def typed_lennard_jones_cutoff(
    distances: jnp.ndarray,
    epsilons: jnp.ndarray,
    sigmas: jnp.ndarray,
    cutoff: float,
) -> jax.Array:
    """DSL: typedLennardJonesCutoff."""
    return cast(
        jax.Array,
        dsl_typed_lj_cutoff(distances, epsilons, sigmas, cutoff),
    )


@jit
def coulomb_cutoff(
    charges1: jnp.ndarray,
    charges2: jnp.ndarray,
    distances: jnp.ndarray,
    cutoff: float,
    dielectric: float = 1.0,
) -> jax.Array:
    """DSL: coulombCutoff."""
    return cast(
        jax.Array,
        dsl_coulomb_cutoff(charges1, charges2, distances, cutoff, dielectric),
    )


@jit
def upper_triangle_masked_sum(values: jnp.ndarray, mask: jnp.ndarray) -> jax.Array:
    """DSL: upperTriangleMaskedSum."""
    return cast(jax.Array, dsl_upper_triangle_masked_sum(values, mask))


@jit
def ewald_real_space_kernel(distances: jnp.ndarray, alpha: float) -> jax.Array:
    """DSL: ewaldRealSpaceKernel."""
    return cast(jax.Array, dsl_ewald_real_space_kernel(distances, alpha))


@jit
def sum_pair_potentials(
    q1: jnp.ndarray, q2: jnp.ndarray, potential_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """DSL: sumPairPotentials. Sums potential_fn(dist) for all pairs."""
    dists = pairwise_distances(q1, q2)
    return jnp.sum(potential_fn(dists))


def map_op(func: Callable[[Any], Any], operand: jnp.ndarray) -> jnp.ndarray:
    """DSL: map. Implemented via jax.vmap."""
    return vmap(func)(operand)


def reduce_sum(operand: jnp.ndarray) -> jnp.ndarray:
    """DSL: reduce. Implemented via jnp.sum."""
    return jnp.sum(operand)
