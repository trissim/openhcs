"""Generated JAX wrappers for Lean ArrayDSL primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def map(f, arr):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.map."""
    return jax.vmap(f)(arr)


def reduce_sum(arr):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.reduce_sum."""
    return jnp.sum(arr)


def elemBinaryAdd(arr1, arr2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.elemBinaryAdd."""
    return arr1 + arr2


def elemBinarySub(arr1, arr2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.elemBinarySub."""
    return arr1 - arr2


def norm(arr):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.norm."""
    return jnp.linalg.norm(arr)


def distance(arr1, arr2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.distance."""
    return jnp.linalg.norm(arr1 - arr2)


def pairwiseDistances(coords1, coords2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.pairwiseDistances."""
    return jnp.abs(coords1[:, None] - coords2[None, :])


def applyCutoff(distances, rc):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.applyCutoff."""
    return jnp.where(distances < rc, distances, 0.0)


def lennardJones(epsilon, sigma, r):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.lennardJones."""
    safe_r = jnp.where(r == 0, 1.0, r)
    sr = sigma / safe_r
    energy = 4.0 * epsilon * (sr ** 12 - sr ** 6)
    return jnp.where(r == 0, 0.0, energy)


def sumPairPotentials(distances, rc, epsilon, sigma):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.sumPairPotentials."""
    masked = applyCutoff(distances, rc)
    energies = jax.vmap(lambda r: lennardJones(epsilon, sigma, r))(masked)
    return jnp.sum(energies)



__all__ = [
    'map',
    'reduce_sum',
    'elemBinaryAdd',
    'elemBinarySub',
    'norm',
    'distance',
    'pairwiseDistances',
    'applyCutoff',
    'lennardJones',
    'sumPairPotentials'
]
