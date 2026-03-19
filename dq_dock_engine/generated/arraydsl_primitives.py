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


def rowWiseNorm(arr):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.rowWiseNorm."""
    return jnp.linalg.norm(arr, axis=-1)


def distance(arr1, arr2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.distance."""
    return jnp.linalg.norm(arr1 - arr2)


def rowWiseDistance(arr1, arr2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.rowWiseDistance."""
    return jnp.linalg.norm(arr1 - arr2, axis=-1)


def rigidTransform3D(coords, quaternion, translation):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.rigidTransform3D."""
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    R = jnp.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w], [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]])
    return (coords @ R.T) + translation


def pairwiseDistances(coords1, coords2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.pairwiseDistances."""
    return jnp.abs(coords1[:, None] - coords2[None, :])


def pairwiseDistances3D(coords1, coords2):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.pairwiseDistances3D."""
    return jnp.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)


def minimumImagePairwiseDistances(coords1, coords2, box_size):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.minimumImagePairwiseDistances."""
    diff = coords1[:, None, :] - coords2[None, :, :]
    wrapped = diff - box_size * jnp.round(diff / box_size)
    return jnp.linalg.norm(wrapped, axis=-1)


def applyCutoff(distances, rc):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.applyCutoff."""
    return jnp.where(distances < rc, distances, 0.0)


def lennardJones(epsilon, sigma, r):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.lennardJones."""
    safe_r = jnp.where(r > 1e-10, r, 1e-10)
    inv_r6 = (sigma / safe_r) ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4.0 * epsilon * (inv_r12 - inv_r6)
    return jnp.where(r > 1e-10, potential, 1e12)


def sumPairPotentials(distances, rc, epsilon, sigma):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.sumPairPotentials."""
    masked = applyCutoff(distances, rc)
    return jnp.sum(lennardJones(epsilon, sigma, masked))


def sumPairPotentialsMatrix(distances, rc, epsilon, sigma):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.sumPairPotentialsMatrix."""
    masked = jnp.where(distances < rc, distances, 0.0)
    return jnp.sum(lennardJones(epsilon, sigma, masked))


def sumPairPotentials3D(coords1, coords2, rc, epsilon, sigma):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.sumPairPotentials3D."""
    distances = pairwiseDistances3D(coords1, coords2)
    return sumPairPotentialsMatrix(distances, rc, epsilon, sigma)


def typedLennardJonesMatrix(distances, epsilons, sigmas):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.typedLennardJonesMatrix."""
    safe_r = jnp.where(distances > 1e-10, distances, 1e-10)
    inv_r6 = (sigmas / safe_r) ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4.0 * epsilons * (inv_r12 - inv_r6)
    return jnp.where(distances > 1e-10, potential, 1e12)


def typedLennardJonesCutoff(distances, epsilons, sigmas, rc):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.typedLennardJonesCutoff."""
    energies = typedLennardJonesMatrix(distances, epsilons, sigmas)
    return jnp.sum(jnp.where(distances < rc, energies, 0.0))


def coulombCutoff(charges1, charges2, distances, rc, dielectric):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.coulombCutoff."""
    charge_product = charges1[:, None] * charges2[None, :]
    within = (distances < rc) & (distances > 1e-10)
    safe_r = jnp.where(within, distances, 1.0)
    return jnp.sum(jnp.where(within, charge_product / (dielectric * safe_r), 0.0))


def upperTriangleMaskedSum(values, mask):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.upperTriangleMaskedSum."""
    upper = jnp.triu(jnp.ones_like(values, dtype=bool), k=1)
    return jnp.sum(jnp.where(upper & mask, values, 0.0))


def ewaldRealSpaceKernel(distances, alpha):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.ewaldRealSpaceKernel."""
    safe_r = jnp.where(distances > 1e-10, distances, 1e-10)
    return jnp.exp(-((alpha * safe_r) ** 2)) / safe_r



__all__ = [
    'map',
    'reduce_sum',
    'elemBinaryAdd',
    'elemBinarySub',
    'norm',
    'rowWiseNorm',
    'distance',
    'rowWiseDistance',
    'rigidTransform3D',
    'pairwiseDistances',
    'pairwiseDistances3D',
    'minimumImagePairwiseDistances',
    'applyCutoff',
    'lennardJones',
    'sumPairPotentials',
    'sumPairPotentialsMatrix',
    'sumPairPotentials3D',
    'typedLennardJonesMatrix',
    'typedLennardJonesCutoff',
    'coulombCutoff',
    'upperTriangleMaskedSum',
    'ewaldRealSpaceKernel'
]
