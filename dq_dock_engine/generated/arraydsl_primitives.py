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


def supportConditioning(probs, mask):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.supportConditioning."""
    return jnp.where(mask, probs, 0.0)


def normalizeProbabilityVector(weights):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.normalizeProbabilityVector."""
    return weights / jnp.sum(weights)


def uniformProbabilityVectorLike(template):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.uniformProbabilityVectorLike."""
    weights = jnp.ones_like(template, dtype=jnp.float32)
    return weights / jnp.sum(weights)


def noopBiasedProbabilityVectorLike(template, noop_mass):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.noopBiasedProbabilityVectorLike."""
    n = template.shape[0]
    if n == 0:
        return jnp.zeros_like(template, dtype=jnp.float32)
    if n == 1:
        return jnp.ones_like(template, dtype=jnp.float32)
    remainder = (1.0 - noop_mass) / (n - 1)
    return jnp.concatenate([jnp.array([noop_mass], dtype=jnp.float32), jnp.full((n - 1,), remainder, dtype=jnp.float32)])


def topKWithTiesMask(utilities, k):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.topKWithTiesMask."""
    n = utilities.shape[0]
    if n == 0:
        return jnp.zeros_like(utilities, dtype=bool)
    if k <= 0:
        return jnp.zeros_like(utilities, dtype=bool)
    if k >= n:
        return jnp.ones_like(utilities, dtype=bool)
    kth_boundary = jnp.partition(utilities, n - k)[n - k]
    return utilities >= kth_boundary


def ambiguityBandMask(utilities, k, epsilon):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.ambiguityBandMask."""
    sorted_utilities = jnp.sort(utilities)[::-1]
    kth_boundary = sorted_utilities[jnp.maximum(k - 1, 0)]
    return utilities >= (kth_boundary - epsilon)


def stableArgmaxMasked(values, mask):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.stableArgmaxMasked."""
    return jnp.argmax(jnp.where(mask, values, -jnp.inf))


def axisAngleQuaternion(axis, angle):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.axisAngleQuaternion."""
    half = angle / 2.0
    s = jnp.sin(half)
    return jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=jnp.float32)


def localTranslationStencil3D(step):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.localTranslationStencil3D."""
    return jnp.array([[step, 0.0, 0.0], [-step, 0.0, 0.0], [0.0, step, 0.0], [0.0, -step, 0.0], [0.0, 0.0, step], [0.0, 0.0, -step]], dtype=jnp.float32)


def localRotationStencil3D(angle):
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.localRotationStencil3D."""
    axes = jnp.eye(3, dtype=jnp.float32)
    return jnp.stack([axisAngleQuaternion(axes[0], angle), axisAngleQuaternion(axes[0], -angle), axisAngleQuaternion(axes[1], angle), axisAngleQuaternion(axes[1], -angle), axisAngleQuaternion(axes[2], angle), axisAngleQuaternion(axes[2], -angle)], axis=0)


def quaternionDictionary8():
    """Generated wrapper for DecisionQuotient.Computation.ArrayDSL.quaternionDictionary8."""
    half = jnp.sqrt(jnp.array(0.5, dtype=jnp.float32))
    return jnp.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[half,half,0.0,0.0],[half,0.0,half,0.0],[half,0.0,0.0,half],[0.5,0.5,0.5,0.5]], dtype=jnp.float32)


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
    'supportConditioning',
    'normalizeProbabilityVector',
    'uniformProbabilityVectorLike',
    'noopBiasedProbabilityVectorLike',
    'topKWithTiesMask',
    'ambiguityBandMask',
    'stableArgmaxMasked',
    'axisAngleQuaternion',
    'localTranslationStencil3D',
    'localRotationStencil3D',
    'quaternionDictionary8',
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
