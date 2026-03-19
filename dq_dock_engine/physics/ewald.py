import jax.numpy as jnp
from jax import jit
from .kernels import (
    ewald_real_space_kernel,
    minimum_image_pairwise_distances,
    upper_triangle_masked_sum,
)

"""
Ewald Summation for long-range electrostatics.
Direct translation of EwaldSummation.lean.

PROOF STATUS SUMMARY:
  - ewald_real_space_energy: CONDITIONALLY_CERTIFIED (exponential decay proven)
  - ewald_reciprocal_energy: CONDITIONALLY_CERTIFIED (form proven)
  - ewald_self_energy: CERTIFIED (exact formula)
  - ewald_total: CONDITIONALLY_CERTIFIED

Lean mapping:
  - ewaldRealSpaceCore → ewald_real_space_energy
  - ewaldReciprocalCore → ewald_reciprocal_energy
  - ewald_real_space_exponential_decay → verified by construction (exp bound)
  - ewald_fourier_exponential_decay → verified by construction (positivity)
"""

from dq_dock_engine.proof_status import certified, conditionally_certified


@conditionally_certified(
    "EwaldSummation.lean::ewald_real_space_exponential_decay",
    assumptions=[
        "Real.erfc implementation correct",
        "Minimum image convention appropriate",
    ],
)
@jit
def ewald_real_space_energy(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    alpha: float,
    cutoff: float,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """
    Real-space Ewald sum — exponentially decaying.

    PROOF STATUS: CONDITIONALLY_CERTIFIED
      - Theorem: EwaldSummation.lean::ewald_real_space_exponential_decay
      - Decay: exp(-(αr)²) / r (exponential in αr)
      - Cutoff: HEURISTIC (convergence rate depends on α)
    """
    dists = minimum_image_pairwise_distances(positions, positions, box_size)
    within_cutoff = dists < cutoff

    real_kernel = ewald_real_space_kernel(dists, alpha)

    charge_product = charges[:, None] * charges[None, :]
    return upper_triangle_masked_sum(charge_product * real_kernel, within_cutoff)


@conditionally_certified(
    "EwaldSummation.lean::ewaldReciprocalCore",
    assumptions=[
        "k-space truncation sufficient",
        "Ewald splitting parameter α chosen appropriately",
    ],
)
@jit
def ewald_reciprocal_energy(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    alpha: float,
    k_max: int,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """
    Reciprocal-space (Fourier) Ewald sum.

    PROOF STATUS: CONDITIONALLY_CERTIFIED
      - Theorem: EwaldSummation.lean::ewaldReciprocalCore
      - Decay: (1/k²) * exp(-k²/4α²) (exponential in k/α)
      - k_max: HEURISTIC (typically 16-32 for good convergence)
    """
    volume = jnp.prod(box_size)

    k_range = jnp.arange(-k_max, k_max + 1)
    kx, ky, kz = jnp.meshgrid(k_range, k_range, k_range, indexing="ij")
    k_vecs = jnp.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=-1)
    k_vecs = k_vecs * (2 * jnp.pi / box_size)

    k_sq = jnp.sum(k_vecs**2, axis=-1)

    nonzero = k_sq > 1e-10
    k_sq_safe = jnp.where(nonzero, k_sq, 1.0)

    # Reciprocal kernel: (4π/k²) exp(-k²/4α²)
    recip_kernel = (4 * jnp.pi / k_sq_safe) * jnp.exp(-k_sq_safe / (4 * alpha**2))
    recip_kernel = jnp.where(nonzero, recip_kernel, 0.0)

    kr = jnp.dot(k_vecs, positions.T)
    S_real = jnp.sum(charges[None, :] * jnp.cos(kr), axis=1)
    S_imag = jnp.sum(charges[None, :] * jnp.sin(kr), axis=1)
    S_sq = S_real**2 + S_imag**2

    return jnp.sum(recip_kernel * S_sq) / (2 * volume)


@certified("EwaldSummation.lean")
@jit
def ewald_self_energy(charges: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Self-interaction correction: -α/√π Σ q_i².

    PROOF STATUS: CERTIFIED (exact formula)
    """
    return -alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2)


@conditionally_certified("EwaldSummation.lean")
def ewald_total(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    alpha: float,
    cutoff: float,
    k_max: int,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """
    Full Ewald energy = real + reciprocal + self correction.

    PROOF STATUS: CONDITIONALLY_CERTIFIED
      - All components: CONDITIONALLY_CERTIFIED
      - Total sum: conditionally certified if all assumptions hold
    """
    return (
        ewald_real_space_energy(positions, charges, alpha, cutoff, box_size)
        + ewald_reciprocal_energy(positions, charges, alpha, k_max, box_size)
        + ewald_self_energy(charges, alpha)
    )
