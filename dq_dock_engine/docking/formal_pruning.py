from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import ambiguityBandMask, topKWithTiesMask


@dataclass(frozen=True)
class CertifiedPruningCertificate:
    exact_top_k_mask: jax.Array
    exact_ambiguity_mask: jax.Array
    coarse_ambiguity_mask: jax.Array
    survivor_mask: jax.Array
    k: int
    delta: float


def _scores_to_utilities(scores: jax.Array) -> jax.Array:
    return -scores


def top_k_with_ties_mask(scores: jax.Array, k: int = 1) -> jax.Array:
    utilities = _scores_to_utilities(scores)
    return topKWithTiesMask(utilities, k)


def ambiguity_band_mask(scores: jax.Array, k: int, epsilon: float) -> jax.Array:
    if k <= 0:
        raise ValueError("k must be positive")
    utilities = _scores_to_utilities(scores)
    return ambiguityBandMask(utilities, k, epsilon)


def coarse_top1_ambiguity_mask(coarse_scores: jax.Array, delta: float) -> jax.Array:
    """Top-1 coarse ambiguity band with width justified by TK7.

    If exact and coarse scores differ uniformly by at most ``delta``, the exact
    top-1 action lies inside this coarse ambiguity band of width ``2 * delta``.
    """
    return ambiguity_band_mask(coarse_scores, k=1, epsilon=2.0 * delta)


def certified_survivor_mask(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    k: int,
    delta: float,
) -> jax.Array:
    return certified_pruning_certificate(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        k=k,
        delta=delta,
    ).survivor_mask


def certified_pruning_certificate(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    k: int,
    delta: float,
) -> CertifiedPruningCertificate:
    coarse_mask = top_k_with_ties_mask(coarse_scores, k)
    if delta <= 0:
        coarse_band = coarse_mask
        return CertifiedPruningCertificate(
            exact_top_k_mask=top_k_with_ties_mask(exact_scores, k),
            exact_ambiguity_mask=coarse_mask,
            coarse_ambiguity_mask=coarse_band,
            survivor_mask=coarse_mask,
            k=k,
            delta=delta,
        )

    exact_ambiguity = ambiguity_band_mask(exact_scores, k, delta)
    coarse_band = (
        coarse_top1_ambiguity_mask(coarse_scores, delta) if k == 1 else coarse_mask
    )
    return CertifiedPruningCertificate(
        exact_top_k_mask=top_k_with_ties_mask(exact_scores, k),
        exact_ambiguity_mask=exact_ambiguity,
        coarse_ambiguity_mask=coarse_band,
        survivor_mask=coarse_band
        if k == 1
        else jnp.logical_or(coarse_mask, exact_ambiguity),
        k=k,
        delta=delta,
    )
