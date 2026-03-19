from __future__ import annotations

import jax
import jax.numpy as jnp


def _scores_to_utilities(scores: jax.Array) -> jax.Array:
    return -scores


def top_k_with_ties_mask(scores: jax.Array, k: int = 1) -> jax.Array:
    utilities = _scores_to_utilities(scores)
    strict_better = utilities[None, :] > utilities[:, None]
    better_counts = jnp.sum(strict_better, axis=1)
    return better_counts < k


def ambiguity_band_mask(scores: jax.Array, k: int, epsilon: float) -> jax.Array:
    if k <= 0:
        raise ValueError("k must be positive")
    utilities = _scores_to_utilities(scores)
    sorted_utilities = jnp.sort(utilities)[::-1]
    kth_boundary = sorted_utilities[min(k - 1, len(sorted_utilities) - 1)]
    return utilities >= (kth_boundary - epsilon)


def certified_survivor_mask(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    k: int,
    delta: float,
) -> jax.Array:
    coarse_mask = top_k_with_ties_mask(coarse_scores, k)
    if delta <= 0:
        return coarse_mask

    exact_ambiguity = ambiguity_band_mask(exact_scores, k, delta)
    return jnp.logical_or(coarse_mask, exact_ambiguity)
