from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CertifiedBeliefState:
    prior: jax.Array
    posterior: jax.Array
    coarse_scores: jax.Array
    exact_scores: jax.Array
    exact_error_bound: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
    selected_action: int
    step_index: int


def uniform_prior(n_actions: int) -> jax.Array:
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    return jnp.full((n_actions,), 1.0 / n_actions)


def likelihood_from_survivor_mask(survivor_mask: jax.Array) -> jax.Array:
    return survivor_mask.astype(jnp.float32)


def update_posterior(prior: jax.Array, survivor_mask: jax.Array) -> jax.Array:
    weights = prior * likelihood_from_survivor_mask(survivor_mask)
    total_weight = float(jnp.sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "Bayes conditioning failed: survivor set has zero posterior mass"
        )
    return weights / total_weight


def select_admissible_action(posterior: jax.Array, ambiguity_mask: jax.Array) -> int:
    masked_posterior = jnp.where(ambiguity_mask, posterior, -jnp.inf)
    if not bool(jnp.any(jnp.isfinite(masked_posterior))):
        masked_posterior = posterior
    return int(jnp.argmax(masked_posterior))
