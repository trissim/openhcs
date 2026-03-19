from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import (
    noopBiasedProbabilityVectorLike,
    normalizeProbabilityVector,
    stableArgmaxMasked,
    supportConditioning,
    uniformProbabilityVectorLike,
)


@dataclass(frozen=True)
class CertifiedBeliefState:
    prior_spec: "CertifiedPriorSpec"
    prior: jax.Array
    posterior: jax.Array
    coarse_scores: jax.Array
    exact_scores: jax.Array
    exact_error_bound: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
    selected_action: int
    step_index: int


@dataclass(frozen=True)
class CertifiedPriorSpec:
    kind: Literal["uniform", "noop_biased"]
    noop_mass: float = 0.0


def build_prior(prior_spec: CertifiedPriorSpec, n_actions: int) -> jax.Array:
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    match prior_spec.kind:
        case "uniform":
            return uniformProbabilityVectorLike(
                jnp.zeros((n_actions,), dtype=jnp.float32)
            )
        case "noop_biased":
            if not 0.0 <= prior_spec.noop_mass < 1.0:
                raise ValueError("noop_mass must lie in [0, 1)")
            return noopBiasedProbabilityVectorLike(
                jnp.zeros((n_actions,), dtype=jnp.float32),
                prior_spec.noop_mass,
            )
        case _:
            raise ValueError(f"Unknown prior kind: {prior_spec.kind}")


def likelihood_from_survivor_mask(survivor_mask: jax.Array) -> jax.Array:
    return survivor_mask.astype(jnp.float32)


def update_posterior(prior: jax.Array, survivor_mask: jax.Array) -> jax.Array:
    weights = supportConditioning(prior, survivor_mask)
    total_weight = float(jnp.sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "Bayes conditioning failed: survivor set has zero posterior mass"
        )
    return normalizeProbabilityVector(weights)


def select_admissible_action(posterior: jax.Array, ambiguity_mask: jax.Array) -> int:
    if bool(jnp.any(ambiguity_mask)):
        return int(stableArgmaxMasked(posterior, ambiguity_mask))
    return int(stableArgmaxMasked(posterior, jnp.ones_like(ambiguity_mask, dtype=bool)))
