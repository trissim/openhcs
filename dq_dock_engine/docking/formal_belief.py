from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import (
    noopBiasedProbabilityVectorLike,
    normalizeProbabilityVector,
    supportConditioning,
    uniformProbabilityVectorLike,
)
from jax.tree_util import register_pytree_node_class
from dq_dock_engine.docking.formal_handles import (
    belief_witness_handle,
    posterior_update_theorem_handle,
    posterior_update_witness_handle,
    selection_branch_membership_handle,
    selection_theorem_handle,
    selection_witness_handle,
)


class SelectionBranch(str, Enum):
    AMBIGUITY_BAND = "ambiguity_band"
    SUPPORT_FALLBACK = "posterior_support"


class PosteriorUpdateBranch(str, Enum):
    SURVIVOR_CONDITIONING = "survivor_conditioning"


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedBeliefState:
    prior_spec: "CertifiedPriorSpec"
    prior: jax.Array
    posterior: jax.Array
    posterior_rule: str
    posterior_theorem: str
    coarse_scores: jax.Array
    exact_scores: jax.Array
    exact_error_bound: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
    selected_action: int
    selected_action_rule: str
    selected_action_theorem: str
    step_index: int

    def tree_flatten(self):
        children = (self.prior, self.posterior, self.coarse_scores, self.exact_scores, self.survivor_mask, self.ambiguity_mask, self.prior_spec)
        aux_data = (self.posterior_rule, self.posterior_theorem, self.exact_error_bound, self.selected_action, self.selected_action_rule, self.selected_action_theorem, self.step_index)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        prior, posterior, coarse_scores, exact_scores, survivor_mask, ambiguity_mask, prior_spec = children
        return cls(
            prior_spec=prior_spec,
            prior=prior,
            posterior=posterior,
            posterior_rule=aux_data[0],
            posterior_theorem=aux_data[1],
            coarse_scores=coarse_scores,
            exact_scores=exact_scores,
            exact_error_bound=aux_data[2],
            survivor_mask=survivor_mask,
            ambiguity_mask=ambiguity_mask,
            selected_action=aux_data[3],
            selected_action_rule=aux_data[4],
            selected_action_theorem=aux_data[5],
            step_index=aux_data[6],
        )


@dataclass(frozen=True)
class PosteriorUpdateWitness:
    branch: PosteriorUpdateBranch
    theorem_handle: str
    witness_handle: str


@dataclass(frozen=True)
class SelectionWitness:
    branch: SelectionBranch
    theorem_handle: str
    witness_handle: str
    selected_action: int


@dataclass(frozen=True)
class CertifiedBeliefWitness:
    posterior_update: PosteriorUpdateWitness
    selection: SelectionWitness
    witness_handle: str


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedPriorSpec:
    kind: Literal["uniform", "noop_biased"]
    noop_mass: float = 0.0

    def tree_flatten(self):
        children = ()
        aux_data = (self.kind, self.noop_mass)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(kind=aux_data[0], noop_mass=aux_data[1])


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
    weights = jnp.asarray(supportConditioning(prior, survivor_mask))
    total_weight = float(jnp.sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "Bayes conditioning failed: survivor set has zero posterior mass"
        )
    return normalizeProbabilityVector(weights)


def _select_first_action(mask: jax.Array) -> int:
    if not bool(jnp.any(mask)):
        raise ValueError("Formal action selection requires a non-empty admissible set")
    return int(jnp.argmax(mask.astype(jnp.int32)))


def select_admissible_action(posterior: jax.Array, ambiguity_mask: jax.Array) -> int:
    posterior_support = posterior > 0
    ambiguity_support = jnp.logical_and(ambiguity_mask, posterior_support)
    if bool(jnp.any(ambiguity_support)):
        return _select_first_action(ambiguity_support)
    return _select_first_action(posterior_support)


def selection_provenance(
    posterior: jax.Array, ambiguity_mask: jax.Array
) -> tuple[str, str]:
    posterior_support = posterior > 0
    ambiguity_support = jnp.logical_and(ambiguity_mask, posterior_support)
    if bool(jnp.any(ambiguity_support)):
        return (
            SelectionBranch.AMBIGUITY_BAND.value,
            selection_branch_membership_handle(SelectionBranch.AMBIGUITY_BAND.value),
        )
    return (
        SelectionBranch.SUPPORT_FALLBACK.value,
        selection_branch_membership_handle(SelectionBranch.SUPPORT_FALLBACK.value),
    )


def posterior_update_witness() -> PosteriorUpdateWitness:
    return PosteriorUpdateWitness(
        branch=PosteriorUpdateBranch.SURVIVOR_CONDITIONING,
        theorem_handle=posterior_update_theorem_handle(),
        witness_handle=posterior_update_witness_handle(),
    )


def selection_witness(
    posterior: jax.Array, ambiguity_mask: jax.Array, selected_action: int
) -> SelectionWitness:
    rule, theorem = selection_provenance(posterior, ambiguity_mask)
    return SelectionWitness(
        branch=SelectionBranch(rule),
        theorem_handle=selection_theorem_handle(rule),
        witness_handle=selection_witness_handle(rule),
        selected_action=selected_action,
    )


def belief_witness(
    posterior: jax.Array, ambiguity_mask: jax.Array, selected_action: int
) -> CertifiedBeliefWitness:
    selection = selection_witness(
        posterior=posterior,
        ambiguity_mask=ambiguity_mask,
        selected_action=selected_action,
    )
    return CertifiedBeliefWitness(
        posterior_update=posterior_update_witness(),
        selection=selection,
        witness_handle=belief_witness_handle(selection.branch.value),
    )


def update_posterior_batch(prior: jax.Array, survivor_masks: jax.Array) -> jax.Array:
    weights = jnp.where(survivor_masks, prior[None, :], 0.0)
    normalizers = jnp.sum(weights, axis=1, keepdims=True)
    return weights / normalizers


def select_admissible_actions(
    posterior_matrix: jax.Array, ambiguity_masks: jax.Array
) -> jax.Array:
    posterior_support = posterior_matrix > 0
    ambiguity_support = jnp.logical_and(ambiguity_masks, posterior_support)
    has_ambiguity = jnp.any(ambiguity_support, axis=1)
    selected_masks = jnp.where(
        has_ambiguity[:, None], ambiguity_support, posterior_support
    )
    return jnp.argmax(selected_masks.astype(jnp.int32), axis=1)
