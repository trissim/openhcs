from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import ambiguityBandMask, topKWithTiesMask
from dq_dock_engine.docking.formal_handles import (
    CP1,
    CP2,
    TK1,
    TK11,
    TK12,
    TK4,
    survivor_set_witness_handle,
)


@dataclass(frozen=True)
class CertifiedPruningCertificate:
    exact_top_k_mask: jax.Array
    exact_ambiguity_mask: jax.Array
    coarse_ambiguity_mask: jax.Array
    survivor_mask: jax.Array
    k: int
    delta: float
    rule: str
    theorem_handle: str


@dataclass(frozen=True)
class CertifiedSurvivorSetWitness:
    survivor_mask: jax.Array
    certificate: CertifiedPruningCertificate
    theorem_handle: str


@dataclass(frozen=True)
class StagedTop1Guarantee:
    branch: Top1PruningBranch
    selected_action: int
    survivor_mask: jax.Array
    theorem_handle: str
    exact_winner_certified: bool


class Top1PruningBranch(str, Enum):
    EXACT_TOP1 = "exact_top1"
    EXACT_SINGLETON_WINNER = "exact_singleton_winner"
    TOP1_COARSE_AMBIGUITY_BAND = "top1_coarse_ambiguity_band"
    TOPK_MARGIN_SURVIVOR = "topk_margin_survivor"
    EXACT_TOPK = "exact_topk"


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


def coarse_top1_pairwise_margin_mask(
    coarse_scores: jax.Array, delta: float
) -> jax.Array:
    if coarse_scores.ndim != 1:
        raise ValueError("coarse_scores must be one-dimensional")
    best_index = int(jnp.argmin(coarse_scores))
    margins = coarse_scores - coarse_scores[best_index]
    strict_mask = margins > 2.0 * delta
    strict_mask = strict_mask.at[best_index].set(True)
    return strict_mask


def has_exact_singleton_winner_proof_condition(
    coarse_scores: jax.Array, delta: float
) -> bool:
    strict_mask = coarse_top1_pairwise_margin_mask(coarse_scores, delta)
    return bool(jnp.all(strict_mask))


def select_top1_pruning_branch(
    coarse_scores: jax.Array,
    delta: float,
) -> Top1PruningBranch:
    if delta <= 0:
        return Top1PruningBranch.EXACT_TOP1
    if has_exact_singleton_winner_proof_condition(coarse_scores, delta):
        return Top1PruningBranch.EXACT_SINGLETON_WINNER
    return Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND


def certified_top1_coarse_ambiguity_certificate(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> CertifiedPruningCertificate:
    return CertifiedPruningCertificate(
        exact_top_k_mask=top_k_with_ties_mask(exact_scores, k=1),
        exact_ambiguity_mask=ambiguity_band_mask(exact_scores, k=1, epsilon=delta),
        coarse_ambiguity_mask=coarse_top1_ambiguity_mask(coarse_scores, delta),
        survivor_mask=coarse_top1_ambiguity_mask(coarse_scores, delta),
        k=1,
        delta=delta,
        rule=Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND.value,
        theorem_handle=TK11,
    )


def certified_exact_singleton_winner_certificate(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> CertifiedPruningCertificate:
    best_index = int(jnp.argmin(coarse_scores))
    singleton_mask = jnp.zeros_like(coarse_scores, dtype=bool).at[best_index].set(True)
    return CertifiedPruningCertificate(
        exact_top_k_mask=top_k_with_ties_mask(exact_scores, k=1),
        exact_ambiguity_mask=singleton_mask,
        coarse_ambiguity_mask=singleton_mask,
        survivor_mask=singleton_mask,
        k=1,
        delta=delta,
        rule=Top1PruningBranch.EXACT_SINGLETON_WINNER.value,
        theorem_handle=TK12,
    )


def exact_path_top1_certificate(exact_scores: jax.Array) -> CertifiedPruningCertificate:
    top1_mask = top_k_with_ties_mask(exact_scores, k=1)
    return CertifiedPruningCertificate(
        exact_top_k_mask=top1_mask,
        exact_ambiguity_mask=top1_mask,
        coarse_ambiguity_mask=top1_mask,
        survivor_mask=top1_mask,
        k=1,
        delta=0.0,
        rule=Top1PruningBranch.EXACT_TOP1.value,
        theorem_handle=CP2,
    )


def certificate_of_top1_branch(
    branch: Top1PruningBranch,
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> CertifiedPruningCertificate:
    if branch == Top1PruningBranch.EXACT_TOP1:
        return exact_path_top1_certificate(exact_scores)
    if branch == Top1PruningBranch.EXACT_SINGLETON_WINNER:
        return certified_exact_singleton_winner_certificate(
            exact_scores=exact_scores,
            coarse_scores=coarse_scores,
            delta=delta,
        )
    if branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND:
        return certified_top1_coarse_ambiguity_certificate(
            exact_scores=exact_scores,
            coarse_scores=coarse_scores,
            delta=delta,
        )
    raise ValueError(f"Unsupported top-1 pruning branch: {branch}")


def survivor_set_of_top1_branch(
    branch: Top1PruningBranch,
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> CertifiedSurvivorSetWitness:
    certificate = certificate_of_top1_branch(
        branch=branch,
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        delta=delta,
    )
    return CertifiedSurvivorSetWitness(
        survivor_mask=certificate.survivor_mask,
        certificate=certificate,
        theorem_handle=survivor_set_witness_handle(branch.value),
    )


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


def certified_survivor_set_witness(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    k: int,
    delta: float,
) -> CertifiedSurvivorSetWitness:
    certificate = certified_pruning_certificate(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        k=k,
        delta=delta,
    )
    if k == 1:
        return survivor_set_of_top1_branch(
            branch=Top1PruningBranch(certificate.rule),
            exact_scores=exact_scores,
            coarse_scores=coarse_scores,
            delta=delta,
        )
    return CertifiedSurvivorSetWitness(
        survivor_mask=certificate.survivor_mask,
        certificate=certificate,
        theorem_handle=CP1,
    )


def certified_pruning_certificate(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    k: int,
    delta: float,
) -> CertifiedPruningCertificate:
    if k == 1 and delta > 0:
        return certificate_of_top1_branch(
            branch=select_top1_pruning_branch(coarse_scores, delta),
            exact_scores=exact_scores,
            coarse_scores=coarse_scores,
            delta=delta,
        )

    coarse_mask = top_k_with_ties_mask(coarse_scores, k)
    if delta <= 0:
        if k == 1:
            return certificate_of_top1_branch(
                branch=Top1PruningBranch.EXACT_TOP1,
                exact_scores=exact_scores,
                coarse_scores=coarse_scores,
                delta=delta,
            )
        coarse_band = coarse_mask
        return CertifiedPruningCertificate(
            exact_top_k_mask=top_k_with_ties_mask(exact_scores, k),
            exact_ambiguity_mask=coarse_mask,
            coarse_ambiguity_mask=coarse_band,
            survivor_mask=coarse_mask,
            k=k,
            delta=delta,
            rule=Top1PruningBranch.EXACT_TOPK.value,
            theorem_handle=TK1,
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
        rule=Top1PruningBranch.TOPK_MARGIN_SURVIVOR.value,
        theorem_handle=TK4,
    )


def staged_top1_guarantee_from_coarse_scores(
    coarse_scores: jax.Array,
    delta: float,
) -> StagedTop1Guarantee:
    branch = select_top1_pruning_branch(coarse_scores, delta)
    selected_action = int(jnp.argmin(coarse_scores))
    if branch == Top1PruningBranch.EXACT_SINGLETON_WINNER:
        survivor_mask = (
            jnp.zeros_like(coarse_scores, dtype=bool).at[selected_action].set(True)
        )
        return StagedTop1Guarantee(
            branch=branch,
            selected_action=selected_action,
            survivor_mask=survivor_mask,
            theorem_handle=TK12,
            exact_winner_certified=True,
        )

    survivor_mask = coarse_top1_ambiguity_mask(coarse_scores, delta)
    theorem_handle = (
        TK11 if branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND else CP2
    )
    return StagedTop1Guarantee(
        branch=branch,
        selected_action=selected_action,
        survivor_mask=survivor_mask,
        theorem_handle=theorem_handle,
        exact_winner_certified=(branch == Top1PruningBranch.EXACT_TOP1),
    )
