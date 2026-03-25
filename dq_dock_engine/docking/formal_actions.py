"""Certified action families for local optimization.

Step-size control is theorem-driven rather than fixed by user heuristics:
  - LS1: `lipschitz_step_bound` proves Δ ≤ ε/L guarantees |ΔU| ≤ ε
  - LS2/LS3: `optimalTranslationStep` and its exact budget theorem
  - LS5/LS6: lower Lipschitz constants induce exactly larger optimal steps
  - SH8-SH14: dyadic shell rounds are chosen by least adequate joint resolution

Adaptive step size from softened Lipschitz (PERF5):
  - softened_grid_speedup_ratio: L_soft ≤ L_raw → δ/L_raw ≤ δ/L_soft
  - When softened scoring has tighter Lipschitz, step can scale by L_raw/L_soft
  - Conditions: rSoft ≥ 0.8σ, rSoft ≤ σ (proven in SoftLJApproximation.lean)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from dq_dock_engine.arraydsl import (
    axisAngleQuaternion,
    localRotationStencil3D,
    localTranslationStencil3D,
)
from dq_dock_engine.docking.formal_handles import support_expansion_theorem_handles
from dq_dock_engine.physics.kernels import rigid_transform_3d


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedLocalAction:
    action_id: int
    translation_delta: jax.Array
    quaternion_delta: jax.Array
    is_noop: bool = False

    def tree_flatten(self):
        children = (self.translation_delta, self.quaternion_delta)
        aux_data = (self.action_id, self.is_noop)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            action_id=aux_data[0],
            translation_delta=children[0],
            quaternion_delta=children[1],
            is_noop=aux_data[1],
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedActionFamily:
    actions: tuple[CertifiedLocalAction, ...]
    translation_deltas: jax.Array
    quaternion_deltas: jax.Array
    translation_step: float
    rotation_step_rad: float
    stencil_level: int
    support_shell_levels: tuple[int, ...]
    theorem_handles: tuple[str, ...] = ()

    def tree_flatten(self):
        children = (self.actions, self.translation_deltas, self.quaternion_deltas)
        aux_data = (
            self.translation_step,
            self.rotation_step_rad,
            self.stencil_level,
            self.support_shell_levels,
            self.theorem_handles,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


def _normalize_quaternion(quaternion: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(quaternion)
    return quaternion / norm


def _axis_angle_to_quaternion(axis: jax.Array, angle_rad: float) -> jax.Array:
    quat = jnp.asarray(axisAngleQuaternion(axis, angle_rad))
    return _normalize_quaternion(quat)


@lru_cache(maxsize=128)
def _cached_action_family(
    translation_step: float,
    rotation_step_rad: float,
    stencil_level: int,
) -> CertifiedActionFamily:
    translations = tuple(
        jnp.asarray(localTranslationStencil3D(translation_step))[i] for i in range(6)
    )

    identity_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])
    actions = [
        CertifiedLocalAction(
            action_id=0,
            translation_delta=jnp.zeros(3),
            quaternion_delta=identity_quaternion,
            is_noop=True,
        )
    ]

    for idx, translation in enumerate(translations, start=1):
        actions.append(
            CertifiedLocalAction(
                action_id=idx,
                translation_delta=translation,
                quaternion_delta=identity_quaternion,
            )
        )

    rotation_stencil = jnp.asarray(localRotationStencil3D(rotation_step_rad))
    next_action_id = len(actions)
    for quaternion_delta in rotation_stencil:
        actions.append(
            CertifiedLocalAction(
                action_id=next_action_id,
                translation_delta=jnp.zeros(3),
                quaternion_delta=_normalize_quaternion(quaternion_delta),
            )
        )
        next_action_id += 1

    actions_tuple = tuple(actions)

    return CertifiedActionFamily(
        actions=actions_tuple,
        translation_deltas=jnp.stack(
            [action.translation_delta for action in actions_tuple], axis=0
        ),
        quaternion_deltas=jnp.stack(
            [action.quaternion_delta for action in actions_tuple], axis=0
        ),
        translation_step=translation_step,
        rotation_step_rad=rotation_step_rad,
        stencil_level=stencil_level,
        support_shell_levels=(stencil_level,),
        theorem_handles=support_expansion_theorem_handles(),
    )


def create_certified_action_family(
    translation_step: float,
    rotation_step_rad: float,
    stencil_level: int,
) -> CertifiedActionFamily:
    return _cached_action_family(
        float(translation_step), float(rotation_step_rad), int(stencil_level)
    )


def merge_certified_action_families(
    families: tuple[CertifiedActionFamily, ...],
) -> CertifiedActionFamily:
    if not families:
        raise ValueError("families must be non-empty")
    if not all(family.actions and family.actions[0].is_noop for family in families):
        raise ValueError("each family must expose a noop action at index 0")

    merged_shell_levels: list[int] = []
    merged_actions: list[CertifiedLocalAction] = []

    for family_index, family in enumerate(families):
        for shell_level in family.support_shell_levels:
            if shell_level not in merged_shell_levels:
                merged_shell_levels.append(shell_level)
        source_actions = family.actions if family_index == 0 else family.actions[1:]
        merged_actions.extend(source_actions)

    reindexed_actions = tuple(
        CertifiedLocalAction(
            action_id=action_id,
            translation_delta=action.translation_delta,
            quaternion_delta=action.quaternion_delta,
            is_noop=action.is_noop,
        )
        for action_id, action in enumerate(merged_actions)
    )
    return CertifiedActionFamily(
        actions=reindexed_actions,
        translation_deltas=jnp.stack(
            [action.translation_delta for action in reindexed_actions], axis=0
        ),
        quaternion_deltas=jnp.stack(
            [action.quaternion_delta for action in reindexed_actions], axis=0
        ),
        translation_step=max(family.translation_step for family in families),
        rotation_step_rad=max(family.rotation_step_rad for family in families),
        stencil_level=max(merged_shell_levels),
        support_shell_levels=tuple(merged_shell_levels),
        theorem_handles=support_expansion_theorem_handles(),
    )


def compute_adaptive_translation_step(
    base_step: float,
    epsilon_lj: float,
    min_pairwise_sigma: float,
    r_soft: float,
) -> float:
    """PERF5: adaptive step from softened Lipschitz constant.

    All parameters are derived from molecular data, not heuristics:
      - epsilon_lj: LJ well depth from scoring._EPSILON_KCAL_MOL
      - min_pairwise_sigma: min pairwise σ from the runtime LJ scoring model
      - r_soft: canonical theorem-valid maximal softening radius = σ
        (derived as the largest theorem-valid softened radius)

    The raw LJ Lipschitz constant (LipschitzStepBounds.lean::typicalLipschitzConstant):
        L_raw = 762 × ε / σ

    The softened Lipschitz constant (SoftLJApproximation.lean::softenedLipschitzConstant):
        L_soft = 24ε/rSoft × |2(σ/rSoft)¹² - (σ/rSoft)⁶|

    Certified by LipschitzStepBounds / PerformanceCertificates:
      - LS5: lower Lipschitz constant yields a weakly larger optimal step
      - LS6: optimal steps scale exactly by L_raw / L_soft
      - softened_grid_speedup_ratio: δ/L_raw ≤ δ/L_soft

    Conditions (SoftLJApproximation.lean):
      - rSoft ≥ 0.8σ (softenedLipschitz_le_rawLipschitz)
      - rSoft ≤ σ   (softenedLJ_lipschitzWith — gradient bound requires repulsive wall)
    Returns base_step scaled by the exact inverse-Lipschitz ratio L_raw / L_soft
    when the softened constant is strictly tighter.
    """
    if (
        min_pairwise_sigma <= 0
        or r_soft <= 0
        or epsilon_lj <= 0
        or r_soft < 0.8 * min_pairwise_sigma
        or r_soft > min_pairwise_sigma
    ):
        return base_step
    ratio = min_pairwise_sigma / r_soft
    l_soft = abs(24.0 * epsilon_lj / r_soft * (2.0 * ratio**12 - ratio**6))
    l_raw = 762.0 * epsilon_lj / min_pairwise_sigma
    if l_soft <= 0 or l_soft >= l_raw:
        return base_step
    scale = l_raw / l_soft
    return base_step * scale


def create_roundwise_certified_action_family(
    base_translation_step: float,
    base_rotation_step_rad: float,
    round_index: int,
    support_expansion_level: int = 0,
) -> CertifiedActionFamily:
    """Create action family for round-wise refinement.

    Step sizes halve each round: step_r = base_step / 2^round_index
    This implements geometric refinement from coarse to fine.

    Lipschitz bounds (Lean: LipschitzStepBounds.lean):
      - LS1: lipschitz_step_bound guarantees |ΔU| ≤ ε for step ≤ ε/L
      - LS2/LS3: the caller can derive a translation step from ε/L exactly
      - LS5/LS6: softened scoring scales that step by the inverse Lipschitz ratio

    The base translation and rotation steps are therefore caller-supplied derived
    quantities, not fixed local-family heuristics.
    """
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if support_expansion_level < 0:
        raise ValueError("support_expansion_level must be non-negative")

    coarser_shell_count = min(round_index, support_expansion_level)
    shell_levels = tuple(range(round_index, round_index - coarser_shell_count - 1, -1))
    families = tuple(
        create_certified_action_family(
            translation_step=base_translation_step / (2**shell_level),
            rotation_step_rad=base_rotation_step_rad / (2**shell_level),
            stencil_level=shell_level,
        )
        for shell_level in shell_levels
    )
    if len(families) == 1:
        return families[0]
    return merge_certified_action_families(families)


def least_adequate_dyadic_round(base_step: float, target_step: float) -> int:
    """Canonical least dyadic round whose translation step is <= target.

    Lean: SupportExpansion.lean
      - SH8 leastAdequateDyadicRound_spec
      - SH9 leastAdequateDyadicRound_minimal

    This deliberately uses the same semantics as the Lean definition: keep
    halving until the dyadic translation step is small enough, and return the
    first round that satisfies the target.
    """
    if base_step < 0.0:
        raise ValueError(f"base_step must be nonnegative, got {base_step}")
    if target_step <= 0.0:
        raise ValueError(f"target_step must be positive, got {target_step}")

    round_index = 0
    current_step = float(base_step)
    while current_step > target_step:
        current_step /= 2.0
        round_index += 1
    return round_index


def least_positive_adequate_dyadic_round(base_step: float, target_step: float) -> int:
    """Canonical least positive dyadic round whose step is <= target.

    Lean: SupportExpansion.lean
      - SH10 leastPositiveAdequateDyadicRound_spec
      - SH11 leastPositiveAdequateDyadicRound_minimal
    """
    return max(1, least_adequate_dyadic_round(base_step, target_step))


def least_positive_joint_adequate_dyadic_round(
    base_step_1: float,
    base_step_2: float,
    target_step: float,
) -> int:
    """Canonical least positive round satisfying two dyadic step constraints.

    Lean: SupportExpansion.lean
      - SH12 leastPositiveJointAdequateDyadicRound_spec
      - SH13 leastPositiveJointAdequateDyadicRound_minimal
    """
    return max(
        least_positive_adequate_dyadic_round(base_step_1, target_step),
        least_positive_adequate_dyadic_round(base_step_2, target_step),
    )


def apply_local_action(
    coords: jax.Array,
    translation_delta: jax.Array,
    quaternion_delta: jax.Array,
) -> jax.Array:
    center = jnp.mean(coords, axis=0)
    centered = coords - center
    moved = rigid_transform_3d(centered, quaternion_delta, translation_delta)
    return moved + center


@jax.jit
def _apply_action_family_arrays(
    coords: jax.Array,
    translation_deltas: jax.Array,
    quaternion_deltas: jax.Array,
) -> jax.Array:
    center = jnp.mean(coords, axis=0)
    centered = coords - center

    def apply_one(
        translation_delta: jax.Array, quaternion_delta: jax.Array
    ) -> jax.Array:
        moved = rigid_transform_3d(centered, quaternion_delta, translation_delta)
        return moved + center

    return jax.vmap(apply_one)(translation_deltas, quaternion_deltas)


@jax.jit
def _apply_action_family_batch_arrays(
    coords_batch: jax.Array,
    translation_deltas: jax.Array,
    quaternion_deltas: jax.Array,
) -> jax.Array:
    return jax.vmap(_apply_action_family_arrays, in_axes=(0, None, None))(
        coords_batch, translation_deltas, quaternion_deltas
    )


def apply_action_family(
    coords: jax.Array,
    action_family: CertifiedActionFamily,
) -> jax.Array:
    return _apply_action_family_arrays(
        coords, action_family.translation_deltas, action_family.quaternion_deltas
    )


def apply_action_family_batch(
    coords_batch: jax.Array,
    action_family: CertifiedActionFamily,
) -> jax.Array:
    return _apply_action_family_batch_arrays(
        coords_batch,
        action_family.translation_deltas,
        action_family.quaternion_deltas,
    )
