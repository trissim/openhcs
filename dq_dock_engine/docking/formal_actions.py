"""Certified action families for local optimization.

Step size derivation (Lean: LipschitzStepBounds.lean):
  - LS1: lipschitz_step_bound proves Δ ≤ ε/L guarantees |ΔU| ≤ ε
  - LS2: For LJ potential, L ≈ 22 kcal/(mol·Å) at typical distances
  - LS3: Translation step 0.5 Å is ~20× theoretical minimum but practical
  - LS4: Rotation step π/12 rad (~15°) balances sampling and convergence
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
      - LS3: base_translation_step=0.5Å is empirically tuned (LS3)
      - LS4: base_rotation_step=π/12 balances exploration/exploitation (LS4)
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
