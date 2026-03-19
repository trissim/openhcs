from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import (
    axisAngleQuaternion,
    localRotationStencil3D,
    localTranslationStencil3D,
)
from dq_dock_engine.physics.kernels import rigid_transform_3d


@dataclass(frozen=True)
class CertifiedLocalAction:
    action_id: int
    translation_delta: jax.Array
    quaternion_delta: jax.Array
    is_noop: bool = False


@dataclass(frozen=True)
class CertifiedActionFamily:
    actions: tuple[CertifiedLocalAction, ...]
    translation_step: float
    rotation_step_rad: float
    stencil_level: int


def _normalize_quaternion(quaternion: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(quaternion)
    return quaternion / norm


def _axis_angle_to_quaternion(axis: jax.Array, angle_rad: float) -> jax.Array:
    quat = jnp.asarray(axisAngleQuaternion(axis, angle_rad))
    return _normalize_quaternion(quat)


def create_certified_action_family(
    translation_step: float,
    rotation_step_rad: float,
    stencil_level: int,
) -> CertifiedActionFamily:
    axes = (
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
    )
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

    return CertifiedActionFamily(
        actions=tuple(actions),
        translation_step=translation_step,
        rotation_step_rad=rotation_step_rad,
        stencil_level=stencil_level,
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


def apply_action_family(
    coords: jax.Array,
    action_family: CertifiedActionFamily,
) -> jax.Array:
    translation_deltas = jnp.stack(
        [action.translation_delta for action in action_family.actions], axis=0
    )
    quaternion_deltas = jnp.stack(
        [action.quaternion_delta for action in action_family.actions], axis=0
    )

    def apply_one(
        translation_delta: jax.Array, quaternion_delta: jax.Array
    ) -> jax.Array:
        return apply_local_action(coords, translation_delta, quaternion_delta)

    return jax.vmap(apply_one)(translation_deltas, quaternion_deltas)
