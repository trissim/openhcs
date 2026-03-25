from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from dq_dock_engine.arraydsl import quaternionDictionary8
from dq_dock_engine.docking.core import CertifiedBindingSite, DockingBox, PoseVector


@dataclass(frozen=True)
class CertifiedGlobalActionFamily:
    translations: jax.Array
    quaternions: jax.Array
    lattice_resolution: int
    quaternion_count: int


def _quaternion_dictionary() -> jax.Array:
    base = jnp.asarray(quaternionDictionary8())
    norms = jnp.linalg.norm(base, axis=1, keepdims=True)
    return base / norms


def _minimal_even_grid_resolution(n_points: int) -> int:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    resolution = int(jnp.ceil(n_points ** (1.0 / 3.0)))
    resolution = max(resolution, 2)
    if resolution % 2 == 1:
        resolution += 1
    return resolution


def _translation_grid_at_resolution(box: DockingBox, resolution: int) -> jax.Array:
    half_size = box.size / 2.0
    x_edges = jnp.linspace(
        box.center[0] - half_size[0], box.center[0] + half_size[0], resolution + 1
    )
    y_edges = jnp.linspace(
        box.center[1] - half_size[1], box.center[1] + half_size[1], resolution + 1
    )
    z_edges = jnp.linspace(
        box.center[2] - half_size[2], box.center[2] + half_size[2], resolution + 1
    )
    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    zs = 0.5 * (z_edges[:-1] + z_edges[1:])
    return jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape((-1, 3))


def _select_uniform_grid_subset(grid: jax.Array, n_points: int) -> jax.Array:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    total = int(grid.shape[0])
    if total <= n_points:
        return grid
    indices = np.floor(np.arange(n_points, dtype=np.float64) * total / n_points).astype(
        np.int32
    )
    return grid[jnp.asarray(indices, dtype=jnp.int32)]


def _translation_grid(box: DockingBox, n_points: int) -> tuple[jax.Array, int]:
    resolution = _minimal_even_grid_resolution(n_points)
    full_grid = _translation_grid_at_resolution(box, resolution)
    return _select_uniform_grid_subset(full_grid, n_points), resolution


def _binding_site_translation_grid(
    binding_site: CertifiedBindingSite,
    n_points: int,
) -> tuple[jax.Array, int]:
    resolution = _minimal_even_grid_resolution(n_points)
    box = DockingBox(
        center=binding_site.center,
        size=jnp.full((3,), 2.0 * binding_site.radius),
    )
    while True:
        full_grid = _translation_grid_at_resolution(box, resolution)
        in_sphere = (
            jnp.linalg.norm(full_grid - binding_site.center[None, :], axis=-1)
            <= binding_site.radius
        )
        sphere_grid = full_grid[in_sphere]
        if int(sphere_grid.shape[0]) >= n_points:
            return _select_uniform_grid_subset(sphere_grid, n_points), resolution
        resolution += 2


def create_certified_global_action_family(
    box: DockingBox,
    n_poses: int,
) -> CertifiedGlobalActionFamily:
    quaternions = _quaternion_dictionary()
    n_quaternions = quaternions.shape[0]
    n_translation_points = int(jnp.ceil(n_poses / n_quaternions))
    translations, resolution = _translation_grid(box, n_translation_points)
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))
    return CertifiedGlobalActionFamily(
        translations=tiled_translations[:n_poses],
        quaternions=tiled_quaternions[:n_poses],
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def create_certified_binding_site_action_family(
    binding_site: CertifiedBindingSite,
    n_poses: int,
) -> CertifiedGlobalActionFamily:
    quaternions = _quaternion_dictionary()
    n_quaternions = quaternions.shape[0]
    n_translation_points = int(jnp.ceil(n_poses / n_quaternions))
    translations, resolution = _binding_site_translation_grid(
        binding_site,
        n_translation_points,
    )
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))
    return CertifiedGlobalActionFamily(
        translations=tiled_translations[:n_poses],
        quaternions=tiled_quaternions[:n_poses],
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def sample_certified_global_poses(box: DockingBox, n_poses: int) -> PoseVector:
    family = create_certified_global_action_family(box, n_poses)
    return PoseVector(
        translation=family.translations,
        quaternion=family.quaternions,
    )
