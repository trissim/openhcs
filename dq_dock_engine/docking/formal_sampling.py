from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
from typing import cast
from dq_dock_engine.arraydsl import quaternionDictionary8
from dq_dock_engine.docking.certified_runtime_plans import (
    CertifiedRigidSeedFamilyPlan,
    CertifiedRigidSeedRegionKind,
)
from dq_dock_engine.docking.core import CertifiedBindingSite, DockingBox, PoseVector
from dq_dock_engine.docking.formal_handles import rigid_seed_family_theorem_handles


@dataclass(frozen=True)
class CertifiedGlobalActionFamily:
    translations: jax.Array
    quaternions: jax.Array
    lattice_resolution: int
    quaternion_count: int


def _cover_stats_against_selected_subset(
    full_points: np.ndarray,
    selected_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    if full_points.size == 0 or selected_points.size == 0:
        raise ValueError("cover stats require non-empty full and selected point sets")
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(selected_points).query(full_points, k=1)
        nearest = selected_points[np.asarray(indices, dtype=np.int64)]
    except Exception:
        deltas = full_points[:, None, :] - selected_points[None, :, :]
        sq = np.sum(deltas * deltas, axis=2)
        indices = np.argmin(sq, axis=1)
        nearest = selected_points[indices]
        distances = np.sqrt(np.min(sq, axis=1))
    axis_half_widths = np.max(np.abs(full_points - nearest), axis=0)
    return axis_half_widths, float(np.max(distances))


def _box_search_volume(box: DockingBox) -> float:
    return float(np.prod(np.asarray(box.size, dtype=np.float64)))


def _binding_site_search_volume(binding_site: CertifiedBindingSite) -> float:
    radius = float(binding_site.radius)
    return (4.0 / 3.0) * math.pi * radius**3


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


def _required_full_lattice_resolution(
    box_size: jax.Array,
    target_translation_cover_radius: float,
) -> int:
    if target_translation_cover_radius <= 0.0:
        raise ValueError(
            "target_translation_cover_radius must be positive for theorem-enforced lattices"
        )
    size = np.asarray(box_size, dtype=np.float64)
    resolution = int(
        math.ceil(np.linalg.norm(size) / (2.0 * float(target_translation_cover_radius)))
    )
    resolution = max(resolution, 2)
    if resolution % 2 == 1:
        resolution += 1
    return resolution


def _select_uniform_grid_subset(grid: jax.Array, n_points: int) -> jax.Array:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    total = int(grid.shape[0])
    if total <= n_points:
        return grid
    grid_np = np.asarray(grid, dtype=np.float64)
    center = np.mean(grid_np, axis=0)
    selected: list[int] = [
        int(np.argmin(np.sum((grid_np - center[None, :]) ** 2, axis=1)))
    ]
    min_sq_dist = np.sum((grid_np - grid_np[selected[0]][None, :]) ** 2, axis=1)
    min_sq_dist[selected[0]] = -1.0
    while len(selected) < n_points:
        next_index = int(np.argmax(min_sq_dist))
        selected.append(next_index)
        new_sq_dist = np.sum((grid_np - grid_np[next_index][None, :]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)
        min_sq_dist[selected] = -1.0
    return grid[jnp.asarray(np.asarray(selected, dtype=np.int32), dtype=jnp.int32)]


def _translation_grid(box: DockingBox, n_points: int) -> tuple[jax.Array, int]:
    resolution = _minimal_even_grid_resolution(n_points)
    full_grid = _translation_grid_at_resolution(box, resolution)
    return _select_uniform_grid_subset(full_grid, n_points), resolution


def _translation_grid_cover_radius(box: DockingBox, n_points: int) -> float:
    resolution = _minimal_even_grid_resolution(n_points)
    full_grid = np.asarray(
        _translation_grid_at_resolution(box, resolution), dtype=np.float64
    )
    selected = np.asarray(
        _select_uniform_grid_subset(jnp.asarray(full_grid), n_points), dtype=np.float64
    )
    step = np.asarray(box.size, dtype=np.float64) / float(resolution)
    subset_axis_half_widths, _ = _cover_stats_against_selected_subset(
        full_grid, selected
    )
    return float(np.linalg.norm(step / 2.0 + subset_axis_half_widths))


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


def _binding_site_translation_grid_cover_radius(
    binding_site: CertifiedBindingSite,
    n_points: int,
) -> float:
    resolution = _minimal_even_grid_resolution(n_points)
    box = DockingBox(
        center=binding_site.center,
        size=jnp.full((3,), 2.0 * binding_site.radius),
    )
    while True:
        full_grid = np.asarray(
            _translation_grid_at_resolution(box, resolution), dtype=np.float64
        )
        in_sphere = (
            np.linalg.norm(
                full_grid - np.asarray(binding_site.center, dtype=np.float64)[None, :],
                axis=1,
            )
            <= float(binding_site.radius) + 1e-9
        )
        sphere_grid = full_grid[in_sphere]
        if int(sphere_grid.shape[0]) >= n_points:
            selected = np.asarray(
                _select_uniform_grid_subset(jnp.asarray(sphere_grid), n_points),
                dtype=np.float64,
            )
            step = np.full((3,), (2.0 * float(binding_site.radius)) / float(resolution))
            subset_axis_half_widths, _ = _cover_stats_against_selected_subset(
                full_grid, selected
            )
            return float(np.linalg.norm(step / 2.0 + subset_axis_half_widths))
        resolution += 2


def _minimal_translation_points_for_cover(
    *,
    base_points: int,
    target_translation_cover_radius: float | None,
    box: DockingBox | None = None,
    binding_site: CertifiedBindingSite | None = None,
) -> int:
    if (
        target_translation_cover_radius is None
        or target_translation_cover_radius <= 0.0
    ):
        return base_points

    def cover_radius(n_points: int) -> float:
        if binding_site is not None:
            return _binding_site_translation_grid_cover_radius(binding_site, n_points)
        assert box is not None
        return _translation_grid_cover_radius(box, n_points)

    low = max(1, base_points)
    if cover_radius(low) <= target_translation_cover_radius:
        return low
    high = low * 2
    while cover_radius(high) > target_translation_cover_radius:
        low = high
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if cover_radius(mid) <= target_translation_cover_radius:
            high = mid
        else:
            low = mid
    return high


def create_certified_global_action_family(
    box: DockingBox,
    n_poses: int,
    *,
    translation_full_lattice: bool = False,
    target_translation_cover_radius: float | None = None,
) -> CertifiedGlobalActionFamily:
    quaternions = _quaternion_dictionary()
    n_quaternions = quaternions.shape[0]
    if translation_full_lattice:
        if target_translation_cover_radius is None:
            raise ValueError(
                "translation_full_lattice requires target_translation_cover_radius"
            )
        resolution = _required_full_lattice_resolution(
            box.size, target_translation_cover_radius
        )
        translations = _translation_grid_at_resolution(box, resolution)
    else:
        n_translation_points = int(jnp.ceil(n_poses / n_quaternions))
        translations, resolution = _translation_grid(box, n_translation_points)
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))
    return CertifiedGlobalActionFamily(
        translations=(
            tiled_translations
            if translation_full_lattice
            else tiled_translations[:n_poses]
        ),
        quaternions=(
            tiled_quaternions
            if translation_full_lattice
            else tiled_quaternions[:n_poses]
        ),
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def create_certified_binding_site_action_family(
    binding_site: CertifiedBindingSite,
    n_poses: int,
    *,
    translation_full_lattice: bool = False,
    target_translation_cover_radius: float | None = None,
) -> CertifiedGlobalActionFamily:
    quaternions = _quaternion_dictionary()
    n_quaternions = quaternions.shape[0]
    if translation_full_lattice:
        if target_translation_cover_radius is None:
            raise ValueError(
                "translation_full_lattice requires target_translation_cover_radius"
            )
        box = DockingBox(
            center=binding_site.center,
            size=jnp.full((3,), 2.0 * binding_site.radius),
        )
        resolution = _required_full_lattice_resolution(
            box.size, target_translation_cover_radius
        )
        translations = _translation_grid_at_resolution(box, resolution)
    else:
        n_translation_points = int(jnp.ceil(n_poses / n_quaternions))
        translations, resolution = _binding_site_translation_grid(
            binding_site,
            n_translation_points,
        )
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))
    return CertifiedGlobalActionFamily(
        translations=(
            tiled_translations
            if translation_full_lattice
            else tiled_translations[:n_poses]
        ),
        quaternions=(
            tiled_quaternions
            if translation_full_lattice
            else tiled_quaternions[:n_poses]
        ),
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def derive_certified_rigid_seed_family_plan(
    box: DockingBox,
    n_poses: int,
    certified_binding_site: CertifiedBindingSite | None = None,
    *,
    target_translation_cover_radius: float | None = None,
) -> CertifiedRigidSeedFamilyPlan:
    enforce_cover = (
        target_translation_cover_radius is not None
        and target_translation_cover_radius > 0.0
    )
    if certified_binding_site is None:
        family = create_certified_global_action_family(
            box,
            n_poses,
            translation_full_lattice=enforce_cover,
            target_translation_cover_radius=target_translation_cover_radius,
        )
        n_translation_points = int(
            family.translations.shape[0] // family.quaternion_count
        )
        pose_count = int(family.translations.shape[0])
        return CertifiedRigidSeedFamilyPlan(
            region_kind=CertifiedRigidSeedRegionKind.BOX,
            pose_count=pose_count,
            translation_point_count=n_translation_points,
            lattice_resolution=family.lattice_resolution,
            quaternion_count=family.quaternion_count,
            translation_search_volume=_box_search_volume(box),
            theorem_handles=rigid_seed_family_theorem_handles(),
            box_center=cast(
                tuple[float, float, float],
                tuple(float(v) for v in np.asarray(box.center).tolist()),
            ),
            box_size=cast(
                tuple[float, float, float],
                tuple(float(v) for v in np.asarray(box.size).tolist()),
            ),
            translation_full_lattice=enforce_cover,
            target_translation_cover_radius=target_translation_cover_radius,
            note=(
                "Certified rigid seed family over the docking box lattice"
                if pose_count == n_poses
                else "Certified rigid seed family over the docking box lattice "
                f"(translation-tightened from {n_poses} to {pose_count} poses)"
            ),
        )
    family = create_certified_binding_site_action_family(
        certified_binding_site,
        n_poses,
        translation_full_lattice=enforce_cover,
        target_translation_cover_radius=target_translation_cover_radius,
    )
    n_translation_points = int(family.translations.shape[0] // family.quaternion_count)
    pose_count = int(family.translations.shape[0])
    return CertifiedRigidSeedFamilyPlan(
        region_kind=CertifiedRigidSeedRegionKind.CERTIFIED_BINDING_SITE,
        pose_count=pose_count,
        translation_point_count=n_translation_points,
        lattice_resolution=family.lattice_resolution,
        quaternion_count=family.quaternion_count,
        translation_search_volume=_binding_site_search_volume(certified_binding_site),
        theorem_handles=tuple(
            dict.fromkeys(
                rigid_seed_family_theorem_handles()
                + tuple(certified_binding_site.theorem_handles)
            )
        ),
        binding_site_center=cast(
            tuple[float, float, float],
            tuple(float(v) for v in np.asarray(certified_binding_site.center).tolist()),
        ),
        binding_site_radius=float(certified_binding_site.radius),
        translation_full_lattice=enforce_cover,
        target_translation_cover_radius=target_translation_cover_radius,
        note=(
            "Certified rigid seed family over the certified binding-site sphere"
            if pose_count == n_poses
            else "Certified rigid seed family over the certified binding-site sphere "
            f"(translation-tightened from {n_poses} to {pose_count} poses)"
        ),
    )


def materialize_certified_rigid_seed_family(
    plan: CertifiedRigidSeedFamilyPlan,
) -> CertifiedGlobalActionFamily:
    if plan.region_kind == CertifiedRigidSeedRegionKind.BOX:
        assert plan.box_center is not None
        assert plan.box_size is not None
        return create_certified_global_action_family(
            DockingBox(
                center=jnp.asarray(
                    cast(tuple[float, float, float], plan.box_center), dtype=jnp.float32
                ),
                size=jnp.asarray(
                    cast(tuple[float, float, float], plan.box_size), dtype=jnp.float32
                ),
            ),
            plan.pose_count,
            translation_full_lattice=plan.translation_full_lattice,
            target_translation_cover_radius=plan.target_translation_cover_radius,
        )
    assert plan.binding_site_center is not None
    assert plan.binding_site_radius is not None
    return create_certified_binding_site_action_family(
        CertifiedBindingSite(
            center=jnp.asarray(
                cast(tuple[float, float, float], plan.binding_site_center),
                dtype=jnp.float32,
            ),
            radius=float(plan.binding_site_radius),
            theorem_handles=plan.theorem_handles,
        ),
        plan.pose_count,
        translation_full_lattice=plan.translation_full_lattice,
        target_translation_cover_radius=plan.target_translation_cover_radius,
    )


def sample_certified_global_poses(box: DockingBox, n_poses: int) -> PoseVector:
    family = create_certified_global_action_family(box, n_poses)
    return PoseVector(
        translation=family.translations,
        quaternion=family.quaternions,
    )
