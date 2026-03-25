from __future__ import annotations

import jax.numpy as jnp
import pytest

from dq_dock_engine.docking.core import CertifiedBindingSite, DockingBox
from dq_dock_engine.docking.formal_sampling import (
    create_certified_binding_site_action_family,
    create_certified_global_action_family,
    derive_certified_rigid_seed_family_plan,
    materialize_certified_rigid_seed_family,
)


def test_global_action_family_uses_spatially_distributed_translations() -> None:
    family = create_certified_global_action_family(
        DockingBox(
            center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            size=jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32),
        ),
        16,
    )

    unique_translations = jnp.unique(family.translations, axis=0)
    assert unique_translations.shape[0] == 2
    assert float(jnp.ptp(unique_translations[:, 0])) > 0.0


def test_binding_site_action_family_stays_inside_certified_sphere() -> None:
    binding_site = CertifiedBindingSite(
        center=jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32),
        radius=2.5,
        theorem_handles=("SD10",),
    )
    family = create_certified_binding_site_action_family(binding_site, 64)
    unique_translations = jnp.unique(family.translations, axis=0)
    distances = jnp.linalg.norm(
        unique_translations - binding_site.center[None, :], axis=-1
    )

    assert unique_translations.shape[0] == 8
    assert bool(jnp.all(distances <= binding_site.radius + 1e-6))


def test_rigid_seed_family_plan_round_trips_box_family() -> None:
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32),
    )

    plan = derive_certified_rigid_seed_family_plan(box, 32)
    family = materialize_certified_rigid_seed_family(plan)

    assert plan.pose_count == 32
    assert plan.translation_search_volume == 48.0
    assert family.translations.shape[0] == 32
    assert family.quaternions.shape[0] == 32


def test_rigid_seed_family_plan_uses_binding_site_volume() -> None:
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
    )
    binding_site = CertifiedBindingSite(
        center=jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32),
        radius=2.0,
        theorem_handles=("SD10",),
    )

    plan = derive_certified_rigid_seed_family_plan(
        box,
        64,
        certified_binding_site=binding_site,
    )
    family = materialize_certified_rigid_seed_family(plan)

    assert plan.binding_site_radius == 2.0
    assert plan.translation_search_volume == pytest.approx(
        (4.0 / 3.0) * 3.141592653589793 * 8.0
    )
    distances = jnp.linalg.norm(
        family.translations - binding_site.center[None, :], axis=-1
    )
    assert bool(jnp.all(distances <= binding_site.radius + 1e-6))
