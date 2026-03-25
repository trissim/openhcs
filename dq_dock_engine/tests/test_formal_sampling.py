from __future__ import annotations

import jax.numpy as jnp

from dq_dock_engine.docking.core import CertifiedBindingSite, DockingBox
from dq_dock_engine.docking.formal_sampling import (
    create_certified_binding_site_action_family,
    create_certified_global_action_family,
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
