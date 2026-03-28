from __future__ import annotations

import jax.numpy as jnp
import pytest
from typing import cast

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
    assert plan.adequate_pose_count == 32
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


def test_rigid_seed_family_plan_uses_cover_floor_without_overwriting_adequacy() -> None:
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
    )

    plan = derive_certified_rigid_seed_family_plan(
        box,
        8,
        target_translation_cover_radius=1.0,
    )
    family = materialize_certified_rigid_seed_family(plan)

    assert plan.adequate_pose_count == 8
    assert plan.pose_count > plan.adequate_pose_count
    assert plan.translation_full_lattice is False
    assert plan.target_translation_cover_radius == pytest.approx(1.0)
    assert family.translations.shape[0] == plan.pose_count


def test_rigid_seed_family_plan_debug_summary_exposes_quaternion_bridge_handles() -> (
    None
):
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([6.0, 6.0, 6.0], dtype=jnp.float32),
    )

    plan = derive_certified_rigid_seed_family_plan(
        box,
        16,
        target_translation_cover_radius=1.0,
    )
    summary = plan.debug_summary()
    rotation_dictionary_handles = cast(
        tuple[str, ...], summary["rotation_dictionary_theorem_handles"]
    )
    rotation_winner_handles = cast(
        tuple[str, ...], summary["rotation_winner_bridge_theorem_handles"]
    )

    assert summary["csc63_csc97_ready"] is True
    assert summary["csc63_csc77_ready"] is True
    assert summary["quaternion_signed_distance_witness_radius"] == pytest.approx(1.0)
    assert {
        "CSC65",
        "CSC70",
        "CSC72",
        "CSC74",
        "CSC76",
        "CSC78",
        "CSC79",
        "CSC80",
        "CSC81",
        "CSC82",
        "CSC84",
        "CSC85",
        "CSC86",
        "CSC87",
        "CSC88",
        "CSC89",
        "CSC90",
        "CSC91",
        "CSC92",
        "CSC93",
        "CSC94",
        "CSC95",
    }.issubset(set(rotation_dictionary_handles))
    assert {
        "CSC67",
        "CSC69",
        "CSC71",
        "CSC73",
        "CSC75",
        "CSC77",
        "CSC83",
        "CSC96",
    }.issubset(set(rotation_winner_handles))
    assert "CSC97" in cast(
        tuple[str, ...], summary["rotation_support_bridge_theorem_handles"]
    )


def test_binding_site_rigid_seed_plan_debug_summary_reports_bridge_obstruction() -> (
    None
):
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
    summary = plan.debug_summary()
    bridge_obstructions = cast(tuple[str, ...], summary["csc63_csc97_obstructions"])

    assert summary["csc63_csc97_ready"] is False
    assert summary["csc63_csc77_ready"] is False
    assert len(bridge_obstructions) > 0
