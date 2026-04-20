"""Semantic fidelity checks for generated ArrayDSL runtime operations."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from dq_dock_engine.codegen.arraydsl_runtime import PrimitiveMetadata


def resolve_backend_symbol(metadata: PrimitiveMetadata) -> Any:
    """Resolve and return the declared backend symbol for one primitive."""

    module = importlib.import_module(metadata.jax_module)
    if not hasattr(module, metadata.jax_symbol):
        raise ValueError(
            f"Missing backend symbol for {metadata.name}: "
            f"{metadata.jax_module}.{metadata.jax_symbol}"
        )
    return getattr(module, metadata.jax_symbol)


def _fixture_and_reference(lowering_kind: str) -> tuple[tuple[Any, ...], Any]:
    vec = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    vec2 = jnp.array([3.5, -1.0, 0.5], dtype=jnp.float32)
    mat = jnp.array([[1.0, 2.0, 2.0], [3.0, 0.0, 4.0]], dtype=jnp.float32)
    mat2 = jnp.array([[0.0, 2.0, 2.0], [0.0, 0.0, 4.0]], dtype=jnp.float32)
    coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    coords2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=jnp.float32)
    quaternion = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    translation = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float32)
    box_size = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    pairwise = jnp.array([[1.1, 2.4], [1.7, 2.2]], dtype=jnp.float32)
    epsilons = jnp.array([[0.5, 0.25], [0.125, 0.75]], dtype=jnp.float32)
    sigmas = jnp.array([[1.2, 1.1], [1.0, 1.3]], dtype=jnp.float32)
    charges1 = jnp.array([1.0, -1.0], dtype=jnp.float32)
    charges2 = jnp.array([-0.5, 0.5], dtype=jnp.float32)
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    mask = jnp.array([[False, True], [True, False]])
    rc = jnp.array(2.0, dtype=jnp.float32)
    epsilon = jnp.array(0.5, dtype=jnp.float32)
    sigma = jnp.array(1.2, dtype=jnp.float32)
    alpha = jnp.array(0.5, dtype=jnp.float32)
    dielectric = jnp.array(1.0, dtype=jnp.float32)

    if lowering_kind == "vmap":
        f = lambda x: x * x + 1.0
        arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        return (f, arr), jax.vmap(f)(arr)
    if lowering_kind == "reduce_sum":
        return (vec,), jnp.sum(vec)
    if lowering_kind == "elem_binary_add":
        return (vec, vec2), vec + vec2
    if lowering_kind == "elem_binary_sub":
        return (vec, vec2), vec - vec2
    if lowering_kind == "norm":
        return (vec,), jnp.linalg.norm(vec)
    if lowering_kind == "row_wise_norm":
        return (mat,), jnp.linalg.norm(mat, axis=-1)
    if lowering_kind == "distance":
        return (vec, vec2), jnp.linalg.norm(vec - vec2)
    if lowering_kind == "row_wise_distance":
        return (mat, mat2), jnp.linalg.norm(mat - mat2, axis=-1)
    if lowering_kind == "rigid_transform_3d":
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        rot = jnp.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ],
            dtype=jnp.float32,
        )
        return (coords1, quaternion, translation), (coords1 @ rot.T) + translation
    if lowering_kind == "pairwise_distances":
        arr1 = jnp.array([0.0, 2.0], dtype=jnp.float32)
        arr2 = jnp.array([-1.0, 3.0], dtype=jnp.float32)
        return (arr1, arr2), jnp.abs(arr1[:, None] - arr2[None, :])
    if lowering_kind == "pairwise_distances_3d":
        return (coords1, coords2), jnp.linalg.norm(
            coords1[:, None, :] - coords2[None, :, :], axis=-1
        )
    if lowering_kind == "minimum_image_pairwise_distances":
        diff = coords1[:, None, :] - coords2[None, :, :]
        wrapped = diff - box_size * jnp.round(diff / box_size)
        return (coords1, coords2, box_size), jnp.linalg.norm(wrapped, axis=-1)
    if lowering_kind == "apply_cutoff":
        return (pairwise, rc), jnp.where(pairwise < rc, pairwise, 0.0)
    if lowering_kind == "lennard_jones":
        safe_r = jnp.where(pairwise > 1e-10, pairwise, 1e-10)
        inv_r6 = (sigma / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilon * (inv_r12 - inv_r6)
        return (epsilon, sigma, pairwise), jnp.where(pairwise > 1e-10, potential, 1e12)
    if lowering_kind == "sum_pair_potentials":
        masked = jnp.where(pairwise < rc, pairwise, 0.0)
        safe_r = jnp.where(masked > 1e-10, masked, 1e-10)
        inv_r6 = (sigma / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilon * (inv_r12 - inv_r6)
        lj = jnp.where(masked > 1e-10, potential, 1e12)
        return (pairwise, rc, epsilon, sigma), jnp.sum(lj)
    if lowering_kind == "sum_pair_potentials_matrix":
        masked = jnp.where(pairwise < rc, pairwise, 0.0)
        safe_r = jnp.where(masked > 1e-10, masked, 1e-10)
        inv_r6 = (sigma / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilon * (inv_r12 - inv_r6)
        lj = jnp.where(masked > 1e-10, potential, 1e12)
        return (pairwise, rc, epsilon, sigma), jnp.sum(lj)
    if lowering_kind == "sum_pair_potentials_3d":
        distances = jnp.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)
        masked = jnp.where(distances < rc, distances, 0.0)
        safe_r = jnp.where(masked > 1e-10, masked, 1e-10)
        inv_r6 = (sigma / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilon * (inv_r12 - inv_r6)
        lj = jnp.where(masked > 1e-10, potential, 1e12)
        return (coords1, coords2, rc, epsilon, sigma), jnp.sum(lj)
    if lowering_kind == "typed_lennard_jones_matrix":
        safe_r = jnp.where(pairwise > 1e-10, pairwise, 1e-10)
        inv_r6 = (sigmas / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilons * (inv_r12 - inv_r6)
        return (pairwise, epsilons, sigmas), jnp.where(
            pairwise > 1e-10, potential, 1e12
        )
    if lowering_kind == "typed_lennard_jones_cutoff":
        safe_r = jnp.where(pairwise > 1e-10, pairwise, 1e-10)
        inv_r6 = (sigmas / safe_r) ** 6
        inv_r12 = inv_r6**2
        potential = 4.0 * epsilons * (inv_r12 - inv_r6)
        energies = jnp.where(pairwise > 1e-10, potential, 1e12)
        return (pairwise, epsilons, sigmas, rc), jnp.sum(
            jnp.where(pairwise < rc, energies, 0.0)
        )
    if lowering_kind == "coulomb_cutoff":
        charge_product = charges1[:, None] * charges2[None, :]
        within = (pairwise < rc) & (pairwise > 1e-10)
        safe_r = jnp.where(within, pairwise, 1.0)
        return (charges1, charges2, pairwise, rc, dielectric), jnp.sum(
            jnp.where(within, charge_product / (dielectric * safe_r), 0.0)
        )
    if lowering_kind == "upper_triangle_masked_sum":
        upper = jnp.triu(jnp.ones_like(values, dtype=bool), k=1)
        return (values, mask), jnp.sum(jnp.where(upper & mask, values, 0.0))
    if lowering_kind == "ewald_real_space_kernel":
        safe_r = jnp.where(pairwise > 1e-10, pairwise, 1e-10)
        return (pairwise, alpha), jnp.exp(-((alpha * safe_r) ** 2)) / safe_r

    raise ValueError(f"Unsupported lowering kind: {lowering_kind}")


def _allclose(observed: Any, expected: Any, *, atol: float) -> bool:
    observed_arr = jnp.asarray(observed)
    expected_arr = jnp.asarray(expected)
    return bool(jnp.allclose(observed_arr, expected_arr, rtol=1e-6, atol=atol))


def check_primitive_semantics(
    metadata: PrimitiveMetadata, *, atol: float = 1e-6
) -> None:
    """Check one generated primitive against its declared lowering semantics."""

    _ = resolve_backend_symbol(metadata)
    args, expected = _fixture_and_reference(metadata.lowering_kind)
    observed = metadata.callable(*args)
    if not _allclose(observed, expected, atol=atol):
        raise ValueError(
            f"Semantic fidelity mismatch for primitive '{metadata.name}' "
            f"(lowering={metadata.lowering_kind}, backend={metadata.jax_module}.{metadata.jax_symbol})"
        )


def check_registry_semantics(
    registry: Mapping[str, PrimitiveMetadata], *, atol: float = 1e-6
) -> None:
    """Check semantic fidelity for every generated primitive in the registry."""

    for name in sorted(registry):
        check_primitive_semantics(registry[name], atol=atol)


__all__ = [
    "check_primitive_semantics",
    "check_registry_semantics",
    "resolve_backend_symbol",
]
