"""Generate Python/JAX wrappers from the exported Lean ArrayDSL IR."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast


DEFAULT_PROOFS_DIR = Path("docs/papers/paper4_decision_quotient/proofs")
DEFAULT_EXPORT_PATH = Path(
    "docs/papers/paper4_decision_quotient/proofs/arraydsl_primitives.json"
)
DEFAULT_OUTPUT_DIR = Path("dq_dock_engine/generated")
DEFAULT_OUTPUT_PACKAGE = "dq_dock_engine.generated"


@dataclass(frozen=True)
class ArgIR:
    name: str
    kind: str
    scalar_type: Optional[str]


@dataclass(frozen=True)
class PrimitiveIR:
    name: str
    args: List[ArgIR]
    result_kind: str
    scalar_type: Optional[str]
    lowering_kind: str
    jax_module: str
    jax_symbol: str
    supports_grad: bool
    lean_symbol: str
    proof_ref: Optional[str]
    proof_status: Optional[str]


LOWERING_BODIES: Dict[str, str] = {
    "vmap": "return jax.vmap(f)(arr)",
    "reduce_sum": "return jnp.sum(arr)",
    "elem_binary_add": "return arr1 + arr2",
    "elem_binary_sub": "return arr1 - arr2",
    "norm": "return jnp.linalg.norm(arr)",
    "row_wise_norm": "return jnp.linalg.norm(arr, axis=-1)",
    "distance": "return jnp.linalg.norm(arr1 - arr2)",
    "row_wise_distance": "return jnp.linalg.norm(arr1 - arr2, axis=-1)",
    "rigid_transform_3d": "w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]\nR = jnp.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w], [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]])\nreturn (coords @ R.T) + translation",
    "pairwise_distances": "return jnp.abs(coords1[:, None] - coords2[None, :])",
    "pairwise_distances_3d": "return jnp.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)",
    "minimum_image_pairwise_distances": "diff = coords1[:, None, :] - coords2[None, :, :]\nwrapped = diff - box_size * jnp.round(diff / box_size)\nreturn jnp.linalg.norm(wrapped, axis=-1)",
    "apply_cutoff": "return jnp.where(distances < rc, distances, 0.0)",
    "lennard_jones": "safe_r = jnp.where(r > 1e-10, r, 1e-10)\ninv_r6 = (sigma / safe_r) ** 6\ninv_r12 = inv_r6 ** 2\npotential = 4.0 * epsilon * (inv_r12 - inv_r6)\nreturn jnp.where(r > 1e-10, potential, 1e12)",
    "sum_pair_potentials": "masked = applyCutoff(distances, rc)\nreturn jnp.sum(lennardJones(epsilon, sigma, masked))",
    "sum_pair_potentials_matrix": "masked = jnp.where(distances < rc, distances, 0.0)\nreturn jnp.sum(lennardJones(epsilon, sigma, masked))",
    "sum_pair_potentials_3d": "distances = pairwiseDistances3D(coords1, coords2)\nreturn sumPairPotentialsMatrix(distances, rc, epsilon, sigma)",
    "typed_lennard_jones_matrix": "safe_r = jnp.where(distances > 1e-10, distances, 1e-10)\ninv_r6 = (sigmas / safe_r) ** 6\ninv_r12 = inv_r6 ** 2\npotential = 4.0 * epsilons * (inv_r12 - inv_r6)\nreturn jnp.where(distances > 1e-10, potential, 1e12)",
    "typed_lennard_jones_cutoff": "energies = typedLennardJonesMatrix(distances, epsilons, sigmas)\nreturn jnp.sum(jnp.where(distances < rc, energies, 0.0))",
    "coulomb_cutoff": "charge_product = charges1[:, None] * charges2[None, :]\nwithin = (distances < rc) & (distances > 1e-10)\nsafe_r = jnp.where(within, distances, 1.0)\nreturn jnp.sum(jnp.where(within, charge_product / (dielectric * safe_r), 0.0))",
    "upper_triangle_masked_sum": "upper = jnp.triu(jnp.ones_like(values, dtype=bool), k=1)\nreturn jnp.sum(jnp.where(upper & mask, values, 0.0))",
    "ewald_real_space_kernel": "safe_r = jnp.where(distances > 1e-10, distances, 1e-10)\nreturn jnp.exp(-((alpha * safe_r) ** 2)) / safe_r",
}


def _load_export(path: Path) -> List[PrimitiveIR]:
    data = json.loads(path.read_text())
    primitives: List[PrimitiveIR] = []
    for entry in data:
        args = [ArgIR(**arg) for arg in entry["args"]]
        primitives.append(
            PrimitiveIR(
                name=entry["name"],
                args=args,
                result_kind=entry["result_kind"],
                scalar_type=entry.get("scalar_type"),
                lowering_kind=entry["lowering_kind"],
                jax_module=entry["jax_module"],
                jax_symbol=entry["jax_symbol"],
                supports_grad=entry["supports_grad"],
                lean_symbol=entry["lean_symbol"],
                proof_ref=entry.get("proof_ref"),
                proof_status=entry.get("proof_status"),
            )
        )
    return primitives


def _render_function(primitive: PrimitiveIR) -> str:
    signature = ", ".join(arg.name for arg in primitive.args)
    body_lines = [
        f"    {line}" for line in LOWERING_BODIES[primitive.lowering_kind].splitlines()
    ]
    body = "\n".join(body_lines)
    return (
        f"def {primitive.name}({signature}):\n"
        f'    """Generated wrapper for {primitive.lean_symbol}."""\n'
        f"{body}\n"
    )


def _proof_status_expr(proof_status: Optional[str]) -> str:
    if proof_status is None:
        return "None"
    return f"ProofStatus.{proof_status}"


def _proof_ref_expr(proof_ref: Optional[str]) -> str:
    if proof_ref is None:
        return "None"
    return repr(proof_ref)


def _render_primitives_module(primitives: List[PrimitiveIR]) -> str:
    functions = "\n\n".join(_render_function(primitive) for primitive in primitives)
    exported_names = ",\n    ".join(repr(primitive.name) for primitive in primitives)
    return f'''"""Generated JAX wrappers for Lean ArrayDSL primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp


{functions}


__all__ = [
    {exported_names}
]
'''


def _render_registry_module(primitives: List[PrimitiveIR], package_name: str) -> str:
    imports = ",\n    ".join(primitive.name for primitive in primitives)
    registrations = []
    for primitive in primitives:
        registrations.append(
            f"""register_primitive(
    PrimitiveMetadata(
        name={primitive.name!r},
        lean_symbol={primitive.lean_symbol!r},
        jax_module={primitive.jax_module!r},
        jax_symbol={primitive.jax_symbol!r},
        lowering_kind={primitive.lowering_kind!r},
        supports_grad={primitive.supports_grad!r},
        proof_ref={_proof_ref_expr(primitive.proof_ref)},
        proof_status={_proof_status_expr(primitive.proof_status)},
        callable={primitive.name},
    )
)"""
        )
    registration_block = "\n\n".join(registrations)
    exported_names = ",\n    ".join(repr(primitive.name) for primitive in primitives)
    return f'''"""Generated registry for Lean ArrayDSL primitives."""

from __future__ import annotations

from dq_dock_engine.codegen.arraydsl_runtime import PrimitiveMetadata, PRIMITIVE_REGISTRY, register_primitive
from {package_name}.arraydsl_primitives import (
    {imports},
)
from dq_dock_engine.proof_status import ProofStatus


{registration_block}


ARRAYDSL_PRIMITIVES = tuple(PRIMITIVE_REGISTRY[name] for name in [
    {exported_names}
])
'''


def _render_init_module(primitives: List[PrimitiveIR]) -> str:
    exported_names = ",\n    ".join(repr(primitive.name) for primitive in primitives)
    return f'''"""Generated Lean-backed JAX wrappers."""


__all__ = [
    {exported_names}
]
'''


def generate_modules(
    export_path: Path,
    output_dir: Path,
    package_name: str = DEFAULT_OUTPUT_PACKAGE,
) -> List[Path]:
    primitives = _load_export(export_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_path = output_dir / "__init__.py"
    primitives_path = output_dir / "arraydsl_primitives.py"
    registry_path = output_dir / "arraydsl_registry.py"

    init_path.write_text(_render_init_module(primitives))
    primitives_path.write_text(_render_primitives_module(primitives))
    registry_path.write_text(_render_registry_module(primitives, package_name))
    return [init_path, primitives_path, registry_path]


def validate_generated_modules(
    export_path: Path,
    output_dir: Path,
    package_name: str = DEFAULT_OUTPUT_PACKAGE,
) -> None:
    from dq_dock_engine.codegen.backend_fidelity import check_registry_semantics
    from dq_dock_engine.codegen.arraydsl_runtime import clear_registry
    from dq_dock_engine.codegen.arraydsl_runtime import PrimitiveMetadata

    primitives = _load_export(export_path)
    generated_names = {primitive.name for primitive in primitives}

    clear_registry()
    namespace: Dict[str, object] = {}
    registry_path = output_dir / "arraydsl_registry.py"
    exec(compile(registry_path.read_text(), str(registry_path), "exec"), namespace)

    registry = cast(Dict[str, PrimitiveMetadata], namespace["PRIMITIVE_REGISTRY"])
    registered_names = set(registry)
    if registered_names != generated_names:
        raise ValueError(
            f"Generated registry mismatch: expected {sorted(generated_names)}, got {sorted(registered_names)}"
        )

    check_registry_semantics(registry)


def export_lean_ir(
    proofs_dir: Path = DEFAULT_PROOFS_DIR,
    export_path: Path = DEFAULT_EXPORT_PATH,
) -> Path:
    subprocess.run(
        ["lake", "build", "arraydsl-export"],
        cwd=proofs_dir,
        check=True,
    )
    subprocess.run(
        [
            str((proofs_dir / ".lake" / "build" / "bin" / "arraydsl-export").resolve()),
            str(export_path.resolve()),
        ],
        cwd=proofs_dir,
        check=True,
    )
    if not export_path.exists():
        raise FileNotFoundError(f"Lean export did not produce {export_path}")
    return export_path


def regenerate_bridge(
    proofs_dir: Path = DEFAULT_PROOFS_DIR,
    export_path: Path = DEFAULT_EXPORT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    package_name: str = DEFAULT_OUTPUT_PACKAGE,
    *,
    run_validation: bool = True,
) -> List[Path]:
    export_lean_ir(proofs_dir=proofs_dir, export_path=export_path)
    generated_paths = generate_modules(
        export_path, output_dir, package_name=package_name
    )
    if run_validation:
        validate_generated_modules(export_path, output_dir, package_name=package_name)
    return generated_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default=DEFAULT_OUTPUT_PACKAGE)
    parser.add_argument("--export-from-lean", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_path = args.input
    if args.export_from_lean:
        export_path = export_lean_ir(export_path=export_path)
    generated_paths = generate_modules(
        export_path, args.output_dir, package_name=args.package_name
    )
    if args.check:
        validate_generated_modules(
            export_path, args.output_dir, package_name=args.package_name
        )
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    sys.exit(main())
