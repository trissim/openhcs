"""Generate Python/JAX wrappers from the exported Lean ArrayDSL IR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast


DEFAULT_EXPORT_PATH = Path(
    "docs/papers/paper4_decision_quotient/proofs/arraydsl_primitives.json"
)
DEFAULT_OUTPUT_DIR = Path("dq_dock_engine/generated")


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
    "distance": "return jnp.linalg.norm(arr1 - arr2)",
    "pairwise_distances": "return jnp.abs(coords1[:, None] - coords2[None, :])",
    "apply_cutoff": "return jnp.where(distances < rc, distances, 0.0)",
    "lennard_jones": "safe_r = jnp.where(r == 0, 1.0, r)\nsr = sigma / safe_r\nenergy = 4.0 * epsilon * (sr ** 12 - sr ** 6)\nreturn jnp.where(r == 0, 0.0, energy)",
    "sum_pair_potentials": "masked = applyCutoff(distances, rc)\nenergies = jax.vmap(lambda r: lennardJones(epsilon, sigma, r))(masked)\nreturn jnp.sum(energies)",
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


def _render_registry_module(primitives: List[PrimitiveIR]) -> str:
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
from dq_dock_engine.generated.arraydsl_primitives import (
    {imports},
)
from dq_dock_engine.proof_status import ProofStatus


{registration_block}


ARRAYDSL_PRIMITIVES = tuple(PRIMITIVE_REGISTRY[name] for name in [
    {exported_names}
])
'''


def generate_modules(export_path: Path, output_dir: Path) -> List[Path]:
    primitives = _load_export(export_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_path = output_dir / "__init__.py"
    primitives_path = output_dir / "arraydsl_primitives.py"
    registry_path = output_dir / "arraydsl_registry.py"

    init_path.write_text('"""Generated Lean-backed JAX wrappers."""\n')
    primitives_path.write_text(_render_primitives_module(primitives))
    registry_path.write_text(_render_registry_module(primitives))
    return [init_path, primitives_path, registry_path]


def validate_generated_modules(export_path: Path, output_dir: Path) -> None:
    from dq_dock_engine.codegen.arraydsl_runtime import clear_registry

    primitives = _load_export(export_path)
    generated_names = {primitive.name for primitive in primitives}

    clear_registry()
    namespace: Dict[str, object] = {}
    registry_path = output_dir / "arraydsl_registry.py"
    exec(compile(registry_path.read_text(), str(registry_path), "exec"), namespace)

    registry = cast(Dict[str, object], namespace["PRIMITIVE_REGISTRY"])
    registered_names = set(registry)
    if registered_names != generated_names:
        raise ValueError(
            f"Generated registry mismatch: expected {sorted(generated_names)}, got {sorted(registered_names)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_paths = generate_modules(args.input, args.output_dir)
    if args.check:
        validate_generated_modules(args.input, args.output_dir)
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
