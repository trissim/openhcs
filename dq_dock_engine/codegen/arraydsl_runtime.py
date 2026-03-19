"""Runtime support for generated ArrayDSL primitive wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from dq_dock_engine.proof_status import ProofStatus


@dataclass(frozen=True)
class PrimitiveMetadata:
    """Generated wrapper metadata exported from Lean IR."""

    name: str
    lean_symbol: str
    jax_module: str
    jax_symbol: str
    lowering_kind: str
    supports_grad: bool
    proof_ref: Optional[str]
    proof_status: Optional[ProofStatus]
    callable: Callable


PRIMITIVE_REGISTRY: Dict[str, PrimitiveMetadata] = {}


def register_primitive(metadata: PrimitiveMetadata) -> PrimitiveMetadata:
    """Register generated primitive metadata and attach proof/runtime attributes."""

    if metadata.name in PRIMITIVE_REGISTRY:
        raise ValueError(f"Primitive already registered: {metadata.name}")

    fn = metadata.callable
    fn._lean_symbol = metadata.lean_symbol
    fn._jax_symbol = f"{metadata.jax_module}.{metadata.jax_symbol}"
    fn._lowering_kind = metadata.lowering_kind
    fn._supports_grad = metadata.supports_grad

    if metadata.proof_ref is not None:
        fn._lean_theorem = metadata.proof_ref

    if metadata.proof_status is not None:
        fn._proof_status = metadata.proof_status

    PRIMITIVE_REGISTRY[metadata.name] = metadata
    return metadata


def clear_registry() -> None:
    """Reset generated primitive registration state."""

    PRIMITIVE_REGISTRY.clear()


def get_registered_primitive(name: str) -> PrimitiveMetadata:
    """Return metadata for a generated primitive."""

    return PRIMITIVE_REGISTRY[name]
