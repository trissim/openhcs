from dq_dock_engine.arraydsl import PRIMITIVE_REGISTRY
from dq_dock_engine.codegen.backend_fidelity import (
    check_registry_semantics,
    resolve_backend_symbol,
)


def test_generated_backend_symbols_resolve():
    for metadata in PRIMITIVE_REGISTRY.values():
        assert resolve_backend_symbol(metadata) is not None


def test_generated_registry_semantic_fidelity():
    check_registry_semantics(PRIMITIVE_REGISTRY)
