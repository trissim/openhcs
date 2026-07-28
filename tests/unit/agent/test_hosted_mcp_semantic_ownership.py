"""Deletion gates against hosted-plugin semantic mirrors."""

from __future__ import annotations

import ast
from pathlib import Path

from openhcs.agent.capabilities import (
    CapabilityTransport,
    get_capability_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
HOSTED_BOUNDARY_FILES = (
    REPO_ROOT / "openhcs" / "mcp" / "http.py",
    REPO_ROOT / "scripts" / "build_hosted_mcp_plugin.py",
)


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _attribute_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_hosted_boundaries_do_not_copy_capability_names():
    hosted_names = {
        capability.name
        for capability in get_capability_registry(
            CapabilityTransport.HOSTED_STREAMABLE_HTTP
        ).capabilities
    }

    for path in HOSTED_BOUNDARY_FILES:
        assert hosted_names.isdisjoint(_string_literals(path)), path


def test_hosted_boundaries_do_not_rederive_capability_mutation_semantics():
    for path in HOSTED_BOUNDARY_FILES:
        assert {"mutating", "side_effects"}.isdisjoint(_attribute_names(path)), path
