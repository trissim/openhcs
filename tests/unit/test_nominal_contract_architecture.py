"""Static and runtime gates for nominal product contracts."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from openhcs.core.pipeline.function_contracts import runtime_bound_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOTS = (
    REPO_ROOT / "openhcs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "packaging",
)


def _protocol_declarations(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_aliases: set[str] = set()
    typing_modules: set[str] = set()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "typing",
            "typing_extensions",
        }:
            for alias in node.names:
                if alias.name == "Protocol":
                    imported_aliases.add(alias.asname or alias.name)
                    violations.append(f"line {node.lineno}: imports Protocol")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"typing", "typing_extensions"}:
                    typing_modules.add(alias.asname or alias.name)

    for node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in imported_aliases:
                violations.append(f"line {node.lineno}: {node.name} inherits Protocol")
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "Protocol"
                and isinstance(base.value, ast.Name)
                and base.value.id in typing_modules
            ):
                violations.append(f"line {node.lineno}: {node.name} inherits Protocol")
    return tuple(violations)


def test_product_code_uses_nominal_abcs_instead_of_typing_protocols() -> None:
    violations = {
        str(path.relative_to(REPO_ROOT)): declarations
        for root in PRODUCT_ROOTS
        for path in root.rglob("*.py")
        if (declarations := _protocol_declarations(path))
    }

    assert violations == {}


def test_runtime_parameter_decorator_rejects_structural_impostor() -> None:
    class StructuralRuntimeParameter:
        @classmethod
        def require_parameter_name(cls) -> str:
            return "runtime_value"

        @classmethod
        def parameter(cls) -> inspect.Parameter:
            return inspect.Parameter(
                cls.require_parameter_name(),
                inspect.Parameter.KEYWORD_ONLY,
            )

    with pytest.raises(TypeError, match="RuntimeParameterDeclarationABC subclasses"):
        runtime_bound_parameters(StructuralRuntimeParameter)
