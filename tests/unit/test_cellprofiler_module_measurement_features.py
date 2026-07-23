"""Static ownership gates for CellProfiler measurement feature authorities."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_AUTHORITY_PATH = (
    PROJECT_ROOT / "openhcs/interop/cellprofiler/module_measurement_features.py"
)


def test_relationship_feature_hook_uses_callable_artifact_authorities() -> None:
    """Keep the retired module artifact wrapper out of feature ownership."""

    tree = ast.parse(FEATURE_AUTHORITY_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "relationship_distance_measurements_apply"
    )

    assert "openhcs.core.module_artifact_contract" not in imported_modules
    assert "ModuleArtifactContract" not in imported_names
    assert [argument.arg for argument in hook.args.args] == [
        "cls",
        "callable_contract",
        "relationship_spec",
    ]
    assert isinstance(hook.args.args[1].annotation, ast.Constant)
    assert hook.args.args[1].annotation.value == "CallableContract"
    assert isinstance(hook.args.args[2].annotation, ast.Constant)
    assert hook.args.args[2].annotation.value == "ArtifactSpec"
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        for node in ast.walk(hook)
    )
