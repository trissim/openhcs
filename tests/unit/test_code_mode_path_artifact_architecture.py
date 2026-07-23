"""Static architecture gates for code-mode paths and Artifact UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
OWNED_BOUNDARIES = (
    ROOT / "openhcs/serialization/source_path_factoring.py",
    ROOT / "openhcs/core/pipeline/compilation_session.py",
    ROOT / "openhcs/core/artifact_inspection.py",
    ROOT / "openhcs/core/progress/runtime_artifacts.py",
    ROOT / "openhcs/runtime/zmq_control.py",
    ROOT / "openhcs/pyqt_gui/widgets/artifact_plan_view.py",
    ROOT
    / "openhcs/pyqt_gui/widgets/shared/services/runtime_artifact_progress_service.py",
)
ARTIFACT_UI_BOUNDARIES = (
    ROOT / "openhcs/pyqt_gui/widgets/artifact_plan_view.py",
    ROOT
    / "openhcs/pyqt_gui/widgets/shared/services/compile_workflow_service.py",
    ROOT
    / "openhcs/pyqt_gui/widgets/shared/services/runtime_artifact_progress_service.py",
    ROOT
    / "openhcs/pyqt_gui/widgets/shared/services/progress_workflow_service.py",
    ROOT / "openhcs/pyqt_gui/windows/dual_editor_window.py",
    ROOT / "openhcs/pyqt_gui/windows/dual_editor_session.py",
    ROOT / "openhcs/pyqt_gui/windows/dual_editor_tab_builder.py",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        return node.attr if owner is None else f"{owner}.{node.attr}"
    return None


def test_owned_boundaries_use_no_reflection_or_ambient_working_directory() -> None:
    violations = []
    for path in OWNED_BOUNDARIES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"getattr", "setattr"}:
                    violations.append((path, node.lineno, node.func.id))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and (owner.id, node.func.attr)
                    in {("Path", "cwd"), ("os", "getcwd")}
                ):
                    violations.append(
                        (path, node.lineno, f"{owner.id}.{node.func.attr}")
                    )

    assert violations == []


def test_generic_compiler_transport_and_ui_import_no_concrete_backends() -> None:
    violations = []
    for path in OWNED_BOUNDARIES:
        for node in ast.walk(_tree(path)):
            imported_modules = ()
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = (node.module,)
            for module in imported_modules:
                if module.startswith("openhcs.processing.backends"):
                    violations.append((path, node.lineno, module))

    assert violations == []


def test_deleted_preview_and_debug_router_cannot_return() -> None:
    deleted_paths = (
        ROOT / "openhcs/pyqt_gui/widgets/artifact_contract_preview.py",
        ROOT / "tests/unit/test_artifact_contract_preview.py",
        ROOT / "openhcs/runtime/zmq_debug_control.py",
    )
    assert tuple(path for path in deleted_paths if path.exists()) == ()


def test_runtime_has_one_registered_control_router() -> None:
    routers = []
    for path in (ROOT / "openhcs/runtime").glob("*.py"):
        for node in _tree(path).body:
            if isinstance(node, ast.ClassDef) and node.name.endswith(
                "ControlMessageRouter"
            ):
                routers.append((path.name, node.name))

    assert routers == [("zmq_control.py", "ZMQControlMessageRouter")]


def test_artifact_ui_never_compiles_or_reconstructs_artifact_semantics() -> None:
    forbidden_import_prefixes = (
        "openhcs.core.pipeline.compiler",
        "openhcs.core.orchestrator",
    )
    forbidden_calls = {
        "ArtifactSpec.input",
        "ArtifactSpec.output",
        "CallableContract.from_callable",
        "CompiledArtifactInspection",
        "CompiledArtifactInspection.from_execution_bundle",
        "PipelineCompiler",
        "PipelineOrchestrator",
        "RuntimeArtifactAddress",
    }
    violations = []
    for path in ARTIFACT_UI_BOUNDARIES:
        for node in ast.walk(_tree(path)):
            imported_modules = ()
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = (node.module,)
            for module in imported_modules:
                if module.startswith(forbidden_import_prefixes):
                    violations.append((path, node.lineno, module))
            if not isinstance(node, ast.Call):
                continue
            call_name = _qualified_name(node.func)
            if call_name in forbidden_calls or call_name == "compile_pipelines":
                violations.append((path, node.lineno, call_name))

    assert violations == []


def test_server_and_worker_are_the_only_artifact_projection_producers() -> None:
    compiled_projection_owners = set()
    runtime_projection_owners = set()
    for path in (ROOT / "openhcs").rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            call_name = _qualified_name(node.func)
            relative_path = str(path.relative_to(ROOT))
            if call_name == "CompiledArtifactInspection.from_execution_bundle":
                compiled_projection_owners.add(relative_path)
            if call_name == "runtime_artifact_context_for_records":
                runtime_projection_owners.add(relative_path)

    assert compiled_projection_owners == {"openhcs/runtime/zmq_control.py"}
    assert runtime_projection_owners == {
        "openhcs/core/orchestrator/worker_execution.py"
    }
