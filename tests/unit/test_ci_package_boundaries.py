"""Release-boundary gates for CI package installation."""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

from scripts.run_installed_tests import _remove_checkout_import_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"


def test_acceptance_workflows_never_install_editable_packages() -> None:
    editable_install = re.compile(r"\bpip(?:3)?\s+install\s+[^\n]*\s-e(?:\s|$)")
    violations = {
        path.name: editable_install.findall(path.read_text(encoding="utf-8"))
        for path in WORKFLOW_ROOT.glob("*.yml")
        if editable_install.search(path.read_text(encoding="utf-8"))
    }

    assert violations == {}


def test_integration_acceptance_defaults_to_public_dependencies_and_wheel_imports() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(
        encoding="utf-8"
    )

    assert "default: 'pypi'" in workflow
    assert "|| 'pypi'" in workflow
    assert "scripts.install_ci_candidate" in workflow
    assert "scripts/run_installed_tests.py" in workflow
    assert "tests/unit/pyqt_gui/test_progress_tree_aggregation.py" in workflow
    assert "pip','install','-e" not in workflow


def test_installed_wheel_integration_uses_headless_qt_platform() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  wheel-integration-test:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )

    assert match is not None
    wheel_job = match.group("body")
    assert "QT_QPA_PLATFORM: offscreen" in wheel_job
    assert "MPLBACKEND: Agg" in wheel_job


def test_source_unit_gate_checks_out_static_authority_submodules() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  unit-tests:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )

    assert match is not None
    unit_job = match.group("body")
    assert "submodules: recursive" in unit_job
    assert "git submodule foreach --recursive git fetch --tags --force" in unit_job


def test_docs_and_publish_gates_fetch_submodule_release_tags() -> None:
    fetch_command = "git submodule foreach --recursive git fetch --tags --force"

    assert fetch_command in (WORKFLOW_ROOT / "docs.yml").read_text(encoding="utf-8")
    assert fetch_command in (WORKFLOW_ROOT / "publish.yml").read_text(
        encoding="utf-8"
    )


def test_documentation_workflow_targets_exist_in_the_recursive_checkout() -> None:
    workflow = (WORKFLOW_ROOT / "docs.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^\s+docs_targets=\(\n(?P<body>.*?)^\s+\)\n",
        workflow,
    )

    assert match is not None
    targets = tuple(
        line.strip() for line in match.group("body").splitlines() if line.strip()
    )
    assert targets
    assert tuple(
        target for target in targets if not (REPO_ROOT / target).exists()
    ) == ()


def test_candidate_builder_discovers_external_projects_from_package_metadata() -> None:
    source = (REPO_ROOT / "scripts" / "install_ci_candidate.py").read_text(
        encoding="utf-8"
    )

    assert "discover_local_projects()" in source
    assert '"external/ObjectState"' not in source
    assert '"external/pyqt-reactive"' not in source
    assert '"-e"' not in source


def test_candidate_builder_installs_outside_the_source_checkout() -> None:
    source = (REPO_ROOT / "scripts" / "install_ci_candidate.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    pip_installs = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "run"
            and call.args
            and isinstance(call.args[0], ast.Tuple)
        ):
            continue
        string_arguments = {
            element.value
            for element in call.args[0].elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if {"pip", "install"} <= string_arguments:
            pip_installs.append(call)

    assert pip_installs
    assert all(
        any(
            keyword.arg == "cwd"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "wheel_directory"
            for keyword in call.keywords
        )
        for call in pip_installs
    )


def test_installed_test_runner_removes_checkout_paths_from_parent_and_children(
    monkeypatch,
    tmp_path: Path,
) -> None:
    external_path = REPO_ROOT / "external" / "pyqt-reactive" / "src"
    safe_path = tmp_path / "installed"
    monkeypatch.setattr(
        sys,
        "path",
        [str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(external_path), str(safe_path)],
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(("", str(REPO_ROOT), str(external_path), str(safe_path))),
    )

    _remove_checkout_import_paths(REPO_ROOT)

    assert sys.path == [str(safe_path)]
    assert os.environ["PYTHONPATH"] == str(safe_path)


def test_installed_test_runner_preserves_venv_nested_under_checkout(
    monkeypatch,
) -> None:
    nested_environment = REPO_ROOT / "test-wheel"
    site_packages = nested_environment / "lib" / "python3.12" / "site-packages"
    monkeypatch.setattr(sys, "prefix", str(nested_environment))
    monkeypatch.setattr(sys, "base_prefix", "/opt/python")
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(site_packages)])
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(REPO_ROOT), str(site_packages))),
    )

    _remove_checkout_import_paths(REPO_ROOT)

    assert sys.path == [str(site_packages)]
    assert os.environ["PYTHONPATH"] == str(site_packages)
