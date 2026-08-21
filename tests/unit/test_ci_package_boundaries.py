"""Release-boundary gates for CI package installation."""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

from scripts.run_installed_tests import (
    _prepare_installed_test_runtime,
    _remove_checkout_import_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
QUALITY_REQUIREMENTS = REPO_ROOT / "scripts" / "requirements-quality.txt"


def test_acceptance_workflows_never_install_editable_packages() -> None:
    editable_install = re.compile(r"\bpip(?:3)?\s+install\s+[^\n]*\s-e(?:\s|$)")
    violations = {
        path.name: editable_install.findall(path.read_text(encoding="utf-8"))
        for path in WORKFLOW_ROOT.glob("*.yml")
        if editable_install.search(path.read_text(encoding="utf-8"))
    }

    assert violations == {}


def test_installed_acceptance_uses_public_dependencies_and_wheels() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(encoding="utf-8")

    assert "default: 'pypi'" in workflow
    assert "|| 'pypi'" in workflow
    assert "scripts.install_ci_candidate" in workflow
    assert "scripts/run_installed_tests.py" in workflow
    assert "scripts/run_installed_tests.py --coverage" in workflow
    assert "tests/unit/pyqt_gui/test_progress_tree_aggregation.py" in workflow
    assert "pip','install','-e" not in workflow


def test_installed_wheel_integration_uses_headless_qt_platform() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  wheel-integration-test:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )

    assert match is not None
    wheel_job = match.group("body")
    assert "QT_QPA_PLATFORM: offscreen" in wheel_job
    assert "MPLBACKEND: Agg" in wheel_job


def test_source_unit_gate_checks_out_static_authority_submodules() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  unit-tests:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )

    assert match is not None
    unit_job = match.group("body")
    assert "submodules: recursive" in unit_job
    assert "git submodule foreach --recursive git fetch --tags --force" in unit_job


def test_code_quality_gate_derives_changed_python_files_from_git() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  code-quality:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )

    assert match is not None
    quality_job = match.group("body")
    assert "fetch-depth: 0" in quality_job
    assert "github.event.pull_request.base.sha" in quality_job
    assert "git diff --name-only --diff-filter=ACMR -z" in quality_job
    assert 'black --check "${changed_python_files[@]}"' in quality_job


def test_code_quality_toolchain_is_reproducibly_pinned() -> None:
    workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(encoding="utf-8")
    requirements = tuple(
        line
        for raw_line in QUALITY_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )

    assert "pip install -r scripts/requirements-quality.txt" in workflow
    assert requirements
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", item) for item in requirements)


def test_coverage_collection_and_publication_fail_closed() -> None:
    integration_workflow = (WORKFLOW_ROOT / "integration-tests.yml").read_text(
        encoding="utf-8"
    )
    coverage_workflow = (WORKFLOW_ROOT / "coverage-pages.yml").read_text(
        encoding="utf-8"
    )
    wheel_match = re.search(
        r"(?ms)^  wheel-integration-test:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        integration_workflow,
    )
    coverage_uploads = re.findall(
        r"(?ms)^      - name: Upload coverage artifact\n(?P<body>.*?)(?=^      - name:|^  [A-Za-z0-9_-]+:\n|\Z)",
        integration_workflow,
    )

    assert wheel_match is not None
    assert coverage_uploads
    assert all("if-no-files-found: error" in upload for upload in coverage_uploads)
    assert "cp .coverage .coverage.wheel" in wheel_match.group("body")
    assert "../.coverage.wheel" not in wheel_match.group("body")
    assert "github.paginate" in coverage_workflow
    assert "workflow_run.head_sha" in coverage_workflow
    assert "status: 'success'" in coverage_workflow
    assert "run.head_sha === context.sha" in coverage_workflow
    assert "workflow_run.conclusion == 'failure'" not in coverage_workflow
    assert "coverage report --fail-under=1" in coverage_workflow
    assert "No Coverage Data" not in coverage_workflow
    assert "if-no-files-found: error" in coverage_workflow
    assert "*/site-packages/openhcs" in (REPO_ROOT / ".coveragerc").read_text(
        encoding="utf-8"
    )


def test_docs_and_publish_gates_fetch_submodule_release_tags() -> None:
    fetch_command = "git submodule foreach --recursive git fetch --tags --force"

    assert fetch_command in (WORKFLOW_ROOT / "docs.yml").read_text(encoding="utf-8")
    assert fetch_command in (WORKFLOW_ROOT / "publish.yml").read_text(encoding="utf-8")


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
    missing_targets = tuple(
        target for target in targets if not (REPO_ROOT / target).exists()
    )

    assert targets
    assert missing_targets == ()


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
        [
            str(REPO_ROOT),
            str(REPO_ROOT / "scripts"),
            str(external_path),
            str(safe_path),
        ],
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


def test_installed_test_runner_resolves_wheel_before_exposing_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_root = tmp_path / "site-packages"
    installed_package = installed_root / "openhcs"
    installed_package.mkdir(parents=True)
    installed_init = installed_package / "__init__.py"
    installed_init.write_text(
        "raise AssertionError('package resolution must not execute OpenHCS')\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "openhcs", raising=False)
    monkeypatch.setattr(sys, "prefix", str(installed_root.parent))
    monkeypatch.setattr(sys, "base_prefix", "/opt/python")
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(installed_root)])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    package_path = _prepare_installed_test_runtime(REPO_ROOT)

    assert package_path == installed_init
    assert sys.path == [str(installed_root), str(REPO_ROOT)]
    assert "openhcs" not in sys.modules
