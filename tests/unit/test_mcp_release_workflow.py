"""Static gates for the tag-driven official MCP publication boundary."""

import json
import re
import tomllib
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "publish.yml"
INTEGRATION_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "integration-tests.yml"
)
SERVER_PATH = REPO_ROOT / "server.json"
MCPB_ROOT = REPO_ROOT / "packaging" / "mcpb" / "openhcs"
RELEASE_DOCUMENTATION_PATH = (
    REPO_ROOT / "docs" / "source" / "development" / "mcp_release.rst"
)
INSTALLER_DOCUMENTATION_PATH = REPO_ROOT / "packaging" / "installers" / "README.md"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _ci_python_versions() -> set[Version]:
    workflow = yaml.safe_load(
        INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    )
    return {
        Version(str(version))
        for job in workflow["jobs"].values()
        for version in job.get("strategy", {})
        .get("matrix", {})
        .get("python-version", ())
    }


def test_registry_metadata_uses_the_project_authority_and_readme_marker():
    server = json.loads(SERVER_PATH.read_text(encoding="utf-8"))
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    package = server["packages"][0]

    assert package["registryType"] == "pypi"
    assert package["identifier"] == project["name"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[0] == f"<!-- mcp-name: {server['name']} -->"


def test_release_commands_read_the_package_version_authority():
    authority_import = (
        "from scripts.sync_mcp_release_metadata import read_package_version"
    )

    assert authority_import in RELEASE_DOCUMENTATION_PATH.read_text(encoding="utf-8")
    assert authority_import in INSTALLER_DOCUMENTATION_PATH.read_text(
        encoding="utf-8"
    )


def test_tag_workflow_publishes_registry_last_after_exact_pypi_signal():
    workflow = _workflow()
    assert "permissions" not in workflow
    triggers = workflow.get("on", workflow.get(True))
    manual_input = triggers["workflow_dispatch"]["inputs"]["release_version"]
    assert manual_input["required"] is True
    assert manual_input["type"] == "string"
    package_input = triggers["workflow_dispatch"]["inputs"][
        "publish_python_package"
    ]
    assert package_input == {
        "description": "Publish the Python package before registering it",
        "required": True,
        "default": False,
        "type": "boolean",
    }

    build_job = workflow["jobs"]["build-and-publish"]
    assert build_job["needs"] == [
        "build-windows-installer",
        "build-macos-installer",
    ]
    build_condition = build_job["if"]
    assert "always()" in build_condition
    assert "github.event_name == 'push'" in build_condition
    assert "needs.build-windows-installer.result == 'success'" in build_condition
    assert "needs.build-macos-installer.result == 'success'" in build_condition
    assert "github.event_name == 'workflow_dispatch'" in build_condition
    assert "inputs.publish_python_package" in build_condition
    assert build_job["permissions"] == {"contents": "write"}
    assert build_job["env"]["OPENHCS_RELEASE_VERSION"] == (
        "${{ github.event_name == 'workflow_dispatch' && "
        "inputs.release_version || github.ref_name }}"
    )
    build_steps = build_job["steps"]
    build_step_names = tuple(step.get("name") for step in build_steps)
    metadata_validation_index = build_step_names.index(
        "Validate release version and generated MCP metadata"
    )
    assert metadata_validation_index < build_step_names.index("Publish to PyPI")
    assert "check-jsonschema" in build_steps[metadata_validation_index]["run"]
    assert '"${OPENHCS_RELEASE_VERSION#v}"' in build_steps[
        metadata_validation_index
    ]["run"]
    installer_download = build_steps[
        build_step_names.index("Download desktop installers")
    ]
    github_release = build_steps[build_step_names.index("Create GitHub Release")]
    assert installer_download["if"] == "github.event_name == 'push'"
    assert github_release["if"] == "github.event_name == 'push'"

    registry_job = workflow["jobs"]["publish-mcp-registry"]
    assert registry_job["needs"] == "build-and-publish"
    registry_condition = registry_job["if"]
    assert "always()" in registry_condition
    assert "github.event_name == 'workflow_dispatch'" in registry_condition
    assert "needs.build-and-publish.result == 'success'" in registry_condition
    assert "!inputs.publish_python_package" in registry_condition
    assert registry_job["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    publisher_environment = registry_job["env"]
    assert publisher_environment["OPENHCS_RELEASE_VERSION"] == (
        "${{ github.event_name == 'workflow_dispatch' && "
        "inputs.release_version || github.ref_name }}"
    )
    assert re.fullmatch(
        r"v\d+\.\d+\.\d+",
        publisher_environment["MCP_PUBLISHER_VERSION"],
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        publisher_environment["MCP_PUBLISHER_LINUX_AMD64_SHA256"],
    )

    steps = registry_job["steps"]
    step_names = tuple(step.get("name") for step in steps)
    registry_metadata_step = steps[
        step_names.index("Validate release tag and generated MCP metadata")
    ]
    assert "scripts/sync_mcp_release_metadata.py --check" in registry_metadata_step[
        "run"
    ]
    assert '"${OPENHCS_RELEASE_VERSION#v}"' in registry_metadata_step["run"]
    registry_validation_index = step_names.index(
        "Validate official MCP Registry metadata"
    )
    wait_index = step_names.index("Wait for the published PyPI release")
    capability_validation_index = step_names.index(
        "Validate published desktop capability requirements"
    )
    assert wait_index < capability_validation_index < registry_validation_index
    capability_validation = steps[capability_validation_index]["run"]
    assert "--print-desktop-extras" in capability_validation
    assert "--capability-requirements" in capability_validation
    assert (
        '"${MCP_PYPI_PROJECT}[${DESKTOP_EXTRAS}]==${OPENHCS_RELEASE_VERSION#v}"'
    ) in capability_validation
    assert '"$RUNNER_TEMP/mcp-publisher" validate' in steps[
        registry_validation_index
    ]["run"]

    wait_step = steps[wait_index]
    assert 'package["registryType"] == "pypi"' in wait_step["run"]
    assert "scripts/wait_for_pypi_release.py" in wait_step["run"]
    assert '"$MCP_PYPI_PROJECT"' in wait_step["run"]
    assert '"${OPENHCS_RELEASE_VERSION#v}"' in wait_step["run"]
    assert "--timeout-seconds 900" in wait_step["run"]
    assert "--poll-interval-seconds 5" in wait_step["run"]

    registry_step = steps[-1]
    assert registry_step["name"] == "Publish to the official MCP Registry"
    assert "login github-oidc" in registry_step["run"]
    assert '"$RUNNER_TEMP/mcp-publisher" publish' in registry_step["run"]


def test_tag_workflow_installs_linux_pyqt_runtime_before_wheel_smoke():
    build_steps = _workflow()["jobs"]["build-and-publish"]["steps"]
    step_names = tuple(step.get("name") for step in build_steps)
    runtime_index = step_names.index("Install PyQt runtime libraries")
    smoke_index = step_names.index("Smoke-test installed MCP wheel outside checkout")

    assert runtime_index < smoke_index
    runtime_setup = build_steps[runtime_index]["run"]
    assert "sudo apt-get install -y libgl1" in runtime_setup
    assert (
        "sudo apt-get install -y libegl1 || sudo apt-get install -y libegl1-mesa"
    ) in runtime_setup
    smoke = build_steps[smoke_index]["run"]
    assert "--print-desktop-extras" in smoke
    assert 'pip install "${WHEEL}[${DESKTOP_EXTRAS}]"' in smoke
    assert "--capability-requirements" in smoke


def test_pypi_wheel_smoke_uses_the_canonical_pipeline_document_boundary():
    workflow_text = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "from openhcs.core.pipeline import Pipeline" not in workflow_text

    workflow = yaml.safe_load(workflow_text)
    steps = workflow["jobs"]["pypi-installation-test"]["steps"]
    headless_step = next(
        step for step in steps if step.get("name") == "Test headless installation (base - no extras)"
    )
    smoke = headless_step["run"]

    assert "from openhcs.core.config import PipelineConfig" in smoke
    assert (
        "from openhcs.core.pipeline_document import PipelineDocumentAuthority" in smoke
    )
    assert "from openhcs.core.steps.function_step import FunctionStep" in smoke
    assert "PipelineDocumentAuthority.from_values(" in smoke
    assert "pipeline_config=PipelineConfig()" in smoke
    assert "pipeline_steps=[FunctionStep()]" in smoke
    assert "Pipeline(" not in smoke

    desktop_step = next(
        step
        for step in steps
        if step.get("name") == "Test declared desktop MCP installation outside checkout"
    )
    desktop_smoke = desktop_step["run"]
    assert "--print-desktop-extras" in desktop_smoke
    assert 'pip install "${WHEEL}[${DESKTOP_EXTRAS}]"' in desktop_smoke
    assert "--capability-requirements" in desktop_smoke


def test_pypi_classifiers_cover_every_ci_matrix_python_version():
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    python_versions = _ci_python_versions()

    classifiers = set(project["classifiers"])
    assert {
        f"Programming Language :: Python :: {version}" for version in python_versions
    } <= classifiers


def test_macos_integration_jobs_disable_x86_only_intel_svml():
    workflow = yaml.safe_load(
        INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    )
    expected = "${{ runner.os == 'macOS' && '1' || '0' }}"

    for job_name, step_name in (
        ("python-boundary-tests", "Run boundary version tests"),
        ("backend-microscope-tests", "Run backend/microscope combination tests"),
    ):
        steps = workflow["jobs"][job_name]["steps"]
        test_step = next(step for step in steps if step.get("name") == step_name)
        assert test_step["env"]["NUMBA_DISABLE_INTEL_SVML"] == expected


def test_native_macos_installer_uses_native_qt_smoke_harness():
    workflow = yaml.safe_load(
        INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["desktop-installer-source-test"]["steps"]
    smoke_step = next(
        step
        for step in steps
        if step.get("name") == "Execute and verify macOS installer"
    )
    assert "QT_QPA_PLATFORM" not in smoke_step.get("env", {})
    assert "scripts.smoke_installed_desktop" in smoke_step["run"]


def test_mcpb_python_ranges_match_the_ci_supported_boundary():
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    wrapper = tomllib.loads(
        (MCPB_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    manifest = json.loads(
        (MCPB_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    ci_versions = _ci_python_versions()

    project_range = SpecifierSet(project["requires-python"])
    wrapper_range = SpecifierSet(wrapper["requires-python"])
    manifest_range = SpecifierSet(
        manifest["compatibility"]["runtimes"]["python"].replace(" ", ",")
    )
    assert wrapper_range == manifest_range
    assert all(version in project_range for version in ci_versions)
    assert all(version in wrapper_range for version in ci_versions)

    oldest = min(ci_versions)
    newest = max(ci_versions)
    previous_minor = Version(f"{oldest.major}.{oldest.minor - 1}")
    next_minor = Version(f"{newest.major}.{newest.minor + 1}")
    assert previous_minor not in wrapper_range
    assert next_minor not in wrapper_range
