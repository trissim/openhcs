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


def _uses_action(step: dict, action: str) -> bool:
    return "uses" in step and step["uses"].partition("@")[0] == action


def _ci_python_versions() -> set[Version]:
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    return {
        Version(str(version))
        for job in workflow["jobs"].values()
        for version in job.get("strategy", {})
        .get("matrix", {})
        .get("python-version", ())
    }


def test_registry_metadata_uses_the_project_authority_and_readme_marker():
    server = json.loads(SERVER_PATH.read_text(encoding="utf-8"))
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
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
    assert authority_import in INSTALLER_DOCUMENTATION_PATH.read_text(encoding="utf-8")


def test_tag_workflow_publishes_registry_last_after_exact_pypi_signal():
    workflow = _workflow()
    assert "permissions" not in workflow
    triggers = workflow.get("on", workflow.get(True))
    manual_input = triggers["workflow_dispatch"]["inputs"]["release_version"]
    assert manual_input["required"] is True
    assert manual_input["type"] == "string"
    package_input = triggers["workflow_dispatch"]["inputs"]["publish_python_package"]
    assert package_input == {
        "description": "Publish the Python package before registering it",
        "required": True,
        "default": False,
        "type": "boolean",
    }

    verification_job = workflow["jobs"]["verify-release-commit"]
    assert verification_job["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert set(verification_job["outputs"]) == {
        "release_sha",
        "release_tag",
        "release_version",
    }
    verification_steps = verification_job["steps"]
    verification_step_names = tuple(step.get("name") for step in verification_steps)
    checkout = verification_steps[0]
    assert checkout["with"]["fetch-depth"] == 0
    assert "refs/tags/v{0}" in checkout["with"]["ref"]
    resolve = verification_steps[
        verification_step_names.index("Resolve the immutable release tag")
    ]["run"]
    assert 'git cat-file -t "refs/tags/$release_tag"' in resolve
    assert 'git rev-list -n 1 "refs/tags/$release_tag"' in resolve
    evidence = verification_steps[
        verification_step_names.index("Verify exact-commit release evidence")
    ]
    assert evidence["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "scripts.release_readiness" in evidence["run"]
    assert '--commit "${{ steps.release.outputs.release_sha }}"' in evidence["run"]

    for installer_job_name in (
        "build-windows-installer",
        "build-macos-installer",
    ):
        installer_job = workflow["jobs"][installer_job_name]
        assert installer_job["needs"] == "verify-release-commit"
        assert installer_job["env"]["OPENHCS_RELEASE_VERSION"] == (
            "${{ needs.verify-release-commit.outputs.release_version }}"
        )
        installer_checkout = next(
            step
            for step in installer_job["steps"]
            if _uses_action(step, "actions/checkout")
        )
        assert installer_checkout["with"]["ref"] == (
            "${{ needs.verify-release-commit.outputs.release_sha }}"
        )
        assert installer_checkout["with"]["submodules"] == "recursive"

    build_job = workflow["jobs"]["build-and-publish"]
    assert build_job["needs"] == [
        "verify-release-commit",
        "build-windows-installer",
        "build-macos-installer",
    ]
    build_condition = build_job["if"]
    assert "always()" in build_condition
    assert "needs.verify-release-commit.result == 'success'" in build_condition
    assert "github.event_name == 'push'" in build_condition
    assert "needs.build-windows-installer.result == 'success'" in build_condition
    assert "needs.build-macos-installer.result == 'success'" in build_condition
    assert "github.event_name == 'workflow_dispatch'" in build_condition
    assert "inputs.publish_python_package" in build_condition
    assert build_job["permissions"] == {"contents": "write"}
    assert build_job["env"]["OPENHCS_RELEASE_VERSION"] == (
        "${{ needs.verify-release-commit.outputs.release_version }}"
    )
    build_steps = build_job["steps"]
    assert build_steps[0]["with"]["ref"] == (
        "${{ needs.verify-release-commit.outputs.release_sha }}"
    )
    build_step_names = tuple(step.get("name") for step in build_steps)
    metadata_validation_index = build_step_names.index(
        "Validate release version and generated MCP metadata"
    )
    assert metadata_validation_index < build_step_names.index("Publish to PyPI")
    assert "check-jsonschema" in build_steps[metadata_validation_index]["run"]
    assert (
        '"${OPENHCS_RELEASE_VERSION#v}"'
        in build_steps[metadata_validation_index]["run"]
    )
    installer_download = build_steps[
        build_step_names.index("Download desktop installers")
    ]
    github_release = build_steps[build_step_names.index("Create GitHub Release")]
    assert installer_download["if"] == (
        "github.event_name == 'push' || inputs.publish_desktop_installers"
    )
    assert "if" not in github_release
    assert github_release["with"]["tag_name"] == (
        "${{ needs.verify-release-commit.outputs.release_tag }}"
    )

    release_recovery = workflow["jobs"]["publish-release-recovery"]
    assert release_recovery["needs"] == [
        "verify-release-commit",
        "build-windows-installer",
        "build-macos-installer",
    ]
    assert "!inputs.publish_python_package" in release_recovery["if"]
    recovery_steps = release_recovery["steps"]
    recovery_step_names = tuple(step.get("name") for step in recovery_steps)
    assert "Require existing release tag" not in recovery_step_names
    recovery_checkout = recovery_steps[0]
    assert recovery_checkout["with"] == {
        "ref": "${{ needs.verify-release-commit.outputs.release_sha }}",
        "submodules": "recursive",
    }
    materialize_step = recovery_steps[
        recovery_step_names.index("Materialize published release distributions")
    ]["run"]
    assert "python -m build" not in materialize_step
    assert "scripts/wait_for_pypi_release.py" in materialize_step
    assert "--release-directory dist" in materialize_step
    assert "scripts.validate_wheel_deployment" in materialize_step
    assert "python -m twine check dist/*" in materialize_step
    release_step = recovery_steps[
        recovery_step_names.index("Create or update complete release")
    ]
    assert release_step["with"]["tag_name"] == (
        "${{ needs.verify-release-commit.outputs.release_tag }}"
    )
    assert "dist/*" in release_step["with"]["files"]
    assert "installer-assets/*" in release_step["with"]["files"]

    registry_job = workflow["jobs"]["publish-mcp-registry"]
    assert registry_job["needs"] == [
        "verify-release-commit",
        "build-and-publish",
        "publish-release-recovery",
    ]
    registry_condition = registry_job["if"]
    assert "always()" in registry_condition
    assert "github.event_name == 'workflow_dispatch'" in registry_condition
    assert "needs.build-and-publish.result == 'success'" in registry_condition
    assert "needs.publish-release-recovery.result == 'success'" in registry_condition
    assert "!inputs.publish_python_package" in registry_condition
    assert "needs.verify-release-commit.result == 'success'" in registry_condition
    assert registry_job["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    publisher_environment = registry_job["env"]
    assert publisher_environment["OPENHCS_RELEASE_VERSION"] == (
        "${{ needs.verify-release-commit.outputs.release_version }}"
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
    assert steps[0]["with"]["ref"] == (
        "${{ needs.verify-release-commit.outputs.release_sha }}"
    )
    step_names = tuple(step.get("name") for step in steps)
    registry_metadata_step = steps[
        step_names.index("Validate release tag and generated MCP metadata")
    ]
    assert (
        "scripts/sync_mcp_release_metadata.py --check" in registry_metadata_step["run"]
    )
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
        '"${MCP_PYPI_PROJECT}[${DESKTOP_EXTRAS}] @ ${MCP_WHEEL_URL}"'
    ) in capability_validation
    assert (
        'MCP_WHEEL_URL=$(< "$RUNNER_TEMP/openhcs-release-wheel-url")'
        in capability_validation
    )
    assert (
        '"$RUNNER_TEMP/mcp-publisher" validate'
        in steps[registry_validation_index]["run"]
    )

    wait_step = steps[wait_index]
    assert "read_package_name" in wait_step["run"]
    assert "scripts/wait_for_pypi_release.py" in wait_step["run"]
    assert '"$MCP_PYPI_PROJECT"' in wait_step["run"]
    assert '"${OPENHCS_RELEASE_VERSION#v}"' in wait_step["run"]
    assert "--timeout-seconds 900" in wait_step["run"]
    assert "--poll-interval-seconds 5" in wait_step["run"]
    assert (
        '--wheel-url-output "$RUNNER_TEMP/openhcs-release-wheel-url"'
        in wait_step["run"]
    )

    registry_step = steps[-1]
    assert registry_step["name"] == "Publish to the official MCP Registry"
    assert "login github-oidc" in registry_step["run"]
    assert '"$RUNNER_TEMP/mcp-publisher" publish' in registry_step["run"]

    publish_step = next(
        step for step in build_steps if step.get("name") == "Publish to PyPI"
    )
    assert "--skip-existing" not in publish_step["run"]


def test_tag_workflow_installs_linux_pyqt_runtime_before_wheel_smoke():
    build_steps = _workflow()["jobs"]["build-and-publish"]["steps"]
    step_names = tuple(step.get("name") for step in build_steps)
    runtime_index = step_names.index("Install PyQt runtime libraries")
    smoke_index = step_names.index(
        "Smoke-test installed desktop wheel outside checkout"
    )

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
    assert "scripts/smoke_installed_mcp.py" in smoke
    assert "scripts/smoke_installed_gui.py" in smoke
    assert build_steps[smoke_index]["env"]["QT_QPA_PLATFORM"] == "offscreen"
    assert smoke.index("DESKTOP_EXTRAS=") < smoke.index("python -m venv")
    assert smoke.index("WHEEL=") < smoke.index("python -m venv")


def test_pypi_wheel_smoke_uses_the_canonical_pipeline_document_boundary():
    workflow_text = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "from openhcs.core.pipeline import Pipeline" not in workflow_text

    workflow = yaml.safe_load(workflow_text)
    steps = workflow["jobs"]["pypi-installation-test"]["steps"]
    headless_step = next(
        step
        for step in steps
        if step.get("name") == "Test headless installation (base - no extras)"
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
    assert "python -m scripts.install_ci_candidate" in desktop_smoke
    assert '--candidate-wheel "$WHEEL"' in desktop_smoke
    assert '--extras "$DESKTOP_EXTRAS"' in desktop_smoke
    assert "--capability-requirements" in desktop_smoke

    gui_step = next(
        step
        for step in steps
        if step.get("name") == "Test installed GUI and live MCP outside checkout"
    )
    gui_smoke = gui_step["run"]
    assert "scripts/smoke_installed_gui.py" in gui_smoke
    assert "--forbid-import-root" in gui_smoke
    assert "QT_QPA_PLATFORM=offscreen" in gui_smoke

    install_steps = tuple(
        step
        for step in steps
        if step.get("name", "").startswith("Test ")
        and "installation" in step.get("name", "")
    )
    assert len(install_steps) == 4
    assert all(
        "python -m scripts.install_ci_candidate" in step["run"]
        and "--published-wheel-requirements-json" in step["run"]
        and '--candidate-wheel "$WHEEL"' in step["run"]
        for step in install_steps
    )


def test_source_candidate_jobs_run_independently_from_publication_readiness():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    source_candidate_jobs = {
        "unit-tests",
        "gui-tests",
        "python-boundary-tests",
        "backend-microscope-tests",
        "omero-tests-linux",
        "official30-headless-parity",
        "official30-real-viewer-smoke",
        "wheel-integration-test",
        "desktop-installer-source-test",
    }

    for job_name in source_candidate_jobs:
        job = jobs[job_name]
        assert "needs" not in job
        checkout = next(
            step for step in job["steps"] if _uses_action(step, "actions/checkout")
        )
        assert checkout["with"]["submodules"] == "recursive"
        install = next(
            step
            for step in job["steps"]
            if "scripts.install_ci_candidate" in step.get("run", "")
        )["run"]
        assert "--dependency-source submodules" in install
        assert "--published-wheel-requirements-json" not in install


def test_pypi_consumers_wait_once_for_metadata_declared_dependencies():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    readiness = jobs["pypi-dependency-readiness"]
    wait_step = next(
        step
        for step in readiness["steps"]
        if step.get("name") == "Wait for installer-visible dependency releases"
    )
    consumers = {"pypi-installation-test"}
    assert all(
        jobs[job_name]["needs"] == "pypi-dependency-readiness" for job_name in consumers
    )
    assert "python -m scripts.validate_local_release_floors" in wait_step["run"]
    assert "--wait-for-pypi" in wait_step["run"]
    assert "--wheel-requirements-output" in wait_step["run"]
    assert readiness["outputs"]["published_wheel_requirements"]
    pypi_install = "\n".join(
        step.get("run", "") for step in jobs["pypi-installation-test"]["steps"]
    )
    assert "--dependency-source pypi" in pypi_install
    assert "--published-wheel-requirements-json" in pypi_install
    assert "pyqt-reactive" not in wait_step["run"]


def test_pypi_classifiers_cover_every_ci_matrix_python_version():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    python_versions = _ci_python_versions()

    classifiers = set(project["classifiers"])
    assert {
        f"Programming Language :: Python :: {version}" for version in python_versions
    } <= classifiers


def test_pypi_metadata_declares_single_beta_maturity_classifier():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    maturity_classifiers = {
        classifier
        for classifier in project["classifiers"]
        if classifier.startswith("Development Status ::")
    }

    assert maturity_classifiers == {"Development Status :: 4 - Beta"}


def test_macos_integration_jobs_disable_x86_only_intel_svml():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    expected = "${{ runner.os == 'macOS' && '1' || '0' }}"

    for job_name, step_name in (
        ("python-boundary-tests", "Run boundary version tests"),
        ("backend-microscope-tests", "Run backend/microscope combination tests"),
    ):
        steps = workflow["jobs"][job_name]["steps"]
        test_step = next(step for step in steps if step.get("name") == step_name)
        assert test_step["env"]["NUMBA_DISABLE_INTEL_SVML"] == expected


def test_cross_platform_integration_jobs_have_bounded_runtime():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert workflow["jobs"]["python-boundary-tests"]["timeout-minutes"] == 45
    assert workflow["jobs"]["backend-microscope-tests"]["timeout-minutes"] == 25


def test_real_viewer_smoke_validates_native_qt_and_prewarms_managed_fiji():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["official30-real-viewer-smoke"]["steps"]
    step_names = tuple(step.get("name") for step in steps)
    runtime_index = step_names.index("Install graphical runtime libraries")
    cache_index = step_names.index("Cache managed Java and ImageJ artifacts")
    prewarm_index = step_names.index("Prewarm managed Fiji runtime")
    bioformats_index = step_names.index("Run real Bio-Formats ImageXpress source smoke")
    viewer_index = step_names.index("Run Fiji and Fiji plus Napari real-viewer smokes")

    runtime_setup = steps[runtime_index]["run"]
    for runtime_package in (
        "libfontconfig1",
        "libopengl0",
        "libxcb-shape0",
    ):
        assert runtime_package in runtime_setup
    assert cache_index < prewarm_index < bioformats_index < viewer_index

    xcb_index = step_names.index("Verify native Qt xcb runtime")
    assert runtime_index < xcb_index < cache_index
    xcb_smoke = steps[xcb_index]["run"]
    assert "xvfb-run" in xcb_smoke
    assert "import cv2" in xcb_smoke
    assert "ViewerQtEnvironmentPolicy().apply_to(os.environ)" in xcb_smoke
    assert "QLibraryInfo.LibraryPath.PluginsPath" in xcb_smoke
    assert "QApplication([])" in xcb_smoke

    prewarm = steps[prewarm_index]["run"]
    assert "from polystore.imagej_runtime import FIJI_IMAGEJ_RUNTIME" in prewarm
    assert "FIJI_IMAGEJ_RUNTIME.initialize(" in prewarm
    assert 'getProperty("java.version")' in prewarm
    assert "FIJI_IMAGEJ_RUNTIME.shutdown(ij, sj)" in prewarm

    bioformats_smoke = steps[bioformats_index]["run"]
    assert "scripts/run_installed_tests.py" in bioformats_smoke
    assert "test_bioformats_imagexpress_synthetic.py" in bioformats_smoke
    assert "for smoke_attempt in 1 2 3" in bioformats_smoke


def test_native_macos_installer_uses_native_qt_smoke_harness():
    workflow = yaml.safe_load(INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8"))
    installer_job = workflow["jobs"]["desktop-installer-source-test"]
    assert "env" not in installer_job
    matrix = installer_job["strategy"]["matrix"]["include"]
    assert {
        "os": "macos-latest",
        "platform": "macos",
        "architecture": "arm64",
    } in matrix
    assert {
        "os": "macos-15-intel",
        "platform": "macos",
        "architecture": "x86_64",
    } in matrix
    steps = installer_job["steps"]
    architecture_step = next(
        step
        for step in steps
        if step.get("name") == "Verify macOS installer architecture"
    )
    assert "uname -m" in architecture_step["run"]
    candidate_step = next(
        step
        for step in steps
        if step.get("name") == "Build native installer candidate wheelhouse"
    )
    assert "scripts/stage_ci_candidate_version.py" in candidate_step["run"]
    assert "--dependency-source submodules" in candidate_step["run"]
    assert "--build-only" in candidate_step["run"]
    assert "GITHUB_RUN_ID" in candidate_step["run"]
    assert "RUNNER_TEMP" in candidate_step["run"]
    assert "wheelhouse=%s" in candidate_step["run"]
    smoke_step = next(
        step
        for step in steps
        if step.get("name") == "Execute and verify macOS installer"
    )
    assert "QT_QPA_PLATFORM" not in smoke_step.get("env", {})
    assert "scripts.smoke_installed_desktop" in smoke_step["run"]
    assert "scripts/smoke_staged_desktop_update.py" in smoke_step["run"]
    assert '--latest-version "$release_version"' in smoke_step["run"]
    assert "OPENHCS_MCP_INSTALLATION_POINTER" in smoke_step["run"]
    assert smoke_step["env"] == {
        "OPENHCS_INSTALLER_CANDIDATE_VERSION": "${{ steps.source_candidate.outputs.release_version }}",
        "OPENHCS_INSTALLER_CANDIDATE_WHEELHOUSE": "${{ steps.source_candidate.outputs.wheelhouse }}",
    }
    assert "steps.source_candidate.outputs" not in smoke_step["run"]
    assert "OPENHCS_INSTALLER_CANDIDATE_VERSION" in smoke_step["run"]
    assert "OPENHCS_INSTALLER_CANDIDATE_WHEELHOUSE" in smoke_step["run"]
    assert "import tkinter as tk" in smoke_step["run"]
    assert '"$managed_python" -I "$installed_worker" --help' in smoke_step["run"]
    assert "macOS staged updater did not switch environments" in smoke_step["run"]


def test_mcpb_python_ranges_match_the_ci_supported_boundary():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    wrapper = tomllib.loads((MCPB_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    manifest = json.loads((MCPB_ROOT / "manifest.json").read_text(encoding="utf-8"))
    ci_versions = _ci_python_versions()

    project_range = SpecifierSet(project["requires-python"])
    wrapper_range = SpecifierSet(wrapper["requires-python"])
    manifest_range = SpecifierSet(
        manifest["compatibility"]["runtimes"]["python"].replace(" ", ",")
    )
    assert project_range == wrapper_range
    assert wrapper_range == manifest_range
    assert all(version in project_range for version in ci_versions)
    assert all(version in wrapper_range for version in ci_versions)

    oldest = min(ci_versions)
    newest = max(ci_versions)
    previous_minor = Version(f"{oldest.major}.{oldest.minor - 1}")
    next_minor = Version(f"{newest.major}.{newest.minor + 1}")
    assert previous_minor not in wrapper_range
    assert next_minor not in wrapper_range
