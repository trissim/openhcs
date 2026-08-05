"""Static contract tests for the native macOS installer sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPOSITORY_ROOT / "packaging" / "installers"
MACOS_ROOT = INSTALLER_ROOT / "macos"
BOOTSTRAP_PATH = MACOS_ROOT / "install-openhcs.sh"
APPLESCRIPT_PATH = MACOS_ROOT / "Install-OpenHCS.applescript"
APP_SOURCE_PATH = MACOS_ROOT / "OpenHCSInstaller.swift"
BUILD_PATH = MACOS_ROOT / "build-installer.sh"
CONTRACT_PATH = INSTALLER_ROOT / "installer_contract.json"
PUBLISH_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"


def _bootstrap() -> str:
    return BOOTSTRAP_PATH.read_text(encoding="utf-8")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _cancellation_function_block() -> str:
    source = _bootstrap()
    start = source.index("run_cancellable() {")
    end = source.index("\ntrap cleanup EXIT", start)
    return source[start:end]


def test_macos_installer_fails_closed_on_validated_shared_contract() -> None:
    source = _bootstrap()
    contract = _contract()

    assert "plutil -extract" in source
    assert "entry_point=$(contract_value entry_point)" in source
    assert "openhcs.installer.v2" in source
    assert "'https://astral.sh/uv'" in source
    assert 'uv_installer_url="$uv_base_url/$uv_version/install.sh"' in source

    for value in (
        contract["python_version"],
        contract["package_requirement"],
        contract["uv_release"]["version"],
    ):
        assert value not in source


def test_macos_installer_uses_uv_without_system_python_or_admin() -> None:
    source = _bootstrap()

    assert "UV_INSTALL_DIR" in source
    assert "UV_PYTHON_INSTALL_DIR" in source
    assert "UV_NO_MODIFY_PATH=1" in source
    assert "UV_NO_CONFIG=1" in source
    for command in ("python install", "venv", "pip install", "pip check"):
        assert f'run_cancellable "$uv_executable" --no-config {command}' in source
    assert '--python "$python_version" "$new_environment"' in source
    assert '--python "$environment_python"' in source
    assert "--prerelease if-necessary-or-explicit" in source
    assert "sudo" not in source
    assert "/usr/bin/python" not in source


def test_macos_update_switches_only_after_verification() -> None:
    source = _bootstrap()

    verify_position = source.index("--no-config pip check")
    entry_position = source.index('if [[ ! -x "$installed_entry" ]]')
    state_switch_position = source.index("-m openhcs.desktop_deployment_cli")

    assert verify_position < entry_position < state_switch_position
    assert "new_environment" in source
    assert "trap cleanup EXIT" in source
    assert "trap cancel_install HUP INT TERM" in source
    assert "run_cancellable()" in source
    assert "active_child_pid=$!" in source
    assert "/bin/kill -TERM" in source
    assert 'export OPENHCS_UV_EXECUTABLE="$uv_executable"' in source
    assert '--installation-pointer="$current_environment"' in source
    assert 'readlink "$current_environment"' in source


@pytest.mark.skipif(
    os.name == "nt",
    reason="the executable cancellation harness requires POSIX signals",
)
def test_macos_cancellation_escalates_and_reaps_a_term_ignoring_child(
    tmp_path: Path,
) -> None:
    cleanup_marker = tmp_path / "cleanup-finished"
    child_pid_path = tmp_path / "child.pid"
    harness_path = tmp_path / "cancel-harness.sh"
    harness_path.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "set -uo pipefail",
                f"cleanup_marker={shlex.quote(str(cleanup_marker))}",
                f"child_pid_path={shlex.quote(str(child_pid_path))}",
                "active_child_pid=",
                'cleanup() { /usr/bin/touch "$cleanup_marker"; }',
                _cancellation_function_block(),
                "trap cleanup EXIT",
                "trap cancel_install HUP INT TERM",
                (
                    "run_cancellable /bin/bash -c "
                    '\'trap "" TERM; printf "%s\\n" "$$" > "$1"; '
                    "while :; do /bin/sleep 1; done' "
                    'cancellable-child "$child_pid_path"'
                ),
                "",
            )
        ),
        encoding="utf-8",
    )

    process = subprocess.Popen(
        ["/bin/bash", str(harness_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 3
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid_path.exists(), "cancellable child did not start"
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        process.send_signal(signal.SIGTERM)
        return_code = process.wait(timeout=5)

        assert return_code == 130
        assert cleanup_marker.is_file()
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("TERM-ignoring child survived KILL escalation")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_macos_installer_builds_a_universal_native_app_with_embedded_contract() -> None:
    app_source = APP_SOURCE_PATH.read_text(encoding="utf-8")
    build = BUILD_PATH.read_text(encoding="utf-8")

    assert not APPLESCRIPT_PATH.exists()
    assert "NSApplication.shared" in app_source
    assert "Process()" in app_source
    assert "process.run()" in app_source
    assert "application.run()" in app_source
    assert "administrator privileges" not in app_source
    assert "swiftc" in build
    assert "x86_64-apple-macosx12.0" not in build
    assert '"$architecture-apple-macosx12.0"' in build
    assert "for architecture in x86_64 arm64" in build
    assert "lipo -create" in build
    assert "CFBundleShortVersionString" not in build
    assert "CFBundleVersion" not in build
    assert "Contents/Resources/installer_contract.json" in build
    assert "Contents/Resources/install-openhcs.sh" in build
    assert "Contents/Resources/OpenHCS.icns" in build
    assert "<key>CFBundleIconFile</key><string>OpenHCS.icns</string>" in build


def test_macos_release_is_one_verified_disk_image() -> None:
    workflow = PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")
    dmg_builder = (
        REPOSITORY_ROOT / "packaging" / "installers" / "macos" / "build-dmg.sh"
    ).read_text(encoding="utf-8")
    macos_job = workflow[
        workflow.index("  build-macos-installer:") : workflow.index(
            "  build-and-publish:"
        )
    ]

    assert "OpenHCS-macOS-Installer.dmg" in macos_job
    assert "packaging/installers/macos/build-dmg.sh" in macos_job
    assert "hdiutil create" in dmg_builder
    assert '-volname "OpenHCS Installer"' in dmg_builder
    assert "-format UDZO" in dmg_builder
    assert "hdiutil verify" in dmg_builder
    assert "path: OpenHCS-macOS-Installer.dmg" in macos_job
    assert "OpenHCS-macOS-Installer.zip" not in macos_job
    assert "ditto -c -k" not in macos_job


def test_macos_app_has_responsive_welcome_progress_and_finish_states() -> None:
    source = APP_SOURCE_PATH.read_text(encoding="utf-8")

    for state in (
        "case .welcome:",
        "case .installing:",
        "case .cancelling:",
        "case .finished:",
        "case .failed:",
        "case .cancelled:",
    ):
        assert state in source

    assert 'primaryButton.title = "Continue"' in source
    assert 'primaryButton.title = "Finish"' in source
    assert "progressIndicator.startAnimation" in source
    assert "process.terminationHandler" in source
    assert "Timer(timeInterval: 0.25" in source
    assert "worker.terminate()" in source
    assert source.index("worker = process") < source.index("try process.run()")
    assert 'installerStateValue(named: "launcher-path") != nil' in source
    assert "NSWorkspace.shared.open" in source
    assert "Terminal" in source
    assert "do shell script" not in source


def test_macos_shell_owns_live_progress_log_and_launcher_projection() -> None:
    source = _bootstrap()

    assert "OPENHCS_INSTALLER_STATE_DIRECTORY" in source
    assert "write_installer_state()" in source
    assert "report_progress()" in source
    assert "write_installer_state log-path" in source
    assert "write_installer_state launcher-path" in source
    assert "write_installer_state agent-registration-status" in source
    assert "report_progress 'Installation complete.'" in source
    touch_position = source.index('/usr/bin/touch "$log_path"')
    regular_file_position = source.index('if [[ ! -f "$log_path" ]]')
    projection_position = source.index("write_installer_state log-path")
    redirect_position = source.index('exec >>"$log_path"')
    assert (
        touch_position < regular_file_position < projection_position < redirect_position
    )
    assert 'if [[ -L "$log_path" ]]' in source
    assert '"$environment_python" -I -m openhcs.desktop_deployment_cli' in source
    assert '--installation-pointer="$current_environment"' in source
    assert "openhcs.resources.brand" not in source
    assert "new_launcher_app" not in source
    assert "CFBundleIconFile" not in source

    app_source = APP_SOURCE_PATH.read_text(encoding="utf-8")
    assert 'installerStateValue(named: "progress")' in app_source
    assert 'installerStateValue(named: "log-path")' in app_source
    assert 'installerStateValue(named: "launcher-path")' in app_source
    assert "installerLogURL()" in app_source
    assert ".isRegularFileKey" in app_source
    assert "values.isRegularFile == true" in app_source
    assert "values.isSymbolicLink != true" in app_source
    assert "activateFileViewerSelecting([logURL])" in app_source
    assert "Library/Logs" not in app_source
    assert "Library/Application Support" not in app_source


def test_macos_installer_registers_agent_clients_through_stable_launcher() -> None:
    source = _bootstrap()
    app_source = APP_SOURCE_PATH.read_text(encoding="utf-8")
    workflow = (
        REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
    ).read_text(encoding="utf-8")

    assert "OPENHCS_INSTALLER_REGISTER_MCP_CLIENTS" in source
    assert '"$new_environment/bin/openhcs-mcp-register"' in source
    assert '--command "$current_environment/launch-openhcs.sh"' in source
    assert "--args-json '[\"mcp\"]'" in source
    assert "--register codex" in source
    assert "--register-detected" in source
    assert "OPENHCS_UV_EXECUTABLE" in source
    assert "openhcs.desktop_deployment_cli" in source
    assert "agent-registration.json" in source
    assert "agent-registration-status connected" in source
    assert "agent-registration-status warning" in source
    assert "agent-registration-summary" in source
    assert 'result["display_name"]' in source
    assert 'json.load(open(sys.argv[1]))["ok"]' in source
    assert '"$registration_status" -ne 0' in source
    assert '"$registration_ok" != true' in source

    assert "connectAgentsCheckbox" in app_source
    assert "Connect OpenHCS to ChatGPT, Codex, and local AI agent apps" in app_source
    assert "connectAgentsCheckbox.state = .on" in app_source
    assert 'environment["OPENHCS_INSTALLER_REGISTER_MCP_CLIENTS"]' in app_source
    assert 'installerStateValue(named: "agent-registration-status")' in app_source
    assert 'installerStateValue(named: "agent-registration-summary")' in app_source
    assert "Restart ChatGPT desktop, Codex, and other listed apps" in app_source

    macos_smoke = workflow[
        workflow.index(
            "      - name: Execute and verify macOS installer"
        ) : workflow.index("      - name: Show macOS installer log on failure")
    ]
    assert 'codex_config="$HOME/.codex/config.toml"' in macos_smoke
    assert "stable_launcher=" in macos_smoke
    assert "['mcp']" in macos_smoke
