"""Static contract tests for the native macOS installer sources."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest

from openhcs.desktop_installation import DESKTOP_INSTALL_PROFILE

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = REPOSITORY_ROOT / "packaging" / "installers"
MACOS_ROOT = INSTALLER_ROOT / "macos"
BOOTSTRAP_PATH = MACOS_ROOT / "install-openhcs.sh"
APPLESCRIPT_PATH = MACOS_ROOT / "Install-OpenHCS.applescript"
APP_SOURCE_PATH = MACOS_ROOT / "OpenHCSInstaller.swift"
BUILD_PATH = MACOS_ROOT / "build-installer.sh"
DMG_BUILD_PATH = MACOS_ROOT / "build-dmg.sh"
DMG_LIFECYCLE_PATH = MACOS_ROOT / "dmg-lifecycle.sh"
PUBLISH_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
INTEGRATION_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
)
WINDOW_SMOKE_PATH = REPOSITORY_ROOT / "scripts" / "smoke_macos_installer_window.sh"
WINDOW_PROBE_PATH = REPOSITORY_ROOT / "scripts" / "macos_installer_window_probe.swift"
MACOS_README_PATH = MACOS_ROOT / "README.md"
GETTING_STARTED_PATH = (
    REPOSITORY_ROOT / "docs" / "source" / "getting_started" / "getting_started.rst"
)


def _bootstrap() -> str:
    return BOOTSTRAP_PATH.read_text(encoding="utf-8")


def _profile_values() -> tuple[str, str, str]:
    profile = DESKTOP_INSTALL_PROFILE
    return (
        profile.python_version,
        profile.select("openhcs", "0.5.22").package_requirement,
        profile.uv_release.version,
    )


def _cancellation_function_block() -> str:
    source = _bootstrap()
    start = source.index("run_cancellable() {")
    end = source.index("\ntrap cleanup EXIT", start)
    return source[start:end]


def test_macos_installer_fails_closed_on_validated_shared_contract() -> None:
    source = _bootstrap()

    assert "plutil -extract" in source
    assert "entry_point=$(contract_value entry_point)" in source
    assert "openhcs.installer.v2" in source
    assert "'https://astral.sh/uv'" in source
    assert 'uv_installer_url="$uv_base_url/$uv_version/install.sh"' in source
    assert "==[A-Za-z0-9][A-Za-z0-9.*+!_-]*$" in source

    for value in _profile_values():
        assert value not in source


def test_macos_installer_uses_uv_for_python_and_pip_for_packages() -> None:
    source = _bootstrap()

    assert "UV_INSTALL_DIR" in source
    assert "UV_PYTHON_INSTALL_DIR" in source
    assert "UV_NO_MODIFY_PATH=1" in source
    assert "UV_NO_CONFIG=1" in source
    assert "PIP_CONFIG_FILE=/dev/null" in source
    assert "unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL" in source
    for command in ("python install", "venv"):
        assert f'run_cancellable "$uv_executable" --no-config {command}' in source
    assert '--python "$python_version" --seed "$new_environment"' in source
    assert 'run_cancellable "$environment_python" -m pip install' in source
    assert "binary_only_packages=$(contract_value binary_only_packages)" in source
    assert '--only-binary "$binary_only_packages"' in source
    assert 'run_cancellable "$environment_python" -m pip check' in source
    assert "--prerelease" not in source
    assert "Preparing the execution catalog" not in source
    assert "--prepare-capabilities" not in source
    assert "sudo" not in source
    assert "/usr/bin/python" not in source
    assert "/usr/bin/sw_vers -productVersion" in source
    assert "/usr/bin/uname -m" in source


def test_macos_update_switches_only_after_verification() -> None:
    source = _bootstrap()

    verify_position = source.index("-m pip check")
    entry_position = source.index('if [[ ! -x "$installed_entry" ]]')
    state_switch_position = source.index("-m openhcs.desktop_deployment_cli")

    assert verify_position < entry_position < state_switch_position
    assert "new_environment" in source
    assert "trap cleanup EXIT" in source
    assert "trap cancel_install HUP INT TERM" in source
    assert "run_cancellable()" in source
    assert "child_pid=$!" in source
    assert "active_child_pid=$child_pid" in source
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
                "install_cancellation_requested=false",
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


def test_macos_cancellation_polls_liveness_then_reaps_child_once() -> None:
    source = _cancellation_function_block()
    termination_start = source.index("terminate_active_child() {")
    cancellation_start = source.index("cancel_install() {")
    termination = source[termination_start:cancellation_start]
    cancellation = source[cancellation_start:]

    assert 'wait "$child_pid"' not in termination
    assert 'wait "$child_pid"' not in cancellation
    assert 'while child_is_running "$child_pid"; do' in source
    assert "termination_grace_seconds=2" in termination
    assert "termination_deadline" in termination
    assert "{1..20}" not in termination
    assert source.count('wait "$child_pid"') == 1
    assert "install_cancellation_requested=true" in cancellation


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
    assert "contract_path=${1:?" in build
    assert "CFBundleShortVersionString" not in build
    assert "CFBundleVersion" not in build
    assert "Contents/Resources/installer_contract.json" in build
    assert "Contents/Resources/install-openhcs.sh" in build
    assert "Contents/Resources/OpenHCS.icns" in build
    assert "<key>CFBundleIconFile</key><string>OpenHCS.icns</string>" in build


def test_macos_release_is_one_verified_disk_image() -> None:
    workflow = PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")
    dmg_builder = DMG_BUILD_PATH.read_text(encoding="utf-8")
    macos_job = workflow[
        workflow.index("  build-macos-installer:") : workflow.index(
            "  build-and-publish:"
        )
    ]

    assert "OpenHCS-macOS-Installer.dmg" in macos_job
    assert "packaging/installers/macos/build-dmg.sh" in macos_job
    assert "hdiutil create" in dmg_builder
    assert '-volname "OpenHCS Installer"' in dmg_builder
    assert "-srcfolder" not in dmg_builder
    assert '-size "${image_size_kib}k"' in dmg_builder
    assert "-fs APFS" in dmg_builder
    assert "-type UDIF" in dmg_builder
    assert "-format UDRW" not in dmg_builder
    assert 'ditto "$installer_app" "$mount_point/OpenHCS Installer.app"' in dmg_builder
    assert "-format UDZO" in dmg_builder
    assert "hdiutil verify" in dmg_builder
    assert "path: OpenHCS-macOS-Installer.dmg" in macos_job
    assert "OpenHCS-macOS-Installer.zip" not in macos_job
    assert "ditto -c -k" not in macos_job


def test_macos_disk_image_cleanup_retains_exact_device_authority() -> None:
    builder = DMG_BUILD_PATH.read_text(encoding="utf-8")
    lifecycle = DMG_LIFECYCLE_PATH.read_text(encoding="utf-8")
    integration = (
        REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
    ).read_text(encoding="utf-8")

    assert 'source "$script_directory/dmg-lifecycle.sh"' in builder
    assert "openhcs_attach_writable_disk_image" in builder
    assert "openhcs_attach_readonly_disk_image" in integration
    assert "/usr/bin/hdiutil attach" in lifecycle
    assert "-plist" in lifecycle
    assert "system-entities.0.dev-entry" in lifecycle
    assert 'diskutil info -plist "$mount_point"' in lifecycle
    assert "plutil -extract DeviceNode" in lifecycle
    assert 'diskutil unmount "$mounted_volume"' in lifecycle
    assert "DeviceIdentifier" not in lifecycle
    assert "/bin/sync" in lifecycle
    assert "local detach_attempt_limit=10" in lifecycle
    assert "attempt <= detach_attempt_limit" in lifecycle
    assert "attempt < detach_attempt_limit" in lifecycle
    assert '/usr/bin/hdiutil detach "$mounted_device"' in lifecycle
    assert '/usr/sbin/diskutil info "$mounted_device"' in lifecycle
    assert '/usr/bin/hdiutil detach -force "$mounted_device"' in lifecycle
    assert 'hdiutil detach "$mount_point"' not in builder
    assert 'hdiutil detach "$mount_point"' not in integration


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


def test_macos_installer_ci_drives_and_captures_the_shipping_app() -> None:
    workflow = INTEGRATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    smoke = WINDOW_SMOKE_PATH.read_text(encoding="utf-8")
    probe = WINDOW_PROBE_PATH.read_text(encoding="utf-8")
    macos_step = workflow[
        workflow.index(
            "      - name: Execute and verify macOS installer"
        ) : workflow.index("      - name: Show macOS installer log on failure")
    ]

    assert 'installer_executable="$installer_app/Contents/MacOS/OpenHCSInstaller"' in (
        smoke
    )
    assert '"$installer_executable" >"$installer_stdout"' in smoke
    assert "press-primary" in smoke
    assert "wait_for_installer_log" in smoke
    assert "installer-progress" in smoke
    assert "installer-finished" in smoke
    assert "Installation completed successfully." in smoke
    assert '/usr/sbin/screencapture -x -l "$window_id"' in smoke
    assert '/bin/kill -TERM "$installer_pid"' in smoke
    assert "CGWindowListCopyWindowInfo(" in probe
    assert "ownerPID == processIdentifier" in probe
    assert "title == expectedTitle" in probe
    assert "matchingWindows.count == 1" in probe
    assert "case pressPrimary" in probe
    assert "keyDown.postToPid(processIdentifier)" in probe

    invocation = "scripts/smoke_macos_installer_window.sh"
    assert invocation in macos_step
    assert macos_step.index('export HOME="$smoke_home"') < macos_step.index(invocation)
    assert '"$installer_app/Contents/Resources/install-openhcs.sh"' not in macos_step
    assert macos_step.index(invocation) < macos_step.index(
        "python -m scripts.smoke_installed_desktop"
    )
    assert "native-installer-ui-${{ matrix.platform }}" in workflow


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
    redirect_position = source.index('exec > >(/usr/bin/tee -a "$log_path") 2>&1')
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


def test_macos_app_streams_real_worker_output_into_a_scrollable_transcript() -> None:
    source = APP_SOURCE_PATH.read_text(encoding="utf-8")
    bootstrap = _bootstrap()

    assert "private let transcriptScrollView = NSScrollView()" in source
    assert "private let transcriptTextView = NSTextView(" in source
    assert "transcriptScrollView.hasVerticalScroller = true" in source
    assert "transcriptTextView.isEditable = false" in source
    assert "transcriptTextView.isSelectable = true" in source
    assert "process.standardOutput = pipe" in source
    assert "process.standardError = pipe" in source
    assert "self?.appendWorkerOutput(data)" in source
    assert "pendingTranscriptOutput.append(data)" in source
    assert "transcriptTextView.textStorage?.append(" in source
    assert "transcriptTextView.scrollToEndOfDocument(nil)" in source
    assert (
        "guard let status = workerTerminationStatus, workerOutputReachedEnd" in source
    )
    assert "let text = transcriptTextView.string" in source
    assert "private var workerOutput = Data()" not in source
    assert source.count("appendWorkerOutput(") == 2

    log_projection = bootstrap.index("write_installer_state log-path")
    tee_redirect = bootstrap.index('exec > >(/usr/bin/tee -a "$log_path") 2>&1')
    first_install_output = bootstrap.index("Starting %s installation")
    assert log_projection < tee_redirect < first_install_output
    assert 'exec >>"$log_path" 2>&1' not in bootstrap


@pytest.mark.skipif(
    os.name == "nt" or not Path("/usr/bin/tee").is_file(),
    reason="the macOS transcript redirect requires POSIX Bash and /usr/bin/tee",
)
def test_macos_transcript_redirect_duplicates_the_real_combined_stream(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "installer.log"
    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                'log_path=$1; exec > >(/usr/bin/tee -a "$log_path") 2>&1; '
                "printf 'worker stdout\\n'; printf 'worker stderr\\n' >&2"
            ),
            "transcript-redirect-harness",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout == "worker stdout\nworker stderr\n"
    assert log_path.read_text(encoding="utf-8") == completed.stdout


def test_unsigned_macos_docs_explain_the_current_open_anyway_path() -> None:
    for path in (MACOS_README_PATH, GETTING_STARTED_PATH):
        documentation = path.read_text(encoding="utf-8")

        assert "System Settings" in documentation
        assert "Privacy & Security" in documentation
        assert "Open Anyway" in documentation
        assert "unsigned" in documentation
        assert "notarised" in documentation


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
    assert 'export PIP_FIND_LINKS="$UV_FIND_LINKS"' in macos_smoke
    assert 'codex_config="$HOME/.codex/config.toml"' in macos_smoke
    assert "stable_launcher=" in macos_smoke
    assert "['mcp']" in macos_smoke
