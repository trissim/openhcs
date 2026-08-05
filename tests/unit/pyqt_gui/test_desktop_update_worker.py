from __future__ import annotations

import json
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

from openhcs.pyqt_gui.services import desktop_update_worker
from openhcs.resources.brand import BrandAsset, brand_asset_path

BACKGROUND_LAUNCH_SPEC = desktop_update_worker.ResolvedProcessLaunchSpec(
    creationflags=73,
    start_new_session=False,
)
DETACHED_LAUNCH_SPEC = desktop_update_worker.ResolvedProcessLaunchSpec(
    creationflags=91,
    start_new_session=False,
)
WORKER_LAUNCH_ARGUMENTS = [
    "--background-creationflags=73",
    "--detached-creationflags=91",
]


class _ProgressProbe:
    def __init__(
        self,
        *,
        action: desktop_update_worker.DesktopUpdateProgressAction = (
            desktop_update_worker.DesktopUpdateProgressAction.EXIT
        ),
    ) -> None:
        self.action = action
        self.phases = []
        self.outputs = []
        self.failures = []
        self.completed = False

    def phase(self, phase) -> None:
        self.phases.append(phase)

    def output(self, message: str) -> None:
        self.outputs.append(message)

    def failure(self, message: str):
        self.failures.append(message)
        return self.action

    def complete(self) -> None:
        self.completed = True

    def run(self, operation) -> int:
        return operation()


class _StreamingProcess:
    def __init__(self, returncode: int, output: str) -> None:
        self._returncode = returncode
        self.stdout = io.StringIO(output)

    def wait(self) -> int:
        return self._returncode


def _progress_arguments(tmp_path: Path) -> list[str]:
    return [
        "--progress-theme-file",
        str(tmp_path / "desktop-update-theme.json"),
        "--progress-brand-file",
        str(tmp_path / "desktop-update-brand.png"),
    ]


def test_worker_reports_bounded_install_failure(monkeypatch) -> None:
    progress = _ProgressProbe()
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _StreamingProcess(7, "failure detail\n"),
    )

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.1",
        verification_executable="/target/venv/python",
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert error == "OpenHCS update failed with exit code 7.\n\nfailure detail"
    assert progress.phases == [desktop_update_worker.DesktopUpdatePhase.INSTALLING]
    assert progress.outputs == ["failure detail"]


def test_worker_verifies_with_target_environment_interpreter(monkeypatch) -> None:
    calls = []
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "resolved packages\ninstalled OpenHCS\n"),
            _StreamingProcess(0, "version verified\n"),
        )
    )

    def _popen(command, **kwargs):
        calls.append((command, kwargs))
        return next(processes)

    monkeypatch.setattr(desktop_update_worker.subprocess, "Popen", _popen)

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.1",
        verification_executable="/target/venv/python",
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert error is None
    assert calls[0][0] == ["uv", "pip", "install"]
    assert calls[1][0][0] == "/target/venv/python"
    assert calls[1][0][-1] == "0.7.1"
    assert calls[0][1]["creationflags"] == 73
    assert calls[1][1]["creationflags"] == 73
    assert progress.phases == [
        desktop_update_worker.DesktopUpdatePhase.INSTALLING,
        desktop_update_worker.DesktopUpdatePhase.VERIFYING,
    ]
    assert progress.outputs == [
        "resolved packages",
        "installed OpenHCS",
        "version verified",
    ]


def test_worker_refreshes_installer_managed_desktop_after_verification(
    monkeypatch,
) -> None:
    calls = []
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "installed OpenHCS\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(0, '{"platform": "windows"}\n'),
        )
    )

    def _popen(command, **kwargs):
        calls.append((command, kwargs))
        return next(processes)

    monkeypatch.setattr(desktop_update_worker.subprocess, "Popen", _popen)

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.15",
        verification_executable="C:/OpenHCS/env/python.exe",
        installation_pointer="C:/OpenHCS/Launch-OpenHCS.ps1",
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert error is None
    assert calls[2][0] == [
        "C:/OpenHCS/env/python.exe",
        "-I",
        "-m",
        "openhcs.desktop_deployment",
        "--installation-pointer=C:/OpenHCS/Launch-OpenHCS.ps1",
        "--json",
    ]
    assert progress.phases == [
        desktop_update_worker.DesktopUpdatePhase.INSTALLING,
        desktop_update_worker.DesktopUpdatePhase.VERIFYING,
        desktop_update_worker.DesktopUpdatePhase.REFRESHING_DESKTOP,
    ]


def test_worker_reports_desktop_refresh_failure_with_repair_path(monkeypatch) -> None:
    progress = _ProgressProbe()
    processes = iter(
        (
            _StreamingProcess(0, "installed OpenHCS\n"),
            _StreamingProcess(0, "version verified\n"),
            _StreamingProcess(1, "shortcut publication failed\n"),
        )
    )
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(processes),
    )

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.15",
        verification_executable="/OpenHCS/env/python",
        installation_pointer="/OpenHCS/current",
        launch_spec=BACKGROUND_LAUNCH_SPEC,
        progress=progress,
    )

    assert error is not None
    assert "Re-run the official installer" in error
    assert "shortcut publication failed" in error


def test_worker_restarts_prior_entry_with_saved_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    launched = []
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )

    desktop_update_worker._restart(
        "openhcs",
        ["--log-level", "INFO"],
        session_directory=tmp_path,
        restore_option="--restore-update-session",
        launch_spec=DETACHED_LAUNCH_SPEC,
    )

    assert launched[0][0] == [
        "openhcs",
        "--log-level",
        "INFO",
        "--restore-update-session",
        str(tmp_path),
    ]
    assert launched[0][1]["close_fds"] is True
    assert launched[0][1]["stdin"] is subprocess.DEVNULL
    assert launched[0][1]["stdout"] is subprocess.DEVNULL
    assert launched[0][1]["stderr"] is subprocess.DEVNULL
    assert launched[0][1]["creationflags"] == 91


def test_worker_relaunches_and_preserves_session_after_update_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    progress = _ProgressProbe(
        action=desktop_update_worker.DesktopUpdateProgressAction.REOPEN
    )
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: progress,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda pid: calls.append(("wait", pid)) or True,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: "network unavailable",
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda executable,
        arguments,
        *,
        session_directory,
        restore_option,
        launch_spec: calls.append(
            ("restart", executable, arguments, session_directory, launch_spec)
        ),
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(
        [
            "--parent-pid",
            "42",
            "--session-directory",
            str(tmp_path),
            "--update-executable",
            "uv",
            "--update-argument=--no-config",
            "--restart-executable",
            "openhcs",
            "--restart-argument=--log-level",
            "--restart-argument=INFO",
            "--expected-version",
            "0.7.1",
            "--verification-executable",
            "/target/venv/python",
            "--error-file",
            str(error_file),
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        ]
    )

    assert result == 1
    assert error_file.read_text(encoding="utf-8") == "network unavailable"
    assert progress.failures == ["network unavailable"]
    assert calls == [
        ("wait", 42),
        (
            "restart",
            "openhcs",
            ["--log-level", "INFO"],
            tmp_path,
            DETACHED_LAUNCH_SPEC,
        ),
    ]


def test_worker_cancels_before_update_when_parent_does_not_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe()
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: progress,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda _pid: False,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("update must not start while parent is alive")
        ),
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(
        [
            "--parent-pid",
            "42",
            "--session-directory",
            str(tmp_path),
            "--update-executable",
            "uv",
            "--update-argument=--no-config",
            "--restart-executable",
            "openhcs",
            "--restart-argument=--log-level",
            "--expected-version",
            "0.7.1",
            "--verification-executable",
            "/target/venv/python",
            "--error-file",
            str(error_file),
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        ]
    )

    assert result == 2
    assert tmp_path.exists()
    assert "cancelled before modifying" in error_file.read_text(encoding="utf-8")
    assert progress.failures == [error_file.read_text(encoding="utf-8")]


def test_worker_fails_closed_and_reopens_when_progress_window_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    monkeypatch.setattr(
        desktop_update_worker.DesktopUpdateProgressWindow,
        "create",
        lambda **_kwargs: (_ for _ in ()).throw(
            desktop_update_worker.DesktopUpdateProgressUnavailable(
                "progress unavailable; environment not modified"
            )
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda pid: calls.append(("wait", pid)) or True,
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_run_update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("update must not run without a visible progress surface")
        ),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda executable,
        arguments,
        *,
        session_directory,
        restore_option,
        launch_spec: calls.append(("restart", session_directory)) or None,
    )
    error_file = tmp_path / "update-error.txt"

    result = desktop_update_worker.main(
        [
            "--parent-pid",
            "42",
            "--session-directory",
            str(tmp_path),
            "--update-executable",
            "uv",
            "--restart-executable",
            "openhcs",
            "--expected-version",
            "0.7.1",
            "--verification-executable",
            "/target/venv/python",
            "--error-file",
            str(error_file),
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        ]
    )

    assert result == 3
    assert error_file.read_text(encoding="utf-8") == (
        "progress unavailable; environment not modified"
    )
    assert calls == [("wait", 42), ("restart", tmp_path)]


def test_unexpected_orchestration_exception_uses_visible_recovery_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress = _ProgressProbe(
        action=desktop_update_worker.DesktopUpdateProgressAction.REOPEN
    )
    error_file = tmp_path / "update-error.txt"
    arguments = desktop_update_worker.parse_arguments(
        [
            "--parent-pid",
            "42",
            "--session-directory",
            str(tmp_path),
            "--update-executable",
            "uv",
            "--restart-executable",
            "openhcs",
            "--expected-version",
            "0.7.1",
            "--verification-executable",
            "/target/venv/python",
            "--error-file",
            str(error_file),
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        ]
    )
    restarts = []
    monkeypatch.setattr(
        desktop_update_worker,
        "_wait_for_parent_exit",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("wait boundary exploded")),
    )
    monkeypatch.setattr(
        desktop_update_worker,
        "_restart",
        lambda *_args, **_kwargs: restarts.append("reopened") or None,
    )

    result = desktop_update_worker._perform_update(
        arguments,
        progress=progress,
        background_launch_spec=BACKGROUND_LAUNCH_SPEC,
        detached_launch_spec=DETACHED_LAUNCH_SPEC,
    )

    expected = "OpenHCS could not complete the update: wait boundary exploded"
    assert result == 1
    assert progress.failures == [expected]
    assert error_file.read_text(encoding="utf-8") == expected
    assert restarts == ["reopened"]


def test_parser_preserves_leading_dash_forwarded_arguments(tmp_path: Path) -> None:
    arguments = desktop_update_worker.parse_arguments(
        [
            "--parent-pid",
            "42",
            "--session-directory",
            "/tmp/session",
            "--update-executable",
            "uv",
            "--update-argument=--no-config",
            "--update-argument=--upgrade",
            "--restart-executable",
            "openhcs",
            "--restart-argument=--log-level",
            "--restart-argument=DEBUG",
            "--expected-version",
            "0.7.1",
            "--verification-executable",
            "/target/venv/python",
            "--error-file",
            "/tmp/session/update-error.txt",
            "--restore-option=--restore-update-session",
            *_progress_arguments(tmp_path),
            *WORKER_LAUNCH_ARGUMENTS,
        ]
    )

    assert arguments.update_argument == ["--no-config", "--upgrade"]
    assert arguments.restart_argument == ["--log-level", "DEBUG"]
    assert arguments.verification_executable == "/target/venv/python"
    assert arguments.background_creationflags == 73
    assert arguments.detached_creationflags == 91
    assert arguments.progress_theme_file == tmp_path / "desktop-update-theme.json"
    assert arguments.progress_brand_file == tmp_path / "desktop-update-brand.png"


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
    reason="Tk progress window requires a display",
)
def test_worker_process_waits_updates_restarts_and_restores_session(
    tmp_path: Path,
) -> None:
    session_directory = tmp_path / "pending"
    session_directory.mkdir()
    (session_directory / "session.py").write_text(
        "canonical session source",
        encoding="utf-8",
    )
    (session_directory / "objectstate-history.objectstate").write_text(
        "canonical history",
        encoding="utf-8",
    )
    desktop_update_worker.DesktopUpdateProgressTheme(
        window_bg="#2b2b2b",
        panel_bg="#1e1e1e",
        text_primary="#ffffff",
        text_secondary="#cccccc",
        text_accent="#00aaff",
        border_color="#555555",
        button_bg="#404040",
        button_text="#ffffff",
        error_color="#ff0000",
        progress_color="#0078d4",
    ).write(session_directory / "desktop-update-theme.json")
    shutil.copyfile(
        brand_asset_path(BrandAsset.ICON_RASTER),
        session_directory / "desktop-update-brand.png",
    )
    update_marker = tmp_path / "updated.txt"
    restore_marker = tmp_path / "restored.json"
    restore_script = tmp_path / "restore.py"
    restore_script.write_text(
        """
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from objectstate.object_state import ObjectStateRegistry
from openhcs.pyqt_gui.services.desktop_update import DesktopUpdateSession

parser = argparse.ArgumentParser()
parser.add_argument("marker", type=Path)
parser.add_argument("--restore-update-session", required=True, type=Path)
args = parser.parse_args()
calls = []
ObjectStateRegistry.load_history_from_file = classmethod(
    lambda cls, path: calls.append(["history", Path(path).read_text(encoding="utf-8")])
)
plate_manager = SimpleNamespace(
    apply_code_document_source=lambda source: calls.append(["source", source]),
    update_item_list=lambda: calls.append(["refresh", None]),
)
main_window = SimpleNamespace(
    embedded_widgets=SimpleNamespace(
        require_plate_manager=lambda: plate_manager,
    ),
    time_travel_widget=SimpleNamespace(
        refresh=lambda: calls.append(["history-ui", None]),
    ),
)
error = DesktopUpdateSession(args.restore_update_session).restore(main_window)
args.marker.write_text(
    json.dumps({"calls": calls, "error": error}),
    encoding="utf-8",
)
""".strip(),
        encoding="utf-8",
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.3)"],
    )
    parent_reaper = threading.Thread(target=parent.wait)
    parent_reaper.start()
    source_root = Path(__file__).resolve().parents[3]
    external_roots = (
        "external/ObjectState/src",
        "external/python-introspect/src",
        "external/metaclass-registry/src",
        "external/arraybridge/src",
        "external/pycodify/src",
        "external/PolyStore/src",
        "external/pyqt-reactive/src",
        "external/zmqruntime/src",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), *(str(source_root / path) for path in external_roots))
    )
    update_code = (
        "from pathlib import Path;"
        f"Path({str(update_marker)!r}).write_text('updated', encoding='utf-8')"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "openhcs.pyqt_gui.services.desktop_update_worker",
            "--parent-pid",
            str(parent.pid),
            "--session-directory",
            str(session_directory),
            "--update-executable",
            sys.executable,
            "--update-argument=-c",
            f"--update-argument={update_code}",
            "--restart-executable",
            sys.executable,
            f"--restart-argument={restore_script}",
            f"--restart-argument={restore_marker}",
            "--expected-version",
            distribution_version("openhcs"),
            "--verification-executable",
            sys.executable,
            "--error-file",
            str(session_directory / "update-error.txt"),
            "--restore-option=--restore-update-session",
            *_progress_arguments(session_directory),
            "--background-creationflags=0",
            "--detached-creationflags=0",
            "--detached-start-new-session",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    parent_reaper.join(timeout=2)
    deadline = time.monotonic() + 10
    while not restore_marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert completed.returncode == 0, completed.stderr
    assert update_marker.read_text(encoding="utf-8") == "updated"
    restored = json.loads(restore_marker.read_text(encoding="utf-8"))
    assert restored == {
        "calls": [
            ["source", "canonical session source"],
            ["history", "canonical history"],
            ["history-ui", None],
            ["refresh", None],
        ],
        "error": None,
    }
    assert not session_directory.exists()
