from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import SimpleNamespace

from openhcs.pyqt_gui.services import desktop_update_worker


def test_worker_reports_bounded_install_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        desktop_update_worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stderr="failure detail",
            stdout="",
        ),
    )

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.1",
        verification_executable="/target/venv/python",
    )

    assert error == "OpenHCS update failed with exit code 7.\n\nfailure detail"


def test_worker_verifies_with_target_environment_interpreter(monkeypatch) -> None:
    calls = []

    def _run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(desktop_update_worker.subprocess, "run", _run)

    error = desktop_update_worker._run_update(
        "uv",
        ["pip", "install"],
        expected_version="0.7.1",
        verification_executable="/target/venv/python",
    )

    assert error is None
    assert calls[0] == ["uv", "pip", "install"]
    assert calls[1][0] == "/target/venv/python"
    assert calls[1][-1] == "0.7.1"


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


def test_worker_relaunches_and_preserves_session_after_update_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
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
        lambda executable, arguments, *, session_directory, restore_option: calls.append(
            ("restart", executable, arguments, session_directory)
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
        ]
    )

    assert result == 1
    assert error_file.read_text(encoding="utf-8") == "network unavailable"
    assert calls == [
        ("wait", 42),
        ("restart", "openhcs", ["--log-level", "INFO"], tmp_path),
    ]


def test_worker_cancels_before_update_when_parent_does_not_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
        ]
    )

    assert result == 2
    assert not tmp_path.exists()


def test_parser_preserves_leading_dash_forwarded_arguments() -> None:
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
        ]
    )

    assert arguments.update_argument == ["--no-config", "--upgrade"]
    assert arguments.restart_argument == ["--log-level", "DEBUG"]
    assert arguments.verification_executable == "/target/venv/python"


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
