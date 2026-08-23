"""Detached desktop restart worker tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from openhcs.pyqt_gui.services import desktop_restart_worker


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--parent-pid",
        "42",
        "--restart-executable",
        str(tmp_path / "openhcs-gui"),
        "--restart-argument=--restore-update-session",
        f"--restart-argument={tmp_path / 'session'}",
        "--creationflags",
        "73",
        "--start-new-session",
    ]


def test_process_wait_distinguishes_live_and_exited_processes() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert not desktop_restart_worker._wait_for_process_exit(
            process.pid,
            timeout_seconds=0.0,
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert desktop_restart_worker._wait_for_process_exit(
        process.pid,
        timeout_seconds=1.0,
    )


def test_windows_process_wait_uses_and_closes_synchronization_handle(
    monkeypatch,
) -> None:
    calls = []

    class _Kernel32Probe:
        def OpenProcess(self, access, inherit_handle, process_id):
            calls.append(("open", access, inherit_handle, process_id))
            return 73

        def WaitForSingleObject(self, handle, timeout_ms):
            calls.append(("wait", handle, timeout_ms))
            return 258

        def CloseHandle(self, handle):
            calls.append(("close", handle))

    monkeypatch.setattr(desktop_restart_worker.os, "name", "nt")
    monkeypatch.setattr(
        desktop_restart_worker.ctypes,
        "WinDLL",
        lambda name, *, use_last_error: (
            calls.append(("library", name, use_last_error)) or _Kernel32Probe()
        ),
        raising=False,
    )

    assert not desktop_restart_worker._wait_for_process_exit(
        42,
        timeout_seconds=1.5,
    )
    assert calls == [
        ("library", "kernel32", True),
        ("open", 0x00100000, False, 42),
        ("wait", 73, 1500),
        ("close", 73),
    ]


def test_worker_launches_restart_only_after_parent_exit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    launched = []
    monkeypatch.setattr(
        desktop_restart_worker,
        "_wait_for_process_exit",
        lambda parent_pid: parent_pid == 42,
    )
    monkeypatch.setattr(
        desktop_restart_worker.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)),
    )

    assert desktop_restart_worker.main(_arguments(tmp_path)) == 0

    command, launch_options = launched[0]
    assert command == [
        str(tmp_path / "openhcs-gui"),
        "--restore-update-session",
        str(tmp_path / "session"),
    ]
    assert launch_options["creationflags"] == 73
    assert launch_options["start_new_session"] is True
    assert launch_options["stdin"] is subprocess.DEVNULL
    assert launch_options["stdout"] is subprocess.DEVNULL
    assert launch_options["stderr"] is subprocess.DEVNULL


def test_worker_fails_closed_when_parent_remains_live(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        desktop_restart_worker,
        "_wait_for_process_exit",
        lambda _parent_pid: False,
    )
    monkeypatch.setattr(
        desktop_restart_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restart must not launch while the parent is live")
        ),
    )

    assert desktop_restart_worker.main(_arguments(tmp_path)) == 2


def test_worker_reports_restart_launch_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        desktop_restart_worker,
        "_wait_for_process_exit",
        lambda _parent_pid: True,
    )
    monkeypatch.setattr(
        desktop_restart_worker.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("restart executable is unavailable")
        ),
    )

    assert desktop_restart_worker.main(_arguments(tmp_path)) == 1
