"""Minimal detached worker that restarts OpenHCS after its current UI exits."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import time


def _wait_for_process_exit(parent_pid: int, timeout_seconds: float = 60.0) -> bool:
    if os.name == "nt":
        synchronize = 0x00100000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return True
        try:
            return kernel32.WaitForSingleObject(
                handle,
                int(timeout_seconds * 1000),
            ) == 0
        finally:
            kernel32.CloseHandle(handle)

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--restart-executable", required=True, type=Path)
    parser.add_argument("--restart-argument", action="append", default=[])
    parser.add_argument("--creationflags", required=True, type=int)
    parser.add_argument("--start-new-session", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    if not _wait_for_process_exit(parsed.parent_pid):
        return 2
    launch_arguments: dict[str, bool | int] = {}
    if parsed.creationflags:
        launch_arguments["creationflags"] = parsed.creationflags
    if parsed.start_new_session:
        launch_arguments["start_new_session"] = True
    try:
        subprocess.Popen(
            [str(parsed.restart_executable), *parsed.restart_argument],
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **launch_arguments,
        )
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
