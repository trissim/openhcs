"""Out-of-process OpenHCS environment update and restart worker."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import time
from pathlib import Path


def _wait_for_parent_exit(
    parent_pid: int,
    *,
    timeout_seconds: float = 60.0,
) -> bool:
    if os.name == "nt":
        synchronize = 0x00100000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return True
        try:
            wait_object_0 = 0
            return (
                kernel32.WaitForSingleObject(handle, int(timeout_seconds * 1000))
                == wait_object_0
            )
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


def _run_update(
    executable: str,
    arguments: list[str],
    *,
    expected_version: str,
    verification_executable: str,
) -> str | None:
    completed = subprocess.run(
        [executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return f"OpenHCS update failed with exit code {completed.returncode}." + (
            f"\n\n{detail[-4000:]}" if detail else ""
        )

    verification = subprocess.run(
        [
            verification_executable,
            "-c",
            (
                "from importlib.metadata import version;"
                "import sys;"
                "sys.exit(0 if version('openhcs') == sys.argv[1] else 1)"
            ),
            expected_version,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if verification.returncode != 0:
        return (
            "OpenHCS finished installing, but the updated environment did not "
            f"report version {expected_version}."
        )
    return None


def _restart(
    executable: str,
    arguments: list[str],
    *,
    session_directory: Path,
    restore_option: str,
) -> None:
    command = [
        executable,
        *arguments,
        restore_option,
        str(session_directory),
    ]
    kwargs: dict[str, object] = {
        "close_fds": True,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--session-directory", required=True, type=Path)
    parser.add_argument("--update-executable", required=True)
    parser.add_argument("--update-argument", action="append", default=[])
    parser.add_argument("--restart-executable", required=True)
    parser.add_argument("--restart-argument", action="append", default=[])
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--verification-executable", required=True)
    parser.add_argument("--error-file", required=True, type=Path)
    parser.add_argument("--restore-option", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if not _wait_for_parent_exit(arguments.parent_pid):
        arguments.error_file.write_text(
            "OpenHCS did not close within 60 seconds, so the update was "
            "cancelled before modifying the environment.",
            encoding="utf-8",
        )
        shutil.rmtree(arguments.session_directory, ignore_errors=True)
        return 2
    error_message = _run_update(
        arguments.update_executable,
        arguments.update_argument,
        expected_version=arguments.expected_version,
        verification_executable=arguments.verification_executable,
    )
    if error_message is not None:
        arguments.error_file.write_text(error_message, encoding="utf-8")
    _restart(
        arguments.restart_executable,
        arguments.restart_argument,
        session_directory=arguments.session_directory,
        restore_option=arguments.restore_option,
    )
    return 0 if error_message is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
