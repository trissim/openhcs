"""Out-of-process OpenHCS environment update and restart worker."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Protocol


class DesktopUpdatePhase(Enum):
    """Closed worker phases projected to the detached progress surface."""

    WAITING_FOR_APPLICATION = "Waiting for OpenHCS to close"
    INSTALLING = "Installing the OpenHCS update"
    VERIFYING = "Verifying the updated environment"
    REFRESHING_DESKTOP = "Refreshing launchers and application icons"
    RESTARTING = "Restarting OpenHCS"


class DesktopUpdateProgressEventKind(Enum):
    """Closed event axis sent from worker orchestration to its progress UI."""

    PHASE = "phase"
    OUTPUT = "output"
    FAILURE = "failure"
    COMPLETE = "complete"


class DesktopUpdateProgressAction(Enum):
    """Closed failure response axis sent from the progress UI to orchestration."""

    REOPEN = "reopen"
    EXIT = "exit"


class DesktopUpdateProgressUnavailable(RuntimeError):
    """Raised before mutation when the detached progress surface cannot start."""


@dataclass(frozen=True, slots=True)
class DesktopUpdateProgressEvent:
    """One typed progress projection emitted by real worker activity."""

    kind: DesktopUpdateProgressEventKind
    message: str
    phase: DesktopUpdatePhase | None = None


@dataclass(frozen=True, slots=True)
class DesktopUpdateProgressTheme:
    """Environment-independent theme projection owned by the progress surface."""

    window_bg: str
    panel_bg: str
    text_primary: str
    text_secondary: str
    text_accent: str
    border_color: str
    button_bg: str
    button_text: str
    error_color: str
    progress_color: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "DesktopUpdateProgressTheme":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Desktop update progress theme must be a JSON object.")
        return cls(**payload)


class DesktopUpdateProgressReporter(Protocol):
    """Progress operations consumed by the authoritative update worker."""

    def phase(self, phase: DesktopUpdatePhase) -> None: ...

    def output(self, message: str) -> None: ...

    def failure(self, message: str) -> DesktopUpdateProgressAction: ...

    def complete(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedProcessLaunchSpec:
    """Already-resolved subprocess arguments supplied by the running GUI."""

    creationflags: int
    start_new_session: bool

    def popen_arguments(self) -> dict[str, bool | int]:
        arguments: dict[str, bool | int] = {}
        if self.creationflags:
            arguments["creationflags"] = self.creationflags
        if self.start_new_session:
            arguments["start_new_session"] = True
        return arguments


@dataclass(frozen=True, slots=True)
class DesktopUpdateExecution:
    """Result of mutation plus the deployment-owned successful restart target."""

    error_message: str | None = None
    restart_executable: str | None = None


def _deployment_restart_executable(output: str) -> str:
    """Read the restart target published by the platform deployment authority."""

    reports: list[dict[str, object]] = []
    for line in output.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "restart_executable" in payload:
            reports.append(payload)
    if len(reports) != 1:
        raise ValueError(
            "Desktop deployment did not publish exactly one restart executable."
        )
    restart_executable = reports[0]["restart_executable"]
    if not isinstance(restart_executable, str) or not restart_executable.strip():
        raise ValueError("Desktop deployment published an invalid restart executable.")
    if not (
        Path(restart_executable).is_absolute()
        or PureWindowsPath(restart_executable).is_absolute()
    ):
        raise ValueError("Desktop deployment published a relative restart executable.")
    return restart_executable


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
    launch_spec: ResolvedProcessLaunchSpec,
    progress: DesktopUpdateProgressReporter,
    installation_pointer: str | None = None,
) -> DesktopUpdateExecution:
    progress.phase(DesktopUpdatePhase.INSTALLING)
    returncode, detail = _run_process_with_progress(
        [executable, *arguments],
        launch_spec=launch_spec,
        progress=progress,
    )
    if returncode != 0:
        return DesktopUpdateExecution(
            error_message=f"OpenHCS update failed with exit code {returncode}."
            + (f"\n\n{detail[-4000:]}" if detail else "")
        )

    progress.phase(DesktopUpdatePhase.VERIFYING)
    verification_returncode, verification_detail = _run_process_with_progress(
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
        launch_spec=launch_spec,
        progress=progress,
    )
    if verification_returncode != 0:
        message = (
            "OpenHCS finished installing, but the updated environment did not "
            f"report version {expected_version}."
        )
        if verification_detail:
            message += f"\n\n{verification_detail[-4000:]}"
        return DesktopUpdateExecution(error_message=message)
    if installation_pointer is not None:
        progress.phase(DesktopUpdatePhase.REFRESHING_DESKTOP)
        deployment_returncode, deployment_detail = _run_process_with_progress(
            [
                verification_executable,
                "-I",
                "-m",
                "openhcs.desktop_deployment_cli",
                f"--installation-pointer={installation_pointer}",
                "--json",
            ],
            launch_spec=launch_spec,
            progress=progress,
        )
        if deployment_returncode != 0:
            message = (
                "OpenHCS was updated, but its launcher, shortcut, or application "
                "icon could not be refreshed. Re-run the official installer to "
                "repair the desktop integration."
            )
            if deployment_detail:
                message += f"\n\n{deployment_detail[-4000:]}"
            return DesktopUpdateExecution(error_message=message)
        try:
            restart_executable = _deployment_restart_executable(deployment_detail)
        except ValueError as exc:
            return DesktopUpdateExecution(
                error_message=(
                    "OpenHCS was updated, but its desktop deployment returned an "
                    f"invalid restart target: {exc} Re-run the official installer "
                    "to repair the desktop integration."
                )
            )
        return DesktopUpdateExecution(restart_executable=restart_executable)
    return DesktopUpdateExecution()


def _run_process_with_progress(
    command: list[str],
    *,
    launch_spec: ResolvedProcessLaunchSpec,
    progress: DesktopUpdateProgressReporter,
) -> tuple[int, str]:
    """Run one worker command while reporting its real merged output."""

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        **launch_spec.popen_arguments(),
    )
    tail: deque[str] = deque(maxlen=200)
    if process.stdout is not None:
        for line in process.stdout:
            message = line.rstrip("\r\n")
            if message:
                progress.output(message)
                tail.append(message)
    return process.wait(), "\n".join(tail)[-4000:]


def _restart(
    executable: str,
    arguments: list[str],
    *,
    session_directory: Path,
    restore_option: str,
    launch_spec: ResolvedProcessLaunchSpec,
) -> str | None:
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
    kwargs.update(launch_spec.popen_arguments())
    try:
        subprocess.Popen(command, **kwargs)
    except OSError as exc:
        return f"OpenHCS could not reopen after the update: {exc}"
    return None


class DesktopUpdateProgressWindow:
    """Branded Tk progress UI owned by the detached, external worker process."""

    _VISIBLE_LOG_LINE_LIMIT = 500

    @classmethod
    def create(
        cls,
        *,
        theme_document: Path,
        brand_document: Path,
    ) -> "DesktopUpdateProgressWindow":
        try:
            return cls(
                theme=DesktopUpdateProgressTheme.read(theme_document),
                brand_document=brand_document,
            )
        except Exception as exc:
            raise DesktopUpdateProgressUnavailable(
                "OpenHCS could not open the detached update progress window. "
                "The environment was not modified. Reopen OpenHCS to recover "
                "the saved session and view details."
            ) from exc

    def __init__(
        self,
        *,
        theme: DesktopUpdateProgressTheme,
        brand_document: Path,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._events: queue.Queue[DesktopUpdateProgressEvent] = queue.Queue()
        self._actions: queue.Queue[DesktopUpdateProgressAction] = queue.Queue()
        self._theme = theme
        self._completed = False
        self._failed = False
        self._root = tk.Tk()
        self._root.title("Updating OpenHCS")
        self._root.configure(background=theme.window_bg)
        self._root.geometry("720x460")
        self._root.minsize(600, 360)

        self._brand_image = tk.PhotoImage(file=str(brand_document))
        display_scale = max(
            1,
            max(self._brand_image.width(), self._brand_image.height()) // 64,
        )
        self._display_brand_image = self._brand_image.subsample(
            display_scale,
            display_scale,
        )
        self._root.iconphoto(True, self._brand_image)

        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure(
            "OpenHCS.Horizontal.TProgressbar",
            background=theme.progress_color,
            troughcolor=theme.panel_bg,
            bordercolor=theme.border_color,
            lightcolor=theme.progress_color,
            darkcolor=theme.progress_color,
        )
        style.configure(
            "OpenHCS.TButton",
            background=theme.button_bg,
            foreground=theme.button_text,
            bordercolor=theme.border_color,
            padding=(14, 7),
        )
        style.map(
            "OpenHCS.TButton",
            background=[("active", theme.text_accent)],
            foreground=[("active", theme.window_bg)],
        )

        content = tk.Frame(self._root, background=theme.window_bg, padx=24, pady=22)
        content.grid(row=0, column=0, sticky="nsew")
        self._root.rowconfigure(0, weight=1)
        self._root.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(3, weight=1)

        logo = tk.Label(
            content,
            image=self._display_brand_image,
            background=theme.window_bg,
        )
        logo.grid(row=0, column=0, rowspan=2, padx=(0, 16), sticky="nw")
        heading = tk.Label(
            content,
            text="OpenHCS Update",
            background=theme.window_bg,
            foreground=theme.text_accent,
            font=("TkDefaultFont", 17, "bold"),
            anchor="w",
        )
        heading.grid(row=0, column=1, sticky="ew")
        self._phase_label = tk.Label(
            content,
            text=DesktopUpdatePhase.WAITING_FOR_APPLICATION.value,
            background=theme.window_bg,
            foreground=theme.text_primary,
            font=("TkDefaultFont", 11),
            anchor="w",
        )
        self._phase_label.grid(row=1, column=1, pady=(4, 14), sticky="ew")

        self._progress_bar = ttk.Progressbar(
            content,
            mode="indeterminate",
            style="OpenHCS.Horizontal.TProgressbar",
        )
        self._progress_bar.grid(
            row=2,
            column=0,
            columnspan=2,
            pady=(0, 14),
            sticky="ew",
        )
        self._progress_bar.start(12)

        output_frame = tk.Frame(
            content,
            background=theme.panel_bg,
            highlightbackground=theme.border_color,
            highlightthickness=1,
        )
        output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self._output = tk.Text(
            output_frame,
            background=theme.panel_bg,
            foreground=theme.text_secondary,
            insertbackground=theme.text_primary,
            borderwidth=0,
            padx=10,
            pady=8,
            wrap="word",
            state="disabled",
        )
        self._output.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            output_frame,
            orient="vertical",
            command=self._output.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._output.configure(yscrollcommand=scrollbar.set)

        footer = tk.Frame(content, background=theme.window_bg)
        footer.grid(row=4, column=0, columnspan=2, pady=(14, 0), sticky="ew")
        footer.columnconfigure(0, weight=1)
        self._status_label = tk.Label(
            footer,
            text="The installation log will appear here.",
            background=theme.window_bg,
            foreground=theme.text_secondary,
            anchor="w",
        )
        self._status_label.grid(row=0, column=0, sticky="ew")
        button_frame = tk.Frame(footer, background=theme.window_bg)
        button_frame.grid(row=0, column=1, sticky="e")
        self._reopen_button = ttk.Button(
            button_frame,
            text="Reopen OpenHCS",
            style="OpenHCS.TButton",
            command=lambda: self._finish_with(DesktopUpdateProgressAction.REOPEN),
        )
        self._exit_button = ttk.Button(
            button_frame,
            text="Exit",
            style="OpenHCS.TButton",
            command=lambda: self._finish_with(DesktopUpdateProgressAction.EXIT),
        )
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def phase(self, phase: DesktopUpdatePhase) -> None:
        self._events.put(
            DesktopUpdateProgressEvent(
                kind=DesktopUpdateProgressEventKind.PHASE,
                phase=phase,
                message=phase.value,
            )
        )

    def output(self, message: str) -> None:
        self._events.put(
            DesktopUpdateProgressEvent(
                kind=DesktopUpdateProgressEventKind.OUTPUT,
                message=message,
            )
        )

    def failure(self, message: str) -> DesktopUpdateProgressAction:
        self._events.put(
            DesktopUpdateProgressEvent(
                kind=DesktopUpdateProgressEventKind.FAILURE,
                message=message,
            )
        )
        return self._actions.get()

    def complete(self) -> None:
        self._events.put(
            DesktopUpdateProgressEvent(
                kind=DesktopUpdateProgressEventKind.COMPLETE,
                message="OpenHCS updated successfully",
            )
        )

    def run(self, operation: Callable[[], int]) -> int:
        """Run orchestration off the Tk main thread and return its exit code."""

        results: queue.Queue[int] = queue.Queue(maxsize=1)

        def execute() -> None:
            results.put(operation())

        worker = threading.Thread(
            target=execute,
            name="openhcs-desktop-update",
            daemon=True,
        )
        worker.start()
        self._root.after(40, self._poll_events)
        self._root.mainloop()
        if worker.is_alive():
            self._actions.put(DesktopUpdateProgressAction.EXIT)
        worker.join(timeout=5)
        if worker.is_alive() or results.empty():
            return 1
        return results.get_nowait()

    def _append_output(self, message: str) -> None:
        self._output.configure(state="normal")
        self._output.insert("end", f"{message}\n")
        line_count = int(self._output.index("end-1c").split(".", maxsplit=1)[0])
        if line_count > self._VISIBLE_LOG_LINE_LIMIT:
            excess = line_count - self._VISIBLE_LOG_LINE_LIMIT
            self._output.delete("1.0", f"{excess + 1}.0")
        self._output.see("end")
        self._output.configure(state="disabled")

    def _show_failure(self, message: str) -> None:
        self._failed = True
        self._progress_bar.stop()
        self._phase_label.configure(
            text="Update failed",
            foreground=self._theme.error_color,
        )
        self._status_label.configure(
            text="Recovery files were preserved. Reopen OpenHCS to view details.",
            foreground=self._theme.error_color,
        )
        self._append_output(message)
        self._reopen_button.grid(row=0, column=0, padx=(0, 8))
        self._exit_button.grid(row=0, column=1)

    def _apply_event(self, event: DesktopUpdateProgressEvent) -> None:
        if event.kind is DesktopUpdateProgressEventKind.PHASE:
            self._phase_label.configure(
                text=event.message,
                foreground=self._theme.text_primary,
            )
            self._status_label.configure(
                text="Update in progress",
                foreground=self._theme.text_secondary,
            )
        elif event.kind is DesktopUpdateProgressEventKind.OUTPUT:
            self._append_output(event.message)
        elif event.kind is DesktopUpdateProgressEventKind.FAILURE:
            self._show_failure(event.message)
        elif event.kind is DesktopUpdateProgressEventKind.COMPLETE:
            self._completed = True
            self._progress_bar.stop()
            self._progress_bar.configure(mode="determinate", value=100)
            self._phase_label.configure(
                text=event.message,
                foreground=self._theme.text_accent,
            )
            self._status_label.configure(text="OpenHCS is reopening.")
            self._root.after(700, self._root.destroy)

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._apply_event(event)
        self._root.after(40, self._poll_events)

    def _finish_with(self, action: DesktopUpdateProgressAction) -> None:
        self._actions.put(action)
        self._root.destroy()

    def _on_close(self) -> None:
        if self._failed:
            self._finish_with(DesktopUpdateProgressAction.EXIT)
        elif self._completed:
            self._root.destroy()
        else:
            self._root.iconify()


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
    parser.add_argument("--installation-pointer")
    parser.add_argument("--progress-theme-file", required=True, type=Path)
    parser.add_argument("--progress-brand-file", required=True, type=Path)
    parser.add_argument("--error-file", required=True, type=Path)
    parser.add_argument("--restore-option", required=True)
    parser.add_argument("--background-creationflags", required=True, type=int)
    parser.add_argument("--background-start-new-session", action="store_true")
    parser.add_argument("--detached-creationflags", required=True, type=int)
    parser.add_argument("--detached-start-new-session", action="store_true")
    return parser.parse_args(argv)


def _write_update_error(path: Path, message: str) -> None:
    path.write_text(message, encoding="utf-8")


def _restart_saved_session(
    arguments: argparse.Namespace,
    *,
    launch_spec: ResolvedProcessLaunchSpec,
    restart_executable: str | None = None,
) -> str | None:
    return _restart(
        arguments.restart_executable
        if restart_executable is None
        else restart_executable,
        arguments.restart_argument,
        session_directory=arguments.session_directory,
        restore_option=arguments.restore_option,
        launch_spec=launch_spec,
    )


def _recover_from_failure(
    error_message: str,
    arguments: argparse.Namespace,
    *,
    progress: DesktopUpdateProgressReporter,
    launch_spec: ResolvedProcessLaunchSpec,
    reopen_available: bool = True,
    restart_executable: str | None = None,
) -> None:
    _write_update_error(arguments.error_file, error_message)
    action = progress.failure(error_message)
    if not reopen_available or action is not DesktopUpdateProgressAction.REOPEN:
        return
    restart_error = _restart_saved_session(
        arguments,
        launch_spec=launch_spec,
        restart_executable=restart_executable,
    )
    if restart_error is not None:
        _write_update_error(
            arguments.error_file,
            f"{error_message}\n\n{restart_error}",
        )


def _perform_update_transaction(
    arguments: argparse.Namespace,
    *,
    progress: DesktopUpdateProgressReporter,
    background_launch_spec: ResolvedProcessLaunchSpec,
    detached_launch_spec: ResolvedProcessLaunchSpec,
) -> int:
    """Execute the real update transaction from the progress UI worker thread."""

    progress.phase(DesktopUpdatePhase.WAITING_FOR_APPLICATION)
    if not _wait_for_parent_exit(arguments.parent_pid):
        error_message = (
            "OpenHCS did not close within 60 seconds, so the update was "
            "cancelled before modifying the environment. The existing application "
            "remains open."
        )
        _recover_from_failure(
            error_message,
            arguments,
            progress=progress,
            launch_spec=detached_launch_spec,
            reopen_available=False,
        )
        return 2

    execution = _run_update(
        arguments.update_executable,
        arguments.update_argument,
        expected_version=arguments.expected_version,
        verification_executable=arguments.verification_executable,
        installation_pointer=arguments.installation_pointer,
        launch_spec=background_launch_spec,
        progress=progress,
    )
    if execution.error_message is not None:
        _recover_from_failure(
            execution.error_message,
            arguments,
            progress=progress,
            launch_spec=detached_launch_spec,
        )
        return 1

    progress.phase(DesktopUpdatePhase.RESTARTING)
    restart_error = _restart_saved_session(
        arguments,
        launch_spec=detached_launch_spec,
        restart_executable=execution.restart_executable,
    )
    if restart_error is not None:
        _recover_from_failure(
            restart_error,
            arguments,
            progress=progress,
            launch_spec=detached_launch_spec,
            restart_executable=execution.restart_executable,
        )
        return 1
    progress.complete()
    return 0


def _perform_update(
    arguments: argparse.Namespace,
    *,
    progress: DesktopUpdateProgressReporter,
    background_launch_spec: ResolvedProcessLaunchSpec,
    detached_launch_spec: ResolvedProcessLaunchSpec,
) -> int:
    """Convert every unexpected orchestration exception into visible recovery."""

    try:
        return _perform_update_transaction(
            arguments,
            progress=progress,
            background_launch_spec=background_launch_spec,
            detached_launch_spec=detached_launch_spec,
        )
    except Exception as exc:
        error_message = f"OpenHCS could not complete the update: {exc}"
        _recover_from_failure(
            error_message,
            arguments,
            progress=progress,
            launch_spec=detached_launch_spec,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    raw_arguments = sys.argv[1:] if argv is None else argv
    arguments = parse_arguments(raw_arguments)
    background_launch_spec = ResolvedProcessLaunchSpec(
        creationflags=arguments.background_creationflags,
        start_new_session=arguments.background_start_new_session,
    )
    detached_launch_spec = ResolvedProcessLaunchSpec(
        creationflags=arguments.detached_creationflags,
        start_new_session=arguments.detached_start_new_session,
    )
    try:
        progress = DesktopUpdateProgressWindow.create(
            theme_document=arguments.progress_theme_file,
            brand_document=arguments.progress_brand_file,
        )
    except DesktopUpdateProgressUnavailable as exc:
        error_message = str(exc)
        _write_update_error(arguments.error_file, error_message)
        if _wait_for_parent_exit(arguments.parent_pid):
            restart_error = _restart_saved_session(
                arguments,
                launch_spec=detached_launch_spec,
            )
            if restart_error is not None:
                _write_update_error(
                    arguments.error_file,
                    f"{error_message}\n\n{restart_error}",
                )
        return 3
    return progress.run(
        lambda: _perform_update(
            arguments,
            progress=progress,
            background_launch_spec=background_launch_spec,
            detached_launch_spec=detached_launch_spec,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
