"""Desktop session capture and relaunch without an environment update."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from pyqt_reactive.process_launch import BackgroundProcessLaunchPolicy

from openhcs.pyqt_gui.services.desktop_update import (
    DesktopRestartEnvironment,
    DesktopRestartPurpose,
    DesktopRestartSession,
    DesktopUpdateError,
    UPDATE_SESSION_ARGUMENT,
)


@dataclass(frozen=True, slots=True)
class DesktopSessionRestart:
    """Captured UI session plus its validated relaunch environment."""

    runtime: DesktopRestartEnvironment
    session: DesktopRestartSession

    @classmethod
    def capture(cls, main_window) -> "DesktopSessionRestart":
        runtime = DesktopRestartEnvironment.current()
        return cls(
            runtime=runtime,
            session=DesktopRestartSession.capture(
                main_window,
                purpose=DesktopRestartPurpose.ZMQ_VERSION,
            ),
        )

    def discard(self) -> None:
        self.session.discard()

    def start(self, *, parent_pid: int | None = None) -> bool:
        if not self.session.is_complete:
            raise DesktopUpdateError("The saved OpenHCS restart session is incomplete.")
        worker = Path(__file__).with_name("desktop_restart_worker.py")
        if not worker.is_file():
            raise DesktopUpdateError("The OpenHCS restart worker is unavailable.")

        restart_spec = BackgroundProcessLaunchPolicy.current(detached=True).resolve()
        worker_policy = BackgroundProcessLaunchPolicy.current(detached=True)
        arguments = [
            worker_policy.python_executable(
                str(self.runtime.worker_python_executable)
            ),
            "-I",
            str(worker),
            "--parent-pid",
            str(os.getpid() if parent_pid is None else parent_pid),
            "--restart-executable",
            str(self.runtime.restart_executable),
            "--creationflags",
            str(restart_spec.creationflags),
        ]
        restart_arguments = (
            *self.runtime.restart_arguments,
            UPDATE_SESSION_ARGUMENT,
            str(self.session.directory),
        )
        arguments.extend(
            f"--restart-argument={argument}" for argument in restart_arguments
        )
        if restart_spec.start_new_session:
            arguments.append("--start-new-session")
        try:
            subprocess.Popen(
                arguments,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **worker_policy.resolve().popen_arguments(),
            )
        except OSError:
            return False
        return True
