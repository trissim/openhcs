"""Early desktop-startup feedback for the OpenHCS GUI.

This module intentionally imports only the Python standard library and
pyqt-reactive's standard-library-only process-launch policy at module load time.
The progress window runs in a small child process so it can be visible while
the main process imports and constructs the real Qt application.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from enum import Enum
from importlib import import_module
from threading import Lock, Thread
from typing import IO, Callable, Protocol

from pyqt_reactive.process_launch import BackgroundProcessLaunchPolicy

_STARTUP_WINDOW_CHILD_ARGUMENT = "--startup-window-child"
_STARTUP_PROGRESS_ENVIRONMENT = "OPENHCS_STARTUP_PROGRESS"


class GuiStartupProgressReporter(Protocol):
    """Progress surface consumed by the authoritative GUI launch path."""

    def ready(self) -> None:
        """Close the progress surface after the main window is painted."""

    def fail(self, message: str, detail: str = "") -> None:
        """Keep the progress surface open and expose a startup failure."""


class _StartupEventKind(str, Enum):
    """Closed event axis for the startup-window IPC protocol."""

    OUTPUT = "output"
    READY = "ready"
    FAILURE = "failure"
    EOF = "eof"


class GuiStartupProgressController:
    """Own one best-effort startup-window child process."""

    def __init__(
        self,
        process: subprocess.Popen[str] | None,
    ) -> None:
        self._process = process
        self._stream: IO[str] | None = process.stdin if process is not None else None
        self._lock = Lock()
        self._failed = False
        self._closed = False
        self._finish_callbacks: list[Callable[[], None]] = []

    @classmethod
    def start(cls) -> "GuiStartupProgressController":
        """Start the lightweight progress window without blocking GUI startup."""
        if os.environ.get(_STARTUP_PROGRESS_ENVIRONMENT, "1") == "0":
            return cls(None)

        launch_policy = BackgroundProcessLaunchPolicy.current()
        command = [
            launch_policy.python_executable(sys.executable),
            "-m",
            "openhcs.gui_startup",
            _STARTUP_WINDOW_CHILD_ARGUMENT,
        ]
        popen_arguments: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "text": True,
            "bufsize": 1,
        }
        popen_arguments.update(launch_policy.popen_arguments())

        try:
            process = subprocess.Popen(command, **popen_arguments)
        except (OSError, ValueError):
            return cls(None)
        return cls(process)

    @property
    def active(self) -> bool:
        """Whether a startup window is currently accepting output."""
        return not self._closed and self._stream is not None

    def output(self, message: str, *, stream_name: str) -> None:
        """Mirror one real terminal line into the startup window."""
        self._send(
            _StartupEventKind.OUTPUT,
            message=message,
            stream_name=stream_name,
        )

    def add_finish_callback(self, callback: Callable[[], None]) -> None:
        """Run a cleanup callback immediately before the progress stream closes."""
        if self._closed:
            callback()
            return
        self._finish_callbacks.append(callback)

    def ready(self) -> None:
        self._run_finish_callbacks()
        self._send(_StartupEventKind.READY)
        self._finish_stream()

    def fail(self, message: str, detail: str = "") -> None:
        self._failed = True
        self._run_finish_callbacks()
        self._send(
            _StartupEventKind.FAILURE,
            message=message,
            detail=detail,
        )
        self._finish_stream()

    def close_unless_failed(self) -> None:
        """Close a still-active non-failure window during early termination."""
        if not self._failed:
            self.ready()

    def _run_finish_callbacks(self) -> None:
        callbacks, self._finish_callbacks = self._finish_callbacks, []
        for callback in callbacks:
            callback()

    def _send(self, kind: _StartupEventKind, **payload: str) -> None:
        with self._lock:
            if self._closed or self._stream is None:
                return
            event = {"kind": kind.value, **payload}
            try:
                self._stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._finish_stream_locked()

    def _finish_stream(self) -> None:
        with self._lock:
            self._finish_stream_locked()

    def _finish_stream_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        if self._process is not None:
            process = self._process
            Thread(
                target=process.wait,
                name="openhcs-startup-window-reaper",
                daemon=True,
            ).start()


class GuiStartupStreamTee:
    """Preserve one terminal stream while mirroring its emitted lines."""

    def __init__(
        self,
        original: IO[str] | None,
        progress: GuiStartupProgressController,
        *,
        stream_name: str,
    ) -> None:
        self._original = original
        self._progress = progress
        self._stream_name = stream_name
        self._pending = ""
        self._mirroring = True
        self._lock = Lock()

    @property
    def encoding(self) -> str:
        if self._original is None:
            return "utf-8"
        return self._original.encoding or "utf-8"

    @property
    def errors(self) -> str:
        if self._original is None:
            return "strict"
        return self._original.errors or "strict"

    @property
    def closed(self) -> bool:
        return bool(self._original is not None and self._original.closed)

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(self._original is not None and self._original.isatty())

    def fileno(self) -> int:
        if self._original is None:
            raise OSError("The GUI process has no terminal file descriptor.")
        return self._original.fileno()

    def write(self, text: str) -> int:
        written = len(text)
        if self._original is not None:
            result = self._original.write(text)
            if result is not None:
                written = result
        with self._lock:
            if self._mirroring:
                self._pending += text
                self._emit_complete_lines_locked()
        return written

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()
        with self._lock:
            if self._mirroring and self._pending:
                self._emit_line_locked(self._pending)
                self._pending = ""

    def stop_mirroring(self) -> None:
        """Stop projecting output after the main window becomes ready."""
        with self._lock:
            if not self._mirroring:
                return
            if self._pending:
                self._emit_line_locked(self._pending)
                self._pending = ""
            self._mirroring = False

    def _emit_complete_lines_locked(self) -> None:
        while True:
            newline_index = self._pending.find("\n")
            if newline_index < 0:
                return
            line = self._pending[:newline_index].rstrip("\r")
            self._pending = self._pending[newline_index + 1 :]
            self._emit_line_locked(line)

    def _emit_line_locked(self, line: str) -> None:
        if line:
            self._progress.output(line, stream_name=self._stream_name)


class GuiStartupOutputMirror:
    """Install and restore the startup-only stdout/stderr tee."""

    def __init__(self, progress: GuiStartupProgressController) -> None:
        self._progress = progress
        self._original_stdout: IO[str] | None = None
        self._original_stderr: IO[str] | None = None
        self._stdout_tee: GuiStartupStreamTee | None = None
        self._stderr_tee: GuiStartupStreamTee | None = None
        self._started = False

    def start(self) -> None:
        if self._started or not self._progress.active:
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_tee = GuiStartupStreamTee(
            self._original_stdout,
            self._progress,
            stream_name="stdout",
        )
        self._stderr_tee = GuiStartupStreamTee(
            self._original_stderr,
            self._progress,
            stream_name="stderr",
        )
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee
        self._started = True
        self._progress.add_finish_callback(self.stop)

    def stop(self) -> None:
        if not self._started:
            return
        assert self._stdout_tee is not None
        assert self._stderr_tee is not None
        self._stdout_tee.stop_mirroring()
        self._stderr_tee.stop_mirroring()
        if sys.stdout is self._stdout_tee:
            sys.stdout = self._original_stdout
        if sys.stderr is self._stderr_tee:
            sys.stderr = self._original_stderr
        self._started = False


def _run_startup_window_child() -> int:
    """Run the isolated Qt progress surface until the parent reports readiness."""
    from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
    )
    from pyqt_reactive.theming import ColorScheme, ThemeManager

    from openhcs.pyqt_gui.branding import (
        openhcs_application_icon,
        openhcs_brand_pixmap,
    )
    from openhcs.resources.brand import BrandAsset

    class _EventBridge(QObject):
        event_received = pyqtSignal(dict)

    class _StartupWindow(QDialog):
        def __init__(self, color_scheme, style_generator) -> None:
            super().__init__()
            self._failed = False
            self._color_scheme = color_scheme
            self.setWindowTitle("Starting OpenHCS")
            self.setMinimumWidth(640)
            self.resize(680, 360)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 22, 24, 20)
            layout.setSpacing(12)

            brand_row = QHBoxLayout()
            brand_row.setContentsMargins(0, 0, 0, 0)
            brand_row.setSpacing(14)

            brand_mark = QLabel()
            brand_mark.setObjectName("startupBrandMark")
            brand_mark.setPixmap(
                openhcs_brand_pixmap(
                    BrandAsset.MARK,
                    QSize(180, 64),
                )
            )
            brand_mark.setFixedSize(180, 64)
            brand_row.addWidget(brand_mark)

            brand_text = QVBoxLayout()
            brand_text.setContentsMargins(0, 0, 0, 0)
            brand_text.setSpacing(2)

            title = QLabel("OpenHCS")
            title_font = QFont()
            title_font.setPointSize(24)
            title_font.setBold(True)
            title.setFont(title_font)
            title.setObjectName("startupTitle")
            brand_text.addWidget(title)

            subtitle = QLabel("Preparing the high-content screening workspace")
            subtitle.setObjectName("startupSubtitle")
            brand_text.addWidget(subtitle)
            brand_text.addStretch(1)
            brand_row.addLayout(brand_text, 1)
            layout.addLayout(brand_row)

            self.phase_label = QLabel("Starting desktop application…")
            self.phase_label.setWordWrap(True)
            self.phase_label.setObjectName("startupPhase")
            layout.addWidget(self.phase_label)

            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(8)
            layout.addWidget(self.progress)

            self.details = QPlainTextEdit()
            self.details.setReadOnly(True)
            self.details.setPlaceholderText("Startup details will appear here.")
            self.details.document().setMaximumBlockCount(300)
            self.details.setObjectName("startupDetails")
            layout.addWidget(self.details, 1)

            actions = QHBoxLayout()
            actions.addStretch(1)
            self.close_button = QPushButton("Close")
            self.close_button.clicked.connect(self.close)
            self.close_button.hide()
            actions.addWidget(self.close_button)
            layout.addLayout(actions)

            to_hex = color_scheme.to_hex
            self.setStyleSheet(
                style_generator.generate_dialog_style()
                + style_generator.generate_button_style()
                + style_generator.generate_progress_bar_style()
                + f"""
                    QLabel#startupTitle {{
                        color: {to_hex(color_scheme.text_accent)};
                    }}
                    QLabel#startupSubtitle {{
                        color: {to_hex(color_scheme.text_secondary)};
                        font-size: 12px;
                    }}
                    QLabel#startupPhase {{
                        color: {to_hex(color_scheme.text_primary)};
                        font-size: 13px;
                        font-weight: 600;
                    }}
                    QPlainTextEdit#startupDetails {{
                        background: {to_hex(color_scheme.panel_bg)};
                        border: 1px solid {to_hex(color_scheme.border_color)};
                        border-radius: 3px;
                        color: {to_hex(color_scheme.text_secondary)};
                        font-family: monospace;
                        font-size: 10px;
                        padding: 6px;
                    }}
                """
            )

        def apply_event(self, event: dict) -> None:
            kind = event.get("kind")
            message = str(event.get("message", ""))
            if kind == _StartupEventKind.OUTPUT.value:
                status = message if len(message) <= 180 else f"{message[:179]}…"
                self.phase_label.setText(status)
                self._append_detail(message)
            elif kind == _StartupEventKind.FAILURE.value:
                self._failed = True
                self.setWindowTitle("OpenHCS startup failed")
                self.phase_label.setText(message or "OpenHCS could not start.")
                self.phase_label.setStyleSheet(
                    "color: "
                    f"{self._color_scheme.to_hex(self._color_scheme.status_error)};"
                    " font-size: 13px; font-weight: 600;"
                )
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._append_detail(message)
                detail = str(event.get("detail", ""))
                if detail:
                    self._append_detail(detail)
                self.close_button.show()
                self.raise_()
                self.activateWindow()
            elif kind == _StartupEventKind.READY.value:
                self.close()
            elif kind == _StartupEventKind.EOF.value and not self._failed:
                self.close()

        def _append_detail(self, message: str) -> None:
            if not message:
                return
            self.details.appendPlainText(message)
            scrollbar = self.details.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    app = QApplication([sys.argv[0]])
    app.setApplicationName("OpenHCS Startup")
    application_icon = openhcs_application_icon()
    app.setWindowIcon(application_icon)
    color_scheme = ColorScheme()
    theme_manager = ThemeManager(color_scheme)
    theme_manager.apply_color_scheme(color_scheme)
    window = _StartupWindow(color_scheme, theme_manager.style_generator)
    window.setWindowIcon(application_icon)
    bridge = _EventBridge()
    bridge.event_received.connect(window.apply_event)

    def _read_parent_events() -> None:
        for line in sys.stdin:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                bridge.event_received.emit(event)
        bridge.event_received.emit({"kind": _StartupEventKind.EOF.value})

    Thread(
        target=_read_parent_events,
        name="openhcs-startup-window-events",
        daemon=True,
    ).start()

    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        window.move(available.center() - window.rect().center())
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


def main() -> int | None:
    """Launch the early progress surface, then run the authoritative GUI path."""
    if sys.argv[1:] == [_STARTUP_WINDOW_CHILD_ARGUMENT]:
        return _run_startup_window_child()

    launch_module = import_module("openhcs.pyqt_gui.launch")
    arguments = launch_module.parse_arguments()
    progress = GuiStartupProgressController.start()
    output_mirror = GuiStartupOutputMirror(progress)
    output_mirror.start()
    try:
        return launch_module.main(
            arguments=arguments,
            startup_progress=progress,
        )
    except ImportError as error:
        if "PyQt6" in str(error) or "pyqt_gui" in str(error):
            message = (
                "PyQt6 GUI dependencies are unavailable. "
                "Install OpenHCS with the 'gui' extra."
            )
            progress.fail(message, traceback.format_exc())
            print(f"ERROR: {message}", file=sys.stderr)
            return 1
        progress.fail("OpenHCS could not import its desktop interface.", traceback.format_exc())
        raise
    except BaseException:
        progress.fail("OpenHCS could not start.", traceback.format_exc())
        raise
    finally:
        output_mirror.stop()
        progress.close_unless_failed()


if __name__ == "__main__":
    raise SystemExit(main())
