"""Focused tests for the early OpenHCS desktop startup surface."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from openhcs import gui_startup


class _FakeProcess:
    def __init__(self) -> None:
        class _InspectableStream(io.StringIO):
            def close(self) -> None:
                pass

        self.stdin = _InspectableStream()
        self.waited = False

    def wait(self) -> int:
        self.waited = True
        return 0


def test_controller_uses_current_interpreter_and_streams_structured_events(
    monkeypatch,
) -> None:
    process = _FakeProcess()
    captured = {}
    monkeypatch.delenv("OPENHCS_STARTUP_PROGRESS", raising=False)

    def _popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(gui_startup.subprocess, "Popen", _popen)
    controller = gui_startup.GuiStartupProgressController.start()

    controller.output(
        "2026-07-27 - openhcs - INFO - configuration loaded",
        stream_name="stdout",
    )
    controller.fail("Startup failed", "traceback")

    events = [
        json.loads(line)
        for line in process.stdin.getvalue().splitlines()
    ]
    assert captured["command"] == [
        sys.executable,
        "-m",
        "openhcs.gui_startup",
        "--startup-window-child",
    ]
    assert captured["kwargs"]["stdin"] is subprocess.PIPE
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL
    assert events == [
        {
            "kind": "output",
            "message": "2026-07-27 - openhcs - INFO - configuration loaded",
            "stream_name": "stdout",
        },
        {
            "kind": "failure",
            "message": "Startup failed",
            "detail": "traceback",
        },
    ]


def test_controller_degrades_when_progress_process_cannot_start(monkeypatch) -> None:
    monkeypatch.setattr(
        gui_startup.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    controller = gui_startup.GuiStartupProgressController.start()
    controller.output("Still launches", stream_name="stderr")
    controller.ready()


def test_help_exits_before_progress_window_is_started(monkeypatch) -> None:
    class _LaunchModule:
        @staticmethod
        def parse_arguments():
            raise SystemExit(0)

    monkeypatch.setattr(gui_startup, "import_module", lambda name: _LaunchModule)
    monkeypatch.setattr(
        gui_startup.GuiStartupProgressController,
        "start",
        lambda: pytest.fail("progress should not start for non-launching options"),
    )

    with pytest.raises(SystemExit, match="0"):
        gui_startup.main()


def test_parent_entrypoint_passes_parsed_arguments_and_progress(monkeypatch) -> None:
    arguments = SimpleNamespace()
    received = {}

    class _Progress:
        active = False

        def close_unless_failed(self):
            received["closed"] = True

    progress = _Progress()

    class _LaunchModule:
        @staticmethod
        def parse_arguments():
            return arguments

        @staticmethod
        def main(**kwargs):
            received.update(kwargs)
            return 23

    monkeypatch.setattr(gui_startup, "import_module", lambda name: _LaunchModule)
    monkeypatch.setattr(
        gui_startup.GuiStartupProgressController,
        "start",
        lambda: progress,
    )
    monkeypatch.setattr(gui_startup.sys, "argv", ["openhcs"])

    assert gui_startup.main() == 23
    assert received["arguments"] is arguments
    assert received["startup_progress"] is progress
    assert received["closed"] is True


def test_authoritative_launch_path_reports_actual_readiness(monkeypatch) -> None:
    from openhcs.pyqt_gui import launch

    events = []

    class _Progress:
        def ready(self):
            events.append(("ready",))

        def fail(self, message, detail=""):
            events.append(("failure", message, detail))

    class _RuntimeContext:
        def __init__(self, ui_config, *, pipeline_runtime):
            self.ui_config = ui_config
            self.pipeline_runtime = pipeline_runtime

    class _Application:
        def __init__(self, argv, *, runtime_context):
            events.append(("application", runtime_context.pipeline_runtime))

        def run(self, *, on_main_window_ready, on_startup_failure):
            on_main_window_ready()
            return 0

    config_module = ModuleType("openhcs.pyqt_gui.config")
    config_module.PyQtGuiRuntimeContext = _RuntimeContext
    config_module.load_cached_ui_config_sync = lambda: "ui-config"
    gpu_module = ModuleType("openhcs.core.orchestrator.gpu_scheduler")
    gpu_module.setup_global_gpu_registry = lambda *, global_config: events.append(
        ("gpu", global_config)
    )
    app_module = ModuleType("openhcs.pyqt_gui.app")
    app_module.OpenHCSPyQtApp = _Application
    window_utils_module = ModuleType("pyqt_reactive.utils.window_utils")
    window_utils_module.install_global_window_bounds_filter = (
        lambda app: events.append(("bounds", app))
    )

    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)
    monkeypatch.setitem(sys.modules, app_module.__name__, app_module)
    monkeypatch.setitem(sys.modules, window_utils_module.__name__, window_utils_module)
    monkeypatch.setattr(launch, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(launch, "setup_qt_platform", lambda: None)
    monkeypatch.setattr(launch, "check_dependencies", lambda: True)
    monkeypatch.setattr(launch, "load_configuration", lambda path: "pipeline-config")

    result = launch.main(
        arguments=SimpleNamespace(
            log_level="INFO",
            log_file=None,
            config=None,
            no_gpu=False,
        ),
        startup_progress=_Progress(),
    )

    assert result == 0
    assert ("gpu", "pipeline-config") in events
    assert events[-1] == ("ready",)


def test_stream_tee_preserves_terminal_text_and_mirrors_real_lines() -> None:
    original = io.StringIO()
    mirrored = []

    class _Progress:
        def output(self, message, *, stream_name):
            mirrored.append((stream_name, message))

    tee = gui_startup.GuiStartupStreamTee(
        original,
        _Progress(),
        stream_name="stdout",
    )
    terminal_text = (
        "2026-07-27 - openhcs - INFO - Starting OpenHCS\n"
        "registry discovery still running"
    )

    assert tee.write(terminal_text) == len(terminal_text)
    tee.flush()

    assert original.getvalue() == terminal_text
    assert mirrored == [
        ("stdout", "2026-07-27 - openhcs - INFO - Starting OpenHCS"),
        ("stdout", "registry discovery still running"),
    ]


def test_parent_entrypoint_mirrors_actual_stdout_and_stderr(monkeypatch) -> None:
    process = _FakeProcess()
    progress = gui_startup.GuiStartupProgressController(process)

    class _LaunchModule:
        @staticmethod
        def parse_arguments():
            return SimpleNamespace()

        @staticmethod
        def main(*, arguments, startup_progress):
            assert arguments is not None
            print("terminal stdout line")
            print("terminal stderr line", file=sys.stderr)
            startup_progress.ready()
            return 0

    monkeypatch.setattr(gui_startup, "import_module", lambda name: _LaunchModule)
    monkeypatch.setattr(
        gui_startup.GuiStartupProgressController,
        "start",
        lambda: progress,
    )
    monkeypatch.setattr(gui_startup.sys, "argv", ["openhcs"])

    assert gui_startup.main() == 0
    events = [
        json.loads(line)
        for line in process.stdin.getvalue().splitlines()
    ]
    assert events == [
        {
            "kind": "output",
            "message": "terminal stdout line",
            "stream_name": "stdout",
        },
        {
            "kind": "output",
            "message": "terminal stderr line",
            "stream_name": "stderr",
        },
        {"kind": "ready"},
    ]


def test_pyqt_gui_package_import_does_not_eagerly_import_main_window() -> None:
    checkout = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import openhcs.pyqt_gui as package; "
                "assert 'openhcs.pyqt_gui.app' not in sys.modules; "
                "assert 'openhcs.pyqt_gui.main' not in sys.modules; "
                "from openhcs.pyqt_gui import OpenHCSPyQtApp, OpenHCSMainWindow; "
                "from openhcs.pyqt_gui.app import OpenHCSPyQtApp as DirectApp; "
                "from openhcs.pyqt_gui.main import OpenHCSMainWindow as DirectWindow; "
                "assert OpenHCSPyQtApp is DirectApp; "
                "assert OpenHCSMainWindow is DirectWindow; "
                "assert package.OpenHCSPyQtApp is DirectApp; "
                "assert package.OpenHCSMainWindow is DirectWindow"
            ),
        ],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_clean_dependency_check_imports_qtcore_explicitly() -> None:
    checkout = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from openhcs.pyqt_gui.launch import check_dependencies; "
                "assert check_dependencies()"
            ),
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_real_child_closes_on_parent_eof_without_orphan(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openhcs.gui_startup",
            "--startup-window-child",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(
        json.dumps({"kind": "phase", "message": "EOF lifecycle smoke"}) + "\n"
    )
    process.stdin.close()

    assert process.wait(timeout=10) == 0


def test_real_failure_child_remains_actionable_then_is_reaped(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openhcs.gui_startup",
            "--startup-window-child",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(
            json.dumps(
                {
                    "kind": "failure",
                    "message": "Synthetic startup failure",
                    "detail": "Focused failure detail",
                }
            )
            + "\n"
        )
        process.stdin.close()
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert process.poll() is not None


def test_application_reports_ready_after_deferred_initialization() -> None:
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    events = []

    class _ApplicationHarness:
        def show_main_window(
            self,
            *,
            on_deferred_initialization_complete,
            on_deferred_initialization_failed,
        ):
            events.append("show")
            on_deferred_initialization_complete()

        def exec(self):
            events.append("exec")
            return 0

        def cleanup(self):
            events.append("cleanup")

    result = OpenHCSPyQtApp.run(
        _ApplicationHarness(),
        on_main_window_ready=lambda: events.append("ready"),
    )

    assert result == 0
    assert events == ["show", "ready", "exec", "cleanup"]


def test_show_main_window_reports_ready_only_after_deferred_work(
    monkeypatch,
) -> None:
    from PyQt6 import QtCore
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    events = []

    class _ImmediateTimer:
        @staticmethod
        def singleShot(interval, callback):
            events.append(("timer", interval))
            callback()

    class _MainWindow:
        def show(self):
            events.append("show")

        def raise_(self):
            events.append("raise")

        def activateWindow(self):
            events.append("activate")

        def deferred_initialization(self):
            events.append("deferred")

    class _ApplicationHarness:
        main_window = _MainWindow()

        def processEvents(self):
            events.append("process")

    monkeypatch.setattr(QtCore, "QTimer", _ImmediateTimer)
    OpenHCSPyQtApp.show_main_window(
        _ApplicationHarness(),
        on_deferred_initialization_complete=lambda: events.append("ready"),
    )

    assert events == [
        "show",
        "raise",
        "activate",
        ("timer", 100),
        "deferred",
        "process",
        "ready",
    ]


def test_application_surfaces_main_window_construction_failure() -> None:
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    failure = []

    class _ApplicationHarness:
        def show_main_window(self, **kwargs):
            raise RuntimeError("workspace failed")

        def cleanup(self):
            pass

    result = OpenHCSPyQtApp.run(
        _ApplicationHarness(),
        on_startup_failure=failure.append,
    )

    assert result == 1
    assert len(failure) == 1
    assert str(failure[0]) == "workspace failed"


def test_application_exits_after_deferred_initialization_failure() -> None:
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    failure = []

    class _ApplicationHarness:
        exit_code = 0

        def show_main_window(
            self,
            *,
            on_deferred_initialization_complete,
            on_deferred_initialization_failed,
        ):
            on_deferred_initialization_failed(RuntimeError("deferred failed"))

        def exit(self, exit_code):
            self.exit_code = exit_code

        def exec(self):
            return self.exit_code

        def cleanup(self):
            pass

    result = OpenHCSPyQtApp.run(
        _ApplicationHarness(),
        on_startup_failure=failure.append,
    )

    assert result == 1
    assert len(failure) == 1
    assert str(failure[0]) == "deferred failed"
