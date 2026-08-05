"""Focused tests for the early OpenHCS desktop startup surface."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from openhcs import __version__ as OPENHCS_VERSION
from openhcs import gui_startup


class _FakeProcess:
    def __init__(self) -> None:
        class _InspectableStream(io.StringIO):
            def close(self) -> None:
                pass

        self.stdin = _InspectableStream()
        self.stdout = io.StringIO('{"kind": "painted"}\n')
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

    events = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert captured["command"] == [
        sys.executable,
        "-m",
        "openhcs.gui_startup",
        "--startup-window-child",
    ]
    assert captured["kwargs"]["stdin"] is subprocess.PIPE
    assert captured["kwargs"]["stdout"] is subprocess.PIPE
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


def test_controller_uses_gui_child_process_policy(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.delenv("OPENHCS_STARTUP_PROGRESS", raising=False)

    class _LaunchPolicy:
        @classmethod
        def current(cls, *, detached=False):
            assert detached is False
            return SimpleNamespace(
                popen_arguments=lambda: {"creationflags": 73},
                python_executable=lambda _executable: "windowed-python",
            )

    monkeypatch.setattr(
        gui_startup,
        "BackgroundProcessLaunchPolicy",
        _LaunchPolicy,
    )
    captured: dict[str, object] = {}

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(gui_startup.subprocess, "Popen", _popen)

    gui_startup.GuiStartupProgressController.start()

    assert captured["command"][0] == "windowed-python"
    assert captured["creationflags"] == 73
    assert "start_new_session" not in captured


def test_windows_native_launcher_handoff_is_signaled_once_after_paint(
    monkeypatch,
) -> None:
    calls = []

    class _WinApi:
        def __init__(self, name, result):
            self._name = name
            self._result = result
            self.argtypes = None
            self.restype = None

        def __call__(self, *arguments):
            calls.append((self._name, arguments))
            return self._result

    kernel32 = SimpleNamespace(
        OpenEventW=_WinApi("open", 41),
        SetEvent=_WinApi("set", 1),
        CloseHandle=_WinApi("close", 1),
    )
    monkeypatch.setattr(gui_startup.sys, "platform", "win32")
    monkeypatch.setenv(
        gui_startup.STARTUP_HANDOFF_EVENT_ENVIRONMENT,
        "Local\\OpenHCSStartup",
    )
    monkeypatch.setattr(
        gui_startup.ctypes,
        "WinDLL",
        lambda name, *, use_last_error: kernel32,
        raising=False,
    )

    assert gui_startup._signal_native_startup_handoff() is True
    assert gui_startup.STARTUP_HANDOFF_EVENT_ENVIRONMENT not in os.environ
    assert calls == [
        ("open", (0x0002, False, "Local\\OpenHCSStartup")),
        ("set", (41,)),
        ("close", (41,)),
    ]


def test_windows_native_launcher_handoff_remains_retryable_after_signal_failure(
    monkeypatch,
) -> None:
    class _WinApi:
        def __init__(self, result):
            self._result = result
            self.argtypes = None
            self.restype = None

        def __call__(self, *arguments):
            return self._result

    kernel32 = SimpleNamespace(
        OpenEventW=_WinApi(41),
        SetEvent=_WinApi(0),
        CloseHandle=_WinApi(1),
    )
    monkeypatch.setattr(gui_startup.sys, "platform", "win32")
    monkeypatch.setenv(
        gui_startup.STARTUP_HANDOFF_EVENT_ENVIRONMENT,
        "Local\\OpenHCSStartup",
    )
    monkeypatch.setattr(
        gui_startup.ctypes,
        "WinDLL",
        lambda name, *, use_last_error: kernel32,
        raising=False,
    )

    assert gui_startup._signal_native_startup_handoff() is False
    assert (
        os.environ[gui_startup.STARTUP_HANDOFF_EVENT_ENVIRONMENT]
        == "Local\\OpenHCSStartup"
    )


def test_controller_does_not_wait_for_timeout_after_child_stdout_closes() -> None:
    process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(""))
    controller = gui_startup.GuiStartupProgressController(process)

    started = time.monotonic()
    assert controller._observe_first_paint(wait=True) is False

    assert time.monotonic() - started < 2.0


def test_controller_degrades_when_progress_process_cannot_start(monkeypatch) -> None:
    handoffs = []
    monkeypatch.setattr(
        gui_startup.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        gui_startup,
        "_signal_native_startup_handoff",
        lambda: handoffs.append("main-window-ready") or True,
    )

    controller = gui_startup.GuiStartupProgressController.start()
    controller.output("Still launches", stream_name="stderr")
    controller.ready()

    assert handoffs == ["main-window-ready"]


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
    config_module.load_cached_ui_config_sync = lambda: SimpleNamespace(
        logging="logging-config"
    )
    gpu_module = ModuleType("openhcs.core.orchestrator.gpu_scheduler")
    gpu_module.setup_global_gpu_registry = lambda *, global_config: events.append(
        ("gpu", global_config)
    )
    app_module = ModuleType("openhcs.pyqt_gui.app")
    app_module.OpenHCSPyQtApp = _Application
    window_utils_module = ModuleType("pyqt_reactive.utils.window_utils")
    window_utils_module.install_global_window_bounds_filter = lambda app: events.append(
        ("bounds", app)
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
            restore_update_session=None,
        ),
        startup_progress=_Progress(),
    )

    assert result == 0
    assert ("gpu", "pipeline-config") in events
    assert events[-1] == ("ready",)


def test_launcher_version_uses_source_version_authority(
    monkeypatch,
    capsys,
) -> None:
    from openhcs.pyqt_gui import launch

    monkeypatch.setattr(sys, "argv", ["openhcs", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        launch.parse_arguments()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"OpenHCS PyQt6 GUI {OPENHCS_VERSION}"


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
    events = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
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

    from openhcs.pyqt_gui import app as app_module
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

    class _Readiness:
        def __init__(self, main_window, *, on_ready, on_failure):
            self._on_ready = on_ready

        def deferred_initialization_complete(self):
            events.append("painted")
            self._on_ready()

        def fail(self, error):
            raise error

    class _ApplicationHarness:
        main_window = _MainWindow()

    monkeypatch.setattr(QtCore, "QTimer", _ImmediateTimer)
    monkeypatch.setattr(app_module, "MainWindowStartupReadiness", _Readiness)
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
        "painted",
        "ready",
    ]


def test_readiness_gate_reports_only_after_initialized_paint(qapp) -> None:
    from PyQt6 import QtCore, QtGui, QtWidgets

    from openhcs.pyqt_gui.app import MainWindowStartupReadiness

    events = []
    main_window = QtWidgets.QWidget()
    gate = MainWindowStartupReadiness(
        main_window,
        on_ready=lambda: events.append("ready"),
        on_failure=lambda error: events.append(f"failed: {error}"),
    )
    paint_event = QtGui.QPaintEvent(QtCore.QRect(0, 0, 10, 10))

    gate.eventFilter(main_window, paint_event)
    qapp.processEvents()
    assert events == []

    gate.deferred_initialization_complete()
    gate.eventFilter(main_window, paint_event)
    gate.eventFilter(main_window, paint_event)
    assert events == []

    qapp.processEvents()
    assert events == ["ready"]


def test_readiness_gate_tracks_a_real_widget_paint(qapp) -> None:
    from PyQt6 import QtWidgets

    from openhcs.pyqt_gui.app import MainWindowStartupReadiness

    events = []
    main_window = QtWidgets.QWidget()
    gate = MainWindowStartupReadiness(
        main_window,
        on_ready=lambda: events.append("ready"),
        on_failure=lambda error: events.append(f"failed: {error}"),
    )
    main_window.show()
    qapp.processEvents()
    assert events == []

    gate.deferred_initialization_complete()
    main_window.repaint()
    assert events == []

    qapp.processEvents()
    assert events == ["ready"]
    main_window.close()


def test_readiness_gate_routes_ready_callback_failure_once(qapp) -> None:
    from PyQt6 import QtCore, QtGui, QtWidgets

    from openhcs.pyqt_gui.app import MainWindowStartupReadiness

    failures = []
    main_window = QtWidgets.QWidget()

    def _fail_ready() -> None:
        raise RuntimeError("restore failed")

    gate = MainWindowStartupReadiness(
        main_window,
        on_ready=_fail_ready,
        on_failure=failures.append,
    )
    gate.deferred_initialization_complete()
    gate.eventFilter(
        main_window,
        QtGui.QPaintEvent(QtCore.QRect(0, 0, 10, 10)),
    )
    qapp.processEvents()
    qapp.processEvents()

    assert len(failures) == 1
    assert str(failures[0]) == "restore failed"


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
