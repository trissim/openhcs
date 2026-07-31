from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, get_default_ui_config
from openhcs.pyqt_gui.windows.managed_windows import (
    ImageBrowserWindow,
    PlateManagerWindow,
)


class _Signal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[str], None]] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value: str) -> None:
        for callback in self._callbacks:
            callback(value)


class _OrchestratorStub:
    pass


class _PlateManagerStub(QWidget):
    def __init__(self, orchestrator: _OrchestratorStub) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self.plate_selected = _Signal()

    def get_selected_orchestrator(self) -> _OrchestratorStub:
        return self._orchestrator

    def select(self, orchestrator: _OrchestratorStub) -> None:
        self._orchestrator = orchestrator
        self.plate_selected.emit("selected-plate")


class _ImageBrowserStub(QWidget):
    instances: list["_ImageBrowserStub"] = []

    def __init__(
        self,
        *,
        orchestrator: _OrchestratorStub | None,
        color_scheme: None,
        zmq_config,
        progress_config,
    ) -> None:
        super().__init__()
        del color_scheme, progress_config
        self.orchestrators = [orchestrator]
        self.zmq_configs = [zmq_config]
        _ImageBrowserStub.instances.append(self)

    def set_orchestrator(self, orchestrator: _OrchestratorStub) -> None:
        self.orchestrators.append(orchestrator)

    def set_zmq_config(self, config) -> None:
        self.zmq_configs.append(config)


class _ServiceAdapterStub:
    def get_current_color_scheme(self) -> None:
        return None


class _EmbeddedWidgetsStub:
    def __init__(self, plate_manager: _PlateManagerStub) -> None:
        self._plate_manager = plate_manager

    def require_plate_manager(self) -> _PlateManagerStub:
        return self._plate_manager


class _MainWindowStub(QWidget):
    def __init__(self, plate_manager: _PlateManagerStub) -> None:
        super().__init__()
        self.runtime_context = PyQtGuiRuntimeContext(get_default_ui_config())
        self.ui_config_changed = _Signal()
        self.embedded_widgets = _EmbeddedWidgetsStub(plate_manager)


def test_managed_image_browser_uses_embedded_plate_manager(
    qtbot,
    monkeypatch,
) -> None:
    first_orchestrator = _OrchestratorStub()
    second_orchestrator = _OrchestratorStub()
    plate_manager = _PlateManagerStub(first_orchestrator)
    main_window = _MainWindowStub(plate_manager)

    import openhcs.pyqt_gui.widgets.image_browser as image_browser_module

    monkeypatch.setattr(
        image_browser_module,
        "ImageBrowserWidget",
        _ImageBrowserStub,
    )
    _ImageBrowserStub.instances.clear()

    window = ImageBrowserWindow(main_window, _ServiceAdapterStub())
    qtbot.addWidget(window)

    browser = _ImageBrowserStub.instances[-1]
    assert browser.orchestrators == [None, first_orchestrator]
    assert browser.zmq_configs == [main_window.runtime_context.ui_config.zmq]

    plate_manager.select(second_orchestrator)

    assert browser.orchestrators == [
        None,
        first_orchestrator,
        second_orchestrator,
    ]


def test_managed_plate_window_passes_resolved_ui_config(
    qtbot,
    monkeypatch,
) -> None:
    resolved_ui_config = get_default_ui_config()
    captured = {}

    class ServiceAdapterStub:
        widget_gui_config = resolved_ui_config

        @staticmethod
        def get_current_color_scheme() -> None:
            return None

    class PlateManagerStub(QWidget):
        def __init__(
            self,
            service_adapter,
            color_scheme,
            *,
            gui_config,
        ) -> None:
            super().__init__()
            del service_adapter, color_scheme
            captured["gui_config"] = gui_config

    import openhcs.pyqt_gui.widgets.plate_manager as plate_manager_module

    monkeypatch.setattr(
        plate_manager_module,
        "PlateManagerWidget",
        PlateManagerStub,
    )
    monkeypatch.setattr(PlateManagerWindow, "_setup_connections", lambda self: None)

    main_window = QWidget()
    window = PlateManagerWindow(main_window, ServiceAdapterStub())
    qtbot.addWidget(main_window)
    qtbot.addWidget(window)

    assert captured["gui_config"] is resolved_ui_config
