from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from openhcs.pyqt_gui.services.main_window_workflows import MainWindowEmbeddedWidgets
from openhcs.pyqt_gui.windows.managed_windows import ImageBrowserWindow


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
    ) -> None:
        super().__init__()
        del color_scheme
        self.orchestrators = [orchestrator]
        _ImageBrowserStub.instances.append(self)

    def set_orchestrator(self, orchestrator: _OrchestratorStub) -> None:
        self.orchestrators.append(orchestrator)


class _ServiceAdapterStub:
    def get_current_color_scheme(self) -> None:
        return None


class _MainWindowStub(QWidget):
    def __init__(self, plate_manager: _PlateManagerStub) -> None:
        super().__init__()
        self.embedded_widgets = MainWindowEmbeddedWidgets(
            plate_manager=plate_manager,
        )


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

    plate_manager.select(second_orchestrator)

    assert browser.orchestrators == [
        None,
        first_orchestrator,
        second_orchestrator,
    ]
