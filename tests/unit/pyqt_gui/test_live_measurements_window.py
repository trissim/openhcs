from __future__ import annotations

import sys
from types import ModuleType

from PyQt6.QtWidgets import QApplication, QWidget

from openhcs.pyqt_gui.windows.live_measurements_window import (
    LiveMeasurementsWindow,
    LiveMeasurementTableModel,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


def test_image_browser_is_created_only_when_its_results_tab_is_selected(
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    orchestrator = object()
    created_for: list[object] = []

    class FakeImageBrowserWidget(QWidget):
        def __init__(self, *, orchestrator, parent, **_kwargs) -> None:
            super().__init__(parent)
            created_for.append(orchestrator)

    image_browser_module = ModuleType("openhcs.pyqt_gui.widgets.image_browser")
    image_browser_module.ImageBrowserWidget = FakeImageBrowserWidget
    monkeypatch.setitem(
        sys.modules,
        "openhcs.pyqt_gui.widgets.image_browser",
        image_browser_module,
    )

    window = LiveMeasurementsWindow(
        LiveMeasurementTableModel(),
        orchestrator=orchestrator,
        zmq_config=OpenHCSZMQConfig(),
    )
    try:
        assert window.tabs.currentIndex() == 0
        assert window.tabs.tabText(1) == "Images / Viewers"
        assert created_for == []

        window.tabs.setCurrentIndex(1)
        app.processEvents()

        assert created_for == [orchestrator]
        assert isinstance(window.tabs.widget(1), FakeImageBrowserWidget)
    finally:
        window.close()
