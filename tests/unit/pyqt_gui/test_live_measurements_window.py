from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from PyQt6.QtWidgets import QApplication, QWidget

from openhcs.constants.constants import AllComponents
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.pyqt_gui.windows.live_measurements_window import (
    LiveMeasurementsWindow,
    LiveMeasurementTableModel,
    _scope_text,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


def test_live_measurement_scope_uses_biological_coordinate_labels() -> None:
    scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="2",
        fixed_component_values=((AllComponents.SITE, "3"),),
    )

    label = _scope_text(scope)

    assert label == "well=A01 / site=3 / channel=2"
    assert "AllComponents" not in label
    assert "RuntimeExecutionAxisScope" not in label


def test_image_browser_is_created_only_when_its_results_tab_is_selected(
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    orchestrator = SimpleNamespace(plate_path="/tmp/live-measurement-plate")
    created_for: list[object] = []
    created_scope_parents: list[str] = []
    cleanup_calls = 0

    class FakeImageBrowserWidget(QWidget):
        def __init__(
            self,
            *,
            orchestrator,
            parent,
            state_scope_parent_id,
            **_kwargs,
        ) -> None:
            super().__init__(parent)
            created_for.append(orchestrator)
            created_scope_parents.append(state_scope_parent_id)

        def cleanup(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

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
        assert created_scope_parents == [
            "/tmp/live-measurement-plate::live_measurements"
        ]
        assert isinstance(window.tabs.widget(1), FakeImageBrowserWidget)
    finally:
        window.close()

    assert cleanup_calls == 1
