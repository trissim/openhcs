from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow


class DummyServiceAdapter:
    def __init__(self) -> None:
        self.main_window = object()
        self.event_bus = object()

    def get_event_bus(self) -> object:
        return self.event_bus


def test_dual_editor_window_uses_explicit_service_adapter_context() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.service_adapter = DummyServiceAdapter()

    assert window._find_main_window() is window.service_adapter.main_window
    assert window._get_event_bus() is window.service_adapter.event_bus
