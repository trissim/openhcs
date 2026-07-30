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


class RecordingSignal:
    def __init__(self) -> None:
        self.connected = []
        self.disconnected = []

    def connect(self, callback) -> None:
        self.connected.append(callback)

    def disconnect(self, callback) -> None:
        self.disconnected.append(callback)


class RecordingEventBus:
    def __init__(self) -> None:
        self.pipeline_changed = RecordingSignal()
        self.config_changed = RecordingSignal()
        self.unregistered = []

    def unregister_window(self, window) -> None:
        self.unregistered.append(window)


def test_dual_editor_window_cleans_cross_window_subscriptions() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    event_bus = RecordingEventBus()
    orchestrator_signal = RecordingSignal()
    window._event_bus = event_bus
    window._orchestrator_config_signal = orchestrator_signal
    window._compiled_artifact_signal = None
    window._runtime_artifact_signal = None
    window._debug_snapshot_signal = None
    window._managed_listener_cleanup_done = False
    window.step_editor = None

    window._cleanup_managed_listeners()

    assert event_bus.pipeline_changed.disconnected == [window._on_pipeline_changed]
    assert event_bus.config_changed.disconnected == [window._on_config_changed]
    assert event_bus.unregistered == [window]
    assert orchestrator_signal.disconnected == [window.on_orchestrator_config_changed]
    assert window._event_bus is None
    assert window._orchestrator_config_signal is None


def test_dual_editor_window_orchestrator_signal_connection_is_owned() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    signal = RecordingSignal()

    window.connect_orchestrator_config_signal(signal)

    assert signal.connected == [window.on_orchestrator_config_changed]
    assert window._orchestrator_config_signal is signal
