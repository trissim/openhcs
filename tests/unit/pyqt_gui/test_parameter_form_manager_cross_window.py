from types import SimpleNamespace

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication

from objectstate import ObjectStateRegistry
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.services.field_change_dispatcher import FieldChangeDispatcher, FieldChangeEvent


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def test_cross_window_change_debounces_placeholder_refresh() -> None:
    """Cross-window changes schedule live inherited placeholder refreshes."""
    QtApplicationHarness.app()

    refresh_calls = []
    visual_updates = []

    class FakeFormManager(QObject):
        CROSS_WINDOW_PLACEHOLDER_REFRESH_MS = 20

        def __init__(self):
            super().__init__()
            self.field_id = "processing_config"
            self._block_cross_window_updates = False
            self._cross_window_refresh_timer = None
            self._parameter_ops_service = SimpleNamespace(
                refresh_with_live_context=lambda manager: refresh_calls.append(manager)
            )

        def queue_visual_update(self):
            visual_updates.append(self)

        def _schedule_cross_window_placeholder_refresh(self):
            return ParameterFormManager._schedule_cross_window_placeholder_refresh(self)

        def _refresh_cross_window_placeholders(self):
            return ParameterFormManager._refresh_cross_window_placeholders(self)

    manager = FakeFormManager()

    ParameterFormManager._on_live_context_changed(manager)

    assert visual_updates == [manager]
    assert manager._cross_window_refresh_timer is not None
    assert manager._cross_window_refresh_timer.isActive()

    ParameterFormManager._refresh_cross_window_placeholders(manager)

    assert refresh_calls == [manager]
    assert visual_updates == [manager, manager]
    manager._cross_window_refresh_timer.stop()


def test_dispatcher_notifies_source_root_for_live_resolved_refresh() -> None:
    """Unsaved edits must fan out to the source root's live preview refresh."""

    class SignalRecorder:
        def __init__(self) -> None:
            self.emissions = []

        def emit(self, *args):
            self.emissions.append(args)

    class FakeState:
        _in_reset = False
        _block_cross_window_updates = False
        _parent_state = None

        def __init__(self) -> None:
            self.updated = []

        def update_parameter(self, path, value):
            self.updated.append((path, value))

        def should_skip_updates(self):
            return False

    class FakeRoot:
        field_id = "step_well_filter_config"
        scope_id = "plate"
        _parent_manager = None
        _dispatching = False
        _in_reset = False
        _block_cross_window_updates = False

        def __init__(self) -> None:
            self.state = FakeState()
            self.parameter_changed = SignalRecorder()
            self.context_changed = SignalRecorder()
            self.synced = []
            self.live_refreshes = 0

        def sync_after_model_field_change(self, field_name, full_path):
            self.synced.append((field_name, full_path))

        def sync_enabled_field_visuals(self, value):
            del value

        def _on_live_context_changed(self):
            if self._block_cross_window_updates:
                return
            self.live_refreshes += 1

    previous_callbacks = list(ObjectStateRegistry._change_callbacks)
    ObjectStateRegistry._change_callbacks[:] = []
    manager = FakeRoot()
    ObjectStateRegistry.connect_listener(manager._on_live_context_changed)

    try:
        FieldChangeDispatcher.instance().dispatch(
            FieldChangeEvent("well_filter", "B02", manager)
        )

        assert manager.state.updated == [("step_well_filter_config.well_filter", "B02")]
        assert manager.synced == [
            ("well_filter", "step_well_filter_config.well_filter")
        ]
        assert manager.live_refreshes == 1
        assert manager.parameter_changed.emissions == [
            ("step_well_filter_config.well_filter", "B02")
        ]
        assert manager.context_changed.emissions == [
            ("plate", "step_well_filter_config.well_filter")
        ]
    finally:
        ObjectStateRegistry._change_callbacks[:] = previous_callbacks
