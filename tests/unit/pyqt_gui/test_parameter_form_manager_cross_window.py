from types import SimpleNamespace

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication

from objectstate import ObjectStateRegistry
from pyqt_reactive.forms.parameter_form_chrome_sync import ParameterFormChromeSync
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.protocols.widget_protocols import ValueSettable
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


def test_chrome_sync_refreshes_widgets_by_exact_objectstate_path() -> None:
    """Value refresh follows the same dotted ObjectState paths as flash/styling."""

    class FakeWidget(ValueSettable):
        def __init__(self) -> None:
            self.values = []

        def set_value(self, value):
            self.values.append(value)

    class FakeWidgetService:
        def __init__(self) -> None:
            self.calls = []

        def update_widget_value(
            self,
            widget,
            value,
            param_name,
            skip_context_behavior,
            manager,
        ) -> None:
            self.calls.append(
                (
                    manager.field_id,
                    param_name,
                    value,
                    skip_context_behavior,
                )
            )
            widget.set_value(value)

    class FakePlaceholderService:
        def __init__(self) -> None:
            self.calls = []

        def refresh_single_placeholder(self, manager, field_name):
            self.calls.append((manager.field_id, field_name))

    class FakeManager:
        def __init__(
            self,
            field_id: str,
            state,
            widget_service,
            placeholder_service,
        ) -> None:
            self.field_id = field_id
            self.state = state
            self.widgets = {"well_filter": FakeWidget()}
            self.nested_managers = {}
            self._widget_service = widget_service
            self._parameter_ops_service = placeholder_service
            self.chrome_sync = ParameterFormChromeSync(self)

    state = SimpleNamespace(
        parameters={
            "well_filter_config.well_filter": "A01",
            "path_planning_config.well_filter": None,
        }
    )
    widget_service = FakeWidgetService()
    placeholder_service = FakePlaceholderService()
    root = FakeManager("", state, widget_service, placeholder_service)
    well_filter = FakeManager(
        "well_filter_config",
        state,
        widget_service,
        placeholder_service,
    )
    path_planning = FakeManager(
        "path_planning_config",
        state,
        widget_service,
        placeholder_service,
    )
    root.nested_managers = {
        "well_filter_config": well_filter,
        "path_planning_config": path_planning,
    }

    root.chrome_sync.refresh_widgets_for_paths(
        {"well_filter_config.well_filter"}
    )

    assert widget_service.calls == [
        ("well_filter_config", "well_filter", "A01", False)
    ]
    assert well_filter.widgets["well_filter"].values == ["A01"]
    assert path_planning.widgets["well_filter"].values == []
    assert placeholder_service.calls == []

    root.chrome_sync.refresh_widgets_for_paths(
        {"path_planning_config.well_filter"}
    )

    assert widget_service.calls[-1] == (
        "path_planning_config",
        "well_filter",
        None,
        True,
    )
    assert placeholder_service.calls == [
        ("path_planning_config", "well_filter")
    ]


def test_resolved_change_refreshes_widget_values_outside_time_travel() -> None:
    """External ObjectState edits update visible widgets without time travel."""

    class FakeChromeSync:
        def __init__(self) -> None:
            self.widget_paths = []

        def refresh_widgets_for_paths(self, paths):
            self.widget_paths.append(set(paths))

    class FakeManager:
        _parent_manager = None
        field_id = ""

        def __init__(self) -> None:
            self.chrome_sync = FakeChromeSync()
            self.flashes = []

        def _queue_leaf_flash_for_path(self, path):
            self.flashes.append(path)

        def _apply_to_nested_managers(self, callback):
            del callback

    previous = ObjectStateRegistry._in_time_travel
    ObjectStateRegistry._in_time_travel = False
    manager = FakeManager()

    try:
        ParameterFormManager._on_resolved_values_changed(
            manager,
            {"well_filter_config.well_filter"},
        )
    finally:
        ObjectStateRegistry._in_time_travel = previous

    assert manager.chrome_sync.widget_paths == [
        {"well_filter_config.well_filter"}
    ]
    assert manager.flashes == ["well_filter_config.well_filter"]
