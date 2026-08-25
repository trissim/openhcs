from types import SimpleNamespace

from PyQt6.QtCore import QEventLoop, QObject, QTimer
from PyQt6.QtWidgets import QApplication

from objectstate import ObjectStateRegistry
from objectstate.global_config import set_global_config_for_editing
from objectstate.object_state import ObjectState
from openhcs.constants.constants import VariableComponents
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from pyqt_reactive.forms.parameter_form_chrome_sync import ParameterFormChromeSync
from pyqt_reactive.forms.parameter_form_tree_index import ParameterFormTreeIndex
from pyqt_reactive.forms.parameter_form_manager import FormManagerConfig
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.protocols.widget_adapters import CheckboxGroupAdapter
from pyqt_reactive.protocols.widget_protocols import ValueSettable
from pyqt_reactive.services.field_change_dispatcher import (
    FieldChangeDispatcher,
    FieldChangeEvent,
)
from pyqt_reactive.theming.color_scheme import ColorScheme


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
    from PyQt6.QtCore import QEventLoop, QTimer

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
            return {path}

        def should_skip_updates(self):
            return False

    class FakeRoot:
        field_id = "step_well_filter_config"
        scope_id = "plate"
        before_mutation = None
        _parent_manager = None
        _dispatching = False
        _in_reset = False
        _block_cross_window_updates = False

        def __init__(self) -> None:
            self.state = FakeState()
            self.parameter_changed = SignalRecorder()
            self.context_changed = SignalRecorder()
            self.synced = []
            self.flash_requests = []
            self.live_refreshes = 0

        def sync_after_model_field_change(
            self,
            field_name,
            full_path,
            *,
            queue_flash=True,
            changed_paths=None,
        ):
            self.synced.append((field_name, full_path))
            if queue_flash:
                self.flash_requests.append(full_path)

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
        loop = QEventLoop()
        QTimer.singleShot(240, loop.quit)
        loop.exec()

        assert manager.state.updated == [("step_well_filter_config.well_filter", "B02")]
        assert manager.synced == [
            ("well_filter", "step_well_filter_config.well_filter")
        ]
        assert manager.flash_requests == ["step_well_filter_config.well_filter"]
        assert manager.live_refreshes == 1
        assert manager.parameter_changed.emissions == [
            ("step_well_filter_config.well_filter", "B02")
        ]
        assert manager.context_changed.emissions == [
            ("plate", "step_well_filter_config.well_filter")
        ]
    finally:
        ObjectStateRegistry._change_callbacks[:] = previous_callbacks


def test_registry_deferred_invalidations_coalesce_until_flush(monkeypatch) -> None:
    """Repeated live invalidations collapse into one descendant recompute request."""

    calls = []

    def record_invalidation(cls, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        ObjectStateRegistry,
        "_invalidate_by_type_and_scope_now",
        classmethod(record_invalidation),
    )
    ObjectStateRegistry._deferred_invalidations.clear()
    ObjectStateRegistry._deferred_invalidation_depth = 0

    with ObjectStateRegistry.defer_live_invalidations():
        ObjectStateRegistry.invalidate_by_type_and_scope("", str, "name")
        ObjectStateRegistry.invalidate_by_type_and_scope("", str, "name")

    assert calls == []
    assert ObjectStateRegistry.has_deferred_invalidations()

    ObjectStateRegistry.flush_deferred_invalidations()

    assert calls == [
        {
            "scope_id": "",
            "changed_type": str,
            "field_name": "name",
            "invalidate_saved": False,
            "include_origin": True,
        }
    ]
    assert not ObjectStateRegistry.has_deferred_invalidations()


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
            self._parent_manager = None
            self.form_tree = None
            self._widget_service = widget_service
            self._parameter_ops_service = placeholder_service
            self.chrome_sync = ParameterFormChromeSync(self)

    class FakeState(SimpleNamespace):
        def get_resolved_value(self, path):
            return self.parameters.get(path)

    state = FakeState(
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
    well_filter._parent_manager = root
    path_planning._parent_manager = root
    form_tree = ParameterFormTreeIndex(root)
    root.form_tree = form_tree
    well_filter.form_tree = form_tree
    path_planning.form_tree = form_tree

    root.chrome_sync.refresh_widgets_for_paths({"well_filter_config.well_filter"})

    assert widget_service.calls == [("well_filter_config", "well_filter", "A01", False)]
    assert well_filter.widgets["well_filter"].values == ["A01"]
    assert path_planning.widgets["well_filter"].values == []
    assert placeholder_service.calls == []

    root.chrome_sync.refresh_widgets_for_paths({"path_planning_config.well_filter"})

    assert widget_service.calls[-1] == (
        "path_planning_config",
        "well_filter",
        None,
        True,
    )
    assert placeholder_service.calls == [("path_planning_config", "well_filter")]


def test_resolved_change_refreshes_widget_values_outside_time_travel(qapp) -> None:
    """External ObjectState edits update visible widgets without time travel."""
    from PyQt6.QtCore import QEventLoop, QTimer

    class FakeChromeSync:
        def __init__(self) -> None:
            self.widget_paths = []
            self.state_paths = []

        def refresh_widgets_for_paths(self, paths):
            self.widget_paths.append(set(paths))
            return set()

        def state_changed_for_paths(self, paths, refreshed_compound_owner_paths=None):
            del refreshed_compound_owner_paths
            self.state_paths.append(set(paths))

    class FakeManager(QObject):
        _parent_manager = None
        field_id = ""

        def __init__(self) -> None:
            super().__init__()
            self.state = SimpleNamespace(scope_id="fake_scope")
            self.chrome_sync = FakeChromeSync()
            self.flash_registrations = []
            self.flash_batches = []
            self._pending_resolved_changed_paths = set()
            self._resolved_changed_flush_scheduled = False
            self._locally_applied_model_paths = set()
            self._pending_path_scoped_state_refresh = None

        def schedule_lifecycle_callback(self, delay_ms, callback):
            return ParameterFormManager.schedule_lifecycle_callback(
                self,
                delay_ms,
                callback,
            )

        def _queue_leaf_flash_for_path(self, path, *, queue_flash=True):
            self.flash_registrations.append((path, queue_flash))
            return path

        def queue_flash_local_batch(self, paths):
            self.flash_batches.append(tuple(paths))

        def _apply_to_nested_managers(self, callback):
            del callback

        def _flush_resolved_values_changed(self):
            return ParameterFormManager._flush_resolved_values_changed(self)

        def _exclude_local_edit_paths(self, changed_paths, local_paths):
            return ParameterFormManager._exclude_local_edit_paths(
                changed_paths,
                local_paths,
            )

        def _widget_refresh_paths_for_changed_paths(self, changed_paths, local_paths):
            return ParameterFormManager._widget_refresh_paths_for_changed_paths(
                self,
                changed_paths,
                local_paths,
            )

    previous = ObjectStateRegistry._in_time_travel
    ObjectStateRegistry._in_time_travel = False
    manager = FakeManager()

    try:
        ParameterFormManager._on_resolved_values_changed(
            manager,
            {"well_filter_config.well_filter"},
        )
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()
    finally:
        ObjectStateRegistry._in_time_travel = previous
        manager.deleteLater()
        qapp.processEvents()

    assert manager.chrome_sync.widget_paths == [{"well_filter_config.well_filter"}]
    assert manager.chrome_sync.state_paths == [{"well_filter_config.well_filter"}]
    assert manager.flash_registrations == [("well_filter_config.well_filter", False)]
    assert manager.flash_batches == [("well_filter_config.well_filter",)]


def test_flash_paths_keep_single_structural_leaf_but_coalesce_rows() -> None:
    """Single-cell edits stay precise; multi-leaf row edits flash the owner section."""

    assert ParameterFormManager._flash_paths_for_changed_paths(
        {
            "source_bindings.source_filters",
            "source_bindings.source_filters[0].match_type",
        }
    ) == ("source_bindings.source_filters[0].match_type",)

    assert ParameterFormManager._flash_paths_for_changed_paths(
        {
            "source_bindings.bindings",
            "source_bindings.bindings[0].alias",
            "source_bindings.bindings[0].required",
        }
    ) == ("source_bindings.bindings",)


def test_resolved_child_path_flashes_inline_dataclass_container() -> None:
    """Inline dataclass child updates flash by ObjectState path."""
    from openhcs.core.source_bindings import NamedSourceBinding
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.pyqt_gui.widgets.source_bindings_editor import (
        SourceBindingsEditorWidget,
    )

    QtApplicationHarness.app()

    manager = ParameterFormManager(
        ObjectState(FunctionStep(func=lambda image: image)),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        for _ in range(20):
            QApplication.processEvents()
            widget = manager.findChild(SourceBindingsEditorWidget)
            if widget is not None:
                break
        else:
            widget = None
        assert widget is not None

        flashes: list[str] = []
        manager.queue_flash_local_batch = flashes.extend
        widget.add_binding_row(NamedSourceBinding(alias="DNA"))
        QApplication.processEvents()

        assert "source_bindings.bindings" in flashes
        assert "source_bindings" not in flashes
    finally:
        manager.deleteLater()


def test_list_enum_placeholder_preview_refreshes_structural_child_paths() -> None:
    """Cross-window list[Enum] inheritance refreshes from structural ObjectState paths."""
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    global_config = GlobalPipelineConfig()
    global_state = ObjectState(global_config, scope_id="")
    ObjectStateRegistry.register(global_state, _skip_snapshot=True)
    set_global_config_for_editing(GlobalPipelineConfig, global_config)

    plate_state = ObjectState(PipelineConfig(), scope_id="/tmp/plate")
    ObjectStateRegistry.register(plate_state, _skip_snapshot=True)

    manager = ParameterFormManager(
        plate_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="/tmp/plate",
        ),
    )

    try:
        for _ in range(120):
            QApplication.processEvents()

        group = next(
            checkbox_group
            for checkbox_group in manager.findChildren(CheckboxGroupAdapter)
            if any(
                enum_value is VariableComponents.SITE
                for enum_value, _ in checkbox_group.checkbox_items()
            )
        )
        checkboxes = dict(group.checkbox_items())

        assert plate_state.parameters["processing_config.variable_components"] is None
        assert group.get_value() is None
        assert checkboxes[VariableComponents.SITE].isChecked()
        assert not checkboxes[VariableComponents.CHANNEL].isChecked()
        assert group.has_placeholder_state()

        global_state.update_parameter(
            "processing_config.variable_components",
            [VariableComponents.SITE, VariableComponents.CHANNEL],
        )
        loop = QEventLoop()
        QTimer.singleShot(300, loop.quit)
        loop.exec()
        QApplication.processEvents()

        assert plate_state.get_resolved_value(
            "processing_config.variable_components"
        ) == [VariableComponents.SITE, VariableComponents.CHANNEL]
        assert group.get_value() is None
        assert checkboxes[VariableComponents.SITE].isChecked()
        assert checkboxes[VariableComponents.CHANNEL].isChecked()
        assert checkboxes[VariableComponents.SITE].get_value() is None
        assert checkboxes[VariableComponents.CHANNEL].get_value() is None
        assert group.has_placeholder_state()
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()
