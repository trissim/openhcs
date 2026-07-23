from __future__ import annotations

from dataclasses import replace

import pytest

from openhcs.config_framework.global_config import set_global_config_for_editing
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import (
    GlobalPipelineConfig,
    PipelineConfig,
)
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.constants.constants import VariableComponents
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.services.window_handlers import (
    OpenHCSWindowCreationAuthority,
    register_openhcs_window_handlers,
)
from openhcs.pyqt_gui.config import UIConfig, get_default_ui_config
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from openhcs.pyqt_gui.windows.config_window import (
    ConfigWindow,
    ConfigWindowTabSpec,
)
from pyqt_reactive.forms.parameter_form_manager import FormManagerConfig, ParameterFormManager
from pyqt_reactive.protocols.widget_adapters import CheckboxGroupAdapter
from pyqt_reactive.services.scope_window_factory import (
    ScopeWindowCreationRequest,
    ScopeWindowRegistry,
)
from pyqt_reactive.theming.color_scheme import ColorScheme


class PipelineConfigHost:
    """Minimal delegated object matching PipelineOrchestrator's ObjectState contract."""

    __objectstate_delegate__ = "pipeline_config"

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        self.pipeline_config = pipeline_config


class UIConfigHost:
    """Delegated object with the wrong saved type for a plate-config scope."""

    __objectstate_delegate__ = "ui_config"

    def __init__(self, ui_config: UIConfig) -> None:
        self.ui_config = ui_config


def teardown_function() -> None:
    ObjectStateRegistry.clear()
    ScopeWindowRegistry.clear()


def test_config_window_tab_uses_exact_caller_owned_state() -> None:
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/analysis.cppipe",
    ).scope_id
    state = ObjectState(
        PipelineConfigHost(PipelineConfig()),
        scope_id=scope_id,
    )
    ObjectStateRegistry.register(state)
    spec = ConfigWindowTabSpec(state=state)

    assert spec.state is state
    assert spec.label == "PipelineConfig"


def test_global_config_window_requires_registered_global_state() -> None:
    with pytest.raises(
        RuntimeError,
        match="GlobalPipelineConfig ObjectState is not registered",
    ):
        OpenHCSWindowCreationAuthority().create_global_config_window(
            ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
        )


def test_global_config_window_rejects_wrong_registered_global_state() -> None:
    ObjectStateRegistry.register(
        ObjectState(PipelineConfig(), scope_id=""),
    )

    with pytest.raises(TypeError, match="must contain GlobalPipelineConfig"):
        OpenHCSWindowCreationAuthority().create_global_config_window(
            ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
        )


def test_global_config_window_requires_registered_ui_state() -> None:
    ObjectStateRegistry.register(
        ObjectState(GlobalPipelineConfig(), scope_id=""),
    )

    with pytest.raises(RuntimeError, match="UIConfig ObjectState is not registered"):
        OpenHCSWindowCreationAuthority().create_global_config_window(
            ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
        )


def test_global_config_window_rejects_wrong_registered_ui_state() -> None:
    ObjectStateRegistry.register(
        ObjectState(GlobalPipelineConfig(), scope_id=""),
    )
    ObjectStateRegistry.register(
        ObjectState(PipelineConfig(), scope_id=UIConfig.object_state_scope_id()),
    )

    with pytest.raises(TypeError, match="must contain UIConfig"):
        OpenHCSWindowCreationAuthority().create_global_config_window(
            ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
        )


def test_ui_config_object_state_reconstructs_nested_zmq_owner() -> None:
    config = replace(
        get_default_ui_config(),
        zmq=OpenHCSZMQConfig(default_port=8123),
    )
    state = ObjectState(config, scope_id=UIConfig.object_state_scope_id())

    state.update_parameter("zmq.default_port", 8124)
    reconstructed = state.to_object()

    assert isinstance(reconstructed, UIConfig)
    assert isinstance(reconstructed.zmq, OpenHCSZMQConfig)
    assert reconstructed.zmq.default_port == 8124
    assert config.zmq.default_port == 8123


def test_ui_config_pycodify_round_trip_preserves_exact_zmq_type() -> None:
    default_config = get_default_ui_config()
    config = replace(
        default_config,
        performance_monitor=replace(
            default_config.performance_monitor,
            update_fps=2.0,
            sampler_config=replace(
                default_config.performance_monitor.sampler_config,
                enable_gpu_monitoring=False,
            ),
        ),
        zmq=OpenHCSZMQConfig(
            default_port=8123,
            ports_per_server_type=4,
        ),
    )
    source = ConfigDocumentAuthority.render(
        config,
        expected_config_type=UIConfig,
    )
    restored = ConfigDocumentAuthority.from_source(
        source,
        expected_config_type=UIConfig,
    )

    assert restored == config
    assert type(restored.performance_monitor) is type(config.performance_monitor)
    assert type(restored.performance_monitor.sampler_config) is type(
        config.performance_monitor.sampler_config
    )
    assert type(restored.zmq) is OpenHCSZMQConfig


def test_two_tab_config_window_saves_both_authoritative_objects(qapp) -> None:
    pipeline_config = GlobalPipelineConfig(num_workers=2)
    ui_config = replace(
        get_default_ui_config(),
        zmq=OpenHCSZMQConfig(default_port=8123),
    )
    pipeline_state = ObjectState(pipeline_config, scope_id="")
    ui_state = ObjectState(ui_config, scope_id=UIConfig.object_state_scope_id())
    ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
    ObjectStateRegistry.register(ui_state, _skip_snapshot=True)
    saved: list[object] = []
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=pipeline_state,
                on_save=saved.append,
            ),
            ConfigWindowTabSpec(
                state=ui_state,
                on_save=saved.append,
            ),
        ),
        scope_id=OpenHCSUiWindowId.global_config,
    )

    try:
        ui_state.update_parameter("zmq.default_port", 8124)
        qapp.processEvents()
        window.save_config(close_window=False)

        assert tuple(type(value) for value in saved) == (
            GlobalPipelineConfig,
            UIConfig,
        )
        assert saved[1].zmq.default_port == 8124
        assert tuple(tab.spec.state for tab in window._tabs) == (
            pipeline_state,
            ui_state,
        )
        assert not pipeline_state.is_raw_dirty
        assert not ui_state.is_raw_dirty
    finally:
        window.close()


def test_config_window_page_count_and_visual_scope_own_presentation(qapp) -> None:
    plate_state = ObjectState(PipelineConfig(), scope_id="/tmp/plate")
    plate_window = ConfigWindow(
        tabs=(ConfigWindowTabSpec(state=plate_state),),
        scope_id=plate_state.scope_id,
    )

    try:
        assert plate_window._tab_body.tab_bar.isHidden()
        assert plate_window._tab_body.current_widget() is plate_window._tabs[0].content
        assert plate_window.windowTitle() == "Config PipelineConfig"
        assert plate_window._header_label.text() == "Config PipelineConfig"
        code_document = plate_window.window_code_document_driver().read_document()
        assert (
            ConfigDocumentAuthority.from_source(
                code_document.source,
                expected_config_type=PipelineConfig,
            )
            == plate_state.saved_object
        )
    finally:
        plate_window.close()

    global_state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ui_state = ObjectState(
        get_default_ui_config(),
        scope_id=UIConfig.object_state_scope_id(),
    )
    global_window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(state=global_state),
            ConfigWindowTabSpec(state=ui_state),
        ),
        scope_id="",
    )

    try:
        qapp.processEvents()

        assert not global_window._tab_body.tab_bar.isHidden()
        assert tuple(
            global_window._tab_body.tab_bar.tabText(index)
            for index in range(global_window._tab_body.tab_bar.count())
        ) == ("GlobalPipelineConfig", "UIConfig")
        assert tuple(tab.form_manager.scope_id for tab in global_window._tabs) == (
            global_state.scope_id,
            ui_state.scope_id,
        )
        assert all(
            tab.form_manager._parent_manager is None
            and tab.form_manager._visual_scope_id == ""
            and tab.form_manager._scope_color_scheme.scope_id == ""
            and tab.form_manager._scope_color_scheme.accent_qcolor().name().lower()
            == "#ffffff"
            for tab in global_window._tabs
        )
    finally:
        global_window.close()


def test_pipeline_config_header_projects_semantic_groups_by_capacity(qapp) -> None:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QPushButton
    from pyqt_reactive.widgets.shared.responsive_layout_widgets import (
        _widget_required_width,
    )

    state = ObjectState(PipelineConfig(), scope_id="/tmp/plate")
    window = ConfigWindow(
        tabs=(ConfigWindowTabSpec(state=state),),
        scope_id=state.scope_id,
    )

    try:
        window.show()
        qapp.processEvents()
        header = window._action_header
        layout = header._layout_widget
        expected_labels = {"Cancel", "Save", "Reset", "View Code", "Help"}
        buttons = {
            button.text(): button
            for button in header.findChildren(QPushButton)
            if button.text() in expected_labels
        }
        assert set(buttons) == expected_labels

        widths = {
            name: _widget_required_width(widget)
            for name, widget in layout._groups
        }
        required_width = layout._row_width(
            ["title", "group_auxiliary", "group_commit"],
            widths,
        )
        outer_width = window.width() - layout.width()
        window.resize(required_width + outer_width, window.height())
        qapp.processEvents()
        layout._update_layout()
        qapp.processEvents()

        assert layout._last_row1 == ["title", "group_auxiliary", "group_commit"]
        assert layout._last_row2 == []
        assert len({button.geometry().center().y() for button in buttons.values()}) == 1

        title_group = dict(layout._groups)["title"]
        assert buttons["Help"].parentWidget() is title_group
        assert header.header_label.geometry().right() < buttons["Help"].geometry().left()
        commit_group = dict(layout._groups)["group_commit"]
        assert commit_group.geometry().right() >= layout.contentsRect().right() - 1

        window.resize(required_width + outer_width - 1, window.height())
        layout._update_layout()
        qapp.processEvents()

        assert layout._last_row1 == ["title", "group_commit"]
        assert layout._last_row2 == ["group_auxiliary"]
        assert buttons["Cancel"].mapTo(header, QPoint()).y() < buttons["Reset"].mapTo(
            header, QPoint()
        ).y()
        auxiliary_group = dict(layout._groups)["group_auxiliary"]
        assert auxiliary_group.geometry().right() >= layout.contentsRect().right() - 1
    finally:
        window.close()


def test_config_window_header_switches_title_help_and_auxiliary_actions_by_tab(qapp) -> None:
    global_state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ui_state = ObjectState(
        get_default_ui_config(),
        scope_id=UIConfig.object_state_scope_id(),
    )
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(state=global_state),
            ConfigWindowTabSpec(state=ui_state),
        ),
        scope_id="",
        title_text="Configure OpenHCS",
    )

    try:
        window.show()
        qapp.processEvents()
        header = window._action_header
        first_tab, second_tab = window._tabs

        assert header.action("tab_0_help") is first_tab.help_button
        assert header.action("tab_1_help") is second_tab.help_button
        assert first_tab.help_button.isVisible()
        assert first_tab.actions.isVisible()
        assert not second_tab.help_button.isVisible()
        assert not second_tab.actions.isVisible()

        window._tab_body.set_current_index(1)
        qapp.processEvents()

        assert not first_tab.help_button.isVisible()
        assert not first_tab.actions.isVisible()
        assert second_tab.help_button.isVisible()
        assert second_tab.actions.isVisible()
        assert window.active_tab is second_tab
        assert window._action_header._layout_widget._last_width == (
            window._action_header._layout_widget.contentsRect().width()
        )
    finally:
        window.close()


def test_pipeline_config_code_document_emits_only_authored_override(qapp) -> None:
    state = ObjectState(
        PipelineConfig(),
        scope_id="/tmp/plate",
    )
    window = ConfigWindow(
        tabs=(ConfigWindowTabSpec(state=state),),
        scope_id=state.scope_id,
    )

    try:
        authored_source = """# OpenHCS configuration

from openhcs.core.config import (
    LazyWellFilterConfig,
    PipelineConfig,
)

config = PipelineConfig(
    well_filter_config=LazyWellFilterConfig(
        well_filter='3'
    )
)"""
        driver = window.window_code_document_driver()
        driver.apply_source(authored_source)
        source = driver.read_document().source

        assert state.signature_diff_fields == {"well_filter_config.well_filter"}
        assert source == authored_source
        restored = ConfigDocumentAuthority.from_source(
            source,
            expected_config_type=PipelineConfig,
        )
        assert restored.well_filter_config.well_filter == "3"
    finally:
        window.close()


def test_ui_config_save_commits_state_before_live_notifications(qapp) -> None:
    from types import MethodType, SimpleNamespace

    from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext
    from openhcs.pyqt_gui.main import OpenHCSMainWindow

    class ConfigConsumer:
        def __init__(self) -> None:
            self.config = None

        def set_ui_config(self, config) -> None:
            self.config = config

    class ZMQConsumer:
        def __init__(self) -> None:
            self.config = None

        def set_zmq_config(self, config, _ports) -> None:
            self.config = config

    class MonitorConsumer:
        def __init__(self) -> None:
            self.config = None

        def update_config(self, config) -> None:
            self.config = config

    class Signal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, config) -> None:
            for callback in self.callbacks:
                callback(config)

    initial = replace(
        get_default_ui_config(),
        zmq=OpenHCSZMQConfig(default_port=8123),
    )
    state = ObjectState(initial, scope_id=UIConfig.object_state_scope_id())
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    main_window = SimpleNamespace(
        runtime_context=PyQtGuiRuntimeContext(initial),
        window_services=SimpleNamespace(widget_gui_config=initial),
        system_monitor=MonitorConsumer(),
        plate_manager_widget=ConfigConsumer(),
        pipeline_editor_widget=SimpleNamespace(gui_config=initial),
        zmq_manager_widget=ZMQConsumer(),
        ui_config_changed=Signal(),
        zmq_server_manager_ports_to_scan=lambda: [8124],
    )
    main_window.set_ui_config = MethodType(OpenHCSMainWindow.set_ui_config, main_window)
    signal_observations: list[tuple[UIConfig, UIConfig]] = []
    registry_observations: list[UIConfig] = []
    main_window.ui_config_changed.connect(
        lambda config: signal_observations.append((config, state.saved_object))
    )
    ObjectStateRegistry.connect_listener(
        lambda: registry_observations.append(state.saved_object)
    )
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=state,
                on_save=main_window.set_ui_config,
            ),
        ),
        scope_id="",
    )

    try:
        state.update_parameter("performance_monitor.update_fps", 2.0)
        state.update_parameter("zmq.default_port", 8124)
        registry_observations.clear()
        window.save_config()
        qapp.processEvents()

        committed = state.saved_object
        assert committed.performance_monitor.update_fps == 2.0
        assert committed.zmq.default_port == 8124
        assert main_window.runtime_context.ui_config is committed
        assert main_window.window_services.widget_gui_config is committed
        assert main_window.system_monitor.config is committed.performance_monitor
        assert main_window.plate_manager_widget.config is committed
        assert main_window.pipeline_editor_widget.gui_config is committed
        assert main_window.zmq_manager_widget.config is committed.zmq
        assert signal_observations == [(committed, committed)]
        assert registry_observations
        assert all(observed is committed for observed in registry_observations)
    finally:
        window.close()


def test_global_config_concrete_variable_components_are_not_placeholder(qapp) -> None:
    """Concrete global defaults must not be painted as inherited lazy values."""
    state = ObjectState(GlobalPipelineConfig(), scope_id="")
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        for _ in range(120):
            qapp.processEvents()

        groups = manager.findChildren(CheckboxGroupAdapter)
        variable_components_group = next(
            group
            for group in groups
            if any(
                enum_value is VariableComponents.SITE
                for enum_value, _ in group.checkbox_items()
            )
        )

        assert variable_components_group.get_value() == [VariableComponents.SITE]
        assert not variable_components_group.has_placeholder_state()
        for _, checkbox in variable_components_group.checkbox_items():
            assert not checkbox.is_placeholder()
            assert not checkbox.has_placeholder_state()
    finally:
        manager.deleteLater()


def test_config_window_save_button_starts_disabled_and_styles_disabled_state(qapp) -> None:
    """Config windows expose managed dirty state through visible save styling."""
    state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=state,
            ),
        ),
        scope_id=OpenHCSUiWindowId.global_config,
    )

    try:
        qapp.processEvents()

        assert window._save_button is not None
        assert not window._save_button.isEnabled()
        assert "QPushButton:disabled" in window._save_button.styleSheet()

        state.update_parameter("num_workers", state.parameters["num_workers"] + 1)
        qapp.processEvents()
        window.detect_changes()

        assert window._save_button.isEnabled()
    finally:
        window.close()
        ObjectStateRegistry.clear()


def test_pipeline_streaming_enableable_placeholders_dim_after_build_and_toggle(qapp) -> None:
    """Lazy inherited streaming configs still receive enableable dimming."""
    ObjectStateRegistry.clear()
    set_global_config_for_editing(GlobalPipelineConfig, GlobalPipelineConfig())

    state = ObjectState(PipelineConfig(), scope_id="/tmp/plate")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="/tmp/plate",
        ),
    )

    try:
        for _ in range(200):
            qapp.processEvents()

        for field_name in (
            "streaming_defaults",
            "napari_streaming_config",
            "fiji_streaming_config",
        ):
            nested_manager = manager.nested_managers[field_name]
            enabled_widget = nested_manager.widgets["enabled"]
            dimmed_widget = nested_manager.widgets["host"]
            enabled_path = f"{field_name}.enabled"

            assert enabled_widget.isChecked() is False
            assert enabled_widget.get_value() is None
            assert state.parameters[enabled_path] is None
            assert dimmed_widget.property("enabled_field_dimmed") is True
            assert dimmed_widget.graphicsEffect() is not None

            enabled_widget.click()
            for _ in range(20):
                qapp.processEvents()

            assert state.parameters[enabled_path] is True
            assert dimmed_widget.property("enabled_field_dimmed") is False
            assert dimmed_widget.graphicsEffect() is None

            enabled_widget.click()
            for _ in range(20):
                qapp.processEvents()

            assert state.parameters[enabled_path] is False
            assert dimmed_widget.property("enabled_field_dimmed") is True
            assert dimmed_widget.graphicsEffect() is not None
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_global_config_window_registers_with_stable_window_id() -> None:
    window = ConfigWindow.__new__(ConfigWindow)
    window.scope_id = ""

    assert window.window_manager_scope_id() == OpenHCSUiWindowId.global_config


def test_scope_global_config_window_save_persists_cache(monkeypatch) -> None:
    created_windows = []
    saved_configs = []
    propagated_configs = []

    class FakeConfigChangedSignal:
        def emit(self, config: GlobalPipelineConfig) -> None:
            propagated_configs.append(config)

    class FakeMainWindow:
        def __init__(self) -> None:
            self.config_changed = FakeConfigChangedSignal()

        def set_ui_config(self, config: UIConfig) -> None:
            self.ui_config = config

    class FakeServiceAdapter:
        def __init__(self) -> None:
            self.main_window = FakeMainWindow()

    class FakePlateManager:
        def __init__(self) -> None:
            self.service_adapter = FakeServiceAdapter()

    class FakeConfigWindow:
        def __init__(
            self,
            tabs,
            **kwargs,
        ) -> None:
            self.tabs = tabs
            self.scope_id = kwargs.get("scope_id")
            self.title_text = kwargs.get("title_text")
            created_windows.append(self)

        def show(self) -> None:
            pass

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

    def save_global_config(config: GlobalPipelineConfig) -> bool:
        saved_configs.append(config)
        return True

    from openhcs.core import config_cache
    from openhcs.pyqt_gui.windows import config_window

    monkeypatch.setattr(config_window, "ConfigWindow", FakeConfigWindow)
    monkeypatch.setattr(
        config_cache,
        "save_global_config_sync",
        save_global_config,
    )
    global_config = GlobalPipelineConfig()
    ui_config = get_default_ui_config()
    global_state = ObjectState(global_config, scope_id="")
    ui_state = ObjectState(ui_config, scope_id=UIConfig.object_state_scope_id())
    ObjectStateRegistry.register(global_state)
    ObjectStateRegistry.register(ui_state)
    window_authority = OpenHCSWindowCreationAuthority()
    monkeypatch.setattr(window_authority, "_plate_manager", lambda: FakePlateManager())
    window = window_authority.create_global_config_window(
        ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
    )
    new_config = GlobalPipelineConfig(num_workers=3)

    window.tabs[0].on_save(new_config)

    assert window is created_windows[0]
    assert window.scope_id == ""
    assert window.title_text == "Configure OpenHCS"
    assert saved_configs == [new_config]
    assert propagated_configs == [new_config]
    assert tuple(tab.state for tab in window.tabs) == (global_state, ui_state)


def test_plate_config_window_factory_rejects_standalone_pipeline_config_scope() -> None:
    scope_id = "/tmp/plate"
    ObjectStateRegistry.register(ObjectState(PipelineConfig(), scope_id=scope_id))

    request = ScopeWindowCreationRequest(scope_id=scope_id)

    assert OpenHCSWindowCreationAuthority().create_plate_config_window(request) is None


def test_plate_config_window_factory_rejects_missing_state() -> None:
    request = ScopeWindowCreationRequest(scope_id="/tmp/missing-plate")

    assert OpenHCSWindowCreationAuthority().create_plate_config_window(request) is None


def test_plate_config_window_factory_rejects_wrong_delegated_state() -> None:
    scope_id = "/tmp/wrong-delegate"
    ObjectStateRegistry.register(
        ObjectState(UIConfigHost(get_default_ui_config()), scope_id=scope_id)
    )

    request = ScopeWindowCreationRequest(scope_id=scope_id)

    assert OpenHCSWindowCreationAuthority().create_plate_config_window(request) is None


def test_plate_config_window_factory_passes_exact_registered_state(monkeypatch) -> None:
    scope_id = "/tmp/plate"
    state = ObjectState(PipelineConfigHost(PipelineConfig()), scope_id=scope_id)
    ObjectStateRegistry.register(state)
    captured: dict[str, object] = {}

    class FakeConfigWindow:
        def __init__(self, *, tabs, scope_id) -> None:
            captured["tabs"] = tabs
            captured["scope_id"] = scope_id

    from openhcs.pyqt_gui.windows import config_window

    monkeypatch.setattr(config_window, "ConfigWindow", FakeConfigWindow)
    authority = OpenHCSWindowCreationAuthority()
    monkeypatch.setattr(authority, "_show_window", lambda window: None)

    window = authority.create_plate_config_window(
        ScopeWindowCreationRequest(scope_id=scope_id)
    )

    assert window is not None
    assert captured["scope_id"] == scope_id
    assert captured["tabs"][0].state is state


def test_window_registry_routes_cppipe_plate_scope_to_plate_config_factory() -> None:
    register_openhcs_window_handlers()

    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/analysis.cppipe",
    ).scope_id
    handler = ScopeWindowRegistry.find_handler(scope_id)

    assert handler is not None
    assert handler.handler.__name__ == "create_plate_config_window"


def test_window_registry_routes_cppipe_step_scope_to_step_editor_factory() -> None:
    register_openhcs_window_handlers()

    plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/analysis.cppipe",
    ).scope_id
    handler = ScopeWindowRegistry.find_handler(
        f"{plate_scope}::functionstep_0"
    )

    assert handler is not None
    assert handler.handler.__name__ == "create_step_editor_window"


def test_window_registry_routes_global_config_to_stable_window_id() -> None:
    register_openhcs_window_handlers()

    legacy_handler = ScopeWindowRegistry.find_handler("")
    stable_handler = ScopeWindowRegistry.find_handler(OpenHCSUiWindowId.global_config)

    assert legacy_handler is not None
    assert stable_handler is not None
    assert legacy_handler.handler.__name__ == "create_global_config_window"
    assert stable_handler.handler.__name__ == "create_global_config_window"
    assert (
        legacy_handler.navigation_target("").window_scope_id
        == OpenHCSUiWindowId.global_config
    )
    assert (
        stable_handler.navigation_target(
            OpenHCSUiWindowId.global_config
        ).window_scope_id
        == OpenHCSUiWindowId.global_config
    )
