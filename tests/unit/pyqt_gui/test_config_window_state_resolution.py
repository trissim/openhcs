from __future__ import annotations

import pytest

from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.constants.constants import VariableComponents
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.services.window_handlers import (
    OpenHCSWindowCreationAuthority,
    register_openhcs_window_handlers,
)
from openhcs.pyqt_gui.windows.config_window import ConfigWindow, ConfigWindowStateResolver
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


def teardown_function() -> None:
    ObjectStateRegistry.clear()
    ScopeWindowRegistry.clear()


def test_pipeline_config_window_requires_existing_orchestrator_state() -> None:
    resolver = ConfigWindowStateResolver(
        config_class=PipelineConfig,
        current_config=PipelineConfig(),
        scope_id="/tmp/plate",
    )

    with pytest.raises(RuntimeError, match="requires an existing orchestrator ObjectState"):
        resolver.resolve()


def test_pipeline_config_window_rejects_standalone_pipeline_config_state() -> None:
    scope_id = "/tmp/plate"
    ObjectStateRegistry.register(ObjectState(PipelineConfig(), scope_id=scope_id))
    resolver = ConfigWindowStateResolver(
        config_class=PipelineConfig,
        current_config=PipelineConfig(),
        scope_id=scope_id,
    )

    with pytest.raises(RuntimeError, match="must resolve to an orchestrator ObjectState"):
        resolver.resolve()


def test_pipeline_config_window_uses_orchestrator_delegate_state() -> None:
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/analysis.cppipe",
    ).scope_id
    state = ObjectState(
        PipelineConfigHost(PipelineConfig()),
        scope_id=scope_id,
    )
    ObjectStateRegistry.register(state)
    resolver = ConfigWindowStateResolver(
        config_class=PipelineConfig,
        current_config=PipelineConfig(),
        scope_id=scope_id,
    )

    assert resolver.resolve() is state


def test_global_config_window_uses_canonical_object_state_scope() -> None:
    state = ObjectState(GlobalPipelineConfig(num_workers=3), scope_id="")
    ObjectStateRegistry.register(state)
    resolver = ConfigWindowStateResolver(
        config_class=GlobalPipelineConfig,
        current_config=GlobalPipelineConfig(),
        scope_id=OpenHCSUiWindowId.global_config,
    )

    assert resolver.resolve() is state


def test_global_config_window_creates_state_at_canonical_scope() -> None:
    resolver = ConfigWindowStateResolver(
        config_class=GlobalPipelineConfig,
        current_config=GlobalPipelineConfig(num_workers=3),
        scope_id=OpenHCSUiWindowId.global_config,
    )

    state = resolver.resolve()

    assert state.scope_id == ""
    assert state.object_instance.num_workers == 3


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
        GlobalPipelineConfig,
        state.object_instance,
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

    class FakeServiceAdapter:
        def __init__(self) -> None:
            self.main_window = FakeMainWindow()

    class FakePlateManager:
        def __init__(self) -> None:
            self.service_adapter = FakeServiceAdapter()

    class FakeConfigWindow:
        def __init__(
            self,
            config_class,
            current_config,
            on_save_callback,
            *args,
            **kwargs,
        ) -> None:
            del config_class, current_config, args
            self.on_save_callback = on_save_callback
            self.scope_id = kwargs.get("scope_id")
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
    window_authority = OpenHCSWindowCreationAuthority()
    monkeypatch.setattr(window_authority, "_plate_manager", lambda: FakePlateManager())
    window = window_authority.create_global_config_window(
        ScopeWindowCreationRequest(scope_id=OpenHCSUiWindowId.global_config)
    )
    new_config = GlobalPipelineConfig(num_workers=3)

    window.on_save_callback(new_config)

    assert window is created_windows[0]
    assert window.scope_id == ""
    assert saved_configs == [new_config]
    assert propagated_configs == [new_config]


def test_plate_config_window_factory_rejects_standalone_pipeline_config_scope() -> None:
    scope_id = "/tmp/plate"
    ObjectStateRegistry.register(ObjectState(PipelineConfig(), scope_id=scope_id))

    request = ScopeWindowCreationRequest(scope_id=scope_id)

    assert OpenHCSWindowCreationAuthority().create_plate_config_window(request) is None


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
