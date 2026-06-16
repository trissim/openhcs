from __future__ import annotations

import pytest

from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import PipelineConfig
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.window_handlers import (
    OpenHCSWindowCreationAuthority,
    register_openhcs_window_handlers,
)
from openhcs.pyqt_gui.windows.config_window import ConfigWindowStateResolver
from pyqt_reactive.services.scope_window_factory import ScopeWindowRegistry


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


def test_plate_config_window_factory_rejects_standalone_pipeline_config_scope() -> None:
    scope_id = "/tmp/plate"
    ObjectStateRegistry.register(ObjectState(PipelineConfig(), scope_id=scope_id))

    assert OpenHCSWindowCreationAuthority().create_plate_config_window(scope_id) is None


def test_window_registry_routes_cppipe_plate_scope_to_plate_config_factory() -> None:
    register_openhcs_window_handlers()

    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/analysis.cppipe",
    ).scope_id
    handler = ScopeWindowRegistry.find_handler(scope_id)

    assert handler is not None
    assert handler.__name__ == "create_plate_config_window"


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
    assert handler.__name__ == "create_step_editor_window"
