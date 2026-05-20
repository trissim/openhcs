from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.scope_token_service import ScopeTokenService

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


class PipelineEditorServiceStub:
    """Minimal service adapter surface needed by PipelineEditorWidget construction."""

    def __init__(self) -> None:
        self.global_config = GlobalPipelineConfig()
        self.color_scheme = ColorScheme()
        self.event_bus = GlobalEventBus()

    def get_global_config(self) -> GlobalPipelineConfig:
        return self.global_config

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus

    def get_file_manager(self):
        return None


def test_pipeline_editor_constructor_connects_debug_toolbar_signal() -> None:
    QtApplicationHarness.app()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())

    assert widget.debug_toolbar is not None
    widget.close()


class RuntimeCallable:
    """Callable object with a non-function scope prefix."""

    def __call__(self, image, threshold: int = 1):
        return image


def test_step_registration_persists_function_editor_scope_tokens() -> None:
    ObjectStateRegistry._states.clear()
    ScopeTokenService.clear_scope("plate::functionstep_0")

    runtime_callable = RuntimeCallable()
    step = FunctionStep(
        func=(runtime_callable, {"threshold": 3}),
        name="Crop",
    )
    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)

    step_state, states = editor._collect_step_registration_states(
        step=step,
        scope_id="plate::functionstep_0",
        parent_state=None,
    )

    assert step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] == [
        "runtimecallable_0",
    ]
    assert [state.scope_id for state in states] == [
        "plate::functionstep_0",
        "plate::functionstep_0::runtimecallable_0",
    ]
