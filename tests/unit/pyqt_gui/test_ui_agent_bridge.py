from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QTabBar
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QPushButton, QWidget
from PyQt6.QtWidgets import QVBoxLayout

from openhcs.agent.dto.ui_bridge import (
    UiActionInvokeRequest,
    UiBridgeConfirmationRequirement,
    UiBridgeConnectionSpec,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentId,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationRequest,
    UiObjectStateFieldFilter,
    UiObjectStateFieldHelpRequest,
    UiObjectStateScopeListRequest,
    UiSelectedPlateWorkflowKind,
    UiSelectedPlateWorkflowRequest,
    UiStateSurfaceId,
    UiStateSurfaceRequest,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiWidgetActionInvokeRequest,
    UiWidgetTreeRequest,
    UiWindowIdentity,
    UiWindowManagerScope,
    UiWindowSummary,
    UiWindowCloseRequest,
    UiWindowFocusRequest,
    UiWindowOpenPolicy,
    UiWindowSnapshotRequest,
)
from openhcs.agent.ui_bridge_identities import PipelineDebugToolbarWidgetIdentity
from openhcs.serialization.json import to_jsonable
from openhcs.agent.services.ui_bridge_service import (
    UiBridgeConnectionResolution,
    UiBridgeService,
)
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.constants.constants import OrchestratorState
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    PipelineConfig,
)
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.debug import (
    DebugCommand,
    DebugCommandType,
    DebugCursor,
    DebugEventType,
    DebugProgressContext,
    DebugSession,
    DebugTerminalSummary,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.services.ui_agent_bridge import (
    UiAgentBridgeService,
    UiBridgeOperationTracker,
    UiCodeDocumentSourcePolicy,
    UiObjectStateSnapshotProvider,
)
from openhcs.pyqt_gui.services.ui_thread_dispatch import UiThreadDispatcher
from openhcs.pyqt_gui.services.embedded_code_documents import (
    EmbeddedCodeDocumentRegistrationABC,
)
from openhcs.pyqt_gui.services.ui_bridge_composition import (
    OpenHCSUiBridgeCompositionRoot,
)
from openhcs.pyqt_gui.config import AgentUiBridgeConfig
from openhcs.pyqt_gui.services.reactor_providers import OpenHCSCodegenProvider
from openhcs.pyqt_gui.services.ui_bridge_object_state import (
    OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX,
    WINDOW_CODE_DOCUMENT_PREFIX,
    ObjectStateScopeCodeDocumentProvider,
    ObjectStateBridgeProviderSet,
    ObjectStateScopeProjectionService,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
    UiBridgeSurfaceRegistry,
)
from openhcs.pyqt_gui.services.ui_bridge_server import UiBridgeControlServer
from openhcs.pyqt_gui.services.pycodified_window_code_document import (
    PycodifiedConfigDocumentSpec,
    PycodifiedObjectCodeDocumentDriver,
    PycodifiedObjectDocumentSpec,
)
from openhcs.ui.shared.plate_scope_identity import PipelineScopeIdentity
from openhcs.pyqt_gui.services.ui_bridge_pipeline_editor import (
    PipelineEditorBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_plate_manager import (
    PlateManagerBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_live_overview import (
    LiveOverviewBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_windows import (
    MainWindowBridgeProviderSet,
    ManagedWindowAction,
    UiWidgetTreeResultFactory,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.windows.live_measurements_window import LiveMeasurementTableModel
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    PlateRuntimeIdentity,
    PlateRuntimeProjection,
    PlateRuntimeState,
)
from openhcs.core.progress.debug_projection import DebugRuntimeProjection
from openhcs.core.progress.debug_projection import (
    RuntimeProjectionBuilder,
    RuntimeProjectionSource,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ExecutionBatchRuntime,
    ManagerExecutionState,
)
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerAction
from openhcs.pyqt_gui.widgets.pipeline_editor import (
    PipelineEditorAction,
    PipelineEditorWidget,
)
from openhcs.pyqt_gui.widgets.debug_toolbar import DebugToolbarWidget
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    PipelineDebugPauseBoundaryState,
    PipelineDebugSessionContext,
    PipelineDebugTargetState,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugToolbarAuxiliaryAction,
    PipelineDebugActionDeclarationBase,
    StepDebugAction,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionRoute,
)
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureScope
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.protocols import register_codegen_provider
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.window_code_document import (
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)
from pyqt_reactive.services.widget_tree_projection import (
    WidgetActionKind,
    WidgetDescriptor,
    WidgetRect,
    WidgetTreeProjection,
)
from pyqt_reactive.widgets.editors import simple_code_editor
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.widgets.shared import (
    BaseFormDialog,
    ManagedWindowActionCapabilities,
)
from pyqt_reactive.widgets.shared.list_item_delegate import (
    DIRTY_FIELDS_ROLE,
    OBJECT_STATE_PATH_ROLE,
    SIG_DIFF_FIELDS_ROLE,
)

DOCUMENT_ID = UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value
PLATE_SCOPE_ID = "plate-1"
PLATE_NAME = "plate 1"
ALL_SELECTION_MODE = UiCodeDocumentSelectionMode.ALL.value
SELECTED_SELECTION_MODE = UiCodeDocumentSelectionMode.SELECTED.value
BRIDGE_INSTANCE_ID = "bridge-test"
BRIDGE_AUTH_TOKEN = "secret-token"
VALID_SOURCE = (
    "from openhcs.core.config import GlobalPipelineConfig, PipelineConfig\n"
    f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
    "global_config = GlobalPipelineConfig()\n"
    f"per_plate_configs = {{'{PLATE_SCOPE_ID}': PipelineConfig()}}\n"
    f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
)


class FakeEmptySelectionPolicy(str, Enum):
    ERROR = "error"


def _bridge_server_config(directory_path: Path) -> AgentUiBridgeConfig:
    return AgentUiBridgeConfig(
        port=0,
        descriptor_directory_path=directory_path,
        bridge_instance_id=BRIDGE_INSTANCE_ID,
        auth_token=BRIDGE_AUTH_TOKEN,
    )


def _json_payload_values(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _json_payload_values(value)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _json_payload_values(value)
        return
    yield payload


@dataclass
class Dummy:
    x: int = 1


@dataclass
class FakeOrchestrator:
    state: OrchestratorState


class FakeWindowCodeDocumentDriver(WindowCodeDocumentDriver):
    def __init__(self, source: str) -> None:
        self.source = source

    def read_document(self, clean: bool = True) -> WindowCodeDocument:
        del clean
        return WindowCodeDocument(
            title="View/Edit GlobalPipelineConfig",
            source=self.source,
        )

    def validate_source(self, source: str) -> None:
        compile(source, "<fake-window-code-document>", "exec")

    def apply_source(self, source: str) -> None:
        self.source = source


def adjustable_function(image, threshold: int = 1, enabled: bool = True):
    return image


def replacement_function(image, threshold: int = 2):
    return image


def test_pycodified_window_code_document_honors_clean_flag() -> None:
    driver = PycodifiedObjectCodeDocumentDriver(
        spec=PycodifiedObjectDocumentSpec(
            assignment_name="dummy",
            title="View/Edit Dummy",
            header="# Dummy",
            expected_type=Dummy,
        ),
        current_object=lambda: Dummy(),
        apply_object=lambda _value: None,
    )

    clean_source = driver.read_document(clean=True).source
    full_source = driver.read_document(clean=False).source

    assert "dummy = Dummy()" in clean_source
    assert "x=1" in full_source


def test_pycodified_config_document_delegates_to_config_authority() -> None:
    original = GlobalPipelineConfig(num_workers=2)
    replacement = GlobalPipelineConfig(num_workers=3)
    applied: list[GlobalPipelineConfig] = []
    driver = PycodifiedObjectCodeDocumentDriver(
        spec=PycodifiedConfigDocumentSpec(
            title="View/Edit GlobalPipelineConfig",
            expected_type=GlobalPipelineConfig,
        ),
        current_object=lambda: original,
        apply_object=applied.append,
    )

    document = driver.read_document()
    replacement_source = ConfigDocumentAuthority.render(
        replacement,
        expected_config_type=GlobalPipelineConfig,
    )
    driver.validate_source(document.source)
    driver.apply_source(replacement_source)

    assert (
        ConfigDocumentAuthority.from_source(
            document.source,
            expected_config_type=GlobalPipelineConfig,
        )
        == original
    )
    assert applied == [replacement]


def test_pyqt_codegen_provider_delegates_config_documents() -> None:
    config = GlobalPipelineConfig(num_workers=4)

    source = OpenHCSCodegenProvider().generate_config_code(
        config,
        config_class=GlobalPipelineConfig,
    )

    assert (
        ConfigDocumentAuthority.from_source(
            source,
            expected_config_type=GlobalPipelineConfig,
        )
        == config
    )

    with pytest.raises(TypeError, match="PipelineConfig"):
        OpenHCSCodegenProvider().generate_config_code(
            config,
            config_class=PipelineConfig,
        )


@dataclass(frozen=True, slots=True)
class FakeRow:
    scope_id: str
    name: str
    plate_root: str = f"/tmp/{PLATE_SCOPE_ID}"
    cppipe_path: str | None = None


@dataclass(frozen=True, slots=True)
class FakeCodeDocumentContext:
    source: str
    selected_scope_ids: tuple[str, ...]


class FakeOperations:
    def __init__(self, state: ObjectState | None = None) -> None:
        self.state = state
        self.pre_count = 0
        self.post_count = 0
        self.applied_namespaces: list[dict] = []

    @contextmanager
    def patch_lazy_constructors(self):
        yield

    def migrate_code_namespace(self, code, error, namespace):
        del code, error, namespace
        return None

    def apply_code_namespace(self, namespace: dict) -> bool:
        self.applied_namespaces.append(namespace)
        if self.state is not None:
            self.state.update_parameter("x", self.state.parameters["x"] + 1)
        return True

    def pre_code_execution(self) -> None:
        self.pre_count += 1

    def post_code_execution(self) -> None:
        self.post_count += 1
        ObjectStateRegistry.increment_token()


@dataclass(frozen=True, slots=True)
class FakeButton:
    enabled: bool = True

    def isEnabled(self) -> bool:
        return self.enabled


class FakeServiceAdapter:
    def execute_async_operation(self, operation):
        raise AssertionError(f"Unexpected async operation in test: {operation!r}")


class EmbeddedManagerServiceStub:
    """Minimal service adapter surface needed by AbstractManagerWidget subclasses."""

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

    def execute_async_operation(self, operation):
        return operation()

    def show_error_dialog(self, message: str) -> None:
        self.last_error_message = message


class InlineDispatcher:
    def call(self, callback, *, timeout_ms: int = 5000):
        del timeout_ms
        return callback()

    def post(self, callback) -> None:
        callback()


class CountingDispatcher(InlineDispatcher):
    def __init__(self) -> None:
        self.call_count = 0

    def call(self, callback, *, timeout_ms: int = 5000):
        self.call_count += 1
        return super().call(callback, timeout_ms=timeout_ms)


class QueuedPostDispatcher(InlineDispatcher):
    def __init__(self) -> None:
        self.callbacks = []

    def post(self, callback) -> None:
        self.callbacks.append(callback)

    def run_next(self) -> None:
        self.callbacks.pop(0)()


class QtApplicationAuthority:
    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def test_ui_thread_dispatcher_post_queues_when_called_on_ui_thread() -> None:
    app = QtApplicationAuthority.app()
    dispatcher = UiThreadDispatcher()
    calls: list[str] = []

    dispatcher.post(lambda: calls.append("posted"))

    assert calls == []
    app.processEvents()
    assert calls == ["posted"]


class FakeEmbeddedWindowWidgets:
    def __init__(self) -> None:
        self.plate_manager = QWidget()
        self.pipeline_editor = QWidget()
        self.zmq_manager = QWidget()

    def require_plate_manager(self) -> QWidget:
        return self.plate_manager

    def require_pipeline_editor(self) -> QWidget:
        return self.pipeline_editor

    def require_zmq_manager(self) -> QWidget:
        return self.zmq_manager

    def show_plate_manager(self) -> None:
        self.plate_manager.show()

    def show_pipeline_editor(self) -> None:
        self.pipeline_editor.show()

    def show_zmq_manager(self) -> None:
        self.zmq_manager.show()


class FakeMainWindow:
    def __init__(self) -> None:
        self.embedded_widgets = FakeEmbeddedWindowWidgets()
        self.window_specs = {}


def test_ui_bridge_composition_discovers_new_provider_set_declarations() -> None:
    factory_calls: list[object] = []

    class DiscoveredProviderSet(UiBridgeProviderSetABC):
        registry_key = "test.discovered_provider_set"

        @classmethod
        def for_main_window(cls, main_window):
            factory_calls.append(main_window)
            return cls()

        def register(self, context):
            del context

    class CompositionMainWindow:
        plate_manager_widget = object()
        pipeline_editor_widget = object()

    main_window = CompositionMainWindow()
    try:
        OpenHCSUiBridgeCompositionRoot.for_main_window(main_window)
    finally:
        UiBridgeProviderSetABC.__registry__.pop(
            DiscoveredProviderSet.registry_key,
            None,
        )

    assert factory_calls == [main_window]


def test_ui_bridge_composition_builds_all_registered_provider_sets() -> None:
    QtApplicationAuthority.app()
    main_window = FakeMainWindow()
    main_window.plate_manager_widget = FakePlateManager()
    main_window.pipeline_editor_widget = FakePipelineEditor()

    bridge = OpenHCSUiBridgeCompositionRoot.for_main_window(main_window).build_service()

    assert isinstance(bridge, UiAgentBridgeService)


def test_embedded_code_document_registrations_are_registry_discovered(
    monkeypatch,
) -> None:
    QtApplicationAuthority.app()
    pipeline_driver = FakeWindowCodeDocumentDriver("pipeline_steps = []")
    additional_driver = FakeWindowCodeDocumentDriver("value = 1")
    pipeline_widget = QWidget()
    pipeline_widget.code_document_driver = lambda: pipeline_driver
    additional_widget = QWidget()
    registrations: list[tuple[str, object, object]] = []

    class AdditionalEmbeddedCodeDocumentRegistration(
        EmbeddedCodeDocumentRegistrationABC
    ):
        scope_id = "test_embedded_code_document"

        @classmethod
        def window_for_main_window(cls, main_window):
            return main_window.additional_widget

        @classmethod
        def code_document_driver_for_window(cls, window):
            assert window is additional_widget
            return additional_driver

    def record_registration(
        cls,
        scope_id,
        window,
        navigation_driver=None,
        code_document_driver=None,
    ):
        del cls, navigation_driver
        registrations.append((scope_id, window, code_document_driver))

    monkeypatch.setattr(
        WindowManager,
        "register",
        classmethod(record_registration),
    )
    main_window = type(
        "EmbeddedMainWindow",
        (),
        {
            "pipeline_editor_widget": pipeline_widget,
            "additional_widget": additional_widget,
        },
    )()
    try:
        EmbeddedCodeDocumentRegistrationABC.register_all_for_main_window(main_window)
    finally:
        EmbeddedCodeDocumentRegistrationABC.__registry__.pop(
            AdditionalEmbeddedCodeDocumentRegistration.scope_id,
            None,
        )

    assert (
        OpenHCSUiWindowId.pipeline_editor,
        pipeline_widget,
        pipeline_driver,
    ) in registrations
    assert (
        AdditionalEmbeddedCodeDocumentRegistration.scope_id,
        additional_widget,
        additional_driver,
    ) in registrations


class FakeManagedFormWindow(BaseFormDialog):
    def __init__(self, scope_id: str) -> None:
        super().__init__()
        self.scope_id = scope_id
        self.save_count = 0
        self.saved_close_window: bool | None = None

    def managed_window_action_capabilities(
        self,
    ) -> ManagedWindowActionCapabilities:
        return ManagedWindowActionCapabilities(
            save_and_close=True,
            save_without_close=True,
            discard_and_close=True,
        )

    def agent_save_managed_window(self, *, close_window: bool) -> None:
        self.save_count += 1
        self.saved_close_window = close_window
        self.finish_managed_save(close_window=close_window)


class FakePlateManager:
    BUTTON_CONFIGS = [
        ("Code", PlateManagerAction.CODE_PLATE.value, "Generate Python code"),
    ]
    ACTION_ROUTES = {
        PlateManagerAction.CODE_PLATE: WidgetActionRoute(
            PlateManagerAction.CODE_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }

    def __init__(
        self,
        *,
        selected: tuple[FakeRow, ...] = (),
        plates: tuple[FakeRow, ...] = (FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
        operations: FakeOperations | None = None,
        pipeline_steps: tuple[FunctionStep, ...] = (),
    ) -> None:
        self.selected = list(selected)
        self.plates = list(plates)
        self.operations = operations or FakeOperations()
        self.pipeline_steps = pipeline_steps
        self.execution_state = ManagerExecutionState.IDLE
        self.plate_execution_ids = {}
        self.runtime_progress_projection = ExecutionRuntimeProjection()
        self.live_measurement_model = LiveMeasurementTableModel()
        self.plate_terminal_activity_status = ExecutionBatchRuntime()
        self.plate_init_pending = set()
        self.plate_compile_pending = set()
        self.plate_compiled_data = {}
        self.execution_server_info = None
        self.global_config = GlobalPipelineConfig()
        self.service_adapter = FakeServiceAdapter()
        self.buttons = {
            PlateManagerAction.CODE_PLATE.value: FakeButton(),
        }
        self.code_action_count = 0

    def get_selected_items(self):
        return list(self.selected)

    def orchestrator_code_document_context(
        self,
        *,
        selection_mode: str = SELECTED_SELECTION_MODE,
        empty_selection_policy: str = FakeEmptySelectionPolicy.ERROR.value,
    ) -> FakeCodeDocumentContext:
        rows_by_mode = {
            UiCodeDocumentSelectionMode.ALL: self.plates,
            UiCodeDocumentSelectionMode.SELECTED: self.selected,
        }
        rows = rows_by_mode[UiCodeDocumentSelectionMode(selection_mode)]
        if (
            not rows
            and FakeEmptySelectionPolicy(empty_selection_policy)
            is FakeEmptySelectionPolicy.ERROR
        ):
            raise ValueError("No plates selected.")
        return FakeCodeDocumentContext(
            source=VALID_SOURCE,
            selected_scope_ids=tuple(row.scope_id for row in rows),
        )

    def code_document_execution_operations(self) -> FakeOperations:
        return self.operations

    def _get_current_pipeline_definition(self, plate_path: str):
        del plate_path
        return list(self.pipeline_steps)

    def action_code_plate(self) -> None:
        self.code_action_count += 1

    def debug_session_for_plate(self, plate_path: str):
        del plate_path
        return None

    def debug_terminal_summary_for_plate(self, plate_path: str):
        del plate_path
        return None

    def debug_session_context_for_plate(
        self,
        plate_path: str,
    ) -> PipelineDebugSessionContext:
        target = PipelineDebugTargetState(
            current_plate_scope_id=plate_path,
            pipeline_scope_id=PipelineScopeIdentity.from_plate_scope(
                plate_path,
            ).scope_id,
            initialized=True,
            compiled=plate_path in self.plate_compiled_data,
            terminal_status=None,
        )
        return PipelineDebugSessionContext(
            target=target,
            session=self.debug_session_for_plate(plate_path),
            terminal_summary=self.debug_terminal_summary_for_plate(plate_path),
            pause_boundaries=PipelineDebugPauseBoundaryState(),
            manager_execution_state=self.execution_state,
        )


class FakePipelineEditor:
    BUTTON_CONFIGS = PipelineEditorWidget.BUTTON_CONFIGS
    STATE_BINDING = PipelineEditorWidget.STATE_BINDING
    ACTION_ROUTES = {
        PipelineEditorAction.ADD_STEP: WidgetActionRoute(
            PipelineEditorAction.ADD_STEP,
            lambda widget: widget.action_add,
        ),
        PipelineEditorAction.DELETE_STEP: WidgetActionRoute(
            PipelineEditorAction.DELETE_STEP,
            lambda widget: widget.action_delete,
        ),
        PipelineEditorAction.EDIT_STEP: WidgetActionRoute(
            PipelineEditorAction.EDIT_STEP,
            lambda widget: widget.action_edit,
        ),
        PipelineEditorAction.AUTO_LOAD_PIPELINE: WidgetActionRoute(
            PipelineEditorAction.AUTO_LOAD_PIPELINE,
            lambda widget: widget.action_auto_load_pipeline,
        ),
        PipelineEditorAction.CODE_PIPELINE: WidgetActionRoute(
            PipelineEditorAction.CODE_PIPELINE,
            lambda widget: widget.action_code_pipeline,
        ),
    }

    def __init__(
        self,
        *,
        current_plate: str = PLATE_SCOPE_ID,
        selected_indices: tuple[int, ...] = (0,),
    ) -> None:
        self.current_plate = current_plate
        self.pipeline_steps = [
            FunctionStep(func=lambda image: image, name="step_one"),
            FunctionStep(func=lambda image: image, name="step_two"),
        ]
        self.selected_indices = selected_indices
        self.service_adapter = FakeServiceAdapter()
        self.buttons = {
            action.value: FakeButton(enabled=True) for action in self.ACTION_ROUTES
        }
        self.debug_toolbar = DebugToolbarWidget()
        self.debug_session_state = None
        self.debug_terminal_summary = None
        self.debug_runtime_projection_state = DebugRuntimeProjection.empty()
        self.initialized = bool(current_plate)
        self.compiled = bool(current_plate)
        self.manager_execution_state = ManagerExecutionState.IDLE
        self.terminal_status = None
        self.debug_toolbar.set_debug_session_context(self.debug_session_context())
        self.debug_workflow = FakePipelineDebugWorkflow()
        self.add_count = 0
        self.delete_count = 0
        self.edit_count = 0
        self.auto_count = 0
        self.code_count = 0

    def get_selected_items(self):
        return [self.pipeline_steps[index] for index in self.selected_indices]

    def _get_item_scope_id(self, item: FunctionStep, index: int) -> str:
        del item
        return f"scope-from-manager-hook-{index}"

    def selected_step_scope_ids(self) -> tuple[str, ...]:
        selected = set(self.selected_indices)
        return tuple(
            self._get_item_scope_id(step, index)
            for index, step in enumerate(self.pipeline_steps)
            if index in selected
        )

    def debug_session_context(self) -> PipelineDebugSessionContext:
        target = None
        if self.current_plate:
            target = PipelineDebugTargetState(
                current_plate_scope_id=self.current_plate,
                pipeline_scope_id=PipelineScopeIdentity.from_plate_scope(
                    self.current_plate
                ).scope_id,
                initialized=self.initialized,
                compiled=self.compiled,
                terminal_status=self.terminal_status,
            )
        return PipelineDebugSessionContext(
            target=target,
            session=self.debug_session_state,
            terminal_summary=self.debug_terminal_summary,
            pause_boundaries=PipelineDebugPauseBoundaryState(
                pause_step_indices=tuple(
                    index
                    for index, step in enumerate(self.pipeline_steps)
                    if step.debug_pause
                )
            ),
            manager_execution_state=self.manager_execution_state,
        )

    def debug_runtime_projection(self) -> DebugRuntimeProjection:
        return self.debug_runtime_projection_state

    def action_add(self) -> None:
        self.add_count += 1

    def action_delete(self) -> None:
        self.delete_count += 1

    def action_edit(self) -> None:
        self.edit_count += 1

    def action_auto_load_pipeline(self) -> None:
        self.auto_count += 1

    def action_code_pipeline(self) -> None:
        self.code_count += 1


class FakePipelineDebugWorkflow:
    def __init__(self) -> None:
        self.commands: list[DebugCommand] = []
        self.runtime_inspections = 0

    def handle_command(self, command: DebugCommand) -> None:
        self.commands.append(command)

    def show_runtime_inspection(self) -> None:
        self.runtime_inspections += 1


@pytest.fixture(autouse=True)
def reset_object_state_registry():
    ObjectStateRegistry._states.clear()
    ObjectStateRegistry._time_travel_limbo.clear()
    ObjectStateRegistry._graveyard.clear()
    ObjectStateRegistry._snapshots.clear()
    ObjectStateRegistry._timelines.clear()
    ObjectStateRegistry._current_timeline = "main"
    ObjectStateRegistry._current_head = None
    ObjectStateRegistry._in_time_travel = False
    ObjectStateRegistry._atomic_depth = 0
    ObjectStateRegistry._atomic_label = None
    ObjectStateRegistry._atomic_triggering_scope = None
    ObjectStateRegistry._token = 0


def test_atomic_success_does_not_record_snapshot_on_failure() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    with pytest.raises(RuntimeError):
        with ObjectStateRegistry.atomic_success("edit failing", state.scope_id):
            state.update_parameter("x", 2)
            raise RuntimeError("boom")

    assert ObjectStateRegistry.get_branch_history() == []


def test_selected_read_fails_loudly_when_no_plate_is_selected() -> None:
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager())
    )

    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )

    assert document.errors
    assert document.errors[0].code == "no_selection"
    assert "selection_mode='all'" in document.errors[0].hint
    assert document.selected_scope_ids == ()


def test_selected_plate_document_does_not_fall_back_to_all_rows() -> None:
    other_row = FakeRow("plate-2", "plate 2")
    manager = FakePlateManager(
        selected=(),
        plates=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME), other_row),
    )
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    selected_document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )
    all_document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    selected_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )
    all_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert selected_document.errors[0].code == "no_selection"
    assert selected_document.selected_scope_ids == ()
    assert selected_document.summary.current_selection_count == 0
    assert all_document.errors == ()
    assert all_document.selected_scope_ids == (PLATE_SCOPE_ID, other_row.scope_id)
    assert all_document.summary.current_selection_count == 0
    assert selected_state.selected_scope_ids == ()
    assert selected_state.summary.current_selection_count == 0
    assert selected_state.payload["rows"] == []
    assert all_state.selected_scope_ids == ()
    assert [row["selected"] for row in all_state.payload["rows"]] == [False, False]
    assert selected_state.current_revision_token != all_state.current_revision_token

    manager.selected = [other_row]
    newly_selected_document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )
    newly_selected_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )

    assert newly_selected_document.errors == ()
    assert newly_selected_document.selected_scope_ids == (other_row.scope_id,)
    assert newly_selected_document.summary.current_selection_count == 1
    assert newly_selected_state.selected_scope_ids == (other_row.scope_id,)
    assert newly_selected_state.summary.current_selection_count == 1
    assert [row["plate_scope_id"] for row in newly_selected_state.payload["rows"]] == [
        other_row.scope_id
    ]
    assert newly_selected_state.payload["rows"][0]["selected"] is True
    assert (
        newly_selected_state.current_revision_token
        != selected_state.current_revision_token
    )


def test_all_plate_document_context_failure_is_not_reported_as_no_selection() -> None:
    class FailingPlateManager(FakePlateManager):
        def orchestrator_code_document_context(self, **kwargs):
            del kwargs
            raise RuntimeError("context construction failed")

    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FailingPlateManager(selected=()))
    )

    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert document.errors[0].code == "ui_code_document_read_failed"
    assert document.errors[0].exception_type == "RuntimeError"


def test_object_state_provider_reads_and_applies_child_function_scope() -> None:
    register_codegen_provider(OpenHCSCodegenProvider())
    parent_scope = "plate::functionstep_0"
    child_token = "runtimecallable_0"
    child_scope = f"{parent_scope}::{child_token}"
    parent_state = ObjectState(
        FunctionStep(
            func=(adjustable_function, {"threshold": 3, "enabled": True}),
            name="Adjust",
        ),
        scope_id=parent_scope,
    )
    parent_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] = [child_token]
    child_state = ObjectState(
        object_instance=adjustable_function,
        scope_id=child_scope,
        parent_state=parent_state,
        exclude_params=["image"],
        initial_values={"threshold": 3, "enabled": True},
    )
    ObjectStateRegistry.register(parent_state, _skip_snapshot=True)
    ObjectStateRegistry.register(child_state, _skip_snapshot=True)
    snapshot_provider = UiObjectStateSnapshotProvider()
    provider = ObjectStateScopeCodeDocumentProvider(snapshot_provider)
    document_id = f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}{child_scope}"

    document = provider.read(UiCodeDocumentRequest(document_id=document_id))
    full_document = provider.read(
        UiCodeDocumentRequest(document_id=document_id, clean=False)
    )

    assert document.errors == ()
    assert "pattern" in document.source
    assert "adjustable_function" in document.source
    assert "'threshold': 3" in document.source
    assert "'enabled': True" not in document.source
    assert "'threshold': 3" in full_document.source
    assert "'enabled': True" in full_document.source

    source = (
        f"from {adjustable_function.__module__} import adjustable_function\n"
        "pattern = (adjustable_function, "
        "{'threshold': 9, 'enabled': False})\n"
    )
    result = provider.apply(
        UiCodeDocumentApplyRequest(
            document_id=document_id,
            source=source,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    updated_child = ObjectStateRegistry.get_by_scope(child_scope)
    assert result.applied
    assert result.receipt.accepted
    assert result.receipt.bridge_operation_id is None
    assert updated_child is not None
    assert updated_child.parameters["threshold"] == 9
    assert updated_child.parameters["enabled"] is False
    assert parent_state.parameters["func"][1] == {
        "threshold": 9,
        "enabled": False,
    }

    threshold_only_source = (
        f"from {adjustable_function.__module__} import adjustable_function\n"
        "pattern = (adjustable_function, {'threshold': 10})\n"
    )
    threshold_only_result = provider.apply(
        UiCodeDocumentApplyRequest(
            document_id=document_id,
            source=threshold_only_source,
            base_revision_token=result.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    threshold_only_child = ObjectStateRegistry.get_by_scope(child_scope)
    threshold_only_document = provider.read(
        UiCodeDocumentRequest(document_id=document_id)
    )
    assert threshold_only_result.applied
    assert threshold_only_child is not None
    assert threshold_only_child.parameters["threshold"] == 10
    assert threshold_only_child.parameters["enabled"] is True
    assert parent_state.parameters["func"][1] == {"threshold": 10}
    assert "'enabled': None" not in threshold_only_document.source
    assert "'enabled': True" not in threshold_only_document.source

    replacement_source = (
        f"from {replacement_function.__module__} import replacement_function\n"
        "pattern = (replacement_function, {'threshold': 12})\n"
    )
    replacement_result = provider.apply(
        UiCodeDocumentApplyRequest(
            document_id=document_id,
            source=replacement_source,
            base_revision_token=threshold_only_result.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    replacement_child = ObjectStateRegistry.get_by_scope(child_scope)
    assert replacement_result.applied
    assert replacement_result.receipt.accepted
    assert replacement_child is not None
    assert replacement_child.object_instance is replacement_function
    assert replacement_child.parameters["threshold"] == 12
    assert parent_state.parameters["func"][0] is replacement_function
    assert parent_state.parameters["func"][1] == {"threshold": 12}
    assert [
        snapshot.label for snapshot in ObjectStateRegistry.get_branch_history()
    ] == [
        "init",
        f"edit {document_id} via MCP [{child_scope}]",
        f"edit {document_id} via MCP [{child_scope}]",
        f"edit {document_id} via MCP [{child_scope}]",
    ]


def test_object_state_code_document_noop_apply_does_not_record_snapshot() -> None:
    register_codegen_provider(OpenHCSCodegenProvider())
    parent_scope = "plate::functionstep_0"
    child_token = "runtimecallable_0"
    child_scope = f"{parent_scope}::{child_token}"
    parent_state = ObjectState(
        FunctionStep(
            func=(adjustable_function, {"threshold": 3}),
            name="Adjust",
        ),
        scope_id=parent_scope,
    )
    parent_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] = [child_token]
    child_state = ObjectState(
        object_instance=adjustable_function,
        scope_id=child_scope,
        parent_state=parent_state,
        exclude_params=["image"],
        initial_values={"threshold": 3},
    )
    ObjectStateRegistry.register(parent_state, _skip_snapshot=True)
    ObjectStateRegistry.register(child_state, _skip_snapshot=True)
    snapshot_provider = UiObjectStateSnapshotProvider()
    provider = ObjectStateScopeCodeDocumentProvider(snapshot_provider)
    document_id = f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}{child_scope}"

    document = provider.read(UiCodeDocumentRequest(document_id=document_id))
    result = provider.apply(
        UiCodeDocumentApplyRequest(
            document_id=document_id,
            source=document.source,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert not result.applied
    assert result.outcome == "unchanged"
    assert result.receipt.accepted
    assert result.current_revision_token == document.current_revision_token
    assert result.new_revision_token == document.current_revision_token
    assert result.current_snapshot == document.current_snapshot
    assert ObjectStateRegistry.get_branch_history() == []
    assert parent_state.parameters["func"][1] == {"threshold": 3}


def test_listed_embedded_window_routes_support_widget_tree_projection() -> None:
    QtApplicationAuthority.app()
    main_window = FakeMainWindow()
    main_window.embedded_widgets.plate_manager.setObjectName("plate_manager")
    main_window.embedded_widgets.plate_manager.show()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(main_window).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    windows = bridge.list_windows()
    window_ids = tuple(summary.window_id for summary in windows.windows)
    assert "plate_manager" in window_ids

    widget_tree = bridge.widget_tree(
        UiWidgetTreeRequest(
            window_id="plate_manager",
            open_policy=UiWindowOpenPolicy(),
            include_tree=True,
        )
    )

    assert widget_tree.errors == ()
    assert widget_tree.projected is True
    assert widget_tree.root is not None
    assert widget_tree.root.object_name == "plate_manager"
    assert widget_tree.include_tree is True


def test_widget_tree_item_rows_carry_shared_object_state_roles() -> None:
    app = QtApplicationAuthority.app()
    main_window = FakeMainWindow()
    pipeline_editor = QWidget()
    pipeline_editor.setObjectName("pipeline_editor")
    layout = QVBoxLayout(pipeline_editor)
    steps = QListWidget(pipeline_editor)
    item = QListWidgetItem("1. Normalize")
    item.setData(OBJECT_STATE_PATH_ROLE, "plate-1::functionstep_0")
    item.setData(DIRTY_FIELDS_ROLE, {"name"})
    item.setData(SIG_DIFF_FIELDS_ROLE, {"func"})
    steps.addItem(item)
    layout.addWidget(steps)
    pipeline_editor.show()
    app.processEvents()
    main_window.embedded_widgets.pipeline_editor = pipeline_editor

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(main_window).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id="pipeline_editor",
                open_policy=UiWindowOpenPolicy(),
                max_nodes=10,
            )
        )

        step_action = next(
            action
            for action in widget_tree.actionable_widgets
            if action.label == "1. Normalize"
        )
        assert step_action.action_role == "item_select"
        assert step_action.object_state_scope_id == "plate-1::functionstep_0"
        assert step_action.field_path is None
        assert step_action.dirty is True
        assert step_action.signature_diff is True
        assert step_action.semantic_markers == ("*", "_")
    finally:
        pipeline_editor.close()


def test_embedded_manager_window_summary_carries_shared_row_semantics() -> None:
    app = QtApplicationAuthority.app()
    main_window = FakeMainWindow()
    pipeline_editor = PipelineEditorWidget(EmbeddedManagerServiceStub())
    pipeline_editor.setObjectName("pipeline_editor")
    row = QListWidgetItem("1. Normalize")
    row.setData(OBJECT_STATE_PATH_ROLE, "plate-1::functionstep_0")
    row.setData(DIRTY_FIELDS_ROLE, {"name", "napari_streaming_config.enabled"})
    row.setData(SIG_DIFF_FIELDS_ROLE, {"func"})
    assert pipeline_editor.item_list is not None
    pipeline_editor.item_list.addItem(row)
    pipeline_editor.show()
    app.processEvents()
    main_window.embedded_widgets.pipeline_editor = pipeline_editor

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(main_window).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        summary = next(
            window
            for window in windows.windows
            if window.window_id == "pipeline_editor"
        )

        assert summary.dirty is True
        assert summary.signature_diff is True
        assert summary.dirty_field_count == 2
        assert summary.signature_diff_field_count == 1
        assert summary.semantic_markers == ("*", "_")
    finally:
        pipeline_editor.close()


def test_widget_tree_projection_defaults_bound_actionable_paths() -> None:
    def descriptor(
        path,
        *,
        class_name: str,
        text: str | None = None,
        actionable: bool = False,
        visible: bool = True,
        children: tuple[WidgetDescriptor, ...] = (),
    ) -> WidgetDescriptor:
        return WidgetDescriptor(
            path=path,
            path_id="root" if not path else ".".join(str(index) for index in path),
            child_index=None if not path else path[-1],
            class_name=class_name,
            object_name="",
            accessible_name="",
            accessible_description="",
            visible=visible,
            enabled=True,
            geometry=WidgetRect(0, 0, 10, 10),
            global_geometry=WidgetRect(0, 0, 10, 10),
            tool_tip="",
            status_tip="",
            whats_this="",
            window_title="",
            text=text,
            text_truncated=False,
            title=None,
            action_kinds=(WidgetActionKind.BUTTON,) if actionable else (),
            clickable=actionable,
            actionable=actionable,
            checkable=False if actionable else None,
            checked=None,
            current_index=None,
            current_text=None,
            item_count=None,
            children=children,
        )

    root = descriptor(
        (),
        class_name="ConfigWindow",
        children=(
            descriptor((0,), class_name="QLabel", text="noise"),
            descriptor(
                (1,),
                class_name="QWidget",
                children=(
                    descriptor(
                        (1, 0), class_name="QPushButton", text="Save", actionable=True
                    ),
                    descriptor(
                        (1, 1), class_name="QPushButton", text="Cancel", actionable=True
                    ),
                ),
            ),
            descriptor((2,), class_name="QScrollBar"),
        ),
    )

    result = UiWidgetTreeResultFactory.from_projection(
        UiWidgetTreeRequest(
            window_id="global_config",
            open_policy=UiWindowOpenPolicy(),
            max_nodes=3,
        ),
        UiWindowSummary(
            schema_version="openhcs.agent.v1",
            identity=UiWindowIdentity(window_id="global_config"),
            window_kind="scope",
            title="Configuration - GlobalPipelineConfig",
            visible=True,
            focusable=True,
            manager_scope=UiWindowManagerScope(""),
        ),
        WidgetTreeProjection(root=root, widget_count=6, actionable_count=2),
    )

    assert result.widget_count == 6
    assert result.actionable_count == 2
    assert result.include_tree is False
    assert result.root is None
    assert result.returned_widget_count == 0
    assert result.returned_actionable_count == 2
    assert result.actionable_widgets_truncated is False
    assert [widget.label for widget in result.actionable_widgets] == ["Save", "Cancel"]

    tree_result = UiWidgetTreeResultFactory.from_projection(
        UiWidgetTreeRequest(
            window_id="global_config",
            open_policy=UiWindowOpenPolicy(),
            include_tree=True,
            max_nodes=3,
        ),
        UiWindowSummary(
            schema_version="openhcs.agent.v1",
            identity=UiWindowIdentity(window_id="global_config"),
            window_kind="scope",
            title="Configuration - GlobalPipelineConfig",
            visible=True,
            focusable=True,
            manager_scope=UiWindowManagerScope(""),
        ),
        WidgetTreeProjection(root=root, widget_count=6, actionable_count=2),
    )

    assert tree_result.returned_widget_count == 3
    assert tree_result.tree_truncated is True
    assert tree_result.root is not None
    assert [child.class_name for child in tree_result.root.children] == ["QWidget"]
    assert [child.text for child in tree_result.root.children[0].children] == ["Save"]

    visible_first_result = UiWidgetTreeResultFactory.from_projection(
        UiWidgetTreeRequest(
            window_id="global_config",
            open_policy=UiWindowOpenPolicy(),
            actionable_only=False,
            include_tree=True,
            max_nodes=2,
        ),
        tree_result.summary,
        WidgetTreeProjection(
            root=descriptor(
                (),
                class_name="ConfigWindow",
                children=(
                    descriptor((0,), class_name="HiddenPanel", visible=False),
                    descriptor((1,), class_name="VisiblePanel"),
                ),
            ),
            widget_count=3,
            actionable_count=0,
        ),
    )

    assert visible_first_result.root is not None
    assert [child.class_name for child in visible_first_result.root.children] == [
        "VisiblePanel"
    ]


def test_widget_tree_action_summaries_keep_reset_actions_with_field_context() -> None:
    def descriptor(
        path,
        *,
        class_name: str,
        text: str | None = None,
        object_name: str = "",
        action_kinds: tuple[WidgetActionKind, ...] = (),
        children: tuple[WidgetDescriptor, ...] = (),
    ) -> WidgetDescriptor:
        actionable = bool(action_kinds)
        return WidgetDescriptor(
            path=path,
            path_id="root" if not path else ".".join(str(index) for index in path),
            child_index=None if not path else path[-1],
            class_name=class_name,
            object_name=object_name,
            accessible_name="",
            accessible_description="",
            visible=True,
            enabled=True,
            geometry=WidgetRect(0, 0, 10, 10),
            global_geometry=WidgetRect(0, 0, 10, 10),
            tool_tip="",
            status_tip="",
            whats_this="",
            window_title="",
            text=text,
            text_truncated=False,
            title=None,
            action_kinds=action_kinds,
            clickable=actionable,
            actionable=actionable,
            checkable=False if WidgetActionKind.BUTTON in action_kinds else None,
            checked=None,
            current_index=None,
            current_text=None,
            item_count=None,
            children=children,
        )

    root = descriptor(
        (),
        class_name="ConfigWindow",
        children=(
            descriptor(
                (0,),
                class_name="ResponsiveParameterRow",
                children=(
                    descriptor((0, 0), class_name="QLabel", text="Microscope"),
                    descriptor(
                        (0, 1),
                        class_name="NoScrollComboBox",
                        text="auto",
                        action_kinds=(WidgetActionKind.CHOICE,),
                    ),
                    descriptor(
                        (0, 2),
                        class_name="QPushButton",
                        text="Reset",
                        object_name="well_filter_config_reset",
                        action_kinds=(WidgetActionKind.BUTTON,),
                    ),
                ),
            ),
        ),
    )

    result = UiWidgetTreeResultFactory.from_projection(
        UiWidgetTreeRequest(
            window_id="global_config",
            open_policy=UiWindowOpenPolicy(),
            max_nodes=10,
        ),
        UiWindowSummary(
            schema_version="openhcs.agent.v1",
            identity=UiWindowIdentity(window_id="global_config"),
            window_kind="scope",
            title="Configuration - GlobalPipelineConfig",
            visible=True,
            focusable=True,
            manager_scope=UiWindowManagerScope(""),
        ),
        WidgetTreeProjection(root=root, widget_count=4, actionable_count=2),
    )

    assert result.actionable_count == 2
    assert result.returned_actionable_count == 2
    assert [widget.label for widget in result.actionable_widgets] == ["auto", "Reset"]
    assert [widget.context_label for widget in result.actionable_widgets] == [
        "Microscope",
        "Microscope",
    ]
    assert result.actionable_widgets[1].action_role == "field_reset"


def test_widget_tree_action_context_skips_disabled_action_siblings() -> None:
    def descriptor(
        path,
        *,
        class_name: str,
        text: str | None = None,
        action_kinds: tuple[WidgetActionKind, ...] = (),
        actionable: bool | None = None,
        children: tuple[WidgetDescriptor, ...] = (),
    ) -> WidgetDescriptor:
        resolved_actionable = bool(action_kinds) if actionable is None else actionable
        return WidgetDescriptor(
            path=path,
            path_id="root" if not path else ".".join(str(index) for index in path),
            child_index=None if not path else path[-1],
            class_name=class_name,
            object_name="",
            accessible_name="",
            accessible_description="",
            visible=True,
            enabled=resolved_actionable,
            geometry=WidgetRect(0, 0, 10, 10),
            global_geometry=WidgetRect(0, 0, 10, 10),
            tool_tip="",
            status_tip="",
            whats_this="",
            window_title="",
            text=text,
            text_truncated=False,
            title=None,
            action_kinds=action_kinds,
            clickable=resolved_actionable,
            actionable=resolved_actionable,
            checkable=False if WidgetActionKind.BUTTON in action_kinds else None,
            checked=None,
            current_index=None,
            current_text=None,
            item_count=None,
            children=children,
        )

    root = descriptor(
        (),
        class_name="PipelineEditorWidget",
        children=(
            descriptor(
                (0,),
                class_name="ButtonPanel",
                children=(
                    descriptor(
                        (0, 0),
                        class_name="QPushButton",
                        text="Add",
                        action_kinds=(WidgetActionKind.BUTTON,),
                        actionable=False,
                    ),
                    descriptor(
                        (0, 1),
                        class_name="QPushButton",
                        text="Del",
                        action_kinds=(WidgetActionKind.BUTTON,),
                    ),
                ),
            ),
        ),
    )

    result = UiWidgetTreeResultFactory.from_projection(
        UiWidgetTreeRequest(
            window_id="pipeline_editor",
            open_policy=UiWindowOpenPolicy(),
            max_nodes=10,
        ),
        UiWindowSummary(
            schema_version="openhcs.agent.v1",
            identity=UiWindowIdentity(window_id="pipeline_editor"),
            window_kind="embedded",
            title="Pipeline Editor",
            visible=True,
            focusable=True,
            manager_scope=UiWindowManagerScope("pipeline_editor"),
        ),
        WidgetTreeProjection(root=root, widget_count=3, actionable_count=1),
    )

    assert len(result.actionable_widgets) == 1
    assert result.actionable_widgets[0].label == "Del"
    assert result.actionable_widgets[0].context_label is None


def test_listed_qt_top_level_window_supports_operation_request_resolution() -> None:
    app = QtApplicationAuthority.app()
    top_level = QWidget()
    top_level.setWindowTitle("Agent visible top-level")
    top_level.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        summaries = tuple(
            summary
            for summary in windows.windows
            if summary.title == "Agent visible top-level"
        )
        assert len(summaries) == 1
        window_id = summaries[0].window_id

        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                include_tree=True,
            )
        )
        close_result = bridge.close_window(UiWindowCloseRequest(window_id=window_id))

        assert widget_tree.errors == ()
        assert widget_tree.projected is True
        assert widget_tree.root is not None
        assert widget_tree.root.class_name == "QWidget"
        assert widget_tree.include_tree is True
        assert close_result.errors == ()
        assert close_result.closed is True
    finally:
        top_level.close()


def test_projected_widget_action_invokes_live_button_by_path_id() -> None:
    app = QtApplicationAuthority.app()
    top_level = QWidget()
    top_level.setWindowTitle("Agent actionable top-level")
    button = QPushButton("Generate", top_level)
    button.setObjectName("generate_button")
    button.resize(120, 32)
    clicked: list[bool] = []
    button.clicked.connect(lambda: clicked.append(True))
    top_level.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        summaries = tuple(
            summary
            for summary in windows.windows
            if summary.title == "Agent actionable top-level"
        )
        assert len(summaries) == 1
        window_id = summaries[0].window_id

        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                include_tree=True,
            )
        )
        action_summary = next(
            action
            for action in widget_tree.actionable_widgets
            if action.object_name == "generate_button"
        )

        result = bridge.invoke_widget_action(
            UiWidgetActionInvokeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                path_id=action_summary.path_id,
                action_kind="button",
            )
        )
        app.processEvents()

        assert result.errors == ()
        assert result.invoked is True
        assert result.receipt.accepted is True
        assert result.receipt.bridge_operation_id is not None
        assert result.summary is not None
        assert result.summary.object_name == "generate_button"
        assert clicked == [True]
    finally:
        top_level.close()


def test_projected_widget_action_selects_live_item_view_row_by_path_id() -> None:
    app = QtApplicationAuthority.app()
    top_level = QWidget()
    top_level.setWindowTitle("Agent selectable list")
    list_widget = QListWidget(top_level)
    list_widget.setObjectName("pipeline_steps")
    list_widget.resize(240, 80)
    list_widget.addItem(QListWidgetItem("segment_cells"))
    list_widget.addItem(QListWidgetItem("measure_intensity"))
    top_level.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        window_id = next(
            summary.window_id
            for summary in windows.windows
            if summary.title == "Agent selectable list"
        )

        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                include_tree=True,
                max_nodes=20,
            )
        )
        action_summary = next(
            action
            for action in widget_tree.actionable_widgets
            if action.label == "measure_intensity"
        )

        result = bridge.invoke_widget_action(
            UiWidgetActionInvokeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                path_id=action_summary.path_id,
            )
        )
        app.processEvents()

        assert result.errors == ()
        assert result.invoked is True
        assert result.action_kind == WidgetActionKind.ITEM_SELECT.value
        assert result.summary is not None
        assert result.summary.action_kinds == (WidgetActionKind.ITEM_SELECT.value,)
        assert list_widget.currentRow() == 1
        assert list_widget.currentItem().text() == "measure_intensity"

        unavailable = bridge.invoke_widget_action(
            UiWidgetActionInvokeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                path_id=action_summary.path_id,
                action_kind=WidgetActionKind.BUTTON.value,
            )
        )

        assert unavailable.invoked is False
        assert unavailable.summary is not None
        assert unavailable.errors[0].code == "ui_widget_action_kind_unavailable"
        assert "item_select" in (unavailable.errors[0].hint or "")
    finally:
        top_level.close()


def test_projected_widget_action_selects_standalone_tab_bar_by_index() -> None:
    app = QtApplicationAuthority.app()
    top_level = QWidget()
    top_level.setWindowTitle("Agent selectable tabs")
    tab_bar = QTabBar(top_level)
    tab_bar.setObjectName("editor_tab_bar")
    tab_bar.addTab("Step Settings")
    tab_bar.addTab("Function Pattern")
    tab_bar.addTab("Artifacts")
    top_level.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        window_id = next(
            summary.window_id
            for summary in bridge.list_windows().windows
            if summary.title == "Agent selectable tabs"
        )
        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                include_tree=True,
            )
        )
        action_summary = next(
            action
            for action in widget_tree.actionable_widgets
            if action.object_name == "editor_tab_bar"
        )
        assert action_summary.item_texts == (
            "Step Settings",
            "Function Pattern",
            "Artifacts",
        )

        result = bridge.invoke_widget_action(
            UiWidgetActionInvokeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                path_id=action_summary.path_id,
                action_kind=WidgetActionKind.TAB_SELECTOR.value,
                target_index=2,
            )
        )
        app.processEvents()

        assert result.errors == ()
        assert result.invoked is True
        assert result.action_kind == WidgetActionKind.TAB_SELECTOR.value
        assert tab_bar.currentIndex() == 2

        invalid = bridge.invoke_widget_action(
            UiWidgetActionInvokeRequest(
                window_id=window_id,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                path_id=action_summary.path_id,
                action_kind=WidgetActionKind.TAB_SELECTOR.value,
                target_index=3,
            )
        )
        assert invalid.invoked is False
        assert invalid.errors[0].code == "ui_widget_tab_index_invalid"
    finally:
        top_level.close()


def test_legacy_empty_scope_window_catalogs_as_global_config(tmp_path: Path) -> None:
    app = QtApplicationAuthority.app()
    global_window = QWidget()
    global_window.setWindowTitle("Configuration - GlobalPipelineConfig")
    WindowManager.register("", global_window)
    global_window.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        summaries = tuple(
            summary
            for summary in windows.windows
            if summary.title == "Configuration - GlobalPipelineConfig"
        )

        assert len(summaries) == 1
        assert summaries[0].window_id == OpenHCSUiWindowId.global_config
        assert summaries[0].manager_scope is not None
        assert summaries[0].manager_scope.value == ""

        focus_result = bridge.focus_window(
            UiWindowFocusRequest(
                window_id=OpenHCSUiWindowId.global_config,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
            )
        )
        widget_tree = bridge.widget_tree(
            UiWidgetTreeRequest(
                window_id=OpenHCSUiWindowId.global_config,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
                include_tree=True,
            )
        )
        snapshot = bridge.snapshot_window(
            UiWindowSnapshotRequest(
                window_id=OpenHCSUiWindowId.global_config,
                output_dir_path=str(tmp_path),
                capture_scope=WindowSnapshotCaptureScope.WINDOW,
                open_policy=UiWindowOpenPolicy(create_if_missing=False),
            )
        )
        close_result = bridge.close_window(
            UiWindowCloseRequest(window_id=OpenHCSUiWindowId.global_config)
        )

        assert focus_result.errors == ()
        assert focus_result.focused is True
        assert focus_result.summary is not None
        assert focus_result.summary.manager_scope is not None
        assert focus_result.summary.manager_scope.value == ""
        assert widget_tree.errors == ()
        assert widget_tree.projected is True
        assert widget_tree.summary is not None
        assert widget_tree.summary.manager_scope is not None
        assert widget_tree.summary.manager_scope.value == ""
        assert snapshot.errors == ()
        assert snapshot.captured is True
        assert snapshot.resource is not None
        assert snapshot.summary is not None
        assert snapshot.summary.window_id == OpenHCSUiWindowId.global_config
        assert snapshot.summary.manager_scope is not None
        assert snapshot.summary.manager_scope.value == ""
        assert snapshot.capture_scope is WindowSnapshotCaptureScope.WINDOW
        assert snapshot.width is not None
        assert snapshot.width > 0
        assert snapshot.height is not None
        assert snapshot.height > 0
        assert close_result.closed is True
        assert close_result.summary is not None
        assert close_result.summary.window_id == OpenHCSUiWindowId.global_config
    finally:
        WindowManager.unregister("")
        global_window.close()


def test_managed_window_catalog_projects_object_state_status() -> None:
    app = QtApplicationAuthority.app()
    state = ObjectState(Dummy(), scope_id="managed-scope")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    state.update_parameter("x", 2)
    window = FakeManagedFormWindow("managed-scope")
    window.state = state
    window.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        windows = bridge.list_windows()
        summary = next(
            window_summary
            for window_summary in windows.windows
            if window_summary.window_id == "managed-scope"
        )

        assert summary.object_state_scope_id == "managed-scope"
        assert summary.dirty is True
        assert summary.signature_diff is True
        assert summary.dirty_field_count == 1
        assert summary.signature_diff_field_count == 1
        assert summary.semantic_markers == ("*", "_")
        assert (
            ManagedWindowAction.SAVE_WITHOUT_CLOSE.value in summary.managed_action_ids
        )
    finally:
        WindowManager.unregister("managed-scope")
        window.close()


def test_open_window_code_mode_documents_are_discoverable_by_window_id() -> None:
    app = QtApplicationAuthority.app()
    source = "config = GlobalPipelineConfig(num_workers=2)\n"
    global_window = QWidget()
    global_window.setWindowTitle("Configuration - GlobalPipelineConfig")
    WindowManager.register(
        "",
        global_window,
        code_document_driver=FakeWindowCodeDocumentDriver(source),
    )
    global_window.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    ObjectStateBridgeProviderSet().register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        document_id = f"{WINDOW_CODE_DOCUMENT_PREFIX}{OpenHCSUiWindowId.global_config}"
        documents = bridge.list_documents().documents
        summaries = tuple(
            summary for summary in documents if summary.document_id == document_id
        )
        document = bridge.get_document(UiCodeDocumentRequest(document_id=document_id))

        assert len(summaries) == 1
        assert summaries[0].widget_id == OpenHCSUiWindowId.global_config
        assert summaries[0].readable is True
        assert summaries[0].writable is True
        assert document.source == source
        assert document.summary.title == "View/Edit GlobalPipelineConfig"
        assert document.selected_scope_ids == ("",)
    finally:
        WindowManager.unregister("")
        global_window.close()


def test_open_window_code_mode_summary_uses_driver_title_when_window_title_empty() -> (
    None
):
    app = QtApplicationAuthority.app()
    source = "pipeline_steps = []\n"
    embedded_widget = QWidget()
    WindowManager.register(
        "embedded-code",
        embedded_widget,
        code_document_driver=FakeWindowCodeDocumentDriver(source),
    )
    embedded_widget.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    ObjectStateBridgeProviderSet().register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        document_id = f"{WINDOW_CODE_DOCUMENT_PREFIX}embedded-code"
        summary = next(
            summary
            for summary in bridge.list_documents().documents
            if summary.document_id == document_id
        )

        assert summary.title == "Code mode - View/Edit GlobalPipelineConfig"
    finally:
        WindowManager.unregister("embedded-code")
        embedded_widget.close()


def test_simple_code_editor_windows_register_code_documents(monkeypatch) -> None:
    app = QtApplicationAuthority.app()
    parent = QWidget()
    applied_sources: list[str] = []
    before_scopes = set(WindowManager.get_code_document_scopes())
    service = SimpleCodeEditorService(parent)
    monkeypatch.setattr(simple_code_editor, "QSCINTILLA_AVAILABLE", False)

    service.edit_code(
        "pipeline_steps = []\n",
        title="Edit Pipeline Steps",
        callback=applied_sources.append,
        code_type="pipeline",
        code_data={"clean_mode": True},
    )
    app.processEvents()

    try:
        new_scopes = set(WindowManager.get_code_document_scopes()) - before_scopes
        assert len(new_scopes) == 1
        scope_id = new_scopes.pop()
        document_id = f"{WINDOW_CODE_DOCUMENT_PREFIX}{scope_id}"

        registry = UiBridgeSurfaceRegistry()
        snapshot_provider = UiObjectStateSnapshotProvider()
        ObjectStateBridgeProviderSet().register(
            UiBridgeRegistrationContext(
                registry=registry,
                snapshot_provider=snapshot_provider,
            )
        )
        bridge = UiAgentBridgeService(
            registry=registry,
            dispatcher=InlineDispatcher(),
            snapshot_provider=snapshot_provider,
        )

        summaries = bridge.list_documents().documents
        document = bridge.get_document(UiCodeDocumentRequest(document_id=document_id))
        bridge.validate_document(
            UiCodeDocumentValidationRequest(
                document_id=document_id,
                source="pipeline_steps = [\n",
            )
        )
        apply_result = bridge.apply_document(
            UiCodeDocumentApplyRequest(
                document_id=document_id,
                source="pipeline_steps = ['edited']\n",
                base_revision_token=document.current_revision_token,
                confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                    False
                ),
            )
        )
        updated = bridge.get_document(UiCodeDocumentRequest(document_id=document_id))

        matching_summaries = tuple(
            summary for summary in summaries if summary.document_id == document_id
        )
        assert len(matching_summaries) == 1
        assert matching_summaries[0].title == "Code mode - Edit Pipeline Steps"
        assert document.source == "pipeline_steps = []\n"
        assert document.summary.title == "Edit Pipeline Steps"
        assert apply_result.applied is True
        assert applied_sources == ["pipeline_steps = ['edited']\n"]
        assert updated.source == "pipeline_steps = ['edited']\n"
    finally:
        for scope_id in set(WindowManager.get_code_document_scopes()) - before_scopes:
            window = WindowManager.get_window(scope_id)
            WindowManager.unregister(scope_id)
            if window is not None:
                window.close()
        parent.close()


def test_object_state_code_mode_documents_are_discoverable_by_scope_id() -> None:
    app = QtApplicationAuthority.app()
    source = "config = GlobalPipelineConfig(num_workers=2)\n"
    state = ObjectState(Dummy(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    global_window = QWidget()
    global_window.setWindowTitle("Configuration - GlobalPipelineConfig")
    WindowManager.register(
        "",
        global_window,
        code_document_driver=FakeWindowCodeDocumentDriver(source),
    )
    global_window.show()
    app.processEvents()

    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    ObjectStateBridgeProviderSet().register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )

    try:
        document_id = (
            f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}"
            f"{OpenHCSUiWindowId.global_config}"
        )
        documents = bridge.list_documents().documents
        summaries = tuple(
            summary for summary in documents if summary.document_id == document_id
        )
        generic_summaries = tuple(
            summary
            for summary in documents
            if summary.document_id
            == ObjectStateScopeCodeDocumentProvider.identity.document_id
        )
        document = bridge.get_document(UiCodeDocumentRequest(document_id=document_id))

        assert len(summaries) == 1
        assert not generic_summaries
        assert summaries[0].widget_id == "object_state_scope"
        assert summaries[0].readable is True
        assert summaries[0].writable is True
        assert document.source == source
        assert document.selected_scope_ids == (OpenHCSUiWindowId.global_config,)
    finally:
        WindowManager.unregister("")
        global_window.close()


def test_global_config_object_state_fields_use_stable_window_id() -> None:
    state = ObjectState(Dummy(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    state.update_parameter("x", 2)

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(
            include_system_scopes=True,
            include_fields=True,
            field_limit=1,
        )
    )

    global_scope = next(
        scope
        for scope in catalog.scopes
        if scope.identity.object_state_scope_id == OpenHCSUiWindowId.global_config
    )

    assert global_scope.fields
    field = global_scope.fields[0]
    assert global_scope.has_unsaved_changes is True
    assert global_scope.has_default_overrides is True
    assert global_scope.dirty_marker == "*"
    assert global_scope.signature_diff_marker == "_"
    assert field.address.object_state_scope_id == OpenHCSUiWindowId.global_config
    assert field.address.window_id == OpenHCSUiWindowId.global_config
    assert field.raw_value is None
    assert field.resolved_value is None
    assert field.raw_value_preview is not None
    assert field.resolved_value_preview is not None
    assert field.raw_value_preview.text == "2"
    assert field.resolved_value_preview.text == "2"
    assert field.raw_value_is_none is False
    assert field.resolved_value_is_none is False
    assert field.semantic_markers == ("*", "_")


def test_object_state_default_visibility_includes_global_config_without_plate_root() -> (
    None
):
    ObjectStateRegistry.register(ObjectState(Dummy(), scope_id=""), _skip_snapshot=True)
    ObjectStateRegistry.register(
        ObjectState(Dummy(), scope_id="__plates__"),
        _skip_snapshot=True,
    )
    ObjectStateRegistry.register(
        ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID),
        _skip_snapshot=True,
    )

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest()
    )
    scope_ids = {scope.identity.object_state_scope_id for scope in catalog.scopes}

    assert OpenHCSUiWindowId.global_config in scope_ids
    assert PLATE_SCOPE_ID in scope_ids
    assert "__plates__" not in scope_ids

    system_catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(include_system_scopes=True)
    )
    system_scope_ids = {
        scope.identity.object_state_scope_id for scope in system_catalog.scopes
    }

    assert OpenHCSUiWindowId.global_config in system_scope_ids
    assert "__plates__" in system_scope_ids


def test_object_state_fields_can_include_full_values_on_request() -> None:
    state = ObjectState(Dummy(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    state.update_parameter("x", 2)

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(
            include_system_scopes=True,
            include_fields=True,
            include_field_values=True,
            field_limit=1,
        )
    )

    global_scope = next(
        scope
        for scope in catalog.scopes
        if scope.identity.object_state_scope_id == OpenHCSUiWindowId.global_config
    )

    field = global_scope.fields[0]
    assert field.raw_value == 2
    assert field.resolved_value == 2
    assert field.raw_value_preview is not None
    assert field.resolved_value_preview is not None


def test_object_state_scope_fields_filter_before_pagination() -> None:
    @dataclass
    class MultiFieldDummy:
        alpha: int = 1
        beta: int = 2
        zeta: int = 3

    state = ObjectState(MultiFieldDummy(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    state.update_parameter("zeta", 4)

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(
            include_system_scopes=True,
            include_fields=True,
            field_filter=UiObjectStateFieldFilter.DIRTY,
            field_limit=1,
        )
    )

    global_scope = next(
        scope
        for scope in catalog.scopes
        if scope.identity.object_state_scope_id == OpenHCSUiWindowId.global_config
    )

    assert [field.address.field_path for field in global_scope.fields] == ["zeta"]
    assert global_scope.field_page is not None
    assert global_scope.field_page.total_count == 1
    assert global_scope.field_page.truncated is False


def test_object_state_exact_field_paths_include_path_type_and_description() -> None:
    state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(
            include_fields=True,
            include_field_descriptions=True,
            field_paths=("napari_display_config",),
            field_limit=1,
        )
    )

    global_scope = next(
        scope
        for scope in catalog.scopes
        if scope.identity.object_state_scope_id == OpenHCSUiWindowId.global_config
    )
    assert len(global_scope.fields) == 1
    field = global_scope.fields[0]
    assert field.address.field_path == "napari_display_config"
    assert field.object_state_path_type == "openhcs.core.config.NapariDisplayConfig"
    assert field.parameter_description is not None
    assert "napari display behavior" in field.parameter_description


def test_object_state_field_help_uses_object_state_path_types() -> None:
    state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    bridge = UiAgentBridgeService()

    section = bridge.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id=OpenHCSUiWindowId.global_config,
            field_path="napari_display_config",
            max_description_chars=500,
        )
    )
    child = bridge.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id=OpenHCSUiWindowId.global_config,
            field_path="napari_display_config.colormap",
            max_description_chars=500,
        )
    )

    assert section.errors == ()
    assert section.help_target_type == "openhcs.core.config.NapariDisplayConfig"
    assert section.parameter_name == "napari_display_config"
    assert section.description is not None
    assert "napari display behavior" in section.description

    assert child.errors == ()
    assert child.help_target_type == "openhcs.core.config.NapariDisplayConfig"
    assert child.parameter_name == "colormap"
    assert child.summary == "• colormap (NapariColormap)"
    assert child.description == "Colormap applied to grayscale image layers in napari."


def test_object_state_field_help_uses_source_binding_field_docstrings() -> None:
    state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    bridge = UiAgentBridgeService()

    source_defaults = bridge.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id=OpenHCSUiWindowId.global_config,
            field_path="source_bindings_config.metadata_rules",
            max_description_chars=500,
        )
    )
    step_bindings = bridge.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id=OpenHCSUiWindowId.global_config,
            field_path="step_source_bindings_config.bindings",
            max_description_chars=500,
        )
    )

    assert source_defaults.errors == ()
    assert (
        source_defaults.help_target_type
        == "openhcs.core.source_bindings.SourceBindingsConfig"
    )
    assert source_defaults.description == (
        "Regex/metadata extraction rules that add semantic fields for matching sources."
    )

    assert step_bindings.errors == ()
    assert (
        step_bindings.help_target_type
        == "openhcs.core.source_bindings.StepSourceBindingsConfig"
    )
    assert step_bindings.description == (
        "Named semantic source bindings available to pipelines and inherited by steps."
    )


def test_managed_window_save_action_returns_before_deferred_save_runs() -> None:
    app = QtApplicationAuthority.app()
    dispatcher = QueuedPostDispatcher()
    registry = UiBridgeSurfaceRegistry()
    snapshot_provider = UiObjectStateSnapshotProvider()
    MainWindowBridgeProviderSet(FakeMainWindow()).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=dispatcher,
        snapshot_provider=snapshot_provider,
    )
    window = FakeManagedFormWindow("managed-scope")
    other_window = FakeManagedFormWindow("other-managed-scope")
    window.show()
    other_window.show()
    app.processEvents()

    try:
        action = next(
            action
            for action in bridge.list_actions().actions
            if action.identity.action_id == ManagedWindowAction.SAVE_WITHOUT_CLOSE.value
        )
        result = bridge.invoke_action(
            UiActionInvokeRequest(
                widget_id=action.identity.widget_id,
                action_id=action.identity.action_id,
                selected_scope_ids=(window.scope_id,),
                confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                    False
                ),
            )
        )

        assert action.invocation_mode == "async"
        assert result.status == "accepted"
        assert result.receipt.bridge_operation_id is not None
        assert result.target_scope_ids == (window.scope_id,)
        assert window.save_count == 0
        assert len(action.target_scope_ids) == 2
        assert len(dispatcher.callbacks) == 1
        assert (
            bridge.get_operation_status(result.receipt.bridge_operation_id).status
            == "running"
        )

        dispatcher.run_next()

        operation = bridge.get_operation_status(result.receipt.bridge_operation_id)
        assert window.save_count == 1
        assert window.saved_close_window is False
        assert operation.status == "completed"
        assert operation.outcome == "accepted"
    finally:
        window.close()
        other_window.close()


def test_all_read_returns_source_hash_and_revision() -> None:
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager())
    )

    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert document.source == VALID_SOURCE
    assert document.size_bytes == len(VALID_SOURCE.encode("utf-8"))
    assert document.sha256
    assert document.current_revision_token
    assert document.selected_scope_ids == (PLATE_SCOPE_ID,)


def test_plate_manager_state_surface_projects_runtime_row_status() -> None:
    manager = FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
    manager.plate_compile_pending.add(PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    poll_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
            base_revision_token=state.current_revision_token,
        )
    )

    assert state.summary.surface_id == UiStateSurfaceId.PLATE_MANAGER.value
    row = state.payload["rows"][0]
    assert row["plate_scope_id"] == PLATE_SCOPE_ID
    assert row["status_prefix"] == "⏳ Compile"
    assert row["compile_pending"] is True
    assert row["selected"] is True
    assert poll_state.unchanged is True


def test_view_results_action_relates_widget_owned_live_measurement_surface() -> None:
    manager = FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
    manager.ACTION_ROUTES = {
        PlateManagerAction.VIEW_RESULTS: WidgetActionRoute(
            PlateManagerAction.VIEW_RESULTS,
            lambda _widget: None,
        ),
    }
    manager.buttons[PlateManagerAction.VIEW_RESULTS.value] = FakeButton(enabled=True)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    action = bridge.list_actions().actions[0]

    assert action.identity.action_id == PlateManagerAction.VIEW_RESULTS.value
    assert action.related_state_surface_ids == (
        "plate_manager.state",
        "plate_manager.live_measurements",
    )


def test_plate_manager_state_surface_links_source_and_output_plate_rows() -> None:
    source_row = FakeRow(
        PLATE_SCOPE_ID,
        PLATE_NAME,
        plate_root="/tmp/source-plate",
    )
    output_row = FakeRow(
        "/tmp/source-plate_openhcs",
        "source-plate_openhcs",
        plate_root="/tmp/source-plate_openhcs",
    )
    manager = FakePlateManager(
        selected=(source_row,),
        plates=(source_row, output_row),
    )
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    source_payload, output_payload = state.payload["rows"]
    assert source_payload["output_plate_scope_id"] == output_row.scope_id
    assert source_payload["output_plate_root"] == output_row.plate_root
    assert source_payload["source_plate_scope_id"] is None
    assert output_payload["source_plate_scope_id"] == source_row.scope_id
    assert output_payload["source_plate_root"] == source_row.plate_root
    assert output_payload["output_plate_scope_id"] is None


def test_plate_manager_state_surface_uses_row_effective_path_config(
    tmp_path: Path,
) -> None:
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    source_root = tmp_path / "source-plate"
    source_root.mkdir()
    output_root = tmp_path / "source-plate_custom"
    source_row = FakeRow(
        PLATE_SCOPE_ID,
        PLATE_NAME,
        plate_root=str(source_root),
    )
    output_row = FakeRow(
        str(output_root),
        "source-plate_custom",
        plate_root=str(output_root),
    )
    orchestrator = PipelineOrchestrator(
        source_root,
        pipeline_config=PipelineConfig(
            path_planning_config=LazyPathPlanningConfig(
                output_dir_suffix="_custom",
            ),
        ),
    )
    ObjectStateRegistry.register(
        ObjectState(orchestrator, scope_id=source_row.scope_id),
        _skip_snapshot=True,
    )
    manager = FakePlateManager(
        selected=(source_row,),
        plates=(source_row, output_row),
    )
    bridge = UiAgentBridgeService(provider_set=PlateManagerBridgeProviderSet(manager))

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    source_payload, output_payload = state.payload["rows"]
    assert source_payload["output_plate_scope_id"] == output_row.scope_id
    assert source_payload["output_plate_root"] == output_row.plate_root
    assert output_payload["source_plate_scope_id"] == source_row.scope_id
    assert output_payload["source_plate_root"] == source_row.plate_root


def test_plate_manager_state_ignores_stale_runtime_without_current_execution_id() -> (
    None
):
    manager = FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
    stale_projection = PlateRuntimeProjection(
        identity=PlateRuntimeIdentity(
            execution_id="old-execution",
            plate_id=PLATE_SCOPE_ID,
        ),
        state=PlateRuntimeState.EXECUTING,
        percent=0.0,
        axis_progress=(),
        latest_timestamp=1.0,
    )
    manager.runtime_progress_projection.add_plate(stale_projection)
    manager.runtime_progress_projection.mark_latest(stale_projection.identity)
    bridge = UiAgentBridgeService(provider_set=PlateManagerBridgeProviderSet(manager))

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    row = state.payload["rows"][0]
    assert row["execution_id"] is None
    assert row["execution_active"] is False
    assert row["runtime_state"] is None
    assert row["runtime_percent"] is None
    assert row["status_prefix"] == ""


def test_plate_manager_state_terminal_status_overrides_stale_executing_state() -> None:
    ObjectStateRegistry.register(
        ObjectState(
            FakeOrchestrator(state=OrchestratorState.EXECUTING),
            scope_id=PLATE_SCOPE_ID,
        ),
        _skip_snapshot=True,
    )
    manager = FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
    manager.plate_execution_ids[PLATE_SCOPE_ID] = "failed-execution"
    manager.plate_terminal_activity_status.mark_terminal(PLATE_SCOPE_ID, "failed")
    bridge = UiAgentBridgeService(provider_set=PlateManagerBridgeProviderSet(manager))

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    row = state.payload["rows"][0]
    assert row["execution_id"] == "failed-execution"
    assert row["terminal_status"] == "failed"
    assert row["orchestrator_state"] == "exec_failed"
    assert row["execution_active"] is False
    assert row["status_prefix"] == "❌ Exec Failed"


def test_plate_manager_action_catalog_token_can_guard_invoke() -> None:
    QtApplicationAuthority.app()
    manager = FakePlateManager(
        selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
    )
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    action = bridge.list_actions().actions[0]
    accepted = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=action.identity.widget_id,
            action_id=action.identity.action_id,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token=action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )
    stale = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=action.identity.widget_id,
            action_id=action.identity.action_id,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token="stale-token",
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert action.selection_revision_token
    assert accepted.status == "accepted"
    assert accepted.selection_revision_token == action.selection_revision_token
    assert manager.code_action_count == 1
    assert stale.status == "rejected"
    assert stale.errors
    assert stale.errors[0].code == "stale_ui_action_revision"
    assert stale.errors[0].hint is not None
    assert "openhcs_ui_list_actions" in stale.errors[0].hint
    assert "selection_revision_token" in stale.errors[0].hint
    assert "base_revision_token" in stale.errors[0].hint


def _pipeline_editor_bridge(manager: FakePipelineEditor) -> UiAgentBridgeService:
    snapshot_provider = UiObjectStateSnapshotProvider()
    registry = UiBridgeSurfaceRegistry()
    PipelineEditorBridgeProviderSet(manager).register(
        UiBridgeRegistrationContext(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
    )
    return UiAgentBridgeService(
        registry=registry,
        dispatcher=InlineDispatcher(),
        snapshot_provider=snapshot_provider,
    )


def test_pipeline_editor_action_catalog_uses_declared_routes_and_scope_hooks() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor(selected_indices=(1,))
    manager.buttons[PipelineEditorAction.EDIT_STEP.value] = FakeButton(enabled=False)
    bridge = _pipeline_editor_bridge(manager)

    actions = {
        action.identity.action_id: action
        for action in bridge.list_actions().actions
        if action.identity.widget_id == OpenHCSUiWindowId.pipeline_editor
    }
    code_action = actions[PipelineEditorAction.CODE_PIPELINE.value]
    edit_action = actions[PipelineEditorAction.EDIT_STEP.value]
    auto_action = actions[PipelineEditorAction.AUTO_LOAD_PIPELINE.value]

    assert set(actions) == {action.value for action in PipelineEditorAction}
    assert code_action.identity.widget_id == OpenHCSUiWindowId.pipeline_editor
    assert code_action.confirmation_required is False
    assert code_action.side_effects == ("opens_code_document_window",)
    assert code_action.target_scope_ids == (
        PipelineScopeIdentity.from_plate_scope(PLATE_SCOPE_ID).scope_id,
    )
    assert code_action.selection_mode == "current_pipeline"
    assert edit_action.target_scope_ids == ("scope-from-manager-hook-1",)
    assert edit_action.selection_mode == "selected_steps"
    assert edit_action.related_state_surface_ids == ("pipeline_editor.state",)
    assert edit_action.disabled_error is not None
    assert edit_action.disabled_error.hint is not None
    assert "selected step" in edit_action.disabled_error.hint
    assert auto_action.confirmation_required is True
    assert auto_action.side_effects == ("loads_basic_pipeline", "mutates_pipeline")


def test_pipeline_debug_toolbar_actions_are_exposed_from_toolbar_declarations() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor()
    bridge = _pipeline_editor_bridge(manager)
    widget_id = PipelineDebugToolbarWidgetIdentity.require_value()

    actions = {
        action.identity.action_id: action
        for action in bridge.list_actions().actions
        if action.identity.widget_id == widget_id
    }
    expected_action_ids = {
        declaration.action_id()
        for declaration in PipelineDebugActionDeclarationBase.toolbar_actions()
    }

    assert set(actions) == expected_action_ids
    step_action = actions[DebugCommandType.STEP.value]
    restart_action = actions[DebugCommandType.RESTART.value]
    stop_action = actions[DebugCommandType.STOP.value]
    runtime_action = actions[DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value]
    assert step_action.title == StepDebugAction.label
    assert step_action.enabled is True
    assert step_action.confirmation_required is True
    assert restart_action.enabled is False
    assert restart_action.disabled_error is not None
    assert restart_action.disabled_error.code == "debug_session_required"
    assert stop_action.enabled is False
    assert stop_action.disabled_error is not None
    assert stop_action.disabled_error.code == "debug_session_required"
    assert runtime_action.enabled is False
    assert runtime_action.disabled_error is not None
    assert runtime_action.disabled_error.code == "debug_session_required"


def test_pipeline_debug_toolbar_projects_pending_execution_to_bridge_actions() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor()
    manager.manager_execution_state = ManagerExecutionState.RUNNING
    manager.debug_toolbar.set_debug_session_context(manager.debug_session_context())
    bridge = _pipeline_editor_bridge(manager)
    widget_id = PipelineDebugToolbarWidgetIdentity.require_value()
    actions = {
        action.identity.action_id: action
        for action in bridge.list_actions().actions
        if action.identity.widget_id == widget_id
    }

    step_action = actions[DebugCommandType.STEP.value]
    stop_action = actions[DebugCommandType.STOP.value]

    assert step_action.enabled is False
    assert step_action.disabled_error is not None
    assert step_action.disabled_error.code == "debug_execution_pending"
    assert stop_action.enabled is True


def test_pipeline_debug_toolbar_action_invoke_routes_to_debug_workflow() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor()
    bridge = _pipeline_editor_bridge(manager)
    widget_id = PipelineDebugToolbarWidgetIdentity.require_value()
    action = next(
        action
        for action in bridge.list_actions().actions
        if (
            action.identity.widget_id == widget_id
            and action.identity.action_id == DebugCommandType.STEP.value
        )
    )

    rejected = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=widget_id,
            action_id=DebugCommandType.STEP.value,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token=action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )
    accepted = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=widget_id,
            action_id=DebugCommandType.STEP.value,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token=action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert rejected.status == "rejected"
    assert rejected.errors
    assert rejected.errors[0].code == "confirmation_required"
    assert accepted.status == "accepted"
    assert manager.debug_workflow.commands == [DebugCommand(DebugCommandType.STEP)]


def test_pipeline_debug_toolbar_runtime_values_action_requires_debug_session() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor()
    bridge = _pipeline_editor_bridge(manager)
    widget_id = PipelineDebugToolbarWidgetIdentity.require_value()
    manager.debug_session_state = DebugSession.create(plate_id=PLATE_SCOPE_ID)
    manager.debug_toolbar.set_debug_session_context(manager.debug_session_context())
    action = next(
        action
        for action in bridge.list_actions().actions
        if (
            action.identity.widget_id == widget_id
            and action.identity.action_id
            == DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value
        )
    )

    result = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=widget_id,
            action_id=DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token=action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )

    assert action.enabled is True
    assert action.confirmation_required is False
    assert result.status == "accepted"
    assert manager.debug_workflow.runtime_inspections == 1


def test_pipeline_debug_session_state_surface_projects_context_and_actions() -> None:
    QtApplicationAuthority.app()
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="scope-from-manager-hook-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    manager = FakePipelineEditor()
    manager.debug_session_state = DebugSession.create(
        plate_id=PLATE_SCOPE_ID,
        execution_id="exec-1",
        axis_id="A01",
    ).with_cursor(cursor)
    bridge = _pipeline_editor_bridge(manager)

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_DEBUG_SESSION.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    actions = {action["action_id"]: action for action in state.payload["actions"]}

    assert state.summary.surface_id == UiStateSurfaceId.PIPELINE_DEBUG_SESSION.value
    assert state.payload["phase"] == "active_session"
    assert state.payload["current_plate_scope_id"] == PLATE_SCOPE_ID
    assert state.payload["pipeline_scope_id"] == (
        PipelineScopeIdentity.from_plate_scope(PLATE_SCOPE_ID).scope_id
    )
    assert (
        state.payload["active_session_id"]
        == manager.debug_session_state.debug_session_id
    )
    assert state.payload["execution_id"] == "exec-1"
    assert state.payload["axis_id"] == "A01"
    assert state.payload["cursor"]["step_scope_id"] == "scope-from-manager-hook-1"
    assert actions[DebugCommandType.RESTART.value]["enabled"] is True
    assert actions[DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value]["enabled"] is True
    assert actions[DebugCommandType.STEP.value]["label"] == StepDebugAction.label


def test_pipeline_debug_session_state_surface_projects_runtime_frame() -> None:
    QtApplicationAuthority.app()
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="scope-from-manager-hook-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    session = DebugSession(
        debug_session_id="debug-1",
        plate_id=PLATE_SCOPE_ID,
        execution_id="exec-1",
        axis_id="A01",
    ).with_cursor(cursor)
    progress_event = ProgressEvent(
        identity=ProgressIdentity(
            execution_id="exec-1",
            plate_id=PLATE_SCOPE_ID,
            axis_id="A01",
            step_name="IdentifyPrimaryObjects",
        ),
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=123.0,
        pid=1234,
        context=DebugProgressContext(
            debug_session_id="debug-1",
            snapshot_id="snapshot-1",
            cursor=cursor,
            event_type=DebugEventType.AFTER_INVOCATION,
            snapshot_store_ref="/debug",
        ).to_progress_context(),
    )
    manager = FakePipelineEditor()
    manager.debug_session_state = session
    manager.debug_runtime_projection_state = (
        RuntimeProjectionBuilder()
        .build(
            RuntimeProjectionSource(
                events_by_execution={"exec-1": [progress_event]},
                session=session,
            )
        )
        .debug
    )
    bridge = _pipeline_editor_bridge(manager)

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_DEBUG_SESSION.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert state.payload["current_frame"]["debug_session_id"] == "debug-1"
    assert state.payload["current_frame"]["snapshot_id"] == "snapshot-1"
    assert state.payload["current_frame"]["event_type"] == "after_invocation"
    assert state.payload["current_frame"]["progress_identity"]["axis_id"] == "A01"
    assert state.payload["last_frame"] == state.payload["current_frame"]


def test_pipeline_debug_session_state_surface_projects_terminal_summary() -> None:
    QtApplicationAuthority.app()
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="scope-from-manager-hook-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    manager = FakePipelineEditor()
    manager.debug_terminal_summary = DebugTerminalSummary(
        debug_session_id="debug-1",
        plate_id=PLATE_SCOPE_ID,
        terminal_status="complete",
        cursor=cursor,
        command_type=DebugCommandType.STEP,
        axis_id="A01",
        snapshot_id="snapshot-1",
        snapshot_store_ref="/debug",
    )
    bridge = _pipeline_editor_bridge(manager)

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_DEBUG_SESSION.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert state.payload["phase"] == "terminal_complete"
    assert state.payload["active_session_id"] is None
    assert state.payload["terminal_summary"]["debug_session_id"] == "debug-1"
    assert state.payload["terminal_summary"]["command_type"] == "step"
    assert state.payload["terminal_summary"]["cursor"]["step_index"] == 1


def test_pipeline_debug_session_state_surface_retire_matching_local_session() -> None:
    QtApplicationAuthority.app()
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="scope-from-manager-hook-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    manager = FakePipelineEditor()
    manager.debug_session_state = (
        DebugSession.create(
            plate_id=PLATE_SCOPE_ID,
            execution_id="exec-1",
            axis_id="A01",
        )
        .with_cursor(cursor)
        .with_command(DebugCommandType.STEP)
    )
    manager.debug_terminal_summary = DebugTerminalSummary(
        debug_session_id=manager.debug_session_state.debug_session_id,
        plate_id=PLATE_SCOPE_ID,
        terminal_status="complete",
        cursor=cursor,
        command_type=DebugCommandType.STEP,
        axis_id="A01",
        snapshot_id="snapshot-1",
        snapshot_store_ref="/debug",
    )
    bridge = _pipeline_editor_bridge(manager)

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_DEBUG_SESSION.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    actions = {action["action_id"]: action for action in state.payload["actions"]}

    assert state.payload["phase"] == "terminal_complete"
    assert state.payload["active_session_id"] is None
    assert state.payload["execution_id"] is None
    assert state.payload["cursor"] is None
    assert state.payload["terminal_summary"]["debug_session_id"] == (
        manager.debug_session_state.debug_session_id
    )
    assert actions[DebugCommandType.RUN.value]["label"] == "Start Debug"
    assert actions[DebugCommandType.RESTART.value]["enabled"] is False
    assert actions[DebugCommandType.STOP.value]["enabled"] is False
    assert actions[DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value]["enabled"] is False


def test_pipeline_editor_state_surface_projects_steps_and_selection(
    monkeypatch,
) -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor(selected_indices=(1,))
    bridge = _pipeline_editor_bridge(manager)
    reference_indexes: dict[int, int] = {}

    @dataclass(frozen=True)
    class _FunctionReference:
        composite_key: str

    def function_reference(function):
        index = reference_indexes.setdefault(id(function), len(reference_indexes))
        return _FunctionReference(f"test:function_{index}")

    monkeypatch.setattr(
        FunctionReferenceTransportAuthority,
        "function_reference",
        staticmethod(function_reference),
    )

    catalog = bridge.list_state_surfaces()
    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_EDITOR.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    selected_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PIPELINE_EDITOR.value,
            selection_mode=SELECTED_SELECTION_MODE,
            base_revision_token=state.current_revision_token,
        )
    )

    assert catalog.surfaces[0].surface_id == UiStateSurfaceId.PIPELINE_EDITOR.value
    assert state.summary.surface_id == UiStateSurfaceId.PIPELINE_EDITOR.value
    assert state.summary.current_selection_count == 1
    assert state.summary.total_scope_count == 2
    assert state.payload["pipeline_scope_id"] == (
        PipelineScopeIdentity.from_plate_scope(PLATE_SCOPE_ID).scope_id
    )
    assert state.payload["selected_scope_ids"] == ["scope-from-manager-hook-1"]
    assert [step["name"] for step in state.payload["steps"]] == [
        "step_one",
        "step_two",
    ]
    assert state.payload["steps"][0]["selected"] is False
    assert state.payload["steps"][1]["selected"] is True
    assert state.payload["steps"][1]["step_scope_id"] == "scope-from-manager-hook-1"
    assert state.payload["steps"][0]["function_ids"] == ["test:function_0"]
    assert state.payload["steps"][1]["function_ids"] == ["test:function_1"]
    assert selected_state.payload["steps"][0]["name"] == "step_two"
    assert selected_state.payload["steps"][0]["function_ids"] == ["test:function_1"]
    assert selected_state.unchanged is False


def test_pipeline_editor_action_invoke_uses_selection_token_and_confirmation() -> None:
    QtApplicationAuthority.app()
    manager = FakePipelineEditor(selected_indices=(0,))
    bridge = _pipeline_editor_bridge(manager)

    actions = {
        action.identity.action_id: action for action in bridge.list_actions().actions
    }
    code_action = actions[PipelineEditorAction.CODE_PIPELINE.value]
    edit_action = actions[PipelineEditorAction.EDIT_STEP.value]

    code_result = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=code_action.identity.widget_id,
            action_id=code_action.identity.action_id,
            selected_scope_ids=code_action.target_scope_ids,
            observed_selection_revision_token=code_action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )
    edit_result = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=edit_action.identity.widget_id,
            action_id=edit_action.identity.action_id,
            selected_scope_ids=edit_action.target_scope_ids,
            observed_selection_revision_token=edit_action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )
    stale_result = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=edit_action.identity.widget_id,
            action_id=edit_action.identity.action_id,
            selected_scope_ids=("wrong-target",),
            observed_selection_revision_token=edit_action.selection_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert code_result.status == "accepted"
    assert manager.code_count == 1
    assert edit_result.status == "rejected"
    assert edit_result.errors
    assert edit_result.errors[0].code == "confirmation_required"
    assert manager.edit_count == 0
    assert stale_result.status == "rejected"
    assert stale_result.errors
    assert stale_result.errors[0].code == "stale_ui_action_selection"
    assert stale_result.errors[0].hint is not None
    assert "selection_mode=selected_steps" in stale_result.errors[0].hint
    assert "pipeline_editor.state" in stale_result.errors[0].hint
    assert "openhcs_ui_navigate_window" in stale_result.errors[0].hint
    assert "window_id='wrong-target'" in stale_result.errors[0].hint


def test_selected_workflow_returns_before_queued_plate_action_runs() -> None:
    QtApplicationAuthority.app()
    dispatcher = QueuedPostDispatcher()
    manager = FakePlateManager(
        selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
    )
    manager.ACTION_ROUTES = {
        PlateManagerAction.INIT_PLATE: WidgetActionRoute(
            PlateManagerAction.INIT_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }
    manager.BUTTON_CONFIGS = [
        ("Init", PlateManagerAction.INIT_PLATE.value, "Initialize plate"),
    ]
    manager.buttons[PlateManagerAction.INIT_PLATE.value] = FakeButton(enabled=True)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=dispatcher,
    )

    result = bridge.selected_plate_workflow(
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.INIT,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    operation_id = result.action_result.receipt.bridge_operation_id
    assert result.action_result.status == "accepted"
    assert operation_id is not None
    assert result.action_result.workflow_status_surface_ids == ("plate_manager.state",)
    assert manager.code_action_count == 0
    assert len(dispatcher.callbacks) == 1
    assert bridge.get_operation_status(operation_id).status == "running"

    dispatcher.run_next()

    assert manager.code_action_count == 1
    operation = bridge.get_operation_status(operation_id)
    assert operation.status == "completed"
    assert operation.outcome == "accepted"


def test_selected_workflow_rejection_includes_selection_recovery_hint() -> None:
    QtApplicationAuthority.app()
    manager = FakePlateManager(selected=(), plates=())
    manager.ACTION_ROUTES = {
        PlateManagerAction.COMPILE_PLATE: WidgetActionRoute(
            PlateManagerAction.COMPILE_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }
    manager.BUTTON_CONFIGS = [
        ("Compile", PlateManagerAction.COMPILE_PLATE.value, "Compile plate pipelines"),
    ]
    manager.buttons[PlateManagerAction.COMPILE_PLATE.value] = FakeButton(enabled=True)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    catalog = bridge.list_actions()
    action = catalog.actions[0]
    result = bridge.selected_plate_workflow(
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert not action.enabled
    assert action.disabled_error is not None
    assert action.disabled_error.code == "plate_selection_required"
    assert action.disabled_error.hint is not None
    assert "plate_manager.state" in action.disabled_error.hint
    assert "plate_manager.orchestrator_config" in action.disabled_error.hint
    assert "openhcs_ui_apply_code_document" in action.disabled_error.hint
    assert catalog.warnings
    assert catalog.warnings[0].code == "plate_path_setup_uses_code_document"
    assert "plate_paths" in catalog.warnings[0].message
    assert result.errors
    assert result.errors[0].code == "plate_selection_required"
    assert result.errors[0].hint is not None
    assert "plate_manager.orchestrator_config" in result.errors[0].hint


def test_selected_compile_reports_init_precondition_when_plate_is_created() -> None:
    QtApplicationAuthority.app()
    ObjectStateRegistry.register(
        ObjectState(
            FakeOrchestrator(state=OrchestratorState.CREATED),
            scope_id=PLATE_SCOPE_ID,
        ),
        _skip_snapshot=True,
    )
    manager = FakePlateManager(
        selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
        pipeline_steps=(FunctionStep(func=lambda image: image, name="Defined"),),
    )
    manager.ACTION_ROUTES = {
        PlateManagerAction.COMPILE_PLATE: WidgetActionRoute(
            PlateManagerAction.COMPILE_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }
    manager.BUTTON_CONFIGS = [
        ("Compile", PlateManagerAction.COMPILE_PLATE.value, "Compile plate pipelines"),
    ]
    manager.buttons[PlateManagerAction.COMPILE_PLATE.value] = FakeButton(enabled=True)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager),
        dispatcher=InlineDispatcher(),
    )

    action = bridge.list_actions().actions[0]
    assert not action.enabled
    assert action.disabled_error is not None
    assert action.disabled_error.code == "orchestrator_not_initialized"
    assert "init_plate" in action.disabled_error.message

    result = bridge.selected_plate_workflow(
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            selected_scope_ids=(PLATE_SCOPE_ID,),
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert result.action_result.status == "rejected"
    assert result.errors
    assert result.errors[0].code == "orchestrator_not_initialized"
    assert "init_plate" in result.errors[0].message


def test_selected_workflow_confirmation_rejection_avoids_ui_preflight() -> None:
    QtApplicationAuthority.app()
    ObjectStateRegistry.register(
        ObjectState(
            FakeOrchestrator(state=OrchestratorState.READY),
            scope_id=PLATE_SCOPE_ID,
        ),
        _skip_snapshot=True,
    )
    manager = FakePlateManager(
        selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
        pipeline_steps=(FunctionStep(func=lambda image: image, name="Defined"),),
    )
    manager.ACTION_ROUTES = {
        PlateManagerAction.COMPILE_PLATE: WidgetActionRoute(
            PlateManagerAction.COMPILE_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }
    manager.BUTTON_CONFIGS = [
        ("Compile", PlateManagerAction.COMPILE_PLATE.value, "Compile plate pipelines"),
    ]
    manager.buttons[PlateManagerAction.COMPILE_PLATE.value] = FakeButton(enabled=True)
    dispatcher = CountingDispatcher()
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(manager), dispatcher=dispatcher
    )

    result = bridge.selected_plate_workflow(
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )

    assert result.action_result.status == "rejected"
    assert result.errors[0].code == "confirmation_required"
    assert result.action_result.target_scope_ids == ()
    assert result.action_result.selection_revision_token is None
    assert result.action_result.workflow_status_surface_ids == ("plate_manager.state",)
    assert dispatcher.call_count == 0


def test_validation_rejects_side_effecting_source_before_execution() -> None:
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager())
    )

    result = bridge.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source="open('/tmp/openhcs-mcp-side-effect', 'w')\n",
        )
    )

    assert not result.valid
    assert result.errors
    assert result.errors[0].code == "unsafe_statement"


def test_validation_rejects_legacy_pipeline_config_assignment() -> None:
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager())
    )

    result = bridge.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source=(
                "from openhcs.core.config import PipelineConfig\n"
                f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
                "pipeline_config = PipelineConfig()\n"
                f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
            ),
        )
    )

    assert not result.valid
    assert any(error.code == "unexpected_assignment" for error in result.errors)


def test_validation_reports_orchestrator_payload_hint() -> None:
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager())
    )

    result = bridge.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source=(
                f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
                "global_config = None\n"
                "per_plate_configs = {}\n"
                f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
            ),
        )
    )

    assert not result.valid
    assert result.errors
    assert result.errors[0].code == "invalid_orchestrator_code_payload"
    assert "global_config must be a GlobalPipelineConfig" in result.errors[0].message
    assert result.errors[0].hint is not None
    assert "plate_paths" in result.errors[0].hint
    assert "pipeline_data" in result.errors[0].hint


def test_apply_rejection_reports_current_revision_and_snapshot() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    ObjectStateRegistry.record_snapshot("before edit", scope_id=PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(
                selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
                operations=FakeOperations(state),
            )
        ),
        dispatcher=InlineDispatcher(),
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token="stale-token",
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert not result.applied
    assert result.errors[0].code == "stale_revision_token"
    assert result.current_revision_token == document.current_revision_token
    assert result.current_snapshot == document.current_snapshot


def test_confirmation_required_apply_rejection_reports_current_context() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    ObjectStateRegistry.record_snapshot("before edit", scope_id=PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(
                selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
                operations=FakeOperations(state),
            )
        ),
        dispatcher=InlineDispatcher(),
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(True),
        )
    )

    assert not result.applied
    assert result.errors[0].code == "confirmation_required"
    assert result.current_revision_token == document.current_revision_token
    assert result.current_snapshot == document.current_snapshot


def test_object_state_apply_rejection_reports_dynamic_revision_key() -> None:
    scope_id = "plate::functionstep_0"
    state = ObjectState(Dummy(), scope_id=scope_id)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    ObjectStateRegistry.record_snapshot("before object edit", scope_id=scope_id)
    snapshot_provider = UiObjectStateSnapshotProvider()
    provider = ObjectStateScopeCodeDocumentProvider(snapshot_provider)
    document_id = f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}{scope_id}"

    result = provider.apply(
        UiCodeDocumentApplyRequest(
            document_id=document_id,
            source="pattern = None\n",
            base_revision_token="stale-token",
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert not result.applied
    assert result.errors[0].code == "stale_revision_token"
    assert result.current_revision_token == snapshot_provider.revision_token(
        f"ui-code-document:{document_id}"
    )
    assert result.current_snapshot == snapshot_provider.current_snapshot()


def test_source_policy_allows_public_function_steps_not_runtime_factories() -> None:
    policy = UiCodeDocumentSourcePolicy()
    cellprofiler_function_step_source = (
        "from openhcs.core.steps.function_step import FunctionStep\n"
        "from openhcs.processing.backends.cellprofiler import "
        "identify_secondary_objects\n"
        f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
        "global_config = None\n"
        f"per_plate_configs = {{'{PLATE_SCOPE_ID}': None}}\n"
        f"pipeline_data = {{'{PLATE_SCOPE_ID}': [FunctionStep("
        "func=identify_secondary_objects, name='IdentifySecondaryObjects')]}\n"
    )
    undeclared_factory_source = (
        "from openhcs.fake import lower_factory\n"
        f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
        f"pipeline_data = {{'{PLATE_SCOPE_ID}': [lower_factory()]}}\n"
    )

    assert policy.validate(cellprofiler_function_step_source) == ()
    errors = policy.validate(undeclared_factory_source)
    assert any(error.code == "unsafe_call" for error in errors)


def test_source_policy_allows_safe_builtin_type_references_only() -> None:
    source = (
        "from openhcs.core.runtime_tabular_values import FieldSpec\n"
        f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
        "global_config = None\n"
        f"per_plate_configs = {{'{PLATE_SCOPE_ID}': "
        "FieldSpec(name='FileLocation', dtype=str)}\n"
        f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
    )

    assert UiCodeDocumentSourcePolicy().validate(source) == ()

    unsafe_source = source.replace("dtype=str", "dtype=eval")
    errors = UiCodeDocumentSourcePolicy().validate(unsafe_source)
    assert any(error.code == "unknown_name" for error in errors)


def test_source_policy_allows_topological_path_bindings() -> None:
    source = (
        "from pathlib import Path\n"
        "path_root = Path('/media/alice/T7/screen')\n"
        "path_1 = path_root / 'plate_A'\n"
        "plate_paths = [path_1]\n"
        "global_config = None\n"
        "per_plate_configs = {path_1: None}\n"
        "pipeline_data = {path_1: []}\n"
    )

    assert UiCodeDocumentSourcePolicy().validate(source) == ()


def test_source_policy_allows_reused_safe_literal_bindings() -> None:
    source = (
        f"plate_path = '{PLATE_SCOPE_ID}'\n"
        "plate_paths = [plate_path]\n"
        "global_config = None\n"
        "per_plate_configs = {plate_path: None}\n"
        "pipeline_data = {plate_path: []}\n"
    )

    assert UiCodeDocumentSourcePolicy().validate(source) == ()


def test_source_policy_rejects_reassigned_literal_bindings() -> None:
    source = (
        f"plate_path = '{PLATE_SCOPE_ID}'\n"
        "plate_path = '/different/plate'\n"
        "plate_paths = [plate_path]\n"
        "pipeline_data = {plate_path: []}\n"
    )

    errors = UiCodeDocumentSourcePolicy().validate(source)

    assert any(error.code == "unexpected_assignment" for error in errors)


@pytest.mark.parametrize(
    "binding",
    (
        "path_root = Path('relative')",
        "path_root = unknown / 'plate_A'",
        "path_root = Path('/data').resolve()",
        "path_root = Path('/data') + 'plate_A'",
        "path_root = Path('/data') / '..' / 'plate_A'",
    ),
)
def test_source_policy_rejects_non_declarative_path_bindings(binding: str) -> None:
    source = (
        "from pathlib import Path\n"
        f"{binding}\n"
        f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
        f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
    )

    assert UiCodeDocumentSourcePolicy().validate(source)


def test_apply_creates_baseline_and_edit_snapshot() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    operations = FakeOperations(state)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(operations=operations)
        ),
        dispatcher=InlineDispatcher(),
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    history = ObjectStateRegistry.get_branch_history()
    assert result.applied
    assert result.operation_id is not None
    assert result.receipt.accepted
    assert result.receipt.bridge_operation_id == result.operation_id
    assert operations.pre_count == 1
    assert operations.post_count == 1
    assert [snapshot.label for snapshot in history] == [
        "init",
        f"edit {DOCUMENT_ID} via MCP [{PLATE_SCOPE_ID}]",
    ]
    assert result.undo_snapshot is not None
    assert result.undo_snapshot == result.pre_apply_snapshot
    assert result.undo_snapshot.label == "init"
    assert result.post_apply_snapshot is not None
    assert result.post_apply_snapshot == result.current_snapshot
    assert bridge.list_snapshots(UiSnapshotListRequest()).current_snapshot_index == 1
    assert (
        bridge.list_object_state_scopes(
            UiObjectStateScopeListRequest()
        ).current_snapshot_index
        == 1
    )


def test_apply_document_returns_running_operation_before_queued_ui_apply_runs() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    operations = FakeOperations(state)
    dispatcher = QueuedPostDispatcher()
    operation_tracker = UiBridgeOperationTracker()
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(operations=operations)
        ),
        dispatcher=dispatcher,
        operation_tracker=operation_tracker,
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert result.applied is False
    assert result.operation_id is not None
    assert result.receipt.accepted is True
    assert result.receipt.bridge_operation_id == result.operation_id
    assert operations.pre_count == 0
    assert bridge.get_operation_status(result.operation_id).status == "running"

    dispatcher.run_next()

    operation = bridge.get_operation_status(result.operation_id)
    assert operations.pre_count == 1
    assert operations.post_count == 1
    assert operation.status == "completed"
    assert operation.outcome == "applied"


def test_queued_apply_document_error_updates_operation_status() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    operations = FakeOperations(state)
    dispatcher = QueuedPostDispatcher()
    operation_tracker = UiBridgeOperationTracker()
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(operations=operations)
        ),
        dispatcher=dispatcher,
        operation_tracker=operation_tracker,
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token="stale-token",
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert result.applied is False
    assert result.operation_id is not None
    assert result.receipt.accepted is True
    assert bridge.get_operation_status(result.operation_id).status == "running"

    dispatcher.run_next()

    operation = bridge.get_operation_status(result.operation_id)
    assert operation.status == "completed"
    assert operation.outcome == "not_applied"
    assert operation.errors
    assert operation.errors[0].code == "stale_revision_token"
    operation_section = operation_tracker.overview_sections()[0]
    assert operation_section.items[0].status == "completed"
    assert operation_section.items[0].severity == "error"


def test_live_overview_reports_running_and_completed_bridge_operations() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    operations = FakeOperations(state)
    dispatcher = QueuedPostDispatcher()
    snapshot_provider = UiObjectStateSnapshotProvider()
    operation_tracker = UiBridgeOperationTracker()
    registry = UiBridgeSurfaceRegistry()
    context = UiBridgeRegistrationContext(
        registry=registry,
        snapshot_provider=snapshot_provider,
    )
    registry.register_live_overview_contributor(operation_tracker)
    PlateManagerBridgeProviderSet(FakePlateManager(operations=operations)).register(
        context
    )
    LiveOverviewBridgeProviderSet().register(context)
    bridge = UiAgentBridgeService(
        registry=registry,
        dispatcher=dispatcher,
        snapshot_provider=snapshot_provider,
        operation_tracker=operation_tracker,
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )
    running_overview = bridge.get_state_surface(
        UiStateSurfaceRequest(surface_id=UiStateSurfaceId.UI_LIVE_OVERVIEW.value)
    )

    running_section = next(
        section
        for section in running_overview.payload["sections"]
        if section["section_id"] == operation_tracker.overview_identity.section_id
    )
    assert running_section["metrics"][0]["key"] == "running"
    assert running_section["metrics"][0]["value"] == "1"
    assert running_section["items"][0]["status"] == "running"
    assert result.operation_id in running_section["items"][0]["detail"]

    dispatcher.run_next()
    completed_overview = bridge.get_state_surface(
        UiStateSurfaceRequest(surface_id=UiStateSurfaceId.UI_LIVE_OVERVIEW.value)
    )
    completed_section = next(
        section
        for section in completed_overview.payload["sections"]
        if section["section_id"] == operation_tracker.overview_identity.section_id
    )

    assert completed_section["metrics"][0]["key"] == "running"
    assert completed_section["metrics"][0]["value"] == "0"
    assert completed_section["metrics"][2]["key"] == "completed"
    assert completed_section["metrics"][2]["value"] == "1"
    assert completed_section["items"][0]["status"] == "completed"


def test_bridge_lists_and_restores_snapshots() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    ObjectStateRegistry.record_snapshot("before", scope_id=PLATE_SCOPE_ID)
    before_id = ObjectStateRegistry.get_branch_history()[-1].id
    state.update_parameter("x", 2)
    ObjectStateRegistry.record_snapshot("after", scope_id=PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(FakePlateManager()),
        dispatcher=InlineDispatcher(),
    )

    catalog = bridge.list_snapshots(UiSnapshotListRequest())
    restore = bridge.restore_snapshot(
        UiSnapshotRestoreRequest(
            snapshot_id=before_id,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert f"before [{PLATE_SCOPE_ID}]" in [
        snapshot.label for snapshot in catalog.snapshots
    ]
    assert f"after [{PLATE_SCOPE_ID}]" in [
        snapshot.label for snapshot in catalog.snapshots
    ]
    assert restore.restored
    assert restore.current_snapshot is not None
    assert restore.current_snapshot.snapshot_id == before_id


def test_ui_bridge_control_server_round_trips_documents_through_descriptor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    bridge = UiAgentBridgeService(
        provider_set=PlateManagerBridgeProviderSet(
            FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
        ),
        dispatcher=InlineDispatcher(),
    )
    server = UiBridgeControlServer(
        bridge,
        _bridge_server_config(tmp_path),
    )

    binding = server.start()
    try:
        service = UiBridgeService()

        status = service.status()
        catalog = service.list_documents()
        state_catalog = service.list_state_surfaces()
        document = service.get_document(
            UiCodeDocumentRequest(
                document_id=DOCUMENT_ID,
                selection_mode=ALL_SELECTION_MODE,
            )
        )
        state = service.get_state_surface(
            UiStateSurfaceRequest(
                surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
                selection_mode=ALL_SELECTION_MODE,
            )
        )

        assert status.reachable is True
        assert status.auth_required is True
        assert status.descriptor_status == "ok"
        assert status.bridge_instance_id == BRIDGE_INSTANCE_ID
        assert BRIDGE_AUTH_TOKEN not in set(_json_payload_values(to_jsonable(status)))
        assert catalog.documents[0].document_id == DOCUMENT_ID
        assert (
            state_catalog.surfaces[0].surface_id == UiStateSurfaceId.PLATE_MANAGER.value
        )
        assert document.source == VALID_SOURCE
        assert state.payload["rows"][0]["plate_scope_id"] == PLATE_SCOPE_ID
        assert Path(binding.descriptor_file_path).exists()
    finally:
        server.stop()

    assert not Path(binding.descriptor_file_path).exists()


def test_ui_bridge_control_server_preserves_bad_auth_error(tmp_path: Path) -> None:
    bridge = UiAgentBridgeService(dispatcher=InlineDispatcher())
    server = UiBridgeControlServer(
        bridge,
        _bridge_server_config(tmp_path),
    )

    binding = server.start()
    try:
        bad_connection = UiBridgeConnectionSpec(
            host=binding.connection.host,
            port=binding.connection.port,
            transport_mode=binding.connection.transport_mode,
            auth_token="wrong-token",
        )
        service = UiBridgeService(
            descriptor_resolver=_StaticUiBridgeDescriptorResolver(bad_connection)
        )

        catalog = service.list_documents()

        assert catalog.documents == ()
        assert catalog.errors[0].code == "ui_bridge_auth_failed"
    finally:
        server.stop()


class _StaticUiBridgeDescriptorResolver:
    def __init__(self, connection: UiBridgeConnectionSpec) -> None:
        self._connection = connection

    def resolve(
        self,
        connection: UiBridgeConnectionSpec | None,
    ) -> UiBridgeConnectionResolution:
        del connection
        return UiBridgeConnectionResolution.from_connection(self._connection)
