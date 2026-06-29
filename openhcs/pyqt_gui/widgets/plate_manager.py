"""
Plate Manager Widget for PyQt6

Manages plate selection, initialization, and execution with full feature parity
to the Textual TUI version. Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import os
import asyncio
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, List, Dict, Optional, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, pyqtSignal
from metaclass_registry import AutoRegisterMeta
from typing_extensions import override

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
)
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings_view import SchemaContextSourceInventoryProvider
from openhcs.core.orchestrator.orchestrator import (
    PipelineOrchestrator,
    OrchestratorState,
)
from openhcs.core.path_cache import PathCacheKey
from openhcs.core.selection import (
    SelectedAllSelectionMode as PlateManagerCodeSelectionMode,
    SelectedScopeIdsCarrier,
)
from polystore.filemanager import FileManager
from polystore.base import _create_storage_registry
from openhcs.config_framework import LiveContextResolver
from openhcs.config_framework.lazy_factory import (
    ensure_global_config_context,
    rebuild_lazy_config_with_new_global_reference,
)
from openhcs.config_framework.global_config import (
    set_global_config_for_editing,
    get_current_global_config,
)
from openhcs.config_framework.context_manager import config_context
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from objectstate import DataclassFieldAccess
from openhcs.config_framework.collection_containers import RootState
from openhcs.core.config_cache import save_global_config_sync
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, BlankLine, CodeBlock, generate_python_source
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    consolidate_multi_plate_summaries,
)
from pyqt_reactive.theming import ColorScheme
from openhcs.pyqt_gui.windows.config_window import ConfigWindow
from openhcs.pyqt_gui.windows.plate_viewer_window import PlateViewerWindow
from openhcs.pyqt_gui.windows.live_measurements_window import (
    LiveMeasurementTableModel,
    LiveMeasurementsWindow,
)
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.widgets.shared.abstract_manager_widget import (
    AbstractManagerWidget,
    ListItemFormat,
)
from openhcs.pyqt_gui.widgets.shared.openhcs_manager_mixins import (
    OpenHCSSingleRowActionManagerMixin,
)
from pyqt_reactive.widgets.shared.manager_selection_controller import (
    ItemIdSelectionPayloadProjection,
)
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.services.zmq_server_info_parser import ExecutionServerInfo
from openhcs.pyqt_gui.services.plate_manager_batch_workflow import (
    PlateManagerBatchWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
)
from openhcs.pyqt_gui.services.plate_manager_state_projection import (
    PlateManagerStateProjectionService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    BUSY_MANAGER_STATES,
    ExecutionBatchRuntime,
    ManagerExecutionState,
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
    TerminalUiPolicy,
    parse_terminal_status,
    terminal_ui_policy,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
    ZMQExecutionClientBoundary,
    ZMQClientService,
)
from pyqt_reactive.widgets.shared.manager_item_hooks import (
    AttributeItemIdProjection,
    ManagerItemHooks,
)
from pyqt_reactive.widgets.shared.manager_state_binding import ManagerStateBinding
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerOrchestratorCodePayload,
    PlateManagerCodeWorkflow,
    PlateManagerDeletionWorkflow,
)
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.services.plate_manager_root_state import (
    root_orchestrator_scope_ids,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionRoute,
    dispatch_widget_action,
)
from openhcs.pyqt_gui.widgets.shared.services.qt_widget_edit_commit import (
    commit_focused_widget_edits,
)
from pyqt_reactive.widgets.shared.scope_visual_config import ListItemType
from openhcs.core.progress import registry
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
)
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugCommandType,
    DebugReplayMode,
    DebugSession,
)
from openhcs.interop.cellprofiler.plate_workspace import (
    CellProfilerPlateWorkspacePreparer,
    CellProfilerPlateWorkspaceRequest,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

# Root ObjectState scope - tracks all plates in the application
# NOTE: Cannot use "" as scope_id - that's already used by GlobalPipelineConfig in app.py
ROOT_SCOPE_ID = "__plates__"


def external_editor_enabled() -> bool:
    """Return the explicit environment policy for launching the code editor."""
    variable_name = "OPENHCS_USE_EXTERNAL_EDITOR"
    if variable_name not in os.environ:
        return False
    return os.environ[variable_name].lower() in ("1", "true", "yes")


@dataclass(frozen=True, slots=True)
class PlateOrchestratorRegistration:
    """One visible plate-manager row backed by one orchestrator scope."""

    identity: PlateScopeIdentity
    select_by_default: bool = False

    @property
    def scope_id(self) -> str:
        return self.identity.scope_id

    @property
    def plate_root(self) -> Path:
        return self.identity.plate_root

    @property
    def cppipe_path(self) -> Path | None:
        return self.identity.cppipe_path

    @property
    def display_name(self) -> str:
        return self.identity.display_name


@dataclass(frozen=True, slots=True)
class PlateScopeNormalization:
    """Normalized root scope ids and orchestrators needed to back new rows."""

    scope_ids: tuple[str, ...]
    registrations_to_create: tuple[PlateOrchestratorRegistration, ...]


class CellProfilerPlateScopeNormalizer:
    """Normalize persisted physical CellProfiler folders into pipeline scopes."""

    def __init__(
        self,
        registration_resolver: Callable[
            [Path],
            tuple[PlateOrchestratorRegistration, ...],
        ],
        registration_creator: Callable[[PlateOrchestratorRegistration], ObjectState],
    ) -> None:
        self._registration_resolver = registration_resolver
        self._registration_creator = registration_creator

    def normalize_root_state(self, root_state: ObjectState) -> None:
        scope_ids = tuple(root_orchestrator_scope_ids(root_state))
        if not scope_ids:
            return

        normalization = self.normalize(scope_ids)
        if normalization.scope_ids == scope_ids:
            return

        with ObjectStateRegistry.atomic("normalize CellProfiler pipeline scopes"):
            for registration in normalization.registrations_to_create:
                self._registration_creator(registration)
            root_state.update_parameter(
                "orchestrator_scope_ids",
                list(normalization.scope_ids),
            )

        logger.info(
            "Normalized CellProfiler plate scopes: %s -> %s",
            list(scope_ids),
            list(normalization.scope_ids),
        )

    def normalize(self, scope_ids: tuple[str, ...]) -> PlateScopeNormalization:
        normalized_scope_ids: list[str] = []
        registrations_to_create: list[PlateOrchestratorRegistration] = []

        for scope_id in scope_ids:
            if PlateScopeIdentity.from_scope_id(scope_id).cppipe_path is not None:
                target_scope_ids = (scope_id,)
            else:
                registrations = self._registration_resolver(Path(scope_id))
                if registrations and any(
                    registration.scope_id != scope_id for registration in registrations
                ):
                    registrations_to_create.extend(registrations)
                    target_scope_ids = tuple(
                        registration.scope_id for registration in registrations
                    )
                else:
                    target_scope_ids = (scope_id,)

            for target_scope_id in target_scope_ids:
                if target_scope_id not in normalized_scope_ids:
                    normalized_scope_ids.append(target_scope_id)

        return PlateScopeNormalization(
            scope_ids=tuple(normalized_scope_ids),
            registrations_to_create=tuple(registrations_to_create),
        )


class PlateOperation(str, Enum):
    """Closed set of batch operations that validate visible plate rows."""

    INIT = "init"
    COMPILE = "compile"
    RUN = "run"


class PlateManagerAction(str, Enum):
    """Closed set of PlateManager button actions and agent-facing semantics."""

    side_effects: tuple[str, ...]
    confirmation_required: bool
    plate_operation: PlateOperation | None

    def __new__(
        cls,
        value: str,
        side_effects: tuple[str, ...],
        confirmation_required: bool,
        plate_operation: PlateOperation | None,
    ) -> "PlateManagerAction":
        member = str.__new__(cls, value)
        member._value_ = value
        member.side_effects = side_effects
        member.confirmation_required = confirmation_required
        member.plate_operation = plate_operation
        return member

    ADD_PLATE = (
        "add_plate",
        ("opens_file_dialog", "mutates_plate_collection"),
        True,
        None,
    )
    DELETE_PLATE = (
        "del_plate",
        ("mutates_plate_collection",),
        True,
        None,
    )
    EDIT_CONFIG = (
        "edit_config",
        ("opens_config_window", "may_mutate_plate_config"),
        True,
        None,
    )
    INIT_PLATE = (
        "init_plate",
        ("starts_initialization_workflow",),
        True,
        PlateOperation.INIT,
    )
    COMPILE_PLATE = (
        "compile_plate",
        ("starts_compile_workflow",),
        True,
        PlateOperation.COMPILE,
    )
    RUN_PLATE = (
        "run_plate",
        ("starts_or_stops_execution_workflow",),
        True,
        PlateOperation.RUN,
    )
    CODE_PLATE = (
        "code_plate",
        ("opens_code_document_window",),
        False,
        None,
    )
    VIEW_RESULTS = (
        "view_results",
        ("opens_results_window",),
        False,
        None,
    )
    VIEW_METADATA = (
        "view_metadata",
        ("opens_metadata_window",),
        False,
        None,
    )


RUNNABLE_ORCHESTRATOR_STATES = frozenset(
    {
        OrchestratorState.COMPILED,
        OrchestratorState.COMPLETED,
    }
)


@dataclass(frozen=True, slots=True)
class PlateValidationResult:
    """Validation outcome for one plate row and one batch operation."""

    valid: bool
    reason: str
    message: str
    recovery_action: PlateManagerAction | None = None


PLATE_VALIDATION_OK = PlateValidationResult(valid=True, reason="ok", message="ok")


def rejected_plate_validation(
    reason: str,
    message: str,
    *,
    recovery_action: PlateManagerAction | None = None,
) -> PlateValidationResult:
    return PlateValidationResult(
        valid=False,
        reason=reason,
        message=message,
        recovery_action=recovery_action,
    )


class ExecutionCompletionField(str, Enum):
    """Serialized completion payload fields produced by the execution services."""

    STATUS = "status"
    AUTO_ADD_OUTPUT_PLATE = "auto_add_output_plate_to_plate_manager"
    OUTPUT_PLATE_ROOT = "output_plate_root"
    TRACEBACK = "traceback"
    MESSAGE = "message"


@dataclass(frozen=True, slots=True)
class ExecutionCompletionPayload:
    """Nominal view of the execution server completion result payload."""

    status: TerminalExecutionStatus
    auto_add_output_plate_to_plate_manager: bool | None
    output_plate_root: str | None
    traceback_text: str
    message: str

    @classmethod
    def from_result(cls, result: dict) -> "ExecutionCompletionPayload":
        if ExecutionCompletionField.STATUS not in result:
            raise RuntimeError("Execution completion result is missing status.")
        status = parse_terminal_status(result[ExecutionCompletionField.STATUS])

        auto_add_output_plate_to_plate_manager = None
        if ExecutionCompletionField.AUTO_ADD_OUTPUT_PLATE in result:
            auto_add_output_plate_to_plate_manager = bool(
                result[ExecutionCompletionField.AUTO_ADD_OUTPUT_PLATE]
            )

        output_plate_root = None
        if ExecutionCompletionField.OUTPUT_PLATE_ROOT in result:
            output_plate_root = str(result[ExecutionCompletionField.OUTPUT_PLATE_ROOT])

        traceback_text = ""
        if ExecutionCompletionField.TRACEBACK in result:
            traceback_text = str(result[ExecutionCompletionField.TRACEBACK])

        message = "Unknown error"
        if ExecutionCompletionField.MESSAGE in result:
            message = str(result[ExecutionCompletionField.MESSAGE])

        return cls(
            status=status,
            auto_add_output_plate_to_plate_manager=(
                auto_add_output_plate_to_plate_manager
            ),
            output_plate_root=output_plate_root,
            traceback_text=traceback_text,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class OrchestratorCodeSource:
    """Nominal source object for orchestrator code generation."""

    code_block: CodeBlock
    header: str = "# Edit this orchestrator configuration and save to apply changes"
    clean_mode: bool = True

    def render(self) -> str:
        return generate_python_source(
            self.code_block,
            self.header,
            self.clean_mode,
        )


class EmptyPlateSelectionPolicy(str, Enum):
    """How code document rendering handles an empty selected-plate set."""

    ALLOW_EMPTY = "allow_empty"
    ERROR = "error"
    FALL_BACK_TO_ALL = "fall_back_to_all"


@dataclass(frozen=True, slots=True)
class PlateManagerCodeDocumentContext(SelectedScopeIdsCarrier):
    """Rendered orchestrator code plus the semantic payload that produced it."""

    source: str
    payload: PlateManagerOrchestratorCodePayload
    clean_mode: bool = True

    def editor_code_data(self) -> "PlateManagerEditorCodeData":
        return PlateManagerEditorCodeData(
            clean_mode=self.clean_mode,
        )


@dataclass(frozen=True, slots=True)
class PlateManagerEditorCodeData:
    """Nominal code-editor metadata for regenerating plate-manager source."""

    clean_mode: bool

    def as_editor_payload(self) -> dict:
        return {"clean_mode": self.clean_mode}


class EmptyPlateSelectionPolicyRunner(ABC, metaclass=AutoRegisterMeta):
    """Strategy for resolving an empty code-document plate selection."""

    __registry_key__ = "policy"
    __skip_if_no_key__ = True

    policy: ClassVar[EmptyPlateSelectionPolicy | None] = None

    @classmethod
    def for_policy(
        cls,
        policy: EmptyPlateSelectionPolicy,
    ) -> "EmptyPlateSelectionPolicyRunner":
        return cls.__registry__[policy]()

    @abstractmethod
    def selected_items(self, manager: "PlateManagerWidget") -> list[PlateManagerRow]:
        raise NotImplementedError


class AllowEmptyPlateSelectionPolicyRunner(EmptyPlateSelectionPolicyRunner):
    policy = EmptyPlateSelectionPolicy.ALLOW_EMPTY

    def selected_items(self, manager: "PlateManagerWidget") -> list[PlateManagerRow]:
        del manager
        return []


class ErrorEmptyPlateSelectionPolicyRunner(EmptyPlateSelectionPolicyRunner):
    policy = EmptyPlateSelectionPolicy.ERROR

    def selected_items(self, manager: "PlateManagerWidget") -> list[PlateManagerRow]:
        del manager
        raise ValueError("No plates selected.")


class FallBackToAllEmptyPlateSelectionPolicyRunner(EmptyPlateSelectionPolicyRunner):
    policy = EmptyPlateSelectionPolicy.FALL_BACK_TO_ALL

    def selected_items(self, manager: "PlateManagerWidget") -> list[PlateManagerRow]:
        return manager._fallback_code_document_items()


class PlateOperationValidator(ABC, metaclass=AutoRegisterMeta):
    """Registered validation strategy for one PlateOperation."""

    __registry_key__ = "operation"
    __skip_if_no_key__ = True

    operation: ClassVar[PlateOperation | None] = None

    @classmethod
    def for_operation(cls, operation: PlateOperation) -> "PlateOperationValidator":
        return cls.__registry__[operation]()

    @abstractmethod
    def validate(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
    ) -> PlateValidationResult:
        ...


class InitPlateOperationValidator(PlateOperationValidator):
    operation = PlateOperation.INIT

    def validate(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
    ) -> PlateValidationResult:
        del manager, row
        return PLATE_VALIDATION_OK


class CompilePlateOperationValidator(PlateOperationValidator):
    operation = PlateOperation.COMPILE

    def validate(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
    ) -> PlateValidationResult:
        orch = ObjectStateRegistry.get_object(row.scope_id)
        if not orch:
            return rejected_plate_validation(
                "no_orchestrator_initialized",
                "Selected plate has no orchestrator; run init_plate before compile_plate.",
                recovery_action=PlateManagerAction.INIT_PLATE,
            )
        if not orch.state.has_completed_initialization:
            return rejected_plate_validation(
                "orchestrator_not_initialized",
                "Selected plate is not initialized; run init_plate before compile_plate.",
                recovery_action=PlateManagerAction.INIT_PLATE,
            )
        pipeline_steps = manager._get_current_pipeline_definition(row.scope_id)
        if not pipeline_steps:
            return rejected_plate_validation(
                "empty_pipeline_definition",
                "Selected plate has no pipeline definition to compile.",
            )
        return PLATE_VALIDATION_OK


class RunPlateOperationValidator(PlateOperationValidator):
    operation = PlateOperation.RUN

    def validate(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
    ) -> PlateValidationResult:
        orch = ObjectStateRegistry.get_object(row.scope_id)
        if not orch:
            return rejected_plate_validation(
                "no_orchestrator_initialized",
                "Selected plate has no orchestrator; run init_plate before run_plate.",
                recovery_action=PlateManagerAction.INIT_PLATE,
            )
        if not orch.state.has_completed_initialization:
            return rejected_plate_validation(
                "orchestrator_not_initialized",
                "Selected plate is not initialized; run init_plate before run_plate.",
                recovery_action=PlateManagerAction.INIT_PLATE,
            )
        if orch.state not in RUNNABLE_ORCHESTRATOR_STATES:
            return rejected_plate_validation(
                "orchestrator_state_not_runnable",
                (
                    "Selected plate is not in a runnable orchestrator state "
                    f"({orch.state.value}); run compile_plate before run_plate."
                ),
                recovery_action=PlateManagerAction.COMPILE_PLATE,
            )
        return PLATE_VALIDATION_OK


@dataclass(frozen=True, slots=True)
class TerminalCompletionUiPolicyAuthority:
    """Applies TerminalUiPolicy side effects for a completed plate."""

    manager: "PlateManagerWidget"
    plate_path: str
    completion: ExecutionCompletionPayload
    policy: TerminalUiPolicy

    def apply(self) -> None:
        self._emit_status_message()
        self._emit_failure_message()
        self._apply_auto_add_output()

    def _emit_status_message(self) -> None:
        status_prefix = self.policy.status_prefix
        if status_prefix is None:
            return
        self.manager.status_message.emit(f"{status_prefix} {self.plate_path}")

    def _emit_failure_message(self) -> None:
        if not self.policy.emit_failure:
            return
        self.manager.execution_error.emit(
            self.manager._build_execution_failure_message(
                self.plate_path,
                self.completion,
            )
        )

    def _apply_auto_add_output(self) -> None:
        if not self.policy.auto_add_output_plate:
            return
        if self.manager.execution_state == ManagerExecutionState.RUNNING:
            self.manager._maybe_auto_add_output_plate_orchestrator(
                self.plate_path,
                self.completion,
            )
            return
        logger.info(
            "Skipping auto-add output plate (execution_state=%s)",
            self.manager.execution_state,
        )


class PlateManagerWidget(OpenHCSSingleRowActionManagerMixin, AbstractManagerWidget):
    """
    PyQt6 Plate Manager Widget.

    Manages plate selection, initialization, compilation, and execution.
    Preserves all business logic from Textual version with clean PyQt6 UI.

    Uses CrossWindowPreviewMixin for reactive preview labels showing orchestrator
    config states (num_workers, well_filter, streaming configs, etc.).

    Auto-registers with ServiceRegistry for decoupled lookup by window factory.
    """

    TITLE = "Plate Manager"
    ENABLE_STATUS_SCROLLING = True  # Marquee animation for long status messages
    BUTTON_CONFIGS = [
        ("Add", PlateManagerAction.ADD_PLATE.value, "Add new plate directory"),
        ("Del", PlateManagerAction.DELETE_PLATE.value, "Delete selected plates"),
        ("Edit", PlateManagerAction.EDIT_CONFIG.value, "Edit plate configuration"),
        ("Init", PlateManagerAction.INIT_PLATE.value, "Initialize selected plates"),
        ("Compile", PlateManagerAction.COMPILE_PLATE.value, "Compile plate pipelines"),
        ("Run", PlateManagerAction.RUN_PLATE.value, "Run/Stop plate execution"),
        ("Code", PlateManagerAction.CODE_PLATE.value, "Generate Python code"),
        ("Results", PlateManagerAction.VIEW_RESULTS.value, "View live measurement results"),
        ("Viewer", PlateManagerAction.VIEW_METADATA.value, "View plate metadata"),
    ]
    ACTION_ROUTES = MappingProxyType(
        {
            route.action: route
            for route in (
                WidgetActionRoute(
                    PlateManagerAction.ADD_PLATE,
                    lambda widget: widget.action_add,
                ),
                WidgetActionRoute(
                    PlateManagerAction.DELETE_PLATE,
                    lambda widget: widget.action_delete,
                ),
                WidgetActionRoute(
                    PlateManagerAction.EDIT_CONFIG,
                    lambda widget: widget.action_edit_config,
                ),
                WidgetActionRoute(
                    PlateManagerAction.INIT_PLATE,
                    lambda widget: widget.action_init_plate,
                ),
                WidgetActionRoute(
                    PlateManagerAction.COMPILE_PLATE,
                    lambda widget: widget.action_compile_plate,
                ),
                WidgetActionRoute(
                    PlateManagerAction.RUN_PLATE,
                    lambda widget: (
                        widget.action_stop_execution
                        if widget.is_any_plate_running()
                        else widget.action_run_plate
                    ),
                ),
                WidgetActionRoute(
                    PlateManagerAction.CODE_PLATE,
                    lambda widget: widget.action_code_plate,
                ),
                WidgetActionRoute(
                    PlateManagerAction.VIEW_RESULTS,
                    lambda widget: widget.action_view_live_results,
                ),
                WidgetActionRoute(
                    PlateManagerAction.VIEW_METADATA,
                    lambda widget: widget.action_view_metadata,
                ),
            )
        }
    )
    ITEM_NAME_SINGULAR = "plate"
    ITEM_NAME_PLURAL = "plates"
    SELECTION_PAYLOAD_PROJECTION = ItemIdSelectionPayloadProjection()
    SELECTION_CLEARED_PAYLOAD = ""
    SCOPE_ITEM_TYPE = ListItemType.ORCHESTRATOR
    STATE_BINDING = ManagerStateBinding(
        items_attr="plates",
        selection_attr="selected_plate_path",
        selection_signal_attr="plate_selected",
    )
    ITEM_HOOKS = ManagerItemHooks(
        id_projection=AttributeItemIdProjection("scope_id"),
        preserve_selection_pred=lambda self: bool(self.plates),
    )
    # Signals
    plate_selected = pyqtSignal(str)
    status_message = pyqtSignal(str)
    orchestrator_state_changed = pyqtSignal(str, str)
    orchestrator_config_changed = pyqtSignal(str, object)
    global_config_changed = pyqtSignal()
    pipeline_data_changed = pyqtSignal()
    clear_subprocess_logs = pyqtSignal()
    progress_started = pyqtSignal(int)
    progress_updated = pyqtSignal(int)
    progress_finished = pyqtSignal()
    debug_snapshot_available = pyqtSignal(object)
    live_measurement_available = pyqtSignal(object)
    compilation_error = pyqtSignal(str, str)
    initialization_error = pyqtSignal(str, str)
    execution_error = pyqtSignal(str)
    _execution_complete_signal = pyqtSignal(dict, str)
    _execution_error_signal = pyqtSignal(str)
    _all_plates_completed_signal = pyqtSignal(int, int)

    def __init__(
        self,
        service_adapter,
        color_scheme: Optional[ColorScheme] = None,
        gui_config=None,
        parent=None,
    ):
        """
        Initialize the plate manager widget.

        Args:
            service_adapter: PyQt service adapter for dialogs and operations
            color_scheme: Color scheme for styling (optional, uses service adapter if None)
            gui_config: GUI configuration (optional, for API compatibility with ABC)
            parent: Parent widget
        """
        # Plate-specific state (BEFORE super().__init__)
        self.global_config = service_adapter.get_global_config()
        self._plate_pipeline_editor: "PipelineEditorWidget | None" = None

        # Business logic state (extracted from Textual version)
        # NOTE: self.plates is now a @property that derives from Root ObjectState
        # NOTE: Orchestrators are now stored in ObjectState (single source of truth for time-travel).
        #       Access via ObjectStateRegistry.get_object(plate_path) instead of self.orchestrators dict.
        self.selected_plate_path: str = ""
        self.plate_configs: Dict[str, Dict] = {}
        self.plate_compiled_data: Dict[str, tuple] = {}  # Store compiled pipeline data
        self.current_execution_id: Optional[str] = (
            None  # Track current execution ID for cancellation
        )
        self.execution_state = ManagerExecutionState.IDLE
        self._active_debug_sessions: Dict[str, DebugSession] = {}
        self.live_measurement_model = LiveMeasurementTableModel()
        self.live_measurements_window: LiveMeasurementsWindow | None = None

        # Track per-plate execution state
        self.plate_execution_ids: Dict[str, str] = {}  # plate_path -> execution_id
        self.plate_terminal_activity_status = ExecutionBatchRuntime()

        # Use shared ExecutionProgressTracker singleton (same instance as ZMQ server browser)
        # This ensures both UI components show the same progress data
        self._progress_tracker = registry()
        self.plate_progress: Dict[str, Dict] = {}
        self.plate_init_pending = set()
        self.plate_compile_pending = set()
        self.runtime_progress_projection = ExecutionRuntimeProjection()
        self.execution_server_info: ExecutionServerInfo | None = None
        self._state_projection_service = PlateManagerStateProjectionService()

        # Unified PlateManager batch workflow
        self._zmq_client_service = ZMQClientService(port=7777)
        self._batch_workflow_service = PlateManagerBatchWorkflow(
            self,
            zmq=ZMQExecutionClientBoundary(self._zmq_client_service),
        )

        # Initialize base class (creates style_generator, event_bus, item_list, buttons, status_label internally)
        super().__init__(service_adapter, color_scheme, gui_config, parent)
        self._batch_workflow_service.add_debug_snapshot_listener(
            self.debug_snapshot_available.emit
        )
        self._batch_workflow_service.add_live_measurement_listener(
            self.live_measurement_available.emit
        )
        self.code_execution_workflow = PlateManagerCodeWorkflow(self)
        self.deletion_workflow = PlateManagerDeletionWorkflow(self)

        def create_registration_orchestrator(
            registration: PlateOrchestratorRegistration,
        ) -> ObjectState:
            return self._create_orchestrator_for_plate(
                registration.scope_id,
                plate_root=registration.plate_root,
                cppipe_path=registration.cppipe_path,
            )

        CellProfilerPlateScopeNormalizer(
            self._orchestrator_registrations_for_selected_path,
            create_registration_orchestrator,
        ).normalize_root_state(self._ensure_root_state())

        # Setup UI (after base and subclass state is ready)
        self.setup_ui()
        self.setup_connections()
        self.update_button_states()

        # Connect internal signals for thread-safe completion handling
        self._execution_complete_signal.connect(self._on_execution_complete)
        self._execution_error_signal.connect(self._on_execution_error)
        self._all_plates_completed_signal.connect(
            self._finalize_all_plates_completed_ui
        )
        self.live_measurement_available.connect(self._on_live_measurement_available)
        self.plate_selected.connect(self._load_cellprofiler_pipeline_from_orchestrator)

        logger.debug("Plate manager widget initialized")

    @property
    def plate_pipeline_editor(self) -> "PipelineEditorWidget | None":
        return self._plate_pipeline_editor

    @property
    def pipeline_editor(self) -> "PipelineEditorWidget | None":
        return self._plate_pipeline_editor

    @pipeline_editor.setter
    def pipeline_editor(self, editor: "PipelineEditorWidget | None") -> None:
        self._plate_pipeline_editor = editor

    def handle_button_action(self, action: str) -> None:
        dispatch_widget_action(
            widget=self,
            action_id=action,
            action_enum=PlateManagerAction,
            routes=self.ACTION_ROUTES,
            async_runner=self.service_adapter.execute_async_operation,
            before_dispatch=commit_focused_widget_edits,
        )

    def cleanup(self):
        """Cleanup resources before widget destruction."""
        logger.info("🧹 Cleaning up PlateManagerWidget resources...")
        self._batch_workflow_service.cleanup()
        self._batch_workflow_service.disconnect()

        logger.info("✅ PlateManagerWidget cleanup completed")

    def on_time_travel_complete(self, dirty_states, triggering_scope):
        """Refresh UI after time travel.

        Called automatically by ObjectStateRegistry when time travel completes.
        Orchestrator lifecycle is now handled by ObjectState limbo - no dict to maintain.

        Args:
            dirty_states: List of (scope_id, ObjectState) tuples that changed
            triggering_scope: The scope that triggered the time travel (if any)
        """
        # Refresh UI to reflect current state
        if self.item_list:
            self.update_item_list()

        # Log for debugging
        root_state = self._ensure_root_state()
        current_paths = set(root_orchestrator_scope_ids(root_state))
        initialized = sum(
            1 for p in current_paths if ObjectStateRegistry.get_object(p) is not None
        )
        logger.info(
            f"🕰️ Time travel complete: {initialized}/{len(current_paths)} plates initialized"
        )

        # Clear plate configs cache - force reload from ObjectState
        # (PipelineConfig is properly restored by ObjectState time travel)
        logger.info(f"🕰️ Clearing {len(self.plate_configs)} plate config cache(s)")
        self.plate_configs.clear()

        # Note: orchestrator_scope_ids list is restored by time travel automatically
        # Update button states (Init Plate button should be enabled for non-initialized plates)
        self.update_button_states()

        # Update UI to reflect restored state
        self.update_item_list()
        logger.info("🕰️ Time travel cleanup complete")

    # ========== Root ObjectState Management ==========

    def _ensure_root_state(self) -> ObjectState:
        """Get or create root ObjectState tracking all plates.

        Returns:
            Root ObjectState with orchestrator_scope_ids parameter
        """
        state = ObjectStateRegistry.get_by_scope(ROOT_SCOPE_ID)
        if not state:
            root = RootState()
            state = ObjectState(object_instance=root, scope_id=ROOT_SCOPE_ID)
            ObjectStateRegistry.register(state, _skip_snapshot=True)
        return state

    @property
    def plates(self) -> list[PlateManagerRow]:
        """Derive plate list from root ObjectState.

        Converts orchestrator_scope_ids to typed visible rows.
        """
        root_state = self._ensure_root_state()
        scope_ids = root_orchestrator_scope_ids(root_state)

        return [
            PlateManagerRow.from_scope(
                scope_id,
                cppipe_path=self._cppipe_path_for_scope_id(scope_id),
            )
            for scope_id in scope_ids
        ]

    def _display_name_for_scope_id(
        self,
        scope_id: str,
    ) -> str:
        return PlateScopeIdentity.from_scope_id(scope_id).display_name

    def _plate_root_for_scope_id(
        self,
        scope_id: str,
    ) -> str:
        return str(PlateScopeIdentity.from_scope_id(scope_id).plate_root)

    def _cppipe_path_for_scope_id(
        self,
        scope_id: str,
    ) -> str | None:
        identity = PlateScopeIdentity.from_scope_id(scope_id)
        if identity.cppipe_path is not None:
            return str(identity.cppipe_path)

        orchestrator = ObjectStateRegistry.get_object(scope_id)
        if not isinstance(orchestrator, PipelineOrchestrator):
            return None
        request = orchestrator.input_workspace_preparation
        if request is not None and request.selected_pipeline_path is not None:
            return str(request.selected_pipeline_path)
        result = orchestrator.input_workspace_preparation_result
        if result is not None and result.pipeline_path is not None:
            return str(result.pipeline_path)
        return None

    def _orchestrator_registrations_for_selected_path(
        self,
        selected_path: Path,
    ) -> tuple[PlateOrchestratorRegistration, ...]:
        plate_root = Path(selected_path)
        preparer = CellProfilerPlateWorkspacePreparer.from_paths(plate_root)
        cppipe_paths = preparer.cppipe_paths()
        if len(cppipe_paths) <= 1:
            if cppipe_paths:
                cppipe_path = cppipe_paths[0]
                identity = PlateScopeIdentity.from_cellprofiler_pipeline(
                    plate_root,
                    cppipe_path,
                )
            else:
                cppipe_path = None
                identity = PlateScopeIdentity.from_plate_root(plate_root)
            return (
                PlateOrchestratorRegistration(
                    identity=identity,
                    select_by_default=True,
                ),
            )

        default_cppipe_path = preparer.default_cppipe_path()
        return tuple(
            PlateOrchestratorRegistration(
                identity=PlateScopeIdentity.from_cellprofiler_pipeline(
                    plate_root,
                    cppipe_path,
                ),
                select_by_default=cppipe_path == default_cppipe_path,
            )
            for cppipe_path in cppipe_paths
        )

    # ExecutionHost interface
    def emit_status(self, msg: str) -> None:
        self.status_message.emit(msg)

    def emit_error(self, msg: str) -> None:
        self.execution_error.emit(msg)

    def emit_orchestrator_state(self, plate_path: str, state: str) -> None:
        self.orchestrator_state_changed.emit(plate_path, state)

    def emit_execution_complete(self, result: dict, plate_path: str) -> None:
        self._execution_complete_signal.emit(result, plate_path)

    def emit_clear_logs(self) -> None:
        self.clear_subprocess_logs.emit()

    # CompilationHost interface
    def emit_progress_started(self, count: int) -> None:
        self.progress_started.emit(count)

    def emit_progress_updated(self, value: int) -> None:
        self.progress_updated.emit(value)

    def emit_progress_finished(self) -> None:
        self.progress_finished.emit()

    def emit_compilation_error(self, plate_name: str, error: str) -> None:
        self.compilation_error.emit(plate_name, error)

    def get_pipeline_definition(self, plate_path: str) -> List:
        return self._get_current_pipeline_definition(plate_path)

    def notify_plate_completed(
        self, plate_path: str, status: str, result: dict
    ) -> None:
        if "status" not in result:
            result["status"] = status
        self._execution_complete_signal.emit(result, plate_path)

    def notify_all_plates_completed(
        self, completed_count: int, failed_count: int
    ) -> None:
        self._all_plates_completed_signal.emit(completed_count, failed_count)

    def _finalize_all_plates_completed_ui(
        self, completed_count: int, failed_count: int
    ) -> None:
        self._batch_workflow_service.disconnect_async()
        self.execution_state = ManagerExecutionState.IDLE
        self.current_execution_id = None
        if (
            completed_count > 1
            and self.global_config.analysis_consolidation_config.enabled
        ):
            try:
                self._consolidate_multi_plate_results()
                self.status_message.emit(
                    f"All done: {completed_count} completed, {failed_count} failed. Global summary created."
                )
            except Exception as e:
                logger.error(f"Failed to create global summary: {e}", exc_info=True)
                self.status_message.emit(
                    f"All done: {completed_count} completed, {failed_count} failed. Global summary failed."
                )
        else:
            self.status_message.emit(
                f"All done: {completed_count} completed, {failed_count} failed"
            )
        self.refresh_execution_ui()

    # Declarative list item format for PlateManager
    # The config source is orchestrator.pipeline_config
    # Field abbreviations are declared on config classes via @global_pipeline_config(field_abbreviations=...)
    # Config indicators (NAP, FIJI, MAT) are auto-discovered via always_viewable_fields
    LIST_ITEM_FORMAT = ListItemFormat(
        first_line=(),  # No fields on first line (just name)
        preview_line=("num_workers",),
        detail_line_field="path",  # Show plate path as detail line
    )

    # ========== CrossWindowPreviewMixin Hooks ==========

    def _handle_full_preview_refresh(self) -> None:
        """Refresh all preview labels."""
        logger.info(
            "🔄 PlateManager._handle_full_preview_refresh: refreshing preview labels"
        )
        self.update_item_list()

    def format_item_for_display(
        self,
        item: PlateManagerRow,
        live_ctx=None,
    ) -> Tuple[str, str]:
        """Format plate item for display with preview."""
        return (self._format_plate_item_with_preview_text(item), item.scope_id)

    def _format_plate_item_with_preview_text(self, row: PlateManagerRow):
        """Format plate item with status and config preview labels.

        Uses declarative LIST_ITEM_FORMAT with orchestrator.pipeline_config as config source.
        """
        row_state = self._state_projection_service.project_row(
            self,
            row,
            selected_scope_ids=set(),
            output_relation=self._state_projection_service.output_relation_for(
                self,
                row,
            ),
        )

        # Preview resolution is keyed by the visible row scope. For CellProfiler
        # pipeline rows, orchestrator.plate_path is the physical plate root while
        # row.scope_id is the delegated PipelineConfig ObjectState scope.
        return self.build_item_display_from_format(
            item=row,
            item_name=row.name,
            status_prefix=row_state.status_prefix,
            detail_line=row.scope_id,
        )

    def setup_connections(self):
        """Setup signal/slot connections (base class + plate-specific)."""
        self.setup_manager_connections()
        self.orchestrator_state_changed.connect(self.on_orchestrator_state_changed)
        self.progress_started.connect(self._on_progress_started)
        self.progress_updated.connect(self._on_progress_updated)
        self.progress_finished.connect(self._on_progress_finished)
        self.compilation_error.connect(self._handle_compilation_error)
        self.initialization_error.connect(self._handle_initialization_error)
        self.execution_error.connect(self._handle_execution_error)

    def _update_orchestrator_global_config(
        self,
        scope_id: str,
        orchestrator,
        new_global_config,
    ):
        """Update orchestrator global config reference and rebuild pipeline config if needed."""
        ensure_global_config_context(GlobalPipelineConfig, new_global_config)

        current_config = orchestrator.pipeline_config or PipelineConfig()
        orchestrator.pipeline_config = rebuild_lazy_config_with_new_global_reference(
            current_config, new_global_config, GlobalPipelineConfig
        )
        logger.info(
            f"Rebuilt orchestrator-specific config for plate: {orchestrator.plate_path}"
        )

        # NOTE: ObjectState now auto-detects delegate changes, so no manual sync needed.
        # When the orchestrator's ObjectState is next accessed, it will automatically
        # detect that pipeline_config has been replaced and re-extract parameters.

        effective_config = orchestrator.get_effective_config()
        self.orchestrator_config_changed.emit(
            scope_id, effective_config
        )

    # ========== Business Logic Methods ==========

    def action_add_plate(self):
        """Handle Add Plate button."""
        # Use cached directory dialog with multi-selection support
        selected_paths = self.service_adapter.show_cached_directory_dialog(
            cache_key=PathCacheKey.PLATE_IMPORT,
            title="Select Plate Directory",
            fallback_path=Path.home(),
            allow_multiple=True,
        )

        if selected_paths:
            self.add_plate_callback(selected_paths)

    def add_plate_callback(self, selected_paths: List[Path]):
        """
        Handle plate directory selection (extracted from Textual version).

        Creates orchestrator immediately on plate addition (in CREATED state).
        This ensures every visible plate has a corresponding orchestrator object
        that can receive pipeline configs and other data before initialization.

        Args:
            selected_paths: List of selected directory paths
        """
        if not selected_paths:
            self.status_message.emit("Plate selection cancelled")
            return

        added_plates = []
        last_added_path = None
        preferred_added_path = None

        # Get current plate paths from root ObjectState
        root_state = self._ensure_root_state()
        current_paths = root_orchestrator_scope_ids(root_state)
        new_paths = list(current_paths)  # Copy for mutation

        for selected_path in selected_paths:
            registrations = self._orchestrator_registrations_for_selected_path(
                selected_path
            )
            for registration in registrations:
                plate_path = registration.scope_id

                # Check if plate already exists
                if plate_path in current_paths or plate_path in new_paths:
                    continue

                # Create orchestrator immediately (in CREATED state, not initialized)
                self._create_orchestrator_for_plate(
                    plate_path,
                    plate_root=registration.plate_root,
                    cppipe_path=registration.cppipe_path,
                )

                # Add plate path to root ObjectState
                new_paths.append(plate_path)
                added_plates.append(registration.display_name)
                last_added_path = plate_path
                if preferred_added_path is None and registration.select_by_default:
                    preferred_added_path = plate_path

        # Update root ObjectState if any plates were added
        if added_plates:
            # Prefer the CellProfiler pipeline row selected by the workspace policy.
            selection_path = preferred_added_path
            if selection_path is None:
                selection_path = last_added_path
            if selection_path is not None:
                self.selected_plate_path = selection_path

            # Atomic: register orchestrator(s) + update orchestrator_scope_ids together
            with ObjectStateRegistry.atomic("register orchestrators"):
                root_state.update_parameter("orchestrator_scope_ids", new_paths)

            self.update_item_list()
            if selection_path:
                logger.info(f"🔔 EMITTING plate_selected signal for: {selection_path}")
                self.plate_selected.emit(selection_path)
            self.status_message.emit(
                f"Added {len(added_plates)} plate(s): {', '.join(added_plates)}"
            )
        else:
            self.status_message.emit("No new plates added (duplicates skipped)")

    def _create_orchestrator_for_plate(
        self,
        plate_path: str,
        *,
        plate_root: Path | str | None = None,
        cppipe_path: Path | str | None = None,
    ) -> ObjectState:
        """
        Create an orchestrator for a plate (in CREATED state, not initialized).

        This is called when a plate is added to ensure every visible plate has
        a corresponding orchestrator object. The orchestrator can receive configs
        and pipeline data before the heavy initialization work is done.

        Args:
            plate_path: Orchestrator scope id.
            plate_root: Physical plate directory. Defaults to plate_path for
                ordinary one-orchestrator-per-folder entries.

        Returns:
            The created PipelineOrchestrator instance
        """
        # Skip if orchestrator already exists
        existing_state = ObjectStateRegistry.get_by_scope(plate_path)
        if existing_state:
            return existing_state

        if plate_root is None:
            physical_plate_root = Path(self._plate_root_for_scope_id(str(plate_path)))
        else:
            physical_plate_root = Path(plate_root)
        input_workspace_preparation = None
        if cppipe_path is not None:
            input_workspace_preparation = InputWorkspacePreparationRequest(
                selected_path=physical_plate_root,
                selected_pipeline_path=Path(cppipe_path),
            )
        plate_registry = _create_storage_registry()
        orchestrator = PipelineOrchestrator(
            plate_path=physical_plate_root,
            storage_registry=plate_registry,
            input_workspace_preparation=input_workspace_preparation,
        )

        # Apply any saved config (e.g., from code loading)
        saved_config = self.plate_configs.get(str(plate_path))
        if saved_config:
            orchestrator.apply_pipeline_config(saved_config)

        # Register Orchestrator ObjectState (single source of truth for time-travel)
        # Uses __objectstate_delegate__ to extract params from pipeline_config
        # Parent is GlobalPipelineConfig state for inheritance resolution
        orchestrator_state = ObjectState(
            object_instance=orchestrator,
            scope_id=str(plate_path),
            parent_state=ObjectStateRegistry.get_by_scope(""),  # Global scope
        )
        ObjectStateRegistry.register(orchestrator_state)

        self.orchestrator_state_changed.emit(
            plate_path, OrchestratorState.CREATED.value
        )
        logger.info(f"Created orchestrator for plate (CREATED state): {plate_path}")

        return orchestrator_state

    # action_delete_plate() REMOVED - now uses ABC's action_delete() template with deletion_workflow

    def _validate_plates_for_operation(
        self,
        plates: list[PlateManagerRow],
        operation_type: PlateOperation,
    ) -> list[PlateManagerRow]:
        """Unified functional validator for all plate operations with debug logging."""
        validator = PlateOperationValidator.for_operation(operation_type)
        invalid: list[PlateManagerRow] = []
        for row in plates:
            result = validator.validate(self, row)
            if not result.valid:
                invalid.append(row)
                # Greppable trace for troubleshooting validation failures
                logger.info(
                    "PLATE_VALIDATION [%s] plate=%s name=%s reason=%s",
                    operation_type.value,
                    row.scope_id,
                    row.name,
                    result.reason,
                )
        return invalid

    def _ensure_context(self):
        """Ensure global config context is set up (for worker threads)."""
        ensure_global_config_context(GlobalPipelineConfig, self.global_config)

    async def action_init_plate(self):
        """Handle Initialize Plate button with unified validation.

        Initializes existing orchestrators (created during plate addition).
        The heavy I/O work (scanning plate, building metadata cache) happens here.
        """
        self._ensure_context()
        selected_items = self.get_selected_items()
        self._validate_plates_for_operation(selected_items, PlateOperation.INIT)
        self.progress_started.emit(len(selected_items))

        async def init_single_plate(i, row: PlateManagerRow):
            plate_path = row.scope_id
            plate_root = Path(row.plate_root)
            cppipe_path = None
            if row.cppipe_path is not None:
                cppipe_path = Path(row.cppipe_path)

            # Get existing orchestrator (created during add) or create if missing
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if not orchestrator:
                # Edge case: orchestrator doesn't exist (e.g., loaded from old session)
                logger.warning(f"Orchestrator not found for {plate_path}, creating now")
                self._create_orchestrator_for_plate(
                    plate_path,
                    plate_root=plate_root,
                    cppipe_path=cppipe_path,
                )
                orchestrator = ObjectStateRegistry.get_object(plate_path)

            if orchestrator.state.skips_initialization:
                logger.info(
                    f"Orchestrator already initialized for {plate_path}, skipping"
                )
                self._load_cellprofiler_pipeline_from_orchestrator(plate_path)
                self.progress_updated.emit(i + 1)
                return

            self.plate_init_pending.add(plate_path)
            self.update_item_list()

            def do_init():
                self._ensure_context()
                if cppipe_path is not None:
                    orchestrator.set_input_workspace_preparation(
                        InputWorkspacePreparationRequest(
                            selected_path=plate_root,
                            selected_pipeline_path=cppipe_path,
                        )
                    )
                orchestrator.initialize()
                return orchestrator.input_workspace_preparation_result

            try:
                input_workspace = await asyncio.get_event_loop().run_in_executor(
                    None,
                    do_init,
                )
                self._load_cellprofiler_pipeline_from_workspace(
                    plate_path,
                    input_workspace,
                )
                self.plate_init_pending.remove(plate_path)
                self.update_item_list()
                self.orchestrator_state_changed.emit(plate_path, "READY")

                # If this plate is currently selected, emit signal to update pipeline editor
                # This ensures pipeline editor gets notified when the selected plate is initialized
                if plate_path == self.selected_plate_path:
                    logger.info(
                        f"🔔 EMITTING plate_selected after init for currently selected plate: {plate_path}"
                    )
                    self.plate_selected.emit(plate_path)
                elif not self.selected_plate_path:
                    # If no plate is selected, select this one
                    self.selected_plate_path = plate_path
                    logger.info(
                        f"🔔 EMITTING plate_selected after init (auto-selecting): {plate_path}"
                    )
                    self.plate_selected.emit(plate_path)
            except Exception as e:
                logger.error(
                    f"Failed to initialize plate {plate_path}: {e}", exc_info=True
                )
                orchestrator._state = OrchestratorState.INIT_FAILED
                self.plate_init_pending.remove(plate_path)
                self.update_item_list()
                self.orchestrator_state_changed.emit(
                    plate_path, OrchestratorState.INIT_FAILED.value
                )
                self.initialization_error.emit(row.name, str(e))

            self.progress_updated.emit(i + 1)

        await asyncio.gather(
            *[init_single_plate(i, p) for i, p in enumerate(selected_items)]
        )

        self.progress_finished.emit()

        # Count successes and failures
        selected_orchestrators = [
            ObjectStateRegistry.get_object(row.scope_id)
            for row in selected_items
        ]
        success_count = sum(
            orchestrator is not None
            and orchestrator.state == OrchestratorState.READY
            for orchestrator in selected_orchestrators
        )
        error_count = sum(
            orchestrator is not None
            and orchestrator.state == OrchestratorState.INIT_FAILED
            for orchestrator in selected_orchestrators
        )

        msg = (
            f"Successfully initialized {success_count} plate(s)"
            if error_count == 0
            else f"Initialized {success_count} plate(s), {error_count} error(s)"
        )
        self.status_message.emit(msg)

    def action_edit_config(self):
        """Handle Edit Config button - per-orchestrator PipelineConfig editing."""
        selected_items = self.get_selected_items()
        if not selected_items:
            self.service_adapter.show_error_dialog(
                "No plates selected for configuration."
            )
            return

        selected_orchestrator_entries = []
        for item in selected_items:
            scope_id = item.scope_id
            orchestrator = ObjectStateRegistry.get_object(scope_id)
            if orchestrator is not None:
                selected_orchestrator_entries.append((scope_id, orchestrator))

        if not selected_orchestrator_entries:
            self.service_adapter.show_error_dialog(
                "No initialized orchestrators selected."
            )
            return

        selected_orchestrators = [
            orchestrator for _, orchestrator in selected_orchestrator_entries
        ]
        representative_scope_id, representative_orchestrator = (
            selected_orchestrator_entries[0]
        )
        current_plate_config = representative_orchestrator.pipeline_config

        def handle_config_save(new_config: PipelineConfig) -> None:
            logger.debug(f"🔍 CONFIG SAVE - new_config type: {type(new_config)}")
            for field in fields(new_config):
                raw_value = DataclassFieldAccess.raw_value(new_config, field.name)
                logger.debug(f"🔍 CONFIG SAVE - new_config.{field.name} = {raw_value}")

            for scope_id, orchestrator in selected_orchestrator_entries:
                self.plate_configs[scope_id] = new_config
                # Direct synchronous call - no async needed
                orchestrator.apply_pipeline_config(new_config)
                # Emit signal for UI components to refresh
                effective_config = orchestrator.get_effective_config()
                self.orchestrator_config_changed.emit(scope_id, effective_config)

            # Auto-sync handles context restoration automatically when pipeline_config is accessed
            if self.selected_plate_path and ObjectStateRegistry.get_object(
                self.selected_plate_path
            ):
                logger.debug(
                    f"Orchestrator context automatically maintained after config save: {self.selected_plate_path}"
                )

            count = len(selected_orchestrators)
            # Success message dialog removed for test automation compatibility

        # Open configuration window using PipelineConfig (not GlobalPipelineConfig)
        # PipelineConfig already imported from openhcs.core.config
        self._open_config_window(
            config_class=PipelineConfig,
            current_config=current_plate_config,
            on_save_callback=handle_config_save,
            scope_id=representative_scope_id,
        )

    def _open_config_window(
        self,
        config_class,
        current_config,
        on_save_callback,
        *,
        scope_id: str | None,
    ):
        """Open configuration window with specified config class and current config.

        Singleton-per-scope behavior is handled automatically by BaseFormDialog.show().
        """
        config_window = ConfigWindow(
            config_class,
            current_config,
            on_save_callback,
            self.color_scheme,
            self,
            scope_id=scope_id,
        )
        # BaseFormDialog.show() handles singleton-per-scope automatically
        config_window.show()
        config_window.raise_()
        config_window.activateWindow()

    def action_edit_global_config(self):
        """Handle global configuration editing - affects all orchestrators."""
        current_global_config = (
            self.service_adapter.get_global_config() or GlobalPipelineConfig()
        )

        def handle_global_config_save(new_config: GlobalPipelineConfig) -> None:
            self.service_adapter.set_global_config(new_config)
            # FIX: Use set_saved_global_config() instead of set_global_config_for_editing()
            # set_global_config_for_editing() sets BOTH saved and live, which overwrites unsaved edits
            # set_saved_global_config() only updates saved, preserving live config for UI preview
            set_saved_global_config(GlobalPipelineConfig, new_config)
            self._save_global_config_to_cache(new_config)
            self.global_config = new_config

            for plate in self.plates:
                row = plate
                orchestrator = ObjectStateRegistry.get_object(row.scope_id)
                if orchestrator:
                    self._update_orchestrator_global_config(
                        row.scope_id,
                        orchestrator,
                        new_config,
                    )
            self.service_adapter.show_info_dialog(
                "Global configuration applied to all orchestrators"
            )

        self._open_config_window(
            config_class=GlobalPipelineConfig,
            current_config=current_global_config,
            on_save_callback=handle_global_config_save,
            scope_id="",  # Global scope - matches app.py registration
        )

    def _save_global_config_to_cache(self, config: GlobalPipelineConfig):
        """Save global config to cache for persistence between sessions."""
        try:
            success = save_global_config_sync(config)

            if success:
                logger.info("Global config saved to cache for session persistence")
            else:
                logger.error(
                    "Failed to save global config to cache - sync save returned False"
                )
        except Exception as e:
            logger.error(f"Failed to save global config to cache: {e}")
            # Don't show error dialog as this is not critical for immediate functionality

    async def action_compile_plate(self):
        """Handle Compile Plate button - compile pipelines for selected plates."""
        selected_items = self.get_selected_items()

        if not selected_items:
            logger.warning("No plates available for compilation")
            return

        # Unified validation using functional validator
        invalid_plates = self._validate_plates_for_operation(
            selected_items,
            PlateOperation.COMPILE,
        )

        # Let validation failures bubble up as status messages
        if invalid_plates:
            invalid_names = [plate.name for plate in invalid_plates]
            logger.info(
                "PLATE_VALIDATION [compile] blocked %d plate(s): %s",
                len(invalid_names),
                ", ".join(invalid_names),
            )
            self.status_message.emit(
                f"Cannot compile invalid plates: {', '.join(invalid_names)}"
            )
            return

        # Delegate to compilation service
        await self._batch_workflow_service.compile_plates(selected_items)

    async def action_run_plate(self):
        """Handle Run Plate button - execute compiled plates using ZMQ."""
        selected_items = self.get_selected_items()
        if not selected_items:
            self.execution_error.emit("No plates selected to run.")
            return

        ready_items = [
            item
            for item in selected_items
            if item.scope_id in self.plate_compiled_data
        ]
        if not ready_items:
            self.execution_error.emit(
                "Selected plates are not compiled. Please compile first."
            )
            return

        await self._batch_workflow_service.run_plates(ready_items)

    async def action_run_debug_plate(
        self,
        plate_path: str | None = None,
        *,
        command_type: DebugCommandType = DebugCommandType.RUN,
        snapshot_store_backend: str | None = None,
        selected_source_group: str | None = None,
        pause_step_indices: tuple[int, ...] = (),
        start_step_index: int = 0,
        start_after_invocation_key: str | None = None,
    ) -> None:
        """Run one selected plate through the bounded debug execution path."""

        target_plate_path = plate_path
        if target_plate_path is None:
            selected_items = self.get_selected_items()
            if not selected_items:
                self.execution_error.emit("No plate selected to debug.")
                return
            target_plate_path = selected_items[0].scope_id

        session = self._active_debug_sessions.get(target_plate_path)
        if session is not None:
            await self._batch_workflow_service.send_debug_worker_command(
                debug_session_id=session.debug_session_id,
                command_type=command_type,
            )
            if command_type is DebugCommandType.STOP:
                self._active_debug_sessions.pop(target_plate_path, None)
            return

        session = DebugSession.create(plate_id=target_plate_path)
        self._active_debug_sessions[target_plate_path] = session
        plate_root = Path(target_plate_path)
        snapshot_root = (
            plate_root if plate_root.is_dir() else plate_root.parent
        ) / ".openhcs_debug"
        await self._batch_workflow_service.run_debug_plate(
            plate_path=target_plate_path,
            debug_session_id=session.debug_session_id,
            snapshot_store_ref=str(snapshot_root),
            snapshot_store_backend=snapshot_store_backend,
            command_type=command_type,
            selected_source_group=selected_source_group,
            pause_step_indices=pause_step_indices,
            start_step_index=start_step_index,
            start_after_invocation_key=start_after_invocation_key,
            replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
        )

    async def action_export_debug_artifact(
        self,
        *,
        debug_session_id: str,
        artifact_ref: DebugArtifactRef,
        export_root: str,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
    ) -> str:
        """Export one debug artifact through the shared workflow service."""

        response = await self._batch_workflow_service.export_debug_artifact(
            debug_session_id=debug_session_id,
            artifact_ref=artifact_ref,
            export_root=export_root,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        )
        return response.exported_ref

    def _maybe_auto_add_output_plate_orchestrator(
        self,
        source_plate_path: str,
        result: ExecutionCompletionPayload,
    ) -> None:
        """Optionally add the computed output plate root as a new orchestrator.

        The ZMQ execution server attaches `output_plate_root` to the completion
        payload (computed by the compiler/path planner). If enabled via global
        config, we add that path to Plate Manager when the run completes.
        """
        auto_add_value = result.auto_add_output_plate_to_plate_manager
        if auto_add_value is None:
            raise RuntimeError(
                "Missing auto-add flag in completion result; expected from compile context."
            )

        auto_add = bool(auto_add_value)

        if not auto_add:
            return

        output_plate_root = result.output_plate_root
        if not output_plate_root:
            return

        output_plate_root = str(output_plate_root)

        root_state = self._ensure_root_state()
        current_paths = root_orchestrator_scope_ids(root_state)
        if output_plate_root in current_paths:
            return

        # PipelineOrchestrator requires a real directory for non-OMERO paths.
        # Ensure it exists so we can register an orchestrator.
        if not output_plate_root.startswith("/omero/"):
            out_path = Path(output_plate_root)
            try:
                out_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Auto-add output plate skipped (mkdir failed): {output_plate_root} ({e})"
                )

            if not out_path.is_dir():
                raise RuntimeError(
                    f"Auto-add output plate skipped (not a dir): {output_plate_root}"
                )

        # Create orchestrator and add to root scope list (do not change selection)
        self._create_orchestrator_for_plate(output_plate_root)
        new_paths = list(current_paths)
        new_paths.append(output_plate_root)

        with ObjectStateRegistry.atomic("auto-add output plate"):
            root_state.update_parameter("orchestrator_scope_ids", new_paths)

        self.update_item_list()
        logger.info(
            "Auto-added output plate orchestrator: %s (from %s)",
            output_plate_root,
            source_plate_path,
        )

    def _on_execution_complete(self, result: dict, plate_path: str):
        """Handle execution completion for a single plate (called from main thread via signal)."""
        completion = ExecutionCompletionPayload.from_result(result)
        status = completion.status
        logger.info("Plate %s completed with status: %s", plate_path, status.value)

        self.plate_progress.pop(plate_path, None)

        policy = terminal_ui_policy(status)
        self.plate_terminal_activity_status.mark_terminal(plate_path, status)

        TerminalCompletionUiPolicyAuthority(
            manager=self,
            plate_path=plate_path,
            completion=completion,
            policy=policy,
        ).apply()

        new_state = policy.orchestrator_state

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator:
            orchestrator._state = new_state
            self.orchestrator_state_changed.emit(plate_path, new_state.value)

        self.clear_plate_execution_tracking(plate_path, clear_terminal=False)
        self._active_debug_sessions.pop(plate_path, None)
        self._maybe_reset_execution_state_after_stop()
        self.refresh_execution_ui()

    @staticmethod
    def _build_execution_failure_message(
        plate_path: str,
        result: ExecutionCompletionPayload,
    ) -> str:
        if result.traceback_text:
            return f"Execution failed for {plate_path}:\n\n{result.traceback_text}"
        return f"Execution failed for {plate_path}: {result.message}"

    def _maybe_reset_execution_state_after_stop(self) -> None:
        """Reset run/stop UI once all plates are terminal after a stop request."""
        if self.execution_state not in STOP_PENDING_MANAGER_STATES:
            return

        if not self.plate_terminal_activity_status.all_batch_terminal():
            return

        server_info = self.execution_server_info
        if server_info is not None and (
            server_info.running_execution_entries
            or server_info.queued_execution_entries
        ):
            return

        self.execution_state = ManagerExecutionState.IDLE
        self.current_execution_id = None

    def _consolidate_multi_plate_results(self):
        """Consolidate results from multiple completed plates into a global summary."""
        summary_paths, plate_names = [], []
        path_config = self.global_config.path_planning_config
        analysis_config = self.global_config.analysis_consolidation_config

        for (
            plate_path_str,
            terminal_status,
        ) in self.plate_terminal_activity_status.terminal_status_by_plate.items():
            if terminal_status != TerminalExecutionStatus.COMPLETE:
                continue
            plate_path = Path(plate_path_str)
            base = (
                Path(path_config.global_output_folder)
                if path_config.global_output_folder
                else plate_path.parent
            )
            output_plate_root = (
                base / f"{plate_path.name}{path_config.output_dir_suffix}"
            )

            materialization_path = self.global_config.materialization_results_path
            results_dir = (
                Path(materialization_path)
                if Path(materialization_path).is_absolute()
                else output_plate_root / materialization_path
            )
            summary_path = results_dir / analysis_config.output_filename

            if summary_path.exists():
                summary_paths.append(str(summary_path))
                plate_names.append(output_plate_root.name)
            else:
                logger.warning(
                    f"No summary found for plate {plate_path} at {summary_path}"
                )

        if len(summary_paths) < 2:
            return

        global_output_dir = (
            Path(path_config.global_output_folder)
            if path_config.global_output_folder
            else Path(summary_paths[0]).parent.parent.parent
        )
        global_summary_path = (
            global_output_dir / analysis_config.global_summary_filename
        )

        logger.info(
            f"Consolidating {len(summary_paths)} summaries to {global_summary_path}"
        )
        consolidate_multi_plate_summaries(
            summary_paths=summary_paths,
            output_path=str(global_summary_path),
            plate_names=plate_names,
        )
        logger.info(f"✅ Global summary created: {global_summary_path}")

    def _on_execution_error(self, error_msg):
        """Handle execution error (called from main thread via signal)."""
        self.execution_error.emit(f"Execution error: {error_msg}")
        self.execution_state = ManagerExecutionState.IDLE
        self.current_execution_id = None
        self.refresh_execution_ui()

    def action_stop_execution(self):
        """Handle Stop Execution via ZMQ.

        First click: Graceful shutdown, button changes to "Force Kill"
        Second click: Force shutdown
        """
        logger.info("🛑 action_stop_execution CALLED")

        is_force_kill = self.buttons["run_plate"].text() == "Force Kill"

        # Change button to "Force Kill" IMMEDIATELY (before any async operations)
        if not is_force_kill:
            logger.info("🛑 Stop button pressed - changing to Force Kill")
            self.execution_state = ManagerExecutionState.FORCE_KILL_READY
            # Clear stale server info so state can properly reset when plates are terminal
            self.execution_server_info = None
            self.update_button_states()
            QApplication.processEvents()
        else:
            # Force-kill requested: immediately disable stop interactions while
            # cancellation propagates from background threads.
            self.execution_state = ManagerExecutionState.STOPPING
            # Clear stale server info so state can properly reset when plates are terminal
            self.execution_server_info = None
            self.update_button_states()

        self._batch_workflow_service.stop_execution(force=is_force_kill)

    def orchestrator_code_document_context(
        self,
        selection_mode: PlateManagerCodeSelectionMode = PlateManagerCodeSelectionMode.SELECTED,
        empty_selection_policy: EmptyPlateSelectionPolicy = EmptyPlateSelectionPolicy.ERROR,
    ) -> PlateManagerCodeDocumentContext:
        """Build the code-mode document shown by the plate-manager Code action."""
        selection_mode = PlateManagerCodeSelectionMode(selection_mode)
        empty_selection_policy = EmptyPlateSelectionPolicy(empty_selection_policy)
        selected_items = self.get_selected_items()
        if selection_mode is PlateManagerCodeSelectionMode.ALL:
            selected_items = list(self.plates)
        elif selected_items:
            selected_items = list(selected_items)
        else:
            selected_items = EmptyPlateSelectionPolicyRunner.for_policy(
                empty_selection_policy
            ).selected_items(self)

        return self.orchestrator_code_document_context_for_rows(selected_items)

    def orchestrator_code_document_context_for_rows(
        self,
        selected_items: list[PlateManagerRow],
    ) -> PlateManagerCodeDocumentContext:
        """Render orchestrator code for an explicit plate row collection."""
        plate_paths: list[str] = []
        pipeline_data: dict[str, list] = {}
        per_plate_configs: dict[str, PipelineConfig] = {}
        global_config = self._current_global_config_for_code_document()

        for row in selected_items:
            plate_path = row.scope_id
            plate_paths.append(plate_path)

            definition_pipeline = self._get_current_pipeline_definition(plate_path)
            if not definition_pipeline:
                logger.warning("No pipeline defined for %s, using empty pipeline", row.name)
                definition_pipeline = []

            pipeline_data[plate_path] = definition_pipeline

            pipeline_config = self._authored_pipeline_config_for_code_document(
                plate_path
            )
            if pipeline_config is not None:
                per_plate_configs[plate_path] = pipeline_config

        code_items = [
            Assignment("plate_paths", plate_paths),
            BlankLine(),
            Assignment("global_config", global_config),
            BlankLine(),
            Assignment("per_plate_configs", per_plate_configs),
            BlankLine(),
            Assignment("pipeline_data", pipeline_data),
        ]

        payload = PlateManagerOrchestratorCodePayload(
            plate_paths=tuple(str(path) for path in plate_paths),
            pipeline_data=pipeline_data,
            global_pipeline_config=global_config,
            per_plate_configs=per_plate_configs,
        )

        source = OrchestratorCodeSource(CodeBlock.from_items(code_items)).render()
        return PlateManagerCodeDocumentContext(
            source=source,
            payload=payload,
            selected_scope_ids=tuple(str(row.scope_id) for row in selected_items),
        )

    def _authored_pipeline_config_for_code_document(
        self,
        plate_path: str,
    ) -> PipelineConfig | None:
        """Return the per-plate config only when it carries authored state."""
        state = ObjectStateRegistry.get_by_scope(plate_path)
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if (
            state is not None
            and orchestrator is not None
            and orchestrator.pipeline_config is not None
        ):
            if state.dirty_fields or state.signature_diff_fields:
                pipeline_config = state.to_object(update_delegate=False)
                if not isinstance(pipeline_config, PipelineConfig):
                    raise TypeError(
                        "Plate ObjectState must reconstruct a PipelineConfig for "
                        "code document serialization."
                    )
                return pipeline_config
            return None

        pipeline_config = self.plate_configs.get(plate_path)
        if pipeline_config is None:
            return None
        if pipeline_config == PipelineConfig():
            return None
        return pipeline_config

    def _current_global_config_for_code_document(self) -> GlobalPipelineConfig:
        """Return the canonical live global config for code-mode rendering."""
        global_state = ObjectStateRegistry.get_by_scope("")
        if global_state is None:
            return self.global_config
        return global_state.to_object(update_delegate=False)

    def _fallback_code_document_items(self) -> list[PlateManagerRow]:
        if self.plates:
            logger.info("Code button pressed with no selection, falling back to all plates.")
            return list(self.plates)
        logger.info("Code button pressed with no plates configured; generating empty template.")
        return []

    def code_document_execution_operations(self):
        """Return the existing manager code-execution operation port."""
        return self._action_operations()

    def action_code_plate(self):
        """Generate Python code for selected plates and their pipelines (Tier 3)."""
        logger.debug("Code button pressed - generating Python code for plates")

        try:
            context = self.orchestrator_code_document_context(
                empty_selection_policy=EmptyPlateSelectionPolicy.FALL_BACK_TO_ALL,
            )

            editor_service = SimpleCodeEditorService(self)
            use_external = external_editor_enabled()
            editor_service.edit_code(
                initial_content=context.source,
                title="Edit Orchestrator Configuration",
                callback=self._handle_edited_code,
                use_external=use_external,
                code_type="orchestrator",
                code_data=context.editor_code_data().as_editor_payload(),
            )

        except Exception as e:
            logger.error(f"Failed to generate plate code: {e}")
            self.service_adapter.show_error_dialog(f"Failed to generate code: {str(e)}")

    def _get_orchestrator_for_path(self, plate_path: str):
        """Return orchestrator instance for the provided plate path string."""
        return ObjectStateRegistry.get_object(str(plate_path))

    # === Code Execution Hooks (ABC _handle_edited_code template) ===

    def _pre_code_execution(self) -> None:
        """Open pipeline editor window before processing orchestrator code."""
        main_window = self._find_main_window()
        if main_window is not None:
            main_window.show_pipeline_editor()

    def action_view_metadata(self):
        """View plate images and metadata in tabbed window."""
        selected_items = self.get_selected_items()
        if not selected_items:
            self.service_adapter.show_error_dialog("No plates selected.")
            return

        for row in selected_items:
            plate_path = row.scope_id

            # Check if orchestrator is initialized
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if not orchestrator:
                self.service_adapter.show_error_dialog(
                    f"Plate must be initialized to view: {plate_path}"
                )
                continue

            try:
                # Create plate viewer window with tabs (Image Browser + Metadata)
                viewer = PlateViewerWindow(orchestrator=orchestrator, parent=self)
                viewer.show()  # Use show() instead of exec() to allow multiple windows
            except Exception as e:
                logger.error(
                    f"Failed to open plate viewer for {plate_path}: {e}", exc_info=True
                )
                self.service_adapter.show_error_dialog(
                    f"Failed to open plate viewer: {str(e)}"
                )

    def action_view_live_results(self):
        """Open the live measurement results window."""
        orchestrator = self._live_results_viewer_orchestrator_for_selection()
        if self.live_measurements_window is None:
            self.live_measurements_window = LiveMeasurementsWindow(
                self.live_measurement_model,
                orchestrator=orchestrator,
                color_scheme=self.color_scheme,
                parent=self,
            )
        else:
            self.live_measurements_window.set_orchestrator(orchestrator)
        self.live_measurements_window.refresh(select_latest=True)
        self.live_measurements_window.show()
        self.live_measurements_window.raise_()
        self.live_measurements_window.activateWindow()

    def _on_live_measurement_available(
        self,
        notification: LiveMeasurementAvailableNotification,
    ) -> None:
        self.live_measurement_model.add_notification(notification)
        if self.live_measurements_window is not None:
            self.live_measurements_window.set_orchestrator(
                self._live_results_viewer_orchestrator_for_plate(
                    notification.event.plate_id
                )
            )
            self.live_measurements_window.refresh(select_latest=True)

    def reset_live_measurements(self) -> None:
        """Clear retained live measurement previews for a new execution batch."""
        self.live_measurement_model.clear()
        if self.live_measurements_window is not None:
            self.live_measurements_window.refresh()

    def _live_results_viewer_orchestrator_for_selection(self):
        selected_items = self.get_selected_items()
        if not selected_items:
            return None
        return self._live_results_viewer_orchestrator_for_row(selected_items[0])

    def _live_results_viewer_orchestrator_for_plate(self, plate_id: str):
        for row in self.plates:
            if row.scope_id == plate_id:
                orchestrator = self._live_results_viewer_orchestrator_for_row(row)
                if orchestrator is not None:
                    return orchestrator
                break
        return self._live_results_viewer_orchestrator_for_selection()

    def _live_results_viewer_orchestrator_for_row(self, row: PlateManagerRow):
        relation = self._state_projection_service.output_relation_for(self, row)
        if relation.output_plate_scope_id is not None:
            output_orchestrator = ObjectStateRegistry.get_object(
                relation.output_plate_scope_id
            )
            if output_orchestrator is not None:
                return output_orchestrator

        orchestrator = ObjectStateRegistry.get_object(row.scope_id)
        if relation.source_plate_scope_id is not None:
            return orchestrator
        if orchestrator is not None and orchestrator.state is not OrchestratorState.CREATED:
            return orchestrator
        return None

    # ========== UI Helper Methods ==========

    # update_item_list() REMOVED - uses ABC template with list update hooks

    def get_selected_orchestrator(self):
        """
        Get the orchestrator for the currently selected plate.

        Returns:
            PipelineOrchestrator or None if no plate selected or not initialized
        """
        if self.selected_plate_path:
            return ObjectStateRegistry.get_object(self.selected_plate_path)
        return None

    def update_button_states(self):
        """Update button enabled/disabled states based on selection."""
        selected_plates = self.get_selected_items()
        has_selection = len(selected_plates) > 0

        def _plate_is_initialized(row: PlateManagerRow):
            orchestrator = ObjectStateRegistry.get_object(row.scope_id)
            return orchestrator and orchestrator.state != OrchestratorState.CREATED

        has_initialized = any(_plate_is_initialized(plate) for plate in selected_plates)
        has_compiled = any(
            plate.scope_id in self.plate_compiled_data
            for plate in selected_plates
        )
        is_running = self.is_any_plate_running()

        # Update button states (logic extracted from Textual version)
        self.buttons["del_plate"].setEnabled(has_selection and not is_running)
        self.buttons["edit_config"].setEnabled(has_initialized and not is_running)
        self.buttons["init_plate"].setEnabled(has_selection and not is_running)
        self.buttons["compile_plate"].setEnabled(has_initialized and not is_running)
        # Code button available even without initialized plates so users can edit templates
        self.buttons["code_plate"].setEnabled(not is_running)
        self.buttons["view_metadata"].setEnabled(has_initialized and not is_running)

        # Run button - enabled if plates are compiled or if currently running (for stop)
        if self.execution_state == ManagerExecutionState.STOPPING:
            # Stopping state - keep button as "Stop" but disable it
            self.buttons["run_plate"].setEnabled(False)
            self.buttons["run_plate"].setText("Stop")
        elif self.execution_state == ManagerExecutionState.FORCE_KILL_READY:
            # Force kill ready state - button is "Force Kill" and enabled
            self.buttons["run_plate"].setEnabled(True)
            self.buttons["run_plate"].setText("Force Kill")
        elif is_running:
            # Running state - button is "Stop" and enabled
            self.buttons["run_plate"].setEnabled(True)
            self.buttons["run_plate"].setText("Stop")
        else:
            # Idle state - button is "Run" and enabled if plates are compiled
            self.buttons["run_plate"].setEnabled(has_compiled)
            self.buttons["run_plate"].setText("Run")

    def refresh_execution_ui(self) -> None:
        """Refresh list row statuses and action buttons after execution state changes."""
        self.update_item_list()
        self.update_button_states()

    @override
    def _get_item_scope_id(
        self,
        item: PlateManagerRow,
        index: int,
    ) -> Optional[str]:
        """Return the ObjectState scope id represented by a plate list item."""
        del index
        return item.scope_id

    def clear_plate_execution_tracking(
        self, plate_path: str, *, clear_terminal: bool = True
    ) -> None:
        """Clear per-plate execution runtime tracking.

        By default this also clears terminal execution status; pass ``clear_terminal=False``
        to preserve a terminal outcome label until the next explicit operation.
        """
        execution_id = self.plate_execution_ids.pop(plate_path, None)
        self.plate_terminal_activity_status.clear_plate(
            plate_path,
            clear_terminal=clear_terminal,
        )
        if execution_id:
            self._batch_workflow_service.clear_progress_execution(execution_id)

    def is_any_plate_running(self) -> bool:
        """
        Check if any plate is currently running.

        Returns:
            True if any plate is running, False otherwise
        """
        return self.execution_state in BUSY_MANAGER_STATES

    # Event handlers (on_selection_changed, on_plates_reordered, on_item_double_clicked)
    # provided by AbstractManagerWidget base class
    # Plate-specific behavior implemented via abstract hooks below

    def on_orchestrator_state_changed(self, plate_path: str, state: str):
        """
        Handle orchestrator state changes.

        Args:
            plate_path: Path of the plate
            state: New orchestrator state
        """
        self.update_item_list()
        logger.debug(f"Orchestrator state changed: {plate_path} -> {state}")

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.

        Args:
            new_config: New global configuration
        """
        self.global_config = new_config

        # Apply new global config to all existing orchestrators
        # This rebuilds their pipeline configs preserving concrete values
        count = 0
        for row in self.plates:
            orchestrator = ObjectStateRegistry.get_object(row.scope_id)
            if orchestrator:
                self._update_orchestrator_global_config(
                    row.scope_id,
                    orchestrator,
                    new_config,
                )
                count += 1

        # REMOVED: Thread-local modification - dual-axis resolver handles orchestrator context automatically

        logger.info(f"Applied new global config to {count} orchestrators")

        # SIMPLIFIED: Dual-axis resolver handles placeholder updates automatically

    # REMOVED: _refresh_all_parameter_form_placeholders and _refresh_widget_parameter_forms
    # SIMPLIFIED: Dual-axis resolver handles placeholder updates automatically

    # ========== Helper Methods ==========

    def _get_current_pipeline_definition(self, plate_path: str) -> List:
        """
        Get the current pipeline definition for a plate.

        Args:
            plate_path: Path to the plate

        Returns:
            List of pipeline steps or empty list if no pipeline
        """
        plate_pipeline_editor = self.plate_pipeline_editor
        if not plate_pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []
        pipeline_steps = plate_pipeline_editor.get_pipeline_for_plate(plate_path)
        logger.debug(
            "Loaded pipeline for plate %s from ObjectState with %d steps",
            plate_path,
            len(pipeline_steps),
        )
        return pipeline_steps

    def set_pipeline_editor(self, pipeline_editor: "PipelineEditorWidget") -> None:
        """
        Set the pipeline editor reference.

        Args:
            pipeline_editor: Pipeline editor widget instance
        """
        if self.plate_pipeline_editor is not None:
            self.debug_snapshot_available.disconnect(
                self.plate_pipeline_editor.show_debug_snapshot
            )
        self._plate_pipeline_editor = pipeline_editor
        self.debug_snapshot_available.connect(pipeline_editor.show_debug_snapshot)
        logger.debug("Pipeline editor reference set in plate manager")
        for row in self.plates:
            self._load_cellprofiler_pipeline_from_orchestrator(row.scope_id)

    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        """Invalidate compiled/run state after the Pipeline ObjectState changes."""
        PlateManagerCodeWorkflow(self).invalidate_orchestrator_compilation_state(
            plate_path
        )
        self.pipeline_data_changed.emit()
        self.update_item_list()

    # _find_main_window() moved to AbstractManagerWidget

    def _load_cellprofiler_pipeline_from_orchestrator(self, plate_path: str) -> None:
        """Seed pipeline editor state from an initialized orchestrator input workspace."""

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if not isinstance(orchestrator, PipelineOrchestrator):
            return
        self._load_cellprofiler_pipeline_from_workspace(
            plate_path,
            orchestrator.input_workspace_preparation_result,
        )

    def _load_cellprofiler_pipeline_from_workspace(
        self,
        plate_path: str,
        input_workspace: InputWorkspacePreparationResult | None,
    ) -> None:
        """Load the prepared CellProfiler pipeline into editor/ObjectState state."""
        plate_pipeline_editor = self.plate_pipeline_editor
        if plate_pipeline_editor is None:
            return
        if input_workspace is None:
            return
        prepared_pipeline = input_workspace.prepared_pipeline
        import_result = None
        if prepared_pipeline is not None:
            import_result = prepared_pipeline.import_result
        if input_workspace.pipeline_import_error is not None:
            self.status_message.emit(
                "CellProfiler source workspace initialized; pipeline import failed: "
                f"{input_workspace.pipeline_import_error.message}"
            )
        if input_workspace.source_schema is not None:
            plate_pipeline_editor.set_source_binding_context_for_plate(
                plate_path,
                SourceBindingContext(
                    logical_plate_id=plate_path,
                    display_plate_root=input_workspace.original_source_root,
                    execution_plate_path=input_workspace.execution_plate_path,
                    cppipe_path=input_workspace.pipeline_path,
                    source_schema=input_workspace.source_schema,
                    inventory_provider=SchemaContextSourceInventoryProvider(
                        input_workspace.execution_plate_path,
                    ),
                    import_result=import_result,
                ),
            )
        if prepared_pipeline is None:
            return
        pipeline_steps = list(prepared_pipeline.pipeline.steps)
        if not pipeline_steps:
            raise RuntimeError(
                f"CellProfiler pipeline import produced no steps for {plate_path}."
            )
        plate_pipeline_editor.cellprofiler_import_results_by_plate[plate_path] = (
            import_result
        )
        plate_pipeline_editor.update_pipeline_for_plate(plate_path, pipeline_steps)
        plate_pipeline_editor.refresh_loaded_pipeline_for_plate(
            plate_path,
            import_result,
            pipeline_steps,
        )
        self.status_message.emit(
            f"Imported {len(pipeline_steps)} CellProfiler step(s) for {Path(plate_path).name}"
        )

    def _on_progress_started(self, max_value: int):
        """Handle progress started signal - route to status bar."""
        # Progress is now displayed in the status bar instead of a separate widget
        # This method is kept for signal compatibility but doesn't need to do anything
        pass

    def _on_progress_updated(self, value: int):
        """Handle progress updated signal - route to status bar."""
        # Progress is now displayed in the status bar instead of a separate widget
        # This method is kept for signal compatibility but doesn't need to do anything
        pass

    def _on_progress_finished(self):
        """Handle progress finished signal - route to status bar."""
        # Progress is now displayed in the status bar instead of a separate widget
        # This method is kept for signal compatibility but doesn't need to do anything
        pass

    # ========== Abstract Hook Implementations (AbstractManagerWidget ABC) ==========

    # === CRUD Hooks ===

    def action_add(self) -> None:
        """Add plates via directory chooser."""
        self.action_add_plate()

    @override
    def show_item_editor(self, item: PlateManagerRow) -> None:
        """Show config window for plate (required abstract method)."""
        del item
        self.action_edit_config()  # Delegate to existing implementation

    # === List Update Hooks (domain-specific) ===

    @override
    def _format_item_content(
        self,
        item: PlateManagerRow,
        index: int,
        context: None,
    ) -> str:
        """Format plate for list display (dirty marker added by ABC)."""
        del index, context
        return self._format_plate_item_with_preview_text(item)

    @override
    def _get_list_item_tooltip(self, item: PlateManagerRow) -> str:
        """Get plate tooltip with orchestrator status."""
        orchestrator = ObjectStateRegistry.get_object(item.scope_id)
        if orchestrator:
            return f"Status: {orchestrator.state.value}"
        return ""

    @override
    def _post_update_list(self) -> None:
        """Keep the visible list row aligned with the semantic plate selection."""
        if not self.plates:
            return
        if self.selected_plate_path:
            from pyqt_reactive.widgets.mixins import restore_selection_by_id

            restore_selection_by_id(
                self.item_list,
                self.selected_plate_path,
                self.ITEM_HOOKS.item_id,
            )
            return
        self.item_list.setCurrentRow(0)

    # === Config Resolution Hooks ===

    @override
    def _get_scope_for_item(self, item: PlateManagerRow) -> str:
        """PlateManager: scope comes from the visible row identity."""
        return item.scope_id

    # === CrossWindowPreviewMixin Hook ===

    def _get_current_orchestrator(self):
        """Get orchestrator for current plate (required abstract method)."""
        return ObjectStateRegistry.get_object(self.selected_plate_path)

    # ========== End Abstract Hook Implementations ==========

    def _handle_compilation_error(self, plate_name: str, error_message: str):
        """Handle compilation error on main thread (slot)."""
        self.service_adapter.show_error_dialog(
            f"Compilation failed for {plate_name}: {error_message}"
        )

    def _handle_initialization_error(self, plate_name: str, error_message: str):
        """Handle initialization error on main thread (slot)."""
        self.service_adapter.show_error_dialog(
            f"Failed to initialize {plate_name}: {error_message}"
        )

    def _handle_execution_error(self, error_message: str):
        """Handle execution error on main thread (slot)."""
        self.service_adapter.show_error_dialog(error_message)
