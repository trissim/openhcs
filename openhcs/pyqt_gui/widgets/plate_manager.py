"""
Plate Manager Widget for PyQt6

Manages plate selection, initialization, and execution with full feature parity
to the Textual TUI version. Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import os
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, List, Dict, Optional, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSignal
from metaclass_registry import AutoRegisterMeta
from typing_extensions import override

from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentTarget
from openhcs.agent.ui_bridge_actions import PlateManagerAction, PlateOperation
from openhcs.agent.ui_bridge_identities import (
    PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
)
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.orchestrator.orchestrator import (
    PipelineOrchestrator,
    OrchestratorState,
)
from openhcs.core.path_cache import PathCacheKey
from openhcs.core.selection import (
    SelectedAllSelectionMode as PlateManagerCodeSelectionMode,
    SelectedScopeIdsCarrier,
)
from polystore.base import _create_storage_registry
from objectstate.lazy_factory import (
    ensure_global_config_context,
)
from objectstate.object_state import ObjectState, ObjectStateRegistry
from objectstate import DataclassFieldAccess
from objectstate.collection_containers import RootState
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
from openhcs.pyqt_gui.services.plate_manager_batch_workflow import (
    PlateManagerBatchWorkflow,
)
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiOwnedStateSurfaceDeclaration,
)
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    PlateCompiledState,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugSnapshotAvailableNotification,
)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.config import UIConfig
from openhcs.pyqt_gui.services.plate_manager_state_projection import (
    PlateManagerStateProjectionService,
)
from openhcs.core.execution_state import (
    BUSY_MANAGER_STATES,
    ManagerExecutionState,
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
    parse_terminal_status,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ExecutionBatchRuntime,
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
    PlateManagerCodeMutationScope,
    PlateManagerCodeWorkflow,
    PlateManagerDeletionWorkflow,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)
from openhcs.serialization.pycodify_formatters import (
    LazyDataclassFieldEmissionState,
)
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineObjectStateBinding,
)
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
from openhcs.core.progress.debug_projection import (
    DebugRuntimeProjection,
    RuntimeProjectionBundle,
)
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugCommandType,
    DebugReplayMode,
    DebugSession,
    DebugSnapshot,
    DebugTerminalSummary,
)
from openhcs.interop.cellprofiler.plate_workspace import (
    CellProfilerPlateWorkspacePreparer,
    prepare_cellprofiler_input_workspace,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
        PipelineDebugSessionContext,
    )

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
    ) -> PlateValidationResult: ...


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
        if manager.is_any_plate_running():
            return PLATE_VALIDATION_OK

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
    """Applies terminal-status-owned UI effects for a completed plate."""

    manager: "PlateManagerWidget"
    plate_path: str
    completion: ExecutionCompletionPayload
    policy: TerminalExecutionStatus

    def apply_before_presentation(self) -> None:
        """Apply non-modal terminal effects while the batch is still active."""

        self._emit_status_message()
        self._apply_auto_add_output()

    def present_failure(self) -> None:
        """Present a terminal failure only after lifecycle finalization."""

        self._emit_failure_message()

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
    """Manage microscopy plates through initialization, compilation, and execution.

    Add a plate directory, edit its configuration, initialize source metadata,
    compile its pipeline, then run it. Results opens live measurement snapshots;
    Viewer opens plate metadata. Reinitialize after changing source bindings, and
    recompile after changing a pipeline before running it again.
    """

    TITLE = "Plate Manager"
    UI_STATE_SURFACE_DECLARATIONS = (
        UiOwnedStateSurfaceDeclaration(
            identity=PlateManagerStateSurfaceIdentityDeclaration,
            title="Plate manager state",
            payload_schema="openhcs.ui.plate_manager_state.v1",
            related_action_ids=tuple(action.value for action in PlateManagerAction),
        ),
        UiOwnedStateSurfaceDeclaration(
            identity=PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
            title="Live measurement results",
            payload_schema="openhcs.ui.live_measurements_state.v1",
            related_action_ids=(PlateManagerAction.VIEW_RESULTS.value,),
        ),
    )
    UI_BRIDGE_WIDGET_IDENTITY = PlateManagerWidgetIdentity
    HELP_KNOWLEDGE_TARGET = KnowledgeBaseDocumentTarget(
        document_id="openhcs_basic_interface",
        section_id="plate-manager",
    )
    ENABLE_STATUS_SCROLLING = True  # Marquee animation for long status messages
    BUTTON_CONFIGS = [
        ("Add", PlateManagerAction.ADD_PLATE.value, "Add new plate directory"),
        ("Del", PlateManagerAction.DELETE_PLATE.value, "Delete selected plates"),
        ("Edit", PlateManagerAction.EDIT_CONFIG.value, "Edit plate configuration"),
        ("Init", PlateManagerAction.INIT_PLATE.value, "Initialize selected plates"),
        ("Compile", PlateManagerAction.COMPILE_PLATE.value, "Compile plate pipelines"),
        ("Run", PlateManagerAction.RUN_PLATE.value, "Run/Stop plate execution"),
        ("Code", PlateManagerAction.CODE_PLATE.value, "Generate Python code"),
        (
            "Results",
            PlateManagerAction.VIEW_RESULTS.value,
            "View live measurement results",
        ),
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
    zmq_connection_status_changed = pyqtSignal(object)
    zmq_endpoint_compatibility_observed = pyqtSignal(object)
    orchestrator_state_changed = pyqtSignal(str, OrchestratorState)
    orchestrator_config_changed = pyqtSignal(str, object)
    manager_execution_state_changed = pyqtSignal(ManagerExecutionState)
    global_config_changed = pyqtSignal()
    pipeline_data_changed = pyqtSignal()
    cellprofiler_pipeline_imported = pyqtSignal(str)
    clear_subprocess_logs = pyqtSignal()
    progress_started = pyqtSignal(int)
    progress_updated = pyqtSignal(int)
    progress_finished = pyqtSignal()
    runtime_progress_projection_changed = pyqtSignal(object)
    debug_snapshot_available = pyqtSignal(object)
    live_measurement_available = pyqtSignal(object)
    runtime_artifact_available = pyqtSignal(object)
    compiled_artifact_inspection_changed = pyqtSignal(str, object)
    compilation_error = pyqtSignal(str, str)
    initialization_error = pyqtSignal(str, str)
    execution_error = pyqtSignal(str)
    _execution_complete_signal = pyqtSignal(dict, str)
    _execution_running_signal = pyqtSignal(str)
    _debug_snapshot_received_signal = pyqtSignal(object)
    _execution_error_signal = pyqtSignal(str)
    _all_plates_completed_signal = pyqtSignal(int, int)

    def __init__(
        self,
        service_adapter,
        color_scheme: Optional[ColorScheme] = None,
        gui_config: "UIConfig | None" = None,
        parent=None,
    ):
        """
        Initialize the plate manager widget.

        Args:
            service_adapter: PyQt service adapter for dialogs and operations
            color_scheme: Color scheme for styling (optional, uses service adapter if None)
            gui_config: Resolved UI configuration used by this manager
            parent: Parent widget
        """
        if gui_config is None:
            raise TypeError("PlateManagerWidget requires the resolved UIConfig")
        self._ui_config: UIConfig = gui_config

        # Plate-specific state (BEFORE super().__init__)
        self.global_config = service_adapter.get_global_config()
        self._debug_terminal_summaries_by_plate: dict[
            str,
            DebugTerminalSummary,
        ] = {}
        self._debug_snapshots_by_plate: dict[str, tuple[DebugSnapshot, ...]] = {}

        # Business logic state (extracted from Textual version)
        # NOTE: self.plates is now a @property that derives from Root ObjectState
        # NOTE: Orchestrators are now stored in ObjectState (single source of truth for time-travel).
        #       Access via ObjectStateRegistry.get_object(plate_path) instead of self.orchestrators dict.
        self.selected_plate_path: str = ""
        self.plate_configs: Dict[str, Dict] = {}
        self.plate_compiled_data: Dict[str, PlateCompiledState] = {}
        self.current_execution_id: Optional[str] = (
            None  # Track current execution ID for cancellation
        )
        self._execution_state = ManagerExecutionState.IDLE
        self._active_debug_sessions: Dict[str, DebugSession] = {}
        self.live_measurement_model = LiveMeasurementTableModel()
        self.live_measurements_window: LiveMeasurementsWindow | None = None

        # Track per-plate execution state
        self.plate_execution_ids: Dict[str, str] = {}  # plate_path -> execution_id
        self.plate_terminal_activity_status = ExecutionBatchRuntime()

        # Use shared ExecutionProgressTracker singleton (same instance as ZMQ server browser)
        # This ensures both UI components show the same progress data
        self._progress_tracker = registry()
        self.plate_init_pending = set()
        self.plate_compile_pending = set()
        self.runtime_progress_projection = ExecutionRuntimeProjection()
        self.debug_runtime_projection = DebugRuntimeProjection.empty(
            self.runtime_progress_projection
        )
        self._state_projection_service = PlateManagerStateProjectionService()

        # Initialize base class (creates event bus, item list, buttons, and status label).
        super().__init__(service_adapter, color_scheme, parent=parent)

        # Compose Qt-signal consumers only after QWidget/QObject construction.
        self.zmq_client_service = ZMQClientService(
            config=gui_config.zmq,
            status_callback=self.zmq_connection_status_changed.emit,
            compatibility_callback=self.zmq_endpoint_compatibility_observed.emit,
        )
        self._batch_workflow_service = PlateManagerBatchWorkflow(
            self,
            zmq=ZMQExecutionClientBoundary(self.zmq_client_service),
            progress_config=gui_config.progress,
        )
        self._batch_workflow_service.start_progress_updates()
        self._batch_workflow_service.add_debug_snapshot_listener(
            self._debug_snapshot_received_signal.emit
        )
        self._batch_workflow_service.add_live_measurement_listener(
            self.live_measurement_available.emit
        )
        self._batch_workflow_service.add_runtime_artifact_listener(
            self.runtime_artifact_available.emit
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
        self._execution_running_signal.connect(self._on_execution_running)
        self._debug_snapshot_received_signal.connect(self._on_debug_snapshot_available)
        self._execution_error_signal.connect(self._on_execution_error)
        self._all_plates_completed_signal.connect(
            self._finalize_all_plates_completed_ui
        )
        self.live_measurement_available.connect(self._on_live_measurement_available)

        logger.debug("Plate manager widget initialized")

    def set_ui_config(self, config: "UIConfig") -> None:
        """Apply process UI settings at their live/future lifecycle owners."""

        self._ui_config = config
        self.zmq_client_service.set_config(config.zmq)
        self._batch_workflow_service.update_progress_config(config.progress)

    async def attach_existing_execution_server(self) -> bool:
        """Restore this manager's client session without starting a server."""

        return await self._batch_workflow_service.attach_existing_server()

    async def ensure_execution_server(self) -> bool:
        """Attach to or start this manager's persistent execution server."""

        return await self._batch_workflow_service.ensure_server()

    @property
    def execution_state(self) -> ManagerExecutionState:
        """Current PlateManager execution state, emitted on transition."""

        return self._execution_state

    @execution_state.setter
    def execution_state(self, state: ManagerExecutionState) -> None:
        if not isinstance(state, ManagerExecutionState):
            state = ManagerExecutionState(state)
        if state is self._execution_state:
            return
        self._execution_state = state
        self.manager_execution_state_changed.emit(state)

    def handle_button_action(self, action: str) -> None:
        dispatch_widget_action(
            widget=self,
            action_id=action,
            action_enum=PlateManagerAction,
            routes=self.ACTION_ROUTES,
            async_runner=self.service_adapter.execute_async_operation,
            before_dispatch=commit_focused_widget_edits,
        )

    def setup_ui(self) -> None:
        """Create the standard manager UI and append contextual help."""
        super().setup_ui()
        self.context_help_button = self.install_context_help_button(
            title_layout=self.manager_header.title_layout,
            object_name="plate_manager_help_button",
        )

    def cleanup(self):
        """Cleanup resources before widget destruction."""
        logger.info("🧹 Cleaning up PlateManagerWidget resources...")
        self._time_travel_binding.disconnect()
        self._list_visual_state.dispose()
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
        result = orchestrator.input_workspace_preparation_result
        if result is not None and result.pipeline_path is not None:
            return str(result.pipeline_path)
        if orchestrator.selected_pipeline_path is not None:
            return str(orchestrator.selected_pipeline_path)
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

    def emit_orchestrator_state(
        self,
        plate_path: str,
        state: OrchestratorState,
    ) -> None:
        self.orchestrator_state_changed.emit(plate_path, state)

    def emit_compiled_state(
        self,
        plate_path: str,
        compiled_state: PlateCompiledState | None,
    ) -> None:
        if compiled_state is None:
            self.plate_compiled_data.pop(plate_path, None)
        else:
            self.plate_compiled_data[plate_path] = compiled_state
        self.compiled_artifact_inspection_changed.emit(plate_path, compiled_state)

    def compiled_artifact_inspection_for_plate(self, plate_path: str):
        """Return the typed compiler projection retained for one plate."""

        state = self.plate_compiled_data.get(plate_path)
        return None if state is None else state.inspection

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

    def apply_runtime_projection(
        self,
        projection_bundle: RuntimeProjectionBundle,
    ) -> None:
        """Install and publish one atomically built runtime projection bundle."""

        self.runtime_progress_projection = projection_bundle.execution
        self.debug_runtime_projection = projection_bundle.debug
        self.runtime_progress_projection_changed.emit(projection_bundle.execution)

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

    def notify_plate_running(self, plate_path: str) -> None:
        """Marshal a background status-poller update onto the Qt thread."""

        self._execution_running_signal.emit(plate_path)

    def notify_all_plates_completed(
        self, completed_count: int, failed_count: int
    ) -> None:
        self._all_plates_completed_signal.emit(completed_count, failed_count)

    def _finalize_all_plates_completed_ui(
        self, completed_count: int, failed_count: int
    ) -> None:
        # Lifecycle state is authoritative and must not depend on transport
        # teardown or presentation succeeding.
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
        self.compilation_error.connect(self._handle_compilation_error)
        self.initialization_error.connect(self._handle_initialization_error)
        self.execution_error.connect(self._handle_execution_error)

    def _update_orchestrator_global_config(
        self,
        scope_id: str,
        orchestrator,
        new_global_config,
    ):
        """Publish a saved global config change to an orchestrator's dependents."""
        ensure_global_config_context(GlobalPipelineConfig, new_global_config)

        # Do not replace orchestrator.pipeline_config here. The registered plate
        # ObjectState owns the live/saved PipelineConfig draft; replacing the
        # delegate makes ObjectState treat the save as an external object
        # replacement and rebases away unsaved per-plate edits.
        logger.info(
            "Refreshed orchestrator global config context for plate: %s",
            orchestrator.plate_path,
        )

        effective_config = orchestrator.get_effective_config()
        self.orchestrator_config_changed.emit(scope_id, effective_config)

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
        plate_registry = _create_storage_registry()
        orchestrator = PipelineOrchestrator(
            plate_path=physical_plate_root,
            storage_registry=plate_registry,
            selected_pipeline_path=cppipe_path,
            transport_config=self._ui_config.zmq,
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

        self.orchestrator_state_changed.emit(plate_path, OrchestratorState.CREATED)
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
                input_workspace = None
                if cppipe_path is not None:
                    input_workspace = prepare_cellprofiler_input_workspace(
                        InputWorkspacePreparationRequest(
                            selected_path=plate_root,
                            selected_pipeline_path=cppipe_path,
                        )
                    )
                    orchestrator.bind_input_workspace(input_workspace)
                orchestrator.initialize()
                return input_workspace

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
                self.orchestrator_state_changed.emit(
                    plate_path,
                    OrchestratorState.READY,
                )

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
                    plate_path,
                    OrchestratorState.INIT_FAILED,
                )
                self.initialization_error.emit(row.name, str(e))

            self.progress_updated.emit(i + 1)

        await asyncio.gather(
            *[init_single_plate(i, p) for i, p in enumerate(selected_items)]
        )

        self.progress_finished.emit()

        # Count successes and failures
        selected_orchestrators = [
            ObjectStateRegistry.get_object(row.scope_id) for row in selected_items
        ]
        success_count = sum(
            orchestrator is not None and orchestrator.state == OrchestratorState.READY
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
            state = ObjectStateRegistry.get_by_scope(scope_id)
            orchestrator = ObjectStateRegistry.get_object(scope_id)
            if state is not None and orchestrator is not None:
                selected_orchestrator_entries.append((scope_id, state, orchestrator))

        if not selected_orchestrator_entries:
            self.service_adapter.show_error_dialog(
                "No initialized orchestrators selected."
            )
            return

        representative_state = selected_orchestrator_entries[0][1]
        current_plate_config = representative_state.saved_object
        if not isinstance(current_plate_config, PipelineConfig):
            self.service_adapter.show_error_dialog(
                "Selected orchestrator state does not delegate to PipelineConfig."
            )
            return

        def handle_config_save(new_config: PipelineConfig) -> None:
            logger.debug(f"🔍 CONFIG SAVE - new_config type: {type(new_config)}")
            for field in fields(new_config):
                raw_value = DataclassFieldAccess.raw_value(new_config, field.name)
                logger.debug(f"🔍 CONFIG SAVE - new_config.{field.name} = {raw_value}")

            for scope_id, _state, orchestrator in selected_orchestrator_entries:
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

            # Success message dialog removed for test automation compatibility

        # Open configuration window using PipelineConfig (not GlobalPipelineConfig)
        # PipelineConfig already imported from openhcs.core.config
        self._open_config_window(
            state=representative_state,
            on_save_callback=handle_config_save,
        )

    def _open_config_window(
        self,
        state: ObjectState,
        on_save_callback,
    ):
        """Open a configuration window for one authoritative config object.

        Singleton-per-scope behavior is handled automatically by BaseFormDialog.show().
        """
        from openhcs.pyqt_gui.windows.config_window import (
            ConfigSaveParticipant,
            ConfigWindowTabSpec,
        )

        config_window = ConfigWindow(
            tabs=(
                ConfigWindowTabSpec(
                    state=state,
                    save_participant=ConfigSaveParticipant(
                        apply=on_save_callback,
                        rollback=on_save_callback,
                    ),
                    before_mutation=(self.require_pipeline_definition_mutation_allowed),
                ),
            ),
            color_scheme=self.color_scheme,
            parent=self,
            scope_id=state.scope_id,
        )
        # BaseFormDialog.show() handles singleton-per-scope automatically
        config_window.show()
        config_window.raise_()
        config_window.activateWindow()

    def action_edit_global_config(self):
        """Open the application-owned Global/UI configuration window."""
        self.service_adapter.main_window.show_configuration()

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
            item for item in selected_items if item.scope_id in self.plate_compiled_data
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
            session = session.with_command(command_type)
            self._active_debug_sessions[target_plate_path] = session
            await self._batch_workflow_service.send_debug_worker_command(
                debug_session_id=session.debug_session_id,
                command_type=command_type,
            )
            if command_type is DebugCommandType.STOP:
                self._active_debug_sessions.pop(target_plate_path, None)
            return

        session = DebugSession.create(
            plate_id=target_plate_path,
            command_type=command_type,
        )
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

    def debug_session_for_plate(self, plate_path: str) -> DebugSession | None:
        """Return the active debug session for one plate."""

        return self._active_debug_sessions.get(plate_path)

    def _on_debug_snapshot_available(
        self,
        notification: DebugSnapshotAvailableNotification,
    ) -> None:
        """Record active debug cursor state, then forward the snapshot notification."""

        plate_path = notification.progress_event.plate_id
        if notification.snapshot is not None:
            self._remember_debug_snapshot(plate_path, notification.snapshot)
        session = self._active_debug_sessions.get(plate_path)
        if session is not None:
            debug_context = notification.debug_context
            self._active_debug_sessions[plate_path] = session.with_snapshot_store(
                snapshot_store_ref=debug_context.snapshot_store_ref,
                snapshot_store_backend=debug_context.snapshot_store_backend,
                axis_id=notification.progress_event.axis_id,
            ).with_cursor(debug_context.cursor)
        self.debug_snapshot_available.emit(notification)

    def _remember_debug_snapshot(
        self,
        plate_path: str,
        snapshot: DebugSnapshot,
    ) -> None:
        """Store the latest typed snapshot metadata for debugger projections."""

        current = self._debug_snapshots_by_plate.get(plate_path, ())
        retained = tuple(
            existing
            for existing in current
            if existing.snapshot_id != snapshot.snapshot_id
        )
        self._debug_snapshots_by_plate[plate_path] = (*retained, snapshot)

    def _last_debug_snapshot_for_plate(self, plate_path: str) -> DebugSnapshot | None:
        snapshots = self._debug_snapshots_by_plate.get(plate_path, ())
        if not snapshots:
            return None
        return snapshots[-1]

    def debug_terminal_summary_for_plate(
        self,
        plate_path: str,
    ) -> DebugTerminalSummary | None:
        """Return the terminal debug summary for one plate."""

        return self._debug_terminal_summaries_by_plate.get(str(plate_path))

    def supersede_debug_terminal_summaries_for_standard_run(
        self,
        plate_paths: list[str],
    ) -> None:
        """Retire completed debug presentation when a standard run begins."""

        for plate_path in plate_paths:
            self._debug_terminal_summaries_by_plate.pop(str(plate_path), None)

    def source_binding_context_for_plate(
        self,
        plate_path: str,
    ) -> SourceBindingContext | None:
        """Return the current orchestrator-owned source-binding context."""

        orchestrator = ObjectStateRegistry.get_object(str(plate_path))
        if not isinstance(orchestrator, PipelineOrchestrator):
            return None
        return orchestrator.source_binding_context(str(plate_path))

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

    async def action_inspect_debug_runtime(
        self,
        *,
        debug_session_id: str,
    ):
        """Read a renderer-independent live runtime inspection view."""

        return await self._batch_workflow_service.inspect_debug_runtime(
            debug_session_id=debug_session_id,
        )

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

    def _on_execution_running(self, plate_path: str) -> None:
        """Refresh execution presentation on the Qt thread."""

        self.update_item_list()
        self.emit_status(f"▶️ Running {plate_path}")

    def _on_execution_complete(self, result: dict, plate_path: str):
        """Handle execution completion for a single plate (called from main thread via signal)."""
        completion = ExecutionCompletionPayload.from_result(result)
        status = completion.status
        logger.info("Plate %s completed with status: %s", plate_path, status.value)

        self.plate_terminal_activity_status.mark_terminal(plate_path, status)
        policy_authority = TerminalCompletionUiPolicyAuthority(
            manager=self,
            plate_path=plate_path,
            completion=completion,
            policy=status,
        )

        new_state = status.orchestrator_state

        try:
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if orchestrator:
                orchestrator._state = new_state
                self.orchestrator_state_changed.emit(plate_path, new_state)
            try:
                policy_authority.apply_before_presentation()
            except Exception:
                logger.exception(
                    "Failed to apply non-modal terminal UI effects for %s",
                    plate_path,
                )
            try:
                self._clear_debug_session_for_plate(plate_path)
            except Exception:
                logger.exception(
                    "Failed to finalize debug-session state for %s",
                    plate_path,
                )
        finally:
            # Terminal bookkeeping and batch finalization must complete before
            # any modal presentation. A QMessageBox runs a nested event loop and
            # must never suspend the lifecycle authority in RUNNING.
            try:
                self.clear_plate_execution_tracking(
                    plate_path,
                    clear_terminal=False,
                )
            except Exception:
                logger.exception(
                    "Failed to clear execution tracking for %s",
                    plate_path,
                )
            self._maybe_reset_execution_state_after_stop()
            self._batch_workflow_service.components.execution_control.check_all_completed()
            self.refresh_execution_ui()
        policy_authority.present_failure()

    def _clear_debug_session_for_plate(self, plate_path: str) -> None:
        active_session = self._active_debug_sessions.pop(plate_path, None)
        if active_session is None or active_session.plate_id != plate_path:
            return

        terminal_status = self.plate_terminal_activity_status.terminal_status(
            plate_path
        )
        if terminal_status is not None:
            terminal_summary = DebugTerminalSummary.from_session(
                active_session,
                terminal_status=terminal_status.value,
                completed_at_unix=time.time(),
            )
            latest_snapshot = self._last_debug_snapshot_for_plate(plate_path)
            self._debug_terminal_summaries_by_plate[plate_path] = (
                terminal_summary.with_snapshot(
                    snapshot=latest_snapshot,
                    snapshot_id=(
                        None if latest_snapshot is None else latest_snapshot.snapshot_id
                    ),
                    snapshot_store_ref=active_session.snapshot_store_ref,
                    snapshot_store_backend=active_session.snapshot_store_backend,
                )
                if latest_snapshot is not None
                else terminal_summary
            )
        else:
            self._debug_terminal_summaries_by_plate.pop(plate_path, None)
        self.manager_execution_state_changed.emit(self.execution_state)

    def debug_session_context_for_plate(
        self,
        plate_path: str,
    ) -> "PipelineDebugSessionContext":
        """Project debug-session state for one plate without depending on an editor."""

        from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
            PipelineDebugPauseBoundaryState,
            PipelineDebugSessionContext,
            PipelineDebugTargetState,
        )
        from openhcs.ui.shared.plate_scope_identity import PipelineScopeIdentity

        plate_key = str(plate_path)
        target = PipelineDebugTargetState(
            current_plate_scope_id=plate_key,
            pipeline_scope_id=PipelineScopeIdentity.from_plate_scope(
                plate_key,
            ).scope_id,
            initialized=self._debug_target_initialized(plate_key),
            compiled=plate_key in self.plate_compiled_data,
            terminal_status=self._debug_terminal_status_value(plate_key),
        )
        return PipelineDebugSessionContext(
            target=target,
            session=self.debug_session_for_plate(plate_key),
            terminal_summary=self.debug_terminal_summary_for_plate(plate_key),
            pause_boundaries=PipelineDebugPauseBoundaryState(
                pause_step_indices=tuple(
                    index
                    for index, step in enumerate(
                        PipelineObjectStateBinding.steps_for_plate(plate_key)
                    )
                    if step.debug_pause
                )
            ),
            snapshots=self._debug_snapshots_by_plate.get(plate_key, ()),
            manager_execution_state=self.execution_state,
        )

    def _debug_target_initialized(self, plate_path: str) -> bool:
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if not isinstance(orchestrator, PipelineOrchestrator):
            return False
        return orchestrator.state.has_completed_initialization

    def _debug_terminal_status_value(self, plate_path: str) -> str | None:
        terminal_status = self.plate_terminal_activity_status.terminal_status(
            plate_path
        )
        if terminal_status is None:
            return None
        if isinstance(terminal_status, Enum):
            return terminal_status.value
        return str(terminal_status)

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

    def action_stop_execution(self, force: bool | None = None):
        """Handle Stop Execution via ZMQ.

        First click: Graceful shutdown, button changes to "Force Kill"
        Second click: Force shutdown
        """
        logger.info("🛑 action_stop_execution CALLED")

        is_force_kill = (
            self.buttons["run_plate"].text() == "Force Kill" if force is None else force
        )

        # Change button to "Force Kill" IMMEDIATELY (before any async operations)
        if not is_force_kill:
            logger.info("🛑 Stop button pressed - changing to Force Kill")
            self.execution_state = ManagerExecutionState.FORCE_KILL_READY
            self.update_button_states()
            QApplication.processEvents()
        else:
            # Force-kill requested: immediately disable stop interactions while
            # cancellation propagates from background threads.
            self.execution_state = ManagerExecutionState.STOPPING
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

        return self.orchestrator_code_document_context_for_rows(
            selected_items,
            selection_mode=selection_mode,
        )

    def orchestrator_code_document_context_for_rows(
        self,
        selected_items: list[PlateManagerRow],
        *,
        selection_mode: PlateManagerCodeSelectionMode = (
            PlateManagerCodeSelectionMode.ALL
        ),
    ) -> PlateManagerCodeDocumentContext:
        """Render orchestrator code for an explicit plate row collection."""
        plate_paths: list[str] = []
        pipeline_data: dict[str, list[FunctionStep]] = {}
        per_plate_configs: dict[str, PipelineConfig] = {}
        global_config = self._current_global_config_for_code_document()

        for row in selected_items:
            plate_path = row.scope_id
            plate_paths.append(plate_path)

            definition_pipeline = self._get_current_pipeline_definition(plate_path)
            pipeline_steps = list(definition_pipeline)
            if not pipeline_steps:
                logger.warning(
                    "No pipeline defined for %s, using empty pipeline",
                    row.name,
                )
                pipeline_steps = []

            pipeline_data[plate_path] = pipeline_steps

            pipeline_config = self.authored_pipeline_config_for_code_document(
                plate_path
            )
            per_plate_configs[plate_path] = pipeline_config

        payload = PlateManagerCodeDocumentAuthority.from_values(
            plate_paths=plate_paths,
            global_pipeline_config=global_config,
            per_plate_configs=per_plate_configs,
            pipeline_data=pipeline_data,
        )

        source = PlateManagerCodeDocumentAuthority.render(payload)
        return PlateManagerCodeDocumentContext(
            source=source,
            payload=payload,
            selection_mode=PlateManagerCodeSelectionMode(selection_mode).value,
            selected_scope_ids=tuple(str(row.scope_id) for row in selected_items),
        )

    def authored_pipeline_config_for_code_document(
        self,
        plate_path: str,
    ) -> PipelineConfig:
        """Return the canonical registered per-plate configuration."""
        state = ObjectStateRegistry.get_by_scope(plate_path)
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if state is None or orchestrator is None:
            raise ValueError(
                f"Plate {plate_path!r} has no registered orchestrator ObjectState."
            )
        pipeline_config = state.to_object(update_delegate=False)
        if not isinstance(pipeline_config, PipelineConfig):
            raise TypeError(
                "Plate ObjectState must reconstruct a PipelineConfig for "
                "code document serialization."
            )
        return LazyDataclassFieldEmissionState.retain_only_authored_paths(
            pipeline_config,
            state.signature_diff_fields,
        )

    def _current_global_config_for_code_document(self) -> GlobalPipelineConfig:
        """Return the canonical live global config for code-mode rendering."""
        global_state = ObjectStateRegistry.get_by_scope("")
        if global_state is None:
            return self.global_config
        return global_state.to_object(update_delegate=False)

    def _fallback_code_document_items(self) -> list[PlateManagerRow]:
        if self.plates:
            logger.info(
                "Code button pressed with no selection, falling back to all plates."
            )
            return list(self.plates)
        logger.info(
            "Code button pressed with no plates configured; generating empty template."
        )
        return []

    def code_document_execution_operations(
        self,
        mutation_scope: PlateManagerCodeMutationScope | None = None,
    ):
        """Return manager operations bound to one code-document mutation scope."""

        operations = self._action_operations()
        if mutation_scope is None:
            return operations
        return replace(
            operations,
            apply_code_namespace=PlateManagerCodeWorkflow(
                self,
                mutation_scope=mutation_scope,
            ).apply_namespace,
        )

    def action_code_plate(
        self,
        *,
        selection_mode: PlateManagerCodeSelectionMode = (
            PlateManagerCodeSelectionMode.SELECTED
        ),
    ):
        """Generate Python code for the requested plates and their pipelines."""
        logger.debug("Code button pressed - generating Python code for plates")

        try:
            context = self.orchestrator_code_document_context(
                selection_mode=selection_mode,
                empty_selection_policy=EmptyPlateSelectionPolicy.FALL_BACK_TO_ALL,
            )

            editor_service = SimpleCodeEditorService(self)
            use_external = external_editor_enabled()
            operations = self.code_document_execution_operations(
                PlateManagerCodeMutationScope.from_carrier(context)
            )
            editor_service.edit_code(
                initial_content=context.source,
                title="Edit Orchestrator Configuration",
                callback=lambda edited_code: self._action_controller.apply_edited_code(
                    operations,
                    edited_code,
                ),
                use_external=use_external,
                declaration_type=PlateManagerOrchestratorCodePayload,
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
        """Prepare for orchestrator code execution."""
        return

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
                viewer = PlateViewerWindow(
                    orchestrator=orchestrator,
                    zmq_config=self._ui_config.zmq,
                    progress_config=self._ui_config.progress,
                    parent=self,
                )
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
                zmq_config=self._ui_config.zmq,
                progress_config=self._ui_config.progress,
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
        if relation.source_plate_scope_id is not None:
            return ObjectStateRegistry.get_object(relation.source_plate_scope_id)

        orchestrator = ObjectStateRegistry.get_object(row.scope_id)
        if (
            orchestrator is not None
            and orchestrator.state is not OrchestratorState.CREATED
        ):
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
            plate.scope_id in self.plate_compiled_data for plate in selected_plates
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

    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        """Reject a pipeline/config mutation while execution owns the manager."""

        del plate_path
        if self.is_any_plate_running():
            raise RuntimeError(
                "Pipeline definitions cannot change while plate execution is active."
            )

    def require_pipeline_definition_mutation_allowed_for_scope(
        self,
        scope_id: str,
    ) -> None:
        """Authorize a bridge mutation when its ObjectState belongs to a plate."""

        if scope_id == "":
            self.require_pipeline_definition_mutation_allowed()
            return

        for row in self.plates:
            if row.identity.owns_object_state_scope(scope_id):
                self.require_pipeline_definition_mutation_allowed(row.scope_id)
                return

    # Event handlers (on_selection_changed, on_plates_reordered, on_item_double_clicked)
    # provided by AbstractManagerWidget base class
    # Plate-specific behavior implemented via abstract hooks below

    def on_orchestrator_state_changed(
        self,
        plate_path: str,
        state: OrchestratorState,
    ) -> None:
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

    def _get_current_pipeline_definition(
        self,
        plate_path: str,
    ) -> list[FunctionStep]:
        """
        Get the current pipeline definition for a plate.

        Args:
            plate_path: Path to the plate

        Returns:
            Mutable FunctionStep list for the plate.
        """
        pipeline_steps = PipelineObjectStateBinding.steps_for_plate(plate_path)
        logger.debug(
            "Loaded pipeline for plate %s from ObjectState with %d steps",
            plate_path,
            len(pipeline_steps),
        )
        return pipeline_steps

    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        """Invalidate compiled/run state after the Pipeline ObjectState changes."""
        self.require_pipeline_definition_mutation_allowed(plate_path)
        PlateManagerCodeWorkflow(self).invalidate_orchestrator_compilation_state(
            plate_path
        )
        self.pipeline_data_changed.emit()
        self.update_item_list()

    def refresh_prepared_cellprofiler_pipelines(self) -> None:
        """Publish imported CellProfiler pipeline state for all initialized rows."""

        for row in self.plates:
            self._load_cellprofiler_pipeline_from_orchestrator(row.scope_id)

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
        if input_workspace is None:
            return
        pipeline_steps = input_workspace.pipeline_steps
        pipeline_config = input_workspace.pipeline_config
        if pipeline_config is not None:
            PlateManagerCodeWorkflow(self).apply_per_plate_configs(
                {plate_path: pipeline_config}
            )
        if input_workspace.pipeline_import_error is not None:
            self.status_message.emit(
                "CellProfiler source workspace initialized; pipeline import failed: "
                f"{input_workspace.pipeline_import_error.message}"
            )
        if pipeline_steps is None:
            self.cellprofiler_pipeline_imported.emit(plate_path)
            return
        if not pipeline_steps:
            raise RuntimeError(
                f"CellProfiler pipeline import produced no steps for {plate_path}."
            )
        PipelineObjectStateBinding.update_plate_steps(plate_path, list(pipeline_steps))
        self.cellprofiler_pipeline_imported.emit(plate_path)
        self.status_message.emit(
            f"Imported {len(pipeline_steps)} CellProfiler step(s) for {Path(plate_path).name}"
        )

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
            )
            return
        self.item_list.setCurrentRow(0)

    @override
    def _handle_items_reordered(self, from_index: int, to_index: int) -> None:
        """Persist visible plate order through the authoritative root ObjectState."""

        root_state = self._ensure_root_state()
        scope_ids = root_orchestrator_scope_ids(root_state)
        scope_id = scope_ids.pop(from_index)
        scope_ids.insert(to_index, scope_id)
        root_state.update_parameter("orchestrator_scope_ids", scope_ids)

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
