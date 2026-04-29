"""
Plate Manager Widget for PyQt6

Manages plate selection, initialization, and execution with full feature parity
to the Textual TUI version. Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import os
import asyncio
import traceback
from dataclasses import fields
from typing import List, Dict, Optional, Any, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.orchestrator.orchestrator import (
    PipelineOrchestrator,
    OrchestratorState,
)
from openhcs.core.path_cache import PathCacheKey
from polystore.filemanager import FileManager
from polystore.base import _create_storage_registry
from openhcs.config_framework import LiveContextResolver
from openhcs.config_framework.lazy_factory import (
    ensure_global_config_context,
    rebuild_lazy_config_with_new_global_reference,
    is_global_config_type,
)
from openhcs.config_framework.global_config import (
    set_global_config_for_editing,
    get_current_global_config,
)
from openhcs.config_framework.context_manager import config_context
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.config_framework.collection_containers import RootState
from openhcs.core.config_cache import _sync_save_config
from openhcs.core.xdg_paths import get_config_file_path
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, BlankLine, CodeBlock, generate_python_source
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    consolidate_multi_plate_summaries,
)
from pyqt_reactive.theming import ColorScheme
from openhcs.pyqt_gui.windows.config_window import ConfigWindow
from openhcs.pyqt_gui.windows.plate_viewer_window import PlateViewerWindow
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.widgets.shared.abstract_manager_widget import (
    AbstractManagerWidget,
    ListItemFormat,
)
from pyqt_reactive.forms import ParameterFormManager
from pyqt_reactive.services import ExecutionServerInfo
from openhcs.pyqt_gui.widgets.shared.services.batch_workflow_service import (
    BatchWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_status_presenter import (
    PlateStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    BUSY_MANAGER_STATES,
    ExecutionBatchRuntime,
    ManagerExecutionState,
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
    parse_terminal_status,
    terminal_ui_policy,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
    ZMQClientService,
)
from pyqt_reactive.widgets.shared.scope_visual_config import ListItemType
from openhcs.core.progress import registry
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
)

logger = logging.getLogger(__name__)

# Root ObjectState scope - tracks all plates in the application
# NOTE: Cannot use "" as scope_id - that's already used by GlobalPipelineConfig in app.py
ROOT_SCOPE_ID = "__plates__"


class PlateManagerWidget(AbstractManagerWidget):
    """
    PyQt6 Plate Manager Widget.

    Manages plate selection, initialization, compilation, and execution.
    Preserves all business logic from Textual version with clean PyQt6 UI.

    Uses CrossWindowPreviewMixin for reactive preview labels showing orchestrator
    config states (num_workers, well_filter, streaming configs, etc.).

    Auto-registers with ServiceRegistry for decoupled lookup by window factory.
    """

    TITLE = "Plate Manager"
    BUTTON_GRID_COLUMNS = 0  # Single row with all buttons
    ENABLE_STATUS_SCROLLING = True  # Marquee animation for long status messages
    BUTTON_CONFIGS = [
        ("Add", "add_plate", "Add new plate directory"),
        ("Del", "del_plate", "Delete selected plates"),
        ("Edit", "edit_config", "Edit plate configuration"),
        ("Init", "init_plate", "Initialize selected plates"),
        ("Compile", "compile_plate", "Compile plate pipelines"),
        ("Run", "run_plate", "Run/Stop plate execution"),
        ("Code", "code_plate", "Generate Python code"),
        ("Viewer", "view_metadata", "View plate metadata"),
    ]
    ACTION_REGISTRY = {
        "add_plate": "action_add",
        "del_plate": "action_delete",
        "edit_config": "action_edit_config",
        "init_plate": "action_init_plate",
        "compile_plate": "action_compile_plate",
        "code_plate": "action_code_plate",
        "view_metadata": "action_view_metadata",
    }
    DYNAMIC_ACTIONS = {"run_plate": "_resolve_run_action"}
    ITEM_NAME_SINGULAR = "plate"
    ITEM_NAME_PLURAL = "plates"
    ITEM_HOOKS = {
        "id_accessor": "path",
        "backing_attr": "plates",
        "selection_attr": "selected_plate_path",
        "selection_signal": "plate_selected",
        "selection_emit_id": True,
        "selection_clear_value": "",
        "items_changed_signal": None,
        "list_item_data": "item",
        "preserve_selection_pred": lambda self: bool(self.plates),
        "scope_item_type": ListItemType.ORCHESTRATOR,
        "scope_id_attr": "path",
    }
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
        self.pipeline_editor = None  # Will be set by main window

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

        # Track per-plate execution state
        self.plate_execution_ids: Dict[str, str] = {}  # plate_path -> execution_id
        self.execution_runtime = ExecutionBatchRuntime()

        # Use shared ExecutionProgressTracker singleton (same instance as ZMQ server browser)
        # This ensures both UI components show the same progress data
        self._progress_tracker = registry()
        self.plate_progress: Dict[str, Dict] = {}  # Deprecated, kept for compatibility
        self.plate_init_pending = set()
        self.plate_compile_pending = set()
        self._runtime_progress_projection = ExecutionRuntimeProjection()
        self._execution_server_info: ExecutionServerInfo | None = None
        self._plate_status_presenter = PlateStatusPresenter()

        # Unified batch workflow service
        self._zmq_client_service = ZMQClientService(port=7777)
        self._batch_workflow_service = BatchWorkflowService(
            self, client_service=self._zmq_client_service
        )

        # Initialize base class (creates style_generator, event_bus, item_list, buttons, status_label internally)
        super().__init__(service_adapter, color_scheme, gui_config, parent)

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

        logger.debug("Plate manager widget initialized")

    def cleanup(self):
        """Cleanup resources before widget destruction."""
        logger.info("🧹 Cleaning up PlateManagerWidget resources...")
        self._batch_workflow_service.cleanup()
        self._batch_workflow_service.disconnect()

        logger.info("✅ PlateManagerWidget cleanup completed")

    def _on_time_travel_complete(self, dirty_states, triggering_scope):
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
        current_paths = set(root_state.parameters.get("orchestrator_scope_ids") or [])
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
    def plates(self) -> List[Dict]:
        """Derive plate list from root ObjectState.

        Converts orchestrator_scope_ids (List[str]) to List[Dict] for backwards compatibility.
        Each plate dict has 'name' (derived from path) and 'path' keys.
        """
        root_state = self._ensure_root_state()
        plate_paths = root_state.parameters.get("orchestrator_scope_ids") or []

        return [{"name": Path(path).name, "path": path} for path in plate_paths]

    # ExecutionHost interface
    def emit_status(self, msg: str) -> None:
        self.status_message.emit(msg)

    def set_runtime_progress_projection(
        self, projection: ExecutionRuntimeProjection
    ) -> None:
        self._runtime_progress_projection = projection

    def set_execution_server_info(self, info: ExecutionServerInfo | None) -> None:
        self._execution_server_info = info

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
        preview_line=(
            "num_workers",
            "vfs_config.materialization_backend",
            "path_planning_config.well_filter",
            "path_planning_config.output_dir_suffix",
            "path_planning_config.global_output_folder",
        ),
        detail_line_field="path",  # Show plate path as detail line
    )

    # ========== CrossWindowPreviewMixin Hooks ==========

    def _handle_full_preview_refresh(self) -> None:
        """Refresh all preview labels."""
        logger.info(
            "🔄 PlateManager._handle_full_preview_refresh: refreshing preview labels"
        )
        self.update_item_list()

    def _update_single_plate_item(self, plate_path: str):
        """Update a single plate item's preview text without rebuilding the list."""
        for i in range(self.item_list.count()):
            item = self.item_list.item(i)
            plate_data = item.data(Qt.ItemDataRole.UserRole)
            if plate_data and plate_data.get("path") == plate_path:
                display_text = self._format_plate_item_with_preview_text(plate_data)
                item.setText(display_text)
                self._set_item_styling_roles(
                    item, display_text, plate_data
                )  # ABC helper
                break

    def format_item_for_display(self, item: Dict, live_ctx=None) -> Tuple[str, str]:
        """Format plate item for display with preview (required abstract method)."""
        return (self._format_plate_item_with_preview_text(item), item["path"])

    def _format_plate_item_with_preview_text(self, plate: Dict):
        """Format plate item with status and config preview labels.

        Uses declarative LIST_ITEM_FORMAT with orchestrator.pipeline_config as config source.
        """
        status_prefix = ""
        plate_path = plate["path"]
        plate_key = str(plate_path)
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        terminal_status = self.execution_runtime.terminal_status(plate_key)
        execution_id = self.plate_execution_ids.get(plate_key)
        runtime_projection = self._runtime_progress_projection.get_plate(
            plate_id=plate_key,
            execution_id=execution_id,
        )
        is_active_execution = self.execution_runtime.is_active(plate_key) or (
            runtime_projection is not None
        )

        status_prefix = self._plate_status_presenter.build_status_prefix(
            orchestrator_state=orchestrator.state if orchestrator else None,
            is_init_pending=plate_key in self.plate_init_pending,
            is_compile_pending=plate_key in self.plate_compile_pending,
            is_execution_active=is_active_execution,
            terminal_status=terminal_status,
            queue_position=self._queued_execution_position_for_plate(plate_key),
            runtime_projection=runtime_projection,
        )

        # Use declarative format with orchestrator.pipeline_config as introspection source
        return self._build_item_display_from_format(
            item=orchestrator,
            item_name=plate["name"],
            status_prefix=status_prefix,
            detail_line=plate["path"],
        )

    def _queued_execution_position_for_plate(self, plate_id: str) -> Optional[int]:
        """Return queue position from latest execution server snapshot for this plate."""
        server_info = self._execution_server_info
        if server_info is None:
            return None

        for queued in server_info.queued_execution_entries:
            if queued.plate_id != plate_id:
                continue
            return queued.queue_position
        return None

    def setup_connections(self):
        """Setup signal/slot connections (base class + plate-specific)."""
        self._setup_connections()
        self.orchestrator_state_changed.connect(self.on_orchestrator_state_changed)
        self.progress_started.connect(self._on_progress_started)
        self.progress_updated.connect(self._on_progress_updated)
        self.progress_finished.connect(self._on_progress_finished)
        self.compilation_error.connect(self._handle_compilation_error)
        self.initialization_error.connect(self._handle_initialization_error)
        self.execution_error.connect(self._handle_execution_error)

    def _resolve_run_action(self) -> str:
        """Resolve run/stop action based on current state."""
        return (
            "action_stop_execution"
            if self.is_any_plate_running()
            else "action_run_plate"
        )

    def _update_orchestrator_global_config(self, orchestrator, new_global_config):
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
            str(orchestrator.plate_path), effective_config
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

        # Get current plate paths from root ObjectState
        root_state = self._ensure_root_state()
        current_paths = root_state.parameters.get("orchestrator_scope_ids") or []
        new_paths = list(current_paths)  # Copy for mutation

        for selected_path in selected_paths:
            plate_path = str(selected_path)

            # Check if plate already exists
            if plate_path in current_paths:
                continue

            # Create orchestrator immediately (in CREATED state, not initialized)
            orchestrator_state = self._create_orchestrator_for_plate(plate_path)

            # Add plate path to root ObjectState
            new_paths.append(plate_path)
            added_plates.append(selected_path.name)
            last_added_path = plate_path

        # Update root ObjectState if any plates were added
        if added_plates:
            # Atomic: register orchestrator(s) + update orchestrator_scope_ids together
            with ObjectStateRegistry.atomic("register orchestrators"):
                root_state.update_parameter("orchestrator_scope_ids", new_paths)

            self.update_item_list()
            # Select the last added plate to ensure pipeline assignment works correctly
            if last_added_path:
                self.selected_plate_path = last_added_path
                logger.info(f"🔔 EMITTING plate_selected signal for: {last_added_path}")
                self.plate_selected.emit(last_added_path)
            self.status_message.emit(
                f"Added {len(added_plates)} plate(s): {', '.join(added_plates)}"
            )
        else:
            self.status_message.emit("No new plates added (duplicates skipped)")

    def _create_orchestrator_for_plate(self, plate_path: str) -> ObjectState:
        """
        Create an orchestrator for a plate (in CREATED state, not initialized).

        This is called when a plate is added to ensure every visible plate has
        a corresponding orchestrator object. The orchestrator can receive configs
        and pipeline data before the heavy initialization work is done.

        Args:
            plate_path: Path to the plate directory

        Returns:
            The created PipelineOrchestrator instance
        """
        # Skip if orchestrator already exists
        existing_state = ObjectStateRegistry.get_by_scope(plate_path)
        if existing_state:
            return existing_state

        plate_registry = _create_storage_registry()
        orchestrator = PipelineOrchestrator(
            plate_path=plate_path, storage_registry=plate_registry
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

    # action_delete_plate() REMOVED - now uses ABC's action_delete() template with _perform_delete() hook

    def _validate_plates_for_operation(self, plates, operation_type):
        """Unified functional validator for all plate operations with debug logging."""

        def _validate_compile(p):
            orch = ObjectStateRegistry.get_object(p["path"])
            if not orch:
                return False, "no_orchestrator_initialized"
            pipeline_steps = self._get_current_pipeline_definition(p["path"])
            if not pipeline_steps:
                return False, "empty_pipeline_definition"
            return True, "ok"

        def _validate_run(p):
            orch = ObjectStateRegistry.get_object(p["path"])
            if not orch:
                return False, "no_orchestrator_initialized"
            if orch.state not in ["COMPILED", "COMPLETED"]:
                return False, f"orchestrator_state_not_runnable:{orch.state}"
            return True, "ok"

        validators = {
            "init": lambda p: (True, "ok"),  # Init can work on any plates
            "compile": _validate_compile,
            "run": _validate_run,
        }

        validator = validators.get(operation_type, lambda p: (True, "ok"))

        invalid = []
        for plate in plates:
            is_valid, reason = validator(plate)
            if not is_valid:
                invalid.append(plate)
                # Greppable trace for troubleshooting validation failures
                logger.info(
                    "PLATE_VALIDATION [%s] plate=%s name=%s reason=%s",
                    operation_type,
                    plate.get("path"),
                    plate.get("name"),
                    reason,
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
        self._validate_plates_for_operation(selected_items, "init")
        self.progress_started.emit(len(selected_items))

        async def init_single_plate(i, plate):
            plate_path = str(plate["path"])

            # Get existing orchestrator (created during add) or create if missing
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if not orchestrator:
                # Edge case: orchestrator doesn't exist (e.g., loaded from old session)
                logger.warning(f"Orchestrator not found for {plate_path}, creating now")
                orchestrator = self._create_orchestrator_for_plate(plate_path)

            # Skip if already initialized
            if orchestrator.state in [
                OrchestratorState.READY,
                OrchestratorState.COMPILED,
                OrchestratorState.COMPLETED,
            ]:
                logger.info(
                    f"Orchestrator already initialized for {plate_path}, skipping"
                )
                self.progress_updated.emit(i + 1)
                return

            self.plate_init_pending.add(plate_path)
            self.update_item_list()

            def do_init():
                self._ensure_context()
                return orchestrator.initialize()

            try:
                await asyncio.get_event_loop().run_in_executor(None, do_init)
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
                self.initialization_error.emit(plate["name"], str(e))

            self.progress_updated.emit(i + 1)

        await asyncio.gather(
            *[init_single_plate(i, p) for i, p in enumerate(selected_items)]
        )

        self.progress_finished.emit()

        # Count successes and failures
        success_count = len(
            [
                p
                for p in selected_items
                if ObjectStateRegistry.get_object(p["path"])
                and ObjectStateRegistry.get_object(p["path"]).state
                == OrchestratorState.READY
            ]
        )
        error_count = len(
            [
                p
                for p in selected_items
                if ObjectStateRegistry.get_object(p["path"])
                and ObjectStateRegistry.get_object(p["path"]).state
                == OrchestratorState.INIT_FAILED
            ]
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

        selected_orchestrators = [
            ObjectStateRegistry.get_object(item["path"])
            for item in selected_items
            if ObjectStateRegistry.get_object(item["path"]) is not None
        ]
        if not selected_orchestrators:
            self.service_adapter.show_error_dialog(
                "No initialized orchestrators selected."
            )
            return

        representative_orchestrator = selected_orchestrators[0]
        current_plate_config = representative_orchestrator.pipeline_config

        def handle_config_save(new_config: PipelineConfig) -> None:
            logger.debug(f"🔍 CONFIG SAVE - new_config type: {type(new_config)}")
            for field in fields(new_config):
                raw_value = object.__getattribute__(new_config, field.name)
                logger.debug(f"🔍 CONFIG SAVE - new_config.{field.name} = {raw_value}")

            for orchestrator in selected_orchestrators:
                plate_key = str(orchestrator.plate_path)
                self.plate_configs[plate_key] = new_config
                # Direct synchronous call - no async needed
                orchestrator.apply_pipeline_config(new_config)
                # Emit signal for UI components to refresh
                effective_config = orchestrator.get_effective_config()
                self.orchestrator_config_changed.emit(
                    str(orchestrator.plate_path), effective_config
                )

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
            orchestrator=representative_orchestrator,  # Pass orchestrator for context persistence
        )

    def _open_config_window(
        self, config_class, current_config, on_save_callback, orchestrator=None
    ):
        """Open configuration window with specified config class and current config.

        Singleton-per-scope behavior is handled automatically by BaseFormDialog.show().
        """
        # CRITICAL: GlobalPipelineConfig uses scope_id="" (empty string), not None
        # The ObjectState is registered with scope_id="" in app.py, so we must match it
        # to reuse the existing ObjectState instead of creating a new one
        if orchestrator:
            scope_id = str(orchestrator.plate_path)
        elif is_global_config_type(config_class):
            scope_id = ""  # Global scope - matches app.py registration
        else:
            scope_id = None

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
                orchestrator = ObjectStateRegistry.get_object(plate["path"])
                if orchestrator:
                    self._update_orchestrator_global_config(orchestrator, new_config)
            self.service_adapter.show_info_dialog(
                "Global configuration applied to all orchestrators"
            )

        self._open_config_window(
            config_class=GlobalPipelineConfig,
            current_config=current_global_config,
            on_save_callback=handle_global_config_save,
        )

    def _save_global_config_to_cache(self, config: GlobalPipelineConfig):
        """Save global config to cache for persistence between sessions."""
        try:
            cache_file = get_config_file_path("global_config.config")
            success = _sync_save_config(config, cache_file)

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
        invalid_plates = self._validate_plates_for_operation(selected_items, "compile")

        # Let validation failures bubble up as status messages
        if invalid_plates:
            invalid_names = [p["name"] for p in invalid_plates]
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
            if item.get("path") in self.plate_compiled_data
        ]
        if not ready_items:
            self.execution_error.emit(
                "Selected plates are not compiled. Please compile first."
            )
            return

        await self._batch_workflow_service.run_plates(ready_items)

    def _maybe_auto_add_output_plate_orchestrator(
        self, source_plate_path: str, result: dict
    ) -> None:
        """Optionally add the computed output plate root as a new orchestrator.

        The ZMQ execution server attaches `output_plate_root` to the completion
        payload (computed by the compiler/path planner). If enabled via global
        config, we add that path to Plate Manager when the run completes.
        """
        auto_add_value = (result or {}).get("auto_add_output_plate_to_plate_manager")
        if auto_add_value is None:
            raise RuntimeError(
                "Missing auto-add flag in completion result; expected from compile context."
            )

        auto_add = bool(auto_add_value)

        if not auto_add:
            return

        output_plate_root = (result or {}).get("output_plate_root")
        if not output_plate_root:
            return

        output_plate_root = str(output_plate_root)

        root_state = self._ensure_root_state()
        current_paths = root_state.parameters.get("orchestrator_scope_ids") or []
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

    def _on_execution_complete(self, result, plate_path):
        """Handle execution completion for a single plate (called from main thread via signal)."""
        status = parse_terminal_status(result.get("status"))
        logger.info("Plate %s completed with status: %s", plate_path, status.value)

        self.plate_progress.pop(plate_path, None)

        policy = terminal_ui_policy(status)
        self.execution_runtime.mark_terminal(plate_path, status)

        if policy.status_prefix:
            self.status_message.emit(f"{policy.status_prefix} {plate_path}")
        if policy.emit_failure:
            self.execution_error.emit(
                self._build_execution_failure_message(plate_path, result)
            )
        if (
            policy.auto_add_output_plate
            and self.execution_state == ManagerExecutionState.RUNNING
        ):
            self._maybe_auto_add_output_plate_orchestrator(plate_path, result)
        elif policy.auto_add_output_plate:
            logger.info(
                "Skipping auto-add output plate (execution_state=%s)",
                self.execution_state,
            )

        new_state = policy.orchestrator_state

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator:
            orchestrator._state = new_state
            self.orchestrator_state_changed.emit(plate_path, new_state.value)

        self.clear_plate_execution_tracking(plate_path, clear_terminal=False)
        self._maybe_reset_execution_state_after_stop()
        self.refresh_execution_ui()

    @staticmethod
    def _build_execution_failure_message(plate_path: str, result: dict) -> str:
        traceback_str = result.get("traceback", "")
        if traceback_str:
            return f"Execution failed for {plate_path}:\n\n{traceback_str}"
        error_msg = result.get("message", "Unknown error")
        return f"Execution failed for {plate_path}: {error_msg}"

    def _maybe_reset_execution_state_after_stop(self) -> None:
        """Reset run/stop UI once all plates are terminal after a stop request."""
        if self.execution_state not in STOP_PENDING_MANAGER_STATES:
            return

        if not self.execution_runtime.all_batch_terminal():
            return

        server_info = self._execution_server_info
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
        ) in self.execution_runtime.terminal_status_by_plate.items():
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
            self.set_execution_server_info(None)
            self.update_button_states()
            QApplication.processEvents()
        else:
            # Force-kill requested: immediately disable stop interactions while
            # cancellation propagates from background threads.
            self.execution_state = ManagerExecutionState.STOPPING
            # Clear stale server info so state can properly reset when plates are terminal
            self.set_execution_server_info(None)
            self.update_button_states()

        self._batch_workflow_service.stop_execution(force=is_force_kill)

    def action_code_plate(self):
        """Generate Python code for selected plates and their pipelines (Tier 3)."""
        logger.debug("Code button pressed - generating Python code for plates")

        selected_items = self.get_selected_items()
        if not selected_items:
            if self.plates:
                logger.info(
                    "Code button pressed with no selection, falling back to all plates."
                )
                selected_items = list(self.plates)
            else:
                logger.info(
                    "Code button pressed with no plates configured; generating empty template."
                )
                selected_items = []

        try:
            # Collect plate paths, pipeline data, and per-plate pipeline configs
            plate_paths = []
            pipeline_data = {}
            per_plate_configs = {}  # Store pipeline config for each plate

            for plate_data in selected_items:
                plate_path = plate_data["path"]
                plate_paths.append(plate_path)

                # Get pipeline definition for this plate
                definition_pipeline = self._get_current_pipeline_definition(plate_path)
                if not definition_pipeline:
                    logger.warning(
                        f"No pipeline defined for {plate_data['name']}, using empty pipeline"
                    )
                    definition_pipeline = []

                pipeline_data[plate_path] = definition_pipeline

                # Get the actual pipeline config from this plate's orchestrator
                orchestrator = ObjectStateRegistry.get_object(plate_path)
                if orchestrator:
                    per_plate_configs[plate_path] = orchestrator.pipeline_config

            pipeline_config = None
            code_items = [
                Assignment("plate_paths", plate_paths),
                BlankLine(),
                Assignment("global_config", self.global_config),
                BlankLine(),
            ]

            if per_plate_configs:
                code_items.append(Assignment("per_plate_configs", per_plate_configs))
                code_items.append(BlankLine())
            else:
                pipeline_config = PipelineConfig()
                code_items.append(Assignment("pipeline_config", pipeline_config))
                code_items.append(BlankLine())

            code_items.append(Assignment("pipeline_data", pipeline_data))

            python_code = generate_python_source(
                CodeBlock.from_items(code_items),
                header="# Edit this orchestrator configuration and save to apply changes",
                clean_mode=True,
            )

            editor_service = SimpleCodeEditorService(self)
            use_external = os.environ.get(
                "OPENHCS_USE_EXTERNAL_EDITOR", ""
            ).lower() in ("1", "true", "yes")
            code_data = {
                "clean_mode": True,
                "plate_paths": plate_paths,
                "pipeline_data": pipeline_data,
                "global_config": self.global_config,
                "per_plate_configs": per_plate_configs,
                "pipeline_config": pipeline_config,
            }
            editor_service.edit_code(
                initial_content=python_code,
                title="Edit Orchestrator Configuration",
                callback=self._handle_edited_code,
                use_external=use_external,
                code_type="orchestrator",
                code_data=code_data,
            )

        except Exception as e:
            logger.error(f"Failed to generate plate code: {e}")
            self.service_adapter.show_error_dialog(f"Failed to generate code: {str(e)}")

    # _patch_lazy_constructors() moved to AbstractManagerWidget

    def _ensure_plate_entries_from_code(self, plate_paths: List[str]) -> None:
        """Ensure that any plates referenced in orchestrator code exist in the UI list.

        Reuses the same logic as add_plate_callback - creates orchestrator for each plate.
        """
        if not plate_paths:
            return

        # Get current plate paths from root ObjectState
        root_state = self._ensure_root_state()
        current_paths = root_state.parameters.get("orchestrator_scope_ids") or []
        existing_paths = set(current_paths)
        new_paths = list(current_paths)  # Copy for mutation
        added_count = 0

        for plate_path in plate_paths:
            plate_str = str(plate_path)
            if plate_str in existing_paths:
                continue

            # Create orchestrator immediately (same as add_plate_callback)
            self._create_orchestrator_for_plate(plate_str)

            plate_name = Path(plate_str).name or plate_str
            new_paths.append(plate_str)
            existing_paths.add(plate_str)
            added_count += 1
            logger.info(f"Added plate '{plate_name}' from orchestrator code")

        if added_count:
            # Atomic: register orchestrator(s) + update orchestrator_scope_ids together
            with ObjectStateRegistry.atomic("register orchestrators"):
                root_state.update_parameter("orchestrator_scope_ids", new_paths)

            if self.item_list:
                self.update_item_list()
            status_message = f"Added {added_count} plate(s) from orchestrator code"
            self.status_message.emit(status_message)
            logger.info(status_message)

    def _get_orchestrator_for_path(self, plate_path: str):
        """Return orchestrator instance for the provided plate path string."""
        return ObjectStateRegistry.get_object(str(plate_path))

    # === Code Execution Hooks (ABC _handle_edited_code template) ===

    def _pre_code_execution(self) -> None:
        """Open pipeline editor window before processing orchestrator code."""
        main_window = self._find_main_window()
        if main_window and hasattr(main_window, "show_pipeline_editor"):
            main_window.show_pipeline_editor()

    def _apply_executed_code(self, namespace: dict) -> bool:
        """Extract orchestrator variables from namespace and apply to widget state."""
        if "plate_paths" not in namespace or "pipeline_data" not in namespace:
            return False

        new_plate_paths = namespace["plate_paths"]
        new_pipeline_data = namespace["pipeline_data"]
        self._ensure_plate_entries_from_code(new_plate_paths)

        # Update global config if present
        if "global_config" in namespace:
            self._apply_global_config_from_code(namespace["global_config"])

        # Handle per-plate configs (preferred) or single pipeline_config (legacy)
        if "per_plate_configs" in namespace:
            self._apply_per_plate_configs_from_code(namespace["per_plate_configs"])
        elif "pipeline_config" in namespace:
            self._apply_legacy_pipeline_config_from_code(
                namespace["pipeline_config"], new_plate_paths
            )

        # Update pipeline data for ALL affected plates
        self._apply_pipeline_data_from_code(new_pipeline_data)

        return True

    def _apply_global_config_from_code(self, new_global_config) -> None:
        """Apply global config from executed code."""
        self.global_config = new_global_config

        # Update the ObjectState for global config to point to new instance (using public API)
        global_state = ObjectStateRegistry.get_by_scope("")
        if global_state:
            global_state.update_object_instance(new_global_config)

        # Apply to all orchestrators
        for plate in self.plates:
            orchestrator = ObjectStateRegistry.get_object(plate["path"])
            if orchestrator:
                self._update_orchestrator_global_config(orchestrator, new_global_config)

        # Update service adapter
        self.service_adapter.set_global_config(new_global_config)
        self.global_config_changed.emit()

        # Broadcast to event bus
        self._broadcast_to_event_bus("config", new_global_config)

    def _apply_per_plate_configs_from_code(self, per_plate_configs: dict) -> None:
        """Apply per-plate pipeline configs from executed code."""
        last_pipeline_config = None
        for plate_path_str, new_pipeline_config in per_plate_configs.items():
            plate_key = str(plate_path_str)
            self.plate_configs[plate_key] = new_pipeline_config

            orchestrator = self._get_orchestrator_for_path(plate_key)
            if orchestrator:
                orchestrator.apply_pipeline_config(new_pipeline_config)
                effective_config = orchestrator.get_effective_config()
                self.orchestrator_config_changed.emit(
                    str(orchestrator.plate_path), effective_config
                )
                logger.debug(
                    f"Applied per-plate pipeline config to orchestrator: {orchestrator.plate_path}"
                )
            else:
                logger.info(
                    f"Stored pipeline config for {plate_key}; will apply when initialized."
                )

            last_pipeline_config = new_pipeline_config

        # Broadcast last config to event bus
        if last_pipeline_config:
            self._broadcast_to_event_bus("config", last_pipeline_config)

    def _apply_legacy_pipeline_config_from_code(
        self, new_pipeline_config, plate_paths: list
    ) -> None:
        """Apply legacy single pipeline_config to all plates."""
        # Broadcast to event bus
        self._broadcast_to_event_bus("config", new_pipeline_config)

        # Apply to all affected orchestrators
        for plate_path in plate_paths:
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if orchestrator:
                orchestrator.apply_pipeline_config(new_pipeline_config)
                effective_config = orchestrator.get_effective_config()
                self.orchestrator_config_changed.emit(str(plate_path), effective_config)
                logger.debug(
                    f"Applied tier 3 pipeline config to orchestrator: {plate_path}"
                )

    def _apply_pipeline_data_from_code(self, new_pipeline_data: dict) -> None:
        """Apply pipeline data for ALL affected plates with proper state invalidation."""
        if not self.pipeline_editor or not hasattr(
            self.pipeline_editor, "_update_pipeline_steps"
        ):
            logger.warning("No pipeline editor available to update pipeline data")
            self.pipeline_data_changed.emit()
            return

        current_plate = getattr(self.pipeline_editor, "current_plate", None)

        for plate_path, new_steps in new_pipeline_data.items():
            # Update pipeline data via ObjectState (not dict assignment - plate_pipelines is read-only)
            self.pipeline_editor._update_pipeline_steps(plate_path, new_steps)
            logger.debug(
                f"Updated pipeline for {plate_path} with {len(new_steps)} steps"
            )

            # Invalidate orchestrator state
            self._invalidate_orchestrator_compilation_state(plate_path)

            # If this is the currently displayed plate, trigger UI cascade
            if plate_path == current_plate:
                self.pipeline_editor.pipeline_steps = new_steps
                self.pipeline_editor.update_item_list()
                self.pipeline_editor.pipeline_changed.emit(new_steps)
                self._broadcast_to_event_bus("pipeline", new_steps)
                logger.debug(
                    f"Triggered UI cascade refresh for current plate: {plate_path}"
                )

        self.pipeline_data_changed.emit()

    # _broadcast_config_to_event_bus() and _broadcast_pipeline_to_event_bus() REMOVED
    # Now using ABC's generic _broadcast_to_event_bus(event_type, data)

    def _invalidate_orchestrator_compilation_state(self, plate_path: str):
        """Invalidate compilation state for an orchestrator when its pipeline changes.

        This ensures that tier 3 changes properly invalidate ALL affected orchestrators,
        not just the currently visible one.

        Args:
            plate_path: Path of the plate whose orchestrator state should be invalidated
        """
        # Clear compiled data from simple state
        if plate_path in self.plate_compiled_data:
            del self.plate_compiled_data[plate_path]
            logger.debug(f"Cleared compiled data for {plate_path}")

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator and orchestrator.state == OrchestratorState.COMPILED:
            orchestrator._state = OrchestratorState.READY
            self.orchestrator_state_changed.emit(plate_path, "READY")

    def action_view_metadata(self):
        """View plate images and metadata in tabbed window."""
        selected_items = self.get_selected_items()
        if not selected_items:
            self.service_adapter.show_error_dialog("No plates selected.")
            return

        for item in selected_items:
            plate_path = item["path"]

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

        def _plate_is_initialized(plate_dict):
            orchestrator = ObjectStateRegistry.get_object(plate_dict["path"])
            return orchestrator and orchestrator.state != OrchestratorState.CREATED

        has_initialized = any(_plate_is_initialized(plate) for plate in selected_plates)
        has_compiled = any(
            plate["path"] in self.plate_compiled_data for plate in selected_plates
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

    def clear_plate_execution_tracking(
        self, plate_path: str, *, clear_terminal: bool = True
    ) -> None:
        """Clear per-plate execution runtime tracking.

        By default this also clears terminal execution status; pass ``clear_terminal=False``
        to preserve a terminal outcome label until the next explicit operation.
        """
        execution_id = self.plate_execution_ids.pop(plate_path, None)
        self.execution_runtime.clear_plate(plate_path, clear_terminal=clear_terminal)
        if execution_id:
            self._progress_tracker.clear_execution(execution_id)

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
        for plate in self.plates:
            orchestrator = ObjectStateRegistry.get_object(plate["path"])
            if orchestrator:
                self._update_orchestrator_global_config(orchestrator, new_config)
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
        if not self.pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []
        pipeline_steps = self.pipeline_editor.get_pipeline_for_plate(plate_path)
        logger.debug(
            "Loaded pipeline for plate %s from ObjectState with %d steps",
            plate_path,
            len(pipeline_steps),
        )
        return pipeline_steps

    def set_pipeline_editor(self, pipeline_editor):
        """
        Set the pipeline editor reference.

        Args:
            pipeline_editor: Pipeline editor widget instance
        """
        self.pipeline_editor = pipeline_editor
        logger.debug("Pipeline editor reference set in plate manager")

    # _find_main_window() moved to AbstractManagerWidget

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

    def _validate_delete(self, items: List[Any]) -> bool:
        """Check if delete is allowed - no running plates (required abstract method)."""
        if self.is_any_plate_running():
            self.service_adapter.show_error_dialog(
                "Cannot delete plates while execution is in progress.\n"
                "Please stop execution first."
            )
            return False
        return True

    def _perform_delete(self, items: List[Any]) -> None:
        """Remove plates from backing list and cleanup orchestrators (required abstract method)."""
        paths_to_delete = {plate["path"] for plate in items}

        # Remove from root ObjectState
        root_state = self._ensure_root_state()
        current_paths = root_state.parameters.get("orchestrator_scope_ids") or []
        new_paths = [p for p in current_paths if p not in paths_to_delete]
        root_state.update_parameter("orchestrator_scope_ids", new_paths)

        # Clean up orchestrators and ObjectStates for deleted plates
        for path in paths_to_delete:
            path_str = str(path)

            # Cascade unregister: plate + all steps + all functions (prevents memory leak)
            # This also removes the orchestrator from ObjectState (single source of truth)
            count = ObjectStateRegistry.unregister_scope_and_descendants(path_str)
            logger.debug(
                f"Cascade unregistered {count} ObjectState(s) for deleted plate: {path}"
            )

            # Delete saved PipelineConfig (prevents resurrection)
            if path_str in self.plate_configs:
                del self.plate_configs[path_str]
                logger.debug(f"Deleted plate_configs entry for: {path}")

            # NOTE: Pipeline steps are cascade-deleted via unregister_scope_and_descendants above
            # The Pipeline ObjectState ({path}::pipeline) is automatically cleaned up

        if self.selected_plate_path in paths_to_delete:
            self.selected_plate_path = ""
            # Notify pipeline editor that no plate is selected (mirrors Textual TUI)
            self.plate_selected.emit("")

    def _show_item_editor(self, item: Any) -> None:
        """Show config window for plate (required abstract method)."""
        self.action_edit_config()  # Delegate to existing implementation

    # === List Update Hooks (domain-specific) ===

    def _format_item_content(self, item: Any, index: int, context: Any) -> str:
        """Format plate for list display (dirty marker added by ABC)."""
        return self._format_plate_item_with_preview_text(item)

    def _get_list_item_tooltip(self, item: Any) -> str:
        """Get plate tooltip with orchestrator status."""
        orchestrator = ObjectStateRegistry.get_object(item["path"])
        if orchestrator:
            return f"Status: {orchestrator.state.value}"
        return ""

    def _post_update_list(self) -> None:
        """Auto-select first plate if no selection."""
        if self.plates and not self.selected_plate_path:
            self.item_list.setCurrentRow(0)

    # === Config Resolution Hooks ===

    def _get_scope_for_item(self, item: Any) -> str:
        """PlateManager: scope = plate_path (from orchestrator or dict)."""
        if isinstance(item, PipelineOrchestrator):
            return str(item.plate_path)
        return item.get("path", "") if isinstance(item, dict) else ""

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
