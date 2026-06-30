"""
Pipeline Editor Widget for PyQt6

Pipeline step management with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, List, Dict, Optional, Callable, Tuple, Any
from pathlib import Path

from typing_extensions import override

from PyQt6.QtWidgets import QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.constants import Backend
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings_view import SchemaContextSourceInventoryProvider
from openhcs.core.steps.function_step import FunctionSpec, FunctionStep
from openhcs.core.pipeline import Pipeline
from openhcs.interop.cellprofiler import (
    CellProfilerPipelineImportResult,
    CellProfilerPipelineImportRequest,
    get_cellprofiler_dialect_compiler,
)

from pyqt_reactive.widgets.shared.scope_visual_config import ListItemType
from pyqt_reactive.theming import ColorScheme
from openhcs.pyqt_gui.config import PyQtGUIConfig
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.function_pattern_code_document import (
    FunctionPatternCodeDocumentService,
)
from pyqt_reactive.animation import WindowFlashOverlay

from pyqt_reactive.widgets.shared.button_panel import ButtonPanel
from pyqt_reactive.widgets.shared.manager_ui_scaffold import (
    create_manager_header,
    create_manager_list_widget,
)
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    ProcessingConfig,
)
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, CodeBlock, generate_python_source
from openhcs.utils.pipeline_migration import (
    load_pipeline_with_migration,
)
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
from openhcs.pyqt_gui.widgets.debug_toolbar import DebugToolbarWidget
from openhcs.pyqt_gui.services.plate_scope_identity import (
    PipelineScopeIdentity,
    PlateScopeIdentity,
)
from openhcs.pyqt_gui.services.step_scope_identity import (
    FunctionStepScopeToken,
    SCOPE_SEGMENT_SEPARATOR,
)
from openhcs.core.debug import (
    DebugCommandType,
    DebugSession,
)
from pyqt_reactive.widgets.shared.manager_item_hooks import (
    AttributeItemIdProjection,
    ManagerItemHooks,
)
from pyqt_reactive.widgets.shared.manager_state_binding import ManagerStateBinding
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorCodeWorkflow,
    PipelineEditorDeletionWorkflow,
    PipelineEditorDebugWorkflow,
    PipelineEditorFunctionPresentation,
    PipelineEditorListWorkflow,
    PipelineStepSaveWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionRoute,
    dispatch_widget_action,
)
from openhcs.pyqt_gui.widgets.shared.services.qt_widget_edit_commit import (
    commit_focused_widget_edits,
)
from openhcs.pyqt_gui.widgets.shared.openhcs_manager_mixins import (
    OpenHCSSingleRowActionManagerMixin,
)

# Import ABC base class (Phase 4 migration)
from pyqt_reactive.widgets.shared.abstract_manager_widget import (
    AbstractManagerWidget,
    ListItemFormat,
)
from pyqt_reactive.widgets.shared.manager_action_controller import CodeEditorPayload
from pyqt_reactive.widgets.shared.manager_selection_controller import (
    ItemSelectionPayloadProjection,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


StepFunctionDeclaration = FunctionSpec | dict[str, FunctionSpec] | None
PIPELINE_EDITOR_EXTERNAL_EDITOR_ENV = "OPENHCS_USE_EXTERNAL_EDITOR"


def pipeline_editor_external_editor_enabled() -> bool:
    """Return the explicit environment policy for launching pipeline code edits."""
    if PIPELINE_EDITOR_EXTERNAL_EDITOR_ENV not in os.environ:
        return False
    return os.environ[PIPELINE_EDITOR_EXTERNAL_EDITOR_ENV].lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass(frozen=True, slots=True)
class PipelineCodeSource:
    """Nominal source object for pipeline-step code generation."""

    code_block: CodeBlock
    header: str = "# Edit this pipeline and save to apply changes"
    clean_mode: bool = True

    def render(self) -> str:
        return generate_python_source(
            self.code_block,
            self.header,
            self.clean_mode,
        )


@dataclass(frozen=True, slots=True)
class PipelineDebugCommandRoute:
    """Nominal route for one pipeline-editor debug command."""

    command_type: DebugCommandType
    dispatch: Callable[["PipelineEditorWidget"], None]


class PipelineEditorActionTargetMode(str, Enum):
    """ObjectState target mode for PipelineEditor action summaries."""

    CURRENT_PIPELINE = "current_pipeline"
    SELECTED_STEPS = "selected_steps"


class PipelineEditorAction(str, Enum):
    """Closed set of PipelineEditor button actions and agent-facing semantics."""

    side_effects: tuple[str, ...]
    confirmation_required: bool
    target_mode: PipelineEditorActionTargetMode

    def __new__(
        cls,
        value: str,
        side_effects: tuple[str, ...],
        confirmation_required: bool,
        target_mode: PipelineEditorActionTargetMode,
    ) -> "PipelineEditorAction":
        member = str.__new__(cls, value)
        member._value_ = value
        member.side_effects = side_effects
        member.confirmation_required = confirmation_required
        member.target_mode = target_mode
        return member

    ADD_STEP = (
        "add_step",
        ("opens_step_editor", "may_mutate_pipeline"),
        True,
        PipelineEditorActionTargetMode.CURRENT_PIPELINE,
    )
    DELETE_STEP = (
        "del_step",
        ("mutates_pipeline",),
        True,
        PipelineEditorActionTargetMode.SELECTED_STEPS,
    )
    EDIT_STEP = (
        "edit_step",
        ("opens_step_editor", "may_mutate_step"),
        True,
        PipelineEditorActionTargetMode.SELECTED_STEPS,
    )
    AUTO_LOAD_PIPELINE = (
        "auto_load_pipeline",
        ("loads_basic_pipeline", "mutates_pipeline"),
        True,
        PipelineEditorActionTargetMode.CURRENT_PIPELINE,
    )
    CODE_PIPELINE = (
        "code_pipeline",
        ("opens_code_document_window",),
        False,
        PipelineEditorActionTargetMode.CURRENT_PIPELINE,
    )


@dataclass(frozen=True, slots=True)
class StepFunctionTooltipSection:
    """Tooltip section for a step's function declaration."""

    function_presentation: PipelineEditorFunctionPresentation

    def lines(self, func: StepFunctionDeclaration) -> list[str]:
        if not func:
            return ["Function: None"]
        if isinstance(func, list):
            return [self.list_line(func)]
        if callable(func):
            return [f"Function: {self.function_presentation.func_name(func)}"]
        if isinstance(func, dict):
            return [f"Function: Dictionary with {len(func)} routing keys"]
        return []

    def list_line(self, functions: list[Callable]) -> str:
        if len(functions) == 1:
            return f"Function: {self.function_presentation.func_name(functions[0])}"

        func_names = [
            self.function_presentation.func_name(func)
            for func in functions[:3]
        ]
        if len(functions) > 3:
            func_names.append(f"... +{len(functions) - 3} more")
        return f"Functions: {', '.join(func_names)}"


class StepProcessingTooltipSection:
    """Tooltip section for a step's processing config."""

    def lines(self, processing_config: ProcessingConfig) -> list[str]:
        return [
            self.variable_components_line(processing_config.variable_components),
            self.group_by_line(processing_config.group_by),
            self.input_source_line(processing_config.input_source),
        ]

    def variable_components_line(
        self,
        variable_components: list[VariableComponents],
    ) -> str:
        if not variable_components:
            return "Variable Components: None"
        comp_names = [component.name for component in variable_components]
        return f"Variable Components: [{', '.join(comp_names)}]"

    def group_by_line(self, group_by: GroupBy) -> str:
        if not group_by or group_by.value is None:
            return "Group By: None"
        return f"Group By: {group_by.name}"

    def input_source_line(self, input_source: InputSource) -> str:
        if not input_source:
            return "Input Source: None"
        return f"Input Source: {input_source.name}"


@dataclass(frozen=True, slots=True)
class StepTooltipBuilder:
    """Build the detailed tooltip for one pipeline step."""

    function_section: StepFunctionTooltipSection
    processing_section: StepProcessingTooltipSection = StepProcessingTooltipSection()

    @classmethod
    def for_function_presentation(
        cls,
        function_presentation: PipelineEditorFunctionPresentation,
    ) -> "StepTooltipBuilder":
        return cls(
            function_section=StepFunctionTooltipSection(function_presentation),
        )

    def build(self, step: FunctionStep) -> str:
        tooltip_lines = [f"Step: {step.name}"]
        tooltip_lines.extend(self.function_section.lines(step.func))
        tooltip_lines.extend(self.processing_section.lines(step.processing_config))

        return "\n".join(tooltip_lines)


def dispatch_pipeline_debug_run_command(editor: "PipelineEditorWidget") -> None:
    editor.debug_workflow.run_command(DebugCommandType.RUN)


def dispatch_pipeline_debug_toggle_command(editor: "PipelineEditorWidget") -> None:
    editor.status_message.emit(
        "Debug toolbar active. Use Debug, Step, Pause, Restart, or Inspect."
    )


def dispatch_pipeline_debug_step_command(editor: "PipelineEditorWidget") -> None:
    editor.debug_workflow.run_command(DebugCommandType.STEP)


def dispatch_pipeline_debug_run_to_pause_command(editor: "PipelineEditorWidget") -> None:
    editor.debug_workflow.run_command(DebugCommandType.RUN_TO_PAUSE)


def dispatch_pipeline_debug_restart_command(editor: "PipelineEditorWidget") -> None:
    editor.debug_workflow.run_command(DebugCommandType.RESTART)


def dispatch_pipeline_debug_choose_source_group_command(
    editor: "PipelineEditorWidget",
) -> None:
    editor.debug_workflow.run_command(DebugCommandType.CHOOSE_SOURCE_GROUP)


def dispatch_pipeline_debug_random_source_group_command(
    editor: "PipelineEditorWidget",
) -> None:
    editor.debug_workflow.run_command(DebugCommandType.RANDOM_SOURCE_GROUP)


def dispatch_pipeline_debug_stop_command(editor: "PipelineEditorWidget") -> None:
    editor.debug_workflow.stop_command()


class PipelineEditorWidget(OpenHCSSingleRowActionManagerMixin, AbstractManagerWidget):
    """
    PyQt6 Pipeline Editor Widget.

    Manages pipeline steps with add, edit, delete, load, save functionality.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """

    # Declarative UI configuration
    TITLE = "Pipeline Editor"
    CODE_EDITOR_PAYLOAD = CodeEditorPayload(
        code_type="pipeline",
        missing_error_message="No 'pipeline_steps = [...]' assignment found in edited code",
    )
    BUTTON_CONFIGS = [
        ("Add", PipelineEditorAction.ADD_STEP.value, "Add new pipeline step"),
        ("Del", PipelineEditorAction.DELETE_STEP.value, "Delete selected steps"),
        ("Edit", PipelineEditorAction.EDIT_STEP.value, "Edit selected step"),
        (
            "Auto",
            PipelineEditorAction.AUTO_LOAD_PIPELINE.value,
            "Load basic_pipeline.py",
        ),
        (
            "Code",
            PipelineEditorAction.CODE_PIPELINE.value,
            "Edit pipeline as Python code",
        ),
    ]
    ACTION_ROUTES = MappingProxyType(
        {
            route.action: route
            for route in (
                WidgetActionRoute(
                    PipelineEditorAction.ADD_STEP,
                    lambda widget: widget.action_add,
                ),
                WidgetActionRoute(
                    PipelineEditorAction.DELETE_STEP,
                    lambda widget: widget.action_delete,
                ),
                WidgetActionRoute(
                    PipelineEditorAction.EDIT_STEP,
                    lambda widget: widget.action_edit,
                ),
                WidgetActionRoute(
                    PipelineEditorAction.AUTO_LOAD_PIPELINE,
                    lambda widget: widget.action_auto_load_pipeline,
                ),
                WidgetActionRoute(
                    PipelineEditorAction.CODE_PIPELINE,
                    lambda widget: widget.action_code_pipeline,
                ),
            )
        }
    )
    ITEM_NAME_SINGULAR = "step"
    ITEM_NAME_PLURAL = "steps"
    SELECTION_PAYLOAD_PROJECTION = ItemSelectionPayloadProjection()
    SELECTION_CLEARED_PAYLOAD = None
    SCOPE_ITEM_TYPE = ListItemType.STEP
    STATE_BINDING = ManagerStateBinding(
        items_attr="pipeline_steps",
        selection_attr="selected_step",
        selection_signal_attr="step_selected",
    )

    ITEM_HOOKS = ManagerItemHooks(
        id_projection=AttributeItemIdProjection("name"),
        preserve_selection_pred=lambda self: bool(self.pipeline_steps),
    )
    DEBUG_COMMAND_ROUTES = MappingProxyType(
        {
            route.command_type: route
            for route in (
                PipelineDebugCommandRoute(
                    DebugCommandType.TOGGLE,
                    dispatch_pipeline_debug_toggle_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.STEP,
                    dispatch_pipeline_debug_step_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.RUN,
                    dispatch_pipeline_debug_run_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.RUN_TO_PAUSE,
                    dispatch_pipeline_debug_run_to_pause_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.RESTART,
                    dispatch_pipeline_debug_restart_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.CHOOSE_SOURCE_GROUP,
                    dispatch_pipeline_debug_choose_source_group_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.RANDOM_SOURCE_GROUP,
                    dispatch_pipeline_debug_random_source_group_command,
                ),
                PipelineDebugCommandRoute(
                    DebugCommandType.STOP,
                    dispatch_pipeline_debug_stop_command,
                ),
            )
        }
    )

    # Declarative list item format (replaces imperative format_item_for_display logic)
    # Config indicators (NAP, FIJI, MAT) are auto-discovered via always_viewable_fields
    LIST_ITEM_FORMAT = ListItemFormat(
        first_line=("func",),  # func= shown after step name
        formatters={},
    )

    # Signals
    pipeline_changed = pyqtSignal(list)  # List[FunctionStep]
    step_selected = pyqtSignal(object)  # FunctionStep
    status_message = pyqtSignal(str)  # status message

    def __init__(
        self,
        service_adapter,
        color_scheme: Optional[ColorScheme] = None,
        gui_config: Optional[PyQtGUIConfig] = None,
        parent=None,
    ):
        """
        Initialize the pipeline editor widget.

        Args:
            service_adapter: PyQt service adapter for dialogs and operations
            color_scheme: Color scheme for styling (optional, uses service adapter if None)
            gui_config: GUI configuration (optional, for DualEditorWindow)
            parent: Parent widget
        """
        # Step-specific state (BEFORE super().__init__)
        self.pipeline_steps: List[FunctionStep] = []
        self.current_plate: str = ""
        self.selected_step: str = ""
        # NOTE: plate_pipelines now derived from Pipeline ObjectState (phase 3)
        # Use _get_steps_from_pipeline_state() and update_pipeline_for_plate()

        # Reference to plate manager (set externally)
        # Note: orchestrator is looked up dynamically via _get_current_orchestrator()
        self.plate_manager: "PlateManagerWidget | None" = None

        # Clipboard for copy-paste operations (in-memory only)
        self._clipboard_steps: List[FunctionStep] = []
        self.debug_toolbar: DebugToolbarWidget | None = None
        self.cellprofiler_import_result = None
        self.cellprofiler_import_results_by_plate: dict[str, Any] = {}
        self.source_binding_contexts_by_plate: dict[str, SourceBindingContext] = {}
        self.debug_inspector_window: Any | None = None
        self.debug_session_state: DebugSession | None = None

        # Initialize base class (creates style_generator, event_bus, item_list, buttons, status_label internally)
        # Also auto-processes PREVIEW_FIELD_CONFIGS declaratively
        super().__init__(service_adapter, color_scheme, gui_config, parent)
        self.code_execution_workflow = PipelineEditorCodeWorkflow(self)
        self.deletion_workflow = PipelineEditorDeletionWorkflow(self)
        self.function_presentation = PipelineEditorFunctionPresentation(self)
        self.step_tooltip_builder = StepTooltipBuilder.for_function_presentation(
            self.function_presentation
        )
        self.debug_workflow = PipelineEditorDebugWorkflow(self)
        self.LIST_ITEM_FORMAT = ListItemFormat(
            first_line=self.LIST_ITEM_FORMAT.first_line,
            preview_line=self.LIST_ITEM_FORMAT.preview_line,
            detail_line_field=self.LIST_ITEM_FORMAT.detail_line_field,
            formatters={
                "func": self.function_presentation.format_func_preview,
            },
        )
        self._handle_debug_command = self.debug_workflow.handle_command
        self.show_debug_snapshot = self.debug_workflow.show_snapshot
        self._handle_debug_artifact_export_request = (
            self.debug_workflow.handle_artifact_export_request
        )
        self._handle_debug_artifact_open_request = (
            self.debug_workflow.handle_artifact_open_request
        )

        # Setup UI (after base and subclass state is ready)
        self.setup_ui()
        self.setup_connections()
        self.update_button_states()

        logger.debug("Pipeline editor widget initialized")

    # UI infrastructure provided by AbstractManagerWidget base class
    # Step-specific customizations via hooks below

    def handle_button_action(self, action: str) -> None:
        dispatch_widget_action(
            widget=self,
            action_id=action,
            action_enum=PipelineEditorAction,
            routes=self.ACTION_ROUTES,
            async_runner=self.service_adapter.execute_async_operation,
            before_dispatch=commit_focused_widget_edits,
        )

    def setup_ui(self):
        """Create pipeline editor UI with a debug/test-mode toolbar."""

        header_parts = create_manager_header(
            title=self.TITLE,
            color_scheme=self.color_scheme,
            enable_status_scrolling=self.ENABLE_STATUS_SCROLLING,
        )
        self.status_label = header_parts.status_label
        self._status_scroll = header_parts.status_scroll
        self.debug_toolbar = DebugToolbarWidget(
            self,
            style_generator=self.style_generator,
        )
        self.item_list = create_manager_list_widget(
            color_scheme=self.color_scheme,
            style_generator=self.style_generator,
            delegate_manager=self,
        )
        button_panel = ButtonPanel(
            button_configs=self.BUTTON_CONFIGS,
            on_action=self.handle_button_action,
            style_generator=self.style_generator,
            grid_columns=self.BUTTON_GRID_COLUMNS,
            parent=self,
        )
        self.buttons = button_panel.buttons

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        main_layout.addWidget(header_parts.header)
        main_layout.addWidget(self.debug_toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.item_list)
        splitter.addWidget(button_panel)
        splitter.setSizes([1000, 1])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

    def setup_connections(self):
        """Setup signal/slot connections (base class + step-specific)."""
        # Call base class connection setup (handles item list selection, double-click, reordering, status)
        self.setup_manager_connections()

        # Step-specific signal
        self.pipeline_changed.connect(self.on_pipeline_changed)
        self.debug_toolbar.runtime_inspection_requested.connect(
            self.debug_workflow.show_runtime_inspection
        )
        self._suppress_pipeline_state_sync = False

        # Keyboard shortcuts for copy-paste
        from PyQt6.QtGui import QShortcut, QKeySequence

        QShortcut(QKeySequence("Ctrl+C"), self, self._action_copy_steps)
        QShortcut(QKeySequence("Ctrl+V"), self, self._action_paste_steps)
        if self.debug_toolbar is not None:
            self.debug_toolbar.command_requested.connect(self._handle_debug_command)

    # ========== Pipeline ObjectState Management ==========

    def _ensure_pipeline_state(
        self, plate_path: str, *, register: bool = True
    ) -> ObjectState:
        """Get or create Pipeline ObjectState for a plate.

        Args:
            plate_path: Path of the plate

        Returns:
            Pipeline ObjectState with step_scope_ids parameter
        """
        if not plate_path:
            return None

        pipeline_scope = PipelineScopeIdentity.from_plate_scope(plate_path).scope_id
        state = ObjectStateRegistry.get_by_scope(pipeline_scope)

        if not state:
            identity = PlateScopeIdentity.from_scope_id(plate_path)
            pipeline = Pipeline(name=identity.display_name, step_scope_ids=[])

            # Create ObjectState
            state = ObjectState(
                object_instance=pipeline,
                scope_id=pipeline_scope,
                parent_state=ObjectStateRegistry.get_by_scope(plate_path),
            )
            if register:
                ObjectStateRegistry.register(state, _skip_snapshot=True)

        return state

    def _get_steps_from_pipeline_state(self, plate_path: str) -> List[FunctionStep]:
        """Derive step list from Pipeline ObjectState.

        Args:
            plate_path: Path of the plate

        Returns:
            List of FunctionStep objects derived from step_scope_ids
        """
        pipeline_state = self._ensure_pipeline_state(plate_path)
        if not pipeline_state:
            return []

        if "step_scope_ids" not in pipeline_state.parameters:
            step_scope_ids = []
        else:
            step_scope_ids = pipeline_state.parameters["step_scope_ids"]

        steps = []
        for scope_id in step_scope_ids:
            step_state = ObjectStateRegistry.get_by_scope(scope_id)
            if step_state:
                steps.append(self._step_from_state(step_state))

        return steps

    def _step_from_state(self, step_state: ObjectState) -> FunctionStep:
        """Return a FunctionStep with function child ObjectState values applied."""
        step = step_state.to_object()
        return step.with_function_spec(
            self._function_pattern_from_child_states(
                step_state.scope_id,
                step.func,
                step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY),
            )
        )

    def _function_pattern_from_child_states(
        self,
        parent_scope_id: str,
        func_value,
        tokens,
    ):
        if isinstance(func_value, dict):
            token_map = tokens if isinstance(tokens, dict) else {}
            return {
                channel_key: self._function_pattern_from_child_states(
                    parent_scope_id,
                    channel_funcs,
                    token_map.get(str(channel_key), []),
                )
                for channel_key, channel_funcs in func_value.items()
            }
        if isinstance(func_value, list):
            token_list = tokens if isinstance(tokens, list) else []
            return [
                self._function_entry_from_child_state(
                    parent_scope_id,
                    item,
                    token_list[index] if index < len(token_list) else None,
                )
                for index, item in enumerate(func_value)
            ]
        token = tokens[0] if isinstance(tokens, list) and tokens else None
        return self._function_entry_from_child_state(parent_scope_id, func_value, token)

    def _function_entry_from_child_state(
        self,
        parent_scope_id: str,
        func_item,
        token: str | None,
    ):
        if token is None:
            return func_item
        child_scope_id = f"{parent_scope_id}::{token}"
        if ObjectStateRegistry.get_by_scope(child_scope_id) is None:
            return func_item
        entry = FunctionPatternCodeDocumentService().child_scope_entry(child_scope_id)
        return self._replace_function_entry(func_item, entry.func, entry.kwargs)

    @staticmethod
    def _replace_function_entry(func_item, func_obj, kwargs: dict):
        if (
            isinstance(func_item, tuple)
            and len(func_item) == 3
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return (func_obj, kwargs, func_item[2])
        if (
            isinstance(func_item, tuple)
            and len(func_item) == 2
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return (func_obj, kwargs)
        if callable(func_item):
            return (func_obj, kwargs) if kwargs else func_obj
        return func_item

    def update_pipeline_for_plate(
        self, plate_path: str, steps: List[FunctionStep]
    ) -> None:
        """Update Pipeline ObjectState with new step list.

        Args:
            plate_path: Path of the plate
            steps: New list of FunctionStep objects
        """
        pipeline_state = self._ensure_pipeline_state(plate_path, register=False)
        if not pipeline_state:
            return

        existing_step_scope_ids = tuple(
            pipeline_state.parameters.get("step_scope_ids", [])
        )
        self._transfer_existing_step_scope_tokens(
            plate_path,
            existing_step_scope_ids,
            steps,
        )

        # Seed tokens for all steps (ensures each has a unique _scope_token)
        ScopeTokenService.seed_from_objects(plate_path, steps)

        # Build scope IDs and register each step with ObjectState
        step_scope_ids = []
        to_register: list[ObjectState] = []
        for step in steps:
            scope_id = ScopeTokenService.build_scope_id(plate_path, step)
            step_scope_ids.append(scope_id)
            _step_state, states = self._collect_step_registration_states(
                step=step,
                scope_id=scope_id,
                parent_state=ObjectStateRegistry.get_by_scope(plate_path),
            )
            to_register.extend(states)

        # Register pipeline + steps + update step_scope_ids
        # NOTE: This is called within an atomic block from the caller (delete/paste/add)
        # Do NOT wrap in atomic() here - let the caller manage the atomic context
        if ObjectStateRegistry.get_by_scope(pipeline_state.scope_id) is None:
            ObjectStateRegistry.register(pipeline_state)
        for state in to_register:
            ObjectStateRegistry.register(state)
            logger.debug(f"Registered ObjectState for step: {state.scope_id}")
        removed_step_scope_ids = set(existing_step_scope_ids).difference(
            step_scope_ids
        )
        for removed_scope_id in removed_step_scope_ids:
            ObjectStateRegistry.unregister_scope_and_descendants(
                removed_scope_id,
                _skip_snapshot=True,
            )
        pipeline_state.update_parameter("step_scope_ids", step_scope_ids)

    def _transfer_existing_step_scope_tokens(
        self,
        plate_path: str,
        existing_step_scope_ids: tuple[str, ...],
        steps: List[FunctionStep],
    ) -> None:
        """Reuse existing same-position step scope tokens for replacement updates."""
        for existing_scope_id, replacement_step in zip(existing_step_scope_ids, steps):
            if ScopeTokenService.object_token(replacement_step) is not None:
                continue
            token = FunctionStepScopeToken.from_segment(
                existing_scope_id.rsplit(SCOPE_SEGMENT_SEPARATOR, 1)[-1]
            )
            if token is None:
                continue
            ScopeTokenService.adopt_token(
                plate_path,
                replacement_step,
                token.raw,
            )

    @property
    def plate_pipelines(self) -> Dict[str, List[FunctionStep]]:
        """Backwards-compatible property for accessing plate pipelines.

        Derives pipeline steps from Pipeline ObjectState for all registered plates.
        This allows external code (e.g., plate_manager.py) to access pipelines
        via self.pipeline_editor.plate_pipelines[plate_path].

        Returns:
            Dict mapping plate paths to their step lists
        """
        root_state = ObjectStateRegistry.get_by_scope("__plates__")
        if not root_state:
            return {}

        if "orchestrator_scope_ids" not in root_state.parameters:
            plate_paths = []
        else:
            plate_paths = root_state.parameters["orchestrator_scope_ids"]

        # Build dict of plate_path -> steps
        result = {}
        for plate_path in plate_paths:
            steps = self._get_steps_from_pipeline_state(plate_path)
            result[plate_path] = steps

        return result

    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        """Notify the owning plate manager that this plate's pipeline changed."""
        if not plate_path:
            return
        if self.plate_manager is None:
            return
        self.plate_manager.notify_pipeline_definition_changed(plate_path)

    # ========== Business Logic Methods (Extracted from Textual) ==========

    def _numbered_step_display_name(
        self, step: FunctionStep, step_index: Optional[int]
    ) -> tuple[str, str]:
        """Return UI display name and semantic step name without mutating the step."""
        if step.name:
            step_name = step.name
        else:
            step_name = "Unknown Step"
        if step.debug_pause:
            step_name = f"Pause | {step_name}"

        if step_index is None:
            return step_name, step_name
        return f"{step_index + 1}. {step_name}", step_name

    def format_item_for_display(
        self,
        step: FunctionStep,
        live_context_snapshot=None,
        step_index: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Format step for display in the list with constructor value preview.

        Uses ObjectState for resolved values (no context stack rebuild).
        Returns StyledText with segments for per-field dirty/sig-diff styling.

        Args:
            step: FunctionStep to format
            live_context_snapshot: IGNORED - kept for API compatibility
            step_index: Zero-based rendered row index used for UI numbering

        Returns:
            Tuple of (StyledText with segments, semantic step_name)
        """
        display_name, step_name = self._numbered_step_display_name(step, step_index)

        # Use declarative format from LIST_ITEM_FORMAT
        styled = self.build_item_display_from_format(
            item=step,
            item_name=display_name,
        )
        return styled, step_name

    def action_add_step(self):
        """Handle Add Step button (adapted from Textual version)."""
        try:
            plate_scope = self._require_current_plate_scope()
        except RuntimeError as exc:
            self.service_adapter.show_error_dialog(str(exc))
            return

        # Get orchestrator for step creation
        orchestrator = self._get_current_orchestrator()

        # Create new step
        step_name = f"Step_{len(self.pipeline_steps) + 1}"
        new_step = FunctionStep(
            func=[],  # Start with empty function list
            name=step_name,
        )
        ScopeTokenService.ensure_token(plate_scope, new_step)

        # CRITICAL: Register ObjectState BEFORE opening editor
        # StepParameterEditor expects ObjectState to exist in registry
        self._register_step_state(new_step)

        def handle_save(edited_step):
            """Handle step save from editor."""
            # Use atomic operation to coalesce all ObjectState changes into one undo step
            is_new = edited_step not in self.pipeline_steps
            label = (
                f"add step {edited_step.name}"
                if is_new
                else f"edit step {edited_step.name}"
            )

            with ObjectStateRegistry.atomic(label):
                # Check if step already exists in pipeline (for Shift+Click saves)
                if is_new:
                    self.pipeline_steps.append(edited_step)
                    ScopeTokenService.ensure_token(plate_scope, edited_step)
                    self.status_message.emit(f"Added new step: {edited_step.name}")
                else:
                    # Step already exists, just update the display
                    self.status_message.emit(f"Updated step: {edited_step.name}")

                # Update Pipeline ObjectState with new step list
                self.update_pipeline_for_plate(plate_scope, self.pipeline_steps)
                self.notify_pipeline_definition_changed(plate_scope)

            self.update_item_list()
            self._suppress_pipeline_state_sync = True
            try:
                self.pipeline_changed.emit(self.pipeline_steps)
            finally:
                self._suppress_pipeline_state_sync = False

        # Create and show editor dialog within the correct config context
        orchestrator = self._get_current_orchestrator()

        # SIMPLIFIED: Orchestrator context is automatically available through type-based registry
        # No need for explicit context management - dual-axis resolver handles it automatically
        if not orchestrator:
            logger.info(
                "No orchestrator found for step editor context, This should not happen."
            )

        editor = DualEditorWindow(
            step_data=new_step,
            is_new=True,
            on_save_callback=handle_save,
            orchestrator=orchestrator,
            gui_config=self.gui_config,
            parent=self,
            service_adapter=self.service_adapter,
            plate_scope=plate_scope,
            source_schema=self._current_source_schema(),
            source_binding_context=self.current_source_binding_context(),
            function_invocation_badge_provider=(
                self.function_presentation.badge_provider(new_step)
            ),
        )
        # Set original step for change detection
        editor.set_original_step_for_change_detection()

        # Connect orchestrator config changes to step editor for live placeholder updates
        # This ensures the step editor's placeholders update when pipeline config is saved
        if self.plate_manager is not None:
            self.plate_manager.orchestrator_config_changed.connect(
                editor.on_orchestrator_config_changed
            )
            logger.debug("Connected orchestrator_config_changed signal to step editor")

        editor.show()
        editor.raise_()
        editor.activateWindow()

    # action_delete_step() REMOVED - now uses ABC's action_delete() template with deletion_workflow
    # action_edit_step() REMOVED - now uses ABC's action_edit() template with show_item_editor()

    def action_auto_load_pipeline(self):
        """Handle Auto button - load basic_pipeline.py automatically."""
        if not self.current_plate:
            self.service_adapter.show_error_dialog("No plate selected")
            return

        try:
            # Use module import to find basic_pipeline.py
            import openhcs.tests.basic_pipeline as basic_pipeline_module
            import inspect

            # Get the source code from the module
            python_code = inspect.getsource(basic_pipeline_module)

            # Use ABC template for unified code execution (handles registration sync)
            self._handle_edited_code(python_code)
            self.status_message.emit(
                f"Auto-loaded {len(self.pipeline_steps)} steps from basic_pipeline.py"
            )

        except Exception as e:
            import traceback

            logger.error(f"Failed to auto-load basic_pipeline.py: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.service_adapter.show_error_dialog(
                f"Failed to auto-load pipeline: {str(e)}"
            )

    def action_code_pipeline(self):
        """Handle Code Pipeline button - edit pipeline as Python code."""
        logger.debug("Code button pressed - opening code editor")

        if not self.current_plate:
            self.service_adapter.show_error_dialog("No plate selected")
            return

        try:
            python_code = self.code_document_source(clean=True)

            # Create simple code editor service
            editor_service = SimpleCodeEditorService(self)

            use_external = pipeline_editor_external_editor_enabled()

            # Launch editor with callback - uses ABC _handle_edited_code template
            editor_service.edit_code(
                initial_content=python_code,
                title="Edit Pipeline Steps",
                callback=self._handle_edited_code,  # ABC template method
                use_external=use_external,
                code_type="pipeline",
                code_data={"clean_mode": True},
            )

        except Exception as e:
            logger.error(f"Failed to open pipeline code editor: {e}")
            self.service_adapter.show_error_dialog(
                f"Failed to open code editor: {str(e)}"
            )

    def code_document_title(self) -> str:
        """Return the live Pipeline Editor code-mode title."""
        return "Edit Pipeline Steps"

    def code_document_source(self, clean: bool = True) -> str:
        """Render the current pipeline steps as editable Python source."""
        return PipelineCodeSource(
            CodeBlock.from_items(
                [Assignment("pipeline_steps", self._code_document_steps())]
            ),
            clean_mode=clean,
        ).render()

    def _code_document_steps(self) -> list[FunctionStep]:
        """Return live ObjectState-backed steps for code-mode rendering."""
        if not self.current_plate:
            return list(self.pipeline_steps)
        pipeline_scope = PipelineScopeIdentity.from_plate_scope(
            self.current_plate
        ).scope_id
        if ObjectStateRegistry.get_by_scope(pipeline_scope) is None:
            return list(self.pipeline_steps)
        return self._get_steps_from_pipeline_state(self.current_plate)

    def load_pipeline_from_file(self, file_path: Path):
        """
        Load pipeline from file with automatic migration for backward compatibility.

        Args:
            file_path: Path to pipeline file
        """
        try:
            if file_path.suffix == ".cppipe":
                self._load_cppipe_pipeline_from_file(file_path)
                return

            # Use migration utility to load with backward compatibility
            steps = load_pipeline_with_migration(file_path)

            if steps is not None:
                self.pipeline_steps = steps
                # Don't register here; update_pipeline_for_plate handles atomic registration
                self._normalize_step_scope_tokens(register=False)

                # Update Pipeline ObjectState with loaded steps
                if self.current_plate:
                    self.update_pipeline_for_plate(self.current_plate, self.pipeline_steps)
                    self.notify_pipeline_definition_changed(self.current_plate)
                    logger.debug(
                        f"Updated Pipeline ObjectState ({len(self.pipeline_steps)} steps) for plate: {self.current_plate}"
                    )

                self.update_item_list()
                self._suppress_pipeline_state_sync = True
                try:
                    self.pipeline_changed.emit(self.pipeline_steps)
                finally:
                    self._suppress_pipeline_state_sync = False
                self.status_message.emit(
                    f"Loaded {len(steps)} steps from {file_path.name}"
                )
            else:
                self.status_message.emit(f"Invalid pipeline format in {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.service_adapter.show_error_dialog(f"Failed to load pipeline: {e}")

    def _load_cppipe_pipeline_from_file(self, file_path: Path) -> None:
        """Compile a CellProfiler `.cppipe` into normal OpenHCS pipeline state."""
        generated_path = file_path.with_name(f"{file_path.stem}_openhcs.py")
        filemanager = self.service_adapter.get_file_manager()
        import_result = get_cellprofiler_dialect_compiler().compile_pipeline(
            CellProfilerPipelineImportRequest(
                cppipe_path=file_path,
                generated_pipeline_path=generated_path,
                filemanager=filemanager,
                cppipe_backend=Backend.DISK,
                generated_pipeline_backend=Backend.DISK,
            )
        )
        self.pipeline_steps = list(import_result.pipeline.steps)
        self.cellprofiler_import_result = import_result
        if self.current_plate:
            self.cellprofiler_import_results_by_plate[self.current_plate] = import_result
            self.set_source_binding_context_for_plate(
                self.current_plate,
                SourceBindingContext(
                    logical_plate_id=self.current_plate,
                    display_plate_root=file_path.parent,
                    execution_plate_path=self._current_execution_plate_path(),
                    cppipe_path=file_path,
                    source_schema=import_result.source_schema,
                    inventory_provider=SchemaContextSourceInventoryProvider(
                        self._current_source_root(),
                    ),
                    import_result=import_result,
                ),
            )
        self._normalize_step_scope_tokens(register=False)

        if self.current_plate:
            self.update_pipeline_for_plate(self.current_plate, self.pipeline_steps)
            self.notify_pipeline_definition_changed(self.current_plate)
            logger.debug(
                "Updated Pipeline ObjectState (%d steps) from .cppipe: %s",
                len(self.pipeline_steps),
                file_path,
            )

        self.update_item_list()
        self._suppress_pipeline_state_sync = True
        try:
            self.pipeline_changed.emit(self.pipeline_steps)
        finally:
            self._suppress_pipeline_state_sync = False
        self.status_message.emit(
            f"Imported {len(self.pipeline_steps)} steps from {file_path.name}"
        )

    def save_pipeline_to_file(self, file_path: Path):
        """
        Save pipeline to file (extracted from Textual version).

        Args:
            file_path: Path to save pipeline
        """
        try:
            import dill as pickle

            with open(file_path, "wb") as f:
                pickle.dump(list(self.pipeline_steps), f)
            self.status_message.emit(f"Saved pipeline to {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            self.service_adapter.show_error_dialog(f"Failed to save pipeline: {e}")

    def save_pipeline_for_plate(self, plate_path: str, pipeline: List[FunctionStep]):
        """
        Save pipeline for specific plate (extracted from Textual version).

        Args:
            plate_path: Path of the plate
            pipeline: Pipeline steps to save
        """
        self.update_pipeline_for_plate(plate_path, pipeline)
        logger.debug(f"Updated Pipeline ObjectState for plate: {plate_path}")

    def get_pipeline_for_plate(self, plate_path: str) -> List[FunctionStep]:
        """Return the current pipeline definition from Pipeline ObjectState."""
        if not plate_path:
            return []
        return self._get_steps_from_pipeline_state(plate_path)

    def set_current_plate(self, plate_path: str):
        """
        Set current plate and load its pipeline (extracted from Textual version).

        Args:
            plate_path: Path of the current plate
        """
        logger.info(f"🔔 RECEIVED set_current_plate signal: {plate_path}")

        # DON'T unregister ObjectStates when switching plates - they should stay
        # registered until the step editor is closed. Switching plates just changes
        # the view, it doesn't delete the step editors.

        self.current_plate = plate_path
        self.cellprofiler_import_result = (
            self.cellprofiler_import_results_by_plate.get(plate_path)
        )

        # Load pipeline for the new plate from Pipeline ObjectState
        if plate_path:
            plate_pipeline = self._get_steps_from_pipeline_state(plate_path)
            self.pipeline_steps = plate_pipeline
            logger.info(
                f"  → Loaded {len(plate_pipeline)} steps for plate from Pipeline ObjectState"
            )
        else:
            self.pipeline_steps = []
            logger.info("  → No plate selected, cleared pipeline")

        self._normalize_step_scope_tokens(register=False)

        # CRITICAL: Force cleanup of flash subscriptions when switching plates
        # This ensures FlashElements don't point to stale QListWidgetItems
        # from the previous plate's list widget
        self.clear_list_visual_state()

        self.update_item_list()

        # CRITICAL: Invalidate flash overlay cache after rebuilding list
        # This forces geometry recalculation for the new list items
        WindowFlashOverlay.invalidate_cache_for_widget(self)

        self.update_button_states()
        logger.info(f"  → Pipeline editor updated for plate: {plate_path}")

    def refresh_loaded_pipeline_for_plate(
        self,
        plate_path: str,
        import_result: CellProfilerPipelineImportResult | None,
        pipeline_steps: list[FunctionStep],
    ) -> None:
        """Refresh visible editor state after an external pipeline import."""
        if self.current_plate != plate_path:
            return

        self.cellprofiler_import_result = import_result
        self.pipeline_steps = pipeline_steps
        self.update_item_list()
        self._suppress_pipeline_state_sync = True
        try:
            self.pipeline_changed.emit(pipeline_steps)
        finally:
            self._suppress_pipeline_state_sync = False

    def on_orchestrator_config_changed(self, plate_path: str, effective_config):
        """
        Handle orchestrator configuration changes for placeholder refresh.

        Args:
            plate_path: Path of the plate whose orchestrator config changed
            effective_config: The orchestrator's new effective configuration
        """
        # Only refresh if this is for the current plate
        if plate_path == self.current_plate:
            logger.debug(
                f"Refreshing placeholders for orchestrator config change: {plate_path}"
            )

            # SIMPLIFIED: Orchestrator context is automatically available through type-based registry
            # No need for explicit context management - dual-axis resolver handles it automatically
            orchestrator = self._get_current_orchestrator()
            if orchestrator:
                # Trigger refresh of any open configuration windows or step forms
                # The type-based registry ensures they resolve against the updated orchestrator config
                logger.debug(
                    f"Step forms will now resolve against updated orchestrator config for: {plate_path}"
                )
            else:
                logger.debug(f"No orchestrator found for config refresh: {plate_path}")

    def on_orchestrator_state_changed(self, plate_path: str, state: str) -> None:
        """Refresh editor controls when the current plate state changes."""
        if plate_path != self.current_plate:
            return

        logger.debug("Refreshing editor controls for plate state: %s -> %s", plate_path, state)
        self.update_button_states()

    # Config-attribute preview resolution is owned by the base list-format path.

    def _require_current_plate_scope(self) -> str:
        """Return the current logical plate scope or fail at the editor boundary."""
        if not self.current_plate:
            raise RuntimeError("No plate selected.")
        return self.current_plate

    def _build_step_scope_id(self, step: FunctionStep) -> str:
        """Return the hierarchical scope id for a step: plate::step_N."""
        return ScopeTokenService.build_scope_id(
            self._require_current_plate_scope(),
            step,
        )

    # ========== Time-Travel Hooks (ABC overrides) ==========

    def get_item_insert_index(
        self,
        item: FunctionStep,
        scope_key: str,
    ) -> Optional[int]:
        """Get correct position for step re-insertion during time-travel."""
        # Token format is e.g. "functionstep_3" - parse index from it
        del item
        token = scope_key.rsplit("::", 1)[-1]
        if token:
            parts = token.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return min(int(parts[1]), len(self.pipeline_steps))
        return None

    def _normalize_step_scope_tokens(self, register: bool = True) -> None:
        """Ensure all steps have tokens and are registered."""
        if not self.current_plate:
            return
        plate_scope = self._require_current_plate_scope()
        ScopeTokenService.seed_from_objects(plate_scope, self.pipeline_steps)
        if not register:
            return
        for step in self.pipeline_steps:
            self._register_step_state(step)

    def _normalize_func_items(self, func_value) -> list[tuple[Callable, dict]]:
        if not func_value:
            return []
        from pyqt_reactive.services.pattern_data_manager import PatternDataManager

        if isinstance(func_value, dict):
            items = []
            for channel_funcs in func_value.values():
                items.extend(self._normalize_func_items(channel_funcs))
            return items
        if isinstance(func_value, list):
            items = []
            for item in func_value:
                func_obj, kwargs = PatternDataManager.extract_func_and_kwargs(item)
                if func_obj:
                    items.append((func_obj, kwargs))
            return items
        func_obj, kwargs = PatternDataManager.extract_func_and_kwargs(func_value)
        if not func_obj:
            return []
        return [(func_obj, kwargs)]

    def _scope_tokens_for_function_pattern(self, scope_id: str, func_value):
        if not func_value:
            return []
        from pyqt_reactive.services.pattern_data_manager import PatternDataManager

        if isinstance(func_value, dict):
            return {
                str(channel_key): self._scope_tokens_for_function_pattern(
                    scope_id,
                    channel_funcs,
                )
                for channel_key, channel_funcs in func_value.items()
            }
        if isinstance(func_value, list):
            tokens = []
            for item in func_value:
                func_obj, _kwargs = PatternDataManager.extract_func_and_kwargs(item)
                if func_obj:
                    tokens.append(ScopeTokenService.ensure_token(scope_id, func_obj))
            return tokens
        func_obj, _kwargs = PatternDataManager.extract_func_and_kwargs(func_value)
        if not func_obj:
            return []
        return [ScopeTokenService.ensure_token(scope_id, func_obj)]

    def _collect_step_registration_states(
        self,
        *,
        step: FunctionStep,
        scope_id: str,
        parent_state: ObjectState | None,
    ) -> tuple[ObjectState, list[ObjectState]]:
        """Build missing ObjectStates for one step and its function pattern."""

        existing = ObjectStateRegistry.get_by_scope(scope_id)
        step_state = existing
        to_register: list[ObjectState] = []
        if step_state is None:
            step_state = ObjectState(
                object_instance=step,
                scope_id=scope_id,
                parent_state=parent_state,
            )
            to_register.append(step_state)
        else:
            step_state.update_object_instance(step)

        step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] = (
            self._scope_tokens_for_function_pattern(scope_id, step.func)
        )

        for func_obj, kwargs in self._normalize_func_items(step.func):
            func_scope_id = ScopeTokenService.build_scope_id(scope_id, func_obj)
            if ObjectStateRegistry.get_by_scope(func_scope_id):
                continue
            exclude_params = FunctionPatternCodeDocumentService.reserved_parameter_names(
                func_obj
            )
            func_state = ObjectState(
                object_instance=func_obj,
                scope_id=func_scope_id,
                parent_state=step_state,
                exclude_params=exclude_params,
                initial_values=kwargs,
            )
            to_register.append(func_state)

        return step_state, to_register

    def _register_step_state(self, step: FunctionStep) -> None:
        """Register ObjectState for a step (creates if not exists)."""
        scope_id = self._build_step_scope_id(step)

        parent_state = ObjectStateRegistry.get_by_scope(
            self._require_current_plate_scope()
        )
        _step_state, to_register = self._collect_step_registration_states(
            step=step,
            scope_id=scope_id,
            parent_state=parent_state,
        )

        if not to_register:
            return

        # NOTE: Registration should be atomic with the calling operation (paste/add)
        # Do NOT wrap in atomic() here - let the caller manage the atomic context
        for state in to_register:
            ObjectStateRegistry.register(state)

        logger.debug(f"Registered ObjectState for step (and functions): {scope_id}")

    # Live-value merging is handled by ObjectState-backed form state.
    # _get_step_preview_instance() DELETED - ObjectState provides resolved values directly

    def _handle_full_preview_refresh(self) -> None:
        """Refresh all step preview labels."""
        self.update_item_list()

    # ========== UI Helper Methods ==========

    # update_item_list() REMOVED - uses ABC template with list update hooks

    def update_button_states(self):
        """Update button enabled/disabled states based on mathematical constraints (mirrors Textual TUI)."""
        has_plate = bool(self.current_plate)
        is_initialized = self._is_current_plate_initialized()
        has_steps = len(self.pipeline_steps) > 0
        has_selection = len(self.get_selected_items()) > 0

        # Mathematical constraints (mirrors Textual TUI logic):
        # - Pipeline editing requires initialization
        # - Step operations require steps to exist
        # - Edit requires valid selection
        self.buttons["add_step"].setEnabled(has_plate and is_initialized)
        self.buttons["auto_load_pipeline"].setEnabled(has_plate and is_initialized)
        self.buttons["del_step"].setEnabled(has_steps and has_selection)
        self.buttons["edit_step"].setEnabled(has_steps and has_selection)
        self.buttons["code_pipeline"].setEnabled(
            has_plate and is_initialized
        )  # Same as add button - orchestrator init is sufficient
        if self.debug_toolbar is not None:
            is_compiled = self._is_current_plate_compiled()
            has_debug_session = self.debug_session_state is not None
            self.debug_toolbar.set_controls_enabled(
                has_plate and is_initialized and is_compiled
            )
            self.debug_toolbar.set_debug_session_active(has_debug_session)
            self.debug_toolbar.set_runtime_inspection_enabled(
                has_plate
                and is_initialized
                and is_compiled
                and has_debug_session
            )

    def _get_item_scope_id(self, item: FunctionStep, index: int) -> str:
        """Return the ObjectState scope id represented by a pipeline step list item."""
        del index
        return self._build_step_scope_id(item)

    def selected_step_scope_ids(self) -> tuple[str, ...]:
        """Return ObjectState scope ids for currently selected pipeline steps."""
        selected_items = tuple(self.get_selected_items())
        if not selected_items:
            return ()

        item_index_by_identity = {
            id(item): index
            for index, item in enumerate(self.STATE_BINDING.items(self))
        }
        scope_ids: list[str] = []
        for item in selected_items:
            try:
                item_index = item_index_by_identity[id(item)]
            except KeyError:
                continue
            scope_id = self._get_item_scope_id(item, item_index)
            if scope_id:
                scope_ids.append(scope_id)
        return tuple(scope_ids)

    def _emit_items_changed(self) -> None:
        """Emit the current pipeline step list."""
        self.pipeline_changed.emit(self.pipeline_steps)

    # Event handlers (update_status, on_selection_changed, on_item_double_clicked, on_steps_reordered)
    # DELETED - provided by AbstractManagerWidget base class
    # Step-specific behavior implemented via abstract hooks (see end of file)

    def on_pipeline_changed(self, steps: List[FunctionStep]):
        """
        Handle pipeline changes.

        Args:
            steps: New pipeline steps
        """
        if self._suppress_pipeline_state_sync:
            return
        # Save pipeline to current plate if one is selected
        if self.current_plate:
            self.save_pipeline_for_plate(self.current_plate, steps)
            self.notify_pipeline_definition_changed(self.current_plate)
        if self.debug_session_state is not None:
            self.debug_session_state = self.debug_session_state.mark_dirty_from_cursor()
            if self.debug_session_state.dirty_from_cursor is not None:
                self.status_message.emit(
                    "Debug snapshots downstream of the current cursor are dirty."
                )

        logger.debug(f"Pipeline changed: {len(steps)} steps")

    def _is_current_plate_initialized(self) -> bool:
        """Check if current plate has an initialized orchestrator (mirrors Textual TUI)."""
        if not self.current_plate:
            return False

        orchestrator = self._get_current_orchestrator()
        if orchestrator is None:
            return False

        is_initialized = orchestrator.state.has_completed_initialization
        logger.debug(
            "PipelineEditor: Plate %s orchestrator state: %s, initialized: %s",
            self.current_plate,
            orchestrator.state,
            is_initialized,
        )
        return is_initialized

    def _is_current_plate_compiled(self) -> bool:
        """Check whether the current plate has a compiled execution artifact."""
        if not self.current_plate or self.plate_manager is None:
            return False

        return self.current_plate in self.plate_manager.plate_compiled_data

    def _get_current_orchestrator(self) -> Optional[PipelineOrchestrator]:
        """Get the orchestrator for the currently selected plate."""
        if not self.current_plate:
            return None

        candidate = ObjectStateRegistry.get_object(self.current_plate)
        if candidate is None:
            return None
        if isinstance(candidate, PipelineOrchestrator):
            return candidate
        logger.debug(
            "PipelineEditor: Current plate scope %s resolved to %s, not PipelineOrchestrator",
            self.current_plate,
            type(candidate).__name__,
        )
        return None

    def _current_source_schema(self) -> PipelineImageSchema | None:
        """Return the imported pipeline image schema available to step editors."""

        context = self.current_source_binding_context()
        if context is not None:
            return context.source_schema
        import_result = self.cellprofiler_import_result_for_current_plate()
        if import_result is None:
            return None
        return import_result.source_schema

    def current_source_binding_context(self) -> SourceBindingContext | None:
        """Return the source-binding context for the selected plate, if any."""

        if not self.current_plate:
            return None
        return self.source_binding_contexts_by_plate.get(self.current_plate)

    def set_source_binding_context_for_plate(
        self,
        plate_path: str,
        context: SourceBindingContext,
    ) -> None:
        """Store one coherent source-binding context for a logical plate."""

        self.source_binding_contexts_by_plate[str(plate_path)] = context
        if str(plate_path) == self.current_plate:
            self.cellprofiler_import_result = context.import_result

    def _current_execution_plate_path(self) -> Path:
        """Return the best available execution path for the current plate."""

        orchestrator = self._get_current_orchestrator()
        if orchestrator is not None and orchestrator.plate_path is not None:
            return Path(orchestrator.plate_path)
        if self.current_plate:
            return Path(self.current_plate)
        return Path.cwd()

    def _current_source_root(self) -> Path | None:
        """Return the best available source root for preview inventory."""

        orchestrator = self._get_current_orchestrator()
        if orchestrator is None:
            return None
        if orchestrator.input_dir is not None:
            return Path(orchestrator.input_dir)
        if orchestrator.plate_path is not None:
            return Path(orchestrator.plate_path)
        return None

    def cellprofiler_import_result_for_current_plate(
        self,
    ) -> CellProfilerPipelineImportResult | None:
        """Return the CellProfiler import record for the selected plate."""
        if self.current_plate:
            import_result = self.cellprofiler_import_results_by_plate.get(
                self.current_plate
            )
            if import_result is not None:
                return import_result
        return self.cellprofiler_import_result

    # _find_main_window() moved to AbstractManagerWidget

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.

        Args:
            new_config: New global configuration
        """
        self.global_config = new_config

    # ========== Abstract Hook Implementations (AbstractManagerWidget ABC) ==========

    # === CRUD Hooks ===

    @override
    def action_add(self) -> None:
        """Add steps via dialog (required abstract method)."""
        self.action_add_step()  # Delegate to existing implementation

    @override
    def show_item_editor(self, item: FunctionStep) -> None:
        """Show DualEditorWindow for step (required abstract method)."""
        step_to_edit = item
        plate_scope = self._require_current_plate_scope()

        # Find step's current position in pipeline for border pattern
        step_index = None
        for i, step in enumerate(self.pipeline_steps):
            if step is step_to_edit:
                step_index = i
                break

        def handle_save(edited_step):
            """Handle step save from editor."""
            PipelineStepSaveWorkflow(self, step_to_edit, plate_scope).save(edited_step)

        orchestrator = self._get_current_orchestrator()

        editor = DualEditorWindow(
            step_data=step_to_edit,
            is_new=False,
            on_save_callback=handle_save,
            orchestrator=orchestrator,
            gui_config=self.gui_config,
            parent=self,
            service_adapter=self.service_adapter,
            step_index=step_index,  # Pass actual position for border pattern
            plate_scope=plate_scope,
            source_schema=self._current_source_schema(),
            source_binding_context=self.current_source_binding_context(),
            function_invocation_badge_provider=(
                self.function_presentation.badge_provider(step_to_edit)
            ),
        )
        # Set original step for change detection
        editor.set_original_step_for_change_detection()

        # Connect orchestrator config changes to step editor for live placeholder updates
        if self.plate_manager is not None:
            self.plate_manager.orchestrator_config_changed.connect(
                editor.on_orchestrator_config_changed
            )
            logger.debug("Connected orchestrator_config_changed signal to step editor")

        editor.show()
        editor.raise_()
        editor.activateWindow()

    # === List Update Hooks (domain-specific) ===

    @override
    def _format_item_content(
        self,
        item: FunctionStep,
        index: int,
        context: None,
    ) -> str:
        """Format step for list display (dirty marker added by ABC)."""
        display_text, _ = self.format_item_for_display(
            item,
            context,
            step_index=index,
        )
        return display_text

    @override
    def _get_list_item_tooltip(self, item: FunctionStep) -> str:
        """Get step tooltip."""
        return self.step_tooltip_builder.build(item)

    @override
    def _get_list_item_extra_data(
        self,
        item: FunctionStep,
        index: int,
    ) -> dict[int, bool]:
        """Get enabled flag in UserRole+1."""
        return {1: not item.enabled}

    @override
    def _get_list_placeholder(self) -> tuple[str, None] | None:
        """Return placeholder when no orchestrator."""
        orchestrator = self._get_current_orchestrator()
        if not orchestrator:
            return ("No plate selected - select a plate to view pipeline", None)
        return None

    @override
    def prepare_list_update(self) -> None:
        """Normalize scope tokens before list update.

        ObjectState provides resolved values directly - no need to collect
        LiveContextSnapshot. Just ensure scope tokens are normalized.
        """
        PipelineEditorListWorkflow(self).prepare_update()
        return None  # ObjectState provides values, no context needed

    @override
    def _post_reorder(self) -> None:
        """Additional cleanup after reorder - normalize tokens and emit signal."""
        PipelineEditorListWorkflow(self).post_reorder()

    # === Config Resolution Hook (domain-specific) ===

    @override
    def _get_scope_for_item(self, item: FunctionStep) -> str:
        """PipelineEditor: scope = plate::step_token."""
        if not self.current_plate:
            return ""
        scope = self._build_step_scope_id(item)
        logger.debug(f"⚡ FLASH_DEBUG _get_scope_for_item: item={item}, scope={scope}")
        return scope

    # === CrossWindowPreviewMixin Hook ===
    # _get_current_orchestrator() is implemented above (line ~795) - does actual lookup from plate manager
    # _configure_preview_fields() REMOVED - now uses declarative PREVIEW_FIELD_CONFIGS (line ~99)

    # ========== End Abstract Hook Implementations ==========

    def closeEvent(self, event):
        """Handle widget close event to disconnect signals and prevent memory leaks."""
        # Unregister from cross-window refresh signals
        ObjectStateRegistry.disconnect_listener(self._on_live_context_changed)
        logger.debug("Pipeline editor: Unregistered from cross-window refresh signals")

        # Call parent closeEvent
        super().closeEvent(event)

    def on_time_travel_complete(self, dirty_states, triggering_scope):
        """Refresh pipeline list after time travel to reflect restored step order."""
        PipelineEditorListWorkflow(self).restore_after_time_travel()

    def _action_copy_steps(self):
        """Copy selected steps to clipboard (Ctrl+C)."""
        selected_steps = self.get_selected_items()
        if not selected_steps:
            self.status_message.emit("No steps selected to copy")
            return

        self._clipboard_steps = [copy.deepcopy(step) for step in selected_steps]
        step_names = [step.name for step in selected_steps]
        self.status_message.emit(
            f"Copied {len(selected_steps)} step(s): {', '.join(step_names)}"
        )

    def _action_paste_steps(self):
        """Paste steps from clipboard after selected step (Ctrl+V)."""
        if not self._clipboard_steps:
            self.status_message.emit("Clipboard is empty")
            return

        if not self.current_plate:
            self.status_message.emit("No plate selected")
            return

        # Calculate insert position: after last selected index, or at end if nothing selected
        selected_indices = self.item_list.selectedIndexes()
        if selected_indices:
            insert_after_index = max(idx.row() for idx in selected_indices)
        else:
            insert_after_index = len(self.pipeline_steps) - 1

        step_names = [step.name for step in self._clipboard_steps]
        label = f"paste {len(self._clipboard_steps)} step(s): {', '.join(step_names)}"

        with ObjectStateRegistry.atomic(label):
            # Insert steps after the selected position
            insert_position = insert_after_index + 1
            for i, step in enumerate(self._clipboard_steps):
                # Ensure fresh scope token for the copied step
                ScopeTokenService.ensure_token(self.current_plate, step)
                # Register with ObjectState (handles flashing automatically)
                self._register_step_state(step)
                # Insert into pipeline
                self.pipeline_steps.insert(insert_position + i, step)

            # Update Pipeline ObjectState
            self.update_pipeline_for_plate(self.current_plate, self.pipeline_steps)

        self.update_item_list()
        self.pipeline_changed.emit(self.pipeline_steps)
        self.status_message.emit(
            f"Pasted {len(self._clipboard_steps)} step(s) after position {insert_after_index + 1}"
        )
