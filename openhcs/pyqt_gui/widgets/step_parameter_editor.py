"""
Step Parameter Editor Widget for PyQt6 GUI.

Mirrors the Textual TUI StepParameterEditorWidget with type hint-based form generation.
Handles FunctionStep parameter editing with nested dataclass support.
"""

import logging
import dataclasses
from functools import partial
from typing import Optional, Union, get_args, get_origin
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from openhcs.introspection import SignatureAnalyzer
from openhcs.core.config import PipelineConfig
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.path_cache import PathCacheKey
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings_view import SchemaContextSourceInventoryProvider
from openhcs.pyqt_gui.widgets.source_bindings_editor import SourceBindingsEditorWidget
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager, FormManagerConfig
from pyqt_reactive.widgets.shared.config_hierarchy_tree import (
    ConfigHierarchyTreeHelper,
)
from pyqt_reactive.widgets.shared.detachable_action_bar import (
    DetachableActionBar,
    DetachableActionBarHost,
)
from pyqt_reactive.widgets.shared.scrollable_form_body import create_scrollable_form_body
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin
from pyqt_reactive.services.parameter_ops_service import ParameterOpsService
from pyqt_reactive.services.window_code_document import WindowCodeDocumentDriver
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from openhcs.pyqt_gui.config import PyQtGUIConfig, get_default_pyqt_gui_config
from openhcs.pyqt_gui.services.pycodified_window_code_document import (
    ExternalCodeEditorPreference,
    PycodifiedObjectCodeDocumentDriver,
    PycodifiedObjectDocumentSpec,
)

# REMOVED: LazyDataclassFactory import - no longer needed since step editor
# uses existing lazy dataclass instances from the step
from pyqt_reactive.forms.parameter_type_utils import ParameterTypeUtils
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StepSettingsDialogRequest:
    """Cached file-dialog request for loading or saving step settings."""
    title: str
    mode: str
    cache_key: PathCacheKey = PathCacheKey.STEP_SETTINGS
    file_filter: str = "Step Files (*.step);;All Files (*)"


@dataclasses.dataclass(frozen=True, slots=True)
class StepEditorGuiConfigRequest:
    """Explicit GUI-config resolution request for the step editor."""

    gui_config: PyQtGUIConfig | None

    def resolve(self) -> PyQtGUIConfig:
        if self.gui_config is not None:
            return self.gui_config
        return get_default_pyqt_gui_config()


class StepSettingsFileController:
    """Own load/save behavior for serialized step settings."""

    def __init__(self, editor: "StepParameterEditorWidget"):
        self.editor = editor

    def load_step_settings(self) -> None:
        file_path = self._show_dialog(
            StepSettingsDialogRequest(
                title="Load Step Settings (.step)",
                mode="open",
            )
        )
        if file_path:
            self._load_from_file(file_path)

    def save_step_settings(self) -> None:
        file_path = self._show_dialog(
            StepSettingsDialogRequest(
                title="Save Step Settings (.step)",
                mode="save",
            )
        )
        if file_path:
            self._save_to_file(file_path)

    def _show_dialog(self, request: StepSettingsDialogRequest) -> Optional[Path]:
        if not self.editor.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return None

        return self.editor.service_adapter.show_cached_file_dialog(
            cache_key=request.cache_key,
            title=request.title,
            file_filter=request.file_filter,
            mode=request.mode,
        )

    def _load_from_file(self, file_path: Path) -> None:
        try:
            import dill as pickle

            with open(file_path, "rb") as handle:
                step_data = pickle.load(handle)

            for param_name, value in step_data.items():
                self.editor.form_manager.update_parameter(param_name, value)
            self.editor.step = self.editor.state.to_object()

            self.editor.form_manager._refresh_all_placeholders()
            logger.debug("Loaded %d parameters from %s", len(step_data), file_path.name)

        except Exception as exc:
            logger.error("Failed to load step settings from %s: %s", file_path, exc)
            if self.editor.service_adapter:
                self.editor.service_adapter.show_error_dialog(
                    f"Failed to load step settings: {exc}"
                )

    def _save_to_file(self, file_path: Path) -> None:
        try:
            import dill as pickle

            step_data = self.editor.state.get_current_values()
            with open(file_path, "wb") as handle:
                pickle.dump(step_data, handle)
            logger.debug("Saved %d parameters to %s", len(step_data), file_path.name)

        except Exception as exc:
            logger.error("Failed to save step settings to %s: %s", file_path, exc)
            if self.editor.service_adapter:
                self.editor.service_adapter.show_error_dialog(
                    f"Failed to save step settings: {exc}"
                )


class StepParameterEditorWidget(ScrollableFormMixin, DetachableActionBarHost, QWidget):
    """
    Step parameter editor using dynamic form generation.

    Mirrors Textual TUI implementation - builds forms based on FunctionStep
    constructor signature with nested dataclass support.

    Inherits from ScrollableFormMixin to provide scroll-to-section functionality.
    """

    # Signals
    step_parameter_changed = pyqtSignal()

    def showEvent(self, event):
        """Override showEvent to apply initial enabled styling when widget becomes visible."""
        super().showEvent(event)

    def __init__(
        self,
        step: FunctionStep,
        service_adapter=None,
        color_scheme: Optional[ColorScheme] = None,
        gui_config: Optional[PyQtGUIConfig] = None,
        parent=None,
        pipeline_config=None,
        scope_id: Optional[str] = None,
        step_index: Optional[int] = None,
        scope_accent_color=None,
        render_header: bool = True,
        button_style: Optional[str] = None,
        source_schema: PipelineImageSchema | None = None,
        source_binding_context: SourceBindingContext | None = None,
        source_root: str | Path | None = None,
    ):
        super().__init__(parent)

        # Initialize color scheme and GUI config
        self.color_scheme = color_scheme or ColorScheme()
        self.gui_config = StepEditorGuiConfigRequest(gui_config).resolve()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        self._render_header = render_header
        self._button_style = button_style  # Store centralized button style

        self.header_label: Optional[QLabel] = None

        self.step = step
        self.service_adapter = service_adapter
        self.pipeline_config = (
            pipeline_config  # Store pipeline config for context hierarchy
        )
        self.scope_id = scope_id  # Store scope_id for cross-window update scoping
        self.step_index = step_index  # Step position index for tree registry
        self.source_schema = source_schema
        self.source_binding_context = source_binding_context
        self.source_root = source_root

        self.header_label: Optional[QLabel] = None

        # Create action buttons container (always, for external access)
        self._action_buttons_container = DetachableActionBar(
            object_name="step_action_buttons_container"
        )
        self.step_settings_files = StepSettingsFileController(self)

        code_btn = QPushButton("Code")
        code_btn.setMaximumWidth(60)
        code_btn.setFixedHeight(CURRENT_LAYOUT.button_height)
        code_btn.setStyleSheet(self._get_button_style())
        code_btn.clicked.connect(self.view_step_code)
        self._action_buttons_container.add_button(code_btn)

        # Live placeholder updates not yet ready - disable for now
        self._step_editor_coordinator = None
        # TODO: Re-enable when live updates feature is fully implemented
        # if self.gui_config and self.gui_config.enable_live_step_parameter_updates:
        #     from openhcs.config_framework.lazy_factory import ContextEventCoordinator
        #     self._step_editor_coordinator = ContextEventCoordinator()
        #     logger.debug("🔍 STEP EDITOR: Created step-editor-specific coordinator for live step parameter updates")

        # ObjectState MUST be registered by PipelineEditorWidget when step was added.
        logger.debug(
            "🔍 STEP_EDITOR: Looking up ObjectState for scope_id=%s",
            self.scope_id,
        )
        registered_states = ObjectStateRegistry.get_all()
        logger.debug(
            "🔍 STEP_EDITOR: Registry has %d scopes",
            len(registered_states),
        )
        self.state = (
            ObjectStateRegistry.get_by_scope(self.scope_id) if self.scope_id else None
        )

        if self.state is None:
            raise RuntimeError(
                f"ObjectState not found for scope_id={self.scope_id}. "
                f"PipelineEditor must register the step before opening the editor. "
                f"Registry has: {[s.scope_id for s in ObjectStateRegistry.get_all()]}"
            )

        logger.debug(
            "🔍 STEP_EDITOR: Using REGISTERED ObjectState, params=%s",
            list(self.state.parameters.keys()),
        )
        state_values = self.state.get_current_values()

        # Analyze AbstractStep signature to get all inherited parameters (mirrors Textual TUI)
        # Auto-detection correctly identifies constructors and includes all parameters
        param_info = SignatureAnalyzer.analyze(AbstractStep.__init__)

        # Get current parameter values from ObjectState, the editor model.
        parameter_types = {}
        self._step_level_configs = {}

        for name, info in param_info.items():
            # All AbstractStep parameters are relevant for editing
            # ParameterFormManager will automatically route lazy dataclass parameters to LazyDataclassEditor
            if name not in state_values:
                raise RuntimeError(
                    f"ObjectState for scope_id={self.scope_id} is missing "
                    f"step parameter {name!r}"
                )
            current_value = state_values[name]

            # CRITICAL FIX: For lazy dataclass parameters, leave current_value as None
            # This allows the UI to show placeholders and use lazy resolution properly
            if current_value is None and self._is_optional_lazy_dataclass_in_pipeline(
                info.param_type, name
            ):
                # Don't create concrete instances - leave as None for placeholder resolution
                # The UI will handle lazy resolution and show appropriate placeholders
                # Mark this as a step-level config for special handling
                self._step_level_configs[name] = True

            parameter_types[name] = info.param_type

        # Track dataclass-backed parameters for the hierarchy tree
        self._tree_dataclass_params = self._collect_dataclass_parameters(
            parameter_types
        )
        self.tree_helper = ConfigHierarchyTreeHelper()

        # SIMPLIFIED: Create parameter form manager using dual-axis resolution

        # CRITICAL FIX: Use pipeline_config as context_obj (parent for inheritance)
        # The step is the overlay (what's being edited), not the parent context
        # Context hierarchy: GlobalPipelineConfig (thread-local) -> PipelineConfig (context_obj) -> Step (overlay)
        config = FormManagerConfig(
            parent=self,  # Pass self as parent widget
            color_scheme=self.color_scheme,  # Pass color scheme for consistent theming
            use_scroll_area=False,  # Step editor manages its own scroll area
            scope_accent_color=scope_accent_color,  # Pass scope accent color from parent window
            scope_step_index=self.step_index,  # Align scope styling with pipeline order
        )

        self.form_manager = ParameterFormManager(
            state=self.state,  # ObjectState (MODEL) from registry
            config=config,  # Pass configuration object
        )
        self._code_document_driver = PycodifiedObjectCodeDocumentDriver(
            spec=PycodifiedObjectDocumentSpec(
                assignment_name="step",
                title=f"Edit Step: {self.step.name}",
                header="# Function Step",
                expected_type=FunctionStep,
            ),
            current_object=self._current_step_for_code_document,
            apply_object=self._apply_step_from_code_document,
            before_read=self._refresh_code_document_context,
        )
        self.hierarchy_tree = None
        self.content_splitter = None

        self.setup_ui()
        self.apply_source_bindings_preview_context()
        self.setup_connections()

        # Ensure placeholders pick up live context (e.g., PipelineConfig edits) after registration.
        QTimer.singleShot(
            0,
            lambda: ParameterOpsService().refresh_with_live_context(self.form_manager),
        )

        logger.debug(
            "Step parameter editor initialized for step: %s",
            self.step.name,
        )

    def apply_scope_color_scheme(self, scheme) -> None:
        from pyqt_reactive.widgets.shared.scope_style_applier import (
            apply_scope_color_scheme_to_widget_tree,
        )

        apply_scope_color_scheme_to_widget_tree(self.form_manager, scheme)

    def code_document_driver(self) -> WindowCodeDocumentDriver:
        """Return this step editor's pycodified code-mode document driver."""
        return self._code_document_driver

    def apply_source_bindings_preview_context(self) -> None:
        """Pass imported pipeline source-schema context to source-binding editors."""

        schema = (
            self.source_binding_context.source_schema
            if self.source_binding_context is not None
            else self.source_schema
        )
        if schema is None:
            return
        for widget in self.findChildren(SourceBindingsEditorWidget):
            if self.source_binding_context is not None:
                inventory = self.source_binding_context.inventory(widget.get_value())
            else:
                inventory = SchemaContextSourceInventoryProvider(
                    self.source_root,
                ).inventory(
                    schema=schema,
                    bindings=widget.get_value(),
                )
            widget.set_preview_context(
                schema=schema,
                inventory=inventory,
            )

    def _is_optional_lazy_dataclass_in_pipeline(self, param_type, param_name):
        """
        Check if parameter is an optional lazy dataclass that exists in PipelineConfig.

        This enables automatic step-level config creation for any parameter that:
        1. Is Optional[SomeDataclass]
        2. SomeDataclass exists as a field type in PipelineConfig (type-based matching)
        3. The dataclass has lazy resolution capabilities

        No manual mappings needed - uses type-based discovery.
        """

        # Check if parameter is Optional[dataclass]
        if not ParameterTypeUtils.is_optional_dataclass(param_type):
            return False

        # Get the inner dataclass type
        inner_type = ParameterTypeUtils.get_optional_inner_type(param_type)

        # Find if this type exists as a field in PipelineConfig (type-based matching)
        pipeline_field_name = self._find_pipeline_field_by_type(inner_type)
        if not pipeline_field_name:
            return False

        # Check if the dataclass has lazy resolution capabilities
        try:
            test_instance = inner_type()
        except Exception:
            return False

        instance_members = set(dir(test_instance))
        return {
            "_resolve_field_value",
            "_lazy_resolution_config",
        }.issubset(instance_members)

    def _find_pipeline_field_by_type(self, target_type):
        """
        Find the field in PipelineConfig that matches the target type.

        This is type-based discovery - no manual mappings needed.
        """
        for field in dataclasses.fields(PipelineConfig):
            # Use string comparison to handle type identity issues
            if str(field.type) == str(target_type):
                return field.name
        return None

    # REMOVED: _create_step_level_config method - dead code
    # The step editor should use the existing lazy dataclass instances from the step,
    # not create new "StepLevel" versions. The AbstractStep already has the correct
    # lazy dataclass types (LazyNapariStreamingConfig, LazyStepMaterializationConfig, etc.)

    def _collect_dataclass_parameters(self, parameter_types):
        """Return dataclass-based parameters for building the hierarchy tree."""
        dataclass_params = {}
        for field_name, param_type in parameter_types.items():
            obj_type = self._extract_dataclass_from_param_type(param_type)
            if obj_type is not None:
                dataclass_params[field_name] = obj_type

        return dataclass_params

    def _extract_dataclass_from_param_type(self, param_type):
        """Resolve the concrete dataclass type from the annotated parameter."""
        resolved_type = param_type

        try:
            origin = get_origin(param_type)
        except Exception:
            origin = None

        if origin is Union:
            args = [arg for arg in get_args(param_type) if arg is not type(None)]
            if len(args) == 1:
                resolved_type = args[0]

        if resolved_type is None or isinstance(resolved_type, str):
            return None

        if dataclasses.is_dataclass(resolved_type):
            return resolved_type

        return None

    def _create_configuration_tree(self) -> Optional[QTreeWidget]:
        """Create and populate the configuration hierarchy tree."""
        if not self._tree_dataclass_params:
            return None

        return self.tree_helper.create_tree_from_mapping(
            dataclass_params=self._tree_dataclass_params,
            form_manager=self.form_manager,
            state=self.state,
            strip_config_suffix=True,
            on_item_double_clicked=self._on_tree_item_double_clicked,
        )

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Scroll to the associated form section when a tree item is activated."""
        self.tree_helper.activate_item(
            item,
            scroll_to_section=self._scroll_to_section,
            field_for_class=partial(
                self.tree_helper.field_for_class_in_mapping,
                self._tree_dataclass_params,
            ),
        )

    # _scroll_to_section is provided by ScrollableFormMixin

    def setup_ui(self):
        """Setup the user interface (matches FunctionListEditorWidget structure)."""
        # Main layout directly on self (like FunctionListEditorWidget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header with controls (only if render_header=True)
        if self._render_header:
            header_layout = QHBoxLayout()

            # Header label (stored for scope accent styling)
            self.header_label = QLabel("Step Parameters")
            self.header_label.setStyleSheet(
                f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; font-weight: bold; font-size: 14px;"
            )
            header_layout.addWidget(self.header_label)

            header_layout.addStretch()

            # Add action buttons to header
            header_layout.addWidget(self._action_buttons_container)

            layout.addLayout(header_layout)
        else:
            # Header not rendered - buttons are still available for external use
            # No header layout added, so buttons remain in _action_buttons_container
            pass

        hierarchy_tree = self._create_configuration_tree()
        body_parts = create_scrollable_form_body(
            form_widget=self.form_manager,
            tree_widget=hierarchy_tree,
            tree_initial_size=280,
            form_initial_size=720,
            parent=self,
        )
        self.scroll_area = body_parts.scroll_area
        self.hierarchy_tree = hierarchy_tree
        self.content_splitter = body_parts.splitter
        self.splitter_helper = body_parts.splitter_helper
        layout.addWidget(body_parts.body_widget, 1)

        # Apply tree widget styling (matches config window)
        self.setStyleSheet(self.style_generator.generate_tree_widget_style())

    def _get_button_style(self) -> str:
        """Get consistent button styling."""
        if self._button_style:
            return self.style_generator.generate_config_button_styles().get(
                self._button_style, ""
            )

        return """
            QPushButton {
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }
            QPushButton:pressed {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_pressed_bg)};
            }
        """

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect form manager parameter changes
        self.form_manager.parameter_changed.connect(self._handle_parameter_change)

    def _handle_parameter_change(self, param_name: str, value):
        """Handle parameter change from form manager (mirrors Textual TUI).

        Args:
            param_name: Full path like "FunctionStep.processing_config.group_by" or "FunctionStep.name"
            value: New value
        """
        del value
        try:
            self.step = self.state.to_object()
            logger.debug("Synchronized step from ObjectState after %s", param_name)
            self.step_parameter_changed.emit()

        except Exception as e:
            logger.error(f"Error updating step parameter {param_name}: {e}")

    def load_step_settings(self):
        """Load step settings from .step file (mirrors Textual TUI)."""
        self.step_settings_files.load_step_settings()

    def save_step_settings(self):
        """Save step settings to .step file (mirrors Textual TUI)."""
        self.step_settings_files.save_step_settings()

    def update_step(self, step: FunctionStep):
        """Update the step and refresh the form."""
        self.step = step

        CodeEditorFormUpdater.update_form_from_instance(
            self.form_manager,
            step,
            broadcast_callback=None,
        )
        self.step = self.state.to_object()

        logger.debug(
            "Updated step parameter editor for step: %s",
            self.step.name,
        )

    def view_step_code(self):
        """View the complete FunctionStep as Python code."""
        try:
            document = self._code_document_driver.read_document()
            editor_service = SimpleCodeEditorService(self)

            editor_service.edit_code(
                initial_content=document.source,
                title=document.title,
                callback=self._handle_edited_step_code,
                use_external=ExternalCodeEditorPreference.use_external_editor(),
                code_type="step",
                code_data={"clean_mode": True},
            )

        except Exception as e:
            logger.error(f"Failed to open step code editor: {e}")
            if self.service_adapter:
                self.service_adapter.show_error_dialog(
                    f"Failed to open code editor: {str(e)}"
                )

    def _handle_edited_step_code(self, edited_code: str) -> None:
        """Handle the edited step code from code editor."""
        try:
            self._code_document_driver.apply_source(edited_code)
            logger.info("Updated step from code editor: %s", self.step.name)

        except Exception as e:
            logger.error(f"Failed to apply edited step code: {e}")
            raise

    def _refresh_code_document_context(self) -> None:
        """Refresh live context before rendering this step as source."""
        ParameterOpsService().refresh_with_live_context(self.form_manager)

    def _current_step_for_code_document(self) -> FunctionStep:
        """Return the current step with the live function-pattern tab applied."""
        current_step = self.state.to_object()
        parent_window = self.window()
        func = parent_window.func_editor.current_pattern
        current_step.func = func
        logger.debug("Using live func from function list editor: %r", func)
        return current_step

    def _apply_step_from_code_document(self, new_step: FunctionStep) -> None:
        """Apply a parsed code-mode step through the normal form path."""
        self.step = new_step

        CodeEditorFormUpdater.update_form_from_instance(
            self.form_manager,
            new_step,
            broadcast_callback=None,
        )

        parent_window = self.window()
        func_editor = parent_window.func_editor
        func_editor._initialize_pattern_data(new_step.func)
        func_editor._populate_function_list()
        logger.debug("Updated function list editor with new func: %r", new_step.func)

        self.step_parameter_changed.emit()
