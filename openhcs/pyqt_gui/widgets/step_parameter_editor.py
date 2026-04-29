"""
Step Parameter Editor Widget for PyQt6 GUI.

Mirrors the Textual TUI StepParameterEditorWidget with type hint-based form generation.
Handles FunctionStep parameter editing with nested dataclass support.
"""

import logging
import dataclasses
import os
from typing import Any, Optional, Union, get_args, get_origin
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from openhcs.introspection import SignatureAnalyzer
from openhcs.core.config import PipelineConfig
from openhcs.core.path_cache import PathCacheKey
from pyqt_reactive.forms import ParameterFormManager, FormManagerConfig
from pyqt_reactive.widgets.shared.config_hierarchy_tree import ConfigHierarchyTreeHelper
from pyqt_reactive.core.collapsible_splitter_helper import CollapsibleSplitterHelper
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin
from pyqt_reactive.services.parameter_ops_service import ParameterOpsService
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from openhcs.pyqt_gui.config import PyQtGUIConfig, get_default_pyqt_gui_config

# REMOVED: LazyDataclassFactory import - no longer needed since step editor
# uses existing lazy dataclass instances from the step
from pyqt_reactive.forms import ParameterTypeUtils
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, generate_python_source
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry

logger = logging.getLogger(__name__)


class StepParameterEditorWidget(ScrollableFormMixin, QWidget):
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
    ):
        super().__init__(parent)

        # Initialize color scheme and GUI config
        self.color_scheme = color_scheme or ColorScheme()
        self.gui_config = gui_config or get_default_pyqt_gui_config()
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

        self.header_label: Optional[QLabel] = None

        # Create action buttons container (always, for external access)
        self._action_buttons_container = QWidget()
        self._action_buttons_container.setObjectName("step_action_buttons_container")
        self._action_buttons_layout = QHBoxLayout(self._action_buttons_container)
        self._action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._action_buttons_layout.setSpacing(2)
        self._action_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        code_btn = QPushButton("Code")
        code_btn.setMaximumWidth(60)
        code_btn.setFixedHeight(CURRENT_LAYOUT.button_height)
        code_btn.setStyleSheet(self._get_button_style())
        code_btn.clicked.connect(self.view_step_code)
        self._action_buttons_layout.addWidget(code_btn)

        # Live placeholder updates not yet ready - disable for now
        self._step_editor_coordinator = None
        # TODO: Re-enable when live updates feature is fully implemented
        # if self.gui_config and self.gui_config.enable_live_step_parameter_updates:
        #     from openhcs.config_framework.lazy_factory import ContextEventCoordinator
        #     self._step_editor_coordinator = ContextEventCoordinator()
        #     logger.debug("🔍 STEP EDITOR: Created step-editor-specific coordinator for live step parameter updates")

        # Analyze AbstractStep signature to get all inherited parameters (mirrors Textual TUI)
        # Auto-detection correctly identifies constructors and includes all parameters
        param_info = SignatureAnalyzer.analyze(AbstractStep.__init__)

        # Get current parameter values from step instance
        parameters = {}
        parameter_types = {}
        param_defaults = {}

        for name, info in param_info.items():
            # All AbstractStep parameters are relevant for editing
            # ParameterFormManager will automatically route lazy dataclass parameters to LazyDataclassEditor
            current_value = getattr(self.step, name, info.default_value)

            # CRITICAL FIX: For lazy dataclass parameters, leave current_value as None
            # This allows the UI to show placeholders and use lazy resolution properly
            if current_value is None and self._is_optional_lazy_dataclass_in_pipeline(
                info.param_type, name
            ):
                # Don't create concrete instances - leave as None for placeholder resolution
                # The UI will handle lazy resolution and show appropriate placeholders
                param_defaults[name] = None
                # Mark this as a step-level config for special handling
                if getattr(self, "_step_level_configs", None) is None:
                    self._step_level_configs = {}
                self._step_level_configs[name] = True
            else:
                param_defaults[name] = info.default_value

            parameters[name] = current_value
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
        # Look up ObjectState from registry using scope_id
        # ObjectState MUST be registered by PipelineEditorWidget when step was added
        logger.info(
            f"🔍 STEP_EDITOR: Looking up ObjectState for scope_id={self.scope_id}"
        )
        logger.info(
            f"🔍 STEP_EDITOR: Registry has scopes: {[s.scope_id for s in ObjectStateRegistry.get_all()]}"
        )
        self.state = (
            ObjectStateRegistry.get_by_scope(self.scope_id) if self.scope_id else None
        )

        if self.state is None:
            # FAIL LOUD: The step MUST be registered by PipelineEditor before opening the editor
            raise RuntimeError(
                f"ObjectState not found for scope_id={self.scope_id}. "
                f"PipelineEditor must register the step before opening the editor. "
                f"Registry has: {[s.scope_id for s in ObjectStateRegistry.get_all()]}"
            )

        logger.info(
            f"🔍 STEP_EDITOR: Using REGISTERED ObjectState, params={list(self.state.parameters.keys())}"
        )

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
        self.hierarchy_tree = None
        self.content_splitter = None

        self.setup_ui()
        self.setup_connections()

        # Ensure placeholders pick up live context (e.g., PipelineConfig edits) after registration.
        QTimer.singleShot(
            0,
            lambda: ParameterOpsService().refresh_with_live_context(self.form_manager),
        )

        logger.debug(
            f"Step parameter editor initialized for step: {getattr(step, 'name', 'Unknown')}"
        )

    def apply_scope_color_scheme(self, scheme) -> None:
        from pyqt_reactive.widgets.shared.scope_style_applier import (
            apply_scope_color_scheme_to_widget_tree,
        )

        apply_scope_color_scheme_to_widget_tree(self.form_manager, scheme)

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
            # Try to create an instance to see if it's a lazy dataclass
            test_instance = inner_type()
            # Check for lazy dataclass methods - direct access will raise AttributeError if missing
            test_instance._resolve_field_value
            test_instance._lazy_resolution_config
            return True
        except AttributeError:
            return False
        except:
            return False

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
            if field_name == "func":
                continue

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
        if not getattr(self, "_tree_dataclass_params", None):
            return None

        # Pass form_manager as flash_manager - tree reads from SAME _flash_colors dict as groupboxes
        # ONE source of truth: form_manager already subscribes to ObjectState.on_resolved_changed
        # Pass state for automatic dirty tracking subscription (handled by helper)
        tree = self.tree_helper.create_tree_widget(
            flash_manager=self.form_manager,
            state=self.state,
            strip_config_suffix=True,
        )
        self.tree_helper.populate_from_mapping(tree, self._tree_dataclass_params)

        # Initialize dirty styling AFTER population (when _field_to_item is filled)
        self.tree_helper.initialize_dirty_styling()

        # Register tree repaint callback so flash animation triggers tree repaint
        self.form_manager.register_repaint_callback(lambda: tree.viewport().update())

        tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        return tree

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Scroll to the associated form section when a tree item is activated."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data.get("ui_hidden"):
            return

        item_type = data.get("type")
        if item_type == "dataclass":
            field_path = data.get("field_path") or data.get("field_name")
            if field_path:
                self._scroll_to_section(field_path)
        elif item_type == "inheritance_link":
            target_class = data.get("target_class")
            if target_class:
                field_name = self._find_field_for_class(target_class)
                if field_name:
                    self._scroll_to_section(field_name)

    def _find_field_for_class(self, target_class) -> Optional[str]:
        """Locate the parameter field that edits the given dataclass."""
        for field_name, obj_type in self._tree_dataclass_params.items():
            base_type = self.tree_helper.get_base_type(obj_type)
            if target_class in (obj_type, base_type):
                return field_name
        return None

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

        # Scrollable parameter form (matches config window pattern)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # No explicit styling - let it inherit from parent

        # Add form manager directly to scroll area (like config window)
        self.scroll_area.setWidget(self.form_manager)
        hierarchy_tree = self._create_configuration_tree()
        if hierarchy_tree:
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(True)  # CRITICAL: Allow tree to collapse
            splitter.setHandleWidth(5)  # Make handle more visible
            splitter.addWidget(hierarchy_tree)
            splitter.addWidget(self.scroll_area)
            splitter.setSizes([280, 720])
            layout.addWidget(splitter, 1)
            self.hierarchy_tree = hierarchy_tree
            self.content_splitter = splitter

            # Install collapsible splitter helper for double-click toggle
            self.splitter_helper = CollapsibleSplitterHelper(
                splitter, left_panel_index=0
            )
            self.splitter_helper.set_initial_size(280)
        else:
            layout.addWidget(self.scroll_area)
            self.hierarchy_tree = None
            self.content_splitter = None

        # Apply tree widget styling (matches config window)
        self.setStyleSheet(self.style_generator.generate_tree_widget_style())

    def get_action_buttons(self) -> Optional[QWidget]:
        """Get the action buttons container for external placement.

        This method allows parent windows (e.g., DualEditorWindow) to
        extract and reposition action buttons without modifying this widget's
        internal structure.

        Returns:
            QWidget: Container widget with action buttons (Code, Load, Save).
                      Returns None if header is rendered (buttons are in use).
        """
        if self._render_header:
            # Header is rendered, buttons are in use internally
            return None
        return self._action_buttons_container

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

    def _handle_parameter_change(self, param_name: str, value: Any):
        """Handle parameter change from form manager (mirrors Textual TUI).

        Args:
            param_name: Full path like "FunctionStep.processing_config.group_by" or "FunctionStep.name"
            value: New value
        """
        try:
            # Extract leaf field name from full path
            # "FunctionStep.processing_config.group_by" -> "group_by"
            # "FunctionStep.name" -> "name"
            path_parts = param_name.split(".")
            if len(path_parts) > 1:
                # Remove type name prefix
                path_parts = path_parts[1:]

            # For nested fields, the form manager already updated self.step via _mark_parents_modified
            # For top-level fields, we need to update self.step
            if len(path_parts) == 1:
                leaf_field = path_parts[0]

                # Get the properly converted value from state
                # The state handles all type conversions including List[Enum]
                final_value = self.state.get_current_values().get(leaf_field, value)

                # CRITICAL FIX: For function parameters, use fresh imports to avoid unpicklable registry wrappers
                if leaf_field == "func" and callable(final_value):
                    try:
                        import importlib

                        module = importlib.import_module(final_value.__module__)
                        final_value = getattr(module, final_value.__name__)
                    except Exception:
                        pass  # Use original if refresh fails

                # Update step attribute
                setattr(self.step, leaf_field, final_value)
                logger.debug(f"Updated step parameter {leaf_field}={final_value}")
            else:
                # Nested field - already updated by _mark_parents_modified
                logger.debug(
                    f"Nested field {'.'.join(path_parts)} already updated by dispatcher"
                )

            self.step_parameter_changed.emit()

        except Exception as e:
            logger.error(f"Error updating step parameter {param_name}: {e}")

    def load_step_settings(self):
        """Load step settings from .step file (mirrors Textual TUI)."""
        if not self.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return

        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.STEP_SETTINGS,
            title="Load Step Settings (.step)",
            file_filter="Step Files (*.step);;All Files (*)",
            mode="open",
        )

        if file_path:
            self._load_step_settings_from_file(file_path)

    def save_step_settings(self):
        """Save step settings to .step file (mirrors Textual TUI)."""
        if not self.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return

        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.STEP_SETTINGS,
            title="Save Step Settings (.step)",
            file_filter="Step Files (*.step);;All Files (*)",
            mode="save",
        )

        if file_path:
            self._save_step_settings_to_file(file_path)

    def _load_step_settings_from_file(self, file_path: Path):
        """Load step settings from file."""
        try:
            import dill as pickle

            with open(file_path, "rb") as f:
                step_data = pickle.load(f)

            # Update form manager with loaded values
            for param_name, value in step_data.items():
                try:
                    self.form_manager.update_parameter(param_name, value)
                    # Also update to step object
                    setattr(self.step, param_name, value)
                except (AttributeError, AttributeError):
                    pass

            # Refresh the form to show loaded values
            self.form_manager._refresh_all_placeholders()
            logger.debug(f"Loaded {len(step_data)} parameters from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load step settings from {file_path}: {e}")
            if self.service_adapter:
                self.service_adapter.show_error_dialog(
                    f"Failed to load step settings: {e}"
                )

    def _save_step_settings_to_file(self, file_path: Path):
        """Save step settings to file."""
        try:
            import dill as pickle

            # Get current values from state
            step_data = self.state.get_current_values()
            with open(file_path, "wb") as f:
                pickle.dump(step_data, f)
            logger.debug(f"Saved {len(step_data)} parameters to {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to save step settings to {file_path}: {e}")
            if self.service_adapter:
                self.service_adapter.show_error_dialog(
                    f"Failed to save step settings: {e}"
                )

    def get_current_step(self) -> FunctionStep:
        """Get the current step with all parameter values."""
        return self.step

    def update_step(self, step: FunctionStep):
        """Update the step and refresh the form."""
        self.step = step

        # Update form manager with new values
        for param_name in self.form_manager.parameters.keys():
            current_value = getattr(self.step, param_name, None)
            self.form_manager.update_parameter(param_name, current_value)

        logger.debug(
            f"Updated step parameter editor for step: {getattr(step, 'name', 'Unknown')}"
        )

    def view_step_code(self):
        """View the complete FunctionStep as Python code."""
        try:
            # CRITICAL: Refresh with live context BEFORE getting current values
            # This ensures code editor shows unsaved changes from other open windows
            ParameterOpsService().refresh_with_live_context(self.form_manager)

            # Get current step from state using to_object() to properly reconstruct nested dataclasses
            # NOTE: get_current_values() returns flat dotted paths which can't be passed to FunctionStep(**kwargs)
            # to_object() reconstructs proper nested structure from flat storage
            current_step = self.state.to_object()

            # CRITICAL: Get func from parent dual editor's function list editor if available
            # The func is managed by to Function Pattern tab in the dual editor
            parent_window = self.window()
            func = parent_window.func_editor.current_pattern
            current_step.func = func
            logger.debug(f"Using live func from function list editor: {func}")

            # Generate code using existing pattern
            python_code = generate_python_source(
                Assignment("step", current_step),
                header="# Function Step",
                clean_mode=True,
            )

            # Launch editor
            editor_service = SimpleCodeEditorService(self)
            use_external = os.environ.get(
                "OPENHCS_USE_EXTERNAL_EDITOR", ""
            ).lower() in ("1", "true", "yes")

            editor_service.edit_code(
                initial_content=python_code,
                title=f"Edit Step: {current_step.name}",
                callback=self._handle_edited_step_code,
                use_external=use_external,
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
            # SIMPLIFIED: Just exec with patched constructors
            # The patched constructors preserve None vs concrete distinction in raw field values
            # No need to parse code - just inspect raw values after exec
            namespace = {}
            with CodeEditorFormUpdater.patch_lazy_constructors():
                exec(edited_code, namespace)

            new_step = namespace.get("step")
            if not new_step:
                raise ValueError("No 'step' variable found in edited code")

            # Update step object
            self.step = new_step

            # IMPORTANT:
            # Do NOT block cross-window updates here. We want code-mode edits
            # to behave like a sequence of normal widget edits so that
            # FieldChangeDispatcher emits the same parameter_changed and
            # context_value_changed signals as manual interaction.
            CodeEditorFormUpdater.update_form_from_instance(
                self.form_manager,
                new_step,
                broadcast_callback=None,
            )

            # CRITICAL: Update function list editor if we're inside a dual editor window
            parent_window = self.window()
            func_editor = parent_window.func_editor
            func_editor._initialize_pattern_data(new_step.func)
            func_editor._populate_function_list()
            logger.debug(f"Updated function list editor with new func: {new_step.func}")

            # Notify parent window that step parameters changed
            self.step_parameter_changed.emit()

            logger.info(f"Updated step from code editor: {new_step.name}")

        except Exception as e:
            logger.error(f"Failed to apply edited step code: {e}")
            raise
