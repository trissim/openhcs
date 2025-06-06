from prompt_toolkit.layout import Container
"""
Dual Step/Function Pattern Editor Pane for OpenHCS TUI.

This pane provides a two-view editor for `FuncStep` objects:
1.  "Step Settings" view: Edits parameters inherited from `AbstractStep`.
2.  "Func Pattern" view: Edits the `func` attribute (the function pattern) of the `FuncStep`.
"""
import asyncio
import logging
import inspect # For signature inspection
import copy # For deepcopy
import pickle # Added for pickling
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union as TypingUnion, get_origin, get_args

from openhcs.constants.constants import VariableComponents, GroupBy # Added imports

from prompt_toolkit.application import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, FormattedTextControl, Window, ScrollablePane, Container, Dimension
from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioList, CheckboxList, Dialog, Checkbox # Added Checkbox

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from .function_pattern_editor import FunctionPatternEditor
from openhcs.tui.utils import show_error_dialog

logger = logging.getLogger(__name__)

# SafeButton eliminated - use Button directly

class DualStepFuncEditorPane:
    """
    A dual-view editor for FuncStep objects, combining AbstractStep parameter
    editing and Function Pattern editing.
    """
    def __init__(self, state: Any, func_step: FunctionStep):
        """
        Initialize the DualStepFuncEditorPane.

        Args:
            state: The TUIState instance.
            func_step: The FuncStep object to be edited.
        """
        self.state = state
        self.original_func_step = func_step
        # Create a working copy to modify, to allow for "cancel"
        self.editing_func_step = copy.deepcopy(func_step)

        self.current_view = "step"  # "step" or "func"

        # UI components will be created in _initialize_ui
        self.step_settings_container: Optional[ScrollablePane] = None
        self.func_pattern_editor_component: Optional[FunctionPatternEditor] = None
        self.func_pattern_container: Optional[Any] = None

        self.step_param_inputs: Dict[str, Any] = {}

        self.save_button: Optional[Button] = None
        self.close_button: Optional[Button] = None
        self.step_view_button: Optional[Button] = None
        self.func_view_button: Optional[Button] = None

        self._container: Optional[HSplit] = None
        self._initialize_ui()

    def _func_pattern_changed(self):
        """Callback for when the func pattern is changed by its editor."""
        if self.func_pattern_editor_component:
            self.editing_func_step.func = self.func_pattern_editor_component.get_pattern()
        # Trigger change detection. Pass the new pattern for consistency,
        # though _something_changed will primarily use self.editing_func_step.func.
        self._something_changed(param_name="func", widget_value=self.editing_func_step.func)

    def _initialize_ui(self):
        """Initialize all UI components for the editor."""

        # Top menu bar with Step/Func toggle and Save/Close buttons
        self.step_view_button = Button("Step",
            handler=lambda: self._switch_view("step"),
            width=10
        )
        self.func_view_button = Button("Func",
            handler=lambda: self._switch_view("func"),
            width=10
        )
        self.save_button = Button("Save",
            handler=self._save_changes,
            width=8
        )
        self.save_button.disabled = True # Disabled until changes are made

        self.close_button = Button("Close",
            handler=self._close_editor,
            width=8
        )

        # Top menu bar with Step/Func toggle and Save/Close buttons
        top_menu_bar = VSplit([
            self.step_view_button,
            self.func_view_button,
            Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
            self.save_button,
            self.close_button,
        ], height=1, padding=1)

        self.step_settings_container = self._create_step_settings_view()

        # Initialize the FunctionPatternEditor component
        if not self.func_pattern_editor_component: # Ensure it's only initialized once
            self.func_pattern_editor_component = FunctionPatternEditor(
                state=self.state,
                initial_pattern=self.editing_func_step.func,
                change_callback=self._func_pattern_changed
            )
        # Ensure func_pattern_container is set from the component's container
        self.func_pattern_container = self.func_pattern_editor_component.container

        # Dynamic container to switch between views
        def get_current_view_container():
            if self.current_view == "step":
                # Create a container with title bar for Step Settings
                return HSplit([
                    # Title bar for Step Settings
                    VSplit([
                        Label(" Step Settings Editor ", style="class:frame.title"),
                        Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
                        Button("Load", handler=self._load_step_object, width=8),
                        Button("Save As", handler=self._save_step_object_as, width=10)
                    ], height=1, style="class:frame.title"),
                    # Content area
                    self.step_settings_container
                ])
            else: # "func"
                # Create a container with title bar for Func Pattern
                return HSplit([
                    # Title bar for Func Pattern
                    VSplit([
                        Label(" Func Pattern Editor ", style="class:frame.title"),
                        Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
                        Button("Add", handler=lambda: self.func_pattern_editor_component._add_function() if hasattr(self.func_pattern_editor_component, '_add_function') else None, width=6),
                        Button("Load", handler=lambda: self.func_pattern_editor_component._load_func_pattern_from_file_handler() if hasattr(self.func_pattern_editor_component, '_load_func_pattern_from_file_handler') else None, width=8),
                        Button("Save As", handler=lambda: self.func_pattern_editor_component._save_func_pattern_as_file_handler() if hasattr(self.func_pattern_editor_component, '_save_func_pattern_as_file_handler') else None, width=10)
                    ], height=1, style="class:frame.title"),
                    # Content area
                    self.func_pattern_container if self.func_pattern_container else HSplit([Label("Func Editor Component Error")])
                ])

        # Create the dynamic content area
        self._dynamic_content_wrapper = DynamicContainer(get_current_view_container)

        # Create the main container with the top menu bar and dynamic content
        self._container = HSplit([
            top_menu_bar,
            Frame(self._dynamic_content_wrapper)
        ])

        self._update_button_styles()


    def _get_current_view_title(self) -> str:
        if self.current_view == "step":
            return "Step Settings Editor"
        return "Function Pattern Editor"

    def _create_step_settings_view(self) -> ScrollablePane:
        """Creates the UI for editing FunctionStep __init__ parameters dynamically."""
        self.step_param_inputs.clear()

        # Get FunctionStep signature - BACKEND API COMPLIANT
        sig = inspect.signature(FunctionStep.__init__)
        rows = self._create_parameter_rows(sig)

        # Create main content
        parameter_fields_container = HSplit(rows)
        step_settings_toolbar = self._create_step_settings_toolbar()

        view_content = HSplit([
            step_settings_toolbar,
            Frame(parameter_fields_container, title="Step Parameters (FunctionStep)")
        ])

        return ScrollablePane(view_content)

    def _create_parameter_rows(self, sig: inspect.Signature) -> List[Any]:
        """Create UI rows for each parameter in the signature."""
        rows = []

        for param_name, param_obj in sig.parameters.items():
            if self._should_skip_parameter(param_name, param_obj):
                continue

            param_info = self._extract_parameter_info(param_name, param_obj)
            widget = self._create_parameter_widget(param_info)

            if widget:
                self.step_param_inputs[param_name] = widget
                row = self._create_parameter_row(param_info, widget)
                rows.append(row)

        return rows

    def _should_skip_parameter(self, param_name: str, param_obj: inspect.Parameter) -> bool:
        """Check if parameter should be skipped in UI."""
        return (param_name == 'self' or
                param_obj.kind == param_obj.VAR_KEYWORD or
                param_obj.kind == param_obj.VAR_POSITIONAL)

    def _extract_parameter_info(self, param_name: str, param_obj: inspect.Parameter) -> dict:
        """Extract parameter information for UI creation."""
        field_label = param_name.replace('_', ' ').title()
        current_value = getattr(self.editing_func_step, param_name, None)

        param_type_hint = param_obj.annotation
        actual_type = param_type_hint
        is_optional = get_origin(param_type_hint) is TypingUnion and type(None) in get_args(param_type_hint)
        if is_optional:
            actual_type = next((t for t in get_args(param_type_hint) if t is not type(None)), actual_type)

        return {
            'param_name': param_name,
            'param_obj': param_obj,
            'field_label': field_label,
            'current_value': current_value,
            'actual_type': actual_type,
            'is_optional': is_optional
        }

    def _create_parameter_widget(self, param_info: dict) -> Any:
        """Create appropriate widget for parameter type."""
        param_name = param_info['param_name']
        actual_type = param_info['actual_type']
        current_value = param_info['current_value']

        if actual_type is bool:
            return self._create_bool_parameter_widget(param_name, current_value)
        elif param_name == "variable_components":
            return self._create_variable_components_widget(current_value)
        elif param_name == "group_by":
            return self._create_group_by_widget(current_value)
        elif self._is_string_like_type(actual_type):
            return self._create_string_parameter_widget(param_name, param_info)
        else:
            return self._create_fallback_widget(param_name, current_value, actual_type)

    def _create_bool_parameter_widget(self, param_name: str, current_value: Any) -> Checkbox:
        """Create checkbox widget for boolean parameters."""
        widget = Checkbox(checked=bool(current_value))
        original_handler = widget.control.mouse_handler

        def new_mouse_handler(mouse_event):
            result = original_handler(mouse_event)
            self._something_changed(param_name, widget)
            return result

        widget.control.mouse_handler = new_mouse_handler
        return widget

    def _create_variable_components_widget(self, current_value: Any) -> RadioList:
        """Create radio list widget for VariableComponents parameter."""
        options = [(None, "(None)")] + [(member, member.name) for member in VariableComponents]
        initial_selection = self._find_enum_selection(current_value, VariableComponents)

        widget = RadioList(values=options, default=initial_selection)
        widget.handler = lambda val: self._something_changed("variable_components", val)
        return widget

    def _create_group_by_widget(self, current_value: Any) -> RadioList:
        """Create radio list widget for GroupBy parameter."""
        options = [(None, "(None)")] + [(member, member.name) for member in GroupBy]
        initial_selection = self._find_enum_selection(current_value, GroupBy)

        widget = RadioList(values=options, default=initial_selection)
        widget.handler = lambda val: self._something_changed("group_by", val)
        return widget

    def _find_enum_selection(self, current_value: Any, enum_class: type) -> Any:
        """Find the enum member that matches the current value."""
        if current_value and isinstance(current_value, list) and len(current_value) > 0:
            current_name = current_value[0]
            for member in enum_class:
                if member.name == current_name:
                    return member
        return None

    def _is_string_like_type(self, actual_type: type) -> bool:
        """Check if type is string-like (str, Path, Any)."""
        return (actual_type is str or
                get_origin(actual_type) is Path or
                isinstance(actual_type, type(Path)) or
                actual_type is Any)

    def _create_string_parameter_widget(self, param_name: str, param_info: dict) -> TextArea:
        """Create text area widget for string-like parameters."""
        current_value = param_info['current_value']
        param_obj = param_info['param_obj']
        is_optional = param_info['is_optional']

        default_text = ""
        if current_value:
            default_text = str(current_value)
        elif not is_optional and param_obj.default is not inspect.Parameter.empty:
            default_text = str(param_obj.default)

        widget = TextArea(
            text=default_text,
            multiline=False,
            height=1,
            style="class:input-field"
        )
        widget.buffer.on_text_changed += lambda buff: self._something_changed(param_name, buff.text)
        return widget

    def _create_fallback_widget(self, param_name: str, current_value: Any, actual_type: type) -> TextArea:
        """Create fallback text area widget for unhandled parameter types."""
        logger.warning(f"Unhandled param type for UI: {param_name} ({actual_type}). Using TextArea.")

        widget = TextArea(
            text=str(current_value or ""),
            multiline=False,
            height=1,
            style="class:input-field"
        )
        widget.buffer.on_text_changed += lambda buff: self._something_changed(param_name, buff.text)
        return widget

    def _create_parameter_row(self, param_info: dict, widget: Any) -> VSplit:
        """Create a UI row for a parameter."""
        param_name = param_info['param_name']
        field_label = param_info['field_label']

        reset_button = Button("Reset",
            handler=lambda: get_app().create_background_task(
                self._reset_step_parameter_field(param_name, widget)
            ),
            width=8
        )

        return VSplit([
            Label(f"{field_label}:", width=25),
            widget,
            Box(reset_button, width=10, padding_left=1)
        ], padding=0)

    def _create_step_settings_toolbar(self) -> VSplit:
        """Create toolbar with load and save buttons."""
        load_step_button = Button("Load .step", handler=self._load_step_object, width=12)
        save_step_as_button = Button("Save .step As", handler=self._save_step_object_as, width=15)

        return VSplit([
            load_step_button,
            save_step_as_button,
            Window(width=0, char=' ')  # Flexible spacer
        ], height=1, padding_left=1)

    def _create_func_pattern_view(self) -> Any:
        """
        Returns the container of the FunctionPatternEditor component.
        The component should be initialized in _initialize_ui.
        """
        if self.func_pattern_editor_component:
            return self.func_pattern_editor_component.container
        else:
            logger.error("FunctionPatternEditor component not available when creating func pattern view.")
            return HSplit([Label("Error: Function Pattern Editor component is not loaded.")])

    def _switch_view(self, view_name: str):
        self.current_view = view_name
        if self._container and len(self._container.children) > 1 and isinstance(self._container.children[1], Frame):
            self._container.children[1].title = self._get_current_view_title
        self._update_button_styles()
        get_app().invalidate()

    def _update_button_styles(self):
        """Update button styles to indicate active view."""
        if self.step_view_button:
            self.step_view_button.style = "class:button.focused" if self.current_view == "step" else "class:button"
        if self.func_view_button:
            self.func_view_button.style = "class:button.focused" if self.current_view == "func" else "class:button"

    def _something_changed(self, param_name: Optional[str] = None, widget_value: Any = None):
        """
        Callback for when an editable field changes.
        Updates self.editing_func_step and enables save button if changes detected.
        """
        if param_name and param_name != "func":
            self._update_step_parameter(param_name, widget_value)

        # Check for changes and update UI state
        has_changed = self._detect_changes()
        self._update_save_button_state(has_changed)

    def _update_step_parameter(self, param_name: str, widget_value: Any):
        """Update a single step parameter from widget value."""
        try:
            sig = inspect.signature(AbstractStep.__init__)
            if param_name not in sig.parameters:
                logger.warning(f"Unknown parameter: {param_name}")
                return

            # Handle special enum parameters
            if param_name in ("variable_components", "group_by"):
                self._update_enum_parameter(param_name, widget_value)
                return

            # Handle regular parameters
            converted_value = self._convert_widget_value(param_name, widget_value, sig.parameters[param_name])
            setattr(self.editing_func_step, param_name, converted_value)

        except Exception as e:
            logger.warning(f"Error updating parameter '{param_name}': {e}")

    def _update_enum_parameter(self, param_name: str, widget_value: Any):
        """Update enum parameters (variable_components, group_by)."""
        enum_class = VariableComponents if param_name == "variable_components" else GroupBy

        if widget_value and isinstance(widget_value, enum_class):
            setattr(self.editing_func_step, param_name, [widget_value.name])
        else:
            setattr(self.editing_func_step, param_name, None)

    def _convert_widget_value(self, param_name: str, widget_value: Any, param_obj) -> Any:
        """Convert widget value to appropriate type for parameter."""
        param_type_hint = param_obj.annotation
        actual_type = param_type_hint
        is_optional = get_origin(param_type_hint) is TypingUnion and type(None) in get_args(param_type_hint)

        if is_optional:
            actual_type = next((t for t in get_args(param_type_hint) if t is not type(None)), actual_type)

        current_text = str(widget_value).strip()

        if not current_text and is_optional:
            return None
        elif actual_type is Path:
            return Path(current_text) if current_text else None
        elif actual_type is str:
            return current_text if not (is_optional and not current_text) else None
        else:
            return current_text if not (is_optional and not current_text) else None

    def _detect_changes(self) -> bool:
        """Detect if any changes have been made to the step."""
        step_changed = self._detect_step_settings_changes()
        func_changed = self._detect_func_pattern_changes()
        return step_changed or func_changed

    def _detect_step_settings_changes(self) -> bool:
        """Detect changes in step settings."""
        sig_abstract = inspect.signature(AbstractStep.__init__)

        for name_to_check in sig_abstract.parameters:
            if name_to_check == 'self':
                continue

            original_val = getattr(self.original_func_step, name_to_check, inspect.Parameter.empty)

            # Handle checkbox widgets specially
            widget_instance = self.step_param_inputs.get(name_to_check)
            if isinstance(widget_instance, Checkbox):
                editing_val = widget_instance.checked
                setattr(self.editing_func_step, name_to_check, editing_val)
            else:
                editing_val = getattr(self.editing_func_step, name_to_check, inspect.Parameter.empty)

            if original_val != editing_val:
                return True

        return False

    def _detect_func_pattern_changes(self) -> bool:
        """Detect changes in function pattern."""
        return (copy.deepcopy(self.editing_func_step.func) !=
                copy.deepcopy(self.original_func_step.func))

    def _update_save_button_state(self, has_changed: bool):
        """Update save button enabled/disabled state."""
        if self.save_button:
            self.save_button.disabled = not has_changed
        get_app().invalidate()

    async def _save_changes(self):
        """Save changes to the step and function pattern."""
        logger.info("DualStepFuncEditorPane: Save changes initiated.")

        try:
            await self._save_step_parameters()
            await self._save_function_pattern()
            await self._finalize_save()
        except Exception as e:
            await self._handle_save_error(e)

    async def _save_step_parameters(self):
        """Save step parameters from UI widgets."""
        sig_abstract = inspect.signature(AbstractStep.__init__)

        for param_name, widget in self.step_param_inputs.items():
            if param_name not in sig_abstract.parameters:
                continue

            param_obj = sig_abstract.parameters[param_name]
            actual_type = self._extract_actual_type(param_obj.annotation)

            self._save_parameter_value(param_name, widget, actual_type)

    def _extract_actual_type(self, annotation: type) -> type:
        """Extract actual type from annotation, handling Optional types."""
        if get_origin(annotation) is TypingUnion and type(None) in get_args(annotation):
            return next((t for t in get_args(annotation) if t is not type(None)), annotation)
        return annotation

    def _save_parameter_value(self, param_name: str, widget: Any, actual_type: type):
        """Save a single parameter value based on widget type."""
        if param_name == "variable_components" and isinstance(widget, RadioList):
            self._save_variable_components_parameter(param_name, widget)
        elif param_name == "group_by" and isinstance(widget, RadioList):
            self._save_group_by_parameter(param_name, widget)
        elif isinstance(widget, Checkbox) and actual_type is bool:
            setattr(self.editing_func_step, param_name, widget.checked)

    def _save_variable_components_parameter(self, param_name: str, widget: RadioList):
        """Save variable_components parameter."""
        selected_enum_member = widget.current_value
        if selected_enum_member and isinstance(selected_enum_member, VariableComponents):
            setattr(self.editing_func_step, param_name, [selected_enum_member.name])
        else:
            setattr(self.editing_func_step, param_name, None)

    def _save_group_by_parameter(self, param_name: str, widget: RadioList):
        """Save group_by parameter."""
        selected_enum_member = widget.current_value
        if selected_enum_member and isinstance(selected_enum_member, GroupBy):
            setattr(self.editing_func_step, param_name, [selected_enum_member.name])
        else:
            setattr(self.editing_func_step, param_name, None)

    async def _save_function_pattern(self):
        """Save function pattern from editor component."""
        if self.func_pattern_editor_component:
            self.editing_func_step.func = self.func_pattern_editor_component.get_pattern()
        else:
            logger.error("Func pattern editor component not found during save operation.")
            await show_error_dialog(
                title="Save Error",
                message="Function pattern editor is not available. Cannot save.",
                app_state=self.state
            )
            raise ValueError("Function pattern editor not available")

    async def _finalize_save(self):
        """Finalize the save operation."""
        step_to_save = copy.deepcopy(self.editing_func_step)
        await self.state.notify('step_pattern_saved', {'step': step_to_save})
        self.original_func_step = copy.deepcopy(self.editing_func_step)

        if self.save_button:
            self.save_button.disabled = True
        get_app().invalidate()
        logger.info(f"Step '{self.original_func_step.name}' saved successfully.")

    async def _handle_save_error(self, error: Exception):
        """Handle errors during save operation."""
        logger.error(f"Error during final save or notification for step '{self.editing_func_step.name}': {error}", exc_info=True)
        await show_error_dialog(
            title="Save Error",
            message=f"Error saving step: {error}",
            app_state=self.state
        )

    async def _close_editor(self):
        logger.info("DualStepFuncEditorPane: Close editor clicked.")
        # Consider checking for unsaved changes before notifying
        # if not self.save_button or self.save_button.disabled: # No unsaved changes
        await self.state.notify('step_editing_cancelled')
        # else:
            # Show confirmation dialog for unsaved changes

    @property
    def container(self) -> Container: # Ensure Container is imported from prompt_toolkit.layout
        if self._container is None:
            logger.error("DualStepFuncEditorPane: Container accessed before initialization.")
            return HSplit([Label("Error: Editor not initialized.")]) # Fallback
        return self._container

    def get_buttons_container(self) -> Container:
        """
        Get the buttons container for the Dual Step/Func Editor.
        This is used by the TUI architecture to get the buttons for the left pane.

        Returns:
            Container with the buttons for the Dual Step/Func Editor
        """
        return VSplit([
            self.step_view_button,
            self.func_view_button,
            Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
            self.save_button,
            self.close_button,
        ], height=1, padding=1)

    async def shutdown(self):
        """Perform any cleanup if necessary."""
        logger.info("DualStepFuncEditorPane shutting down.")
        pass

    async def _load_step_object(self):
        logger.info("DualStepFuncEditorPane: Load .step object clicked.")
        from openhcs.tui.utils.dialog_helpers import prompt_for_file_dialog
        file_path_str = await prompt_for_file_dialog(
            title="Load .step File",
            prompt_message="Select .step file to load:",
            app_state=self.state,
            filemanager=getattr(self.context, 'filemanager', None),
            selection_mode="files",
            filter_extensions=[".step"]
        )

        if not file_path_str:
            logger.info("Load .step operation cancelled by user.")
            return

        file_path = Path(file_path_str)

        if not file_path.exists() or not file_path.is_file():
            await show_error_dialog("Load Error", f"File not found or is not a file: {file_path}", app_state=self.state)
            return

        try:
            with open(file_path, "rb") as f:
                loaded_object = pickle.load(f)

            if not isinstance(loaded_object, FunctionStep):
                await show_error_dialog("Load Error", "File does not contain a valid FunctionStep.", app_state=self.state)
                return

            self.original_func_step = loaded_object # Keep the loaded one as original for comparison
            self.editing_func_step = copy.deepcopy(loaded_object)

            # Refresh UI with loaded step data
            self._initialize_ui() # Re-initialize UI to reflect new step
            self._something_changed() # Recalculate save button state (should be disabled initially)
            logger.info(f"Successfully loaded step from {file_path}")

        except pickle.UnpicklingError as e:
            logger.error(f"Error unpickling step from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Error", f"Error unpickling file: {e}", app_state=self.state)
        except Exception as e:
            logger.error(f"Failed to load step from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Error", f"Failed to load step: {e}", app_state=self.state)

    async def _save_step_object_as(self):
        logger.info("DualStepFuncEditorPane: Save .step object As clicked.")
        file_path_str = await prompt_for_path_dialog(
            title="Save .step File As",
            prompt_message="Enter path to save .step file:",
            app_state=self.state,
            initial_value=f"{self.editing_func_step.name}.step" # Suggest a default filename
        )

        if not file_path_str:
            logger.info("Save .step As operation cancelled by user.")
            return

        file_path = Path(file_path_str)

        try:
            # Ensure the func pattern is up-to-date before pickling
            if self.func_pattern_editor_component:
                 self.editing_func_step.func = self.func_pattern_editor_component.get_pattern()

            with open(file_path, "wb") as f:
                pickle.dump(self.editing_func_step, f)
            logger.info(f"Successfully saved step to {file_path}")
            # Optionally, update original_func_step and disable save button if this implies a "save"
            # self.original_func_step = copy.deepcopy(self.editing_func_step)
            # if self.save_button: self.save_button.disabled = True
            # get_app().invalidate()

            # For "Save As", typically we don't automatically consider it "saved" in the editor session
            # unless it's also a primary save action.
            # For now, just log success. A success dialog could be added.

        except pickle.PicklingError as e:
            logger.error(f"Error pickling step to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Error", f"Error pickling step: {e}", app_state=self.state)
        except Exception as e:
            logger.error(f"Failed to save step to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Error", f"Failed to save step: {e}", app_state=self.state)

    async def _reset_step_parameter_field(self, param_name_to_reset: str, associated_widget: Any):
        """Resets a specific step parameter field to its original value."""
        logger.info(f"Resetting parameter: {param_name_to_reset}")

        original_value = self._get_original_parameter_value(param_name_to_reset)
        self._update_editing_step_parameter(param_name_to_reset, original_value)
        self._update_widget_with_original_value(associated_widget, param_name_to_reset, original_value)
        self._trigger_change_detection(param_name_to_reset, associated_widget, original_value)

    def _get_original_parameter_value(self, param_name: str) -> Any:
        """Get the original value for a parameter."""
        return getattr(self.original_func_step, param_name, None)

    def _update_editing_step_parameter(self, param_name: str, original_value: Any):
        """Update the editing step with the original value."""
        setattr(self.editing_func_step, param_name, original_value)

    def _update_widget_with_original_value(self, widget: Any, param_name: str, original_value: Any):
        """Update the UI widget with the original value."""
        if isinstance(widget, TextArea):
            self._reset_text_area_widget(widget, original_value)
        elif isinstance(widget, Checkbox):
            self._reset_checkbox_widget(widget, original_value)
        elif isinstance(widget, RadioList):
            self._reset_radio_list_widget(widget, param_name, original_value)
        else:
            logger.warning(f"Cannot update UI for unknown widget type during reset: {type(widget)}")

    def _reset_text_area_widget(self, widget: TextArea, original_value: Any):
        """Reset text area widget to original value."""
        widget.text = str(original_value or "")

    def _reset_checkbox_widget(self, widget: Checkbox, original_value: Any):
        """Reset checkbox widget to original value."""
        widget.checked = bool(original_value)

    def _reset_radio_list_widget(self, widget: RadioList, param_name: str, original_value: Any):
        """Reset radio list widget to original value."""
        enum_class = self._get_enum_class_for_parameter(param_name)

        if enum_class:
            initial_selection = self._find_enum_member_by_value(enum_class, original_value)
            widget.current_value = initial_selection
        else:
            logger.warning(f"Cannot reset RadioList for unknown enum param: {param_name}")

    def _get_enum_class_for_parameter(self, param_name: str) -> Optional[type]:
        """Get the enum class for a parameter name."""
        if param_name == "variable_components":
            return VariableComponents
        elif param_name == "group_by":
            return GroupBy
        return None

    def _find_enum_member_by_value(self, enum_class: type, original_value: Any) -> Any:
        """Find enum member that matches the original value."""
        if not original_value or not isinstance(original_value, list) or len(original_value) == 0:
            return None

        original_name = original_value[0]
        for member in enum_class:
            if member.name == original_name:
                return member
        return None

    def _trigger_change_detection(self, param_name: str, widget: Any, original_value: Any):
        """Trigger change detection to update save button state."""
        widget_value = original_value if not isinstance(widget, RadioList) else widget.current_value
        self._something_changed(param_name=param_name, widget_value=widget_value)
        get_app().invalidate()