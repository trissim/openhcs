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
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, FormattedTextControl, Window, ScrollablePane, Container
from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioList, CheckboxList, Dialog # Added Dialog

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from .function_pattern_editor import FunctionPatternEditor
from .utils import show_error_dialog, prompt_for_path_dialog # Import the new dialog

logger = logging.getLogger(__name__)

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
        
        self.step_view_button = Button(
            "Step Settings", 
            handler=lambda: self._switch_view("step")
        )
        self.func_view_button = Button(
            "Func Pattern",
            handler=lambda: self._switch_view("func")
        )
        self.save_button = Button(
            "Save",
            handler=self._save_changes # Async handler
        )
        self.save_button.disabled = True # Disabled until changes are made

        self.close_button = Button(
            "Close",
            handler=self._close_editor # Async handler
        )

        menu_bar = VSplit([
            self.step_view_button,
            self.func_view_button,
            Window(width=5), # Spacer
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
                return self.step_settings_container
            else: # "func"
                return self.func_pattern_container if self.func_pattern_container else HSplit([Label("Func Editor Component Error")])

        dynamic_content_area = DynamicContainer(get_current_view_container)
        
        self._container = HSplit([
            menu_bar,
            Frame(dynamic_content_area, title=self._get_current_view_title())
        ])
        self._update_button_styles()


    def _get_current_view_title(self) -> str:
        if self.current_view == "step":
            return "Step Settings Editor"
        return "Function Pattern Editor"

    def _create_step_settings_view(self) -> ScrollablePane:
        """Creates the UI for editing AbstractStep __init__ parameters dynamically."""
        self.step_param_inputs.clear()
        rows = []
        
        sig = inspect.signature(AbstractStep.__init__)
        
        for param_name, param_obj in sig.parameters.items():
            if param_name == 'self' or param_obj.kind == param_obj.VAR_KEYWORD or param_obj.kind == param_obj.VAR_POSITIONAL:
                continue

            field_label = param_name.replace('_', ' ').title()
            current_value = getattr(self.editing_func_step, param_name, None)
            widget = None

            param_type_hint = param_obj.annotation
            actual_type = param_type_hint
            is_optional = get_origin(param_type_hint) is TypingUnion and type(None) in get_args(param_type_hint)
            if is_optional:
                actual_type = next((t for t in get_args(param_type_hint) if t is not type(None)), actual_type)
            
            if actual_type is bool:
                widget = CheckboxList(values=[(param_name, "")])
                if current_value:
                    widget.current_values = [param_name]
                # For CheckboxList, changes are typically handled on save or via a dedicated callback
                # if immediate _something_changed is needed. Here, we'll update on save.
            elif param_name == "variable_components": # Special handling for VariableComponents
                options = [(None, "(None)")] + [(member, member.name) for member in VariableComponents]
                
                # Determine initial selection for RadioList
                initial_selection = None
                if current_value and isinstance(current_value, list) and len(current_value) > 0:
                    current_name = current_value[0]
                    for member in VariableComponents:
                        if member.name == current_name:
                            initial_selection = member
                            break
                
                widget = RadioList(values=options, default=initial_selection)
                # The handler for RadioList is set directly on the widget instance
                # and it's called with the selected value (enum member or None)
                widget.handler = lambda val, n=param_name: self._something_changed(n, val)
            elif param_name == "group_by": # Special handling for GroupBy
                options = [(None, "(None)")] + [(member, member.name) for member in GroupBy]
                
                initial_selection = None
                if current_value and isinstance(current_value, list) and len(current_value) > 0:
                    current_name = current_value[0]
                    for member in GroupBy:
                        if member.name == current_name:
                            initial_selection = member
                            break
                
                widget = RadioList(values=options, default=initial_selection)
                widget.handler = lambda val, n=param_name: self._something_changed(n, val)
            elif actual_type is str or get_origin(actual_type) is Path or isinstance(actual_type, type(Path)) or actual_type is Any:
                widget = TextArea(
                    text=str(current_value or ("" if is_optional else param_obj.default if param_obj.default is not inspect.Parameter.empty else "")),
                    multiline=False, height=1, style="class:input-field"
                )
                widget.buffer.on_text_changed += lambda buff, n=param_name: self._something_changed(n, buff.text)
            else:
                logger.warning(f"Unhandled param type for UI: {param_name} ({actual_type}). Using TextArea.")
                widget = TextArea(
                    text=str(current_value or ""),
                    multiline=False, height=1, style="class:input-field"
                )
                widget.buffer.on_text_changed += lambda buff, n=param_name: self._something_changed(n, buff.text)

            if widget:
                self.step_param_inputs[param_name] = widget
                # Create reset button for this parameter
                reset_button = Button(
                    "Reset",
                    handler=lambda p_name=param_name, w=widget: get_app().create_background_task(
                        self._reset_step_parameter_field(p_name, w)
                    ),
                    width=8
                )
                rows.append(VSplit([
                    Label(f"{field_label}:", width=25),
                    widget,
                    Box(reset_button, width=10, padding_left=1) # Added Box for padding
                ], padding=0))
        
        parameter_fields_container = HSplit(rows)

        # Define these buttons earlier for the toolbar
        load_step_button = Button("Load .step", handler=self._load_step_object, width=12)
        save_step_as_button = Button("Save .step As...", handler=self._save_step_object_as, width=18)
        
        # Create a toolbar for these buttons
        step_settings_toolbar = VSplit([
            load_step_button,
            save_step_as_button,
            # Add a flexible spacer to push buttons to the left if desired, or manage alignment via Box
            Window(width=0, char=' ') # Flexible spacer
        ], height=1, padding_left=1)


        view_content = HSplit([
            # Toolbar at the top of the step settings view's content
            step_settings_toolbar,
            # Then the frame containing the parameters
            Frame(parameter_fields_container, title="Step Parameters (AbstractStep)")
            # Removed the old step_object_buttons VSplit from here
        ])
        return ScrollablePane(view_content)

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
        if param_name and param_name != "func": # Change from Step Settings
            try:
                sig = inspect.signature(AbstractStep.__init__)
                if param_name not in sig.parameters:
                    logger.warning(f"_something_changed called for unknown param: {param_name}")
                    return

                param_obj = sig.parameters[param_name]
                param_type_hint = param_obj.annotation
                actual_type = param_type_hint
                is_optional = get_origin(param_type_hint) is TypingUnion and type(None) in get_args(param_type_hint)
                if is_optional:
                    actual_type = next((t for t in get_args(param_type_hint) if t is not type(None)), actual_type)
                
                current_text = str(widget_value).strip()

                converted_value: Any
                if not current_text and is_optional: converted_value = None
                elif actual_type is Path: converted_value = Path(current_text) if current_text else None
                # Removed old TextArea handling for variable_components
                elif actual_type is str:
                    converted_value = current_text if not (is_optional and not current_text) else None
                else: 
                    converted_value = current_text if not (is_optional and not current_text) else None
                
                # Handle variable_components specifically if it was changed by RadioList
                if param_name == "variable_components":
                    # widget_value is the enum member or None
                    if widget_value and isinstance(widget_value, VariableComponents):
                        setattr(self.editing_func_step, param_name, [widget_value.name])
                    else: # widget_value is None (no selection)
                        setattr(self.editing_func_step, param_name, None)
                elif param_name == "group_by":
                    if widget_value and isinstance(widget_value, GroupBy):
                        setattr(self.editing_func_step, param_name, [widget_value.name])
                    else: # widget_value is None
                        setattr(self.editing_func_step, param_name, None)
                else: # Handle other types as before
                    setattr(self.editing_func_step, param_name, converted_value)
            except Exception as e:
                logger.warning(f"Error updating editing_func_step attribute '{param_name}' from text input '{widget_value}': {e}")
        # If param_name == "func", self.editing_func_step.func was already updated in _func_pattern_changed.

        has_changed = False
        sig_abstract = inspect.signature(AbstractStep.__init__)
        for name_to_check in sig_abstract.parameters:
            if name_to_check == 'self': continue
            original_val = getattr(self.original_func_step, name_to_check, inspect.Parameter.empty)
            editing_val = getattr(self.editing_func_step, name_to_check, inspect.Parameter.empty)
            if original_val != editing_val:
                has_changed = True
                break
        
        if not has_changed:
            if copy.deepcopy(self.editing_func_step.func) != copy.deepcopy(self.original_func_step.func):
                has_changed = True
        
        if self.save_button:
            self.save_button.disabled = not has_changed
        get_app().invalidate()

    async def _save_changes(self):
        logger.info("DualStepFuncEditorPane: Save changes initiated.")
        
        sig_abstract = inspect.signature(AbstractStep.__init__)
        for param_name, widget in self.step_param_inputs.items():
            if param_name not in sig_abstract.parameters: continue
            param_obj = sig_abstract.parameters[param_name]
            actual_type = param_obj.annotation 
            if get_origin(actual_type) is TypingUnion and type(None) in get_args(actual_type):
                actual_type = next((t for t in get_args(actual_type) if t is not type(None)), actual_type)

            if param_name == "variable_components" and isinstance(widget, RadioList):
                selected_enum_member = widget.current_value
                if selected_enum_member and isinstance(selected_enum_member, VariableComponents):
                    setattr(self.editing_func_step, param_name, [selected_enum_member.name])
                else: # No selection or (None) was selected
                    setattr(self.editing_func_step, param_name, None)
            elif param_name == "group_by" and isinstance(widget, RadioList):
                selected_enum_member = widget.current_value
                if selected_enum_member and isinstance(selected_enum_member, GroupBy):
                    setattr(self.editing_func_step, param_name, [selected_enum_member.name])
                else:
                    setattr(self.editing_func_step, param_name, None)
            elif isinstance(widget, CheckboxList) and actual_type is bool: # Existing bool handling
                setattr(self.editing_func_step, param_name, param_name in widget.current_values)

        if self.func_pattern_editor_component:
            self.editing_func_step.func = self.func_pattern_editor_component.get_pattern()
        else:
            logger.error("Func pattern editor component not found during save operation.")
            await show_error_dialog(
                title="Save Error",
                message="Function pattern editor is not available. Cannot save.",
                app_state=self.state
            )
            return

        try:
            step_to_save = copy.deepcopy(self.editing_func_step)
            await self.state.notify('step_pattern_saved', {'step': step_to_save})
            self.original_func_step = copy.deepcopy(self.editing_func_step)
            
            if self.save_button:
                self.save_button.disabled = True
            get_app().invalidate()
            logger.info(f"Step '{self.original_func_step.name}' saved successfully.")
        except Exception as e:
            logger.error(f"Error during final save or notification for step '{self.editing_func_step.name}': {e}", exc_info=True)
            await show_error_dialog(
                title="Save Error",
                message=f"Error saving step: {e}",
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

    async def shutdown(self):
        """Perform any cleanup if necessary."""
        logger.info("DualStepFuncEditorPane shutting down.")
        pass

    async def _load_step_object(self):
        logger.info("DualStepFuncEditorPane: Load .step object clicked.")
        file_path_str = await prompt_for_path_dialog(
            title="Load .step File",
            prompt_message="Enter path to .step file:",
            app_state=self.state
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
        original_value = getattr(self.original_func_step, param_name_to_reset, None)
        
        # Update the editing_func_step
        setattr(self.editing_func_step, param_name_to_reset, original_value)

        # Update the UI widget
        if isinstance(associated_widget, TextArea):
            associated_widget.text = str(original_value or "")
        elif isinstance(associated_widget, CheckboxList):
            # Assuming param_name_to_reset is the value for boolean CheckboxList
            associated_widget.current_values = [param_name_to_reset] if original_value else []
        elif isinstance(associated_widget, RadioList):
            # For VariableComponents and GroupBy
            enum_class = None
            if param_name_to_reset == "variable_components":
                enum_class = VariableComponents
            elif param_name_to_reset == "group_by":
                enum_class = GroupBy
            
            if enum_class:
                initial_selection = None
                if original_value and isinstance(original_value, list) and len(original_value) > 0:
                    original_name = original_value[0]
                    for member in enum_class:
                        if member.name == original_name:
                            initial_selection = member
                            break
                associated_widget.current_value = initial_selection
            else:
                logger.warning(f"Cannot reset RadioList for unknown enum param: {param_name_to_reset}")
        else:
            logger.warning(f"Cannot update UI for unknown widget type during reset: {type(associated_widget)}")

        # Trigger change detection to update save button state and redraw
        # Pass the original_value to ensure _something_changed correctly compares with the new state
        self._something_changed(param_name=param_name_to_reset, widget_value=original_value if not isinstance(associated_widget, RadioList) else associated_widget.current_value)
        get_app().invalidate() # Ensure UI redraws