"""
Step Settings Editor View Component for OpenHCS TUI.

This module defines the StepSettingsEditorView class, responsible for
dynamically building and managing UI widgets for editing parameters of a step
based on CoreStepData and its associated function signature/schema.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Dimension, FormattedTextControl, Window
from prompt_toolkit.widgets import Label, TextArea, Checkbox, RadioList, Button, Box, Frame
from prompt_toolkit.formatted_text import HTML

if TYPE_CHECKING:
    from openhcs.tui.interfaces import CoreStepData, ParamSchema # ParamSchema would define expected types and constraints

logger = logging.getLogger(__name__)

# Helper to create a unique ID for a widget if needed for focus management
def _create_widget_id(step_id: str, param_name: str) -> str:
    return f"step_{step_id}_param_{param_name}"

class StepSettingsEditorView:
    """
    Dynamically builds and manages UI widgets for editing step parameters.
    """
    def __init__(self,
                 on_parameter_change: Callable[[str, Any], None], # Callback: param_name, new_value
                 get_current_step_schema: Callable[[], Optional[Dict[str, 'ParamSchema']]] # Callback to get schema
                ):
        self.current_step_data: Optional[Dict[str, Any]] = None # CoreStepData-like dict
        self.current_step_schema: Optional[Dict[str, 'ParamSchema']] = None # Parameter schema
        
        self.on_parameter_change = on_parameter_change
        self.get_current_step_schema = get_current_step_schema # To fetch schema when step data is set

        self.widgets: Dict[str, Any] = {} # Stores the actual input widgets, keyed by param_name
        self._form_container: HSplit = HSplit([Label("No step selected or schema missing.")])
        self.container = DynamicContainer(lambda: self._form_container)

    async def set_step_data(self, step_data: Optional[Dict[str, Any]]): # CoreStepData-like dict
        """
        Sets the step data to be edited and rebuilds the form.
        """
        self.current_step_data = step_data
        self.widgets = {} # Clear old widgets

        if self.current_step_data:
            # Fetch the schema for the current step's function
            # The controller, when setting step_data, should ensure schema is available or fetched.
            # For now, assume get_current_step_schema() can provide it.
            self.current_step_schema = self.get_current_step_schema()
            if not self.current_step_schema:
                logger.warning(f"No parameter schema found for step '{step_data.get('name')}'. Cannot build editor form.")
                self._form_container = HSplit([Label(HTML("<style fg='ansired'>Error: Parameter schema missing for this step.</style>"))])
            else:
                await self._build_form()
        else:
            self.current_step_schema = None
            self._form_container = HSplit([Label("No step selected for settings editing.")])
        
        if get_app().is_running:
            get_app().invalidate()

    async def _build_form(self):
        """Builds the form widgets based on current_step_data and its schema."""
        if not self.current_step_data or not self.current_step_schema:
            self._form_container = HSplit([Label("No step/schema for form.")])
            return

        form_items = []
        parameters = self.current_step_data.get('parameters', {})

        for param_name, schema_entry in self.current_step_schema.items():
            # schema_entry would be a ParamSchema like object/dict
            # e.g., {'type': 'str' | 'int' | 'float' | 'bool' | 'enum', 'label': 'User Label', 
            #        'default': ..., 'choices': [...], 'min': ..., 'max': ...}
            
            param_label = schema_entry.get('label', param_name)
            param_type = schema_entry.get('type', 'str') # Default to string if type not specified
            current_value = parameters.get(param_name, schema_entry.get('default'))

            widget_id = _create_widget_id(self.current_step_data.get('id', 'unknown_step'), param_name)
            
            # Label for the parameter
            form_items.append(Label(f"{param_label}:", width=20)) # Fixed width for labels for alignment

            # Create widget based on type
            if param_type == 'bool':
                checkbox = Checkbox(checked=(current_value is True))
                def on_bool_change(cb: Checkbox, name:str = param_name): # Capture param_name
                    self.on_parameter_change(name, cb.checked)
                checkbox.show_cursor = False # type: ignore # PTK property
                checkbox.control.key_bindings.add('enter')(lambda event, cb=checkbox: setattr(cb, 'checked', not cb.checked) or on_bool_change(cb))
                # To make it interactive via space too:
                checkbox.control.key_bindings.add(' ')(lambda event, cb=checkbox: setattr(cb, 'checked', not cb.checked) or on_bool_change(cb))

                self.widgets[param_name] = checkbox
                form_items.append(checkbox)

            elif param_type in ['str', 'int', 'float']: # Use TextArea for these
                # For int/float, validation would be needed on change or save
                ta = TextArea(
                    text=str(current_value) if current_value is not None else "",
                    multiline=False,
                    height=1,
                    style="class:settings-editor.textarea", # Apply custom style if needed
                    # PTK doesn't have direct on_change for TextArea that gives final value easily.
                    # Usually, value is read on 'save'. We need to notify on each change for reactive UI.
                    # This can be done by patching accept_handler or using a more complex setup.
                    # For simplicity, we might rely on an "Update" button or focus loss to trigger on_parameter_change.
                    # A more reactive way:
                    # accept_handler=lambda buff, name=param_name: self.on_parameter_change(name, buff.text)
                )
                # Simplistic change notification: on focus lost (blur) or if user presses Enter
                def _text_area_changed(buffer, name=param_name, p_type=param_type):
                    new_text = buffer.text
                    try:
                        if p_type == 'int': new_val = int(new_text)
                        elif p_type == 'float': new_val = float(new_text)
                        else: new_val = new_text
                        self.on_parameter_change(name, new_val)
                    except ValueError:
                        logger.warning(f"Invalid input for {name} ({p_type}): {new_text}")
                        # Optionally show error in UI or revert
                
                ta.buffer.on_text_changed.add_handler(lambda buff, name=param_name, p_type=param_type: _text_area_changed(buff, name, p_type))


                self.widgets[param_name] = ta
                form_items.append(ta)
            
            elif param_type == 'enum' and 'choices' in schema_entry:
                choices = schema_entry['choices'] # List of (value, label) tuples or just values
                if choices and isinstance(choices[0], str): # Simple list of string values
                    choices_tuples = [(val, val) for val in choices]
                else: # Assume list of (value, label) tuples
                    choices_tuples = choices
                
                # Find current choice for RadioList default
                current_choice_value = current_value
                
                radio_list = RadioList(values=choices_tuples, default=current_choice_value)
                # RadioList updates its 'current_value' when changed
                # We need to hook into that. This might require a custom RadioList or careful handling.
                # For now, let's assume we can attach a handler or read it on demand.
                # A simple way is to wrap its handler:
                original_handler = radio_list.control.get_changed_handler()
                def on_enum_change(name=param_name, rl=radio_list):
                    if original_handler: original_handler() # Call original if it does something important
                    self.on_parameter_change(name, rl.current_value)
                
                radio_list.control.get_changed_handler = lambda: on_enum_change # type: ignore
                self.widgets[param_name] = radio_list
                form_items.append(radio_list)

            else: # Default for unknown or complex types for now
                form_items.append(Label(f"(Unsupported type: {param_type}) {str(current_value)}"))
            
            form_items.append(Window(height=1, char=' ')) # Spacer after each param row

        if not form_items:
            form_items.append(Label("No editable parameters for this step."))
            
        self._form_container = HSplit(form_items)

    def get_widget_value(self, param_name: str) -> Any:
        """Gets the current value from a specific widget."""
        widget = self.widgets.get(param_name)
        if not widget:
            return None
        
        if isinstance(widget, Checkbox):
            return widget.checked
        elif isinstance(widget, TextArea):
            # Need to parse for int/float based on schema
            schema_entry = self.current_step_schema.get(param_name, {}) if self.current_step_schema else {}
            param_type = schema_entry.get('type', 'str')
            try:
                if param_type == 'int': return int(widget.text)
                elif param_type == 'float': return float(widget.text)
                return widget.text
            except ValueError:
                return widget.text # Return raw text if parsing fails
        elif isinstance(widget, RadioList):
            return widget.current_value
        return None # Should not happen if widget exists

    def __pt_container__(self) -> Container:
        return self.container
