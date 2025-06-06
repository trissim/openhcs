"""
Step Parameter Editor for OpenHCS TUI.

Clean, focused component for editing AbstractStep parameters only.
Follows the same composition pattern as other TUI components.
"""
import inspect
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Union as TypingUnion, get_origin, get_args
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, ScrollablePane, Window, Dimension
from prompt_toolkit.widgets import Button, Frame, Label, TextArea, RadioList, Checkbox, Box

from openhcs.constants.constants import VariableComponents, GroupBy
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)

# Use Button directly - no defensive programming


class StepParameterEditor:
    """
    Pure AbstractStep parameter editor.
    
    Handles only step parameter editing with change callbacks.
    No file I/O, no tab coordination, no god class responsibilities.
    """
    
    def __init__(self, 
                 step: FunctionStep, 
                 on_change: Callable[[str, Any], None],
                 on_load: Optional[Callable[[], None]] = None,
                 on_save_as: Optional[Callable[[], None]] = None):
        """
        Initialize step parameter editor.
        
        Args:
            step: The FunctionStep to edit
            on_change: Callback when parameter changes (param_name, value)
            on_load: Optional callback for load button
            on_save_as: Optional callback for save as button
        """
        self.step = step
        self.on_change = on_change
        self.on_load = on_load
        self.on_save_as = on_save_as
        
        # UI state
        self.param_inputs: Dict[str, Any] = {}
        
        # Build UI
        self._container = self._build_ui()
    
    @property
    def container(self) -> Container:
        """Get the container for this editor."""
        return self._container
    
    def _build_ui(self) -> Container:
        """Build the step parameter editor UI."""
        # Create parameter rows - BACKEND API COMPLIANT
        sig = inspect.signature(FunctionStep.__init__)
        rows = self._create_parameter_rows(sig)

        # Create toolbar
        toolbar = self._create_toolbar()

        # Combine
        content = HSplit([
            toolbar,
            Frame(HSplit(rows), title="Step Parameters (FunctionStep)")
        ])
        
        return ScrollablePane(content)
    
    def _create_toolbar(self) -> VSplit:
        """Create toolbar with load and save buttons."""
        buttons = []
        
        if self.on_load:
            buttons.append(Button("Load .step", handler=self.on_load, width=12))

        if self.on_save_as:
            buttons.append(Button("Save .step As", handler=self.on_save_as, width=18))
        
        buttons.append(Window(width=0, char=' '))  # Spacer
        
        return Box(VSplit(buttons, height=1), padding_left=1)
    
    def _create_parameter_rows(self, sig: inspect.Signature) -> List[Container]:
        """Create UI rows for each parameter."""
        rows = []
        self.param_inputs.clear()
        
        for param_name, param_obj in sig.parameters.items():
            if self._should_skip_parameter(param_name, param_obj):
                continue
            
            param_info = self._extract_parameter_info(param_name, param_obj)
            widget = self._create_parameter_widget(param_info)
            
            if widget:
                self.param_inputs[param_name] = widget
                row = self._create_parameter_row(param_info, widget)
                rows.append(row)
        
        return rows
    
    def _should_skip_parameter(self, param_name: str, param_obj: inspect.Parameter) -> bool:
        """Check if parameter should be skipped."""
        return (param_name == 'self' or
                param_obj.kind == param_obj.VAR_KEYWORD or
                param_obj.kind == param_obj.VAR_POSITIONAL)
    
    def _extract_parameter_info(self, param_name: str, param_obj: inspect.Parameter) -> dict:
        """Extract parameter information for UI creation."""
        field_label = param_name.replace('_', ' ').title()
        current_value = getattr(self.step, param_name, None)
        
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
            return self._create_bool_widget(param_name, current_value)
        elif param_name == "variable_components":
            return self._create_variable_components_widget(current_value)
        elif param_name == "group_by":
            return self._create_group_by_widget(current_value)
        elif self._is_string_like_type(actual_type):
            return self._create_string_widget(param_name, param_info)
        else:
            return self._create_fallback_widget(param_name, current_value, actual_type)
    
    def _create_bool_widget(self, param_name: str, current_value: Any) -> Checkbox:
        """Create checkbox widget for boolean parameters."""
        widget = Checkbox(checked=bool(current_value))
        original_handler = widget.control.mouse_handler
        
        def new_mouse_handler(mouse_event):
            result = original_handler(mouse_event)
            self.on_change(param_name, widget.checked)
            return result
        
        widget.control.mouse_handler = new_mouse_handler
        return widget
    
    def _create_variable_components_widget(self, current_value: Any) -> RadioList:
        """Create radio list widget for VariableComponents parameter."""
        options = [(None, "(None)")] + [(member, member.name) for member in VariableComponents]
        initial_selection = self._find_enum_selection(current_value, VariableComponents)
        
        widget = RadioList(values=options, default=initial_selection)
        widget.handler = lambda val: self.on_change("variable_components", val)
        return widget
    
    def _create_group_by_widget(self, current_value: Any) -> RadioList:
        """Create radio list widget for GroupBy parameter."""
        options = [(None, "(None)")] + [(member, member.name) for member in GroupBy]
        initial_selection = self._find_enum_selection(current_value, GroupBy)
        
        widget = RadioList(values=options, default=initial_selection)
        widget.handler = lambda val: self.on_change("group_by", val)
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
        """Check if type is string-like."""
        return (actual_type is str or
                get_origin(actual_type) is Path or
                isinstance(actual_type, type(Path)) or
                actual_type is Any)
    
    def _create_string_widget(self, param_name: str, param_info: dict) -> TextArea:
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
            height=1
        )
        widget.buffer.on_text_changed += lambda buff: self.on_change(param_name, buff.text)
        return widget
    
    def _create_fallback_widget(self, param_name: str, current_value: Any, actual_type: type) -> TextArea:
        """Create fallback text area widget for unhandled parameter types."""
        logger.warning(f"Unhandled param type for UI: {param_name} ({actual_type}). Using TextArea.")
        
        widget = TextArea(
            text=str(current_value or ""),
            multiline=False,
            height=1
        )
        widget.buffer.on_text_changed += lambda buff: self.on_change(param_name, buff.text)
        return widget
    
    def _create_parameter_row(self, param_info: dict, widget: Any) -> VSplit:
        """Create a UI row for a parameter."""
        param_name = param_info['param_name']
        field_label = param_info['field_label']
        
        reset_button = Button("Reset",
            handler=lambda: self._reset_parameter(param_name, widget),
            width=8
        )
        
        # RadioList can be used directly in VSplit - no wrapping needed

        return VSplit([
            Label(f"{field_label}:", width=25),
            widget,
            Box(reset_button, width=10, padding_left=1)
        ], padding=0)
    
    def _reset_parameter(self, param_name: str, widget: Any):
        """Reset parameter to original value."""
        # Get original value from step
        original_value = getattr(self.step, param_name, None)
        
        # Update widget
        if isinstance(widget, TextArea):
            widget.text = str(original_value or "")
        elif isinstance(widget, Checkbox):
            widget.checked = bool(original_value)
        elif isinstance(widget, RadioList):
            enum_class = self._get_enum_class_for_parameter(param_name)
            if enum_class:
                initial_selection = self._find_enum_selection(original_value, enum_class)
                widget.current_value = initial_selection
        
        # Notify change
        self.on_change(param_name, original_value)
        get_app().invalidate()
    
    def _get_enum_class_for_parameter(self, param_name: str) -> Optional[type]:
        """Get the enum class for a parameter name."""
        if param_name == "variable_components":
            return VariableComponents
        elif param_name == "group_by":
            return GroupBy
        return None
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current values from all widgets."""
        values = {}
        for param_name, widget in self.param_inputs.items():
            if isinstance(widget, TextArea):
                values[param_name] = widget.text
            elif isinstance(widget, Checkbox):
                values[param_name] = widget.checked
            elif isinstance(widget, RadioList):
                values[param_name] = widget.current_value
        return values
