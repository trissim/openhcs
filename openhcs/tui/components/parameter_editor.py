"""
Parameter Editor Component for Hybrid TUI.

Ported from TUI's parameter_editor.py with adaptations for:
- Component interface compliance
- Schema-free operation using static analysis
- Enhanced type handling and validation

This provides dynamic parameter form generation for any callable
using function signature introspection.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, Dimension
from prompt_toolkit.widgets import Button, Label, TextArea, Checkbox, Box
from prompt_toolkit.formatted_text import HTML

from ..interfaces.component_interfaces import ComponentInterface
from ..utils.static_analysis import get_function_signature
from ..utils.dialogs import prompt_for_path_dialog

logger = logging.getLogger(__name__)

class ParameterEditor(ComponentInterface):
    """
    A component for editing function parameters.

    This component displays a form for editing the parameters of a function,
    with input fields for each parameter and buttons to reset parameters.
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        current_kwargs: Optional[Dict[str, Any]] = None,
        on_parameter_change: Optional[Callable[[str, Any], None]] = None,
        on_reset_parameter: Optional[Callable[[str], None]] = None,
        on_reset_all_parameters: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the parameter editor.

        Args:
            func: The function whose parameters to edit
            current_kwargs: Current parameter values
            on_parameter_change: Callback when a parameter value changes
            on_reset_parameter: Callback when a parameter is reset
            on_reset_all_parameters: Callback when all parameters are reset
        """
        self.func = func
        self.current_kwargs = current_kwargs or {}
        self.on_parameter_change = on_parameter_change
        self.on_reset_parameter = on_reset_parameter
        self.on_reset_all_parameters = on_reset_all_parameters

        # Create parameter editors
        self.parameter_widgets = {}
        self.parameter_containers = []
        self._container = None

        # Build UI
        self._build_ui()

    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        return self._container

    def update_data(self, data: Any) -> None:
        """Update component with new function and kwargs."""
        if isinstance(data, tuple) and len(data) == 2:
            func, kwargs = data
            self.update_function(func, kwargs)
        elif callable(data):
            self.update_function(data, {})

    def update_function(self, func: Callable, kwargs: Dict[str, Any]):
        """Update the function and parameters being edited."""
        self.func = func
        self.current_kwargs = kwargs

        # Rebuild UI
        self.parameter_widgets = {}
        self.parameter_containers = []
        self._build_ui()

        # Invalidate UI
        get_app().invalidate()

    def _build_ui(self):
        """Build the UI components."""
        if not self.func:
            self._container = HSplit([Label("No function selected")])
            return

        # Get function signature
        try:
            sig_info = get_function_signature(self.func)
            parameters = sig_info.get('parameters', {})
        except Exception as e:
            logger.error(f"Failed to get function signature: {e}")
            self._container = HSplit([Label(f"Error: {e}")])
            return

        # Create parameter editors
        form_widgets = []

        # Function name label
        func_name = getattr(self.func, '__name__', str(self.func))
        title = Label(HTML(f"<b>Parameters for {func_name}</b>"))
        form_widgets.append(title)

        # Create widgets for each parameter
        for param_name, param_info in parameters.items():
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue

            widget = self._create_parameter_widget(param_name, param_info)
            if widget:
                form_widgets.append(widget)
                self.parameter_containers.append(widget)

        # Reset all parameters button
        if self.parameter_containers:
            reset_all_button = Button(
                "Reset All Parameters",
                handler=lambda: self._handle_reset_all()
            )
            form_widgets.append(Box(reset_all_button, padding_top=1))

        # Create container
        self._container = HSplit(form_widgets)

    def _create_parameter_widget(self, param_name: str, param_info: Dict) -> Optional[Container]:
        """Create widget for a single parameter."""
        try:
            param_type = param_info.get('type')
            param_default = param_info.get('default')
            param_required = param_info.get('required', False)

            # Get current value
            current_value = self.current_kwargs.get(param_name, param_default)

            # Create label with required indicator
            label_text = f"{param_name}{'*' if param_required else ''}: "
            label = Label(label_text, width=20)

            # Create input widget based on type
            input_widget = self._create_input_widget(param_name, param_type, current_value)

            # Create reset button
            reset_button = Button(
                "Reset",
                handler=lambda name=param_name: self._handle_reset_parameter(name)
            )

            # Store widget reference
            self.parameter_widgets[param_name] = input_widget

            # Combine components
            return VSplit([
                label,
                input_widget,
                Box(reset_button, width=8)
            ])

        except Exception as e:
            logger.error(f"Failed to create widget for parameter {param_name}: {e}")
            return None

    def _create_input_widget(self, param_name: str, param_type: Any, current_value: Any):
        """Create appropriate input widget based on parameter type."""
        type_str = str(param_type).lower() if param_type else 'str'

        if 'bool' in type_str:
            # Boolean parameter - use checkbox
            checkbox = Checkbox(text="", checked=bool(current_value) if current_value is not None else False)

            def on_change():
                self._update_parameter(param_name, checkbox.checked)

            checkbox.on_change = on_change
            return checkbox

        elif 'path' in type_str or 'union[str, path]' in type_str:
            # Path parameter - use text area with file dialog button
            text_area = TextArea(
                text=str(current_value) if current_value is not None else "",
                multiline=False,
                height=1,
                width=Dimension(preferred=30)
            )

            def on_text_change():
                self._update_parameter(param_name, text_area.text.strip())

            text_area.buffer.on_text_changed += lambda: on_text_change()

            # Browse button for path selection
            browse_button = Button(
                "Browse",
                handler=lambda: get_app().create_background_task(self._browse_for_path(param_name, text_area))
            )

            return VSplit([text_area, Box(browse_button, width=10)])

        elif 'list' in type_str:
            # List parameter - use text area with comma separation
            if isinstance(current_value, list):
                display_value = ", ".join(str(item) for item in current_value)
            else:
                display_value = str(current_value) if current_value is not None else ""

            text_area = TextArea(
                text=display_value,
                multiline=False,
                height=1,
                width=Dimension(preferred=30)
            )

            def on_text_change():
                text = text_area.text.strip()
                if text:
                    value_list = [item.strip() for item in text.split(',') if item.strip()]
                else:
                    value_list = []
                self._update_parameter(param_name, value_list)

            text_area.buffer.on_text_changed += lambda: on_text_change()
            return text_area

        else:
            # Default to text area for strings, numbers, etc.
            text_area = TextArea(
                text=str(current_value) if current_value is not None else "",
                multiline=False,
                height=1,
                width=Dimension(preferred=30)
            )

            def on_text_change():
                text = text_area.text.strip()
                converted_value = self._convert_value(text, param_type)
                self._update_parameter(param_name, converted_value)

            text_area.buffer.on_text_changed += lambda: on_text_change()
            return text_area

    def _convert_value(self, value_str: str, param_type: Any) -> Any:
        """Convert string value to appropriate type."""
        if not value_str:
            return None

        # Handle special string values
        if value_str.lower() == 'none':
            return None
        elif value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False

        # Try type conversion based on parameter type
        type_str = str(param_type).lower() if param_type else ''

        try:
            if 'int' in type_str:
                return int(value_str)
            elif 'float' in type_str:
                return float(value_str)
            elif 'path' in type_str:
                return Path(value_str)
            else:
                # Try automatic conversion
                try:
                    return int(value_str)
                except ValueError:
                    try:
                        return float(value_str)
                    except ValueError:
                        return value_str
        except (ValueError, TypeError):
            return value_str

    async def _browse_for_path(self, param_name: str, text_area: TextArea):
        """Open file browser for path selection."""
        try:
            current_path = Path(text_area.text) if text_area.text else None

            selected_path = await prompt_for_path_dialog(
                title=f"Select {param_name}",
                initial_path=current_path,
                save_mode=False
            )

            if selected_path:
                text_area.text = str(selected_path)
                self._update_parameter(param_name, selected_path)

        except Exception as e:
            logger.error(f"Failed to browse for path: {e}")

    def _update_parameter(self, param_name: str, new_value: Any):
        """Update parameter value and notify change."""
        try:
            self.current_kwargs[param_name] = new_value
            if self.on_parameter_change:
                self.on_parameter_change(param_name, new_value)
        except Exception as e:
            logger.error(f"Failed to update parameter {param_name}: {e}")

    def _handle_reset_parameter(self, param_name: str):
        """Handle reset parameter button click."""
        try:
            # Get default value from function signature
            if self.func:
                sig = inspect.signature(self.func)
                if param_name in sig.parameters:
                    param = sig.parameters[param_name]
                    default_value = param.default if param.default != inspect.Parameter.empty else None

                    if default_value is not None:
                        self.current_kwargs[param_name] = default_value
                    elif param_name in self.current_kwargs:
                        del self.current_kwargs[param_name]

                    # Update widget display
                    if param_name in self.parameter_widgets:
                        widget = self.parameter_widgets[param_name]
                        if hasattr(widget, 'text'):
                            widget.text = str(default_value) if default_value is not None else ""
                        elif hasattr(widget, 'checked'):
                            widget.checked = bool(default_value) if default_value is not None else False

                    if self.on_reset_parameter:
                        self.on_reset_parameter(param_name)

        except Exception as e:
            logger.error(f"Failed to reset parameter {param_name}: {e}")

    def _handle_reset_all(self):
        """Handle reset all parameters button click."""
        try:
            if self.func:
                sig = inspect.signature(self.func)
                for param_name in list(self.current_kwargs.keys()):
                    if param_name in sig.parameters:
                        self._handle_reset_parameter(param_name)

                if self.on_reset_all_parameters:
                    self.on_reset_all_parameters()

        except Exception as e:
            logger.error(f"Failed to reset all parameters: {e}")
