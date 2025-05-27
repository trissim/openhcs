"""
Step Settings Editor for Hybrid TUI.

Ported from TUI2's placeholder implementation and TUI's parameter editor patterns.
Uses static analysis to replace schema dependencies and provides dynamic form
generation for AbstractStep parameters.

Key Features:
- Dynamic form generation from AbstractStep introspection
- Type-aware widget creation
- Reset functionality (individual + reset all)
- Change callbacks with validation
- Schema-free operation using static analysis
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, Container, Dimension
from prompt_toolkit.widgets import Button, Label, TextArea, Checkbox, Box
from prompt_toolkit.formatted_text import HTML

from ..interfaces.component_interfaces import EditorComponentInterface
from ..utils.static_analysis import get_abstractstep_parameters, get_type_widget_mapping
from ..utils.dialogs import prompt_for_path_dialog

logger = logging.getLogger(__name__)

class StepSettingsEditor(EditorComponentInterface):
    """
    Step Settings Editor component for hybrid TUI.

    Provides editing capabilities for AbstractStep parameters using
    static analysis instead of schema dependencies.
    """

    def __init__(
        self,
        initial_step_data: Optional[Dict[str, Any]] = None,
        change_callback: Optional[Callable[[str, Any], None]] = None
    ):
        """
        Initialize the Step Settings Editor.

        Args:
            initial_step_data: Initial step data to edit
            change_callback: Callback to notify when parameters change
        """
        self.change_callback = change_callback

        # Step data state
        self.original_step_data = initial_step_data.copy() if initial_step_data else {}
        self.current_step_data = initial_step_data.copy() if initial_step_data else {}

        # UI components
        self.parameter_widgets = {}
        self.parameter_containers = []
        self._container = None

        # Initialize UI
        self._initialize_ui()

    def _initialize_ui(self):
        """Initialize UI components."""
        self._build_form()

    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        return self._container

    def update_data(self, data: Any) -> None:
        """Update component with new step data."""
        if isinstance(data, dict):
            self.current_step_data = data.copy()
            self._rebuild_form()

    def get_current_value(self) -> Dict[str, Any]:
        """Get current edited step data."""
        return self.current_step_data.copy()

    def set_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Set callback for parameter changes."""
        self.change_callback = callback

    def reset_to_original(self) -> None:
        """Reset to original step data."""
        self.current_step_data = self.original_step_data.copy()
        self._rebuild_form()
        self._notify_change('reset', None)

    def has_changes(self) -> bool:
        """Check if component has unsaved changes."""
        return self.current_step_data != self.original_step_data

    def _build_form(self):
        """Build the form using static analysis of AbstractStep parameters."""
        try:
            # Get AbstractStep parameters via introspection
            step_params = get_abstractstep_parameters()

            # Create form widgets
            form_widgets = []
            self.parameter_widgets = {}
            self.parameter_containers = []

            # Add title
            title = Label(HTML("<b>Step Settings</b>"))
            form_widgets.append(title)

            # Create parameter widgets
            for param_name, param_info in step_params.items():
                current_value = self.current_step_data.get(param_name, param_info.get('default'))
                widget = self._create_parameter_widget(param_name, param_info, current_value)
                if widget:
                    form_widgets.append(widget)
                    self.parameter_containers.append(widget)

            # Add reset all button
            reset_all_button = Button(
                "Reset All Parameters",
                handler=lambda: self._handle_reset_all()
            )
            form_widgets.append(Box(reset_all_button, padding_top=1))

            # Create main container
            self._container = HSplit(form_widgets)

        except Exception as e:
            logger.error(f"Failed to build step settings form: {e}")
            self._container = HSplit([
                Label(HTML(f"<ansired>Error building form: {e}</ansired>"))
            ])

    def _create_parameter_widget(self, param_name: str, param_info: Dict, current_value: Any) -> Optional[Container]:
        """Create widget for a single parameter based on its type."""
        try:
            param_type = param_info.get('type', str)
            param_default = param_info.get('default')
            param_required = param_info.get('required', False)

            # Create label with required indicator
            label_text = f"{param_name}{'*' if param_required else ''}: "
            label = Label(label_text, width=20)

            # Create input widget based on type
            input_widget = self._create_input_widget(param_name, param_type, current_value)

            # Create reset button
            reset_button = Button(
                "Reset",
                handler=lambda name=param_name: self._handle_reset_parameter(name, param_default)
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
        type_str = str(param_type)

        if 'bool' in type_str.lower():
            # Boolean parameter - use checkbox
            checkbox = Checkbox(text="", checked=bool(current_value) if current_value is not None else False)

            def on_change():
                self._update_parameter(param_name, checkbox.checked)

            checkbox.on_change = on_change
            return checkbox

        elif 'path' in type_str.lower() or 'union[str, path]' in type_str.lower():
            # Path parameter - use text area with file dialog button
            text_area = TextArea(
                text=str(current_value) if current_value is not None else "",
                multiline=False,
                height=1,
                width=Dimension(preferred=40)
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

        elif 'list[str]' in type_str.lower():
            # List of strings - use text area with comma separation
            list_value = current_value if isinstance(current_value, list) else []
            text_area = TextArea(
                text=", ".join(list_value) if list_value else "",
                multiline=False,
                height=1,
                width=Dimension(preferred=40)
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
                width=Dimension(preferred=40)
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

        type_str = str(param_type).lower()

        try:
            if 'int' in type_str:
                return int(value_str)
            elif 'float' in type_str:
                return float(value_str)
            elif 'bool' in type_str:
                return value_str.lower() in ('true', '1', 'yes', 'on')
            elif 'path' in type_str:
                return Path(value_str)
            else:
                return value_str
        except (ValueError, TypeError):
            # Return as string if conversion fails
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
            self.current_step_data[param_name] = new_value
            self._notify_change(param_name, new_value)
        except Exception as e:
            logger.error(f"Failed to update parameter {param_name}: {e}")

    def _handle_reset_parameter(self, param_name: str, default_value: Any):
        """Reset parameter to default value."""
        try:
            if default_value is not None:
                self.current_step_data[param_name] = default_value
            elif param_name in self.current_step_data:
                del self.current_step_data[param_name]

            # Update widget display
            if param_name in self.parameter_widgets:
                widget = self.parameter_widgets[param_name]
                if hasattr(widget, 'text'):
                    widget.text = str(default_value) if default_value is not None else ""
                elif hasattr(widget, 'checked'):
                    widget.checked = bool(default_value) if default_value is not None else False

            self._notify_change(param_name, default_value)

        except Exception as e:
            logger.error(f"Failed to reset parameter {param_name}: {e}")

    def _handle_reset_all(self):
        """Reset all parameters to default values."""
        try:
            step_params = get_abstractstep_parameters()

            for param_name, param_info in step_params.items():
                default_value = param_info.get('default')
                self._handle_reset_parameter(param_name, default_value)

            self._notify_change('reset_all', None)

        except Exception as e:
            logger.error(f"Failed to reset all parameters: {e}")

    def _rebuild_form(self):
        """Rebuild the form with current data."""
        self._build_form()
        get_app().invalidate()

    def _notify_change(self, param_name: str, new_value: Any):
        """Notify parent component of parameter change."""
        if self.change_callback:
            try:
                self.change_callback(param_name, new_value)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")
