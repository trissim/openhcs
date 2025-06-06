"""
Parameter Editor component for the OpenHCS TUI.

This module provides a component for editing function parameters.
"""
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding.key_bindings import KeyBindingsBase
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.widgets import Box, Button, Label, TextArea

logger = logging.getLogger(__name__)

# SafeButton eliminated - use Button directly


class ParameterEditor(Container):
    """
    A component for editing function parameters.

    This component displays a form for editing the parameters of a function,
    with input fields for each parameter and buttons to reset parameters.
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        current_kwargs: Optional[Dict[str, Any]] = None,
        on_parameter_change: Optional[Callable[[str, str, int], Any]] = None,
        on_reset_parameter: Optional[Callable[[str, int], Any]] = None,
        on_reset_all_parameters: Optional[Callable[[int], Any]] = None,
        func_index: int = 0
    ):
        """
        Initialize the parameter editor.

        Args:
            func: The function whose parameters to edit
            current_kwargs: Current parameter values
            on_parameter_change: Callback when a parameter value changes
            on_reset_parameter: Callback when a parameter is reset
            on_reset_all_parameters: Callback when all parameters are reset
            func_index: Index of the function in a list (for callbacks)
        """
        self.func = func
        self.current_kwargs = current_kwargs or {}
        self.on_parameter_change = on_parameter_change
        self.on_reset_parameter = on_reset_parameter
        self.on_reset_all_parameters = on_reset_all_parameters
        self.func_index = func_index

        # Create parameter editors
        self.parameter_editors = {}
        self.parameter_containers = []

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""
        # Reset all parameters button
        reset_all_button = Button("Reset All Parameters",
            handler=lambda: self._handle_reset_all(),
            width=len("Reset All Parameters") + 2
        )

        # Create parameter editors
        self._create_parameter_editors()

        # Create container
        if self.func:
            self.container = HSplit([
                Box(reset_all_button),
                HSplit(self.parameter_containers)
            ])
        else:
            self.container = HSplit([
                Label("No function selected")
            ])

    def _create_parameter_editors(self):
        """Create editors for each parameter of the function."""
        if not self.func:
            return

        # Get function signature
        sig = inspect.signature(self.func)

        # Create editors for each parameter
        for name, param in sig.parameters.items():
            # Skip self and cls parameters
            if name in ('self', 'cls'):
                continue

            # Get default value
            default_value = param.default if param.default is not inspect.Parameter.empty else None

            # Get current value
            current_value = self.current_kwargs.get(name, default_value)

            # Create editor
            self._create_parameter_editor(name, current_value, default_value)

    def _create_parameter_editor(self, name: str, current_value: Any, default_value: Any):
        """Create an editor for a single parameter."""
        # Convert current value to string for display
        current_value_str = str(current_value) if current_value is not None else ""

        # Create text area for editing
        text_area = TextArea(
            text=current_value_str,
            multiline=False,
            height=1,
            width=Dimension(preferred=30)
        )

        # Set accept handler
        def accept_handler(buffer):
            if self.on_parameter_change:
                from openhcs.tui.utils.unified_task_manager import get_task_manager
                get_task_manager().fire_and_forget(
                    self.on_parameter_change(name, buffer.text, self.func_index), f"param_change_{name}"
                )
            return True

        text_area.buffer.accept_handler = accept_handler

        # Create reset button
        reset_button = Button("Reset",
            handler=lambda: self._handle_reset_parameter(name),
            width=len("Reset") + 2
        )

        # Create container
        param_container = VSplit([
            Label(f"{name}: ", width=15),
            text_area,
            Box(reset_button, width=8)
        ])

        # Store editor
        self.parameter_editors[name] = text_area
        self.parameter_containers.append(param_container)

    def _handle_reset_parameter(self, name: str):
        """Handle reset parameter button click."""
        if self.on_reset_parameter:
            from openhcs.tui.utils.unified_task_manager import get_task_manager
            get_task_manager().fire_and_forget(
                self.on_reset_parameter(name, self.func_index), f"reset_param_{name}"
            )

    def _handle_reset_all(self):
        """Handle reset all parameters button click."""
        if self.on_reset_all_parameters:
            from openhcs.tui.utils.unified_task_manager import get_task_manager
            get_task_manager().fire_and_forget(
                self.on_reset_all_parameters(self.func_index), f"reset_all_{self.func_index}"
            )

    def update_function(self, func: Callable, kwargs: Dict[str, Any]):
        """Update the function and parameters being edited."""
        self.func = func
        self.current_kwargs = kwargs

        # Rebuild UI
        self.parameter_editors = {}
        self.parameter_containers = []
        self._build_ui()

        # Invalidate UI
        get_app().invalidate()

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self.container

    # Implement abstract methods by delegating to the container
    def get_children(self):
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        # Container should return Dimension, not int
        return Dimension(min=40, preferred=max(40, max_available_width - 10))

    def preferred_height(self, width, max_available_height):
        # Delegate to the container but ensure a minimum height
        container_height = self.container.preferred_height(width, max_available_height)

        # If the container has a valid preferred height, use it
        if container_height.preferred is not None and container_height.preferred > 0:
            return container_height

        # Otherwise, use a reasonable default based on number of parameters
        # At least 3 lines, or more if we have many parameters
        param_count = len(self.parameter_editors) if hasattr(self, 'parameter_editors') else 0
        min_height = min(max(3, param_count + 2), max_available_height)
        return Dimension(min=min_height, preferred=min_height)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                      parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events."""
        return self.container.mouse_handler(mouse_event)

    def is_modal(self):
        """Return whether this container is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings for this container."""
        return None
