"""Individual function pane widget with parameter editing."""

from typing import Tuple, Any, Callable, Dict
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static
from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive

from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from .shared.parameter_form_manager import ParameterFormManager
from .shared.signature_analyzer import SignatureAnalyzer


class FunctionPaneWidget(Container):
    """
    Individual function pane with parameter editing capabilities.

    Displays function name, editable parameters, and control buttons.
    """

    # Reactive properties for automatic UI updates
    function_callable = reactive(None, recompose=False)
    kwargs = reactive(dict, recompose=False)  # Prevent recomposition during parameter changes
    show_parameters = reactive(True, recompose=False)

    def __init__(self, func_item: Tuple[Callable, dict], index: int):
        super().__init__()
        self.func_item = func_item
        self.index = index

        # Extract function and kwargs using existing business logic
        self.func, self.kwargs = PatternDataManager.extract_func_and_kwargs(func_item)

        # Set reactive properties
        self.function_callable = self.func
        self.kwargs = self.kwargs or {}

        # Create parameter form manager using shared components
        if self.func:
            param_info = SignatureAnalyzer.analyze(self.func)
            parameters = {name: self.kwargs.get(name, info.default_value) for name, info in param_info.items()}
            parameter_types = {name: info.param_type for name, info in param_info.items()}

            self.form_manager = ParameterFormManager(parameters, parameter_types, f"func_{index}")
            self.param_defaults = {name: info.default_value for name, info in param_info.items()}
        else:
            self.form_manager = None
            self.param_defaults = {}

    def compose(self) -> ComposeResult:
        """Compose the function pane with parameter editing."""
        func_name = getattr(self.func, '__name__', 'Unknown Function')
        module_name = getattr(self.func, '__module__', '').split('.')[-1] if self.func else ''

        # Function header with module name
        title = f"{self.index + 1}: {func_name}"
        if module_name:
            title += f" ({module_name})"
        yield Static(f"[bold]{title}[/bold]")

       # # Control buttons row with Reset All Parameters on the right
       # with Horizontal() as button_row:
       #     button_row.styles.height = "auto"  # Only take height needed for buttons
       #     # Left side: movement and edit buttons
       #     yield Button("↑", id=f"move_up_{self.index}", compact=True)
       #     yield Button("↓", id=f"move_down_{self.index}", compact=True)
       #     yield Button("Add", id=f"add_func_{self.index}", compact=True)
       #     yield Button("Delete", id=f"remove_func_{self.index}", compact=True)

       #     # Right side: Reset All Parameters button
       #     yield Static("", classes="spacer")  # Spacer to push button right
       #     yield Button("Reset All Parameters", id=f"reset_all_{self.index}", compact=True)

        with Horizontal() as button_row:
            # Empty space (flex-grows)
            yield Static("")

            # Centered action button group
            with Horizontal() as action_group:
                action_group.styles.width = "auto"

                # Create buttons with explicit width: auto styling
                up_btn = Button("↑", id=f"move_up_{self.index}", compact=True)
                up_btn.styles.width = "auto"
                yield up_btn

                down_btn = Button("↓", id=f"move_down_{self.index}", compact=True)
                down_btn.styles.width = "auto"
                yield down_btn

                add_btn = Button("Add", id=f"add_func_{self.index}", compact=True)
                add_btn.styles.width = "auto"
                yield add_btn

                delete_btn = Button("Delete", id=f"remove_func_{self.index}", compact=True)
                delete_btn.styles.width = "auto"
                yield delete_btn

                reset_btn = Button("Reset All Parameters", id=f"reset_all_{self.index}", compact=True)
                reset_btn.styles.width = "auto"
                yield reset_btn

            # Empty space (flex-grows)  
            yield Static("")

            # Reset button (or remove if not needed)

        # Parameter form (if function exists and parameters shown)
        if self.func and self.show_parameters and self.form_manager:
            yield from self._build_parameter_form()

    def _build_parameter_form(self) -> ComposeResult:
        """Generate form widgets using shared ParameterFormManager."""
        if not self.form_manager:
            return

        try:
            # Use shared form manager to build form
            yield from self.form_manager.build_form()
        except Exception as e:
            yield Static(f"[red]Error building parameter form: {e}[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle function pane button presses."""
        button_id = event.button.id

        if button_id.startswith("add_func_"):
            self._add_function()
        elif button_id.startswith("remove_func_") or button_id.startswith("delete_"):
            self._remove_function()
        elif button_id.startswith("move_up_"):
            self._move_function(-1)
        elif button_id.startswith("move_down_"):
            self._move_function(1)
        elif button_id.startswith("reset_all_"):
            self._reset_all_parameters()
        elif button_id.startswith(f"reset_func_{self.index}_"):
            # Individual parameter reset
            param_name = button_id.split("_", 3)[3]
            self._reset_parameter(param_name)

    def on_input_changed(self, event) -> None:
        """Handle input changes from shared components."""
        if event.input.id.startswith(f"func_{self.index}_"):
            param_name = event.input.id.split("_", 2)[2]
            if self.form_manager:
                self.form_manager.update_parameter(param_name, event.value)
                final_value = self.form_manager.parameters[param_name]
                self._handle_parameter_change(param_name, final_value)

    def on_checkbox_changed(self, event) -> None:
        """Handle checkbox changes from shared components."""
        if event.checkbox.id.startswith(f"func_{self.index}_"):
            param_name = event.checkbox.id.split("_", 2)[2]
            if self.form_manager:
                self.form_manager.update_parameter(param_name, event.value)
                final_value = self.form_manager.parameters[param_name]
                self._handle_parameter_change(param_name, final_value)

    def on_radio_set_changed(self, event) -> None:
        """Handle RadioSet changes from shared components."""
        if event.radio_set.id.startswith(f"func_{self.index}_"):
            param_name = event.radio_set.id.split("_", 2)[2]
            if event.pressed and event.pressed.id:
                enum_value = event.pressed.id[5:]  # Remove "enum_" prefix
                if self.form_manager:
                    self.form_manager.update_parameter(param_name, enum_value)
                    final_value = self.form_manager.parameters[param_name]
                    self._handle_parameter_change(param_name, final_value)

    def _change_function(self) -> None:
        """Change the function (launch function selector)."""
        # Post message to parent widget to launch function selector
        self.post_message(self.ChangeFunction(self.index))

    def _handle_parameter_change(self, param_name: str, value: Any) -> None:
        """Update kwargs and emit change message."""
        # Update local kwargs without triggering reactive update
        # This prevents recomposition and focus loss during typing
        if not hasattr(self, '_internal_kwargs'):
            self._internal_kwargs = self.kwargs.copy()

        self._internal_kwargs[param_name] = value

        # Emit parameter changed message
        self.post_message(self.ParameterChanged(self.index, param_name, value))

    def _sync_kwargs(self) -> None:
        """Sync internal kwargs to reactive property when safe to do so."""
        if hasattr(self, '_internal_kwargs'):
            self.kwargs = self._internal_kwargs.copy()

    def get_current_kwargs(self) -> dict:
        """Get current kwargs values (from internal storage if available)."""
        if hasattr(self, '_internal_kwargs'):
            return self._internal_kwargs.copy()
        return self.kwargs.copy()

    def _remove_function(self) -> None:
        """Remove this function."""
        # Post message to parent widget
        self.post_message(self.RemoveFunction(self.index))

    def _move_function(self, direction: int) -> None:
        """Move function up or down."""
        # Post message to parent widget
        self.post_message(self.MoveFunction(self.index, direction))

    def _reset_parameter(self, param_name: str) -> None:
        """Reset a specific parameter to its default value."""
        if not self.form_manager or param_name not in self.param_defaults:
            return

        # Use form manager to reset parameter
        default_value = self.param_defaults[param_name]
        self.form_manager.reset_parameter(param_name, default_value)

        # Update local kwargs and notify parent
        self._handle_parameter_change(param_name, default_value)

    def _reset_all_parameters(self) -> None:
        """Reset all parameters to their default values."""
        if not self.form_manager:
            return

        # Use form manager to reset all parameters
        self.form_manager.reset_all_parameters(self.param_defaults)

        # Update internal kwargs and notify parent
        self._internal_kwargs = self.form_manager.get_current_values()
        self.post_message(self.ParameterChanged(self.index, 'all', self._internal_kwargs))

    # Custom messages for parent communication
    class ParameterChanged(Message):
        """Message sent when parameter value changes."""
        def __init__(self, index: int, param_name: str, value: Any):
            super().__init__()
            self.index = index
            self.param_name = param_name
            self.value = value

    class ChangeFunction(Message):
        """Message sent when function should be changed."""
        def __init__(self, index: int):
            super().__init__()
            self.index = index

    class RemoveFunction(Message):
        """Message sent when function should be removed."""
        def __init__(self, index: int):
            super().__init__()
            self.index = index

    class MoveFunction(Message):
        """Message sent when function should be moved up or down."""
        def __init__(self, index: int, direction: int):
            super().__init__()
            self.index = index
            self.direction = direction