"""Individual function pane widget."""

from typing import Tuple, Any, Callable
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.app import ComposeResult

from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager


class FunctionPaneWidget(Container):
    """
    Individual function pane display.

    Ports function display logic from existing components.
    """

    def __init__(self, func_item: Tuple[Callable, dict], index: int):
        super().__init__()
        self.func_item = func_item
        self.index = index

        # Extract function and kwargs using existing business logic
        self.func, self.kwargs = PatternDataManager.extract_func_and_kwargs(func_item)

    def compose(self) -> ComposeResult:
        """Compose the function pane."""
        func_name = getattr(self.func, '__name__', 'Unknown Function')
        param_count = len(self.kwargs)

        yield Static(f"[bold]{self.index + 1}: {func_name}[/bold]")
        yield Static(f"Parameters: {param_count} configured")

        with Horizontal():
            yield Button("Edit", id=f"edit_func_{self.index}", compact=True)
            yield Button("Remove", id=f"remove_func_{self.index}", compact=True)
            yield Button("Move Up", id=f"move_up_{self.index}", compact=True)
            yield Button("Move Down", id=f"move_down_{self.index}", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle function pane button presses."""
        button_id = event.button.id

        if button_id.startswith("edit_func_"):
            self._edit_function()
        elif button_id.startswith("remove_func_"):
            self._remove_function()
        elif button_id.startswith("move_up_"):
            self._move_function(-1)
        elif button_id.startswith("move_down_"):
            self._move_function(1)

    def _edit_function(self) -> None:
        """Edit function parameters."""
        # TODO: Implement in later sprint
        pass

    def _remove_function(self) -> None:
        """Remove this function."""
        # Post message to parent widget
        self.post_message(self.RemoveFunction(self.index))

    def _move_function(self, direction: int) -> None:
        """Move function up or down."""
        # Post message to parent widget
        self.post_message(self.MoveFunction(self.index, direction))

    # Custom messages for parent communication
    class RemoveFunction:
        def __init__(self, index: int):
            self.index = index

    class MoveFunction:
        def __init__(self, index: int, direction: int):
            self.index = index
            self.direction = direction
