"""Function list editor widget - port of function_list_manager.py to Textual."""

from typing import List, Union, Dict, Any
from textual.containers import ScrollableContainer, Container, Horizontal
from textual.widgets import Button, Static
from textual.app import ComposeResult
from textual.reactive import reactive

from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.widgets.function_pane import FunctionPaneWidget


class FunctionListEditorWidget(ScrollableContainer):
    """
    Scrollable function list editor.

    Ports the function display and management logic from function_list_manager.py
    """

    # Reactive properties for automatic UI updates
    functions = reactive(list)

    def __init__(self, initial_functions: Union[List, Dict, None] = None):
        super().__init__()

        # Initialize services (reuse existing business logic)
        self.registry_service = FunctionRegistryService()
        self.data_manager = PatternDataManager()

        # Initialize function list
        if initial_functions is None:
            self.functions = []
        elif isinstance(initial_functions, (list, dict)):
            self.functions = self._extract_functions_from_pattern(initial_functions)
        else:
            self.functions = []

    def compose(self) -> ComposeResult:
        """Compose the function list."""
        # Header with add button
        with Container(id="function_list_header"):
            with Horizontal():
                yield Static("[bold]Functions[/bold]")
                yield Button("Add Function", id="add_function_btn", compact=True)

        # Function panes (scrolling works automatically!)
        for i, func_item in enumerate(self.functions):
            yield FunctionPaneWidget(func_item, i)

        # Empty state
        if not self.functions:
            yield Static("No functions defined. Click 'Add Function' to add functions.")

    def watch_functions(self, functions: List) -> None:
        """Automatically update UI when functions change."""
        # Textual handles this automatically - just refresh compose
        self.refresh()

    def _extract_functions_from_pattern(self, pattern: Union[List, Dict]) -> List:
        """Extract function list from pattern using existing business logic."""
        if isinstance(pattern, list):
            return pattern
        elif isinstance(pattern, dict) and pattern:
            # Get first key's functions
            first_key = next(iter(pattern))
            return pattern[first_key]
        else:
            return []

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add_function_btn":
            self._add_function()

    def _add_function(self) -> None:
        """Add new function using existing service logic."""
        # Get default function from registry
        default_func = self.registry_service.find_default_function()
        if default_func:
            new_func_item = (default_func, {})
            self.functions = self.functions + [new_func_item]
