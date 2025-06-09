"""Function list editor widget - port of function_list_manager.py to Textual."""

import logging
from typing import List, Union, Dict, Any, Optional, Callable # Added Optional, Callable
from textual.containers import ScrollableContainer, Container, Horizontal, Center
from textual.widgets import Button, Static
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.message import Message # Added Message

from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.widgets.function_pane import FunctionPaneWidget

logger = logging.getLogger(__name__)


class FunctionListEditorWidget(Container):
    """
    Scrollable function list editor.

    Ports the function display and management logic from function_list_manager.py
    """

    class FunctionPatternChanged(Message):
        """Message to indicate the function pattern has changed."""
        pass

    # Reactive properties for automatic UI updates
    functions = reactive(list, recompose=True) # This will hold List[(callable, kwargs_dict)]

    def __init__(self, initial_functions: Union[List, Dict, callable, None] = None):
        super().__init__()

        # Initialize services (reuse existing business logic)
        self.registry_service = FunctionRegistryService()
        self.data_manager = PatternDataManager() # Not heavily used yet, but available

        # Initialize function list
        if initial_functions is None:
            self.functions = []
        elif callable(initial_functions):
            self.functions = [(initial_functions, {})]
        elif isinstance(initial_functions, list):
            self.functions = self._normalize_function_list(initial_functions)
        elif isinstance(initial_functions, dict):
            # Extract from first key for Phase 1 simplicity
            if initial_functions:
                first_key = next(iter(initial_functions))
                self.functions = self._normalize_function_list(initial_functions.get(first_key, []))
            else:
                self.functions = []
        else:
            logger.warning(f"Unknown initial_functions type: {type(initial_functions)}, using empty list")
            self.functions = []
        
        logger.debug(f"FunctionListEditorWidget initialized with {len(self.functions)} functions.")

    def _normalize_function_list(self, func_list: List[Any]) -> List[tuple[Callable, Dict]]:
        """Ensures all items in a function list are (callable, kwargs) tuples."""
        normalized = []
        for item in func_list:
            if isinstance(item, tuple) and len(item) == 2 and callable(item[0]) and isinstance(item[1], dict):
                normalized.append(item)
            elif callable(item):
                normalized.append((item, {}))
            else:
                logger.warning(f"Skipping invalid item in function list: {item}")
        return normalized

    def _commit_and_notify(self) -> None:
        """Post a change message to notify parent of function pattern changes."""
        self.post_message(self.FunctionPatternChanged())
        logger.debug("Posted FunctionPatternChanged message to parent.")

    def compose(self) -> ComposeResult:
        """Compose the function list using the common interface pattern."""
        from textual.containers import Vertical

        with Vertical():
            # Fixed header with title
            yield Static("[bold]Functions[/bold]")

            # Button row - takes minimal height needed for buttons
            with Horizontal(id="function_list_header") as button_row:
                button_row.styles.height = "auto"  # CRITICAL: Take only needed height
                
                # Empty space (flex-grows)
                yield Static("")
                
                # Centered main button group
                with Horizontal() as main_button_group:
                    main_button_group.styles.width = "auto"
                    yield Button("Add Function", id="add_function_btn", compact=True)
                    yield Button("Load .func", id="load_func_btn", compact=True)
                    yield Button("Save .func As", id="save_func_as_btn", compact=True)
                    yield Button("Edit in Vim", id="edit_vim_btn", compact=True)
                
                # Empty space (flex-grows)
                yield Static("")

            # Scrollable content area - expands to fill ALL remaining vertical space
            with ScrollableContainer(id="function_list_content") as container:
                container.styles.height = "1fr"  # CRITICAL: Fill remaining space

                if not self.functions:
                    content = Static("No functions defined. Click 'Add Function' to begin.")
                    content.styles.width = "100%"
                    content.styles.height = "100%"
                    yield content
                else:
                    for i, func_item in enumerate(self.functions):
                        pane = FunctionPaneWidget(func_item, i)
                        pane.styles.width = "100%"
                        pane.styles.height = "auto"  # CRITICAL: Only take height needed for content
                        yield pane



    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add_function_btn":
            self._add_function()
        elif event.button.id == "load_func_btn":
            self._load_func()
        elif event.button.id == "save_func_as_btn":
            self._save_func_as()
        elif event.button.id == "edit_vim_btn":
            self._edit_in_vim()

    def on_function_pane_widget_parameter_changed(self, event: Message) -> None:
        """Handle parameter change message from FunctionPaneWidget."""
        if hasattr(event, 'index') and hasattr(event, 'param_name') and hasattr(event, 'value'):
            if 0 <= event.index < len(self.functions):
                # Update function kwargs
                func, kwargs = self.functions[event.index]
                new_kwargs = kwargs.copy()
                new_kwargs[event.param_name] = event.value

                # Update functions list
                new_functions = self.functions.copy()
                new_functions[event.index] = (func, new_kwargs)
                self.functions = new_functions

                self._commit_and_notify()
                logger.debug(f"Updated parameter {event.param_name}={event.value} for function {event.index}")

    def on_function_pane_widget_change_function(self, event: Message) -> None:
        """Handle change function message from FunctionPaneWidget."""
        if hasattr(event, 'index') and 0 <= event.index < len(self.functions):
            self._change_function(event.index)

    def on_function_pane_widget_remove_function(self, event: Message) -> None:
        """Handle remove function message from FunctionPaneWidget."""
        if hasattr(event, 'index') and 0 <= event.index < len(self.functions):
            new_functions = self.functions[:event.index] + self.functions[event.index+1:]
            self.functions = new_functions
            self._commit_and_notify()
            logger.debug(f"Removed function at index {event.index}")
        else:
            logger.warning(f"Invalid index for remove function: {getattr(event, 'index', 'N/A')}")

    def on_function_pane_widget_move_function(self, event: Message) -> None:
        """Handle move function message from FunctionPaneWidget."""
        if not (hasattr(event, 'index') and hasattr(event, 'direction')):
            return

        index = event.index
        direction = event.direction

        if not (0 <= index < len(self.functions)):
            return

        new_index = index + direction
        if not (0 <= new_index < len(self.functions)):
            return

        new_functions = self.functions.copy()
        new_functions[index], new_functions[new_index] = new_functions[new_index], new_functions[index]
        self.functions = new_functions
        self._commit_and_notify()
        logger.debug(f"Moved function from index {index} to {new_index}")

    def _add_function(self) -> None:
        """Add a new function to the list."""
        from openhcs.textual_tui.screens.function_selector import FunctionSelectorScreen

        def handle_function_selection(selected_function: Optional[Callable]) -> None:
            if selected_function:
                self.functions = self.functions + [(selected_function, {})]
                self._commit_and_notify()
                logger.debug(f"Added function: {selected_function.__name__}")

        self.app.push_screen(FunctionSelectorScreen(), handle_function_selection)

    def _change_function(self, index: int) -> None:
        """Change function at specified index."""
        from openhcs.textual_tui.screens.function_selector import FunctionSelectorScreen

        if 0 <= index < len(self.functions):
            current_func, _ = self.functions[index]

            def handle_function_selection(selected_function: Optional[Callable]) -> None:
                if selected_function:
                    # Replace function but keep existing kwargs structure
                    new_functions = self.functions.copy()
                    new_functions[index] = (selected_function, {})
                    self.functions = new_functions
                    self._commit_and_notify()
                    logger.debug(f"Changed function at index {index} to: {selected_function.__name__}")

            self.app.push_screen(
                FunctionSelectorScreen(current_function=current_func),
                handle_function_selection
            )

    def _commit_and_notify(self) -> None:
        """Commit changes and notify parent of function pattern change."""
        # Post message to notify parent (DualEditorScreen) of changes
        self.post_message(self.FunctionPatternChanged())

    def _load_func(self) -> None:
        """Load function pattern from .func file."""
        # TODO: Implement file loading functionality
        logger.debug("Load .func button pressed - not implemented yet")

    def _save_func_as(self) -> None:
        """Save function pattern to .func file."""
        # TODO: Implement file saving functionality
        logger.debug("Save .func As button pressed - not implemented yet")

    def _edit_in_vim(self) -> None:
        """Edit function pattern in Vim."""
        # TODO: Implement Vim editing functionality
        logger.debug("Edit in Vim button pressed - not implemented yet")

