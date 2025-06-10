"""Function list editor widget - port of function_list_manager.py to Textual."""

import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Callable # Added Optional, Callable
from textual.containers import ScrollableContainer, Container, Horizontal, Center
from textual.widgets import Button, Static, Select
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
    pattern_data = reactive(list, recompose=True)  # The actual pattern (List or Dict)
    is_dict_mode = reactive(False, recompose=True)  # Whether we're in channel-specific mode
    selected_channel = reactive(None, recompose=True)  # Currently selected channel (for dict mode)
    available_channels = reactive(list)  # Available channels from orchestrator

    def __init__(self, initial_functions: Union[List, Dict, callable, None] = None):
        super().__init__()

        # Initialize services (reuse existing business logic)
        self.registry_service = FunctionRegistryService()
        self.data_manager = PatternDataManager() # Not heavily used yet, but available

        # Initialize pattern data and mode
        self._initialize_pattern_data(initial_functions)

        logger.debug(f"FunctionListEditorWidget initialized with {len(self.functions)} functions, dict_mode={self.is_dict_mode}")

    @property
    def current_pattern(self) -> Union[List, Dict]:
        """Get the current pattern data (for parent widgets to access)."""
        self._update_pattern_data()  # Ensure it's up to date
        return self.pattern_data

    def watch_functions(self, new_functions: List) -> None:
        """Watch for changes to functions and update pattern data."""
        self._update_pattern_data()

    def _initialize_pattern_data(self, initial_functions: Union[List, Dict, callable, None]) -> None:
        """Initialize pattern data and determine mode."""
        if initial_functions is None:
            self.pattern_data = []
            self.is_dict_mode = False
            self.functions = []
        elif callable(initial_functions):
            self.pattern_data = [(initial_functions, {})]
            self.is_dict_mode = False
            self.functions = [(initial_functions, {})]
        elif isinstance(initial_functions, list):
            self.pattern_data = initial_functions
            self.is_dict_mode = False
            self.functions = self._normalize_function_list(initial_functions)
        elif isinstance(initial_functions, dict):
            self.pattern_data = initial_functions
            self.is_dict_mode = True
            # Set first channel as selected, or empty if no channels
            if initial_functions:
                first_channel = next(iter(initial_functions))
                self.selected_channel = first_channel
                self.functions = self._normalize_function_list(initial_functions.get(first_channel, []))
            else:
                self.selected_channel = None
                self.functions = []
        else:
            logger.warning(f"Unknown initial_functions type: {type(initial_functions)}, using empty list")
            self.pattern_data = []
            self.is_dict_mode = False
            self.functions = []

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

    def _update_pattern_data(self) -> None:
        """Update pattern_data based on current functions and mode."""
        if self.is_dict_mode and self.selected_channel is not None:
            # Save current functions to the selected channel
            if not isinstance(self.pattern_data, dict):
                self.pattern_data = {}
            self.pattern_data[self.selected_channel] = self.functions
        else:
            # List mode - pattern_data is just the functions list
            self.pattern_data = self.functions

    def _switch_to_channel(self, channel: Any) -> None:
        """Switch to editing functions for a specific channel."""
        if not self.is_dict_mode:
            return

        # Save current functions first
        self._update_pattern_data()

        # Switch to new channel
        self.selected_channel = channel
        if isinstance(self.pattern_data, dict):
            self.functions = self.pattern_data.get(channel, [])
        else:
            self.functions = []

    def _add_channel_to_pattern(self, channel: Any) -> None:
        """Add a new channel (converts to dict mode if needed)."""
        if not self.is_dict_mode:
            # Convert to dict mode
            self.pattern_data = {channel: self.functions}
            self.is_dict_mode = True
            self.selected_channel = channel
        else:
            # Add new channel with empty functions
            if not isinstance(self.pattern_data, dict):
                self.pattern_data = {}
            self.pattern_data[channel] = []
            self.selected_channel = channel
            self.functions = []

    def _remove_current_channel(self) -> None:
        """Remove the currently selected channel."""
        if not self.is_dict_mode or self.selected_channel is None:
            return

        if isinstance(self.pattern_data, dict):
            new_pattern = self.pattern_data.copy()
            if self.selected_channel in new_pattern:
                del new_pattern[self.selected_channel]

            if len(new_pattern) == 0:
                # Revert to list mode
                self.pattern_data = []
                self.is_dict_mode = False
                self.selected_channel = None
                self.functions = []
            else:
                # Switch to first remaining channel
                self.pattern_data = new_pattern
                first_channel = next(iter(new_pattern))
                self.selected_channel = first_channel
                self.functions = new_pattern[first_channel]

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

                    # Channel management button
                    channel_text = self._get_channel_button_text()
                    yield Button(channel_text, id="channel_btn", compact=True)

                    # Channel navigation buttons (only in dict mode with multiple channels)
                    if self.is_dict_mode and isinstance(self.pattern_data, dict) and len(self.pattern_data) > 1:
                        yield Button("<", id="prev_channel_btn", compact=True)
                        yield Button(">", id="next_channel_btn", compact=True)
                
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
        elif event.button.id == "channel_btn":
            self._show_channel_selection_dialog()
        elif event.button.id == "prev_channel_btn":
            self._navigate_channel(-1)
        elif event.button.id == "next_channel_btn":
            self._navigate_channel(1)



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
        # Update pattern data before notifying
        self._update_pattern_data()
        # Post message to notify parent (DualEditorScreen) of changes
        self.post_message(self.FunctionPatternChanged())

    def _load_func(self) -> None:
        """Load function pattern from .func file."""
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend

        def handle_result(result):
            if result and isinstance(result, Path):
                self._load_pattern_from_file(result)

        # Launch enhanced file browser for .func files
        browser = EnhancedFileBrowserScreen(
            file_manager=self.app.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Load Function Pattern (.func)",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func']
        )
        self.app.push_screen(browser, handle_result)

    def _save_func_as(self) -> None:
        """Save function pattern to .func file."""
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend

        def handle_result(result):
            if result and isinstance(result, Path):
                self._save_pattern_to_file(result)

        # Launch enhanced file browser for saving .func files
        browser = EnhancedFileBrowserScreen(
            file_manager=self.app.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Save Function Pattern (.func)",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func'],
            default_filename="pattern.func"
        )
        self.app.push_screen(browser, handle_result)

    def _load_pattern_from_file(self, file_path: Path) -> None:
        """Load pattern from .func file."""
        import pickle
        try:
            with open(file_path, 'rb') as f:
                pattern = pickle.load(f)
            self._initialize_pattern_data(pattern)
            self._commit_and_notify()
        except Exception as e:
            logger.error(f"Failed to load pattern: {e}")

    def _save_pattern_to_file(self, file_path: Path) -> None:
        """Save pattern to .func file."""
        import pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.current_pattern, f)
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")

    def _edit_in_vim(self) -> None:
        """Edit function pattern in Vim."""
        # TODO: Implement Vim editing functionality
        logger.debug("Edit in Vim button pressed - not implemented yet")

    def _get_channel_button_text(self) -> str:
        """Get text for the channel button."""
        if self.is_dict_mode and self.selected_channel is not None:
            return f"Channel: {self.selected_channel}"
        return "Channel: None"

    def _show_channel_selection_dialog(self) -> None:
        """Show the channel selection dialog."""
        try:
            # Get available channels from orchestrator
            orchestrator = self._get_current_orchestrator()
            if orchestrator is None:
                logger.warning("No orchestrator available for channel selection")
                return

            available_channels = orchestrator.get_channels()
            if not available_channels:
                logger.warning("No channels found in current plate")
                return

            # Get currently selected channels
            if self.is_dict_mode and isinstance(self.pattern_data, dict):
                selected_channels = list(self.pattern_data.keys())
            else:
                selected_channels = []

            # Show dialog
            from openhcs.textual_tui.screens.channel_selection_dialog import ChannelSelectionDialog

            def handle_selection(result_channels):
                if result_channels is not None:
                    self._update_channels(result_channels)

            dialog = ChannelSelectionDialog(
                available_channels=available_channels,
                selected_channels=selected_channels,
                callback=handle_selection
            )
            self.app.push_screen(dialog)

        except Exception as e:
            logger.error(f"Failed to show channel selection dialog: {e}")

    def _update_channels(self, new_channels: List[int]) -> None:
        """Update the pattern based on new channel selection."""
        if not new_channels:
            # No channels selected - revert to list mode
            if self.is_dict_mode:
                # Save current functions to list mode
                self.pattern_data = self.functions
                self.is_dict_mode = False
                self.selected_channel = None
                logger.debug("Reverted to list mode (no channels selected)")
        else:
            # Channels selected - ensure dict mode
            if not self.is_dict_mode:
                # Convert to dict mode
                current_functions = self.functions
                self.pattern_data = {new_channels[0]: current_functions}
                self.is_dict_mode = True
                self.selected_channel = new_channels[0]

                # Add other channels with empty functions
                for channel in new_channels[1:]:
                    self.pattern_data[channel] = []
            else:
                # Already in dict mode - update channels
                old_pattern = self.pattern_data.copy() if isinstance(self.pattern_data, dict) else {}
                new_pattern = {}

                # Keep existing functions for channels that remain
                for channel in new_channels:
                    if channel in old_pattern:
                        new_pattern[channel] = old_pattern[channel]
                    else:
                        new_pattern[channel] = []

                self.pattern_data = new_pattern

                # Update selected channel if needed
                if self.selected_channel not in new_channels:
                    self.selected_channel = new_channels[0] if new_channels else None
                    if self.selected_channel:
                        self.functions = new_pattern.get(self.selected_channel, [])

        self._commit_and_notify()
        logger.debug(f"Updated channels: {new_channels}")

    def _navigate_channel(self, direction: int) -> None:
        """Navigate to next/previous channel (with looping)."""
        if not self.is_dict_mode or not isinstance(self.pattern_data, dict):
            return

        channels = sorted(self.pattern_data.keys())
        if len(channels) <= 1:
            return

        try:
            current_index = channels.index(self.selected_channel)
            new_index = (current_index + direction) % len(channels)
            new_channel = channels[new_index]

            self._switch_to_channel(new_channel)
            logger.debug(f"Navigated to channel {new_channel}")
        except (ValueError, IndexError):
            logger.warning(f"Failed to navigate channels: current={self.selected_channel}, channels={channels}")



    def _get_current_orchestrator(self):
        """Get the current orchestrator from the app."""
        try:
            # Get from app
            if hasattr(self.app, 'query_one'):
                from openhcs.textual_tui.widgets.main_content import MainContent
                from openhcs.textual_tui.widgets.plate_manager import PlateManagerWidget

                main_content = self.app.query_one(MainContent)
                plate_manager = main_content.query_one(PlateManagerWidget)

                # Use selected_plate (not current_plate!)
                selected_plate = plate_manager.selected_plate
                if selected_plate and selected_plate in plate_manager.orchestrators:
                    orchestrator = plate_manager.orchestrators[selected_plate]
                    if not orchestrator.is_initialized():
                        orchestrator.initialize()
                    return orchestrator
            return None
        except Exception as e:
            logger.error(f"Failed to get orchestrator: {e}")
            return None

