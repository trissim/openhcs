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
from openhcs.constants.constants import GroupBy, VariableComponents

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
    functions = reactive(list, recompose=True)  # Structural changes (add/remove) should trigger recomposition
    pattern_data = reactive(list, recompose=False)  # The actual pattern (List or Dict)
    is_dict_mode = reactive(False, recompose=True)  # Whether we're in channel-specific mode - affects UI layout
    selected_channel = reactive(None, recompose=True)  # Currently selected channel - affects button text
    available_channels = reactive(list)  # Available channels from orchestrator

    # Step configuration reactive properties for dynamic component selection
    current_group_by = reactive(None, recompose=True)  # Current GroupBy setting from step editor
    current_variable_components = reactive(list, recompose=True)  # Current VariableComponents list from step editor

    def __init__(self, initial_functions: Union[List, Dict, callable, None] = None, step_identifier: str = None):
        super().__init__()

        # Initialize services (reuse existing business logic)
        self.registry_service = FunctionRegistryService()
        self.data_manager = PatternDataManager() # Not heavily used yet, but available

        # Step identifier for cache isolation (optional, defaults to widget instance id)
        self.step_identifier = step_identifier or f"widget_{id(self)}"

        # Component selection cache per GroupBy (instance-specific, not shared between steps)
        self.component_selections: Dict[GroupBy, List[str]] = {}

        # Initialize pattern data and mode
        self._initialize_pattern_data(initial_functions)

        logger.debug(f"FunctionListEditorWidget initialized for step '{self.step_identifier}' with {len(self.functions)} functions, dict_mode={self.is_dict_mode}")

    @property
    def current_pattern(self) -> Union[List, Dict]:
        """Get the current pattern data (for parent widgets to access)."""
        self._update_pattern_data()  # Ensure it's up to date

        # Migration fix: Convert any integer keys to string keys for compatibility
        # with pattern detection system which always uses string component values
        if isinstance(self.pattern_data, dict):
            migrated_pattern = {}
            for key, value in self.pattern_data.items():
                str_key = str(key)
                migrated_pattern[str_key] = value
            return migrated_pattern

        return self.pattern_data

    def watch_functions(self, new_functions: List) -> None:
        """Watch for changes to functions and update pattern data."""
        # Update pattern data when functions change (structural changes only)
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
            # Convert any integer keys to string keys for consistency
            normalized_dict = {}
            for key, value in initial_functions.items():
                str_key = str(key)  # Ensure all keys are strings
                normalized_dict[str_key] = value
                logger.debug(f"Converted channel key {key} ({type(key)}) to '{str_key}' (str)")

            self.pattern_data = normalized_dict
            self.is_dict_mode = True
            # Set first channel as selected, or empty if no channels
            if normalized_dict:
                first_channel = next(iter(normalized_dict))
                self.selected_channel = first_channel
                self.functions = self._normalize_function_list(normalized_dict.get(first_channel, []))
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

    def watch_current_group_by(self, old_group_by: Optional[GroupBy], new_group_by: Optional[GroupBy]) -> None:
        """Handle group_by changes by saving/loading component selections."""
        if old_group_by is not None and old_group_by != GroupBy.NONE:
            # Save current selection for old group_by
            if self.is_dict_mode and isinstance(self.pattern_data, dict):
                current_selection = list(self.pattern_data.keys())
                self.component_selections[old_group_by] = current_selection
                logger.debug(f"Step '{self.step_identifier}': Saved selection for {old_group_by.value}: {current_selection}")

        # Note: We don't automatically load selection for new_group_by here
        # because the dialog will handle loading from cache when opened
        logger.debug(f"Group by changed from {old_group_by} to {new_group_by}")

    def _update_pattern_data(self) -> None:
        """Update pattern_data based on current functions and mode."""
        if self.is_dict_mode and self.selected_channel is not None:
            # Save current functions to the selected channel
            if not isinstance(self.pattern_data, dict):
                self.pattern_data = {}
            logger.debug(f"Saving {len(self.functions)} functions to channel {self.selected_channel}")
            self.pattern_data[self.selected_channel] = self.functions.copy()  # Make a copy to avoid reference issues
        else:
            # List mode - pattern_data is just the functions list
            self.pattern_data = self.functions



    def _switch_to_channel(self, channel: Any) -> None:
        """Switch to editing functions for a specific channel."""
        if not self.is_dict_mode:
            return

        # Save current functions first
        old_channel = self.selected_channel
        logger.debug(f"Switching from channel {old_channel} to {channel}")
        logger.debug(f"Current functions before save: {len(self.functions)} functions")

        self._update_pattern_data()

        # Verify the save worked
        if old_channel and isinstance(self.pattern_data, dict):
            saved_functions = self.pattern_data.get(old_channel, [])
            logger.debug(f"Saved {len(saved_functions)} functions to channel {old_channel}")

        # Switch to new channel
        self.selected_channel = channel
        if isinstance(self.pattern_data, dict):
            self.functions = self.pattern_data.get(channel, [])
            logger.debug(f"Loaded {len(self.functions)} functions for channel {channel}")
        else:
            self.functions = []

        # Update button text to show new channel
        self._refresh_component_button()

        # Channel switch will automatically trigger recomposition via reactive system

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
                    yield Button("Add", id="add_function_btn", compact=True)
                    yield Button("Load", id="load_func_btn", compact=True)
                    yield Button("Save As", id="save_func_as_btn", compact=True)
                    yield Button("Edit", id="edit_vim_btn", compact=True)

                    # Component selection button (dynamic based on group_by setting)
                    component_text = self._get_component_button_text()
                    component_button = Button(component_text, id="component_btn", compact=True)
                    component_button.disabled = self._is_component_button_disabled()
                    yield component_button

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



    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add_function_btn":
            await self._add_function()
        elif event.button.id == "load_func_btn":
            await self._load_func()
        elif event.button.id == "save_func_as_btn":
            await self._save_func_as()
        elif event.button.id == "edit_vim_btn":
            self._edit_in_vim()
        elif event.button.id == "component_btn":
            await self._show_component_selection_dialog()
        elif event.button.id == "prev_channel_btn":
            self._navigate_channel(-1)
        elif event.button.id == "next_channel_btn":
            self._navigate_channel(1)



    def on_function_pane_widget_parameter_changed(self, event: Message) -> None:
        """Handle parameter change message from FunctionPaneWidget."""
        if hasattr(event, 'index') and hasattr(event, 'param_name') and hasattr(event, 'value'):
            if 0 <= event.index < len(self.functions):
                # Update function kwargs with proper type conversion
                func, kwargs = self.functions[event.index]
                new_kwargs = kwargs.copy()

                # Convert value to proper type based on function signature
                converted_value = event.value
                try:
                    from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
                    from enum import Enum

                    param_info = SignatureAnalyzer.analyze(func)
                    if event.param_name in param_info:
                        param_details = param_info[event.param_name]
                        param_type = param_details.param_type
                        is_required = param_details.is_required
                        default_value = param_details.default_value

                        # Handle empty strings - always convert to None or default value
                        if isinstance(event.value, str) and event.value.strip() == "":
                            # Empty parameter - use default value (None for required params)
                            converted_value = default_value
                        elif isinstance(event.value, str) and event.value.strip() != "":
                            # Non-empty string - convert to proper type
                            if param_type == float:
                                converted_value = float(event.value)
                            elif param_type == int:
                                converted_value = int(event.value)
                            elif param_type == bool:
                                converted_value = event.value.lower() in ('true', '1', 'yes', 'on')
                            elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                                converted_value = param_type(event.value)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to convert parameter '{event.param_name}' value '{event.value}': {e}")
                    converted_value = event.value  # Keep original value on conversion failure

                new_kwargs[event.param_name] = converted_value

                # Update functions list WITHOUT triggering recomposition (to prevent focus loss)
                # We modify the underlying list directly instead of assigning a new list
                self.functions[event.index] = (func, new_kwargs)

                # Manually update pattern data
                self._update_pattern_data()

                # Notify parent
                self.post_message(self.FunctionPatternChanged())
                logger.debug(f"Updated parameter {event.param_name}={converted_value} (type: {type(converted_value)}) for function {event.index}")

    async def on_function_pane_widget_change_function(self, event: Message) -> None:
        """Handle change function message from FunctionPaneWidget."""
        if hasattr(event, 'index') and 0 <= event.index < len(self.functions):
            await self._change_function(event.index)

    def on_function_pane_widget_remove_function(self, event: Message) -> None:
        """Handle remove function message from FunctionPaneWidget."""
        if hasattr(event, 'index') and 0 <= event.index < len(self.functions):
            new_functions = self.functions[:event.index] + self.functions[event.index+1:]
            self.functions = new_functions
            self._commit_and_notify()
            logger.debug(f"Removed function at index {event.index}")
        else:
            logger.warning(f"Invalid index for remove function: {getattr(event, 'index', 'N/A')}")

    async def on_function_pane_widget_add_function(self, event: Message) -> None:
        """Handle add function message from FunctionPaneWidget."""
        if hasattr(event, 'insert_index'):
            insert_index = min(event.insert_index, len(self.functions))  # Clamp to valid range
            await self._add_function_at_index(insert_index)
        else:
            logger.warning(f"Invalid add function event: missing insert_index")

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

    async def _add_function(self) -> None:
        """Add a new function to the end of the list."""
        await self._add_function_at_index(len(self.functions))

    async def _add_function_at_index(self, insert_index: int) -> None:
        """Add a new function at the specified index."""
        from openhcs.textual_tui.windows import FunctionSelectorWindow
        from textual.css.query import NoMatches

        def handle_function_selection(selected_function: Optional[Callable]) -> None:
            if selected_function:
                # Insert function at the specified index
                new_functions = self.functions.copy()
                new_functions.insert(insert_index, (selected_function, {}))
                self.functions = new_functions
                self._commit_and_notify()
                logger.debug(f"Added function: {selected_function.__name__} at index {insert_index}")

        # Use window-based function selector (follows ConfigWindow pattern)
        try:
            window = self.app.query_one(FunctionSelectorWindow)
            # Window exists, update it and open
            window.on_result_callback = handle_function_selection
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = FunctionSelectorWindow(on_result_callback=handle_function_selection)
            await self.app.mount(window)
            window.open_state = True

    async def _change_function(self, index: int) -> None:
        """Change function at specified index."""
        from openhcs.textual_tui.windows import FunctionSelectorWindow
        from textual.css.query import NoMatches

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

            # Use window-based function selector (follows ConfigWindow pattern)
            try:
                window = self.app.query_one(FunctionSelectorWindow)
                # Window exists, update it and open
                window.current_function = current_func
                window.on_result_callback = handle_function_selection
                window.open_state = True
            except NoMatches:
                # Expected case: window doesn't exist yet, create new one
                window = FunctionSelectorWindow(current_function=current_func, on_result_callback=handle_function_selection)
                await self.app.mount(window)
                window.open_state = True

    def _commit_and_notify(self) -> None:
        """Commit changes and notify parent of function pattern change."""
        # Update pattern data before notifying
        self._update_pattern_data()
        # Post message to notify parent (DualEditorScreen) of changes
        self.post_message(self.FunctionPatternChanged())

    async def _load_func(self) -> None:
        """Load function pattern from .func file."""
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey

        def handle_result(result):
            if result and isinstance(result, Path):
                self._load_pattern_from_file(result)

        # Use window-based file browser
        from openhcs.textual_tui.services.file_browser_service import SelectionMode
        await open_file_browser_window(
            app=self.app,
            file_manager=self.app.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.FUNCTION_PATTERNS),
            backend=Backend.DISK,
            title="Load Function Pattern (.func)",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func'],
            cache_key=PathCacheKey.FUNCTION_PATTERNS,
            on_result_callback=handle_result
        )

    async def _save_func_as(self) -> None:
        """Save function pattern to .func file."""
        from openhcs.textual_tui.windows import open_file_browser_window, BrowserMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey

        def handle_result(result):
            if result and isinstance(result, Path):
                self._save_pattern_to_file(result)

        # Use window-based file browser
        from openhcs.textual_tui.services.file_browser_service import SelectionMode
        await open_file_browser_window(
            app=self.app,
            file_manager=self.app.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.FUNCTION_PATTERNS),
            backend=Backend.DISK,
            title="Save Function Pattern (.func)",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func'],
            default_filename="pattern.func",
            cache_key=PathCacheKey.FUNCTION_PATTERNS,
            on_result_callback=handle_result
        )

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

    def _get_component_button_text(self) -> str:
        """Get text for the component selection button based on current group_by setting."""
        if self.current_group_by is None or self.current_group_by == GroupBy.NONE:
            return "Component: None"

        # Use group_by.value.title() for dynamic component type display
        component_type = self.current_group_by.value.title()

        if self.is_dict_mode and self.selected_channel is not None:
            # Try to get metadata name for the selected component
            display_name = self._get_component_display_name(self.selected_channel)
            return f"{component_type}: {display_name}"
        return f"{component_type}: None"

    def _get_component_display_name(self, component_key: str) -> str:
        """Get display name for component key, using metadata if available."""
        # Try to get metadata name from orchestrator
        orchestrator = self._get_current_orchestrator()
        if orchestrator and self.current_group_by:
            try:
                metadata_name = orchestrator.get_component_metadata(self.current_group_by, component_key)
                if metadata_name:
                    return metadata_name
            except Exception as e:
                logger.debug(f"Could not get metadata for {self.current_group_by.value} {component_key}: {e}")

        # Fallback to component key
        return component_key

    def _refresh_component_button(self) -> None:
        """Refresh the component button text and state."""
        try:
            component_button = self.query_one("#component_btn", Button)
            component_button.label = self._get_component_button_text()
            component_button.disabled = self._is_component_button_disabled()
        except Exception as e:
            logger.debug(f"Could not refresh component button: {e}")

    def _is_component_button_disabled(self) -> bool:
        """Check if component selection button should be disabled."""
        return (
            self.current_group_by is None or
            self.current_group_by == GroupBy.NONE or
            (self.current_variable_components and
             self.current_group_by.value in [vc.value for vc in self.current_variable_components])
        )

    async def _show_component_selection_dialog(self) -> None:
        """Show the component selection dialog for the current group_by setting."""
        try:
            # Check if component selection is disabled
            if self._is_component_button_disabled():
                logger.debug("Component selection is disabled")
                return

            # Get available components from orchestrator using current group_by
            orchestrator = self._get_current_orchestrator()
            if orchestrator is None:
                logger.warning("No orchestrator available for component selection")
                return

            available_components = orchestrator.get_component_keys(self.current_group_by)
            if not available_components:
                component_type = self.current_group_by.value
                logger.warning(f"No {component_type} values found in current plate")
                return

            # Get currently selected components from cache or current pattern
            if self.current_group_by in self.component_selections:
                # Use cached selection for this group_by
                selected_components = self.component_selections[self.current_group_by]
                logger.debug(f"Step '{self.step_identifier}': Using cached selection for {self.current_group_by.value}: {selected_components}")
            elif self.is_dict_mode and isinstance(self.pattern_data, dict):
                # Fallback to current pattern keys
                selected_components = list(self.pattern_data.keys())
            else:
                selected_components = []

            # Show window with dynamic component type
            from openhcs.textual_tui.windows import GroupBySelectorWindow
            from textual.css.query import NoMatches

            def handle_selection(result_components):
                if result_components is not None:
                    self._update_components(result_components)

            # Use window-based group-by selector (follows ConfigWindow pattern)
            try:
                window = self.app.query_one(GroupBySelectorWindow)
                # Window exists, update it and open
                window.available_channels = available_components
                window.selected_channels = selected_components
                window.component_type = self.current_group_by.value
                window.orchestrator = orchestrator
                window.on_result_callback = handle_selection
                window.open_state = True
            except NoMatches:
                # Expected case: window doesn't exist yet, create new one
                window = GroupBySelectorWindow(
                    available_channels=available_components,
                    selected_channels=selected_components,
                    on_result_callback=handle_selection,
                    component_type=self.current_group_by.value,
                    orchestrator=orchestrator
                )
                await self.app.mount(window)
                window.open_state = True

        except Exception as e:
            component_type = self.current_group_by.value if self.current_group_by else "component"
            logger.error(f"Failed to show {component_type} selection dialog: {e}")

    def _update_components(self, new_components: List[str]) -> None:
        """
        Update the pattern based on new component selection.

        Uses string component keys directly to match the pattern detection system.
        Pattern detection always returns string component values (e.g., '1', '2', '3') when
        grouping by any component, so function patterns use string keys for consistency.
        """
        if not new_components:
            # No components selected - revert to list mode
            if self.is_dict_mode:
                # Save current functions to list mode
                self.pattern_data = self.functions
                self.is_dict_mode = False
                self.selected_channel = None
                logger.debug("Reverted to list mode (no components selected)")
        else:
            # Use component strings directly - no conversion needed
            component_keys = new_components

            # Components selected - ensure dict mode
            if not self.is_dict_mode:
                # Convert to dict mode
                current_functions = self.functions
                self.pattern_data = {component_keys[0]: current_functions}
                self.is_dict_mode = True
                self.selected_channel = component_keys[0]

                # Add other components with empty functions
                for component_key in component_keys[1:]:
                    self.pattern_data[component_key] = []
            else:
                # Already in dict mode - update components
                old_pattern = self.pattern_data.copy() if isinstance(self.pattern_data, dict) else {}
                new_pattern = {}

                # Keep existing functions for components that remain
                for component_key in component_keys:
                    # Check both string and integer keys for backward compatibility
                    if component_key in old_pattern:
                        new_pattern[component_key] = old_pattern[component_key]
                    else:
                        # Try integer key for backward compatibility
                        try:
                            component_int = int(component_key)
                            if component_int in old_pattern:
                                new_pattern[component_key] = old_pattern[component_int]
                            else:
                                new_pattern[component_key] = []
                        except ValueError:
                            new_pattern[component_key] = []

                self.pattern_data = new_pattern

                # Update selected component if needed
                if self.selected_channel not in component_keys:
                    self.selected_channel = component_keys[0] if component_keys else None
                    if self.selected_channel:
                        self.functions = new_pattern.get(self.selected_channel, [])

        # Save selection to cache for current group_by
        if self.current_group_by is not None and self.current_group_by != GroupBy.NONE:
            self.component_selections[self.current_group_by] = new_components
            logger.debug(f"Step '{self.step_identifier}': Cached selection for {self.current_group_by.value}: {new_components}")

        self._commit_and_notify()
        self._refresh_component_button()
        logger.debug(f"Updated components: {new_components}")

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
            # Get from app - PlateManagerWidget is now in PipelinePlateWindow
            if hasattr(self.app, 'query_one'):
                from openhcs.textual_tui.windows import PipelinePlateWindow
                from openhcs.textual_tui.widgets.plate_manager import PlateManagerWidget

                # Try to find the PipelinePlateWindow first
                try:
                    pipeline_plate_window = self.app.query_one(PipelinePlateWindow)
                    plate_manager = pipeline_plate_window.plate_widget
                except:
                    # Fallback: try to find PlateManagerWidget directly in the app
                    plate_manager = self.app.query_one(PlateManagerWidget)

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

