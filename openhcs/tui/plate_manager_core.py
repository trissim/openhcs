"""
Plate Manager Pane for OpenHCS TUI - Core Implementation.

This module implements the left pane of the OpenHCS TUI, which displays
a list of plates (filesystem directories) and allows the user to select,
add, and manage them.

🔒 Clause 295: Component Boundaries
Maintain clear boundaries between components with explicit interfaces.

🔒 Clause 306: Backend Positional Parameters
All backend parameters must be passed positionally, not as keywords.

🔒 Clause 310: Function Backend Propagation
Any function accepting filemanager must also accept backend.

🔒 Clause 315: TUI_FILEMANAGER_INJECTION
All TUI managers must receive context.filemanager and pass backend positionally.

🔒 Clause 317: TUI_STATUS_THREADSAFETY
All updates to state.operation_status must be serialized through a lock.

🔒 Clause 319: TUI_NO_VIRTUALPATH_EXPOSURE
TUI must work only with plain strings, not VirtualPath objects.
"""
import asyncio
import logging # ADDED
import os
import shutil  # For terminal size fallback
import signal  # For signal handling in register_with_app
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, VSplit
from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea
from tui.constants import STATUS_ICONS
from tui.dialogs.plate_dialog_manager import PlateDialogManager
from tui.services.plate_validation_service import PlateValidationService

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.orchestrator import BackendRegistry

logger = logging.getLogger(__name__) # ADDED


# Define interfaces for component communication
class PlateEventHandler(Protocol):
    """Protocol defining the interface for plate event handling."""
    async def on_plate_added(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_removed(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_selected(self, plate: Dict[str, Any]) -> None: ...
    async def on_plate_status_changed(self, plate_id: str, status: str) -> None: ...


class PlateManagerPane:
    """
    Left pane for managing plates in the OpenHCS TUI.

    Displays a list of plates (filesystem directories) and allows
    the user to select, add, and manage them. When a plate is selected,
    it emits an event to the ProcessingContext through the TUIState.

    This class uses composition with PlateDialogManager for dialog handling.
    """
    def __init__(self, state, context: ProcessingContext, filemanager, backend_registry=None):
        """
        Initialize the Plate Manager pane.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
            filemanager: Optional FileManager instance (defaults to context.filemanager)
            backend_registry: Optional BackendRegistry instance for memory sharing
        """
        self.state = state
        self.context = context

        # Strict filemanager validation (Clause 231)
        if filemanager is None:
            raise RuntimeError("PlateManagerCore requires non-None filemanager at initialization")
        self.filemanager = filemanager

        # Store backend registry for memory sharing
        self.backend_registry = backend_registry

        # Initialize state with thread safety
        self.plates: List[Dict[str, Any]] = []
        self.selected_index = 0
        self.is_loading = False
        self.plates_lock = asyncio.Lock()

        # Create dedicated executor for I/O operations
        self.io_executor = ThreadPoolExecutor(
            max_workers=3,  # Limit concurrent I/O operations
            thread_name_prefix="plate-io-"
        )

        # Create dialog manager with callbacks
        self.dialog_manager = PlateDialogManager(
            on_add_dialog_result=self._handle_add_dialog_result,
            on_remove_dialog_result=self._handle_remove_dialog_result,
            on_error=self._handle_error,
            backend_registry=self.backend_registry
        )

        # Create validation service with callbacks
        self.validation_service = PlateValidationService(
            context=self.context,
            on_validation_result=self._handle_validation_result,
            on_error=self._handle_error,
            filemanager=self.filemanager,
            io_executor=self.io_executor
        )

        # Initialize UI components that don't require app context
        # Actual UI initialization will happen in _initialize_and_refresh
        # This prevents duplicate initialization (Clause 12)
        self._ui_initialized = False

    async def _initialize_ui(self):
        """
        Initialize UI components that need filemanager.

        🔒 Clause 12: Explicit Error Handling
        Made idempotent to prevent duplicate initialization.

        🔒 Clause 317: Runtime Correctness
        Defers app context access until TUI is running.
        """
        # Guard against duplicate initialization
        if self._ui_initialized:
            return

        # Create UI components
        self.plate_list = await self._create_plate_list()

        # Create button handlers that safely access app context
        # Only create these when the app is actually running
        app = get_app()
        if not hasattr(app, 'is_running') or not app.is_running:
            raise RuntimeError("Cannot initialize UI before application is running")

        self.add_button = Button(
            "Add Plate",
            handler=lambda: app.create_background_task(self._show_add_plate_dialog())
        )
        self.remove_button = Button(
            "Remove Plate",
            handler=lambda: app.create_background_task(self._show_remove_plate_dialog())
        )
        self.refresh_button = Button(
            "Refresh",
            handler=lambda: app.create_background_task(self._refresh_plates())
        )

        # Mark as initialized to prevent duplicate initialization
        self._ui_initialized = True

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Create container
        self.container = HSplit([
            self.plate_list,
            VSplit([
                Box(self.add_button, padding=1),
                Box(self.remove_button, padding=1),
                Box(self.refresh_button, padding=1)
            ])
        ])

        # Register for events
        self.state.add_observer('refresh_plates', self._refresh_plates)
        self.state.add_observer('plate_status_changed', self._update_plate_status)
        # Listen for requests to show the add plate dialog
        self.state.add_observer('ui_request_show_add_plate_dialog',
                                lambda data=None: get_app().create_background_task(self._handle_request_show_add_plate_dialog(data)))
        # Listen for requests to add a predefined test plate
        self.state.add_observer('add_predefined_plate',
                                lambda data=None: get_app().create_background_task(self._handle_add_predefined_plate(data)))

    async def _handle_request_show_add_plate_dialog(self, data=None):
        """Handles the TUIState event 'ui_request_show_add_plate_dialog'."""
        logger.info("PlateManagerPane: Received ui_request_show_add_plate_dialog event.")
        await self._show_add_plate_dialog()

    async def _handle_add_predefined_plate(self, data: Optional[Dict[str, Any]] = None):
        """Handles the TUIState event 'add_predefined_plate'."""
        if not data or 'path' not in data or 'backend' not in data:
            logger.error("PlateManagerPane: Received 'add_predefined_plate' event with missing data.")
            await self._handle_error("Invalid data for predefined plate.",
                                     f"Received: {data}")
            return

        path = data['path']
        backend = data['backend']
        logger.info(f"PlateManagerPane: Received 'add_predefined_plate' event for path='{path}', backend='{backend}'.")
        
        try:
            # Use the existing validation service to add and validate the plate
            await self.validation_service.validate_plate(path, backend)
            logger.info(f"PlateManagerPane: Validation initiated for predefined plate '{path}'.")
        except Exception as e:
            logger.error(f"PlateManagerPane: Error initiating validation for predefined plate '{path}': {e}", exc_info=True)
            await self._handle_error(f"Error adding test plate {Path(path).name}", str(e))

    def _on_filemanager_available(self, data):
        """Handle filemanager becoming available."""
        if 'filemanager' in data:
            self.filemanager = data['filemanager']

            # Update validation service with filemanager
            self.validation_service.filemanager = self.filemanager

            # Initialize UI asynchronously
            # Use await to properly handle async method (Clause 12)
            get_app().create_background_task(self._initialize_and_refresh)

    async def _initialize_and_refresh(self):
        """
        Initialize UI and refresh plates asynchronously.

        🔒 Clause 317: Runtime Correctness
        Ensures app is running before initializing UI.
        """
        # Wait for app to be fully running before initializing UI
        app = get_app()
        while not hasattr(app, 'is_running') or not app.is_running:
            await asyncio.sleep(0.1)

        # Initialize UI first
        await self._initialize_ui()

        # Then refresh plates if any were added before filemanager was available
        await self._refresh_plates()

        # Register shutdown hook with application
        self.register_with_app()

    async def _create_plate_list(self) -> TextArea:
        """Create the plate list component with proper scrolling."""
        # Get initial text with thread safety
        initial_text = await self._format_plate_list()

        # Create TextArea with proper scrolling support for PTK ≥3.0
        # Remove horizontal_scroll which is not valid in PTK ≥3.0 (Clause 24)
        from prompt_toolkit.layout.margins import ScrollbarMargin

        text_area = TextArea(
            text=initial_text,
            read_only=True,
            scrollbar=True,
            wrap_lines=False,
            width=None,  # Let TextArea determine appropriate width
            right_margins=[ScrollbarMargin()]  # Use ScrollbarMargin instead of horizontal_scroll
        )
        return text_area

    async def _format_plate_list(self, lock_already_held: bool = False) -> str:
        """
        Format the plate list for display with proper path handling.

        Args:
            lock_already_held: Whether the plates_lock is already held by the caller

        🔒 Clause 317: TUI_STATUS_THREADSAFETY
        Uses plates_lock to prevent read-while-write races.
        """
        # Define the formatting function to avoid code duplication
        async def _do_format():
            if self.is_loading:
                return "Loading plates..."

            if not self.plates:
                return "No plates added. Click 'Add Plate' to add a plate."

            lines = []
            # Get filemanager from instance (set in constructor)
            fm = self.filemanager

            # Get terminal width for dynamic path truncation
            # Use shutil.get_terminal_size() as a fallback (Clause 12)
            terminal_width = 80  # Default fallback
            try:
                # Try prompt_toolkit's method first
                app = get_app()
                if hasattr(app, 'output') and app.output is not None:
                    size = app.output.get_size()
                    if size and size.columns > 0:
                        terminal_width = size.columns
                    else:
                        # Fallback to shutil if prompt_toolkit returns invalid size
                        terminal_width = shutil.get_terminal_size((80, 20)).columns
                else:
                    # Fallback to shutil if app.output is not available
                    terminal_width = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                # Final fallback if all else fails
                terminal_width = shutil.get_terminal_size((80, 20)).columns

            max_path_length = max(30, terminal_width - 30)  # Dynamic width based on terminal

            for i, plate in enumerate(self.plates):
                # Format: [status] plate_name | plate_path
                status_icon = STATUS_ICONS.get(plate['status'], "?")
                selected = ">" if i == self.selected_index else " "
                name = plate['name']

                # 🔒 Clause 319: TUI_NO_VIRTUALPATH_EXPOSURE
                # Convert to OS path to prevent leaking VFS URIs
                backend = plate['backend']
                virtual_path = fm.get_path(plate['path'], backend)
                # Use os_path (canonical accessor) to get OS path without VFS URI
                # VirtualPath guarantees .os_path, not .local_path (Clause 88)
                path_str = str(virtual_path.os_path)

                # Truncate long paths for display
                if len(path_str) > max_path_length:
                    path_str = path_str[:max_path_length-3] + "..."

                line = f"{selected} {status_icon} {name} | {path_str}"
                lines.append(line)

            return "\n".join(lines)

        # Acquire lock only if not already held
        if lock_already_held:
            return await _do_format()
        else:
            # Acquire lock to prevent read-while-write races
            async with self.plates_lock:
                return await _do_format()

    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings for the plate list with Vim-style navigation."""
        kb = KeyBindings()

        # Arrow key navigation
        @kb.add('up', filter=has_focus(self.plate_list))
        def _(event):
            """Move selection up."""
            if self.plates:
                get_app().create_background_task(self._move_selection(-1))

        @kb.add('down', filter=has_focus(self.plate_list))
        def _(event):
            """Move selection down."""
            if self.plates:
                get_app().create_background_task(self._move_selection(1))

        # Vim-style navigation
        # Use getattr with default for vim_mode to prevent AttributeError (Clause 24/41)
        vim_mode_condition = Condition(lambda: getattr(self.state, "vim_mode", False))

        @kb.add('k', filter=has_focus(self.plate_list) & vim_mode_condition)
        def _(event):
            """Move selection up (Vim style)."""
            if self.plates:
                get_app().create_background_task(self._move_selection(-1))

        @kb.add('j', filter=has_focus(self.plate_list) & vim_mode_condition)
        def _(event):
            """Move selection down (Vim style)."""
            if self.plates:
                get_app().create_background_task(self._move_selection(1))

        # Selection
        @kb.add('enter', filter=has_focus(self.plate_list))
        def _(event):
            """Select the current plate."""
            if self.plates and 0 <= self.selected_index < len(self.plates):
                get_app().create_background_task(self._select_plate(self.selected_index))

        return kb

    # Dialog interface methods
    async def _show_add_plate_dialog(self):
        """Show dialog to add a plate using the dialog manager."""
        await self.dialog_manager.show_add_plate_dialog()

    async def _show_remove_plate_dialog(self):
        """Show dialog to remove a plate using the dialog manager."""
        if not self.plates or not (0 <= self.selected_index < len(self.plates)):
            return

        plate = self.plates[self.selected_index]
        await self.dialog_manager.show_remove_plate_dialog(plate)

    # Event handlers for dialog manager callbacks
    async def _handle_add_dialog_result(self, result: Dict[str, Any]):
        """Handle add dialog result from dialog manager."""
        if 'path' in result and 'backend' in result:
            # Start validation process
            try:
                await self.validation_service.validate_plate(result['path'], result['backend'])
            except Exception as e:
                # Assuming validation_service handles user notification of this error.
                # Logging that this path was taken for debug purposes.
                logger.debug(f"Exception caught in _handle_add_dialog_result after validation_service.validate_plate call: {e}", exc_info=True)
                pass

    async def _handle_remove_dialog_result(self, plate: Dict[str, Any]):
        """Handle remove dialog result from dialog manager."""
        async with self.plates_lock:
            # Remove plate from list
            self.plates = [p for p in self.plates if p['id'] != plate['id']]

            # Update selection
            if self.selected_index >= len(self.plates):
                self.selected_index = max(0, len(self.plates) - 1)

            # Update UI - lock already held
            formatted = await self._format_plate_list(lock_already_held=True)
            self.plate_list.text = formatted

            # Notify state
            self.state.notify('plate_removed', plate)

    # Event handlers for validation service callbacks
    async def _handle_validation_result(self, plate: Dict[str, Any]):
        """Handle validation result from validation service."""
        async with self.plates_lock:
            # Check if plate already exists
            existing_plate = next((p for p in self.plates if p['id'] == plate['id']), None)

            if existing_plate:
                # Update existing plate
                for p in self.plates:
                    if p['id'] == plate['id']:
                        p.update(plate)
                        break
            else:
                # Add new plate
                self.plates.append(plate)
                self.selected_index = len(self.plates) - 1

            # Update UI - lock already held
            formatted = await self._format_plate_list(lock_already_held=True)
            self.plate_list.text = formatted

            # Notify state if plate is ready
            if plate['status'] == 'ready':
                self.state.notify('plate_added', plate)

    async def _handle_error(self, message: str, details: str = None):
        """Handle error from components."""
        # Store in state for logging with bounded size
        if not hasattr(self.state, 'error_logs'):
            # Use collections.deque with maxlen for bounded size
            from collections import deque
            self.state.error_logs = deque(maxlen=200)

        if details:
            # Add error to the bounded log
            self.state.error_logs.append(f"--- Error ---\n{details}")

        # Notify state
        self.state.notify('error', {
            'source': 'PlateManagerPane',
            'message': message,
            'details': details
        })

    # Core plate management methods
    async def _move_selection(self, delta: int) -> None:
        """Move selection up or down with thread safety."""
        async with self.plates_lock:
            if self.plates:
                self.selected_index = max(0, min(len(self.plates) - 1, self.selected_index + delta))
                # Use async _format_plate_list with lock already acquired
                # Update UI - lock already held
                formatted = await self._format_plate_list(lock_already_held=True)
                self.plate_list.text = formatted

                # Ensure selection is visible
                if 0 <= self.selected_index < len(self.plates):
                    # Schedule this to run after rendering
                    get_app().call_later(self._ensure_selection_visible)

    def _ensure_selection_visible(self) -> None:
        """Ensure the selected item is visible in the viewport."""
        if hasattr(self, 'plate_list') and self.plates and 0 <= self.selected_index < len(self.plates):
            # Scroll to make selection visible if needed
            self.plate_list.buffer.cursor_position = self.plate_list.buffer.document.translate_row_col_to_index(
                self.selected_index, 0
            )

    async def _select_plate(self, index: int) -> None:
        """Select a plate and emit selection event."""
        if not self.plates or not (0 <= index < len(self.plates)):
            return

        async with self.plates_lock:
            plate = self.plates[index]
            self.selected_index = index

            # Update UI to reflect selection - lock already held
            # Update UI - lock already held
            formatted = await self._format_plate_list(lock_already_held=True)
            self.plate_list.text = formatted

            # Emit event to TUIState (not directly to context)
            self.state.notify('plate_selected', {
                'id': plate['id'],
                'backend': plate['backend'],
                'path': plate['path']
            })

    async def _update_plate_status(self, data):
        """Update the status of a plate with thread safety."""
        if not data or 'plate_id' not in data or 'status' not in data:
            return

        # Use thread-safe lock for plate list modifications (TUI_STATUS_THREADSAFETY)
        async with self.plates_lock:
            # Find plate by ID
            for plate in self.plates:
                if plate['id'] == data['plate_id']:
                    # Update status
                    plate['status'] = data['status']
                    break

            # Update UI - lock already held
            # Update UI - lock already held
            formatted = await self._format_plate_list(lock_already_held=True)
            self.plate_list.text = formatted

            # Notify state of status change
            self.state.notify('plate_status_updated', {
                'plate_id': data['plate_id'],
                'status': data['status']
            })

    async def _refresh_plates(self, _=None):
        """Refresh the plate list with thread safety."""
        async with self.plates_lock:
            # Set loading state
            self.is_loading = True
            self.plate_list.text = await self._format_plate_list(lock_already_held=True)

            try:
                # In a real implementation, this would query the filesystem
                # or ProcessingContext for updated plate information
                # For now, we'll just simulate work
                await asyncio.sleep(0.5)

                # Update plate statuses from context
                for plate in self.plates:
                    # TODO: Implement plate status update logic here.
                    # This should involve:
                    # 1. Getting the relevant ProcessingContext for the plate.
                    # 2. Using the FileManager in the context to inspect VFS
                    #    for output artifacts, logs, or status markers.
                    # 3. Updating `plate['status']` based on VFS inspection.
                    # For now, the plate's existing status is preserved during refresh.
                    pass
            finally:
                # Hide loading indicator
                self.is_loading = False
                self.plate_list.text = await self._format_plate_list(lock_already_held=True)

    async def shutdown(self):
        """
        Explicit cleanup method for deterministic resource release.

        🔒 Clause 317: Runtime Correctness
        Ensures ThreadPool is properly shut down.

        🔒 Clause 241: Resource Lifecycle Management
        Explicitly manages executor and validation service lifecycle.
        """
        # Close validation service first if it exists
        if hasattr(self, 'validation_service') and self.validation_service is not None:
            await self.validation_service.close()
            self.validation_service = None

        # Then shut down our own executor
        if hasattr(self, 'io_executor') and self.io_executor is not None:
            # Use asyncio to ensure thread pool shutdown is non-blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.io_executor.shutdown(wait=True)
            )
            self.io_executor = None
```
