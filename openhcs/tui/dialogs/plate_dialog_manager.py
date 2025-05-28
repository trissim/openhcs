# plan_02a2_plate_dialog_manager.md
# Component: Plate Dialog Manager

"""
Plate Dialog Manager for OpenHCS TUI.

This module implements a separate dialog manager component for handling
plate-related dialogs, using composition to maintain clear boundaries
with the Plate Manager pane. This component focuses exclusively on
dialog creation, display, and lifecycle management.

🔒 Clause 295: Component Boundaries
Maintain clear boundaries between components with explicit interfaces.

🔒 Clause 316: TUI_BACKEND_SELECT_WIDGET
UI elements for backend selection must emit explicit strings.

🔒 Clause 298: Dialog Future Namespacing
Dialog futures must use namespaced attributes to avoid conflicts.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Coroutine # Added List

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import (ConditionalContainer, Container, Float,
                                   HSplit, Dimension) # Added Dimension
from prompt_toolkit.widgets import Box, Button, Dialog, Label
from prompt_toolkit.widgets import RadioList as Dropdown
from prompt_toolkit.widgets import TextArea

# OpenHCS imports
from openhcs.io.filemanager import FileManager
from openhcs.constants.constants import Backend
from openhcs.tui.file_browser import FileManagerBrowser # Import the new browser

# For logging
import logging
logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]


# Define callback protocols for clean interfaces
class DialogResultCallback(Protocol):
    """Protocol for dialog result callback."""
    async def __call__(self, result: Any) -> None: ...

class ErrorCallback(Protocol):
    """Protocol for error callback."""
    async def __call__(self, message: str, details: Optional[str] = None) -> None: ...

class ErrorBanner(Container):
    """
    A dedicated error banner component for dialogs.

    This component provides a consistent way to display error messages
    in dialogs, with proper visibility control and styling.
    """
    def __init__(self):
        """Initialize the error banner with empty message and hidden state."""
        self.message = ""
        self.visible = False

        # Create the label with error styling
        self.label = Label(lambda: HTML(f'<ansired>{self.message}</ansired>'))

        # Create the container with the label
        self.container = Box(
            self.label,
            padding=1,
            style="class:error-banner"
        )

        # Set up visibility filter and apply it to the container
        self.filter = Condition(lambda: self.visible)
        # Wrap container with filter to enforce visibility control
        self.filtered_container = ConditionalContainer(
            self.container,
            filter=self.filter
        )

    def show(self, message: str):
        """
        Show the error banner with the given message.

        Args:
            message: The error message to display
        """
        self.message = message
        self.visible = True
        # Force UI refresh to ensure visibility
        get_app().invalidate()

    def hide(self):
        """Hide the error banner."""
        self.visible = False
        # Force UI refresh to ensure visibility update
        get_app().invalidate()

    def reset(self):
        """Reset the error banner to its initial state."""
        self.message = ""
        self.visible = False

    def __pt_container__(self):
        """Return the container to render."""
        return self.filtered_container

    # Implement abstract methods by delegating to the internal container
    def get_children(self):
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        return self.container.preferred_width(max_available_width)

    def preferred_height(self, max_available_height, width):
        return self.container.preferred_height(max_available_height, width)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                       parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events for the error banner (no interaction)."""
        return False # Event not handled

    @staticmethod
    def find_in_container(container):
        """
        Recursively find ErrorBanner in container hierarchy.

        Args:
            container: Container to search in

        Returns:
            ErrorBanner instance or None if not found
        """
        # Check if this is an ErrorBanner
        if isinstance(container, ErrorBanner):
            return container

        # Check if container has children
        if hasattr(container, 'children'):
            for child in container.children:
                result = ErrorBanner.find_in_container(child)
                if result:
                    return result

        # Check if container has content or body
        if hasattr(container, 'content'):
            return ErrorBanner.find_in_container(container.content)

        if hasattr(container, 'body'):
            return ErrorBanner.find_in_container(container.body)

        # Not found
        return None


class PlateDialogManager:
    """
    Dialog manager for plate-related operations.

    This class handles all dialog operations for the Plate Manager pane,
    focusing exclusively on dialog creation, display, and lifecycle management.
    It uses callbacks to communicate with the parent component, maintaining
    clear boundaries.
    """
    def __init__(
        self,
        on_add_dialog_result: DialogResultCallback,
        on_remove_dialog_result: DialogResultCallback,
        on_error: ErrorCallback,
        # For now, directly accept FileManager and default backend.
        # Later, this would be a StorageRegistry.
        file_manager: FileManager,
        default_backend: Backend
    ):
        """
        Initialize the dialog manager.

        Args:
            on_add_dialog_result: Callback for add dialog result.
            on_remove_dialog_result: Callback for remove dialog result.
            on_error: Callback for error handling.
            file_manager: An instance of FileManager.
            default_backend: The default Backend to use.
        """
        self.on_add_dialog_result = on_add_dialog_result
        self.on_remove_dialog_result = on_remove_dialog_result
        self.on_error = on_error
        self.file_manager = file_manager # Store FileManager instance
        self.default_backend = default_backend # Store default backend
        self.current_dialog_instance: Optional[Dialog] = None # To manage dialog future

    async def show_add_plate_dialog(self) -> None:
        """Show dialog to add a plate using FileManagerBrowser."""

        app = get_app()

        # Callbacks for FileManagerBrowser
        async def browser_on_path_selected(selected_paths: List[Path]): # Expects a List[Path]
            logger.info(f"PlateDialogManager: Paths selected from browser: {selected_paths}")
            # Ensure paths are strings for the result dict
            path_strs = [str(p) for p in selected_paths]
            await self.on_add_dialog_result({'paths': path_strs}) # Pass list of paths
            # Close the dialog by setting its future
            if self.current_dialog_instance and hasattr(self.current_dialog_instance, '__ohcs_future'):
                future = getattr(self.current_dialog_instance, '__ohcs_future')
                if future and not future.done():
                    future.set_result({'paths': path_strs}) # Result for _show_dialog
            else: # Fallback if future mechanism isn't set up by _show_dialog for this custom body
                # This might mean we need to adjust how _show_dialog handles custom bodies
                # or the browser dialog needs to manage its own exit.
                # For now, assume _show_dialog's future is primary.
                pass


        async def browser_on_cancel():
            logger.info("PlateDialogManager: File browser cancelled.")
            # Close the dialog by setting its future to None
            if self.current_dialog_instance and hasattr(self.current_dialog_instance, '__ohcs_future'):
                future = getattr(self.current_dialog_instance, '__ohcs_future')
                if future and not future.done():
                    future.set_result(None)
            await self.on_add_dialog_result(None) # Notify main callback of cancellation


        file_browser_component = FileManagerBrowser(
            file_manager=self.file_manager,
            on_path_selected=browser_on_path_selected,
            on_cancel=browser_on_cancel,
            initial_path=Path.home(), # Or a more relevant default
            backend=self.default_backend,
            select_files=False, # We are selecting directories for plates
            select_multiple=True # Enable multi-select
        )

        dialog = Dialog(
            title="Select Plate Directory",
            body=file_browser_component, # Embed the browser
            # FileManagerBrowser has its own Select/Cancel buttons, so we might not need them here.
            # If Dialog buttons are kept, they need to interact with file_browser_component.
            # For now, let's remove dialog-level buttons and rely on browser's internal ones.
            buttons=[],
            width=Dimension(preferred=100, max=120), # Wider for file browser
            height=Dimension(preferred=30, max=40), # Taller for file browser
            modal=True,
            key_bindings=file_browser_component.get_key_bindings(), # Pass browser's KBs
            initial_focus=file_browser_component.get_initial_focus_target() # Set initial focus
        )
        self.current_dialog_instance = dialog # Store for callbacks

        # Show dialog and wait for result (path dict or None)
        # _show_dialog will handle the future for this dialog instance
        selected_data = await self._show_dialog(dialog)

        # The browser_on_path_selected/browser_on_cancel already called on_add_dialog_result.
        # So, no further action needed here with selected_data unless for logging.
        if selected_data:
            logger.info(f"Add Plate Dialog completed with selection: {selected_data}")
        else:
            logger.info("Add Plate Dialog completed with no selection or cancellation.")


    # _create_file_browser_dialog is now replaced by the logic within show_add_plate_dialog
    # We can remove it or comment it out. For now, let's remove it.

    async def _show_dialog(self, dialog: Dialog) -> Optional[Any]:
        """
        Show a dialog and properly manage its lifecycle.

        🔒 Clause 12: Explicit Error Handling
        Adds guard for previous_focus to prevent crashes.
        """
        # Get app reference
        app = get_app()

        # Store current focus for restoration with explicit None check
        # 🔒 Clause 12: Explicit Error Handling
        previous_focus = app.layout.current_window if hasattr(app.layout, 'current_window') else None

        # Create float container
        float_container = Float(dialog)

        # Store app, float_container and previous_focus in dialog for access in cancel handler
        # 🔒 Clause 3: Declarative Primacy
        # Ensure cancel handler has access to same state
        setattr(dialog, '__ohcs_app', app)
        setattr(dialog, '__ohcs_float', float_container)
        setattr(dialog, '__ohcs_prev_focus', previous_focus)

        # Create future for result
        # 🔒 Clause 298: Use namespaced attribute to avoid conflicts
        future = asyncio.Future()
        setattr(dialog, '__ohcs_future', future)

        try:
            # Show dialog
            app.layout.container.floats.append(float_container)
            app.layout.focus(dialog)

            # Wait for result
            return await future
        finally:
            # Ensure dialog is closed in all cases
            # Store reference to the Float object, not just the dialog
            if float_container in app.layout.container.floats:
                app.layout.container.floats.remove(float_container)

            # Restore previous focus with robust guard for None
            # 🔒 Clause 12: Explicit Error Handling
            # Use walk() instead of find_all_windows() for Textual 0.40+ compatibility
            if previous_focus is not None and hasattr(app.layout, 'walk'):
                try:
                    if previous_focus in app.layout.walk():
                        app.layout.focus(previous_focus)
                except Exception as e_focus:
                    # Silently continue if focus restoration fails, but log for debugging.
                    logger.debug(f"Failed to restore focus to {previous_focus}: {e_focus}", exc_info=True)
                    pass

    async def _dialog_ok(self, dlg, result):
        """
        Handle dialog OK button with defensive backend validation.

        Args:
            dlg: The dialog instance
            result: The dialog result data
        """
        # 🔒 Clause 316: Defensive check for backend
        if isinstance(result, dict) and 'backend' in result:
            backend = result['backend']
            # Belt-and-suspenders check at dialog boundary
            # Simplified check: backend must be a non-empty string
            if not (isinstance(backend, str) and backend):
                # Find the error banner in the dialog
                error_banner = None
                for child in dlg.body.children:
                    if isinstance(child, ErrorBanner):
                        error_banner = child
                        break

                # Show error in banner and abort
                if error_banner:
                    error_banner.show("Backend must be explicitly chosen.")
                return

        # Set dialog result
        # 🔒 Clause 298: Use getattr to safely access namespaced attribute
        future = getattr(dlg, '__ohcs_future', None)
        if future is not None and not future.done():
            future.set_result(result)

    async def _dialog_cancel(self, dlg):
        """
        Handle dialog Cancel button.

        Args:
            dlg: The dialog instance

        🔒 Clause 3: Declarative Primacy
        Ensures float removal happens in cancel path as well.
        """
        # Get stored app and float container
        app = getattr(dlg, '__ohcs_app', None)
        float_container = getattr(dlg, '__ohcs_float', None)
        previous_focus = getattr(dlg, '__ohcs_prev_focus', None)

        # Remove float if it exists
        # This handles the case where cancel is called before _show_dialog awaits
        if app is not None and float_container is not None:
            if hasattr(app, 'layout') and hasattr(app.layout, 'container'):
                if hasattr(app.layout.container, 'floats') and float_container in app.layout.container.floats:
                    app.layout.container.floats.remove(float_container)

                    # Restore previous focus with robust guard for None
                    if previous_focus is not None and hasattr(app.layout, 'walk'):
                        try:
                            if previous_focus in app.layout.walk():
                                app.layout.focus(previous_focus)
                        except Exception as e_focus_cancel:
                            # Silently continue if focus restoration fails, but log for debugging.
                            logger.debug(f"Failed to restore focus during dialog cancel to {previous_focus}: {e_focus_cancel}", exc_info=True)
                            pass

        # 🔒 Clause 298: Use getattr to safely access namespaced attribute
        future = getattr(dlg, '__ohcs_future', None)
        if future is not None and not future.done():
            future.set_result(None)

    def _handle_ok_button_press(self, dialog, path_input: TextArea):
        """
        Handle OK button press. Extracts path(s) and calls dialog_ok.
        Basic path validation (non-empty) is done here.
        More extensive validation is deferred.

        Args:
            dialog: The dialog instance
            path_input: The path input TextArea
        """
        paths_text = path_input.text.strip()
        error_banner = ErrorBanner.find_in_container(dialog)

        if not self._validate_input_not_empty(paths_text, error_banner):
            return

        paths = self._extract_valid_paths(paths_text)
        if not self._validate_paths_exist(paths, error_banner):
            return

        if not self._validate_path_format(paths, error_banner):
            return

        self._hide_error_banner(error_banner)
        self._submit_dialog_result(dialog, paths)

    def _validate_input_not_empty(self, paths_text: str, error_banner: Optional[ErrorBanner]) -> bool:
        """Validate that input is not empty."""
        if not paths_text:
            if error_banner:
                error_banner.show("Path input cannot be empty.")
            return False
        return True

    def _extract_valid_paths(self, paths_text: str) -> List[str]:
        """Extract valid paths from input text."""
        return [p.strip() for p in paths_text.splitlines() if p.strip()]

    def _validate_paths_exist(self, paths: List[str], error_banner: Optional[ErrorBanner]) -> bool:
        """Validate that at least one path exists."""
        if not paths:
            if error_banner:
                error_banner.show("No valid paths entered.")
            return False
        return True

    def _validate_path_format(self, paths: List[str], error_banner: Optional[ErrorBanner]) -> bool:
        """Validate path format (absolute, no scheme)."""
        for p_str in paths:
            if not self._is_valid_path_format(p_str):
                if error_banner:
                    if "://" in p_str:
                        error_banner.show(f"Path cannot contain URI schemes: {p_str}")
                    else:
                        error_banner.show(f"Path must be absolute: {p_str}")
                return False
        return True

    def _is_valid_path_format(self, path_str: str) -> bool:
        """Check if path format is valid."""
        if "://" in path_str:
            return False
        path_obj = Path(path_str)
        return path_obj.is_absolute()

    def _hide_error_banner(self, error_banner: Optional[ErrorBanner]) -> None:
        """Hide error banner if it exists."""
        if error_banner:
            error_banner.hide()

    def _submit_dialog_result(self, dialog, paths: List[str]) -> None:
        """Submit dialog result."""
        result_path = paths if len(paths) > 1 else paths[0]
        get_app().create_background_task(
            self._dialog_ok(dialog, {'path': result_path})
        )

    # Note: _show_error_inline method removed in favor of ErrorBanner component
    async def _show_error_dialog(self, message: str, details: str = None):
        """Show an error dialog with proper error details."""
        # Create error dialog
        dialog = self._create_error_dialog(message, details)

        # Show dialog
        await self._show_dialog(dialog)

        # Notify parent component of error
        await self.on_error(message, details)

    def _create_error_dialog(self, message: str, details: str = None):
        """Create an error dialog with optional detailed logs."""
        # Create components
        message_label = Label(message)
        error_banner = ErrorBanner()  # Add error banner for consistency

        if details:
            # Create dialog with error details and logs
            body = HSplit([
                message_label,
                error_banner,
                Label("Details:"),
                TextArea(
                    text=details,
                    read_only=True,
                    scrollbar=True,
                    height=10,
                    width=78
                )
            ])
        else:
            # Simple error message
            body = HSplit([
                message_label,
                error_banner
            ])

        # Create dialog without button handlers first
        dialog = Dialog(
            title="Error",
            body=body,
            buttons=[
                # Create button without handler initially
                SafeButton("OK", handler=None)
            ],
            width=80,
            modal=True
        )

        # Now attach handler after dialog is created
        dialog.buttons[0].handler = lambda: get_app().create_background_task(
            self._dialog_ok(dialog, None)
        )

        return dialog

    async def show_remove_plate_dialog(self, plate: Dict[str, Any]):
        """Show dialog to remove a plate."""
        # Create error banner for validation feedback
        error_banner = ErrorBanner()

        # Create confirmation dialog without button handlers first
        dialog = Dialog(
            title="Confirm Removal",
            body=HSplit([
                Label(f"Remove plate '{plate['name']}'?"),
                error_banner
            ]),
            buttons=[
                # Create buttons without handlers initially
                SafeButton("Yes", handler=None),
                SafeButton("No", handler=None)
            ],
            width=50,
            modal=True
        )

        # Now attach handlers after dialog is created
        dialog.buttons[0].handler = lambda: get_app().create_background_task(
            self._dialog_ok(dialog, True)
        )
        dialog.buttons[1].handler = lambda: get_app().create_background_task(
            self._dialog_cancel(dialog)
        )

        # Show dialog and wait for result
        result = await self._show_dialog(dialog)

        # Notify parent component of result
        if result:
            await self.on_remove_dialog_result(plate)