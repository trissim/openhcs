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
from typing import Any, Dict, Optional, Protocol

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import (ConditionalContainer, Container, Float,
                                   HSplit)
from prompt_toolkit.widgets import Box, Button, Dialog, Label
from prompt_toolkit.widgets import RadioList as Dropdown
from prompt_toolkit.widgets import TextArea


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
        backend_registry=None
    ):
        """
        Initialize the dialog manager.

        Args:
            on_add_dialog_result: Callback for add dialog result
            on_remove_dialog_result: Callback for remove dialog result
            on_error: Callback for error handling
            backend_registry: BackendRegistry instance for backend options
        """
        self.on_add_dialog_result = on_add_dialog_result
        self.on_remove_dialog_result = on_remove_dialog_result
        self.on_error = on_error
        self.backend_registry = backend_registry

    async def show_add_plate_dialog(self) -> None:
        """Show dialog to add a plate."""
        # Create file browser dialog with backend selection
        dialog = self._create_file_browser_dialog()

        # Show dialog and wait for result
        result = await self._show_dialog(dialog)

        if result and isinstance(result, dict) and 'path' in result and 'backend' in result:
            path = result['path']
            backend = result['backend']

            # 🔒 Clause 316: Strict validation for backend
            # Simplified check: backend must be a non-empty string
            if not (isinstance(backend, str) and backend):
                await self._show_error_dialog(
                    "Backend selection required",
                    "A valid storage backend must be explicitly selected."
                )
                return

            # Notify parent component of result
            await self.on_add_dialog_result({
                'path': path,
                'backend': backend
            })

    def _create_file_browser_dialog(self) -> Dialog:
        """Create a file browser dialog with backend selection."""
        # Path input
        path_input = TextArea(
            text="",  # Empty default to force explicit input
            multiline=False
        )

        # 🔒 Clause 316: Must use current_value, not selected_option
        # Initialize with empty string to force explicit selection
        backend_options = []
        if self.backend_registry:
            # 🔒 Clause 316: PTK expects (label, value) order
            backend_options = [(backend, backend) for backend in self.backend_registry.get_available_backends()]

        # Fallback to standard backends if registry not available
        if not backend_options:
            # 🔒 Clause 316: PTK expects (label, value) order
            backend_options = [
                ("Local Disk", "disk"),  # (label, value)
                ("Memory", "memory"),    # (label, value)
                ("Zarr Store", "zarr")   # (label, value)
            ]

        # 🔒 Clause 316: Must use current_value for initialization
        # prompt-toolkit Dropdown takes current_value, not initial_option
        # Initialize with first option to prevent runtime crash (Clause 316/231)
        # But disable OK button until user explicitly changes selection
        first_option_value = backend_options[0][1] if backend_options else None
        backend_selector = Dropdown(
            options=backend_options,
            current_value=first_option_value  # Initialize with first option to prevent crash
        )
        
        # Track if user has explicitly selected a backend
        user_selected = {'value': False}
        
        # Create a change handler to track user selection
        original_handle_mouse = backend_selector.handle_mouse
        def selection_tracking_handler(mouse_event):
            result = original_handle_mouse(mouse_event)
            if result:
                user_selected['value'] = True
            return result
        backend_selector.handle_mouse = selection_tracking_handler

        # Create error banner for validation feedback
        error_banner = ErrorBanner()

        # Create dialog without button handlers first
        dialog = Dialog(
            title="Select Plate Directory",
            body=HSplit([
                Label("Enter path to plate directory:"),
                path_input,
                Label("Select storage backend (required):"),
                backend_selector,
                error_banner
            ]),
            buttons=[
                # Create buttons without handlers initially
                # OK button will be enabled only after user explicitly selects backend
                # 🔒 Clause 231: Deferred-Default Enforcement
                ok_button = Button(
                    "OK",
                    handler=None,
                    # Disable until user explicitly selects backend AND backend has a value
                    filter=Condition(lambda: user_selected['value'] and bool(backend_selector.current_value))
                )
                cancel_button = Button("Cancel", handler=None)
            ],
            width=80,
            modal=True
        )

        # Now attach handlers after dialog is created
        # 🔒 Clause 231: Deferred-Default Enforcement
        # 🔒 Clause 88: No-Inference
        # Capture backend_selector at button press time, not dialog creation time
        ok_button.handler = lambda: self._handle_ok_button_press(
            dialog, path_input, backend_selector, user_selected
        )
        cancel_button.handler = lambda: get_app().create_background_task(
            self._dialog_cancel(dialog)
        )
        
        # Set buttons on dialog
        dialog.buttons = [ok_button, cancel_button]

        return dialog

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

    def _handle_ok_button_press(self, dialog, path_input, backend_selector, user_selected):
        """
        Handle OK button press with immediate validation.

        Args:
            dialog: The dialog instance
            path_input: The path input TextArea
            backend_selector: The backend selector Dropdown
            user_selected: Dictionary tracking if user has explicitly selected backend
        """
        # Get current values at button press time
        path = path_input.text
        backend = backend_selector.current_value

        # Find the error banner in the dialog using recursive search
        error_banner = ErrorBanner.find_in_container(dialog)
        
        # Verify user has explicitly selected a backend
        if not user_selected['value']:
            if error_banner:
                error_banner.show("Backend must be explicitly chosen.")
            return

        # 🔒 Clause 231: Immediate validation at button press time
        # Validate path is non-empty
        if not path.strip():
            if error_banner:
                error_banner.show("Path cannot be empty.")
            return
            
        # 🔒 Clause 319: TUI_NO_VIRTUALPATH_EXPOSURE
        # Validate path is absolute and doesn't contain a scheme (no VFS URIs)
        path_obj = Path(path)
        if "://" in path:
            if error_banner:
                error_banner.show("Path cannot contain URI schemes (e.g., zarr://).")
            return
            
        # Validate path is absolute to prevent VFS injection
        if not path_obj.is_absolute():
            if error_banner:
                error_banner.show("Path must be an absolute filesystem path.")
            return
            
        # Validate path exists and is a directory
        if not path_obj.is_dir():
            if error_banner:
                error_banner.show("Directory does not exist or is unreadable.")
            return

        # Validate backend selection before proceeding
        if not (isinstance(backend, str) and backend):
            # Show error in banner and abort
            if error_banner:
                error_banner.show("Backend must be explicitly chosen.")
            return

        # Hide error banner if validation passes
        if error_banner:
            error_banner.hide()

        # Schedule the async dialog OK handler
        get_app().create_background_task(
            self._dialog_ok(
                dialog,
                {
                    'path': path,
                    'backend': backend
                }
            )
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
                Button("OK", handler=None)
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
                Button("Yes", handler=None),
                Button("No", handler=None)
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
```