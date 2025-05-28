"""
Dialog Service - Centralized Dialog Management.

Handles all dialog operations in a consistent, reusable manner.
Provides a clean API for showing various types of dialogs.

🔒 Clause 295: Component Boundaries
Clear separation between dialog service and UI components.
"""
import asyncio
import logging
from typing import Any, Dict, Optional, Callable

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import FloatContainer, Float
from prompt_toolkit.widgets import Dialog, Label, Button
from prompt_toolkit.layout.containers import HSplit

logger = logging.getLogger(__name__)


class DialogService:
    """
    Service for managing dialogs in the TUI application.
    
    Provides:
    - Consistent dialog lifecycle management
    - Standard dialog types (error, confirmation, info)
    - Proper focus and layout handling
    - Resource cleanup
    """
    
    def __init__(self, state):
        self.state = state
        self.active_dialogs = []
        self.dialog_lock = asyncio.Lock()
    
    async def show_error_dialog(self, title: str, message: str, details: str = "") -> None:
        """
        Show an error dialog.
        
        Args:
            title: Dialog title
            message: Error message
            details: Optional detailed error information
        """
        async with self.dialog_lock:
            try:
                # Create dialog content
                content_parts = [Label(message)]
                
                if details:
                    content_parts.extend([
                        Label(""),  # Spacer
                        Label("Details:"),
                        Label(details, style="class:error-details")
                    ])
                
                # Create dialog
                dialog = Dialog(
                    title=title,
                    body=HSplit(content_parts),
                    buttons=[
                        Button("OK", handler=lambda: self._close_dialog(dialog))
                    ],
                    width=80,
                    modal=True
                )
                
                await self._show_dialog(dialog)
                
            except Exception as e:
                logger.error(f"DialogService: Error showing error dialog: {e}", exc_info=True)
    
    async def show_confirmation_dialog(self, title: str, message: str, default_yes: bool = False) -> bool:
        """
        Show a confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            default_yes: Whether "Yes" should be the default button
            
        Returns:
            True if user confirmed, False otherwise
        """
        async with self.dialog_lock:
            try:
                result_future = asyncio.Future()
                
                # Create buttons
                yes_button = Button(
                    "Yes", 
                    handler=lambda: result_future.set_result(True)
                )
                no_button = Button(
                    "No", 
                    handler=lambda: result_future.set_result(False)
                )
                
                buttons = [yes_button, no_button] if default_yes else [no_button, yes_button]
                
                # Create dialog
                dialog = Dialog(
                    title=title,
                    body=HSplit([Label(message)]),
                    buttons=buttons,
                    width=60,
                    modal=True
                )
                
                # Show dialog and wait for result
                await self._show_dialog_with_result(dialog, result_future)
                return await result_future
                
            except Exception as e:
                logger.error(f"DialogService: Error showing confirmation dialog: {e}", exc_info=True)
                return False
    
    async def show_info_dialog(self, title: str, message: str) -> None:
        """
        Show an information dialog.
        
        Args:
            title: Dialog title
            message: Information message
        """
        async with self.dialog_lock:
            try:
                # Create dialog
                dialog = Dialog(
                    title=title,
                    body=HSplit([Label(message)]),
                    buttons=[
                        Button("OK", handler=lambda: self._close_dialog(dialog))
                    ],
                    width=60,
                    modal=True
                )
                
                await self._show_dialog(dialog)
                
            except Exception as e:
                logger.error(f"DialogService: Error showing info dialog: {e}", exc_info=True)
    
    async def show_custom_dialog(self, dialog, result_future: Optional[asyncio.Future] = None):
        """
        Show a custom dialog.
        
        Args:
            dialog: The dialog to show
            result_future: Optional future to wait for result
            
        Returns:
            Result from the future if provided, None otherwise
        """
        async with self.dialog_lock:
            try:
                if result_future:
                    await self._show_dialog_with_result(dialog, result_future)
                    return await result_future
                else:
                    await self._show_dialog(dialog)
                    return None
                    
            except Exception as e:
                logger.error(f"DialogService: Error showing custom dialog: {e}", exc_info=True)
                return None
    
    async def _show_dialog(self, dialog):
        """Show a dialog without waiting for a specific result."""
        result_future = asyncio.Future()
        
        # Set up close handler
        original_close = getattr(dialog, '_close_handler', None)
        
        def close_handler():
            if original_close:
                original_close()
            if not result_future.done():
                result_future.set_result(None)
        
        dialog._close_handler = close_handler
        
        await self._show_dialog_with_result(dialog, result_future)
        await result_future
    
    async def _show_dialog_with_result(self, dialog, result_future: asyncio.Future):
        """Show a dialog and manage its lifecycle."""
        app = get_app()
        
        # Store current layout
        previous_layout = app.layout
        previous_focus = app.layout.current_window if hasattr(app.layout, 'current_window') else None
        
        try:
            # Add to active dialogs
            self.active_dialogs.append(dialog)
            
            # Create float container
            float_container = FloatContainer(
                content=previous_layout.container,
                floats=[
                    Float(
                        content=dialog,
                        transparent=False,
                    )
                ]
            )
            
            # Set new layout
            app.layout = Layout(float_container)
            
            # Focus the dialog
            app.layout.focus(dialog)
            
            # Wait for dialog to complete
            await result_future
            
        finally:
            # Clean up
            if dialog in self.active_dialogs:
                self.active_dialogs.remove(dialog)
            
            # Restore previous layout
            app.layout = previous_layout
            
            # Restore focus
            if previous_focus:
                try:
                    app.layout.focus(previous_focus)
                except Exception:
                    pass  # Focus restoration is best-effort
    
    def _close_dialog(self, dialog):
        """Close a specific dialog."""
        if hasattr(dialog, '_close_handler'):
            dialog._close_handler()
    
    async def close_all_dialogs(self):
        """Close all active dialogs."""
        async with self.dialog_lock:
            for dialog in list(self.active_dialogs):
                self._close_dialog(dialog)
            self.active_dialogs.clear()
    
    def has_active_dialogs(self) -> bool:
        """Check if there are any active dialogs."""
        return len(self.active_dialogs) > 0
    
    async def shutdown(self):
        """Clean up the dialog service."""
        logger.info("DialogService: Shutting down...")
        await self.close_all_dialogs()
        logger.info("DialogService: Shutdown complete")
