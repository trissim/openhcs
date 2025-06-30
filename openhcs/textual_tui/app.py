"""
OpenHCS Textual TUI Main Application

A modern terminal user interface built with Textual framework.
This is the main application class that orchestrates the entire TUI.
"""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Static, Button, TextArea
from rich.syntax import Syntax

# OpenHCS imports
from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.io.base import storage_registry
from openhcs.io.filemanager import FileManager

# Widget imports (will be created)
from .widgets.main_content import MainContent
from .widgets.status_bar import StatusBar

# Textual-window imports
from textual_window import Window, WindowSwitcher
from openhcs.textual_tui.widgets.custom_window_bar import CustomWindowBar
from openhcs.textual_tui.windows import HelpWindow, ConfigWindow, DualEditorWindow, PipelinePlateWindow
from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

logger = logging.getLogger(__name__)


class ErrorDialog(BaseOpenHCSWindow):
    """Error dialog with syntax highlighting using textual-window system."""

    def __init__(self, error_message: str, error_details: str = ""):
        self.error_message = error_message
        self.error_details = error_details
        super().__init__(
            window_id="error_dialog",
            title="🚨 ERROR",
            mode="temporary"
        )

    def compose(self) -> ComposeResult:
        """Compose the error dialog content."""
        # Error message
        yield Static(self.error_message, classes="error-message", markup=False)

        # Error details with syntax highlighting if available
        if self.error_details:
            yield TextArea(
                text=self.error_details,
                language="python",  # Python syntax highlighting for tracebacks
                theme="monokai",
                read_only=True,  # Make it read-only but selectable
                show_line_numbers=True,
                soft_wrap=True,
                id="error_content"
            )

        # Close button
        with Container(classes="dialog-buttons"):
            yield Button("Close", id="close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.close_window()

    DEFAULT_CSS = """
    .error-message {
        color: $error;
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
    }

    #error_content {
        height: auto;
        margin: 0;
        max-height: 20;
        min-height: 10;
        border: solid $primary;
    }
    """


class OpenHCSTUIApp(App):
    """
    Main OpenHCS Textual TUI Application.

    This app provides a complete interface for OpenHCS pipeline management
    with proper reactive state management and clean architectural boundaries.
    """
    CSS_PATH = "styles.css"

    # Blocking window for pseudo-modal behavior
    blocking_window = None
#    CSS = """
#    /* General dialog styling */
#    .dialog {
#        background: $surface;
#        border: tall $primary;
#        padding: 1 2;
#        width: auto;
#        height: auto;
#    }
#
#    /* SelectionList styling - remove circles, keep highlighting */
#    SelectionList > .selection-list--option {
#        padding-left: 1;
#        text-style: none;
#    }
#
#    SelectionList > .selection-list--option-highlighted {
#        background: $accent;
#        color: $text;
#    }
#
#    /* MenuBar */
#    MenuBar {
#        height: 3;
#        border: solid white;
#    }
#
#    /* All buttons - uniform height and styling */
#    Button {
#        height: 1;
#    }
#
#    /* MenuBar buttons - content-based width */
#    MenuBar Button {
#        margin: 0 1;
#        width: auto;
#    }
#
#    /* MenuBar title - properly centered */
#    MenuBar Static {
#        text-align: center;
#        content-align: center middle;
#        width: 1fr;
#        text-style: bold;
#    }
#
#    /* Function list header buttons - specific styling for spacing */
#    #function_list_header Button {
#        margin: 0 1; /* 0 vertical, 1 horizontal margin */
#        width: auto; /* Let buttons size to their content */
#    }
#
#    /* Main content containers with proper borders and responsive sizing */
#    #plate_manager_container {
#        border: solid white;
#        width: 1fr;
#        min-width: 0;
#    }
#
#    #pipeline_editor_container {
#        border: solid white;
#        width: 1fr;
#        min-width: 0;
#    }
#
#    /* StatusBar */
#    StatusBar {
#        height: 3;
#        border: solid white;
#    }
#
#    /* Button containers - full width */
#    #plate_manager_container Horizontal,
#    #pipeline_editor_container Horizontal {
#        width: 100%;
#    }
#
#    /* Content area buttons - responsive width distribution */
#    #plate_manager_container Button,
#    #pipeline_editor_container Button {
#        width: 1fr;
#        margin: 0;
#        min-width: 0;
#    }
#
#    /* App fills terminal height properly */
#    OpenHCSTUIApp {
#        height: 100vh;
#    }
#
#    /* Main content layout fills remaining space and is responsive */
#    MainContent {
#        height: 1fr;
#        width: 100%;
#    }
#
#    /* Main horizontal layout is responsive */
#    MainContent > Horizontal {
#        width: 100%;
#        height: 100%;
#    }
#
#
#    /* Content areas adapt to available space */
#    ScrollableContainer {
#        height: 1fr;
#    }
#
#    /* Static content styling */
#    Static {
#        text-align: center;
#    }
#
#
#    """
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("f1", "toggle_window_switcher", "Switch Windows"),
    ]
    
    # App-level reactive state
    current_status = reactive("Ready")
    
    def __init__(self, global_config: Optional[GlobalPipelineConfig] = None):
        """
        Initialize the OpenHCS TUI App.
        
        Args:
            global_config: Global configuration (uses default if None)
        """
        super().__init__()
        
        # Core configuration - minimal TUI responsibility
        self.global_config = global_config or get_default_global_config()
        
        # Create shared components (pattern from SimpleOpenHCSTUILauncher)
        self.storage_registry = storage_registry
        self.filemanager = FileManager(self.storage_registry)

        # Toolong compatibility attributes
        self.save_merge = None  # For Toolong LogView compatibility

        logger.info("OpenHCSTUIApp initialized with Textual reactive system")
    
    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        # TEMPORARILY DISABLED - testing if WindowSwitcher causes hangs
        # yield WindowSwitcher()  # Invisible Alt-Tab overlay

        # Custom WindowBar with no left button
        yield CustomWindowBar(dock="bottom", start_open=True)



        # Status bar for status messages
        yield StatusBar()

        # Main content fills the rest
        yield MainContent(
            filemanager=self.filemanager,
            global_config=self.global_config
        )
    
    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        logger.info("OpenHCS TUI mounted and ready")
        self.current_status = "OpenHCS TUI Ready"

        # Status bar will automatically show this log message
        # No need to manually update it anymore

        # Add our start menu button to the WindowBar using the same pattern as window buttons
        logger.info("🚀 APP MOUNT: About to add start menu button")
        try:
            await self._add_start_menu_button()
            logger.info("🚀 APP MOUNT: Start menu button added")
        except Exception as e:
            logger.error(f"🚀 APP MOUNT: Start menu button failed: {e}")
            # Continue without start menu button for now

    def watch_current_status(self, status: str) -> None:
        """Watch for status changes and log them (status bar will show automatically)."""
        # Log the status change - status bar will pick it up automatically
        logger.info(status)
    
    def action_quit(self) -> None:
        """Handle quit action."""
        logger.info("OpenHCS TUI shutting down")
        self.exit()

    def action_toggle_window_switcher(self):
        """Toggle the window switcher."""
        switcher = self.query_one(WindowSwitcher)
        switcher.action_toggle()  # Correct textual-window API method

    async def _add_start_menu_button(self):
        """Add our start menu button to the WindowBar at the leftmost position."""
        try:
            logger.info("🚀 START MENU: Creating start menu button")
            from openhcs.textual_tui.widgets.start_menu_button import StartMenuButton

            # Get the CustomWindowBar
            logger.info("🚀 START MENU: Getting CustomWindowBar")
            window_bar = self.query_one(CustomWindowBar)
            logger.info(f"🚀 START MENU: Found window bar: {window_bar}")

            # Check if right button exists (no left button in CustomWindowBar)
            logger.info("🚀 START MENU: Looking for right button")
            right_button = window_bar.query_one("#windowbar_button_right")
            logger.info(f"🚀 START MENU: Found right button: {right_button}")

            # Add our start menu button at the very beginning (leftmost position)
            # Mount before the right button to be at the far left
            logger.info("🚀 START MENU: Creating StartMenuButton")
            start_button = StartMenuButton(window_bar=window_bar, id="start_menu_button")
            logger.info(f"🚀 START MENU: Created start button: {start_button}")

            logger.info("🚀 START MENU: Mounting start button")
            await window_bar.mount(start_button, before=right_button)
            logger.info("🚀 START MENU: Start menu button mounted successfully")

        except Exception as e:
            logger.error(f"🚀 START MENU: Failed to add start menu button: {e}")
            import traceback
            logger.error(f"🚀 START MENU: Traceback: {traceback.format_exc()}")
            raise



    def open_blocking_window(self, window_class, *args, **kwargs):
        """Open a blocking window that disables main UI interactions."""
        if self.blocking_window:
            return  # Only allow one blocking window at a time

        window = window_class(*args, **kwargs)
        self.blocking_window = window
        self._disable_main_interactions()
        self.mount(window)
        return window

    def _disable_main_interactions(self):
        """Disable main UI interactions when modal window is open."""
        # Note: MenuBar removed - interactions now handled by start menu
        pass

    def _enable_main_interactions(self):
        """Re-enable main UI interactions when modal window closes."""
        # Note: MenuBar removed - interactions now handled by start menu
        pass

    def on_window_closed(self, event: Window.Closed) -> None:
        """Handle window closed events from textual-window."""
        # Check if this is our blocking window
        # Event has window reference through WindowMessage base
        if event.control == self.blocking_window:
            self.blocking_window = None
            self._enable_main_interactions()

    def show_error(self, error_message: str, exception: Exception = None) -> None:
        """Show a global error dialog with optional exception details."""
        error_details = ""
        if exception:
            error_details = f"Exception: {type(exception).__name__}\n"
            error_details += f"Message: {str(exception)}\n\n"
            error_details += "Traceback:\n"
            error_details += traceback.format_exc()

        logger.error(f"Global error: {error_message}", exc_info=exception)

        # Show error dialog using window system
        from textual.css.query import NoMatches

        try:
            # Check if error dialog already exists
            window = self.query_one(ErrorDialog)
            # Update existing dialog
            window.error_message = error_message
            window.error_details = error_details
            window.open_state = True
        except NoMatches:
            # Create new error dialog window
            error_dialog = ErrorDialog(error_message, error_details)
            self.run_worker(self._mount_error_dialog(error_dialog))

    async def _mount_error_dialog(self, error_dialog):
        """Mount error dialog window."""
        await self.mount(error_dialog)
        error_dialog.open_state = True

    def _handle_exception(self, error: Exception) -> None:
        """Handle exceptions with special cases for Toolong internal errors."""
        # Check for known Toolong internal timing errors that are non-fatal
        error_str = str(error)
        if (
            "No nodes match" in error_str and
            ("FindDialog" in error_str or "Label" in error_str) and
            ("InfoOverlay" in error_str or "LogView" in error_str)
        ):
            # This is a known Toolong internal timing issue - log but don't crash
            logger.warning(f"Ignoring Toolong internal timing error: {error_str}")
            return

        # Log the error for debugging
        logger.error(f"Unhandled exception in TUI: {str(error)}", exc_info=True)

        # Re-raise the exception to let it crash loudly
        # This allows the global error handler to catch it
        raise error

    async def _on_exception(self, error: Exception) -> None:
        """Let async exceptions bubble up."""
        self._handle_exception(error)

    def _on_unhandled_exception(self, error: Exception) -> None:
        """Let unhandled exceptions bubble up."""
        self._handle_exception(error)

    async def on_unmount(self) -> None:
        """Clean up when app is shutting down with aggressive thread cleanup."""
        logger.info("OpenHCS TUI app unmounting, cleaning up threads...")

        # Force cleanup of any ReactiveLogMonitor instances
        try:
            from openhcs.textual_tui.widgets.reactive_log_monitor import ReactiveLogMonitor
            monitors = self.query(ReactiveLogMonitor)
            for monitor in monitors:
                monitor.stop_monitoring()
        except Exception as e:
            logger.debug(f"Error cleaning up ReactiveLogMonitors: {e}")

        # Force immediate exit if threads don't stop cleanly
        try:
            import threading
            import time
            time.sleep(0.1)  # Give threads a moment to stop
            active_threads = [t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()]
            if active_threads:
                logger.warning(f"Found {len(active_threads)} active threads during shutdown")
                # Don't try to modify running threads - just log and continue
        except Exception as e:
            logger.debug(f"Error checking threads: {e}")

        logger.info("OpenHCS TUI app cleanup complete")


async def main():
    """
    Main entry point for the OpenHCS Textual TUI.

    This function handles initialization and runs the application.
    Note: Logging is setup by the main entry point, not here.
    """
    logger.info("Starting OpenHCS Textual TUI from app.py...")

    try:
        # Load configuration with cache support
        from openhcs.textual_tui.services.global_config_cache import load_cached_global_config
        global_config = await load_cached_global_config()

        # REMOVED: setup_global_gpu_registry - this is now ONLY done in __main__.py
        # to avoid duplicate initialization
        logger.info("Using global_config with GPU registry already initialized by __main__.py")

        # Create and run the app
        app = OpenHCSTUIApp(global_config=global_config)
        await app.run_async()
        
    except KeyboardInterrupt:
        logger.info("TUI terminated by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
    finally:
        logger.info("OpenHCS Textual TUI finished")


if __name__ == "__main__":
    asyncio.run(main())
