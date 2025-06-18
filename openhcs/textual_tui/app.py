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
from .widgets.menu_bar import MenuBar
from .widgets.status_bar import StatusBar
from .widgets.floating_window import BaseFloatingWindow

logger = logging.getLogger(__name__)


class ErrorDialog(BaseFloatingWindow):
    """Error dialog with syntax highlighting using global floating window system."""

    def __init__(self, error_message: str, error_details: str = ""):
        self.error_message = error_message
        self.error_details = error_details
        super().__init__(title="🚨 ERROR")

    def compose_content(self) -> ComposeResult:
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

    def compose_buttons(self) -> ComposeResult:
        """Provide Close button."""
        yield Button("Close", id="close", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Close button dismisses dialog."""
        return False  # Dismiss with False result

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
        
        logger.info("OpenHCSTUIApp initialized with Textual reactive system")
    
    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield MenuBar(global_config=self.global_config)
        yield MainContent(
            filemanager=self.filemanager,
            global_config=self.global_config
        )
        yield StatusBar()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        logger.info("OpenHCS TUI mounted and ready")
        self.current_status = "OpenHCS TUI Ready"

        # Status bar will automatically show this log message
        # No need to manually update it anymore

    def watch_current_status(self, status: str) -> None:
        """Watch for status changes and log them (status bar will show automatically)."""
        # Log the status change - status bar will pick it up automatically
        logger.info(status)
    
    def action_quit(self) -> None:
        """Handle quit action."""
        logger.info("OpenHCS TUI shutting down")
        self.exit()

    def show_error(self, error_message: str, exception: Exception = None) -> None:
        """Show a global error dialog with optional exception details."""
        error_details = ""
        if exception:
            error_details = f"Exception: {type(exception).__name__}\n"
            error_details += f"Message: {str(exception)}\n\n"
            error_details += "Traceback:\n"
            error_details += traceback.format_exc()

        logger.error(f"Global error: {error_message}", exc_info=exception)

        # Show error dialog
        error_dialog = ErrorDialog(error_message, error_details)
        self.push_screen(error_dialog)

    def _handle_exception(self, error: Exception) -> None:
        """Let exceptions bubble up to global handler instead of silencing them."""
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
