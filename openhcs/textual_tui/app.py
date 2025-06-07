"""
OpenHCS Textual TUI Main Application

A modern terminal user interface built with Textual framework.
This is the main application class that orchestrates the entire TUI.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.reactive import reactive

# OpenHCS imports
from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.io.base import storage_registry
from openhcs.io.filemanager import FileManager

# Widget imports (will be created)
from .widgets.main_content import MainContent
from .widgets.menu_bar import MenuBar
from .widgets.status_bar import StatusBar

logger = logging.getLogger(__name__)


class OpenHCSTUIApp(App):
    """
    Main OpenHCS Textual TUI Application.

    This app provides a complete interface for OpenHCS pipeline management
    with proper reactive state management and clean architectural boundaries.
    """

    CSS = """
    /* Consistent dialog styling */
    .dialog {
        background: $surface;
        border: tall $primary;
        padding: 1 2;
        width: auto;
        height: auto;
    }

    .dialog-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .dialog-buttons {
        dock: bottom;
        height: 3;
        content-align: center middle;
    }

    /* MenuBar */
    MenuBar {
        height: 3;
        border: solid white;
    }

    /* All buttons - uniform height and styling */
    Button {
        height: 1;
    }

    /* MenuBar buttons - content-based width */
    MenuBar Button {
        margin: 0 1;
        width: auto;
    }

    /* MenuBar title - properly centered */
    MenuBar Static {
        text-align: center;
        content-align: center middle;
        width: 1fr;
        text-style: bold;
    }

    /* Main content containers with proper borders and responsive sizing */
    #plate_manager_container {
        border: solid white;
        width: 1fr;
        min-width: 0;
    }

    #pipeline_editor_container {
        border: solid white;
        width: 1fr;
        min-width: 0;
    }

    /* StatusBar */
    StatusBar {
        height: 3;
        border: solid white;
    }

    /* Button containers - full width */
    #plate_manager_container Horizontal,
    #pipeline_editor_container Horizontal {
        width: 100%;
    }

    /* Content area buttons - responsive width distribution */
    #plate_manager_container Button,
    #pipeline_editor_container Button {
        width: 1fr;
        margin: 0;
        min-width: 0;
    }

    /* App fills terminal height properly */
    OpenHCSTUIApp {
        height: 100vh;
    }

    /* Main content layout fills remaining space and is responsive */
    MainContent {
        height: 1fr;
        width: 100%;
    }

    /* Main horizontal layout is responsive */
    MainContent > Horizontal {
        width: 100%;
        height: 100%;
    }

    /* Content areas adapt to available space */
    ScrollableContainer {
        height: 1fr;
    }

    /* Static content styling */
    Static {
        text-align: center;
    }
    """
    
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
        
        # Update status bar
        status_bar = self.query_one(StatusBar)
        status_bar.status_message = self.current_status
    
    def watch_current_status(self, status: str) -> None:
        """Watch for status changes and update status bar."""
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.status_message = status
        except Exception:
            # Status bar might not be mounted yet
            pass
    
    def action_quit(self) -> None:
        """Handle quit action."""
        logger.info("OpenHCS TUI shutting down")
        self.exit()


async def main():
    """
    Main entry point for the OpenHCS Textual TUI.
    
    This function handles initialization and runs the application.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting OpenHCS Textual TUI...")
    
    try:
        # Load configuration
        global_config = get_default_global_config()
        
        # Setup GPU registry
        setup_global_gpu_registry(global_config=global_config)
        logger.info("GPU registry setup completed")
        
        # Create and run the app
        app = OpenHCSTUIApp(global_config=global_config)
        await app.run_async()
        
    except KeyboardInterrupt:
        logger.info("TUI terminated by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        print(f"ERROR: {e}")
    finally:
        logger.info("OpenHCS Textual TUI finished")


if __name__ == "__main__":
    asyncio.run(main())
