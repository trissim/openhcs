"""
MainContent Widget for OpenHCS Textual TUI

Main content area with horizontal split between PlateManager and PipelineEditor.
Matches the layout from the current prompt-toolkit TUI.
"""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from textual.widget import Widget
from textual.css.query import NoMatches
from textual.reactive import reactive

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from .system_monitor import SystemMonitorTextual

from .plate_manager import PlateManagerWidget
from .pipeline_editor import PipelineEditorWidget

logger = logging.getLogger(__name__)


class MainContent(Widget):
    """
    Main content area widget.
    
    Layout: Horizontal split with PlateManager (left) and PipelineEditor (right)
    Uses proper frame containers with titles matching the current TUI.
    """
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """
        Initialize the main content area.

        Args:
            filemanager: FileManager instance for file operations
            global_config: Global configuration (for initial setup only)
        """
        super().__init__()
        self.filemanager = filemanager
        # Note: We don't store global_config as it can become stale
        # Child widgets should use self.app.global_config to get current config
        logger.debug("MainContent initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the main content layout."""
        # Use the system monitor as the main background
        yield SystemMonitorTextual()
    
    def on_mount(self) -> None:
        """Called when the main content is mounted."""
        # Widgets are now in floating windows, no setup needed here
        pass

    def open_pipeline_editor(self):
        """Open the pipeline editor in shared window."""
        window = self._get_or_create_shared_window()
        window.show_pipeline_editor()
        window.open_state = True

    def open_plate_manager(self):
        """Open the plate manager in shared window."""
        window = self._get_or_create_shared_window()
        window.show_plate_manager()
        window.open_state = True

    def _get_or_create_shared_window(self):
        """Get existing shared window or create new one."""
        from openhcs.textual_tui.windows import PipelinePlateWindow

        # Try to find existing window - if it doesn't exist, query_one will raise NoMatches
        try:
            window = self.app.query_one(PipelinePlateWindow)
            return window
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = PipelinePlateWindow(self.filemanager, self.app.global_config)
            self.app.mount(window)
            return window
