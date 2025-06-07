"""
StatusBar Widget for OpenHCS Textual TUI

Bottom status bar with current operation status and messages.
Matches the layout from the current prompt-toolkit TUI.
"""

import logging
from datetime import datetime

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget

logger = logging.getLogger(__name__)


class StatusBar(Widget):
    """
    Bottom status bar widget.
    
    Layout: |————————————————————————— Status |
    Shows current operation status, error messages, and progress.
    """
    
    # Reactive state
    status_message = reactive("Ready")
    last_updated = reactive("")
    
    def __init__(self):
        """Initialize the status bar."""
        super().__init__()
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        logger.debug("StatusBar initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the status bar layout."""
        yield Static(self.get_status_display(), id="status_display")
    
    def get_status_display(self) -> str:
        """Get the formatted status display string."""
        return f"Status: {self.status_message}"
    
    def watch_status_message(self, message: str) -> None:
        """Update the status display when status_message changes."""
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        try:
            status_display = self.query_one("#status_display")
            status_display.update(self.get_status_display())
        except Exception:
            # Widget might not be mounted yet
            pass
    
    def set_status(self, message: str) -> None:
        """
        Set the status message.
        
        Args:
            message: Status message to display
        """
        self.status_message = message
        logger.debug(f"Status updated: {message}")
    
    def set_error(self, error_message: str) -> None:
        """
        Set an error status message.
        
        Args:
            error_message: Error message to display
        """
        self.status_message = f"ERROR: {error_message}"
        logger.error(f"Status error: {error_message}")
    
    def set_info(self, info_message: str) -> None:
        """
        Set an info status message.
        
        Args:
            info_message: Info message to display
        """
        self.status_message = f"INFO: {info_message}"
        logger.info(f"Status info: {info_message}")
    
    def clear_status(self) -> None:
        """Clear the status message back to ready."""
        self.status_message = "Ready"
