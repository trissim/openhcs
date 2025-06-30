"""
Toolong Log Viewer Window for OpenHCS TUI

A window that displays OpenHCS logs using the reactive log monitoring system.
Provides professional log viewing with syntax highlighting, search, and live tailing.
"""

import logging
from pathlib import Path
from typing import List

from textual.app import ComposeResult
from textual.widgets import Static

# Import OpenHCS window base class
from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

# Import our reactive log monitor
from openhcs.textual_tui.widgets.reactive_log_monitor import ReactiveLogMonitor

# Import Toolong components for LogViewWrapper
from toolong.log_view import LogView



logger = logging.getLogger(__name__)


class ToolongWindow(BaseOpenHCSWindow):
    """
    Window that displays OpenHCS logs using Toolong.

    Features:
    - Professional log viewing with syntax highlighting
    - Live tailing of active log files
    - Search and filtering capabilities
    - Multi-file support with tabs
    - Merge view for combined log analysis
    """



    DEFAULT_CSS = """
    ToolongWindow {
        width: 80;
        height: 25;
        min-width: 60;
        min-height: 15;
    }

    .error-message {
        color: $error;
        text-style: bold;
        text-align: center;
        padding: 2;
    }
    """
    
    def __init__(self, log_files: List[str] = None, base_log_path: str = "", **kwargs):
        """
        Initialize the Toolong window.

        Args:
            log_files: List of log file paths to display (legacy support)
            base_log_path: Base path for reactive log monitoring
        """
        # Determine base path from log files if not provided
        if not base_log_path and log_files:
            # Extract base path from first log file
            first_log = Path(log_files[0])
            if first_log.suffix == '.log':
                base_log_path = str(first_log.with_suffix(''))

        # Determine window title
        if base_log_path:
            base_name = Path(base_log_path).name
            title = f"Log Viewer - {base_name}"
        elif log_files:
            if len(log_files) == 1:
                title = f"Log Viewer - {Path(log_files[0]).name}"
            else:
                title = f"Log Viewer - {len(log_files)} files"
        else:
            title = "Log Viewer"

        super().__init__(
            window_id="toolong_viewer",
            title=title,
            mode="temporary",
            **kwargs
        )
        self.base_log_path = base_log_path
        self.log_files = log_files or []
        
    def compose(self) -> ComposeResult:
        """Compose the Toolong window layout using ReactiveLogMonitor."""
        try:
            # Always use ReactiveLogMonitor - it can show TUI logs even without subprocess
            yield ReactiveLogMonitor(
                base_log_path=self.base_log_path,
                auto_start=True,
                include_tui_log=True  # Always show TUI log
            )

        except Exception as e:
            logger.error(f"Failed to create ReactiveLogMonitor: {e}")
            yield Static(
                f"Error loading log viewer: {e}\n\n" +
                "Please check that log files exist and are readable.",
                classes="error-message"
            )
    
    async def on_mount(self) -> None:
        """Set up the window when mounted."""
        # ReactiveLogMonitor handles its own setup
        logger.info(f"Toolong window opened with base path: {self.base_log_path}")



    async def on_unmount(self) -> None:
        """Clean up when window is unmounted."""
        logger.info("Toolong window unmounting, ensuring cleanup...")

        # Explicitly stop ReactiveLogMonitor to ensure thread cleanup
        try:
            reactive_monitor = self.query_one(ReactiveLogMonitor)
            if reactive_monitor:
                reactive_monitor.stop_monitoring()
        except Exception as e:
            logger.debug(f"Could not find ReactiveLogMonitor to stop: {e}")

        logger.info("Toolong window closed")
