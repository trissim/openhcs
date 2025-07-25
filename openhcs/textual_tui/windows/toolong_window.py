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

# Import our OpenHCS Toolong widget
from openhcs.textual_tui.widgets.openhcs_toolong_widget import OpenHCSToolongWidget



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
    
    def __init__(self, base_log_path: str = "", **kwargs):
        """
        Initialize the Toolong window.

        Args:
            base_log_path: Base path for subprocess log monitoring (optional)
        """
        super().__init__(
            window_id="toolong_viewer",
            title="Log Viewer",
            mode="permanent",  # Make it permanent so it can't be closed, only minimized
            allow_maximize=True,
            **kwargs
        )

        # Find current TUI log file
        self.tui_log_file = self._find_current_tui_log()

        # Store base_log_path for reference
        self.base_log_path = base_log_path

        # Determine logs directory for file watching
        if base_log_path:
            # Use directory of base_log_path for watching subprocess logs
            self.logs_directory = Path(base_log_path).parent
        elif self.tui_log_file:
            # Fall back to TUI log directory
            self.logs_directory = Path(self.tui_log_file).parent
        else:
            self.logs_directory = None

    def _find_current_tui_log(self) -> str:
        """Find the current TUI process log file (not subprocess or worker logs)."""
        import glob

        # Look for TUI log files (exclude subprocess and worker logs)
        # Use cross-platform path resolution instead of hardcoded path
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_pattern = str(log_dir / "openhcs_unified_*.log")
        all_logs = sorted(glob.glob(log_pattern))

        # Filter to only actual TUI logs (not subprocess or worker logs)
        tui_logs = []
        for log_file in all_logs:
            log_name = Path(log_file).name
            # TUI logs don't contain "subprocess" or "worker" in the name
            if "_subprocess_" not in log_name and "_worker_" not in log_name:
                tui_logs.append(log_file)

        if tui_logs:
            # Return the most recent TUI log
            logger.debug(f"Found {len(tui_logs)} TUI logs, using most recent: {Path(tui_logs[-1]).name}")
            return tui_logs[-1]
        else:
            logger.warning("No TUI log files found")
            return None

    def _find_session_logs(self) -> List[str]:
        """Find all logs belonging to the current TUI session."""
        import glob

        if not self.tui_log_file:
            return []

        # Extract session base from TUI log
        # openhcs_unified_20250630_092636.log -> openhcs_unified_20250630_092636
        tui_base = Path(self.tui_log_file).stem

        # Find all logs that start with this session base
        log_pattern = str(self.logs_directory / f"{tui_base}*.log")
        session_logs = sorted(glob.glob(log_pattern))

        logger.debug(f"Found {len(session_logs)} logs for session '{tui_base}':")
        for log_file in session_logs:
            logger.debug(f"  - {Path(log_file).name}")

        return session_logs

    def _find_all_openhcs_logs(self) -> List[str]:
        """Find all existing OpenHCS log files."""
        import glob

        if not self.logs_directory:
            return []

        # Look for all OpenHCS log files
        log_pattern = str(self.logs_directory / "openhcs_*.log")
        all_logs = sorted(glob.glob(log_pattern))

        logger.debug(f"Found {len(all_logs)} existing OpenHCS log files")
        for log_file in all_logs:
            logger.debug(f"  - {Path(log_file).name}")

        return all_logs

    def compose(self) -> ComposeResult:
        """Compose the Toolong window layout using OpenHCSToolongWidget."""
        try:
            # Find all logs for current session
            session_logs = self._find_session_logs()

            if session_logs:
                # Start with all existing session logs, watch for new ones
                yield OpenHCSToolongWidget(
                    file_paths=session_logs,  # All session logs
                    show_tabs=False,  # Hide tabs, use dropdown only
                    show_dropdown=True,  # Show dropdown selector
                    show_controls=True,  # Show control buttons
                    base_log_path=str(self.logs_directory) if self.logs_directory else None
                )
            else:
                yield Static("No session log files found", classes="error-message")

        except Exception as e:
            logger.error(f"Failed to create OpenHCSToolongWidget: {e}")
            yield Static(
                f"Error loading log viewer: {e}\n\n" +
                "Please check that log files exist and are readable.",
                classes="error-message"
            )
    
    async def on_mount(self) -> None:
        """Set up the window when mounted."""
        # ReactiveLogMonitor handles its own setup
        logger.debug(f"Toolong window opened with base path: {self.base_log_path}")



    async def on_unmount(self) -> None:
        """Clean up when window is unmounted."""
        logger.debug("Toolong window unmounting, ensuring cleanup...")

        # Explicitly stop ReactiveLogMonitor to ensure thread cleanup
        try:
            reactive_monitor = self.query_one(ReactiveLogMonitor)
            if reactive_monitor:
                reactive_monitor.stop_monitoring()
        except Exception as e:
            logger.debug(f"Could not find ReactiveLogMonitor to stop: {e}")

        logger.debug("Toolong window closed")


def clear_toolong_logs(app):
    """Clear subprocess logs from the singleton toolong window (mounted at startup)."""
    try:
        # Find the singleton toolong window (should always exist)
        window = app.query_one(ToolongWindow)
        logger.info("Found singleton toolong window")

        # Find the widget inside the window
        from openhcs.textual_tui.widgets.openhcs_toolong_widget import OpenHCSToolongWidget
        widgets = window.query(OpenHCSToolongWidget)
        for widget in widgets:
            logger.info("Clearing logs from singleton toolong window")
            widget._clear_all_logs_except_tui()
            logger.info("Logs cleared from singleton toolong window")
    except Exception as e:
        logger.error(f"Failed to clear logs from singleton toolong window: {e}")
        import traceback
        logger.error(traceback.format_exc())
