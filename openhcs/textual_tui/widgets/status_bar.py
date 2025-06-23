"""
StatusBar Widget for OpenHCS Textual TUI

Bottom status bar with real-time log streaming.
Shows live log messages from OpenHCS operations.
"""

import logging
import os
from datetime import datetime
from collections import deque
from typing import Optional
from pathlib import Path

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget

logger = logging.getLogger(__name__)


class TUILogHandler(logging.Handler):
    """Custom logging handler that feeds log messages to the TUI status bar."""

    def __init__(self, status_bar: 'StatusBar'):
        super().__init__()
        self.status_bar = status_bar
        self.setLevel(logging.INFO)  # Only show INFO and above in status bar

        # Format for status bar (compact)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        self.setFormatter(formatter)

    def emit(self, record):
        """Emit a log record to the status bar."""
        try:
            msg = self.format(record)
            # Use call_from_thread to safely update UI from any thread
            if hasattr(self.status_bar.app, 'call_from_thread'):
                self.status_bar.app.call_from_thread(self.status_bar.add_log_message, msg, record.levelname)
            else:
                # Fallback for direct calls
                self.status_bar.add_log_message(msg, record.levelname)
        except Exception:
            # Don't let logging errors crash the app
            pass


class StatusBar(Widget):
    """
    Top status bar widget with real-time log streaming.

    Layout: |————————————————————————— Live Log Messages |
    Shows live log messages from OpenHCS operations in real-time.
    """

    # Reactive state
    current_log_message = reactive("Ready")
    last_updated = reactive("")

    def __init__(self, max_history: int = 100):
        """Initialize the status bar with log streaming."""
        super().__init__()
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        self.max_history = max_history
        self.log_history = deque(maxlen=max_history)  # Keep recent log messages
        self.log_handler: Optional[TUILogHandler] = None

        # Log file monitoring for subprocess logs
        self.log_file_path: Optional[str] = None
        self.log_file_position: int = 0
        self.log_monitor_timer = None

        logger.debug("StatusBar initialized with real-time logging")
    
    def compose(self) -> ComposeResult:
        """Compose the status bar layout."""
        yield Static(self.get_log_display(), id="log_display", markup=False)

    def get_log_display(self) -> str:
        """Get the formatted log display string."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {self.current_log_message}"

    def watch_current_log_message(self, message: str) -> None:
        """Update the display when current_log_message changes."""
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        try:
            log_display = self.query_one("#log_display")
            log_display.update(self.get_log_display())
        except Exception:
            # Widget might not be mounted yet
            pass
    
    def on_mount(self) -> None:
        """Set up log handler when widget is mounted."""
        self.setup_log_handler()
        self.start_log_file_monitoring()

    def on_unmount(self) -> None:
        """Clean up log handler when widget is unmounted."""
        self.cleanup_log_handler()
        self.stop_log_file_monitoring()

    def setup_log_handler(self) -> None:
        """Set up the custom log handler to capture OpenHCS logs."""
        if self.log_handler is None:
            self.log_handler = TUILogHandler(self)

            # Add handler to OpenHCS root logger to capture all OpenHCS logs
            openhcs_logger = logging.getLogger("openhcs")
            openhcs_logger.addHandler(self.log_handler)

            logger.debug("Real-time log handler attached to OpenHCS logger")

    def cleanup_log_handler(self) -> None:
        """Remove the log handler."""
        if self.log_handler is not None:
            openhcs_logger = logging.getLogger("openhcs")
            openhcs_logger.removeHandler(self.log_handler)
            self.log_handler = None
            logger.debug("Real-time log handler removed")

    def add_log_message(self, message: str, level: str) -> None:
        """
        Add a new log message to the status bar.

        Args:
            message: Log message to display
            level: Log level (INFO, WARNING, ERROR, etc.)
        """
        # Store in history
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_history.append(log_entry)

        # Update current display
        self.current_log_message = message

    def get_recent_logs(self, count: int = 10) -> list:
        """Get the most recent log messages."""
        return list(self.log_history)[-count:]

    # Legacy methods for compatibility
    def set_status(self, message: str) -> None:
        """Legacy method - now adds as INFO log."""
        self.add_log_message(f"Status: {message}", "INFO")

    def set_error(self, error_message: str) -> None:
        """Legacy method - now adds as ERROR log."""
        self.add_log_message(f"ERROR: {error_message}", "ERROR")

    def set_info(self, info_message: str) -> None:
        """Legacy method - now adds as INFO log."""
        self.add_log_message(f"INFO: {info_message}", "INFO")

    def clear_status(self) -> None:
        """Legacy method - now shows ready message."""
        self.current_log_message = "Ready"

    def start_log_file_monitoring(self) -> None:
        """Start monitoring the log file for subprocess updates."""
        # Get current log file path from the logging system
        self.log_file_path = self.get_current_log_file_path()
        if self.log_file_path and Path(self.log_file_path).exists():
            # Start from end of file to only show new logs
            self.log_file_position = Path(self.log_file_path).stat().st_size
            # Monitor every 0.1 seconds for very responsive updates
            self.log_monitor_timer = self.set_interval(0.1, self.check_log_file_updates)
            logger.debug(f"Started log file monitoring: {self.log_file_path}")

    def stop_log_file_monitoring(self) -> None:
        """Stop monitoring the log file."""
        if self.log_monitor_timer:
            self.log_monitor_timer.stop()
            self.log_monitor_timer = None
        logger.debug("Stopped log file monitoring")

    def get_current_log_file_path(self) -> Optional[str]:
        """Get the current log file path from the logging system."""
        try:
            # Get the root logger and find the FileHandler
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    return handler.baseFilename

            # Fallback: try to get from openhcs logger
            openhcs_logger = logging.getLogger("openhcs")
            for handler in openhcs_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    return handler.baseFilename

            return None
        except Exception as e:
            logger.debug(f"Could not determine log file path: {e}")
            return None

    async def check_log_file_updates(self) -> None:
        """Check for new log entries and update status bar."""
        if not self.log_file_path or not Path(self.log_file_path).exists():
            return

        try:
            # Wrap all file I/O operations in executor to avoid blocking UI
            import asyncio

            def _read_log_updates():
                current_size = Path(self.log_file_path).stat().st_size
                if current_size > self.log_file_position:
                    # Read new content
                    with open(self.log_file_path, 'r') as f:
                        f.seek(self.log_file_position)
                        new_lines = f.readlines()
                        new_position = f.tell()
                    return new_lines, new_position
                return [], self.log_file_position

            new_lines, new_position = await asyncio.get_event_loop().run_in_executor(None, _read_log_updates)
            self.log_file_position = new_position

            # Process new log lines
            for line in new_lines:
                line = line.strip()
                if line and self.is_subprocess_log(line):
                    # Extract the message part for display
                    message = self.extract_log_message(line)
                    if message:
                        self.add_log_message(message, "INFO")

        except Exception as e:
            logger.debug(f"Error reading log file: {e}")

    def is_subprocess_log(self, line: str) -> bool:
        """Accept ANY log line - show all subprocess output."""
        return True  # Show everything

    def extract_log_message(self, line: str) -> Optional[str]:
        """Extract the message part from any log line, removing timestamp."""
        try:
            # Remove timestamp: "2025-06-18 01:00:50,281 - logger - level - message"
            # Find first " - " and take everything after it
            if " - " in line:
                parts = line.split(" - ", 1)
                if len(parts) > 1:
                    clean_line = parts[1].strip()
                else:
                    clean_line = line.strip()
            else:
                clean_line = line.strip()
            
            # Allow more characters (200 instead of 100)
            if len(clean_line) > 200:
                return clean_line[:197] + "..."
            return clean_line
        except Exception:
            return line.strip() if line else None
