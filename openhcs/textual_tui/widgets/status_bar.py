"""
StatusBar Widget for OpenHCS Textual TUI

Bottom status bar with real-time log streaming.
Shows live log messages from OpenHCS operations.
"""

import logging
import os
import time
from datetime import datetime
from collections import deque
from typing import Optional, List
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.widget import Widget
from textual.worker import get_current_worker
from textual.events import Click

logger = logging.getLogger(__name__)


class TUILogHandler(logging.Handler):
    """Custom logging handler that feeds log messages to the TUI status bar with batching."""

    def __init__(self, status_bar: 'StatusBar'):
        super().__init__()
        self.status_bar = status_bar
        self.setLevel(logging.INFO)  # Only show INFO and above in status bar

        # Format for status bar (compact)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        self.setFormatter(formatter)

        # Batching to reduce call_from_thread frequency
        self.pending_messages = []
        self.last_update_time = 0
        self.batch_interval = 0.5  # Batch messages for 500ms

    def emit(self, record):
        """Emit a log record to the status bar with batching."""
        try:
            msg = self.format(record)

            # Add to pending messages
            self.pending_messages.append((msg, record.levelname))

            # Only update UI if enough time has passed (batching)
            current_time = time.time()
            if current_time - self.last_update_time >= self.batch_interval:
                self._flush_pending_messages()
                self.last_update_time = current_time

        except (AttributeError, ValueError, TypeError) as e:
            # Don't let logging errors crash the app, but log the issue
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Status bar emit error (non-critical): {e}")

    def _flush_pending_messages(self):
        """Flush all pending messages to the UI."""
        if not self.pending_messages:
            return

        # Take the most recent message as the current display
        latest_msg, latest_level = self.pending_messages[-1]

        # Clear pending messages
        self.pending_messages.clear()

        # Update UI with latest message only (avoid spam)
        try:
            if hasattr(self.status_bar.app, 'call_from_thread'):
                self.status_bar.app.call_from_thread(self.status_bar.add_log_message, latest_msg, latest_level)
            else:
                # Fallback for direct calls
                self.status_bar.add_log_message(latest_msg, latest_level)
        except Exception:
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

        # Log file monitoring for subprocess logs - using Textual workers
        self.log_file_path: Optional[str] = None
        self.log_file_position: int = 0
        self.log_monitor_worker = None
        self.subprocess_base_log_path: Optional[str] = None  # For ReactiveLogMonitor

        logger.debug("StatusBar initialized with real-time logging")
    
    def compose(self) -> ComposeResult:
        """Compose the status bar layout."""
        # Make the status bar clickable to open Toolong
        yield Static(
            self.get_log_display() + " 📋",
            id="log_display",
            markup=False,
            classes="clickable-status"
        )

    def get_log_display(self) -> str:
        """Get the formatted log display string."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {self.current_log_message}"

    async def on_click(self, event: Click) -> None:
        """Handle click on status bar to open Toolong log viewer."""
        try:
            await self.open_toolong_viewer()
        except Exception as e:
            logger.error(f"Failed to open Toolong viewer: {e}")
            self.add_log_message(f"Error opening log viewer: {e}", "ERROR")

    async def open_toolong_viewer(self) -> None:
        """Open Toolong log viewer in a window using ReactiveLogMonitor."""
        from openhcs.textual_tui.windows.toolong_window import ToolongWindow
        from textual.css.query import NoMatches

        # Determine base log path for subprocess logs (if any)
        base_log_path = ""
        if hasattr(self, 'subprocess_base_log_path') and self.subprocess_base_log_path:
            base_log_path = self.subprocess_base_log_path

        # Try to find existing window - if it doesn't exist, create new one
        try:
            window = self.app.query_one(ToolongWindow)
            # Window exists, just open it
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = ToolongWindow(base_log_path=base_log_path)
            await self.app.mount(window)
            window.open_state = True

    def watch_current_log_message(self, message: str) -> None:
        """Update the display when current_log_message changes."""
        self.last_updated = datetime.now().strftime("%H:%M:%S")
        # Use call_later to defer DOM operations to next event loop cycle
        self.call_later(self._update_log_display)

    def _update_log_display(self) -> None:
        """Update the log display - deferred to avoid blocking reactive watchers."""
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
        """Clean up log handler and background thread when widget is unmounted."""
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

    def collect_log_files(self) -> List[str]:
        """Collect all available OpenHCS log files for viewing."""
        log_files = []

        # Add main TUI log file if available
        if self.log_file_path and Path(self.log_file_path).exists():
            log_files.append(self.log_file_path)

        # Look for subprocess and worker log files in the log directory
        if self.log_file_path:
            log_dir = Path(self.log_file_path).parent

            # Find subprocess logs
            subprocess_logs = list(log_dir.glob("openhcs_subprocess_*.log"))
            log_files.extend(str(f) for f in subprocess_logs)

            # Find worker logs
            worker_logs = list(log_dir.glob("openhcs_subprocess_*_worker_*.log"))
            log_files.extend(str(f) for f in worker_logs)

        # Remove duplicates and sort
        log_files = sorted(list(set(log_files)))

        logger.info(f"Collected {len(log_files)} log files for Toolong viewer")
        return log_files

    def start_log_monitoring(self, base_log_path: str) -> None:
        """Start log monitoring for subprocess with base path tracking."""
        self.subprocess_base_log_path = base_log_path
        logger.info(f"StatusBar: Started log monitoring for base path: {base_log_path}")

    def stop_log_monitoring(self) -> None:
        """Stop log monitoring and clear subprocess base path."""
        self.subprocess_base_log_path = None
        logger.info("StatusBar: Stopped log monitoring")

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
        """Start monitoring the log file for subprocess updates using Textual worker."""
        # Get current log file path from the logging system
        self.log_file_path = self.get_current_log_file_path()
        if self.log_file_path and Path(self.log_file_path).exists():
            # Start from end of file to only show new logs
            self.log_file_position = Path(self.log_file_path).stat().st_size

            # Stop any existing monitoring
            self.stop_log_file_monitoring()

            # Start Textual worker using @work decorated method
            self.log_monitor_worker = self._log_monitor_worker()
            logger.debug(f"Started worker log file monitoring: {self.log_file_path}")

    def stop_log_file_monitoring(self) -> None:
        """Stop monitoring the log file and cleanup worker."""
        if self.log_monitor_worker and not self.log_monitor_worker.is_finished:
            self.log_monitor_worker.cancel()
        self.log_monitor_worker = None
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

    @work(thread=True, exclusive=True)
    def _log_monitor_worker(self) -> None:
        """Textual worker that monitors log file for changes."""
        worker = get_current_worker()

        while not worker.is_cancelled:
            try:
                if not self.log_file_path or not Path(self.log_file_path).exists():
                    time.sleep(0.5)  # Check every 500ms (reduced frequency)
                    continue

                # Check for new content
                current_size = Path(self.log_file_path).stat().st_size
                if current_size > self.log_file_position:
                    # Read new content
                    with open(self.log_file_path, 'r') as f:
                        f.seek(self.log_file_position)
                        new_lines = f.readlines()
                        self.log_file_position = f.tell()

                    # Process new log lines and batch UI updates
                    messages_to_add = []
                    for line in new_lines:
                        line = line.strip()
                        if line and self.is_subprocess_log(line):
                            # Extract the message part for display
                            message = self.extract_log_message(line)
                            if message:
                                messages_to_add.append(message)

                    # Batch update UI if we have messages and not cancelled
                    if messages_to_add and not worker.is_cancelled:
                        # Thread-safe UI update - batch all messages at once
                        self.app.call_from_thread(self._batch_add_log_messages, messages_to_add)

                # Sleep for 2 seconds before next check (much reduced frequency)
                time.sleep(2.0)

            except Exception as e:
                logger.debug(f"Error in log monitor worker: {e}")
                time.sleep(1.0)  # Longer sleep on error

    def _batch_add_log_messages(self, messages: List[str]) -> None:
        """Add multiple log messages at once to reduce reactive watcher triggers."""
        if not messages:
            return

        # Add all messages to history without triggering reactive updates
        timestamp = datetime.now().strftime("%H:%M:%S")
        for message in messages:
            log_entry = f"[{timestamp}] {message}"
            self.log_history.append(log_entry)

        # Only trigger ONE reactive update with the latest message
        self.current_log_message = messages[-1]

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
