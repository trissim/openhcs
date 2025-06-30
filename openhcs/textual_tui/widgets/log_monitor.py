"""
OpenHCS Log Monitor Widget

Integrates the modernized Toolong log viewing capabilities into the OpenHCS TUI
for monitoring subprocess and worker process logs in real-time.

This uses our Textual 3.x compatible fork of Toolong.
"""

import logging
from pathlib import Path
from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Label, Static
from textual.widget import Widget
from textual.reactive import reactive

# Import modernized Toolong components (Textual 3.x compatible)
from toolong.ui import UI
from toolong.log_view import LogView
from toolong.log_file import LogFile
from toolong.watcher import get_watcher

logger = logging.getLogger(__name__)


class LogMonitorWidget(Widget):
    """
    Widget for monitoring OpenHCS subprocess and worker logs using Toolong components.
    
    Features:
    - Live tailing of multiple log files
    - Syntax highlighting for log formats
    - Search and filtering capabilities
    - Automatic detection of new worker log files
    """
    
    # Reactive properties
    log_files: reactive[List[str]] = reactive([], layout=True)
    current_log: reactive[Optional[str]] = reactive(None)
    
    def __init__(
        self,
        log_file_base: Optional[str] = None,
        auto_detect_workers: bool = True,
        **kwargs
    ):
        """
        Initialize the log monitor.
        
        Args:
            log_file_base: Base path for log files (e.g., "openhcs_subprocess_plates_A_B_1234567890")
            auto_detect_workers: Whether to automatically detect new worker log files
        """
        super().__init__(**kwargs)
        self.log_file_base = log_file_base
        self.auto_detect_workers = auto_detect_workers
        self.log_views = {}  # Map of log file path to LogView widget
        self.watchers = {}   # Map of log file path to file watcher
        
    def compose(self) -> ComposeResult:
        """Compose the log monitor layout."""
        with Vertical():
            # Header with controls
            with Horizontal(classes="log-monitor-header"):
                yield Label("📋 Log Monitor", classes="log-monitor-title")
                yield Button("🔄 Refresh", id="refresh-logs", variant="primary")
                yield Button("🔍 Search", id="search-logs", variant="default")
                yield Button("⏸️ Pause", id="pause-tailing", variant="default")
            
            # Log file tabs/selector
            with Horizontal(classes="log-file-selector"):
                yield Label("Active Logs:", classes="selector-label")
                # Dynamic log file buttons will be added here
            
            # Main log viewing area
            with Container(classes="log-view-container"):
                yield Static("Select a log file to view", classes="log-placeholder")
    
    def on_mount(self) -> None:
        """Set up the log monitor when mounted."""
        if self.log_file_base:
            self.discover_log_files()
            if self.auto_detect_workers:
                self.start_worker_detection()
    
    def discover_log_files(self) -> None:
        """Discover existing log files based on the base path."""
        if not self.log_file_base:
            return
            
        base_path = Path(self.log_file_base)
        log_dir = base_path.parent
        base_name = base_path.name
        
        # Find main subprocess log
        main_log = f"{self.log_file_base}.log"
        if Path(main_log).exists():
            self.add_log_file(main_log, "Main Process")
        
        # Find worker logs
        if log_dir.exists():
            for log_file in log_dir.glob(f"{base_name}_worker_*.log"):
                worker_id = log_file.stem.split('_worker_')[-1]
                self.add_log_file(str(log_file), f"Worker {worker_id}")
    
    def add_log_file(self, log_path: str, display_name: str = None) -> None:
        """Add a log file to the monitor."""
        if log_path in self.log_views:
            return  # Already monitoring this file

        try:
            # Verify file exists
            if not Path(log_path).exists():
                logger.warning(f"Log file does not exist: {log_path}")
                return

            # Create a simple UI wrapper for this log file
            # We'll use Toolong's UI components directly
            self.log_views[log_path] = {
                'path': log_path,
                'display_name': display_name or Path(log_path).name
            }

            # Update the UI
            self.update_log_selector()

            logger.info(f"Added log file to monitor: {log_path}")

        except Exception as e:
            logger.error(f"Failed to add log file {log_path}: {e}")
    
    def update_log_selector(self) -> None:
        """Update the log file selector buttons."""
        selector = self.query_one(".log-file-selector")
        
        # Remove existing log buttons (keep the label)
        for button in selector.query("Button.log-file-button"):
            button.remove()
        
        # Add buttons for each log file
        for i, (log_path, log_info) in enumerate(self.log_views.items()):
            button = Button(
                log_info['display_name'],
                id=f"log-{i}-{abs(hash(log_path)) % 10000}",  # More unique ID
                classes="log-file-button",
                variant="default"
            )
            selector.mount(button)
    
    def switch_to_log(self, log_path: str) -> None:
        """Switch the main view to show a specific log file using Toolong UI."""
        if log_path not in self.log_views:
            return

        try:
            container = self.query_one(".log-view-container")

            # Remove current content
            container.query("*").remove()

            # Create Toolong UI for this log file
            toolong_ui = UI([log_path], merge=False)

            # Mount the Toolong UI
            container.mount(toolong_ui)

            self.current_log = log_path
            logger.info(f"Switched to log: {log_path}")

        except Exception as e:
            logger.error(f"Failed to switch to log {log_path}: {e}")
            # Show error message in container
            container = self.query_one(".log-view-container")
            container.query("*").remove()
            container.mount(Static(f"Error loading log: {e}", classes="log-error"))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "refresh-logs":
            self.discover_log_files()
        elif button_id == "search-logs":
            self.show_search_dialog()
        elif button_id == "pause-tailing":
            self.toggle_tailing()
        elif button_id and button_id.startswith("log-"):
            # Find the log file for this button
            for i, (log_path, log_info) in enumerate(self.log_views.items()):
                if f"log-{i}-{abs(hash(log_path)) % 10000}" == button_id:
                    self.switch_to_log(log_path)
                    break
    
    def show_search_dialog(self) -> None:
        """Show search dialog for the current log."""
        # TODO: Implement search functionality using Toolong's FindDialog
        logger.info("Search functionality not yet implemented")
    
    def toggle_tailing(self) -> None:
        """Toggle live tailing on/off."""
        # TODO: Implement tailing toggle
        logger.info("Tailing toggle not yet implemented")
    
    def start_worker_detection(self) -> None:
        """Start monitoring for new worker log files."""
        # TODO: Implement automatic detection of new worker logs
        logger.info("Worker detection not yet implemented")
    
    def cleanup(self) -> None:
        """Clean up watchers and resources."""
        for watcher in self.watchers.values():
            if hasattr(watcher, 'stop'):
                watcher.stop()
        self.watchers.clear()
        self.log_views.clear()
