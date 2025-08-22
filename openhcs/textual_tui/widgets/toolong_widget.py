"""
Complete Toolong Widget for OpenHCS TUI

A self-contained widget that embeds the full Toolong log viewer functionality
into the OpenHCS TUI. This provides professional log viewing with all Toolong
features while being properly integrated into the OpenHCS window system.
"""

import logging
from pathlib import Path
from typing import List, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import TabbedContent, TabPane, Select
from textual.lazy import Lazy

# Import Toolong components
from toolong.ui import LogScreen
from toolong.watcher import get_watcher

logger = logging.getLogger(__name__)

# Import shared watcher to prevent conflicts
from openhcs.textual_tui.widgets.openhcs_toolong_widget import get_shared_watcher


class ToolongWidget(Widget):
    """
    Complete Toolong log viewer widget.
    
    This widget encapsulates the full Toolong application functionality,
    providing professional log viewing with tabs, search, tailing, and
    all other Toolong features in a self-contained widget.
    """
    
    DEFAULT_CSS = """
    ToolongWidget {
        width: 100%;
        height: 100%;
    }
    
    ToolongWidget TabbedContent {
        width: 100%;
        height: 100%;
    }
    
    ToolongWidget TabPane {
        padding: 0;
    }
    """
    
    def __init__(
        self,
        log_files: List[str],
        merge: bool = False,
        can_tail: bool = True,
        **kwargs
    ):
        """
        Initialize the Toolong widget.
        
        Args:
            log_files: List of log file paths to display
            merge: Whether to merge multiple files into one view
            can_tail: Whether tailing is enabled
        """
        super().__init__(**kwargs)
        self.log_files = self._sort_paths(log_files)
        self.merge = merge
        self.can_tail = can_tail
        self.watcher = get_shared_watcher()  # Use shared watcher to prevent conflicts
        
    @classmethod
    def _sort_paths(cls, paths: List[str]) -> List[str]:
        """Sort paths for consistent display order."""
        return sorted(paths, key=lambda p: Path(p).name)
    
    def compose(self) -> ComposeResult:
        """Compose the Toolong widget using dropdown selector instead of tabs."""
        if not self.log_files:
            # No log files - show empty state
            from textual.widgets import Static
            yield Static("No log files available", classes="empty-state")
            return

        from textual.widgets import Select, Container
        from textual.containers import Horizontal

        # Create dropdown options with friendly names
        options = []
        for log_file in self.log_files:
            display_name = self._get_friendly_name(log_file)
            options.append((display_name, log_file))

        # Dropdown selector
        with Horizontal(classes="log-selector"):
            yield Select(
                options=options,
                value=self.log_files[0] if self.log_files else None,
                id="log_selector",
                compact=True
            )

        # Container for log view
        with Container(id="log_container"):
            if self.log_files:
                yield Lazy(
                    LogView(
                        [self.log_files[0]],  # Start with first file
                        self.watcher,
                        can_tail=self.can_tail,
                    )
                )

    def _get_friendly_name(self, log_file: str) -> str:
        """Convert log file path to friendly display name."""
        import re
        file_name = Path(log_file).name

        if "unified" in file_name:
            if "worker_" in file_name:
                # Extract worker ID: openhcs_unified_20250630_094200_subprocess_123_worker_456_789.log
                worker_match = re.search(r'worker_(\d+)', file_name)
                return f"Worker {worker_match.group(1)}" if worker_match else "Worker"
            elif "_subprocess_" in file_name:
                return "Subprocess"
            else:
                return "TUI Main"
        elif "subprocess" in file_name:
            return "Subprocess"
        else:
            return Path(log_file).name

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown selection change."""
        if event.control.id == "log_selector" and event.value:
            selected_path = event.value

            # Update the log view with selected file
            container = self.query_one("#log_container")

            # Remove existing log view
            container.query("*").remove()

            # Add new log view for selected file
            new_view = LogView(
                [selected_path],
                self.watcher,
                can_tail=self.can_tail,
            )
            container.mount(new_view)
    
    def on_mount(self) -> None:
        """Start the watcher when widget is mounted."""

        # Start the watcher and enable tailing (only if not already running)
        if not hasattr(self.watcher, '_thread') or self.watcher._thread is None:
            self.watcher.start()
        else:
            pass  # Watcher already running

        # Focus LogLines (no tabs in dropdown structure)
        try:
            log_lines = self.query("LogView > LogLines")
            if log_lines:
                log_lines.first().focus()
        except Exception as e:
            logger.debug(f"Could not focus LogLines: {e}")

        # Simple tailing setup
        if self.can_tail:
            self.call_after_refresh(self._enable_tailing)

    def _enable_tailing(self) -> None:
        """Enable tailing on LogView widgets."""
        try:
            for log_view in self.query("LogView"):
                if hasattr(log_view, 'can_tail') and log_view.can_tail:
                    log_view.tail = True

                    # Also start the file watcher for single-file LogLines
                    for log_lines in log_view.query("LogLines"):
                        if hasattr(log_lines, 'start_tail') and len(log_lines.log_files) == 1:
                            log_lines.start_tail()
        except Exception as e:
            logger.debug(f"Could not enable tailing: {e}")

    def on_unmount(self) -> None:
        """Clean up the watcher when widget is unmounted."""
        # Don't close shared watcher - other widgets might be using it
        # The shared watcher will be cleaned up when the app exits
    
    def add_log_file(self, log_file: str) -> None:
        """Add a new log file to the widget."""
        if log_file not in self.log_files:
            self.log_files.append(log_file)
            self.log_files = self._sort_paths(self.log_files)
            # Would need to refresh the widget to show new file
    
    def remove_log_file(self, log_file: str) -> None:
        """Remove a log file from the widget."""
        if log_file in self.log_files:
            self.log_files.remove(log_file)
            # Would need to refresh the widget to remove file
    
    @classmethod
    def from_single_file(cls, log_file: str, can_tail: bool = True) -> "ToolongWidget":
        """Create a ToolongWidget for a single log file."""
        return cls([log_file], merge=False, can_tail=can_tail)
    
    @classmethod
    def from_multiple_files(cls, log_files: List[str], merge: bool = False) -> "ToolongWidget":
        """Create a ToolongWidget for multiple log files."""
        return cls(log_files, merge=merge, can_tail=True)
