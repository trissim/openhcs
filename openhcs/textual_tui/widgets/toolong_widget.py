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
from textual.widgets import TabbedContent, TabPane
from textual.lazy import Lazy

# Import Toolong components
from toolong.ui import LogScreen
from toolong.watcher import get_watcher

logger = logging.getLogger(__name__)


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
        self.watcher = get_watcher()
        
    @classmethod
    def _sort_paths(cls, paths: List[str]) -> List[str]:
        """Sort paths for consistent display order."""
        return sorted(paths, key=lambda p: Path(p).name)
    
    def compose(self) -> ComposeResult:
        """Compose the Toolong widget using the same structure as Toolong UI."""
        if not self.log_files:
            # No log files - show empty state
            from textual.widgets import Static
            yield Static("No log files available", classes="empty-state")
            return
            
        # Use the exact same structure as Toolong's LogScreen
        with TabbedContent():
            if self.merge and len(self.log_files) > 1:
                # Merged view - single tab with all files
                tab_name = " + ".join(Path(path).name for path in self.log_files)
                with TabPane(tab_name):
                    yield Lazy(
                        LogView(
                            self.log_files,
                            self.watcher,
                            can_tail=False,  # Merged views don't tail
                        )
                    )
            else:
                # Individual tabs for each file
                for log_file in self.log_files:
                    tab_name = Path(log_file).name
                    with TabPane(tab_name):
                        yield Lazy(
                            LogView(
                                [log_file],
                                self.watcher,
                                can_tail=self.can_tail,
                            )
                        )
    
    def on_mount(self) -> None:
        """Start the watcher when widget is mounted."""
        logger.info(f"ToolongWidget mounting with {len(self.log_files)} log files")

        # Start the watcher and enable tailing
        self.watcher.start()

        # Hide tabs if only one file
        try:
            tabbed_content = self.query_one(TabbedContent)
            tab_panes = self.query(TabPane)
            tabbed_content.query("Tabs").set(display=len(tab_panes) > 1)

            # Focus the first LogLines and enable tailing
            if tab_panes:
                active_pane = tabbed_content.active_pane
                if active_pane:
                    log_lines = active_pane.query("LogView > LogLines")
                    if log_lines:
                        log_lines.first().focus()

        except Exception as e:
            logger.debug(f"Could not configure tabs or focus: {e}")

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
        logger.info("ToolongWidget unmounting, stopping watcher")
        try:
            self.watcher.close()
        except Exception as e:
            logger.debug(f"Error stopping watcher: {e}")
    
    def add_log_file(self, log_file: str) -> None:
        """Add a new log file to the widget."""
        if log_file not in self.log_files:
            self.log_files.append(log_file)
            self.log_files = self._sort_paths(self.log_files)
            # Would need to refresh the widget to show new file
            logger.info(f"Added log file: {log_file}")
    
    def remove_log_file(self, log_file: str) -> None:
        """Remove a log file from the widget."""
        if log_file in self.log_files:
            self.log_files.remove(log_file)
            # Would need to refresh the widget to remove file
            logger.info(f"Removed log file: {log_file}")
    
    @classmethod
    def from_single_file(cls, log_file: str, can_tail: bool = True) -> "ToolongWidget":
        """Create a ToolongWidget for a single log file."""
        return cls([log_file], merge=False, can_tail=can_tail)
    
    @classmethod
    def from_multiple_files(cls, log_files: List[str], merge: bool = False) -> "ToolongWidget":
        """Create a ToolongWidget for multiple log files."""
        return cls(log_files, merge=merge, can_tail=True)
