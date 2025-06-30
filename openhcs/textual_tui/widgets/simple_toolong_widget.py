"""
Simple Toolong Widget for OpenHCS TUI

A minimal widget that embeds Toolong's LogScreen directly to provide
identical functionality to standalone Toolong.
"""

import logging
from pathlib import Path
from typing import List

from textual.app import ComposeResult
from textual.widget import Widget

# Import Toolong components
from toolong.ui import LogScreen, UI
from toolong.watcher import get_watcher

logger = logging.getLogger(__name__)


class SimpleToolongWidget(Widget):
    """
    A simple widget that embeds Toolong functionality.

    Instead of trying to embed LogScreen directly, this creates the actual
    Toolong UI components that work properly.
    """

    def __init__(self, log_files: List[str], **kwargs):
        """Initialize with log files."""
        super().__init__(**kwargs)
        self.log_files = log_files
        self.watcher = get_watcher()

        logger.info(f"SimpleToolongWidget initialized with {len(self.log_files)} files")
    
    def compose(self) -> ComposeResult:
        """Create Toolong structure directly without LogScreen."""
        from textual.widgets import TabbedContent, TabPane
        from textual.lazy import Lazy
        from toolong.log_view import LogView

        with TabbedContent():
            for log_file in self.log_files:
                tab_name = Path(log_file).name
                with TabPane(tab_name):
                    yield Lazy(
                        LogView(
                            [log_file],
                            self.watcher,
                            can_tail=True,
                        )
                    )
    
    def on_mount(self) -> None:
        """Start watcher and initialize like standalone Toolong."""
        logger.info("SimpleToolongWidget mounting")

        # Start watcher
        self.watcher.start()

        # Hide tabs if only one file and focus LogLines
        try:
            from textual.widgets import TabbedContent, TabPane
            tabbed_content = self.query_one(TabbedContent)
            tab_panes = self.query(TabPane)
            tabbed_content.query("Tabs").set(display=len(tab_panes) > 1)

            # Focus LogLines
            active_pane = tabbed_content.active_pane
            if active_pane:
                log_lines = active_pane.query("LogView > LogLines")
                if log_lines:
                    log_lines.first().focus()
                    logger.info("Focused LogLines")
        except Exception as e:
            logger.debug(f"Could not configure tabs or focus: {e}")

    def on_unmount(self) -> None:
        """Clean up watcher."""
        logger.info("SimpleToolongWidget unmounting")
        try:
            if self.watcher:
                self.watcher.close()
        except Exception as e:
            logger.debug(f"Error stopping watcher: {e}")
