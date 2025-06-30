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
from textual.widgets import Button
from textual.containers import Container, Horizontal

# Import Toolong components
from toolong.ui import LogScreen, UI
from toolong.watcher import get_watcher
from toolong.log_lines import TailFile, LogLines
from toolong.log_view import LogView

logger = logging.getLogger(__name__)

# Global shared watcher to prevent conflicts
_shared_watcher = None

# CSS for the widget
TOOLONG_CSS = """
SimpleToolongWidget {
    layout: vertical;
}

.toolong-controls {
    dock: top;
    height: 3;
    padding: 1;
}

.toolong-viewer {
    height: 1fr;
}

.toolong-controls Button {
    margin-right: 1;
}
"""

def get_shared_watcher():
    """Get or create a shared watcher instance to prevent conflicts."""
    global _shared_watcher
    if _shared_watcher is None:
        _shared_watcher = get_watcher()
    return _shared_watcher


class PersistentTailLogLines(LogLines):
    """LogLines that doesn't automatically disable tailing on user interaction."""

    def __init__(self, watcher, file_paths):
        super().__init__(watcher, file_paths)
        self._persistent_tail = True

    def post_message(self, message):
        """Override to block TailFile(False) messages when persistent tailing is enabled."""
        if (isinstance(message, TailFile) and
            not message.tail and
            self._persistent_tail):
            # Block the message - don't send TailFile(False)
            return
        super().post_message(message)


class PersistentTailLogView(LogView):
    """LogView that uses PersistentTailLogLines."""

    def compose(self):
        """Override to use our custom LogLines."""
        # Get the original compose result but replace LogLines
        for widget in super().compose():
            if isinstance(widget, LogLines):
                # Replace with our persistent version
                yield PersistentTailLogLines(
                    widget.watcher,
                    widget.file_paths
                )
            else:
                yield widget


class SimpleToolongWidget(Widget):
    """
    A simple widget that embeds Toolong functionality with tailing controls.

    Provides toggle buttons for auto-scroll behavior and manual tailing control.
    """

    def __init__(self, log_files: List[str], auto_tail: bool = True, **kwargs):
        """Initialize with log files."""
        super().__init__(**kwargs)
        self.log_files = log_files
        self.auto_tail = auto_tail  # Whether to auto-scroll on new content
        self.manual_tail_enabled = True  # Whether tailing is manually enabled
        self.watcher = get_shared_watcher()  # Use shared watcher to prevent conflicts

        logger.info(f"SimpleToolongWidget initialized with {len(self.log_files)} files, auto_tail={auto_tail}")
    
    def compose(self) -> ComposeResult:
        """Create Toolong structure with tailing controls."""
        from textual.widgets import TabbedContent, TabPane
        from textual.lazy import Lazy

        # Tailing control buttons
        with Horizontal(classes="toolong-controls"):
            yield Button("Auto-Scroll", id="toggle_auto_tail", compact=True)
            yield Button("Pause", id="toggle_manual_tail", compact=True)
            yield Button("Bottom", id="scroll_to_bottom", compact=True)

        # Log viewer
        with Container(classes="toolong-viewer"):
            with TabbedContent():
                for log_file in self.log_files:
                    tab_name = Path(log_file).name
                    with TabPane(tab_name):
                        yield Lazy(
                            PersistentTailLogView(
                                [log_file],
                                self.watcher,
                                can_tail=True,
                            )
                        )
    
    def on_mount(self) -> None:
        """Start watcher and initialize like standalone Toolong."""
        logger.info("SimpleToolongWidget mounting")

        # Start watcher (only if not already started)
        if not hasattr(self.watcher, '_thread') or self.watcher._thread is None:
            logger.info("Starting shared watcher")
            self.watcher.start()
        else:
            logger.info("Shared watcher already running")



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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tailing control button presses."""
        if event.button.id == "toggle_auto_tail":
            self.auto_tail = not self.auto_tail
            event.button.label = f"Auto-Scroll {'ON' if self.auto_tail else 'OFF'}"
            logger.info(f"Auto-tail toggled: {self.auto_tail}")

        elif event.button.id == "toggle_manual_tail":
            self.manual_tail_enabled = not self.manual_tail_enabled
            event.button.label = f"{'Resume' if not self.manual_tail_enabled else 'Pause'}"

            # Control persistent tailing and send TailFile message
            for log_lines in self.query("PersistentTailLogLines"):
                log_lines._persistent_tail = self.manual_tail_enabled
                log_lines.post_message(TailFile(self.manual_tail_enabled))
            logger.info(f"Manual tailing toggled: {self.manual_tail_enabled}")

        elif event.button.id == "scroll_to_bottom":
            # Scroll all LogLines to bottom and enable tailing
            for log_lines in self.query("LogLines"):
                log_lines.scroll_to(y=log_lines.max_scroll_y, duration=0.3)
                log_lines.post_message(TailFile(True))
            logger.info("Scrolled to bottom and enabled tailing")

    def on_tail_file(self, event: TailFile) -> None:
        """Handle TailFile events from LogLines."""
        # If auto_tail is disabled, prevent ALL automatic tailing disable
        if not self.auto_tail and not event.tail:
            # Block any attempt to disable tailing - only buttons can control it
            event.stop()
            return

        # If manual tailing is enabled, prevent automatic disable
        if self.manual_tail_enabled and not event.tail:
            # Block automatic disable - only manual button can turn off tailing
            event.stop()
            return

        # Update button states based on tailing status
        try:
            manual_button = self.query_one("#toggle_manual_tail")
            if event.tail != self.manual_tail_enabled:
                self.manual_tail_enabled = event.tail
                manual_button.label = f"{'Resume' if not self.manual_tail_enabled else 'Pause'}"
        except Exception:
            pass  # Button might not be mounted yet

    def on_unmount(self) -> None:
        """Clean up watcher."""
        logger.info("SimpleToolongWidget unmounting")
        # Don't close shared watcher - other widgets might be using it
        # The shared watcher will be cleaned up when the app exits
