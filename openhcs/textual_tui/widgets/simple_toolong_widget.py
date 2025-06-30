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
from toolong.ui import ToolongWidget as ToolongDropdownWidget  # Explicit: this is the dropdown-based widget
from toolong.watcher import get_watcher
from toolong.log_lines import TailFile, LogLines
from toolong.log_view import LogView

# Import file system watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)

# Global shared watcher to prevent conflicts
_shared_watcher = None


class LogFileHandler(FileSystemEventHandler):
    """File system event handler for detecting new log files."""

    def __init__(self, widget: 'SimpleToolongWidget'):
        self.widget = widget

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.log'):
            file_path = Path(event.src_path)
            if self.widget._is_relevant_log_file(file_path):
                logger.info(f"New log file detected: {file_path}")
                self.widget._add_log_file(str(file_path))

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

/* Force hide tabs */
.force-hidden {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Also hide all ContentTabs by default in SimpleToolongWidget */
SimpleToolongWidget ContentTabs {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Override global Select rules for SimpleToolongWidget */
SimpleToolongWidget Select {
    height: auto !important;
    width: auto !important;
    padding: 0;
    margin: 0;
    text-style: underline;
    border: none;
}

/* Fix SelectOverlay to expand over other widgets - using undocumented overlay properties */
SimpleToolongWidget SelectOverlay {
    width: 1fr !important;
    display: none !important;
    height: auto !important;
    max-height: 15 !important;
    overlay: screen !important;
    constrain: none inside !important;
}

/* Show SelectOverlay when Select is expanded */
SimpleToolongWidget Select.-expanded > SelectOverlay {
    display: block !important;
}
"""

def get_shared_watcher():
    """Get or create a shared watcher instance to prevent conflicts."""
    global _shared_watcher
    if _shared_watcher is None:
        _shared_watcher = get_watcher()
    return _shared_watcher





class SimpleToolongWidget(Widget):
    """
    A simple widget that embeds Toolong functionality with tailing controls.

    Provides toggle buttons for auto-scroll behavior and manual tailing control.
    """

    CSS = TOOLONG_CSS

    def __init__(self, log_files: List[str], auto_tail: bool = True, base_log_path: str = None, **kwargs):
        """Initialize with log files and optional dynamic detection."""
        print(f"DEBUG: SimpleToolongWidget.__init__ called with {len(log_files)} files")
        super().__init__(**kwargs)
        self.log_files = log_files.copy()  # Make a copy so we can modify it
        self.auto_tail = auto_tail  # Whether to auto-scroll on new content
        self.manual_tail_enabled = True  # Whether tailing is manually enabled
        self.watcher = get_shared_watcher()  # Use shared watcher to prevent conflicts

        # Dynamic log detection
        self.base_log_path = base_log_path
        self._file_observer = None

        logger.info(f"SimpleToolongWidget initialized with {len(self.log_files)} files, auto_tail={auto_tail}, base_log_path={base_log_path}")

        # Start file watcher if we have a base log path for dynamic detection
        if self.base_log_path:
            self._start_file_watcher()
    
    def compose(self) -> ComposeResult:
        """Create Toolong structure with tailing controls."""
        logger.info(f"SimpleToolongWidget compose() called with {len(self.log_files)} files")

        # Tailing control buttons
        with Horizontal(classes="toolong-controls"):
            yield Button("Auto-Scroll", id="toggle_auto_tail", compact=True)
            yield Button("Pause", id="toggle_manual_tail", compact=True)
            yield Button("Bottom", id="scroll_to_bottom", compact=True)

        # Log viewer using ToolongDropdownWidget (entire Toolong app as widget)
        with Container(classes="toolong-viewer"):
            # Use the ToolongDropdownWidget that embeds the entire Toolong app with dropdown
            logger.info(f"About to create ToolongDropdownWidget with files: {[Path(f).name for f in self.log_files]}")
            yield ToolongDropdownWidget(
                file_paths=self.log_files,
                merge=False,  # Use dropdown for multiple files
                show_tabs=False,  # Hide tabs, use dropdown only
                show_dropdown=True  # Show dropdown selector
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

        # Wait a bit for LogScreen to mount, then configure tabs
        logger.info("About to call _configure_logscreen after refresh")
        self.call_after_refresh(self._configure_logscreen)

        # Also try to hide tabs immediately and after a delay
        self.call_after_refresh(lambda: self._force_hide_all_tabs("on initial mount"))

    def _configure_logscreen(self):
        """Configure LogScreen after it has mounted."""
        try:
            logger.info("Configuring LogScreen...")

            # Hide tabs in the ToolongDropdownWidget (force single-tab behavior)
            toolong_widgets = self.query(ToolongDropdownWidget)
            if toolong_widgets:
                toolong_widget = toolong_widgets.first()
                # Force tabs to be hidden by making it think there's only 1 tab
                tabs_elements = toolong_widget.query("#main_tabs Tabs")
                if tabs_elements:
                    tabs_elements.set(display=False)
                    logger.info("Hidden tabs in ToolongDropdownWidget")
                else:
                    logger.warning("Could not find #main_tabs Tabs to hide")
        except Exception as e:
            logger.error(f"_configure_logscreen failed: {e}")

    def _hide_tabs_immediately(self):
        """Try to hide tabs immediately."""
        try:
            logger.info("_hide_tabs_immediately called")

            # Try multiple approaches to find and hide tabs
            all_tabs = self.query("Tabs")
            logger.info(f"Found {len(all_tabs)} Tabs elements total")

            for i, tabs_element in enumerate(all_tabs):
                logger.info(f"Hiding Tabs element {i}: {tabs_element}")
                tabs_element.set(display=False)

            # Also try the specific query
            main_tabs = self.query("#main_tabs Tabs")
            logger.info(f"Found {len(main_tabs)} #main_tabs Tabs elements")
            for tabs_element in main_tabs:
                tabs_element.set(display=False)

        except Exception as e:
            logger.error(f"_hide_tabs_immediately failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _force_hide_all_tabs(self, context=""):
        """Force hide all tabs - used both initially and after refresh."""
        try:
            logger.info(f"_force_hide_all_tabs called {context}")

            # Find and hide ALL Tabs elements
            all_tabs = self.query("Tabs")
            logger.info(f"Found {len(all_tabs)} Tabs elements {context}")

            for i, tabs_element in enumerate(all_tabs):
                logger.info(f"Force hiding Tabs element {i} {context}: {tabs_element}")
                # Try multiple approaches to hide tabs
                tabs_element.display = False
                tabs_element.styles.display = "none"
                tabs_element.add_class("force-hidden")

            if len(all_tabs) == 0:
                logger.warning(f"No Tabs elements found to hide {context}")

        except Exception as e:
            logger.error(f"_force_hide_all_tabs failed {context}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info(f"SimpleToolongWidget has {len(self.log_files)} files:")
            for i, file_path in enumerate(self.log_files):
                logger.info(f"  File {i}: {Path(file_path).name}")

            # Check if ToolongDropdownWidget mounted properly
            toolong_widget = self.query(ToolongDropdownWidget)  # Query for the actual class
            if toolong_widget:
                logger.info(f"Found {len(toolong_widget)} ToolongDropdownWidget(s)")

                # Debug ToolongDropdownWidget content
                for i, widget in enumerate(toolong_widget):
                    logger.info(f"ToolongDropdownWidget {i}: {widget}")
                    children = widget.children
                    logger.info(f"ToolongDropdownWidget {i} children: {len(children)} - {[type(c).__name__ for c in children]}")

                    # Check if the widget is mounted
                    logger.info(f"ToolongDropdownWidget {i} is_mounted: {widget.is_mounted}")
                    logger.info(f"ToolongDropdownWidget {i} app: {widget.app}")

                    # Since the widget is mounted but on_mount() wasn't called, manually trigger dropdown update
                    if widget.is_mounted:
                        logger.info(f"ToolongDropdownWidget {i} is mounted, manually triggering dropdown update")
                        try:
                            # Manually call the dropdown update method
                            widget._update_dropdown_from_tabs()
                        except Exception as e:
                            logger.error(f"Manual dropdown update failed: {e}")
                    else:
                        logger.warning(f"ToolongDropdownWidget {i} is not mounted! This explains why on_mount() isn't called.")
            else:
                logger.warning("No ToolongDropdownWidget found!")
                return

            # Check container and content dimensions
            container = self.query_one(".toolong-viewer")
            logger.info(f"Container size: {container.size}")
            logger.info(f"Container region: {container.region}")

            # Focus LogLines in the dropdown-based structure
            log_lines = self.query("LogView > LogLines")
            if log_lines:
                log_lines.first().focus()
                logger.info("Focused LogLines")
        except Exception as e:
            logger.error(f"Could not configure LogScreen: {e}")
            import traceback
            traceback.print_exc()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tailing control button presses."""
        if event.button.id == "toggle_auto_tail":
            self.auto_tail = not self.auto_tail
            event.button.label = f"Auto-Scroll {'ON' if self.auto_tail else 'OFF'}"
            logger.info(f"Auto-tail toggled: {self.auto_tail}")

        elif event.button.id == "toggle_manual_tail":
            self.manual_tail_enabled = not self.manual_tail_enabled
            event.button.label = f"{'Resume' if not self.manual_tail_enabled else 'Pause'}"

            # Control persistent tailing through ToolongWidget
            for toolong_widget in self.query("ToolongWidget"):
                toolong_widget.toggle_persistent_tailing(self.manual_tail_enabled)
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

        # Stop file observer if running
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
            logger.info("File observer stopped")

        # Don't close shared watcher - other widgets might be using it
        # The shared watcher will be cleaned up when the app exits

    def _start_file_watcher(self):
        """Start watching for new OpenHCS log files."""
        if not self.base_log_path:
            return

        try:
            # base_log_path is now the logs directory
            log_dir = Path(self.base_log_path)
            if not log_dir.exists():
                logger.warning(f"Log directory does not exist: {log_dir}")
                return

            self._file_observer = Observer()
            handler = LogFileHandler(self)
            self._file_observer.schedule(handler, str(log_dir), recursive=False)
            self._file_observer.start()
            logger.info(f"Started file watcher for OpenHCS logs in: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")

    def _is_relevant_log_file(self, file_path: Path) -> bool:
        """Check if a log file belongs to the current session."""
        if not self.base_log_path or not self.log_files:
            return False

        file_name = file_path.name

        # Extract session base from existing log files
        # Use the first log file to determine session pattern
        first_log = Path(self.log_files[0]).name
        if first_log.startswith("openhcs_unified_"):
            # Extract session base: openhcs_unified_20250630_092636.log -> openhcs_unified_20250630_092636
            session_base = first_log.replace(".log", "")

            # Check if new file belongs to this session
            return (file_name.startswith(session_base) and
                    file_name.endswith(".log") and
                    str(file_path) not in self.log_files)

        return False

    def _add_log_file(self, log_file_path: str):
        """Add a new log file and refresh the ToolongWidget."""
        if log_file_path not in self.log_files:
            self.log_files.append(log_file_path)
            self.log_files.sort()  # Keep sorted for consistent display
            logger.info(f"Added new log file: {log_file_path}")

            # Refresh the ToolongWidget to show the new file
            self._refresh_toolong_widget()

    def _refresh_toolong_widget(self):
        """Refresh the ToolongWidget with updated log files."""
        try:
            logger.info(f"Refreshing ToolongWidget with {len(self.log_files)} files")

            # Use Textual's call_after_refresh to safely update the widget
            self.call_after_refresh(self._safe_refresh_dropdown)

        except Exception as e:
            logger.error(f"Failed to refresh ToolongWidget: {e}")

    def _safe_refresh_dropdown(self):
        """Safely refresh dropdown using Textual's reactive system."""
        try:
            # Always recreate the widget for reliable tab updates
            container = self.query_one(".toolong-viewer")

            # Remove existing ToolongDropdownWidget
            old_widgets = container.query(ToolongDropdownWidget)
            for widget in old_widgets:
                widget.remove()

            # Create new ToolongDropdownWidget with updated file list
            logger.info(f"Creating new ToolongDropdownWidget with {len(self.log_files)} files")
            new_widget = ToolongDropdownWidget(
                file_paths=self.log_files,
                merge=False,  # Use dropdown for multiple files
                show_tabs=False,  # Hide tabs, use dropdown only
                show_dropdown=True  # Show dropdown selector
            )
            container.mount(new_widget)
            logger.info(f"Successfully refreshed with {len(self.log_files)} files")

            # Hide tabs in the newly created widget - force hide all tabs
            self.call_after_refresh(lambda: self._force_hide_all_tabs("after refresh"))
            # Also hide tabs after a delay to override any tab activation logic
            self.set_timer(0.1, lambda: self._force_hide_all_tabs("delayed after refresh"))
            self.set_timer(0.5, lambda: self._force_hide_all_tabs("final delayed after refresh"))

        except Exception as e:
            logger.error(f"Failed to safely refresh dropdown: {e}")
            logger.info("New files detected but couldn't refresh dropdown automatically")
            logger.info(f"Files: {[Path(f).name for f in self.log_files]}")
            logger.info("Close and reopen the Log Viewer to see new files")

    def _hide_tabs_in_new_widget(self):
        """Hide tabs in the newly created ToolongDropdownWidget."""
        try:
            logger.info("_hide_tabs_in_new_widget called")

            # Find all Tabs elements and hide them
            all_tabs = self.query("Tabs")
            logger.info(f"Found {len(all_tabs)} Tabs elements after refresh")

            for i, tabs_element in enumerate(all_tabs):
                logger.info(f"Hiding Tabs element {i} after refresh: {tabs_element}")
                tabs_element.set(display=False)

        except Exception as e:
            logger.error(f"_hide_tabs_in_new_widget failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
