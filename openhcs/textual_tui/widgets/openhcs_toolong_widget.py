"""
OpenHCS Toolong Widget

Consolidated toolong widget that combines the best of the external toolong library
with OpenHCS-specific dropdown selection logic and file management.

This widget is moved from src/toolong/ui.py to be a native OpenHCS component
while still importing the core toolong functionality.
"""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.lazy import Lazy
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import TabbedContent, TabPane, Select, Button

# Import toolong core components
from toolong.log_view import LogView, LogLines
from toolong.messages import TailFile
from toolong.ui import UI
from toolong.watcher import get_watcher

# Import file system watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)

# Global shared watcher to prevent conflicts
_shared_watcher = None





def get_shared_watcher():
    """Get or create a shared watcher instance to prevent conflicts."""
    global _shared_watcher
    if _shared_watcher is None:
        _shared_watcher = get_watcher()
    return _shared_watcher


def clear_all_subprocess_logs_from_ui(app):
    """Clear all subprocess logs from UI viewers, keeping only the main TUI log. Can be called from anywhere."""
    try:
        # Try to find any existing toolong widgets and clear them
        from openhcs.textual_tui.widgets.openhcs_toolong_widget import OpenHCSToolongWidget
        toolong_widgets = app.query(OpenHCSToolongWidget)
        logger.info(f"Found {len(toolong_widgets)} toolong widgets")

        for widget in toolong_widgets:
            logger.info("Calling clear logs on toolong widget")
            widget._clear_all_logs_except_tui()
            logger.info("Clear logs completed")

        # Also try to find toolong windows
        from openhcs.textual_tui.windows.toolong_window import ToolongWindow
        toolong_windows = app.query(ToolongWindow)
        logger.info(f"Found {len(toolong_windows)} toolong windows")

        for window in toolong_windows:
            # Find the widget inside the window
            widgets = window.query(OpenHCSToolongWidget)
            for widget in widgets:
                logger.info("Calling clear logs on toolong widget in window")
                widget._clear_all_logs_except_tui()
                logger.info("Clear logs completed")

    except Exception as e:
        logger.error(f"Failed to clear logs from UI: {e}")
        import traceback
        logger.error(traceback.format_exc())



class LogFileHandler(FileSystemEventHandler):
    """File system event handler for detecting new log files."""

    def __init__(self, widget: 'OpenHCSToolongWidget'):
        self.widget = widget

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.log'):
            file_path = Path(event.src_path)
            if self.widget._is_relevant_log_file(file_path):
                logger.debug(f"New log file detected: {file_path}")
                self.widget._add_log_file(str(file_path))


class HiddenTabsTabbedContent(TabbedContent):
    """TabbedContent that can force-hide tabs regardless of tab count."""

    def __init__(self, *args, force_hide_tabs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_hide_tabs = force_hide_tabs

    def compose(self) -> ComposeResult:
        """Override compose to control tab visibility."""
        result = super().compose()
        if self.force_hide_tabs:
            # Hide tabs immediately after composition
            self.call_after_refresh(self._force_hide_tabs)
        return result

    def _force_hide_tabs(self):
        """Force hide tabs regardless of count."""
        try:
            tabs = self.query("ContentTabs")
            for tab in tabs:
                tab.display = False
                tab.styles.display = "none"
        except Exception as e:
            logger.debug(f"Could not force hide tabs: {e}")

    def _on_mount(self, event):
        """Override mount to prevent automatic tab showing."""
        super()._on_mount(event)
        if self.force_hide_tabs:
            self._force_hide_tabs()


class PersistentTailLogLines(LogLines):
    """LogLines that doesn't automatically disable tailing on user interaction."""

    def __init__(self, watcher, file_paths):
        super().__init__(watcher, file_paths)
        self._persistent_tail = True

    def post_message(self, message):
        """Override to block TailFile(False) messages when persistent tailing is enabled."""
        # Handle FileError messages safely when app context is not available
        from toolong.messages import FileError

        if (isinstance(message, TailFile) and
            not message.tail and
            self._persistent_tail):
            # Block the message - don't send TailFile(False)
            return

        # Handle FileError messages safely when app context is not available
        if isinstance(message, FileError):
            try:
                super().post_message(message)
            except Exception as e:
                # Log the error instead of crashing if app context is not available
                logger.warning(f"Could not post FileError message (app context unavailable): {message.error}")
                return
        else:
            super().post_message(message)

    def on_scan_complete(self, event) -> None:
        """Override to ensure start_tail is actually called after scan completes."""
        # Call parent first
        super().on_scan_complete(event)

        # Force start_tail if conditions are met
        if len(self.log_files) == 1 and self.can_tail:
            try:
                self.start_tail()
            except Exception as e:
                logger.error(f"❌ PersistentTailLogLines: start_tail() failed in on_scan_complete: {e}")
                import traceback
                traceback.print_exc()

    def action_scroll_up(self) -> None:
        """Override scroll up to not disable tailing when persistent."""
        if self.pointer_line is None:
            super().action_scroll_up()
        else:
            self.advance_search(-1)
        # Don't send TailFile(False) when persistent tailing is enabled
        if not self._persistent_tail:
            self.post_message(TailFile(False))

    def action_scroll_home(self) -> None:
        """Override scroll home to not disable tailing when persistent."""
        if self.pointer_line is not None:
            self.pointer_line = 0
        self.scroll_to(y=0, duration=0)
        # Don't send TailFile(False) when persistent tailing is enabled
        if not self._persistent_tail:
            self.post_message(TailFile(False))

    def action_scroll_end(self) -> None:
        """Override scroll end to not disable tailing when persistent."""
        if self.pointer_line is not None:
            self.pointer_line = self.line_count
        if self.scroll_offset.y == self.max_scroll_y:
            self.post_message(TailFile(True))
        else:
            self.scroll_to(y=self.max_scroll_y, duration=0)
            # Don't send TailFile(False) when persistent tailing is enabled
            if not self._persistent_tail:
                self.post_message(TailFile(False))

    def action_page_down(self) -> None:
        """Override page down to not disable tailing when persistent."""
        if self.pointer_line is None:
            super().action_page_down()
        else:
            self.pointer_line = (
                self.pointer_line + self.scrollable_content_region.height
            )
            self.scroll_pointer_to_center()
        # Don't send TailFile(False) when persistent tailing is enabled
        if not self._persistent_tail:
            self.post_message(TailFile(False))

    def action_page_up(self) -> None:
        """Override page up to not disable tailing when persistent."""
        if self.pointer_line is None:
            super().action_page_up()
        else:
            self.pointer_line = max(
                0, self.pointer_line - self.scrollable_content_region.height
            )
            self.scroll_pointer_to_center()
        # Don't send TailFile(False) when persistent tailing is enabled
        if not self._persistent_tail:
            self.post_message(TailFile(False))


class PersistentTailLogView(LogView):
    """LogView that uses PersistentTailLogLines."""

    def on_mount(self) -> None:
        """Override to ensure tailing is enabled after mount."""
        # Force enable tailing for persistent behavior
        self.tail = True
        logger.debug(f"PersistentTailLogView mounted with tail={self.tail}, can_tail={self.can_tail}")

    async def watch_tail(self, old_value: bool, new_value: bool) -> None:
        """Watch for changes to the tail property."""
        if hasattr(super(), 'watch_tail'):
            await super().watch_tail(old_value, new_value)

    def compose(self):
        """Override to use our custom LogLines with proper data binding."""
        # Create PersistentTailLogLines with proper data binding (this is critical!)
        yield (
            log_lines := PersistentTailLogLines(self.watcher, self.file_paths).data_bind(
                LogView.tail,
                LogView.show_line_numbers,
                LogView.show_find,
                LogView.can_tail,
            )
        )

        # Import the other components from toolong
        from toolong.line_panel import LinePanel
        from toolong.find_dialog import FindDialog
        from toolong.log_view import InfoOverlay, LogFooter

        yield LinePanel()
        yield FindDialog(log_lines._suggester)
        yield InfoOverlay().data_bind(LogView.tail)

        # Create LogFooter with error handling for mount_keys
        footer = LogFooter().data_bind(LogView.tail, LogView.can_tail)

        # Monkey patch mount_keys to add error handling
        original_mount_keys = footer.mount_keys
        async def safe_mount_keys():
            try:
                await original_mount_keys()
            except Exception as e:
                logger.error(f"LogFooter mount_keys failed: {e}")
                # Continue without crashing
        footer.mount_keys = safe_mount_keys

        yield footer

    def on_mount(self) -> None:
        """Override to ensure tailing is enabled after mount."""
        # Force enable tailing for persistent behavior
        self.tail = True

    async def on_scan_complete(self, event) -> None:
        """Override to ensure tailing remains enabled after scan."""
        # Call parent method first
        await super().on_scan_complete(event)
        # Force enable tailing (this is critical!)
        self.tail = True


class OpenHCSToolongWidget(Widget):
    """
    OpenHCS native toolong widget with dropdown selection and file management.
    
    This widget provides professional log viewing with:
    - Dropdown selection for multiple log files
    - Automatic switching to latest log files
    - Tab-based viewing with optional tab hiding
    - Persistent tailing functionality
    - OpenHCS-specific file naming and organization
    """

    # CSS to control Select dropdown height
    DEFAULT_CSS = """
    OpenHCSToolongWidget SelectOverlay {
        max-height: 20;
    }
    """

    # Reactive variable to track when tabs are ready
    tabs_ready = reactive(False)

    def __init__(
        self,
        file_paths: List[str],
        merge: bool = False,
        save_merge: str | None = None,
        show_tabs: bool = True,
        show_dropdown: bool = False,
        show_controls: bool = True,
        base_log_path: Optional[str] = None,
        **kwargs
    ) -> None:
        logger.debug(f"OpenHCSToolongWidget.__init__ called with {len(file_paths)} files: {[Path(f).name for f in file_paths]}")
        super().__init__(**kwargs)
        self.file_paths = UI.sort_paths(file_paths)
        self.merge = merge
        self.save_merge = save_merge
        self.show_tabs = show_tabs
        self.show_dropdown = show_dropdown
        self.show_controls = show_controls
        self.watcher = get_shared_watcher()  # Use shared watcher to prevent conflicts
        self._current_file_path = file_paths[0] if file_paths else None  # Track currently viewed file

        # Control states
        self.auto_tail = True  # Whether to auto-scroll on new content
        self.manual_tail_enabled = True  # Whether tailing is manually enabled

        # Dynamic log detection
        self.base_log_path = base_log_path
        self._file_observer = None

        # Timing protection
        self._last_tab_switch_time = 0

        # Tab creation protection
        self._tab_creation_in_progress = False

        # Thread-safe log addition system to prevent race conditions
        self._log_addition_lock = threading.Lock()
        self._pending_logs = set()  # Track logs being processed
        self._debounce_timer = None
        self._debounce_delay = 0.1  # 100ms debounce delay

        logger.debug(f"OpenHCSToolongWidget.__init__ completed with show_tabs={show_tabs}, show_dropdown={show_dropdown}, show_controls={show_controls}")

        # Start file watcher if we have a base log path for dynamic detection
        if self.base_log_path:
            self._start_file_watcher()

    def compose(self) -> ComposeResult:
        """Compose the Toolong widget using persistent tailing LogViews."""
        logger.debug(f"OpenHCSToolongWidget compose() called with {len(self.file_paths)} files")

        # Conditionally add control buttons
        if self.show_controls:
            with Horizontal(classes="toolong-controls"):
                yield Button("Auto-Scroll", id="toggle_auto_tail", compact=True)
                yield Button("Pause", id="toggle_manual_tail", compact=True)
                yield Button("Bottom", id="scroll_to_bottom", compact=True)
                yield Button("Clear All", id="clear_all_logs", compact=True)

        # Conditionally add dropdown selector
        if self.show_dropdown:
            if self.file_paths:
                # Create initial options from file paths
                initial_options = []
                current_index = 0  # Default to first file

                for i, path in enumerate(self.file_paths):
                    # Create friendly tab names
                    tab_name = self._create_friendly_tab_name(path)
                    initial_options.append((tab_name, i))

                    # Check if this is the currently viewed file
                    if self._current_file_path and path == self._current_file_path:
                        current_index = i

                yield Select(initial_options, id="log_selector", compact=True, allow_blank=False, value=current_index)
                logger.debug(f"Yielded Select widget with {len(initial_options)} initial options, selected index: {current_index}")
            else:
                # Create empty Select that will be populated later
                yield Select([("Loading...", -1)], id="log_selector", compact=True, allow_blank=False, value=-1)
                logger.debug("Yielded Select widget with placeholder option")
        else:
            logger.debug("Skipped Select widget (show_dropdown=False)")

        # Always create tabs (needed for dropdown), but conditionally hide them
        with HiddenTabsTabbedContent(id="main_tabs", force_hide_tabs=not self.show_tabs):
            if self.merge and len(self.file_paths) > 1:
                tab_name = " + ".join(Path(path).name for path in self.file_paths)
                with TabPane(tab_name):
                    # Create separate watcher for merged view (like original toolong)
                    from toolong.watcher import get_watcher
                    watcher = get_watcher()
                    watcher.start()  # CRITICAL: Start the watcher thread!
                    yield Lazy(
                        PersistentTailLogView(
                            self.file_paths,
                            watcher,  # Separate watcher
                            can_tail=False,
                        )
                    )
            else:
                for path in self.file_paths:
                    # Create friendly tab names
                    tab_name = self._create_friendly_tab_name(path)

                    with TabPane(tab_name):
                        # Create separate watcher for each LogView (like original toolong)
                        from toolong.watcher import get_watcher
                        watcher = get_watcher()
                        watcher.start()  # CRITICAL: Start the watcher thread!
                        yield Lazy(
                            PersistentTailLogView(
                                [path],
                                watcher,  # Separate watcher for each tab
                                can_tail=True,
                            )
                        )

        logger.debug(f"OpenHCSToolongWidget compose() completed")

    def _create_friendly_tab_name(self, path: str) -> str:
        """Create a friendly display name for a log file path."""
        tab_name = Path(path).name

        # Check for most specific patterns first
        if "worker_" in tab_name:
            # Extract worker ID
            worker_match = re.search(r'worker_(\d+)', tab_name)
            tab_name = f"Worker {worker_match.group(1)}" if worker_match else "Worker"
        elif "subprocess" in tab_name:
            # Subprocess runner (plate manager spawned process)
            tab_name = "Subprocess"
        elif "unified" in tab_name:
            # Main TUI thread (only if no subprocess/worker indicators)
            tab_name = "TUI Main"
        else:
            # Fallback to filename
            tab_name = Path(path).stem

        return tab_name

    def _is_tui_main_log(self, path: str) -> bool:
        """Check if a log file is the main TUI log (not subprocess or worker)."""
        log_name = Path(path).name
        # TUI main logs don't contain "subprocess" or "worker" in the name
        return "_subprocess_" not in log_name and "_worker_" not in log_name



    def on_mount(self) -> None:
        """Update dropdown when widget mounts and enable persistent tailing."""
        try:
            logger.debug("OpenHCSToolongWidget on_mount called")

            # Start the watcher (critical for real-time updates!)
            if not hasattr(self.watcher, '_thread') or self.watcher._thread is None:
                logger.debug("Starting watcher for real-time log updates")
                self.watcher.start()
            else:
                logger.debug("Watcher already running")

            # Set tabs_ready to True after mounting to trigger dropdown update
            self.call_after_refresh(self._mark_tabs_ready)
            # Enable persistent tailing by default
            self.call_after_refresh(self._enable_persistent_tailing)
            logger.debug("OpenHCSToolongWidget on_mount completed successfully")
        except Exception as e:
            logger.error(f"OpenHCSToolongWidget on_mount failed: {e}")
            import traceback
            logger.error(f"OpenHCSToolongWidget on_mount traceback: {traceback.format_exc()}")

    def _enable_persistent_tailing(self) -> None:
        """Enable persistent tailing on all LogViews and LogLines."""
        try:
            logger.debug("Enabling persistent tailing by default")

            # Enable tailing on LogView widgets (this is critical!)
            for log_view in self.query("PersistentTailLogView"):
                if hasattr(log_view, 'can_tail') and log_view.can_tail:
                    log_view.tail = True
                    logger.debug(f"Enabled tail=True on LogView: {log_view}")

            # Enable persistent tailing on LogLines and start individual file tailing
            log_lines_widgets = self.query("PersistentTailLogLines")
            logger.debug(f"🔍 Found {len(log_lines_widgets)} PersistentTailLogLines widgets for tailing setup")

            for log_lines in log_lines_widgets:
                logger.debug(f"🔍 Processing LogLines: {log_lines}, log_files={len(getattr(log_lines, 'log_files', []))}")

                log_lines._persistent_tail = True
                log_lines.post_message(TailFile(True))

                # Check if file is opened before starting tailing
                if hasattr(log_lines, 'start_tail') and len(log_lines.log_files) == 1:
                    log_file = log_lines.log_files[0]
                    file_opened = hasattr(log_file, 'file') and log_file.file is not None
                    logger.debug(f"🔍 File status: path={getattr(log_file, 'path', 'unknown')}, file_opened={file_opened}")

                    if file_opened:
                        try:
                            log_lines.start_tail()
                            logger.debug(f"✅ Started file tailing on LogLines: {log_lines}")
                        except Exception as e:
                            logger.error(f"❌ Failed to start tailing on LogLines: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.debug(f"⏰ File not opened yet, tailing will start after scan completes")
                else:
                    logger.warning(f"⚠️ Cannot start tailing: has_start_tail={hasattr(log_lines, 'start_tail')}, log_files={len(getattr(log_lines, 'log_files', []))}")

                logger.debug(f"Enabled persistent tailing for {log_lines}")
        except Exception as e:
            logger.error(f"Failed to enable persistent tailing: {e}")

    def _mark_tabs_ready(self) -> None:
        """Mark tabs as ready, which will trigger the watcher."""
        try:
            logger.debug("Marking tabs as ready")

            # Control tab visibility using the existing logic pattern
            # When show_tabs=False, pretend there's only 1 tab so tabs are hidden
            # When show_tabs=True, use actual tab count
            actual_tab_count = len(self.query(TabPane))
            effective_tab_count = actual_tab_count if self.show_tabs else 1

            self.query("#main_tabs Tabs").set(display=effective_tab_count > 1)
            logger.debug(f"Tab visibility: show_tabs={self.show_tabs}, actual_tabs={actual_tab_count}, effective_tabs={effective_tab_count}, display={effective_tab_count > 1}")

            self.tabs_ready = True
            logger.debug("tabs_ready set to True")
        except Exception as e:
            logger.error(f"_mark_tabs_ready failed: {e}")
            import traceback
            logger.error(f"_mark_tabs_ready traceback: {traceback.format_exc()}")

    def _force_hide_tabs_after_activation(self):
        """Force hide tabs after tab activation events."""
        try:
            tabs_elements = self.query("#main_tabs Tabs")
            if tabs_elements:
                for tabs_element in tabs_elements:
                    tabs_element.display = False
                    tabs_element.styles.display = "none"
        except Exception as e:
            logger.error(f"_force_hide_tabs_after_activation failed: {e}")

    def watch_tabs_ready(self, tabs_ready: bool) -> None:
        """Watcher that updates dropdown when tabs are ready."""
        if tabs_ready:
            logger.debug("tabs_ready watcher triggered")
            self._update_dropdown_from_tabs()

    def _update_dropdown_from_tabs(self) -> None:
        """Update dropdown options to match current tabs."""
        logger.debug("_update_dropdown_from_tabs called")

        # Check if dropdown exists
        try:
            select = self.query_one("#log_selector", Select)
        except:
            logger.debug("No dropdown selector found, skipping update")
            return

        tabbed_content = self.query_one("#main_tabs", TabbedContent)
        tab_panes = tabbed_content.query(TabPane)

        logger.debug(f"Found {len(tab_panes)} tab panes")

        # Check if we need to update options (either placeholder or different count)
        current_value = select.value
        options_need_update = (current_value == -1 or  # Placeholder value
                              len(tab_panes) != len(getattr(select, '_options', [])))

        # Check if selection needs to be updated to match current file
        selection_needs_update = False
        if self._current_file_path:
            try:
                current_file_index = self.file_paths.index(self._current_file_path)
                if current_value != current_file_index:
                    selection_needs_update = True
                    logger.debug(f"Selection needs update: current={current_value}, should be={current_file_index}")
            except ValueError:
                logger.warning(f"Current file {self._current_file_path} not found in file_paths")
                selection_needs_update = True  # Force update if current file not found

        if not options_need_update and not selection_needs_update:
            logger.debug("Dropdown already has correct options and selection, skipping update")
            return

        # Only update options if needed
        if options_need_update:
            logger.debug(f"Found {len(tab_panes)} tab panes")
            logger.debug(f"Tab pane IDs: {[getattr(pane, 'id', 'no-id') for pane in tab_panes]}")

            # Create dropdown options from tab labels
            options = []
            logger.debug("Starting to process tab panes...")
            for i, tab_pane in enumerate(tab_panes):
                logger.debug(f"Processing tab_pane {i}: {tab_pane}")
                # Get the tab title from the TabPane - this is what shows in the tab
                tab_label = getattr(tab_pane, '_title', str(tab_pane))
                logger.debug(f"Tab {i}: {tab_label}")
                options.append((tab_label, i))

            logger.debug(f"Created options: {options}")

            # Update dropdown - let Textual handle sizing automatically
            logger.debug("About to call select.set_options...")
            select.set_options(options)
            logger.debug(f"Set dropdown options: {options}")
        else:
            logger.debug("Options don't need updating, only updating selection")

        # Update selection - prioritize current file path over active tab
        if self._current_file_path:
            try:
                current_file_index = self.file_paths.index(self._current_file_path)
                select.value = current_file_index
                logger.debug(f"Set dropdown to current file index: {current_file_index} ({Path(self._current_file_path).name})")
                return
            except ValueError:
                logger.warning(f"Current file {self._current_file_path} not found in file_paths")

        # Fallback to active tab if current file not found
        if len(tab_panes) > 0:
            try:
                active_tab = tabbed_content.active_pane
                if active_tab:
                    try:
                        active_index = list(tab_panes).index(active_tab)
                        select.value = active_index
                        logger.debug(f"Set dropdown to active tab index: {active_index}")
                    except ValueError:
                        # Active tab not found in list, default to first option
                        select.value = 0
                        logger.debug("Active tab not found, defaulting to first option")
                else:
                    # No active tab, select first option
                    select.value = 0
                    logger.debug("No active tab, selecting first option")
            except Exception as e:
                # Tab access failed (probably due to tabs being removed), default to first option
                logger.debug(f"Tab access failed during update: {e}, defaulting to first option")
                select.value = 0
        else:
            logger.warning("No tab panes available for dropdown")

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab activation - update dropdown to match."""
        logger.debug(f"Tab activated: {event.tab}")

        # Update current file path tracking based on active tab
        try:
            tabbed_content = self.query_one("#main_tabs", TabbedContent)
            tab_panes = tabbed_content.query(TabPane)
            active_tab = tabbed_content.active_pane

            if active_tab:
                active_index = list(tab_panes).index(active_tab)
                if 0 <= active_index < len(self.file_paths):
                    self._current_file_path = self.file_paths[active_index]
                    logger.debug(f"Updated current file path to: {Path(self._current_file_path).name}")
        except Exception as e:
            logger.error(f"Error updating current file path: {e}")

        # Update dropdown when tabs change
        self._update_dropdown_from_tabs()

        # Force hide tabs again after activation if show_tabs=False
        if not self.show_tabs:
            self.call_after_refresh(lambda: self._force_hide_tabs_after_activation())

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown selection - switch to corresponding tab."""
        if (event.control.id == "log_selector" and
            event.value is not None and
            isinstance(event.value, int) and
            event.value >= 0):  # Ignore placeholder value (-1)

            try:
                tabbed_content = self.query_one("#main_tabs", TabbedContent)
                tab_panes = tabbed_content.query(TabPane)

                if 0 <= event.value < len(tab_panes):
                    target_tab = tab_panes[event.value]
                    tabbed_content.active = target_tab.id

                    # Update current file path tracking
                    if event.value < len(self.file_paths):
                        self._current_file_path = self.file_paths[event.value]
                        logger.debug(f"Switched to tab {event.value}, file: {Path(self._current_file_path).name}")
                    else:
                        logger.warning(f"Tab index {event.value} out of range for file_paths")
                else:
                    logger.warning(f"Invalid tab index: {event.value}, available: {len(tab_panes)}")

            except Exception as e:
                logger.error(f"Exception switching tab: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tailing controls."""
        if event.button.id == "toggle_auto_tail":
            self.auto_tail = not self.auto_tail
            event.button.label = f"Auto-Scroll {'On' if self.auto_tail else 'Off'}"
            logger.debug(f"Auto-scroll toggled: {self.auto_tail}")

        elif event.button.id == "toggle_manual_tail":
            self.manual_tail_enabled = not self.manual_tail_enabled
            event.button.label = f"{'Resume' if not self.manual_tail_enabled else 'Pause'}"
            logger.debug(f"Manual tailing toggled: {self.manual_tail_enabled}")

            # Control persistent tailing through OpenHCSToolongWidget
            self.toggle_persistent_tailing(self.manual_tail_enabled)

        elif event.button.id == "scroll_to_bottom":
            # Use OpenHCSToolongWidget method to scroll to bottom and enable tailing
            self.scroll_to_bottom_and_tail()
            logger.debug("Scrolled to bottom and enabled tailing")

        elif event.button.id == "clear_all_logs":
            self._show_clear_confirmation()
            logger.debug("Showing clear confirmation dialog")



    def update_file_paths(self, new_file_paths: List[str], old_file_paths: List[str] = None) -> None:
        """Update tabs when new file paths are detected."""
        logger.debug(f"OpenHCSToolongWidget.update_file_paths called with {len(new_file_paths)} files")

        # Use provided old_file_paths or current self.file_paths
        if old_file_paths is None:
            old_file_paths = self.file_paths

        old_file_paths_set = set(old_file_paths)
        self.file_paths = new_file_paths
        new_file_paths_set = set(new_file_paths)

        # Find newly added files
        newly_added = new_file_paths_set - old_file_paths_set
        logger.debug(f"DEBUG: old_file_paths_set={len(old_file_paths_set)}, new_file_paths_set={len(new_file_paths_set)}, newly_added={len(newly_added)}")
        if newly_added:
            logger.debug(f"Found {len(newly_added)} newly added files: {[Path(p).name for p in newly_added]}")
            # Find the most recent file
            most_recent_file = max(newly_added, key=lambda p: os.path.getmtime(p))
            logger.debug(f"Most recent file: {Path(most_recent_file).name}")

            # ALWAYS switch to the most recent file when new files are added
            self._current_file_path = most_recent_file
            logger.debug(f"Switching to most recent file: {Path(most_recent_file).name}")
        else:
            logger.debug("No newly added files detected")

        # If no current file is set, default to first file
        if not self._current_file_path and new_file_paths:
            self._current_file_path = new_file_paths[0]
            logger.debug(f"No current file set, defaulting to first: {Path(self._current_file_path).name}")

        # If new files were added, trigger a full recompose
        if newly_added:
            logger.debug(f"New files detected, triggering recompose for {len(newly_added)} new files")
            self.refresh(recompose=True)
        else:
            # Just update dropdown if no new files
            self.call_after_refresh(self._update_dropdown_from_tabs)

    def toggle_persistent_tailing(self, enabled: bool):
        """Enable or disable persistent tailing for all LogLines."""
        for log_lines in self.query("PersistentTailLogLines"):
            log_lines._persistent_tail = enabled
            if enabled:
                log_lines.post_message(TailFile(True))
            else:
                log_lines.post_message(TailFile(False))

    def scroll_to_bottom_and_tail(self):
        """Scroll all LogLines to bottom and ensure tailing is enabled."""
        for log_lines in self.query("PersistentTailLogLines"):
            log_lines.scroll_to(y=log_lines.max_scroll_y, duration=0.3)
            log_lines._persistent_tail = True
            log_lines.post_message(TailFile(True))

    def on_unmount(self) -> None:
        """Clean up watcher and LogLines when widget is unmounted."""
        try:
            logger.debug("OpenHCSToolongWidget unmounting, cleaning up resources")

            # Stop file observer if running
            if self._file_observer:
                self._file_observer.stop()
                self._file_observer.join()
                self._file_observer = None
                logger.debug("File observer stopped")

            # No shared watcher cleanup needed - each LogView has its own watcher
            # The individual watchers will be cleaned up when their LogLines widgets unmount
            logger.debug("No shared watcher cleanup needed - using separate watchers per LogView")

            # Don't close shared watcher - other widgets might be using it
            # The shared watcher will be cleaned up when the app exits
            logger.debug("OpenHCSToolongWidget unmount completed")
        except Exception as e:
            logger.error(f"Error during OpenHCSToolongWidget unmount: {e}")

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
            logger.debug(f"Started file watcher for OpenHCS logs in: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")

    def _is_relevant_log_file(self, file_path: Path) -> bool:
        """Check if a log file is relevant to the current session."""
        if not file_path.name.endswith('.log'):
            return False

        # Check if it's an OpenHCS unified log file
        return (file_path.name.startswith('openhcs_unified_') and
                str(file_path) not in self.file_paths)

    def _add_log_file(self, log_file_path: str):
        """Thread-safe log file addition with debouncing to prevent race conditions."""
        with self._log_addition_lock:
            # Check if already processing this log
            if log_file_path in self._pending_logs:
                logger.debug(f"Log file {log_file_path} already being processed, skipping")
                return

            # Check if already exists
            if log_file_path in self.file_paths:
                logger.debug(f"Log file {log_file_path} already exists, skipping")
                return

            # Add to pending set to prevent duplicate processing
            self._pending_logs.add(log_file_path)
            logger.debug(f"Adding new log file to pending: {log_file_path}")

        # Use debouncing to handle rapid successive additions
        self._debounced_log_addition(log_file_path)

    def _debounced_log_addition(self, log_file_path: str):
        """Debounced log addition to handle rapid file creation."""
        # Cancel existing timer if any
        if self._debounce_timer:
            self._debounce_timer.cancel()

        # Start new timer
        self._debounce_timer = threading.Timer(
            self._debounce_delay,
            self._process_pending_logs
        )
        self._debounce_timer.start()

    def _process_pending_logs(self):
        """Process all pending log additions in a single batch."""
        with self._log_addition_lock:
            if not self._pending_logs:
                return

            # Store old file paths before modifying
            old_file_paths = self.file_paths.copy()
            new_logs = list(self._pending_logs)

            # Add all pending logs
            for log_path in new_logs:
                if log_path not in self.file_paths:
                    self.file_paths.append(log_path)
                    logger.debug(f"Processed pending log file: {log_path}")

            # Sort for consistent display
            self.file_paths = UI.sort_paths(self.file_paths)

            # Clear pending logs
            self._pending_logs.clear()

            # Update UI in a single batch operation
            if len(self.file_paths) != len(old_file_paths):
                logger.debug(f"Batch updating UI: {len(old_file_paths)} → {len(self.file_paths)} files")
                self.call_after_refresh(self.update_file_paths, self.file_paths, old_file_paths)

    def _show_clear_confirmation(self) -> None:
        """Show confirmation dialog before clearing logs."""
        # Find the TUI main log(s) to keep
        tui_logs = [path for path in self.file_paths if self._is_tui_main_log(path)]

        if not tui_logs:
            logger.warning("No TUI main log found, cannot clear logs safely")
            return

        # Keep only the most recent TUI log
        tui_log_to_keep = max(tui_logs, key=lambda p: os.path.getmtime(p))
        logs_to_remove = [path for path in self.file_paths if path != tui_log_to_keep]

        if not logs_to_remove:
            logger.info("No logs to remove")
            return

        # Show confirmation window
        from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
        from textual.widgets import Static, Button
        from textual.containers import Horizontal
        from textual.app import ComposeResult

        tui_log_name = Path(tui_log_to_keep).name

        class ClearLogsConfirmationWindow(BaseOpenHCSWindow):
            def __init__(self, tui_log_name: str, logs_to_remove_count: int, on_result_callback):
                super().__init__(
                    window_id="clear_logs_confirmation",
                    title="Clear All Logs",
                    mode="temporary"
                )
                self.tui_log_name = tui_log_name
                self.logs_to_remove_count = logs_to_remove_count
                self.on_result_callback = on_result_callback

            def compose(self) -> ComposeResult:
                message = f"This will remove {self.logs_to_remove_count} log entries.\nOnly '{self.tui_log_name}' will remain.\nThis action cannot be undone."
                yield Static(message, classes="dialog-content")

                with Horizontal(classes="dialog-buttons"):
                    yield Button("Cancel", id="cancel", compact=True)
                    yield Button("Clear All", id="clear", compact=True)

            def on_button_pressed(self, event: Button.Pressed) -> None:
                result = event.button.id == "clear"
                if self.on_result_callback:
                    self.on_result_callback(result)
                self.close_window()

        def handle_confirmation(result):
            if result:
                self._clear_all_logs_except_tui()

        # Create and mount confirmation window
        confirmation = ClearLogsConfirmationWindow(tui_log_name, len(logs_to_remove), handle_confirmation)
        self.app.mount(confirmation)
        confirmation.open_state = True

    def _clear_all_logs_except_tui(self) -> None:
        """Clear all log entries except the main TUI log and properly clean up resources."""
        logger.info("Clearing all logs except TUI main...")

        # Find the TUI main log(s) to keep
        tui_logs = [path for path in self.file_paths if self._is_tui_main_log(path)]

        if not tui_logs:
            logger.warning("No TUI main log found, keeping all logs")
            return

        # Keep only the most recent TUI log
        tui_log_to_keep = max(tui_logs, key=lambda p: os.path.getmtime(p))
        logs_to_remove = [path for path in self.file_paths if path != tui_log_to_keep]

        logger.info(f"Keeping TUI log: {Path(tui_log_to_keep).name}")
        logger.info(f"Removing {len(logs_to_remove)} logs: {[Path(p).name for p in logs_to_remove]}")

        # Set active tab to first one (TUI main) BEFORE clearing to avoid race conditions
        try:
            tabbed_content = self.query_one("#main_tabs", HiddenTabsTabbedContent)
            tab_panes = list(tabbed_content.query(TabPane))
            if tab_panes:
                # Set active tab to first one (should be TUI main)
                tabbed_content.active = tab_panes[0].id
                logger.info("Set active tab to first tab (TUI main) before clearing")
        except Exception as e:
            logger.warning(f"Could not set active tab before clearing: {e}")

        # Clean up resources for logs being removed
        self._cleanup_removed_logs(logs_to_remove)

        # Update file paths to only include the TUI log
        old_file_paths = self.file_paths.copy()
        self.file_paths = [tui_log_to_keep]

        # Update UI
        self.call_after_refresh(self.update_file_paths, self.file_paths, old_file_paths)

        logger.info("Log clearing completed")

    def _cleanup_removed_logs(self, logs_to_remove: List[str]) -> None:
        """Clean up resources (watchers, threads) for removed log files."""
        try:
            # Get all TabPanes and their associated LogViews
            tabbed_content = self.query_one("#main_tabs", HiddenTabsTabbedContent)
            tab_panes = list(tabbed_content.query(TabPane))

            # Find tabs corresponding to logs being removed
            tabs_to_remove = []
            for i, path in enumerate(self.file_paths):
                if path in logs_to_remove and i < len(tab_panes):
                    tabs_to_remove.append(tab_panes[i])

            # Clean up each tab's resources
            for tab_pane in tabs_to_remove:
                try:
                    # Find LogView in this tab
                    log_views = tab_pane.query("PersistentTailLogView")
                    for log_view in log_views:
                        # Clean up LogLines and their watchers
                        log_lines_widgets = log_view.query("LogLines")
                        for log_lines in log_lines_widgets:
                            # Stop the watcher for this LogLines
                            if hasattr(log_lines, 'watcher') and log_lines.watcher:
                                try:
                                    log_lines.watcher.close()
                                    logger.debug(f"Closed watcher for {getattr(log_lines, 'log_files', 'unknown')}")
                                except Exception as e:
                                    logger.warning(f"Error closing watcher: {e}")

                            # Stop line reader thread if it exists
                            if hasattr(log_lines, '_line_reader') and log_lines._line_reader:
                                try:
                                    log_lines._line_reader.exit_event.set()
                                    logger.debug("Stopped line reader thread")
                                except Exception as e:
                                    logger.warning(f"Error stopping line reader: {e}")

                    # Remove the tab pane
                    tab_pane.remove()
                    logger.debug(f"Removed tab pane")

                except Exception as e:
                    logger.warning(f"Error cleaning up tab resources: {e}")

            logger.debug(f"Cleaned up resources for {len(tabs_to_remove)} tabs")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")




