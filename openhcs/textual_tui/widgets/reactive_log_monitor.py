"""
OpenHCS Reactive Log Monitor Widget

A clean, low-entropy reactive log monitoring system that provides real-time
log viewing with a dropdown selector interface.

Mathematical properties:
- Reactive: UI updates are pure functions of file system events
- Monotonic: Logs only get added during execution
- Deterministic: Same file system state always produces same UI
"""

import logging
import os
import re
from pathlib import Path
from typing import Set, Dict, Optional, Callable, List
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, Select
from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Horizontal, Vertical

# Import file system watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

# Toolong components are imported in ToolongWidget

logger = logging.getLogger(__name__)


@dataclass
class LogFileInfo:
    """Information about a log file."""
    path: Path
    log_type: str  # "tui", "main", "worker", "unknown"
    well_id: Optional[str] = None
    display_name: Optional[str] = None
    
    def __post_init__(self):
        """Generate display name if not provided."""
        if not self.display_name:
            if self.log_type == "tui":
                self.display_name = "TUI Process"
            elif self.log_type == "main":
                self.display_name = "Main Subprocess"
            elif self.log_type == "worker" and self.well_id:
                self.display_name = f"Worker {self.well_id}"
            else:
                self.display_name = self.path.name


class ReactiveLogFileHandler(FileSystemEventHandler):
    """File system event handler for reactive log monitoring."""
    
    def __init__(self, monitor: 'ReactiveLogMonitor'):
        self.monitor = monitor
        
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.log'):
            file_path = Path(event.src_path)
            if self.monitor._is_relevant_log_file(file_path):
                logger.debug(f"Log file created: {file_path}")
                self.monitor._handle_log_file_created(file_path)
                
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith('.log'):
            file_path = Path(event.src_path)
            if self.monitor._is_relevant_log_file(file_path):
                self.monitor._handle_log_file_modified(file_path)


class ReactiveLogMonitor(Widget):
    """
    Reactive log monitor with dropdown selector.
    
    Provides real-time monitoring of OpenHCS log files with a clean dropdown
    interface for selecting which log to view.
    """
    
    # Reactive properties
    active_logs: reactive[Set[Path]] = reactive(set())
    base_log_path: reactive[str] = reactive("")
    
    def __init__(
        self,
        base_log_path: str = "",
        auto_start: bool = True,
        include_tui_log: bool = True,
        **kwargs
    ):
        """
        Initialize ReactiveLogMonitor.
        
        Args:
            base_log_path: Base path for subprocess log files
            auto_start: Whether to automatically start monitoring when mounted
            include_tui_log: Whether to include the current TUI process log
        """
        super().__init__(**kwargs)
        self.base_log_path = base_log_path
        self.auto_start = auto_start
        self.include_tui_log = include_tui_log
        
        # Internal state
        self._log_info_cache: Dict[Path, LogFileInfo] = {}
        # ToolongWidget will manage its own watcher
        
        # File system watcher (will be set up in on_mount)
        self._file_observer = None
        
    def compose(self) -> ComposeResult:
        """Compose the reactive log monitor layout with dropdown selector."""
        # Simple layout like other widgets - no complex containers
        yield Static("Log File:")
        yield Select(
            options=[("Loading...", "loading")],
            value="loading",
            id="log_selector",
            compact=True
        )
        yield Container(id="log_view_container")
    
    def on_mount(self) -> None:
        """Set up log monitoring when widget is mounted."""
        logger.info(f"ReactiveLogMonitor.on_mount() called, auto_start={self.auto_start}")
        if self.auto_start:
            logger.info("Starting monitoring from on_mount")
            self.start_monitoring()
        else:
            logger.warning("Auto-start disabled, not starting monitoring")

    def on_unmount(self) -> None:
        """Clean up when widget is unmounted."""
        logger.info("ReactiveLogMonitor unmounting, cleaning up watchers...")
        self.stop_monitoring()
    
    def start_monitoring(self, base_log_path: str = None) -> None:
        """
        Start monitoring for log files.

        Args:
            base_log_path: Optional new base path to monitor
        """
        logger.info(f"start_monitoring() called with base_log_path='{base_log_path}'")

        if base_log_path:
            self.base_log_path = base_log_path

        logger.info(f"Current state: base_log_path='{self.base_log_path}', include_tui_log={self.include_tui_log}")

        # We can monitor even without base_log_path if include_tui_log is True
        if not self.base_log_path and not self.include_tui_log:
            raise RuntimeError("Cannot start log monitoring: no base log path and TUI log disabled")

        if self.base_log_path:
            logger.info(f"Starting reactive log monitoring for subprocess: {self.base_log_path}")
        else:
            logger.info("Starting reactive log monitoring for TUI log only")

        # Discover existing logs - THIS SHOULD CRASH IF NO TUI LOG FOUND
        logger.info("About to discover existing logs...")
        self._discover_existing_logs()
        logger.info("Finished discovering existing logs")

        # Start file system watcher (only if we have subprocess logs to watch)
        if self.base_log_path:
            self._start_file_watcher()

        logger.info("Log monitoring started successfully")
    
    def stop_monitoring(self) -> None:
        """Stop all log monitoring with proper thread cleanup."""
        logger.info("Stopping reactive log monitoring")
        
        try:
            # Stop file system watcher first
            self._stop_file_watcher()
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")
            
        # ToolongWidget now manages its own watcher, so no need to stop it here
        logger.debug("ReactiveLogMonitor stopped (ToolongWidget manages its own watcher)")
            
        # Clear state
        self.active_logs = set()
        self._log_info_cache.clear()
        
        logger.info("Reactive log monitoring stopped")

    def get_current_tui_log_path(self) -> Path:
        """Get the current TUI process log file path - FAIL LOUD if not found."""
        # Get the root logger and find the FileHandler (same as status bar)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)

        # Fallback: try to get from openhcs logger (same as status bar)
        openhcs_logger = logging.getLogger("openhcs")
        for handler in openhcs_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)

        # FAIL LOUD - this should never happen in a running TUI
        raise RuntimeError(
            f"CRITICAL: No FileHandler found in logging system!\n"
            f"Root logger handlers: {[type(h).__name__ for h in root_logger.handlers]}\n"
            f"OpenHCS logger handlers: {[type(h).__name__ for h in openhcs_logger.handlers]}\n"
            f"The TUI process should always have a log file!"
        )

    def discover_logs(self) -> Set[Path]:
        """
        Pure function: Discover all relevant log files.

        Returns:
            Set of Path objects for discovered log files
        """
        discovered_logs = set()
        
        # Always include current TUI log if requested
        if self.include_tui_log:
            tui_log = self.get_current_tui_log_path()  # This will raise if not found
            # FAIL LOUD if log file doesn't exist
            if not tui_log.exists():
                raise RuntimeError(f"TUI log file does not exist: {tui_log}")
            discovered_logs.add(tui_log)
            logger.info(f"Added TUI log to monitoring: {tui_log}")
        
        # Include subprocess logs if base_log_path is provided
        if self.base_log_path:
            base_path = Path(self.base_log_path)
            log_dir = base_path.parent
            base_name = base_path.name
            
            # Fail fast if directory doesn't exist
            if log_dir.exists():
                # Find main subprocess log: {base_name}.log
                main_log = base_path.with_suffix('.log')
                if main_log.exists() and main_log.is_file():
                    discovered_logs.add(main_log)
                    
                # Find worker logs: {base_name}_worker_*.log
                worker_pattern = f"{base_name}_worker_*.log"
                for log_file in log_dir.glob(worker_pattern):
                    if log_file.is_file() and self._is_valid_worker_log(log_file, base_name):
                        discovered_logs.add(log_file)
                
        return discovered_logs

    def classify_log_file(self, log_path: Path) -> LogFileInfo:
        """
        Pure function: Classify a log file and extract metadata.

        Args:
            log_path: Path to log file

        Returns:
            LogFileInfo with classification and metadata
        """
        file_name = log_path.name

        # Check if it's the current TUI log
        if self.include_tui_log:
            tui_log = self.get_current_tui_log_path()
            if tui_log and log_path == tui_log:
                return LogFileInfo(log_path, "tui", display_name="TUI Process")

        # Check subprocess logs if base_log_path is provided
        if self.base_log_path:
            base_name = Path(self.base_log_path).name

            # Check if it's the main subprocess log: exact match
            if file_name == f"{base_name}.log":
                return LogFileInfo(log_path, "main", display_name="Main Subprocess")

            # Check if it's a worker log: {base_name}_worker_{well_id}.log
            worker_pattern = rf"^{re.escape(base_name)}_worker_([A-Za-z0-9_]+)\.log$"
            match = re.match(worker_pattern, file_name)
            if match:
                well_id = match.group(1)
                # Validate well_id format (basic sanity check)
                if self._is_valid_well_id(well_id):
                    return LogFileInfo(log_path, "worker", well_id)
                else:
                    logger.warning(f"Invalid well ID format in log file: {file_name}")

        # Unknown or malformed log file
        logger.debug(f"Unrecognized log file pattern: {file_name}")
        return LogFileInfo(log_path, "unknown")

    def _is_relevant_log_file(self, log_path: Path) -> bool:
        """Check if a log file is relevant for monitoring."""
        if not self.base_log_path:
            return False

        base_name = Path(self.base_log_path).name
        file_name = log_path.name

        # Check if it matches our patterns
        if file_name == f"{base_name}.log":
            return True

        if file_name.startswith(f"{base_name}_worker_") and file_name.endswith('.log'):
            return True

        return False

    def _is_valid_worker_log(self, log_path: Path, base_name: str) -> bool:
        """Validate that a file is a properly formatted worker log."""
        file_name = log_path.name
        worker_pattern = rf"^{re.escape(base_name)}_worker_([A-Za-z0-9_]+)\.log$"
        match = re.match(worker_pattern, file_name)

        if not match:
            return False

        well_id = match.group(1)
        return self._is_valid_well_id(well_id)

    def _is_valid_well_id(self, well_id: str) -> bool:
        """Validate well ID format (e.g., A01, B12, etc.)."""
        # Allow alphanumeric well IDs with underscores
        # Common formats: A01, B12, A_01, etc.
        return bool(re.match(r'^[A-Za-z0-9_]+$', well_id)) and len(well_id) <= 10

    def _discover_existing_logs(self) -> None:
        """Discover and add existing log files."""
        discovered = self.discover_logs()
        for log_path in discovered:
            self._add_log_file(log_path)

    def _add_log_file(self, log_path: Path) -> None:
        """Add a log file to monitoring (internal method)."""
        if log_path in self.active_logs:
            return  # Already monitoring

        # Classify the log file
        log_info = self.classify_log_file(log_path)
        self._log_info_cache[log_path] = log_info

        # Add to active logs (triggers reactive update)
        new_logs = set(self.active_logs)
        new_logs.add(log_path)
        self.active_logs = new_logs

        logger.info(f"Added log file to monitoring: {log_info.display_name} ({log_path})")

    def watch_active_logs(self, logs: Set[Path]) -> None:
        """Reactive: Update dropdown when active logs change."""
        logger.info(f"Active logs changed: {len(logs)} logs")
        # Always try to update - the _update_log_selector method has its own safety checks
        logger.info("Updating log selector")
        self._update_log_selector()

    def _update_log_selector(self) -> None:
        """Update dropdown selector with available logs."""
        logger.info(f"_update_log_selector called, is_mounted={self.is_mounted}")

        try:
            # Check if the selector exists (might not be ready yet or removed during unmount)
            try:
                log_selector = self.query_one("#log_selector", Select)
                logger.info(f"Found log selector widget: {log_selector}")
            except Exception as e:
                logger.debug(f"Log selector not found (widget not ready or unmounting?): {e}")
                return
            logger.info(f"Found log selector widget: {log_selector}")

            # Sort logs: TUI first, then main subprocess, then workers by well ID
            sorted_logs = self._sort_logs_for_display(self.active_logs)
            logger.info(f"Active logs: {[str(p) for p in self.active_logs]}")
            logger.info(f"Sorted logs: {[str(p) for p in sorted_logs]}")

            # Build dropdown options
            options = []
            for log_path in sorted_logs:
                log_info = self._log_info_cache.get(log_path)
                logger.info(f"Log path {log_path} -> log_info: {log_info}")
                if log_info:
                    options.append((log_info.display_name, str(log_path)))
                else:
                    logger.warning(f"No log_info found for {log_path} in cache: {list(self._log_info_cache.keys())}")

            logger.info(f"Built options: {options}")

            if not options:
                logger.error("CRITICAL: No options built! This should never happen with TUI log.")
                options = [("No logs available", "none")]

            # Update selector options
            log_selector.set_options(options)
            logger.info(f"Set options on selector, current value: {log_selector.value}")

            # Force refresh the selector
            log_selector.refresh()
            logger.info("Forced selector refresh")

            # Auto-select first option (TUI log) if nothing selected
            if options and (log_selector.value == "loading" or log_selector.value not in [opt[1] for opt in options]):
                logger.info(f"Auto-selecting first option: {options[0]}")
                log_selector.value = options[0][1]
                logger.info(f"About to show log file: {options[0][1]}")
                self._show_log_file(Path(options[0][1]))
                logger.info(f"Finished showing log file: {options[0][1]}")
            else:
                logger.info("Not auto-selecting, current selection is valid")

        except Exception as e:
            # FAIL LOUD - UI updates should not silently fail
            raise RuntimeError(f"Failed to update log selector: {e}") from e

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle log file selection change."""
        logger.info(f"Select changed: value={event.value}, type={type(event.value)}")

        # Handle NoSelection/BLANK - this should not happen if we always have TUI log
        if event.value == Select.BLANK or event.value is None:
            logger.error("CRITICAL: Select widget has no selection! This should never happen.")
            return

        # Handle valid selections
        if event.value and event.value != "loading" and event.value != "none":
            logger.info(f"Showing log file: {event.value}")
            self._show_log_file(Path(event.value))
        else:
            logger.warning(f"Ignoring invalid selection: {event.value}")

    def _show_log_file(self, log_path: Path) -> None:
        """Show the selected log file using proper Toolong structure."""
        logger.info(f"_show_log_file called with: {log_path}")
        try:
            log_container = self.query_one("#log_view_container", Container)
            logger.info(f"Found log container: {log_container}")

            # Clear existing content
            existing_widgets = log_container.query("*")
            logger.info(f"Clearing {len(existing_widgets)} existing widgets")
            existing_widgets.remove()

            # Create complete ToolongWidget - this encapsulates all Toolong functionality
            from openhcs.textual_tui.widgets.toolong_widget import ToolongWidget

            logger.info(f"Creating ToolongWidget for: {log_path}")

            # Create ToolongWidget for the selected file
            toolong_widget = ToolongWidget.from_single_file(
                str(log_path),
                can_tail=True
            )
            logger.info(f"Created ToolongWidget: {toolong_widget}")

            # Mount the complete ToolongWidget
            logger.info("Mounting ToolongWidget to container")
            log_container.mount(toolong_widget)

            logger.info(f"Successfully showing log file with ToolongWidget: {log_path}")

        except Exception as e:
            logger.error(f"Failed to show log file {log_path}: {e}", exc_info=True)
            # Show error message
            try:
                log_container = self.query_one("#log_view_container", Container)
                log_container.query("*").remove()
                log_container.mount(Static(f"Error loading log: {e}", classes="error-message"))
                logger.info("Mounted error message")
            except Exception as e2:
                logger.error(f"Failed to show error message: {e2}")

    def _sort_logs_for_display(self, logs: Set[Path]) -> List[Path]:
        """Sort logs for display: TUI first, then main subprocess, then workers by well ID."""
        tui_logs = []
        main_logs = []
        worker_logs = []
        unknown_logs = []

        for log_path in logs:
            log_info = self._log_info_cache.get(log_path)
            if not log_info:
                unknown_logs.append(log_path)
                continue

            if log_info.log_type == "tui":
                tui_logs.append(log_path)
            elif log_info.log_type == "main":
                main_logs.append(log_path)
            elif log_info.log_type == "worker":
                worker_logs.append((log_info.well_id or "", log_path))
            else:
                unknown_logs.append(log_path)

        # Sort workers by well ID
        worker_logs.sort(key=lambda x: x[0])

        return tui_logs + main_logs + [log_path for _, log_path in worker_logs] + unknown_logs

    def _start_file_watcher(self) -> None:
        """Start file system watcher for new log files."""
        if not self.base_log_path:
            logger.warning("Cannot start file watcher: no base log path")
            return

        base_path = Path(self.base_log_path)
        watch_directory = base_path.parent

        if not watch_directory.exists():
            logger.warning(f"Watch directory does not exist: {watch_directory}")
            return

        try:
            # Stop any existing watcher
            self._stop_file_watcher()

            # Create new watcher as daemon thread
            self._file_observer = Observer()
            self._file_observer.daemon = True  # Don't block app shutdown

            # Create event handler
            event_handler = ReactiveLogFileHandler(self)

            # Schedule watching
            self._file_observer.schedule(
                event_handler,
                str(watch_directory),
                recursive=False  # Only watch the log directory, not subdirectories
            )

            # Start watching
            self._file_observer.start()

            logger.info(f"Started file system watcher for: {watch_directory}")

        except Exception as e:
            logger.error(f"Failed to start file system watcher: {e}")
            self._file_observer = None

    def _stop_file_watcher(self) -> None:
        """Stop file system watcher with aggressive thread cleanup."""
        if self._file_observer:
            try:
                logger.debug("Stopping file system observer...")
                self._file_observer.stop()

                # Wait for observer thread to finish with timeout
                logger.debug("Waiting for file system observer thread to join...")
                self._file_observer.join(timeout=0.5)  # Shorter timeout

                if self._file_observer.is_alive():
                    logger.warning("File system observer thread did not stop cleanly, forcing cleanup")
                    # Force cleanup by setting daemon flag
                    try:
                        for thread in self._file_observer._threads:
                            if hasattr(thread, 'daemon'):
                                thread.daemon = True
                    except:
                        pass
                else:
                    logger.debug("File system observer stopped cleanly")

            except Exception as e:
                logger.error(f"Error stopping file system watcher: {e}")
            finally:
                self._file_observer = None

    def _handle_log_file_created(self, file_path: Path) -> None:
        """Handle creation of a new log file."""
        self._add_log_file(file_path)

    def _handle_log_file_modified(self, file_path: Path) -> None:
        """Handle modification of an existing log file."""
        # Toolong LogView handles live tailing automatically
        pass
