"""
PyQt6 Log Viewer Window

Provides comprehensive log viewing capabilities with real-time tailing, search functionality,
and integration with OpenHCS subprocess execution. Reimplements log viewing using Qt widgets
for native desktop integration.
"""

import logging
from typing import Optional, List, Set, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QTextEdit, QToolBar, QLineEdit, QCheckBox, QPushButton, QDialog
)
from PyQt6.QtGui import QSyntaxHighlighter, QTextDocument
from PyQt6.QtCore import QObject, QTimer, QFileSystemWatcher, pyqtSignal, Qt, QRegularExpression
from PyQt6.QtGui import QTextCharFormat, QColor, QAction, QFont, QTextCursor

from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.widgets.reactive_log_monitor import LogFileInfo
from openhcs.pyqt_gui.utils.log_detection_utils import (
    get_current_tui_log_path, discover_logs, classify_log_file, is_relevant_log_file
)

logger = logging.getLogger(__name__)


class LogFileDetector(QObject):
    """
    Detects new log files in directory using efficient file monitoring.
    
    Uses QFileSystemWatcher to monitor directory changes and set operations
    for efficient new file detection. Handles base_log_path as file prefix
    and watches the parent directory.
    """
    
    # Signals
    new_log_detected = pyqtSignal(object)  # LogFileInfo object

    def __init__(self, base_log_path: Optional[str] = None):
        """
        Initialize LogFileDetector.
        
        Args:
            base_log_path: Base path for subprocess log files (file prefix, not directory)
        """
        super().__init__()
        self._base_log_path = base_log_path
        self._previous_files: Set[Path] = set()
        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._watching_directory: Optional[Path] = None
        
        logger.debug(f"LogFileDetector initialized with base_log_path: {base_log_path}")

    def start_watching(self, directory: Path) -> None:
        """
        Start watching directory for new log files.
        
        Args:
            directory: Directory to watch for new log files
        """
        if not directory.exists():
            logger.warning(f"Cannot watch non-existent directory: {directory}")
            return
            
        # Stop any existing watching
        self.stop_watching()
        
        # Add directory to watcher
        success = self._watcher.addPath(str(directory))
        if success:
            self._watching_directory = directory
            # Initialize previous files set
            self._previous_files = self.scan_directory(directory)
            logger.debug(f"Started watching directory: {directory}")
            logger.debug(f"Initial file count: {len(self._previous_files)}")
        else:
            logger.error(f"Failed to add directory to watcher: {directory}")

    def stop_watching(self) -> None:
        """Stop file watching and cleanup."""
        if self._watching_directory:
            self._watcher.removePath(str(self._watching_directory))
            self._watching_directory = None
            self._previous_files.clear()
            logger.debug("Stopped file watching")

    def scan_directory(self, directory: Path) -> Set[Path]:
        """
        Scan directory for .log files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Set[Path]: Set of Path objects for .log files found
        """
        try:
            log_files = set(directory.glob("*.log"))
            logger.debug(f"Scanned directory {directory}: found {len(log_files)} .log files")
            return log_files
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Error scanning directory {directory}: {e}")
            return set()

    def detect_new_files(self, current_files: Set[Path]) -> Set[Path]:
        """
        Use set.difference() to find new files efficiently.
        
        Args:
            current_files: Current set of files in directory
            
        Returns:
            Set[Path]: Set of newly discovered files
        """
        new_files = current_files.difference(self._previous_files)
        if new_files:
            logger.debug(f"Detected {len(new_files)} new files: {[f.name for f in new_files]}")
        
        # Update previous files set
        self._previous_files = current_files
        return new_files

    def _on_directory_changed(self, directory_path: str) -> None:
        """
        Handle QFileSystemWatcher directory change signal.
        
        Args:
            directory_path: Path of directory that changed
        """
        directory = Path(directory_path)
        logger.debug(f"Directory changed: {directory}")
        
        # Scan directory for current files
        current_files = self.scan_directory(directory)
        
        # Detect new files
        new_files = self.detect_new_files(current_files)
        
        # Process new files
        for file_path in new_files:
            if file_path.exists() and is_relevant_log_file(file_path, self._base_log_path):
                try:
                    log_info = classify_log_file(file_path, self._base_log_path, include_tui_log=False)
                    logger.info(f"New relevant log file detected: {file_path} (type: {log_info.log_type})")
                    self.new_log_detected.emit(log_info)
                except Exception as e:
                    logger.error(f"Error classifying new log file {file_path}: {e}")


class LogHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter for log files.

    Provides highlighting for OpenHCS log format: YYYY-MM-DD HH:MM:SS,mmm - logger - level - message
    Highlights log levels (INFO, WARNING, ERROR, DEBUG) and timestamps.
    """

    def __init__(self, parent: QTextDocument):
        super().__init__(parent)
        self.highlighting_rules = []
        self.setup_highlighting_rules()

    def setup_highlighting_rules(self) -> None:
        """Setup regex patterns and formats for log highlighting."""
        # Log level patterns with colors
        log_levels = [
            ("INFO", QColor(0, 100, 200)),      # Blue
            ("WARNING", QColor(255, 165, 0)),   # Orange/Yellow
            ("ERROR", QColor(200, 0, 0)),       # Red
            ("DEBUG", QColor(128, 128, 128)),   # Gray
        ]

        for level, color in log_levels:
            pattern = QRegularExpression(rf"\b{level}\b")
            format = QTextCharFormat()
            format.setForeground(color)
            format.setFontWeight(QFont.Weight.Bold)
            self.highlighting_rules.append((pattern, format))

        # Timestamp pattern: YYYY-MM-DD HH:MM:SS,mmm
        timestamp_pattern = QRegularExpression(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}")
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(QColor(100, 100, 100))  # Dark gray
        timestamp_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((timestamp_pattern, timestamp_format))

        logger.debug(f"Setup {len(self.highlighting_rules)} highlighting rules")

    def highlightBlock(self, text: str) -> None:
        """
        Apply highlighting to text block.

        Args:
            text: Text block to highlight
        """
        for pattern, format in self.highlighting_rules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, format)


class LogViewerWindow(QMainWindow):
    """Main log viewer window with dropdown, search, and real-time tailing."""
    
    window_closed = pyqtSignal()

    def __init__(self, file_manager: FileManager, service_adapter, parent=None):
        super().__init__(parent)
        self.file_manager = file_manager
        self.service_adapter = service_adapter

        # State
        self.current_log_path: Optional[Path] = None
        self.current_file_position: int = 0
        self.auto_scroll_enabled: bool = True
        self.tailing_paused: bool = False

        # Search state
        self.current_search_text: str = ""
        self.search_highlights: List[QTextCursor] = []

        # Components
        self.log_selector: QComboBox = None
        self.search_toolbar: QToolBar = None
        self.log_display: QTextEdit = None
        self.file_detector: LogFileDetector = None
        self.tail_timer: QTimer = None
        self.highlighter: LogHighlighter = None

        self.setup_ui()
        self.setup_connections()
        self.initialize_logs()

    def setup_ui(self) -> None:
        """Setup complete UI layout with exact widget hierarchy."""
        self.setWindowTitle("Log Viewer")
        self.setMinimumSize(800, 600)

        # Central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Log selector dropdown
        self.log_selector = QComboBox()
        self.log_selector.setMinimumHeight(30)
        main_layout.addWidget(self.log_selector)

        # Search toolbar (initially hidden)
        self.search_toolbar = QToolBar("Search")
        self.search_toolbar.setVisible(False)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs...")
        self.search_toolbar.addWidget(self.search_input)

        # Search options
        self.case_sensitive_cb = QCheckBox("Case sensitive")
        self.search_toolbar.addWidget(self.case_sensitive_cb)

        self.regex_cb = QCheckBox("Regex")
        self.search_toolbar.addWidget(self.regex_cb)

        # Search navigation buttons
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.close_search_button = QPushButton("Close")

        self.search_toolbar.addWidget(self.prev_button)
        self.search_toolbar.addWidget(self.next_button)
        self.search_toolbar.addWidget(self.close_search_button)

        main_layout.addWidget(self.search_toolbar)

        # Log display area
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))  # Monospace font for logs
        main_layout.addWidget(self.log_display)

        # Control buttons layout
        control_layout = QHBoxLayout()

        self.auto_scroll_btn = QPushButton("Auto-scroll")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)

        self.clear_btn = QPushButton("Clear")
        self.bottom_btn = QPushButton("Bottom")

        control_layout.addWidget(self.auto_scroll_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.bottom_btn)
        control_layout.addStretch()  # Push buttons to left

        main_layout.addLayout(control_layout)

        # Setup syntax highlighting
        self.highlighter = LogHighlighter(self.log_display.document())

        # Setup window-local Ctrl+F shortcut
        search_action = QAction("Search", self)
        search_action.setShortcut("Ctrl+F")
        search_action.triggered.connect(self.toggle_search_toolbar)
        self.addAction(search_action)

        logger.debug("LogViewerWindow UI setup complete")

    def setup_connections(self) -> None:
        """Setup signal/slot connections."""
        # Log selector
        self.log_selector.currentIndexChanged.connect(self.on_log_selection_changed)

        # Search functionality
        self.search_input.returnPressed.connect(self.perform_search)
        self.prev_button.clicked.connect(self.find_previous)
        self.next_button.clicked.connect(self.find_next)
        self.close_search_button.clicked.connect(self.toggle_search_toolbar)

        # Control buttons
        self.auto_scroll_btn.toggled.connect(self.toggle_auto_scroll)
        self.pause_btn.toggled.connect(self.toggle_pause_tailing)
        self.clear_btn.clicked.connect(self.clear_log_display)
        self.bottom_btn.clicked.connect(self.scroll_to_bottom)

        logger.debug("LogViewerWindow connections setup complete")

    def initialize_logs(self) -> None:
        """Initialize with TUI log and start monitoring."""
        try:
            # Discover initial logs (TUI log only initially)
            initial_logs = discover_logs(base_log_path=None, include_tui_log=True)
            self.populate_log_dropdown(initial_logs)

            # Switch to TUI log if available
            if initial_logs:
                tui_logs = [log for log in initial_logs if log.log_type == "tui"]
                if tui_logs:
                    self.switch_to_log(tui_logs[0].path)

            logger.info("Log viewer initialized with TUI log")
        except Exception as e:
            logger.error(f"Failed to initialize logs: {e}")
            self.log_display.setText(f"Error initializing logs: {e}")

    # Dropdown Management Methods
    def populate_log_dropdown(self, log_files: List[LogFileInfo]) -> None:
        """
        Populate QComboBox with log files. Store LogFileInfo as item data.

        Args:
            log_files: List of LogFileInfo objects to add to dropdown
        """
        self.log_selector.clear()

        # Sort logs: TUI first, main subprocess, then workers by timestamp
        sorted_logs = sorted(log_files, key=self._log_sort_key)

        for log_info in sorted_logs:
            self.log_selector.addItem(log_info.display_name, log_info)

        logger.debug(f"Populated dropdown with {len(log_files)} log files")

    def _log_sort_key(self, log_info: LogFileInfo) -> tuple:
        """
        Generate sort key for log files.

        Args:
            log_info: LogFileInfo to generate sort key for

        Returns:
            tuple: Sort key (priority, timestamp)
        """
        # Priority: TUI=0, main=1, worker=2, unknown=3
        priority_map = {"tui": 0, "main": 1, "worker": 2, "unknown": 3}
        priority = priority_map.get(log_info.log_type, 3)

        # Use file modification time as secondary sort
        try:
            timestamp = log_info.path.stat().st_mtime
        except (OSError, AttributeError):
            timestamp = 0

        return (priority, -timestamp)  # Negative timestamp for newest first

    def clear_subprocess_logs(self) -> None:
        """Remove all non-TUI logs from dropdown and switch to TUI log."""
        current_logs = []

        # Collect TUI logs only
        for i in range(self.log_selector.count()):
            log_info = self.log_selector.itemData(i)
            if log_info and log_info.log_type == "tui":
                current_logs.append(log_info)

        # Repopulate with TUI logs only
        self.populate_log_dropdown(current_logs)

        # Auto-select TUI log if available
        if current_logs:
            self.switch_to_log(current_logs[0].path)

        logger.info("Cleared subprocess logs, kept TUI logs")

    def add_new_log(self, log_file_info: LogFileInfo) -> None:
        """
        Add new log to dropdown maintaining sort order.

        Args:
            log_file_info: New LogFileInfo to add
        """
        # Get current logs
        current_logs = []
        for i in range(self.log_selector.count()):
            log_info = self.log_selector.itemData(i)
            if log_info:
                current_logs.append(log_info)

        # Add new log
        current_logs.append(log_file_info)

        # Repopulate with updated list
        self.populate_log_dropdown(current_logs)

        logger.info(f"Added new log to dropdown: {log_file_info.display_name}")

    def on_log_selection_changed(self, index: int) -> None:
        """
        Handle dropdown selection change - switch log display.

        Args:
            index: Selected index in dropdown
        """
        if index >= 0:
            log_info = self.log_selector.itemData(index)
            if log_info:
                self.switch_to_log(log_info.path)

    def switch_to_log(self, log_path: Path) -> None:
        """
        Switch log display to show specified log file.

        Args:
            log_path: Path to log file to display
        """
        try:
            # Stop current tailing
            if self.tail_timer and self.tail_timer.isActive():
                self.tail_timer.stop()

            # Validate file exists
            if not log_path.exists():
                self.log_display.setText(f"Log file not found: {log_path}")
                return

            # Load log file content
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            self.log_display.setText(content)
            self.current_log_path = log_path
            self.current_file_position = len(content.encode('utf-8'))

            # Start tailing if not paused
            if not self.tailing_paused:
                self.start_log_tailing(log_path)

            # Scroll to bottom if auto-scroll enabled
            if self.auto_scroll_enabled:
                self.scroll_to_bottom()

            logger.info(f"Switched to log file: {log_path}")

        except Exception as e:
            logger.error(f"Error switching to log {log_path}: {e}")
            raise

    # Search Functionality Methods
    def toggle_search_toolbar(self) -> None:
        """Show/hide search toolbar (Ctrl+F handler)."""
        if self.search_toolbar.isVisible():
            # Hide toolbar and clear highlights
            self.search_toolbar.setVisible(False)
            self.clear_search_highlights()
        else:
            # Show toolbar and focus search input
            self.search_toolbar.setVisible(True)
            self.search_input.setFocus()
            self.search_input.selectAll()

    def perform_search(self) -> None:
        """Search in log display using QTextEdit.find()."""
        search_text = self.search_input.text()
        if not search_text:
            self.clear_search_highlights()
            return

        # Clear previous highlights if search text changed
        if search_text != self.current_search_text:
            self.clear_search_highlights()
            self.current_search_text = search_text
            self.highlight_all_matches(search_text)

        # Find next occurrence
        flags = QTextDocument.FindFlag(0)
        if self.case_sensitive_cb.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively

        found = self.log_display.find(search_text, flags)
        if not found:
            # Try from beginning
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.log_display.setTextCursor(cursor)
            self.log_display.find(search_text, flags)

    def highlight_all_matches(self, search_text: str) -> None:
        """
        Highlight all matches of search text in the document.

        Args:
            search_text: Text to search and highlight
        """
        if not search_text:
            return

        # Create highlight format
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor(255, 255, 0, 100))  # Yellow with transparency

        # Search through entire document
        document = self.log_display.document()
        cursor = QTextCursor(document)

        flags = QTextDocument.FindFlag(0)
        if self.case_sensitive_cb.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively

        self.search_highlights.clear()

        while True:
            cursor = document.find(search_text, cursor, flags)
            if cursor.isNull():
                break

            # Apply highlight
            cursor.mergeCharFormat(highlight_format)
            self.search_highlights.append(cursor)

        logger.debug(f"Highlighted {len(self.search_highlights)} search matches")

    def clear_search_highlights(self) -> None:
        """Clear all search highlights from the document."""
        # Reset format for all highlighted text
        for cursor in self.search_highlights:
            if not cursor.isNull():
                # Reset to default format
                default_format = QTextCharFormat()
                cursor.setCharFormat(default_format)

        self.search_highlights.clear()
        self.current_search_text = ""

    def find_next(self) -> None:
        """Find next search result."""
        self.perform_search()

    def find_previous(self) -> None:
        """Find previous search result."""
        search_text = self.search_input.text()
        if not search_text:
            return

        flags = QTextDocument.FindFlag.FindBackward
        if self.case_sensitive_cb.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively

        found = self.log_display.find(search_text, flags)
        if not found:
            # Try from end
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_display.setTextCursor(cursor)
            self.log_display.find(search_text, flags)

    # Control Button Methods
    def toggle_auto_scroll(self, enabled: bool) -> None:
        """Toggle auto-scroll to bottom."""
        self.auto_scroll_enabled = enabled
        logger.debug(f"Auto-scroll {'enabled' if enabled else 'disabled'}")

    def toggle_pause_tailing(self, paused: bool) -> None:
        """Toggle pause/resume log tailing."""
        self.tailing_paused = paused
        if paused and self.tail_timer:
            self.tail_timer.stop()
        elif not paused and self.current_log_path:
            self.start_log_tailing(self.current_log_path)
        logger.debug(f"Log tailing {'paused' if paused else 'resumed'}")

    def clear_log_display(self) -> None:
        """Clear current log display content."""
        self.log_display.clear()
        logger.debug("Log display cleared")

    def scroll_to_bottom(self) -> None:
        """Scroll log display to bottom."""
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())



    # Real-time Tailing Methods
    def start_log_tailing(self, log_path: Path) -> None:
        """
        Start tailing log file with QTimer (100ms interval).

        Args:
            log_path: Path to log file to tail
        """
        # Stop any existing timer
        if self.tail_timer:
            self.tail_timer.stop()

        # Create new timer
        self.tail_timer = QTimer()
        self.tail_timer.timeout.connect(self.read_log_incremental)
        self.tail_timer.start(100)  # 100ms interval

        logger.debug(f"Started tailing log file: {log_path}")

    def stop_log_tailing(self) -> None:
        """Stop current log tailing."""
        if self.tail_timer:
            self.tail_timer.stop()
            self.tail_timer = None
        logger.debug("Stopped log tailing")

    def read_log_incremental(self) -> None:
        """Read new content from current log file (track file position)."""
        if not self.current_log_path or not self.current_log_path.exists():
            return

        try:
            # Get current file size
            current_size = self.current_log_path.stat().st_size

            # Handle log rotation (file size decreased)
            if current_size < self.current_file_position:
                logger.info(f"Log rotation detected for {self.current_log_path}")
                self.current_file_position = 0
                # Optionally clear display or add rotation marker
                self.log_display.append("\n--- Log rotated ---\n")

            # Read new content if file grew
            if current_size > self.current_file_position:
                with open(self.current_log_path, 'rb') as f:
                    f.seek(self.current_file_position)
                    new_data = f.read(current_size - self.current_file_position)

                # Decode new content
                try:
                    new_content = new_data.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    new_content = new_data.decode('latin-1', errors='replace')

                if new_content:
                    # Check if user has scrolled up (disable auto-scroll)
                    scrollbar = self.log_display.verticalScrollBar()
                    was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

                    # Append new content
                    cursor = self.log_display.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    cursor.insertText(new_content)

                    # Auto-scroll if enabled and user was at bottom
                    if self.auto_scroll_enabled and was_at_bottom:
                        self.scroll_to_bottom()

                    # Update file position
                    self.current_file_position = current_size

        except (OSError, PermissionError) as e:
            logger.warning(f"Error reading log file {self.current_log_path}: {e}")
            # Handle file deletion/recreation
            if not self.current_log_path.exists():
                logger.info(f"Log file deleted: {self.current_log_path}")
                self.log_display.append(f"\n--- Log file deleted: {self.current_log_path} ---\n")
                # Try to reconnect after a delay
                QTimer.singleShot(1000, self._attempt_reconnection)
        except Exception as e:
            logger.error(f"Unexpected error in log tailing: {e}")
            raise

    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to log file after deletion."""
        if self.current_log_path and self.current_log_path.exists():
            logger.info(f"Log file recreated, reconnecting: {self.current_log_path}")
            self.current_file_position = 0
            self.log_display.append(f"\n--- Reconnected to: {self.current_log_path} ---\n")
            # File will be read on next timer tick

    # External Integration Methods
    def start_monitoring(self, base_log_path: str) -> None:
        """Start monitoring for new subprocess logs."""
        if self.file_detector:
            self.file_detector.stop_watching()

        # Extract directory from base_log_path (file prefix)
        log_directory = Path(base_log_path).parent

        # Create new detector
        self.file_detector = LogFileDetector(base_log_path)
        self.file_detector.new_log_detected.connect(self.add_new_log)
        self.file_detector.start_watching(log_directory)

        logger.info(f"Started monitoring for new logs in: {log_directory}")

    def stop_monitoring(self) -> None:
        """Stop monitoring for new logs."""
        if self.file_detector:
            self.file_detector.stop_watching()
            self.file_detector = None
        logger.info("Stopped monitoring for new logs")

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        if self.file_detector:
            self.file_detector.stop_watching()
        if self.tail_timer:
            self.tail_timer.stop()
        self.window_closed.emit()
        super().closeEvent(event)
