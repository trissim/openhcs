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
from openhcs.core.log_utils import LogFileInfo
from openhcs.pyqt_gui.utils.log_detection_utils import (
    get_current_tui_log_path, discover_logs, discover_all_logs
)
from openhcs.core.log_utils import (
    classify_log_file, is_openhcs_log_file, infer_base_log_path
)

# Import Pygments for advanced syntax highlighting
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.formatters import get_formatter_by_name
from pygments.token import Token
from pygments.style import Style
from pygments.styles import get_style_by_name
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LogColorScheme:
    """
    Centralized color scheme for log highlighting with semantic color names.

    Supports light/dark theme variants and ensures WCAG accessibility compliance.
    All colors meet minimum 4.5:1 contrast ratio for normal text readability.
    """

    # Log level colors with semantic meaning (WCAG 4.5:1 compliant)
    log_critical_fg: Tuple[int, int, int] = (255, 255, 255)  # White text
    log_critical_bg: Tuple[int, int, int] = (139, 0, 0)      # Dark red background
    log_error_color: Tuple[int, int, int] = (255, 85, 85)    # Brighter red - WCAG compliant
    log_warning_color: Tuple[int, int, int] = (255, 140, 0)  # Dark orange - attention grabbing
    log_info_color: Tuple[int, int, int] = (100, 160, 210)   # Brighter steel blue - WCAG compliant
    log_debug_color: Tuple[int, int, int] = (160, 160, 160)  # Lighter gray - better contrast

    # Metadata and structural colors
    timestamp_color: Tuple[int, int, int] = (105, 105, 105)      # Dim gray - unobtrusive
    logger_name_color: Tuple[int, int, int] = (147, 112, 219)   # Medium slate blue - distinctive
    memory_address_color: Tuple[int, int, int] = (255, 182, 193) # Light pink - technical data
    file_path_color: Tuple[int, int, int] = (34, 139, 34)       # Forest green - file system

    # Python syntax colors (following VS Code dark theme conventions)
    python_keyword_color: Tuple[int, int, int] = (86, 156, 214)    # Blue - language keywords
    python_string_color: Tuple[int, int, int] = (206, 145, 120)    # Orange - string literals
    python_number_color: Tuple[int, int, int] = (181, 206, 168)    # Light green - numeric values
    python_operator_color: Tuple[int, int, int] = (212, 212, 212)  # Light gray - operators/punctuation
    python_name_color: Tuple[int, int, int] = (156, 220, 254)      # Light blue - identifiers
    python_function_color: Tuple[int, int, int] = (220, 220, 170)  # Yellow - function names
    python_class_color: Tuple[int, int, int] = (78, 201, 176)      # Teal - class names
    python_builtin_color: Tuple[int, int, int] = (86, 156, 214)    # Blue - built-in functions
    python_comment_color: Tuple[int, int, int] = (106, 153, 85)    # Green - comments

    # Special highlighting colors
    exception_color: Tuple[int, int, int] = (255, 69, 0)       # Red orange - error types
    function_call_color: Tuple[int, int, int] = (255, 215, 0)  # Gold - function invocations
    boolean_color: Tuple[int, int, int] = (86, 156, 214)       # Blue - True/False/None

    # Enhanced syntax colors (Phase 1 additions)
    tuple_parentheses_color: Tuple[int, int, int] = (255, 215, 0)     # Gold - tuple delimiters
    set_braces_color: Tuple[int, int, int] = (255, 140, 0)            # Dark orange - set delimiters
    class_representation_color: Tuple[int, int, int] = (78, 201, 176) # Teal - <class 'name'>
    function_representation_color: Tuple[int, int, int] = (220, 220, 170) # Yellow - <function name>
    module_path_color: Tuple[int, int, int] = (147, 112, 219)         # Medium slate blue - module.path
    hex_number_color: Tuple[int, int, int] = (181, 206, 168)          # Light green - 0xFF
    scientific_notation_color: Tuple[int, int, int] = (181, 206, 168) # Light green - 1.23e-4
    binary_number_color: Tuple[int, int, int] = (181, 206, 168)       # Light green - 0b1010
    octal_number_color: Tuple[int, int, int] = (181, 206, 168)        # Light green - 0o755
    python_special_color: Tuple[int, int, int] = (255, 20, 147)       # Deep pink - __name__
    single_quoted_string_color: Tuple[int, int, int] = (206, 145, 120) # Orange - 'string'
    list_comprehension_color: Tuple[int, int, int] = (156, 220, 254)  # Light blue - [x for x in y]
    generator_expression_color: Tuple[int, int, int] = (156, 220, 254) # Light blue - (x for x in y)

    @classmethod
    def create_dark_theme(cls) -> 'LogColorScheme':
        """
        Create a dark theme variant with adjusted colors for dark backgrounds.

        Returns:
            LogColorScheme: Dark theme color scheme with higher contrast
        """
        return cls(
            # Enhanced colors for dark backgrounds with better contrast
            log_error_color=(255, 100, 100),    # Brighter red
            log_info_color=(120, 180, 230),     # Brighter steel blue
            timestamp_color=(160, 160, 160),    # Lighter gray
            python_string_color=(236, 175, 150), # Brighter orange
            python_number_color=(200, 230, 190), # Brighter green
            # Other colors remain the same as they work well on dark backgrounds
        )

    @classmethod
    def create_light_theme(cls) -> 'LogColorScheme':
        """
        Create a light theme variant with adjusted colors for light backgrounds.

        Returns:
            LogColorScheme: Light theme color scheme with appropriate contrast
        """
        return cls(
            # Darker colors for light backgrounds with WCAG compliance
            log_error_color=(180, 20, 40),       # Darker red
            log_info_color=(30, 80, 130),        # Darker steel blue
            log_warning_color=(200, 100, 0),     # Darker orange
            timestamp_color=(60, 60, 60),        # Darker gray
            logger_name_color=(100, 60, 160),    # Darker slate blue
            python_string_color=(150, 80, 60),   # Darker orange
            python_number_color=(120, 140, 100), # Darker green
            memory_address_color=(200, 120, 140), # Darker pink
            file_path_color=(20, 100, 20),       # Darker forest green
            exception_color=(200, 40, 0),        # Darker red orange
            # Adjust other colors for light background contrast
        )

    def to_qcolor(self, color_tuple: Tuple[int, int, int]) -> QColor:
        """
        Convert RGB tuple to QColor object.

        Args:
            color_tuple: RGB color tuple (r, g, b)

        Returns:
            QColor: Qt color object
        """
        return QColor(*color_tuple)


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
            if file_path.exists() and is_openhcs_log_file(file_path):
                try:
                    # For general watching, try to infer base_log_path from the file name
                    effective_base_log_path = self._base_log_path
                    if not effective_base_log_path and 'subprocess_' in file_path.name:
                        effective_base_log_path = infer_base_log_path(file_path)

                    log_info = classify_log_file(file_path, effective_base_log_path,
                                                include_tui_log=False)

                    logger.info(f"New relevant log file detected: {file_path} (type: {log_info.log_type})")
                    self.new_log_detected.emit(log_info)
                except Exception as e:
                    logger.error(f"Error classifying new log file {file_path}: {e}")


class LogHighlighter(QSyntaxHighlighter):
    """
    Advanced syntax highlighter for log files using Pygments.

    Provides sophisticated highlighting for OpenHCS log format with support for:
    - Log levels and timestamps
    - Python code snippets and data structures
    - Memory addresses and function signatures
    - Complex nested dictionaries and lists
    - Exception tracebacks and file paths
    """

    def __init__(self, parent: QTextDocument, color_scheme: LogColorScheme = None):
        """
        Initialize the log highlighter with optional color scheme.

        Args:
            parent: QTextDocument to apply highlighting to
            color_scheme: Color scheme to use (defaults to dark theme)
        """
        super().__init__(parent)
        self.color_scheme = color_scheme or LogColorScheme()
        self.setup_pygments_styles()
        self.setup_highlighting_rules()

    def setup_pygments_styles(self) -> None:
        """Setup Pygments token to QTextCharFormat mapping using color scheme."""
        cs = self.color_scheme  # Shorthand for readability

        # Create a mapping from Pygments tokens to Qt text formats
        self.token_formats = {
            # Log levels with distinct colors and backgrounds
            'log_critical': self._create_format(
                cs.to_qcolor(cs.log_critical_fg),
                cs.to_qcolor(cs.log_critical_bg),
                bold=True
            ),
            'log_error': self._create_format(cs.to_qcolor(cs.log_error_color), bold=True),
            'log_warning': self._create_format(cs.to_qcolor(cs.log_warning_color), bold=True),
            'log_info': self._create_format(cs.to_qcolor(cs.log_info_color), bold=True),
            'log_debug': self._create_format(cs.to_qcolor(cs.log_debug_color)),

            # Timestamps and metadata
            'timestamp': self._create_format(cs.to_qcolor(cs.timestamp_color)),
            'logger_name': self._create_format(cs.to_qcolor(cs.logger_name_color), bold=True),

            # Python syntax highlighting (for complex data structures)
            Token.Keyword: self._create_format(cs.to_qcolor(cs.python_keyword_color), bold=True),
            Token.String: self._create_format(cs.to_qcolor(cs.python_string_color)),
            Token.String.Single: self._create_format(cs.to_qcolor(cs.python_string_color)),
            Token.String.Double: self._create_format(cs.to_qcolor(cs.python_string_color)),
            Token.Number: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Number.Integer: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Number.Float: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Number.Hex: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Number.Oct: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Number.Bin: self._create_format(cs.to_qcolor(cs.python_number_color)),
            Token.Operator: self._create_format(cs.to_qcolor(cs.python_operator_color)),
            Token.Punctuation: self._create_format(cs.to_qcolor(cs.python_operator_color)),
            Token.Name: self._create_format(cs.to_qcolor(cs.python_name_color)),
            Token.Name.Function: self._create_format(cs.to_qcolor(cs.python_function_color), bold=True),
            Token.Name.Class: self._create_format(cs.to_qcolor(cs.python_class_color), bold=True),
            Token.Name.Builtin: self._create_format(cs.to_qcolor(cs.python_builtin_color)),
            Token.Comment: self._create_format(cs.to_qcolor(cs.python_comment_color)),
            Token.Literal: self._create_format(cs.to_qcolor(cs.python_number_color)),

            # Special patterns for log content
            'memory_address': self._create_format(cs.to_qcolor(cs.memory_address_color)),
            'file_path': self._create_format(cs.to_qcolor(cs.file_path_color)),
            'exception': self._create_format(cs.to_qcolor(cs.exception_color), bold=True),
            'function_call': self._create_format(cs.to_qcolor(cs.function_call_color)),
            'dict_key': self._create_format(cs.to_qcolor(cs.python_name_color)),
            'boolean': self._create_format(cs.to_qcolor(cs.boolean_color), bold=True),

            # Enhanced Python syntax elements (Phase 1)
            'tuple_parentheses': self._create_format(cs.to_qcolor(cs.tuple_parentheses_color)),
            'set_braces': self._create_format(cs.to_qcolor(cs.set_braces_color)),
            'class_representation': self._create_format(cs.to_qcolor(cs.class_representation_color), bold=True),
            'function_representation': self._create_format(cs.to_qcolor(cs.function_representation_color), bold=True),
            'module_path': self._create_format(cs.to_qcolor(cs.module_path_color)),
            'hex_number': self._create_format(cs.to_qcolor(cs.hex_number_color)),
            'scientific_notation': self._create_format(cs.to_qcolor(cs.scientific_notation_color)),
            'binary_number': self._create_format(cs.to_qcolor(cs.binary_number_color)),
            'octal_number': self._create_format(cs.to_qcolor(cs.octal_number_color)),
            'python_special': self._create_format(cs.to_qcolor(cs.python_special_color), bold=True),
            'single_quoted_string': self._create_format(cs.to_qcolor(cs.single_quoted_string_color)),
            'list_comprehension': self._create_format(cs.to_qcolor(cs.list_comprehension_color)),
            'generator_expression': self._create_format(cs.to_qcolor(cs.generator_expression_color)),
        }

    def _create_format(self, fg_color: QColor, bg_color: QColor = None, bold: bool = False) -> QTextCharFormat:
        """Create a QTextCharFormat with specified properties."""
        format = QTextCharFormat()
        format.setForeground(fg_color)
        if bg_color:
            format.setBackground(bg_color)
        if bold:
            format.setFontWeight(QFont.Weight.Bold)
        return format

    def setup_highlighting_rules(self) -> None:
        """Setup regex patterns for log-specific highlighting."""
        self.highlighting_rules = []

        # Log level patterns (highest priority)
        log_levels = [
            ("CRITICAL", self.token_formats['log_critical']),
            ("ERROR", self.token_formats['log_error']),
            ("WARNING", self.token_formats['log_warning']),
            ("INFO", self.token_formats['log_info']),
            ("DEBUG", self.token_formats['log_debug']),
        ]

        for level, format in log_levels:
            pattern = QRegularExpression(rf"\b{level}\b")
            self.highlighting_rules.append((pattern, format))

        # Timestamp pattern: YYYY-MM-DD HH:MM:SS,mmm
        timestamp_pattern = QRegularExpression(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}")
        self.highlighting_rules.append((timestamp_pattern, self.token_formats['timestamp']))

        # Logger names (e.g., openhcs.core.orchestrator)
        logger_pattern = QRegularExpression(r"openhcs\.[a-zA-Z0-9_.]+")
        self.highlighting_rules.append((logger_pattern, self.token_formats['logger_name']))

        # Memory addresses (e.g., 0x7f1640dd8e00)
        memory_pattern = QRegularExpression(r"0x[0-9a-fA-F]+")
        self.highlighting_rules.append((memory_pattern, self.token_formats['memory_address']))

        # File paths in tracebacks
        filepath_pattern = QRegularExpression(r'["\']?/[^"\'\s]+\.py["\']?')
        self.highlighting_rules.append((filepath_pattern, self.token_formats['file_path']))

        # Exception names
        exception_pattern = QRegularExpression(r'\b[A-Z][a-zA-Z]*Error\b|\b[A-Z][a-zA-Z]*Exception\b')
        self.highlighting_rules.append((exception_pattern, self.token_formats['exception']))

        # Function calls with parentheses
        function_pattern = QRegularExpression(r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)')
        self.highlighting_rules.append((function_pattern, self.token_formats['function_call']))

        # Boolean values
        boolean_pattern = QRegularExpression(r'\b(True|False|None)\b')
        self.highlighting_rules.append((boolean_pattern, self.token_formats['boolean']))

        # Enhanced Python syntax elements

        # Single-quoted strings (complement to double-quoted)
        single_quote_pattern = QRegularExpression(r"'[^']*'")
        self.highlighting_rules.append((single_quote_pattern, self.token_formats['single_quoted_string']))

        # Class representations: <class 'module.ClassName'>
        class_repr_pattern = QRegularExpression(r"<class '[^']*'>")
        self.highlighting_rules.append((class_repr_pattern, self.token_formats['class_representation']))

        # Function representations: <function name at 0xaddress>
        function_repr_pattern = QRegularExpression(r"<function [^>]+ at 0x[0-9a-fA-F]+>")
        self.highlighting_rules.append((function_repr_pattern, self.token_formats['function_representation']))

        # Extended module paths (beyond just openhcs)
        module_path_pattern = QRegularExpression(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){2,}")
        self.highlighting_rules.append((module_path_pattern, self.token_formats['module_path']))

        # Hexadecimal numbers (beyond memory addresses): 0xFF, 0x1A2B
        hex_number_pattern = QRegularExpression(r"\b0[xX][0-9a-fA-F]+\b")
        self.highlighting_rules.append((hex_number_pattern, self.token_formats['hex_number']))

        # Scientific notation: 1.23e-4, 5.67E+10
        scientific_pattern = QRegularExpression(r"\b\d+\.?\d*[eE][+-]?\d+\b")
        self.highlighting_rules.append((scientific_pattern, self.token_formats['scientific_notation']))

        # Binary literals: 0b1010
        binary_pattern = QRegularExpression(r"\b0[bB][01]+\b")
        self.highlighting_rules.append((binary_pattern, self.token_formats['binary_number']))

        # Octal literals: 0o755
        octal_pattern = QRegularExpression(r"\b0[oO][0-7]+\b")
        self.highlighting_rules.append((octal_pattern, self.token_formats['octal_number']))

        # Python special constants: __name__, __main__, __file__, etc.
        python_special_pattern = QRegularExpression(r"\b__[a-zA-Z_][a-zA-Z0-9_]*__\b")
        self.highlighting_rules.append((python_special_pattern, self.token_formats['python_special']))

        logger.debug(f"Setup {len(self.highlighting_rules)} highlighting rules")

    def set_color_scheme(self, color_scheme: LogColorScheme) -> None:
        """
        Update the color scheme and refresh highlighting.

        Args:
            color_scheme: New color scheme to apply
        """
        self.color_scheme = color_scheme
        self.setup_pygments_styles()
        self.setup_highlighting_rules()
        # Trigger re-highlighting of the entire document
        self.rehighlight()
        logger.debug(f"Applied new color scheme with {len(self.token_formats)} token formats")

    def switch_to_dark_theme(self) -> None:
        """Switch to dark theme color scheme."""
        self.set_color_scheme(LogColorScheme.create_dark_theme())

    def switch_to_light_theme(self) -> None:
        """Switch to light theme color scheme."""
        self.set_color_scheme(LogColorScheme.create_light_theme())

    @classmethod
    def load_color_scheme_from_config(cls, config_path: str = None) -> LogColorScheme:
        """
        Load color scheme from external configuration file.

        Args:
            config_path: Path to JSON/YAML config file (optional)

        Returns:
            LogColorScheme: Loaded color scheme or default if file not found
        """
        if config_path and Path(config_path).exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Create color scheme from config
                scheme_kwargs = {}
                for key, value in config.items():
                    if key.endswith('_color') or key.endswith('_fg') or key.endswith('_bg'):
                        if isinstance(value, list) and len(value) == 3:
                            scheme_kwargs[key] = tuple(value)

                return LogColorScheme(**scheme_kwargs)

            except Exception as e:
                logger.warning(f"Failed to load color scheme from {config_path}: {e}")

        return LogColorScheme()  # Return default scheme

    def highlightBlock(self, text: str) -> None:
        """
        Apply advanced highlighting to text block.

        Uses both regex patterns for log-specific content and Pygments
        for Python syntax highlighting of complex data structures.
        """
        # First apply log-specific patterns
        for pattern, format in self.highlighting_rules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, format)

        # Then apply Pygments highlighting for Python-like content
        self._highlight_python_content(text)

    def _highlight_python_content(self, text: str) -> None:
        """
        Apply Pygments Python syntax highlighting to parts of the text that contain
        Python data structures (dictionaries, lists, function signatures, etc.).
        """
        try:
            # Look for Python-like patterns in the log line
            python_patterns = [
                # Dictionary patterns: {'key': 'value', ...}
                r'\{[^{}]*:[^{}]*\}',
                # List patterns: [item1, item2, ...]
                r'\[[^\[\]]*,.*?\]',
                # Function signatures: function_name(arg1=value, arg2=value)
                r'\b[a-zA-Z_][a-zA-Z0-9_]*\([^)]*=.*?\)',
                # Complex nested structures
                r'\{.*?:\s*\[.*?\].*?\}',

                # Enhanced patterns for Phase 1

                # Tuple patterns: (item1, item2, item3)
                r'\([^()]*,.*?\)',
                # Set patterns: {item1, item2, item3} (no colons, distinguishes from dict)
                r'\{[^{}:]*,.*?\}',
                # List comprehensions: [x for x in items]
                r'\[[^\[\]]*\s+for\s+[^\[\]]*\s+in\s+[^\[\]]*\]',
                # Generator expressions: (x for x in items)
                r'\([^()]*\s+for\s+[^()]*\s+in\s+[^()]*\)',
                # Class representations: <class 'module.ClassName'>
                r"<class '[^']*'>",
                # Function representations: <function name at 0xaddress>
                r"<function [^>]+ at 0x[0-9a-fA-F]+>",
                # Complex function calls with keyword arguments
                r'\b[a-zA-Z_][a-zA-Z0-9_]*\([^)]*[a-zA-Z_][a-zA-Z0-9_]*\s*=.*?\)',
                # Multi-line dictionary/list structures (single line representation)
                r'\{[^{}]*:\s*[^{}]*,\s*[^{}]*:\s*[^{}]*\}',
                # Nested collections: [{...}, {...}] or [(...), (...)]
                r'\[[\{\(][^[\]]*[\}\)],\s*[\{\(][^[\]]*[\}\)]\]',
            ]

            for pattern in python_patterns:
                regex = QRegularExpression(pattern)
                iterator = regex.globalMatch(text)

                while iterator.hasNext():
                    match = iterator.next()
                    start = match.capturedStart()
                    length = match.capturedLength()
                    python_text = match.captured(0)

                    # Use Pygments to highlight this Python-like content
                    self._apply_pygments_highlighting(python_text, start)

        except Exception as e:
            # Don't let highlighting errors break the log viewer
            logger.debug(f"Error in Python content highlighting: {e}")

    def _apply_pygments_highlighting(self, python_text: str, start_offset: int) -> None:
        """
        Apply Pygments highlighting to a specific piece of Python-like text.
        """
        try:
            from pygments.lexers import PythonLexer

            lexer = PythonLexer()
            tokens = list(lexer.get_tokens(python_text))

            current_pos = 0
            for token_type, token_value in tokens:
                if token_value.strip():  # Skip whitespace-only tokens
                    token_start = start_offset + current_pos
                    token_length = len(token_value)

                    # Apply format if we have a mapping for this token type
                    if token_type in self.token_formats:
                        format = self.token_formats[token_type]
                        self.setFormat(token_start, token_length, format)

                current_pos += len(token_value)

        except Exception as e:
            # Don't let Pygments errors break the highlighting
            logger.debug(f"Error in Pygments highlighting: {e}")


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
        """Initialize with main process log only and start monitoring."""
        # Only discover the current main process log, not old logs
        try:
            from openhcs.core.log_utils import get_current_log_file_path, classify_log_file
            from pathlib import Path

            main_log_path = get_current_log_file_path()
            main_log = Path(main_log_path)
            if main_log.exists():
                log_info = classify_log_file(main_log, None, True)
                self.populate_log_dropdown([log_info])
                self.switch_to_log(main_log)
        except Exception:
            # Main log not available, continue without it
            pass

        # Start monitoring for new logs
        self.start_monitoring()

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
        import traceback
        logger.error(f"🔥 DEBUG: clear_subprocess_logs called! Stack trace:")
        for line in traceback.format_stack():
            logger.error(f"🔥 DEBUG: {line.strip()}")

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
    def start_monitoring(self, base_log_path: Optional[str] = None) -> None:
        """Start monitoring for new logs."""
        if self.file_detector:
            self.file_detector.stop_watching()

        # Get log directory
        log_directory = Path(base_log_path).parent if base_log_path else Path.home() / ".local" / "share" / "openhcs" / "logs"

        # Start file watching
        self.file_detector = LogFileDetector(base_log_path)
        self.file_detector.new_log_detected.connect(self.add_new_log)
        self.file_detector.start_watching(log_directory)

    def stop_monitoring(self) -> None:
        """Stop monitoring for new logs."""
        if self.file_detector:
            self.file_detector.stop_watching()
            self.file_detector = None
        logger.info("Stopped monitoring for new logs")

    def cleanup(self) -> None:
        """Cleanup all resources and background processes."""
        try:
            # Stop tailing timer
            if hasattr(self, 'tail_timer') and self.tail_timer and self.tail_timer.isActive():
                self.tail_timer.stop()
                self.tail_timer.deleteLater()
                self.tail_timer = None

            # Stop file monitoring
            self.stop_monitoring()

            # Clean up file detector
            if hasattr(self, 'file_detector') and self.file_detector:
                self.file_detector.stop_watching()
                self.file_detector = None

        except Exception as e:
            logger.warning(f"Error during log viewer cleanup: {e}")

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        if self.file_detector:
            self.file_detector.stop_watching()
        if self.tail_timer:
            self.tail_timer.stop()
        self.window_closed.emit()
        super().closeEvent(event)
