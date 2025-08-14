"""
Enhanced Path Widget for PyQt6 GUI

Provides intelligent path selection with browse button functionality.
Uses standard Qt dialogs for consistency with the rest of OpenHCS.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QHBoxLayout, QFileDialog
from PyQt6.QtCore import pyqtSignal

from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.core.path_cache import PathCacheKey, get_cached_dialog_path, cache_dialog_path
from openhcs.textual_tui.widgets.shared.signature_analyzer import ParameterInfo

logger = logging.getLogger(__name__)


@dataclass
class PathBehavior:
    """Defines behavior for path widget based on parameter analysis."""
    is_directory: bool = False
    extensions: Optional[List[str]] = None
    cache_key: PathCacheKey = PathCacheKey.GENERAL
    description: str = "path"

    @property
    def title(self) -> str:
        """Generate appropriate dialog title."""
        if self.is_directory:
            return "Select Directory"
        elif self.extensions:
            ext_str = ", ".join(self.extensions)
            return f"Select File ({ext_str})"
        else:
            return "Select Path"

    @property
    def file_filter(self) -> str:
        """Generate Qt file filter string."""
        if self.extensions:
            # Create filter like "Image Files (*.tiff *.png);;All Files (*)"
            ext_pattern = " ".join(f"*{ext}" for ext in self.extensions)
            filter_name = f"{self.extensions[0].upper()} Files" if len(self.extensions) == 1 else "Files"
            return f"{filter_name} ({ext_pattern});;All Files (*)"
        else:
            return "All Files (*)"


class PathBehaviorDetector:
    """Detects appropriate path behavior from parameter names and docstring hints."""

    @staticmethod
    def detect_behavior(param_name: str, param_info: Optional[ParameterInfo] = None) -> PathBehavior:
        """
        Detect path behavior from parameter name and optional parameter info.

        Args:
            param_name: Parameter name to analyze
            param_info: Optional parameter info with docstring description

        Returns:
            PathBehavior with detected settings
        """
        # Get base behavior from parameter name
        base_behavior = PathBehaviorDetector._detect_from_parameter_name(param_name)

        # Try to enhance with docstring info
        if param_info and param_info.description:
            docstring_behavior = PathBehaviorDetector._parse_docstring_hints(param_info.description)
            if docstring_behavior:
                # Merge: docstring extensions + parameter name cache key
                return PathBehavior(
                    is_directory=docstring_behavior.is_directory,
                    extensions=docstring_behavior.extensions,
                    cache_key=base_behavior.cache_key if base_behavior else PathCacheKey.GENERAL,
                    description=docstring_behavior.description
                )

        # Fall back to base behavior or smart default
        return base_behavior or PathBehavior(
            is_directory=False,
            extensions=None,
            cache_key=PathCacheKey.GENERAL,
            description="file or directory"
        )

    @staticmethod
    def _parse_docstring_hints(description: str) -> Optional[PathBehavior]:
        """Parse docstring for path behavior hints."""
        desc_lower = description.lower()

        # Directory specification
        if any(pattern in desc_lower for pattern in ["directory only", "folder only", "dir only"]):
            return PathBehavior(is_directory=True, cache_key=PathCacheKey.DIRECTORY_SELECTION, description="directory")

        # Extension patterns: (.ext only), (.ext1, .ext2), (.ext1/.ext2), etc.
        patterns = [
            r'\(\.([a-zA-Z0-9]+(?:\s*[,/]\s*\.?[a-zA-Z0-9]+)*)\)',  # (.json, .yaml) or (.json/.yaml)
            r'\(\.([a-zA-Z0-9]+)\s+only\)',                         # (.tiff only)
            r'\(([a-zA-Z0-9]+)\s+only\)',                           # (tiff only)
            r'\.([a-zA-Z0-9]+)\s+only',                             # .tiff only
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                ext_string = match.group(1)
                # Split by comma or slash and clean up
                raw_exts = re.split(r'[,/]', ext_string)
                extensions = [f".{ext.strip().lstrip('.')}" for ext in raw_exts if ext.strip()]

                if extensions:
                    desc = f"{extensions[0].upper()} file" if len(extensions) == 1 else f"file ({', '.join(ext.upper() for ext in extensions)})"
                    return PathBehavior(is_directory=False, extensions=extensions, cache_key=PathCacheKey.FILE_SELECTION, description=desc)

        return None

    @staticmethod
    def _detect_from_parameter_name(param_name: str) -> Optional[PathBehavior]:
        """Detect behavior from parameter name patterns."""
        name_lower = param_name.lower()

        # Directory patterns
        if any(pattern in name_lower for pattern in ['dir', 'folder', 'directory']):
            return PathBehavior(is_directory=True, cache_key=PathCacheKey.DIRECTORY_SELECTION, description="directory")

        # File patterns
        if any(pattern in name_lower for pattern in ['file']):
            return PathBehavior(is_directory=False, cache_key=PathCacheKey.FILE_SELECTION, description="file")

        # Context-specific cache keys (NO EXTENSIONS - docstring handles that)
        if 'pipeline' in name_lower:
            return PathBehavior(is_directory=False, cache_key=PathCacheKey.PIPELINE_FILES, description="pipeline file")
        if 'step' in name_lower:
            return PathBehavior(is_directory=False, cache_key=PathCacheKey.STEP_SETTINGS, description="step file")
        if 'function' in name_lower or 'func' in name_lower:
            return PathBehavior(is_directory=False, cache_key=PathCacheKey.FUNCTION_PATTERNS, description="function file")

        return None


class EnhancedPathWidget(QWidget):
    """Enhanced path widget with browse button using standard Qt dialogs."""

    path_changed = pyqtSignal(str)

    def __init__(self, param_name: str, current_value: Any, param_info: Optional[ParameterInfo] = None, color_scheme=None):
        """
        Initialize enhanced path widget.

        Args:
            param_name: Parameter name for behavior detection
            current_value: Current path value
            param_info: Optional parameter info with docstring
            color_scheme: Color scheme for styling
        """
        super().__init__()
        self.behavior = PathBehaviorDetector.detect_behavior(param_name, param_info)
        self.color_scheme = color_scheme or PyQt6ColorScheme()

        # Layout: [QLineEdit] [Browse Button]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(f"Enter {self.behavior.description} path...")
        self.browse_button = QPushButton("📁 Browse")
        self.browse_button.setMaximumWidth(80)

        layout.addWidget(self.path_input, 1)
        layout.addWidget(self.browse_button, 0)

        self._apply_styling()
        self._setup_signals()
        self.set_path(current_value)

    def _apply_styling(self):
        """Apply color scheme styling to widgets."""
        self.path_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px; padding: 5px; font-family: 'Courier New', monospace;
            }}
        """)

        self.browse_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_normal_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.button_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px; padding: 5px 10px; font-size: 11px;
            }}
        """)

    def _setup_signals(self):
        """Setup signal connections."""
        self.path_input.textChanged.connect(self._on_text_changed)
        self.browse_button.clicked.connect(self._open_dialog)

    def _on_text_changed(self, text: str):
        """Handle text change in path input."""
        self.path_changed.emit(text)

    def set_path(self, value: Any):
        """Set path value without triggering signals."""
        self.path_input.blockSignals(True)
        try:
            if value is not None:
                # Set actual value
                text = str(value)
                self.path_input.setText(text)
            else:
                # For None values, don't set empty text - let placeholder system handle it
                # This allows lazy placeholder text to be visible instead of hardcoded placeholder
                pass
        finally:
            self.path_input.blockSignals(False)

    def get_path(self) -> str:
        """Get current path value."""
        return self.path_input.text()

    def _open_dialog(self):
        """Open appropriate Qt dialog based on behavior."""
        try:
            # Get cached initial directory
            initial_dir = str(get_cached_dialog_path(self.behavior.cache_key, fallback=Path.home()))

            # Use None as parent to create a clean, top-level dialog
            # This prevents inheriting the dark styling from nested containers
            # and matches the simple appearance of ServiceAdapter dialogs
            parent = None

            if self.behavior.is_directory:
                # Use directory dialog
                selected_path = QFileDialog.getExistingDirectory(
                    parent,
                    self.behavior.title,
                    initial_dir
                )
            else:
                # Use file dialog
                selected_path, _ = QFileDialog.getOpenFileName(
                    parent,
                    self.behavior.title,
                    initial_dir,
                    self.behavior.file_filter
                )

            if selected_path:
                path_obj = Path(selected_path)
                self.set_path(path_obj)
                self.path_changed.emit(str(path_obj))

                # Cache the selection (directory for files, path itself for directories)
                cache_path = path_obj.parent if path_obj.is_file() else path_obj
                cache_dialog_path(self.behavior.cache_key, cache_path)

        except Exception as e:
            logger.error(f"Failed to open dialog: {e}")
