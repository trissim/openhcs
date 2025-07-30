"""
File Browser Window for PyQt6

File and directory browser dialog with filtering and selection capabilities.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from typing import Optional, Callable, List
from pathlib import Path
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeView, QLineEdit, QComboBox, QFrame, QWidget,
    QSplitter, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import Qt, QDir, pyqtSignal, QModelIndex
from PyQt6.QtGui import QFont

from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
logger = logging.getLogger(__name__)


class BrowserMode(Enum):
    """Browser mode enumeration (extracted from Textual version)."""
    LOAD = "load"
    SAVE = "save"


class SelectionMode(Enum):
    """Selection mode enumeration (extracted from Textual version)."""
    FILES_ONLY = "files_only"
    DIRECTORIES_ONLY = "directories_only"
    FILES_AND_DIRECTORIES = "files_and_directories"


class FileBrowserWindow(QDialog):
    """
    PyQt6 File Browser Window.
    
    File and directory browser with filtering, selection, and navigation.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    file_selected = pyqtSignal(list)  # List of selected paths
    selection_cancelled = pyqtSignal()
    
    def __init__(self, initial_path: Optional[Path] = None, 
                 mode: BrowserMode = BrowserMode.LOAD,
                 selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
                 filter_extensions: Optional[List[str]] = None,
                 title: str = "File Browser",
                 on_result_callback: Optional[Callable] = None, color_scheme: Optional[PyQt6ColorScheme] = None,
                 parent=None):
        """
        Initialize the file browser window.
        
        Args:
            initial_path: Initial directory path
            mode: Browser mode (load/save)
            selection_mode: Selection mode (files/directories/both)
            filter_extensions: List of file extensions to filter
            title: Window title
            on_result_callback: Callback for selection result
            parent: Parent widget
        """
        super().__init__(parent)

        # Initialize color scheme
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        
        # Business logic state (extracted from Textual version)
        self.initial_path = initial_path or Path.home()
        self.mode = mode
        self.selection_mode = selection_mode
        self.filter_extensions = filter_extensions or []
        self.on_result_callback = on_result_callback
        
        # Current state
        self.current_path = self.initial_path
        self.selected_paths: List[Path] = []
        
        # UI components
        self.file_model: Optional[QFileSystemModel] = None
        self.tree_view: Optional[QTreeView] = None
        self.path_edit: Optional[QLineEdit] = None
        self.filter_combo: Optional[QComboBox] = None
        self.filename_edit: Optional[QLineEdit] = None
        
        # Setup UI
        self.setup_ui(title)
        self.setup_connections()
        self.navigate_to_path(self.initial_path)
        
        logger.debug(f"File browser window initialized (mode={mode.value})")
    
    def setup_ui(self, title: str):
        """Setup the user interface."""
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 500)
        self.resize(900, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header with path navigation
        header_frame = self.create_header()
        layout.addWidget(header_frame)
        
        # Main content area
        content_splitter = self.create_content_area()
        layout.addWidget(content_splitter)
        
        # Footer with filename input (for save mode)
        if self.mode == BrowserMode.SAVE:
            footer_frame = self.create_footer()
            layout.addWidget(footer_frame)
        
        # Button panel
        button_panel = self.create_button_panel()
        layout.addWidget(button_panel)
        
        # Set styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.window_bg)};
                color: white;
            }}
            QTreeView {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                selection-background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
            QLineEdit, QComboBox {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
    
    def create_header(self) -> QWidget:
        """
        Create the header with path navigation.
        
        Returns:
            Widget containing navigation controls
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(frame)
        
        # Path label
        path_label = QLabel("Path:")
        path_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; font-weight: bold;")
        layout.addWidget(path_label)
        
        # Path edit
        self.path_edit = QLineEdit()
        self.path_edit.setText(str(self.current_path))
        self.path_edit.returnPressed.connect(self.on_path_entered)
        layout.addWidget(self.path_edit)
        
        # Up button
        up_button = QPushButton("↑ Up")
        up_button.setMaximumWidth(60)
        up_button.clicked.connect(self.navigate_up)
        up_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border-radius: 3px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }}
        """)
        layout.addWidget(up_button)
        
        # Filter combo
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; font-weight: bold;")
        layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Files (*)", "*")
        
        # Add extension filters
        if self.filter_extensions:
            for ext in self.filter_extensions:
                self.filter_combo.addItem(f"{ext.upper()} Files (*{ext})", f"*{ext}")
        
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        layout.addWidget(self.filter_combo)
        
        return frame
    
    def create_content_area(self) -> QWidget:
        """
        Create the main content area with file tree.
        
        Returns:
            Widget containing file browser
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # File system model and tree view
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(str(self.current_path))
        
        # Set name filters based on selection mode
        self.update_name_filters()
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(str(self.current_path)))
        
        # Configure tree view
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        
        # Hide unnecessary columns
        self.tree_view.hideColumn(1)  # Size
        self.tree_view.hideColumn(2)  # Type
        self.tree_view.hideColumn(3)  # Date Modified
        
        splitter.addWidget(self.tree_view)
        
        # Selection info panel
        info_panel = self.create_selection_info_panel()
        splitter.addWidget(info_panel)
        
        # Set splitter proportions
        splitter.setSizes([600, 200])
        
        return splitter
    
    def create_selection_info_panel(self) -> QWidget:
        """
        Create the selection information panel.
        
        Returns:
            Widget showing selection details
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        
        # Selection label
        selection_label = QLabel("Selection:")
        selection_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        selection_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};")
        layout.addWidget(selection_label)
        
        # Selection list
        self.selection_list = QListWidget()
        self.selection_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.selection_list)
        
        return frame
    
    def create_footer(self) -> QWidget:
        """
        Create the footer with filename input (save mode only).
        
        Returns:
            Widget containing filename input
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(frame)
        
        # Filename label
        filename_label = QLabel("Filename:")
        filename_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; font-weight: bold;")
        layout.addWidget(filename_label)
        
        # Filename edit
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Enter filename...")
        layout.addWidget(self.filename_edit)
        
        return frame
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 10px;
            }}
        """)
        
        layout = QHBoxLayout(panel)
        layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.cancel_selection)
        cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.status_error)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.status_error)};
                border-radius: 3px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.status_error)};
            }}
        """)
        layout.addWidget(cancel_button)
        
        # Select/Open button
        action_text = "Save" if self.mode == BrowserMode.SAVE else "Open"
        self.select_button = QPushButton(action_text)
        self.select_button.setMinimumWidth(80)
        self.select_button.clicked.connect(self.confirm_selection)
        self.select_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
                border-radius: 3px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
        """)
        layout.addWidget(self.select_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        if self.tree_view:
            self.tree_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
            self.tree_view.doubleClicked.connect(self.on_item_double_clicked)
    
    def update_name_filters(self):
        """Update file model name filters based on current settings."""
        if not self.file_model:
            return
        
        filters = []
        
        # Add extension filters
        if self.filter_extensions:
            for ext in self.filter_extensions:
                filters.append(f"*{ext}")
        
        # If no specific filters, show all files
        if not filters:
            filters = ["*"]
        
        self.file_model.setNameFilters(filters)
        
        # Set filter mode based on selection mode
        if self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            self.file_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        else:
            self.file_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
    
    def navigate_to_path(self, path: Path):
        """
        Navigate to specific path.
        
        Args:
            path: Path to navigate to
        """
        if path.exists():
            self.current_path = path
            
            if self.file_model and self.tree_view:
                self.file_model.setRootPath(str(path))
                self.tree_view.setRootIndex(self.file_model.index(str(path)))
            
            if self.path_edit:
                self.path_edit.setText(str(path))
            
            logger.debug(f"Navigated to: {path}")
    
    def navigate_up(self):
        """Navigate to parent directory."""
        parent_path = self.current_path.parent
        if parent_path != self.current_path:  # Not at root
            self.navigate_to_path(parent_path)
    
    def on_path_entered(self):
        """Handle manual path entry."""
        path_text = self.path_edit.text().strip()
        if path_text:
            try:
                path = Path(path_text)
                if path.exists():
                    self.navigate_to_path(path)
                else:
                    self.path_edit.setText(str(self.current_path))  # Reset to current
            except Exception as e:
                logger.warning(f"Invalid path entered: {e}")
                self.path_edit.setText(str(self.current_path))  # Reset to current
    
    def on_filter_changed(self, filter_text: str):
        """Handle filter changes."""
        self.update_name_filters()
    
    def on_selection_changed(self):
        """Handle tree view selection changes."""
        if not self.tree_view:
            return
        
        selected_indexes = self.tree_view.selectionModel().selectedIndexes()
        self.selected_paths = []
        
        for index in selected_indexes:
            if index.column() == 0:  # Only process name column
                file_path = Path(self.file_model.filePath(index))
                self.selected_paths.append(file_path)
        
        # Update selection list
        self.update_selection_list()
        
        # Update button state
        self.select_button.setEnabled(len(self.selected_paths) > 0)
    
    def on_item_double_clicked(self, index: QModelIndex):
        """Handle double-click on items."""
        file_path = Path(self.file_model.filePath(index))
        
        if file_path.is_dir():
            # Navigate into directory
            self.navigate_to_path(file_path)
        else:
            # Select file and confirm
            self.selected_paths = [file_path]
            self.confirm_selection()
    
    def update_selection_list(self):
        """Update the selection list display."""
        self.selection_list.clear()
        
        for path in self.selected_paths:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.selection_list.addItem(item)
    
    def confirm_selection(self):
        """Confirm the current selection."""
        if self.mode == BrowserMode.SAVE and self.filename_edit:
            # For save mode, use filename from input
            filename = self.filename_edit.text().strip()
            if filename:
                save_path = self.current_path / filename
                self.selected_paths = [save_path]
            elif not self.selected_paths:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Filename", "Please enter a filename or select a file.")
                return
        
        if not self.selected_paths:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please select a file or directory.")
            return
        
        # Emit signal and call callback
        self.file_selected.emit([str(path) for path in self.selected_paths])
        
        if self.on_result_callback:
            self.on_result_callback(self.selected_paths)
        
        self.accept()
        logger.debug(f"Selection confirmed: {[str(p) for p in self.selected_paths]}")
    
    def cancel_selection(self):
        """Cancel the selection."""
        self.selection_cancelled.emit()
        self.reject()
        logger.debug("Selection cancelled")


# Convenience function for opening file browser
def open_file_browser_window(initial_path: Optional[Path] = None,
                            mode: BrowserMode = BrowserMode.LOAD,
                            selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
                            filter_extensions: Optional[List[str]] = None,
                            title: str = "File Browser",
                            on_result_callback: Optional[Callable] = None, color_scheme: Optional[PyQt6ColorScheme] = None,
                            parent=None):
    """
    Open file browser window.
    
    Args:
        initial_path: Initial directory path
        mode: Browser mode (load/save)
        selection_mode: Selection mode (files/directories/both)
        filter_extensions: List of file extensions to filter
        title: Window title
        on_result_callback: Callback for selection result
        parent: Parent widget
        
    Returns:
        Dialog result
    """
    browser = FileBrowserWindow(
        initial_path=initial_path,
        mode=mode,
        selection_mode=selection_mode,
        filter_extensions=filter_extensions,
        title=title,
        on_result_callback=on_result_callback,
        parent=parent
    )
    return browser.exec()
