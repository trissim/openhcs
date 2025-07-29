"""
Function Selector Window for PyQt6

Function selection dialog with search and filtering capabilities.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from typing import Optional, Callable, List, Dict, Any
import inspect

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QTextEdit, QFrame,
    QSplitter, QGroupBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService

logger = logging.getLogger(__name__)


class FunctionSelectorWindow(QDialog):
    """
    PyQt6 Function Selector Window.
    
    Function selection dialog with search, filtering, and preview capabilities.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    function_selected = pyqtSignal(object)  # Selected function
    selection_cancelled = pyqtSignal()
    
    def __init__(self, on_result_callback: Optional[Callable] = None, parent=None):
        """
        Initialize the function selector window.
        
        Args:
            on_result_callback: Callback for function selection result
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Business logic state (extracted from Textual version)
        self.on_result_callback = on_result_callback
        self.function_registry = FunctionRegistryService()
        
        # Current state
        self.available_functions: Dict[str, Any] = {}
        self.filtered_functions: Dict[str, Any] = {}
        self.selected_function = None
        self.search_text = ""
        
        # UI components
        self.search_edit: Optional[QLineEdit] = None
        self.function_tree: Optional[QTreeWidget] = None
        self.preview_text: Optional[QTextEdit] = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.load_functions()
        
        logger.debug("Function selector window initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Select Function")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header_label = QLabel("Function Library")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(header_label)
        
        # Search section
        search_frame = self.create_search_section()
        layout.addWidget(search_frame)
        
        # Main content area
        content_splitter = self.create_content_area()
        layout.addWidget(content_splitter)
        
        # Button panel
        button_panel = self.create_button_panel()
        layout.addWidget(button_panel)
        
        # Set styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QTreeWidget {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                selection-background-color: #0078d4;
            }
            QTreeWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333333;
            }
            QTreeWidget::item:hover {
                background-color: #333333;
            }
            QLineEdit, QTextEdit {
                background-color: #404040;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 5px;
            }
        """)
    
    def create_search_section(self) -> QWidget:
        """
        Create the search section.
        
        Returns:
            Widget containing search controls
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Search label
        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        layout.addWidget(search_label)
        
        # Search edit
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search functions by name, module, or description...")
        self.search_edit.textChanged.connect(self.on_search_changed)
        layout.addWidget(self.search_edit)
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.setMaximumWidth(60)
        clear_button.clicked.connect(self.clear_search)
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        layout.addWidget(clear_button)
        
        return frame
    
    def create_content_area(self) -> QWidget:
        """
        Create the main content area.
        
        Returns:
            Widget containing function tree and preview
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Function tree
        function_group = QGroupBox("Available Functions")
        function_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00aaff;
            }
        """)
        
        function_layout = QVBoxLayout(function_group)
        
        self.function_tree = QTreeWidget()
        self.function_tree.setHeaderLabels(["Function", "Module"])
        self.function_tree.setRootIsDecorated(True)
        self.function_tree.setSortingEnabled(True)
        self.function_tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        function_layout.addWidget(self.function_tree)
        
        splitter.addWidget(function_group)
        
        # Preview panel
        preview_group = QGroupBox("Function Preview")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00aaff;
            }
        """)
        
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Courier New", 10))
        self.preview_text.setPlaceholderText("Select a function to see its details...")
        preview_layout.addWidget(self.preview_text)
        
        splitter.addWidget(preview_group)
        
        # Set splitter proportions
        splitter.setSizes([500, 400])
        
        return splitter
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(panel)
        
        # Function count label
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self.count_label)
        
        layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.cancel_selection)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                border: 1px solid #ff0000;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #dd0000;
            }
        """)
        layout.addWidget(cancel_button)
        
        # Select button
        self.select_button = QPushButton("Select")
        self.select_button.setMinimumWidth(80)
        self.select_button.setEnabled(False)
        self.select_button.clicked.connect(self.confirm_selection)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 1px solid #106ebe;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
        """)
        layout.addWidget(self.select_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        if self.function_tree:
            self.function_tree.itemSelectionChanged.connect(self.on_selection_changed)
            self.function_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
    
    def load_functions(self):
        """Load available functions from the registry."""
        try:
            # Get functions from registry (extracted from Textual version)
            self.available_functions = self.function_registry.get_all_functions()
            self.filtered_functions = self.available_functions.copy()
            
            self.update_function_tree()
            self.update_count_label()
            
            logger.debug(f"Loaded {len(self.available_functions)} functions")
            
        except Exception as e:
            logger.error(f"Failed to load functions: {e}")
            self.available_functions = {}
            self.filtered_functions = {}
    
    def update_function_tree(self):
        """Update the function tree display."""
        self.function_tree.clear()
        
        # Group functions by module
        modules = {}
        for func_name, func_info in self.filtered_functions.items():
            module_name = getattr(func_info.get('function'), '__module__', 'Unknown')
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append((func_name, func_info))
        
        # Create tree items
        for module_name, functions in sorted(modules.items()):
            module_item = QTreeWidgetItem(self.function_tree)
            module_item.setText(0, module_name)
            module_item.setText(1, f"({len(functions)} functions)")
            module_item.setExpanded(True)
            
            # Add function items
            for func_name, func_info in sorted(functions):
                func_item = QTreeWidgetItem(module_item)
                func_item.setText(0, func_name)
                func_item.setText(1, module_name)
                func_item.setData(0, Qt.ItemDataRole.UserRole, func_info)
                
                # Add icon or styling based on function type
                func_item.setToolTip(0, func_info['function'].__doc__ or "No description available")
    
    def update_count_label(self):
        """Update the function count label."""
        total = len(self.available_functions)
        filtered = len(self.filtered_functions)
        
        if filtered == total:
            self.count_label.setText(f"{total} functions")
        else:
            self.count_label.setText(f"{filtered} of {total} functions")
    
    def on_search_changed(self, text: str):
        """Handle search text changes."""
        self.search_text = text.lower().strip()
        self.filter_functions()
    
    def filter_functions(self):
        """Filter functions based on search text."""
        if not self.search_text:
            self.filtered_functions = self.available_functions.copy()
        else:
            self.filtered_functions = {}
            
            for func_name, func_info in self.available_functions.items():
                # Search in function name
                if self.search_text in func_name.lower():
                    self.filtered_functions[func_name] = func_info
                    continue
                
                # Search in module name
                module_name = getattr(func_info.get('function'), '__module__', '')
                if self.search_text in module_name.lower():
                    self.filtered_functions[func_name] = func_info
                    continue
                
                # Search in docstring
                docstring = getattr(func_info.get('function'), '__doc__', '') or ''
                if self.search_text in docstring.lower():
                    self.filtered_functions[func_name] = func_info
                    continue
        
        self.update_function_tree()
        self.update_count_label()
    
    def clear_search(self):
        """Clear the search text."""
        self.search_edit.clear()
    
    def on_selection_changed(self):
        """Handle function tree selection changes."""
        selected_items = self.function_tree.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            func_info = item.data(0, Qt.ItemDataRole.UserRole)
            
            if func_info:  # Function item selected
                self.selected_function = func_info['function']
                self.update_preview(func_info)
                self.select_button.setEnabled(True)
            else:  # Module item selected
                self.selected_function = None
                self.preview_text.clear()
                self.select_button.setEnabled(False)
        else:
            self.selected_function = None
            self.preview_text.clear()
            self.select_button.setEnabled(False)
    
    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on function items."""
        func_info = item.data(0, Qt.ItemDataRole.UserRole)
        if func_info:  # Function item double-clicked
            self.selected_function = func_info['function']
            self.confirm_selection()
    
    def update_preview(self, func_info: Dict[str, Any]):
        """
        Update the function preview.
        
        Args:
            func_info: Function information dictionary
        """
        func = func_info['function']
        
        preview_text = f"Function: {func.__name__}\n"
        preview_text += f"Module: {func.__module__}\n\n"
        
        # Function signature
        try:
            sig = inspect.signature(func)
            preview_text += f"Signature:\n{func.__name__}{sig}\n\n"
        except Exception:
            preview_text += "Signature: Not available\n\n"
        
        # Docstring
        docstring = func.__doc__
        if docstring:
            preview_text += f"Description:\n{docstring.strip()}\n\n"
        else:
            preview_text += "Description: No documentation available\n\n"
        
        # Additional info
        if func.__file__:
            preview_text += f"Source: {func.__file__}\n"
        
        self.preview_text.setPlainText(preview_text)
    
    def confirm_selection(self):
        """Confirm the function selection."""
        if not self.selected_function:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please select a function.")
            return
        
        # Emit signal and call callback
        self.function_selected.emit(self.selected_function)
        
        if self.on_result_callback:
            self.on_result_callback(self.selected_function)
        
        self.accept()
        logger.debug(f"Function selected: {self.selected_function.__name__}")
    
    def cancel_selection(self):
        """Cancel the function selection."""
        self.selection_cancelled.emit()
        self.reject()
        logger.debug("Function selection cancelled")


# Convenience function for opening function selector
def open_function_selector_window(on_result_callback: Optional[Callable] = None, parent=None):
    """
    Open function selector window.
    
    Args:
        on_result_callback: Callback for function selection result
        parent: Parent widget
        
    Returns:
        Dialog result
    """
    selector = FunctionSelectorWindow(on_result_callback, parent)
    return selector.exec()
