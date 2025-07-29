"""
Function Selector Dialog for PyQt6 GUI.

Mirrors the Textual TUI FunctionSelectorWindow functionality using the same
FunctionRegistryService and business logic.
"""

import logging
from typing import Callable, Optional, Dict, List, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, 
    QTreeWidgetItem, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# REUSE the actual working Textual TUI services
from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService

logger = logging.getLogger(__name__)


class FunctionSelectorDialog(QDialog):
    """
    Function selector dialog that mirrors Textual TUI FunctionSelectorWindow.
    
    Uses the same FunctionRegistryService and business logic for consistency.
    """
    
    # Signals
    function_selected = pyqtSignal(object)  # Selected function
    
    def __init__(self, current_function: Optional[Callable] = None, parent=None):
        """
        Initialize function selector dialog.
        
        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.current_function = current_function
        self.selected_function = None
        
        # Load function data using Textual TUI service (reuse working logic)
        self.functions_by_backend = FunctionRegistryService.get_functions_by_backend()
        self.all_functions = []
        for backend, functions in self.functions_by_backend.items():
            self.all_functions.extend(functions)
        
        self.setup_ui()
        self.setup_connections()
        self.populate_function_tree()
        
        logger.debug(f"Function selector initialized with {len(self.all_functions)} functions")
    
    def setup_ui(self):
        """Setup the user interface (mirrors Textual TUI layout)."""
        self.setWindowTitle("Select Function")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Select Function")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Search input (mirrors Textual TUI)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search functions...")
        layout.addWidget(self.search_input)
        
        # Function tree (mirrors Textual TUI tree structure)
        self.function_tree = QTreeWidget()
        self.function_tree.setHeaderLabel("Functions")
        self.function_tree.setRootIsDecorated(True)
        layout.addWidget(self.function_tree)
        
        # Buttons (mirrors Textual TUI dialog-buttons)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.select_btn = QPushButton("Select")
        self.select_btn.setEnabled(False)
        self.select_btn.setDefault(True)
        button_layout.addWidget(self.select_btn)
        
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Connect buttons
        self.select_btn.clicked.connect(self.accept_selection)
        cancel_btn.clicked.connect(self.reject)
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Search functionality (mirrors Textual TUI)
        self.search_input.textChanged.connect(self.filter_functions)
        
        # Tree selection (mirrors Textual TUI)
        self.function_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        self.function_tree.itemDoubleClicked.connect(self.on_tree_double_click)
    
    def populate_function_tree(self, functions_by_backend: Optional[Dict] = None):
        """Populate function tree (mirrors Textual TUI _build_function_tree)."""
        if functions_by_backend is None:
            functions_by_backend = self.functions_by_backend
        
        self.function_tree.clear()
        
        # Add backend nodes (same structure as Textual TUI)
        for backend, functions in functions_by_backend.items():
            backend_item = QTreeWidgetItem(self.function_tree)
            backend_item.setText(0, f"{backend} ({len(functions)} functions)")
            backend_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "backend", "name": backend})
            backend_item.setExpanded(True)  # Expand all backend nodes by default
            
            # Add function nodes
            for func, display_name in functions:
                func_item = QTreeWidgetItem(backend_item)
                func_item.setText(0, display_name)
                func_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "function", "func": func, "name": display_name})
                
                # Highlight current function if it matches
                if self.current_function and func == self.current_function:
                    func_item.setSelected(True)
                    self.function_tree.setCurrentItem(func_item)
                    self.selected_function = func
                    self.select_btn.setEnabled(True)
    
    def filter_functions(self, search_term: str):
        """Filter functions based on search term (mirrors Textual TUI _filter_functions)."""
        if not search_term.strip():
            # Show all functions
            self.populate_function_tree()
        else:
            # Filter functions (same logic as Textual TUI)
            search_lower = search_term.lower()
            filtered_functions = {}
            
            for backend, functions in self.functions_by_backend.items():
                matching_functions = [
                    (func, display_name) for func, display_name in functions
                    if search_lower in display_name.lower()
                ]
                if matching_functions:
                    filtered_functions[backend] = matching_functions
            
            self.populate_function_tree(filtered_functions)
    
    def on_tree_selection_changed(self):
        """Handle tree selection changes (mirrors Textual TUI on_tree_node_selected)."""
        selected_items = self.function_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            data = item.data(0, Qt.ItemDataRole.UserRole)
            
            if data and data.get("type") == "function":
                self.selected_function = data["func"]
                self.select_btn.setEnabled(True)
            else:
                self.selected_function = None
                self.select_btn.setEnabled(False)
        else:
            self.selected_function = None
            self.select_btn.setEnabled(False)
    
    def on_tree_double_click(self, item, column):
        """Handle tree double-click (quick selection)."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get("type") == "function":
            self.selected_function = data["func"]
            self.accept_selection()
    
    def accept_selection(self):
        """Accept the selected function."""
        if self.selected_function:
            self.function_selected.emit(self.selected_function)
            self.accept()
    
    def get_selected_function(self) -> Optional[Callable]:
        """Get the selected function."""
        return self.selected_function
    
    @staticmethod
    def select_function(current_function: Optional[Callable] = None, parent=None) -> Optional[Callable]:
        """
        Static method to show function selector and return selected function.
        
        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget
            
        Returns:
            Selected function or None if cancelled
        """
        dialog = FunctionSelectorDialog(current_function, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_selected_function()
        return None
