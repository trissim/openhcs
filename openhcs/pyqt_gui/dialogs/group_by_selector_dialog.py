"""
Group By Selector Dialog for PyQt6 GUI.

Mirrors the Textual TUI GroupBySelectorWindow functionality with dual list selection.
"""

import logging
from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, 
    QLabel, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class GroupBySelectorDialog(QDialog):
    """
    Group by selector dialog that mirrors Textual TUI GroupBySelectorWindow.
    
    Uses dual list selection with mathematical operations for moving items.
    """
    
    # Signals
    selection_changed = pyqtSignal(list)  # Selected components
    
    def __init__(self, available_components: List[str], selected_components: List[str], 
                 component_type: str = "channel", orchestrator=None, parent=None):
        """
        Initialize group by selector dialog.
        
        Args:
            available_components: List of available components
            selected_components: List of currently selected components
            component_type: Type of component (for display)
            orchestrator: Orchestrator for metadata access
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.available_components = available_components.copy()
        self.selected_components = selected_components.copy()
        self.component_type = component_type
        self.orchestrator = orchestrator
        
        # Calculate initial lists (same logic as Textual TUI) - sorted for consistency
        self.current_available = sorted([ch for ch in self.available_components if ch not in self.selected_components])
        self.current_selected = sorted(self.selected_components.copy())
        
        self.setup_ui()
        self.setup_connections()
        self.update_lists()
        
        logger.debug(f"Group by selector initialized: {len(self.current_available)} available, {len(self.current_selected)} selected")
    
    def setup_ui(self):
        """Setup the user interface (mirrors Textual TUI layout)."""
        self.setWindowTitle(f"Select {self.component_type.title()}s")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"Select {self.component_type.title()}s")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Top button row (mirrors Textual TUI)
        top_button_layout = QHBoxLayout()
        
        self.move_right_btn = QPushButton("→")
        self.move_right_btn.setMaximumWidth(40)
        self.move_right_btn.setToolTip("Move selected items to selected list")
        top_button_layout.addWidget(self.move_right_btn)
        
        self.move_left_btn = QPushButton("←")
        self.move_left_btn.setMaximumWidth(40)
        self.move_left_btn.setToolTip("Move selected items to available list")
        top_button_layout.addWidget(self.move_left_btn)
        
        top_button_layout.addStretch()
        
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.setMaximumWidth(50)
        self.select_all_btn.setToolTip("Select all available items")
        top_button_layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("None")
        self.select_none_btn.setMaximumWidth(50)
        self.select_none_btn.setToolTip("Clear all selections")
        top_button_layout.addWidget(self.select_none_btn)
        
        layout.addLayout(top_button_layout)
        
        # Dual lists container (mirrors Textual TUI)
        lists_layout = QHBoxLayout()
        
        # Available list
        available_container = QVBoxLayout()
        available_label = QLabel("Available")
        available_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        available_container.addWidget(available_label)
        
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        available_container.addWidget(self.available_list)
        
        available_widget = QWidget()
        available_widget.setLayout(available_container)
        lists_layout.addWidget(available_widget)
        
        # Selected list
        selected_container = QVBoxLayout()
        selected_label = QLabel("Selected")
        selected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        selected_container.addWidget(selected_label)
        
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        selected_container.addWidget(self.selected_list)
        
        selected_widget = QWidget()
        selected_widget.setLayout(selected_container)
        lists_layout.addWidget(selected_widget)
        
        layout.addLayout(lists_layout)
        
        # Bottom buttons (mirrors Textual TUI dialog-buttons)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Connect buttons
        ok_btn.clicked.connect(self.accept_selection)
        cancel_btn.clicked.connect(self.reject)
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Movement buttons (mirrors Textual TUI button handling)
        self.move_right_btn.clicked.connect(self.move_right)
        self.move_left_btn.clicked.connect(self.move_left)
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_none_btn.clicked.connect(self.select_none)
        
        # Double-click for quick movement
        self.available_list.itemDoubleClicked.connect(lambda: self.move_right())
        self.selected_list.itemDoubleClicked.connect(lambda: self.move_left())
    
    def update_lists(self):
        """Update both list widgets (mirrors Textual TUI _update_lists)."""
        # Clear and populate available list
        self.available_list.clear()
        for item in sorted(self.current_available):
            display_text = self._format_component_display(item)
            self.available_list.addItem(display_text)

        # Clear and populate selected list (sorted for consistency)
        self.selected_list.clear()
        for item in sorted(self.current_selected):
            display_text = self._format_component_display(item)
            self.selected_list.addItem(display_text)

        logger.debug(f"Updated lists: available={self.current_available}, selected={self.current_selected}")

    def _format_component_display(self, component_key: str) -> str:
        """
        Format component key for display with metadata if available (mirrors Textual TUI).

        Args:
            component_key: Component key (e.g., "1", "2", "A01")

        Returns:
            Formatted display string (e.g., "Channel 1 | HOECHST 33342" or "Channel 1")
        """
        base_text = f"{self.component_type.title()} {component_key}"

        # Try to get metadata name if orchestrator is available
        if self.orchestrator:
            try:
                # Convert component_type string back to GroupBy enum
                from openhcs.constants.constants import GroupBy
                group_by = GroupBy(self.component_type)
                metadata_name = self.orchestrator.get_component_metadata(group_by, component_key)

                if metadata_name:
                    return f"{base_text} | {metadata_name}"
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not get metadata for {self.component_type} {component_key}: {e}")

        return base_text

    def _extract_component_key(self, display_text: str) -> str:
        """
        Extract component key from formatted display text.

        Args:
            display_text: Formatted text like "Channel 1 | HOECHST 33342" or "Channel 1"

        Returns:
            Component key like "1"
        """
        # Extract the key from "Component_Type KEY" or "Component_Type KEY | metadata"
        parts = display_text.split(' | ')[0]  # Remove metadata part if present
        component_key = parts.split(' ')[-1]  # Get the last part (the key)
        return component_key

    def move_right(self):
        """Move selected items from available to selected (mirrors Textual TUI _move_right)."""
        selected_items = [item.text() for item in self.available_list.selectedItems()]

        for display_text in selected_items:
            component_key = self._extract_component_key(display_text)
            if component_key in self.current_available:
                self.current_available.remove(component_key)
                self.current_selected.append(component_key)

        self.update_lists()
    
    def move_left(self):
        """Move selected items from selected to available (mirrors Textual TUI _move_left)."""
        selected_items = [item.text() for item in self.selected_list.selectedItems()]

        for display_text in selected_items:
            component_key = self._extract_component_key(display_text)
            if component_key in self.current_selected:
                self.current_selected.remove(component_key)
                self.current_available.append(component_key)

        self.update_lists()
    
    def select_all(self):
        """Select all available items (mirrors Textual TUI _select_all)."""
        self.current_selected.extend(self.current_available)
        self.current_available.clear()
        self.update_lists()
    
    def select_none(self):
        """Clear all selections (mirrors Textual TUI _select_none)."""
        self.current_available.extend(self.current_selected)
        self.current_selected.clear()
        self.update_lists()
    
    def accept_selection(self):
        """Accept the current selection."""
        self.selection_changed.emit(self.current_selected.copy())
        self.accept()
    
    def get_selected_components(self) -> List[str]:
        """Get the selected components."""
        return self.current_selected.copy()
    
    @staticmethod
    def select_components(available_components: List[str], selected_components: List[str], 
                         component_type: str = "channel", orchestrator=None, parent=None) -> Optional[List[str]]:
        """
        Static method to show group by selector and return selected components.
        
        Args:
            available_components: List of available components
            selected_components: List of currently selected components
            component_type: Type of component (for display)
            orchestrator: Orchestrator for metadata access
            parent: Parent widget
            
        Returns:
            Selected components or None if cancelled
        """
        dialog = GroupBySelectorDialog(available_components, selected_components, component_type, orchestrator, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_selected_components()
        return None
