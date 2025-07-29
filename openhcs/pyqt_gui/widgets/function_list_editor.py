"""
Function List Editor Widget for PyQt6 GUI.

Mirrors the Textual TUI FunctionListEditorWidget with sophisticated parameter forms.
Displays a scrollable list of function panes with Add/Load/Save/Code controls.
"""

import logging
from typing import List, Union, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.pyqt_gui.widgets.function_pane import FunctionPaneWidget
from openhcs.constants.constants import GroupBy, VariableComponents

logger = logging.getLogger(__name__)


class FunctionListEditorWidget(QWidget):
    """
    Function list editor widget that mirrors Textual TUI functionality.
    
    Displays functions with parameter editing, Add/Delete/Reset buttons,
    and Load/Save/Code functionality.
    """
    
    # Signals
    function_pattern_changed = pyqtSignal()
    
    def __init__(self, initial_functions: Union[List, Dict, callable, None] = None, 
                 step_identifier: str = None, service_adapter=None, parent=None):
        super().__init__(parent)
        
        # Initialize services (reuse existing business logic)
        self.registry_service = FunctionRegistryService()
        self.data_manager = PatternDataManager()
        self.service_adapter = service_adapter

        # Step identifier for cache isolation
        self.step_identifier = step_identifier or f"widget_{id(self)}"

        # Step configuration properties (mirrors Textual TUI)
        self.current_group_by = None  # Current GroupBy setting from step editor
        self.current_variable_components = []  # Current VariableComponents list from step editor
        self.selected_channel = None  # Currently selected channel
        self.available_channels = []  # Available channels from orchestrator
        self.is_dict_mode = False  # Whether we're in channel-specific mode

        # Component selection cache per GroupBy (mirrors Textual TUI)
        self.component_selections = {}

        # Initialize pattern data and mode
        self._initialize_pattern_data(initial_functions)
        
        # UI components
        self.function_panes = []
        
        self.setup_ui()
        self.setup_connections()
        
        logger.debug(f"Function list editor initialized with {len(self.functions)} functions")
    
    def _initialize_pattern_data(self, initial_functions):
        """Initialize pattern data from various input formats (mirrors Textual TUI logic)."""
        if initial_functions is None:
            self.pattern_data = []
            self.is_dict_mode = False
            self.functions = []
        elif callable(initial_functions):
            self.pattern_data = [(initial_functions, {})]
            self.is_dict_mode = False
            self.functions = [(initial_functions, {})]
        elif isinstance(initial_functions, list):
            self.pattern_data = initial_functions
            self.is_dict_mode = False
            self.functions = self._normalize_function_list(initial_functions)
        elif isinstance(initial_functions, dict):
            # Convert any integer keys to string keys for consistency
            normalized_dict = {}
            for key, value in initial_functions.items():
                str_key = str(key)
                normalized_dict[str_key] = self._normalize_function_list(value) if value else []

            self.pattern_data = normalized_dict
            self.is_dict_mode = True

            # Set selected channel to first key and load its functions
            if normalized_dict:
                self.selected_channel = next(iter(normalized_dict.keys()))
                self.functions = normalized_dict[self.selected_channel]
            else:
                self.selected_channel = None
                self.functions = []
        else:
            logger.warning(f"Unknown initial_functions type: {type(initial_functions)}")
            self.pattern_data = []
            self.is_dict_mode = False
            self.functions = []

    def _normalize_function_list(self, func_list):
        """Normalize function list using PatternDataManager."""
        normalized = []
        for item in func_list:
            func, kwargs = self.data_manager.extract_func_and_kwargs(item)
            if func:
                normalized.append((func, kwargs))
        return normalized
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header with controls (mirrors Textual TUI)
        header_layout = QHBoxLayout()
        
        functions_label = QLabel("Functions")
        functions_label.setStyleSheet("color: #4a9eff; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(functions_label)
        
        header_layout.addStretch()
        
        # Control buttons (mirrors Textual TUI)
        add_btn = QPushButton("Add")
        add_btn.setMaximumWidth(60)
        add_btn.setStyleSheet(self._get_button_style())
        add_btn.clicked.connect(self.add_function)
        header_layout.addWidget(add_btn)
        
        load_btn = QPushButton("Load")
        load_btn.setMaximumWidth(60)
        load_btn.setStyleSheet(self._get_button_style())
        load_btn.clicked.connect(self.load_function_pattern)
        header_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save As")
        save_btn.setMaximumWidth(80)
        save_btn.setStyleSheet(self._get_button_style())
        save_btn.clicked.connect(self.save_function_pattern)
        header_layout.addWidget(save_btn)
        
        code_btn = QPushButton("Code")
        code_btn.setMaximumWidth(60)
        code_btn.setStyleSheet(self._get_button_style())
        code_btn.clicked.connect(self.edit_function_code)
        header_layout.addWidget(code_btn)

        # Component selection button (mirrors Textual TUI)
        self.component_btn = QPushButton(self._get_component_button_text())
        self.component_btn.setMaximumWidth(120)
        self.component_btn.setStyleSheet(self._get_button_style())
        self.component_btn.clicked.connect(self.show_component_selection_dialog)
        self.component_btn.setEnabled(not self._is_component_button_disabled())
        header_layout.addWidget(self.component_btn)

        # Channel navigation buttons (only in dict mode with multiple channels, mirrors Textual TUI)
        self.prev_channel_btn = QPushButton("<")
        self.prev_channel_btn.setMaximumWidth(30)
        self.prev_channel_btn.setStyleSheet(self._get_button_style())
        self.prev_channel_btn.clicked.connect(lambda: self._navigate_channel(-1))
        header_layout.addWidget(self.prev_channel_btn)

        self.next_channel_btn = QPushButton(">")
        self.next_channel_btn.setMaximumWidth(30)
        self.next_channel_btn.setStyleSheet(self._get_button_style())
        self.next_channel_btn.clicked.connect(lambda: self._navigate_channel(1))
        header_layout.addWidget(self.next_channel_btn)

        # Update navigation button visibility
        self._update_navigation_buttons()

        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Scrollable function list (mirrors Textual TUI)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        
        # Function list container
        self.function_container = QWidget()
        self.function_layout = QVBoxLayout(self.function_container)
        self.function_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.function_layout.setSpacing(8)
        
        # Populate function list
        self._populate_function_list()
        
        self.scroll_area.setWidget(self.function_container)
        layout.addWidget(self.scroll_area)
    
    def _get_button_style(self) -> str:
        """Get consistent button styling."""
        return """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
        """
    
    def _populate_function_list(self):
        """Populate function list with panes (mirrors Textual TUI)."""
        # Clear existing panes
        for pane in self.function_panes:
            pane.setParent(None)
        self.function_panes.clear()
        
        # Clear layout
        while self.function_layout.count():
            child = self.function_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        
        if not self.functions:
            # Show empty state
            empty_label = QLabel("No functions defined. Click 'Add' to begin.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
            self.function_layout.addWidget(empty_label)
        else:
            # Create function panes
            for i, func_item in enumerate(self.functions):
                pane = FunctionPaneWidget(func_item, i, self.service_adapter)
                
                # Connect signals (using actual FunctionPaneWidget signal names)
                pane.move_function.connect(self._move_function)
                pane.add_function.connect(self._add_function_at_index)
                pane.remove_function.connect(self._remove_function)
                pane.parameter_changed.connect(self._on_parameter_changed)
                
                self.function_panes.append(pane)
                self.function_layout.addWidget(pane)
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        pass
    
    def add_function(self):
        """Add a new function (mirrors Textual TUI)."""
        from openhcs.pyqt_gui.dialogs.function_selector_dialog import FunctionSelectorDialog

        # Show function selector dialog (reuses Textual TUI logic)
        selected_function = FunctionSelectorDialog.select_function(parent=self)

        if selected_function:
            # Add function to list (same logic as Textual TUI)
            new_func_item = (selected_function, {})
            self.functions.append(new_func_item)
            self._update_pattern_data()
            self._populate_function_list()
            self.function_pattern_changed.emit()
            logger.debug(f"Added function: {selected_function.__name__}")
    
    def load_function_pattern(self):
        """Load function pattern from file (mirrors Textual TUI)."""
        if self.service_adapter:
            from openhcs.pyqt_gui.utils.path_cache import PathCacheKey
            
            file_path = self.service_adapter.show_cached_file_dialog(
                cache_key=PathCacheKey.FUNCTION_PATTERNS,
                title="Load Function Pattern",
                file_filter="Function Files (*.func);;All Files (*)",
                mode="open"
            )
            
            if file_path:
                self._load_function_pattern_from_file(file_path)
    
    def save_function_pattern(self):
        """Save function pattern to file (mirrors Textual TUI)."""
        if self.service_adapter:
            from openhcs.pyqt_gui.utils.path_cache import PathCacheKey
            
            file_path = self.service_adapter.show_cached_file_dialog(
                cache_key=PathCacheKey.FUNCTION_PATTERNS,
                title="Save Function Pattern",
                file_filter="Function Files (*.func);;All Files (*)",
                mode="save"
            )
            
            if file_path:
                self._save_function_pattern_to_file(file_path)
    
    def edit_function_code(self):
        """Edit function pattern as code (mirrors Textual TUI)."""
        # TODO: Implement code editor
        logger.debug("Edit function code clicked - TODO: implement code editor")
    
    def _move_function(self, index, direction):
        """Move function up or down."""
        if 0 <= index < len(self.functions):
            new_index = index + direction
            if 0 <= new_index < len(self.functions):
                # Swap functions
                self.functions[index], self.functions[new_index] = self.functions[new_index], self.functions[index]
                self._update_pattern_data()
                self._populate_function_list()
                self.function_pattern_changed.emit()
    
    def _add_function_at_index(self, index):
        """Add function at specific index (mirrors Textual TUI)."""
        from openhcs.pyqt_gui.dialogs.function_selector_dialog import FunctionSelectorDialog

        # Show function selector dialog (reuses Textual TUI logic)
        selected_function = FunctionSelectorDialog.select_function(parent=self)

        if selected_function:
            # Insert function at specific index (same logic as Textual TUI)
            new_func_item = (selected_function, {})
            self.functions.insert(index, new_func_item)
            self._update_pattern_data()
            self._populate_function_list()
            self.function_pattern_changed.emit()
            logger.debug(f"Added function at index {index}: {selected_function.__name__}")
    
    def _remove_function(self, index):
        """Remove function at index."""
        if 0 <= index < len(self.functions):
            self.functions.pop(index)
            self._update_pattern_data()
            self._populate_function_list()
            self.function_pattern_changed.emit()
    
    def _on_parameter_changed(self, index, param_name, value):
        """Handle parameter change from function pane."""
        if 0 <= index < len(self.functions):
            func, kwargs = self.functions[index]
            kwargs[param_name] = value
            self.functions[index] = (func, kwargs)
            self._update_pattern_data()
            self.function_pattern_changed.emit()
    

    
    def _load_function_pattern_from_file(self, file_path):
        """Load function pattern from file."""
        # TODO: Implement file loading
        logger.debug(f"Load function pattern from {file_path} - TODO: implement")
    
    def _save_function_pattern_to_file(self, file_path):
        """Save function pattern to file."""
        # TODO: Implement file saving
        logger.debug(f"Save function pattern to {file_path} - TODO: implement")
    
    def get_current_functions(self):
        """Get current function list."""
        return self.functions.copy()

    @property
    def current_pattern(self):
        """Get the current pattern data (for parent widgets to access)."""
        self._update_pattern_data()  # Ensure it's up to date

        # Migration fix: Convert any integer keys to string keys for compatibility
        # with pattern detection system which always uses string component values
        if isinstance(self.pattern_data, dict):
            migrated_pattern = {}
            for key, value in self.pattern_data.items():
                str_key = str(key)
                migrated_pattern[str_key] = value
            return migrated_pattern

        return self.pattern_data
    
    def set_functions(self, functions):
        """Set function list and refresh display."""
        self.functions = functions.copy() if functions else []
        self._update_pattern_data()
        self._populate_function_list()

    def _get_component_button_text(self) -> str:
        """Get text for the component selection button (mirrors Textual TUI)."""
        if self.current_group_by is None or self.current_group_by == GroupBy.NONE:
            return "Component: None"

        # Use group_by.value.title() for dynamic component type display
        component_type = self.current_group_by.value.title()

        if self.is_dict_mode and self.selected_channel is not None:
            # Try to get metadata name for the selected component
            display_name = self._get_component_display_name(self.selected_channel)
            return f"{component_type}: {display_name}"
        return f"{component_type}: None"

    def _get_component_display_name(self, component_key: str) -> str:
        """Get display name for component key, using metadata if available (mirrors Textual TUI)."""
        orchestrator = self._get_current_orchestrator()
        if orchestrator and self.current_group_by:
            metadata_name = orchestrator.get_component_metadata(self.current_group_by, component_key)
            if metadata_name:
                return metadata_name
        return component_key

    def _is_component_button_disabled(self) -> bool:
        """Check if component selection button should be disabled (mirrors Textual TUI)."""
        return (
            self.current_group_by is None or
            self.current_group_by == GroupBy.NONE or
            (self.current_variable_components and
             self.current_group_by.value in [vc.value for vc in self.current_variable_components])
        )

    def show_component_selection_dialog(self):
        """Show the component selection dialog (mirrors Textual TUI)."""
        from openhcs.pyqt_gui.dialogs.group_by_selector_dialog import GroupBySelectorDialog

        # Check if component selection is disabled
        if self._is_component_button_disabled():
            logger.debug("Component selection is disabled")
            return

        # Get available components from orchestrator using current group_by - MUST exist, no fallbacks
        orchestrator = self._get_current_orchestrator()

        available_components = orchestrator.get_component_keys(self.current_group_by)
        assert available_components, f"No {self.current_group_by.value} values found in current plate"

        # Get current selection from pattern data (mirrors Textual TUI logic)
        selected_components = self._get_current_component_selection()

        # Show group by selector dialog (reuses Textual TUI logic)
        result = GroupBySelectorDialog.select_components(
            available_components=available_components,
            selected_components=selected_components,
            component_type=self.current_group_by.value,
            orchestrator=orchestrator,
            parent=self
        )

        if result is not None:
            self._handle_component_selection(result)

    def _get_current_orchestrator(self):
        """Get current orchestrator instance - MUST exist, no fallbacks allowed."""
        # Use stored main window reference to get plate manager
        main_window = self.main_window
        plate_manager_window = main_window.floating_windows['plate_manager']

        # Find the actual plate manager widget
        plate_manager_widget = None
        for child in plate_manager_window.findChildren(QWidget):
            if hasattr(child, 'orchestrators') and hasattr(child, 'selected_plate_path'):
                plate_manager_widget = child
                break

        # Get current plate from plate manager's selection
        current_plate = plate_manager_widget.selected_plate_path
        orchestrator = plate_manager_widget.orchestrators[current_plate]

        # Orchestrator must be initialized
        assert orchestrator.is_initialized(), f"Orchestrator for plate {current_plate} is not initialized"

        return orchestrator

    def _get_current_component_selection(self):
        """Get current component selection from pattern data (mirrors Textual TUI logic)."""
        # If in dict mode, return the keys of the dict as the current selection (sorted)
        if self.is_dict_mode and isinstance(self.pattern_data, dict):
            return sorted(list(self.pattern_data.keys()))

        # If not in dict mode, check the cache (sorted)
        cached_selection = self.component_selections.get(self.current_group_by, [])
        return sorted(cached_selection)

    def _handle_component_selection(self, new_components):
        """Handle component selection result (mirrors Textual TUI)."""
        # Save selection to cache for current group_by
        if self.current_group_by is not None and self.current_group_by != GroupBy.NONE:
            self.component_selections[self.current_group_by] = new_components
            logger.debug(f"Step '{self.step_identifier}': Cached selection for {self.current_group_by.value}: {new_components}")

        # Update pattern structure based on component selection (mirrors Textual TUI)
        self._update_components(new_components)

        # Update component button text and navigation
        self._refresh_component_button()
        logger.debug(f"Updated components: {new_components}")

    def _update_components(self, new_components):
        """Update function pattern structure based on component selection (mirrors Textual TUI)."""
        # Sort new components for consistent ordering
        if new_components:
            new_components = sorted(new_components)

        if not new_components:
            # No components selected - revert to list mode
            if self.is_dict_mode:
                # Save current functions to list mode
                self.pattern_data = self.functions
                self.is_dict_mode = False
                self.selected_channel = None
                logger.debug("Reverted to list mode (no components selected)")
        else:
            # Use component strings directly - no conversion needed
            component_keys = new_components

            # Components selected - ensure dict mode
            if not self.is_dict_mode:
                # Convert to dict mode
                current_functions = self.functions
                self.pattern_data = {component_keys[0]: current_functions}
                self.is_dict_mode = True
                self.selected_channel = component_keys[0]

                # Add other components with empty functions
                for component_key in component_keys[1:]:
                    self.pattern_data[component_key] = []
            else:
                # Already in dict mode - update components
                old_pattern = self.pattern_data.copy() if isinstance(self.pattern_data, dict) else {}

                # Create a persistent storage for deselected components (mirrors Textual TUI)
                if not hasattr(self, '_deselected_components_storage'):
                    self._deselected_components_storage = {}

                # Save currently deselected components to storage
                for old_key, old_functions in old_pattern.items():
                    if old_key not in component_keys:
                        self._deselected_components_storage[old_key] = old_functions
                        logger.debug(f"Saved {len(old_functions)} functions for deselected component {old_key}")

                new_pattern = {}

                # Restore functions for components (from current pattern or storage)
                for component_key in component_keys:
                    if component_key in old_pattern:
                        # Component was already selected - keep its functions
                        new_pattern[component_key] = old_pattern[component_key]
                    elif component_key in self._deselected_components_storage:
                        # Component was previously deselected - restore its functions
                        new_pattern[component_key] = self._deselected_components_storage[component_key]
                        logger.debug(f"Restored {len(new_pattern[component_key])} functions for reselected component {component_key}")
                    else:
                        # New component - start with empty functions
                        new_pattern[component_key] = []

                self.pattern_data = new_pattern

                # Update selected channel if current one is no longer available
                if self.selected_channel not in component_keys:
                    self.selected_channel = component_keys[0]
                    self.functions = new_pattern[self.selected_channel]

        # Update UI to reflect changes
        self._populate_function_list()
        self._update_navigation_buttons()

    def _refresh_component_button(self):
        """Refresh the component button text and state (mirrors Textual TUI)."""
        if hasattr(self, 'component_btn'):
            self.component_btn.setText(self._get_component_button_text())
            self.component_btn.setEnabled(not self._is_component_button_disabled())

        # Also update navigation buttons when component button is refreshed
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        """Update visibility of channel navigation buttons (mirrors Textual TUI)."""
        if hasattr(self, 'prev_channel_btn') and hasattr(self, 'next_channel_btn'):
            # Show navigation buttons only in dict mode with multiple channels
            show_nav = (self.is_dict_mode and
                       isinstance(self.pattern_data, dict) and
                       len(self.pattern_data) > 1)

            self.prev_channel_btn.setVisible(show_nav)
            self.next_channel_btn.setVisible(show_nav)

    def _navigate_channel(self, direction: int):
        """Navigate to next/previous channel (with looping, mirrors Textual TUI)."""
        if not self.is_dict_mode or not isinstance(self.pattern_data, dict):
            return

        channels = sorted(self.pattern_data.keys())
        if len(channels) <= 1:
            return

        try:
            current_index = channels.index(self.selected_channel)
            new_index = (current_index + direction) % len(channels)
            new_channel = channels[new_index]

            self._switch_to_channel(new_channel)
            logger.debug(f"Navigated to channel {new_channel}")
        except (ValueError, IndexError):
            logger.warning(f"Failed to navigate channels: current={self.selected_channel}, channels={channels}")

    def _switch_to_channel(self, channel: str):
        """Switch to editing functions for a specific channel (mirrors Textual TUI)."""
        if not self.is_dict_mode:
            return

        # Save current functions first
        old_channel = self.selected_channel
        logger.debug(f"Switching from channel {old_channel} to {channel}")

        self._update_pattern_data()

        # Switch to new channel
        self.selected_channel = channel
        if isinstance(self.pattern_data, dict):
            self.functions = self.pattern_data.get(channel, [])
            logger.debug(f"Loaded {len(self.functions)} functions for channel {channel}")
        else:
            self.functions = []

        # Update UI
        self._refresh_component_button()
        self._populate_function_list()

    def _update_pattern_data(self):
        """Update pattern_data based on current functions and mode (mirrors Textual TUI)."""
        if self.is_dict_mode and self.selected_channel is not None:
            # Save current functions to the selected channel
            if not isinstance(self.pattern_data, dict):
                self.pattern_data = {}
            logger.debug(f"Saving {len(self.functions)} functions to channel {self.selected_channel}")
            self.pattern_data[self.selected_channel] = self.functions.copy()
        else:
            # List mode - pattern_data is just the functions list
            self.pattern_data = self.functions
