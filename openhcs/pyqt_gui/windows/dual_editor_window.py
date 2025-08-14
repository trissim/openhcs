"""
Dual Editor Window for PyQt6

Step and function editing dialog with tabbed interface.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from typing import Optional, Callable, Any, Dict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QWidget, QFrame, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager

from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
logger = logging.getLogger(__name__)


class DualEditorWindow(QDialog):
    """
    PyQt6 Dual Editor Window.
    
    Step and function editing dialog with tabbed interface for comprehensive editing.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    step_saved = pyqtSignal(object)  # FunctionStep
    step_cancelled = pyqtSignal()
    changes_detected = pyqtSignal(bool)  # has_changes
    
    def __init__(self, step_data: Optional[FunctionStep] = None, is_new: bool = False,
                 on_save_callback: Optional[Callable] = None, color_scheme: Optional[PyQt6ColorScheme] = None, parent=None):
        """
        Initialize the dual editor window.
        
        Args:
            step_data: FunctionStep to edit (None for new step)
            is_new: Whether this is a new step
            on_save_callback: Function to call when step is saved
            parent: Parent widget
        """
        super().__init__(parent)

        # Make window non-modal (like plate manager and pipeline editor)
        self.setModal(False)

        # Initialize color scheme
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        
        # Business logic state (extracted from Textual version)
        self.is_new = is_new
        self.on_save_callback = on_save_callback
        
        # Pattern management (extracted from Textual version)
        self.pattern_manager = PatternDataManager()
        
        if step_data:
            self.editing_step = step_data
        else:
            self.editing_step = self.pattern_manager.create_new_step()
        
        # Store original for change detection
        self.original_step = self.pattern_manager.clone_pattern(self.editing_step)
        
        # Change tracking
        self.has_changes = False
        self.current_tab = "step"
        
        # UI components
        self.tab_widget: Optional[QTabWidget] = None
        self.step_editor: Optional[QWidget] = None
        self.func_editor: Optional[QWidget] = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        logger.debug(f"Dual editor window initialized (new={is_new})")
    
    def setup_ui(self):
        """Setup the user interface."""
        title = "New Step" if self.is_new else f"Edit Step: {getattr(self.editing_step, 'name', 'Unknown')}"
        self.setWindowTitle(title)
        # Keep non-modal (already set in __init__)
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header_label = QLabel(title)
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; padding: 10px;")
        layout.addWidget(header_label)
        
        # Tabbed content
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
            }}
            QTabBar::tab {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
            QTabBar::tab:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }}
        """)
        
        # Create tabs
        self.create_step_tab()
        self.create_function_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Button panel
        button_panel = self.create_button_panel()
        layout.addWidget(button_panel)
        
        # Set styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.window_bg)};
                color: white;
            }}
        """)
    
    def create_step_tab(self):
        """Create the step settings tab (using dedicated widget)."""
        from openhcs.pyqt_gui.widgets.step_parameter_editor import StepParameterEditorWidget

        # Create step parameter editor widget (mirrors Textual TUI)
        self.step_editor = StepParameterEditorWidget(self.editing_step, service_adapter=None, color_scheme=self.color_scheme)

        # Connect parameter changes
        self.step_editor.step_parameter_changed.connect(self.detect_changes)

        self.tab_widget.addTab(self.step_editor, "Step Settings")

    def create_function_tab(self):
        """Create the function pattern tab (using dedicated widget)."""
        from openhcs.pyqt_gui.widgets.function_list_editor import FunctionListEditorWidget

        # Convert step func to function list format
        initial_functions = self._convert_step_func_to_list()

        # Create function list editor widget (mirrors Textual TUI)
        step_id = getattr(self.editing_step, 'name', 'unknown_step')
        self.func_editor = FunctionListEditorWidget(
            initial_functions=initial_functions,
            step_identifier=step_id,
            service_adapter=None
        )

        # Store main window reference for orchestrator access (find it through parent chain)
        main_window = self._find_main_window()
        if main_window:
            self.func_editor.main_window = main_window

        # Initialize step configuration settings in function editor (mirrors Textual TUI)
        self.func_editor.current_group_by = self.editing_step.group_by
        self.func_editor.current_variable_components = self.editing_step.variable_components or []

        # Refresh component button to show correct text and state (mirrors Textual TUI reactive updates)
        self.func_editor._refresh_component_button()

        # Connect function pattern changes
        self.func_editor.function_pattern_changed.connect(self._on_function_pattern_changed)

        self.tab_widget.addTab(self.func_editor, "Function Pattern")

    def _on_function_pattern_changed(self):
        """Handle function pattern changes from function editor."""
        # Update step func from function editor - use current_pattern to get full pattern data
        current_pattern = self.func_editor.current_pattern
        self.editing_step.func = current_pattern
        self.detect_changes()
        logger.debug(f"Function pattern changed: {current_pattern}")



    def create_button_panel(self) -> QWidget:
        """
        Create the button panel with save/cancel actions.

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

        # Changes indicator
        self.changes_label = QLabel("")
        self.changes_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.status_warning)}; font-style: italic;")
        layout.addWidget(self.changes_label)

        layout.addStretch()

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.cancel_edit)
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

        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumWidth(80)
        self.save_button.setEnabled(False)  # Initially disabled
        self.save_button.clicked.connect(self.save_edit)
        self.save_button.setStyleSheet(f"""
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
            QPushButton:disabled {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.separator_color)};
            }}
        """)
        layout.addWidget(self.save_button)

        return panel

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Tab change tracking
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Change detection
        self.changes_detected.connect(self.on_changes_detected)
        func_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(func_label)

        header_layout.addStretch()

        # Function management buttons (mirrors Textual TUI)
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

        save_as_btn = QPushButton("Save As")
        save_as_btn.setMaximumWidth(80)
        save_as_btn.setStyleSheet(self._get_button_style())
        save_as_btn.clicked.connect(self.save_function_pattern)
        header_layout.addWidget(save_as_btn)

        code_btn = QPushButton("Code")
        code_btn.setMaximumWidth(60)
        code_btn.setStyleSheet(self._get_button_style())
        code_btn.clicked.connect(self.edit_function_code)
        header_layout.addWidget(code_btn)

        layout.addLayout(header_layout)

        # Function list scroll area (mirrors Textual TUI)
        self.function_scroll = QScrollArea()
        self.function_scroll.setWidgetResizable(True)
        self.function_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.function_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 4px;
            }}
        """)

        # Function list container
        self.function_container = QWidget()
        self.function_layout = QVBoxLayout(self.function_container)
        self.function_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.function_layout.setSpacing(8)

        # Initialize function list from step
        self.function_panes = []
        self._populate_function_list()

        self.function_scroll.setWidget(self.function_container)
        layout.addWidget(self.function_scroll)

        return frame

    def _get_button_style(self) -> str:
        """Get consistent button styling."""
        return """
            QPushButton {
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }
            QPushButton:pressed {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_pressed_bg)};
            }
        """

    def _populate_function_list(self):
        """Populate function list from current step (mirrors Textual TUI)."""
        # Clear existing panes
        for pane in self.function_panes:
            pane.setParent(None)
        self.function_panes.clear()

        # Convert step func to function list
        functions = self._convert_step_func_to_list()

        if not functions:
            # Show empty state
            empty_label = QLabel("No functions defined. Click 'Add' to begin.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_disabled)}; font-style: italic; padding: 20px;")
            self.function_layout.addWidget(empty_label)
        else:
            # Create function panes
            for i, func_item in enumerate(functions):
                pane = self._create_function_pane(func_item, i)
                self.function_panes.append(pane)
                self.function_layout.addWidget(pane)

    def _convert_step_func_to_list(self):
        """Convert step func to initial pattern format for function list editor."""
        if not hasattr(self.editing_step, 'func') or not self.editing_step.func:
            return []

        # Return the step func directly - the function list editor will handle the conversion
        return self.editing_step.func



    def _find_main_window(self):
        """Find the main window through the parent chain."""
        try:
            # Navigate up the parent chain to find OpenHCSMainWindow
            current = self.parent()
            while current:
                # Check if this is the main window (has floating_windows attribute)
                if hasattr(current, 'floating_windows') and hasattr(current, 'service_adapter'):
                    logger.debug(f"Found main window: {type(current).__name__}")
                    return current
                current = current.parent()

            logger.warning("Could not find main window in parent chain")
            return None

        except Exception as e:
            logger.error(f"Error finding main window: {e}")
            return None

    def _get_current_plate_from_pipeline_editor(self):
        """Get current plate from pipeline editor (mirrors Textual TUI pattern)."""
        try:
            # Navigate up to find pipeline editor widget
            current = self.parent()
            while current:
                # Check if this is a pipeline editor widget
                if hasattr(current, 'current_plate') and hasattr(current, 'pipeline_steps'):
                    current_plate = getattr(current, 'current_plate', None)
                    if current_plate:
                        logger.debug(f"Found current plate from pipeline editor: {current_plate}")
                        return current_plate

                # Check children for pipeline editor widget
                for child in current.findChildren(QWidget):
                    if hasattr(child, 'current_plate') and hasattr(child, 'pipeline_steps'):
                        current_plate = getattr(child, 'current_plate', None)
                        if current_plate:
                            logger.debug(f"Found current plate from pipeline editor child: {current_plate}")
                            return current_plate

                current = current.parent()

            logger.warning("Could not find current plate from pipeline editor")
            return None

        except Exception as e:
            logger.error(f"Error getting current plate from pipeline editor: {e}")
            return None

    # Old function pane methods removed - now using dedicated FunctionListEditorWidget

    def create_button_panel(self) -> QWidget:
        """
        Create the button panel with save/cancel actions.
        
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
        
        # Changes indicator
        self.changes_label = QLabel("")
        self.changes_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.status_warning)}; font-style: italic;")
        layout.addWidget(self.changes_label)
        
        layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.cancel_edit)
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
        
        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumWidth(80)
        self.save_button.clicked.connect(self.save_step)
        self.save_button.setStyleSheet(f"""
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
        layout.addWidget(self.save_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.changes_detected.connect(self.on_changes_detected)
    
    def get_function_info(self) -> str:
        """
        Get function information for display.
        
        Returns:
            Function information string
        """
        if not self.editing_step or not hasattr(self.editing_step, 'func'):
            return "No function assigned"
        
        func = self.editing_step.func
        func_name = getattr(func, '__name__', 'Unknown Function')
        func_module = getattr(func, '__module__', 'Unknown Module')
        
        info = f"Function: {func_name}\n"
        info += f"Module: {func_module}\n"
        
        # Add parameter info if available
        if hasattr(self.editing_step, 'parameters'):
            params = self.editing_step.parameters
            if params:
                info += f"\nParameters ({len(params)}):\n"
                for param_name, param_value in params.items():
                    info += f"  {param_name}: {param_value}\n"
        
        return info
    
    def on_step_parameter_changed(self, param_name: str, value):
        """Handle step parameter changes from form manager."""
        try:
            # Update the editing step
            setattr(self.editing_step, param_name, value)
            self.detect_changes()
            logger.debug(f"Step parameter changed: {param_name} = {value}")
        except Exception as e:
            logger.error(f"Failed to update step parameter {param_name}: {e}")
    
    def on_tab_changed(self, index: int):
        """Handle tab changes."""
        tab_names = ["step", "function"]
        if 0 <= index < len(tab_names):
            self.current_tab = tab_names[index]
            logger.debug(f"Tab changed to: {self.current_tab}")
    
    def detect_changes(self):
        """Detect if changes have been made."""
        has_changes = False

        # Check step parameters
        for attr in ['name', 'variable_components', 'group_by', 'force_disk_output', 'input_dir', 'output_dir']:
            original_value = getattr(self.original_step, attr, None)
            current_value = getattr(self.editing_step, attr, None)
            if original_value != current_value:
                has_changes = True
                break

        # Check function pattern
        if not has_changes:
            original_func = getattr(self.original_step, 'func', None)
            current_func = getattr(self.editing_step, 'func', None)
            # Simple comparison - could be enhanced for deep comparison
            has_changes = str(original_func) != str(current_func)

        if has_changes != self.has_changes:
            self.has_changes = has_changes
            self.changes_detected.emit(has_changes)
    
    def on_changes_detected(self, has_changes: bool):
        """Handle changes detection."""
        if has_changes:
            self.changes_label.setText("● Unsaved changes")
            self.save_button.setEnabled(True)
        else:
            self.changes_label.setText("")
            self.save_button.setEnabled(False)
    
    def save_step(self):
        """Save the edited step."""
        try:
            # Validate step
            step_name = getattr(self.editing_step, 'name', None)
            if not step_name or not step_name.strip():
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Validation Error", "Step name cannot be empty.")
                return
            
            # Emit signals and call callback
            self.step_saved.emit(self.editing_step)
            
            if self.on_save_callback:
                self.on_save_callback(self.editing_step)
            
            self.accept()
            logger.debug(f"Step saved: {getattr(self.editing_step, 'name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to save step: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", f"Failed to save step:\n{e}")
    
    def cancel_edit(self):
        """Cancel editing and close dialog."""
        if self.has_changes:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to cancel?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.step_cancelled.emit()
        self.reject()
        logger.debug("Step editing cancelled")



    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.has_changes:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        event.accept()
