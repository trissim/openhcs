"""
Function Pane Widget for PyQt6

Individual function display with parameter editing capabilities.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from typing import Any, Dict, Callable, Optional, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFrame, QScrollArea, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from openhcs.textual_tui.services.pattern_data_manager import PatternDataManager
from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager as TextualParameterFormManager
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer

# Import PyQt6 help components (using same pattern as Textual TUI)
from openhcs.pyqt_gui.widgets.shared.clickable_help_components import HelpIndicator

logger = logging.getLogger(__name__)


class FunctionPaneWidget(QWidget):
    """
    PyQt6 Function Pane Widget.
    
    Displays individual function with editable parameters and control buttons.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    parameter_changed = pyqtSignal(int, str, object)  # index, param_name, value
    function_changed = pyqtSignal(int)  # index
    add_function = pyqtSignal(int)  # index
    remove_function = pyqtSignal(int)  # index
    move_function = pyqtSignal(int, int)  # index, direction
    reset_parameters = pyqtSignal(int)  # index
    
    def __init__(self, func_item: Tuple[Callable, Dict], index: int, service_adapter, parent=None):
        """
        Initialize the function pane widget.
        
        Args:
            func_item: Tuple of (function, kwargs)
            index: Function index in the list
            service_adapter: PyQt service adapter for dialogs and operations
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Core dependencies
        self.service_adapter = service_adapter
        
        # Business logic state (extracted from Textual version)
        self.func, self.kwargs = func_item
        self.index = index
        self.show_parameters = True
        
        # Parameter management (extracted from Textual version)
        if self.func:
            param_info = SignatureAnalyzer.analyze(self.func)
            parameters = {name: self.kwargs.get(name, info.default_value) for name, info in param_info.items()}
            parameter_types = {name: info.param_type for name, info in param_info.items()}
            
            self.form_manager = TextualParameterFormManager(parameters, parameter_types, f"func_{index}", param_info)
            self.param_defaults = {name: info.default_value for name, info in param_info.items()}
        else:
            self.form_manager = None
            self.param_defaults = {}
        
        # Internal kwargs tracking (extracted from Textual version)
        self._internal_kwargs = self.kwargs.copy()
        
        # UI components
        self.parameter_widgets: Dict[str, QWidget] = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        logger.debug(f"Function pane widget initialized for index {index}")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Function header
        header_frame = self.create_function_header()
        layout.addWidget(header_frame)
        
        # Control buttons
        button_frame = self.create_button_panel()
        layout.addWidget(button_frame)
        
        # Parameter form (if function exists and parameters shown)
        if self.func and self.show_parameters and self.form_manager:
            parameter_frame = self.create_parameter_form()
            layout.addWidget(parameter_frame)
        
        # Set styling
        self.setStyleSheet("""
            FunctionPaneWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 5px;
                margin: 2px;
            }
        """)
    
    def create_function_header(self) -> QWidget:
        """
        Create the function header with name and info.
        
        Returns:
            Widget containing function header
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        # Function name with help functionality (reuses Textual TUI help logic)
        if self.func:
            func_name = self.func.__name__
            func_module = self.func.__module__

            # Function name with help
            name_label = QLabel(f"🔧 {func_name}")
            name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            name_label.setStyleSheet("color: #00aaff;")
            layout.addWidget(name_label)

            # Help indicator for function (import locally to avoid circular imports)
            from openhcs.pyqt_gui.widgets.shared.clickable_help_components import HelpIndicator
            help_indicator = HelpIndicator(help_target=self.func)
            layout.addWidget(help_indicator)

            # Module info
            if func_module:
                module_label = QLabel(f"({func_module})")
                module_label.setFont(QFont("Arial", 8))
                module_label.setStyleSheet("color: #888888;")
                layout.addWidget(module_label)
        else:
            name_label = QLabel("No Function Selected")
            name_label.setStyleSheet("color: #ff6666;")
            layout.addWidget(name_label)

        layout.addStretch()
        
        return frame
    
    def create_button_panel(self) -> QWidget:
        """
        Create the control button panel.
        
        Returns:
            Widget containing control buttons
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.addStretch()  # Center the buttons
        
        # Button configurations (extracted from Textual version)
        button_configs = [
            ("↑", "move_up", "Move function up"),
            ("↓", "move_down", "Move function down"),
            ("Add", "add_func", "Add new function"),
            ("Delete", "remove_func", "Delete this function"),
            ("Reset", "reset_all", "Reset all parameters"),
        ]
        
        for name, action, tooltip in button_configs:
            button = QPushButton(name)
            button.setToolTip(tooltip)
            button.setMaximumWidth(60)
            button.setMaximumHeight(25)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #666666;
                    border-radius: 2px;
                    padding: 2px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:pressed {
                    background-color: #303030;
                }
            """)
            
            # Connect button to action
            button.clicked.connect(lambda checked, a=action: self.handle_button_action(a))
            
            layout.addWidget(button)
        
        layout.addStretch()  # Center the buttons
        
        return frame
    
    def create_parameter_form(self) -> QWidget:
        """
        Create the parameter form using extracted business logic.
        
        Returns:
            Widget containing parameter form
        """
        group_box = QGroupBox("Parameters")
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 3px;
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
        
        layout = QVBoxLayout(group_box)

        # Use the enhanced ParameterFormManager that has help and reset functionality
        if self.form_manager:
            # Import the enhanced PyQt6 ParameterFormManager
            from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager as PyQtParameterFormManager

            # Create enhanced parameter form manager with help and reset buttons
            enhanced_form_manager = PyQtParameterFormManager(
                parameters=self.form_manager.parameters,
                parameter_types=self.form_manager.parameter_types,
                field_id=f"func_{self.index}",
                parameter_info=self.form_manager.parameter_info,
                use_scroll_area=False,  # Don't use scroll area in function panes
                function_target=self.func  # Pass function for docstring fallback
            )

            # Connect parameter changes
            enhanced_form_manager.parameter_changed.connect(
                lambda param_name, value: self.handle_parameter_change(param_name, value)
            )

            layout.addWidget(enhanced_form_manager)

            # Store reference for parameter updates
            self.enhanced_form_manager = enhanced_form_manager
        
        return group_box
    
    def create_parameter_widget(self, param_name: str, param_type: type, current_value: Any) -> Optional[QWidget]:
        """
        Create parameter widget based on type (simplified TypedWidgetFactory).
        
        Args:
            param_name: Parameter name
            param_type: Parameter type
            current_value: Current parameter value
            
        Returns:
            Widget for parameter editing or None
        """
        from PyQt6.QtWidgets import QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox
        from PyQt6.QtGui import QWheelEvent

        # No-scroll widget classes to prevent accidental value changes
        class NoScrollSpinBox(QSpinBox):
            def wheelEvent(self, event: QWheelEvent):
                event.ignore()

        class NoScrollDoubleSpinBox(QDoubleSpinBox):
            def wheelEvent(self, event: QWheelEvent):
                event.ignore()

        class NoScrollComboBox(QComboBox):
            def wheelEvent(self, event: QWheelEvent):
                event.ignore()
        
        # Boolean parameters
        if param_type == bool:
                widget = QCheckBox()
                widget.setChecked(bool(current_value))
                widget.toggled.connect(lambda checked: self.handle_parameter_change(param_name, checked))
                return widget
            
            # Integer parameters
            elif param_type == int:
                widget = NoScrollSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(int(current_value) if current_value is not None else 0)
                widget.valueChanged.connect(lambda value: self.handle_parameter_change(param_name, value))
                return widget

            # Float parameters
            elif param_type == float:
                widget = NoScrollDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
                widget.setValue(float(current_value) if current_value is not None else 0.0)
                widget.valueChanged.connect(lambda value: self.handle_parameter_change(param_name, value))
                return widget

            # Enum parameters
            elif any(base.__name__ == 'Enum' for base in param_type.__bases__):
                widget = NoScrollComboBox()
                for enum_value in param_type:
                    widget.addItem(str(enum_value.value), enum_value)
                
                # Set current value
                if current_value is not None:
                    for i in range(widget.count()):
                        if widget.itemData(i) == current_value:
                            widget.setCurrentIndex(i)
                            break
                
                widget.currentIndexChanged.connect(
                    lambda index: self.handle_parameter_change(param_name, widget.itemData(index))
                )
                return widget
            
            # String and other parameters
            else:
                widget = QLineEdit()
                widget.setText(str(current_value) if current_value is not None else "")
                widget.textChanged.connect(lambda text: self.handle_parameter_change(param_name, text))
                return widget
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        pass  # Connections are set up in widget creation
    
    def handle_button_action(self, action: str):
        """
        Handle button actions (extracted from Textual version).
        
        Args:
            action: Action identifier
        """
        if action == "move_up":
            self.move_function.emit(self.index, -1)
        elif action == "move_down":
            self.move_function.emit(self.index, 1)
        elif action == "add_func":
            self.add_function.emit(self.index + 1)
        elif action == "remove_func":
            self.remove_function.emit(self.index)
        elif action == "reset_all":
            self.reset_all_parameters()
    
    def handle_parameter_change(self, param_name: str, value: Any):
        """
        Handle parameter value changes (extracted from Textual version).
        
        Args:
            param_name: Name of the parameter
            value: New parameter value
        """
        # Update internal kwargs without triggering reactive update
        self._internal_kwargs[param_name] = value
        
        # Update form manager
        if self.form_manager:
            self.form_manager.update_parameter(param_name, value)
            final_value = self.form_manager.parameters[param_name]
        else:
            final_value = value
        
        # Emit parameter changed signal
        self.parameter_changed.emit(self.index, param_name, final_value)
        
        logger.debug(f"Parameter changed: {param_name} = {final_value}")
    
    def reset_all_parameters(self):
        """Reset all parameters to default values (extracted from Textual version)."""
        
        for param_name, default_value in self.param_defaults.items():
            # Update internal kwargs
            self._internal_kwargs[param_name] = default_value
            
            # Update form manager
            if self.form_manager:
                self.form_manager.reset_parameter(param_name, default_value)
            
            # Update UI widget
            if param_name in self.parameter_widgets:
                widget = self.parameter_widgets[param_name]
                self.update_widget_value(widget, default_value)
            
            # Emit parameter changed signal
            self.parameter_changed.emit(self.index, param_name, default_value)
        
        self.reset_parameters.emit(self.index)
        logger.debug(f"Reset all parameters for function {self.index}")
    
    def update_widget_value(self, widget: QWidget, value: Any):
        """
        Update widget value without triggering signals.
        
        Args:
            widget: Widget to update
            value: New value
        """
        from PyQt6.QtWidgets import QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox
        # Import the no-scroll classes from the same module scope
        from openhcs.pyqt_gui.shared.typed_widget_factory import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox
        
        # Temporarily block signals to avoid recursion
        widget.blockSignals(True)
        
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, NoScrollSpinBox)):
                widget.setValue(int(value) if value is not None else 0)
            elif isinstance(widget, (QDoubleSpinBox, NoScrollDoubleSpinBox)):
                widget.setValue(float(value) if value is not None else 0.0)
            elif isinstance(widget, (QComboBox, NoScrollComboBox)):
                for i in range(widget.count()):
                    if widget.itemData(i) == value:
                        widget.setCurrentIndex(i)
                        break
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
        finally:
            widget.blockSignals(False)
    
    def get_current_kwargs(self) -> Dict[str, Any]:
        """
        Get current kwargs values (extracted from Textual version).
        
        Returns:
            Current parameter values
        """
        return self._internal_kwargs.copy()
    
    def sync_kwargs(self):
        """Sync internal kwargs to main kwargs (extracted from Textual version)."""
        self.kwargs = self._internal_kwargs.copy()
    
    def update_function(self, func_item: Tuple[Callable, Dict]):
        """
        Update the function and parameters.
        
        Args:
            func_item: New function item tuple
        """
        self.func, self.kwargs = func_item
        self._internal_kwargs = self.kwargs.copy()
        
        # Recreate form manager
        if self.func:
            param_info = SignatureAnalyzer.analyze(self.func)
            parameters = {name: self.kwargs.get(name, info.default_value) for name, info in param_info.items()}
            parameter_types = {name: info.param_type for name, info in param_info.items()}
            
            self.form_manager = TextualParameterFormManager(parameters, parameter_types, f"func_{self.index}", param_info)
            self.param_defaults = {name: info.default_value for name, info in param_info.items()}
        else:
            self.form_manager = None
            self.param_defaults = {}
        
        # Rebuild UI
        self.setup_ui()
        
        logger.debug(f"Updated function for index {self.index}")


class FunctionListWidget(QWidget):
    """
    PyQt6 Function List Widget.
    
    Container for multiple FunctionPaneWidgets with list management.
    """
    
    # Signals
    functions_changed = pyqtSignal(list)  # List of function items
    
    def __init__(self, service_adapter, parent=None):
        """
        Initialize the function list widget.
        
        Args:
            service_adapter: PyQt service adapter
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.service_adapter = service_adapter
        self.functions: List[Tuple[Callable, Dict]] = []
        self.function_panes: List[FunctionPaneWidget] = []
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Scroll area for function panes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container widget for function panes
        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setSpacing(5)
        
        scroll_area.setWidget(self.container_widget)
        layout.addWidget(scroll_area)
        
        # Add function button
        add_button = QPushButton("Add Function")
        add_button.clicked.connect(lambda: self.add_function_at_index(len(self.functions)))
        layout.addWidget(add_button)
    
    def update_function_list(self):
        """Update the function list display."""
        # Clear existing panes
        for pane in self.function_panes:
            pane.setParent(None)
        self.function_panes.clear()
        
        # Create new panes
        for i, func_item in enumerate(self.functions):
            pane = FunctionPaneWidget(func_item, i, self.service_adapter)
            
            # Connect signals
            pane.parameter_changed.connect(self.on_parameter_changed)
            pane.add_function.connect(self.add_function_at_index)
            pane.remove_function.connect(self.remove_function_at_index)
            pane.move_function.connect(self.move_function)
            
            self.function_panes.append(pane)
            self.container_layout.addWidget(pane)
        
        self.container_layout.addStretch()
    
    def add_function_at_index(self, index: int):
        """Add function at specific index."""
        # Placeholder function
        new_func_item = (lambda x: x, {})
        self.functions.insert(index, new_func_item)
        self.update_function_list()
        self.functions_changed.emit(self.functions)
    
    def remove_function_at_index(self, index: int):
        """Remove function at specific index."""
        if 0 <= index < len(self.functions):
            self.functions.pop(index)
            self.update_function_list()
            self.functions_changed.emit(self.functions)
    
    def move_function(self, index: int, direction: int):
        """Move function up or down."""
        new_index = index + direction
        if 0 <= new_index < len(self.functions):
            self.functions[index], self.functions[new_index] = self.functions[new_index], self.functions[index]
            self.update_function_list()
            self.functions_changed.emit(self.functions)
    
    def on_parameter_changed(self, index: int, param_name: str, value: Any):
        """Handle parameter changes."""
        if 0 <= index < len(self.functions):
            func, kwargs = self.functions[index]
            new_kwargs = kwargs.copy()
            new_kwargs[param_name] = value
            self.functions[index] = (func, new_kwargs)
            self.functions_changed.emit(self.functions)
