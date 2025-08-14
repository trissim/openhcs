"""
Typed Widget Factory for PyQt6

Factory for creating typed parameter widgets in PyQt6.
Adapted from Textual TUI version with PyQt6 widget types.
"""

import logging
from typing import Any, Optional, Type, get_origin, get_args, Union, List
from enum import Enum
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTextEdit, QSlider, QDateEdit, QTimeEdit
)
from PyQt6.QtCore import Qt, QDate, QTime
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QWheelEvent

from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.textual_tui.widgets.shared.signature_analyzer import ParameterInfo

logger = logging.getLogger(__name__)


class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores wheel events to prevent accidental value changes."""

    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores wheel events to prevent accidental value changes."""

    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores wheel events to prevent accidental value changes."""

    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class TypedWidgetFactory:
    """
    Factory for creating typed parameter widgets in PyQt6.
    
    Creates appropriate PyQt6 widgets based on parameter types,
    similar to the Textual TUI TypedWidgetFactory.
    """
    
    def __init__(self, color_scheme: Optional[PyQt6ColorScheme] = None):
        """
        Initialize the widget factory.

        Args:
            color_scheme: Color scheme for styling (optional, uses default if None)
        """
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        self.widget_creators = {
            bool: self._create_bool_widget,
            int: self._create_int_widget,
            float: self._create_float_widget,
            str: self._create_str_widget,
            Path: self._create_path_widget,
            list: self._create_list_widget,
            tuple: self._create_tuple_widget,
        }
        
        logger.debug("Typed widget factory initialized")

    def create_enhanced_widget(self, param_name: str, param_type: Type, current_value: Any,
                             param_info: Optional[ParameterInfo] = None) -> Optional[QWidget]:
        """
        Create enhanced widget with parameter info support.

        Args:
            param_name: Parameter name
            param_type: Parameter type
            current_value: Current parameter value
            param_info: Optional parameter info with docstring

        Returns:
            Enhanced widget for parameter editing or None
        """
        if self._is_path_type(param_type):
            from openhcs.pyqt_gui.widgets.enhanced_path_widget import EnhancedPathWidget
            return EnhancedPathWidget(param_name, current_value, param_info, self.color_scheme)
        else:
            return self.create_widget(param_name, param_type, current_value)

    def create_widget(self, param_name: str, param_type: Type, current_value: Any) -> Optional[QWidget]:
        """
        Create a widget for the given parameter type.
        
        Args:
            param_name: Parameter name
            param_type: Parameter type
            current_value: Current parameter value
            
        Returns:
            Widget for parameter editing or None
        """
        try:
            # Handle complex types first (Optional, Union, List)
            resolved_type = self._resolve_complex_type(param_type)
            if resolved_type != param_type:
                # Recursively handle the resolved type
                return self.create_widget(param_name, resolved_type, current_value)

            # Special case: if current_value is None for basic types, use placeholder widget
            if current_value is None and resolved_type in [int, float, bool]:
                return self._create_placeholder_widget(param_name, resolved_type)

            # Handle enum types
            if self._is_enum_type(param_type):
                return self._create_enum_widget(param_type, current_value)

            # Handle List[Enum] types
            if self._is_list_of_enums(param_type):
                enum_type = self._get_enum_from_list(param_type)
                return self._create_enum_widget(enum_type, current_value)

            # Handle dataclass types (missing from original implementation!)
            if self._is_dataclass_type(param_type):
                # Return None to indicate this should be handled by the parameter form manager
                # The parameter form manager will detect the dataclass and create nested widgets
                return None

            # Handle basic types
            if param_type in self.widget_creators:
                return self.widget_creators[param_type](param_name, current_value)

            # Handle special types
            if param_type == type(None):
                return self._create_none_widget(param_name, current_value)

            # Default to string widget
            logger.warning(f"Unknown parameter type {param_type}, using string widget")
            return self._create_str_widget(param_name, current_value)
            
        except Exception as e:
            logger.error(f"Failed to create widget for {param_name} ({param_type}): {e}")
            return None

    def _resolve_complex_type(self, param_type):
        """
        Resolve complex types like Optional[str], Union[str, Path], List[Enum] to simpler types.

        Args:
            param_type: The type to resolve

        Returns:
            Simplified type that can be handled by basic widget creators
        """
        try:
            origin = get_origin(param_type)
            args = get_args(param_type)

            # Handle Optional[T] (which is Union[T, None])
            if origin is Union:
                # Filter out NoneType and get the first non-None type
                non_none_types = [arg for arg in args if arg != type(None)]
                if non_none_types:
                    return non_none_types[0]  # Use first non-None type
                return str  # Fallback to string

            # Handle List[T] - for now, treat as string input
            elif origin is list:
                # Keep the list type for special handling
                return param_type

            # No complex type found, return as-is
            return param_type

        except Exception:
            # If anything goes wrong, return original type
            return param_type

    def _is_list_of_enums(self, param_type) -> bool:
        """Check if parameter type is List[Enum]."""
        try:
            origin = get_origin(param_type)
            if origin is list:
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    return self._is_enum_type(inner_type)
            return False
        except Exception:
            return False

    def _get_enum_from_list(self, param_type):
        """Extract enum type from List[Enum] type."""
        try:
            args = get_args(param_type)
            if args and len(args) > 0:
                return args[0]  # Return the enum type
            return None
        except Exception:
            return None
    
    def update_widget_value(self, widget: QWidget, value: Any):
        """
        Update widget value without triggering signals.

        Args:
            widget: Widget to update
            value: New value
        """
        # Handle EnhancedPathWidget FIRST (duck typing)
        if hasattr(widget, 'set_path'):
            widget.set_path(value)
            return

        # Temporarily block signals to avoid recursion for standard widgets
        widget.blockSignals(True)
        
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value) if value is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value) if value is not None else 0.0)
            elif isinstance(widget, QComboBox):
                # Find and set the matching item
                for i in range(widget.count()):
                    if widget.itemData(i) == value:
                        widget.setCurrentIndex(i)
                        break
                else:
                    # If not found, try text matching
                    text_value = str(value) if value is not None else ""
                    index = widget.findText(text_value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(value) if value is not None else "")
            elif isinstance(widget, QSlider):
                widget.setValue(int(value) if value is not None else 0)
            elif isinstance(widget, QDateEdit):
                if isinstance(value, QDate):
                    widget.setDate(value)
                elif value is not None:
                    widget.setDate(QDate.fromString(str(value)))
            elif isinstance(widget, QTimeEdit):
                if isinstance(value, QTime):
                    widget.setTime(value)
                elif value is not None:
                    widget.setTime(QTime.fromString(str(value)))
                    
        except Exception as e:
            logger.warning(f"Failed to update widget value: {e}")
        finally:
            widget.blockSignals(False)

    def _is_path_type(self, param_type: Type) -> bool:
        """Check if parameter type is Path or Optional[Path]."""
        import typing

        # Direct Path type
        if param_type == Path:
            return True

        # Check for Optional[Path] (Union[Path, None])
        if hasattr(typing, 'get_origin') and hasattr(typing, 'get_args'):
            origin = typing.get_origin(param_type)
            if origin is typing.Union:
                args = typing.get_args(param_type)
                # Optional[Path] is Union[Path, None]
                if len(args) == 2 and Path in args and type(None) in args:
                    return True

        return False

    def _is_enum_type(self, param_type: Type) -> bool:
        """Check if type is an enum."""
        return any(base.__name__ == 'Enum' for base in param_type.__bases__)

    def _is_dataclass_type(self, param_type: Type) -> bool:
        """Check if type is a dataclass."""
        import dataclasses
        return dataclasses.is_dataclass(param_type)
    
    def _create_bool_widget(self, param_name: str, current_value: Any) -> QCheckBox:
        """Create checkbox widget for boolean parameters."""
        widget = QCheckBox()
        widget.setChecked(bool(current_value) if current_value is not None else False)
        widget.setStyleSheet(f"""
            QCheckBox {{
                color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
                border-radius: 3px;
            }}
        """)
        return widget
    
    def _create_int_widget(self, param_name: str, current_value: Any) -> NoScrollSpinBox:
        """Create spinbox widget for integer parameters."""
        widget = NoScrollSpinBox()
        widget.setRange(-999999, 999999)
        widget.setValue(int(current_value) if current_value is not None else 0)
        widget.setStyleSheet(f"""
            QSpinBox {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
            }}
        """)
        return widget
    
    def _create_float_widget(self, param_name: str, current_value: Any) -> NoScrollDoubleSpinBox:
        """Create double spinbox widget for float parameters."""
        widget = NoScrollDoubleSpinBox()
        widget.setRange(-999999.0, 999999.0)
        widget.setDecimals(6)
        widget.setValue(float(current_value) if current_value is not None else 0.0)
        widget.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
            }}
        """)
        return widget
    
    def _create_str_widget(self, param_name: str, current_value: Any) -> QLineEdit:
        """Create line edit widget for string parameters."""
        widget = QLineEdit()
        widget.setText(str(current_value) if current_value is not None else "")
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
            }}
        """)
        return widget

    def _create_placeholder_widget(self, param_name: str, param_type: Type) -> QLineEdit:
        """Create a QLineEdit widget for None values that will show placeholder text."""
        widget = QLineEdit()
        widget.setText("")  # Empty text - placeholder will be applied later

        # Store the original type so we can convert back when user enters a value
        widget.setProperty("original_type", param_type)
        widget.setProperty("is_placeholder_widget", True)

        # Add helpful placeholder text that will be overridden by the placeholder system
        if param_type == int:
            widget.setPlaceholderText("Enter integer value...")
        elif param_type == float:
            widget.setPlaceholderText("Enter decimal value...")
        elif param_type == bool:
            widget.setPlaceholderText("Enter true/false...")

        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
                font-style: italic;  /* Italic to indicate placeholder state */
            }}
            QLineEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
                font-style: normal;  /* Normal when focused */
            }}
        """)
        return widget

    def _create_list_widget(self, param_name: str, current_value: Any) -> QTextEdit:
        """Create text edit widget for list parameters."""
        widget = QTextEdit()
        widget.setMaximumHeight(100)
        
        # Convert list to text representation
        if isinstance(current_value, list):
            text = '\n'.join(str(item) for item in current_value)
        else:
            text = str(current_value) if current_value is not None else ""
        
        widget.setPlainText(text)
        widget.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New', monospace;
            }}
            QTextEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
            }}
        """)
        return widget

    def _create_path_widget(self, param_name: str, current_value: Any) -> QLineEdit:
        """Create line edit widget for Path parameters."""
        widget = QLineEdit()

        # Convert Path to string
        if isinstance(current_value, Path):
            text = str(current_value)
        else:
            text = str(current_value) if current_value is not None else ""

        widget.setText(text)
        widget.setPlaceholderText("Enter file or directory path...")
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New', monospace;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
            }}
        """)
        return widget

    def _create_tuple_widget(self, param_name: str, current_value: Any) -> QLineEdit:
        """Create line edit widget for tuple parameters."""
        widget = QLineEdit()
        
        # Convert tuple to text representation
        if isinstance(current_value, tuple):
            text = ', '.join(str(item) for item in current_value)
        else:
            text = str(current_value) if current_value is not None else ""
        
        widget.setText(text)
        widget.setPlaceholderText("Enter comma-separated values...")
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New', monospace;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
            }}
        """)
        return widget
    
    def _create_enum_widget(self, param_type: Type, current_value: Any) -> NoScrollComboBox:
        """Create combobox widget for enum parameters."""
        widget = NoScrollComboBox()
        
        # Add enum values
        for enum_value in param_type:
            widget.addItem(str(enum_value.value), enum_value)
        
        # Set current value
        if current_value is not None:
            for i in range(widget.count()):
                if widget.itemData(i) == current_value:
                    widget.setCurrentIndex(i)
                    break
        
        widget.setStyleSheet(f"""
            QComboBox {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
            }}
            QComboBox::drop-down {{
                border: none;
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }}
            QComboBox::down-arrow {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                selection-background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
        """)
        return widget
    
    def _create_none_widget(self, param_name: str, current_value: Any) -> QLineEdit:
        """Create widget for None type parameters."""
        widget = QLineEdit()
        widget.setText(str(current_value) if current_value is not None else "None")
        widget.setPlaceholderText("None")
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.text_disabled)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
                font-style: italic;
            }}
        """)
        return widget
