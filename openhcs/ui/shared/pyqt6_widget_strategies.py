"""Magicgui-based PyQt6 Widget Creation with OpenHCS Extensions"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, Type

from PyQt6.QtWidgets import QCheckBox, QLineEdit, QComboBox, QGroupBox, QVBoxLayout
from magicgui.widgets import create_widget
from magicgui.type_map import register_type

from openhcs.pyqt_gui.widgets.shared.no_scroll_spinbox import (
    NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox
)
from openhcs.pyqt_gui.widgets.enhanced_path_widget import EnhancedPathWidget
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from .widget_creation_registry import WidgetRegistry, TypeCheckers, TypeResolution

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WidgetConfig:
    """Immutable widget configuration constants."""
    NUMERIC_RANGE_MIN: int = -999999
    NUMERIC_RANGE_MAX: int = 999999
    FLOAT_PRECISION: int = 6


def create_enhanced_path_widget(param_name: str = "", current_value: Any = None, parameter_info: Any = None):
    """Factory function for OpenHCS enhanced path widgets."""
    return EnhancedPathWidget(param_name, current_value, parameter_info, PyQt6ColorScheme())


def register_openhcs_widgets():
    """Register OpenHCS custom widgets with magicgui type system."""
    # Register using string widget types that magicgui recognizes
    register_type(int, widget_type="SpinBox")
    register_type(float, widget_type="FloatSpinBox")
    register_type(Path, widget_type="FileEdit")





# Functional widget replacement registry
WIDGET_REPLACEMENT_REGISTRY: Dict[Type, callable] = {
    bool: lambda current_value, **kwargs: (
        lambda w: w.setChecked(bool(current_value)) or w
    )(QCheckBox()),
    int: lambda current_value, **kwargs: (
        lambda w: w.setValue(int(current_value) if current_value else 0) or w
    )(NoScrollSpinBox()),
    float: lambda current_value, **kwargs: (
        lambda w: w.setValue(float(current_value) if current_value else 0.0) or w
    )(NoScrollDoubleSpinBox()),
    Path: lambda current_value, param_name, parameter_info, **kwargs:
        create_enhanced_path_widget(param_name, current_value, parameter_info),
}

# String fallback widget for any type magicgui cannot handle
def create_string_fallback_widget(current_value: Any, **kwargs) -> QLineEdit:
    """Create string fallback widget for unsupported types."""
    widget = QLineEdit()
    widget.setText(str(current_value) if current_value is not None else "")
    return widget


def create_enum_widget_unified(enum_type: Type, current_value: Any, **kwargs) -> QComboBox:
    """Unified enum widget creator."""
    widget = NoScrollComboBox()
    for enum_value in enum_type:
        widget.addItem(enum_value.value, enum_value)

    # Set current selection
    if current_value and hasattr(current_value, '__class__') and isinstance(current_value, enum_type):
        for i in range(widget.count()):
            if widget.itemData(i) == current_value:
                widget.setCurrentIndex(i)
                break

    return widget

# Functional configuration registry
CONFIGURATION_REGISTRY: Dict[Type, callable] = {
    int: lambda widget: widget.setRange(WidgetConfig.NUMERIC_RANGE_MIN, WidgetConfig.NUMERIC_RANGE_MAX)
        if hasattr(widget, 'setRange') else None,
    float: lambda widget: (
        widget.setRange(WidgetConfig.NUMERIC_RANGE_MIN, WidgetConfig.NUMERIC_RANGE_MAX)
        if hasattr(widget, 'setRange') else None,
        widget.setDecimals(WidgetConfig.FLOAT_PRECISION)
        if hasattr(widget, 'setDecimals') else None
    )[-1],
}


@dataclasses.dataclass(frozen=True)
class MagicGuiWidgetFactory:
    """OpenHCS widget factory using functional mapping dispatch."""

    def create_widget(self, param_name: str, param_type: Type, current_value: Any,
                     widget_id: str, parameter_info: Any = None) -> Any:
        """Create widget using functional registry dispatch."""
        resolved_type = TypeResolution.resolve_optional(param_type)

        # Handle list-wrapped enum pattern in Union
        if TypeCheckers.is_union_with_list_wrapped_enum(resolved_type):
            enum_type = TypeCheckers.extract_enum_type_from_union(resolved_type)
            extracted_value = TypeCheckers.extract_enum_from_list_value(current_value)
            return create_enum_widget_unified(enum_type, extracted_value)

        # Handle direct List[Enum] types
        if TypeCheckers.is_list_of_enums(resolved_type):
            enum_type = TypeCheckers.get_enum_from_list(resolved_type)
            extracted_value = TypeCheckers.extract_enum_from_list_value(current_value)
            return create_enum_widget_unified(enum_type, extracted_value)

        # Extract enum from list wrapper for other cases
        extracted_value = TypeCheckers.extract_enum_from_list_value(current_value)

        # Handle direct enum types
        if TypeCheckers.is_enum(resolved_type):
            return create_enum_widget_unified(resolved_type, extracted_value)

        # Check for OpenHCS custom widget replacements
        replacement_factory = WIDGET_REPLACEMENT_REGISTRY.get(resolved_type)
        if replacement_factory:
            widget = replacement_factory(
                current_value=extracted_value,
                param_name=param_name,
                parameter_info=parameter_info
            )
        else:
            # Try magicgui for standard types, with string fallback for unsupported types
            try:
                widget = create_widget(annotation=resolved_type, value=extracted_value)
                # Extract native PyQt6 widget from magicgui wrapper if needed
                if hasattr(widget, 'native'):
                    native_widget = widget.native
                    native_widget._magicgui_widget = widget  # Store reference for signal connections
                    widget = native_widget
            except (ValueError, TypeError) as e:
                # Fallback to string widget for any type magicgui cannot handle
                logger.warning(f"Widget creation failed for {param_name} ({resolved_type}): {e}", exc_info=True)
                widget = create_string_fallback_widget(current_value=extracted_value)

        # Functional configuration dispatch
        configurator = CONFIGURATION_REGISTRY.get(resolved_type, lambda w: w)
        configurator(widget)

        return widget


def create_pyqt6_registry() -> WidgetRegistry:
    """Create PyQt6 widget registry leveraging magicgui's automatic type system."""
    register_openhcs_widgets()

    registry = WidgetRegistry()
    factory = MagicGuiWidgetFactory()

    # Register single factory for all types - let magicgui handle type dispatch
    all_types = [bool, int, float, str, Path]
    for type_key in all_types:
        registry.register(type_key, factory.create_widget)

    # Register for complex types that magicgui handles automatically
    complex_type_checkers = [TypeCheckers.is_enum, dataclasses.is_dataclass, TypeCheckers.is_list_of_enums]
    for checker in complex_type_checkers:
        registry.register(checker, factory.create_widget)

    return registry


# Functional placeholder strategy registry
PLACEHOLDER_STRATEGIES: Dict[str, callable] = {
    'setPlaceholderText': lambda widget, text: widget.setPlaceholderText(text),
    'setSpecialValueText': lambda widget, text: (
        widget.setSpecialValueText(text),
        widget.setValue(widget.minimum()) if hasattr(widget, 'minimum') else None
    )[-1],
}

# Functional signal connection registry
SIGNAL_CONNECTION_REGISTRY: Dict[str, callable] = {
    'stateChanged': lambda widget, param_name, callback:
        widget.stateChanged.connect(lambda: callback(param_name, widget.isChecked())),
    'textChanged': lambda widget, param_name, callback:
        widget.textChanged.connect(lambda v: callback(param_name, v)),
    'valueChanged': lambda widget, param_name, callback:
        widget.valueChanged.connect(lambda v: callback(param_name, v)),
    'currentTextChanged': lambda widget, param_name, callback:
        widget.currentTextChanged.connect(lambda: callback(param_name,
            widget.currentData() if hasattr(widget, 'currentData') else widget.currentText())),
    'path_changed': lambda widget, param_name, callback:
        widget.path_changed.connect(lambda v: callback(param_name, v)),
    # Magicgui-specific widget signals
    'changed': lambda widget, param_name, callback:
        widget.changed.connect(lambda: callback(param_name, widget.value)),
}


@dataclasses.dataclass(frozen=True)
class PyQt6WidgetEnhancer:
    """Widget enhancement using functional mapping dispatch."""

    @staticmethod
    def apply_placeholder_text(widget: Any, placeholder_text: str) -> None:
        """Apply placeholder using functional strategy dispatch."""
        strategy = next((strategy for method_name, strategy in PLACEHOLDER_STRATEGIES.items()
                        if hasattr(widget, method_name)), None)

        if strategy:
            strategy(widget, placeholder_text)
        else:
            raise ValueError(f"Widget {type(widget).__name__} does not support placeholder text")

    @staticmethod
    def connect_change_signal(widget: Any, param_name: str, callback: Any) -> None:
        """Connect signal using functional registry dispatch with magicgui support."""
        # Check if we need to get the magicgui wrapper for signal connection
        magicgui_widget = PyQt6WidgetEnhancer._get_magicgui_wrapper(widget)

        # Prioritize magicgui's standard 'changed' signal first
        if magicgui_widget and hasattr(magicgui_widget, 'changed'):
            magicgui_widget.changed.connect(lambda: callback(param_name, magicgui_widget.value))
            return

        # Fall back to native PyQt6 signal patterns
        connector = next((connector for signal_name, connector in SIGNAL_CONNECTION_REGISTRY.items()
                         if hasattr(widget, signal_name)), None)

        if connector:
            connector(widget, param_name, callback)
        else:
            raise ValueError(f"Widget {type(widget).__name__} has no supported change signal")

    @staticmethod
    def _get_magicgui_wrapper(widget: Any) -> Any:
        """Get magicgui wrapper if widget was created by magicgui."""
        # Check if widget has a reference to its magicgui wrapper
        if hasattr(widget, '_magicgui_widget'):
            return widget._magicgui_widget
        # If widget itself is a magicgui widget, return it
        if hasattr(widget, 'changed') and hasattr(widget, 'value'):
            return widget
        return None
