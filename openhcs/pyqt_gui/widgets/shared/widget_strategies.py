"""Magicgui-based PyQt6 Widget Creation with OpenHCS Extensions"""

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type, Callable, Optional, Union

from PyQt6.QtWidgets import QCheckBox, QLineEdit, QComboBox, QGroupBox, QVBoxLayout, QSpinBox, QDoubleSpinBox
from magicgui.widgets import create_widget
from magicgui.type_map import register_type

from openhcs.pyqt_gui.widgets.shared.no_scroll_spinbox import (
    NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox
)
from openhcs.pyqt_gui.widgets.enhanced_path_widget import EnhancedPathWidget
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.ui.shared.widget_creation_registry import resolve_optional, is_enum, is_list_of_enums, get_enum_from_list

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
    # Import here to avoid circular imports
    from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import NoneAwareLineEdit

    # Use NoneAwareLineEdit for proper None handling
    widget = NoneAwareLineEdit()
    widget.set_value(current_value)
    return widget


def create_enum_widget_unified(enum_type: Type, current_value: Any, **kwargs) -> QComboBox:
    """Unified enum widget creator with consistent display text."""
    from openhcs.ui.shared.ui_utils import format_enum_display

    widget = NoScrollComboBox()
    for enum_value in enum_type:
        display_text = format_enum_display(enum_value)
        widget.addItem(display_text, enum_value)

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
        resolved_type = resolve_optional(param_type)

        # Handle direct List[Enum] types - create multi-selection checkbox group
        if is_list_of_enums(resolved_type):
            return self._create_checkbox_group_widget(param_name, resolved_type, current_value)

        # Extract enum from list wrapper for other cases
        extracted_value = (current_value[0] if isinstance(current_value, list) and
                          len(current_value) == 1 and isinstance(current_value[0], Enum)
                          else current_value)

        # Handle direct enum types
        if is_enum(resolved_type):
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
            # For string types, use our NoneAwareLineEdit instead of magicgui
            if resolved_type == str:
                widget = create_string_fallback_widget(current_value=extracted_value)
            else:
                # Try magicgui for non-string types, with string fallback for unsupported types
                try:
                    # Handle None values to prevent magicgui from converting None to literal "None" string
                    magicgui_value = extracted_value
                    if extracted_value is None:
                        # Use appropriate default values for magicgui to prevent "None" string conversion
                        if resolved_type == int:
                            magicgui_value = 0
                        elif resolved_type == float:
                            magicgui_value = 0.0
                        elif resolved_type == bool:
                            magicgui_value = False
                        # For other types, let magicgui handle None (might still cause issues but less common)

                    widget = create_widget(annotation=resolved_type, value=magicgui_value)

                    # If original value was None, clear the widget to show placeholder behavior
                    if extracted_value is None and hasattr(widget, 'native'):
                        native_widget = widget.native
                        if hasattr(native_widget, 'setText'):
                            native_widget.setText("")  # Clear text for None values
                        elif hasattr(native_widget, 'setChecked') and resolved_type == bool:
                            native_widget.setChecked(False)  # Uncheck for None bool values

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

    def _create_checkbox_group_widget(self, param_name: str, param_type: Type, current_value: Any):
        """Create multi-selection checkbox group for List[Enum] parameters."""
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox

        enum_type = get_enum_from_list(param_type)
        widget = QGroupBox(param_name.replace('_', ' ').title())
        layout = QVBoxLayout(widget)

        # Store checkboxes for value retrieval
        widget._checkboxes = {}

        for enum_value in enum_type:
            checkbox = QCheckBox(enum_value.value)
            checkbox.setObjectName(f"{param_name}_{enum_value.value}")
            widget._checkboxes[enum_value] = checkbox
            layout.addWidget(checkbox)

        # Set current values (check boxes for items in the list)
        if current_value and isinstance(current_value, list):
            for enum_value in current_value:
                if enum_value in widget._checkboxes:
                    widget._checkboxes[enum_value].setChecked(True)

        # Add method to get selected values
        def get_selected_values():
            return [enum_val for enum_val, checkbox in widget._checkboxes.items()
                   if checkbox.isChecked()]
        widget.get_selected_values = get_selected_values

        return widget


# Registry pattern removed - use create_pyqt6_widget from widget_creation_registry.py instead


class PlaceholderConfig:
    """Declarative placeholder configuration."""
    PLACEHOLDER_PREFIX = "Pipeline default: "
    # Stronger styling that overrides application theme
    PLACEHOLDER_STYLE = "color: #888888 !important; font-style: italic !important; opacity: 0.7;"
    INTERACTION_HINTS = {
        'checkbox': 'click to set your own value',
        'combobox': 'select to set your own value'
    }


# Functional placeholder strategies
PLACEHOLDER_STRATEGIES: Dict[str, Callable[[Any, str], None]] = {
    'setPlaceholderText': lambda widget, text: _apply_lineedit_placeholder(widget, text),
    'setSpecialValueText': lambda widget, text: _apply_spinbox_placeholder(widget, text),
}


def _extract_default_value(placeholder_text: str) -> str:
    """Extract default value from placeholder text, handling enum values properly."""
    value = placeholder_text.replace(PlaceholderConfig.PLACEHOLDER_PREFIX, "").strip()

    # Handle enum values like "Microscope.AUTO" -> "AUTO"
    if '.' in value and not value.startswith('('):  # Avoid breaking "(none)" values
        parts = value.split('.')
        if len(parts) == 2:
            # Return just the enum member name
            return parts[1]

    return value


def _extract_numeric_value_from_placeholder(placeholder_text: str) -> Optional[Union[int, float]]:
    """
    Extract numeric value from placeholder text for integer/float fields.

    Args:
        placeholder_text: Full placeholder text like "Pipeline default: 42"

    Returns:
        Numeric value if found and valid, None otherwise
    """
    try:
        # Extract the value part after the prefix
        value_str = placeholder_text.replace(PlaceholderConfig.PLACEHOLDER_PREFIX, "").strip()

        # Try to parse as int first, then float
        if value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
            return int(value_str)
        else:
            # Try float parsing
            return float(value_str)
    except (ValueError, AttributeError):
        return None


def _apply_placeholder_styling(widget: Any, interaction_hint: str, placeholder_text: str) -> None:
    """Apply consistent placeholder styling and tooltip."""
    # Get widget-specific styling that's strong enough to override application theme
    widget_type = type(widget).__name__

    if widget_type == "QComboBox":
        # Strong combobox-specific styling
        style = """
            QComboBox {
                color: #888888 !important;
                font-style: italic !important;
                opacity: 0.7;
            }
        """
    elif widget_type == "QCheckBox":
        # Strong checkbox-specific styling
        style = """
            QCheckBox {
                color: #888888 !important;
                font-style: italic !important;
                opacity: 0.7;
            }
        """
    else:
        # Fallback to general styling
        style = PlaceholderConfig.PLACEHOLDER_STYLE

    widget.setStyleSheet(style)
    widget.setToolTip(f"{placeholder_text} ({interaction_hint})")
    widget.setProperty("is_placeholder_state", True)


def _apply_lineedit_placeholder(widget: Any, text: str) -> None:
    """Apply placeholder to line edit with proper state tracking."""
    # Clear existing text so placeholder becomes visible
    widget.clear()
    widget.setPlaceholderText(text)
    # Set placeholder state property for consistency with other widgets
    widget.setProperty("is_placeholder_state", True)
    # Add tooltip for consistency
    widget.setToolTip(text)


def _apply_spinbox_placeholder(widget: Any, text: str) -> None:
    """Apply placeholder to spinbox using numeric-only special value text."""
    # Extract numeric value from placeholder text for integer/float fields
    numeric_value = _extract_numeric_value_from_placeholder(text)

    # For numeric fields, show only the number, not the full text
    if numeric_value is not None:
        widget.setSpecialValueText(str(numeric_value))
    else:
        # Fallback to full text for non-numeric placeholders
        widget.setSpecialValueText(text)

    # Set widget to minimum value to show the special value text
    if hasattr(widget, 'minimum'):
        widget.setValue(widget.minimum())

    # Apply visual styling to indicate this is a placeholder
    _apply_placeholder_styling(
        widget,
        'change value to set your own',
        text  # Keep full text in tooltip
    )


def _apply_checkbox_placeholder(widget: QCheckBox, placeholder_text: str) -> None:
    """Apply placeholder to checkbox with visual preview without triggering signals."""
    try:
        default_value = _extract_default_value(placeholder_text).lower() == 'true'
        # Block signals to prevent checkbox state changes from triggering parameter updates
        widget.blockSignals(True)
        try:
            widget.setChecked(default_value)
        finally:
            widget.blockSignals(False)
        _apply_placeholder_styling(
            widget,
            PlaceholderConfig.INTERACTION_HINTS['checkbox'],
            placeholder_text
        )
    except Exception:
        widget.setToolTip(placeholder_text)


def _apply_path_widget_placeholder(widget: Any, placeholder_text: str) -> None:
    """Apply placeholder to Path widget by targeting the inner QLineEdit."""
    try:
        # Path widgets have a path_input attribute that's a QLineEdit
        if hasattr(widget, 'path_input'):
            # Clear any existing text and apply placeholder to the inner QLineEdit
            widget.path_input.clear()
            widget.path_input.setPlaceholderText(placeholder_text)
            widget.path_input.setProperty("is_placeholder_state", True)
            widget.path_input.setToolTip(placeholder_text)
        else:
            # Fallback to tooltip if structure is different
            widget.setToolTip(placeholder_text)
    except Exception:
        widget.setToolTip(placeholder_text)


def _apply_combobox_placeholder(widget: QComboBox, placeholder_text: str) -> None:
    """Apply placeholder to combobox with visual preview using robust enum matching."""
    try:
        default_value = _extract_default_value(placeholder_text)

        # Find matching item using robust enum matching
        matching_index = next(
            (i for i in range(widget.count())
             if _item_matches_value(widget, i, default_value)),
            -1
        )

        if matching_index >= 0:
            widget.setCurrentIndex(matching_index)

        # Always apply placeholder styling to indicate this is a placeholder value
        _apply_placeholder_styling(
            widget,
            PlaceholderConfig.INTERACTION_HINTS['combobox'],
            placeholder_text
        )
    except Exception:
        widget.setToolTip(placeholder_text)


def _item_matches_value(widget: QComboBox, index: int, target_value: str) -> bool:
    """Check if combobox item matches target value using robust enum matching."""
    item_data = widget.itemData(index)
    item_text = widget.itemText(index)
    target_normalized = target_value.upper()

    # Primary: Match enum name (most reliable)
    if item_data and hasattr(item_data, 'name'):
        if item_data.name.upper() == target_normalized:
            return True

    # Secondary: Match enum value (case-insensitive)
    if item_data and hasattr(item_data, 'value'):
        if str(item_data.value).upper() == target_normalized:
            return True

    # Tertiary: Match display text (case-insensitive)
    if item_text.upper() == target_normalized:
        return True

    return False


# Declarative widget-to-strategy mapping
WIDGET_PLACEHOLDER_STRATEGIES: Dict[Type, Callable[[Any, str], None]] = {
    QCheckBox: _apply_checkbox_placeholder,
    QComboBox: _apply_combobox_placeholder,
    QSpinBox: _apply_spinbox_placeholder,
    QDoubleSpinBox: _apply_spinbox_placeholder,
    NoScrollSpinBox: _apply_spinbox_placeholder,
    NoScrollDoubleSpinBox: _apply_spinbox_placeholder,
    NoScrollComboBox: _apply_combobox_placeholder,
    QLineEdit: _apply_lineedit_placeholder,  # Add standard QLineEdit support
}

# Add Path widget support dynamically to avoid import issues
def _register_path_widget_strategy():
    """Register Path widget strategy dynamically to avoid circular imports."""
    try:
        from openhcs.pyqt_gui.widgets.enhanced_path_widget import EnhancedPathWidget
        WIDGET_PLACEHOLDER_STRATEGIES[EnhancedPathWidget] = _apply_path_widget_placeholder
    except ImportError:
        pass  # Path widget not available

def _register_none_aware_lineedit_strategy():
    """Register NoneAwareLineEdit strategy dynamically to avoid circular imports."""
    try:
        from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import NoneAwareLineEdit
        WIDGET_PLACEHOLDER_STRATEGIES[NoneAwareLineEdit] = _apply_lineedit_placeholder
    except ImportError:
        pass  # NoneAwareLineEdit not available

# Register widget strategies
_register_path_widget_strategy()
_register_none_aware_lineedit_strategy()

# Functional signal connection registry
SIGNAL_CONNECTION_REGISTRY: Dict[str, callable] = {
    'stateChanged': lambda widget, param_name, callback:
        widget.stateChanged.connect(lambda: callback(param_name, widget.isChecked())),
    'textChanged': lambda widget, param_name, callback:
        widget.textChanged.connect(lambda v: callback(param_name,
            widget.get_value() if hasattr(widget, 'get_value') else v)),
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
    # Checkbox group signal (custom attribute for multi-selection widgets)
    'get_selected_values': lambda widget, param_name, callback:
        PyQt6WidgetEnhancer._connect_checkbox_group_signals(widget, param_name, callback),
}





@dataclasses.dataclass(frozen=True)
class PyQt6WidgetEnhancer:
    """Widget enhancement using functional dispatch patterns."""

    @staticmethod
    def apply_placeholder_text(widget: Any, placeholder_text: str) -> None:
        """Apply placeholder using declarative widget-strategy mapping."""
        # Direct widget type mapping for enhanced placeholders
        widget_strategy = WIDGET_PLACEHOLDER_STRATEGIES.get(type(widget))
        if widget_strategy:
            return widget_strategy(widget, placeholder_text)

        # Method-based fallback for standard widgets
        strategy = next(
            (strategy for method_name, strategy in PLACEHOLDER_STRATEGIES.items()
             if hasattr(widget, method_name)),
            lambda w, t: w.setToolTip(t) if hasattr(w, 'setToolTip') else None
        )
        strategy(widget, placeholder_text)

    @staticmethod
    def apply_global_config_placeholder(widget: Any, field_name: str, global_config: Any = None) -> None:
        """
        Apply placeholder to standalone widget using global config.

        This method allows applying placeholders to widgets that are not part of
        a dataclass form by directly using the global configuration.

        Args:
            widget: The widget to apply placeholder to
            field_name: Name of the field in the global config
            global_config: Global config instance (uses thread-local if None)
        """
        try:
            if global_config is None:
                from openhcs.core.config import _current_pipeline_config
                if hasattr(_current_pipeline_config, 'value') and _current_pipeline_config.value:
                    global_config = _current_pipeline_config.value
                else:
                    return  # No global config available

            # Get the field value from global config
            if hasattr(global_config, field_name):
                field_value = getattr(global_config, field_name)

                # Format the placeholder text appropriately for different types
                if hasattr(field_value, 'name'):  # Enum
                    from openhcs.ui.shared.ui_utils import format_enum_placeholder
                    placeholder_text = format_enum_placeholder(field_value)
                else:
                    placeholder_text = f"Pipeline default: {field_value}"

                PyQt6WidgetEnhancer.apply_placeholder_text(widget, placeholder_text)
        except Exception:
            # Silently fail if placeholder can't be applied
            pass

    @staticmethod
    def connect_change_signal(widget: Any, param_name: str, callback: Any) -> None:
        """Connect signal with placeholder state management."""
        magicgui_widget = PyQt6WidgetEnhancer._get_magicgui_wrapper(widget)

        # Create placeholder-aware callback wrapper
        def create_wrapped_callback(original_callback, value_getter):
            def wrapped():
                PyQt6WidgetEnhancer._clear_placeholder_state(widget)
                original_callback(param_name, value_getter())
            return wrapped

        # Prioritize magicgui signals
        if magicgui_widget and hasattr(magicgui_widget, 'changed'):
            magicgui_widget.changed.connect(
                create_wrapped_callback(callback, lambda: magicgui_widget.value)
            )
            return

        # Fallback to native PyQt6 signals
        connector = next(
            (connector for signal_name, connector in SIGNAL_CONNECTION_REGISTRY.items()
             if hasattr(widget, signal_name)),
            None
        )

        if connector:
            placeholder_aware_callback = lambda pn, val: (
                PyQt6WidgetEnhancer._clear_placeholder_state(widget),
                callback(pn, val)
            )[-1]
            connector(widget, param_name, placeholder_aware_callback)
        else:
            raise ValueError(f"Widget {type(widget).__name__} has no supported change signal")

    @staticmethod
    def _connect_checkbox_group_signals(widget: Any, param_name: str, callback: Any) -> None:
        """Connect signals for checkbox group widgets."""
        if hasattr(widget, '_checkboxes'):
            # Connect to each checkbox's stateChanged signal
            for checkbox in widget._checkboxes.values():
                checkbox.stateChanged.connect(
                    lambda: callback(param_name, widget.get_selected_values())
                )

    @staticmethod
    def _clear_placeholder_state(widget: Any) -> None:
        """Clear placeholder state using functional approach."""
        if not widget.property("is_placeholder_state"):
            return

        widget.setStyleSheet("")
        widget.setProperty("is_placeholder_state", False)

        # Clean tooltip using functional pattern
        current_tooltip = widget.toolTip()
        cleaned_tooltip = next(
            (current_tooltip.replace(f" ({hint})", "")
             for hint in PlaceholderConfig.INTERACTION_HINTS.values()
             if f" ({hint})" in current_tooltip),
            current_tooltip
        )
        widget.setToolTip(cleaned_tooltip)

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
