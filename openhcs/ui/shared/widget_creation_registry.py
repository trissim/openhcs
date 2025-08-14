"""Declarative Widget Creation Registry for OpenHCS UI"""

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Type, Callable, Dict, get_origin, get_args, Union


@dataclasses.dataclass(frozen=True)
class TypeResolution:
    """Immutable type resolution configuration."""
    UNION_NONE_ARGS_COUNT: int = 2

    @staticmethod
    def resolve_optional(param_type: Type) -> Type:
        """Resolve Optional[T] to T using functional composition."""
        return (
            next(arg for arg in get_args(param_type) if arg is not type(None))
            if (origin := get_origin(param_type)) is Union
            and len(args := get_args(param_type)) == TypeResolution.UNION_NONE_ARGS_COUNT
            and type(None) in args
            else param_type
        )


@dataclasses.dataclass(frozen=True)
class TypeCheckers:
    """Declarative type checking functions."""

    @staticmethod
    def is_enum(param_type: Type) -> bool:
        """Check if type is an Enum."""
        return isinstance(param_type, type) and issubclass(param_type, Enum)

    @staticmethod
    def is_list_of_enums(param_type: Type) -> bool:
        """Check if type is List[Enum]."""
        return (get_origin(param_type) is list and
                get_args(param_type) and
                TypeCheckers.is_enum(get_args(param_type)[0]))

    @staticmethod
    def get_enum_from_list(param_type: Type) -> Type:
        """Extract enum type from List[Enum]."""
        return get_args(param_type)[0]

    @staticmethod
    def is_union_with_list_wrapped_enum(param_type: Type) -> bool:
        """Check if Union contains List[Enum]."""
        if get_origin(param_type) is not Union:
            return False
        return any(get_origin(arg) is list and get_args(arg) and TypeCheckers.is_enum(get_args(arg)[0])
                  for arg in get_args(param_type))

    @staticmethod
    def extract_enum_type_from_union(param_type: Type) -> Type:
        """Extract enum type from Union containing List[Enum]."""
        for arg in get_args(param_type):
            if get_origin(arg) is list and get_args(arg) and TypeCheckers.is_enum(get_args(arg)[0]):
                return get_args(arg)[0]
        raise ValueError(f"No enum type found in union {param_type}")

    @staticmethod
    def extract_enum_from_list_value(current_value: Any) -> Any:
        """Extract enum value from list wrapper."""
        return (current_value[0] if isinstance(current_value, list) and
                len(current_value) == 1 and isinstance(current_value[0], Enum)
                else current_value)


@dataclasses.dataclass
class WidgetRegistry:
    """Immutable widget creation registry with functional dispatch."""
    _creators: Dict[Type, Callable] = dataclasses.field(default_factory=dict)
    _type_checkers: Dict[Callable, Callable] = dataclasses.field(default_factory=dict)

    def register(self, type_or_checker: Type | Callable, creator_func: Callable) -> None:
        """Register widget creator using declarative dispatch."""
        target_dict = self._creators if isinstance(type_or_checker, type) else self._type_checkers
        target_dict[type_or_checker] = creator_func

    def create_widget(self, param_name: str, param_type: Type, current_value: Any,
                     widget_id: str, parameter_info: Any = None) -> Any:
        """Create widget using functional composition and fail-loud dispatch."""
        resolved_type = TypeResolution.resolve_optional(param_type)

        # Functional dispatch with early return pattern
        if creator := self._creators.get(resolved_type):
            return creator(param_name, resolved_type, current_value, widget_id, parameter_info)

        # Type checker dispatch using functional composition
        if creator := next((creator for checker, creator in self._type_checkers.items()
                           if checker(resolved_type)), None):
            return creator(param_name, resolved_type, current_value, widget_id, parameter_info)

        # Fail-loud fallback
        if fallback := self._creators.get(str):
            return fallback(param_name, resolved_type, current_value, widget_id, parameter_info)

        raise ValueError(f"No widget creator registered for type: {resolved_type}")


# Declarative registry factory functions
def create_textual_registry() -> WidgetRegistry:
    """Create Textual TUI widget registry using functional composition."""
    from .textual_widget_strategies import create_textual_registry as _create_registry
    return _create_registry()


def create_pyqt6_registry() -> WidgetRegistry:
    """Create PyQt6 widget registry using functional composition."""
    from .pyqt6_widget_strategies import create_pyqt6_registry as _create_registry
    return _create_registry()


# Direct widget creation functions - no unnecessary abstraction layers
def create_textual_widget(param_name: str, param_type: Type, current_value: Any, widget_id: str, parameter_info: Any = None) -> Any:
    """Create Textual TUI widget directly."""
    from textual.widgets import Input, Checkbox, Collapsible

    param_type = resolve_optional(param_type)

    if param_type == bool:
        return Checkbox(value=bool(current_value), id=widget_id, compact=True)
    elif param_type == int:
        return Input(value=str(current_value or ""), type="integer", id=widget_id)
    elif param_type == float:
        return Input(value=str(current_value or ""), type="number", id=widget_id)
    elif param_type == str:
        return Input(value=str(current_value or ""), type="text", id=widget_id)
    elif is_enum(param_type):
        from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
        return EnumRadioSet(param_type, current_value, id=widget_id)
    elif dataclasses.is_dataclass(param_type):
        return Collapsible(title=param_name.replace('_', ' ').title(), collapsed=current_value is None)
    elif is_list_of_enums(param_type):
        from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
        enum_type = get_enum_from_list(param_type)
        display_value = (current_value[0].value if current_value and isinstance(current_value, list) and current_value else None)
        return EnumRadioSet(enum_type, display_value, id=widget_id)
    else:
        return Input(value=str(current_value or ""), type="text", id=widget_id)


def create_pyqt6_widget(param_name: str, param_type: Type, current_value: Any, widget_id: str, parameter_info: Any = None) -> Any:
    """Create PyQt6 widget directly."""
    from PyQt6.QtWidgets import QCheckBox, QLineEdit, QComboBox, QGroupBox, QVBoxLayout
    from openhcs.pyqt_gui.widgets.shared.no_scroll_spinbox import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox

    param_type = resolve_optional(param_type)

    if param_type == bool:
        widget = QCheckBox()
        widget.setChecked(bool(current_value))
        return widget
    elif param_type == int:
        widget = NoScrollSpinBox()
        widget.setRange(-999999, 999999)
        widget.setValue(int(current_value) if current_value else 0)
        return widget
    elif param_type == float:
        widget = NoScrollDoubleSpinBox()
        widget.setRange(-999999.0, 999999.0)
        widget.setValue(float(current_value) if current_value else 0.0)
        return widget
    elif param_type == str:
        widget = QLineEdit()
        widget.setText(str(current_value or ""))
        return widget
    elif param_type == Path:
        from openhcs.pyqt_gui.widgets.enhanced_path_widget import EnhancedPathWidget
        from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
        return EnhancedPathWidget(param_name, current_value, parameter_info, PyQt6ColorScheme())
    elif is_enum(param_type):
        widget = NoScrollComboBox()
        for enum_value in param_type:
            widget.addItem(enum_value.value, enum_value)
        if current_value:
            for i in range(widget.count()):
                if widget.itemData(i) == current_value:
                    widget.setCurrentIndex(i)
                    break
        return widget
    elif dataclasses.is_dataclass(param_type):
        group_box = QGroupBox(param_name.replace('_', ' ').title())
        QVBoxLayout(group_box)
        return group_box
    elif is_list_of_enums(param_type):
        enum_type = get_enum_from_list(param_type)
        widget = QComboBox()
        for enum_value in enum_type:
            widget.addItem(enum_value.value, enum_value)
        if current_value and isinstance(current_value, list) and current_value:
            first_item = current_value[0]
            for i in range(widget.count()):
                if widget.itemData(i) == first_item:
                    widget.setCurrentIndex(i)
                    break
        return widget
    else:
        widget = QLineEdit()
        widget.setText(str(current_value or ""))
        return widget
