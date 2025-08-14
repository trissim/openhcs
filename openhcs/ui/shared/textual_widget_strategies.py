"""Textual TUI Widget Creation Functions"""

import dataclasses
from textual.widgets import Input, Checkbox, Collapsible
from .widget_creation_registry import WidgetRegistry, TypeCheckers


# Widget creation functions - simple and direct
def create_bool_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    return Checkbox(value=bool(current_value), id=widget_id, compact=True)


def create_int_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    return Input(value=str(current_value or ""), type="integer", id=widget_id)


def create_float_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    return Input(value=str(current_value or ""), type="number", id=widget_id)


def create_str_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    return Input(value=str(current_value or ""), type="text", id=widget_id)


def create_enum_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
    return EnumRadioSet(param_type, current_value, id=widget_id)


def create_dataclass_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    return Collapsible(title=param_name.replace('_', ' ').title(), collapsed=current_value is None)


def create_list_of_enums_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
    enum_type = TypeCheckers.get_enum_from_list(param_type)
    display_value = (current_value[0].value if current_value and isinstance(current_value, list) and current_value else None)
    return EnumRadioSet(enum_type, display_value, id=widget_id)


# Registry creation function
def create_textual_registry() -> WidgetRegistry:
    """Create Textual TUI widget registry."""
    registry = WidgetRegistry()

    # Register direct type mappings
    registry.register(bool, create_bool_widget)
    registry.register(int, create_int_widget)
    registry.register(float, create_float_widget)
    registry.register(str, create_str_widget)

    # Register type checker mappings
    registry.register(TypeCheckers.is_enum, create_enum_widget)
    registry.register(dataclasses.is_dataclass, create_dataclass_widget)
    registry.register(TypeCheckers.is_list_of_enums, create_list_of_enums_widget)

    return registry


# Simplified different values widget creation
def create_different_values_widget(param_name: str, param_type: type, default_value, widget_id: str):
    """Create different values widget for batch editing."""
    if param_type in (str, int, float):
        from openhcs.textual_tui.widgets.different_values_input import DifferentValuesInput
        return DifferentValuesInput(default_value, param_name, id=widget_id)
    elif param_type == bool:
        from openhcs.textual_tui.widgets.different_values_checkbox import DifferentValuesCheckbox
        return DifferentValuesCheckbox(default_value, param_name, id=widget_id)
    elif TypeCheckers.is_enum(param_type):
        from openhcs.textual_tui.widgets.different_values_radio_set import DifferentValuesRadioSet
        return DifferentValuesRadioSet(param_type, default_value, param_name, id=widget_id)
    else:
        from openhcs.textual_tui.widgets.shared.typed_widget_factory import TypedWidgetFactory
        return TypedWidgetFactory.create_different_values_widget(param_type, default_value, widget_id, param_name)
