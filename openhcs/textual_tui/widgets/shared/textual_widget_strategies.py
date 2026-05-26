"""Textual TUI Widget Creation Functions"""

import dataclasses
from .widget_creation_registry import resolve_optional, is_enum, is_list_of_enums, get_enum_from_list


def create_textual_widget(param_name: str, param_type: type, current_value, widget_id: str, parameter_info=None):
    """Create Textual TUI widget directly."""
    # Lazy import to avoid requiring textual when only PyQt GUI is used
    from textual.widgets import Checkbox, Collapsible
    from openhcs.ui.shared.ui_utils import format_param_name
    from openhcs.textual_tui.widgets.shared.textual_input_spec import TextualInputSpec

    param_type = resolve_optional(param_type)

    if param_type == bool:
        return Checkbox(value=bool(current_value), id=widget_id, compact=True)
    elif param_type == int:
        return TextualInputSpec.for_kind("integer", current_value, widget_id).build()
    elif param_type == float:
        return TextualInputSpec.for_kind("number", current_value, widget_id).build()
    elif param_type == str:
        return TextualInputSpec.for_kind("text", current_value, widget_id).build()
    elif is_enum(param_type):
        from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
        return EnumRadioSet(param_type, current_value, id=widget_id)
    elif dataclasses.is_dataclass(param_type):
        return Collapsible(title=format_param_name(param_name), collapsed=current_value is None)
    elif is_list_of_enums(param_type):
        from openhcs.textual_tui.widgets.shared.enum_radio_set import EnumRadioSet
        enum_type = get_enum_from_list(param_type)
        display_value = (current_value[0].value if current_value and isinstance(current_value, list) and current_value else None)
        return EnumRadioSet(enum_type, display_value, id=widget_id)
    else:
        return TextualInputSpec.for_kind("text", current_value, widget_id).build()


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
