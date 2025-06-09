# File: openhcs/textual_tui/widgets/shared/typed_widget_factory.py

import dataclasses
from enum import Enum
from textual.widgets import Input, Checkbox, Collapsible
from .enum_radio_set import EnumRadioSet

class TypedWidgetFactory:
    """Simple type → widget mapping. That's it."""

    @staticmethod
    def create_widget(param_type, current_value, widget_id):
        """Mathematical mapping: type → widget."""

        if param_type == bool:
            return Checkbox(value=bool(current_value or False), id=widget_id)
        elif param_type == int:
            return Input(value=str(current_value or ""), type="integer", id=widget_id)
        elif param_type == float:
            return Input(value=str(current_value or ""), type="number", id=widget_id)
        elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
            return EnumRadioSet(param_type, current_value, id=widget_id)
        elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
            return EnumRadioSet(param_type, current_value, id=widget_id)
        elif dataclasses.is_dataclass(param_type):
            return TypedWidgetFactory._create_nested_dataclass_widget(param_type, current_value, widget_id)
        else:
            # Everything else is text input
            return Input(value=str(current_value or ""), type="text", id=widget_id)

    @staticmethod
    def _create_nested_dataclass_widget(dataclass_type, current_value, widget_id):
        """Create collapsible widget for nested dataclass."""
        # Create human-readable title
        title = dataclass_type.__name__.replace('Config', '').replace('_', ' ').title()

        # Create collapsible container with NO ID - it's just structure
        collapsible = Collapsible(title=title, collapsed=False)

        # The nested form will be added by ParameterFormManager
        return collapsible
