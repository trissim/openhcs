# File: openhcs/textual_tui/widgets/shared/typed_widget_factory.py

import dataclasses
from enum import Enum
from typing import get_origin, get_args
from textual.widgets import Input, Checkbox, Collapsible
from .enum_radio_set import EnumRadioSet

class TypedWidgetFactory:
    """Simple type → widget mapping. That's it."""

    @staticmethod
    def create_widget(param_type, current_value, widget_id):
        """Mathematical mapping: type → widget."""

        if param_type == bool:
            return Checkbox(value=bool(current_value or False), id=widget_id, compact=True)
        elif param_type == int:
            return Input(value=str(current_value or ""), type="integer", id=widget_id)
        elif param_type == float:
            return Input(value=str(current_value or ""), type="number", id=widget_id)
        elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
            return EnumRadioSet(param_type, current_value, id=widget_id)
        elif TypedWidgetFactory._is_list_of_enums(param_type):
            # Handle List[Enum] types (like List[VariableComponents])
            enum_type = TypedWidgetFactory._get_enum_from_list(param_type)
            # For list of enums, current_value might be a list, so get first item or None
            display_value = None
            if current_value and isinstance(current_value, list) and len(current_value) > 0:
                first_item = current_value[0]
                display_value = first_item.value if hasattr(first_item, 'value') else str(first_item)
            return EnumRadioSet(enum_type, display_value, id=widget_id)
        elif dataclasses.is_dataclass(param_type):
            return TypedWidgetFactory._create_nested_dataclass_widget(param_type, current_value, widget_id)
        else:
            # Everything else is text input
            return Input(value=str(current_value or ""), type="text", id=widget_id)

    @staticmethod
    def _is_list_of_enums(param_type) -> bool:
        """Check if parameter type is List[Enum]."""
        try:
            # Check if it's a generic type (like List[Something])
            origin = get_origin(param_type)
            if origin is list:
                # Get the type arguments (e.g., VariableComponents from List[VariableComponents])
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    # Check if the inner type is an enum
                    return hasattr(inner_type, '__bases__') and Enum in inner_type.__bases__
            return False
        except Exception:
            return False

    @staticmethod
    def _get_enum_from_list(param_type):
        """Extract enum type from List[Enum] type."""
        try:
            args = get_args(param_type)
            if args and len(args) > 0:
                return args[0]  # Return the enum type (e.g., VariableComponents)
            return None
        except Exception:
            return None

    @staticmethod
    def _create_nested_dataclass_widget(dataclass_type, current_value, widget_id):
        """Create collapsible widget for nested dataclass."""
        # Create human-readable title
        title = dataclass_type.__name__.replace('Config', '').replace('_', ' ').title()

        # Create collapsible container with NO ID - it's just structure
        collapsible = Collapsible(title=title, collapsed=False)

        # The nested form will be added by ParameterFormManager
        return collapsible
