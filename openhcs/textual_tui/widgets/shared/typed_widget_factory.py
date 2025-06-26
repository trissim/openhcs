# File: openhcs/textual_tui/widgets/shared/typed_widget_factory.py

import dataclasses
from enum import Enum
from typing import get_origin, get_args, Any, Optional
from textual.widgets import Input, Checkbox, Collapsible
from .enum_radio_set import EnumRadioSet
from ..different_values_wrapper import create_different_values_widget

class TypedWidgetFactory:
    """Simple type → widget mapping with universal 'DIFFERENT VALUES' support."""

    @staticmethod
    def create_widget(param_type, current_value, widget_id, is_different_values=False, default_value=None):
        """Mathematical mapping: type → widget with optional 'DIFFERENT VALUES' support."""

        # Create the appropriate widget based on type
        if param_type == bool:
            widget = Checkbox(value=bool(current_value or False), id=widget_id, compact=True)
        elif param_type == int:
            widget = Input(value=str(current_value or ""), type="integer", id=widget_id)
        elif param_type == float:
            widget = Input(value=str(current_value or ""), type="number", id=widget_id)
        elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
            widget = EnumRadioSet(param_type, current_value, id=widget_id)
        elif TypedWidgetFactory._is_list_of_enums(param_type):
            # Handle List[Enum] types (like List[VariableComponents])
            enum_type = TypedWidgetFactory._get_enum_from_list(param_type)
            # For list of enums, current_value might be a list, so get first item or None
            display_value = None
            if current_value and isinstance(current_value, list) and len(current_value) > 0:
                first_item = current_value[0]
                display_value = first_item.value if hasattr(first_item, 'value') else str(first_item)
            widget = EnumRadioSet(enum_type, display_value, id=widget_id)
        elif dataclasses.is_dataclass(param_type):
            widget = TypedWidgetFactory._create_nested_dataclass_widget(param_type, current_value, widget_id)
        else:
            # Everything else is text input
            widget = Input(value=str(current_value or ""), type="text", id=widget_id)

        # If this is a different values field, wrap it with universal functionality
        if is_different_values and default_value is not None:
            from ..different_values_wrapper import DifferentValuesWrapper
            wrapper = DifferentValuesWrapper(
                widget=widget,
                default_value=default_value,
                field_name=widget_id
            )
            widget._different_values_wrapper = wrapper

        return widget

    @staticmethod
    def create_different_values_widget(param_type, default_value, widget_id, field_name=""):
        """Create a widget specifically for 'DIFFERENT VALUES' state."""
        return TypedWidgetFactory.create_widget(
            param_type=param_type,
            current_value=None,  # No current value in different state
            widget_id=widget_id,
            is_different_values=True,
            default_value=default_value
        )

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
