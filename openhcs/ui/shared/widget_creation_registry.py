"""
Functional widget creation registry for OpenHCS UI frameworks.

Provides extensible type-to-widget dispatch using simple dicts and functions,
eliminating class-based abstractions while maintaining clean extensibility.
"""

import dataclasses
from pathlib import Path
from typing import Type, get_origin, get_args, Union
from enum import Enum


def resolve_optional(param_type: Type) -> Type:
    """Resolve Optional[T] to T."""
    if get_origin(param_type) is Union:
        args = get_args(param_type)
        if len(args) == 2 and type(None) in args:
            return next(arg for arg in args if arg is not type(None))
    return param_type


def is_enum(param_type: Type) -> bool:
    """Check if type is an Enum."""
    return isinstance(param_type, type) and issubclass(param_type, Enum)


def is_list_of_enums(param_type: Type) -> bool:
    """Check if type is List[Enum]."""
    return (get_origin(param_type) is list and
            get_args(param_type) and
            is_enum(get_args(param_type)[0]))


def get_enum_from_list(param_type: Type) -> Type:
    """Extract enum type from List[Enum]."""
    return get_args(param_type)[0]


# Registry factory functions - return actual widget creators
def create_textual_registry():
    """Return Textual widget creator function."""
    from openhcs.ui.shared.textual_widget_strategies import create_textual_widget
    return create_textual_widget


def create_pyqt6_registry():
    """Return PyQt6 widget creator function."""
    try:
        from openhcs.pyqt_gui.widgets.shared.widget_strategies import MagicGuiWidgetFactory
        factory = MagicGuiWidgetFactory()
        return factory.create_widget
    except ImportError:
        def fallback_creator(*args, **kwargs):
            raise ImportError("PyQt6 not available - install with pip install 'openhcs[gui]'")
        return fallback_creator
