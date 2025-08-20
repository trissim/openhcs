"""
Consolidated UI utilities for PyQt6 parameter forms.

This module replaces four over-engineered utility modules (979 lines) with simple
functional implementations (~45 lines), adhering to OpenHCS coding standards.

Replaces:
- debug_config.py (321 lines)
- enum_display_formatter.py (170 lines) 
- parameter_name_formatter.py (276 lines)
- field_id_generator.py (208 lines)
"""

import logging
from typing import Any
from enum import Enum


def format_param_name(name: str) -> str:
    """Convert snake_case to Title Case: 'param_name' -> 'Param Name'"""
    return name.replace('_', ' ').title()


def format_checkbox_label(name: str) -> str:
    """Create checkbox label: 'param_name' -> 'Enable Param Name'"""
    return f"Enable {format_param_name(name)}"


def format_field_label(name: str) -> str:
    """Create field label: 'param_name' -> 'Param Name:'"""
    return f"{format_param_name(name)}:"


def format_field_id(parent: str, param: str) -> str:
    """Generate field ID: 'parent', 'param' -> 'parent_param'"""
    return f"{parent}_{param}"


def format_reset_button_id(widget_id: str) -> str:
    """Generate reset button ID: 'widget_id' -> 'reset_widget_id'"""
    return f"reset_{widget_id}"


def format_enum_display(enum_value: Enum) -> str:
    """Get enum display text: Enum.VALUE -> 'VALUE'"""
    return enum_value.name.upper()


def format_enum_placeholder(enum_value: Enum, prefix: str = "Pipeline default: ") -> str:
    """Get enum placeholder: Enum.VALUE -> 'Pipeline default: VALUE'"""
    return f"{prefix}{format_enum_display(enum_value)}"


def debug_param(param_name: str, value: Any, context: str = "") -> None:
    """Simple parameter debug logging"""
    context_str = f" [{context}]" if context else ""
    logging.debug(f"PARAM: {param_name} = {value}{context_str}")
