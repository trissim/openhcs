"""
Component interface definitions for hybrid TUI.

Defines standard interfaces that all components must implement
to ensure consistent behavior and clean architecture.
"""

from .component_interfaces import (
    ComponentInterface,
    EditorComponentInterface,
    ControllerInterface
)

__all__ = [
    'ComponentInterface',
    'EditorComponentInterface', 
    'ControllerInterface'
]
