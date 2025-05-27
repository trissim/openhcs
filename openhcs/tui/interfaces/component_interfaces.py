"""
Component interface definitions for hybrid TUI architecture.

Defines protocols that components must implement to ensure
consistent behavior and clean separation of concerns.
"""

from typing import Any, Callable, Optional, Protocol
from prompt_toolkit.layout import Container

class ComponentInterface(Protocol):
    """
    Base interface for all UI components.
    
    All components must provide a container property and support data updates.
    """
    
    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        ...
        
    def update_data(self, data: Any) -> None:
        """Update component with new data."""
        ...

class EditorComponentInterface(ComponentInterface):
    """
    Interface for editor components that manage editable values.
    
    Extends ComponentInterface with value management and change callbacks.
    """
    
    def get_current_value(self) -> Any:
        """Get current edited value."""
        ...
        
    def set_change_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for value changes."""
        ...
        
    def reset_to_original(self) -> None:
        """Reset to original value."""
        ...
        
    def has_changes(self) -> bool:
        """Check if component has unsaved changes."""
        ...

class ControllerInterface(Protocol):
    """
    Interface for controller components that manage state and coordination.
    
    Controllers coordinate between components and manage application state.
    """
    
    async def initialize_controller(self) -> None:
        """Initialize controller and its managed components."""
        ...
        
    async def cleanup_controller(self) -> None:
        """Clean up controller resources."""
        ...
        
    def get_container(self) -> Container:
        """Get the main container for this controller."""
        ...

class AsyncComponentInterface(ComponentInterface):
    """
    Interface for components that require async initialization.
    
    Some components need to perform async operations during setup.
    """
    
    async def initialize_async(self) -> None:
        """Perform async initialization."""
        ...
        
    async def cleanup_async(self) -> None:
        """Perform async cleanup."""
        ...

class StatefulComponentInterface(ComponentInterface):
    """
    Interface for components that manage internal state.
    
    Provides state management capabilities for complex components.
    """
    
    def get_state(self) -> dict:
        """Get current component state."""
        ...
        
    def set_state(self, state: dict) -> None:
        """Set component state."""
        ...
        
    def reset_state(self) -> None:
        """Reset component to initial state."""
        ...

class ValidatedComponentInterface(ComponentInterface):
    """
    Interface for components that perform validation.
    
    Provides validation capabilities for form components.
    """
    
    def validate(self) -> dict:
        """
        Validate current component state.
        
        Returns:
            Dict with 'valid' bool and optional 'errors' list
        """
        ...
        
    def set_validation_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for validation results."""
        ...
