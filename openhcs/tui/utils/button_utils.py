"""
Button utilities for consistent TUI button creation.

Provides a single, predictable pattern for all button creation.
"""
from typing import Callable, Optional, Any
from prompt_toolkit.widgets import Button


def create_button(text: str, handler: Optional[Callable] = None, padding: int = 2, **kwargs) -> Button:
    """
    Create a button with calculated width - single pattern for all buttons.
    
    Args:
        text: Button text
        handler: Click handler function
        padding: Extra width padding (default: 2)
        **kwargs: Additional Button arguments
        
    Returns:
        Button with calculated width = len(text) + padding
    """
    return Button(text, handler=handler, width=len(text) + padding, **kwargs)


def action_button(text: str, handler: Optional[Callable] = None, **kwargs) -> Button:
    """Create an action button (standard padding)."""
    return create_button(text, handler, padding=2, **kwargs)


def nav_button(text: str, handler: Optional[Callable] = None, **kwargs) -> Button:
    """Create a navigation button (minimal padding for compact nav)."""
    return create_button(text, handler, padding=1, **kwargs)


def dialog_button(text: str, handler: Optional[Callable] = None, **kwargs) -> Button:
    """Create a dialog button (extra padding for prominence)."""
    return create_button(text, handler, padding=4, **kwargs)
