"""
Async Handler Architecture - Elegant sync→async bridging for TUI.

This module provides the architectural solution for the systematic sync/async
boundary friction throughout the OpenHCS TUI. Instead of manual fire_and_forget
calls everywhere, this provides transparent async capability to all UI handlers.

ELIMINATES: Manual fire_and_forget() calls scattered throughout codebase
PROVIDES: Clean, natural async handlers that "just work"
PATTERN: Declarative monkey patching like the button override system
"""

import asyncio
import logging
from typing import Any, Callable, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)


class AsyncCapableHandler:
    """
    Wrapper that automatically handles sync→async bridging for UI handlers.
    
    This is the core of the architectural solution. It allows handlers to
    return coroutines naturally, and automatically schedules them without
    manual fire_and_forget calls.
    """
    
    def __init__(self, handler: Callable, context_name: Optional[str] = None):
        """
        Initialize async-capable handler wrapper.
        
        Args:
            handler: The original handler function
            context_name: Optional context for task naming
        """
        self.handler = handler
        self.context_name = context_name or "async_handler"
        
        # Preserve original function metadata
        if hasattr(handler, '__name__'):
            self.__name__ = f"async_{handler.__name__}"
        if hasattr(handler, '__doc__'):
            self.__doc__ = handler.__doc__
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute handler with automatic async bridging.
        
        Returns:
            Handler result, with coroutines automatically scheduled
        """
        try:
            result = self.handler(*args, **kwargs)
            
            # Check if result is a coroutine
            if asyncio.iscoroutine(result):
                # Automatically schedule async operation
                from openhcs.tui.utils.unified_task_manager import get_task_manager
                task_name = f"{self.context_name}_{id(result)}"
                get_task_manager().fire_and_forget(result, task_name)
                
                # Return True for UI event handling (indicates event was handled)
                return True
            
            # Return original result for sync handlers
            return result
            
        except Exception as e:
            logger.error(f"Error in async handler {self.context_name}: {e}", exc_info=True)
            # Show error using sync error handler to avoid recursion
            from openhcs.tui.utils.dialog_helpers import show_global_error_sync
            show_global_error_sync(e, f"handler_{self.context_name}")
            return False


def async_handler(context_name: Optional[str] = None):
    """
    Decorator for creating async-capable handlers.
    
    Usage:
        @async_handler("button_click")
        def my_handler():
            return await some_async_operation()
    
    Args:
        context_name: Optional context for task naming
        
    Returns:
        Decorated function with async capability
    """
    def decorator(func: Callable) -> AsyncCapableHandler:
        return AsyncCapableHandler(func, context_name)
    return decorator


def make_async_capable(handler: Callable, context_name: Optional[str] = None) -> AsyncCapableHandler:
    """
    Convert any handler to be async-capable.
    
    This is the functional interface for the wrapper.
    
    Args:
        handler: Handler function to wrap
        context_name: Optional context for task naming
        
    Returns:
        Async-capable handler wrapper
    """
    return AsyncCapableHandler(handler, context_name)


# Architectural monkey patching functions
def _patch_frame_mouse_handlers():
    """Patch Frame class to auto-wrap mouse handlers."""
    from prompt_toolkit.widgets import Frame as OriginalFrame

    # Patch the mouse_handler setter to auto-wrap async handlers
    original_setattr = OriginalFrame.__setattr__

    def patched_setattr(self, name, value):
        if name == 'mouse_handler' and value and callable(value):
            # Auto-wrap mouse handlers when they're assigned
            title = getattr(self, 'title', 'untitled') if hasattr(self, 'title') else 'untitled'
            value = make_async_capable(value, f"frame_{title}")
        original_setattr(self, name, value)

    OriginalFrame.__setattr__ = patched_setattr


# Button patching removed - handled in _StyledButton class directly


def _patch_window_mouse_handlers():
    """Patch Window class to auto-wrap mouse handlers."""
    from prompt_toolkit.layout.containers import Window as OriginalWindow
    
    original_setattr = OriginalWindow.__setattr__
    
    def patched_setattr(self, name, value):
        if name == 'mouse_handler' and value and callable(value):
            # Auto-wrap mouse handlers
            value = make_async_capable(value, "window_mouse")
        original_setattr(self, name, value)
    
    OriginalWindow.__setattr__ = patched_setattr


def install_frame_and_window_async_handlers():
    """
    Install async handler architecture for Frame and Window components only.

    Button async handling is done in _StyledButton class directly to avoid conflicts.
    """
    logger.info("Installing Frame and Window async handler architecture...")

    try:
        _patch_frame_mouse_handlers()
        _patch_window_mouse_handlers()

        logger.info("✅ Frame and Window async handlers installed successfully")
        logger.info("🎯 Frame and Window mouse handlers now support async operations")

    except Exception as e:
        logger.error(f"Failed to install async handler architecture: {e}", exc_info=True)
        raise


def install_async_handler_architecture():
    """Legacy function - use install_frame_and_window_async_handlers instead."""
    install_frame_and_window_async_handlers()


# Convenience functions for manual usage
def async_mouse_handler(context_name: str = "mouse"):
    """Decorator specifically for mouse handlers."""
    return async_handler(context_name)


def async_button_handler(context_name: str = "button"):
    """Decorator specifically for button handlers."""
    return async_handler(context_name)


def async_key_handler(context_name: str = "key"):
    """Decorator specifically for key handlers."""
    return async_handler(context_name)
