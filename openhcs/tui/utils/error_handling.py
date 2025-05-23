"""
Error handling utilities for the TUI.

This module provides utilities for handling errors in async functions.
"""

import logging
import traceback
from typing import Callable, Awaitable, TypeVar, Optional, Dict, Any, Union, Type, Tuple

logger = logging.getLogger(__name__)

T = TypeVar('T')

async def handle_async_errors(
    coro: Awaitable[T],
    error_message: str = "An error occurred",
    notify_state: Optional[Any] = None,
    operation_name: str = "operation",
    source: str = "unknown",
    error_types: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None
) -> Optional[T]:
    """
    Handle errors in async functions.

    Args:
        coro: The coroutine to execute
        error_message: The error message to log
        notify_state: Optional AppState to notify of the error
        operation_name: The name of the operation for notification
        source: The source of the error for notification
        error_types: Specific error types to catch, or None to catch all exceptions

    Returns:
        The result of the coroutine, or None if an error occurred
    """
    try:
        return await coro
    except Exception as e:
        if error_types is None or isinstance(e, error_types):
            logger.error(f"{error_message}: {e}", exc_info=True)

            # Notify state if provided
            if notify_state is not None and hasattr(notify_state, 'notify'):
                await notify_state.notify('operation_status_changed', {
                    'operation': operation_name,
                    'status': 'error',
                    'message': f"{error_message}: {str(e)}",
                    'source': source
                })
            return None
        else:
            # Re-raise if not a specified error type
            raise

def handle_async_errors_decorator(
    error_message: str = "An error occurred",
    operation_name: str = "operation",
    source: str = "unknown",
    error_types: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[Optional[T]]]]:
    """
    Decorator for handling errors in async functions.

    Args:
        error_message: The error message to log
        operation_name: The name of the operation for notification
        source: The source of the error for notification
        error_types: Specific error types to catch, or None to catch all exceptions

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
        async def wrapper(*args, **kwargs) -> Optional[T]:
            # Find state argument if present
            state = None
            if args and hasattr(args[0], 'state'):
                state = args[0].state
            elif 'state' in kwargs:
                state = kwargs['state']

            return await handle_async_errors(
                func(*args, **kwargs),
                error_message=error_message,
                notify_state=state,
                operation_name=operation_name,
                source=source,
                error_types=error_types
            )
        return wrapper
    return decorator

async def show_error_dialog(
    title: str = "Error",
    message: str = "An error occurred",
    app_state: Optional[Any] = None
) -> None:
    """
    Show an error dialog.

    Args:
        title: The title of the dialog
        message: The error message to display
        app_state: Optional AppState to notify of the error

    Returns:
        None
    """
    from prompt_toolkit.shortcuts import message_dialog

    # Show the dialog
    await message_dialog(
        title=title,
        text=message
    ).run_async()

    # Notify state if provided
    if app_state is not None and hasattr(app_state, 'notify'):
        await app_state.notify('operation_status_changed', {
            'operation': 'error_dialog',
            'status': 'error',
            'message': message,
            'source': 'error_dialog'
        })
