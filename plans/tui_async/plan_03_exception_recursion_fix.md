# plan_03_exception_recursion_fix.md
## Component: Exception Handler Recursion Prevention

### Objective
Fix the exception handler recursion bomb in dialog_helpers.py where error handlers call async functions that can trigger more error handlers, creating infinite loops and stack overflows.

### Plan
1. **Implement recursion-safe error handling (25 minutes)**
   - Add recursion detection to prevent infinite loops
   - Create fallback logging when dialog display fails
   - Eliminate exception handler calling async functions

2. **Fix show_global_error_sync recursion (15 minutes)**
   - Remove dangerous recursive pattern
   - Implement proper sync/async boundary
   - Add circuit breaker for error display

3. **Fix exception handler patching (10 minutes)**
   - Remove prompt_toolkit monkey patching
   - Implement safer exception handling
   - Prevent handler-calling-handler loops

### Findings

#### Current Exception Recursion Disasters

**File: `openhcs/tui/utils/dialog_helpers.py`**

**RECURSION BOMB 1: Lines 595-606**
```python
def show_global_error_sync(exception: Exception, context: str = "operation", app_state: Optional[Any] = None) -> None:
    """Schedule the global-error dialog on the running PTK event-loop"""
    from prompt_toolkit.application import get_app
    get_app().create_background_task(show_global_error(exception, context, app_state))
    # ↑ CALLS ASYNC FUNCTION FROM SYNC CONTEXT
    # ↑ IF show_global_error FAILS, IT TRIGGERS ANOTHER EXCEPTION HANDLER
    # ↑ WHICH CALLS show_global_error_sync AGAIN = INFINITE RECURSION
```

**RECURSION BOMB 2: Lines 651-663**
```python
def handle_asyncio_exception(loop, context):
    """Handle asyncio exceptions (background tasks, etc)."""
    exception = context.get('exception')
    if exception:
        if isinstance(exception, KeyboardInterrupt):
            return

        # Show error dialog for async exceptions using sync wrapper
        try:
            show_global_error_sync(exception, "async task", app_state)  # ← CALLS RECURSION BOMB 1
        except Exception:
            # Fallback to logging if we can't show dialog
            logger.error(f"Asyncio exception: {exception}", exc_info=exception)
```

**RECURSION BOMB 3: Lines 714-725**
```python
async def patched_run_async(self, *args, **kwargs):
    """Patched run_async that catches prompt_toolkit internal exceptions."""
    try:
        return await original_run_async(self, *args, **kwargs)
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            # Show error dialog for prompt_toolkit exceptions using sync wrapper
            try:
                show_global_error_sync(e, "prompt_toolkit", app_state)  # ← CALLS RECURSION BOMB 1
            except Exception:
                logger.error(f"Prompt_toolkit exception: {e}", exc_info=e)
        raise
```

**The Mathematical Problem:**
```
Exception → show_global_error_sync → create_background_task(show_global_error) 
    → show_global_error fails → Exception Handler → show_global_error_sync → INFINITE LOOP
```

### Implementation Draft

#### Step 1: Create Recursion-Safe Error Handler

**File: `openhcs/tui/utils/safe_error_handler.py`**
```python
"""
Recursion-safe error handling for OpenHCS TUI.

Prevents infinite loops when error handlers themselves fail.
"""
import asyncio
import logging
import threading
from typing import Optional, Any, Set
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SafeErrorHandler:
    """
    Error handler with built-in recursion prevention and circuit breaking.
    
    Prevents infinite loops when error dialogs themselves cause errors.
    """
    
    def __init__(self):
        self._showing_error = threading.local()
        self._error_count = 0
        self._max_errors_per_session = 10
        self._circuit_breaker_active = False
    
    def _is_showing_error(self) -> bool:
        """Check if we're currently showing an error in this thread/task."""
        return getattr(self._showing_error, 'active', False)
    
    def _set_showing_error(self, value: bool) -> None:
        """Set the error showing flag for this thread/task."""
        self._showing_error.active = value
    
    @contextmanager
    def _error_context(self):
        """Context manager to track error handling state."""
        if self._is_showing_error():
            # Already showing error - prevent recursion
            yield False
            return
        
        if self._circuit_breaker_active:
            # Circuit breaker active - only log
            yield False
            return
        
        self._error_count += 1
        if self._error_count > self._max_errors_per_session:
            self._circuit_breaker_active = True
            logger.error("Error handler circuit breaker activated - too many errors")
            yield False
            return
        
        self._set_showing_error(True)
        try:
            yield True
        finally:
            self._set_showing_error(False)
    
    async def show_error_safe(self, exception: Exception, context: str, app_state: Optional[Any] = None) -> bool:
        """
        Show error dialog with recursion protection.
        
        Args:
            exception: The exception to display
            context: Context description
            app_state: TUI state for dialog management
            
        Returns:
            True if dialog was shown, False if prevented by safety mechanisms
        """
        with self._error_context() as can_show:
            if not can_show:
                logger.error(f"Error in {context} (recursion prevented): {exception}", exc_info=True)
                return False
            
            try:
                from openhcs.tui.utils.dialog_helpers import show_scrollable_error_dialog
                await show_scrollable_error_dialog(
                    title=f"Error in {context}",
                    message=f"An error occurred during {context}",
                    exception=exception,
                    app_state=app_state
                )
                return True
            except Exception as dialog_error:
                # Dialog failed - log both errors and prevent recursion
                logger.error(f"Failed to show error dialog: {dialog_error}", exc_info=True)
                logger.error(f"Original error in {context}: {exception}", exc_info=True)
                return False
    
    def show_error_sync(self, exception: Exception, context: str, app_state: Optional[Any] = None) -> None:
        """
        Schedule error dialog from synchronous context with recursion protection.
        
        Args:
            exception: The exception to display
            context: Context description
            app_state: TUI state for dialog management
        """
        with self._error_context() as can_show:
            if not can_show:
                logger.error(f"Error in {context} (recursion prevented): {exception}", exc_info=True)
                return
            
            try:
                from openhcs.tui.utils.unified_task_manager import get_task_manager
                get_task_manager().fire_and_forget(
                    self.show_error_safe(exception, context, app_state),
                    f"safe_error_{context}"
                )
            except Exception as schedule_error:
                # Task scheduling failed - just log
                logger.error(f"Failed to schedule error dialog: {schedule_error}", exc_info=True)
                logger.error(f"Original error in {context}: {exception}", exc_info=True)
    
    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker (for testing or new sessions)."""
        self._circuit_breaker_active = False
        self._error_count = 0
        logger.info("Error handler circuit breaker reset")


# Global safe error handler
_global_error_handler: Optional[SafeErrorHandler] = None


def get_safe_error_handler() -> SafeErrorHandler:
    """Get the global safe error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = SafeErrorHandler()
    return _global_error_handler


async def show_error_safe(exception: Exception, context: str, app_state: Optional[Any] = None) -> bool:
    """Convenience function for safe error display."""
    return await get_safe_error_handler().show_error_safe(exception, context, app_state)


def show_error_sync_safe(exception: Exception, context: str, app_state: Optional[Any] = None) -> None:
    """Convenience function for safe sync error display."""
    get_safe_error_handler().show_error_sync(exception, context, app_state)
```

#### Step 2: Replace Broken show_global_error_sync

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 595-606: COMPLETE REPLACEMENT**

**REMOVE:**
```python
def show_global_error_sync(exception: Exception, context: str = "operation", app_state: Optional[Any] = None) -> None:
    """
    Schedule the global-error dialog on the running PTK event-loop and return
    immediately. Use this from synchronous code.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    from prompt_toolkit.application import get_app
    get_app().create_background_task(show_global_error(exception, context, app_state))
```

**REPLACE WITH:**
```python
def show_global_error_sync(exception: Exception, context: str = "operation", app_state: Optional[Any] = None) -> None:
    """
    Schedule the global-error dialog with recursion protection.
    Safe to call from synchronous code and exception handlers.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
    show_error_sync_safe(exception, context, app_state)
```

#### Step 3: Fix Exception Handler Functions

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 651-663: REPLACE handle_asyncio_exception**

**REMOVE:**
```python
    def handle_asyncio_exception(loop, context):
        """Handle asyncio exceptions (background tasks, etc)."""
        exception = context.get('exception')
        if exception:
            if isinstance(exception, KeyboardInterrupt):
                return

            # Show error dialog for async exceptions using sync wrapper
            try:
                show_global_error_sync(exception, "async task", app_state)
            except Exception:
                # Fallback to logging if we can't show dialog
                logger.error(f"Asyncio exception: {exception}", exc_info=exception)
```

**REPLACE WITH:**
```python
    def handle_asyncio_exception(loop, context):
        """Handle asyncio exceptions with recursion protection."""
        exception = context.get('exception')
        if exception:
            if isinstance(exception, KeyboardInterrupt):
                return

            # Use safe error handler to prevent recursion
            from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
            show_error_sync_safe(exception, "async task", app_state)
```

#### Step 4: Remove Dangerous Prompt Toolkit Patching

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 709-730: COMPLETE REMOVAL**

**REMOVE ENTIRE SECTION:**
```python
    # Also patch prompt_toolkit Application to catch internal exceptions
    try:
        from prompt_toolkit.application import Application
        original_run_async = Application.run_async

        async def patched_run_async(self, *args, **kwargs):
            """Patched run_async that catches prompt_toolkit internal exceptions."""
            try:
                return await original_run_async(self, *args, **kwargs)
            except Exception as e:
                if not isinstance(e, KeyboardInterrupt):
                    # Show error dialog for prompt_toolkit exceptions using sync wrapper
                    try:
                        show_global_error_sync(e, "prompt_toolkit", app_state)
                    except Exception:
                        logger.error(f"Prompt_toolkit exception: {e}", exc_info=e)
                raise

        Application.run_async = patched_run_async
        logger.info("Patched prompt_toolkit Application.run_async for global error handling")
    except Exception as e:
        logger.warning(f"Failed to patch prompt_toolkit: {e}")
```

**REPLACE WITH:**
```python
    # Note: Removed dangerous prompt_toolkit monkey patching that caused recursion.
    # Prompt_toolkit exceptions are now handled by the unified task manager's error handler.
    logger.info("Global exception handler setup complete (without prompt_toolkit patching)")
```

#### Step 5: Fix run_in_executor Exception Handling

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 695-697: REPLACE EXCEPTION HANDLER CALL**

**FIND:**
```python
                # Call the handler properly - it schedules its own background task
                try:
                    handle_asyncio_exception(loop, context)
                except Exception:
                    # If handler fails, just log the original exception
                    logger.error(f"Exception in run_in_executor: {e}", exc_info=e)
```

**REPLACE WITH:**
```python
                # Use safe error handler to prevent recursion
                try:
                    from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
                    show_error_sync_safe(e, "run_in_executor", app_state)
                except Exception:
                    # If handler fails, just log the original exception
                    logger.error(f"Exception in run_in_executor: {e}", exc_info=e)
```

#### Step 6: Update show_global_error Function

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 569-593: REPLACE WITH SAFE VERSION**

**REMOVE:**
```python
async def show_global_error(exception: Exception, context: str = "operation", app_state: Optional[Any] = None):
    """
    Global error handler - shows any error in a scrollable dialog with syntax highlighting.
    Takes focus immediately and restores it when dismissed.

    NOTE: This function is *async*. You must either:
        • `await show_global_error(...)` from async code, or
        • schedule it with `get_app().create_background_task(...)`
          from synchronous code, or
        • use `show_global_error_sync(...)` from synchronous code.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    logger.error(f"Global error in {context}: {exception}", exc_info=True)

    # Show the error dialog - it handles focus management internally
    await show_scrollable_error_dialog(
        title=f"Error in {context}",
        message=f"An error occurred during {context}",
        exception=exception,
        app_state=app_state
    )
```

**REPLACE WITH:**
```python
async def show_global_error(exception: Exception, context: str = "operation", app_state: Optional[Any] = None):
    """
    Global error handler with recursion protection.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    from openhcs.tui.utils.safe_error_handler import show_error_safe
    await show_error_safe(exception, context, app_state)
```

### Verification Steps

1. **Test Recursion Prevention:**
   - Trigger an error in show_scrollable_error_dialog itself
   - Verify no infinite loop occurs
   - Check that fallback logging works

2. **Test Circuit Breaker:**
   - Generate multiple rapid errors
   - Verify circuit breaker activates after threshold
   - Check that logging continues when circuit breaker is active

3. **Test Exception Handler Safety:**
   - Trigger asyncio exceptions
   - Verify no recursion in exception handlers
   - Check that errors are properly displayed or logged

4. **Verify No Monkey Patching:**
   - Confirm prompt_toolkit is not patched
   - Check that TUI still handles exceptions properly
   - Verify no unexpected behavior from removed patching
