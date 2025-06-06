"""
Mathematically correct error handling for OpenHCS TUI.

ELIMINATES: All error handling recursion through circuit breaker pattern.
GUARANTEES: No recursive error display possible.
"""
import asyncio
import logging
from typing import Optional, Any
from openhcs.tui.utils.dialog_helpers import show_scrollable_error_dialog

logger = logging.getLogger(__name__)


class SafeErrorHandler:
    """
    Circuit breaker pattern for error display.
    
    MATHEMATICAL GUARANTEE: _showing_dialog prevents all recursive error display.
    """
    
    def __init__(self):
        self._showing_dialog: bool = False
        self._error_queue: list = []
        self._circuit_breaker_active: bool = False
    
    async def handle_error(self, exception: Exception, context: str = "operation", app_state: Any = None) -> None:
        """
        MATHEMATICALLY SAFE error handling.
        
        GUARANTEE: If _showing_dialog == True, no new dialogs start (no recursion).
        FALLBACK CHAIN: Dialog → Log → Console → Silent (never crash).
        """
        # Circuit breaker - if already showing dialog, just log
        if self._showing_dialog:
            logger.error(f"Error during error handling in {context}: {exception}", exc_info=True)
            return
        
        # If circuit breaker is active, only log
        if self._circuit_breaker_active:
            logger.error(f"Circuit breaker active - logging error in {context}: {exception}", exc_info=True)
            return
        
        # Try to show dialog
        self._showing_dialog = True
        try:
            # Use existing excellent show_scrollable_error_dialog
            await show_scrollable_error_dialog(
                title=f"Error in {context}",
                message=str(exception),
                exception=exception,
                app_state=app_state
            )
        except Exception as dialog_error:
            # Dialog failed - activate circuit breaker and log
            self._circuit_breaker_active = True
            logger.error(f"Error dialog failed: {dialog_error}", exc_info=True)
            logger.error(f"Original error in {context}: {exception}", exc_info=True)
            
            # Try console fallback
            try:
                print(f"CRITICAL ERROR in {context}: {exception}")
            except Exception:
                # Even console failed - silent mode (app keeps running)
                pass
        finally:
            self._showing_dialog = False
    
    def handle_error_sync(self, exception: Exception, context: str = "operation", app_state: Any = None) -> None:
        """
        Synchronous wrapper for error handling.

        USES: UnifiedTaskManager to schedule async error display.
        GUARANTEE: No blocking in sync context.
        """
        coro = None
        try:
            from openhcs.tui.utils.unified_task_manager import get_task_manager
            task_manager = get_task_manager()
            # Create coroutine only after confirming task manager is available
            coro = self.handle_error(exception, context, app_state)
            task_manager.fire_and_forget(coro, f"error_handling_{context}")
            coro = None  # Successfully scheduled, don't close it
        except RuntimeError as e:
            # Task manager not initialized or event loop not running
            # Fall back to logging only
            logger.error(f"Failed to schedule async error handling for {context}: {e}")
            logger.error(f"Original error: {exception}", exc_info=True)
            if coro:
                coro.close()  # Close unscheduled coroutine
        except Exception as e:
            # Any other error in task scheduling
            logger.error(f"Unexpected error scheduling error handler for {context}: {e}")
            logger.error(f"Original error: {exception}", exc_info=True)
            if coro:
                coro.close()  # Close unscheduled coroutine
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after error dialog system is stable."""
        self._circuit_breaker_active = False
        logger.info("Error handling circuit breaker reset")


# Global singleton
_global_error_handler: Optional[SafeErrorHandler] = None


def get_error_handler() -> SafeErrorHandler:
    """Get global error handler singleton."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = SafeErrorHandler()
    return _global_error_handler


async def safe_show_error(exception: Exception, context: str = "operation", app_state: Any = None) -> None:
    """Convenience function for async error display."""
    await get_error_handler().handle_error(exception, context, app_state)


def safe_show_error_sync(exception: Exception, context: str = "operation", app_state: Any = None) -> None:
    """Convenience function for sync error display."""
    get_error_handler().handle_error_sync(exception, context, app_state)
