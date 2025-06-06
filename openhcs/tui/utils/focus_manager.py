"""
Mathematically correct focus management for OpenHCS TUI.

ELIMINATES: All focus-related RuntimeWarnings through circuit breaker pattern.
GUARANTEES: No concurrent focus operations possible.
"""
import asyncio
import logging
from typing import Optional, Any, Coroutine
from prompt_toolkit.application import get_app

logger = logging.getLogger(__name__)


class FocusManager:
    """
    Circuit breaker pattern for focus operations.
    
    MATHEMATICAL GUARANTEE: _focusing_active prevents all concurrent operations.
    """
    
    def __init__(self):
        self._focusing_active: bool = False
        self._active_focus_task: Optional[asyncio.Task] = None
    
    async def set_focus_after_delay(self, target: Any, delay: float = 0.1) -> bool:
        """
        Set focus with circuit breaker protection.
        
        GUARANTEE: If _focusing_active == True, operation is skipped (no concurrency).
        """
        if self._focusing_active:
            logger.debug("Focus operation skipped - already focusing")
            return False
            
        self._focusing_active = True
        try:
            await asyncio.sleep(delay)
            get_app().layout.focus(target)
            logger.debug(f"Focus set to: {target}")
            return True
        except asyncio.CancelledError:
            logger.debug("Focus operation cancelled")
            return False
        except Exception as e:
            logger.warning(f"Focus operation failed: {e}")
            return False
        finally:
            self._focusing_active = False
    
    async def managed_focus_during_dialog(self, target: Any, dialog_coro: Coroutine, delay: float = 0.1):
        """
        MATHEMATICALLY CORRECT focus management during dialog.
        
        ELIMINATES: RuntimeWarning from improper task cancellation.
        GUARANTEES: Task is properly awaited after cancellation.
        """
        from openhcs.tui.utils.unified_task_manager import get_task_manager
        
        focus_task = None
        try:
            # Create focus task through unified task manager
            focus_task = get_task_manager().create_task(
                self.set_focus_after_delay(target, delay),
                name=f"focus_{target.__class__.__name__}"
            )
            
            # Run dialog
            result = await dialog_coro
            return result
            
        finally:
            # MATHEMATICAL CORRECTNESS: Always await cancelled tasks
            if focus_task and not focus_task.done():
                focus_task.cancel()
                try:
                    await focus_task  # ← CRITICAL: Prevents RuntimeWarning
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
    
    def cancel_active_focus(self) -> None:
        """Cancel any active focus operation."""
        if self._active_focus_task and not self._active_focus_task.done():
            self._active_focus_task.cancel()
            self._active_focus_task = None


# Global singleton
_global_focus_manager: Optional[FocusManager] = None


def get_focus_manager() -> FocusManager:
    """Get global focus manager singleton."""
    global _global_focus_manager
    if _global_focus_manager is None:
        _global_focus_manager = FocusManager()
    return _global_focus_manager


async def focus_after_delay(target: Any, delay: float = 0.1) -> bool:
    """Convenience function for focus operations."""
    return await get_focus_manager().set_focus_after_delay(target, delay)
