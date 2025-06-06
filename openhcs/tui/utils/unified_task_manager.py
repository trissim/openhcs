"""
Unified Background Task Manager for OpenHCS TUI.

Replaces the clusterfuck of mixed async patterns with a single, 
mathematically correct task management system.
"""
import asyncio
import logging
import sys
from typing import Set, Optional, Coroutine, Any, Callable
from weakref import WeakSet

logger = logging.getLogger(__name__)


class UnifiedTaskManager:
    """
    Single point of truth for all background task management in TUI.
    
    Eliminates RuntimeWarnings and provides proper lifecycle management.
    """
    
    def __init__(self, max_tasks: int = 1000):
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown_requested = False
        self._error_handler: Optional[Callable[[Exception, str], None]] = None
        self._max_tasks = max_tasks
        self._total_tasks_created = 0
    
    def set_error_handler(self, handler: Callable[[Exception, str], None]) -> None:
        """Set global error handler for task exceptions."""
        self._error_handler = handler
    
    def create_task(self, coro: Coroutine[Any, Any, Any], name: str = None) -> asyncio.Task:
        """
        Create and track a background task.
        
        Args:
            coro: Coroutine to run
            name: Optional task name for debugging
            
        Returns:
            The created task
        """
        if self._shutdown_requested:
            # Close the coroutine to prevent leak
            coro.close()
            # FAIL LOUD: Don't lie with dummy tasks, tell the truth
            raise RuntimeError(f"Task manager is shutting down, cannot create task '{name}'. Fix your shutdown sequence.")

        # Check task limit
        active_count = len([t for t in self._tasks if not t.done()])
        if active_count >= self._max_tasks:
            # Close the coroutine to prevent leak
            coro.close()
            # FAIL LOUD: Don't hide resource exhaustion with dummy tasks
            raise RuntimeError(f"Task limit exceeded ({active_count}/{self._max_tasks}). Cannot create task '{name}'. Your code is creating too many tasks - fix it.")

        # Create task with wrapped coroutine
        wrapped_coro = self._wrap_task(coro, name or "unnamed")
        try:
            task = asyncio.create_task(wrapped_coro, name=name)
        except Exception as e:
            # If task creation fails, close the wrapped coroutine to prevent leak
            wrapped_coro.close()
            raise  # Re-raise the exception

        self._tasks.add(task)
        self._total_tasks_created += 1
        task.add_done_callback(self._tasks.discard)

        # Log task creation for monitoring
        if self._total_tasks_created % 100 == 0:
            logger.info(f"Task manager stats: {self._total_tasks_created} total created, {active_count} active")

        return task
    
    async def _wrap_task(self, coro: Coroutine, name: str) -> Any:
        """Wrap task with error handling."""
        try:
            return await coro
        except asyncio.CancelledError:
            logger.debug(f"Task '{name}' was cancelled")
            raise
        except Exception as e:
            logger.error(f"Task '{name}' failed: {e}", exc_info=True)
            if self._error_handler:
                try:
                    self._error_handler(e, f"background task '{name}'")
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}", exc_info=True)
            raise
    
    def fire_and_forget(self, coro: Coroutine[Any, Any, Any], name: str = None) -> None:
        """
        Fire and forget a coroutine - don't return the task.

        Use this for UI callbacks that don't need result tracking.
        Exceptions are handled by the error handler, preventing "never retrieved" warnings.

        FAILS LOUD: If task cannot be created (shutdown, limit exceeded), logs error and closes coroutine.
        No bullshit dummy tasks that lie about what happened.
        """
        try:
            task = self.create_task(coro, name)
            # Add done callback to ensure exceptions are retrieved and handled
            def handle_fire_forget_result(task_ref):
                try:
                    # This retrieves the exception if any, preventing the warning
                    task_ref.result()
                except Exception:
                    # Exception already handled by _wrap_task, just prevent the warning
                    pass
            task.add_done_callback(handle_fire_forget_result)
        except RuntimeError as e:
            # Task creation failed (shutdown, limit exceeded, event loop issues)
            # LOG THE TRUTH about what happened
            if "shutting down" in str(e):
                logger.warning(f"Ignoring task '{name}' during shutdown: {e}")
            elif "limit exceeded" in str(e):
                logger.error(f"Task limit exceeded, dropping task '{name}': {e}")
            else:
                logger.error(f"Failed to schedule fire_and_forget task '{name}': {e}")

            # Clean up the coroutine (already done by create_task, but be explicit)
            if hasattr(coro, 'close'):
                coro.close()
        except Exception as e:
            # Any other error
            logger.error(f"Unexpected error in fire_and_forget for task '{name}': {e}")
            # Clean up the coroutine
            if hasattr(coro, 'close'):
                coro.close()
    
    async def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shutdown all running tasks.
        
        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        self._shutdown_requested = True
        
        if not self._tasks:
            return
        
        logger.info(f"Shutting down {len(self._tasks)} background tasks")
        
        # Cancel all tasks
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(self._tasks), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Task shutdown timed out after {timeout}s")
        
        logger.info("Background task shutdown complete")
    
    @property
    def active_task_count(self) -> int:
        """Get number of active tasks."""
        return len([t for t in self._tasks if not t.done()])

    @property
    def total_task_count(self) -> int:
        """Get total number of tasks ever created."""
        return self._total_tasks_created

    @property
    def completed_task_count(self) -> int:
        """Get number of completed tasks."""
        return len([t for t in self._tasks if t.done()])

    def get_task_stats(self) -> dict:
        """Get comprehensive task statistics."""
        active = self.active_task_count
        completed = self.completed_task_count
        total_tracked = len(self._tasks)

        return {
            'active': active,
            'completed': completed,
            'total_tracked': total_tracked,
            'total_created': self._total_tasks_created,
            'max_tasks': self._max_tasks,
            'shutdown_requested': self._shutdown_requested
        }


# Global instance - initialized by TUI
_global_task_manager: Optional[UnifiedTaskManager] = None


def get_task_manager() -> UnifiedTaskManager:
    """Get the global task manager instance."""
    if _global_task_manager is None:
        raise RuntimeError("UnifiedTaskManager not initialized. Call initialize_task_manager() first.")
    return _global_task_manager


def initialize_task_manager() -> UnifiedTaskManager:
    """Initialize the global task manager with error handler connection."""
    global _global_task_manager
    if _global_task_manager is not None:
        logger.warning("Task manager already initialized")
        return _global_task_manager

    _global_task_manager = UnifiedTaskManager()

    # Immediately connect to global error handler to prevent race conditions
    def global_error_handler(exception: Exception, context: str):
        """Route task manager errors to global error system."""
        try:
            from openhcs.tui.utils.dialog_helpers import show_global_error_sync
            show_global_error_sync(exception, context, None)  # No app_state available yet
        except Exception as handler_error:
            logger.error(f"Global error handler failed: {handler_error}", exc_info=True)
            logger.error(f"Original error: {exception}", exc_info=True)

    _global_task_manager.set_error_handler(global_error_handler)
    logger.info("UnifiedTaskManager initialized with global error handler")
    return _global_task_manager


async def shutdown_task_manager() -> None:
    """Shutdown the global task manager."""
    global _global_task_manager
    if _global_task_manager is not None:
        await _global_task_manager.shutdown()
        _global_task_manager = None
        logger.info("UnifiedTaskManager shutdown complete")
