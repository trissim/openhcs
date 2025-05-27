"""
Asynchronous UI Task Manager for OpenHCS TUI.

This module provides the AsyncUIManager and ManagedTask classes to help
manage and track asynchronous operations within the TUI, particularly those
that might interact with the TUIState or need to be cancelled gracefully.
"""
import asyncio
import logging
import functools
from typing import Any, Callable, Coroutine, Dict, Generic, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    # Assuming TUIState is the primary state object, replace with actual if different
    # from .tui_architecture import TUIState # Avoid direct import if possible, pass as Any
    TUIState = Any

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ManagedTask(Generic[T]):
    """
    A wrapper around an asyncio Task to provide lifecycle management
    and easier tracking, potentially with UI state interaction.
    """
    def __init__(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None, ui_state: Optional['TUIState'] = None):
        self.name = name or coro.__name__
        self.ui_state = ui_state # Optional TUIState for status updates
        self._coro = coro # Store original coroutine for potential restart (if applicable)
        self._task: Optional[asyncio.Task[T]] = None
        self._future: Optional[asyncio.Future[T]] = None
        self.status: str = "pending" # pending, running, completed, cancelled, failed

    async def _execute_wrapper(self):
        """Internal wrapper to execute the coroutine and handle outcomes."""
        if not self._future: # Should not happen if run() is used
            self._future = asyncio.get_running_loop().create_future()

        try:
            self.status = "running"
            # logger.debug(f"Task '{self.name}' started.") # Handled by AsyncUIManager
            if self.ui_state and hasattr(self.ui_state, 'notify'):
                # Example: await self.ui_state.notify('task_status_changed', {'name': self.name, 'status': self.status})
                pass # Actual notification would depend on TUIState's API

            result = await self._coro
            self.status = "completed"
            logger.debug(f"Task '{self.name}' completed successfully.")
            if self.ui_state and hasattr(self.ui_state, 'notify'):
                # Example: await self.ui_state.notify('task_status_changed', {'name': self.name, 'status': self.status, 'result': result})
                pass
            self._future.set_result(result)
        except asyncio.CancelledError:
            self.status = "cancelled"
            logger.info(f"Task '{self.name}' was cancelled.")
            if self.ui_state and hasattr(self.ui_state, 'notify'):
                # Example: await self.ui_state.notify('task_status_changed', {'name': self.name, 'status': self.status})
                pass
            # Propagate cancellation to the future if it's not already done
            if not self._future.done():
                 self._future.cancel()
        except Exception as e:
            self.status = "failed"
            logger.error(f"Task '{self.name}' failed with error: {e}", exc_info=True)
            if self.ui_state and hasattr(self.ui_state, 'notify'):
                # Example: await self.ui_state.notify('task_status_changed', {'name': self.name, 'status': self.status, 'error': str(e)})
                pass
            self._future.set_exception(e)
        finally:
            # Clean up reference to the asyncio.Task object once it's done
            # The future holds the result/exception.
            if self._task and self._task.done() and self._task in asyncio.all_tasks():
                 # This check is to ensure we don't try to operate on a task from a different loop
                 # or a task that's already been collected.
                 pass # asyncio automatically manages task cleanup usually.
            self._task = None


    def run(self) -> asyncio.Future[T]:
        """
        Schedules the coroutine to run as an asyncio.Task.
        Returns a future that will eventually hold the result or exception.
        """
        if self._task and not self._task.done():
            logger.warning(f"Task '{self.name}' is already running.")
            if self._future: return self._future
            # Fall through to recreate future if it's missing, though this state is unusual.

        loop = asyncio.get_running_loop()
        self._future = loop.create_future() # Create a new future for this run
        self._task = loop.create_task(self._execute_wrapper(), name=self.name)
        logger.debug(f"Task '{self.name}' scheduled to run.")
        return self._future

    def cancel(self) -> bool:
        """
        Requests cancellation of the running task.
        Returns True if cancellation was requested, False otherwise (e.g., task already done).
        """
        if self._task and not self._task.done():
            logger.info(f"Requesting cancellation for task '{self.name}'.")
            cancelled = self._task.cancel()
            # The _execute_wrapper will handle the asyncio.CancelledError
            # and update the future and status.
            return cancelled
        logger.debug(f"Cannot cancel task '{self.name}': not running or no task exists.")
        return False

    def get_future(self) -> asyncio.Future[T]:
        """
        Returns the future associated with this managed task.
        If the task hasn't been run yet, it will create a future but not schedule the task.
        """
        if self._future is None:
            # This case implies run() hasn't been called.
            # Create a future so a reference can be obtained, but it won't complete
            # until run() is called.
            self._future = asyncio.get_running_loop().create_future()
            logger.debug(f"Future created on demand for task '{self.name}' (not yet run).")
        return self._future

    def done(self) -> bool:
        """Checks if the task's future is done."""
        return self._future.done() if self._future else False

    def result(self) -> Optional[T]:
        """Returns the result of the task if completed, else None. Raises exceptions if task failed."""
        if self._future and self._future.done():
            # Future.result() will raise exception if task failed or was cancelled.
            try:
                return self.get_future().result()
            except asyncio.CancelledError:
                logger.info(f"Attempted to get result of cancelled task '{self.name}'.")
                return None # Or re-raise, depending on desired API
            except Exception as e:
                logger.error(f"Attempted to get result of failed task '{self.name}': {e}")
                raise # Re-raise the original exception
        return None


class AsyncUIManager:
    """
    Manages asynchronous UI-related tasks.
    """
    def __init__(self, state: Optional['TUIState'] = None):
        self.ui_state = state
        self.active_tasks: Dict[str, ManagedTask[Any]] = {} # Track tasks by name or ID

    def submit_task(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None) -> ManagedTask[T]:
        """
        Creates and schedules a new ManagedTask from a coroutine.
        The task is run in a "fire-and-forget" manner from the manager's perspective,
        but its lifecycle can be tracked via the returned ManagedTask object.

        Args:
            coro: The coroutine to run.
            name: An optional name for the task. If None, uses coroutine name.

        Returns:
            The ManagedTask instance wrapping the coroutine.
        """
        task_name = name or coro.__name__
        # Ensure unique names if needed, or allow replacement
        if task_name in self.active_tasks and not self.active_tasks[task_name].done():
            logger.warning(f"Task with name '{task_name}' is already active. Replacing.")
            self.active_tasks[task_name].cancel() # Cancel previous if still running

        managed_task = ManagedTask[T](coro, name=task_name, ui_state=self.ui_state)
        self.active_tasks[task_name] = managed_task
        
        logger.info(f"AsyncUIManager: Submitted task '{task_name}'.")
        managed_task.run() # Schedules the task
        return managed_task

    async def run_task_and_wait(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None) -> T:
        """
        Submits a task and waits for its completion, returning its result
        or raising its exception.

        Args:
            coro: The coroutine to run.
            name: An optional name for the task.

        Returns:
            The result of the coroutine.
        
        Raises:
            Exception: If the coroutine raises an exception.
            asyncio.CancelledError: If the task is cancelled.
        """
        task_name = name or coro.__name__
        logger.info(f"AsyncUIManager: Running task '{task_name}' and waiting for completion.")
        
        managed_task = self.submit_task(coro, name=task_name)
        try:
            return await managed_task.get_future()
        finally:
            # Clean up from active_tasks dict once done, if desired
            if task_name in self.active_tasks and self.active_tasks[task_name].done():
                del self.active_tasks[task_name]
                logger.debug(f"AsyncUIManager: Task '{task_name}' removed from active tracking after completion/failure.")


    def managed_async_task(self, name: Optional[str] = None) -> Callable[..., Callable[..., Coroutine[Any, Any, Any]]]:
        """
        Decorator to wrap a coroutine function, automatically submitting it as a ManagedTask
        when called. The decorated function, when called, will return the ManagedTask instance.

        Args:
            name: Optional name for tasks created by the decorated function.
                  If None, the function's name is used.
        """
        def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., ManagedTask[T]]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> ManagedTask[T]:
                coro = func(*args, **kwargs)
                task_name = name or func.__name__
                logger.debug(f"AsyncUIManager: Creating managed task for '{task_name}' via decorator.")
                return self.submit_task(coro, name=task_name)
            return wrapper
        return decorator # type: ignore

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any], name: Optional[str] = None) -> None:
        """
        Submits a task to be run without returning its ManagedTask instance or future.
        Useful for tasks where the caller does not need to track completion or result directly.
        The task is still managed internally for logging/cancellation if named.
        """
        task_name = name or coro.__name__
        logger.info(f"AsyncUIManager: Firing and forgetting task '{task_name}'.")
        self.submit_task(coro, name=task_name)
        # The ManagedTask instance is not returned to the caller.

    def cancel_task(self, name: str) -> bool:
        """Cancels an active task by its name."""
        if name in self.active_tasks:
            return self.active_tasks[name].cancel()
        logger.warning(f"AsyncUIManager: No active task found with name '{name}' to cancel.")
        return False

    def get_task_status(self, name: str) -> Optional[str]:
        """Gets the status of a managed task."""
        if name in self.active_tasks:
            return self.active_tasks[name].status
        return None

    def get_all_tasks_status(self) -> Dict[str, str]:
        """Returns a dictionary of all known tasks and their statuses."""
        return {name: task.status for name, task in self.active_tasks.items()}

    async def shutdown(self, timeout: Optional[float] = 5.0):
        """
        Attempts to gracefully cancel all active, managed tasks.
        Waits for them to complete cancellation up to a timeout.
        """
        logger.info("AsyncUIManager: Shutting down. Cancelling active managed tasks...")
        active_task_futures = []
        for task_name, managed_task in list(self.active_tasks.items()): # Iterate on a copy
            if not managed_task.done():
                logger.debug(f"Requesting cancellation for '{task_name}' during shutdown.")
                managed_task.cancel()
                active_task_futures.append(managed_task.get_future())
            else: # Remove already completed tasks
                 del self.active_tasks[task_name]


        if active_task_futures:
            # Wait for cancellations to be processed or timeout
            # Using asyncio.shield might be relevant if the shutdown itself should not be cancelled.
            # For now, a simple gather with return_exceptions=True.
            results = await asyncio.gather(*active_task_futures, return_exceptions=True)
            for i, res in enumerate(results):
                task_name = active_task_futures[i]._asyncio_future_blocking_shield_asyncio_future_obj_task.get_name() if hasattr(active_task_futures[i], '_asyncio_future_blocking_shield_asyncio_future_obj_task') else f"Task {i}" # Heuristic to get name
                if isinstance(res, asyncio.CancelledError):
                    logger.info(f"Task '{task_name}' confirmed cancelled during shutdown.")
                elif isinstance(res, Exception):
                    logger.error(f"Task '{task_name}' raised an error during shutdown cancellation: {res}", exc_info=res)
                else:
                    logger.info(f"Task '{task_name}' completed with result during shutdown: {res}")
        
        self.active_tasks.clear() # Clear all tasks after attempting cancellation
        logger.info("AsyncUIManager: Shutdown complete. All managed tasks processed.")
