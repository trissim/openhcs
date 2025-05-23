# Plan 03: TUI Asynchronous Orchestration and Typing

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package utilizes `asyncio` for non-blocking operations, but the current implementation, as highlighted by the `reports/code_analysis/tui_comprehensive.md/async_patterns_tui.md` report, shows inconsistencies:
    *   Numerous `async def` functions lack explicit return type annotations (e.g., in `pipeline_editor.py`, `plate_manager_core.py`, `function_pattern_editor.py`). This reduces code clarity and hinders static analysis.
    *   Several instances of unawaited coroutines were detected (e.g., `pipeline_editor.py` calling `execute` from `setup`, `file_browser.py` in `main_async`). These are potential bugs, as the coroutine might not run to completion or its result/exceptions might be lost.
    *   Asynchronous task management (creation, tracking, cancellation, error handling) is decentralized, often relying on `get_app().create_background_task(...)` scattered throughout various UI components and command handlers. This makes it difficult to manage the lifecycle of async operations globally, handle errors consistently, or implement features like task cancellation.

**Goal**: To standardize asynchronous operations within the TUI by:
    1.  Introducing a centralized `AsyncUIManager` (or `TUITaskManager`) responsible for managing the lifecycle of UI-initiated asynchronous tasks.
    2.  Ensuring all `async def` functions have explicit return type annotations.
    3.  Systematically reviewing and fixing all identified unawaited coroutine calls.
    4.  Implementing consistent error handling for asynchronous operations, potentially facilitated by the `AsyncUIManager`.

**Architectural Principles**:
*   **Clarity and Explicitness**: Code should clearly state its intentions, including asynchronous behavior and return types.
*   **Robustness**: Asynchronous operations should be managed to prevent lost tasks, unhandled exceptions, and resource leaks.
*   **Centralized Control (for cross-cutting concerns)**: Common aspects of async task management (e.g., error logging, cancellation hooks) should be handled consistently.
*   **Process Calculi Insights**: Treat async tasks as managed processes with defined lifecycles.

## 2. Proposed `AsyncUIManager` (New file: `openhcs.tui.async_manager.py`)

This new class will be responsible for managing UI-related asynchronous tasks.

```python
# openhcs/tui/async_manager.py
import asyncio
import logging
from typing import Callable, Coroutine, Any, Optional, TypeVar, Generic
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ManagedTask(Generic[T]):
    """Represents a task managed by the AsyncUIManager."""
    def __init__(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None):
        self.name = name or getattr(coro, '__name__', 'unnamed_task')
        self._coro = coro
        self._task: Optional[asyncio.Task[T]] = None
        self._future: asyncio.Future[T] = asyncio.Future()

    async def run(self):
        """Runs the coroutine and sets the future's result or exception."""
        if self._task is not None:
            logger.warning(f"Task '{self.name}' is already running or has run.")
            return

        try:
            self._task = asyncio.create_task(self._coro, name=self.name)
            result = await self._task
            self._future.set_result(result)
            logger.debug(f"Managed task '{self.name}' completed successfully.")
        except asyncio.CancelledError:
            self._future.cancel()
            logger.info(f"Managed task '{self.name}' was cancelled.")
            raise # Re-raise CancelledError if needed by caller
        except Exception as e:
            logger.error(f"Error in managed task '{self.name}': {e}", exc_info=True)
            self._future.set_exception(e)
            # Optionally, notify a global error handler via TUIState
            # if get_app_state_notifier_is_available():
            #    app_state = get_app_state()
            #    app_state.notify('async_error', {'name': self.name, 'error': e})
        finally:
            self._task = None # Clear the task reference once done

    def cancel(self):
        """Cancels the underlying asyncio.Task if it's running."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info(f"Cancellation requested for task '{self.name}'.")
        elif self._task and self._task.done():
            logger.debug(f"Task '{self.name}' already done, cannot cancel.")
        else:
            logger.warning(f"No running task to cancel for '{self.name}'.")

    @property
    def future(self) -> asyncio.Future[T]:
        return self._future

    @property
    def task(self) -> Optional[asyncio.Task[T]]:
        return self._task

    def done(self) -> bool:
        return self._future.done()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def result(self) -> T: # Can raise an exception or CancelledError
        return self._future.result()

    def exception(self) -> Optional[BaseException]:
        return self._future.exception()


class AsyncUIManager:
    def __init__(self, state: Optional[Any] = None): # TUIState can be passed for notifications
        self.state = state
        self._active_managed_tasks: Dict[str, ManagedTask[Any]] = {}
        self._task_id_counter = 0

    def _generate_task_id(self, name_hint: Optional[str] = None) -> str:
        self._task_id_counter += 1
        prefix = name_hint or "task"
        return f"{prefix}_{self._task_id_counter}"

    def submit_task(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None) -> ManagedTask[T]:
        """
        Submits a coroutine to be run as a managed task.
        The task is created but not immediately awaited by this method.
        The caller can await managed_task.future or managed_task.run() if needed.
        Typically, this is for fire-and-forget tasks that update UI upon completion.
        """
        task_name = name or getattr(coro, '__name__', 'unnamed_coro')
        managed_task = ManagedTask(coro, name=task_name)
        
        # Run the managed task in the background.
        # asyncio.create_task is used here to schedule ManagedTask.run()
        # This allows ManagedTask.run() to handle its own lifecycle and future.
        asyncio.create_task(managed_task.run(), name=f"wrapper_for_{managed_task.name}")
        
        # Store if tracking is needed, though ManagedTask itself holds the future.
        # For now, let's not store it here unless specific tracking/cancellation features are added to AsyncUIManager.
        # task_id = self._generate_task_id(managed_task.name)
        # self._active_managed_tasks[task_id] = managed_task
        # managed_task.future.add_done_callback(lambda f: self._active_managed_tasks.pop(task_id, None))

        logger.debug(f"Submitted task '{managed_task.name}' to AsyncUIManager.")
        return managed_task

    async def run_task_and_wait(self, coro: Coroutine[Any, Any, T], name: Optional[str] = None) -> T:
        """
        Runs a coroutine as a managed task and awaits its completion, returning its result or raising its exception.
        """
        task_name = name or getattr(coro, '__name__', 'unnamed_coro')
        managed_task = ManagedTask(coro, name=task_name)
        await managed_task.run() # This awaits the internal task and sets the future
        return managed_task.result() # This will return result or raise exception/CancelledError

    # Optional: Decorator for wrapping functions
    def managed_async_task(self, name: Optional[str] = None) -> Callable[..., Coroutine[Any, Any, Any]]:
        def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                task_name = name or func.__name__
                # When used as a decorator, we typically want to run and await.
                return await self.run_task_and_wait(func(*args, **kwargs), name=task_name)
            return wrapper
        return decorator

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any], name: Optional[str] = None) -> None:
        """
        Submits a task to be run in the background. Errors are logged by ManagedTask.
        No direct way to get result/exception from this call.
        """
        self.submit_task(coro, name=name)

    # Add methods for cancelling tasks if needed, e.g., cancel_all_tasks_by_name_prefix()
```

## 3. Refactoring Steps

1.  **Integrate `AsyncUIManager`**:
    *   Instantiate `AsyncUIManager` in `OpenHCSTUI.__init__` and pass it to components that need to run async tasks (or make it accessible via `TUIState` or a global accessor if appropriate for `prompt_toolkit`'s app structure).
    *   Replace direct calls to `get_app().create_background_task(...)` or `asyncio.create_task(...)` with `async_ui_manager.submit_task(coro)` for fire-and-forget tasks, or `await async_ui_manager.run_task_and_wait(coro)` if the result is needed immediately.

2.  **Add Return Type Annotations**:
    *   Systematically review all `async def` functions in the `openhcs.tui` package.
    *   Add explicit return type annotations. For functions that don't return a value, use `-> None`.
    *   **Target files from report**: `pipeline_editor.py`, `file_browser.py`, `tui_launcher.py`, `dual_step_func_editor.py`, `__main__.py`, `menu_bar.py`, `utils.py`, `function_pattern_editor.py`, `plate_manager_core.py`, `tui_architecture.py`, `status_bar.py`, `dialogs/*`, `services/*`, `utils/*`.

3.  **Fix Unawaited Coroutines**:
    *   Review each instance of "Unawaited Coroutines" from the `async_patterns_tui.md` report.
    *   **`pipeline_editor.py`**: In `PipelineEditorPane.setup`, the calls to `execute` for various commands (e.g., `AddStepCommand().execute(...)`) are likely intended to be fire-and-forget. Wrap these in `self.async_ui_manager.fire_and_forget(AddStepCommand().execute(...))`.
    *   **`file_browser.py`**: In `main_async`, the call to `Application(...).run_async()` should likely be `await Application(...).run_async()` if `main_async` is the top-level entry point for this specific test/example. If it's meant to run in background, use `async_ui_manager.fire_and_forget()`.
    *   **`__main__.py`**: In `main`, the call to `OpenHCSTUILauncher(...).run()` should be `await OpenHCSTUILauncher(...).run()` if `run` is async and `main` is the top-level async entry.
    *   **`utils.py` / `dialog_helpers.py`**: In `focus_text_area` (called by `prompt_for_path_dialog`), the call `app.layout.focus(text_area)` is synchronous. The issue might be that `prompt_for_path_dialog` itself is called without `await` if it's an `async` function. Ensure all calls to `async` dialog functions are awaited or managed by `AsyncUIManager`.
    *   **`plate_manager_core.py`**: In `PlateManagerPane._initialize_ui`, similar to `pipeline_editor.py`, command executions should be wrapped with `self.async_ui_manager.fire_and_forget(...)`.
    *   **`tui_architecture.py`**: In `OpenHCSTUI` constructor and methods, calls to command `execute` methods should be wrapped with `self.async_ui_manager.fire_and_forget(...)`.
    *   **`dialogs/plate_dialog_manager.py`**: Calls to `dialog.show()` or similar async dialog methods within `PlateDialogManager` should be `await`ed if their result is used or if they must complete before proceeding. If they are fire-and-forget UI updates, use `async_ui_manager.fire_and_forget()`.

4.  **Consistent Error Handling**:
    *   The `ManagedTask` class provides basic error logging.
    *   Consider adding a global error notification mechanism via `TUIState` within `ManagedTask.run`'s `except Exception` block, so that `StatusBar` or a dedicated error display component can show async task errors to the user.
    *   For tasks where specific error handling is needed by the caller (not just logging), use `await async_ui_manager.run_task_and_wait(coro)` and wrap the call in a `try...except` block.

## 4. Verification

1.  **Static Analysis**:
    *   Re-run `python tools/code_analysis/code_analyzer_cli.py async-patterns openhcs/tui --output-dir reports/code_analysis/tui_comprehensive.md`.
    *   Verify that the number of "Async Functions Without Return Type" is zero or significantly reduced.
    *   Verify that the number of "Unawaited Coroutines" is zero.
2.  **Code Review**:
    *   Ensure all `async def` have return types.
    *   Confirm that `asyncio.create_task` and `get_app().create_background_task` are largely replaced by `AsyncUIManager` methods.
    *   Check for consistent error handling patterns for async tasks.
3.  **Runtime Testing**:
    *   Thoroughly test UI interactions that trigger asynchronous operations.
    *   Verify that operations complete as expected and that UI updates correctly.
    *   Introduce deliberate errors in mocked async operations to test error handling and logging.
    *   Test task cancellation if this feature is implemented in `AsyncUIManager`.

This plan aims to improve the robustness, clarity, and maintainability of asynchronous code within the TUI.