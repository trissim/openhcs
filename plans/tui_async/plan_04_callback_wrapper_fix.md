# plan_04_callback_wrapper_fix.md
## Component: Async Callback Wrapper Elimination

### Objective
Eliminate the fire-and-forget async callback wrappers that silently swallow exceptions and create unawaited coroutines. Replace with proper error handling and task management.

### Plan
1. **Replace _run_callback patterns (20 minutes)**
   - 3 exact instances of broken callback wrappers
   - Add proper exception handling to callbacks
   - Use unified task manager for async callbacks

2. **Fix _wrap_handler patterns (15 minutes)**
   - 1 instance in list_manager.py
   - Replace with proper async handling
   - Maintain same interface, fix implementation

3. **Fix _run_async patterns (10 minutes)**
   - 1 instance in config_editor.py
   - Replace with unified task manager
   - Add proper error context

### Findings

#### Current Callback Wrapper Disasters

**File: `openhcs/tui/components/interactive_list_item.py`**
**Lines 142-147: FIRE-AND-FORGET NIGHTMARE**
```python
def _run_callback(self, callback: Callable, *args):
    """Run callback with proper async handling."""
    if asyncio.iscoroutinefunction(callback):
        get_app().create_background_task(callback(*args))  # ← SILENT FAILURE
    else:
        callback(*args)  # ← NO ERROR HANDLING
```

**File: `openhcs/tui/components/list_manager.py`**
**Lines 496-503: WRAPPER HELL**
```python
def _wrap_handler(self, handler: Callable) -> Callable:
    """Wrap handler for async support."""
    def wrapped():
        if asyncio.iscoroutinefunction(handler):
            get_app().create_background_task(handler())  # ← SILENT FAILURE
        else:
            handler()  # ← NO ERROR HANDLING
    return wrapped
```

**File: `openhcs/tui/components/config_editor.py`**
**Lines 399-401: CENTRALIZED DISASTER**
```python
def _run_async(self, coro: Coroutine) -> None:
    """Centralized async task dispatch."""
    get_app().create_background_task(coro)  # ← SILENT FAILURE
```

**The Problem:** All these patterns launch coroutines into the void with zero error handling. When callbacks fail, they fail silently and you never know why your UI is broken.

### Implementation Draft

#### Step 1: Create Proper Callback Manager

**File: `openhcs/tui/utils/callback_manager.py`**
```python
"""
Proper callback management for OpenHCS TUI.

Replaces fire-and-forget callback patterns with proper error handling.
"""
import asyncio
import logging
from typing import Callable, Any, Optional
from openhcs.tui.utils.unified_task_manager import get_task_manager

logger = logging.getLogger(__name__)


class CallbackManager:
    """
    Manages UI callbacks with proper async handling and error reporting.
    
    Eliminates silent failures from fire-and-forget callback patterns.
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
    
    def run_callback(self, callback: Callable, *args, **kwargs) -> None:
        """
        Run a callback with proper error handling.
        
        Args:
            callback: The callback function to run
            *args: Arguments to pass to callback
            **kwargs: Keyword arguments to pass to callback
        """
        if asyncio.iscoroutinefunction(callback):
            # Async callback - use task manager with error handling
            async def wrapped_callback():
                try:
                    return await callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Async callback {callback.__name__} in {self.component_name} failed: {e}", exc_info=True)
                    # Show error dialog instead of silent failure
                    from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
                    show_error_sync_safe(e, f"{self.component_name} callback {callback.__name__}")
            
            get_task_manager().fire_and_forget(
                wrapped_callback(),
                f"{self.component_name}_{callback.__name__}"
            )
        else:
            # Sync callback - run directly with error handling
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Sync callback {callback.__name__} in {self.component_name} failed: {e}", exc_info=True)
                from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
                show_error_sync_safe(e, f"{self.component_name} callback {callback.__name__}")
    
    def wrap_handler(self, handler: Callable) -> Callable:
        """
        Wrap a handler function with proper error handling.
        
        Args:
            handler: The handler function to wrap
            
        Returns:
            Wrapped handler function
        """
        def wrapped():
            self.run_callback(handler)
        return wrapped
    
    def run_async_task(self, coro, task_name: Optional[str] = None) -> None:
        """
        Run an async task with proper error handling.
        
        Args:
            coro: The coroutine to run
            task_name: Optional name for the task
        """
        if task_name is None:
            task_name = f"{self.component_name}_async_task"
        
        async def wrapped_task():
            try:
                return await coro
            except Exception as e:
                logger.error(f"Async task {task_name} in {self.component_name} failed: {e}", exc_info=True)
                from openhcs.tui.utils.safe_error_handler import show_error_sync_safe
                show_error_sync_safe(e, f"{self.component_name} task {task_name}")
        
        get_task_manager().fire_and_forget(wrapped_task(), task_name)


def create_callback_manager(component_name: str) -> CallbackManager:
    """
    Create a callback manager for a component.
    
    Args:
        component_name: Name of the component for error reporting
        
    Returns:
        CallbackManager instance
    """
    return CallbackManager(component_name)
```

#### Step 2: Fix InteractiveListItem

**File: `openhcs/tui/components/interactive_list_item.py`**
**Add import at top:**
```python
from openhcs.tui.utils.callback_manager import create_callback_manager
```

**In __init__ method, add:**
```python
        # Create callback manager for proper error handling
        self.callback_manager = create_callback_manager("InteractiveListItem")
```

**Lines 142-147: COMPLETE REPLACEMENT**

**REMOVE:**
```python
    def _run_callback(self, callback: Callable, *args):
        """Run callback with proper async handling."""
        if asyncio.iscoroutinefunction(callback):
            get_app().create_background_task(callback(*args))
        else:
            callback(*args)
```

**REPLACE WITH:**
```python
    def _run_callback(self, callback: Callable, *args):
        """Run callback with proper error handling."""
        self.callback_manager.run_callback(callback, *args)
```

#### Step 3: Fix ListManager

**File: `openhcs/tui/components/list_manager.py`**
**Add import at top:**
```python
from openhcs.tui.utils.callback_manager import create_callback_manager
```

**In __init__ method, add:**
```python
        # Create callback manager for proper error handling
        self.callback_manager = create_callback_manager("ListManager")
```

**Lines 496-503: COMPLETE REPLACEMENT**

**REMOVE:**
```python
    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap handler for async support."""
        def wrapped():
            if asyncio.iscoroutinefunction(handler):
                get_app().create_background_task(handler())
            else:
                handler()
        return wrapped
```

**REPLACE WITH:**
```python
    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap handler with proper error handling."""
        return self.callback_manager.wrap_handler(handler)
```

#### Step 4: Fix ConfigEditor

**File: `openhcs/tui/components/config_editor.py`**
**Add import at top:**
```python
from openhcs.tui.utils.callback_manager import create_callback_manager
```

**In __init__ method, add:**
```python
        # Create callback manager for proper error handling
        self.callback_manager = create_callback_manager("ConfigEditor")
```

**Lines 399-401: COMPLETE REPLACEMENT**

**REMOVE:**
```python
    def _run_async(self, coro: Coroutine) -> None:
        """Centralized async task dispatch."""
        get_app().create_background_task(coro)
```

**REPLACE WITH:**
```python
    def _run_async(self, coro: Coroutine) -> None:
        """Centralized async task dispatch with error handling."""
        self.callback_manager.run_async_task(coro, "config_async")
```

#### Step 5: Fix Parameter Editor Background Task

**File: `openhcs/tui/components/parameter_editor.py`**
**Add import at top:**
```python
from openhcs.tui.utils.callback_manager import create_callback_manager
```

**In __init__ method, add:**
```python
        # Create callback manager for proper error handling
        self.callback_manager = create_callback_manager("ParameterEditor")
```

**Lines 124-129: REPLACE BACKGROUND TASK CREATION**

**FIND:**
```python
        # Set accept handler
        def accept_handler(buffer):
            if self.on_parameter_change:
                get_app().create_background_task(
                    self.on_parameter_change(name, buffer.text, self.func_index)
                )
            return True
```

**REPLACE WITH:**
```python
        # Set accept handler
        def accept_handler(buffer):
            if self.on_parameter_change:
                self.callback_manager.run_async_task(
                    self.on_parameter_change(name, buffer.text, self.func_index),
                    f"param_change_{name}"
                )
            return True
```

#### Step 6: Update Import Statements

**Files to update imports:**
- `openhcs/tui/components/interactive_list_item.py`
- `openhcs/tui/components/list_manager.py`
- `openhcs/tui/components/config_editor.py`
- `openhcs/tui/components/parameter_editor.py`

**Remove these imports:**
```python
from prompt_toolkit.application import get_app
```

**Add these imports:**
```python
from openhcs.tui.utils.callback_manager import create_callback_manager
```

### Verification Steps

1. **Test Callback Error Handling:**
   - Trigger an exception in an async callback
   - Verify error dialog appears instead of silent failure
   - Check that error is logged with proper context

2. **Test Sync Callback Errors:**
   - Trigger an exception in a sync callback
   - Verify error dialog appears
   - Check that component continues functioning

3. **Test Task Manager Integration:**
   - Verify callbacks use unified task manager
   - Check that tasks are properly tracked and cleaned up
   - Confirm no RuntimeWarnings from unawaited coroutines

4. **Test UI Functionality:**
   - Verify all interactive elements still work
   - Check that async operations complete properly
   - Confirm no regression in user experience
