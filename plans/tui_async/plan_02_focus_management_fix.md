# plan_02_focus_management_fix.md
## Component: Focus Management Race Condition Fix

### Objective
Fix the broken focus management pattern in dialog_helpers.py that creates RuntimeWarnings by cancelling tasks without awaiting them. This is the most critical async bug causing visible warnings.

### Plan
1. **Fix focus task cancellation pattern (20 minutes)**
   - Replace broken asyncio.create_task() + cancel() pattern
   - Implement proper task cancellation with await
   - Eliminate RuntimeWarning: coroutine was never awaited

2. **Create proper focus management utility (15 minutes)**
   - Centralized focus management with proper lifecycle
   - Exception-safe focus operations
   - Timeout handling for focus operations

3. **Replace all focus-related asyncio.create_task calls (10 minutes)**
   - 3 exact instances identified
   - Surgical replacement with fixed pattern

### Findings

#### Current Focus Management Disaster

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 487-495: THE BROKEN PATTERN**
```python
# Start the focus task
focus_task = asyncio.create_task(set_focus_after_delay())

# Show the dialog
await app_state.show_dialog(dialog, result_future=future)

# Cancel focus task if still running
if not focus_task.done():
    focus_task.cancel()  # ← BROKEN: Not awaited, causes RuntimeWarning
```

**This is mathematically incorrect.** When you cancel a task, you MUST await it to handle the CancelledError. The current pattern creates unawaited coroutines.

#### Other Focus-Related Issues
- `openhcs/tui/utils/dialog_helpers.py:564` - focus_text_area function
- `openhcs/tui/layout/canonical_layout.py:389` - Pipeline editor async creation

### Implementation Draft

#### Step 1: Create Proper Focus Management Utility

**File: `openhcs/tui/utils/focus_manager.py`**
```python
"""
Proper focus management for OpenHCS TUI.

Replaces the broken focus task pattern with mathematically correct implementation.
"""
import asyncio
import logging
from typing import Optional, Any
from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Window

logger = logging.getLogger(__name__)


class FocusManager:
    """
    Manages focus operations with proper async lifecycle.
    
    Eliminates RuntimeWarnings from improper task cancellation.
    """
    
    def __init__(self):
        self._active_focus_task: Optional[asyncio.Task] = None
    
    async def set_focus_after_delay(self, target: Any, delay: float = 0.1) -> bool:
        """
        Set focus to target after a delay, with proper cancellation handling.
        
        Args:
            target: The UI element to focus
            delay: Delay in seconds before focusing
            
        Returns:
            True if focus was set successfully, False if cancelled or failed
        """
        try:
            await asyncio.sleep(delay)
            get_app().layout.focus(target)
            logger.debug(f"Focus set to: {target}")
            return True
        except asyncio.CancelledError:
            logger.debug("Focus operation was cancelled")
            return False
        except Exception as e:
            logger.warning(f"Failed to set focus: {e}")
            return False
    
    async def managed_focus_during_dialog(self, target: Any, dialog_coro, delay: float = 0.1):
        """
        Manage focus during dialog display with proper task lifecycle.
        
        Args:
            target: The UI element to focus after dialog is shown
            dialog_coro: The dialog coroutine to await
            delay: Delay before focusing
            
        Returns:
            Result of the dialog coroutine
        """
        focus_task = None
        try:
            # Start focus task
            focus_task = asyncio.create_task(
                self.set_focus_after_delay(target, delay),
                name="managed_focus"
            )
            
            # Run dialog
            result = await dialog_coro
            
            return result
            
        finally:
            # Properly cancel and await focus task
            if focus_task and not focus_task.done():
                focus_task.cancel()
                try:
                    await focus_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
    
    def cancel_active_focus(self) -> None:
        """Cancel any active focus operation."""
        if self._active_focus_task and not self._active_focus_task.done():
            self._active_focus_task.cancel()
            self._active_focus_task = None


# Global focus manager instance
_global_focus_manager: Optional[FocusManager] = None


def get_focus_manager() -> FocusManager:
    """Get the global focus manager instance."""
    global _global_focus_manager
    if _global_focus_manager is None:
        _global_focus_manager = FocusManager()
    return _global_focus_manager


async def focus_after_delay(target: Any, delay: float = 0.1) -> bool:
    """
    Convenience function for setting focus after delay.
    
    Args:
        target: The UI element to focus
        delay: Delay in seconds
        
    Returns:
        True if successful, False if failed
    """
    return await get_focus_manager().set_focus_after_delay(target, delay)
```

#### Step 2: Fix Broken Dialog Focus Pattern

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 477-495: COMPLETE REPLACEMENT**

**REMOVE LINES 477-495:**
```python
        # Create a task to set focus after dialog is shown
        async def set_focus_after_delay():
            await asyncio.sleep(0.1)  # Small delay to ensure dialog is rendered
            try:
                get_app().layout.focus(file_browser.item_list_control)
                logger.info("🎯 Focus set to file browser")
            except Exception as e:
                logger.warning(f"Could not set focus to file browser: {e}")

        # Start the focus task
        focus_task = asyncio.create_task(set_focus_after_delay())

        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)

        # Cancel focus task if still running
        if not focus_task.done():
            focus_task.cancel()
```

**REPLACE WITH:**
```python
        # Use proper focus management
        from openhcs.tui.utils.focus_manager import get_focus_manager
        
        # Show dialog with managed focus
        await get_focus_manager().managed_focus_during_dialog(
            target=file_browser.item_list_control,
            dialog_coro=app_state.show_dialog(dialog, result_future=future),
            delay=0.1
        )
```

#### Step 3: Fix focus_text_area Function

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Lines 554-565: REPLACE ENTIRE FUNCTION**

**REMOVE:**
```python
async def focus_text_area(text_area: TextArea) -> None:
    """
    Focus a text area after a short delay.

    Args:
        text_area: The text area to focus

    Returns:
        None
    """
    await asyncio.sleep(0.1)  # Short delay to ensure dialog is rendered
    get_app().layout.focus(text_area)
```

**REPLACE WITH:**
```python
async def focus_text_area(text_area: TextArea) -> bool:
    """
    Focus a text area after a short delay.

    Args:
        text_area: The text area to focus

    Returns:
        True if successful, False if failed
    """
    from openhcs.tui.utils.focus_manager import focus_after_delay
    return await focus_after_delay(text_area, 0.1)
```

#### Step 4: Fix Pipeline Editor Async Creation

**File: `openhcs/tui/layout/canonical_layout.py`**
**Lines 377-389: REPLACE ASYNC TASK CREATION**

**REMOVE:**
```python
        async def create_pipeline_editor():
            from prompt_toolkit.widgets import Label, Frame
            from prompt_toolkit.layout import HSplit
            from prompt_toolkit.application import get_app

            self.pipeline_editor = await PipelineEditorPane.create(self.state, self.context)
            # Use the pipeline editor container directly - it already has proper title and structure
            self._pipeline_editor_container = self.pipeline_editor.container

            get_app().invalidate()
            logger.info("PipelineEditorPane: Async initialization complete")

        asyncio.create_task(create_pipeline_editor())
```

**REPLACE WITH:**
```python
        async def create_pipeline_editor():
            from prompt_toolkit.widgets import Label, Frame
            from prompt_toolkit.layout import HSplit
            from prompt_toolkit.application import get_app

            self.pipeline_editor = await PipelineEditorPane.create(self.state, self.context)
            # Use the pipeline editor container directly - it already has proper title and structure
            self._pipeline_editor_container = self.pipeline_editor.container

            get_app().invalidate()
            logger.info("PipelineEditorPane: Async initialization complete")

        # Use unified task manager instead of raw asyncio.create_task
        from openhcs.tui.utils.unified_task_manager import get_task_manager
        get_task_manager().fire_and_forget(create_pipeline_editor(), "pipeline_editor_init")
```

#### Step 5: Add Import Statements

**File: `openhcs/tui/utils/dialog_helpers.py`**
**Add at top of file:**
```python
from openhcs.tui.utils.focus_manager import get_focus_manager, focus_after_delay
```

**File: `openhcs/tui/layout/canonical_layout.py`**
**Add at top of file:**
```python
from openhcs.tui.utils.unified_task_manager import get_task_manager
```

### Verification Steps

1. **Check for RuntimeWarnings:**
   - Run TUI and open dialogs
   - Verify no "coroutine was never awaited" warnings
   - Focus should work properly in file browser dialogs

2. **Test Focus Behavior:**
   - File browser dialog should focus on file list after opening
   - Text areas should focus properly after delay
   - No hanging tasks after dialog closure

3. **Verify Task Cleanup:**
   - Check task manager active count during dialog operations
   - Ensure tasks are properly cleaned up after dialogs close
   - No memory leaks from uncancelled tasks
