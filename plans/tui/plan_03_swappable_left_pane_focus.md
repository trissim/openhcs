# plan_03_swappable_left_pane_focus.md
## Component: Swappable Left Pane with Focus Management

### Objective
Create a robust container swapping system for the left pane that allows switching between PlateManager and DualStepFuncEditor while maintaining proper focus for mouse handlers. The solution must be mathematically precise and handle all focus edge cases.

### Plan

#### Step 1: Understand Current Architecture Issues
**Current Problem:**
- PlateManager uses DynamicContainer which loses focus on content changes
- No mechanism exists to swap PlateManager with DualStepFuncEditor
- Mouse handlers don't work because FormattedTextControl loses focus

**Root Cause Analysis:**
- DynamicContainer rebuilds content → focus lost → mouse events not received
- No explicit focus restoration after content changes
- Focus applied to wrong container level (ScrollablePane instead of Window)

#### Step 2: Design Swappable Container Architecture
**Container Hierarchy (Mathematical):**
```
VSplit (main_content)
├── DynamicContainer (left_pane_container)  ← SWAP POINT
│   └── Current Pane (PlateManager OR DualStepFuncEditor)
│       └── Frame
│           └── HSplit
│               ├── Button Bar
│               └── DynamicContainer (content_container)
│                   └── ScrollablePane
│                       └── Window ← FOCUS TARGET
│                           └── FormattedTextControl
└── Pipeline Editor (right_pane)
```

**Focus Chain (Validated):**
1. Content changes → `container.children = [new_content]`
2. Focus immediately → `get_app().layout.focus(window)`
3. Invalidate → `get_app().invalidate()`

#### Step 3: Implement Container Swapping Manager
**SwappableLeftPane Class:**
- Manages container.children manipulation
- Tracks current pane state (PlateManager | DualStepFuncEditor)
- Handles focus restoration for both pane types
- Provides clean swap interface

**Focus Management Rules:**
- Always focus the Window containing FormattedTextControl
- Focus immediately after content change (synchronous)
- Store Window references for both pane types
- Restore focus on every swap operation

#### Step 4: Fix DynamicContainer Focus Loss (BULLETPROOF)

**Current Issue:** ListView._on_model_changed() calls invalidate() but loses focus

**ALL Possible Failure Modes:**
1. `scrollable_window` is None → Focus fails silently
2. `scrollable_window` exists but not in layout → Focus throws exception
3. App is not running → `get_app()` fails
4. Layout is being rebuilt → Focus target becomes invalid
5. Multiple rapid model changes → Race conditions
6. Window is not focusable → Focus ignored
7. Window is destroyed during focus → Crash
8. Focus called before Window is added to layout → No effect

**RETARD-PROOF Fix with ALL Edge Cases:**
```python
def _on_model_changed(self):
    # 1. Update cursor position (safe)
    self._ensure_focused_visible()

    # 2. BULLETPROOF focus restoration
    self._restore_focus_bulletproof()

    # 3. Invalidate last
    get_app().invalidate()

def _restore_focus_bulletproof(self):
    """Restore focus with ALL edge cases handled."""
    try:
        # Check 1: App exists and is running
        app = get_app()
        if not app or not hasattr(app, 'layout'):
            return False

        # Check 2: Window exists and is valid
        if not hasattr(self, 'scrollable_window') or not self.scrollable_window:
            return False

        # Check 3: Window is in the layout tree
        if not self._is_window_in_layout(self.scrollable_window, app.layout):
            return False

        # Check 4: Window is focusable
        if not getattr(self.scrollable_window, 'content', None):
            return False

        # Check 5: FormattedTextControl exists and is focusable
        control = self.scrollable_window.content
        if not hasattr(control, 'focusable') or not control.focusable:
            return False

        # SAFE focus call with exception handling
        app.layout.focus(self.scrollable_window)
        return True

    except Exception as e:
        # Log error but don't crash
        print(f"Focus restoration failed: {e}")
        return False

def _is_window_in_layout(self, window, layout):
    """Check if window is actually in the layout tree."""
    try:
        # Traverse layout tree to find window
        def find_window(container):
            if container == window:
                return True
            if hasattr(container, 'children'):
                for child in container.children:
                    if find_window(child):
                        return True
            if hasattr(container, 'content') and container.content == window:
                return True
            return False

        return find_window(layout.container)
    except:
        return False
```

**Window Reference Storage (BULLETPROOF):**
```python
def _create_current_list(self):
    # ... existing code ...

    # CRITICAL: Store Window reference for focus management
    self.scrollable_window = Window(content=self.list_control)

    # VALIDATION: Ensure Window is properly configured
    assert self.scrollable_window.content == self.list_control
    assert hasattr(self.list_control, 'focusable')
    assert self.list_control.focusable

    # Return ScrollablePane but keep Window reference
    self.scrollable_pane = ScrollablePane(
        self.scrollable_window,
        height=Dimension(weight=1),
        show_scrollbar=True,
        display_arrows=True
    )
    return self.scrollable_pane
```

#### Step 5: Implement Pane Swap Interface
**TUIState Integration:**
- Add `current_left_pane: str` field ("plate_manager" | "dual_editor")
- Add `swap_left_pane(new_pane: str)` method
- Trigger swaps from button handlers

**Swap Method (BULLETPROOF - ALL Edge Cases):**

**ALL Possible Failure Modes:**
1. Invalid pane name → Crash
2. Pane objects don't exist → AttributeError
3. Container is None → Focus fails
4. Focus target is None → Focus fails
5. Focus target not in new layout → Focus throws exception
6. App not running during swap → Crash
7. Layout being rebuilt during swap → Race condition
8. Multiple rapid swaps → State corruption
9. Pane not properly initialized → Missing methods
10. Container children is read-only → Assignment fails

```python
def swap_left_pane(self, new_pane: str) -> bool:
    """Swap left pane with BULLETPROOF error handling.

    Returns:
        bool: True if swap successful, False if failed
    """
    try:
        # VALIDATION 1: Input validation
        if not isinstance(new_pane, str):
            raise ValueError(f"new_pane must be string, got {type(new_pane)}")
        if new_pane not in ["plate_manager", "dual_editor"]:
            raise ValueError(f"Invalid pane: {new_pane}")

        # VALIDATION 2: Current state check
        if not hasattr(self, 'current_left_pane'):
            self.current_left_pane = "plate_manager"  # Default
        if self.current_left_pane == new_pane:
            return True  # Already showing requested pane

        # VALIDATION 3: App state check
        app = get_app()
        if not app or not hasattr(app, 'layout'):
            raise RuntimeError("App not running or no layout")

        # VALIDATION 4: Container existence check
        if not hasattr(self, 'left_pane_container') or not self.left_pane_container:
            raise RuntimeError("Left pane container not initialized")

        # VALIDATION 5: Pane objects exist and are initialized
        if new_pane == "plate_manager":
            if not hasattr(self, 'plate_manager') or not self.plate_manager:
                raise RuntimeError("PlateManager not initialized")
            pane_obj = self.plate_manager
        else:  # dual_editor
            if not hasattr(self, 'dual_editor') or not self.dual_editor:
                raise RuntimeError("DualEditor not initialized")
            pane_obj = self.dual_editor

        # VALIDATION 6: Pane has required methods
        if not hasattr(pane_obj, 'container'):
            raise RuntimeError(f"{new_pane} missing container property")
        if not hasattr(pane_obj, 'get_focus_window'):
            raise RuntimeError(f"{new_pane} missing get_focus_window method")

        # VALIDATION 7: Get container and focus target
        new_container = pane_obj.container
        if not new_container:
            raise RuntimeError(f"{new_pane} container is None")

        focus_target = pane_obj.get_focus_window()
        if not focus_target:
            raise RuntimeError(f"{new_pane} focus target is None")

        # VALIDATION 8: Container children is writable
        if not hasattr(self.left_pane_container, 'children'):
            raise RuntimeError("Left pane container has no children attribute")

        # ATOMIC SWAP OPERATION
        # Step 1: Update container children
        old_children = self.left_pane_container.children[:]  # Backup
        try:
            self.left_pane_container.children = [new_container]
        except Exception as e:
            # Restore old children if assignment fails
            self.left_pane_container.children = old_children
            raise RuntimeError(f"Failed to update container children: {e}")

        # Step 2: Focus immediately (with validation)
        focus_success = self._focus_target_bulletproof(focus_target)
        if not focus_success:
            # Swap succeeded but focus failed - log warning but continue
            print(f"Warning: Pane swap succeeded but focus failed for {new_pane}")

        # Step 3: Update state
        self.current_left_pane = new_pane

        # Step 4: Invalidate
        app.invalidate()

        return True

    except Exception as e:
        print(f"Pane swap failed: {e}")
        return False

def _focus_target_bulletproof(self, target) -> bool:
    """Focus target with ALL edge cases handled."""
    try:
        app = get_app()
        if not app or not hasattr(app, 'layout'):
            return False

        # Check if target is in layout tree
        if not self._is_window_in_layout(target, app.layout):
            return False

        # Attempt focus
        app.layout.focus(target)
        return True

    except Exception as e:
        print(f"Focus failed: {e}")
        return False
```

### Findings

#### Research Validation
Based on GitHub issues #677 and #1324:
- **Timing:** Focus immediately after content change, not after invalidate()
- **Target:** Focus Window containing FormattedTextControl, not parent containers
- **Method:** Use `container.children` manipulation for swapping
- **Persistence:** Focus must be manually restored each time

#### Container Focus Rules
From prompt-toolkit documentation:
- `get_app().layout.focus()` accepts Window, Buffer, UIControl
- Focus propagates down container tree
- Modal containers break focus inheritance
- Windows are "leaves in the tree structure that represent the UI"

#### Critical Implementation Details
1. **Store Window References:** Both panes must expose their focusable Window
2. **Synchronous Focus:** No delays or async timing needed
3. **Focus Deepest Element:** Target the Window, not ScrollablePane
4. **State Tracking:** Track current pane for proper restoration

#### MANDATORY Interface Requirements (BULLETPROOF)

**Both PlateManager and DualStepFuncEditor MUST implement:**

```python
class SwappablePaneInterface:
    """MANDATORY interface for swappable panes."""

    @property
    def container(self) -> Container:
        """Return the root container for this pane.

        MUST NOT return None.
        MUST be a valid prompt-toolkit Container.
        """
        raise NotImplementedError

    def get_focus_window(self) -> Window:
        """Return the Window that should receive focus.

        MUST NOT return None.
        MUST return a Window containing a focusable UIControl.
        MUST be the deepest focusable element in the container tree.
        """
        raise NotImplementedError

    def on_focus_gained(self) -> None:
        """Called when this pane gains focus.

        Optional: Implement if pane needs focus notifications.
        """
        pass

    def on_focus_lost(self) -> None:
        """Called when this pane loses focus.

        Optional: Implement if pane needs focus notifications.
        """
        pass
```

**PlateManager Implementation Requirements:**
```python
class PlateManager(SwappablePaneInterface):
    def __init__(self, ...):
        # ... existing init ...
        self._focus_window = None  # Will be set in list creation

    @property
    def container(self) -> Container:
        """Return the PlateManager Frame container."""
        if not hasattr(self, '_container') or not self._container:
            raise RuntimeError("PlateManager container not initialized")
        return self._container

    def get_focus_window(self) -> Window:
        """Return the ListView's scrollable_window."""
        if not hasattr(self, 'list_manager') or not self.list_manager:
            raise RuntimeError("PlateManager list_manager not initialized")
        if not hasattr(self.list_manager, 'scrollable_window'):
            raise RuntimeError("ListView scrollable_window not initialized")
        if not self.list_manager.scrollable_window:
            raise RuntimeError("ListView scrollable_window is None")
        return self.list_manager.scrollable_window
```

**DualStepFuncEditor Implementation Requirements:**
```python
class DualStepFuncEditor(SwappablePaneInterface):
    def __init__(self, ...):
        # ... existing init ...
        self._focus_window = None  # Will be set during creation

    @property
    def container(self) -> Container:
        """Return the DualStepFuncEditor root container."""
        if not hasattr(self, '_container') or not self._container:
            raise RuntimeError("DualStepFuncEditor container not initialized")
        return self._container

    def get_focus_window(self) -> Window:
        """Return the primary focusable Window."""
        # Implementation depends on DualStepFuncEditor internal structure
        # MUST return the Window containing the primary interactive element
        if not hasattr(self, '_primary_window') or not self._primary_window:
            raise RuntimeError("DualStepFuncEditor primary window not initialized")
        return self._primary_window
```

**VALIDATION Requirements:**
- Both implementations MUST pass validation tests
- get_focus_window() MUST return a Window with focusable=True UIControl
- container property MUST return a Container that contains the focus Window
- All methods MUST handle edge cases and never return None unexpectedly

### Implementation Draft

*Implementation will be added after plan approval via smell loop*
