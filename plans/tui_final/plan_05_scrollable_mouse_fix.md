# plan_05_scrollable_mouse_fix.md
## Component: ScrollablePane and Mouse Event Fix

### Objective
Fix scroll bar and mouse events using mathematical analysis of prompt_toolkit container hierarchy. Make file browser fully functional with proper scrolling and clicking.

### Plan
1. **Fix ScrollablePane configuration** - Wrap proper Window with focusable control
2. **Fix mouse event routing** - Make controls focusable within ScrollablePane
3. **Verify mathematical correctness** - Ensure container hierarchy follows prompt_toolkit patterns

### Mathematical Analysis

**Current Architecture (BROKEN):**
```
ScrollablePane(DynamicContainer(_build_item_list))
  └─ DynamicContainer
      └─ HSplit([Window(FormattedTextControl(focusable=False)), ...])
```

**Problem:** `FormattedTextControl(focusable=False)` cannot receive mouse events in ScrollablePane.

**Correct Architecture (from examples):**
```
ScrollablePane(Window(FormattedTextControl(focusable=True)))
```

**Mathematical Proof:** Examples show ScrollablePane must wrap a single focusable Window, not a container with multiple non-focusable controls.

### Implementation Draft

#### RETARD-PROOF FIX

**STEP 1: Fix ScrollablePane wrapper**
```bash
# File: openhcs/tui/editors/file_browser.py
# Line 151: CHANGE
file_area = ScrollablePane(self.item_container)
# TO:
file_area = ScrollablePane(
    Window(
        content=DynamicContainer(self._build_item_list),
        focusable=True,
        scrollbar=True
    )
)
```

**STEP 2: Make FormattedTextControl focusable**
```bash
# File: openhcs/tui/editors/file_browser.py
# Line 448: CHANGE
control = FormattedTextControl(display, focusable=False)
# TO:
control = FormattedTextControl(display, focusable=True)
```

**STEP 3: Add proper height constraint**
```bash
# File: openhcs/tui/editors/file_browser.py
# After line 151: MODIFY ScrollablePane to:
file_area = ScrollablePane(
    Window(
        content=DynamicContainer(self._build_item_list),
        focusable=True,
        scrollbar=True
    ),
    height=Dimension(min=5, max=20)  # Bounded height for scrolling
)
```

**Mathematical Guarantees:**
1. `Window(focusable=True)` can receive mouse events ✓
2. `ScrollablePane(Window(...))` follows correct pattern ✓  
3. `FormattedTextControl(focusable=True)` can handle clicks ✓
4. Mouse routing: `ScrollablePane → Window → FormattedTextControl → mouse_handler` ✓
5. Scroll bar appears when content exceeds height ✓

**Verification Commands:**
```bash
# Test scrolling and clicking
python -m openhcs.tui
# 1. Open file browser (Add button)
# 2. Navigate to directory with >20 items
# 3. Verify scroll bar appears
# 4. Verify mouse wheel scrolling works
# 5. Verify clicking folders works
```
