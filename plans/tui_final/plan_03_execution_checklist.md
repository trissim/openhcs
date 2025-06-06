# plan_03_execution_checklist.md
## Component: File Browser Fix Execution Checklist

### Objective
Retard-proof execution checklist that any LLM can follow to fix file browser selection. Each step has exact file paths, line numbers, and verification commands.

### Plan
Execute steps in exact order. Each step is atomic and verifiable.

### Implementation Draft

#### EXECUTION CHECKLIST

**STEP 1: Remove InteractiveListItem import**
```bash
# File: openhcs/tui/editors/file_browser.py
# Action: DELETE line 27
# Before: from openhcs.tui.components.interactive_list_item import InteractiveListItem
# After: (line deleted)
```

**STEP 2: Add required imports**
```bash
# File: openhcs/tui/editors/file_browser.py
# Action: ADD after line 25 (after ScrollablePane import)
# Add: from prompt_toolkit.layout.controls import FormattedTextControl

# Action: ADD after line 13 (after MouseEventType import if exists, else after enum import)
# Add: from prompt_toolkit.mouse_events import MouseEventType
```

**STEP 3: Replace _build_item_list method**
```bash
# File: openhcs/tui/editors/file_browser.py
# Action: REPLACE lines 424-467 (entire _build_item_list method)
# Verification: Method starts with "def _build_item_list(self) -> Container:"
```

**STEP 4: Clean debug pollution**
```bash
# File: openhcs/tui/editors/file_browser.py
# Action: DELETE these exact lines in _on_activate method:
# - logger.info(f"FileBrowser: _on_activate - invalid focus")
# - logger.info(f"FileBrowser: _on_activate - item={item.name}, is_dir={item.is_dir}, allow_multiple={self.allow_multiple}")
# - logger.info("FileBrowser: Multi-selection mode - toggling selection")
# - logger.info("FileBrowser: Single-selection mode - navigating to directory")
# - logger.info("FileBrowser: Single-selection mode - handling selection")
```

**VERIFICATION COMMANDS**
```bash
# Check imports are correct
grep -n "FormattedTextControl" openhcs/tui/editors/file_browser.py
grep -n "MouseEventType" openhcs/tui/editors/file_browser.py

# Check InteractiveListItem removed
grep -n "InteractiveListItem" openhcs/tui/editors/file_browser.py
# Should return: (no results)

# Check debug pollution removed
grep -n "logger.info" openhcs/tui/editors/file_browser.py
# Should return: (no results in _on_activate method)

# Test TUI launches
python -m openhcs.tui
```

**MATHEMATICAL GUARANTEE**
- Mouse routing: HSplit → Window → FormattedTextControl → mouse_handler
- Same pattern as working FramedButton (openhcs/tui/components/framed_button.py:49-55)
- No duck typing = no mouse event routing failure
- Selection state: is_selected = item.path in self.selected_paths (mathematically correct)

**ROLLBACK PLAN**
If fix fails:
```bash
git checkout openhcs/tui/editors/file_browser.py
```

**SUCCESS CRITERIA**
1. TUI launches without errors
2. File browser opens when clicking "Add" in PlateManager
3. Clicking folders shows visual feedback (reverse video)
4. Selected folders have [x] prefix and reverse styling
5. No debug messages in logs
