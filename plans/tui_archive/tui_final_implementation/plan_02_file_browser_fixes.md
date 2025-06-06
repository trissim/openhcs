# plan_02a_file_browser_complexity_reduction.md
## Component: File Browser Button Handler Complexity Reduction

### Objective
Fix the DNA-identified high-complexity issues in file browser button handlers that are causing complete button failure. Reduce complexity from 16 and 11 to manageable levels through method decomposition.

### Plan
1. **Analyze DNA-Identified Root Causes**
   - Examine `openhcs/tui/components/file_browser.py::_handle_ok()` (complexity 16)
   - Examine `openhcs/tui/components/file_browser.py::_handle_item_activated()` (complexity 11)
   - Map complexity sources: nested conditionals, error handling, state management
   - Identify specific code paths causing button non-responsiveness

2. **Decompose _handle_ok() Method (Complexity 16 → <5)**
   - Extract path validation logic into `_validate_selected_path()`
   - Extract callback execution logic into `_execute_selection_callback()`
   - Extract dialog closing logic into `_close_dialog_with_result()`
   - Extract error handling into `_handle_selection_error()`
   - Ensure each method has single responsibility

3. **Decompose _handle_item_activated() Method (Complexity 11 → <5)**
   - Extract directory navigation logic into `_navigate_to_directory()`
   - Extract file selection logic into `_select_file_item()`
   - Extract UI update logic into `_update_selection_display()`
   - Remove nested conditional chains

4. **Implement Proper Async/Await Patterns**
   - Ensure all button handlers are properly async
   - Fix callback execution to use proper async patterns
   - Eliminate blocking operations in UI thread
   - Add proper error propagation for async operations

5. **Validate Button Responsiveness**
   - Test OK button actually responds to clicks
   - Test Cancel/Escape buttons close dialogs
   - Verify keyboard navigation works
   - Ensure mouse clicks register properly

### Findings
**DNA Analysis Results:**
- `PJLJ:handleok:16` = `openhcs/tui/components/file_browser.py::_handle_ok()` with complexity 16
- `PJLJ:handleitemactiv:11` = `openhcs/tui/components/file_browser.py::_handle_item_activated()` with complexity 11
- These are the exact functions causing button failures in dialogs
- High complexity suggests nested conditionals, error handling, or complex state management

**Current Issues:**
- Add plate dialog opens but buttons don't respond
- Dialog close buttons don't work
- File selection doesn't trigger proper callbacks
- Escape key doesn't dismiss dialogs

**Root Cause Hypothesis:**
- Overly complex button handlers with nested logic
- Improper async/await patterns in event handling
- Dialog lifecycle management issues
- Missing or broken callback execution

### Implementation Draft
*Implementation will be added after smell loop approval*
