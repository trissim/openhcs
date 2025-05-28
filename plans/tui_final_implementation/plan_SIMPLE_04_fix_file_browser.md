# plan_SIMPLE_04_command_integration_fixes.md
## Component: Fix Command System Integration and File Browser Issues

### Objective
Implement canonical button functionality as specified in tui_final.md: orchestrator method integration, multi-folder selection for add plate, and fix DNA-identified file browser complexity issues.

### Plan
1. **CRITICAL: Implement Canonical Button Functionality (4 hours)**
   - **Add button**: Multi-folder selection → creates orchestrator with default config → `?` status
   - **Del button**: Removes selected plates from plate manager list
   - **Edit button**: Opens static reflection config editor for selected plates
   - **Init button**: Calls `orchestrator.initialize()` → changes status to `!` (yellow)
   - **Compile button**: Runs pipeline compiler → changes status to `o` (green)
   - **Run button**: Executes plate processing → maintains `o` status while running
   - **Error handling**: Errors display in status bar with OK dialog, return to previous state

2. **Implement Multi-Folder Selection for Add Plate (3 hours)**
   - **Canonical requirement**: "multiple folders may be selected at once" for add plate
   - **File browser enhancement**: Support multi-folder selection mode
   - **UI integration**: Multi-select checkboxes or ctrl+click selection
   - **Orchestrator creation**: Create one orchestrator per selected folder
   - **Status integration**: Each new plate starts with `?` (uninitialized) status

3. **Fix DNA-Identified File Browser Complexity Issues (2 hours)**
   - **Target**: `openhcs/tui/components/file_browser.py::_handle_ok()` (complexity 16)
   - **Target**: `openhcs/tui/components/file_browser.py::_handle_item_activated()` (complexity 11)
   - **Method**: Extract complex logic into separate methods
   - **Goal**: Reduce complexity to <5 for each method

4. **Decompose _handle_ok() Method (1 hour)**
   - Extract multi-selection logic → `_handle_multi_select_ok()`
   - Extract single selection logic → `_handle_single_select_ok()`
   - Extract validation → `_validate_selection_type(item_info)`
   - Extract error handling → `_handle_selection_error()`

5. **Decompose _handle_item_activated() Method (1 hour)**
   - Extract multi-select toggle → `_handle_multi_select_toggle(index, item_info)`
   - Extract directory navigation → `_handle_directory_navigation(item_info)`
   - Extract UI updates → `_update_selection_display()`
   - Add `select_multiple` parameter to FileManagerBrowser
   - Implement multi-selection UI with checkboxes/indicators
   - Track multiple selected directories in internal state
   - Update add plate workflow to handle multiple results

5. **Test Button Functionality**
   - Verify OK/Cancel buttons respond to clicks
   - Test multi-folder selection workflow
   - Validate integration with add plate command
   - Ensure escape key closes dialogs properly

### Findings
**CANONICAL BUTTON FUNCTIONALITY FROM tui_final.md:**
- **Add button**: "make an orchestrator with the default config using a filepath obtained through file select dialog. multiple folders may be selected at once"
- **Init button**: "makes each selected plate run their initialize() method. if no errors then update with a yellow ! symbol"
- **Compile button**: "runs the pipeline compiler using the list of steps"
- **Error handling**: "if errors then update with error symbol and message in status bar with OK dialog"

**ORCHESTRATOR INTEGRATION REQUIREMENTS:**
- **Status progression**: `?` (uninitialized) → `!` (initialized) → `o` (compiled/running)
- **Method calls**: `orchestrator.initialize()`, pipeline compiler execution
- **Error recovery**: Return to previous state on error, display in status bar
- **Multi-folder support**: Create multiple orchestrators from single add operation

**DNA ANALYSIS RESULTS:**
- `PJLJ:handleok:16` = File browser OK button handler with complexity 16
- `PJLJ:handleitemactiv:11` = File browser item activation handler with complexity 11
- These complex handlers prevent proper multi-folder selection functionality

**ROOT CAUSE ANALYSIS:**
- **Primary**: Missing orchestrator integration - buttons don't call orchestrator methods
- **Secondary**: No multi-folder selection support in file browser
- **Tertiary**: Overly complex button handlers with nested conditionals

**SPEC REQUIREMENTS:**
- "file select dialog that allows multiple folders to be selected"
- Each selected folder becomes a separate plate (orchestrator)
- All plates start with `?` status (not initialized)

**CURRENT FILE BROWSER STATUS:**
- ✅ **Basic functionality exists** - file browser component works
- ✅ **Directory navigation works** - can browse and select directories
- ❌ **Button handlers too complex** - causing click failures
- ❌ **No multi-selection** - only single directory selection

**IMPLEMENTATION SCOPE:**
- Method decomposition to reduce complexity
- Multi-selection UI and logic
- Integration with add plate command
- Button responsiveness fixes

**ESTIMATED EFFORT:**
- **Canonical button functionality**: ~4 hours (orchestrator integration)
- **Multi-folder selection**: ~3 hours (file browser enhancement)
- **DNA complexity reduction**: ~2 hours (method decomposition)
- **Testing and integration**: ~2 hours (workflow validation)
- **Total**: ~11 hours (focused on canonical specification requirements)

### Implementation Draft
*Implementation will be added after smell loop approval*
