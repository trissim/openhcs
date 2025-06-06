# plan_SIMPLE_04_command_integration_fixes.md
## Component: Fix Command System Integration and File Browser Issues

### Objective
Implement canonical button functionality as specified in tui_final.md: orchestrator method integration, multi-folder selection for add plate, and fix DNA-identified file browser complexity issues.

### Plan
1. **CRITICAL: Implement Canonical Button Functionality (4 hours)**
   - **Add button**: Multi-folder selection → `PipelineOrchestrator(plate_path, global_config)` → `?` status
   - **Del button**: Removes selected plates from TUI state (orchestrator instances)
   - **Edit button**: Opens static reflection editor for `orchestrator.global_config` (GlobalPipelineConfig)
   - **Init button**: Calls `orchestrator.initialize()` → status `?` → `-` (yellow) if successful
   - **Compile button**: Calls `orchestrator.compile_pipelines(pipeline_definition)` → status `-` → `o` (green)
  - **Note**: `pipeline_definition` is `List[FunctionStep]` (TUI only uses FunctionStep, though method accepts List[AbstractStep])
   - **Run button**: Calls `orchestrator.execute_compiled_plate()` → status `o` → `!` (red during execution)
   - **Error handling**: Catch orchestrator exceptions, display in status bar + modal dialog, maintain TUI state

2. **Implement Multi-Folder Selection for Add Plate (3 hours)**
   - **Canonical requirement**: "multiple folders may be selected at once" for add plate
   - **File browser enhancement**: Support multi-folder selection mode
   - **UI integration**: Multi-select checkboxes or ctrl+click selection
   - **Orchestrator creation**: Create one `PipelineOrchestrator(plate_path, global_config)` per selected folder
   - **TUI state management**: Track multiple orchestrator instances in TUI state
   - **Status integration**: Each new plate starts with `?` (created) status

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
- **Status progression**: `?` (created) → `-` (initialized) → `o` (compiled) → `!` (running/error)
- **Method calls**: `orchestrator.initialize()`, `orchestrator.compile_pipelines(pipeline_definition)`, `orchestrator.execute_compiled_plate()`
- **Pipeline definition**: `List[FunctionStep]` objects (TUI only uses FunctionStep, though orchestrator accepts List[AbstractStep])
- **Error recovery**: Catch orchestrator exceptions, display in status bar, maintain TUI state at previous level
- **Multi-folder support**: Create multiple `PipelineOrchestrator` instances from single add operation
- **Clean separation**: TUI never accesses `ProcessingContext`, `step_plans`, or other orchestrator internals

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

**CRITICAL IMPLEMENTATION DETAILS:**

#### **Multi-Folder Orchestrator Creation Pattern**
```python
def handle_add_plates_button():
    # Multi-folder selection from file dialog
    selected_folders = file_dialog.select_multiple_directories()

    for folder_path in selected_folders:
        # Create orchestrator for each folder
        orchestrator = PipelineOrchestrator(folder_path, global_config)

        # Track in TUI state
        tui_state.orchestrators[folder_path] = orchestrator
        tui_state.orchestrator_status[folder_path] = "?"  # gray - created but not initialized
        tui_state.current_pipelines[folder_path] = []     # empty pipeline list

        # Update UI display
        plate_list_view.add_plate(folder_path, status="?")
```

#### **Button Handler Error Recovery Pattern**
```python
def handle_button_with_error_recovery(operation_name, operation_func, success_status):
    try:
        operation_func()
        # Only update status if successful
        tui_state.orchestrator_status[current_plate] = success_status
        status_bar.update_message(f"{operation_name} completed successfully")
    except Exception as e:
        # DON'T update status flag - keep at previous state
        error_msg = f"{operation_name} failed: {str(e)}"
        status_bar.update_message(f"Error: {error_msg}")
        show_modal_dialog(title="Error", message=error_msg, buttons=["OK"])

# Usage examples:
handle_button_with_error_recovery("Initialization",
                                 lambda: orchestrator.initialize(),
                                 "-")  # yellow

handle_button_with_error_recovery("Compilation",
                                 lambda: orchestrator.compile_pipelines(pipeline_definition),
                                 "o")  # green
```

### Implementation Draft
*Implementation will be added after smell loop approval*
