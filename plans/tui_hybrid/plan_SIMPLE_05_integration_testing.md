# plan_SIMPLE_05_integration_testing.md
## Component: Integration and Manual Verification of Complete TUI

### Objective
Integrate all components into the canonical TUI layout and perform manual verification against tui_final.md specification. Note: No automated testing infrastructure exists yet - testing will be added later.

### Plan
1. **Wire Up the Complete Layout**
   - Integrate ThreeBarLayout with all copied/created components
   - Connect MenuBar, SectionTitleBar, ActionToolbars, StatusBar
   - Test 3-bar layout structure matches specification
   - Verify proper component spacing and alignment

2. **Test Complete Plate Management Workflow**
   - **Add Plate**: Multi-folder selection → `PipelineOrchestrator(plate_path, global_config)` → `?` status (gray)
   - **Initialize**: Click init → `orchestrator.initialize()` → status `?` → `-` (yellow) if successful
   - **Compile**: Click compile → `orchestrator.compile_pipelines(pipeline_definition)` → status `-` → `o` (green)
  - **Note**: `pipeline_definition` is `List[FunctionStep]` (TUI only uses FunctionStep)
   - **Run**: Click run → `orchestrator.execute_compiled_plate()` → status `o` → `!` (red during execution)
   - **Delete**: Select plates → click delete → remove from TUI state (orchestrator instances)
   - **Error handling**: Test orchestrator exception handling, status bar display, state recovery

3. **Test Pipeline Editing Workflow**
   - **Add Step**: Click add → create `FunctionStep(func=pattern, name=name, ...)` → add to pipeline
   - **Edit Step**: Click edit → dual editor replaces left pane → test Step/Func toggle
   - **Function Pattern Editor**: Test existing `FunctionPatternEditor` integration with `FUNC_REGISTRY`
   - **Pattern Building**: Test all 4 pattern types (single, tuple, list, dict) via visual interface
   - **Save/Close**: Test save button → construct new `FunctionStep` → close restores layout
   - **Load/Save Pipeline**: Test pickle load/save operations for `.step` files

4. **Test Dialog and Navigation Systems**
   - **Global Settings**: Modal dialog blocks interaction, takes most of screen
   - **Help Dialog**: Simple text + OK button functionality
   - **File Dialogs**: Multi-folder selection for add plate
   - **Focus Management**: Proper tab order and keyboard navigation

5. **Validate Against Specification**
   - **Layout**: Exact 3-bar structure with correct content
   - **Status Symbols**: `?` (gray), `-` (yellow), `o` (green), `!` (red) progression with proper colors
   - **Button Functionality**: All buttons clickable and responsive
   - **Orchestrator Integration**: Verify clean separation - TUI never accesses internal orchestrator state
   - **Workflows**: Complete plate and pipeline management
   - **Visual Match**: Compare with specification diagram

6. **Performance and Stability Testing**
   - Test with multiple plates and complex pipelines
   - Verify no memory leaks or resource issues
   - Test error handling and recovery
   - Validate proper shutdown and cleanup

### Findings
**CANONICAL INTEGRATION REQUIREMENTS FROM tui_final.md:**
- **Exact layout match**: 3-row structure with specific content in each row
- **Orchestrator integration**: Buttons must call actual `PipelineOrchestrator` methods
- **Status symbol progression**: `?` → `-` → `o` → `!` based on orchestrator lifecycle (TUI-managed)
- **Dual editor system**: Step/Func toggle with pane replacement using existing `FunctionPatternEditor`
- **Multi-folder selection**: "multiple folders may be selected at once" for add plate
- **Clean architecture**: TUI manages orchestrator instances externally, never accesses internal state

**INTEGRATION SCOPE:**
- Wire together canonical layout components (top bar, section titles, button bars, dual panes)
- Connect FramedButton and StatusBar from archive
- Integrate orchestrator method calls with button functionality
- Implement dual editor pane replacement system

**TESTING PRIORITIES:**
1. **Critical**: Canonical layout structure matches specification exactly
2. **Critical**: Orchestrator integration (init, compile, run methods)
3. **Critical**: Status symbol progression with proper colors
4. **Critical**: Dual editor pane replacement functionality
5. **Important**: Multi-folder plate selection workflow

**SUCCESS CRITERIA:**
- All buttons respond to clicks and perform expected actions
- Status symbols update correctly through plate lifecycle
- Dual editor opens/closes properly with pane replacement
- Multi-folder selection works for batch plate creation
- Layout matches specification exactly
- No crashes or hangs during normal operation

**RISK AREAS:**
- Component integration issues
- Focus management during pane replacement
- State synchronization between components
- Error handling in workflow operations

**ESTIMATED EFFORT:**
- Component integration: ~4 hours
- Workflow testing: ~6 hours
- Bug fixes and polish: ~4 hours
- Total: ~14 hours

**CRITICAL IMPLEMENTATION DETAILS:**

#### **Complete Pipeline Workflow Testing**
```python
# Test sequence for complete workflow:

# 1. Add plates (multi-folder selection)
selected_folders = ["/path/to/plate1", "/path/to/plate2"]
for folder in selected_folders:
    orchestrator = PipelineOrchestrator(folder, global_config)
    tui_state.orchestrators[folder] = orchestrator
    tui_state.orchestrator_status[folder] = "?"  # gray

# 2. Initialize (with error recovery)
try:
    orchestrator.initialize()
    tui_state.orchestrator_status[folder] = "-"  # yellow
except Exception as e:
    # Status stays "?" - show error but don't update flag
    show_error_dialog(f"Init failed: {e}")

# 3. Build pipeline (FunctionStep construction)
function_step = FunctionStep(
    func=pattern_editor.get_pattern(),  # From FunctionPatternEditor
    name=step_editor.get_name(),
    variable_components=step_editor.get_components(),
    group_by=step_editor.get_group_by(),
    force_disk_output=step_editor.get_force_disk(),
    input_dir=step_editor.get_input_dir(),
    output_dir=step_editor.get_output_dir()
)
pipeline_definition = [function_step]  # List[FunctionStep]

# 4. Compile (with error recovery)
try:
    orchestrator.compile_pipelines(pipeline_definition)
    tui_state.orchestrator_status[folder] = "o"  # green
except Exception as e:
    # Status stays "-" - show error but don't update flag
    show_error_dialog(f"Compile failed: {e}")

# 5. Execute (with error recovery)
try:
    results = orchestrator.execute_compiled_plate()
    tui_state.orchestrator_status[folder] = "!"  # red during execution
except Exception as e:
    # Status stays "o" - show error but don't update flag
    show_error_dialog(f"Execution failed: {e}")
```

#### **File Operations Testing**
```python
# Test save/load operations:

# Save pipeline
with open("test.pipeline", "wb") as f:
    pickle.dump(pipeline_definition, f)  # List[FunctionStep]

# Load pipeline
with open("test.pipeline", "rb") as f:
    loaded_pipeline = pickle.load(f)  # → List[FunctionStep]
    tui_state.current_pipelines[plate_path] = loaded_pipeline

# Save individual step
with open("test.step", "wb") as f:
    pickle.dump(function_step, f)  # FunctionStep object

# Load individual step
with open("test.step", "rb") as f:
    loaded_step = pickle.load(f)  # → FunctionStep object
    pipeline_definition.append(loaded_step)

# Save function pattern
with open("test.func", "wb") as f:
    pickle.dump(pattern_editor.current_pattern, f)  # Function pattern

# Load function pattern
with open("test.func", "rb") as f:
    loaded_pattern = pickle.load(f)  # → Function pattern
    pattern_editor.current_pattern = loaded_pattern
```

### Implementation Draft
*Implementation will be added after smell loop approval*
