# plan_SIMPLE_02_create_layout_components.md
## Component: Create Canonical TUI Layout Structure

### Objective
Create the exact layout structure specified in tui_final.md: 3-row header + dual-pane main area + status bar, with dual editor system that can replace plate manager pane.

### Plan
1. **Create Canonical Top Bar (Row 1) - 2 hours**
   - **Layout**: `[Global Settings] [Help] | OpenHCS V1.0`
   - **Left side**: Global Settings and Help buttons (using FramedButton)
   - **Right side**: "OpenHCS V1.0" title text, right-aligned
   - **Functionality**: Global Settings opens config dialog, Help opens help dialog
   - **Integration**: Replace current MenuBar with this canonical top bar

2. **Create Section Title Bar (Row 2) - 1 hour**
   - **Layout**: `1 plate manager | 2 Pipeline editor`
   - **Left pane title**: "1 plate manager"
   - **Right pane title**: "2 Pipeline editor"
   - **Visual separation**: Vertical divider `|` between panes
   - **Dynamic replacement**: When editing step, left becomes "Step/Func Editor"

3. **Create Button Bars (Row 3) - 2 hours**
   - **Plate Manager Buttons**: `[add][del][edit][init][compile][run]`
   - **Pipeline Editor Buttons**: `[add][del][edit][load][save]`
   - **Button functionality**: Each button calls specific orchestrator methods
   - **Visual consistency**: All buttons use FramedButton component
   - **Integration**: Connect to existing command system

4. **Create Dual-Pane Main Layout - 2 hours**
   - **Left pane**: Plate list with status symbols (`?`/`!`/`o`) in left margin
   - **Right pane**: Step list for selected plate's pipeline
   - **Format**: `|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/`
   - **Scrollable content**: Both panes support scrolling with `^/v` indicators
   - **Integration**: Use existing PlateListView and StepListView

5. **Create Dual Editor System - 4 hours**
   - **Step/Func toggle**: `|_X_Step_X_|___Func___|_[save]__[close]_|`
   - **Step editor**: Static reflection of `FunctionStep` parameters (name, input_dir, output_dir, force_disk_output, variable_components, group_by)
   - **Func editor**: Use existing `FunctionPatternEditor` from `openhcs/tui/function_pattern_editor.py`
   - **Function registry integration**: `get_functions_by_memory_type("numpy")` for dropdowns
   - **Pattern building**: Support all 4 pattern types (single, tuple, list, dict) via visual interface
   - **Save/Close buttons**: Save creates `FunctionStep(func=pattern_editor_output, name=step_name, variable_components=step_components, group_by=step_group_by, force_disk_output=step_force_disk, input_dir=step_input_dir, output_dir=step_output_dir)`, Close returns to plate manager
   - **Replacement logic**: Dual editor replaces left pane when editing step

6. **Implement Pane Replacement System - 1 hour**
   - **Method to replace left pane** with dual editor when "edit" clicked on step
   - **Method to restore plate manager** when "close" clicked in dual editor
   - **Focus management**: Proper focus transitions during pane replacement
   - **State preservation**: Remember plate manager state when switching

### Findings
**CANONICAL LAYOUT FROM tui_final.md:**
```
Row 1: [Global Settings] [Help] | OpenHCS V1.0
Row 2: 1 plate manager | 2 Pipeline editor
Row 3: [add][del][edit][init][compile][run] | [add][del][edit][load][save]
Main:  |o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/ | |o| ^/v 1: pos_gen_pattern |
Bottom: Status: ...
```

**DUAL EDITOR REPLACEMENT SYSTEM:**
- When "edit" clicked on step in pipeline editor → left pane becomes Step/Func dual editor
- Dual editor has toggle buttons: `|_X_Step_X_|___Func___|_[save]__[close]_|`
- Step editor shows AbstractStep parameters with reset buttons and file dialogs
- Func editor shows function patterns from func registry with parameter editing
- Save button grayed until changes made, then becomes available
- Close button returns to normal plate manager view

**ORCHESTRATOR INTEGRATION REQUIREMENTS:**
- **Add button**: Multi-folder selection → `PipelineOrchestrator(plate_path, global_config)` → status `?`
- **Init button**: Calls `orchestrator.initialize()` → status `?` → `-` (yellow) if successful
- **Compile button**: Calls `orchestrator.compile_pipelines(pipeline_steps)` → status `-` → `o` (green)
- **Run button**: Calls `orchestrator.execute_compiled_plate()` → status `o` → `!` (red during execution)
- **Edit button**: Opens static reflection editor for `orchestrator.global_config` (GlobalPipelineConfig)
- **Error handling**: Catch orchestrator exceptions, show in status bar + modal dialog, maintain TUI state

**CRITICAL ARCHITECTURAL BOUNDARIES:**
- **TUI SHOULD ACCESS**: `PipelineOrchestrator` public methods only
- **TUI SHOULD NOT ACCESS**: `ProcessingContext`, `step_plans`, `FileManager`, `MicroscopeHandler`
- **CLEAN SEPARATION**: TUI manages UI state, orchestrator manages internal pipeline state
- **STATE TRACKING**: TUI tracks orchestrator lifecycle externally (`?`/`-`/`o`/`!` progression)

**IMPLEMENTATION SCOPE:**
- **5 major layout components**: Top bar, section titles, button bars, dual panes, dual editor
- **Pane replacement system**: Dynamic switching between plate manager and dual editor
- **Orchestrator integration**: Connect buttons to actual orchestrator method calls
- **Static reflection**: Config editing for both global settings and plate-specific settings

**ESTIMATED EFFORT:**
- **Top bar**: ~2 hours (Global Settings + Help buttons + title)
- **Section titles**: ~1 hour (simple dual-pane titles)
- **Button bars**: ~2 hours (orchestrator method integration)
- **Dual panes**: ~2 hours (status symbols + scrolling)
- **Dual editor**: ~4 hours (Step/Func toggle + static reflection)
- **Pane replacement**: ~1 hour (switching logic)
- **Total**: ~12 hours (was 2 hours before canonical specification)

**CRITICAL IMPLEMENTATION DETAILS:**

#### **FunctionStep Construction Pattern**
```python
# When saving from dual editor:
function_step = FunctionStep(
    func=pattern_editor_output,  # Object returned by FunctionPatternEditor
    name=step_name,              # From step editor text field
    variable_components=step_components,  # From step editor list
    group_by=step_group_by,      # From step editor dropdown
    force_disk_output=step_force_disk,    # From step editor checkbox
    input_dir=step_input_dir,    # From step editor path field
    output_dir=step_output_dir   # From step editor path field
)
```

#### **File Operations Pattern**
```python
# Loading .pipeline, .step, .func files:
with open(filepath, 'rb') as f:
    loaded_object = pickle.load(f)
    # Then assign to appropriate variable:
    # - .pipeline → List[FunctionStep] → pipeline_list
    # - .step → FunctionStep → individual step
    # - .func → function pattern → pattern_editor.current_pattern

# Saving follows same pattern with pickle.dump()
```

### Implementation Draft
*Implementation will be added after smell loop approval*
