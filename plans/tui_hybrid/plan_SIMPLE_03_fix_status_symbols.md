# plan_SIMPLE_03_fix_status_symbols.md
## Component: Centralized Status Symbol System Redesign

### Objective
Implement the canonical status symbol system as specified in tui_final.md: `?` (created), `-` (initialized), `o` (compiled), `!` (running/error). Create centralized system that tracks PipelineOrchestrator lifecycle externally in TUI state.

### Plan
1. **CRITICAL: Implement Canonical Status Symbol System (2 hours)**
   - Create `openhcs/tui/constants/status_symbols.py` with canonical mappings:
     - `?` = created (gray) - `PipelineOrchestrator(plate_path)` created but not initialized
     - `-` = initialized (yellow) - `orchestrator.initialize()` completed successfully
     - `o` = compiled (green) - `orchestrator.compile_pipelines()` completed successfully
     - `!` = running/error (red) - during `orchestrator.execute_compiled_plate()` or error state
   - Define TUI state tracking for orchestrator lifecycle (TUI manages this, not orchestrator)
   - Include color coding: gray for `?`, yellow for `-`, green for `o`, red for `!`

2. **CRITICAL: Update Plate Manager Status Display (2 hours)**
   - **PlateListView**: Update to use canonical `?`/`!`/`o` symbols
   - **Connect to orchestrator state**: Map orchestrator status to symbols
   - **Status transitions**: uninitialized → initialized → compiled
   - **Error handling**: Display error symbol and message in status bar

3. **CRITICAL: Update Pipeline Editor Status Display (2 hours)**
   - **StepListView**: Use consistent symbol system for step status
   - **Step status logic**: Based on step completion state
   - **Visual consistency**: Same symbols and colors as plate manager

4. **Integrate with Orchestrator Lifecycle Tracking (2 hours)**
   - **TUI state management**: Track orchestrator instances and their lifecycle states externally
   - **Status updates**: When `orchestrator.initialize()` succeeds → `?` to `-` transition
   - **Compilation updates**: When `orchestrator.compile_pipelines()` succeeds → `-` to `o` transition
   - **Execution updates**: When `orchestrator.execute_compiled_plate()` starts → `o` to `!` transition
   - **Error states**: When orchestrator methods throw exceptions → show in status bar, maintain previous state
   - **Real-time updates**: Status changes reflected immediately in UI

### Findings
**CANONICAL SPECIFICATION FROM tui_final.md:**
- **Status symbols**: `?` (created), `-` (initialized), `o` (compiled), `!` (running/error)
- **Status progression**: `PipelineOrchestrator` created → `?` → `initialize()` → `-` → `compile_pipelines()` → `o` → `execute_compiled_plate()` → `!`
- **Error handling**: Orchestrator exceptions display in status bar with OK dialog, TUI maintains previous state
- **Visual integration**: Status symbols appear in left vertical bar next to plate names
- **TUI responsibility**: Track orchestrator lifecycle externally, orchestrator doesn't expose internal state

**CURRENT IMPLEMENTATION GAPS:**
- **Missing orchestrator integration**: Status not connected to actual `PipelineOrchestrator` method calls
- **Wrong symbol mappings**: Current code doesn't match canonical 4-state progression
- **No error state handling**: Missing orchestrator exception handling and state recovery logic
- **Inconsistent across components**: Each component defines own symbols instead of centralized system

**CANONICAL STATUS FLOW:**
```
Multi-folder selection → PipelineOrchestrator(plate_path) → `?` (created)
    ↓ [init] button clicked
orchestrator.initialize() succeeds → `-` (initialized, yellow)
    ↓ [compile] button clicked
orchestrator.compile_pipelines(steps) succeeds → `o` (compiled, green)
    ↓ [run] button clicked
orchestrator.execute_compiled_plate() running → `!` (running, red)
```

**CRITICAL ARCHITECTURAL UNDERSTANDING:**
- **TUI tracks state**: Orchestrator lifecycle managed externally by TUI, not exposed by orchestrator
- **Clean separation**: TUI calls orchestrator methods, catches exceptions, manages UI state transitions
- **No internal access**: TUI never accesses `ProcessingContext`, `step_plans`, or other orchestrator internals

**IMPLEMENTATION REQUIREMENTS:**
- **Orchestrator integration**: Connect status changes to actual orchestrator method calls
- **Error handling**: Implement error dialogs and state recovery as specified
- **Visual consistency**: Ensure symbols appear correctly in plate list left margin
- **Real-time updates**: Status changes must be immediately visible

**ESTIMATED EFFORT:**
- **Create canonical constants**: ~2 hours
- **Update plate manager display**: ~2 hours
- **Update pipeline editor display**: ~2 hours
- **Integrate with orchestrator**: ~2 hours
- **Total**: ~8 hours (architectural redesign to match canonical specification)

**INTELLECTUAL HONESTY WIN:**
- Canonical specification provides exact requirements for status symbols
- Implementation must match orchestrator state transitions exactly
- Proper specification adherence prevents inconsistent user experience

**CRITICAL IMPLEMENTATION DETAILS:**

#### **Error Recovery Pattern**
```python
# Button click handlers - DON'T update status flag if operation fails:

def handle_init_button():
    try:
        orchestrator.initialize()
        # Only update status if successful:
        tui_state.orchestrator_status[plate_path] = "-"  # yellow
    except Exception as e:
        # DON'T update status, keep at previous state
        show_error_message(f"Initialization failed: {str(e)}")
        # Status remains "?" (gray)

def handle_compile_button():
    try:
        orchestrator.compile_pipelines(pipeline_definition)
        # Only update status if successful:
        tui_state.orchestrator_status[plate_path] = "o"  # green
    except Exception as e:
        # DON'T update status, keep at previous state
        show_error_message(f"Compilation failed: {str(e)}")
        # Status remains "-" (yellow)
```

#### **Error Display Pattern**
```python
def show_error_message(error_text: str):
    # Option 1: Status bar at bottom
    status_bar.update_message(f"Error: {error_text}")

    # Option 2: Modal dialog box
    show_modal_dialog(title="Error", message=error_text, buttons=["OK"])

    # Option 3: Both (recommended)
    status_bar.update_message(f"Error: {error_text}")
    show_modal_dialog(title="Error", message=error_text, buttons=["OK"])
```

### Implementation Draft
*Implementation will be added after smell loop approval*
