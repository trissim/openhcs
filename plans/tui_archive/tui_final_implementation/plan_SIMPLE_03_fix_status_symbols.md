# plan_SIMPLE_03_fix_status_symbols.md
## Component: Centralized Status Symbol System Redesign

### Objective
Implement the canonical status symbol system as specified in tui_final.md: `?` (uninitialized), `!` (initialized but not compiled), `o` (compiled/ready/running). Create centralized system to ensure consistency across all components.

### Plan
1. **CRITICAL: Implement Canonical Status Symbol System (2 hours)**
   - Create `openhcs/tui/constants/status_symbols.py` with canonical mappings:
     - `?` = uninitialized (red/default) - plate not initialized yet
     - `!` = initialized but not compiled (yellow) - after orchestrator.initialize()
     - `o` = compiled/ready/running (green) - after pipeline compilation
   - Define orchestrator state → symbol mapping functions
   - Include color coding: red for `?`, yellow for `!`, green for `o`

2. **CRITICAL: Update Plate Manager Status Display (2 hours)**
   - **PlateListView**: Update to use canonical `?`/`!`/`o` symbols
   - **Connect to orchestrator state**: Map orchestrator status to symbols
   - **Status transitions**: uninitialized → initialized → compiled
   - **Error handling**: Display error symbol and message in status bar

3. **CRITICAL: Update Pipeline Editor Status Display (2 hours)**
   - **StepListView**: Use consistent symbol system for step status
   - **Step status logic**: Based on step completion state
   - **Visual consistency**: Same symbols and colors as plate manager

4. **Integrate with Orchestrator State Management (2 hours)**
   - **Status updates**: When orchestrator.initialize() called → `!` symbol
   - **Compilation updates**: When pipeline compiled → `o` symbol
   - **Error states**: When errors occur → error symbol + status bar message
   - **Real-time updates**: Status changes reflected immediately in UI

### Findings
**CANONICAL SPECIFICATION FROM tui_final.md:**
- **Status symbols**: `?` (uninitialized), `!` (initialized), `o` (compiled/running)
- **Status progression**: Plate starts as `?` → orchestrator.initialize() → `!` → pipeline compile → `o`
- **Error handling**: Errors display in status bar with OK dialog, plate returns to previous state
- **Visual integration**: Status symbols appear in left vertical bar next to plate names

**CURRENT IMPLEMENTATION GAPS:**
- **Missing orchestrator integration**: Status not connected to orchestrator.initialize() calls
- **Wrong symbol mappings**: Current code uses `-` and `✓` instead of canonical `!` and `o`
- **No error state handling**: Missing error display and state recovery logic
- **Inconsistent across components**: Each component defines own symbols

**CANONICAL STATUS FLOW:**
```
Plate added → `?` (uninitialized)
    ↓ [init] button clicked
orchestrator.initialize() → `!` (initialized but not compiled)
    ↓ [compile] button clicked
pipeline compiled → `o` (compiled/ready)
    ↓ [run] button clicked
pipeline running → `o` (running - same symbol)
```

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

### Implementation Draft
*Implementation will be added after smell loop approval*
