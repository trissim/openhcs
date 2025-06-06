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
   - **Add Plate**: Multi-folder selection → create orchestrators → `?` status
   - **Initialize**: Click init → `orchestrator.initialize()` → `!` status
   - **Compile**: Click compile → `orchestrator.compile_pipelines()` → `o` status
   - **Run**: Click run → `orchestrator.execute_compiled_plate()` → monitor progress
   - **Delete**: Select plates → click delete → confirm removal

3. **Test Pipeline Editing Workflow**
   - **Add Step**: Click add → create FunctionStep → add to pipeline
   - **Edit Step**: Click edit → dual editor replaces left pane → test Step/Func toggle
   - **Save/Close**: Test save button → construct new step → close restores layout
   - **Load/Save Pipeline**: Test pickle load/save operations

4. **Test Dialog and Navigation Systems**
   - **Global Settings**: Modal dialog blocks interaction, takes most of screen
   - **Help Dialog**: Simple text + OK button functionality
   - **File Dialogs**: Multi-folder selection for add plate
   - **Focus Management**: Proper tab order and keyboard navigation

5. **Validate Against Specification**
   - **Layout**: Exact 3-bar structure with correct content
   - **Status Symbols**: `?`, `!`, `o` progression with proper colors
   - **Button Functionality**: All buttons clickable and responsive
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
- **Orchestrator integration**: Buttons must call actual orchestrator methods
- **Status symbol progression**: `?` → `!` → `o` based on orchestrator state
- **Dual editor system**: Step/Func toggle with pane replacement
- **Multi-folder selection**: "multiple folders may be selected at once" for add plate

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

### Implementation Draft
*Implementation will be added after smell loop approval*
