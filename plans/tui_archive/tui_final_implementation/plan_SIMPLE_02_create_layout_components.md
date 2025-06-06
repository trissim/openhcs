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
   - **Step editor**: Static reflection of AbstractStep parameters with reset buttons
   - **Func editor**: Static reflection of function patterns with dropdown from func registry
   - **Save/Close buttons**: Save grayed until changes, Close returns to plate manager
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
- **Add button**: Multi-folder selection → creates orchestrator with default config
- **Init button**: Calls `orchestrator.initialize()` → changes status to `!`
- **Compile button**: Runs pipeline compiler → changes status to `o`
- **Edit button**: Opens static reflection config editor for selected plates
- **Error handling**: Errors show in status bar with OK dialog, return to previous state

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

### Implementation Draft
*Implementation will be added after smell loop approval*
