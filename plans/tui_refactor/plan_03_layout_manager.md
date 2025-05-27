# plan_03_layout_manager.md
## Component: Hierarchical Layout Manager

### Objective
Implement the 5-bar hierarchical layout system with shared container walls, dynamic pane switching, and proper scrolling. Architecture uses DynamicContainer for left pane replacement and shared borders to avoid bloat.

### Plan
1. Create main window with shared container walls (not separate frames)
2. Implement VSplit with single shared border between left and right panes
3. Build DynamicContainer system for left pane switching (plate_manager ↔ step_func_editor)
4. Create status bar with expandable log drawer
5. Add global menu bar with settings/help/exit
6. Implement scrollbars for overflow content in both panes

### Findings
Crystal clear architecture:

**5-Bar Layout Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────────┐┌──────┐┌──────┐    OpenHCS V1.0        │ ← Global menu bar with squared buttons
│ │ Global Settings ││ Help ││ Exit │                        │
│ └─────────────────┘└──────┘└──────┘                        │
├─────────────────────────┬───────────────────────────────────┤
│ Left Pane Title         │ Pipeline Editor                   │ ← Two separate title areas
├─────────────────────────┼───────────────────────────────────┤
│ Left Pane Buttons       │ [add][del][edit][load][save]      │ ← Two separate button areas
├─────────────────────────┼───────────────────────────────────┤
│ LEFT CONTAINER          │ RIGHT CONTAINER                   │ ← Two SEPARATE containers
│ (completely separate)   │ (always pipeline editor)          │ ← NOT shared content
│ (scrollable)            │ (scrollable)                      │
├─────────────────────────┴───────────────────────────────────┤
│ Status:_...                                                 │ ← Status bar format
└─────────────────────────────────────────────────────────────┘
```

**Left Pane Dynamic Switching:**
- **Mode 1**: Plate Manager (default)
- **Mode 2**: Step/Func Editor (when edit button clicked)

**Container Architecture:**
```python
main_layout = HSplit([
    global_menu_bar,                    # Top bar with squared buttons
    VSplit([                           # Shared walls
        DynamicContainer(get_left_pane), # Switches between modes
        Window(width=1, char='│'),       # Shared border
        pipeline_editor_pane             # Always pipeline editor
    ]),
    status_bar                          # Bottom bar
])

def get_left_pane():
    if mode == "step_func_editor":
        return step_func_editor.container
    else:
        return plate_manager.container
```

**Button Style Requirements:**
- **Shared Wall Architecture**: Buttons share walls with surrounding containers
- **No Double Walls**: Use ┌┬┐├┼┤└┴┘ characters to connect seamlessly
- **Global menu**: `┌─────────────────┬──────┬──────┐` (shared walls)
- **All buttons**: Integrate directly into container walls, no gaps
- **Consistent styling**: All buttons throughout the TUI use shared wall architecture

**Icon Positioning:**
- Icons on LEFT side of items: `o plate1    ↑↓`
- Steps have arrows: `↑↓ step1`
- Scrollbars appear when content overflows

### Implementation Draft
(Only after smell loop passes)
