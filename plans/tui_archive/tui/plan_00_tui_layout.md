# plan_00_tui_layout.md
## Component: TUI Layout Visualization

### Objective
Provide a clear visual representation of the OpenHCS TUI layout and its components to guide implementation.

### Plan
1. Create ASCII art representations of the main TUI layout
2. Illustrate each major component (Plate Manager, Step Viewer, Action Menu)
3. Show the Function Pattern Editor layout
4. Demonstrate state transitions and modal dialogs

### Findings

#### Main TUI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OpenHCS TUI - Menu Bar                           │
├───────────────────────┬───────────────────────┬───────────────────────────┤
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│   Plate Manager       │    Step Viewer        │     Action Menu          │
│   (Left Pane)         │    (Middle Pane)      │     (Right Pane)         │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
│                       │                       │                           │
├───────────────────────┴───────────────────────┴───────────────────────────┤
│                             Status Bar                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Plate Manager Pane (Left)

```
┌─────────────────────────────────────────────┐
│ Plate Manager                               │
├─────────────────────────────────────────────┤
│ ○ Plate 1 | /path/to/plate_1               │
│ ● Plate 2 | /path/to/plate_2               │
│ ✗ Plate 3 | /path/to/plate_3               │
│ ○ Plate 4 | /path/to/plate_4               │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
├─────────────────────────────────────────────┤
│ [Add Plate] [Remove Plate] [Refresh]        │
└─────────────────────────────────────────────┘
```

Legend:
- ○: Ready
- ●: Completed
- ✗: Error
- ◔: Compiling
- ◕: Running

#### Step Viewer Pane (Middle)

```
┌─────────────────────────────────────────────┐
│ Step Viewer                                 │
├─────────────────────────────────────────────┤
│ ─── Position Generation Pipeline ───        │
│ ○ Generate Positions: position_gen → numpy  │
│                                             │
│ ─── Image Enhancement Pipeline ───          │
│ ● Enhance Images: enhance_func → numpy      │
│ ✗ Assemble Stack: stack_func → numpy        │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
│                                             │
├─────────────────────────────────────────────┤
│ [Edit Step] [Add Step] [Remove Step]        │
└─────────────────────────────────────────────┘
```

Legend:
- ○: Pending
- ●: Validated
- ✗: Error

#### Action Menu Pane (Right)

```
┌─────────────────────────────────────────────┐
│ Action Menu                                 │
├─────────────────────────────────────────────┤
│ ERROR: Compilation failed: Invalid memory   │
│ type in function 'stack_func'               │
├─────────────────────────────────────────────┤
│                                             │
│ [    Add    ]                               │
│                                             │
│ [ Pre-compile ]                             │
│                                             │
│ [  Compile  ]                               │
│                                             │
│ [    Run    ]                               │
│                                             │
│ [   Save    ]                               │
│                                             │
│ [   Test    ]                               │
│                                             │
│ [ Settings  ]                               │
│                                             │
└─────────────────────────────────────────────┘
```

#### Function Pattern Editor (Replaces Left Pane)

```
________________________________________________
| Func Pattern editor  [load] [save]            |
|_______________________________________________|____________________________________________
| dict_keys: |None|V| +/-  |  [edit in vim] ?   | <- you can have mor than one key, making he list of funcs change.
|_______________________________________________|____________________________________________
|^| Func 1: |percentile stack normalize|v|      | <- drop down menu generated from using the func register and static analysis of func name definition
|X| --------------------------------------------|--------------------------------------------
|X|  move  [reset] percentile_low:  0.1 ...     | <- these two kwargs with editabel fields are autogen from the func definition)
|X|   /\   [reset]  percentile_high: 99.9 ...   |
|X|   \/   [add]                                |
|X|        [delete]                             |
|X| ____________________________________________|
|X| Func 2: |n2v2|V|                            | <- drop down menu generated from using the func register and static analysis of func name definition
|X|---------------------------------------------|
|X|  	    [reset] random_seed: 42 ...         | < - also kwargs found through reflections
|X|   move  [reset] device: cuda ...            |
|X|    /\   [reset] blindspot_prob: 0.05 ...    |
|X|    \/   [reset] max_epochs: 10 ...          |
|X|    	    [reset] batch_size: 4 ...           |
|X|    	    [reset] patch_size: 64 ...          |
|X|    	    [reset] learning_rate: 1e-4 ...     |
|X|    	    [reset] save_model_path: ...        |
|X|    	    [reset all]                         |
|V|    	    [add]                               |
| |    	    [delete]	                        |
| | ____________________________________________|
| | Func 3: |3d_deconv|V|                       |
| |   	    [reset] random_seed:  42 ...        |
| |    move [reset] device: cuda  ...           |
| |     /\  [reset] blindspot_prob: 0.05 ...    |
| | vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv| <- scrollable, show more funcs if scroll down
|_| ____________________________________________|





```

#### Status Bar with Expanded Log Drawer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Status: Compiling pipeline... [Click to expand]                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [2023-07-15 14:32:01] INFO: Starting compilation for plate_2                │
│ [2023-07-15 14:32:01] INFO: Validating step 'Generate Positions'            │
│ [2023-07-15 14:32:02] INFO: Validating step 'Enhance Images'                │
│ [2023-07-15 14:32:02] ERROR: Invalid memory type in function 'stack_func'   │
│ [2023-07-15 14:32:02] ERROR: Compilation failed                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Settings Dialog (Modal)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                                                                             │
│                                                                             │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │                      Settings                                │        │
│     ├─────────────────────────────────────────────────────────────┤        │
│     │ [x] Show logger in status bar                               │        │
│     │ [ ] Auto-compile on step change                             │        │
│     │ [x] Confirm before deleting steps                           │        │
│     │                                                             │        │
│     │ Editor: [vim____________] ▼                                 │        │
│     │                                                             │        │
│     │ Theme:  ( ) Light  (•) Dark  ( ) System                     │        │
│     │                                                             │        │
│     │                [Save]        [Cancel]                       │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                                                             │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### File Browser Dialog (Modal)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                                                                             │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │                     Select Plate Directory                   │        │
│     ├─────────────────────────────────────────────────────────────┤        │
│     │ Current path: /home/user/data/                              │        │
│     │                                                             │        │
│     │ [D] experiments/                                            │        │
│     │ [D] plates/                                                 │        │
│     │ [D] results/                                                │        │
│     │ [D] temp/                                                   │        │
│     │                                                             │        │
│     │ Enter path: [/home/user/data/plates/____________]          │        │
│     │                                                             │        │
│     │                [Select]       [Cancel]                      │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Draft

The visual layouts above illustrate the TUI components described in the plan files. The implementation should follow these layouts while adhering to the architectural principles and rules outlined in the plans.

Key implementation considerations:

1. **Resizable Panes**: All three main panes should be independently resizable with minimum size constraints (≥20 cols/5 rows).

2. **Consistent Navigation**: Both mouse and keyboard (Vim-style) navigation should be supported throughout the interface.

3. **Modal Dialogs**: Settings, file browser, and other dialogs should be implemented as modal overlays.

4. **Status Indicators**: Consistent status indicators should be used across the interface (○, ●, ✗, etc.).

5. **Error Display**: Errors should be prominently displayed in the Action Menu pane and detailed in the log drawer.

6. **Responsive Layout**: The layout should adapt to different terminal sizes while maintaining usability.

7. **Keyboard Shortcuts**: Common actions should have keyboard shortcuts displayed in the interface.

8. **Visual Hierarchy**: Clear visual separation between components and logical grouping of related elements.

The implementation should use prompt-toolkit's layout containers (HSplit, VSplit, etc.) to create the structure, with appropriate widgets (Button, TextArea, RadioList, etc.) for interactive elements.
