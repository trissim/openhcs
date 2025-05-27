# OpenHCS TUI - Visual User Workflow Simulation

This document illustrates the key user interactions and corresponding TUI states based on the design in `tui_final.md`.

## Frame 1: Initial Application Launch

**User Action**: User launches the OpenHCS TUI.

**TUI State**:
*   No plates loaded.
*   No steps visible.
*   Focus is likely on the Plate Manager or its `[add]` button.

**Visual Representation**:

```
__________________________________________________________________________________________
| |_[_Global_Settings_]_[_Help]_|_OpenHCS_V1.0___________________________________________|
| |_____________1_plate_manager______________________|_|__2_Pipeline_editor______________|
|_|[add]_[del]_[edit]_[init]_[compile]_[run]_________|_|[add]_[del]_[edit]_[load]_[save]_|
| |                                                  | |                                 | 
| | (No plates. Click [add] to add a plate)          | | (No plate selected)             | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
|_|_________________________________________________ |_|_________________________________|
|_Status:_Ready______________________________________|_|_________________________________|
```

**Internal State Notes**:
*   `TUIState.selected_plate` is `None`.
*   `PlateManagerPane.plates` list is empty.
*   `PipelineEditorPane.steps` list is empty.

---
## Frame 2: Adding a Plate

**User Action**: Clicks the `[add]` button in the Plate Manager's contextual button bar. A file dialog appears (not shown here, as it's a system dialog). User selects one or more directories (e.g., `/path/to/plate1`).

**TUI State**:
*   A new plate entry is added to the Plate Manager list with `?` status.
*   The new plate is selected.
*   Pipeline Editor shows context for the new, empty pipeline.

**Visual Representation**:

```
__________________________________________________________________________________________
| |_[_Global_Settings_]_[_Help]_|_OpenHCS_V1.0___________________________________________|
| |_____________1_plate_manager______________________|_|__2_Pipeline_editor______________|
|_|[add]_[del]_[edit]_[init]_[compile]_[run]_________|_|[add]_[del]_[edit]_[load]_[save]_|
|?| ^/v 1: plate1              | (/path/to/...)     | | (Pipeline for plate1 is empty.  | 
| |                                                  | | Click [add] to add a step.)     | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
|_|_________________________________________________ |_|_________________________________|
|_Status:_Plate 'plate1' added. Ready for init._____|_|_________________________________|
```

**Internal State Notes**:
*   `PlateManagerPane._show_add_plate_dialog()` is called.
*   `PlateDialogManager` handles path selection.
*   `PlateManagerPane._handle_add_dialog_result()` creates a new `PipelineOrchestrator` for `/path/to/plate1` with default global config.
*   New plate data (name: "plate1", path: "/path/to/plate1", status: "?", orchestrator: new_instance) is added to `PlateManagerPane.plates`.
*   `TUIState.notify('plate_added', ...)` and `TUIState.set_selected_plate(...)` are called.
*   `PipelineEditorPane` updates to reflect the new empty pipeline of the selected plate.

---