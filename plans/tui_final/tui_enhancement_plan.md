# TUI Enhancement and Feature Completion Plan

This plan outlines the tasks to refine the OpenHCS TUI, aligning it with the detailed user diagram and functional requirements.

## Phase 0: Prerequisite - Ensure Core TUI Structure for Main Layout

**Target Files**: 
- `openhcs/tui/tui_launcher.py`
- `openhcs/tui/menu_bar.py`
- `openhcs/tui/plate_manager_core.py`
- `openhcs/tui/step_viewer.py` (PipelineEditorPane)
- `openhcs/tui/status_bar.py`

**Sub-Tasks:**

- [x] **0.1: Top Horizontal Bar (Global Menu)**
    - [x] `MenuBar` in `openhcs/tui/menu_bar.py` defines "Global Settings" and "Help" items.
    - [x] "Global Settings" item handler dispatches to `ShowGlobalSettingsDialogCommand`.
    - [x] "Help" item handler dispatches to `ShowHelpCommand`.
    - [x] "OpenHCS_V1.0" label implemented in `OpenHCSTUI._create_root_container`.
- [x] **0.2: Second Horizontal Bar (Pane Titles)**
    - [x] `OpenHCSTUI._create_root_container` creates `VSplit` for Plate Manager and Pipeline Editor.
    - [x] Panes are `Frame` instances with titles "1 Plate Manager" and "2 Pipeline Editor".
- [x] **0.3: Third Horizontal Bar (Action Button Bars)**
    - [x] `PlateManagerPane` has `get_buttons_container()` for its buttons.
    - [x] `PipelineEditorPane` has `get_buttons_container()` for its buttons (layout confirmed horizontal).
    - [x] `OpenHCSTUI._create_root_container` integrates these button bars.
- [x] **0.4: List Panes (Left: Plates, Right: Steps)**
    - [x] `PlateManagerPane` displays scrollable list of plates (status symbol, reorder arrows, index, name, path) via `InteractiveListItem` and `ScrollablePane`.
    - [x] `PipelineEditorPane` displays scrollable list of steps (status symbol, reorder arrows, index, name) via `InteractiveListItem` and `ScrollablePane`.
- [x] **0.5: Bottom Status Bar**
    - [x] `StatusBar` component is present and integrated at the bottom of the TUI layout.

## Phase 1: Plate Manager Functionality

**Target File**: `openhcs/tui/plate_manager_core.py` (and related command/dialog files)

**Sub-Tasks:**

- [x] **1.1: "add" Button Functionality**
    - [x] Handler triggers `ShowAddPlateDialogCommand`.
    - [x] Dialog allows multiple folder selections via `FileManagerBrowser` (multi-select implemented).
    - [x] `PlateManagerPane._handle_add_dialog_result` creates `PipelineOrchestrator` for each selection using global config.
    - [x] Updates `PlateManagerPane.plates` list and UI (with `not_initialized` status, typically displayed as `?`).
    - [x] Notifies `TUIState` (`plate_orchestrator_added`, `plate_selected`).
- [x] **1.2: "del" Button Functionality**
    - [x] `DeleteSelectedPlatesCommand` gets confirmation and notifies `delete_plates_requested`.
    - [x] `PlateManagerPane` observes `delete_plates_requested`, removes plates from its list, updates UI.
    - [x] Notifies `TUIState` (`plate_removed` for each, which `TUILauncher` observes to clean up orchestrators).
- [x] **1.3: "edit" Button Functionality (Plate-Specific Config)**
    - [x] `ShowEditPlateConfigDialogCommand` notifies `TUIState` to show editor.
    - [x] `OpenHCSTUI` switches view to `PlateConfigEditorPane`, passing selected orchestrator.
    - [x] `PlateConfigEditorPane` loads orchestrator's config (or global as base), allows editing of `GlobalPipelineConfig` fields.
    - [x] "Save" in `PlateConfigEditorPane` updates `orchestrator.config` with the edited (deep-copied) configuration and notifies `TUIState` to close editor.
- [x] **1.4: "init" Button Functionality**
    *   [x] `InitializePlatesCommand` calls `orchestrator.initialize()` for selected plates (run in executor).
    *   [x] Notifies `plate_status_changed` ("initialized" or "error_init"); `PlateManagerPane` updates UI symbol.
- [x] **1.5: "compile" Button Functionality**
    *   [x] `CompilePlatesCommand` calls `orchestrator.compile_pipelines(orchestrator.pipeline_definition)` for selected plates (run in executor).
    *   [x] `compile_pipelines` performs step instantiation, validation, etc.
    *   [x] Notifies `plate_status_changed` ("compiled_ok" or "error_compile"); `PlateManagerPane` updates UI symbol. `TUIState.is_compiled` updated.
- [x] **1.6: "run" Button Functionality**
    *   [x] `RunPlatesCommand` calls `orchestrator.execute_compiled_plate(orchestrator.pipeline_definition, orchestrator.last_compiled_contexts)` for selected plates (run in executor).
    *   [x] Notifies `plate_status_changed` ("run_completed" or "error_run"); `PlateManagerPane` updates UI symbol. `TUIState.is_running` updated.

## Phase 2: Dual STEP/FUNC Editor System - View Switching

**Target Files**: 
- `openhcs/tui/tui_launcher.py`
- `openhcs/tui/step_viewer.py` (PipelineEditorPane)
- `openhcs/tui/dual_step_func_editor.py`

**Sub-Tasks:**

- [x] **2.1: Verify/Implement View Switching Logic**
    - [x] "edit" in `PipelineEditorPane` (via `ShowEditStepDialogCommand` & `_handle_edit_step_request`) sets `TUIState.editing_step_config = True` and `TUIState.step_to_edit_config`.
    - [x] `OpenHCSTUI._get_left_pane()` dynamically returns `DualStepFuncEditorPane` or `PlateConfigEditorPane` or `PlateManagerPane` based on `TUIState`.
    - [x] "Close" in `DualStepFuncEditorPane` notifies `step_editing_cancelled`; `OpenHCSTUI._handle_step_editing_cancelled` resets state.

## Phase 3: Visual Details - `|X|` Markers (Lower Priority)

**Target Files**: `PlateManagerPane`, `PipelineEditorPane`, `DualStepFuncEditorPane`, `FunctionPatternEditor`.

**Sub-Tasks:**

- [x] **3.1: Implement `|X|` Markers**
    - [x] Modified `_get_plate_display_text` in `PlateManagerPane` and `_get_step_display_text` in `PipelineEditorPane` to prepend "X | " or "  | " based on selection state, for use with `InteractiveListItem`.

## Completed Enhancements (from previous session)

- **Step Settings Editor (`DualStepFuncEditorPane`):**
    - [x] Dropdown for `variable_components`.
    - [x] Dropdown for `group_by`.
    - [x] Individual "[Reset]" buttons for each step setting.
    - [x] Toolbar for ".step" Load/Save buttons.
- **Func Pattern Editor (`FunctionPatternEditor`):**
    - [x] Header Toolbar for ".func" Load/Save/Add Func buttons.
    - [x] Load/Save functionality for `.func` pattern files.