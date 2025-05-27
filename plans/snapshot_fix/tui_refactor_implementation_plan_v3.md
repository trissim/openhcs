# OpenHCS TUI Refactoring Implementation Plan (V3)

**Goal:** To refactor the OpenHCS TUI to align with the new canonical mental model (as described in `plan_tui_comprehensive_fix_v2.md` and `tui_final.md`), ensuring a clear, consistent, and robust user experience.

**Core Architectural Principles (from `plan_tui_comprehensive_fix_v2.md` and `tui_core_concepts.md`):**
*   Deferred Validation (at "Init" or "Compile").
*   Minimal "Add Plate" Dialog (path(s) only).
*   Configuration via Defaults and "Edit" dialogs.
*   Plate Status Symbols (`?`, `-`, `✓`, `✗`).
*   Plates as Orchestrators.
*   Dynamic UI from Static Reflection.
*   Centralized `storage_registry`.
*   Asynchronous Operations and `TUIState.notify` calls.

---

## Phase 1: Core Layout and Foundational Component Adjustments

**Objective:** Establish the new 3-bar TUI layout in `OpenHCSTUI` and integrate the button containers from `PlateManagerPane` and `PipelineEditorPane`. Dismantle `ActionMenuPane`.

**1.1. Refactor `OpenHCSTUI` Layout ([`openhcs/tui/tui_architecture.py`](openhcs/tui/tui_architecture.py))**
    *   **Action:** Modify `OpenHCSTUI._create_root_container()`:
        *   **Top Bar (1st):** Instantiate `MenuBar` (using `self.menu_bar` which is already initialized).
            *   File: [`openhcs/tui/tui_architecture.py:331`](openhcs/tui/tui_architecture.py:331)
        *   **Titles Bar (2nd):** Create an `HSplit` containing two `Frame` widgets for "1 Plate Manager" and "2 Pipeline Editor" titles.
            *   File: [`openhcs/tui/tui_architecture.py:332-338`](openhcs/tui/tui_architecture.py:332) (adjust existing structure)
        *   **Contextual Buttons Bar (3rd):** Create an `HSplit`. This HSplit will contain two `VSplit` children (or directly the button containers).
            *   The left side will be the container returned by `PlateManagerPane.get_buttons_container()`.
            *   The right side will be the container returned by `PipelineEditorPane.get_buttons_container()`.
            *   File: [`openhcs/tui/tui_architecture.py:339-347`](openhcs/tui/tui_architecture.py:339) (replace placeholder with actual button containers)
        *   **Main Panes:** Create an `HSplit` containing:
            *   Left: `self._get_left_pane()` (which dynamically provides `PlateManagerPane` or `DualStepFuncEditorPane`).
            *   Right: `self._get_step_viewer()` (which provides `PipelineEditorPane`).
            *   File: [`openhcs/tui/tui_architecture.py:348-352`](openhcs/tui/tui_architecture.py:348) (change `VSplit` to `HSplit`)
        *   **Bottom Bar:** Instantiate `StatusBar` (using `self.status_bar`).
            *   File: [`openhcs/tui/tui_architecture.py:353-354`](openhcs/tui/tui_architecture.py:353)
    *   **Action:** Remove any instantiation or usage of `ActionMenuPane` from `OpenHCSTUI._validate_components_present()` and `_create_root_container()`.
        *   File: [`openhcs/tui/tui_architecture.py`](openhcs/tui/tui_architecture.py) (around lines 309-311)
    *   **Verification:** `_get_left_pane()` already handles switching to `DualStepFuncEditorPane`. This part is largely aligned.

**1.2. Adapt `PlateManagerPane` ([`openhcs/tui/plate_manager_core.py`](openhcs/tui/plate_manager_core.py))**
    *   **Action:** Ensure `_initialize_ui()` correctly creates the "add", "del", "edit", "init", "compile", "run" `Button` widgets. (Already mostly done, lines 144-167).
    *   **Action:** Verify `get_buttons_container()` (lines 196-206) returns these buttons in an `HSplit` suitable for the 3rd bar.
    *   **Action:** Implement/complete handlers for these buttons (many are stubs or need orchestrator interaction logic from `ActionMenuPane`):
        *   `_show_add_plate_dialog()` (for "Add"): Already calls `PlateDialogManager`. Ensure it aligns with Phase 2.
        *   `_show_remove_plate_dialog()` (for "Del"): Already calls `PlateDialogManager`.
        *   `_on_edit_plate_clicked()`: **New.** To launch `PlateConfigEditorDialog` (Phase 4). Stub for now.
        *   `_on_init_plate_clicked()`: **Implement.** Adapt logic from `ActionMenuPane._pre_compile_handler()`. Calls `self.state.active_orchestrator.initialize()`.
        *   `_on_compile_plate_clicked()`: **Implement.** Adapt logic from `ActionMenuPane._compile_handler()`. Calls `self.state.active_orchestrator.compile_pipelines()`.
        *   `_on_run_plate_clicked()`: **Implement.** Adapt logic from `ActionMenuPane._run_handler()`. Calls `self.state.active_orchestrator.execute_compiled_plate()`.
    *   **Action:** Update `_format_plate_list()` (lines 296-389) to render new status symbols (`?`, `-`, `✓`, `✗`) and `^/v` reordering symbols.
    *   **Action:** Implement plate reordering logic in `_move_plate_up()` and `_move_plate_down()` and connect to keybindings (or new buttons if added).

**1.3. Adapt `PipelineEditorPane` (formerly `StepViewerPane`) ([`openhcs/tui/step_viewer.py`](openhcs/tui/step_viewer.py))**
    *   **Action:** Class is already renamed to `PipelineEditorPane`.
    *   **Action:** Ensure `setup()` correctly creates "add", "del", "edit", "load", "save" `Button` widgets. (Already mostly done, lines 90-94).
    *   **Action:** Verify `get_buttons_container()` (lines 120-128) returns these buttons in an `HSplit`.
    *   **Action:** Implement/complete handlers for these buttons:
        *   `_add_step()`: **Refine.** Plan suggests dialog/logic to add a new step. Currently basic.
        *   `_remove_step()`: Already has logic.
        *   `_edit_step()`: Already notifies state to trigger `DualStepFuncEditorPane`.
        *   `_load_pipeline()`: **Implement.** Adapt logic from `ActionMenuPane._load_handler()` (if it existed, or implement new). Involves file dialog and updating `active_orchestrator.pipeline_definition`.
        *   `_save_pipeline()`: **Implement.** Adapt logic from `ActionMenuPane._save_handler()`. Involves file dialog and saving `active_orchestrator.pipeline_definition`.
    *   **Action:** Update `_format_step_list()` (lines 145-191) to render `^/v` reordering symbols.
    *   **Action:** Ensure step reordering logic (`_move_step_up`, `_move_step_down`) is functional.

**1.4. Dismantle `ActionMenuPane` ([`openhcs/tui/action_menu_pane.py`](openhcs/tui/action_menu_pane.py))**
    *   **Action:** After migrating all relevant button handlers (`_pre_compile_handler`, `_compile_handler`, `_run_handler`, `_save_handler`) and settings dialog logic (see Phase 4 for `GlobalSettingsEditorDialog`), delete the `action_menu_pane.py` file.
    *   **Action:** Remove imports and references to `ActionMenuPane` from other files (e.g., `tui_architecture.py`).

---

## Phase 2: Implement "Add Plate" Flow (Minimal Dialog)

**Objective:** Simplify the "Add Plate" dialog to only collect paths and update `PlateManagerPane` to handle the new orchestrator creation flow.

**2.1. Verify `PlateDialogManager` ([`openhcs/tui/dialogs/plate_dialog_manager.py`](openhcs/tui/dialogs/plate_dialog_manager.py))**
    *   **Action:** `_create_file_browser_dialog()` (lines 205-238) already seems to *only* include `path_input` and allows multiple paths. Backend/microscope selection is removed. This is aligned.
    *   **Action:** `_handle_ok_button_press()` (lines 363-414) and `_dialog_ok()` (lines 294-324) correctly extract only path(s). This is aligned.
    *   **Outcome:** This component appears to require minimal changes for this phase.

**2.2. Update `PlateManagerPane._handle_add_dialog_result` ([`openhcs/tui/plate_manager_core.py:459-531`](openhcs/tui/plate_manager_core.py:459))**
    *   **Action:** This method already receives path(s) and creates `PipelineOrchestrator` instances using `self.state.global_config`.
    *   **Action:** Ensure it adds the new plate/orchestrator to `self.plates` with status `'not_initialized'` (or equivalent for `?` symbol). (Currently sets to `'not_initialized'`, line 500).
    *   **Action:** `PlateManagerPane` will continue to create the `PipelineOrchestrator` instance directly. It should then notify `self.state.notify('plate_orchestrator_added', {'plate_id': ..., 'orchestrator': new_orchestrator_instance, 'path': path_str})` so `OpenHCSTUILauncher` can begin tracking it. `PlateManagerPane` will manage its own list of these orchestrators for display.

**2.3. Verify `PipelineOrchestrator.initialize()` ([`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py))**
    *   **Action:** (External to TUI code, but crucial) Ensure `initialize()` performs all necessary validation (path existence, type), applies default backend/microscope from its `self.config`, and sets up the plate.
    *   **Action:** Ensure it raises exceptions on validation failure, to be caught by `PlateManagerPane._on_init_plate_clicked()` and displayed.

---

## Phase 3: Implement Dual Step/Func Editor

**Objective:** Implement the `DualStepFuncEditorPane` for editing `FuncStep` objects.

**3.1. Implement `DualStepFuncEditorPane` ([`openhcs/tui/dual_step_func_editor.py`](openhcs/tui/dual_step_func_editor.py))**
    *   **Action:** `__init__` already takes `state` and `func_step`, and creates a working copy. This is good.
    *   **Action:** Menu bar with "Step Settings", "Func Pattern", "Save", "Close" buttons is partially implemented (lines 60-85). Ensure handlers are robust.
        *   `_switch_view()`: Handles view toggling. (Implemented)
        *   `_save_changes()`: **Implement fully.** Update `self.editing_func_step` from UI values from *both* views. Notify `self.state.notify('step_pattern_saved', {'step': new_func_step})`. Handle `save_button.disabled` state based on changes.
        *   `_close_editor()`: Notifies `self.state.notify('step_editing_cancelled')`. (Implemented)
    *   **Action:** Implement "Step Settings" view (`_create_step_settings_view`, lines 112-140):
        *   Dynamically generate UI elements (Labels, TextAreas, Dropdowns for enums like `VariableComponents`, `GroupBy` from [`openhcs/core/steps/abstract.py`](openhcs/core/steps/abstract.py)) based on `AbstractStep` parameters.
        *   Populate with values from `self.editing_func_step`.
        *   Implement "Load" and "Save As" buttons for `.step` objects (persisting/loading `AbstractStep` parameters).
        *   Implement `_something_changed` (line 169) to enable/disable save button.
    *   **Action:** Implement "Func Pattern" view (`_create_func_pattern_view`, lines 142-150):
        *   Adapt logic from [`openhcs/tui/function_pattern_editor.py`](openhcs/tui/function_pattern_editor.py) for displaying and editing the function pattern (`self.editing_func_step.func`). This includes:
            *   Dynamic UI for functions and parameters.
            *   Handling list/dictionary patterns.
            *   Add/Delete/Move functions.
            *   Parameter editing.
        *   Implement "Load" and "Save As" buttons for `.func` pattern objects.
        *   Implement "Edit in Vim" button, adapting validation from `FunctionPatternEditor._edit_in_vim` and `_validate_pattern_file`.
        *   Connect changes in this view to `_something_changed`.
    *   **Action:** Ensure static inspection is used robustly for all dynamic UI generation.

**3.2. Verify `OpenHCSTUI._get_left_pane` ([`openhcs/tui/tui_architecture.py:357-412`](openhcs/tui/tui_architecture.py:357))**
    *   **Action:** This method already switches to `DualStepFuncEditorPane` when `self.state.editing_step_config` is true, passing `self.state.step_to_edit_config`. This is aligned. Ensure `step_to_edit_config` is correctly populated as a `FunctionStep` instance by `PipelineEditorPane._edit_step()`.

---

## Phase 4: Implement Global and Plate-Specific Configuration Editors

**Objective:** Provide UI for editing global and plate-specific configurations.

**4.1. Implement `GlobalSettingsEditorDialog` ([`openhcs/tui/dialogs/global_settings_editor.py`](openhcs/tui/dialogs/global_settings_editor.py))**
    *   **Action:** This dialog is launched from the "Global Settings" button in `MenuBar`.
    *   **Action:** Complete the dynamic UI generation in `_build_dialog()` (lines 43-94) for all fields in `GlobalPipelineConfig` (including nested `VFSConfig`, `PathPlanningConfig`, etc.). Use appropriate widgets (TextArea, RadioList for enums like `Backend`, `Microscope`, Checkbox for booleans).
    *   **Action:** Implement robust `_save_settings()` (lines 96-119) to update `self.editing_config` from all UI elements and notify `self.state.notify('global_config_changed', new_global_config)`.
    *   **Action:** Ensure `MenuBar._on_settings` handler correctly instantiates and shows this dialog, then passes the result to `OpenHCSTUILauncher` via `self.state.notify('global_config_needs_update', ...)`.

**4.2. Create and Implement `PlateConfigEditorDialog` (New file: `openhcs/tui/dialogs/plate_config_editor.py`)**
    *   **Action:** This dialog is launched from the "Edit" button in `PlateManagerPane` for a selected plate.
    *   **Action:** `__init__` should take `state`, and the selected `PipelineOrchestrator` instance. It should work on a copy of `orchestrator.config`.
    *   **Action:** Dynamically generate UI for parameters that can be overridden from `GlobalPipelineConfig` (e.g., specific backends, microscope type for this plate). This will involve inspecting the orchestrator's config structure (which might be a `GlobalPipelineConfig` instance or a derivative).
    *   **Action:** Implement save/cancel logic. Save should update the `orchestrator.config` (the actual instance held by the launcher or `PlateManagerPane`) and potentially trigger persistence of this plate-specific config (e.g., to a file in the plate's directory). Notify `self.state.notify('plate_config_changed', {'plate_id': ..., 'new_config': ...})`.
    *   **Action:** `PlateManagerPane._on_edit_plate_clicked()` should instantiate and show this dialog.

---

## Phase 5: Complete `async` Propagation and Final Cleanup

**Objective:** Ensure all asynchronous operations and TUI state notifications are correctly implemented.

**5.1. Complete `async` Propagation:**
    *   **Action:** Review all new and modified button handlers and other methods that perform I/O or interact with `TUIState.notify`. Ensure they are `async def`.
    *   **Action:** Ensure all calls to `self.state.notify` are `await`ed.
    *   **Action:** Pay special attention to `lambda` handlers used for buttons; they should typically call `get_app().create_background_task(self.actual_async_handler())`.
    *   **Action:** Review `TUIStatusBarLogHandler.emit` for correct async scheduling. (Current implementation in [`status_bar.py`](openhcs/tui/status_bar.py) seems to handle this).

**5.2. Address Remaining Architectural Inconsistencies (from `plan_tui_comprehensive_fix_v2.md`):**
    *   **Action:** Review `PlateValidationService.close()` ([`openhcs/tui/services/plate_validation.py`](openhcs/tui/services/plate_validation.py)). The plan mentioned a duplicate; the current code has `close` and `__del__`. The explicit `close` seems appropriate.
    *   **Action:** Verify `container` properties in `PlateManagerPane`, `PipelineEditorPane`, and `DualStepFuncEditorPane` return the main content area for each pane, excluding their specific button bars (which are handled by `OpenHCSTUI`'s 3rd bar). (Current review indicates these are mostly correct).

---

## Phase 6: Final Verification

**Objective:** Ensure the refactored TUI is stable and correct.

**6.1. Code Compilation:**
    *   **Action:** Run `python -m py_compile <file>` on all modified TUI Python files.
    *   **Action:** Run `python -m openhcs.tui` to ensure the application launches.

**6.2. Manual Testing:**
    *   Test all new layouts and button functionalities.
    *   Test "Add Plate" flow.
    *   Test "Init", "Compile", "Run" for a sample plate.
    *   Test "Edit Step" flow with `DualStepFuncEditorPane` (both Step and Func views).
    *   Test "Global Settings" dialog.
    *   Test "Edit Plate Config" dialog.
    *   Verify status updates and log messages in `StatusBar`.
    *   Verify asynchronous operations do not block the UI.

**6.3. (Optional) Automated Tests:**
    *   If unit/integration tests exist for TUI components, update or create new ones to cover refactored logic.

---

## Mermaid Diagram (Conceptual - from `plan_tui_comprehensive_fix_v2.md`)

```mermaid
graph TD
    App[OpenHCSTUILauncher] --> TUI[OpenHCSTUI]

    TUI --> TopBar[TopBar (MenuBar)]
    TUI --> TitlesBar[2ndBar (Titles)]
    TUI --> ContextButtonsBar[3rdBar (Contextual Buttons)]
    TUI --> MainPanes[HSplit: PlateMgr | PipelineEditor]
    TUI --> StatusBarPane[BottomBar (StatusBar)]

    TopBar --> GlobalSettingsBtn([Global Settings])
    TopBar --> HelpBtn([Help])

    TitlesBar --> PlateMgrTitle[Title: Plate Manager]
    TitlesBar --> PipelineEditorTitle[Title: Pipeline Editor]

    ContextButtonsBar --> PlateMgrButtons[Button Container from PlateManagerPane]
    ContextButtonsBar --> PipelineEditorButtons[Button Container from PipelineEditorPane]

    MainPanes --> PMPaneUser[PlateManagerPane]
    MainPanes --> PEPaneUser[PipelineEditorPane]
    
    subgraph "Left Main Pane Content (Dynamic)"
        PMPaneUser
        DualEditor[DualStepFuncEditorPane]
    end

    PMPaneUser -- Edit Step --> EditorSwitcher{State Change: editing_step_config}
    EditorSwitcher -- true --> DualEditor
    DualEditor -- Close --> EditorSwitcher
    EditorSwitcher -- false --> PMPaneUser


    TUIState[TUIState]

    TUI -. uses .-> TUIState
    PMPaneUser -. uses .-> TUIState
    PEPaneUser -. uses .-> TUIState
    DualEditor -. uses .-> TUIState
    TopBar -. uses .-> TUIState
    StatusBarPane -. uses .-> TUIState

    PMPaneUser -- Add Plate --> AddPlateDialog[PlateDialogManager.show_add_plate_dialog (Path Only)]
    PMPaneUser -- Edit Plate Config --> PlateConfigDialog[PlateConfigEditorDialog]
    TopBar -- Global Settings --> GlobalSettingsDialog[GlobalSettingsEditorDialog]

    PMPaneUser -- Init/Compile/Run --> OrchestratorUser[PipelineOrchestrator for selected plate]
    App -- Manages --> AllOrchestrators[Dict of PipelineOrchestrators]
    OrchestratorUser -- uses --> GlobalCfg[GlobalPipelineConfig]
    OrchestratorUser -- uses --> PlateCfg[PlateSpecificConfig (optional)]
    
    DualEditor -- Edits --> FuncStep[FuncStep]