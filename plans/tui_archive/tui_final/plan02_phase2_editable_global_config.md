# Plan: TUI Action Menu - Phase 2: Editable and Persisted GlobalPipelineConfig

**Date:** 2025-05-22

**Author:** Roo (Architect Mode)

**Depends On:** Completion of Phase 1 ([`plans/tui_final/plan01_phase1.md`](plans/tui_final/plan01_phase1.md:1))

## 1. Goal

Enable users to view, modify, apply, and persist changes to `GlobalPipelineConfig` via the TUI settings dialog. This will make the core application configuration manageable through the user interface.

## 2. Key Tasks

### Task 2.1: Enhance Settings Dialog UI for Editable `GlobalPipelineConfig`
**File:** `openhcs/tui/action_menu_pane.py`
**Method:** `_create_main_settings_dialog()`

*   Modify the `GlobalPipelineConfig` section of the settings dialog.
*   Replace existing read-only `Label` widgets with appropriate editable `prompt_toolkit` input widgets for each field in `GlobalPipelineConfig` and its nested dataclasses (`VFSConfig`, `PathPlanningConfig`, `GPUMemoryConfig`). Examples:
    *   `TextArea(multiline=False)` for string and numeric types (e.g., `num_workers`, `vfs.persistent_storage_root_path`).
    *   `Checkbox` for boolean types.
    *   `Dropdown` or `RadioList` for fields with enumerated choices (e.g., `vfs.default_intermediate_backend`, using choices from `openhcs.core.config.Backend`).
*   Ensure these widgets are populated with the current values from `self.global_config` when the dialog opens.
*   Add or repurpose a dialog button (e.g., "Save Global Config" or "Apply & Save Global Config") to trigger the saving and application process.

### Task 2.2: Implement `GlobalPipelineConfig` Modification and Application Logic
**File:** `openhcs/tui/action_menu_pane.py`

*   Create a new handler method for the "Save Global Config" button (e.g., `async def _apply_and_save_global_config_handler(self)`).
*   **Collect Data:** Retrieve current values from all editable `GlobalPipelineConfig` widgets in the dialog.
*   **Validate Data:** Implement input validation (e.g., numeric types, valid path characters, valid enum choices). Display errors within the dialog if validation fails.
*   **Create New `GlobalPipelineConfig` Instance:** If validation succeeds, construct a new, frozen `GlobalPipelineConfig` object using the validated values.
*   **Notify for Propagation:** Emit an event via `self.state.notify('global_config_needs_update', new_global_config_instance)`. This event will be handled by `OpenHCSTUILauncher`.
*   **Persist Configuration:** Call a new helper method `_persist_global_config_to_file(new_global_config_instance)` (see Task 2.3).
*   Provide user feedback (e.g., "Global settings applied and saved.") and close the settings dialog.

### Task 2.3: Implement `GlobalPipelineConfig` Persistence
*   **Saving Logic (e.g., in `action_menu_pane.py` or a new utility module):**
    *   Method: `_persist_global_config_to_file(self, config_to_save: GlobalPipelineConfig)`
    *   Define a user-specific configuration file path (e.g., `~/.config/openhcs/global_pipeline_config.yaml` or `.json`). Use `pathlib.Path.home()`.
    *   Serialize `config_to_save` to YAML or JSON (e.g., using `dataclasses.asdict` then `yaml.dump` or `json.dump`).
    *   Use `self.filemanager.write_file()` to save the serialized data.
*   **Loading Logic (Modify `openhcs/tui/__main__.py`):**
    *   During application startup, before `GlobalPipelineConfig` is passed to `OpenHCSTUILauncher`:
        *   Attempt to load the persisted configuration file.
        *   If the file exists and is valid: Deserialize it and create a `GlobalPipelineConfig` instance. Implement robust loading to handle missing/new fields gracefully (e.g., by merging with defaults).
        *   If the file doesn't exist or is invalid, fall back to `get_default_global_config()`.
        *   Pass the resulting `GlobalPipelineConfig` (loaded or default) to `OpenHCSTUILauncher`.

### Task 2.4: Implement `GlobalPipelineConfig` Propagation
**File:** `openhcs/tui/tui_launcher.py`

*   **Event Handling in `OpenHCSTUILauncher.__init__`:**
    *   Register an observer for the `'global_config_needs_update'` event from `TUIState`. The callback will be `self._handle_global_config_update`.
*   **New Method in `OpenHCSTUILauncher`:** `async def _handle_global_config_update(self, new_config: GlobalPipelineConfig)`:
    *   Update its primary copy: `self.core_global_config = new_config`.
    *   Update the `global_config` in `self.initial_tui_context` (passed to `OpenHCSTUI`). This may require `self.initial_tui_context.global_config = new_config` or a more careful update if the context's reference is immutable.
    *   **Update Active `PipelineOrchestrator` Instances:** Iterate through `self.orchestrators`. For each active orchestrator:
        *   Call a new method `orchestrator.apply_new_global_config(new_config)` (see Task 2.5).
        *   Decision: Determine if changes apply live or if orchestrators need re-initialization. Live update is preferred.

### Task 2.5: Update `PipelineOrchestrator` for Live Config Changes
**File:** `openhcs/core/orchestrator/orchestrator.py`

*   Add a new method: `apply_new_global_config(self, new_config: GlobalPipelineConfig)`.
*   This method will:
    *   Store the `new_config` as `self.global_config`.
    *   Re-initialize or update internal components that depend on `GlobalPipelineConfig`. This includes:
        *   `self.path_planner`
        *   `self.materialization_flag_planner`
        *   Any other component that caches or uses parts of `global_config`.

## 3. Diagram: Config Update and Propagation Flow

```mermaid
graph TD
    subgraph SettingsDialog [ActionMenuPane - Settings Dialog]
        SD_UI[Editable Global Config UI] -- User Edits & Saves --> SD_Handler[_apply_and_save_global_config_handler];
        SD_Handler -- Validates & Creates --> NewGPC_Instance[New GlobalPipelineConfig instance];
        SD_Handler -- Calls --> Persist[_persist_global_config_to_file];
        Persist -- Uses self.filemanager & saves NewGPC_Instance --> UserConfigFile[User Config File (e.g., YAML/JSON)];
        SD_Handler -- Emits Event self.state.notify --> ConfigUpdateEvent[('global_config_needs_update', NewGPC_Instance)];
    end

    ConfigUpdateEvent --> TUI_Launcher_Observer;

    subgraph TUI_Launcher [openhcs.tui.tui_launcher.OpenHCSTUILauncher]
        TUI_Launcher_Observer[Observer for 'global_config_needs_update'] -- Receives NewGPC_Instance --> HandleUpdate[_handle_global_config_update];
        HandleUpdate -- Updates --> Launcher_MasterGPC[self.core_global_config];
        HandleUpdate -- Updates --> Launcher_InitialContextGPC[self.initial_tui_context.global_config];
        HandleUpdate -- Calls apply_new_global_config on --> ActiveOrchs[Active PipelineOrchestrator Instances];
    end
    
    subgraph Pipeline_Orchestrator [openhcs.core.orchestrator.PipelineOrchestrator]
        ActiveOrchs -- apply_new_global_config(NewGPC_Instance) --> Orch_InternalUpdate[Updates internal components like PathPlanner];
    end

    subgraph AppStartup [openhcs.tui.__main__.py]
        AppStartup -- On Start, Loads --> UserConfigFile;
        AppStartup -- Creates Initial GPC (from file or default) --> InitialGPC_To_Launcher;
        InitialGPC_To_Launcher -- Passed to --> TUI_Launcher;
    end
```

## 4. Considerations

*   **Mutability of `ProcessingContext.global_config`:** If `ProcessingContext.global_config` cannot be directly reassigned, `OpenHCSTUILauncher` might need to recreate `initial_tui_context` or `OpenHCSTUI` might need a method to update its `global_config` reference.
*   **Complexity of Live Updates to `PipelineOrchestrator`:** Thoroughly test the impact of live config updates on ongoing or future operations of `PipelineOrchestrator`.
*   **User Experience:** Ensure clear feedback to the user when settings are applied and if a restart is recommended for certain fundamental changes (though the goal is live apply).
*   **Error Handling:** Robust error handling for file I/O (loading/saving config) and data validation.

This plan for Phase 2 provides a clear path to making `GlobalPipelineConfig` fully manageable through the TUI.