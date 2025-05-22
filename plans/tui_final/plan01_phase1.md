# Plan: TUI Action Menu - Phase 1

**Date:** 2025-05-22

**Author:** Roo (Architect Mode)

## 1. Goal

Implement a foundational `ActionMenuPane` for the OpenHCS TUI. This initial phase will focus on:
1.  Displaying and allowing edits for TUI-specific settings (e.g., Vim mode, TUI log level, editor path). These settings will be managed as part of the global `TUIState`.
2.  Providing a read-only display of the `GlobalPipelineConfig` values.
3.  Establishing the basic structure for all action buttons in the menu, with most handlers being stubs for future implementation.

This phase aims to integrate the `ActionMenuPane` into the existing TUI structure and provide initial visibility into system configurations.

## 2. Key Components & Interactions

The following diagram illustrates the primary components involved and the flow of configuration/state data for this phase:

```mermaid
graph TD
    A[__main__.py] -- creates & passes --> L(OpenHCSTUILauncher);
    L -- creates --> GPC[GlobalPipelineConfig instance];
    L -- creates --> FM[FileManager instance];
    L -- creates & will extend --> TS[TUIState (global, with TUI-specific settings)];
    L -- creates initial_context with GPC, FM --> ITC[ProcessingContext (initial_tui_context)];
    L -- instantiates --> OHT(OpenHCSTUI);
    
    OHT -- receives --> ITC;
    OHT -- receives --> TS;
    OHT -- receives --> FM;
    OHT -- receives --> GPC;
    
    OHT -- instantiates --> AMP(ActionMenuPane);
    AMP -- receives --> TS;
    AMP -- receives --> ITC;
    
    AMP -- accesses .global_config from ITC --> GPC;
    AMP -- accesses/modifies TUI-specific settings in --> TS;

    AMP -- SettingsButton Click --> SD[Settings Dialog];
    SD -- displays (read-only) values from --> GPC;
    SD -- displays/edits TUI-specific settings from --> TS;
```

## 3. Detailed Steps for Phase 1

### Step 3.1: Extend `TUIState`
**File:** `openhcs/tui/tui_architecture.py`

*   Modify the existing `TUIState` class.
*   Add new attributes for TUI-specific settings. Examples:
    *   `vim_mode: bool = False` (or a default from environment/config file later)
    *   `tui_log_level: str = "INFO"` (with potential validation against `logging` levels)
    *   `editor_path: str = os.environ.get('EDITOR', 'vim')`
*   Ensure `TUIState`'s observer pattern (`add_observer`, `notify`) can handle changes to these new settings if other components need to react to them.

### Step 3.2: Create `ActionMenuPane` Class
**New File:** `openhcs/tui/action_menu_pane.py`

*   Define the `ActionMenuPane` class.
*   **Imports:** Include necessary `prompt_toolkit` widgets, `TUIState`, `ProcessingContext`, `GlobalPipelineConfig`.
*   **`__init__(self, state: TUIState, initial_tui_context: ProcessingContext)`:**
    *   Store `self.state = state`.
    *   Store `self.global_config = initial_tui_context.global_config`.
    *   Store `self.filemanager = initial_tui_context.filemanager`.
    *   Initialize any internal component state (e.g., for error banners, status indicators as per `plan_04_action_menu.md`).
    *   Call `self._create_buttons()` and set up the main container for the pane.
*   **`_create_buttons(self) -> List[Container]`:**
    *   Create `prompt_toolkit.widgets.Button` instances for all actions shown in the `tui.md` sketch:
        *   `[ add ]` (e.g., Add Plate/Pipeline)
        *   `[ pre-compile ]`
        *   `[ compile ]`
        *   `[ run ]`
        *   `[ save ]` (e.g., Save Pipeline Definition)
        *   `[ test ]`
        *   `[ settings ]`
    *   Assign handlers to each button. Most handlers (except for `_settings_handler`) will initially be stubs (e.g., `async def _compile_handler(self): self._show_error("Compile: Not yet implemented.")`).
    *   The `[ settings ]` button will call `self._settings_handler()`.
    *   Arrange buttons vertically with separators as per `plan_04_action_menu.md`.
*   **`_settings_handler(self)`:**
    *   This asynchronous method will call `self._create_main_settings_dialog()` and display it as a modal float.
*   **`_create_main_settings_dialog(self) -> Dialog`:**
    *   Constructs and returns a `prompt_toolkit.widgets.Dialog`.
    *   **Dialog Title:** "Settings"
    *   **Dialog Body (HSplit):**
        *   **Section 1: TUI-Specific Settings (Editable)**
            *   `Checkbox` for `vim_mode` (reads from/updates `self.state.vim_mode`).
            *   `RadioList` or `Dropdown` for `tui_log_level` (reads from/updates `self.state.tui_log_level`).
            *   `TextArea(multiline=False)` for `editor_path` (reads from/updates `self.state.editor_path`).
        *   **Section 2: Global Pipeline Configuration (Read-Only)**
            *   Add a clear title label, e.g., "Global Pipeline Configuration (Read-Only)".
            *   Iterate through `self.global_config` (and its nested `VFSConfig`, `PathPlanningConfig`, `GPUMemoryConfig` dataclasses).
            *   For each field, create a `VSplit` containing a `Label` for the field name and a `Label` (or `TextArea(read_only=True)`) for its current value.
            *   Group these into logical sub-sections within the dialog.
    *   **Dialog Buttons:**
        *   `Button("Save TUI Settings", handler=self._save_tui_settings_handler)`: This handler will take the current values from the TUI settings widgets and update `self.state` (which should then notify observers if needed).
        *   `Button("Close", handler=lambda: get_app().exit_dialog())`: Closes the dialog.
*   **Helper methods:** `_show_error(self, message: str)` for displaying errors, `_update_ui(self)` if button states need to change based on `self.state`.

### Step 3.3: Integrate `ActionMenuPane` into `OpenHCSTUI`
**File:** `openhcs/tui/tui_architecture.py`

*   **Import:** Add `from openhcs.tui.action_menu_pane import ActionMenuPane`.
*   **`OpenHCSTUI.__init__`:** Ensure it correctly receives and stores `self.state: TUIState` and `self.context: ProcessingContext` (the `initial_tui_context`). This is already done from previous refactoring.
*   **`OpenHCSTUI._validate_components_present(self)`:**
    *   Replace the line `self.action_menu = ActionMenuPane(self.state, self.context)` (which currently uses the stub) with the instantiation of the new `ActionMenuPane` class:
        `self.action_menu = ActionMenuPane(state=self.state, initial_tui_context=self.context)`
*   **`OpenHCSTUI` Layout Method (e.g., `_get_right_pane` or `_create_root_container`):**
    *   Ensure this method returns `self.action_menu.container` to place the actual `ActionMenuPane` in the TUI's rightmost pane.

## 4. Future Phases (Brief Overview)

*   **Phase 2: Editable `GlobalPipelineConfig`:**
    *   Modify the settings dialog to make `GlobalPipelineConfig` fields editable.
    *   Implement validation and the creation of a new `GlobalPipelineConfig` instance.
    *   Design and implement the propagation mechanism for the new config instance (notifying `OpenHCSTUILauncher` to update itself, `initial_tui_context`, and active `PipelineOrchestrator` instances).
*   **Phase 3: Persistent `GlobalPipelineConfig`:**
    *   Save user-modified `GlobalPipelineConfig` to a file (e.g., YAML/JSON in a user config directory).
    *   Load these settings in `__main__.py` on startup.
*   **Phase 4: Implement Other Action Menu Button Functionalities:**
    *   Develop the logic for `[ add ]`, `[ pre-compile ]`, `[ compile ]`, `[ run ]`, `[ save ]`, `[ test ]` buttons, integrating with `PipelineOrchestrator` and other TUI components.

This Phase 1 plan provides a solid step towards a more functional TUI and integrates the viewing of the new `GlobalPipelineConfig`.