# Plan 02: TUI Component Modularization and Responsibility Refinement

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: Several TUI classes, notably `OpenHCSTUI` (in `tui_architecture.py`), `PlateManagerPane` (in `plate_manager_core.py`), `PipelineEditorPane` (in `pipeline_editor.py`), and `DualStepFuncEditorPane` (in `dual_step_func_editor.py`), have grown large and accumulate multiple responsibilities. `OpenHCSTUI` handles main layout construction, state observation, and acts as a central hub. The pane classes manage their own complex UI, state, event handling, and sometimes direct core interactions (which Plan 01 aims to decouple via adapters). This violates the Single Responsibility Principle and makes the components difficult to understand, test, and maintain.

**Goal**: To decompose these large UI classes into smaller, more focused, and cohesive sub-components. Each new component will have a single, well-defined responsibility (e.g., displaying a list, handling a specific set of user interactions, managing a sub-section of the UI). This will improve modularity, testability, and adherence to the Law of Demeter by ensuring components primarily interact through a central `TUIState` object or a dedicated event bus, rather than direct, complex inter-dependencies.

**Architectural Principles**:
*   **Single Responsibility Principle (SRP)**: Each class/module should have one reason to change.
*   **Composition over Inheritance**: Favor composing UIs from smaller, independent components.
*   **Law of Demeter**: Minimize direct knowledge between components; interactions mediated by `TUIState` or an event bus.
*   **Information Hiding**: Internal state and implementation details of sub-components should be encapsulated.

## 2. Proposed Component Decomposition

### 2.1. `OpenHCSTUI` (from `openhcs.tui.tui_architecture.py`)

**Current Responsibilities**:
*   Overall application layout (`_create_root_container`, `_get_left_pane_with_frame`, etc.).
*   Instantiation and management of major panes (`PlateManagerPane`, `PipelineEditorPane`, `DualStepFuncEditorPane`, `StatusBar`).
*   Global key bindings.
*   Central `TUIState` management and observation hub for some cross-pane events.
*   Handling display logic for switching between editor panes (e.g., `DualStepFuncEditorPane`, `PlateConfigEditorPane`).

**Proposed Decomposition**:
*   **`openhcs.tui.layout_manager.py` -> `LayoutManager` class**:
    *   **Responsibility**: Constructing the main application layout (top bar, main content VSplit, status bar HSplit).
    *   **Interaction**: Takes instances of the primary pane components (e.g., `PlateManagerView`, `PipelineEditorView`, `StatusBarView`) and arranges them. It will no longer contain logic for *which* editor to show in the left pane; that will be handled by a dynamic container within the left pane slot, driven by `TUIState`.
    *   The methods like `_get_left_pane_with_frame`, `_get_pipeline_editor_pane_with_frame` will be simplified or moved here, focusing purely on layout assembly.
*   **`openhcs.tui.app_controller.py` -> `AppController` class (or enhance `TUIState`)**:
    *   **Responsibility**: Managing the lifecycle of primary UI components, handling global application events (like shutdown), and orchestrating high-level UI state changes (e.g., which editor pane is active).
    *   The `OpenHCSTUI`'s `__init__` logic related to component instantiation and observer registration for high-level state changes will move here.
    *   The `TUIState` will become more of a pure state holder and event bus, with `AppController` acting on those state changes to manage UI visibility.
*   **`OpenHCSTUI` itself becomes leaner**: It might remain as the top-level application entry point that instantiates `LayoutManager`, `AppController`, and `TUIState`, and starts the `prompt_toolkit` application. Its direct UI building methods will be significantly reduced.

### 2.2. `PlateManagerPane` (from `openhcs.tui.plate_manager_core.py`)

**Current Responsibilities**:
*   Displaying list of plates (`_build_plate_items_container`, `_get_plate_display_text`).
*   Handling selection, addition, removal, reordering of plates.
*   Managing its own action buttons (Add, Del, Edit, Init, Compile, Run).
*   Interacting with `PlateDialogManager` and `PlateValidationService`.
*   Directly invoking `PipelineOrchestrator` methods (to be addressed by Plan 01).

**Proposed Decomposition**:
*   **`openhcs.tui.components.plate_list_view.py` -> `PlateListView` class**:
    *   **Responsibility**: Displaying the list of plates using `InteractiveListItem`. Handles rendering, selection highlighting, and scrollability.
    *   **Interaction**: Takes `List[CorePlateData]` from `TUIState` (via `PlateManagerController`). Emits selection events.
*   **`openhcs.tui.components.plate_actions_toolbar.py` -> `PlateActionsToolbar` class**:
    *   **Responsibility**: Displaying and handling actions for the plate list (Add, Del, Edit, Init, Compile, Run buttons).
    *   **Interaction**: Dispatches `Command` objects when buttons are clicked. Button enablement driven by `TUIState`.
*   **`openhcs.tui.controllers.plate_manager_controller.py` -> `PlateManagerController` class**:
    *   **Responsibility**: Orchestrating the `PlateListView` and `PlateActionsToolbar`. Responding to `TUIState` changes related to plates (e.g., new plate added, plate status changed). Fetches plate data via `CoreApplicationAdapterInterface`. Handles logic for invoking commands related to plate management.
    *   The current `PlateManagerPane` class will evolve into this controller, shedding its direct UI rendering responsibilities.
    *   Dialog interactions (via `PlateDialogManager`) and validation (via `PlateValidationService`) will be managed here.

### 2.3. `PipelineEditorPane` (from `openhcs.tui.pipeline_editor.py`)

**Current Responsibilities**:
*   Displaying list of steps for the active pipeline.
*   Handling selection, addition, removal, reordering of steps.
*   Managing its own action buttons (Add, Del, Edit, Load, Save).
*   Interacting with `TUIState` to show step editor.

**Proposed Decomposition**:
*   **`openhcs.tui.components.step_list_view.py` -> `StepListView` class**:
    *   **Responsibility**: Displaying the list of steps using `InteractiveListItem`. Handles rendering, selection highlighting.
    *   **Interaction**: Takes `List[CoreStepData]` from `TUIState` (via `PipelineEditorController`). Emits selection events.
*   **`openhcs.tui.components.pipeline_actions_toolbar.py` -> `PipelineActionsToolbar` class**:
    *   **Responsibility**: Displaying and handling actions for the step list (Add, Del, Edit, Load, Save buttons).
    *   **Interaction**: Dispatches `Command` objects. Button enablement driven by `TUIState`.
*   **`openhcs.tui.controllers.pipeline_editor_controller.py` -> `PipelineEditorController` class**:
    *   **Responsibility**: Orchestrating `StepListView` and `PipelineActionsToolbar`. Responding to `TUIState` changes (e.g., active plate changed, steps updated). Fetches step data for the active plate via `CoreOrchestratorAdapterInterface`. Handles logic for invoking commands related to pipeline/step editing.
    *   The current `PipelineEditorPane` class will evolve into this controller.

### 2.4. `DualStepFuncEditorPane` (from `openhcs.tui.dual_step_func_editor.py`)

**Current Responsibilities**:
*   Manages two sub-views: "Step Settings" and "Func Pattern".
*   Builds UI dynamically for `AbstractStep` parameters.
*   Integrates `FunctionPatternEditor` for `func` attribute.
*   Handles saving/loading of individual `FuncStep` objects.

**Proposed Decomposition**:
*   **`openhcs.tui.components.step_settings_editor.py` -> `StepSettingsEditorView` class**:
    *   **Responsibility**: Dynamically building and managing UI widgets for editing parameters of an `AbstractStep` (or `CoreStepData`). Handles input validation and state updates for these parameters.
    *   **Interaction**: Takes `CoreStepData` and its schema/signature. Emits events when parameters change.
*   **`openhcs.tui.components.func_pattern_view.py` -> `FuncPatternView` class**:
    *   **Responsibility**: Encapsulates the existing `FunctionPatternEditor` component, providing a consistent interface for the `DualStepFuncEditorController`.
    *   **Interaction**: Manages the `FunctionPatternEditor` instance.
*   **`openhcs.tui.controllers.dual_editor_controller.py` -> `DualEditorController` class**:
    *   **Responsibility**: Manages the visibility and interaction between `StepSettingsEditorView` and `FuncPatternView`. Handles the overall "Save" and "Close" logic for the dual editor. Loads the `FuncStep` (via `CoreStepData`) to be edited from `TUIState`. Dispatches commands for saving the step or its pattern.
    *   The current `DualStepFuncEditorPane` will evolve into this controller.

## 3. State Management and Communication (`TUIState` and Event Bus)

*   **`TUIState` (`openhcs.tui.tui_architecture.TUIState`)**:
    *   Will continue to be the central observable state holder.
    *   Attributes like `editing_step_config`, `step_to_edit_config`, `editing_plate_config`, `orchestrator_for_plate_config_edit` will determine which editor/view is active in dynamic layout areas.
    *   It will store lists of `CorePlateData` and `CoreStepData` (for the active plate) that views will observe.
*   **Event Bus (potentially enhancing `TUIState.notify`)**:
    *   UI components (e.g., `PlateListView`, `StepListView`) will emit fine-grained events (e.g., `plate_selected(plate_id)`, `step_action_requested(command_name, step_id)`).
    *   Controller components (`PlateManagerController`, `PipelineEditorController`, `DualEditorController`, `AppController`) will subscribe to relevant events from `TUIState` and from their managed view components.
    *   Commands will be dispatched by controllers or action toolbars, and their execution (via adapters) will result in `TUIState` updates, triggering view refreshes.

## 4. Refactoring Steps (High-Level)

1.  **Create New Component Files**: Create the new Python files for the decomposed views and controllers as outlined above (e.g., `plate_list_view.py`, `plate_manager_controller.py`, etc., within `openhcs.tui.components` and `openhcs.tui.controllers` sub-packages).
2.  **Migrate UI Rendering Logic**:
    *   Move UI element creation (e.g., `_build_plate_items_container` from `PlateManagerPane` to `PlateListView`) and associated display logic (e.g., `_get_plate_display_text`) into the new view components.
    *   Views will become primarily responsible for rendering data they receive and emitting user interaction events.
3.  **Migrate Control Logic**:
    *   Move event handling, state management logic, and command dispatching logic from the old large pane classes into the new controller classes.
    *   Controllers will observe `TUIState`, fetch data via core adapters (as per Plan 01), prepare data for their views, and handle actions initiated by views or toolbars.
4.  **Refactor `OpenHCSTUI`**:
    *   Delegate layout construction to `LayoutManager`.
    *   Delegate component lifecycle and high-level state orchestration to `AppController`.
5.  **Update `TUIState`**:
    *   Ensure `TUIState` holds data in the form of `CorePlateData` and `CoreStepData` where appropriate.
    *   Refine event types for more granular communication if needed.
6.  **Testing**:
    *   Write unit tests for new view components (testing rendering based on input data, and event emission).
    *   Write unit tests for new controller components (testing state handling, command dispatch, and interaction with mocked adapters and views).
    *   Update/create integration tests to ensure composed components work together correctly.

## 5. Verification

*   **Code Structure**: Verify that the new file structure reflects the decomposition.
*   **Class Size and Responsibility**: Check that the new classes are smaller and have more focused responsibilities (e.g., using `wily` or manual inspection).
*   **Reduced Coupling**: Analyze dependencies (e.g., using `tools/code_analysis/code_analyzer_cli.py dependencies`) to ensure new view components primarily depend on `TUIState` or their controller, and controllers depend on `TUIState` and core adapter interfaces. Direct dependencies between view components should be minimized.
*   **Test Coverage**: Ensure adequate test coverage for the new and refactored components.
*   **Functional Equivalence**: The TUI should retain its existing functionality after refactoring.

This plan aims to create a more maintainable, testable, and understandable TUI codebase by adhering to established software design principles.