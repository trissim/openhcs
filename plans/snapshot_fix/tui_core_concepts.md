# OpenHCS TUI Core Architectural Concepts

This document outlines the fundamental architectural principles and core concepts that underpin the OpenHCS TUI. A clear understanding of these concepts is crucial for any agent or developer working with the system, as they define the "pure" and predictable behavior of the application.

## 1. Plates as Orchestrators

*   **Concept**: In OpenHCS, each "plate" (representing a dataset or experimental unit, typically a directory on disk) is managed by its own dedicated `PipelineOrchestrator` instance.
*   **Implication**: A plate, in the context of its processing lifecycle, *is* an orchestrator. The `PlateManagerPane` in the TUI is responsible for managing these plate/orchestrator instances.
*   **Key Component**: [`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py) (`PipelineOrchestrator` class).

## 2. Pipelines of Steps

*   **Concept**: Each `PipelineOrchestrator` (and thus each plate) can have a "pipeline," which is a sequential list of `Step` objects. These steps define the ordered operations to be performed on the plate's data.
*   **Implication**: The TUI's "Pipeline Editor" (formerly "Step Viewer") visually represents and allows manipulation of this sequence of steps.
*   **Key Components**: [`openhcs/core/steps/abstract.py`](openhcs/core/steps/abstract.py) (`AbstractStep` class), [`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py) (`FunctionStep` class).

## 3. Steps with Function Patterns (The Dual STEP/FUNC Editor System)

*   **Concept**: A `Step` (specifically a `FunctionStep`) encapsulates a "pattern" of functions. This pattern defines the specific processing logic for that step, allowing for complex, chained operations. The `FunctionStep` itself extends `AbstractStep`, inheriting its configurable parameters.
*   **Implication**: The `DualStepFuncEditorPane` (formerly `FunctionPatternEditor`) is designed to allow users to configure these steps and their function patterns through two distinct views:
    *   **"Step Menu"**: This view allows editing of parameters defined in `AbstractStep` (e.g., `name`, `input_dir`, `output_dir`, `force_disk_output`, `variable_components`, `group_by`). These parameters are optional and provide high-level control over the step's behavior.
    *   **"Func Menu"**: This view is an editor for the actual `func` pattern object, which is the *only* and *required* input of `FunctionStep`. This `func` pattern can take various forms, allowing for flexible and powerful processing definitions:
        *   A single callable function.
        *   A tuple `(callable, dict_kwargs)` where `dict_kwargs` are keyword arguments for the callable.
        *   A list of callables or `(callable, dict_kwargs)` tuples, representing a sequence of functions to be executed.
        *   A dictionary mapping string keys to callables, `(callable, dict_kwargs)` tuples, or lists of the same, enabling complex branching or named function groups.
        *   All parameters within the `func` pattern (e.g., `dict_kwargs`) are optional.
*   **Key Components**: [`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py) (`FunctionStep.func` attribute), [`openhcs/core/steps/abstract.py`](openhcs/core/steps/abstract.py) (parameters of `AbstractStep.__init__`).

## 4. Functions and the `FUNC_REGISTRY` (Static Discovery)

*   **Concept**: The actual processing logic resides in Python functions decorated with OpenHCS-specific decorators. These decorators provide crucial metadata (e.g., `backend`, `input_memory_type`, `output_memory_type`, `special_inputs`, `special_outputs`). All such decorated functions are collected into a global `FUNC_REGISTRY`.
*   **Implication**: The system leverages static discovery (by inspecting these decorators and function signatures) to understand the available functions and their interfaces *without* needing to dynamically load or execute them. This static discovery is fundamental for building the TUI's dynamic UI elements (like dropdowns in the Function Pattern Editor) and for performing validation.
*   **Key Component**: [`openhcs/processing/func_registry.py`](openhcs/processing/func_registry.py) (`FUNC_REGISTRY`, `get_function_info`).

## 5. Centralized `TUIState` for UI State Management

*   **Concept**: The `TUIState` class acts as the single source of truth for all UI-related state within the TUI application. Components observe and react to changes in `TUIState`.
*   **Implication**: All UI updates and inter-component communication should flow through `TUIState.notify`. This ensures consistency and predictability. All `notify` calls and their handlers must be `async` to maintain responsiveness.
*   **Key Component**: [`openhcs/tui/tui_architecture.py`](openhcs/tui/tui_architecture.py) (`TUIState` class).

## 6. Single, Shared `storage_registry`

*   **Concept**: There is one central `storage_registry` instance created by the main `OpenHCSTUI` application. This registry manages all available storage backends (disk, memory, zarr, etc.).
*   **Implication**: Other components (e.g., `PlateManagerPane`, `PlateValidationService`, `PipelineOrchestrator`) that need to interact with the filesystem or different storage backends *receive* this shared `storage_registry` and then create their *own* `FileManager` instances using it. This ensures consistency and avoids fragmented storage management.
*   **Key Components**: [`openhcs/io/base.py`](openhcs/io/base.py) (`storage_registry`), [`openhcs/io/filemanager.py`](openhcs/io/filemanager.py) (`FileManager`).

## 7. Deferred Validation

*   **Concept**: User input validation is explicitly deferred to later stages of the workflow. The initial collection of data (e.g., adding a plate path) is minimal and performs no validation.
*   **Implication**: The "Add Plate" dialog only collects the path. Comprehensive validation (e.g., path existence, correct format, backend compatibility) occurs when the user explicitly triggers an "Init" or "Pre-compile" action for a selected plate. Errors are then reported visually and via the status bar.
*   **Key Components**: [`openhcs/tui/dialogs/plate_dialog_manager.py`](openhcs/tui/dialogs/plate_dialog_manager.py) (simplified), [`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py) (`PipelineOrchestrator.initialize()`).

## 8. Configuration Hierarchy and Overrides

*   **Concept**: OpenHCS uses a hierarchical configuration system, with global defaults defined in `GlobalPipelineConfig`. Plate-specific configurations can override these global defaults.
*   **Implication**: The TUI provides dedicated interfaces ("Global Settings" and "Edit Plate Config") for users to inspect and modify these configurations. UI elements for configuration are dynamically generated based on the structure of these configuration objects.
*   **Key Component**: [`openhcs/core/config.py`](openhcs/core/config.py) (`GlobalPipelineConfig`).

By adhering to these core concepts, any changes or additions to the OpenHCS TUI will naturally align with its intended architecture, leading to a more robust, predictable, and maintainable system.