# OpenHCS Non-TUI System: Conceptual Architecture and Data Flow

This document crystallizes the understanding of the OpenHCS non-TUI system's architecture, focusing on pipeline processing, data management, and core component interactions. It serves as a reference for future work.

## 1. High-Level Architecture

**Purpose:** OpenHCS is an open-source platform for high-content screening (HCS) image analysis. It provides a flexible and extensible framework for creating and executing image processing pipelines, primarily for microscopy images.

**Core Paradigm: Two-Phase Pipeline (Compile-then-Execute)**
The system operates on a two-phase model to process data (typically organized by "wells" from a microplate):

1.  **Compilation Phase:** A user-defined logical pipeline (a list of `Step` objects) is taken. For each well, a detailed, immutable execution plan (`step_plans`) is created within a `ProcessingContext`. This phase resolves paths, backends, memory types, and GPU assignments.
2.  **Execution Phase:** The (now stateless) original `Step` definitions are executed against these compiled and frozen `ProcessingContexts`, potentially in parallel for multiple wells.

**Key Components (Conceptual Overview):**
*   **`PipelineOrchestrator`:** Manages both the compilation and execution phases. Initializes a workspace, handles microscope-specific configurations, and orchestrates parallel processing of wells.
*   **`PipelineCompiler`:** Coordinates various sub-planners to build the `step_plans` within each `ProcessingContext`.
*   **`ProcessingContext`:** A central object holding all information relevant to processing a single well, including the `step_plans`, configuration, and references to services like `FileManager` and `MicroscopeHandler`. It's mutable during compilation and frozen before execution.
*   **`AbstractStep` / `FunctionStep`:** Define individual processing operations. `FunctionStep` is the primary means for users to apply custom Python functions to image data.
*   **`FileManager`:** An abstraction layer for I/O, routing operations to specific `StorageBackend` instances (disk, memory, Zarr).
*   **`MicroscopeHandler`:** Handles microscope-specific aspects like file naming conventions and metadata parsing to detect image patterns.
*   **Memory Utilities (`stack_utils`, `MemoryWrapper`):** Manage the conversion of image data between different memory types (NumPy, CuPy, Torch, etc.) and ensure correct GPU device placement.

---

## 2. Compilation Phase Deep Dive

The compilation phase translates a logical pipeline into a concrete, executable plan for each well.

**Lead: `PipelineOrchestrator`**
*   `compile_pipelines()` method:
    *   Iterates through wells identified by the `MicroscopeHandler`.
    *   For each `well_id`, creates a `ProcessingContext`.
    *   Invokes `PipelineCompiler` static methods sequentially to populate `context.step_plans`.
    *   Freezes the `ProcessingContext` to make it immutable.
    *   Calls `StepAttributeStripper` to make the original `pipeline_definition` (list of `AbstractStep` objects) stateless for the execution phase.

**Central Object: `ProcessingContext`**
*   Holds `global_config`, `well_id`, `filemanager`, `microscope_handler`, `input_dir`, `workspace_path`.
*   Crucially, it contains `step_plans` (a dictionary), which is built up during compilation and becomes the detailed execution plan for that well.

**Core Compiler: `PipelineCompiler`**
This class uses several specialized static methods (planners) to populate `context.step_plans`:

1.  **`initialize_step_plans_for_context(context, steps_definition)` (invokes `PipelinePathPlanner`)**
    *   **`PipelinePathPlanner.prepare_pipeline_paths(context, steps_definition)`:**
        *   **Purpose:** Determines primary input/output directory paths and special I/O paths for each step. Modifies `context.step_plans` in place.
        *   **Main I/O Paths:**
            *   Calculates `input_dir` and `output_dir` for each step based on its position, user overrides, previous/next step linkage, and "chain breaker" logic.
            *   Chain breakers (FunctionSteps with `__chain_breaker__ = True`) force the *next* step to read from the *initial pipeline input* (effectively restarting the primary data flow from disk) and set its `READ_BACKEND` to 'disk'.
            *   Output directories for first/last steps and steps with specific naming conventions (e.g., "position", "stitch") use configured suffixes relative to the workspace path. Intermediate steps often work in-place.
        *   **Special I/O Paths:**
            *   Resolves paths for `special_inputs` and `special_outputs` (declared by `FunctionStep` via `__special_inputs__`, `__special_outputs__`). These are typically `.pkl` files within the step's `output_dir`.
            *   Tracks `declared_outputs` to link `special_inputs` of later steps.
            *   **Metadata Injection:** If a `special_input` key matches a registered `METADATA_RESOLVERS` (e.g., "grid_dimensions"), the resolver function fetches the metadata (using `context`), and `PipelinePathPlanner` modifies the `FunctionStep`'s `func` attribute (if it's a `(callable, kwargs)` pattern) to inject this metadata as a keyword argument.
        *   Adds `input_dir`, `output_dir`, `pipeline_position`, `special_inputs` (map of key to path info), `special_outputs` (map of key to path info) to each `step_plan`.

2.  **`plan_materialization_flags_for_context(context, steps_definition)` (invokes `MaterializationFlagPlanner`)**
    *   **`MaterializationFlagPlanner.prepare_pipeline_flags(context, steps_definition)`:**
        *   **Purpose:** Determines the storage backend (e.g., 'disk', 'memory', 'zarr') for reading and writing main data for each step. Modifies `context.step_plans` in place.
        *   **Flags:** Based on `step.requires_disk_input`, `step.requires_disk_output`, `step.force_disk_output`, and step position:
            *   First step always `requires_disk_input = True`.
            *   Last step always `requires_disk_output = True`.
        *   **Backend Selection (`READ_BACKEND`, `WRITE_BACKEND`):**
            *   If `requires_disk_input/output` is true, uses persistent backends (e.g., 'disk', or `vfs_config.default_materialization_backend`).
            *   `FunctionStep`s (if not requiring disk I/O) can use `vfs_config.default_intermediate_backend` (e.g., 'memory') for both reading and writing.
            *   Non-`FunctionStep`s default to persistent backends for safety even if not strictly requiring disk.
            *   `READ_BACKEND` set by `PipelinePathPlanner` (for chain breakers) is preserved.
        *   Adds `REQUIRES_DISK_READ`, `REQUIRES_DISK_WRITE`, `FORCE_DISK_WRITE`, `READ_BACKEND`, `WRITE_BACKEND` to each `step_plan`.

3.  **`validate_memory_contracts_for_context(context, steps_definition)` (invokes `FuncStepContractValidator`)**
    *   **Purpose:** Determines and validates `input_memory_type` and `output_memory_type` (e.g., 'numpy', 'cupy', 'torch') for each `FunctionStep`.
    *   Uses `step.input_memory_type_hint` and `step.output_memory_type_hint` (if provided on the `FunctionStep` instance) from the `step_plan` as initial guides.
    *   The validator analyzes connectivity and function signatures (potentially looking for type hints or backend-specific function attributes) to ensure compatibility.
    *   Writes final `input_memory_type` and `output_memory_type` into each `step_plan`. Last step writing to disk is forced to '''numpy''' output.

4.  **`assign_gpu_resources_for_context(context)` (invokes `GPUMemoryTypeValidator`)**
    *   **`GPUMemoryTypeValidator.validate_step_plans(context.step_plans)`:**
        *   **Purpose:** Assigns a specific `gpu_id` (GPU device ID) to steps that use GPU memory types (cupy, torch, etc.).
        *   Validates that `gpu_id` is non-negative.
        *   Adds `gpu_id` to relevant `step_plans`.

**Result of Compilation:**
Each `ProcessingContext` in the `compiled_contexts` dictionary (returned by `Orchestrator.compile_pipelines`) contains a fully populated and frozen `step_plans` map. This map is the definitive, immutable guide for executing each step for that specific well.

---

## 3. Execution Phase Deep Dive

The execution phase runs the (now stateless) pipeline against the compiled `ProcessingContexts`.

**Lead: `PipelineOrchestrator`**
*   `execute_compiled_plate(pipeline_definition, compiled_contexts, max_workers, visualizer)` method:
    *   Takes the stateless `pipeline_definition` (list of reusable `AbstractStep` objects) and the `compiled_contexts` map.
    *   Uses `concurrent.futures.ThreadPoolExecutor` to process multiple wells in parallel if `max_workers > 1`.
    *   For each well, it calls its internal `_execute_single_well` method in a separate thread.
*   **`_execute_single_well(pipeline_definition, frozen_context, visualizer)`:**
    *   This is the core sequential execution loop for a single well.
    *   Iterates through each `step` object in the (stateless) `pipeline_definition`.
    *   For each `step`, it calls `step.process(frozen_context)`.
    *   The `frozen_context` provides all necessary configuration and plans for the step to execute correctly for that specific well.
    *   Handles optional visualization calls after each step if enabled in its plan.

**`AbstractStep.process(context)` Method:**
*   Each concrete step (like `FunctionStep`) implements this method.
*   It uses its own `step_id` (derived via `get_step_id(self)` from its object reference) to retrieve its specific `step_plan` from `context.step_plans[step_id]`.
*   All parameters needed for execution (input/output paths, backends, memory types, function to call, etc.) are sourced from this `step_plan`.

---

## 4. `FunctionStep` - The Workhorse

`FunctionStep` is the most versatile step, allowing arbitrary Python functions to be applied to image data.

**`FunctionStep.process(context)` - Detailed Flow:**
1.  **Retrieve Plan:** Gets its `step_plan` from `context.step_plans`. Extracts paths, backends, memory types, `gpu_id`, `variable_components`, `group_by` settings, and importantly, the `func` (callable or list/dict of callables) and special I/O plans.
2.  **Pattern Detection:**
    *   Calls `context.microscope_handler.auto_detect_patterns(input_dir, filemanager, read_backend, well_filter, group_by, variable_components)` to find and group relevant image files in the `step_input_dir` using the specified `read_backend`.
    *   The `group_by` (e.g., "channel") and `variable_components` (e.g., "site") from the plan guide this grouping. Result is `patterns_by_well[well_id]`.
3.  **Function and Argument Preparation:**
    *   Calls `prepare_patterns_and_functions(patterns_by_well[well_id], func_from_plan, group_by)`:
        *   This utility (from `openhcs.formats.func_arg_prep`) maps the detected file patterns to the specific processing function(s) and base keyword arguments defined in `func_from_plan` (which could be a single callable, a `(callable, kwargs)` tuple, a list of these for a chain, or a dictionary mapping `group_by` values to these structures).
        *   Returns `grouped_patterns`, `component_to_funcs` (mapping e.g., 'DAPI' to its function/chain), and `component_to_base_args`.
4.  **Iterate and Process Pattern Groups:**
    *   Loops through each component value (e.g., 'DAPI') and its list of `pattern_item`s from `grouped_patterns`.
    *   For each `pattern_item`, calls internal `_process_single_pattern_group()`:
        *   **Load Main Data:**
            *   `context.microscope_handler.path_list_from_pattern()` (called within `_process_single_pattern_group` via `auto_detect_patterns` logic) gets list of relative file paths for the current pattern.
            *   Files are loaded using `context.filemanager.load(full_path, read_backend)`.
            *   Loaded 2D image slices are stacked into a 3D array (`main_data_stack`) by `stack_slices(slices, input_memory_type, gpu_id)`.
        *   **Execute Function(s) (via `_execute_function_core` or `_execute_chain_core`):**
            *   `_execute_function_core(func_callable, main_data_arg, base_kwargs, context, special_inputs_plan, special_outputs_plan, well_id)`:
                *   **Special Inputs:** Loads special input data from VFS paths specified in `special_inputs_plan` using `context.filemanager.load(path, Backend.MEMORY.value)`. These paths are namespaced with `well_id_`.
                *   **Argument Injection:** Injects loaded special inputs as kwargs. Also injects `context` as a kwarg if the function signature expects it.
                *   **Call:** `raw_output = func_callable(main_data_stack, **all_kwargs)`.
                *   **Special Outputs:** If `special_outputs_plan` exists, `raw_output` is expected as `(main_out, sp_val1, ...)`. These special values are saved positionally to VFS paths from `special_outputs_plan` using `context.filemanager.save(value, path, Backend.MEMORY.value)`. Paths are namespaced with `well_id_`.
                *   Returns the `main_output_data`.
            *   `_execute_chain_core` handles a list of functions, passing the output of one as the `main_data_arg` to the next. Only the last function in the chain handles the step's overall `special_outputs_plan`.
        *   **Save Main Data:**
            *   The `processed_stack` (output from function execution) is unstacked into 2D slices by `unstack_slices(processed_stack, output_memory_type, gpu_id)`.
            *   Each slice is saved using `context.filemanager.save(slice, output_path, write_backend)`. Output filenames are typically derived from input filenames.

---

## 5. Core Services & Utilities

**`FileManager` (`openhcs.io.filemanager`)**
*   **Role:** Acts as a backend-agnostic router for all file and directory operations.
*   **Initialization:** Takes a `registry` (typically the global `storage_registry` from `openhcs.io.base`).
*   **Operation:** Methods like `load`, `save`, `exists`, `list_files`, `ensure_directory` take an explicit `backend` string argument (e.g., "disk", "memory", "zarr"). The `FileManager` retrieves the corresponding `StorageBackend` instance from its registry and delegates the call to it.
*   **No Conversions:** Does not perform data type (e.g., NumPy to CuPy) or file format (e.g., TIFF to Zarr array) conversions itself. These are responsibilities of the specific `StorageBackend` or the calling code.
*   **Shared Backends:** Multiple `FileManager` instances using the same registry will share the same `StorageBackend` instances (e.g., one global `MemoryStorageBackend`).

**`StorageBackend` (`openhcs.io.base.StorageBackend` and implementations like `DiskStorageBackend`, `MemoryStorageBackend`, `ZarrStorageBackend`)**
*   **Interface:** Defines abstract methods for `load`, `save`, `list_files`, `exists`, `ensure_directory`, `delete`, `copy`, `move`, etc.
*   **Implementations:** Each concrete backend (e.g., `DiskStorageBackend`) implements these methods for a specific storage medium. This is where actual file reading/writing, image decoding/encoding (e.g., using `tifffile`, `zarr`), and memory VFS dictionary management occurs.

**`stack_utils` (`openhcs.core.memory.stack_utils`)**
*   **`stack_slices(slices: List[Any], memory_type: str, gpu_id: int) -> Any`:**
    *   Takes a list of 2D array slices (any type).
    *   Converts each slice to the target `memory_type` (e.g., 'numpy', 'cupy') and places it on the target `gpu_id` if applicable. This is done by wrapping each slice in `MemoryWrapper`, calling its conversion methods (e.g., `to_cupy()`).
    *   Stacks the converted slices into a single 3D array using the native stacking function of the target library (e.g., `np.stack`, `cp.stack`).
*   **`unstack_slices(array: Any, memory_type: str, gpu_id: int) -> List[Any]`:**
    *   Takes a 3D array.
    *   Converts the entire 3D array to the target `memory_type` and `gpu_id` using `MemoryWrapper`.
    *   Splits the converted 3D array into a list of 2D slices.

**`MemoryWrapper` (conceptually, implementation likely in `openhcs.core.memory.wrapper` or `__init__`)**
*   Encapsulates an array-like data object along with its current `memory_type` (e.g., 'numpy', 'cupy') and `gpu_id`.
*   Provides unified conversion methods like `to_numpy()`, `to_cupy(target_gpu_id)`, `to_torch(target_gpu_id)`, etc. These methods handle the actual data transfer between CPU/GPU and library-specific type conversions.

---

## 6. Key Design Principles Observed

*   **Immutability & Statelessness:** Compiled plans (`step_plans` in frozen `ProcessingContexts`) are immutable. `Step` definitions become stateless after compilation, making them reusable templates. This promotes predictability and safer parallel execution.
*   **Explicit Configuration & Dependency Injection:** Components like `FileManager` and `MicroscopeHandler` are explicitly passed into contexts or constructors. Parameters like I/O backends and memory types are explicitly determined and stored in plans, avoiding runtime fallbacks or inference.
*   **Abstraction Layers:** `FileManager` abstracts specific storage backends. `MicroscopeHandler` abstracts microscope-specific file organization and parsing. `MemoryWrapper` abstracts array memory types and conversions.
*   **Separation of Concerns:** Planning (compilation) is strictly separated from execution. Path/backend planning is distinct from memory type validation. Data loading/saving is distinct from data processing logic.
*   **Doctrinal Clauses:** Numerous comments refer to internal "Doctrinal Clauses" (e.g., "Clause 66 — Immutability After Construction," "Clause 273 — Backend Authorization Doctrine"). These appear to be a set of guiding principles or coding standards that enforce the system's architectural rigor and explicitness.

---

This document aims to provide a comprehensive yet conceptual overview. Specific implementation details within individual backend methods or complex algorithms would require further code inspection.
