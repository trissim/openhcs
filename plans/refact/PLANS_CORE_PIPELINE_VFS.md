# Core Pipeline and VFS Refactoring Plan for OpenHCS

**Date:** 2025-05-20

**Version:** 3.4 (Drastically Simplified `step_plan` for Special I/O: Single Key Model; Full Document)

## Overarching Goal

Align the OpenHCS pipeline system with key architectural principles:
1.  Strict two-phase (compile-all-then-run-all) execution.
2.  **Stateless Steps:** Step instances are stateless after compilation. All execution parameters are from the `step_plan`.
3.  VFS-exclusive inter-step communication.
    *   **I/O:** `FunctionStep` uses `FileManager` for all data operations. `FileManager` routes operations to the appropriate backend.
    *   **Image Format (De)serialization:** Handled by backends when needed (e.g., disk backend for first read and final write).
    *   **`FileManager`** is a routing layer that forwards operations to appropriate backends for all I/O operations.
    *   **`stack_utils.py` (`stack_slices`, `unstack_slices`)** are UNCHANGED (expect/return array-like objects, no image format (de)serialization).
4.  **`SpecialKey` I/O (Raw Data Transfer - Single Input/Output Model):**
    *   A function can have at most one `@special_input(SpecialKey.KEY_NAME)` and at most one `@special_output(SpecialKey.KEY_NAME)` that are managed through dedicated singular keys in the `step_plan`.
    *   `@special_input`: Function expects a single optional positional argument (after primary 3D stack) which will be raw bytes loaded by `FunctionStep`. User function deserializes.
    *   `@special_output`: Function's direct return is raw bytes (pre-serialized). `FunctionStep` writes these.
5.  **`@chain_breaker`:** Modifies `primary_input` path planning.
6.  `ProcessingContext` immutability. `StepResult` eliminated.
7.  **`step_plan` Structure:** Adheres to flat key structure. For Special I/O, `step_plan` contains singular keys for the key name, path, backend name, and materialize flag for THE special input (if any) and THE special output (if any). E.g., `special_input_key: Optional[str]`, `special_input_path: Optional[str]`.

## I. Identified Areas for Refactoring ("Rot") - Core Pipeline & VFS

### 1. `PipelineOrchestrator.run` Method Phasing
*   **Current State:** The `run` method in [`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py:1) currently iterates through wells, performing compilation (via `PipelineCompiler.compile`) and context creation for each well *within the same loop* that prepares for batch execution.
*   **Desired State:** A strict two-phase approach:
    1.  **Phase 1 (Compile All):** Iterate through all specified wells. For each well, create its `ProcessingContext` and compile its pipeline using `PipelineCompiler.compile`. Collect all compiled `ProcessingContext` objects (now containing immutable step plans) and the corresponding lists of `AbstractStep` objects.
    2.  **Phase 2 (Execute All):** Pass the collected lists of compiled contexts and step lists to `PipelineExecutor.execute` (or `execute_parallel`) for batch execution.
*   **How-To Refactor `PipelineOrchestrator.run`:**
    *   Modify `run` to first loop through wells, calling `PipelineCompiler.compile` for each and storing the `(ProcessingContext, List[AbstractStep])` tuples.
    *   After the loop, pass all collected tuples to `PipelineExecutor.execute_batch` (a new method or an adapted `execute_parallel`).
    *   `PipelineExecutor` will then iterate through these, calling `step.process(context)` for each step in each pipeline.

### 2. Inter-Step Communication and `ProcessingContext` Mutability
*   **Canonical Principle:** All inter-step communication via VFS, orchestrated by `FunctionStep.process` using backend instances for raw byte transfer. Image format (de)serialization for primary data path is handled by `FunctionStep`. User functions handle special data content (de)serialization.
*   **Desired State:**
    *   `StepResult` class will be removed entirely. `step.process()` methods will signal failure by raising exceptions. Success is indicated by normal completion (e.g., returning `None`). `PipelineExecutor` will handle these exceptions.
    *   `ProcessingContext`: The `update_from_step_result` method will be removed. The `ProcessingContext` will not be mutated by the `PipelineExecutor` after a step completes.
*   **How-To Refactor `ProcessingContext`:**
    *   Remove `update_from_step_result` method.
    *   Ensure all attributes are set during `__init__` and treated as immutable thereafter by execution components.
    *   `ProcessingContext` will still hold the `filemanager` instance and the fully compiled `step_plans` dictionary.

### 3. VFS Path, Backend, and Materialization Planning & Execution
*   **`PipelineCompiler` Responsibilities:**
    *   Assembles the flat `step_plan` for each step.
    *   For primary I/O: Populates `input_dir` (path string), `output_dir` (path string), `read_backend` (backend name string), `write_backend` (backend name string), `force_disk_output` (bool).
    *   For special I/O (Strict Single Model):
        *   Introspects function for `@special_input(KEY)` decorator. If multiple are found, this is an error or requires a policy on which one to use for these singular `step_plan` keys. Assuming for now that validation ensures only one, or the first one found, is used.
            *   Populates `step_plan['special_input_key']: Optional[str]` (the `KEY.value` string).
            *   Populates `step_plan['special_input_path']: Optional[str]` (resolved VFS path string).
            *   Populates `step_plan['special_input_backend']: Optional[str]` (backend name string).
            *   Populates `step_plan['special_input_materialize_flag']: Optional[bool]`.
        *   Introspects function for `@special_output(KEY)` decorator. (A function can only have one `@special_output`).
            *   Populates `step_plan['special_output_key']: Optional[str]` (the `KEY.value` string).
            *   Populates `step_plan['special_output_path']: Optional[str]` (resolved VFS path string).
            *   Populates `step_plan['special_output_backend']: Optional[str]` (backend name string).
            *   Populates `step_plan['special_output_materialize_flag']: Optional[bool]`.
    *   (Other params: `func` (resolved pattern), `variable_components`, `group_by`, memory types, `gpu_id`, `chain_breaker` remain).
*   **How-To Refactor Planners & Compiler:**
    *   `PipelinePathPlanner`: Resolves paths for `special_input_path` and `special_output_path` (singular).
    *   `MaterializationFlagPlanner`: Resolves backends and flags for `special_input_backend`/`_materialize_flag` and `special_output_backend`/`_materialize_flag` (singular).
    *   `PipelineCompiler`: Assembles. **Must include validation logic: if a function is decorated with multiple `@special_input`s, how is this handled? Current model implies only one can be represented in `step_plan`'s singular keys. This needs to be a strict rule or the `step_plan` model for special inputs needs to be a list/dict again.**

### 4. `FunctionStep` Refactoring (Core of VFS Integration)
*   **`FunctionStep.process(self, context: ProcessingContext) -> None`:**
    *   (Pseudo-code structure as in v3.3, passes `backend_registry` and `file_manager` to helper).
*   **Helper `_process_single_pattern_vfs`:**
    *   **Signature:** Update to reflect simplified special I/O params from `step_plan` (e.g., `special_input_key_name: Optional[str]`, `special_input_path_str: Optional[str]`, etc.).
    *   **Logic (Focus on single special input/output from `step_plan`):**
        ```python
        # # (Inside FunctionStep class or as a static/standalone helper)
        # # Parameters: backend_registry, file_manager, and all relevant step_plan values
        # # (e.g., input_dir_path_str, read_backend_name, 
        # #  special_input_key_name, special_input_path_str, special_input_backend_name, special_input_materialize_flag,
        # #  special_output_key_name, special_output_path_str, special_output_backend_name, special_output_materialize_flag, etc.)
        #
        # # 1. Primary Input (As in v3.3 - FunctionStep deserializes image bytes)
        # # ...
        # # stacked_3d_input_array = ...
        #
        # # 2. Prepare THE Positional Argument for THE Special Input (if any)
        # special_args_list = [] # Will contain zero or one element
        # if special_input_key_name and special_input_path_str and special_input_backend_name:
        #   SpecialInputBackendClass = backend_registry[special_input_backend_name]
        #   special_input_backend_instance = SpecialInputBackendClass()
        #   raw_data_for_special_arg = special_input_backend_instance.load(special_input_path_str)
        #   special_args_list.append(raw_data_for_special_arg)
        #
        # # 3. Execute Function(s)
        # # User function receives primary 3D typed array, then AT MOST ONE raw byte special arg.
        # # The function signature must align: def user_func(primary_stack, special_arg=None, *, kwarg1=...)
        # result_from_func = actual_callable(stacked_3d_input_array, *special_args_list, **final_kwargs)
        #
        # # 4. Handle Output
        # if special_output_key_name and special_output_path_str and special_output_backend_name: # Special Output
        #   raw_bytes_to_save_special = result_from_func 
        #   
        #   SpecialOutputBackendClass = backend_registry[special_output_backend_name]
        #   special_output_backend_instance = SpecialOutputBackendClass()
        #   special_output_backend_instance.save(raw_bytes_to_save_special, special_output_path_str)
        #
        #   if special_output_materialize_flag and special_output_backend_name != "disk":
        #     DiskBackendClass = backend_registry["disk"]
        #     disk_backend_instance = DiskBackendClass()
        #     disk_backend_instance.save(raw_bytes_to_save_special, special_output_path_str)
        # else: # Standard Primary 3D Image Output
        #   # (As in v3.3 - FunctionStep serializes 2D array slices to bytes)
        #   # ...
        #
        # # No return value
        ```

## II. Well-Structured ("Solved") Areas (Positive Controls) - Core
        #   special_output_backend_instance = SpecialOutputBackendClass()
        #   special_output_backend_instance.save(raw_bytes_to_save_special, special_output_path_str)
        #
        #   if special_output_materialize_flag and special_output_backend_name != "disk":
        #     DiskBackendClass = backend_registry["disk"]
        #     disk_backend_instance = DiskBackendClass()
        #     disk_backend_instance.save(raw_bytes_to_save_special, special_output_path_str)
        # else: # Standard Primary 3D Image Output
        #   processed_3d_output_array = result_from_func # This is a 3D typed array
        #   # unstack_slices returns List[ArrayLike] (UNCHANGED from its current signature)
        #   list_of_2d_typed_slices = unstack_slices(
        #       processed_3d_output_array, output_memory_type, gpu_id
        #   ) 
        #   
        #   PrimaryOutputBackendClass = backend_registry[write_backend_name]
        #   primary_output_backend_instance = PrimaryOutputBackendClass()
        #   file_manager.ensure_directory(output_dir_path_str, write_backend_name) # FileManager for dir ops
        #
        #   # ... (determine output_filenames_to_use) ...
        #   for i, array_slice in enumerate(list_of_2d_typed_slices):
        #     abs_path_to_save = str(Path(output_dir_path_str) / output_filenames_to_use[i])
        #     try:
        #         with io.BytesIO() as buffer:
        #             # FunctionStep serializes (e.g., to TIFF, make format configurable if needed)
        #             iio.imwrite(buffer, array_slice, format='TIFF') 
        #             raw_bytes_slice_to_save = buffer.getvalue()
        #         primary_output_backend_instance.save(raw_bytes_slice_to_save, abs_path_to_save)
        #     except Exception as e:
        #         logger.error(f"Failed to serialize or save image slice {abs_path_to_save}: {e}")
        #         raise # Or handle error
        #
        #     if force_disk_output and write_backend_name != "disk":
        #       DiskBackendClass = backend_registry["disk"]
        #       disk_backend_instance = DiskBackendClass()
        #       disk_backend_instance.save(raw_bytes_slice_to_save, abs_path_to_save)
        #
        # # No return value
        ```

## II. Well-Structured ("Solved") Areas (Positive Controls) - Core
*   `stack_utils.py` functions (`stack_slices`, `unstack_slices`) are finalized components that only perform tensor format conversion and stacking/unstacking (no serialization).
*   `FileManager` acts as a routing layer to appropriate backends which handle the operations specific to their storage type.
*   Storage backends (disk, memory, zarr) handle the actual I/O operations, with disk backend handling serialization when needed.

## III. Architectural Insights and Considerations - Core
*   This model maintains clear separation of responsibilities: `FunctionStep` orchestrates operations, `FileManager` routes I/O to backends, backends handle storage-specific operations, and `stack_utils` handles tensor conversions. The disk backend handles serialization/deserialization only when needed (first read, final write).

## IV. Next Steps in Analysis - Core
*   Define error handling strategy for I/O operation failures.
*   Ensure proper interface documentation for backend implementations.

## V. Core Infrastructure Issues & Refactoring

### 1. Memory Tracker Loading System
*   (No changes from v3.2)

---
*This is a working document and will be updated as the audit and planning progresses.*