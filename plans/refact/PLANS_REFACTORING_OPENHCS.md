# Partial Refactoring and Analysis Plan for OpenHCS

**Date:** 2025-05-20

**Version:** 1.9 (Aligned with Core v3.0, Modules v2.8 - `stack_utils` handles image (de)serialization)

## Overarching Goal

Align the OpenHCS pipeline system with key architectural principles:
1.  A strict two-phase (compile-all-then-run-all) execution model for batch processing.
2.  **VFS-exclusive inter-step communication for all data.**
    *   `FunctionStep.process` (and its helpers) orchestrates all I/O using a `step_plan`.
    *   **`FileManager` ([`openhcs/io/filemanager.py`](openhcs/io/filemanager.py:1)) is a pure raw byte I/O layer.** `FileManager.open()` returns raw bytes; `FileManager.save()` expects and writes raw bytes. It performs no (de)serialization of complex data formats (like images or pickled objects).
    *   **`stack_slices` (in [`openhcs/core/memory/stack_utils.py`](openhcs/core/memory/stack_utils.py:1))** takes a list of raw image file bytes (from `FileManager.open()`), deserializes them into 2D arrays, then type-converts and stacks them into a 3D typed array. (Requires enhancement).
    *   **`unstack_slices` (in [`openhcs/core/memory/stack_utils.py`](openhcs/core/memory/stack_utils.py:1))** takes a 3D typed array, type-converts and unstacks it to a list of 2D typed arrays, then serializes each 2D array into raw image file bytes (e.g., TIFF). It returns a list of raw bytes. (Requires enhancement).
    *   **`SpecialKey` I/O** involves `FunctionStep` passing raw bytes (from `FileManager.open()`) as positional arguments to user functions (which deserialize them), or receiving raw bytes (pre-serialized by user functions) as direct returns for special outputs, which are then saved by `FileManager.save()`. Decorators `@special_input(KeyName)` and `@special_output(KeyName)` are simple markers.
3.  **`ProcessingContext` immutability** after its initial compilation/setup phase. `StepResult` is eliminated.
4.  **Stateless Steps:** Step instances are stateless at execution time; all parameters come from the `step_plan`.
5.  **`step_plan` Structure:** Uses a flat key structure (e.g., `input_dir`, `read_backend` for primary I/O; `special_input_vfs_info`, `special_output_vfs_info` for special I/O VFS details).

## I. Identified Areas for Refactoring ("Rot")

### 1. `PipelineOrchestrator.run` Method Phasing
*   **Current State:** (As in v1.8)
*   **Desired State:** Strict two-phase (Compile All, then Execute All).
*   **Impact:** (As in v1.8)

### 2. Inter-Step Communication and `ProcessingContext` Mutability (Elimination of `StepResult`)
*   **Canonical Principle:** All inter-step communication via VFS, orchestrated by `FunctionStep.process` using `FileManager` for raw byte transfer. `stack_utils` handles primary image (de)serialization. User functions handle special data (de)serialization.
*   **Desired State:**
    *   `StepResult` class removed. `step.process()` returns `None`.
    *   `ProcessingContext` immutable post-compilation. `update_from_step_result` removed.
    *   All data I/O managed by `FunctionStep.process` (and helpers) using `FileManager` (for raw bytes) and `stack_utils` (for primary image (de)serialization & stacking), guided by the `step_plan`.
*   **Impact:**
    *   Removal of `StepResult`.
    *   Changes to `PipelineExecutor.execute` for exception handling.
    *   `AbstractStep.process` signature changes.
    *   `FunctionStep.process` implements full I/O orchestration.
    *   Planners (`PipelinePathPlanner`, `MaterializationFlagPlanner`) and `PipelineCompiler` are critical for creating the correct flat `step_plan`.

### 3. VFS Path, Backend, and Materialization Planning & Execution
*   **Responsibilities & Data Flow:**
    1.  **`PipelinePathPlanner`:** Determines string path structures for all primary and special I/O. Aware of `@chain_breaker`. Links `SpecialKey` producer/consumer paths.
    2.  **`MaterializationFlagPlanner`:** Determines `read_backend`, `write_backend`, `force_disk_output` (for primary I/O), and `backend`, `materialize` flags (within `special_input_vfs_info` / `special_output_vfs_info`) for all VFS data items.
    3.  **`PipelineCompiler`:** Assembles the flat `step_plan` for each step, including:
        *   Primary I/O: `input_dir` (path), `output_dir` (path), `read_backend` (str), `write_backend` (str), `force_disk_output` (bool).
        *   Special I/O: `expected_special_inputs` (list of SpecialKey names), `produces_special_output` (SpecialKey name or None), `special_input_vfs_info` (Dict mapping SpecialKey name to `{'path':..., 'backend':..., 'materialize':...}`), `special_output_vfs_info` (single `{'path':..., 'backend':..., 'materialize':...}` or None).
        *   Other params: `func` (resolved pattern), `variable_components`, `group_by`, memory types, `gpu_id`.
    4.  **`FunctionStep.process` (and its helper `_process_single_pattern_vfs`):**
        *   Retrieves all parameters from its `step_plan`.
        *   For primary input: Calls `FileManager.open()` to get `List[raw_image_bytes]`, then `stack_slices()` to get a 3D typed array.
        *   For special inputs: Calls `FileManager.open()` to get `raw_bytes` for each, passes these directly as positional args to the user function (which deserializes).
        *   User function executes.
        *   For special output: User function returns `raw_bytes` (pre-serialized). `FunctionStep` calls `FileManager.save()` with these bytes.
        *   For primary output: User function returns 3D typed array. `FunctionStep` calls `unstack_slices()` to get `List[raw_image_bytes]`, then calls `FileManager.save()` for each.
        *   Handles materialization (e.g., `force_disk_output` for primary, `materialize` flag for special) by making additional `FileManager.save()` calls with `backend="disk"`.
    5.  **`FileManager`:** Pure raw byte I/O.
*   **Action:**
    *   **Refactor Planners & Compiler:** Ensure they generate the correct flat `step_plan` structure with all VFS info.
    *   **Refactor `FunctionStep.process`:** Implement the full orchestration logic as described above.
    *   **Enhance `stack_utils.py`:** `stack_slices` to deserialize raw image bytes; `unstack_slices` to serialize 2D typed arrays to raw image bytes.

### 4. Deep Learning Function Integration and Refinements (`openhcs/processing/backends/`)
*   **General Requirement:** DL functions must align with the `FunctionStep` model:
    *   Primary input is a 3D typed array (from `stack_slices`).
    *   `@special_input(KeyName)` parameters receive raw bytes positionally; the DL function deserializes (e.g., `torch.load(io.BytesIO(raw_model_bytes))`).
    *   `@special_output(KeyName)` functions return raw bytes (e.g., `model_bytes = io.BytesIO(); torch.save(state_dict, model_bytes); return model_bytes.getvalue()`).
    *   Other parameters are `kwargs`.
*   (Specific file actions in [`PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) v2.8 provide detailed examples).

### 5. Function Linkage and `ImageProcessorInterface` Adherence
*   (No change from v1.8, but ensure alignment with new data flow).

### 6. `ezstitcher` Naming Discrepancy
*   (No change from v1.8).

### 7. Duplicate `create_microscope_handler` Factory
*   (No change from v1.8).

### 8. `PatternPath.is_fully_instantiated()` Validation in `PatternDiscoveryEngine`
*   (No change from v1.8).

## II. Well-Structured ("Solved") Areas (Positive Controls)
*   **VFS Layer (`openhcs/io/`)**: `FileManager` methods operate on direct string paths and backend strings, exclusively handling raw byte I/O. This is aligned.
*   (Other points as in v1.8, interpreted through the lens of the new data flow).

## III. Architectural Insights and Considerations for Microscopy Workflows
*   (As in v1.8, with `step_plan` VFS info now flat and direct).

## IV. Next Steps in Analysis (To Identify Other Problematic Files/Areas or Confirm Alignment)

1.  **Pipeline Planners & Compiler Refactoring (`PipelinePathPlanner`, `MaterializationFlagPlanner`, `PipelineCompiler`):**
    *   **Priority Action:** Refactor to produce the flat `step_plan` structure with direct VFS info keys (e.g., `input_dir`, `special_input_vfs_info`) as per [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0. Ensure `@chain_breaker` and `SpecialKey` linking logic is correct. Validate `step_plan` contents during compilation.

2.  **`FunctionStep` ([`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py:1)) Refactoring:**
    *   **Priority Action:** Modify `FunctionStep.process` and its helper `_process_single_pattern_vfs` to align with [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0: use flat `step_plan` keys, orchestrate `FileManager.open()` for raw bytes, call `stack_slices` (which deserializes), pass raw bytes for special inputs, receive raw bytes for special outputs, call `unstack_slices` (which serializes for primary output), and use `FileManager.save()` for all raw byte outputs. Remove `StepResult`.

3.  **`stack_utils.py` ([`openhcs/core/memory/stack_utils.py`](openhcs/core/memory/stack_utils.py:1)) Interface:**
    *   **Note:** This is a finalized component that should not be modified. Other components must adapt to its current interface:
        *   `stack_slices` accepts array-like objects (not bytes) and performs type conversion and stacking.
        *   `unstack_slices` returns a list of array-like objects (not bytes) after unstacking and type conversion.

4.  **`FileManager` ([`openhcs/io/filemanager.py`](openhcs/io/filemanager.py:1)) Interface:**
    *   **Note:** This is a finalized component that should not be modified. Its `open` and `save` methods strictly deal with raw bytes and do no (de)serialization.

5.  **Microscope Handler and Pattern Detection (`openhcs/microscopes/`, `openhcs/formats/pattern/`) Interaction:**
    *   Ensure they use `FileManager.open()` for metadata and pass raw content to parsers.

6.  **Remaining `openhcs/processing/` Audit (including DL functions):**
    *   Align all functions with the raw data model for `SpecialKey` I/O and (de)serialization responsibilities.

7.  **Other `AbstractStep` Implementations:**
    *   Review for alignment with VFS-only I/O and `StepResult` removal.

8.  **TUI Code Update (`openhcs/tui/function_pattern_editor.py`):**
    *   **Action:** Modify parameter editing UI to differentiate between VFS path linking for mandatory `SpecialKey` raw data inputs/outputs and direct value input for `kwargs`, as detailed in [`PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) v2.8.

## V. TUI Design Considerations and Alignment with Core Refactoring
*   The TUI must reflect the clear distinction between configuring VFS paths for `SpecialKey` raw data artifacts and setting values for behavioral `kwargs`. `FunctionStep.process` orchestrates `FileManager` (raw bytes) and `stack_utils` ((de)serialization for primary images) based on the `step_plan`.

---
*This is a working document and will be updated as the audit and planning progresses.*