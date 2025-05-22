# Refactoring Implementation Context & Key Code References

**Date:** 2025-05-20
**Version:** 1.1 (Added SpecialKey Enum Definition)

## 1. Introduction

This document provides essential context, summarizes key architectural decisions, and references relevant existing code components crucial for implementing the OpenHCS refactoring as detailed in the following primary plan documents:

*   **Core Logic & VFS:** [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0)
*   **Module & TUI Integration:** [`plans/PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) (v2.8)
*   **Configuration System:** [`plans/PLANS_CONFIG_REFACTOR.md`](plans/PLANS_CONFIG_REFACTOR.md:1) (v1.0)
*   **Overall Refactoring Summary:** [`plans/PLANS_REFACTORING_OPENHCS.md`](plans/PLANS_REFACTORING_OPENHCS.md:1) (v1.9)

The aim is to provide a practical, consolidated reference for developers undertaking the implementation.

## 2. Key Data Structures, Definitions, and Conventions

### 2.1. `SpecialKey` Enum/Class
*   **Purpose:** `SpecialKey`s are unique identifiers for specific, named data artifacts that are passed between pipeline steps via VFS. They are used to decorate functions to declare these data dependencies.
*   **Definition:** Found in [`openhcs/constants/constants.py`](openhcs/constants/constants.py:1). The relevant definition is:
    ```python
    from enum import Enum

    class SpecialKey(Enum):
        POSITION_ARRAY = "position_array"
        GRAPH = "graph"
        # Add other keys as needed, e.g.:
        # MODEL_WEIGHTS = "model_weights"
        # TRAINED_MODEL = "trained_model"
        # DXF_FILE = "dxf_file" 
        # STITCHED_POSITIONS = "stitched_positions"
        # NEURITE_TRACES = "neurite_traces"
    ```
*   **Usage in Decorators:**
    *   `@special_input(SpecialKey.KEY_NAME)`
    *   `@special_output(SpecialKey.KEY_NAME)`
    *   These decorators are simple markers and take *only* the `SpecialKey` identifier (e.g., `SpecialKey.POSITION_ARRAY`). They do **not** include `data_type` or `optional` parameters. The value of the enum member (e.g., `"position_array"`) will be used as the string key in `step_plan` dictionaries like `special_input_vfs_info`.

### 2.2. `step_plan` Dictionary Structure
The `step_plan` is a flat dictionary populated by `PipelineCompiler` for each step instance. It is the sole source of truth for step execution parameters at runtime. Based on [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0) and current [`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py:1) structure, it will contain:

*   **Core Parameters (from existing `function_step.py`):**
    *   `'well_id': str`
    *   `'input_dir': str` (Path to primary input directory)
    *   `'output_dir': str` (Path to primary output directory)
    *   `'variable_components': List[str]`
    *   `'group_by': str`
    *   `'func': Any` (The resolved function pattern: callable, list of callables, or dict of lists of `(callable, kwargs_dict)` tuples)
    *   `'read_backend': str` (Backend for primary input from `input_dir`)
    *   `'write_backend': str` (Backend for primary output to `output_dir`)
    *   `'force_disk_output': bool` (For primary output materialization)
    *   `'input_memory_type': str`
    *   `'output_memory_type': str`
    *   `'gpu_id': Optional[int]` (Also aliased as `device_id`)
*   **Added for Special I/O and Enhanced Control:**
    *   `'expected_special_inputs': List[str]` (List of `SpecialKey.KEY_NAME.value` strings, representing mandatory positional arguments after the primary 3D stack)
    *   `'produces_special_output': Optional[str]` (A single `SpecialKey.KEY_NAME.value` string if the function has a special output, otherwise `None`)
    *   `'special_input_vfs_info': Dict[str, {'path': str, 'backend': str, 'materialize': bool}]`
        *   Maps each `SpecialKey.KEY_NAME.value` string from `expected_special_inputs` to its VFS information dictionary.
    *   `'special_output_vfs_info': Optional[{'path': str, 'backend': str, 'materialize': bool}]`
        *   VFS information dictionary for the special output, present if `produces_special_output` is not `None`.
    *   `'chain_breaker': bool` (From function decorator)

## 3. Core Components: Roles & Required Modifications

### 3.1. `FileManager` ([`openhcs/io/filemanager.py`](openhcs/io/filemanager.py:1))
*   **Refactored Role:** Pure raw byte I/O layer for VFS.
*   **Key Methods:**
    *   `open(path: str, backend: str)`: Routes the open operation to the appropriate backend, which returns data in the backend's format. For disk backend, this includes deserialization from disk.
    *   `save(path: str, data, backend: str) -> None`: Routes the save operation to the appropriate backend. Each backend handles its storage format appropriately (e.g., disk backend handles serialization).
    *   Other existing utility methods like `list_files`, `ensure_directory`, `exists`, `delete` remain, operating on paths and backends.
*   **No Modifications Required:** This component is a finalized routing layer that forwards operations to appropriate backends.

### 3.2. `stack_utils.py` ([`openhcs/core/memory/stack_utils.py`](openhcs/core/memory/stack_utils.py:1))
*   **`stack_slices` and `unstack_slices`:**
    *   **No Modifications Required:** These are finalized components that:
        1. Accept array-like objects (not bytes)
        2. Perform tensor format conversion between different memory types
        3. Handle stacking/unstacking operations
        4. Do not perform any serialization/deserialization (this is handled by the appropriate backends)
    *   The function signatures remain unchanged.

### 3.3. `FunctionStep` ([`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py:1))
*   **Refactored Role:** Stateless orchestrator of I/O and function execution, driven entirely by its `step_plan`.
*   **`process(self, context: ProcessingContext) -> None`:** (Details in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0)
*   **`_process_single_pattern_vfs(...)` (New/Refactored Helper):** (Details in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0)
    *   Handles primary I/O using `FileManager.open`, `stack_slices`, `unstack_slices`, `FileManager.save` (all raw bytes for `FileManager`).
    *   Handles SpecialKey I/O: `FileManager.open` for raw input bytes passed to user func; user func returns raw output bytes saved by `FileManager.save`. User func (de)serializes content of these raw bytes.

### 3.4. `PipelineCompiler` ([`openhcs/core/pipeline/compiler.py`](openhcs/core/pipeline/compiler.py:1))
*   **Responsibility:** Creates the valid, flat `step_plan` dictionary (Section 2.2).
*   Introspects `FunctionStep` instances and function callables (for `@special_input`/`@special_output` decorators using attributes like `func.__special_inputs__`) to populate `step_plan`.
*   Uses planners for VFS info. Validates `step_plan`.

### 3.5. `PipelinePathPlanner` ([`openhcs/core/pipeline/path_planner.py`](openhcs/core/pipeline/path_planner.py:1))
*   Determines VFS paths for primary and special I/O. Handles `@chain_breaker`. Links `SpecialKey` paths.

### 3.6. `MaterializationFlagPlanner` ([`openhcs/core/pipeline/materialization_flag_planner.py`](openhcs/core/pipeline/materialization_flag_planner.py:1))
*   Determines `read_backend`, `write_backend`, `force_disk_output` (primary) and `backend`/`materialize` flags (special I/O).

## 4. Decorator Implementation Notes
*   `@special_input(SpecialKey.KEY_NAME)`:
    *   Attaches `SpecialKey.KEY_NAME.value` (string) to `func.__special_inputs__` (list of strings). Order matters.
*   `@special_output(SpecialKey.KEY_NAME)`:
    *   Attaches `SpecialKey.KEY_NAME.value` (string) to `func.__special_output__`.
*   Other decorators (e.g., `@torch_func`) set relevant attributes.

## 5. TUI Modification Context (`openhcs/tui/function_pattern_editor.py`)
*   **File to Modify:** [`openhcs/tui/function_pattern_editor.py`](openhcs/tui/function_pattern_editor.py:1)
*   **Required Changes (as per [`PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) v2.8, Section V):**
    *   Parameter editing UI (e.g., in `_create_parameter_editor`) must:
        1.  Introspect `func.signature` and `func.__special_inputs__`/`func.__special_output__`.
        2.  For `SpecialKey` inputs: UI for VFS path linking (mandatory raw data).
        3.  For `SpecialKey` output: UI for VFS path definition (raw data).
        4.  Other parameters: UI for direct value input (`kwargs`).
*   Ensures TUI distinguishes VFS path config for raw `SpecialKey` data from direct value setting for behavioral `kwargs`.

This document aims to provide a clear, factual basis for the implementation phase, grounded in the agreed-upon plans and existing codebase structure.