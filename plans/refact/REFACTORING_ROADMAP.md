# OpenHCS Refactoring Implementation Roadmap

**Date:** 2025-05-20
**Version:** 1.0

## 1. Introduction

This document outlines a recommended sequence of implementation steps for the OpenHCS refactoring effort. It is designed to guide developers through the changes detailed in the primary planning documents, ensuring that dependencies are addressed logically.

**Core Planning Documents to Reference:**
*   Overall Summary: [`PLANS_REFACTORING_OPENHCS.md`](plans/PLANS_REFACTORING_OPENHCS.md:1) (v1.9)
*   Core Pipeline & VFS Logic: [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0)
*   Module & TUI Integration: [`plans/PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) (v2.8)
*   Configuration System: [`plans/PLANS_CONFIG_REFACTOR.md`](plans/PLANS_CONFIG_REFACTOR.md:1) (v1.0)
*   Implementation Context & Code Refs: [`plans/REFACTORING_IMPLEMENTATION_CONTEXT.md`](plans/REFACTORING_IMPLEMENTATION_CONTEXT.md:1) (v1.0)

## 2. Prerequisites
*   Thoroughly review all linked planning documents.
*   Familiarize yourself with the existing OpenHCS codebase, particularly the components targeted for refactoring.
*   Set up a dedicated development branch for this refactoring work.

## 3. Implementation Phases and Tasks

The refactoring is broken down into phases to manage dependencies and allow for iterative testing.

### Phase 1: Core Pipeline Logic Refactoring
*Goal: Implement the new stateless `FunctionStep` execution model, VFS-driven I/O, and updated `step_plan` structure.*

*   **Task 2.1: Refactor Pipeline Planners & Compiler**
    *   **Files:** [`openhcs/core/pipeline/compiler.py`](openhcs/core/pipeline/compiler.py:1), [`openhcs/core/pipeline/path_planner.py`](openhcs/core/pipeline/path_planner.py:1), [`openhcs/core/pipeline/materialization_flag_planner.py`](openhcs/core/pipeline/materialization_flag_planner.py:1).
    *   **Action:**
        *   Modify planners to determine VFS information (paths, backends, materialization flags).
        *   Modify `PipelineCompiler` to populate the **flat `step_plan` structure** as defined in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0 (Section I.3 & 2.2 of [`REFACTORING_IMPLEMENTATION_CONTEXT.md`](plans/REFACTORING_IMPLEMENTATION_CONTEXT.md:1)). This includes `input_dir`, `output_dir`, `read_backend`, `write_backend`, `force_disk_output`, `expected_special_inputs`, `produces_special_output`, `special_input_vfs_info` (dict), `special_output_vfs_info` (dict/None), `func` (resolved pattern), etc.
        *   Incorporate `@chain_breaker` logic in `PipelinePathPlanner`.
        *   Ensure `PipelineCompiler` handles linking of `SpecialKey` outputs to inputs by assigning consistent VFS paths in respective `step_plan`s.
        *   Move `step_plan` validation logic into the `PipelineCompiler`.
    *   **Reference:** [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0 - Section I.3).

*   **Task 2.2: Refactor `FunctionStep`**
    *   **File:** [`openhcs/core/steps/function_step.py`](openhcs/core/steps/function_step.py:1)
    *   **Action:**
        *   Implement `FunctionStep.process` and its helper `_process_single_pattern_vfs` according to the detailed pseudo-code in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0 - Section I.4). This involves:
            *   Using the flat `step_plan` from `context`.
            *   Using `FileManager` to retrieve and store data.
            *   Preparing array-like objects for `stack_slices()`.
            *   Passing data from backends as positional arguments for `@special_input`s.
            *   Receiving data as return for `@special_output`s.
            *   Handling result from `unstack_slices()` before storage.
            *   Using `FileManager` for all I/O operations.
        *   Remove `StepResult` usage and return `None`.
        *   Remove the runtime `validate_step_plan` helper function.
    *   **Reference:** [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0 - Section I.4).

*   **Task 2.3: Refactor `PipelineOrchestrator`**
    *   **File:** [`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py:1)
    *   **Action:** Implement the two-phase (Compile All, then Execute All) execution model.
    *   **Reference:** [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0 - Section I.1).

*   **Task 2.4: Update `AbstractStep` and Other Step Types**
    *   **Files:** [`openhcs/core/steps/abstract.py`](openhcs/core/steps/abstract.py:1) and other step implementations.
    *   **Action:** Change `process` method signature to return `None`. Remove any reliance on `StepResult`. Ensure other step types align with VFS-only I/O if they perform file operations.

### Phase 2: Module & Function Adjustments
*Goal: Align all processing functions and modules with the new data flow and SpecialKey conventions.*

*   **Task 2.1: Update Processing Functions**
    *   **Files:** Various files in `openhcs/processing/backends/`.
    *   **Action:**
        *   Ensure functions decorated with `@special_input(SpecialKey.KEY_NAME)` expect corresponding raw bytes as positional arguments and perform internal deserialization.
        *   Ensure functions decorated with `@special_output(SpecialKey.KEY_NAME)` perform internal serialization and return raw bytes.
        *   Update regular `kwargs` as needed.
    *   **Reference:** [`plans/PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) (v2.8 - Section I.1).

*   **Task 2.2: Update Other Modules**
    *   **Files:** E.g., [`openhcs/microscopes/opera_phenix.py`](openhcs/microscopes/opera_phenix.py:1) (for `OperaPhenixXmlParser`).
    *   **Action:** Ensure components like metadata parsers correctly use `FileManager.open()` and handle the raw byte/string data.
    *   **Reference:** [`plans/PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) (v2.8 - Sections I.2, etc.).

### Phase 3: TUI Adjustments
*Goal: Update the TUI to correctly configure steps according to the new parameter model.*

*   **Task 3.1: Modify `FunctionPatternEditor`**
    *   **File:** [`openhcs/tui/function_pattern_editor.py`](openhcs/tui/function_pattern_editor.py:1)
    *   **Action:**
        *   Enhance function introspection logic (e.g., in `_create_parameter_editor` or helpers) to check `func.__special_inputs__` and `func.__special_output__` (attributes set by decorators).
        *   Implement differentiated UI:
            *   For `SpecialKey` inputs: UI for VFS path linking (raw data).
            *   For `SpecialKey` outputs: UI for VFS path definition (raw data).
            *   For other parameters: UI for direct value input (`kwargs`).
    *   **Reference:** [`plans/PLANS_MODULES_TUI_INTEGRATION.md`](plans/PLANS_MODULES_TUI_INTEGRATION.md:1) (v2.8 - Section V) and [`plans/REFACTORING_IMPLEMENTATION_CONTEXT.md`](plans/REFACTORING_IMPLEMENTATION_CONTEXT.md:1) (v1.0 - Section 5).

### Phase 4: Configuration System Refactoring (Potentially Parallelizable)
*Goal: Implement changes to the configuration loading and management system.*

*   **Task 4.1: Implement Configuration Changes**
    *   **Action:** Apply the refactoring detailed in [`plans/PLANS_CONFIG_REFACTOR.md`](plans/PLANS_CONFIG_REFACTOR.md:1) (v1.0).
    *   Ensure alignment with new VFS paths and `step_plan` structure where configuration might influence default paths or backends.

## 4. Testing Strategy
*   **Unit Tests:** Crucial for `FunctionStep` logic, individual planners, and the compiler. Test how the system interacts with `FileManager` and storage backends.
*   **Integration Tests:**
    *   Test `PipelineCompiler`'s ability to generate correct `step_plan`s.
    *   Test `PipelineOrchestrator` and `PipelineExecutor` with full pipeline runs involving primary I/O and various `SpecialKey` I/O patterns (including linked special I/O between steps and `@chain_breaker` effects).
    *   Verify VFS operations across different backends ("memory", "disk").
*   **TUI Tests:** Test the `FunctionPatternEditor`'s ability to correctly introspect functions and generate the differentiated UI for `SpecialKey` VFS path linking versus `kwarg` editing. Test saving and loading of pipeline definitions with these configurations.

## 5. Document Review
*   After each major phase or significant component refactor, revisit all five planning documents to ensure they remain consistent and to note any deviations or new insights discovered during implementation.

This roadmap provides a structured approach to a complex refactoring task. Flexibility may be needed, but adhering to the phase dependencies should minimize integration issues.