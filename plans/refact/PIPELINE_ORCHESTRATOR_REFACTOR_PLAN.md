# Refactoring Plan: `PipelineOrchestrator` and Core Execution Model

**Date:** 2025-05-21
**Version:** 1.3 (Incorporate Napari FileManager Integration details)

## 1. Overview

This document outlines the refactoring plan for the `PipelineOrchestrator` and associated core components (`ProcessingContext`, `PipelineCompiler`, `PipelineExecutor`, `AbstractStep`) within the OpenHCS project. The primary goal is to transition to a strict two-phase (compile-all-then-execute-all) execution model, enforce `ProcessingContext` immutability after compilation, and centralize pipeline compilation for a batch of wells. This plan also incorporates considerations for integrating with a refactored `NapariStreamVisualizer` that uses `FileManager`.

## 2. Core Principles Guiding the Refactor

1.  **Two Orchestrator Phases:** A distinct compilation phase that prepares all wells, followed by an execution phase.
2.  **Per-Well Compilation into Context:** For each well, a `ProcessingContext` is created. The `PipelineCompiler` (and its sub-planners) will sequentially populate/inject data into the `step_plans` dictionary within that specific context.
3.  **Context Immutability:** After all compilation sub-steps for a specific well are complete, its `ProcessingContext` is "frozen" to prevent further modifications.
4.  **Stateless Steps for Execution:** The original list of `AbstractStep` objects (the pipeline definition) is stripped of its attributes after all wells are compiled. These (now stateless) step instances are then used for execution against each well's frozen `ProcessingContext`.
5.  **Elimination of `StepResult`:** Steps will no longer return `StepResult`. All inter-step data flow will occur via VFS (managed by `FileManager` and orchestrated through `step_plans`), and steps will return `None`.
6.  **VFS-Based Visualization:** Visualization via `NapariStreamVisualizer` will use `FileManager` to load data from VFS paths specified in step plans, rather than receiving direct tensor data.

## 3. Phased Implementation Plan

### Phase 1: Enhance `ProcessingContext` for Immutability

*   **File:** [`openhcs/core/context/processing_context.py`](openhcs/core/context/processing_context.py:1)
*   **Actions:** (As per [`plans/refact/PROCESSING_CONTEXT_IMMUTABILITY.md`](plans/refact/PROCESSING_CONTEXT_IMMUTABILITY.md:1))
    1.  Implement `_is_frozen` attribute, `freeze()`, `is_frozen()` methods.
    2.  Override `__setattr__` for immutability.
    3.  Remove `update_from_step_result()`.
    4.  Ensure `inject_plan()` checks `_is_frozen`.

### Phase 2: Refactor `PipelineCompiler` for Sequential Plan Injection

*   **File:** [`openhcs/core/pipeline/compiler.py`](openhcs/core/pipeline/compiler.py:1)
*   **Actions:**
    1.  **Decompose `PipelineCompiler.compile()`** into sequential, injectable phases.
    2.  **New/Modified Compiler Methods (Static):**
        *   `initialize_step_plans_for_well(...)`
        *   `plan_paths_for_well(...)`
        *   `plan_materialization_flags_for_well(...)`
        *   `validate_memory_contracts_for_well(...)`
        *   `assign_gpu_resources_for_well(...)`
        *   Ensure these methods correctly populate `output_dir`, `write_backend`, and `visualize` flags in `context.step_plans` for use by the visualizer.
    3.  `StepAttributeStripper` invoked by Orchestrator post-compilation.

### Phase 3: Refactor `PipelineOrchestrator`

*   **File:** [`openhcs/core/orchestrator/orchestrator.py`](openhcs/core/orchestrator/orchestrator.py:1)
*   **Actions:**
    1.  **New Method: `compile_plate_for_processing(...)`** (as previously detailed).
    2.  **New Method: `execute_compiled_plate(...)`** (as previously detailed, with parallel execution).
        *   **Napari Integration Point:** When `visualizer_instance` is created within this method (or the `run` method that calls it), it must be instantiated with `self.filemanager`:
            ```python
            if enable_visualizer_override or any(...): # Logic to check if visualization is needed
                logger.info("Visualization requested.")
                # Assuming NapariStreamVisualizer is imported
                visualizer_instance = NapariStreamVisualizer(filemanager=self.filemanager) 
            ```
    3.  **New Helper Method (private): `_execute_single_well(...)`** (as previously detailed).
        *   **Napari Integration Point:** Within this method, after `step.process(frozen_context)`, add logic to call the refactored visualizer:
            ```python
            # Inside _execute_single_well(), after a step.process(frozen_context) call:
            if visualizer_instance: # Check if visualizer is active
                step_plan = frozen_context.get_step_plan(step.uid) 
                if step_plan.get('visualize', False):
                    output_dir = step_plan.get('output_dir')
                    # Default to 'disk' if write_backend not in plan, or ensure it's always populated
                    write_backend = step_plan.get('write_backend', 'disk') 
                    if output_dir:
                        visualizer_instance.visualize_path(
                            step_id=step.uid,
                            path=str(output_dir), 
                            backend=write_backend,
                            well_id=frozen_context.well_id
                        )
            ```
    4.  **Update `run()` Method:** (Signature: `run(self, pipeline_definition: List[AbstractStep], ...)` as previously detailed).
        *   Coordinates calls to `compile_plate_for_processing` and `execute_compiled_plate`.
        *   Handles instantiation of `NapariStreamVisualizer` (passing `self.filemanager`) and its lifecycle (`stop_viewer`).

### Phase 4: Update `PipelineExecutor` and `AbstractStep` Implementations

*   **`PipelineExecutor` ([`openhcs/core/pipeline/executor.py`](openhcs/core/pipeline/executor.py:1)):**
    *   Role likely diminishes; core loop moves to `PipelineOrchestrator._execute_single_well`.
    *   If retained for any utility, must align with frozen context and `step.process() -> None`.
*   **`AbstractStep` ([`openhcs/core/steps/abstract.py`](openhcs/core/steps/abstract.py:1)) and subclasses:**
    *   `process()` method signature changes to `-> None`.
    *   Remove `StepResult` usage. Rely on frozen `context.step_plans` and `context.filemanager`.

## 4. Mermaid Diagram: New Orchestrator Flow

*(Mermaid diagram from Version 1.2 remains largely applicable, as visualization is a detail within the execution flow of `_execute_single_well`)*
```mermaid
graph TD
    subgraph PipelineOrchestrator
        direction LR
        ORUN[run method]
        OCPL[compile_plate_for_processing]
        OEXC[execute_compiled_plate]
        OESW[_execute_single_well]
    end

    subgraph PipelineCompilerPhases
        direction TB
        C_INIT[initialize_step_plans_for_well]
        C_PATH[plan_paths_for_well]
        C_MAT[plan_materialization_flags_for_well]
        C_MEM[validate_memory_contracts_for_well]
        C_GPU[assign_gpu_resources_for_well]
    end
    
    subgraph StepAttributeStripper
        direction TB
        STRIP[strip_step_attributes]
    end
    
    subgraph ProcessingContextLifeCycle
        direction TB
        CTX_CREATE[Create Context (per well)]
        CTX_PLANS[context.step_plans {}]
        CTX_POPULATE[Populate context.step_plans via Compiler Phases]
        CTX_FREEZE[context.freeze()]
    end
    
    subgraph NapariStreamVisualizer
        direction TB
        NAP_INIT[new NapariStreamVisualizer(filemanager)]
        NAP_VIS[visualize_path(path, backend)]
    end

    UserInputPipeline[User Pipeline Definition (List of Steps)] --> ORUN;
    
    ORUN -- Creates if needed --> NAP_INIT;
    ORUN --> OCPL;
    UserInputPipeline -- Passed to --> OCPL;
    
    OCPL --> ForEachWell{For Each Well};
    ForEachWell --> CTX_CREATE;
    CTX_CREATE --> CTX_PLANS;
    
    CTX_PLANS -- Target for --> C_INIT;
    C_INIT -- Injects into --> CTX_PLANS;
    C_INIT --> C_PATH;
    C_PATH -- Injects into --> CTX_PLANS;
    C_PATH --> C_MAT;
    C_MAT -- Injects into --> CTX_PLANS;
    C_MAT --> C_MEM;
    C_MEM -- Injects into --> CTX_PLANS;
    C_MEM --> C_GPU;
    C_GPU -- Injects into --> CTX_PLANS;
    C_GPU --> CTX_POPULATE;
    CTX_POPULATE --> CTX_FREEZE;
    
    CTX_FREEZE --> OCPL_CollectCtx{Collect Dict[well_id, FrozenContext]};
    OCPL_CollectCtx -- All Wells Done --> STRIP;
    UserInputPipeline -- Target for --> STRIP; 
    STRIP -- Returns Stateless Steps --> UserInputPipeline_Stateless[User Pipeline Definition (Stateless)];
    
    OCPL_CollectCtx --> OEXC_InputDict[Dict[well_id, FrozenContext]];
    OEXC_InputDict -- Passed to --> OEXC;
    UserInputPipeline_Stateless -- Passed to --> OEXC;
    NAP_INIT -- Passed to --> OEXC;
    ORUN --> OEXC;

    OEXC -- Manages ThreadPool --> OESW;
    NAP_INIT -- Passed to --> OESW;
    OESW -- Calls if step.visualize --> NAP_VIS;
    OESW -- Executes steps for one well --> FinalWellStatus[Well Execution Status / VFS Outputs];
    FinalWellStatus --> OEXC_CollectResults{Collect Well Results};
    OEXC_CollectResults --> FinalPlateResults[Overall Plate Results];

    classDef Orchestrator fill:#f9f,stroke:#333,stroke-width:2px;
    classDef Compiler fill:#9cf,stroke:#333,stroke-width:2px;
    classDef Context fill:#9f9,stroke:#333,stroke-width:2px;
    classDef Visualizer fill:#ff9,stroke:#333,stroke-width:2px;
    classDef Data fill:#fff,stroke:#333,stroke-width:1px,color:black;

    class ORUN,OCPL,OEXC,OESW Orchestrator;
    class C_INIT,C_PATH,C_MAT,C_MEM,C_GPU Compiler;
    class STRIP Compiler; 
    class CTX_CREATE,CTX_PLANS,CTX_POPULATE,CTX_FREEZE Context;
    class NAP_INIT,NAP_VIS Visualizer;
    class UserInputPipeline,ForEachWell,OCPL_CollectCtx,OEXC_InputDict,FinalWellStatus,FinalPlateResults,UserInputPipeline_Stateless Data;
```

## 5. Considerations & Open Questions

*   **`input_dir` per well:** (As before)
*   **`StepAttributeStripper` argument:** (As before)
*   **Error Handling in `compile_plate_for_processing`:** (As before)
*   **Error Handling in `execute_compiled_plate`:** (As before)
*   **Napari `FileManager` Synchronization:** Ensure the `FileManager` instance used by the `PipelineOrchestrator` (and thus passed to `NapariStreamVisualizer`) is the same one (or a compatible one) whose backends are populated with data by the pipeline steps.
*   **`NapariStreamVisualizer` Refactoring:** This plan assumes `NapariStreamVisualizer` is refactored as per [`plans/refact/NAPARI_FILEMANAGER_INTEGRATION.md`](plans/refact/NAPARI_FILEMANAGER_INTEGRATION.md:1) to accept `filemanager` and have the `visualize_path` method.

This plan provides a comprehensive roadmap for the refactoring, now updated to include Napari integration details.