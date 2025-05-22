# Plan for Refining `PipelineCompiler` for Special I/O (Take 12)

**Overall Objective:**
Modify `PipelineCompiler` to accurately plan and resolve VFS paths for `special_outputs` and `special_inputs`. The compiler will read lists/tuples of special I/O *keys* from attributes on the `Step` instances themselves (e.g., `step.special_outputs_keys`, `step.special_inputs_keys`). These attributes are assumed to be populated by the `Step`'s `__init__` method, which inspects the wrapped function for decorator-set attributes.

The outcome will be that `context.step_plans[step_id]` will contain:
*   `'special_outputs': OrderedDict[str, str]` (mapping output keys to their VFS save paths)
*   `'special_inputs': OrderedDict[str, str]` (mapping input argument names to the VFS paths of the data they consume, preserving order from the step's attribute)

**Affected File:** `openhcs/core/pipeline/compiler.py`

**Prerequisite/Assumption:**
*   `FunctionStep` (and similar step classes) will be updated so their `__init__` method inspects the passed `func` object for decorator-set attributes (like `__special_inputs__`, `__special_outputs__`) and copies these key lists/tuples to instance attributes (e.g., `self.special_inputs_keys`, `self.special_outputs_keys`).

---

## Phase A: Modifying `initialize_step_plans_for_context` for `special_outputs` Planning

1.  **Location:** Inside the loop `for step in steps_definition:` within `PipelineCompiler.initialize_step_plans_for_context`.
2.  **Logic to Add/Modify:**
    *   After `current_plan` is initialized with basic details.
    *   **Remove Old Logic:** Delete or comment out lines that previously attempted to get special I/O info from `PipelinePathPlanner` (approx. lines 129-135 in the version reviewed prior to this plan).
    *   **Plan `special_outputs`:**
        *   Retrieve `output_keys_tuple = getattr(step, 'special_outputs_keys', ())`. (This attribute is expected to be set on the `step` instance by its `__init__` method, derived from decorators on the original function).
        *   Import `OrderedDict` from `collections`.
        *   `special_outputs_map = OrderedDict()`
        *   If `output_keys_tuple`:
            *   For `key` in `output_keys_tuple`:
                *   `step_output_dir = Path(current_plan["output_dir"])`
                *   `step_uid = step.uid`
                *   Define a standardized path, e.g., `special_output_path = str(step_output_dir / step_uid / "special_outputs" / key)`.
                *   `special_outputs_map[key] = special_output_path`
        *   `current_plan['special_outputs'] = special_outputs_map` (Always set, even if empty).
    *   **Initialize `special_inputs`:**
        *   `current_plan['special_inputs'] = OrderedDict()` (To be populated in the next phase, ensuring key order is maintained if derived from an ordered `step.special_inputs_keys`).

---

## Phase B: Creating/Modifying `resolve_special_input_paths_for_context`

1.  **Static Method Signature (if new, or ensure it matches):**
    ```python
    @staticmethod
    def resolve_special_input_paths_for_context(
        context: ProcessingContext, 
        steps_definition: List[AbstractStep]
    ) -> None:
    ```
2.  **Invocation by Orchestrator:** This method must be called by `PipelineOrchestrator.compile_pipelines` *after* `initialize_step_plans_for_context` and *before* subsequent validation steps.
3.  **Logic within `resolve_special_input_paths_for_context`:**
    *   Iterate `for i, current_step in enumerate(steps_definition):`
        *   `step_id = current_step.uid`
        *   `current_plan = context.step_plans.get(step_id)`
        *   If not `current_plan`: continue.
        *   Retrieve `input_keys_tuple = getattr(current_step, 'special_inputs_keys', ())`. (Attribute set on `step` instance by its `__init__`).
        *   `resolved_inputs_map = OrderedDict()`
        *   If `input_keys_tuple`:
            *   For `consumed_key` in `input_keys_tuple`:
                *   `producer_path_found = False`
                *   Iterate backwards through preceding steps: `for j in range(i - 1, -1, -1):`
                    *   `producer_step = steps_definition[j]`
                    *   `producer_plan = context.step_plans.get(producer_step.uid)`
                    *   If `producer_plan` and `consumed_key` in producer_plan.get('special_outputs', {}):
                        *   `resolved_inputs_map[consumed_key] = producer_plan['special_outputs'][consumed_key]`
                        *   `producer_path_found = True`
                        *   Break from inner loop.
                *   If not `producer_path_found`:
                    *   Log an error and raise `ValueError(f"Unresolved special input '{consumed_key}' for step {current_step.name} (ID: {step_id}). No preceding step provides this as a special_output.")`.
        *   `current_plan['special_inputs'] = resolved_inputs_map` (Update existing plan, preserving order).

---

## Helper Method `_get_core_callable` in `PipelineCompiler`
*   This helper is **NO LONGER NEEDED** in `PipelineCompiler`. The responsibility of interpreting `step.func` and exposing contract attributes (like `special_inputs_keys`, `special_outputs_keys`) lies with the `Step` class's `__init__` method (e.g., `FunctionStep.__init__`).

---

## Updated Mermaid Diagram (Illustrating Compiler Phases - Take 12)

```mermaid
sequenceDiagram
    participant Orch as PipelineOrchestrator
    participant Compiler as PipelineCompiler
    participant PathPlanner as PipelinePathPlanner
    participant Ctx as ProcessingContext
    participant StepInstance as "Step Instance (e.g., FunctionStep)"

    Orch->>Compiler: compile_for_well(Ctx, pipeline_def, well_id, base_input_dir)
    
    %% Phase 1: Initialize Step Plans (including Special Outputs)
    Compiler->>Compiler: initialize_step_plans_for_context(Ctx, pipeline_def, base_input_dir, well_id)
    Note right of Compiler: - Calls PathPlanner for primary paths <br> - Initializes basic step_plan entries <br> - For each StepInstance: <br>   - Reads `StepInstance.special_outputs_keys` (tuple of keys) <br>   - Generates VFS paths (e.g., output_dir/step_uid/special_outputs/key) <br>   - Stores OrderedDict in Ctx.step_plans[step_id]['special_outputs'] <br>   - Initializes Ctx.step_plans[step_id]['special_inputs'] as empty OrderedDict
    
    %% Phase 2: Resolve Special Inputs
    Compiler->>Compiler: resolve_special_input_paths_for_context(Ctx, pipeline_def)
    Note right of Compiler: For each StepInstance: <br> - Reads `StepInstance.special_inputs_keys` (tuple of keys) <br> - For each input_key, searches previous steps' <br>   Ctx.step_plans[...]['special_outputs'] for a matching key & its path. <br> - Stores OrderedDict in Ctx.step_plans[step_id]['special_inputs']
    
    %% Subsequent Phases (Materialization, Contracts, GPU, Viz) - as before
    Compiler->>Compiler: plan_materialization_flags_for_context(...)
    Compiler->>Compiler: validate_memory_contracts_for_context(...)
    Compiler->>Compiler: assign_gpu_resources_for_context(...)
    Compiler->>Compiler: apply_global_visualizer_override_for_context(...)
    
    Compiler-->>Orch: Compiled Ctx (now frozen by Orchestrator after all phases)