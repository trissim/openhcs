# Clause 283 — Well-Scoped Materialization Enforcement

## Canonical Definition

All step executions in OpenHCS are scoped to a single well. Even image-to-image steps must receive `well_id`, as their declared `input_path` and `output_path` refer to shared plate-level directories. Each step is responsible for isolating its materialization behavior to its assigned `well_id`. This requirement is universal and must be enforced structurally across all step types.

## Rationale

1. **Isolation**: Each well's processing must be isolated from other wells to prevent cross-contamination of data.
2. **Parallelization**: Well-scoped processing enables parallel execution of pipelines for different wells.
3. **Traceability**: Explicit well identification enables better tracing and debugging of pipeline execution.
4. **Determinism**: Well-scoped materialization ensures deterministic behavior regardless of execution order.

## Implementation Requirements

1. **Universal Requirement**: All steps must receive `well_id` as a constructor parameter, regardless of whether they appear to use it internally.
2. **First-Class Field**: `well_id` must be a first-class field in all pipeline construction structures (`PipelineBlueprint`, `StepDeclaration`, `PipelineAssembler`), never stored in metadata, dynamic attributes, or side channels.
3. **Explicit Declaration**: Steps must declare `well_id` in their `get_required_fields()` method if they consume it.
4. **Context Binding**: `well_id` must be passed explicitly via the pipeline execution context, validated by each step at runtime.
5. **No Inference**: Steps must never infer `well_id` from paths or other fields.
6. **No Defaults**: `well_id` must never have a default value; it must always be explicitly provided.
7. **Validation**: Steps must validate that `well_id` is provided and non-empty.
8. **Logging**: All step execution logs must include `well_id` for traceability.

## Enforcement

1. **Constructor Validation**: All step constructors must validate that `well_id` is provided and non-empty.
2. **StepDeclaration Validation**: `StepDeclaration.__post_init__()` must validate that `well_id` is provided and non-empty.
3. **Context Validation**: Steps must validate that `well_id` is present in the context before execution.
4. **Dependency Declaration**: All steps must declare `well_id` in their `get_required_fields()` method.
5. **Blueprint Validation**: `PipelineBlueprint.from_path_planner()` must validate that `well_id` is provided and non-empty.

## Related Clauses

- **Clause 21 — Context Immunity**: Steps must not modify the context directly; all modifications must be returned as part of the `StepResult`.
- **Clause 66 — Immutability After Construction**: Steps must be immutable after construction, including their `well_id`.
- **Clause 88 — No Inferred Capabilities**: Steps must not infer capabilities from runtime properties; `well_id` must be explicitly declared.
- **Clause 281 — Context-Bound Identifiers**: All execution-variant information (e.g., `well_id`, `plate_id`, `device_id`) must be passed explicitly via the pipeline execution context.

## Examples

### Correct Usage

```python
# Step declaration with explicit well_id
step_decl = StepDeclaration(
    step_cls=ImageAssemblyStep,
    input_path=input_path,
    output_path=output_path,
    positions_folder=positions_folder,
    input_memory_type="numpy",
    output_memory_type="numpy",
    well_id="A01"  # Explicitly provided
)

# Step instantiation with explicit well_id
step = ImageAssemblyStep(
    backend=backend,
    input_memory_type="numpy",
    output_memory_type="numpy",
    positions_folder=positions_folder,
    well_id="A01"  # Explicitly provided
)

# Context access with explicit well_id validation
well_id = context.get_validated("well_id")
if well_id is None:
    raise ValueError("well_id is required for ImageAssemblyStep (Clause 281, Clause 283)")
```

### Incorrect Usage

```python
# INCORRECT: Step declaration without well_id
step_decl = StepDeclaration(
    step_cls=ImageAssemblyStep,
    input_path=input_path,
    output_path=output_path,
    positions_folder=positions_folder,
    input_memory_type="numpy",
    output_memory_type="numpy"
    # Missing well_id
)

# INCORRECT: Step instantiation without well_id
step = ImageAssemblyStep(
    backend=backend,
    input_memory_type="numpy",
    output_memory_type="numpy",
    positions_folder=positions_folder
    # Missing well_id
)

# INCORRECT: Step instantiation with well_id in metadata
step = ImageAssemblyStep(
    backend=backend,
    input_memory_type="numpy",
    output_memory_type="numpy",
    positions_folder=positions_folder,
    metadata={"well_id": "A01"}  # well_id in metadata instead of as a first-class field
)

# INCORRECT: Inferring well_id from path
well_id = extract_well_id_from_path(context.input_path)  # Inferring well_id from path
```
