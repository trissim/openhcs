# Runtime Value and Artifact Upgrade Plan

**Date:** 2026-04-25
**Branch:** `benchmark-platform`
**Status:** In progress; Phase 0 compiler contracts and Phase 1A runtime value validation are implemented
**Primary goal:** make OpenHCS runtime state rich enough to support CellProfiler-style named images, object labels, measurements, relationships, and native feature outputs without introducing fake wrapper layers.

---

## 1. Why This Plan Exists

The compiler/runtime refactor has already moved the critical execution contract away from raw `FunctionStep.func` and string-keyed step-plan dictionaries.

Current source-of-truth chain:

1. `ProcessingContext.step_plans[int]`
2. `CompiledStepPlan`
3. `CompiledStepPlan.compiled_function_pattern`
4. `CompiledFunctionGroup`
5. `CompiledFunctionInvocation`
6. `FunctionStepExecutionPlan`
7. `PatternGroupRuntime`

That means the next blocker is no longer "which callable runs where." The next blocker is "what kind of value is moving through runtime, where is it stored, and what semantic contract does it satisfy."

CellProfiler support requires OpenHCS to represent these concepts natively:

1. Named images
2. Named object label sets
3. Per-image and per-object measurements
4. Parent-child relationships
5. Exportable feature tables

These must become OpenHCS runtime concepts, not a CellProfiler workspace wrapper over opaque dicts.

---

## 2. Architectural Rule

Do not solve this by wrapping untyped dictionaries.

A new object is acceptable only if it owns a real invariant, validation rule, identity, schema, storage policy, or lifecycle transition. A class that only forwards into a dict is a local minimum and should be rejected.

Correct direction:

```text
compiled invocation contract
    -> typed runtime value
    -> typed store / filemanager boundary
    -> kind-aware materialization
    -> optional CellProfiler adapter
```

Incorrect direction:

```text
CellProfilerContextAdapter
    -> dict-backed object_set
    -> dict-backed measurements
    -> runtime guesses later
```

The adapter can exist later, but it must be thin over OpenHCS-owned state.

---

## 3. Current State

Completed refactor foundation:

1. `CompiledStepPlan` is a typed dataclass and the compiler/runtime source of truth.
2. Validation writes memory types and GPU/runtime facts into `CompiledStepPlan`.
3. `FunctionStepExecutionPlan` snapshots typed compiled plans and has no raw backing dict.
4. Function patterns are compiled into `CompiledFunctionPattern`.
5. Runtime executes `CompiledFunctionInvocation` groups instead of rediscovering callable identity.
6. Invocation-level artifact input/output keys are known before runtime execution.
7. `CallableContract` centralizes callable name, module, memory type, and artifact declaration extraction.
8. `NormalizedFunctionPattern` lowers raw `FunctionStep.func` syntax into grouped callable items before compilation.
9. `ArtifactGraph` owns artifact producer/consumer declarations, grouped output scope, invocation ownership, kind consistency, and materialization intent.
10. `ArtifactOutputPlan.kind` now preserves declared `ArtifactSpec.kind`, and producer/consumer kind mismatches fail during path planning.
11. `RuntimeValue`, `RuntimeValueSchema`, and `RuntimeStoragePolicy` exist as typed runtime artifact values with validation invariants.
12. Runtime normalizes and validates `StepResult` and tuple artifact outputs against compiled `ArtifactOutputPlan.kind` before saving to the memory VFS.
13. Existing tests cover compiled plans, compiled function pattern behavior, artifact graph behavior, runtime artifact validation, `StepResult`, and ZMQ integration smoke.

Known remaining weaknesses:

1. Runtime validation currently saves raw payloads to memory VFS after validation; no runtime value store is attached yet.
2. `StepResult` still accepts `artifacts: Mapping[str, Any]`; normalization now validates values, but the public return type is still permissive for compatibility.
3. Materialization is mostly path/materializer driven, not kind/schema driven.
4. `ProcessingContext` does not yet own named image/object/measurement/relationship runtime state.
5. Some compiler phases still mutate `FunctionStep.func` while preparing normalized patterns.
6. `StepSnapshot` and `CompilationSession` are not implemented yet.
7. `zarr_config` and filemanager payloads remain external boundary mappings, which is acceptable for now.

---

## 4. Target Architecture

### 4.1 Runtime Value Model

Introduce a typed value layer in `openhcs/core/artifacts.py` or a focused sibling module.

Candidate source-of-truth types:

```python
@dataclass(frozen=True, slots=True)
class RuntimeValue:
    key: ArtifactKey
    data: Any
    schema: RuntimeValueSchema | None = None
    storage: RuntimeStoragePolicy | None = None
```

```python
@dataclass(frozen=True, slots=True)
class RuntimeValueSchema:
    kind: ArtifactKind
    fields: tuple[FieldSpec, ...] = ()
    dimensions: tuple[str, ...] = ()
    object_name: str | None = None
    source_image_name: str | None = None
```

Important: this is not a dict wrapper. It owns:

1. The artifact identity.
2. The artifact kind.
3. Optional schema fields.
4. Optional dimensional semantics.
5. Validation against the compiled artifact contract.
6. Storage/materialization policy selection.

### 4.2 Native Runtime Stores

Add native stores only where there is a real invariant.

Recommended stores:

1. `NamedImageStore`
2. `ObjectLabelStore`
3. `MeasurementStore`
4. `RelationshipStore`

Each store must have explicit row/value types:

```python
@dataclass(frozen=True, slots=True)
class ObjectLabelSet:
    name: str
    labels: Any
    source_image: str | None
    dimensions: tuple[str, ...]
```

```python
@dataclass(frozen=True, slots=True)
class MeasurementTable:
    name: str
    rows: Any
    object_name: str | None
    schema: RuntimeValueSchema
```

```python
@dataclass(frozen=True, slots=True)
class ObjectRelationship:
    parent_object: str
    child_object: str
    parent_ids: Any
    child_ids: Any
```

The stores should be attached to `ProcessingContext` only after their lifecycle is clear. Until then, prefer a `RuntimeValueStore` owned by context and keyed by `ArtifactKey`.

### 4.3 Compiler Contract Additions

Extend artifact declarations so functions can declare the semantic kind at the source:

```python
@artifact_outputs(
    ArtifactSpec("nuclei", ArtifactKind.OBJECT_LABELS),
    ArtifactSpec("nuclei_measurements", ArtifactKind.MEASUREMENTS),
)
def identify_primary_objects(image):
    ...
```

Needed compiler behavior:

1. Preserve `ArtifactSpec.kind` into `ArtifactOutputPlan`.
2. Preserve `ArtifactSpec.kind` into `ArtifactInputPlan`.
3. Compile invocation ownership into `CompiledFunctionInvocation.artifact_output_keys`.
4. Fail at compile time if a consumer requests a name with incompatible kind.
5. Fail at runtime if a returned value does not satisfy the compiled kind contract.

### 4.4 Runtime Contract Additions

Upgrade `StepResult` from "image plus mapping of unknown values" to "image plus typed runtime values or coercible typed payloads."

Proposed transition:

1. Keep current tuple return support for compatibility.
2. Keep `StepResult(image, artifacts={...})`.
3. Add runtime normalization that turns returned payloads into `RuntimeValue`.
4. Require the normalized value kind to match the compiled `ArtifactOutputPlan.kind`.
5. Gradually encourage explicit `RuntimeValue` returns for richer outputs.

Runtime flow:

```text
_execute_function_core
    -> call function
    -> normalize return into main image + artifact values
    -> validate artifact names against invocation output keys
    -> validate artifact value kinds against ArtifactOutputPlan.kind
    -> store via RuntimeValueStore / filemanager boundary
```

### 4.5 Materialization Contract Additions

Materialization should become kind-aware.

Initial mapping:

| ArtifactKind | Default storage/materialization |
| --- | --- |
| `IMAGE` | image backend / zarr / disk image writer |
| `OBJECT_LABELS` | label image, ROI zip, zarr labels |
| `MEASUREMENTS` | CSV first, Parquet later |
| `RELATIONSHIPS` | CSV table with parent/child ids |
| `TABLE` | CSV or Parquet |
| `METADATA` | JSON/YAML |
| `SPECIAL` | existing materializer path |

Rules:

1. Explicit `MaterializationSpec` still wins.
2. If no explicit materializer exists, choose by `ArtifactKind`.
3. If no default exists for a kind, fail loud with the artifact name and invocation key.

---

## 5. Implementation Phases

### Phase 0: Nominal Compiler Contracts

Goal: make compiler metadata extraction more type-safe before runtime values depend on it.

This phase prevents the runtime-value upgrade from building on scattered callable attribute probes. Compiler phases should read callable memory types and artifact declarations through one typed source of truth.

Source-of-truth types:

1. `CallableContract`: function name, module, memory types, declared artifact inputs, and declared artifact outputs.
2. `NormalizedFunctionPattern`: grouped callable items with stable invocation identity.
3. `ArtifactGraph`: producer/consumer edges, artifact names, kinds, group scopes, and materialization intent.
4. `StepSnapshot`: ObjectState-derived resolved step config used by compiler phases instead of live step attribute probing.
5. `CompilationSession`: axis-scoped compiler context that owns config, ObjectState map, plan map, and orchestrator access.

Tasks:

1. Add `CallableContract` and route artifact planning plus compiled invocation construction through it.
2. Extend the compiled pattern model so invocation metadata comes from `CallableContract`.
3. Replace direct `getattr(func, "__artifact_outputs__", {})` and memory-type probes in compiler phases with contract reads.
4. Add `NormalizedFunctionPattern` once callable contracts are centralized.
5. Add `ArtifactGraph` to own artifact producer/consumer validation before runtime execution.
6. Add `StepSnapshot` so path planning and validation stop reading live step attributes directly.
7. Add `CompilationSession` to replace loose parameter threading across compiler phases.

Acceptance criteria:

1. Callable memory types and artifact declarations are extracted in exactly one nominal module.
2. Artifact planning, function-pattern compilation, and memory validation all use the same callable contract.
3. Artifact kind mismatches can be detected at compile time once runtime values are added.
4. No new class is a pass-through wrapper over a dict; each type owns an invariant or phase boundary.

Current progress:

1. Done: `CallableContract` is the single callable metadata extraction boundary for memory types and artifact declarations.
2. Done: `NormalizedFunctionPattern` is the compiler input for grouped callable chains and stable invocation identity.
3. Done: `ArtifactGraph` replaced the loose artifact declaration bag and now validates producer/consumer kinds before runtime.
4. Remaining: `StepSnapshot` for ObjectState-derived step config.
5. Remaining: `CompilationSession` for axis-scoped compiler state and reduced loose parameter threading.

### Phase 1: Runtime Value Contract

Goal: typed value semantics without changing external pipeline behavior.

Tasks:

1. Add `RuntimeValue`, `RuntimeValueSchema`, and `RuntimeStoragePolicy`.
2. Add `normalize_artifact_value(...)`.
3. Add `validate_runtime_value(...)`.
4. Update `_execute_function_core` to normalize and validate planned artifact outputs.
5. Keep current tuple and `StepResult` behavior passing.
6. Add tests for each `ArtifactKind` mismatch path.

Acceptance criteria:

1. A function returning `StepResult(image, artifacts={"measurements": rows})` produces a `RuntimeValue` with `ArtifactKind.MEASUREMENTS`.
2. Returning a measurements payload for an `OBJECT_LABELS` output fails loud.
3. Returning an undeclared artifact fails loud.
4. Missing a declared artifact continues to fail loud.
5. Current unit tests still pass.

Current progress:

1. Done: `RuntimeValue`, `RuntimeValueSchema`, `RuntimeStoragePolicy`, `normalize_artifact_value`, and `validate_runtime_value` exist in `openhcs/core/runtime_values.py`.
2. Done: Runtime validates `StepResult` and tuple artifact values against compiled `ArtifactOutputPlan.kind`.
3. Done: The memory VFS still receives the raw payload after validation, preserving existing materializer behavior for this phase.
4. Remaining: explicit runtime value store attachment so validated `RuntimeValue` objects remain discoverable by typed key.
5. Remaining: broader kind mismatch coverage for every `ArtifactKind` path as defaults/materializers are added.

### Phase 2: Runtime Value Store

Goal: centralize artifact runtime state and remove direct generic VFS writes from function execution.

Tasks:

1. Add `RuntimeValueStore`.
2. Attach it to `ProcessingContext` or create it during `FunctionStepExecutionPlan` construction.
3. Move `_save_artifact_value` from raw filemanager writes to the store.
4. Keep filemanager memory VFS as the persistence boundary until materialization is upgraded.
5. Add retrieval by `ArtifactKey`, artifact name, group, and invocation.

Acceptance criteria:

1. Artifact values are discoverable by typed key.
2. Existing artifact inputs still load correctly.
3. Function execution no longer treats all artifact values as untyped VFS blobs internally.

### Phase 3: Kind-Aware Materialization

Goal: materialization follows artifact kind and schema rather than only ad hoc materializer options.

Tasks:

1. Add a default materializer registry keyed by `ArtifactKind`.
2. Route `materialize_artifact_outputs` through the registry when `ArtifactOutputPlan.materialization` is absent.
3. Add default CSV materialization for `MEASUREMENTS`, `RELATIONSHIPS`, and `TABLE`.
4. Add default JSON materialization for `METADATA`.
5. Preserve existing explicit `MaterializationSpec` behavior.

Acceptance criteria:

1. Measurement artifacts materialize to table output without custom per-function glue.
2. Relationship artifacts materialize to table output.
3. Metadata artifacts materialize to JSON.
4. Existing ROI/materializer tests still pass.

### Phase 4: Native Object and Measurement Concepts

Goal: support CellProfiler semantics as OpenHCS-native runtime concepts.

Tasks:

1. Add `ObjectLabelSet`.
2. Add `MeasurementTable`.
3. Add `ObjectRelationship`.
4. Add stores for named images, object labels, measurements, and relationships, or a unified `RuntimeValueStore` with typed views.
5. Add validators for object/measurement relationship integrity.
6. Add producer/consumer tests:
   - identify objects produces `OBJECT_LABELS`
   - measurement step consumes `OBJECT_LABELS` and produces `MEASUREMENTS`
   - relationship step consumes two object sets and produces `RELATIONSHIPS`

Acceptance criteria:

1. Named object labels can be produced and consumed by later invocations.
2. Measurement rows can reference object names and object ids.
3. Relationships can validate parent/child object names.
4. No CellProfiler adapter is required for these semantics.

### Phase 5: Compile-Time Symbol Resolution for CellProfiler

Goal: translate CellProfiler named workspace references into compiled OpenHCS artifact contracts.

Tasks:

1. Build a `.cppipe` symbol table:
   - image name -> input channel or upstream image artifact
   - object name -> object label artifact
   - measurement feature -> measurement artifact
   - relationship name -> relationship artifact
2. Generate `ArtifactSpec` inputs/outputs from the symbol table.
3. Fail at conversion time for unresolved names.
4. Preserve the compiled graph as the execution source of truth.

Acceptance criteria:

1. `get_objects("Nuclei")` becomes an explicit artifact input.
2. `add_objects(..., "Nuclei")` becomes an explicit `OBJECT_LABELS` output.
3. `measurements.add_measurement("Nuclei", ...)` becomes an explicit `MEASUREMENTS` output.
4. The generated OpenHCS pipeline does not need a mutable CellProfiler workspace to route data.

### Phase 6: Thin CellProfiler Compatibility Adapter

Goal: provide compatibility only after OpenHCS owns the real runtime model.

Tasks:

1. Implement `CellProfilerContextAdapter` as a view over OpenHCS runtime stores.
2. Keep the adapter read/write methods delegated to typed stores.
3. Reject adapter-only hidden state.
4. Add parity tests against small real CellProfiler modules.

Acceptance criteria:

1. Adapter can run a minimal CellProfiler module.
2. All produced values are visible as OpenHCS `RuntimeValue` objects.
3. Removing the adapter does not remove runtime semantics.

---

## 6. Testing Strategy

### Unit Tests

Add tests for:

1. `RuntimeValue` validation.
2. `StepResult` normalization.
3. Tuple return normalization.
4. Artifact kind mismatch failure.
5. Invocation-level artifact ownership.
6. Runtime store put/get/list behavior.
7. Kind-aware materialization dispatch.

### Integration Tests

Keep existing integration coverage:

1. ImageXpress 3D disk backend.
2. Multiprocessing execution.
3. ZMQ mode.
4. No streaming visualizer.

Add new integration tests:

1. FunctionStep produces `OBJECT_LABELS`.
2. Later FunctionStep consumes `OBJECT_LABELS`.
3. Measurement FunctionStep produces `MEASUREMENTS`.
4. Relationship FunctionStep produces `RELATIONSHIPS`.
5. Materialized outputs are written to expected backend paths.

### Regression Commands

Use the project venv:

```bash
source /home/ts/code/projects/openhcs/.venv/bin/activate
```

Use current external source paths:

```bash
export PYTHONPATH="external/PolyStore/src:external/ObjectState/src:external/arraybridge/src:external/metaclass-registry/src:external/pycodify/src:external/pyqt-reactive/src:external/python-introspect/src:external/zmqruntime/src:${PYTHONPATH}"
```

Baseline unit run:

```bash
/home/ts/code/projects/openhcs/.venv/bin/pytest -q tests/unit
```

ZMQ ImageXpress 3D smoke:

```bash
OPENHCS_HEADLESS=true \
OPENHCS_DISABLE_NAPARI=true \
OPENHCS_DISABLE_FIJI=true \
/home/ts/code/projects/openhcs/.venv/bin/pytest -q tests/integration/test_main.py \
  --it-backends disk \
  --it-microscopes ImageXpress \
  --it-dims 3d \
  --it-exec-mode multiprocessing \
  --it-zmq-mode zmq \
  --it-visualizers none \
  --it-sequential none
```

---

## 7. External Dependency Policy

Before making changes that depend on external repositories:

1. Fetch the latest upstream branch for the external repo.
2. Verify the checked-out SHA.
3. Prefer current upstream APIs over stale vendored assumptions.
4. Record any external SHA changes in the commit message or PR description.
5. Do not silently rely on stale submodule state for CellProfiler, zmqruntime, ObjectState, PolyStore, or arraybridge integration behavior.

---

## 8. File-Level Work Map

Likely files to touch:

1. `openhcs/core/artifacts.py`
2. `openhcs/core/steps/function_runtime.py`
3. `openhcs/core/steps/function_artifact_materialization.py`
4. `openhcs/core/steps/function_plan.py`
5. `openhcs/core/context/processing_context.py`
6. `openhcs/core/pipeline/function_contracts.py`
7. `openhcs/core/pipeline/artifact_planning.py`
8. `openhcs/core/pipeline/path_planner.py`
9. `openhcs/core/function_patterns.py`
10. `openhcs/processing/materialization.py`
11. `benchmark/converter/*`
12. `benchmark/cellprofiler_library/functions/*`

New files likely needed:

1. `openhcs/core/runtime_values.py` if `artifacts.py` becomes too large.
2. `openhcs/core/runtime_stores.py` if context-owned stores are introduced.
3. `tests/unit/test_runtime_values.py`
4. `tests/unit/test_runtime_value_store.py`
5. `tests/integration/test_runtime_artifacts.py`

---

## 9. Advisor Checkpoints

Run `nominal-refactor-advisor` after each phase on:

```bash
nominal-refactor-advisor --plans-only openhcs/core/artifacts.py
nominal-refactor-advisor --plans-only openhcs/core/steps
nominal-refactor-advisor --plans-only openhcs/core/pipeline
nominal-refactor-advisor --plans-only openhcs/processing
```

Use advisor findings as pressure tests, not as automatic instructions.

Reject suggested refactors that only create wrappers without new invariants.

Prioritize findings about:

1. semantic dict bags,
2. repeated projection dicts,
3. hidden attribute probing,
4. missing source-of-truth mappings,
5. oversized orchestration boundaries.

---

## 10. Migration Rules

1. Do not break existing `@artifact_outputs("name")` declarations.
2. Do not break tuple return compatibility in the first phase.
3. Prefer adding typed contracts before changing existing functions.
4. Move CellProfiler absorbed functions only after the runtime can represent their outputs.
5. Keep commits small enough to review:
   - one source-of-truth change,
   - one runtime behavior change,
   - one materialization change,
   - one integration test slice.

---

## 11. Completion Criteria

This upgrade is complete when:

1. Artifact values have a typed runtime representation.
2. Runtime validates artifact kind and schema against compiled invocation plans.
3. Named object labels, measurements, and relationships are OpenHCS-native values.
4. Materialization has default behavior for measurements, relationships, tables, metadata, and labels.
5. A generated CellProfiler pipeline can compile to explicit artifact contracts.
6. A minimal CellProfiler-style object measurement pipeline passes end-to-end without hidden mutable workspace state.
7. Existing ZMQ/ImageXpress 3D integration smoke still passes.
