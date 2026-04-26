# Runtime Value and Artifact Upgrade Plan

**Date:** 2026-04-25
**Branch:** `benchmark-platform`
**Status:** In progress; Phase 0 compiler contracts, Phase 1 runtime value validation, and Phase 2 runtime value store/VFS access are implemented
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
13. `RuntimeValueStore` is attached to `ProcessingContext` and records validated runtime values by typed artifact key while preserving VFS as the payload storage boundary.
14. `RuntimeArtifactLocation` and `RuntimeArtifactQuery` are the SSOT for VFS-backed runtime artifact lookup.
15. Artifact inputs and explicit artifact materialization resolve through typed store records before loading VFS payloads; missing or ambiguous records fail loudly.
16. Existing tests cover compiled plans, compiled function pattern behavior, artifact graph behavior, runtime artifact validation, runtime value store behavior, materialization store lookup, `StepResult`, and ZMQ integration smoke.

Known remaining weaknesses:

1. VFS is intentionally still the payload persistence boundary, but artifact reads now require typed runtime-store records.
2. `StepResult` still accepts `artifacts: Mapping[str, Any]`; normalization now validates values, but the public return type is still permissive for compatibility.
3. Materialization now requires typed store metadata for explicit materializers, but default materialization is not yet selected by `ArtifactKind`.
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
2. If no explicit materializer exists for a semantic kind, choose an existing writer-backed `MaterializationSpec` by `ArtifactKind`.
3. If no default exists for a semantic kind, fail loud with the artifact name and invocation key.
4. `SPECIAL` remains explicit-only for legacy side-channel artifacts.

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
4. Done: `RuntimeValueStore` records validated values by typed `ArtifactKey`.
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

Current progress:

1. Done: `RuntimeValueStore` is attached to `ProcessingContext`.
2. Done: `_save_artifact_value` records validated `RuntimeValue` objects before saving raw payloads to memory VFS.
3. Done: artifact values are discoverable by typed key and by semantic filters such as name, kind, axis, and group.
4. Done: artifact inputs load from VFS through the typed `RuntimeValueStore` record; missing or ambiguous records fail loudly instead of falling back to direct VFS reads.
5. Done: explicit artifact materialization loads VFS payloads through typed `RuntimeValueStore` records and fails loudly when typed metadata or VFS payloads are missing.
6. Done: `RuntimeArtifactLocation` and `RuntimeArtifactQuery` collapse repeated path/backend/name/kind/axis matching into the store.
7. Done: default semantic materialization reuses existing `MaterializationSpec`, `CsvOptions`, `JsonOptions`, and presets instead of creating new writer infrastructure.
8. Remaining: extend default policy only after native schemas clarify `OBJECT_LABELS` and `IMAGE` semantics.

### Phase 3: Kind-Aware Materialization

Goal: materialization follows artifact kind and schema by selecting existing writer-backed `MaterializationSpec` presets, rather than creating parallel materialization infrastructure.

Tasks:

1. Add a default materialization policy keyed by `ArtifactKind`.
2. Route `materialize_artifact_outputs` through the policy when `ArtifactOutputPlan.materialization` is absent.
3. Reuse existing `csv_only(...)` for `MEASUREMENTS`, `RELATIONSHIPS`, and `TABLE`.
4. Reuse existing `json_only(...)` for `METADATA`.
5. Preserve existing explicit `MaterializationSpec` behavior.
6. Keep `SPECIAL` explicit-only.

Acceptance criteria:

1. Measurement artifacts materialize to table output without custom per-function glue.
2. Relationship artifacts materialize to table output.
3. Metadata artifacts materialize to JSON.
4. Existing ROI/materializer tests still pass.

Current progress:

1. Done: explicit `MaterializationSpec` outputs are resolved through typed runtime-store records before loading VFS payloads.
2. Done: materialization no longer silently skips planned artifacts with missing typed runtime metadata.
3. Done: added a thin `ArtifactKind` policy layer over existing materialization presets.
4. Done: `MEASUREMENTS`, `RELATIONSHIPS`, and `TABLE` default to existing CSV materialization.
5. Done: `METADATA` defaults to existing JSON materialization.
6. Done: `SPECIAL` remains explicit-only.
7. Remaining: decide `OBJECT_LABELS` and `IMAGE` defaults after native runtime schemas define label/image semantics.

---

## 5A. Next Refactor Passes

This section is the planned series of cleanup passes before adding broad CellProfiler compatibility. The goal is to keep increasing type safety, modularity, SSOT ownership, and structural symmetry without introducing low-value wrappers.

### Pass 1: StepSnapshot as Compiler Input SSOT

Primary advisor pressure:

1. `path_planner.py` still probes live step/config attributes.
2. `compiler.py` still contains attribute-probe recovery paths.
3. `funcstep_contract_validator.py` still has fallback-style `getattr` access.

Goal:

Make ObjectState-resolved step facts a typed compiler input instead of repeatedly reading live `FunctionStep` and nested config attributes.

Source-of-truth type:

```python
@dataclass(frozen=True, slots=True)
class StepSnapshot:
    index: int
    name: str
    step_type: str
    func: Any
    processing: StepProcessingSnapshot
    materialization: StepMaterializationSnapshot | None
    input_conversion: StepInputConversionSnapshot | None
```

Tasks:

1. Add `StepSnapshot.from_step(...)` with ObjectState-aware resolved values.
2. Move `group_by`, `input_source`, materialization config, input conversion config, injectable config values, and step name/type into snapshot fields.
3. Route `PathPlanner._get_execution_groups`, `_get_dir`, `_materialized_output_dir_for_step`, `_get_input_source`, and injectable parameter extraction through snapshots.
4. Fail loudly when a required resolved field is missing instead of falling back to live probing.
5. Keep `FunctionStep.func` mutation cleanup visible: snapshot owns compiler input, compiled plan owns runtime output.

Acceptance criteria:

1. Path planning no longer directly probes `step.processing_config` for core compiler facts.
2. ObjectState-derived values have one typed carrier.
3. Existing path planning and ZMQ smoke tests still pass.
4. Advisor findings in `path_planner.py` for config/attribute probing decrease materially.

### Pass 2: CompilationSession as Axis-Scoped Compiler Boundary

Primary advisor pressure:

1. `compiler.py` is still an oversized orchestration hub.
2. Compiler phases thread loose `context`, `pipeline_config`, `orchestrator`, and `step_state_map` values.
3. Provenance of compiler facts is still hard to inspect.

Goal:

Introduce a nominal session object that owns one axis compilation context and exposes typed access to plans, snapshots, config, and orchestrator services.

Source-of-truth type:

```python
@dataclass(slots=True)
class CompilationSession:
    context: ProcessingContext
    pipeline_config: Any
    orchestrator: Any
    step_state_map: Mapping[int, Any]
    snapshots: tuple[StepSnapshot, ...]
    plans: MutableMapping[int, CompiledStepPlan]
```

Tasks:

1. Build `CompilationSession` after step plans are initialized.
2. Route path planning, validation, GPU assignment, and function-pattern compilation through the session.
3. Move repeated setup/teardown and plan lookup helpers out of `PipelineCompiler`.
4. Split compiler orchestration into named stages that accept a session:
   - initialize plans
   - snapshot steps
   - validate contracts
   - plan paths/artifacts
   - validate memory/GPU
   - freeze executable plans
5. Keep stages procedural at first; do not introduce ABCs until the stage interface has real variants.

Acceptance criteria:

1. `PipelineCompiler` reads as orchestration over stages, not a control hub.
2. Compiler phases use session/snapshot/plan objects rather than loose dicts and repeated attribute probes.
3. Advisor oversized-hub findings in `compiler.py` reduce.

### Pass 3: Contract Validation Cleanup

Primary advisor pressure:

1. `funcstep_contract_validator.py` still has semantic-role recovery via `getattr`.
2. Memory and artifact validation overlap with `CallableContract`, `NormalizedFunctionPattern`, and `ArtifactGraph`.

Goal:

Make validation consume the same compiler contracts as compilation.

Tasks:

1. Route all function memory validation through `CallableContract`.
2. Route pattern traversal through `NormalizedFunctionPattern`.
3. Route artifact producer/consumer validation through `ArtifactGraph`.
4. Replace fallback attribute handling with explicit invalid-contract errors.
5. Add tests for invalid callable metadata, invalid artifact declaration kinds, and missing memory type contracts.

Acceptance criteria:

1. Validator does not rediscover callable metadata differently from the compiler.
2. Error messages identify the callable, invocation key, and declared contract field.
3. Advisor `funcstep_contract_validator.py` attribute-probe findings reduce.

### Pass 4: Default ArtifactKind Materialization Registry

Primary runtime pressure:

1. Explicit materializers now use typed store records, but default materialization still does not exist.
2. CellProfiler-style outputs need default behavior for measurements, relationships, metadata, tables, and labels.

Goal:

Make materialization dispatch by `ArtifactKind` when no explicit `MaterializationSpec` is declared, by selecting existing writer-backed presets rather than creating another materialization stack.

Source-of-truth type:

```python
@dataclass(frozen=True, slots=True)
class ArtifactMaterializationRule:
    kind: ArtifactKind
    spec_factory: Callable[[RuntimeValueSchema], MaterializationSpec]
```

Tasks:

1. Add an `ArtifactKind` materialization policy.
2. Preserve explicit `ArtifactOutputPlan.materialization` precedence.
3. Add defaults:
   - `MEASUREMENTS`: existing `csv_only(...)`
   - `RELATIONSHIPS`: existing `csv_only(...)`
   - `TABLE`: existing `csv_only(...)`
   - `METADATA`: existing `json_only(...)`
   - `OBJECT_LABELS`: fail loud until label writer is defined, or add an explicit label-image writer if scope is clear
   - `IMAGE`: fail loud unless image materialization policy is explicit
4. Route `materialize_artifact_outputs` through `RuntimeValue` schema plus registry.
5. Add tests that missing default for a kind fails loudly.

Acceptance criteria:

1. No path-only or materializer-only artifact dispatch remains for planned artifacts.
2. Measurement metadata can materialize without per-function glue.
3. Unsupported kinds fail with artifact name, kind, and invocation/step context.

### Pass 5: Native Runtime Value Schemas

Primary CellProfiler pressure:

1. `RuntimeValueSchema` is still shallow.
2. Object labels, measurements, and relationships need named semantics before any adapter is justified.

Goal:

Add native value contracts for the CellProfiler concepts OpenHCS must own.

Types:

1. `ObjectLabelSet`
2. `MeasurementTable`
3. `ObjectRelationship`
4. `NamedImage`

Tasks:

1. Add typed schema fields for object name, source image name, dimensions, feature names, object id columns, and relationship endpoints.
2. Add `RuntimeValue` normalization for these native values.
3. Add validation that measurements reference declared object/image names.
4. Add validation that relationships reference declared parent/child object names.
5. Add producer/consumer unit tests without any CellProfiler adapter.

Acceptance criteria:

1. `OBJECT_LABELS`, `MEASUREMENTS`, and `RELATIONSHIPS` are native OpenHCS runtime semantics.
2. A measurement function can consume object labels and produce object measurements without hidden workspace state.
3. The runtime can reject inconsistent object/measurement/relationship contracts before adapter work.

### Pass 6: CellProfiler Symbol Table Compiler

Primary compatibility pressure:

1. `.cppipe` names must become compile-time artifact contracts.
2. Adapter-owned mutable state must not become the source of truth.

Goal:

Build a converter-level symbol table that maps CellProfiler names to OpenHCS artifacts.

Tasks:

1. Map image names to channels or image artifacts.
2. Map object names to `OBJECT_LABELS` artifacts.
3. Map measurement features to `MEASUREMENTS` artifacts.
4. Map parent/child references to `RELATIONSHIPS` artifacts.
5. Generate `ArtifactSpec` inputs/outputs from the symbol table.
6. Fail at conversion time for unresolved names or incompatible symbol kinds.

Acceptance criteria:

1. `get_objects("Nuclei")` compiles to an artifact input.
2. `add_objects(..., "Nuclei")` compiles to an `OBJECT_LABELS` artifact output.
3. Measurement writes compile to `MEASUREMENTS` artifact outputs.
4. Generated OpenHCS pipelines do not require a mutable CellProfiler workspace for data routing.

### Pass 7: Thin Compatibility Adapter

Primary rule:

Only after passes 1-6 should a CellProfiler adapter exist.

Goal:

Expose CellProfiler-like APIs as views over OpenHCS-owned runtime state.

Tasks:

1. Implement adapter reads/writes by delegating to typed runtime stores.
2. Prohibit adapter-only hidden object sets, image sets, or measurements.
3. Add parity tests against minimal CellProfiler modules.

Acceptance criteria:

1. Removing the adapter does not remove runtime semantics.
2. All values produced through the adapter are visible as OpenHCS `RuntimeValue`s.
3. Adapter state is inspectable through OpenHCS artifacts and schemas.

### Deferred Low-Value Advisor Items

These should not be prioritized unless they become blockers:

1. `function_runtime.py` shared request-field ABC: currently saves only a few lines and risks abstraction theater.
2. `pipeline/__init__.py` export-map builder: cleanup-only, no CellProfiler/runtime leverage.
3. `gpu_memory_validator.py` string dispatch: small enough to batch with a memory-contract pass.

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
