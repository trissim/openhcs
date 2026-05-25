# CellProfiler Runtime Semantic Authority Refactor Plan

Date: 2026-05-24

## Current Checkpoint

Recent commits moved several CellProfiler runtime concepts toward core semantic
authorities:

- `dc3da1e5 Strengthen CellProfiler runtime abstractions`
- `f79d0a41 Unify measurement scope semantics`
- `0e48ce63 Promote measurement row semantics`
- `6096fed0 Strengthen measurement row queries`

The latest full unit gate at this checkpoint was:

- `tests/unit`: `1862 passed, 9 warnings`

The worktree had only one unrelated untracked probe file:

- `openhcs_zmq_probe_wOVp.py`

## Purpose

This plan captures the next architecture-first refactor sequence for the
CellProfiler runtime boundary. The goal is not to add more wrappers or make the
advisor quiet by moving code around. The goal is to make the runtime model
mathematically coherent:

- one core authority for measurement row axes and scalar literal semantics,
- one core authority for measurement table union/schema preservation,
- one nominal source-plane projection model shared by image and object-label
  payloads,
- CellProfiler-specific code limited to CellProfiler dialect, parser, and
  external numbering vocabulary,
- fail-loud behavior instead of broad fallback paths.

## Architectural Diagnosis

The recent branch is directionally correct. Measurement scope, row-axis state,
row ownership, object-label vector alignment, and feature queries have moved
toward core runtime semantics.

The remaining incoherence is concentrated at the adapter/materialization
boundary:

- CellProfiler still owns projection behavior that is fundamentally runtime row
  axis behavior.
- Numeric scalar parsing exists in multiple incompatible forms.
- Current-source plane projection is split between image and object-label
  payload implementations.
- Measurement-table union claims to be lossless while discarding schema.
- A few adapter paths still treat invalid semantic state as absence.
- Runtime profiling has two parallel sinks with the same environment behavior.

## Non-Negotiable Rules

- Do not add broad `except Exception` fallback paths.
- Do not add `hasattr` or structural probes where a nominal interface can own
  the contract.
- Do not add CellProfiler-local copies of core row, table, scalar, or source
  identity semantics.
- Do not introduce trivial forwarding wrappers to satisfy call sites.
- If a case has no supported semantic model, fail loudly with a precise error.
- If absence is valid, encode it as a typed result with an explicit reason.
- Keep compatibility aliases only at public interop boundaries.

## Target Architecture

### Core Runtime Authorities

Core should own the following semantic authorities:

- `MeasurementScalarLiteral`
  - Parses scalar tokens once.
  - Exposes typed facts such as absent, finite numeric, non-finite numeric,
    non-numeric present value, integer label, and padding.
  - Supports policy views such as `finite_required`, `nan_is_padding`, and
    `non_numeric_is_present`.

- `MeasurementRowAxisProjection`
  - Owns runtime slice/image-number row projection.
  - Owns row-axis value presence rules through `MeasurementScalarLiteral`.
  - Exposes one projection API for row sequences and `ColumnarRows`.

- `MeasurementTableUnionSchema`
  - Proves when a union can preserve fields, subject, object ID field,
    source image name, and validated schema state.
  - Emits a typed schema-less union only with an explicit reason.

- `SourcePlaneIdentitySequence`
  - Names the repeated shape currently written as
    `tuple[frozenset[SourceImageSetIdentity], ...]`.
  - Owns matching, ambiguity detection, and absence reasons for source-plane
    identity comparisons.

- `CurrentSourcePlaneProjection`
  - Projects stack-like payloads to the current source plane.
  - Has payload-specific leaves for image payloads and object-label payloads.
  - Shares the same source-plane identity selection model.

### CellProfiler Interop Authorities

CellProfiler interop should own only the vocabulary and external compatibility
rules:

- CellProfiler target-scope parser vocabulary:
  - `CellProfilerMeasurementTargetScope`
  - mapping to core `MeasurementScopeSelection`

- CellProfiler ImageNumber dialect:
  - how pipeline-start source order maps to 1-based ImageNumber,
  - how source binding metadata resolves external image-set identity,
  - how CP row names are rendered.

- CellProfiler module parser/settings details:
  - legacy threshold version parsing,
  - module-specific setting names,
  - raw generated function variant selection.

### Adapter Responsibilities

`CellProfilerRuntimeAdapter` should coordinate the boundary but should not own
generic semantics:

- resolving runtime records,
- resolving source bindings,
- delegating source-plane projection,
- delegating measurement-table queries,
- recording native values,
- invalidating caches.

It should not own scalar parsing, row-axis projection, table-union schema
semantics, or image/object-label projection differences beyond choosing the
correct nominal projector.

## Refactor Sequence

### Stage 1: Remove Fail-Soft Declared Output Detection

Primary file:

- `openhcs/interop/cellprofiler/runtime/adapter.py`

Problem:

`_is_declared_output()` catches `Exception` and returns `False`. This can turn a
bad output declaration into "not declared" and send callers through unrelated
availability checks.

Target:

- Replace `_is_declared_output()` with a nominal resolution:
  - `DeclaredOutputResolution.present(plan)`
  - `DeclaredOutputResolution.absent()`
  - invalid kind/name errors remain errors.

Migration:

1. Add a small typed result or inline fail-loud method.
2. Update `add_relationship()` and any other callers.
3. Delete the broad exception fallback.

Verification:

- Adapter-focused tests for declared outputs.
- `tests/unit/test_cellprofiler_runtime_adapter.py`
- `tests/unit/test_cellprofiler_module_execution.py`
- Advisor scan on `adapter.py`.

Completion criteria:

- No broad exception fallback remains in declared-output detection.
- Invalid output declarations raise directly.

### Stage 2: Collapse Numeric Literal Semantics

Primary files:

- `openhcs/core/runtime_semantics.py`
- `openhcs/core/runtime_artifact_queries.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/interop/cellprofiler/module_settings_binding.py`

Problem:

The code currently has several scalar parsers:

- `MeasurementNumericLiteral`
- `FiniteNumericLiteralAuthority`
- `CellProfilerNumericLiteral`

They differ in whether `nan`, `inf`, booleans, empty strings, and non-numeric
strings count as absent, padding, numeric, or present.

Target:

Introduce a core `MeasurementScalarLiteral` with policy views:

- `token`
- `is_absent`
- `is_numeric`
- `is_finite_numeric`
- `is_nonfinite_numeric`
- `numeric_value`
- `integer_value`
- `is_present_axis_value`
- `is_present_measurement_value`
- `is_padding_measurement_value`

CellProfiler setting parsing can use a strict finite policy, but the parser
should be a view over the same scalar classification rather than another regex.

Migration:

1. Add core scalar classification with focused tests.
2. Rewire `MeasurementObjectLabelResolution`.
3. Rewire `MeasurementAxisValueProjection.value_is_present()`.
4. Rewire `CellProfilerMeasurementRowsProjection.axis_value_is_present()`.
5. Rewire texture/object measurement padding checks.
6. Rewire legacy CP threshold version parsing through the strict finite view.
7. Delete the duplicate parser classes.

Verification:

- `tests/unit/test_runtime_semantics.py`
- `tests/unit/test_runtime_artifact_queries.py`
- `tests/unit/test_cellprofiler_module_execution.py`
- Full `tests/unit`
- Advisor scan on touched files.

Completion criteria:

- One scalar literal authority exists.
- Non-finite and padding behavior is policy-explicit at call sites.
- No duplicate numeric regex parser remains in CP interop.

### Stage 3: Make Measurement Table Union Schema-Preserving

Primary file:

- `openhcs/core/runtime_artifact_queries.py`

Problem:

`MeasurementTableUnion.as_table()` claims a lossless row-owned view, but when it
unions multiple tables it drops fields and schema validation state. That can
change later layout inference and materialization behavior.

Target:

Introduce `MeasurementTableUnionSchema`:

- verifies compatible `fields`,
- verifies compatible `subject`,
- verifies compatible `object_id_field`,
- verifies compatible `object_name`,
- verifies compatible `source_image_name`,
- preserves `validated_runtime_schema` only when every table proves it,
- otherwise returns schema-less rows with an explicit reason.

Migration:

1. Add `MeasurementTableUnionSchema.from_tables(...)`.
2. Teach `MeasurementTableUnion.as_table()` to use it.
3. Add tests for compatible columnar union, mixed schema union, subject
   mismatch, and schema-less fallback reason.
4. Update adapter expectations only if tests expose current accidental schema
   loss.

Verification:

- `tests/unit/test_runtime_artifact_queries.py`
- `tests/unit/test_cellprofiler_runtime_adapter.py`
- `tests/unit/test_cellprofiler_module_execution.py`
- Full `tests/unit`

Completion criteria:

- Compatible unions preserve schema.
- Incompatible unions are explicit, not accidentally schema-less.

### Stage 4: Promote Row-Axis Projection Out of CellProfiler

Primary files:

- `openhcs/core/runtime_artifact_queries.py`
- `openhcs/core/runtime_semantics.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Problem:

`CellProfilerMeasurementAxisStateStrategy` consumes core
`MeasurementRowAxisState` and core row fields, but still owns the projection
machinery. This keeps CP semantics separate from OpenHCS row-axis semantics.

Target:

Core owns the row-axis projection algorithm. CellProfiler supplies:

- external axis name vocabulary,
- ImageNumber start resolver,
- source-path to ImageNumber mapping.

Candidate core shape:

- `MeasurementRowAxisProjectionRequest`
- `MeasurementRowAxisProjectionPolicy`
- `RuntimeSliceToExternalImageNumberProjection`
- `ProjectedMeasurementRows`

Migration:

1. Move pure row/column projection code into core with no CP imports.
2. Define a CP ImageNumber policy that supplies source-order numbering.
3. Make `CellProfilerMeasurementMaterializer` call the core projector.
4. Remove or shrink `CellProfilerMeasurementAxisStateStrategy`.
5. Keep public CP names only as compatibility vocabulary if needed.

Verification:

- Existing CP module execution tests around ImageNumber.
- Runtime artifact query tests for row-axis projection.
- Generated pipeline execution tests if available in the checkout.
- Full `tests/unit`.

Completion criteria:

- Runtime row-axis projection is testable without CellProfiler.
- CP materialization does not own generic slice/image-number row semantics.

### Stage 5: Unify Current Source Plane Projection

Primary file:

- `openhcs/interop/cellprofiler/runtime/adapter.py`

Supporting core candidate:

- `openhcs/core/runtime_semantics.py` or a new focused core module if the
  source-plane identity model is broadly reusable.

Problem:

Image and object-label current-source projection duplicate projectability,
plane selection, and absence behavior. The raw nested type
`tuple[frozenset[SourceImageSetIdentity], ...]` appears in multiple signatures.

Target:

- `SourcePlaneIdentitySequence`
- `CurrentSourcePayloadPlaneSelection`
  - matched plane,
  - unmatched reason,
  - ambiguous match error.
- `CurrentSourcePlaneProjector` nominal family
  - image payload projector,
  - object-label payload projector.

Migration:

1. Name source-plane identity sequence.
2. Add typed absence reasons for no identity, template current source, no match,
   multiple current identities, and non-projectable stack.
3. Replace `CurrentSourceImagePayloadProjection` and
   `CurrentSourceObjectLabelPayloadProjection` with leaves of one projector
   family.
4. Keep object-label-specific fallback to label-axis projection as an explicit
   second strategy, not an implicit `if plane_index is None` path.

Verification:

- `tests/unit/test_cellprofiler_runtime_adapter.py`
- object-label/current-image focused module execution tests.
- Advisor scan on `adapter.py`.

Completion criteria:

- One source-plane selection model.
- Image and object-label projection share selection semantics.
- Absence is observable and named.

### Stage 6: Normalize Resolution Result Shapes

Primary file:

- `openhcs/interop/cellprofiler/runtime/adapter.py`

Problem:

`OptionalResolution` is a generic `None` carrier without absence reasons. Other
nearby result types have ad hoc `matched/absent` classmethods but no shared
variant shape.

Target:

Use a small number of result shapes:

- Direct `T | None` for local, obvious optional values.
- Domain-specific result with absence reason when absence crosses a boundary.
- No generic monad unless it is reused across several resolver families and
  carries typed reasons.

Migration:

1. Replace `OptionalResolution` in `CellProfilerImageNumberResolver` with direct
   local control flow or a typed `ImageNumberResolution`.
2. Give `PipelineStartCurrentStepPayloadResolution` and
   `CurrentSourcePayloadPlaneSelection` absence reasons if they cross decision
   boundaries.
3. Do not introduce a constructor-variant framework for two-method dataclasses
   unless multiple result classes converge on the same variant algebra.

Verification:

- Adapter source-binding tests.
- Advisor scan on `adapter.py`.

Completion criteria:

- Optional results either stay local or carry domain reasons.
- No generic abstraction exists only to rename `None`.

### Stage 7: Unify Runtime Profile Logging

Primary files:

- `openhcs/interop/cellprofiler/runtime/adapter.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Problem:

Adapter and module execution both implement the same environment-gated
`RUNTIME_PROFILE` sink.

Target:

- One shared `RuntimeProfileLogger` sink.
- Local event-field builders remain close to the adapter/module surfaces.

Migration:

1. Add shared sink in core or runtime support module.
2. Rewire `AdapterProfileLog` and `CellProfilerRuntimeProfileLogger`.
3. Keep event names unchanged.
4. Add a small test around env/path behavior if not already covered.

Verification:

- Existing profiling-sensitive tests, if any.
- Focused smoke test with env var and temp profile path.
- Advisor scan on both files.

Completion criteria:

- One environment/path writer.
- No duplicate `_runtime_profile_enabled()` or profile file append logic.

### Stage 8: Collapse Backend Object Measurement Row Identity Duplication

Primary files:

- `openhcs/processing/backends/cellprofiler/intensity.py`
- related object measurement backend modules
- `openhcs/core/runtime_semantics.py`

Problem:

The adjacent advisor scan found repeated object-measurement row identity state
in backend request/result classes. This matches the larger architecture smell:
row identity lives partly in backend measurement construction and partly in
runtime materialization.

Target:

- A core/backend-shared object measurement row identity component.
- Backend request/result records compose that component instead of restating
  `slice_index`, `object_domain`, and `row_identity`.

Migration:

1. Start with intensity only.
2. Add a small shared record only if it is used by both request and rows.
3. Re-run advisor on `processing/backends/cellprofiler/intensity.py`.
4. Extend only to other measurement modules that show the same shape.

Verification:

- Backend measurement unit tests.
- CP module execution tests involving object intensity.
- Full `tests/unit`.

Completion criteria:

- Shared state has one owner.
- Backend records remain domain-specific and readable.

## Suggested Commit Slices

1. Fail-loud declared output resolution.
2. Core scalar literal authority.
3. Rewire scalar literal call sites and delete duplicate parsers.
4. Schema-preserving measurement table union.
5. Core row-axis projection extraction.
6. CP materializer rebind to core row-axis projection.
7. Source-plane identity sequence and selection result.
8. Unified current-source image/object-label projection family.
9. Resolution result cleanup.
10. Shared runtime profile logger.
11. Backend object-measurement row identity consolidation.

Each slice should be independently testable and pushed as a revertible
checkpoint.

## Advisor Strategy

Use the advisor as a detector, not the decision maker.

Recommended scan scopes:

- touched production files only,
- `openhcs/core/runtime_semantics.py`,
- `openhcs/core/runtime_artifact_queries.py`,
- `openhcs/core/runtime_values.py`,
- `openhcs/interop/cellprofiler/runtime/adapter.py`,
- `openhcs/interop/cellprofiler/runtime/module_execution.py`,
- targeted backend files when working on Stage 8.

Avoid repo-wide scans until generated CellProfiler files are confirmed parseable.

Treat these findings as high signal:

- broad exception fallback,
- attribute probing for semantic role recovery,
- repeated structural nested annotations,
- repeated numeric/scalar parsers,
- repeated source-plane selection logic,
- schema projections that drop facts silently.

Treat these findings skeptically:

- public `__all__` lists in generated or compatibility modules,
- tiny two-constructor result classes when absence is local,
- behaviorful registry leaves that exist for import-time family membership.

## Test Strategy

Minimum gates for each behavior-changing slice:

- `git diff --check`
- targeted unit tests for the touched semantic family,
- advisor scan on touched production files.

Required gates before claiming a stage complete:

- `tests/unit/test_runtime_semantics.py`
- `tests/unit/test_runtime_artifact_queries.py`
- `tests/unit/test_cellprofiler_runtime_adapter.py`
- `tests/unit/test_cellprofiler_module_execution.py`
- full `.venv/bin/pytest tests/unit -q`

Additional gates after row-axis or source-plane work:

- generated CellProfiler pipeline tests if available,
- benchmark parity checks for official CellProfiler workloads,
- `runtime_measurement_equivalence(...)` on representative outputs.

## Stop Conditions

A stage is complete only when:

- the targeted abstraction owns real duplicated semantics,
- old duplicate/fallback paths are deleted,
- tests pass at the agreed scope,
- advisor findings are resolved or explicitly documented as false positives,
- no new CP-local duplicate of a core semantic authority is introduced.

Do not stop at a passing targeted test if the refactor moved shared semantics.
Run the broader unit slice and then the full unit suite before committing.

## Expected End State

After this plan, the CellProfiler runtime boundary should read as:

- OpenHCS core owns rows, scalar literals, measurement table shape, source-plane
  identity, and object-label measurement alignment.
- CellProfiler interop owns parser vocabulary, external ImageNumber dialect,
  generated function selection, and module-specific settings.
- The adapter coordinates runtime stores and source bindings without embedding
  generic semantics.
- The module executor/materializer records measurements through core row/table
  authorities instead of parallel CP-local rules.

That is the shape where abstractions pay rent: each abstraction removes a real
duplicate semantic decision, not just a repeated line of code.
