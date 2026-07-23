# CellProfiler Runtime Boundary Decomposition Plan

Date: 2026-05-16

## Problem

Most simple runtime-family cleanup is already complete, but two forwarding-wrapper findings remain in `CellProfilerModuleExecutor` adjacency:

- `IntensityImageSpecialInputValueStrategy.runtime_input_value`
- `SliceAlignedObjectMeasurementLabelArgumentPolicy.label_argument`

These should not be solved by moving wrappers around. A wrapper disappears only when the underlying responsibility moves to the class that owns the semantic operation.

## Target

CellProfiler runtime execution should have crisp ownership boundaries:

- Invocation request construction belongs to invocation/request builders.
- Special input value resolution belongs to the special-input value family.
- Object-label argument construction belongs to object/label measurement policy families.
- Measurement materialization owns image-number projection and output row shape.
- The executor coordinates collaborators but does not host one-off semantic adapters.

## Non-Goals

- Do not reintroduce generated pipeline runtime boilerplate.
- Do not hide policy differences behind permissive `getattr`/`hasattr` probes.
- Do not introduce CellProfiler-only fallback behavior in core runtime abstractions.
- Do not merge behaviorful policy leaves into a base just to silence a forwarding finding.

## Staged Work

### Stage 1: Ownership Trace

- Trace all call sites for the remaining wrapper methods.
- Identify whether callers should depend on the policy family root or a more specific value object.
- Document whether the wrapper is a true duplicate or a public compatibility bridge.

### Stage 2: Move Caller Dependencies

- Update callers to use the owning semantic method directly.
- If a method name is wrong for the family, rename the family method once and migrate all leaves.
- Keep old names only if tests prove external API usage; otherwise remove them.

### Stage 3: Measurement Materializer Authority

- Audit direct construction of image-number projection/materialization records.
- Ensure measurement materializer remains the single authority for output rows and projection behavior.
- Do not combine this with registry-key refactoring.

### Stage 4: Generated Pipeline Parity

- Run generated-pipeline execution tests after each behavior move.
- If parity fails, fix at the model boundary rather than reintroducing compatibility shims.

## Verification

- `tests/unit/test_cellprofiler_module_execution.py`
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
- `tests/unit/test_cellprofiler_strategy_registries.py`
- Full `tests/unit`
- Targeted advisor scan on `openhcs/interop/cellprofiler/runtime/module_execution.py`

## Completion Criteria

- Remaining forwarding-wrapper findings are gone or justified as intentionally public compatibility APIs.
- Executor code has fewer semantic bridge methods, not more.
- Generated CellProfiler pipeline execution still passes.
- No parity regression in focused tests.

