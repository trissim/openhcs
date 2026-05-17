# Source Assignment Model Cleanup Plan

## Goal

Remove the remaining source-assignment model debt without weakening the typed
source-binding editor or CellProfiler import semantics.

The source-binding UI is already advisor-clean. The remaining smell is in the
core model layer: repeated validation loops, metadata-only source-role leaves,
and compatibility property aliases split source-assignment semantics across
`source_bindings.py` and `pipeline_image_schema.py`.

## Current Evidence

Fresh advisor spot-check from `plan_completion_audit_20260517.md`:

- `openhcs/core/source_bindings.py`: `4` findings.
- `openhcs/core/pipeline_image_schema.py`: `3` findings.
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`: `0` findings.

Relevant files:

- `openhcs/core/source_bindings.py`
- `openhcs/core/pipeline_image_schema.py`
- `openhcs/core/source_bindings_view.py`
- `openhcs/core/source_schema_workspace.py`
- `openhcs/interop/cellprofiler/source_schema.py`
- `tests/unit/test_source_bindings.py`
- `tests/unit/test_pipeline_image_schema.py`
- `tests/unit/test_source_bindings_view.py`
- `tests/unit/test_source_schema_workspace.py`
- `tests/unit/test_cellprofiler_source_schema.py`
- `tests/unit/pyqt_gui/test_source_bindings_editor.py`

## Target Shape

- Source assignment identity is owned by one nominal source-assignment domain.
- Dataclass validation uses a typed validation/coercion record only when it owns
  real source-binding semantics.
- Source roles are either declared through a generated leaf table or explicitly
  justified as behavior-bearing registered leaves.
- Compatibility aliases remain only when public API or serialized state requires
  them; otherwise callers consume the nominal field directly.

## Non-Goals

- Do not rewrite the source-binding editor; it is already advisor-clean.
- Do not convert CellProfiler source-bound images into artifact consumers.
- Do not add generic private helper functions that merely hide repeated loops.
- Do not change serialized field names for `NamedSourceBinding`,
  `ImageAssignment`, `SourceArtifactAssignment`, or `StepSourceBindingsConfig`.

## Implementation Sequence

### Stage 1: Source Role Decision

1. Inspect `SourceRole` leaves and call sites.
2. If leaves are metadata-only and closed over stable fields, replace hand-written
   leaves with a generated declaration table that preserves registry lookup.
3. If leaves carry enough documentation/behavior to remain explicit, encode that
   decision in this plan and do not chase the advisor finding locally.

Verification:

- `tests/unit/test_pipeline_image_schema.py`
- advisor on `openhcs/core/pipeline_image_schema.py`

### Stage 2: Validation Domain Object

1. Identify the repeated `__post_init__` loops in source-binding dataclasses.
2. Introduce a named validation/coercion value object only if it owns a semantic
   domain, for example "non-empty tuple of typed source filters" or "match plan
   dimensions".
3. Replace repeated loops through that object while preserving constructor
   compatibility.

Verification:

- `tests/unit/test_source_bindings.py`
- `tests/unit/test_source_schema_workspace.py`
- advisor on `openhcs/core/source_bindings.py`

### Stage 3: Alias Audit

1. Audit `SourceFilterMatchType.requires_value` call sites.
2. Audit `SourceArtifactAssignment.artifact_kind` call sites.
3. Replace internal callers with nominal fields where safe.
4. Keep aliases only if external/generated compatibility needs them, with a
   short comment and no extra wrapper abstractions.

Verification:

- `rg "requires_value|artifact_kind" openhcs tests`
- focused tests above

### Stage 4: Integrated Source-Binding Gate

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_source_bindings.py \
  tests/unit/test_pipeline_image_schema.py \
  tests/unit/test_source_bindings_view.py \
  tests/unit/test_source_schema_workspace.py \
  tests/unit/test_cellprofiler_source_schema.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  -q --tb=short --disable-warnings
```

Then run:

```bash
.venv/bin/python -m nominal_refactor_advisor openhcs/core/source_bindings.py --json --min-hardcoded-string-sites 3 --min-builder-keywords 3
.venv/bin/python -m nominal_refactor_advisor openhcs/core/pipeline_image_schema.py --json --min-hardcoded-string-sites 3 --min-builder-keywords 3
```

## Completion Criteria

- Source-assignment advisor findings are either resolved or explicitly justified
  in this plan as stable compatibility/noise.
- Focused source-binding and CP source-schema tests pass.
- No generated or serialized source-binding shape changes unless covered by
  migration tests.

