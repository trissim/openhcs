# Source Binding Context Architecture Refactor Plan

## Status

Drafted 2026-05-26 as a critique-driven plan for the parts of the current
source-binding architecture that are still too weak, despite the
`SourceBindingsEditorWidget` itself being a useful generic abstraction.

This plan complements:

- `docs/plans/cellprofiler_gui_source_bindings.md`
- `docs/plans/cellprofiler_bioformats_workspace_binding_20260525.md`
- `docs/plans/cellprofiler_bioformats_pyqt_binding_ui_20260526.md`

## Thesis

The source-binding editor is in the right architectural family: it edits
`StepSourceBindingsConfig`, not CellProfiler-specific GUI state. The remaining
problem is the context around it.

The current UI and ingestion flow still allow source context to be inferred in
several places:

- Plate Manager prepares a CellProfiler source-schema workspace but initializes
  an orchestrator that was created for the physical plate root.
- Pipeline Editor stores a CellProfiler import result, but not a full binding
  context that includes execution workspace and source inventory.
- Step Parameter Editor receives `PipelineImageSchema`, then builds preview
  inventory locally through `SourceInventory.from_schema_context(...)`.
- Source inventory is therefore still partly a view concern, while
  source-schema materialization has its own candidate discovery path.

That is the architectural debt to remove. Source context should be a nominal
runtime/UI contract, not a set of parallel dictionaries, path strings, and local
preview scans.

## Current Strong Parts

### Typed Source Contract

Relevant files:

- `openhcs/core/source_bindings.py`
- `openhcs/core/pipeline_image_schema.py`
- `openhcs/core/source_bindings_view.py`
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`

The core source-binding types are not CellProfiler-only:

- `PipelineImageSchema`
- `StepSourceBindingsConfig`
- `NamedSourceBinding`
- `SourceSelector`
- `MetadataExtractionRule`
- `SourceBindingMatchPlan`

`SourceBindingsEditorWidget` is registered for `StepSourceBindingsConfig`.
That means custom OpenHCS functions can use the same editor when they express
source needs through the OpenHCS source-binding model.

### Generic Inline Widget Route

The editor is reached through the form system's inline dataclass widget route,
not a CellProfiler branch in the step editor. That is the right ownership
boundary.

### Pure Preview Model

`openhcs/core/source_bindings_view.py` provides PyQt-free view/preview types:

- `SourceBindingsViewModel`
- `SourceInventory`
- `SourceBindingsPreview`

This is useful because GUI, CLI diagnostics, tests, and benchmark reporting can
share source-binding presentation semantics.

## Remaining Criticisms

### 1. Execution Path Identity Is Not Load-Bearing Enough

Relevant files:

- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/interop/cellprofiler/plate_workspace.py`
- `openhcs/interop/cellprofiler/source_schema_ingestion.py`

`CellProfilerSourceSchemaWorkspace.execution_plate_path` already models the
difference between the original source root and a materialized workspace root.
But the Plate Manager flow currently creates or retrieves an orchestrator by
logical plate id and physical plate root, then calls:

```python
cellprofiler_workspace = CellProfilerPlateWorkspacePreparer(...).prepare()
orchestrator.initialize()
```

The result of preparation is used to load converted pipeline steps, but the
execution workspace identity is not clearly applied to orchestrator
initialization.

This is risky because the UI has at least three identities:

- the folder the user selected;
- the logical ObjectState/orchestrator scope id;
- the workspace path the orchestrator should execute.

Those must not collapse into one string.

### 2. Preview Inventory Discovery Is Still Duplicated

Relevant files:

- `openhcs/core/source_schema_workspace.py`
- `openhcs/core/source_bindings_view.py`
- `openhcs/pyqt_gui/widgets/step_parameter_editor.py`

`materialize_source_schema_workspace(...)` discovers source-schema candidates
for actual execution/materialization.

`StepParameterEditorWidget.apply_source_bindings_preview_context()` separately
builds preview inventory with:

```python
SourceInventory.from_schema_context(
    self.source_schema,
    bindings=widget.get_value(),
    source_root=self.source_root,
)
```

That means the editor can show a preview from one source universe while
materialization/execution uses another. This is exactly the kind of divergence
that produces "works in preview, fails at runtime" or the reverse.

### 3. Pipeline Editor Stores Import Result, Not Binding Context

Relevant file:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`

`PipelineEditorWidget.cellprofiler_import_results_by_plate` stores the import
result per plate. That is necessary but not sufficient once source binding
depends on the initialized plate workspace.

The per-plate state should include:

- the import result;
- the source schema;
- the source inventory or provider descriptor;
- the display plate root;
- the execution plate path;
- the `.cppipe` identity for multi-pipeline folders.

Keeping only the import result makes later code rediscover or infer the rest.

### 4. Source Inventory Has No Single Provider Authority

Relevant files:

- `openhcs/core/source_schema_workspace.py`
- `openhcs/core/source_bindings_view.py`

`SourceInventory` is useful as a preview data structure, but it is currently not
the same authority as source-schema candidate discovery. The codebase needs one
nominal candidate/inventory provider family that can feed both:

- workspace materialization;
- GUI preview.

Without that, adding Bio-Formats/OpenHCS virtual workspace candidates risks
being implemented twice.

### 5. Ambiguity Reporting Is Presentation-Only

Relevant files:

- `openhcs/core/source_bindings_view.py`
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`

The editor can show matched counts and image-set rows, but unresolved or
ambiguous binding states are not yet a typed diagnostic object. They are mostly
implicit in table values.

For user workflows and tests, ambiguity should become explicit:

- alias has zero matches;
- alias has multiple matches when one was expected;
- aliases cannot be assembled into complete image sets;
- a metadata match plan does not distinguish image sets;
- a selector relies on unavailable metadata.

This should be generated by the pure source-binding layer, not by Qt widgets.

## Target Architecture

### Source Binding Context

Introduce a nominal context object that represents the source-binding state for
one logical plate.

Sketch:

```python
@dataclass(frozen=True, slots=True)
class SourceBindingContext:
    logical_plate_id: str
    display_plate_root: Path
    execution_plate_path: Path
    cppipe_path: Path | None
    source_schema: PipelineImageSchema
    inventory_provider: SourceInventoryProvider
    import_result: CellProfilerPipelineImportResult | None = None
```

This object is not CellProfiler-only. `cppipe_path` and `import_result` are
optional because native OpenHCS pipelines can still have source-binding
contexts.

### Source Inventory Provider

Add a provider interface that bridges materialization candidates and preview
inventory.

Sketch:

```python
class SourceInventoryProvider(Protocol):
    def inventory(
        self,
        *,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig,
    ) -> SourceInventory:
        ...
```

Implementations should include:

- `LocalDirectorySourceInventoryProvider`
- `ExplicitImagePlaneSourceInventoryProvider`
- `OpenHCSWorkspaceSourceInventoryProvider`
- possibly a `CompositeSourceInventoryProvider` that applies precedence without
  hiding ambiguity.

The OpenHCS workspace implementation should reuse the same candidate semantics
as `materialize_source_schema_workspace(...)`, especially for Bio-Formats
virtual workspaces.

### Source Binding Diagnostics

Add a pure diagnostic result:

```python
@dataclass(frozen=True, slots=True)
class SourceBindingDiagnostic:
    severity: SourceBindingDiagnosticSeverity
    code: str
    alias: str | None
    message: str
    candidate_count: int | None = None
```

`SourceBindingsPreview` or a sibling validator should produce these diagnostics
from schema + bindings + inventory. The widget should render diagnostics; it
should not invent them.

## Refactor Plan

### Stage 1: Lock Current Behavior With Tests

Add tests around existing boundaries before changing ownership:

- Plate Manager keeps multi-`.cppipe` logical plate rows distinct.
- Pipeline Editor restores import result by logical plate id.
- Step Editor receives imported `PipelineImageSchema`.
- `SourceBindingsEditorWidget` preview renders when given explicit inventory.

Run:

```bash
.venv/bin/pytest \
  tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  -q
```

### Stage 2: Introduce SourceBindingContext

Add the context record in a non-GUI module. Candidate homes:

- `openhcs/core/source_binding_context.py`
- `openhcs/core/source_bindings_view.py` if the type is strictly
  presentation-oriented

Prefer a new module if the context includes execution path and plate identity.

Update Plate Manager / Pipeline Editor plumbing to pass the context without
changing behavior yet.

Acceptance criteria:

- no new CellProfiler-specific widget branches;
- no parallel dict for context fields;
- multi-`.cppipe` plate rows still isolate state by logical plate id.

### Stage 3: Make Execution Workspace Explicit

Teach Plate Manager to use the prepared execution path when source-schema
preparation materializes a workspace.

Options:

1. Recreate/rebind the orchestrator after preparation using
   `execution_plate_path`.
2. Let orchestrator registration accept both `scope_id` and execution path.
3. Introduce a small `PlateExecutionTarget` object consumed by orchestrator
   initialization.

The third option is cleanest if the current API has too much `plate_path`
overloading.

Acceptance criteria:

- UI row label and ObjectState scope stay stable;
- orchestrator execution path comes from preparation;
- no string parsing in runtime code to recover physical root or `.cppipe`.

### Stage 4: Unify Candidate And Inventory Providers

Create one provider family that can feed both materialization and preview.

Implementation order:

1. Extract current local-directory behavior into a provider.
2. Extract explicit `ImagePlaneSource` behavior into a provider.
3. Add OpenHCS workspace metadata provider.
4. Make `SourceInventory.from_schema_context(...)` a compatibility wrapper
   around the provider family.
5. Make `materialize_source_schema_workspace(...)` use the same provider family
   for candidate discovery.

Acceptance criteria:

- no separate GUI directory scan when a prepared context exists;
- Bio-Formats/OpenHCS virtual workspace candidates are emitted once by a shared
  provider;
- provider precedence is explicit and tested.

### Stage 5: Pass Context Into Step Editor

Change the editor chain from:

```text
PipelineEditorWidget -> source_schema -> DualEditorWindow -> StepParameterEditorWidget
```

to:

```text
PipelineEditorWidget -> SourceBindingContext -> DualEditorWindow
  -> StepParameterEditorWidget -> SourceBindingsEditorWidget
```

`StepParameterEditorWidget` should pass prepared inventory into
`SourceBindingsEditorWidget.set_preview_context(...)`. It should only use the
legacy `SourceInventory.from_schema_context(...)` path when no context/provider
exists.

Acceptance criteria:

- Step Editor preview and materialization use the same source universe;
- custom OpenHCS source-binding functions can receive a context without
  CellProfiler provenance;
- no widget rescans the plate root behind the context's back.

### Stage 6: Add Typed Diagnostics

Extend the pure preview layer to produce diagnostics for unresolved or ambiguous
bindings.

Acceptance criteria:

- tests can assert diagnostics without Qt;
- Source Bindings Editor renders diagnostics but does not compute policy;
- ambiguous aliases are visible rather than guessed;
- fallback-free behavior is testable.

### Stage 7: Advisor And Regression Sweep

Run advisor on full affected module groups after each implementation slice:

```bash
nominal-refactor-advisor \
  openhcs/core/source_binding_context.py \
  openhcs/core/source_bindings_view.py \
  openhcs/core/source_schema_workspace.py \
  openhcs/core/source_matching.py \
  openhcs/interop/cellprofiler/plate_workspace.py \
  openhcs/interop/cellprofiler/source_schema_ingestion.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/pyqt_gui/windows/dual_editor_window.py \
  openhcs/pyqt_gui/widgets/step_parameter_editor.py \
  openhcs/pyqt_gui/widgets/source_bindings_editor.py
```

Focused tests:

```bash
.venv/bin/pytest \
  tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  tests/unit/test_cellprofiler_source_schema_ingestion.py \
  tests/unit/test_source_schema_workspace_openhcs_candidates.py \
  -q
```

Integration tests:

```bash
.venv/bin/pytest \
  tests/integration/test_cellprofiler_bioformats_workspace_binding.py \
  tests/integration/test_bioformats_handler_runtime.py \
  -q
```

## Smells To Avoid

- Adding `if cellprofiler` branches in `StepParameterEditorWidget`.
- Adding `if bioformats` branches in `SourceBindingsEditorWidget`.
- Keeping import result, execution path, source root, and inventory in parallel
  dictionaries.
- Reconstructing execution identity from a scoped plate string.
- Letting widgets scan directories when a prepared source-binding context
  exists.
- Guessing aliases from channel names without typed selector evidence.
- Treating `None` or empty rows as a silent unavailable state.

## Completion Criteria

- There is one nominal per-logical-plate source-binding context.
- Display plate root, logical scope id, `.cppipe`, and execution workspace are
  separate fields.
- Preview inventory and materialization candidates share one provider family.
- Step Editor receives context and does not rediscover sources independently.
- Ambiguity/unavailable states are represented as typed diagnostics.
- Existing CellProfiler imports and custom OpenHCS source-binding functions use
  the same editor path.
- Advisor and focused regression tests pass.
