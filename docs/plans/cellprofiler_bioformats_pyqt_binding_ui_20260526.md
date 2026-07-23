# CellProfiler Bio-Formats PyQt Binding UI Plan

## Status

Drafted 2026-05-26 after inspecting the current PyQt Plate Manager, Pipeline
Editor, Dual Editor, Step Parameter Editor, Source Bindings Editor, and
CellProfiler plate-workspace preparer paths.

This plan is a UI integration plan for the broader source-schema bridge in
`docs/plans/cellprofiler_bioformats_workspace_binding_20260525.md`. It does not
replace that core materialization plan. The core bridge should remain generic
and owned by source-schema workspace abstractions.

## Goal

When a user adds an HCS dataset plus one or more `.cppipe` files through the
PyQt UI, OpenHCS should:

1. Detect the `.cppipe` file or files in Plate Manager.
2. Initialize the selected plate through the normal OpenHCS/Bio-Formats
   microscope path.
3. Let the converted CellProfiler pipeline bind source aliases to the existing
   OpenHCS virtual workspace axes.
4. Populate the Step Editor source-binding preview from that same source
   universe.
5. Require manual `StepSourceBindingsConfig` edits only when alias selection is
   ambiguous or incomplete.

The user should not need a separate source layout config for Bio-Formats-backed
HCS data. Bio-Formats should supply HCS axes; CellProfiler source schema should
supply alias semantics; the Step Editor should display and edit the resulting
typed binding state.

## Current UI Architecture

### Plate Manager

Relevant files:

- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/interop/cellprofiler/plate_workspace.py`

`PlateManagerWidget` owns selected folders, logical orchestrator scopes,
`.cppipe` discovery, and plate initialization.

When a folder is added, Plate Manager asks
`CellProfilerPlateWorkspacePreparer.cppipe_paths()` for direct `.cppipe` files.
If there is one `.cppipe`, the folder is represented by one plate row. If there
are multiple `.cppipe` files, the UI creates one logical plate row per pipeline.
CellProfiler pipeline identity is encoded inside the plate scope segment using:

```text
<physical plate root>#openhcs-cppipe=<encoded cppipe file name>
```

The logical scope id is used for ObjectState and per-plate UI state. The
physical plate root remains the selected folder. `::` remains reserved for real
ObjectState nesting such as `<plate scope>::functionstep_0`.

During initialization, Plate Manager calls:

```python
CellProfilerPlateWorkspacePreparer(
    CellProfilerPlateWorkspaceRequest(plate_root, cppipe_path=cppipe_path)
).prepare()
```

and then initializes the registered orchestrator. If the preparer returns an
ingestion result and the plate has no pipeline yet, Plate Manager loads the
converted steps into Pipeline Editor.

### Pipeline Editor

Relevant file:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`

`PipelineEditorWidget` owns per-plate pipeline steps and the per-plate
CellProfiler import result.

For `.cppipe` imports, it stores the `CellProfilerPipelineImportResult` in
`cellprofiler_import_results_by_plate`. The active import result is the source
of `PipelineImageSchema` for step editors.

### Step Editor

Relevant files:

- `openhcs/pyqt_gui/windows/dual_editor_window.py`
- `openhcs/pyqt_gui/widgets/step_parameter_editor.py`
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`
- `openhcs/core/source_bindings_view.py`

`PipelineEditorWidget` passes `import_result.source_schema` to
`DualEditorWindow`. `DualEditorWindow` passes it to
`StepParameterEditorWidget`. The step editor then calls
`apply_source_bindings_preview_context()` and supplies that schema to each
`SourceBindingsEditorWidget`.

`SourceBindingsEditorWidget` is already the inline editor for
`StepSourceBindingsConfig`. It can render:

- pipeline source counts;
- pipeline-level alias assignments;
- step-local bindings;
- metadata rules;
- match plans;
- preview matches;
- assembled image sets.

The missing piece is that the preview inventory is still derived from explicit
`PipelineImageSchema.image_plane_sources` or directory scans. It does not yet
share the OpenHCS/Bio-Formats virtual workspace candidate authority that the
core materializer needs.

## Architectural Rule

Do not add a CellProfiler-specific or Bio-Formats-specific branch inside the
step editor.

The UI should consume a generic source candidate/inventory abstraction. The
same candidate authority should serve:

- `materialize_source_schema_workspace(...)`;
- Step Editor source-binding preview;
- future diagnostics and benchmark reporting.

Plate Manager may coordinate this flow, but it should not parse source aliases
or Bio-Formats metadata itself. Pipeline Editor may store the import result, but
it should not duplicate source-schema state in UI-only structures.

## Proposed UI Contract

Add or expose a small result object that carries the source-binding context
needed by the UI:

```python
@dataclass(frozen=True, slots=True)
class PreparedPlateSourceBindingContext:
    display_plate_root: Path
    execution_plate_path: Path
    cppipe_path: Path | None
    import_result: CellProfilerPipelineImportResult | None
    source_schema: PipelineImageSchema | None
    source_inventory: SourceInventory | None
```

The exact name can change, but the concepts should stay separate:

- `display_plate_root`: the folder the user selected.
- `execution_plate_path`: the workspace path the orchestrator should execute.
- `cppipe_path`: the selected pipeline for this logical plate row.
- `import_result`: converted CellProfiler pipeline result.
- `source_schema`: pipeline-level image/source semantics.
- `source_inventory`: preview/materialization candidates from the same
  authority used by the core source-schema workspace path.

If `source_inventory` is expensive to build, store an immutable descriptor or a
lazy provider instead of concrete rows. Do not make widgets rescan the plate
directory independently.

## Implementation Plan

### Stage 1: Characterize Current UI Behavior

Add or extend PyQt unit tests for the existing behavior before changing
execution-path semantics:

- multiple `.cppipe` files create multiple logical plate rows;
- each logical row stores its own `cppipe_path`;
- each logical row keeps the physical plate root separate from scope id;
- imported CellProfiler steps are stored per logical plate;
- imported source schema reaches `StepParameterEditorWidget`.

Relevant tests:

- `tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py`
- `tests/unit/pyqt_gui/test_source_bindings_editor.py`

### Stage 2: Make Execution Path Explicit

`CellProfilerSourceSchemaWorkspace.execution_plate_path` already distinguishes
source root from materialized workspace root. Plate Manager should honor that
distinction.

Required behavior:

- The plate row continues to display the selected physical folder.
- ObjectState scope id remains the logical plate id.
- The orchestrator initializes against the prepared execution workspace when
  source-schema materialization is needed.
- Pipeline Editor still keys pipeline/import-result state by logical plate id.

Do not overload `plate_path` strings to mean both display identity and execution
workspace. If the current orchestrator API only accepts one path, introduce a
small nominal bridge at Plate Manager/orchestrator registration time rather than
threading string conditionals through the UI.

### Stage 3: Share Source Inventory With Step Editor

Extend the source-schema candidate provider family from the core plan so the UI
can request preview inventory from the same authority.

Likely shape:

```python
class SourceSchemaInventoryProvider(ABC):
    def inventory(self, request: SourceSchemaInventoryRequest) -> SourceInventory:
        ...
```

or reuse `SourceSchemaCandidateProvider` directly and adapt candidates to
`SourceInventory`.

Required behavior:

- `SourceInventory.from_schema_context(...)` should stop being the only
  authority for step preview when a prepared workspace context exists.
- Step Editor should receive either `SourceInventory` or an inventory provider
  from Pipeline Editor / Plate Manager.
- Source Bindings Editor should remain a pure consumer of schema + bindings +
  inventory.

### Stage 4: Store Binding Context Per Logical Plate

Pipeline Editor already stores `cellprofiler_import_results_by_plate`. Add the
parallel binding context there or in a small service owned by Plate Manager.

Requirements:

- switching selected plates restores the correct import result and inventory;
- two `.cppipe` files in the same folder do not overwrite each other's import
  result or preview inventory;
- a non-CellProfiler plate has no import result and falls back to existing
  native OpenHCS behavior;
- refreshing/reinitializing a plate invalidates the old inventory context.

Prefer a small record keyed by logical plate id over parallel dictionaries if
more than import result must be stored.

### Stage 5: Step Editor Preview Wiring

Change the Step Editor creation path so it receives source-binding context, not
only `PipelineImageSchema`.

Current path:

```text
PipelineEditorWidget -> DualEditorWindow -> StepParameterEditorWidget
```

Target path:

```text
PipelineEditorWidget current plate binding context
  -> DualEditorWindow
  -> StepParameterEditorWidget
  -> SourceBindingsEditorWidget.set_preview_context(...)
```

`StepParameterEditorWidget.apply_source_bindings_preview_context()` should pass
the prepared inventory directly when available. Only fall back to
`SourceInventory.from_schema_context(...)` when no prepared inventory exists.

### Stage 6: UI Feedback For Ambiguous Bindings

Do not silently guess aliases from channel names.

The Step Editor should show ambiguity through the existing source-bindings
tables:

- alias has zero matches;
- alias has more matches than expected;
- image-set assembly is incomplete;
- metadata/match-plan dimensions do not distinguish aliases.

The first implementation can be table/status-row based. A later UX pass can add
focused highlighting or a plate-level "Sources" panel, but the source of truth
must remain `PipelineImageSchema` plus `StepSourceBindingsConfig`.

### Stage 7: Integration Tests

Add a PyQt-facing integration test around a small synthetic HCS fixture:

1. Create or load a Bio-Formats/OpenHCS virtual workspace with wells, sites,
   channels, z, and time metadata.
2. Add a folder containing more than one `.cppipe`.
3. Initialize each logical plate row.
4. Assert that each row gets the correct converted pipeline/import result.
5. Open a step editor and assert source-binding preview rows come from the
   OpenHCS/Bio-Formats workspace inventory.

If full Qt window creation is too expensive for CI, test the service boundary
that prepares the binding context and a focused widget test for
`SourceBindingsEditorWidget.set_preview_context(...)`.

## Advisor And Validation

Run advisor on complete affected modules, not only the file just edited:

```bash
nominal-refactor-advisor \
  openhcs/interop/cellprofiler/plate_workspace.py \
  openhcs/interop/cellprofiler/source_schema_ingestion.py \
  openhcs/core/source_schema_workspace.py \
  openhcs/core/source_bindings_view.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/pyqt_gui/windows/dual_editor_window.py \
  openhcs/pyqt_gui/widgets/step_parameter_editor.py \
  openhcs/pyqt_gui/widgets/source_bindings_editor.py \
  tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py
```

Focused tests:

```bash
.venv/bin/pytest \
  tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  tests/unit/test_source_schema_workspace_openhcs_candidates.py \
  tests/unit/test_cellprofiler_source_schema_ingestion.py \
  -q
```

Integration tests after the core provider exists:

```bash
.venv/bin/pytest \
  tests/integration/test_cellprofiler_bioformats_workspace_binding.py \
  tests/integration/test_bioformats_handler_runtime.py \
  -q
```

## Non-Goals

- Do not create a separate CellProfiler wizard.
- Do not make Bio-Formats code understand CellProfiler aliases.
- Do not make CellProfiler UI code parse vendor filenames.
- Do not store a second source-layout model beside `PipelineImageSchema` and
  `StepSourceBindingsConfig`.
- Do not guess aliases from channel labels without typed selector evidence.
- Do not hide ambiguous or unmatched aliases behind permissive fallbacks.

## Completion Criteria

- Plate Manager can keep display folder, logical scope id, `.cppipe` identity,
  and execution workspace distinct.
- Pipeline Editor stores one coherent binding context per logical plate.
- Step Editor preview uses the same candidate/inventory authority as source
  materialization.
- Source Bindings Editor remains the only step-level editing surface for
  `StepSourceBindingsConfig`.
- Bio-Formats/OpenHCS axes appear in preview rows as wells, sites, channels,
  z-planes, and timepoints.
- Ambiguous aliases are visible and editable rather than guessed.
- Advisor and focused tests pass on the affected module set.
