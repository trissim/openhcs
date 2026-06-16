# CellProfiler Binding To Bio-Formats/OpenHCS Virtual Workspaces

## Status

Drafted 2026-05-25 after inspecting the current CellProfiler source-schema
ingestion path, OpenHCS virtual workspace metadata path, and Bio-Formats
handler output. This is an implementation plan, not a claim that converted
CellProfiler pipelines already bind directly to Bio-Formats workspaces without
source-schema materialization.

## Goal

Let converted CellProfiler `.cppipe` pipelines bind their source image aliases
to an existing OpenHCS virtual workspace, including a Bio-Formats-generated
workspace, without requiring users to provide a separate source layout config or
requiring OpenHCS to parse raw vendor filenames a second time.

The intended behavior is:

1. Bio-Formats/OpenHCS initializes a plate and writes normalized virtual files
   such as `A01_s001_w1_z001_t001.tif`.
2. That metadata declares `workspace_mapping`, `source_metadata`, and component
   values for wells, sites, channels, z-planes, and timepoints.
3. CellProfiler `Images` / `Metadata` / `NamesAndTypes` / `Groups` source
   schema compilation still produces a normal `PipelineImageSchema`.
4. `materialize_source_schema_workspace(...)` can discover candidates from the
   existing virtual workspace metadata instead of only scanning raw source files.
5. CellProfiler aliases bind to the mapped OpenHCS axes and channel metadata.

## Current Architecture Facts

### Bio-Formats Handler Output

Relevant files:

- `openhcs/microscopes/bioformats.py`
- `openhcs/microscopes/bioformats_adapter.py`
- `openhcs/microscopes/bioformats_spw_projector.py`
- `openhcs/microscopes/bioformats_well_key.py`
- `benchmark/bioformats_hcs_validation.py`
- `benchmark/datasets/bioformats_hcs.py`

The Bio-Formats handler now projects Bio-Formats-readable HCS sources into a
normal OpenHCS virtual workspace. `BioFormatsWorkspaceMetadataWriter.write(...)`
writes:

- `image_files`
- `channels`
- `wells`
- `sites`
- `z_indexes`
- `timepoints`
- `available_backends`
- `workspace_mapping`
- `source_metadata`

The virtual filenames are generated through `BioFormatsFilenameParser`, which is
a `SourceSchemaFilenameParser` subclass. The result is intentionally compatible
with normal OpenHCS source component parsing.

Live validation evidence currently covers 25 public Bio-Formats HCS catalog
rows and checks observed OpenHCS axes against declared expected wells, sites,
channels, z-planes, and timepoints.

### Runtime Virtual Workspace Metadata Path

Relevant files:

- `openhcs/core/steps/function_runtime.py`
- `openhcs/core/steps/function_execution.py`
- `openhcs/core/source_schema_workspace.py`
- `openhcs/core/source_matching.py`

`FunctionRuntime._virtual_workspace_source_projection_from_metadata(...)` already
reads existing OpenHCS metadata and builds runtime source projections from:

- `workspace_mapping`
- `source_metadata`
- virtual filename components parsed by
  `source_schema_metadata_with_virtual_components(...)`

That means runtime source binding can already reason from mapped OpenHCS axes
once a workspace exists.

### CellProfiler Source-Schema Ingestion Path

Relevant files:

- `openhcs/interop/cellprofiler/source_schema_ingestion.py`
- `openhcs/interop/cellprofiler/runtime_pipeline.py`
- `openhcs/interop/cellprofiler/source_schema.py`
- `openhcs/core/pipeline_image_schema.py`
- `openhcs/core/source_schema_workspace.py`

`prepare_cellprofiler_source_schema_workspace(...)` compiles a `.cppipe` into a
`PreparedGeneratedPipeline`. If `prepared.source_schema.is_empty` is false, it
calls `materialize_source_schema_workspace(...)`.

`materialize_source_schema_workspace(...)` currently discovers source candidates
from `_source_files(source_root, ...)`, then derives metadata from the
CellProfiler source schema rules and filename fallback policy. This is correct
for flat CellProfiler example folders, but it misses an important case: the
source root may already be an OpenHCS/Bio-Formats virtual workspace whose
metadata has better axis semantics than raw filenames.

### PyQt Plate, Pipeline, And Step Editor Flow

Relevant files:

- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/windows/dual_editor_window.py`
- `openhcs/pyqt_gui/widgets/step_parameter_editor.py`
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`
- `openhcs/core/source_bindings_view.py`

The current UI already has the right high-level separation:

- Plate Manager owns selected folders, `.cppipe` discovery, logical
  orchestrator scopes, and plate initialization.
- Pipeline Editor owns per-plate pipeline steps and the per-plate
  CellProfiler import result.
- Dual Editor / Step Parameter Editor owns step-local parameter editing.
- Source Bindings Editor is an inline editor for `StepSourceBindingsConfig`.

When a user adds a folder, `PlateManagerWidget` asks
`CellProfilerPlateWorkspacePreparer.cppipe_paths()` for direct `.cppipe` files.
If there is one `.cppipe`, the folder remains one plate row. If there are
multiple `.cppipe` files, the UI creates one logical plate row per pipeline.
CellProfiler pipeline identity is encoded inside the plate scope segment using:

```text
<physical plate root>#openhcs-cppipe=<encoded cppipe file name>
```

The logical scope id is used for ObjectState and per-plate UI state; the
physical `plate_root` remains the selected folder. `::` remains reserved for
real ObjectState nesting such as `<plate scope>::functionstep_0`. During
initialization, the plate manager runs:

```python
CellProfilerPlateWorkspacePreparer(
    CellProfilerPlateWorkspaceRequest(plate_root, cppipe_path=cppipe_path)
).prepare()
```

and then initializes the orchestrator. If the preparer returns a CellProfiler
ingestion result and the plate has no pipeline yet, the plate manager loads the
converted steps into Pipeline Editor and stores the import result in
`cellprofiler_import_results_by_plate[plate_scope_id]`.

Pipeline Editor passes `import_result.source_schema` into `DualEditorWindow`
whenever a step is created or edited. `DualEditorWindow` passes that schema into
`StepParameterEditorWidget`, and the step editor applies it to each
`SourceBindingsEditorWidget` through `set_preview_context(...)`.

The step editor therefore already has a source-binding preview surface. It can
show pipeline-level aliases, step-local bindings, match plans, preview matches,
and assembled image sets through the GUI-neutral `SourceInventory` /
`SourceBindingsPreview` types. What it does not yet have is a plate-level
inventory authority that says "these candidates came from the initialized
OpenHCS/Bio-Formats virtual workspace." Today the preview inventory is derived
from `PipelineImageSchema.image_plane_sources` or by scanning `source_root`.

This means the UI bridge should not be implemented as special CellProfiler
widgets or Bio-Formats widgets. The generic candidate provider added for
`materialize_source_schema_workspace(...)` should also be the source of preview
inventory for the step editor. Plate Manager should keep exposing the original
selected folder and `.cppipe` in the row, while the preparer/import result
should expose the execution workspace path and source-schema candidate universe
used for binding.

## Problem

The current architecture has two valid authorities that do not meet cleanly:

- Bio-Formats/OpenHCS microscope initialization owns generic HCS axis discovery
  and virtual workspace projection.
- CellProfiler source-schema materialization owns `.cppipe` alias binding and
  image-set assembly.

Today, the CellProfiler materializer starts from source files and source-schema
rules. It does not first ask whether `source_root` is already an OpenHCS
workspace with mapped axes. As a result, a Bio-Formats-backed plate can be
runtime-readable, but converted `.cppipe` alias binding still tends to require
source-schema file discovery or a separate layout rule.

That is the architectural mismatch to fix. The fix should not duplicate
Bio-Formats parsers in CellProfiler code, and it should not make Bio-Formats
know CellProfiler aliases.

## Proposed Load-Bearing Abstraction

Add a nominal candidate source family to `openhcs/core/source_schema_workspace.py`.

The existing `SourceSchemaCandidate` record is already the correct unit:

```python
@dataclass(frozen=True, slots=True)
class SourceSchemaCandidate:
    path: Path
    relative_path: str
    metadata: Mapping[str, str]
```

The missing abstraction is the authority that produces candidates:

```python
class SourceSchemaCandidateProvider(ABC):
    def candidates(self, request: SourceSchemaCandidateDiscoveryRequest) -> tuple[SourceSchemaCandidate, ...]:
        ...
```

Two provider implementations should exist:

- `LocalFileSourceSchemaCandidateProvider`
  - Current behavior: scan physical source files and apply schema metadata rules.
- `OpenHCSWorkspaceSourceSchemaCandidateProvider`
  - New behavior: read `openhcs_metadata.json`, inspect subdirectory
    `workspace_mapping` and `source_metadata`, and emit one
    `SourceSchemaCandidate` per virtual file.

`SourceSchemaCandidateDiscovery` becomes a coordinator over providers, not the
place that hard-codes one source universe.

## OpenHCS Workspace Candidate Semantics

For each virtual path in `workspace_mapping`, the provider should:

1. Resolve the mapped source path only for provenance and backend routing.
2. Use the virtual path as `relative_path`.
3. Merge existing `source_metadata[virtual_path]`.
4. Enrich that metadata through
   `source_schema_metadata_with_virtual_components(virtual_path, metadata)`.
5. Preserve channel labels from OpenHCS metadata when available.

The candidate `path` should be the workspace virtual path, not the raw vendor
file path, because CellProfiler alias filtering and OpenHCS runtime execution
should select the normalized OpenHCS source plane. The raw source path remains
owned by `workspace_mapping` and the storage backend.

This is important for multi-plane Bio-Formats files. A single real file may
contain many C/Z/T planes. Candidate identity must be the virtual plane, not the
real file.

## Channel Name Binding

CellProfiler `NamesAndTypes` often binds aliases using string rules or metadata
values that correspond to channel labels rather than numeric channel indexes.
Bio-Formats/OpenHCS metadata currently writes channel component values as:

```json
"channels": {
  "1": "DAPI",
  "2": "GFP"
}
```

The workspace candidate provider should attach both numeric and label metadata
for channel selection:

- `channel = "1"`
- `wavelength = "1"` as a compatibility alias when appropriate
- `channel_name = "DAPI"` or another normalized channel display value
- `OpenHCSImageType` only when a CellProfiler assignment explicitly supplies
  that image type; do not guess aliases from channel names

Alias selection should still flow through `SourceSelector` and
`PipelineImageSchema`. The provider only exposes metadata; it does not decide
that `DNA` means `DAPI`.

## Materialization Behavior

When source candidates come from an existing OpenHCS workspace, the source-schema
materializer should avoid copying or rematerializing the underlying images.

The output workspace can map alias-specific virtual filenames to the input
workspace virtual paths:

```text
CP materialized workspace virtual file -> existing Bio-Formats/OpenHCS virtual file
existing Bio-Formats/OpenHCS virtual file -> Bio-Formats structured backend ref
```

There are two acceptable implementation shapes:

1. Nested virtual mapping support
   - Allow `workspace_mapping_source_path(...)` and storage backends to resolve a
     mapped virtual source that is itself a mapped virtual file.
   - This preserves a thin materialized CP workspace.

2. Direct mapping flattening
   - When writing the CP materialized workspace, flatten the source reference by
     copying the original workspace reference payload into the new workspace
     mapping.
   - This avoids nested lookup at runtime but requires a nominal reference-copy
     policy so it does not become ad hoc dictionary copying.

The first implementation pass should prefer direct mapping flattening only if
the existing backend resolver cannot safely handle nested virtual references.
Either way, the flattening/nesting decision must be owned by a small named
policy, not scattered conditionals.

## Implementation Plan

### Stage 1: Characterization Tests

Add tests before changing materialization behavior:

- A unit fixture that writes an OpenHCS-style metadata workspace with:
  - two channels named `DAPI` and `GFP`
  - one well
  - one site
  - one z-plane
  - `workspace_mapping`
  - `source_metadata`
- A `PipelineImageSchema` with two aliases that select those channels by
  metadata or component selectors.
- Assert that candidate discovery can emit candidates from the virtual
  workspace and that candidate metadata contains well/site/channel/z/time plus
  channel label.

Recommended test file:

- `tests/unit/test_source_schema_workspace_openhcs_candidates.py`

### Stage 2: Candidate Provider Family

Refactor `SourceSchemaCandidateDiscovery` into a provider-backed coordinator.

Add:

- `SourceSchemaCandidateProvider`
- `LocalFileSourceSchemaCandidateProvider`
- `OpenHCSWorkspaceSourceSchemaCandidateProvider`
- `OpenHCSWorkspaceSourceCandidateReader` or similarly named helper if reading
  metadata starts to make the provider too large

Keep `SourceSchemaCandidateDiscovery(...).candidates()` as the public call
surface so existing callers do not churn.

Do not change CellProfiler-specific code in this stage.

### Stage 3: Workspace Candidate Metadata

Implement OpenHCS workspace candidate metadata rules:

- Parse `openhcs_metadata.json`.
- Iterate `subdirectories`.
- Include only subdirectories with `workspace_mapping`.
- Ignore non-image virtual paths unless their backing source is an allowed image
  source for the schema.
- Merge declared source metadata and virtual filename components.
- Attach channel label metadata from the same subdirectory's `channels` map.
- Fail if `source_metadata` is malformed.
- Do not infer missing axes from raw vendor filenames.

This gives CellProfiler source schemas access to the mapped axes that
Bio-Formats already established.

### Stage 4: Mapping Materialization Policy

Teach `_primary_workspace_mappings(...)` and `_auxiliary_workspace_mappings(...)`
how to preserve virtual workspace candidates without treating the candidate
`path` as a plain disk file.

Likely shape:

```python
@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateSourceRef:
    path: Path
    workspace_ref: Mapping[str, object] | None = None
```

or a small nominal source-ref policy attached beside the candidate rather than
overloading `path`.

Do not add special cases like `if "backend" in mapping`. Create a named source
reference abstraction that says whether a candidate maps to:

- a workspace-relative plain file path
- a structured OpenHCS workspace reference
- a materialized auxiliary payload

### Stage 5: CellProfiler Ingestion Integration

Once `materialize_source_schema_workspace(...)` can consume OpenHCS workspace
candidates, `prepare_cellprofiler_source_schema_workspace(...)` should not need
Bio-Formats-specific logic. It should keep calling the generic materializer.

Add an integration test that:

1. Builds a Bio-Formats/OpenHCS manifest fixture.
2. Initializes it with `BioFormatsHandler`.
3. Compiles a small `.cppipe` whose source schema selects two image aliases by
   channel metadata.
4. Materializes the CP source-schema workspace from the Bio-Formats workspace.
5. Executes a minimal converted pipeline or at least validates the generated
   workspace metadata and source mappings.

Recommended test file:

- `tests/integration/test_cellprofiler_bioformats_workspace_binding.py`

### Stage 6: PyQt Integration

Keep the UI thin and route it through the same generic ingestion result:

- `CellProfilerPlateWorkspaceResult` should expose the execution plate path
  chosen by ingestion. If a materialized source-schema workspace is needed, the
  orchestrator must initialize against that execution path, while the plate row
  and logical ObjectState scope continue to identify the user's selected folder
  and `.cppipe`.
- `PipelineEditorWidget` should continue storing the per-plate
  `CellProfilerPipelineImportResult`; it should not duplicate source-schema
  state in separate UI-only structures.
- `StepParameterEditorWidget.apply_source_bindings_preview_context()` should
  build preview inventory from the same `SourceSchemaCandidateProvider` family
  used by materialization, not from a second directory-scan path.
- `SourceBindingsEditorWidget` should remain the editing surface for
  `StepSourceBindingsConfig`. If automatic binding is complete, it displays the
  resolved preview. If aliases are ambiguous or unmatched, it should expose
  that through the existing preview/match-plan tables rather than inventing a
  parallel source-layout dialog.

Add PyQt unit coverage for:

- multi-`.cppipe` logical plate rows keeping separate import results;
- source-schema preview rows appearing in the step editor from an
  OpenHCS/Bio-Formats virtual workspace inventory;
- ambiguous bindings staying editable as `StepSourceBindingsConfig` rather than
  being hidden behind automatic source-layout config.

Recommended test files:

- `tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py`
- `tests/unit/pyqt_gui/test_source_bindings_editor.py`

### Stage 7: Real Dataset Validation

After the fixture path passes, run a real catalog subset:

- Pick one OME-SPW or Bio-Formats filename-layout dataset with at least two
  channels from `benchmark/datasets/bioformats_hcs.py`.
- Initialize with `BioFormatsHandler`.
- Run the candidate discovery/materialization path with a simple CP source
  schema.
- Confirm aliases bind to the expected observed axes from
  `bioformats_hcs_validation.csv`.

Do not claim broad `.cppipe` parity on all 25 HCS rows until at least one
representative converted CP pipeline has executed through the bridge.

## Non-Goals

- Do not make Bio-Formats parse CellProfiler aliases.
- Do not make CellProfiler ingestion parse Bio-Formats vendor filenames.
- Do not guess aliases from channel names.
- Do not add permissive fallbacks for unknown file layouts.
- Do not replace explicit source schemas for flat folders that have no OpenHCS
  metadata.
- Do not force Bio-Formats datasets through TIFF materialization just to satisfy
  CellProfiler ingestion.

## Advisor Risks To Watch

Run `nominal-refactor-advisor` on the full affected Python module set after each
stage:

```bash
nominal-refactor-advisor \
  openhcs/core/source_schema_workspace.py \
  openhcs/core/pipeline_image_schema.py \
  openhcs/core/source_matching.py \
  openhcs/interop/cellprofiler/source_schema_ingestion.py \
  openhcs/interop/cellprofiler/source_schema.py \
  openhcs/microscopes/bioformats.py \
  openhcs/microscopes/bioformats_adapter.py \
  openhcs/microscopes/bioformats_spw_projector.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/pyqt_gui/widgets/step_parameter_editor.py \
  openhcs/pyqt_gui/widgets/source_bindings_editor.py \
  tests/unit/test_source_schema_workspace_openhcs_candidates.py \
  tests/unit/pyqt_gui/test_plate_manager_cppipe_import.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  tests/integration/test_cellprofiler_bioformats_workspace_binding.py
```

Specific smells to avoid:

- Role-prefixed field bundles for expected/observed or virtual/real source refs.
- Dict-shape tests for source reference type such as `if "backend" in ref`.
- CellProfiler-specific conditionals inside Bio-Formats modules.
- Bio-Formats-specific conditionals inside CellProfiler schema compilation.
- Raw string fallback parsing of wells/sites/channels when OpenHCS metadata is
  absent.
- A second materialization path outside `materialize_source_schema_workspace`.

The planning-time advisor scan over the current modules reported two existing
findings:

- `openhcs/core/pipeline_image_schema.py` dynamically materializes public
  `ImageTypeSourceRole` classes with `type(...)`. This is outside the immediate
  bridge scope, but the bridge should not add another generated public class
  family. New source-candidate providers should be explicit nominal classes.
- `openhcs/microscopes/bioformats_adapter.py` currently has a medium-confidence
  warning that `BioFormatsCompositeAdapter` / `BioFormatsMetadataAdapter` are
  single-consumer matcher infrastructure. The bridge should not add more public
  matcher infrastructure around Bio-Formats. It should consume Bio-Formats
  through the already-written OpenHCS metadata surface.

## Verification Gates

Focused unit tests:

```bash
.venv/bin/pytest \
  tests/unit/test_source_schema_workspace_openhcs_candidates.py \
  tests/unit/test_bioformats_microscope_handler.py \
  tests/unit/test_bioformats_spw_projector.py \
  tests/unit/test_bioformats_java_adapter.py \
  tests/unit/test_bioformats_hcs_validation.py \
  tests/unit/test_cellprofiler_source_schema_ingestion.py \
  -q
```

Focused integration tests:

```bash
.venv/bin/pytest \
  tests/integration/test_bioformats_handler_runtime.py \
  tests/integration/test_bioformats_imagexpress_synthetic.py \
  tests/integration/test_cellprofiler_bioformats_workspace_binding.py \
  -q
```

Real Bio-Formats HCS semantic validation:

```bash
.venv/bin/python scripts/benchmark_cellprofiler_vs_openhcs.py \
  bioformats-hcs-validate \
  --output-dir /tmp/openhcs-bioformats-hcs-validation-cp-bridge \
  --dataset-cache-root /tmp/openhcs-bioformats-hcs-cache-20 \
  --max-size-mb 50 \
  --load-sample-count 1 \
  --continue-on-error
```

Regression tests for existing `.cppipe` paths:

```bash
.venv/bin/pytest \
  tests/integration/test_cellprofiler_generated_pipeline.py \
  tests/integration/test_benchmark_openhcs_adapter_cppipe.py \
  -q
```

## Completion Criteria

- Existing OpenHCS/Bio-Formats virtual workspaces are discoverable as
  `SourceSchemaCandidate` sources.
- Candidate metadata includes canonical well/site/channel/z/time components and
  channel labels from OpenHCS metadata.
- `materialize_source_schema_workspace(...)` remains the only generic source
  materialization authority.
- Converted CellProfiler aliases can bind to mapped Bio-Formats/OpenHCS axes in
  tests without a separate user source-layout config.
- Existing flat-folder CellProfiler example behavior still passes.
- Bio-Formats HCS semantic catalog validation still reports all current rows as
  passed.
- Advisor scan over the affected modules reports no unhandled nominal
  abstraction findings.
