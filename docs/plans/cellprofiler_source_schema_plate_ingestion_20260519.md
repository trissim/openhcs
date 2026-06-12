# CellProfiler Source-Schema Plate Ingestion

## Problem

The PyQt Plate Manager currently has to prepare CellProfiler folders before it
can initialize an orchestrator. That is the architectural problem. The GUI
should pass the selected source and optional `.cppipe` choice to an
orchestrator-owned input preparation boundary, then call
`orchestrator.initialize()`.

CellProfiler example folders should enter OpenHCS as native source workspaces.
A `.cppipe` file is a source metadata dialect: its `Images`, `Metadata`,
`NamesAndTypes`, and `Groups` modules define source selection, metadata
extraction, image aliases, and image-set grouping. Those semantics should be
normalized into native OpenHCS metadata before ordinary orchestrator
initialization continues.

The current implementation crosses the wrong boundary. Plate initialization
uses `CellProfilerPlateWorkspacePreparer`, which calls full generated-pipeline
preparation before the workspace exists. Full preparation compiles processing
artifact contracts, so an intentionally incomplete tutorial pipeline such as
`BBBC022_Analysis_Start.cppipe` fails during plate init when `MaskImage`
references an object artifact (`Nuclei`) that no prior processing module
produces. That diagnostic is correct, but it belongs to pipeline import/compile,
not source workspace initialization.

## Verified Existing Seams

- `benchmark/adapters/openhcs.py`
  - `OpenHCSAdapter._run_converted_cppipe_pipeline(...)` is the current
    working CLI/benchmark authority for converted `.cppipe` execution.
  - It resolves `.cppipe` sources, calls `prepare_generated_pipeline(...)`,
    materializes `prepared.source_schema` through
    `materialize_source_schema_workspace(...)`, switches execution to the
    materialized `source_workspace.workspace_root`, initializes a
    `PipelineOrchestrator`, and executes `prepared.pipeline`.
  - This is proof that the GUI fix should extract and reuse the same
    source-ingestion subflow, not design a second materialization path.

- `benchmark/adapters/cppipe_source.py`
  - `CPPipeSourceRequest`, `CPPipeSourceResolution`, and
    `resolve_cppipe_source(...)` already provide a typed `.cppipe` resolver for
    local files and reference URLs.
  - The resolver is benchmark-oriented because it expects `dataset_id`,
    `output_dir`, and benchmark pipeline params, but its local-file/reference
    semantics are reusable once moved or wrapped behind a product contract.

- `scripts/run_cellprofiler_cppipe_parity.py`
  - The CLI accepts `--dataset-path` and `--cppipe-path`.
  - `--openhcs-runtime-only` calls `OpenHCSAdapter().run(...)` with
    `cppipe_path` in pipeline params, demonstrating the minimal source inputs
    required to run OpenHCS without native CellProfiler parity.

- `benchmark/runner.py`
  - `run_cellprofiler_cppipe_parity(...)` resolves visible source paths,
    constructs base pipeline params with `cppipe_path`, then delegates converted
    execution to `OpenHCSAdapter`.
  - This gives the product-level API shape the GUI likely needs: source dataset
    path, `.cppipe` path, optional pipeline name/id, and output/workspace root.

- `benchmark/cellprofiler_comparison.py`
  - `load_comparison_cases(...)` resolves manifest `dataset_path` and
    `cppipe_path` through `ComparisonManifest`.
  - Benchmark manifests already model the exact pair the GUI needs to acquire:
    source image root plus `.cppipe`.

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
  - `PipelineEditorWidget.load_pipeline_from_file(...)` already dispatches
    `.cppipe` files to `_load_cppipe_pipeline_from_file(...)`.
  - `_load_cppipe_pipeline_from_file(...)` compiles the `.cppipe` with
    `CellProfilerPipelineImportRequest` and stores
    `self.cellprofiler_import_result`, including the imported
    `PipelineImageSchema`.

- `openhcs/interop/cellprofiler/source_schema.py`
  - `compile_image_schema(...)` already lowers CellProfiler setup modules into
    `PipelineImageSchema`.
  - `Images`, `Metadata`, `NamesAndTypes`, and `Groups` are already represented
    as generic OpenHCS source-schema/source-binding concepts.
  - Verified against `BBBC022_Analysis_Start.cppipe`: source schema compilation
    succeeds and produces native image assignments/source artifacts even though
    full generated-pipeline preparation fails later on the missing `Nuclei`
    object dependency.

- `openhcs/core/source_schema_workspace.py`
  - `materialize_source_schema_workspace(...)` already converts a
    `PipelineImageSchema` plus source root into an OpenHCS metadata workspace.
  - The materialized metadata uses `workspace_mapping`,
    `SourceSchemaFilenameParser`, channel/well/site/z/time component values, and
    virtual-workspace backend metadata.

- `openhcs/microscopes/openhcs.py`
  - `OpenHCSMetadataHandler.find_metadata_file(...)` auto-detects
    `openhcs_metadata.json`.
  - `OpenHCSMicroscopeHandler.initialize_workspace(...)` reads that metadata,
    determines the main subdirectory, and registers the virtual-workspace backend
    when `workspace_mapping` is present.

- `openhcs/core/orchestrator/orchestrator.py`
  - `PipelineOrchestrator.initialize()` already owns microscope handler
    creation, workspace initialization, component-key caching, and OpenHCS
    metadata completion.
  - The missing seam is before microscope handler creation: an input dialect
    resolver that can normalize selected source folders into native OpenHCS
    workspaces. That seam should live under the orchestrator/session boundary,
    not in PyQt.

- `openhcs/pyqt_gui/widgets/plate_manager.py`
  - The current direct calls to `CellProfilerPlateWorkspacePreparer` are a
    boundary violation. Plate Manager is choosing a CellProfiler preparation
    strategy and may rebind the orchestrator to a different execution path.
    That should move behind an orchestrator-owned source initialization
    contract.

- `tests/integration/test_cellprofiler_generated_pipeline.py`
  - Existing tests already execute converted CellProfiler pipelines by manually
    calling `materialize_source_schema_workspace(...)`, then creating a
    `PipelineOrchestrator` on the resulting `workspace_root`.
  - This proves the runtime path exists. The missing piece is product
    orchestration and GUI ingestion.

## Target Architecture

Treat `.cppipe` setup modules as a native OpenHCS input metadata dialect. The
bridge translates CellProfiler source semantics into an OpenHCS source-schema
workspace, then hands normal OpenHCS metadata to the existing
microscope/orchestrator path.

After ingestion, there should be no "CellProfiler folder" special case from the
orchestrator's perspective. There is a native OpenHCS workspace with canonical
axis metadata, source bindings, workspace mapping/projection, and provenance
that records CellProfiler as the source-schema provider.

The benchmark CLI already implements this bridge inside
`OpenHCSAdapter._run_converted_cppipe_pipeline(...)`, but that is the wrong
long-term ownership boundary. The architectural goal is to move the reusable
source-ingestion authority into OpenHCS product code and leave the benchmark
adapter as a thin compatibility/measurement layer. The benchmark adapter should
keep ownership of metrics, runtime cache, native-reference parity, phase timing,
and summary artifacts. It should not own generic CellProfiler source-schema
plate ingestion.

The intended ownership split is:

- `openhcs.interop.cellprofiler`
  - Owns `.cppipe` parsing/import, CellProfiler setup-module source schema
    compilation, processing pipeline import, and CP-specific source resolution
    policy.
  - Must expose source-schema-only preparation separately from full
    generated-pipeline preparation.

- `openhcs.core`
  - Owns `PipelineImageSchema`, `StepSourceBindingsConfig`,
    `materialize_source_schema_workspace(...)`, image-set selection, and generic
    source-schema workspace materialization semantics.
  - Owns the generic input dialect preparation interface used by the
    orchestrator, if that interface is broader than CellProfiler.

- `openhcs.microscopes`
  - Owns OpenHCS metadata detection and virtual-workspace initialization after a
    source-schema workspace has been materialized.
  - Should not learn CellProfiler pipeline semantics; it should only see normal
    `openhcs_metadata.json` and `SourceSchemaFilenameParser`.

- `benchmark`
  - Owns benchmark manifests, dataset acquisition, native CellProfiler
    references, parity comparison, timing/memory metrics, benchmark caches, and
    report artifacts.
  - Calls OpenHCS product APIs instead of embedding OpenHCS ingestion logic.

The desired user flow is:

1. User selects a `.cppipe`, a folder containing a `.cppipe`, or a flat image
   folder with an associated `.cppipe`.
2. The GUI records that selection as orchestrator initialization intent. It does
   not parse or prepare CellProfiler semantics.
3. `PipelineOrchestrator.initialize()` asks an input dialect preparer to
   normalize the selected source.
4. The CellProfiler preparer compiles setup modules only and materializes an
   `openhcs_metadata.json` source-schema workspace.
5. The orchestrator initializes on the returned native workspace through the
   normal `OpenHCSMicroscopeHandler` path.
6. Pipeline import/compile can then load converted `FunctionStep` declarations
   and validate processing artifact dependencies strictly.
7. From this point onward, compilation, debug, execution, and viewer streaming
   use the existing OpenHCS path.

## Proposed New Product Contract

Add a product-level input dialect preparation service, not a PyQt-only helper
and not a benchmark-only adapter method. The service should have one
orchestrator-facing contract and one CellProfiler-specific implementation.

```python
@dataclass(frozen=True, slots=True)
class InputWorkspacePreparationRequest:
    selected_path: Path
    selected_pipeline_path: Path | None = None
    workspace_root: Path | None = None
    filemanager: FileManagerLike | None = None
    source_backend: Backend = Backend.DISK
    workspace_backend: Backend = Backend.DISK
    max_image_set_count: int | None = None


@dataclass(frozen=True, slots=True)
class InputWorkspacePreparationResult:
    original_source_root: Path
    execution_plate_path: Path
    source_schema: PipelineImageSchema
    materialization: SourceSchemaWorkspaceMaterialization | None
    provenance: InputWorkspacePreparationProvenance
    pipeline_import: CellProfilerPipelineImportResult | None = None
    pipeline_import_error: CellProfilerPipelineImportDiagnostic | None = None
```

CellProfiler then implements this generic input workspace contract:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaPreparationResult:
    source_root: Path
    workspace_root: Path
    cppipe_path: Path
    source_schema: PipelineImageSchema
    materialization: SourceSchemaWorkspaceMaterialization
```

The service should live outside PyQt and outside `benchmark`. Split it across
the real OpenHCS layers:

- `openhcs/interop/cellprofiler/source_schema_ingestion.py`
  - CP-specific request resolution and source-schema-only preparation.
  - Full `.cppipe` processing import remains available here or in
    `runtime_pipeline.py`, but it is not a prerequisite for source workspace
    initialization.

- `openhcs/core/source_schema_workspace.py`
  - Existing generic workspace materialization remains here.

- `openhcs/core/orchestrator` or a small adjacent module
  - Orchestrator-owned input workspace preparation boundary. It selects the
    appropriate preparer before microscope handler initialization and updates
    the orchestrator's execution path from the result.

- `openhcs/microscopes/openhcs.py`
  - Existing OpenHCS metadata/virtual-workspace initialization remains here.

The interop ingestion service responsibilities are:

- Resolve `selected_path` into `cppipe_path` and `source_root`, productizing the
  local-source subset of `benchmark.adapters.cppipe_source`.
- Parse the `.cppipe` once and compile the setup/source schema through
  `compile_image_schema(...)`.
- Choose a deterministic default `workspace_root`, for example:
  - sibling: `<selected_root>_openhcs_source_schema`
  - cache: configured OpenHCS GUI workspace/cache root
  - explicit request override for tests and CLI.
- Call `materialize_source_schema_workspace(...)`.
- Return the materialized native workspace and source-schema provenance.

Full processing import responsibilities are separate:

- call `prepare_generated_pipeline(...)`;
- compile the `CellProfilerSymbolTable`;
- generate/import `FunctionStep` declarations;
- fail if processing artifacts are missing or type-conflicting.

An incomplete tutorial `.cppipe` should therefore allow source workspace
initialization but report a pipeline import diagnostic such as
`MaskImage(7) references unknown objects symbol 'Nuclei'`.

The interop service should also expose a lower-level request shape matching the
benchmark path:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaWorkspaceRequest:
    source_root: Path
    cppipe_path: Path
    workspace_root: Path
    generated_pipeline_path: Path
    filemanager: FileManagerLike | None = None
    cppipe_filemanager: FileManagerLike | None = None
    generated_pipeline_filemanager: FileManagerLike | None = None
    source_backend: Backend = Backend.DISK
    workspace_backend: Backend = Backend.DISK
    cppipe_backend: Backend = Backend.DISK
    generated_pipeline_backend: Backend = Backend.DISK
    image_set_selection: SourceSchemaImageSetSelection | None = None
```

This lets the benchmark adapter call the OpenHCS product service with its already-known
`request.dataset_path`, resolved `cppipe_path`, and
`request.axis_selection.source_schema_selection()` without going through GUI
path guessing.  The three FileManager fields are intentionally separate:
local `.cppipe` text, generated Python, and source workspace materialization are
different IO roles and should not be inferred from one benchmark FileManager.

## Refactor Out Of Benchmark

Current benchmark-owned logic to move:

- `benchmark.adapters.cppipe_source.CPPipeSourceRequest`
  - Move or mirror product-neutral local/reference `.cppipe` resolution into
    `openhcs.interop.cellprofiler`.
  - Benchmark can keep a thin adapter that builds the product request from
    benchmark pipeline params.

- `benchmark.adapters.openhcs.OpenHCSAdapter._run_converted_cppipe_pipeline`
  - Extract the block from `prepare_generated_pipeline(...)` through
    `materialize_source_schema_workspace(...)` into the product ingestion
    service.
  - Keep runtime cache, metrics, phase timing, and equivalence comparison in
    benchmark.

- `scripts/run_cellprofiler_cppipe_parity.py`
  - Runtime-only mode should continue to call `OpenHCSAdapter`, but after the
    refactor the adapter delegates ingestion to OpenHCS product code.

Do not move benchmark-only concerns into OpenHCS core:

- native CellProfiler reference execution
- semantic parity comparison against benchmark references
- benchmark runtime cache manifests
- speedup/memory/report CSV fields
- official30 manifest acquisition policy.

## Source Resolution Rules

The resolver should be explicit and fail-loud:

- If `selected_path` is a `.cppipe` file:
  - `cppipe_path = selected_path`.
  - Prefer sibling `images/` as `source_root` if it exists.
  - Otherwise use the `.cppipe` parent as `source_root`.

- If `selected_path` is a directory:
  - Prefer exactly one `.cppipe` in the directory.
  - Otherwise prefer exactly one `.cppipe` in the parent if selected directory is
    named `images`.
  - Otherwise prefer exactly one `.cppipe` in a direct child only when the
    selected directory contains a single obvious CellProfiler example package.
  - If zero or multiple candidates are found, return a typed ambiguous/missing
    result that the GUI can render as a selection dialog.

- For `source_root`:
  - If the `.cppipe` parent has an `images/` directory with image files, use it.
  - Else use the `.cppipe` parent.
  - Allow GUI/CLI override because real CellProfiler projects can keep images
    elsewhere.

## Microscope/Metadata Integration

Do not add CP branches to `create_microscope_handler(...)`.

The ingestion service should materialize an ordinary OpenHCS metadata workspace.
Then auto-detection naturally sees `openhcs_metadata.json` and selects
`OpenHCSMicroscopeHandler`. This keeps the registered microscope format as
`openhcsdata`, while the source provenance says the workspace came from a
CellProfiler source schema.

Add provenance metadata to the generated `openhcs_metadata.json` if the current
metadata schema allows it. If the schema does not yet allow it, add a typed
metadata extension rather than stuffing GUI-only state into the plate manager.
Useful fields:

- `source_schema_provider`: `cellprofiler`
- `source_schema_file`: relative path to the `.cppipe`
- `source_root`: relative or absolute path used during materialization
- `generated_pipeline_path`: generated OpenHCS pipeline path

This provenance is not needed by the microscope handler to parse filenames, but
it is useful for GUI reload, debugging, and reproducibility.

## PyQt Integration

PyQt should become thinner than the current implementation.

Required direction:

- Plate Manager records selected source path and optional `.cppipe` choice as
  orchestrator/source initialization intent.
- Plate Manager calls `orchestrator.initialize()` and observes the resulting
  state.
- It does not call `prepare_generated_pipeline(...)`.
- It does not call `prepare_cellprofiler_source_schema_workspace(...)`.
- It does not decide whether a CellProfiler source root should be materialized
  in place, as a sibling, or in a cache.
- It does not rebind orchestrators by interpreting CellProfiler execution paths.

Pipeline Editor can still request pipeline import for the selected `.cppipe`,
but that is compiler/import behavior. If import fails, the UI should attach the
diagnostic to the pipeline/editor state while keeping the initialized native
workspace visible.

The visible UX can still offer a targeted action such as:

- `Import CellProfiler Pipeline + Images...`

but the action should only build source selection intent and delegate to the
orchestrator-owned input workspace preparation boundary. It should not be the
owner of CellProfiler semantics.

## Benchmark Integration Refactor

After adding the product ingestion service, simplify
`OpenHCSAdapter._run_converted_cppipe_pipeline(...)`:

- Keep benchmark-owned phases:
  - `RESOLVE_SOURCE`
  - `READ_CACHE`
  - benchmark runtime cache read/write
  - metrics
  - watchdog
  - native-reference equivalence
  - benchmark provenance.
- Move product-owned phases into the ingestion service:
  - `.cppipe` local/reference resolution where not benchmark-specific
  - `prepare_generated_pipeline(...)`
  - source-schema workspace materialization
  - selected image-set filtering.
- Have the adapter call:

```python
ingestion = prepare_cellprofiler_source_schema_workspace(
    CellProfilerSourceSchemaWorkspaceRequest(
        source_root=request.dataset_path,
        cppipe_path=cppipe_path,
        workspace_root=source_workspace_path,
        generated_pipeline_path=generated_module_path,
        filemanager=self._filemanager,
        image_set_selection=request.axis_selection.source_schema_selection(),
    )
)
execution_plate_path = ingestion.execution_plate_path
prepared = ingestion.prepared_pipeline
```

The exact return type should avoid leaking benchmark terminology. It may expose
`prepared_pipeline` instead of only `import_result` because the benchmark adapter
needs `prepared.pipeline`, `prepared.source_schema`, and validation expectations.

Completion goal: benchmark CLI behavior stays unchanged, while orchestrator
initialization and benchmark runtime execution use the same source-schema
ingestion authority through different request front doors.

The final benchmark adapter should read like:

1. Resolve benchmark case inputs and cache keys.
2. Ask OpenHCS interop to prepare/materialize the converted source workspace.
3. Create `PipelineOrchestrator` on the returned workspace root.
4. Execute and measure.
5. Validate/compare/report.

It should not know how CellProfiler `NamesAndTypes` becomes OpenHCS
`workspace_mapping`.

## Tests

### Unit Tests

Add tests for source resolution:

- `.cppipe` selected with sibling `images/`.
- `images/` directory selected with parent `.cppipe`.
- package root selected with one `.cppipe` and `images/`.
- ambiguous multiple `.cppipe` candidates fail with actionable diagnostics.
- no `.cppipe` candidate fails with actionable diagnostics.

Add tests for ingestion:

- Use a small synthetic `.cppipe` with `NamesAndTypes` file rules.
- Materialize the workspace.
- Assert `openhcs_metadata.json` exists.
- Assert `workspace_mapping` points to the flat source files.
- Assert `source_filename_parser_name == "SourceSchemaFilenameParser"`.
- Assert OpenHCS auto-detection can initialize the materialized workspace.

### Integration Tests

Extend existing CellProfiler generated-pipeline integration coverage:

- Use official `ExampleFly` when available.
- Run ingestion service instead of manually calling
  `materialize_source_schema_workspace(...)`.
- Initialize through the orchestrator input preparation boundary and assert the
  resulting execution path is the materialized native workspace.
- Execute at least one selected well/image set.

### PyQt Tests

Add a GUI workflow test that does not require manual interaction:

- Create/acquire ExampleFly-like folder.
- Invoke the import action with a selected path override.
- Assert Plate Manager passes source selection intent to the orchestrator rather
  than calling CellProfiler preparation directly.
- Assert Pipeline Editor receives either converted steps or a compiler/import
  diagnostic, and always receives source schema for initialized workspaces.
- Assert initializing the added plate reaches `READY`.

## Implementation Phases

### Phase 1: Product Source-Schema Preparation

- Add typed CellProfiler source resolver/result classes.
- Parse `.cppipe` and compile setup modules through `compile_image_schema(...)`.
- Materialize source-schema workspace without full generated-pipeline
  preparation.
- Preserve strict full generated-pipeline preparation for compile/import.
- Cover with unit tests.
- Add a direct lower-level API for the benchmark path where `source_root` and
  `cppipe_path` are already known.

Completion gate:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_source_schema_ingestion.py -q
.venv/bin/python -m nominal_refactor_advisor openhcs/interop/cellprofiler/source_schema_ingestion.py
```

### Phase 2: Runtime Integration

- Add an orchestrator-owned input workspace preparation boundary.
- Teach `PipelineOrchestrator.initialize()` to call it before microscope handler
  creation.
- Ensure the boundary returns a native OpenHCS workspace path and provenance,
  not GUI state.
- Add ingestion-based integration test for ExampleFly or a small synthetic CP
  dataset.
- Ensure orchestrator initialization uses `OpenHCSMicroscopeHandler` through
  existing auto-detection after input preparation returns the materialized
  workspace.
- Confirm no CP-specific condition is added to microscope auto-detection.
- Refactor `benchmark/adapters/openhcs.py` to call the product ingestion service
  for converted `.cppipe` source workspace preparation.
- Run the runtime-only CLI path against ExampleFly:

```bash
.venv/bin/python scripts/run_cellprofiler_cppipe_parity.py \
  --dataset-path /tmp/cellprofiler_examples_gui_smoke/ExampleFly \
  --cppipe-path /tmp/cellprofiler_examples_gui_smoke/ExampleFly/ExampleFly.cppipe \
  --openhcs-runtime-only \
  --output-root /tmp/openhcs_examplefly_runtime_smoke
```

Completion gate:

```bash
.venv/bin/python -m pytest tests/integration/test_cellprofiler_generated_pipeline.py -q -k source_schema
.venv/bin/python -m pytest tests/integration/test_benchmark_openhcs_adapter_cppipe.py -q
```

### Phase 3: PyQt Simplification

- Remove direct CellProfiler preparation from Plate Manager.
- Pass selected `.cppipe` intent into orchestrator initialization.
- Associate pipeline import diagnostics with Pipeline Editor state without
  making plate init fail for processing-only contract errors.
- Add a noninteractive PyQt workflow test.

Completion gate:

```bash
.venv/bin/python -m pytest tests/pyqt_gui -q -k cellprofiler
timeout 60 openhcs
```

### Phase 4: UX Polish

- If a user selects a CP example root through normal `Add Plate`, detect that it
  contains `.cppipe` metadata and show a targeted message:
  "This looks like a CellProfiler source-schema dataset. Use Import
  CellProfiler Pipeline + Images."
- Optionally offer to run the import workflow directly.
- Do not silently mutate the raw example folder unless the user explicitly
  chooses an in-place workspace mode.

## Non-Goals

- Do not make CellProfiler execution a separate runner in the GUI.
- Do not teach every microscope handler about `.cppipe`.
- Do not put output artifact semantics into `StepSourceBindingsConfig`.
- Do not treat arbitrary flat image folders as valid plates unless there is a
  source schema, explicit user mapping, or a separate flat-folder ingestion
  provider.

## Open Questions

- Should the default materialized workspace live beside the source folder or in
  an OpenHCS GUI cache/workspace directory?
- Should provenance metadata use an existing extension field or require a small
  schema extension to `OpenHCSMetadata`?
- Should normal `Add Plate` auto-offer the CP import workflow, or should this be
  a separate explicit import action only?
- Should the source resolver support `.cpproj` or CellProfiler batch files in
  the same pass, or defer that until `.cppipe` ingestion is stable?
