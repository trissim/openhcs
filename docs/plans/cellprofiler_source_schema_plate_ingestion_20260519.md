# CellProfiler Source-Schema Plate Ingestion

## Problem

The PyQt Plate Manager currently initializes a selected folder by creating a
`PipelineOrchestrator` and calling `orchestrator.initialize()`. That path calls
`create_microscope_handler(..., microscope_type="auto")`, which only
auto-detects registered microscope metadata handlers such as OpenHCS metadata,
ImageXpress, OMERO, and Opera Phenix.

Official CellProfiler examples such as `ExampleFly` are not microscope plates in
that sense. They are flat image datasets plus a `.cppipe` file whose `Images`,
`Metadata`, `NamesAndTypes`, and `Groups` modules define source selection,
metadata extraction, image aliases, and image-set grouping. Loading the
`.cppipe` through Pipeline Editor works, but adding the image/example folder as
a plate fails before CellProfiler source-schema semantics are used.

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

- `tests/integration/test_cellprofiler_generated_pipeline.py`
  - Existing tests already execute converted CellProfiler pipelines by manually
    calling `materialize_source_schema_workspace(...)`, then creating a
    `PipelineOrchestrator` on the resulting `workspace_root`.
  - This proves the runtime path exists. The missing piece is product
    orchestration and GUI ingestion.

## Target Architecture

Treat `.cppipe` as source-schema metadata for flat image datasets, not as a
native microscope metadata format. The bridge should translate CellProfiler
source semantics into an OpenHCS source-schema workspace, then hand normal
OpenHCS metadata to the existing microscope/orchestrator path.

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
    compilation, generated pipeline preparation, and CP-specific source
    resolution policy.

- `openhcs.core`
  - Owns `PipelineImageSchema`, `StepSourceBindingsConfig`,
    `materialize_source_schema_workspace(...)`, image-set selection, and generic
    source-schema workspace materialization semantics.

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
2. OpenHCS resolves the source dataset into a typed ingestion request.
3. OpenHCS compiles the `.cppipe` source schema.
4. OpenHCS materializes an `openhcs_metadata.json` source-schema workspace.
5. Plate Manager adds the materialized workspace root as the plate.
6. Pipeline Editor loads the converted `FunctionStep` declarations from the
   same import result.
7. From this point onward, orchestrator initialization, compilation, debug, and
   execution use the existing OpenHCS path.

## Proposed New Product Contract

Add a small product-level ingestion service, not a PyQt-only helper and not a
benchmark-only adapter method:

```python
@dataclass(frozen=True, slots=True)
class SourceSchemaPlateIngestionRequest:
    selected_path: Path
    workspace_root: Path | None = None
    filemanager: FileManagerLike | None = None
    source_backend: Backend = Backend.DISK
    workspace_backend: Backend = Backend.DISK
    max_image_set_count: int | None = None


@dataclass(frozen=True, slots=True)
class SourceSchemaPlateIngestionResult:
    source_root: Path
    workspace_root: Path
    cppipe_path: Path
    generated_pipeline_path: Path
    import_result: CellProfilerPipelineImportResult
    materialization: SourceSchemaWorkspaceMaterialization
```

The service should live outside PyQt and outside `benchmark`. Split it across
the real OpenHCS layers:

- `openhcs/interop/cellprofiler/source_schema_ingestion.py`
  - CP-specific request resolution and `.cppipe` import/preparation.

- `openhcs/core/source_schema_workspace.py`
  - Existing generic workspace materialization remains here.

- `openhcs/microscopes/openhcs.py`
  - Existing OpenHCS metadata/virtual-workspace initialization remains here.

The interop ingestion service responsibilities are:

- Resolve `selected_path` into `cppipe_path` and `source_root`, productizing the
  local-source subset of `benchmark.adapters.cppipe_source`.
- Compile the `.cppipe` once using the existing CellProfiler dialect compiler.
- Choose a deterministic default `workspace_root`, for example:
  - sibling: `<selected_root>_openhcs_source_schema`
  - cache: configured OpenHCS GUI workspace/cache root
  - explicit request override for tests and CLI.
- Call `materialize_source_schema_workspace(...)`.
- Return both the generated pipeline/import result and materialized workspace.

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

Add a user-visible action that represents the real operation:

- `Import CellProfiler Pipeline + Images...`

This should not be hidden inside normal `Add Plate` without explanation. The
normal `Add Plate` path can optionally detect a `.cppipe` folder and offer to
run the import action, but source-schema ingestion is conceptually different
from adding a native microscope plate.

Implementation sketch:

- Add `MainWindowCellProfilerImportActions` or extend
  `MainWindowPipelineActions` only if the method remains thin.
- The action calls the product ingestion service.
- It then calls `plate_manager.add_plate_callback([result.workspace_root])`.
- It updates Pipeline Editor with `result.import_result.pipeline.steps` and
  `result.import_result.source_schema`.
- It associates the imported pipeline with the materialized workspace plate key
  so `PlateManagerWidget._get_current_pipeline_definition(...)` returns the
  converted steps.

Avoid making `PlateManagerWidget.add_plate_callback(...)` parse `.cppipe`
directly. Plate Manager should continue to add orchestrator scopes; the import
workflow should prepare a valid scope first.

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

Completion goal: benchmark CLI behavior stays unchanged, but the GUI imports and
benchmark runtime execution call the same source-schema ingestion authority.

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
- Initialize `PipelineOrchestrator(result.workspace_root)`.
- Execute at least one selected well/image set.

### PyQt Tests

Add a GUI workflow test that does not require manual interaction:

- Create/acquire ExampleFly-like folder.
- Invoke the import action with a selected path override.
- Assert Plate Manager receives `result.workspace_root`, not the raw flat image
  folder.
- Assert Pipeline Editor has converted steps and source schema.
- Assert initializing the added plate reaches `READY`.

## Implementation Phases

### Phase 1: Product Ingestion Service

- Add typed resolver/result classes.
- Compile `.cppipe` through the existing dialect compiler.
- Materialize source-schema workspace.
- Cover with unit tests.
- Add a direct lower-level API for the benchmark path where `source_root` and
  `cppipe_path` are already known.

Completion gate:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_source_schema_ingestion.py -q
.venv/bin/python -m nominal_refactor_advisor openhcs/interop/cellprofiler/source_schema_ingestion.py
```

### Phase 2: Runtime Integration

- Add ingestion-based integration test for ExampleFly or a small synthetic CP
  dataset.
- Ensure `PipelineOrchestrator(result.workspace_root).initialize()` uses
  `OpenHCSMicroscopeHandler` through existing auto-detection.
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

### Phase 3: PyQt Import Workflow

- Add GUI action for `Import CellProfiler Pipeline + Images...`.
- Keep the action as orchestration over the product ingestion service.
- Associate the converted pipeline with the materialized workspace plate.
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
