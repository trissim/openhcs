# CellProfiler Analyst Export Integration Plan

Date: 2026-06-11

## Purpose

OpenHCS currently produces CellProfiler-like images and measurement CSVs for converted `.cppipe` pipelines, but it does not yet produce a CellProfiler Analyst-loadable project. This plan defines the missing integration between existing OpenHCS result materialization infrastructure and the contract implemented by CellProfiler's `ExportToDatabase` module.

The target is explicit: when a converted `.cppipe` contains `ExportToDatabase` configured for CellProfiler Analyst output, OpenHCS should write a CPA project bundle that CellProfiler Analyst can open without manual reconstruction.

## Current OpenHCS Result Infrastructure

### Runtime Artifacts

OpenHCS already has typed runtime artifacts:

- `ArtifactKind.IMAGE`
- `ArtifactKind.OBJECT_LABELS`
- `ArtifactKind.MEASUREMENTS`
- `ArtifactKind.RELATIONSHIPS`
- `ArtifactKind.TABLE`
- `ArtifactKind.SPATIAL_GRID`
- `ArtifactKind.METADATA`

These are stored through `RuntimeValueStore` records. Measurement records carry `MeasurementTable` schemas, including measurement subject scope, object name, object id field, fields, and source image name. Relationship records carry `ObjectRelationship` semantics, including source/target endpoints and relationship ids.

This is the correct source of truth for CPA export. CPA export should not parse already-written CSV filenames as its primary input.

### Per-Artifact Materialization

The current materialization layer is writer-driven:

- `CsvOptions` writes one CSV file.
- `JsonOptions` writes one JSON file.
- `ROIOptions` writes ImageJ ROI archives.
- `TiffStackOptions` writes TIFF stack/slice outputs.
- `TextOptions` writes text files.

Function outputs declare `special_outputs(...)` / `artifact_outputs(...)`; compiled step plans materialize each runtime artifact record through `materialize_artifact_outputs(...)`. Default artifact-kind rules write measurements/relationships/tables as CSV, object labels as ROI ZIP, metadata/spatial grids as JSON.

This layer is useful for file outputs and inspection, but it is not the right abstraction for CPA because CPA is a run-level export, not a per-artifact file writer.

### Image Materialization And OpenHCS Metadata

The normal step output path writes image outputs and OpenHCS metadata:

- step images go to a planned output directory such as `images`;
- analysis artifacts go to the sibling results directory such as `images_results`;
- `openhcs_metadata.json` records image files, component values, available backends, source parser, workspace mapping, and `results_dir`.

For source-schema workspaces, filenames are normalized by `SourceSchemaFilenameParser` into:

```text
{well}_s{site}_w{channel}_z{z_index}_t{timepoint}.{ext}
```

Those filenames carry the virtual image axes that CPA path/file columns will need to reference or derive.

### CellProfiler Conversion Runtime

Converted `.cppipe` pipelines split modules into:

- processing modules that become OpenHCS steps;
- infrastructure modules such as `LoadData`, `ExportToSpreadsheet`, `ExportToDatabase`, `SaveImages`, and `CreateBatchFiles`.

`ExportToSpreadsheet` is currently represented as table materialization. `SaveImages` is represented as image export validation. `ExportToDatabase` is currently classified as infrastructure, but OpenHCS has no real database/project writer for it. The existing `openhcs.interop.cellprofiler.database_export.export_to_database` function is a stub: it passes the image through and records small export metadata only.

### Existing Validation

`validate_cppipe_execution(...)` currently validates:

- runtime artifacts declared by generated contracts;
- table exports when `ExportToSpreadsheet` is present;
- image exports when `SaveImages` is present.

It does not validate `ExportToDatabase`, SQLite files, CPA `.properties` files, `Per_Image`, `Per_Object`, per-object tables, or relationship tables.

### Existing Result Consolidation

OpenHCS has CSV consolidation helpers that scan result directories and emit MetaXpress-style summaries. These are downstream summaries over existing CSV files. They are not CPA-compatible exports and should not become the CPA authority.

## What CellProfiler Analyst Expects

The bundled CellProfiler `ExportToDatabase` source defines the relevant contract.

### Project Bundle

`ExportToDatabase` writes measurements directly to a database or database-readable output and can create a CellProfiler Analyst properties file.

For OpenHCS, the first target should be SQLite plus `.properties`, because SQLite is local, testable, and does not require a MySQL server.

Expected files:

- SQLite database file, configured by `db_sqlite_file`.
- One or more CPA `.properties` files.

### Database Tables

CellProfiler describes two primary table families:

- `Per_Image`: one row per CellProfiler image cycle/image set, keyed by `ImageNumber`.
- `Per_Object` or `Per_<ObjectName>`: object measurement rows, keyed by `ImageNumber` and object number.

Depending on `ExportToDatabase` settings, object measurements may be represented as:

- one combined `Per_Object` table;
- one table per object type, such as `Per_Nuclei`;
- one combined object view over per-object tables.

CellProfiler also supports:

- experiment tables/properties;
- aggregate columns on `Per_Image`;
- optional relationship type and relationship edge tables;
- optional thumbnails;
- optional `Per_Well` SQL for MySQL-oriented workflows.

The first OpenHCS implementation should target a conservative SQLite subset:

- `Per_Image`;
- one table per object type, or one combined `Per_Object` if the pipeline settings request that and the data are compatible;
- relationship tables when relationship artifacts exist;
- experiment/properties metadata sufficient for CPA to open.

### CPA Properties File

The CPA properties file declares at least:

- database connection info, including `db_type` and `db_sqlite_file`;
- `image_table`;
- `object_table`;
- `image_id`;
- `object_id`;
- `plate_id`;
- `well_id`;
- grouping/timepoint identifiers;
- object coordinate columns (`cell_x_loc`, `cell_y_loc`, `cell_z_loc`);
- `image_path_cols`;
- `image_file_cols`;
- `image_names`;
- `image_channel_colors`;
- `channels_per_image`;

CPA also expects image path/file column lists to line up with `Per_Image` columns. CellProfiler's source states that individual image files are expected to be monochromatic and represent a single channel unless `channels_per_image` says otherwise.

## Gap Analysis

### Already Available

- Runtime artifacts retain typed image/object/measurement/relationship semantics.
- Source-schema virtual filenames preserve well, site, channel, z, and timepoint.
- CellProfiler runtime code already projects measurement rows into CellProfiler `ImageNumber` space.
- Relationships are represented as typed parent/child runtime artifacts.
- Existing materialization can still write inspection CSVs, ROI ZIPs, and images.
- The parser keeps `ExportToDatabase` module settings as infrastructure module blocks.

### Missing

- No nominal CPA export request/model.
- No parser for `ExportToDatabase` settings into an OpenHCS export plan.
- No run-level exporter over all compiled runtime stores.
- No SQLite schema writer for `Per_Image`, object tables, relationships, and experiment metadata.
- No CPA `.properties` writer.
- No table/name/column shortening policy matching CellProfiler's database constraints.
- No explicit object-table strategy matching `ExportToDatabase` settings.
- No image path/file column projection from OpenHCS source-schema virtual images and/or SaveImages outputs.
- No CPA validation in `validate_cppipe_execution(...)`.
- Current manuscript wording that says OpenHCS exports CPA-compatible results is ahead of implementation evidence.

## Architecture Decision

CPA export should be a post-execution CellProfiler interop export, not a generic materialization writer.

Reason:

- Generic materialization owns one artifact value and one output path.
- CPA export owns the whole run: all image sets, all measurement subjects, all object tables, relationships, database metadata, and `.properties`.
- Putting CPA into `csv_materializer` would duplicate `ExportToSpreadsheet` semantics and still fail to produce the database/project-level contract.

The new owner should live under `openhcs/interop/cellprofiler/`, tentatively:

```text
openhcs/interop/cellprofiler/analyst_export.py
openhcs/interop/cellprofiler/export_to_database_settings.py
openhcs/interop/cellprofiler/analyst_image_planes.py
openhcs/interop/cellprofiler/analyst_export_validation.py
tests/unit/test_cellprofiler_analyst_export.py
tests/integration/test_cellprofiler_generated_pipeline.py
```

Advisor scan note: the existing runtime export observation path still recovers
execution output roots through structural attribute probing. The CPA exporter
should not extend that pattern. It should receive a nominal export context with
declared output roots, runtime stores, prepared pipeline metadata, and source
workspace metadata.

Tentative boundary:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerExecutionExportContext:
    prepared: PreparedGeneratedPipeline
    execution: DirectPipelineExecution
    runtime_stores_by_axis: Mapping[str, RuntimeValueStore]
    output_roots: tuple[Path, ...]
    source_workspace_root: Path
    export_root: Path
```

Build this context once at the execution boundary. Downstream CPA table builders
and writers should depend on this nominal context rather than calling
`getattr(...)` against arbitrary orchestrator/compiled-context objects.

This is a product execution boundary, not a benchmark-only convenience hook.
Integration tests may be the first caller, but GUI, batch, direct execution,
and benchmark execution should all route through the same post-execution export
service once a converted `.cppipe` declares `ExportToDatabase`.

### Dry-Run Implementation Result

A minimal implementation spike in
`openhcs/interop/cellprofiler/analyst_export.py` compiled and passed focused
unit tests against real `RuntimeValueStore`, `MeasurementTable`, and
`ObjectRelationship` records.

The spike found two plan constraints that should stay load-bearing:

- `runtime_stores_by_axis` must be an explicit field on
  `CellProfilerExecutionExportContext`. Deriving runtime stores later by
  probing `execution` or compiled context objects would recreate the structural
  fallback problem the exporter is meant to avoid.
- Table row conversion and CPA identity-column checks need a named projection
  authority, currently sketched as `CPATableRowProjection`. Reusing or copying
  private row-conversion helpers from progress/runtime-export modules would make
  CPA export another hidden table dialect.

The dry-run exporter intentionally fails when CPA identity columns such as
`ImageNumber` and `ObjectNumber` are absent. Those fields must be produced by
CellProfiler runtime semantics or by an explicit upstream projection step, not
invented inside the CPA writer.

A second dry-run added `export_to_database_settings.py` and a minimal CPA
properties renderer. It compiled and passed focused tests against a real
`ModuleBlock` / `ModuleSetting` shape. That spike added two more constraints:

- `ExportToDatabase` setting values that choose closed families must be parsed
  through nominal literal families, not string-key dictionaries or inline
  `if choice == ...` branches. The dry run uses registered parser families for
  object table mode and object selection policy.
- CPA properties rendering can be derived from
  `CellProfilerAnalystExportRequest`, `CPAImageChannelSpec`, and
  `CellProfilerAnalystProjection`; it does not need to inspect materialized CSV
  outputs. The remaining hard boundary is column dialect: OpenHCS runtime rows
  currently expose generic `ObjectNumber`, while native CellProfiler per-object
  properties often refer to object-qualified columns such as
  `<Object>_Number_Object_Number`. The real implementation needs one explicit
  column-naming/dialect authority before SQLite writing is added.

A third dry-run should make that column dialect load-bearing as generic OpenHCS
infrastructure, not a CPA-only helper. The proposed split:

- Core owns semantic projection identities in `openhcs/core/runtime_table_projection.py`.
  Core identities name roles such as image id, object id, metadata, source image
  path/file, and measurement feature. Core does not encode CellProfiler strings.
- CellProfiler owns a dialect implementation in
  `openhcs/interop/cellprofiler/database_column_dialect.py`. This dialect renders
  core identities into `Per_Image`, `Per_<Object>`, `ImageNumber`,
  `ObjectNumber`, `<Object>_Number_Object_Number`, `Image_Metadata_*`, and
  `Image_PathName_*` / `Image_FileName_*` names.
- CPA properties rendering and SQLite writing must consume this dialect. They
  must not construct table or column names directly.

This creates the missing write-side counterpart to
`RuntimeMeasurementLookupDialect`: lookup dialects resolve external feature names
back to runtime semantics; projection dialects render runtime semantics out to an
external tabular contract.

The third dry-run implemented this split as a reversible spike:

- `RuntimeProjectedColumnIdentity`, `RuntimeProjectedColumnRole`,
  `RuntimeProjectedTable`, and `RuntimeTableProjectionDialect` live in core.
- `CellProfilerDatabaseColumnDialect` renders those core identities into
  CellProfiler/CPA table and column names.
- `CellProfilerColumnNameRenderer` is a registered role-renderer family, so
  column-role rendering is not an enum dispatch ladder.
- `CPAImageChannelSpec` is semantic-only: it names the image/channel and display
  color. It no longer carries `path_column` or `file_column`, because those are
  dialect-rendered database columns.
- `CPAPropertiesRenderer` and `CellProfilerAnalystProjectionBuilder` consume the
  dialect instead of hardcoding names such as `Per_Image`, `ImageNumber`,
  `Image_Metadata_Plate`, `Image_PathName_*`, or object location columns.

This dry run confirms the abstraction can be load-bearing for CPA and reusable
for other external tabular exports. The remaining implementation work is to move
SQLite row/table writing onto the same `RuntimeTableProjectionDialect` rather
than creating writer-local naming rules.

## Proposed Data Flow

1. Parse `.cppipe`.
2. Partition infrastructure modules as today.
3. Build `PreparedGeneratedPipeline`.
4. Execute generated OpenHCS pipeline.
5. If infrastructure modules contain enabled `ExportToDatabase`, build `CellProfilerAnalystExportRequest` from:
   - the `ExportToDatabase` module block;
   - `PreparedGeneratedPipeline.source_schema`;
   - `PreparedGeneratedPipeline.generated_pipeline.artifact_contracts`;
   - compiled contexts and their `RuntimeValueStore` records;
   - output roots and source-schema workspace metadata.
6. `CellProfilerAnalystExporter.write(request)` emits:
   - SQLite database file;
   - CPA `.properties` file(s);
   - optional diagnostic manifest JSON for OpenHCS validation only.
7. `validate_cppipe_execution(...)` checks the CPA export if `ExportToDatabase` is present.

## New Nominal Types

### Export Settings

```python
@dataclass(frozen=True, slots=True)
class CellProfilerDatabaseExportSettings:
    database_type: Literal["sqlite"]
    sqlite_file: str
    experiment_name: str
    table_prefix: str
    object_table_mode: CellProfilerObjectTableMode
    selected_objects: tuple[str, ...] | None
    wants_properties_file: bool
    wants_relationship_tables: bool
```

This should be built from setting-name helpers, not raw string lookups scattered through the exporter.

Channel/image presentation belongs in a separate source-schema-derived projection
because it is not owned only by `ExportToDatabase` settings.

```python
@dataclass(frozen=True, slots=True)
class CPAImageChannelSpec:
    alias: str
    image_name: str
    path_column: str
    file_column: str
    channel_color: str
    channels_per_image: int = 1
```

`CPAImageChannelSpec` is derived from source schema, NamesAndTypes/LoadData
semantics, and any `ExportToDatabase` channel-display settings. It is not
derived from materialized CSV files.

### Export Request

```python
@dataclass(frozen=True, slots=True)
class CellProfilerAnalystExportRequest:
    settings: CellProfilerDatabaseExportSettings
    context: CellProfilerExecutionExportContext
    image_channels: tuple[CPAImageChannelSpec, ...]
```

The request should fail at construction if required paths or module settings are missing. No implicit fallback.

### Export Model

```python
@dataclass(frozen=True, slots=True)
class CPAImageRow:
    image_number: int
    metadata: Mapping[str, object]
    image_path_columns: Mapping[str, str]
    image_file_columns: Mapping[str, str]
    measurements: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class CPAObjectTable:
    object_name: str
    rows: tuple[Mapping[str, object], ...]
    object_id_column: str

@dataclass(frozen=True, slots=True)
class CPATableRowProjection:
    ...

@dataclass(frozen=True, slots=True)
class CPADatabaseProject:
    image_rows: tuple[CPAImageRow, ...]
    object_tables: tuple[CPAObjectTable, ...]
    relationship_tables: tuple[CPARelationshipTable, ...]
    properties: CPAProperties
```

These records are render projections only. They do not own image, object,
measurement, relationship, or source identity semantics. Every field must trace
back to one of the existing authorities:

- `RuntimeValueStore`
- `RuntimeValueSchema`
- `MeasurementTable`
- `ObjectRelationship`
- source-schema workspace metadata
- parsed `ExportToDatabase` settings

The exporter must rebuild these projections from those authorities for each
run. It must not cache semantic state, parse OpenHCS CSV outputs, or create a
second source of truth.

### Image Plane Export Policy

CPA path/file columns must resolve to channel images that CPA can load. This
cannot be an implicit "use the original path if it works" rule; it needs a
nominal policy with explicit validation.

```python
class CPAImagePlaneExportPolicy(ABC):
    @abstractmethod
    def plane_ref(self, request: CPAImagePlaneExportRequest) -> CPAImagePlaneRef: ...

@dataclass(frozen=True, slots=True)
class OriginalMonochromeFile(CPAImagePlaneExportPolicy):
    ...

@dataclass(frozen=True, slots=True)
class GeneratedMonochromeTiff(CPAImagePlaneExportPolicy):
    ...
```

Policy rules:

- `OriginalMonochromeFile` is valid only when the source candidate already
  resolves to exactly one CP-readable monochrome image plane.
- `GeneratedMonochromeTiff` writes an explicit monochrome TIFF plane from the
  existing OpenHCS/source-schema image payload and records that generated file
  in the CPA image path/file columns.
- Multichannel, Bio-Formats-only, virtual-only, Z-stack, or time-series sources
  must be split into explicit image-plane refs or fail.
- The policy result is a file reference for CPA rendering, not a new source
  image identity.

## Implementation Phases

### Phase 1: Contract Extraction And Guardrails

Add a real `ExportToDatabase` infrastructure note that says CPA export is handled by the post-execution CPA exporter, not by `@special_outputs`.

Add settings parsing for the subset needed by SQLite CPA export:

- database type;
- SQLite filename;
- experiment name;
- table prefix;
- object-table mode;
- selected objects;
- properties file enabled;
- relationship tables enabled;
- image/channel display settings where present.

Add fail-loud behavior:

- unsupported MySQL export should raise a clear unsupported-export error for OpenHCS CPA export;
- disabled properties output should still allow database writing, but validation must not claim CPA-loadable output unless `.properties` exists;
- unsupported object table modes should be explicit errors, not silent downgrade.
- image-channel settings that cannot be reconciled with the source schema should
  raise before writing any partial CPA project.

### Phase 2: Runtime Store To CPA Tables

Create a collector over all compiled contexts:

- Collect `MEASUREMENTS` records and group by `MeasurementSubject`.
- Collect `OBJECT_LABELS` records for object counts and object identity.
- Collect `RELATIONSHIPS` records for relationship tables.
- Collect image/source metadata and image paths from source-schema workspace metadata.
- Build `CPAImageChannelSpec` values from source schema and infrastructure settings.
- Resolve each image row's CPA path/file columns through `CPAImagePlaneExportPolicy`.

Rules:

- `ImageNumber` comes from the existing CellProfiler image-number projection machinery.
- Image-scope measurement rows go to `Per_Image`.
- Object-scope measurement rows go to the object table for that subject.
- `ObjectNumber` must be present or derived from the declared object id field. If it cannot be derived, fail.
- Relationship rows use typed `ObjectRelationship` endpoints and must map to CPA relationship table columns.
- CPA projection records are discarded after rendering and validation. They are
  never persisted as semantic state.

Do not read OpenHCS materialized CSV files to build this model unless validating a compatibility fixture. The runtime store is the semantic authority.

### Phase 3: SQLite Writer

Write a small CPA database writer that accepts `CPADatabaseProject`.

Responsibilities:

- create `Per_Image`;
- create `Per_Object` or `Per_<ObjectName>` tables according to settings;
- create relationship type and relationship edge tables when requested;
- insert rows with stable column ordering;
- coerce values to SQLite-compatible types;
- preserve NULLs/NaNs in a CPA-compatible way;
- create indexes/primary keys for `ImageNumber` and object ids;
- write experiment/properties metadata sufficient for CPA.

This writer may use Python's `sqlite3`, but it is not an OpenHCS
materialization backend. It is a renderer for the CPA project contract and
should write through the configured filesystem/output-root boundary.

Column naming needs a nominal policy:

```python
class CellProfilerDatabaseColumnNamePolicy:
    max_length: int
    def column_name(object_name: str, feature_name: str) -> str: ...
```

The policy must detect collisions after shortening and fail or disambiguate deterministically.

### Phase 4: CPA Properties Writer

Generate `.properties` from the same `CPADatabaseProject`, not from filesystem scanning.

Minimum properties:

- `db_type = sqlite`
- `db_sqlite_file = <absolute or project-relative sqlite path>`
- `image_table = <prefix>Per_Image`
- `object_table = <prefix>Per_Object` or selected object table
- `image_id = ImageNumber`
- `object_id = ObjectNumber` or `<ObjectName>_Number_Object_Number`
- `plate_id`, `well_id`, `series_id`, `group_id`, `timepoint_id`
- `cell_x_loc`, `cell_y_loc`, `cell_z_loc` when available
- `image_path_cols`
- `image_file_cols`
- `image_names`
- `image_channel_colors`
- `channels_per_image`

Path/file columns should be generated from the source-schema virtual image set:

- each CP image alias/channel gets `Image_PathName_<alias>` and `Image_FileName_<alias>`;
- path/file values come from `CPAImagePlaneExportPolicy`;
- every path/file pair must point to exactly one CP-readable monochrome image plane;
- generated monochrome TIFFs are allowed when the original source is not already a valid CPA plane;
- originals are allowed only through the explicit `OriginalMonochromeFile` policy.

No implicit conversion fallback.

### Phase 5: Execution Hook

Add a post-execution hook for converted `.cppipe` execution.

Initial implementation:

- Add a service such as `CellProfilerPostExecutionExporter.run(context)`.
- Build `CellProfilerExecutionExportContext` at the same boundary that knows
  prepared pipeline metadata, execution results, output roots, and source
  workspace root.
- Call this service from integration/benchmark direct execution first only as
  the first rollout path, not as a separate benchmark-only API.

Full integration:

- The prepared CellProfiler workspace should carry `source_workspace_root` and export target root.
- Batch/GUI execution should call the same post-execution export service after successful execution when `ExportToDatabase` exists.
- Execution status should mark export failure as a pipeline failure if `ExportToDatabase` was part of the original `.cppipe`.

### Phase 6: Validation

Extend `CPPipeInfrastructureFeature` with `EXPORT_TO_DATABASE`.

Add CPA export validation:

- `.sqlite` file exists when SQLite export is requested;
- `.properties` exists when CPA properties are requested;
- properties file references existing database file;
- database contains `Per_Image`;
- object tables match selected object mode;
- `Per_Image.ImageNumber` row count matches source image sets;
- all object-table `ImageNumber` values are present in `Per_Image`;
- object rows have `ImageNumber` and object id columns;
- object ids are unique per `ImageNumber` within each object table;
- object coordinate columns required by properties exist when object
  classification/viewing is configured;
- image path/file columns exist and resolve to files;
- image path/file columns are non-null for every `Per_Image` row;
- image path/file pairs resolve to CP-readable monochrome image planes;
- relationship tables exist when relationship export is requested.
- relationship rows reference existing image/object ids.

Add a stricter optional validation gate:

- run a small CPA import/open smoke test if CellProfiler Analyst is installed;
- otherwise run schema-level validation only.

## Tests

### Unit Tests

- Parse `ExportToDatabase` settings from synthetic module blocks.
- Build `CPAImageChannelSpec` from source-schema aliases and infrastructure settings.
- Build `CPADatabaseProject` from hand-constructed runtime stores.
- Project image-scope measurements to `Per_Image`.
- Project object-scope measurements to per-object tables.
- Reject object measurements without object id.
- Reject image columns that cannot resolve to exactly one CP-readable image plane.
- Select `OriginalMonochromeFile` only for valid original monochrome files.
- Select or require `GeneratedMonochromeTiff` for virtual/multichannel/non-CP-readable sources.
- Write SQLite schema and verify table/column names.
- Write `.properties` and verify required keys.
- Verify column-name collision policy.

### Integration Tests

- Synthetic `.cppipe` with `ExportToDatabase` and one object type:
  - execute OpenHCS converted pipeline;
  - write SQLite + properties;
  - validate schema and row counts.
- Synthetic `.cppipe` with two object types and relationships:
  - verify relationship tables;
  - verify per-object table mode.
- Official CellProfiler example with `ExportToDatabase` if available:
  - compare OpenHCS export schema/row counts against native CP output.

### Benchmark/Parity Tests

- Extend benchmark comparison to include CPA export artifacts only for pipelines that contain `ExportToDatabase`.
- Keep existing runtime equivalence for CSV/image parity separate from CPA project validation.

## Non-Goals For First Slice

- MySQL export.
- Oracle export.
- MySQL-only `Per_Well` SQL.
- Thumbnail BLOB export.
- Full CP column shortening parity for every historical edge case.
- CPA classifier/model compatibility beyond project loading.
- Reading OpenHCS CSV result files as the primary data source.

## Risks

- CPA may be stricter about column naming than our first projection. The column-name policy must be isolated and heavily tested.
- Combined `Per_Object` tables are only valid when object rows have compatible one-to-one identities. The exporter should prefer per-object tables unless the `.cppipe` explicitly requests combined output and the data prove compatible.
- Source images may not be directly CP-readable if they are only virtual mappings or nonstandard formats. The exporter must either point to real CP-readable files or fail with a clear message.
- 3D/time series support needs explicit mapping to CPA `series_id`, `group_id`, `timepoint_id`, and possibly `cell_z_loc`. Do not silently collapse axes.
- Treating SQLite as an OpenHCS materialization backend would blur the
  architecture. CPA database writing must remain a project renderer over runtime
  semantics.
- The current manuscript claim that OpenHCS exports CPA-compatible results should stay gated until this validation passes.

## Review Checklist

- Does the plan preserve OpenHCS runtime semantics as the source of truth?
- Does it avoid treating CSV materialization as a database/project export?
- Does it fail loudly for unsupported `ExportToDatabase` modes?
- Does it keep source-schema virtual filenames load-bearing for image axis semantics?
- Does it make CPA compatibility a validated export feature rather than a broad claim?
