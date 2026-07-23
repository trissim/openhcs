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

The implementation should be core-first. CellProfiler Analyst export is the
first product requirement, but most of the needed machinery is not
CellProfiler-specific:

- final-state runtime record selection;
- source image-set row construction;
- source image plane resolution/export;
- external table projection identities;
- SQLite table writing from projected tables;
- project-level export validation.

Those owners should live in core and accept dialect/profile objects. The
CellProfiler package should stay thin: it parses `ExportToDatabase`, provides a
CellProfiler/CPA table dialect, defines the CPA project profile/properties
renderer, and supplies CellProfiler-readable image-plane compatibility rules.

Tentative core-side owners:

```text
openhcs/core/runtime_record_selection.py
openhcs/core/runtime_table_projection.py
openhcs/core/source_image_set_projection.py
openhcs/core/image_plane_export.py
openhcs/core/tabular_project_export.py
openhcs/core/project_export_validation.py
```

Dry-run note: these names are architectural roles, not necessarily all new
files. Prefer extending the current core seams where they already own the
concept:

- `runtime_table_projection.py` already owns projected column/table identities;
- `runtime_artifact_queries.py` already owns typed runtime-artifact query
  helpers and measurement table projections;
- `runtime_exports.py` and `runtime_execution_validation.py` already own export
  expectations/observations, but currently only for table/image files;
- `source_projection.py` already owns `SourcePixelRef`, `SourcePlaneProjection`,
  and `SourceProjectionSet`;
- `source_workspace_projection.py` and `source_schema_workspace.py` already own
  virtual workspace/source metadata projection;
- `image_file_serialization.py` already owns file-format-aware image payload
  preparation.

New files should be introduced only when those existing owners would become
cohesion sinks. The goal is core ownership, not module sprawl.

Tentative CellProfiler edge owners:

```text
openhcs/interop/cellprofiler/analyst_export.py
openhcs/interop/cellprofiler/export_to_database_settings.py
openhcs/interop/cellprofiler/database_column_dialect.py
openhcs/interop/cellprofiler/analyst_project_profile.py
openhcs/interop/cellprofiler/analyst_image_planes.py
openhcs/interop/cellprofiler/analyst_export_validation.py
tests/unit/test_cellprofiler_analyst_export.py
tests/integration/test_cellprofiler_generated_pipeline.py
```

Ownership rule: if a helper can be named without `CellProfiler`, `CPA`, or a
specific `.cppipe` setting, it belongs in core. CellProfiler interop code should
compose core services with CellProfiler dialect/profile objects, not traverse
runtime stores, source workspace metadata, or SQLite schema concerns itself.

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
  authority, now better framed as core `RuntimeTableRowProjection`. Reusing or copying
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
- CellProfiler owns only the dialect implementation in
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

The same split should apply to the rest of the exporter. The generic operation
is "render runtime/source semantics into an external project profile"; the
CellProfiler-specific operation is "provide the CPA profile and naming/settings
semantics." New code that only knows about rows, records, source image sets,
file references, SQLite tables, or validation references should be core code.

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

### Fourth Dry Run Against Native `ExportToDatabase` Fixtures

A fourth dry run used the three native `ExportToDatabase` fixtures under
`benchmark/native_refs/official30_scoped_rows`:

- `BBBC022_Analysis_Final.cppipe`
- `BBBC022_QC.cppipe`
- `Translocation_final.cppipe`

The current nominal settings parser successfully extracts the core SQLite CPA
settings from all three fixtures:

- SQLite file;
- experiment name;
- table prefix;
- object table mode;
- selected-object policy;
- CPA properties toggle;
- relationship-table toggle.

The same dry run exposed gaps that the implementation plan must close before
the exporter can be considered CPA-compatible.

First, the actual `ExportToDatabase` module surface is larger than the current
`CellProfilerDatabaseExportSettings` subset. The QC fixture has 173 ordered
settings, including plate/well metadata names, image classification mode,
thumbnail/workspace toggles, group fields, filter fields, CPA workspace plot
definitions, phenotype class table name, object-location source, and image
presentation blocks. The first implementation may deliberately reject or defer
some of these, but it must parse them into nominal supported/unsupported setting
families. Ignoring unknown CPA-bearing settings would make validation lie about
CPA compatibility.

Second, image presentation cannot be copied blindly from one setting family.
`BBBC022_QC.cppipe` contains five repeated image-presentation blocks, but the
native properties file renders source aliases as `image_names` and default CPA
colors, not the UI display names/colors from those blocks. The advanced
segmentation and translocation fixtures contain a single `Select an image to
include = None` block even though their source schemas compile to real aliases.
This needs a nominal `CPAImagePresentationPolicy`, with explicit modes such as:

- render explicit image blocks when they name concrete source aliases;
- render all source-schema image assignments when the module requests default
  image information;
- reject ambiguous or unreconciled image settings before writing a partial
  project.

This is not a fallback. It is an `ExportToDatabase` setting semantics policy
that must validate against `PipelineImageSchema.assignments_by_alias`.

Third, source-schema compilation succeeds for all three fixtures and gives the
right image universe:

- advanced segmentation: `OrigHoechst`, `OrigER`, `OrigSyto`, `OrigPh_golgi`,
  `OrigMito`, plus illumination source artifacts;
- quality control: `OrigER`, `OrigHoechst`, `OrigMito`, `OrigPh_golgi`,
  `OrigSyto`;
- translocation: `rawDNA`, `rawGFP`.

This confirms that source schema is the image/channel authority for CPA row
construction. `ExportToDatabase` image blocks are presentation instructions that
must be reconciled with that authority, not a separate image universe.

Fourth, the native QC SQLite/properties fixture is more specific than the
earlier plan:

- `BBBC022QC_Per_Image` has 2 rows and 169 columns;
- `BBBC022QC_Per_Object` exists even for image-classification mode and has only
  `ImageNumber`, `ObjectNumber` with zero rows;
- `BBBC022QC_Per_Experiment`, `Experiment`, and `Experiment_Properties` exist;
- the properties file sets `classification_type = image`;
- `group_SQL_PerWell` references the prefixed `Per_Image` table.

The exporter therefore needs an image-classification-aware properties/database
mode. It cannot omit the object table just because there are no object rows.
For image classification, a minimal empty object table may still be part of the
CPA project contract.

Fifth, validation currently only recognizes `ExportToSpreadsheet` and
`SaveImages` as infrastructure features. `ExportToDatabase` must become its own
validation feature with SQLite/properties/database checks. Runtime-artifact
validation is insufficient for this export because the CPA bundle is a
post-execution project-level artifact.

### Fifth Dry Run Of The Core-First Integration

A fifth dry run walked the core-first plan against the current OpenHCS core APIs
instead of only the CellProfiler fixtures. It exposed several implementation
gaps that should be fixed in core before the CPA edge grows more code.

First, final runtime-record selection is not currently a public core operation.
`RuntimeValueStore` tracks current bindings internally, but public
`find(...)`/`values()` still return every stored location. A synthetic
`record(...)` followed by `replace(...)` produces:

```text
find_count = 2
values_count = 2
observed_count = 2
current_path = /memory/second.pkl
```

CPA export needs the current binding, while execution validation often needs the
observation history. The implementation should add a core selection surface,
for example:

```python
class RuntimeRecordSelectionMode(Enum):
    CURRENT_BINDINGS = "current_bindings"
    OBSERVED_HISTORY = "observed_history"

@dataclass(frozen=True, slots=True)
class RuntimeRecordSelectionPolicy:
    mode: RuntimeRecordSelectionMode
```

`RuntimeValueStore` should expose current records through a public method, or
the policy should be implemented beside the store without inspecting private
attributes. CPA must not rely on `find(...)` until this distinction is explicit.

Second, source image-set projection exists before and during workspace
materialization, but there is no post-materialization row projector for CPA-like
exports. Core already has `ImageSetRecord`, `ImageSetAssembler`,
`SourceSchemaImageSetIdentity`, `VirtualWorkspaceSourceProjection`, and
OpenHCS metadata with `image_files`, `workspace_mapping`, `source_metadata`,
`channels`, and sometimes `source_projection`. The missing core owner should
build export image rows from those authorities after execution:

```python
@dataclass(frozen=True, slots=True)
class SourceImageSetRow:
    image_number: int
    component_values: Mapping[AllComponents, str]
    images_by_alias: Mapping[str, SourceImagePlaneRef]
    metadata: Mapping[str, str]
```

This row builder should not re-run CellProfiler setup parsing and should not
scan materialized CSVs. It should consume `PipelineImageSchema` plus source
workspace metadata and preserve the same image-set ordering used by the runtime
measurement image-number projection.

Third, image-plane export must preserve structured source refs. Core already has
`SourcePixelRef` with fields such as `plane_index`, `c`, `z`, and `t`, and
`SourceProjectionSet` can serialize those refs into OpenHCS metadata. By
contrast, `VirtualWorkspaceSourceProjection.source_path_for(...)` returns only a
physical path string. That is appropriate for some source-binding lookups, but
not enough for CPA image path/file export when a virtual image is one plane
inside a multi-plane file. The core export path needs a typed
`SourceImagePlaneRef` / `ImagePlaneRef` that keeps both path and plane address.

Fourth, image serialization should reuse existing core file-preparation logic.
`image_file_serialization.py` and `RuntimeImageExportSpec.prepare_payload(...)`
already define concrete file-compatible payload conversion. The new
`ImagePlaneExportPolicy` should delegate pixel conversion to those owners rather
than introducing CPA-local TIFF conversion semantics.

Fifth, projected table writing is still missing. Core has
`RuntimeProjectedColumnIdentity`, `RuntimeProjectedTable`, and table-row
projection helpers for measurement-axis narrowing, but it does not yet have a
generic SQLite writer or project writer over projected tables. The CPA SQLite
writer should therefore be implemented as a core `TabularProjectWriter` over
`RuntimeProjectedTable`/`TabularProject`, with CellProfiler contributing only
the dialect/profile.

Sixth, project-level export validation should extend the runtime export
validation model rather than bypass it. `runtime_exports.py` currently observes
table/image files under output roots. CPA needs a sibling project-export
expectation/observation for SQLite database files, properties files, table
schemas, and cross-file references. This can live beside the current runtime
export validation, but it should remain generic enough for future external
tabular project exports.

### Sixth Dry Run Of Source-Row Reconstruction

A sixth dry run used the native QC selected source workspace:

```text
benchmark/native_refs/official30_scoped_rows/
  CellProfiler_tutorials_cp_tutorial_quality_control_samples_first1wells/
  native_cellprofiler_selected_source_workspace/openhcs_metadata.json
```

The metadata contains one subdirectory with:

- 10 virtual image files;
- `channels = {1: OrigER, 2: OrigHoechst, 3: OrigMito, 4: OrigPh_golgi, 5: OrigSyto}`;
- 10 workspace mappings;
- 10 source metadata records;
- no `source_projection` section.

Grouping `image_files` by `well`, `site`, `z_index`, and `timepoint`, then
mapping each `channel` through `channels`, produces exactly two image-set rows:

```text
1: A01, site 1, z 1, t 1 -> OrigER, OrigHoechst, OrigMito, OrigPh_golgi, OrigSyto
2: A01, site 2, z 1, t 1 -> OrigER, OrigHoechst, OrigMito, OrigPh_golgi, OrigSyto
```

That matches the native `BBBC022QC_Per_Image` row count and ordering:

```text
[(1, "A01", "1"), (2, "A01", "2")]
```

This validates the planned `SourceImageSetRow` shape, but adds several
requirements:

- the row builder should use the canonical `image_files` list from metadata as
  the row/plane enumeration source, not `VirtualWorkspaceSourceProjection`
  key iteration;
- workspace roots should be normalized to absolute paths before projection,
  because relative roots can make `pipeline_start_files()` expose both relative
  and root-prefixed keys;
- `source_projection` is optional in existing metadata, so the row builder must
  support both legacy string `workspace_mapping` values and structured
  `SourcePixelRef` mapping values;
- `SourceImageSetRow.metadata` must preserve non-axis source metadata columns,
  not only OpenHCS components. Native QC uses metadata fields such as
  `ChannelNumber`, `FileLocation`, `Frame`, `Plate`, `Series`, `Site`, and
  `Well`;
- `ImageNumber` assignment must follow the canonical image-set row order after
  filtering/selection, and validation must compare this against `Per_Image`.

The same native QC fixture also contains a CPA `.workspace` file, and its
`ExportToDatabase` module requests workspace output. The SQLite/properties
subset can be the first loadability target, but requested `.workspace` output is
a real CellProfiler feature. The implementation must either render it or fail
with a named unsupported feature. It must not silently ignore the workspace
settings while claiming full parity with native CellProfiler output.

### Seventh Dry Run Of Native SQLite Topology

A seventh dry run inspected the native SQLite/properties outputs for the same
three `ExportToDatabase` fixtures rather than only their pipeline settings:

```text
QC:
  BBBC022QC.db
  BBBC022QC_BBBC022QC.properties
  BBBC022QC_BBBC022QC.workspace
Translocation:
  DefaultDB.db
  DefaultDB.properties
Advanced segmentation:
  BBBC022.db
  BBBC022_<Object>.properties
```

That exposed several requirements that the earlier projection spike did not yet
model:

- image-classification mode can still write an object table. Native QC writes
  `BBBC022QC_Per_Object` with only `ImageNumber` and `ObjectNumber` columns and
  zero rows, and its properties file still points `object_table` at that table;
- combined-object mode writes one `Per_Object` table containing object-qualified
  columns for multiple object types. The translocation fixture has 306 rows in
  `Per_Object` with columns such as `Cells_Number_Object_Number` and
  `Cells_Location_Center_X`;
- per-object mode writes one table and one properties file per object. Advanced
  segmentation writes seven `Per_<Object>` tables and seven
  `BBBC022_<Object>.properties` files;
- relationship export is not one table per runtime relationship record. Native
  CellProfiler writes a relationship type catalog and one edge table:
  `Per_RelationshipTypes(relationship_type_id, module_number, relationship,
  object_name1, object_name2)` and
  `Per_Relationships(relationship_type_id, image_number1, object_number1,
  image_number2, object_number2)`;
- `Experiment_Properties` is not simply "the properties files in table form".
  QC and translocation populate it for their single-properties-file bundles,
  while advanced segmentation leaves it empty despite writing seven external
  properties files;
- native properties include CPA profile fields that are easy to miss in a
  minimal renderer, including `object_name`, `classifier_ignore_columns`,
  `image_channel_colors`, `group_SQL_*`, and `classification_type`.

The plan therefore needs a core relationship graph projection, not a
CPA-local list of relationship tables. Core should represent typed relationship
definitions and edges once; the CellProfiler profile should render that graph
as `Per_RelationshipTypes` and `Per_Relationships`. A projection builder that
turns each `ObjectRelationship` record into its own table is only a spike and
must not become the final SQLite contract.

The SQLite/project profile also needs separate concepts for:

- external CPA properties files;
- database experiment metadata rows;
- database experiment properties rows;
- object-table layout policy.

Those are related CPA artifacts, but native output proves they are not the same
artifact. The core writer should accept a project profile that declares which
of these artifacts to render; it should not infer one from another by scanning
filenames or duplicating property text into SQLite.

## Current Architecture Reconciliation

Investigation of the current source-schema and runtime artifact paths adds a few
load-bearing details to the original plan.

### Source Workspace Is The Image-Plane Authority

`PipelineImageSchema` already owns setup-derived image semantics:

- `assignments_by_alias` declares CellProfiler image aliases and image types;
- `source_artifacts_by_alias` declares non-stack source artifacts such as
  objects and illumination functions;
- `source_stack_components` declares which OpenHCS components remain inside one
  logical source image;
- `measurement_source_names` declares aliases that may appear in measurement
  feature names.

`materialize_source_schema_workspace(...)` lowers this schema into an OpenHCS
virtual workspace and writes `openhcs_metadata.json`. The metadata contains the
CPA-relevant source facts:

- `image_files`: virtual source image paths;
- `workspace_mapping`: virtual path to real source path, or a structured
  `SourcePixelRef` when one virtual image is a plane inside a source file;
- `source_metadata`: per-virtual-path metadata enriched with canonical OpenHCS
  components and image type;
- `channels`: source-stack channel index to CellProfiler alias.

The core image-plane policy and the CellProfiler compatibility profile should
consume these authorities. They should not scan the filesystem or infer channel
names from materialized result CSVs. Core can build source-channel projections
from source-stack assignments and `channels`; CellProfiler can render
`CPAImageChannelSpec` values as a profile view over that projection. Per-row
image file/path values should come from source workspace metadata and typed
image-plane refs.

This also means CPA image rows must be seeded from source image sets, not only
from image-scope measurement artifacts. A pipeline may have valid source images
and path/file columns even when no image measurement module emitted rows. The
correct construction is:

1. Build the `Per_Image` row universe from source workspace image sets and their
   CellProfiler `ImageNumber` ordering.
2. Fill `Image_Metadata_*`, `Image_Group_*`, `Image_PathName_*`, and
   `Image_FileName_*` columns from source workspace metadata and
   the core image-plane export policy plus the CellProfiler compatibility
   profile.
3. Left-merge image-scope `MeasurementTable` rows by `ImageNumber`.

### Runtime Store Selection Must Be Explicit

`RuntimeValueStore` records both current runtime artifact bindings and an
observation stream. Validation currently uses `observed_values`, which is right
for checking that declared artifacts were produced. CPA export is different: it
must write the final database view for one completed run. It therefore needs a
named runtime-record selection policy rather than iterating every write event.

The first implementation should add a small CPA-facing collector that states
which records it consumes:

- final measurement tables by axis and artifact identity;
- final relationship records by axis and artifact identity;
- source/image records only when needed to resolve image-plane provenance.

If the collector intentionally consumes replacement history for a specific table
family, that must be a declared mode. Silent inclusion of stale replaced records
would duplicate rows in `Per_Image` or object tables.

### ImageNumber Is Already Runtime Semantics

The CellProfiler runtime already projects measurement rows into CellProfiler
`ImageNumber` space through `CellProfilerMeasurementMaterializer`,
`MeasurementRowsAxisProjection`, and `CellProfilerImageNumberResolver`. The CPA
exporter should verify `ImageNumber` presence and consistency; it should not
invent a second image-number algorithm inside the SQLite writer.

For rows that still lack `ImageNumber`, the exporter should fail with the
current explicit error unless the missing field is supplied by a reusable
upstream projection authority. Any such authority belongs near the existing
measurement row materialization/image-number resolver code, not in
`analyst_export.py`.

### Projection Rows Must Be Rendered Through The Dialect

The current dry-run projection already uses `RuntimeTableProjectionDialect` for
table names and properties, but row payloads still carry runtime/native field
names. The SQLite writer must close that gap by rendering every table column
through the same dialect or by producing an explicit `RuntimeProjectedTable`
model before writing.

This is not only a formatting concern. The properties file can reference
dialect-rendered columns such as object-qualified ids or `Image_PathName_*`
columns. The database writer must guarantee those columns actually exist in the
SQLite tables it writes. Properties rendering and SQLite row rendering should
therefore share one column projection object, including object-id behavior for
combined `Per_Object`, per-object tables, and object views.

### Execution Boundary

`PreparedGeneratedPipeline` already carries the parsed infrastructure modules,
source schema, generated artifact contracts, and executable pipeline. Direct
execution returns `DirectPipelineExecution` with compiled contexts and execution
results. This is enough to build `CellProfilerExecutionExportContext` after a
successful run.

The missing product owner is a post-execution service, not another materializer.
Its orchestration can live at the CellProfiler integration edge, but every
generic sub-operation should delegate into core:

```text
CellProfilerPostExecutionExporter
  -> find enabled ExportToDatabase modules
  -> parse settings with export_to_database_settings(...)
  -> build CellProfilerExecutionExportContext
  -> core: select final runtime records
  -> core: build source image-set rows and image-plane refs
  -> core: build projected runtime tables through a dialect
  -> core: write SQLite tables from projected tables
  -> CellProfiler: render CPA properties/profile metadata
  -> core + CellProfiler profile: validate project references
```

The first caller can be direct/integration execution, but the service should not
live in benchmark-only code. GUI, batch, and benchmark execution should call the
same service once the export context factory is available.

### Native Fixture Targets

The repo already contains native CellProfiler SQLite/properties fixtures for
several `ExportToDatabase` modes. The quality-control tutorial bundle includes:

- a prefixed `Per_Image` table;
- a prefixed `Per_Object` table;
- a prefixed `Per_Experiment` table;
- `Experiment` / `Experiment_Properties` metadata tables;
- a CPA `.properties` file whose image path/file columns reference multiple
  source image aliases.

The first OpenHCS slice should therefore write the experiment/properties tables
as part of the conservative SQLite subset. They can be minimal, but they should
exist and validate against the properties file and table prefix.

The translocation fixture adds the combined-object case: one `Per_Object` table
with object-qualified columns for multiple object types. The advanced
segmentation fixture adds the per-object case: multiple `Per_<Object>` tables,
one `.properties` file per object table, and relationship type/edge tables.
Even if the rollout order starts with QC, the core projection model should be
shaped to cover all three modes so the first implementation does not encode a
single-fixture shortcut.

## Proposed Data Flow

1. Parse `.cppipe`.
2. Partition infrastructure modules as today.
3. Build `PreparedGeneratedPipeline`.
4. Execute generated OpenHCS pipeline.
5. If infrastructure modules contain enabled `ExportToDatabase`, run
   `CellProfilerPostExecutionExporter` and build `CellProfilerAnalystExportRequest` from:
   - the `ExportToDatabase` module block;
   - `PreparedGeneratedPipeline.source_schema`;
   - `PreparedGeneratedPipeline.generated_pipeline.artifact_contracts`;
   - compiled contexts and their `RuntimeValueStore` records;
   - output roots and source-schema workspace metadata;
   - a declared runtime-record selection policy.
6. `CellProfilerPostExecutionExporter` lowers the request to a core
   `TabularProject` and passes it to profile-driven renderers:
   - core SQLite project writer emits the database file from projected tables;
   - CellProfiler CPA properties renderer emits `.properties` file(s);
   - CellProfiler CPA workspace renderer emits `.workspace` when supported;
   - optional diagnostic manifest JSON is for OpenHCS validation only.
7. `validate_cppipe_execution(...)` checks the CPA export if `ExportToDatabase` is present.

## New Nominal Types

Nominal types should be introduced at the broadest truthful owner. The
CellProfiler module may define type aliases only when the alias names a real CPA
contract; it should not wrap core records just to make them sound
CellProfiler-specific.

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
    channel_color: str
    channels_per_image: int = 1
```

`CPAImageChannelSpec` is derived from source schema, NamesAndTypes/LoadData
semantics, and any `ExportToDatabase` channel-display settings. It is not
derived from materialized CSV files. Path/file column names are dialect-rendered
from `RuntimeProjectedColumnIdentity(SOURCE_IMAGE_PATH/SOURCE_IMAGE_FILE)`, not
stored on the channel spec.

The generic channel/source-image projection should be core-owned. The
CellProfiler edge may expose `CPAImageChannelSpec` as the CPA profile view over
that core projection, but source alias enumeration and source-image validation
must come from core/source-schema owners.

Image presentation settings need their own nominal policy at the CellProfiler
edge because they interpret `ExportToDatabase` UI settings:

```python
class CPAImagePresentationPolicy(ABC):
    @abstractmethod
    def channels(
        self,
        source_schema: PipelineImageSchema,
        export_module: ModuleBlock,
    ) -> tuple[CPAImageChannelSpec, ...]: ...
```

The first concrete policies should cover explicit CPA image blocks and the
default-all-source-images mode seen in native fixtures. Both policies must
validate that every rendered channel maps to one source-schema image assignment.

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
class ProjectedImageRow:
    image_number: int
    metadata: Mapping[str, object]
    image_path_columns: Mapping[str, str]
    image_file_columns: Mapping[str, str]
    measurements: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class ProjectedObjectTable:
    object_name: str
    rows: tuple[Mapping[str, object], ...]
    object_id_column: str

@dataclass(frozen=True, slots=True)
class ProjectedRelationshipType:
    relationship_type_id: int
    module_number: int | None
    relationship: str
    object_name1: str
    object_name2: str

@dataclass(frozen=True, slots=True)
class ProjectedRelationshipEdge:
    relationship_type_id: int
    image_number1: int
    object_number1: int
    image_number2: int
    object_number2: int

@dataclass(frozen=True, slots=True)
class ProjectedRelationshipGraph:
    relationship_types: tuple[ProjectedRelationshipType, ...]
    edges: tuple[ProjectedRelationshipEdge, ...]

@dataclass(frozen=True, slots=True)
class RuntimeTableRowProjection:
    ...

@dataclass(frozen=True, slots=True)
class TabularProject:
    image_rows: tuple[ProjectedImageRow, ...]
    object_tables: tuple[ProjectedObjectTable, ...]
    relationship_graph: ProjectedRelationshipGraph | None
    profile_metadata: ProjectProfileMetadata
```

These records should be core render projections. They do not own image, object,
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
cannot be an implicit "use the original path if it works" rule. The plane
export mechanism should be core-owned, with CellProfiler contributing a
compatibility profile that states what CPA can read.

```python
class ImagePlaneExportPolicy(ABC):
    @abstractmethod
    def plane_ref(self, request: ImagePlaneExportRequest) -> ImagePlaneRef: ...

@dataclass(frozen=True, slots=True)
class OriginalMonochromeFile(ImagePlaneExportPolicy):
    ...

@dataclass(frozen=True, slots=True)
class GeneratedMonochromeTiff(ImagePlaneExportPolicy):
    ...
```

Policy rules:

- `OriginalMonochromeFile` is valid only when the source candidate already
  resolves to exactly one profile-readable monochrome image plane.
- `GeneratedMonochromeTiff` writes an explicit monochrome TIFF plane from the
  existing OpenHCS/source-schema image payload and records that generated file.
- Multichannel, Bio-Formats-only, virtual-only, Z-stack, or time-series sources
  must be split into explicit image-plane refs or fail according to the export
  profile.
- The policy result is a file reference for project rendering, not a new source
  image identity.
- The CellProfiler profile should add only the CPA-readable constraints and CPA
  path/file column rendering; file resolution, generated plane writing, and
  source-image provenance stay in core.

## Implementation Phases

### Phase 1: Contract Extraction And Guardrails

Add a real `ExportToDatabase` infrastructure note that says CPA export is handled by the post-execution CPA exporter, not by `@special_outputs`.

Add settings parsing for the subset needed by SQLite CPA export. This remains
CellProfiler-specific because it interprets `.cppipe` module settings:

- database type;
- SQLite filename;
- experiment name;
- table prefix;
- object-table mode;
- selected objects;
- properties file enabled;
- relationship tables enabled;
- image/channel display settings where present.
- plate and well metadata setting names;
- classification type;
- object-location source;
- group/filter field declarations;
- workspace-file, workspace plot definitions, thumbnail, and URL settings as explicit supported or
  unsupported feature declarations.

Add fail-loud behavior:

- unsupported MySQL export should raise a clear unsupported-export error for OpenHCS CPA export;
- disabled properties output should still allow database writing, but validation must not claim CPA-loadable output unless `.properties` exists;
- unsupported object table modes should be explicit errors, not silent downgrade.
- image-channel settings that cannot be reconciled with the source schema should
  raise before writing any partial CPA project.
- settings that affect the CPA properties/database contract must not be silently
  ignored. Unsupported values should be represented by nominal unsupported
  feature errors so validation cannot claim a loadable CPA project.
- requested `.workspace` output must either be rendered or represented as a
  named unsupported feature. Do not claim native output parity if the workspace
  request is skipped.

### Phase 2: Runtime Store To CPA Tables

Create a core collector over all compiled contexts:

- Add a public current-record selection surface on `RuntimeValueStore` or a
  core `RuntimeRecordSelectionPolicy` that can select `CURRENT_BINDINGS` versus
  `OBSERVED_HISTORY`.
- Seed the image row universe from core `SourceImageSetRow` records derived
  from source-schema workspace metadata and the same `ImageNumber` ordering used
  by CellProfiler measurement projection.
- Collect `MEASUREMENTS` records selected by the runtime-record policy and
  group by `MeasurementSubject`.
- Collect `OBJECT_LABELS` records for object counts and object identity.
- Collect `RELATIONSHIPS` records for relationship tables.
- Collect image/source metadata and typed `SourceImagePlaneRef` values from
  source-schema workspace metadata without collapsing structured `SourcePixelRef`
  payloads to path-only strings.
- Build core source-channel/source-image projections from source schema.
- Build `CPAImageChannelSpec` values as the CellProfiler profile view over those
  core projections plus infrastructure settings.
- Resolve each image row's project path/file columns through core
  `ImagePlaneExportPolicy` plus a CellProfiler compatibility profile.
- Build image presentation through `CPAImagePresentationPolicy` and reconcile it
  with `PipelineImageSchema.assignments_by_alias`.

Rules:

- `ImageNumber` comes from the existing CellProfiler image-number projection machinery.
- Image-scope measurement rows go to `Per_Image`.
- Object-scope measurement rows go to the object table layout declared by the
  project profile: per-object tables for per-object mode, one combined object
  table for combined mode, or a declared unsupported feature for object views
  until view rendering is implemented.
- `ObjectNumber` must be present or derived from the declared object id field. If it cannot be derived, fail.
- Relationship rows use typed `ObjectRelationship` endpoints and must first
  coalesce into a core `ProjectedRelationshipGraph`; the CellProfiler profile
  renders that graph as relationship type and edge tables.
- Runtime records must be collected through a named final-state selection policy.
  Do not implicitly consume every `RuntimeValueStore.observed_values` entry as a
  database row.
- Source image-set row construction must use core source workspace authorities
  (`ImageSetRecord`, `SourceProjectionSet`, `VirtualWorkspaceSourceProjection`,
  and `openhcs_metadata.json`) rather than re-running CellProfiler parsing.
- When reconstructing rows after workspace materialization, enumerate planes
  from the metadata `image_files` list and group by canonical components. Do not
  use projection map key iteration as the row universe.
- Normalize workspace roots to absolute paths before projection to avoid
  relative/full virtual path duplication.
- Preserve non-axis source metadata fields in `SourceImageSetRow.metadata`; CPA
  properties and tables need fields such as `Plate`, `Series`, `Frame`,
  `FileLocation`, and channel metadata.
- Structured source refs must preserve plane-level fields such as `plane_index`,
  `c`, `z`, and `t` until the image-plane export profile decides whether to
  reference the original file or generate a monochrome plane.
- Project projection records are discarded after rendering and validation. They are
  never persisted as semantic state.
- Image-classification mode still needs CPA database/properties consistency.
  If CellProfiler would emit a minimal object table for the selected mode, the
  OpenHCS exporter should do the same rather than dropping `object_table`.
- External CPA properties files, `Experiment` rows, and
  `Experiment_Properties` rows are separate profile artifacts. Do not assume a
  properties file must always be duplicated into `Experiment_Properties`, and
  do not use an empty `Experiment_Properties` table as evidence that external
  `.properties` files are unnecessary.

Do not read OpenHCS materialized CSV files to build this model unless validating a compatibility fixture. The runtime store is the semantic authority.
The collector should not import CellProfiler modules; CellProfiler should call
it with a dialect/profile.

### Phase 3: SQLite Writer

Write a small core SQLite project writer that accepts projected tables and a
project profile. The CPA layer should specify the required tables, properties,
and naming dialect; it should not own generic SQLite row insertion.

Responsibilities:

- accept `RuntimeProjectedTable` / `TabularProject` records produced by the
  core projection layer;
- create `Per_Image`;
- create `Per_Object`, `Per_<ObjectName>` tables, or an explicit unsupported
  object-view artifact according to the lowered object table profile;
- create `Per_Experiment`;
- create `Experiment` / `Experiment_Properties` metadata tables;
- create profile-named relationship type and relationship edge tables when
  requested. For the CPA profile these are `Per_RelationshipTypes` and
  `Per_Relationships`, not one table per relationship kind;
- create a minimal object table when required by image-classification CPA
  properties, even when there are no object measurement rows;
- insert rows with stable column ordering;
- coerce values to SQLite-compatible types;
- preserve NULLs/NaNs in a CPA-compatible way;
- create indexes/primary keys for `ImageNumber` and object ids;
- write experiment/properties metadata sufficient for CPA.

This writer may use Python's `sqlite3`, but it is not an OpenHCS
materialization backend. It is a renderer for an external tabular project
contract and should write through the configured filesystem/output-root
boundary.
It should not inspect `ExportToDatabase` settings directly; those settings must
already have been lowered to a project profile and dialect.

Column naming needs a nominal policy:

```python
class RuntimeTableColumnNamePolicy:
    max_length: int
    def column_name(object_name: str, feature_name: str) -> str: ...
```

The policy must detect collisions after shortening and fail or disambiguate deterministically.
CellProfiler should provide the CPA shortening/profile rules; collision
detection and deterministic disambiguation belong in core.

### Phase 4: CPA Properties Writer

Generate `.properties` from the same core `TabularProject`, not from filesystem scanning.
This is one of the few intentionally CellProfiler-specific renderers.

Minimum properties:

- `db_type = sqlite`
- `db_sqlite_file = <absolute or project-relative sqlite path>`
- `image_table = <prefix>Per_Image`
- `object_table = <prefix>Per_Object` or selected object table
- `image_id = ImageNumber`
- `object_id = ObjectNumber` or `<ObjectName>_Number_Object_Number`
- `object_name`
- `plate_id`, `well_id`, `series_id`, `group_id`, `timepoint_id`
- `cell_x_loc`, `cell_y_loc`, `cell_z_loc` when available
- `image_path_cols`
- `image_file_cols`
- `image_names`
- `image_channel_colors`
- `channels_per_image`
- `classifier_ignore_columns`
- `classification_type`
- group SQL entries declared by `ExportToDatabase`;
- optional thumbnail/workspace/image-URL fields only when supported.

Properties file cardinality is part of the profile. Combined-object and
image-classification fixtures write one properties file pointing at
`Per_Object`; per-object fixtures write one file per `Per_<Object>` table. The
renderer should receive that decision from the lowered project profile rather
than deciding from `projection.object_tables or (None,)`.

Path/file columns should be generated from the source-schema virtual image set:

- each CP image alias/channel gets `Image_PathName_<alias>` and `Image_FileName_<alias>`;
- path/file values come from `SourceImageSetRow.images_by_alias`, the core
  image-plane export policy, and the CellProfiler compatibility profile;
- every path/file pair must point to exactly one CP-readable monochrome image plane;
- generated monochrome TIFFs are allowed when the original source is not already a valid CPA plane;
- originals are allowed only through the explicit `OriginalMonochromeFile` policy.

No implicit conversion fallback.

### Phase 4b: CPA Workspace Writer

If `ExportToDatabase` requests a CellProfiler Analyst workspace file, render it
from parsed workspace plot settings and the same profile/dialect used by the
database and properties renderers.

Rules:

- workspace plot definitions are CellProfiler-specific and belong at the
  interop edge;
- table and column names inside the workspace file must be rendered through the
  same CellProfiler database dialect as SQLite/properties;
- unsupported workspace display tools or measurement references should raise a
  named unsupported feature error;
- validation must distinguish "CPA loadable SQLite/properties bundle" from
  "native output parity including `.workspace`".

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
Add a core project-export validation model beside the current file-export
validation model:

```python
@dataclass(frozen=True, slots=True)
class RuntimeProjectExportExpectation:
    project_type: str
    required_files: tuple[ProjectFileRequirement, ...]
    required_tables: tuple[ProjectTableRequirement, ...]

@dataclass(frozen=True, slots=True)
class RuntimeProjectExportObservation:
    files: tuple[Path, ...]
    tables: tuple[RuntimeProjectedTable, ...]
    references: tuple[ProjectReference, ...]
```

CellProfiler validation should instantiate that model with the CPA profile. The
core validation should check files, tables, row counts, references, and
schema-level contracts; the CellProfiler profile should provide the CPA-specific
required table/property names and CP-readable image-plane predicate.

Add CPA export validation:

- `.sqlite` file exists when SQLite export is requested;
- `.properties` exists when CPA properties are requested;
- `.workspace` exists when workspace output is requested and supported;
- properties file references existing database file;
- database contains `Per_Image`;
- object tables match selected object mode;
- properties file cardinality matches selected object mode;
- `Per_Image.ImageNumber` row count matches source image sets;
- all object-table `ImageNumber` values are present in `Per_Image`;
- object rows have `ImageNumber` and object id columns;
- object ids are unique per `ImageNumber` within each object table;
- experiment/properties tables exist and agree with the CPA properties file;
- image-classification mode has a database/properties combination that CPA can
  load, including any required minimal object table;
- group SQL entries reference existing tables and columns;
- workspace plot table/column references resolve when a workspace file is
  emitted;
- object coordinate columns required by properties exist when object
  classification/viewing is configured;
- image path/file columns exist and resolve to files;
- image path/file columns are non-null for every `Per_Image` row;
- image path/file pairs resolve to CP-readable monochrome image planes;
- relationship type and edge tables exist when relationship export is requested;
- every relationship edge references a declared relationship type;
- relationship rows reference existing image/object ids.

Add a stricter optional validation gate:

- run a small CPA import/open smoke test if CellProfiler Analyst is installed;
- otherwise run schema-level validation only.

## Tests

### Unit Tests

- Parse `ExportToDatabase` settings from synthetic module blocks.
- Select current runtime records after `RuntimeValueStore.replace(...)` without
  including stale observation-history records.
- Build source image-set rows from source-schema workspace metadata.
- Verify source image-set reconstruction from the native QC selected source
  workspace produces two rows and five aliases per row.
- Preserve structured `SourcePixelRef` plane fields through image-plane export
  planning.
- Preserve non-axis metadata fields on `SourceImageSetRow`.
- Build `CPAImageChannelSpec` as the CellProfiler profile view over core
  source-channel projections and infrastructure settings.
- Build core `TabularProject` projections from hand-constructed runtime stores.
- Project image-scope measurements to `Per_Image`.
- Project object-scope measurements to per-object tables.
- Project object-scope measurements to a combined object table when requested.
- Render one properties file for combined/image-classification mode and one
  properties file per object table for per-object mode.
- Render CPA relationship graph as relationship type and edge tables.
- Reject object measurements without object id.
- Reject image columns that cannot resolve to exactly one CP-readable image plane.
- Select `OriginalMonochromeFile` only for valid original monochrome files.
- Select or require `GeneratedMonochromeTiff` for virtual/multichannel/non-CP-readable sources.
- Write SQLite schema and verify table/column names.
- Write `.properties` and verify required keys.
- Render or explicitly reject requested `.workspace` output.
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
