# Source Projection Overhaul Code Audit - 2026-05-27

> Current disposition (2026-08-22): BBBC021 and BBBC038 handlers referenced
> in this audit were removed. Their ordinary-file ingestion is now declared by
> `SourceBindingsConfig`. The remaining text preserves the pre-removal audit
> evidence.

## Scope

This audit checks the source projection overhaul plan against the current
OpenHCS codebase. It focuses on places where source semantics are represented,
converted, serialized, reconstructed, or consumed.

## Confirmed Current Authorities

### CellProfiler setup intent

`openhcs.interop.cellprofiler.source_schema.NamesAndTypesModuleCompiler`
compiles CellProfiler setup modules into `PipelineImageSchema`. It records
aliases, image types, selectors, and match plans. It does not project source
file internal C/Z/T axes.

Relevant files:

- `openhcs/interop/cellprofiler/source_schema.py`
- `openhcs/core/pipeline_image_schema.py`
- `openhcs/core/source_bindings.py`

Plan implication: keep this layer as source intent, but do not make it own
pixel-plane projection.

### Source-schema candidate and workspace materialization

`SourceSchemaCandidate` is file-level. It carries `path`, `relative_path`, and
metadata. It is appropriate for selector matching but not sufficient for
sub-file plane identity.

`materialize_source_schema_workspace()` currently:

- discovers source files;
- matches candidates by alias;
- assembles image sets;
- calls `_primary_workspace_mappings()`;
- writes OpenHCS metadata.

`_primary_workspace_mappings()` directly emits virtual filenames,
`workspace_mapping`, `source_metadata`, component dictionaries, and hardcoded
`Z_INDEX={"1": None}`.

Relevant file:

- `openhcs/core/source_schema_workspace.py`

Plan implication: `_primary_workspace_mappings()` is the main replacement
target. Its responsibilities should move to `SourceProjectionSet` construction
and serialization.

Filename-contract detail:

- `SourceSchemaFilenameParser` defines the normalized source-schema filename
  grammar: `well_sSITE_wCHANNEL_zZ_tT`.
- `SourceSchemaVirtualFilename` and `SourceSchemaFilenameProjection` already
  form a small virtual filename abstraction.
- `source_schema_metadata_with_virtual_components()` overlays parsed virtual
  filename components into source metadata.
- `_primary_workspace_mappings()` bypasses the missing projection authority by
  constructing one virtual path per assignment with `z_index=1` and
  `extension=candidate.path.suffix`.

Plan implication: the target abstraction should keep the formatter/parsing
idea, but make it stricter. Source-schema virtual image filenames must be
rendered from `OpenHCSPlaneAddress`, parse back to the same address, and use a
canonical image-plane extension. Source file extensions and source internal axes
belong to the source ref, not to the OpenHCS virtual address.

### OpenHCS metadata

`OpenHCSMetadata.workspace_mapping` is typed as `Optional[Dict[str, Any]]`.
Current mapping values can be plain path strings or structured backend refs.

Relevant file:

- `openhcs/microscopes/openhcs.py`

Plan implication: metadata already permits structured refs, but the structure is
not modeled as an OpenHCS core contract. Add `source_projection` and generate
legacy `workspace_mapping` from typed refs.

Metadata-contract detail:

- `image_files` and component dictionaries are expected to describe OpenHCS
  logical images.
- `workspace_mapping` is a backing map from those logical paths to source data.
- `source_metadata` currently mixes source-derived metadata with components
  overlaid from the virtual filename.

Plan implication: metadata writing must enforce that `image_files`, component
dictionaries, `source_metadata` component values, and `workspace_mapping` keys
all agree with the same canonical virtual filename parse. A structured
`workspace_mapping` value may point at a source plane, but it may not redefine
the OpenHCS logical axes.

### Bio-Formats projection

Bio-Formats already has an almost-correct per-plane projection model:

- `BioFormatsImageEntry`;
- `BioFormatsDataset`;
- `BioFormatsSPWProjector`;
- `BioFormatsLayoutProjector`;
- `BioFormatsWorkspaceMetadataWriter`;
- `BioFormatsStorageBackend`.

These preserve `source_path`, `series_index`, `plane_index`, OpenHCS
well/site/channel/z/timepoint, and source C/Z/T.

Relevant files:

- `openhcs/microscopes/bioformats_spw_projector.py`
- `openhcs/microscopes/bioformats.py`
- `openhcs/microscopes/bioformats_adapter.py`
- `external/PolyStore/src/polystore/bioformats_storage.py`

Plan implication: Bio-Formats should be adapted first because it gives a
known-good projection behavior to preserve.

Filename-contract detail: `BioFormatsWorkspaceMetadataWriter.virtual_path()`
already renders virtual paths with `extension=".tif"` from entry
well/site/channel/Z/timepoint, while the source path and plane coordinates are
kept in the structured ref payload. This matches the tightened target contract
more closely than source-schema materialization does.

### Runtime source projection

`VirtualWorkspaceSourceProjection` reconstructs source paths and metadata from
OpenHCS metadata. It is path-oriented:

- `source_paths_by_virtual_path`;
- `source_metadata_by_path`;
- `workspace_root`.

Structured refs are reduced to physical `source_path` through
`workspace_mapping_source_path()`.

Relevant files:

- `openhcs/core/steps/function_runtime.py`
- `openhcs/core/steps/function_execution.py`
- `openhcs/microscopes/openhcs.py`
- `openhcs/core/source_bindings.py`
- `openhcs/core/runtime_values.py`

Plan implication: this should become a legacy adapter. Runtime should prefer a
projection set that preserves structured ref identity.

Second-pass detail:

- `SourceBindingRuntimeContext` stores `step_input_source_paths` and
  `source_metadata_by_path`, both keyed by path strings.
- `SourceRuntimePathLookup` creates lookup keys from a filesystem path and
  optional input directory.
- `ImagePayloadSourcePathResolver` can resolve `VirtualWorkspaceBackend`
  redirects, but it has no structured-ref path for Bio-Formats or future
  source projections.

Plan implication: these helpers need a projection-aware compatibility layer.
Leaving them path-only would reintroduce the same source identity collapse after
metadata deserialization.

### Worker backend registration

`ProcessingContext.__setstate__()` recreates worker-local storage backends for
virtual workspaces and Bio-Formats based on boolean flags and `plate_path`.
This is necessary for multiprocessing, but it means projection-aware source refs
must be backed by worker-available backend instances and connection params.

Relevant file:

- `openhcs/core/context/processing_context.py`

Plan implication: runtime projection deserialization and ref-aware loading must
verify that workers can recreate every backend named by `SourcePixelRef.backend`.
The core projection model should not rely on main-process-only backend state.

### Runtime payload metadata

`ImagePayloadMetadata` carries source path and component metadata through
runtime payloads. It is useful for measurements and source identity after load,
but it cannot prevent incorrect loading if the source was already collapsed or
the wrong plane was selected.

Relevant files:

- `openhcs/core/runtime_values.py`
- `openhcs/core/source_image_semantics.py`
- `openhcs/interop/cellprofiler/runtime/adapter.py`
- `openhcs/core/aligned_image_payload.py`
- `openhcs/processing/materialization/core.py`

Plan implication: keep payload metadata, but make it the runtime carrier of
projection identity, not the original projection authority.

Second-pass detail:

- `aligned_image_payload.py` derives source-spatial domains from
  `ImagePayloadMetadata`.
- materialization writers normalize payloads into `MaterializationInputItem`
  values that preserve `ImagePayloadMetadata`.
- `CurrentSourcePlaneProjectionBase` and its subclasses select stack planes from
  current-source identity carried in payload metadata.

Plan implication: payload metadata remains a load-bearing semantic carrier. The
overhaul should extend it with projection identity rather than bypassing it.
CellProfiler current-source projection should consume the new identity instead
of reconstructing identity from paths and component metadata.

### Input conversion and preload

Input conversion and preload use paths and backend names:

- `FunctionStepExecutor._convert_input_if_needed()`;
- `bulk_preload_step_images()`;
- `get_all_image_paths()`;
- `save_materialized_data()`.

Relevant files:

- `openhcs/core/steps/function_execution.py`
- `openhcs/core/steps/function_io.py`
- `openhcs/core/pipeline/compiler.py`
- `openhcs/core/pipeline/path_planner.py`

Plan implication: this is a second-stage migration target. It needs ref-aware
loading after projection serialization exists.

Second-pass detail:

- `PipelineCompiler._configure_input_conversion_if_needed()` chooses conversion
  from `available_backends` and whether the original subdirectory uses
  `virtual_workspace`.
- `PipelineCompiler` validates `read_backend` against
  `microscope_handler.get_available_backends(...)`.
- `SourceWorkspaceAnchorProjection` in the execution path is another
  path/metadata bridge used before step execution.

Plan implication: `available_backends` must become projection-derived, and
input conversion plans need to carry enough projection context to load the
declared source plane in the first materialization step.

### Component discovery and filename parsing

The orchestrator uses metadata cache first, then falls back to parsing listed
filenames when discovering component keys. This fallback is valid only when the
listed filenames are canonical OpenHCS virtual filenames or materialized output
filenames.

Relevant files:

- `openhcs/core/orchestrator/orchestrator.py`
- `openhcs/microscopes/openhcs.py`

Plan implication: after the overhaul, source-schema and Bio-Formats workspaces
must expose canonical virtual filenames to `list_files()`. Component discovery
must not parse raw source filenames as if they were OpenHCS addresses.

## Confirmed Failure Mode

For the cached 3DNoiseNuclei dataset:

- raw source file `nuclei1_out_c00_dr90_image.tif` reads as `(100, 258, 258)`
  with axes `ZYX`;
- source-schema OpenHCS metadata has `z_indexes={"1": None}`;
- source-schema `workspace_mapping` maps selected c00 TIFFs to single `z001`
  virtual paths;
- converted OpenHCS output `C00_s001_w1_z001_t001.tif` reads as `(258, 258)`
  with axes `YX`.

This confirms the plan must address pre-runtime materialization, not only
viewer metadata.

## Plan Corrections From Audit

### 1. Bio-Formats is not just an inspiration

The plan should treat Bio-Formats as the initial implementation substrate.
`BioFormatsImageEntry` is close enough that the core model should be designed
to accept all of its fields without loss.

### 2. Source-schema projection cannot replace candidates

`SourceSchemaCandidate` is still needed for selector matching and imported
metadata joins. The new projection layer starts after alias/image-set matching,
not before it.

### 3. Runtime flattening must be fixed explicitly

`workspace_mapping_source_path()` currently extracts only the source path from a
structured ref. Any source identity path that passes through it loses
`series_index`, `plane_index`, and source C/Z/T. The plan must include runtime
projection identity, not just metadata serialization.

### 4. Input conversion is a semantic boundary

`_convert_input_if_needed()` loads paths from the step plan and materializes
them. If this remains path-only, a projection-aware workspace can still lose
semantics during the first conversion to Zarr. The plan must include ref-aware
conversion.

### 5. Metadata generator remains output-state authority

`OpenHCSMetadataGenerator` intentionally derives output metadata from actual
written filenames. That is correct for produced outputs. The new source
projection authority should own source workspace metadata, not override all
post-processing output metadata behavior.

### 6. Current-source projection already exists, but at the wrong layer

`CurrentSourcePlaneProjectionBase` in the CellProfiler runtime adapter already
selects a current source plane from stack-like runtime payloads. It operates
after loading, using `ImagePayloadMetadata` and source image-set identities.
That is useful runtime behavior, but it cannot be the source workspace
projection authority because incorrect loading may already have happened.

The plan should integrate it by feeding projection-derived identities into
payload metadata, then letting the existing current-source selection logic
continue to work against richer identities.

### 7. Backend availability is part of projection semantics

Current compile-time backend validation relies on
`microscope_handler.get_available_backends(...)`, while source-schema
materialization writes backend lists directly into metadata. Once source refs can
mix whole-file, virtual-workspace, and structured Bio-Formats references,
backend availability must be derived from the projection set and checked against
worker recreatability.

### 8. Virtual filenames are the semantic contract

The plan should explicitly distinguish two parsing roles:

- source filename parsing: evidence for matching, metadata extraction, and
  source-layout policy;
- OpenHCS virtual filename parsing: canonical semantic address for one logical
  image plane.

The second role is mandatory. A source projection is invalid if its virtual
filename does not parse, parses to different well/site/channel/Z/timepoint
values, inherits a source extension for a primary image plane, or points to a
multi-plane source without a structured ref/policy that names the exact source
plane.

### 9. VirtualWorkspace remains a path resolver, not a plane reader

`VirtualWorkspaceBackend` maps one virtual path to one real path string and
delegates loading to disk. It cannot express `series_index`, `plane_index`, or
source C/Z/T. That is acceptable only for whole-file single-plane payloads.

Plan implication: source projections that require sub-file addressing must not
be serialized as plain `virtual_workspace` string mappings. They need structured
refs and a backend that can honor them.

## Cross-System Leak Audit

The filename-contract audit exposes related leaks beyond source-schema
materialization. They fall into three groups.

### Actual leaks

Source-schema tests already encode the leak as expected behavior. Several tests
expect virtual paths such as `A01_s001_w1_z001_t001.TIF`, preserving the source
suffix instead of canonical image-plane `.tif`. Those tests should be rewritten
when the projection serializer lands.

Relevant file:

- `tests/unit/test_source_schema_workspace.py`

`SourceWorkspaceAnchorProjection` flattens structured `workspace_mapping` values
to `source_path`. Bio-Formats tests currently assert that flattening. That is
fine for path-only anchor matching, but it is not a valid source-plane identity
once structured refs matter.

Relevant files:

- `openhcs/core/steps/function_execution.py`
- `tests/unit/test_bioformats_microscope_handler.py`

Input conversion and preload still load by parsed/listed paths, then attach
payload metadata after loading. This can preserve the wrong semantics if the
path layer already collapsed a source plane.

Relevant files:

- `openhcs/core/steps/function_execution.py`
- `openhcs/core/steps/function_io.py`

`ImagePayloadSourcePathResolver` can resolve a `VirtualWorkspaceBackend` path
redirect, but not a structured Bio-Formats source ref. Payload metadata can
therefore lose source-plane identity even when the backend loaded the correct
plane.

Relevant file:

- `openhcs/core/runtime_values.py`

`BBBC038FilenameParser` treats the image id as `well` and returns default
site/channel/Z/timepoint values that are not encoded in the filename. That is
workable as a dataset-specific adapter, but it violates the tightened virtual
filename contract if those filenames are treated as canonical OpenHCS image
plane addresses.

Relevant file:

- `openhcs/microscopes/bbbc.py`

### Dependent-on-upstream-guarantees

`OpenHCSMetadataGenerator` parses actual output filenames without defensive
checks. That is correct for processed outputs only if all writers have already
emitted canonical OpenHCS filenames.

Relevant file:

- `openhcs/microscopes/openhcs.py`

Zarr materialization dimensions are calculated from parsed filenames. If the
input conversion paths are not canonical virtual plane paths, Zarr dimensions
can reflect collapsed or invented axes.

Relevant file:

- `openhcs/core/steps/function_io.py`

Generic virtual-workspace preparation in `MicroscopeHandlerBase.post_workspace`
lists files through the chosen backend, parses names, and reconstructs names.
For source workspaces, this is safe only if the backend exposes canonical
virtual filenames. It should not be the layer that repairs source semantics.

Relevant file:

- `openhcs/microscopes/microscope_base.py`

ImageXpress, Opera Phenix, and BBBC021 virtual mapping builders parse source
filenames and reconstruct virtual filenames. These are mostly safe for
file-per-plane source formats, but they do not validate that a source file is
actually a single plane. If a multi-plane file appears, they have the same class
of risk as source-schema.

Relevant files:

- `openhcs/microscopes/imagexpress.py`
- `openhcs/microscopes/opera_phenix.py`
- `openhcs/microscopes/bbbc.py`

### Mostly safe or already aligned

Bio-Formats is the best-aligned subsystem: it renders canonical `.tif` virtual
paths from known OpenHCS axes and stores source-plane coordinates in structured
refs. The leaks are downstream consumers that flatten those refs, not the
Bio-Formats projection writer or storage backend.

Relevant files:

- `openhcs/microscopes/bioformats.py`
- `external/PolyStore/src/polystore/bioformats_storage.py`

OMERO is also structurally aligned because the backend generates virtual
filenames from OMERO plate dimensions and then uses those filenames to load one
remote plane. The remaining weakness is that `OMEROFilenameParser` delegates to
the permissive ImageXpress parser even though comments say all components should
be present.

Relevant files:

- `openhcs/microscopes/omero.py`
- `external/PolyStore/src/polystore/omero_local.py`

Streaming is moving in the right direction: PolyStore streaming accepts explicit
component metadata and fails if neither metadata nor a parser-readable filename
is available. The remaining architectural requirement is to supply streaming
metadata from projection addresses instead of source path or artifact filename
fallbacks.

Relevant files:

- `external/PolyStore/src/polystore/streaming/_streaming_backend.py`
- `tests/unit/test_function_artifact_materialization.py`

## Additional Plan Implications

- Add a projection-level validator for source files that are claimed to be
  whole-file single planes. The validator can be conservative and fail if a
  local TIFF/OME-TIFF exposes multiple C/Z/T planes without an explicit policy.
- Treat Bio-Formats and OMERO as reference implementations for the canonical
  virtual filename contract.
- Convert `SourceWorkspaceAnchorProjection`, `VirtualWorkspaceSourceProjection`,
  and runtime source-binding contexts to carry projection identity, not only
  virtual path -> source path.
- Make tests assert canonical virtual filenames and structured source refs
  explicitly; remove source suffix preservation from expected virtual image
  paths.
- Add a compatibility exception or migration path for dataset adapters such as
  BBBC038 that currently encode an arbitrary sample id as `well` without
  filename-visible site/channel/Z/timepoint fields.

## Required New Tests

Add tests before or during implementation:

- `SourceProjectionSet` serializes one structured ref into legacy
  `workspace_mapping` and component dictionaries.
- `SourcePlaneProjection.address` renders a canonical virtual image filename
  and parsing that filename returns the same address.
- Primary source-schema image virtual filenames use `.tif` and do not preserve
  source image suffixes as semantic output.
- Metadata writing rejects component dictionaries or `source_metadata` whose
  component values disagree with the parsed virtual filename.
- Bio-Formats writer parity through `SourceProjectionSet`.
- Source-schema 2D materialization parity through `SourceProjectionSet`.
- Source-schema multi-plane TIFF fails loudly before Z expansion is implemented.
- Source-schema multi-plane TIFF expands to multiple `z` addresses once a
  reader-backed projection policy is implemented.
- Runtime projection deserialization preserves `plane_index`.
- Input conversion of a structured projection loads the declared plane, not the
  whole source file or first plane.
- Existing `OpenHCSMetadataGenerator` output-file-derived metadata behavior is
  unchanged for processed outputs.
- Component discovery sees only canonical OpenHCS virtual/materialized
  filenames for source-projected workspaces.

## High-Risk Code Areas

- `openhcs/core/source_schema_workspace.py`: broad responsibility and many
  tests depend on exact legacy filenames.
- `openhcs/microscopes/source_schema.py`: parser is the canonical grammar for
  source-schema virtual paths and needs round-trip enforcement.
- `openhcs/microscopes/openhcs.py`: metadata generation assumes filenames are
  properly formed and parses them without defensive checks.
- `openhcs/core/steps/function_runtime.py`: source-binding context and runtime
  source universes are path-heavy.
- `openhcs/interop/cellprofiler/runtime/adapter.py`: source identity logic has
  many fallbacks and caches.
- `openhcs/core/steps/function_io.py`: preload/conversion path APIs assume
  concrete file paths.
- `external/PolyStore/src/polystore/virtual_workspace.py`: supports only
  string path redirection.
- `external/PolyStore/src/polystore/bioformats_storage.py`: structured loading
  exists but is backend-specific.

## Recommended Execution Order

1. Add core projection dataclasses and serialization tests.
2. Add canonical virtual filename rendering, parser round-trip validation, and
   component conflict tests.
3. Make Bio-Formats writer serialize through the core projection model.
4. Add source-schema projection builder for current 2D behavior using
   canonical virtual image filenames.
5. Switch `materialize_source_schema_workspace()` to serialize from projection
   sets for primary image mappings.
6. Add fail-loud multi-plane TIFF detection for source-schema paths.
7. Add reader-backed projection expansion for TIFF Z stacks.
8. Add runtime projection deserialization and prefer it over path-only virtual
   workspace projection.
9. Make input conversion/preload ref-aware.
10. Collapse or demote legacy projection helpers after parity tests pass.

## Audit Status

The plan is consistent with the current source layout after tightening the
filename contract. The biggest adjustments from the second code pass are:

- OpenHCS virtual image filenames must remain canonical semantic addresses;
- source filename parsing is evidence, not the final address authority;
- source-schema materialization currently leaks by hardcoding `z001` and using
  the source suffix in virtual paths;
- runtime path flattening and input conversion must be planned as explicit
  phases, otherwise a typed metadata model would still allow semantic loss
  after metadata is written.
