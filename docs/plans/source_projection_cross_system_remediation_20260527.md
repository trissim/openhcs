# Source Projection Cross-System Remediation Plan - 2026-05-27

> Current disposition (2026-08-22): BBBC021 and BBBC038 now use
> `SourceBindingsConfig`; their former dataset-specific handlers were removed.
> BBBC references below describe the historical inputs to this remediation.

## Purpose

This plan explains how the source projection overhaul fixes every leak found in
the cross-system audit without creating separate ad hoc repairs for each
handler, backend, runtime path, or UI consumer.

It is a companion to:

- `docs/plans/source_projection_overhaul_20260527.md`
- `docs/plans/source_projection_overhaul_code_audit_20260527.md`

The rule is simple: if a system claims a path represents an OpenHCS logical
image plane, that path must be a canonical OpenHCS virtual filename and must be
backed by a `SourcePlaneProjection`.

## Architectural Split

The overhaul creates three roles.

### Producers

Producers discover source data and emit `SourceProjectionSet`.

Initial producers:

- Bio-Formats projection, via `BioFormatsImageEntry`;
- source-schema materialization after candidate/image-set matching;
- CellProfiler `.cppipe` source-schema ingestion through setup-module
  compilation, before processing-module contract validation;
- existing file-per-plane microscope handlers through an adapter;
- OMERO virtual plate projection;
- compatibility adapters for legacy OpenHCS metadata.

Producer responsibilities:

- inspect source files enough to know whether they are whole-file single planes
  or structured multi-plane sources;
- render canonical virtual image filenames from `OpenHCSPlaneAddress`;
- write legacy `workspace_mapping`, component dictionaries, `image_files`, and
  `source_metadata` from the projection set;
- fail before metadata write if source axes cannot be mapped.

### Consumers

Consumers read projection identity and must stop reconstructing source truth
from paths.

Initial consumers:

- runtime preload and input conversion;
- `SourceWorkspaceAnchorProjection`;
- `VirtualWorkspaceSourceProjection`;
- `SourceBindingRuntimeContext`;
- CellProfiler current-source plane selection;
- streaming metadata construction;
- component discovery and pattern planning.

Consumer responsibilities:

- use canonical virtual filenames for OpenHCS axes;
- use `SourcePixelRef` for backend/source pixel loading;
- carry projection identity through payload metadata;
- never flatten structured refs to source path when plane identity matters.

### Compatibility Views

Compatibility views keep old metadata and APIs usable.

Allowed compatibility views:

- legacy `workspace_mapping`;
- path-only virtual workspace mappings for whole-file single-plane sources;
- legacy metadata reconstruction into best-effort projections;
- dataset-specific adapters that intentionally encode non-HCS samples.

Compatibility views are not allowed to become semantic authorities.

## Remediation Phases

### Phase A: Core Projection Contract

Add `openhcs.core.source_projection`.

Required types:

- `OpenHCSPlaneAddress`;
- `SourcePixelRef`;
- `SourcePlaneProjection`;
- `SourceProjectionSet`;
- serializer/deserializer for OpenHCS metadata;
- canonical virtual filename renderer;
- metadata/component validator.

Required validation:

- every projection address renders to one parser-readable filename;
- parsing the rendered filename returns the same address;
- `source_metadata` component values do not conflict with the address;
- whole-file path refs are allowed only when a source is known to be one
  monochrome plane;
- structured refs are required for internal C/Z/T/series/plane addressing.

This phase fixes the missing authority. It does not yet move every producer or
consumer.

### Phase B: Move Correct Producers First

Start with Bio-Formats and OMERO because they already model one logical virtual
plane as one backend-resolved source plane.

Bio-Formats:

- adapt `BioFormatsImageEntry` into `SourcePlaneProjection`;
- make `BioFormatsWorkspaceMetadataWriter` serialize through the core
  projection serializer;
- keep `BioFormatsStorageBackend` loading structured refs;
- preserve current Bio-Formats metadata and loading tests.

OMERO:

- add an OMERO projection adapter from plate structure to `SourceProjectionSet`;
- make generated filenames validate against the canonical filename renderer;
- tighten `OMEROFilenameParser` so missing site/channel/Z/timepoint is rejected
  for virtual OMERO paths;
- keep OMERO backend path parsing as a compatibility entry point for actual
  backend load calls.

This phase establishes two reference implementations for the rest of the repo.

### Phase C: Repair Source-Schema as a Projection Producer

Keep current source-schema candidate machinery:

- `SourceSchemaCandidate`;
- metadata extraction rules;
- alias filtering;
- image-set assembly;
- imported metadata joins.

Replace only the final materialization authority:

- `_primary_workspace_mappings()` becomes a projection builder plus serializer;
- virtual image filenames render from `OpenHCSPlaneAddress`;
- primary virtual image-plane paths use canonical `.tif`;
- source suffixes move to `SourcePixelRef` or source metadata;
- source files with hidden C/Z/T axes fail unless a source-layout policy maps
  them;
- reader-backed TIFF Z stacks expand to one virtual filename per OpenHCS Z
  plane.

Tests to rewrite:

- source-schema tests that expect `.TIF` source suffix preservation in virtual
  image paths;
- tests that assume `z_indexes={"1": None}` for a multi-plane source;
- tests that assert path-only `workspace_mapping` for sources requiring
  sub-file addressing.

This phase fixes the 3DNoiseNuclei class of bug at the metadata boundary.

### Phase D: Add Single-Plane Validation for File-Based Handlers

ImageXpress, Opera Phenix, and BBBC021 mostly operate on file-per-plane layouts.
They can remain source-filename-driven producers only if they validate the
file-per-plane assumption.

Required changes:

- add a shared `SinglePlaneSourceValidator`;
- run it when building virtual mappings from disk files;
- accept ordinary 2D monochrome TIFF/PNG paths;
- fail on local TIFF/OME-TIFF sources that expose multiple C/Z/T planes unless
  the handler provides an explicit projection policy;
- emit `SourceProjectionSet` through a file-per-plane adapter instead of
  hand-built `workspace_mapping` only.

Handler-specific notes:

- ImageXpress: safe when source files are one plane; validate before mapping.
- Opera Phenix: safe when source files are one plane; placeholder generation
  remains output/preparation behavior and should not become source projection.
- BBBC021: safe when TIFFs are one plane; validate before mapping.
- BBBC038: needs a compatibility classification because image id is treated as
  `well` and site/channel/Z/timepoint are not encoded in the filename.

### Phase E: Replace Runtime Path Flattening

Move runtime source identity from path maps to projection maps.

Targets:

- `SourceWorkspaceAnchorProjection`;
- `VirtualWorkspaceSourceProjection`;
- `SourceBindingRuntimeContext`;
- `SourceRuntimePathLookup`;
- `ImagePayloadSourcePathResolver`.

Required behavior:

- runtime projection caches prefer `source_projection`;
- legacy path maps are derived views;
- structured refs remain structured until backend load;
- source-binding contexts carry projection identity and canonical address;
- CellProfiler source-image-set identity can include structured source-plane
  coordinates.

Compatibility:

- whole-file single-plane virtual workspaces can still use path-only lookup;
- legacy metadata without `source_projection` can be reconstructed best effort;
- any reconstructed legacy projection that appears multi-plane must fail rather
  than guess.

### Phase F: Make Preload and Input Conversion Ref-Aware

Current preload and conversion load by paths first and attach metadata after
load. That must change.

Targets:

- `FunctionStepExecutor._convert_input_if_needed()`;
- `bulk_preload_step_images()`;
- `get_all_image_paths()`;
- `calculate_zarr_dimensions()`;
- `save_materialized_data()`;
- `update_metadata_for_zarr_conversion()`.

Required behavior:

- select work by canonical virtual filename/address;
- load pixels through `SourcePixelRef`;
- attach projection identity before saving to memory;
- calculate Zarr dimensions from projection addresses, not from raw or collapsed
  source paths;
- write converted metadata from the projection set.

This phase closes the largest remaining post-metadata semantic-loss path.

### Phase G: Update CellProfiler Identity and Current-Source Selection

CellProfiler already has useful downstream selection machinery:

- `CurrentSourcePlaneProjectionBase`;
- source image-set identities;
- payload metadata propagation.

Do not replace that logic first. Feed it better identity.

Required changes:

- put projection identity into `ImagePayloadMetadata`;
- make current-source plane selection prefer projection identity;
- keep component-metadata fallback only for legacy payloads;
- remove path-only identity fallbacks after compatibility tests pass.

This phase turns existing CellProfiler runtime projection into a consumer of the
new authority.

### Phase H: Streaming, Artifacts, and UI

Streaming and UI should use projection addresses for component metadata.

Targets:

- PolyStore streaming `component_metadata`;
- ROI/image artifact materialization;
- plate viewer metadata display;
- source-binding preview UI;
- analysis-result consolidation that needs parser semantics.

Required behavior:

- streaming gets explicit component metadata from projection addresses;
- artifact filenames can use canonical address prefixes, but artifact suffixes
  do not define image-plane semantics;
- UI displays projection-derived axes when available;
- source-binding previews distinguish source filename evidence from OpenHCS
  virtual address semantics.

## Compatibility Matrix

| System | Current state | Migration path |
| --- | --- | --- |
| Bio-Formats | Mostly correct structured refs | First producer migrated to core projection |
| OMERO | Mostly correct virtual plane generation | Add projection adapter and stricter parser validation |
| Source-schema | Major leak | Replace final materialization with projection builder |
| ImageXpress | Conditionally safe file-per-plane adapter | Add single-plane validation and projection serialization |
| Opera Phenix | Conditionally safe file-per-plane adapter | Add single-plane validation and projection serialization |
| BBBC021 | Conditionally safe file-per-plane adapter | Add single-plane validation and projection serialization |
| BBBC038 | Dataset-specific non-HCS filename semantics | Add compatibility mode or emit canonical virtual aliases |
| VirtualWorkspaceBackend | Path resolver only | Keep only as whole-file single-plane compatibility view |
| BioFormatsStorageBackend | Correct structured load backend | Keep as structured ref loader |
| Runtime preload/conversion | Leaky path-first loading | Convert to projection/ref-aware loading |
| CellProfiler current-source selection | Useful downstream consumer | Feed projection identity |
| Streaming | Accepts explicit metadata | Supply projection-derived metadata everywhere |

## Enforcement Gates

Do not merge implementation slices unless these gates pass for the touched
system:

- virtual image paths parse back to their projection addresses;
- component dictionaries are generated from projection addresses;
- source metadata cannot override address components;
- structured refs survive serialization, worker transport, preload, and input
  conversion;
- whole-file mappings are only used for validated single-plane sources;
- tests cover both a 2D source and a multi-plane source for the migrated path.

## Implementation Order

1. Core projection model and serializer tests.
2. Bio-Formats producer migration.
3. OMERO projection adapter and parser strictness.
4. Source-schema projection builder for current 2D behavior.
5. Source-schema fail-loud multi-plane detection.
6. Source-schema TIFF Z expansion policy.
7. File-per-plane handler adapters with single-plane validation.
8. Runtime projection deserialization and source-binding context migration.
9. Ref-aware preload and input conversion.
10. CellProfiler projection identity consumption.
11. Streaming/UI projection metadata consumption.
12. Remove or quarantine legacy path-only semantic fallbacks.

## Residual Risks

- Some historical metadata may not be reconstructable into exact projections.
  The migration should label those records as legacy/best-effort rather than
  silently promising exact semantics.
- File inspection can be expensive for very large plates. The single-plane
  validator should cache results and support conservative fail-loud sampling
  policies where full inspection is impractical.
- Dataset adapters such as BBBC038 may need an explicit non-HCS compatibility
  mode instead of being forced into plate-like axes.
- Zarr conversion will require careful parity tests because it currently derives
  dimensions from filenames and may expose hidden assumptions in downstream
  storage code.
