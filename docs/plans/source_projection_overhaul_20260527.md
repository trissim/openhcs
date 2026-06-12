# Source Projection Architecture Overhaul - 2026-05-27

## Goal

Replace the current path-string source workspace model with a typed source
projection authority that preserves source semantics before OpenHCS metadata,
runtime loading, CellProfiler source binding, and viewer streaming consume the
data.

The non-negotiable contract is that OpenHCS virtual image filenames are
canonical semantic addresses. They encode OpenHCS well, site, channel, Z, and
timepoint for one logical image plane. Source filenames may be used as evidence
for matching and metadata extraction, but they are not allowed to be the final
semantic authority.

This plan is motivated by the 3DNoiseNuclei failure mode: raw TIFF files are
100-plane `ZYX` stacks, but source-schema materialization emits only `z001`
virtual files, and the converted OpenHCS workspace contains a single `YX`
plane. The bug is architectural because source semantics are split across
several representations that can silently disagree.

## Current Problem

OpenHCS currently has at least three separate authorities for source semantics:

- `PipelineImageSchema` records CellProfiler setup intent: aliases, filters,
  image types, image-set matching, and explicit `ImagePlaneSource` URI fields.
- OpenHCS metadata records virtual filenames, component dictionaries,
  `workspace_mapping`, `source_metadata`, and `available_backends`.
- Runtime and backend code reconstruct source identity from virtual paths,
  physical paths, parser metadata, payload metadata, and backend-specific
  structured refs.

Those authorities are coordinated by dictionary keys and mixed filename
parsing. That allows semantic loss when a source file has internal C/Z/T axes
that are not encoded in the source filename or are not projected into canonical
OpenHCS virtual filenames.

The specific leak is not filename parsing itself. The leak is treating a source
filename or a path mapping as if it were already an OpenHCS plane address. In
the target design, source filenames contribute evidence; OpenHCS virtual
filenames carry the logical axes.

## Target Architecture

Introduce a core source projection model, owned outside CellProfiler and outside
any specific microscope handler:

```python
@dataclass(frozen=True, slots=True)
class SourcePixelRef:
    backend: str
    source_path: str
    reader: str | None = None
    series_index: int | None = None
    plane_index: int | None = None
    source_channel: int | None = None
    source_z_index: int | None = None
    source_timepoint: int | None = None


@dataclass(frozen=True, slots=True)
class OpenHCSPlaneAddress:
    well: str
    site: str
    channel: str
    z_index: str
    timepoint: str


@dataclass(frozen=True, slots=True)
class SourcePlaneProjection:
    address: OpenHCSPlaneAddress
    ref: SourcePixelRef
    source_alias: str | None
    image_type: str | None
    source_metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SourceProjectionSet:
    projections: tuple[SourcePlaneProjection, ...]
```

The exact names can change, but the ownership must not: OpenHCS logical plane
identity and the concrete source pixel reference need to live in one value.

Everything currently hand-built from source workspaces should derive from this
projection set:

- normalized OpenHCS virtual filenames;
- `image_files`;
- `workspace_mapping`;
- `source_metadata`;
- `channels`, `wells`, `sites`, `z_indexes`, `timepoints`;
- `available_backends`;
- runtime source-binding source universe;
- runtime source identity and source-plane identity;
- streaming/display component metadata.

Input dialects such as CellProfiler `.cppipe` setup modules must be projection
producers, not GUI-side preprocessing steps. A `.cppipe` `NamesAndTypes` block
is source metadata that should normalize into the same native OpenHCS projection
model as Bio-Formats, OMERO, or existing OpenHCS metadata. Full CellProfiler
processing pipeline import remains compiler work and must not be required
before a projection set can initialize a native OpenHCS workspace.

## Canonical Virtual Filename Contract

For source image workspaces, every virtual image path emitted by OpenHCS must be
a canonical OpenHCS plane address:

- it is parser-readable by the workspace `source_filename_parser_name`;
- it contains well, site, channel, Z, and timepoint;
- its parsed components exactly equal `SourcePlaneProjection.address`;
- for materialized image planes, the virtual image format is a monochrome plane
  TIFF, normally `.tif`;
- source file extension, source filename, source series, source plane index,
  and source C/Z/T coordinates live in `SourcePixelRef`, not in the canonical
  address;
- non-image artifacts may use the same canonical address prefix plus an
  artifact suffix, but they do not define image-plane semantics.

The projection builder must validate round trip:

```text
SourcePlaneProjection.address
  -> canonical virtual filename
  -> parser.parse_filename(...)
  -> same address
```

`workspace_mapping` is only a backing pointer from that canonical address to a
source pixel reference. It cannot add, erase, or reinterpret OpenHCS axes.

## Non-Goals

- Do not rewrite CellProfiler runtime measurement logic in the first slice.
- Do not delete legacy `workspace_mapping` immediately; it remains the
  compatibility serialization.
- Do not force all image sources through Bio-Formats if a simpler exact reader
  can produce the same `SourcePixelRef`.
- Do not treat every 3D array as Z. Internal axes must come from reader metadata
  or a declared source-layout policy, not from shape guessing alone.
- Do not preserve source image extensions in canonical virtual image-plane
  filenames as a semantic signal. The source extension belongs to the source
  ref.

## Required Invariants

The new projection authority must validate these before metadata is written:

- Every emitted virtual filename has exactly one `SourcePlaneProjection`.
- Every `SourcePlaneProjection.address` serializes to one parser-readable
  OpenHCS virtual filename.
- Parsing that virtual filename returns exactly the projection address.
- Component dictionaries are derived from projection addresses, never manually
  constructed.
- `source_metadata` component values cannot conflict with the projection
  address.
- `available_backends` is derived from projection ref types.
- A string-only `virtual_workspace` mapping is allowed only for whole-file
  payloads whose source semantics do not require sub-file plane addressing.
- A multi-plane source that is projected as separate OpenHCS planes must use a
  structured ref with `plane_index` or equivalent reader coordinates.
- If a source file has internal axes and no source-layout policy can map them,
  materialization fails before runtime.
- Runtime cannot flatten a structured ref to only `source_path` when source
  identity requires `series_index`, `plane_index`, or source C/Z/T.
- Primary image-plane virtual filenames do not inherit source extensions or
  hidden source axes. They are generated from the OpenHCS plane address.
- A source-schema image assignment that resolves to a multichannel, multi-Z, or
  multi-T source payload must either emit one canonical virtual image plane per
  OpenHCS address or fail loudly.

## Refactor Phases

### Phase 1: Projection Model and Metadata Serialization

Add a core module, likely `openhcs.core.source_projection`.

Responsibilities:

- define the projection dataclasses;
- validate projection set uniqueness and component consistency;
- render canonical virtual image filenames from `OpenHCSPlaneAddress`;
- reject virtual filenames whose parsed components disagree with their
  projection address;
- serialize a projection set to OpenHCS metadata-compatible dictionaries;
- deserialize existing OpenHCS metadata into a projection set when possible;
- provide stable source identity and source-plane identity keys.

Compatibility:

- Keep writing legacy `workspace_mapping`.
- Add an optional `source_projection` metadata section with typed refs.
- For legacy metadata without `source_projection`, reconstruct a best-effort
  projection set from current `workspace_mapping` and `source_metadata`.

### Phase 2: Move Bio-Formats Onto the Core Projection Model

`BioFormatsImageEntry` already has the shape of a source-plane projection. Move
or adapt it so Bio-Formats becomes a producer of `SourceProjectionSet` rather
than the owner of a parallel model.

Expected changes:

- `BioFormatsSPWProjector` and `BioFormatsLayoutProjector` produce core
  projections or are wrapped by an adapter that does.
- `BioFormatsWorkspaceMetadataWriter` serializes through the core projection
  serializer.
- `BioFormatsStorageBackend` continues to load structured refs, but its ref
  payload should be generated from `SourcePixelRef`.

This phase should be behavior-preserving for current Bio-Formats tests.

### Phase 3: Split Source-Schema Candidate Matching From Pixel Projection

Keep `SourceSchemaCandidate` as a file-level candidate used for selector and
image-set matching. It should not be the final source identity.

After image sets are assembled, add a projection builder:

- file candidate plus assignment plus image-set metadata -> one or more
  `SourcePlaneProjection` values;
- ordinary 2D image -> one projection with whole-file or single-plane ref;
- reader-known multi-plane image -> one projection per declared OpenHCS plane;
- unsupported internal axes -> fail loudly.

This replaces `_primary_workspace_mappings()` as the source of virtual
filenames, mappings, source metadata, and component values.

Target behavior for source-schema image planes:

- render virtual image filenames from `OpenHCSPlaneAddress`, not from the source
  filename suffix;
- use `.tif` for canonical virtual image-plane paths;
- put the original source path and extension only in `SourcePixelRef`;
- keep CellProfiler source filename filters and metadata extraction in the
  candidate-matching layer;
- if source metadata proposes a component value that differs from the canonical
  virtual filename, reject the projection before writing metadata.

### Phase 4: Runtime Consumes SourceProjectionSet

Replace path-only `VirtualWorkspaceSourceProjection` as the primary runtime
source-binding authority.

Runtime should ask the projection set for:

- source universe for an axis;
- source metadata for a virtual path;
- physical/structured source ref for loading;
- source image-set identity;
- source plane identity.

Legacy path projection can stay as a compatibility adapter, but it should not
be the data model used by new source-schema and Bio-Formats workspaces.

This phase must also migrate the existing runtime helper types that currently
encode source identity as paths:

- `SourceBindingRuntimeContext`;
- `SourceRuntimePathLookup`;
- `ImagePayloadSourcePathResolver`;
- `SourceWorkspaceAnchorProjection`;
- worker-side backend recreation in `ProcessingContext`.

These can keep path compatibility fields, but the authoritative lookup should
be by projection identity or virtual address, not by resolved source path.

### Phase 5: Ref-Aware Input Conversion and Loading

Input conversion currently asks for paths and loads them through a backend. That
is not enough for structured source refs.

Update preload and conversion paths so they can consume projection refs:

- load `SourcePixelRef` via the correct backend;
- attach source metadata before saving to memory;
- preserve `SourcePlaneProjection` identity in payload metadata;
- when materializing to Zarr/disk, write metadata derived from projections, not
  only from output filenames.

### Phase 6: CellProfiler Runtime Identity Cleanup

CellProfiler runtime currently reconstructs source identity from paths,
payload metadata, and parsed metadata. Preserve the existing behavior during
migration, then switch identity resolution to use projection identities when
available.

Expected reductions:

- less reliance on `source_path_identity_key`;
- less reverse lookup from resolved physical path to virtual path;
- fewer fallbacks from context metadata to parser metadata;
- source-plane identity can include structured ref coordinates, not just
  component metadata or source path.

This must account for the existing `CurrentSourcePlaneProjectionBase` hierarchy
in the CellProfiler runtime adapter. That code already owns current-source
plane selection for stack-like runtime payloads. It should become a consumer of
projection-derived identities, not a competing projection model.

### Phase 7: UI and Streaming Alignment

Viewer and streaming metadata should consume the same projection addresses used
by runtime.

Expected changes:

- ROI/image streaming gets explicit component metadata from projection
  addresses;
- Fiji/Napari display config can classify channel/slice/frame dimensions from
  projection address fields;
- source review of a structured source workspace can show Z/T/C axes without
  relying on filenames alone.

## First Implementation Milestones

1. Add `SourceProjectionSet` and serializer tests with synthetic projections.
2. Adapt Bio-Formats writer to serialize through the new model without changing
   observed metadata.
3. Add virtual filename round-trip tests that prove projection addresses render
   canonical `.tif` image-plane filenames and reject conflicting components.
4. Add a source-schema characterization test for a multi-plane TIFF and assert
   that current behavior fails loudly instead of silently collapsing Z.
5. Add source-schema projection builder for 2D files while preserving logical
   behavior through canonical virtual filenames.
6. Add source-schema projection builder for reader-known TIFF Z stacks.
7. Switch runtime projection cache to prefer `source_projection`.
8. Remove direct `_primary_workspace_mappings()` metadata construction once the
   projection builder owns all equivalent outputs.

## Test Strategy

Unit tests:

- projection set uniqueness and address serialization;
- canonical virtual filename rendering and parse round-trip;
- canonical image-plane filenames use OpenHCS plane extensions, not source
  extensions;
- structured and whole-file ref serialization;
- component dictionary derivation;
- source metadata conflict detection;
- legacy metadata reconstruction;
- source-schema 2D parity;
- source-schema multi-plane TIFF fail-loud behavior, then Z expansion behavior;
- Bio-Formats metadata parity through the new serializer;
- source-binding runtime context preserves projection identities through
  serialization;
- `ImagePayloadSourcePathResolver` resolves or rejects structured refs without
  silently falling back to whole-file paths.

Integration tests:

- Bio-Formats runtime preload still loads the expected plane;
- CellProfiler source-schema workspace with 3D TIFF exposes Z axes;
- input conversion to Zarr preserves logical Z count;
- ROI/image streaming receives component metadata from projection address.

Regression checks for 3DNoiseNuclei:

- raw c00 TIFF shape is `ZYX` with 100 slices;
- materialized OpenHCS metadata has 100 Z indexes for selected c00 source files
  when the source-layout policy is active;
- no selected source stack is represented as one `z001` whole-file redirect;
- pipeline review output remains addressable by Z.

## Open Design Questions

- Should a CellProfiler pipeline that says `Process as 3D?: Yes` be the only
  trigger for TIFF Z expansion, or should source-schema materialization always
  preserve reader-declared Z when a file is matched as an image stack?
- How should multi-channel internal TIFFs interact with CellProfiler
  `NamesAndTypes` aliases that also assign channels by filename filters?
- Should source-schema projection use Bio-Formats for all TIFFs, or a lighter
  TIFF reader when the file is local and simple?
- How much of `source_projection` should be public metadata versus internal
  cache?
- Should old `workspace_mapping` eventually become a pure compatibility view
  generated from `source_projection`?
