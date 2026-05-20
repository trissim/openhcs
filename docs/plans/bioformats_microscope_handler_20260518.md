# Bio-Formats Microscope Handler Plan

## Status

Drafted 2026-05-18. This plan is intentionally implementation-facing. The paper can describe Bio-Formats-backed microscope support only after this plan is implemented, tested, and verified against real or representative datasets.

## Goal

Add a `BioFormatsHandler` to OpenHCS that lets users point OpenHCS at Bio-Formats-readable microscopy datasets and have source dimensions discovered into the same workflow model used by explicit microscope handlers.

The handler should make this claim true:

> Bio-Formats improves file readability; OpenHCS microscope handlers turn readable files and vendor plate folders into analysis-ready workflow sources with explicit well, site, channel, z-plane, and timepoint identities.

The handler must not replace explicit vendor handlers. `ImageXpressHandler`, `OperaPhenixHandler`, `OMEROHandler`, `OpenHCSMicroscopeHandler`, and future `CellObserverHandler` should remain preferred when they encode source semantics that Bio-Formats does not reliably expose.

## Motivation

Bio-Formats and CellProfiler solve important parts of the file-readability problem. CellProfiler includes Bio-Formats and uses it to read images from disk. CellProfiler input modules still require users or pipelines to establish file lists, metadata extraction, and image-set naming through `Images`, `Metadata`, `NamesAndTypes`, `Groups`, `LoadImages`, or `LoadData`.

OpenHCS should handle the next layer: converting readable datasets and vendor plate layouts into workflow-safe source identities. This matters because the bottleneck is often not just whether a file opens. It is whether a lab can select a plate folder and get correct well/site/channel/z/time source semantics without writing regexes, reorganizing folders, or relying on vendor software.

Concrete current examples:

- `ImageXpressHandler` already flattens `TimePoint_*` and `ZStep_*` folders virtually so timepoint and z-plane identity enter normalized filenames instead of forcing users to reorganize exports.
- `OperaPhenixHandler` already parses `Index.xml`, remaps field IDs to spatial layout, and fills inferred missing images with black placeholders for autofocus failures.
- `OMEROHandler` already handles a non-disk plate namespace and exposes OMERO-backed image lists/metadata.

The Bio-Formats handler should provide a broad fallback for datasets where Bio-Formats can discover dimensions, while preserving fail-loud behavior when biological plate semantics are ambiguous.

## Current Architecture Facts

### Microscope Handler Registry

Relevant files:

- `openhcs/microscopes/__init__.py`
- `openhcs/microscopes/microscope_base.py`
- `openhcs/microscopes/microscope_interfaces.py`
- `openhcs/microscopes/handler_registry_service.py`

Current design:

- `MicroscopeHandler` is an `AutoRegisterMeta` family using `_microscope_type` as the registry key.
- `METADATA_HANDLERS` is populated through the secondary registry from each handler's `_metadata_handler_class`.
- `create_microscope_handler(...)` supports explicit handler type or `"auto"`.
- Auto-detection currently tries `openhcsdata` first, then metadata handlers in registry order, and returns the first handler whose metadata file is found.
- The `Microscope` enum currently exposes `AUTO`, `OPENHCS`, `IMAGEXPRESS`, `OPERAPHENIX`, `BBBC021`, `BBBC022`, and `OMERO`.

Design implication:

- Add a `BioFormatsHandler` as a normal registered microscope family with `_microscope_type = "bioformats"`.
- Add `Microscope.BIOFORMATS = "bioformats"`.
- Auto-detection should try Bio-Formats late, after explicit vendor handlers, because Bio-Formats is a broad fallback and could otherwise mask a more semantic handler.

### Handler Responsibilities

Relevant files:

- `openhcs/microscopes/microscope_base.py`
- `openhcs/microscopes/microscope_interfaces.py`
- `openhcs/formats/pattern/pattern_discovery.py`
- `openhcs/core/orchestrator/orchestrator.py`
- `openhcs/core/pipeline/compiler.py`
- `openhcs/core/steps/function_execution.py`
- `openhcs/core/steps/function_io.py`

Current responsibilities:

- `FilenameParser` parses normalized filenames into component dictionaries with `well`, `site`, `channel`, `z_index`, `timepoint`, and `extension`.
- `MetadataHandler` provides grid dimensions, pixel size, channel values, well values, site values, z-index values, and image files.
- `MicroscopeHandler.initialize_workspace(...)` may build a virtual workspace mapping and register `Backend.VIRTUAL_WORKSPACE`.
- `MicroscopeHandler.post_workspace(...)` lists images from the selected backend, parses filenames, validates required components, normalizes filenames, and returns the image directory.
- Runtime pattern detection and loading go through `microscope_handler.auto_detect_patterns(...)` and `microscope_handler.path_list_from_pattern(...)`.
- Compiler validation asks the handler what read backends are supported.

Design implication:

- A Bio-Formats handler must either produce normalized virtual filenames compatible with `PatternDiscoveryEngine`, or override pattern discovery/listing methods to use Bio-Formats series/plane identities directly.
- The lower-risk implementation is to build an OpenHCS virtual workspace mapping whose keys are normalized names and whose values point to Bio-Formats-backed file/series/plane references.

### Existing Virtual Workspace Pattern

Relevant files:

- `openhcs/microscopes/microscope_base.py`
- `openhcs/microscopes/imagexpress.py`
- `openhcs/microscopes/opera_phenix.py`
- `openhcs/microscopes/openhcs.py`

Current design:

- Vendor handlers write `openhcs_metadata.json` with `workspace_mapping`, `available_backends`, `microscope_handler_name`, and `source_filename_parser_name`.
- `ImageXpressHandler._build_virtual_mapping(...)` maps normalized virtual filenames to nested `TimePoint_*` / `ZStep_*` source paths.
- `OperaPhenixHandler._build_virtual_mapping(...)` maps spatially remapped filenames under `Images/` to original files and may create black placeholder files when expected combinations are missing.
- `VirtualWorkspaceBackend` is then registered on the `FileManager`.

Design implication:

- Bio-Formats should reuse the same virtual workspace metadata convention wherever possible.
- If the underlying Bio-Formats reader requires file + series + plane indexing rather than just a path, we need either:
  - a new `Backend.BIOFORMATS` / `BioFormatsStorageBackend` in `polystore`, or
  - a materialized normalized workspace generated from Bio-Formats planes.
- A metadata-only mapping to a plain file path is insufficient when one physical file contains multiple series or planes that must become distinct OpenHCS source images.

### Storage Backend Gap

Current read backends include disk, zarr, memory, virtual workspace, and OMERO-local. There is no current `bioformats` backend or Bio-Formats dependency in `pyproject.toml`.

Design implication:

- The cleanest design is a dedicated optional storage backend that can load one plane by a structured reference.
- Avoid forcing microscope handlers to materialize every Bio-Formats plane to TIFF just to make existing disk loading work. Materialization can be an optional cache/export mode, not the core abstraction.

## Proposed Architecture

### New Optional Extra

Add a `bioformats` optional dependency group.

Candidate dependency strategy:

- Preferred: a maintained Python-facing Bio-Formats stack that avoids custom JVM lifecycle code in OpenHCS if available.
- Alternative: use the same Java Bio-Formats jars through `scyjava`/ImageJ infrastructure already used by the Fiji extra.
- Avoid making Bio-Formats a core dependency. JVM startup and Java jars should not affect users who do not need Bio-Formats.

Plan action:

1. Spike dependency options in a separate implementation pass.
2. Pick one adapter interface that hides dependency choice from the rest of OpenHCS.
3. Keep the public OpenHCS handler API stable if the underlying Bio-Formats package changes.

### New Adapter Layer

Add a small Bio-Formats adapter module, for example:

- `openhcs/microscopes/bioformats_adapter.py`
- `openhcs/microscopes/bioformats.py`

Core records:

```python
@dataclass(frozen=True, slots=True)
class BioFormatsDataset:
    root: Path
    entries: tuple[BioFormatsImageEntry, ...]

@dataclass(frozen=True, slots=True)
class BioFormatsImageEntry:
    source_path: Path
    series_index: int
    plane_index: int
    well: str | None
    site: int | None
    channel: int
    z_index: int
    timepoint: int
    channel_name: str | None = None
    pixel_size: float | None = None
```

Responsibilities:

- Discover candidate Bio-Formats-readable datasets under a root.
- Query series and dimension sizes.
- Convert C/Z/T plane coordinates into one `BioFormatsImageEntry` per 2D plane.
- Extract channel names and pixel sizes where present.
- Preserve source path, series index, and plane index for loading.
- Return explicit uncertainty when well/site cannot be inferred.

Fail-loud rules:

- If C/Z/T can be inferred but well/site cannot, the adapter can produce entries only for non-plate datasets or require a user-provided source schema.
- If multiple series have unclear identity, do not guess a plate layout.
- If Bio-Formats reports dimensions but no stable plane-to-coordinate mapping, fail with a remediation message.

### New Storage Backend

Add a `BioFormatsStorageBackend` in `polystore` if cross-package changes are acceptable, or an OpenHCS-local backend wrapper if not.

Required interface:

- `load(path_or_ref, backend="bioformats")` must load a single plane from a structured reference.
- `load_batch(...)` must load multiple plane refs efficiently.
- References should be serializable in virtual workspace metadata.

Recommended reference format:

```json
{
  "backend": "bioformats",
  "source_path": "relative/or/absolute/file.ext",
  "series_index": 0,
  "plane_index": 12,
  "c": 2,
  "z": 1,
  "t": 3
}
```

Virtual workspace key:

- Keep normalized OpenHCS filename keys such as `A01_s001_w2_z001_t003.tif`.
- Map each key to a structured Bio-Formats reference rather than a plain file path.

Risk:

- `VirtualWorkspaceBackend` may currently assume string path mappings. If so, either extend it to accept structured refs or introduce a `BioFormatsVirtualWorkspaceBackend` that resolves normalized paths through Bio-Formats refs.

### New Handler

Add `openhcs/microscopes/bioformats.py`.

```python
class BioFormatsHandler(MicroscopeHandler):
    _microscope_type = "bioformats"
    _metadata_handler_class = BioFormatsMetadataHandler
```

Handler behavior:

- `compatible_backends = [Backend.BIOFORMATS]` if a new backend is added.
- `root_dir = "."` for folder roots unless a dataset-specific root is discovered.
- `initialize_workspace(...)` discovers Bio-Formats entries, writes normalized virtual metadata, registers the Bio-Formats backend, and returns a virtual image directory.
- `post_workspace(...)` should avoid base-class file renaming assumptions if the backend is Bio-Formats rather than disk.
- `auto_detect_patterns(...)` and `path_list_from_pattern(...)` should continue to work through normalized keys. Prefer reusing `PatternDiscoveryEngine` with `SourceSchemaFilenameParser` or a `BioFormatsFilenameParser`.

Parser behavior:

- `BioFormatsFilenameParser` should parse normalized virtual filenames, not raw vendor filenames.
- It can reuse the same convention as `SourceSchemaFilenameParser`: `well_s{site}_w{channel}_z{z_index}_t{timepoint}.tif`.

Metadata handler behavior:

- `find_metadata_file(...)` should detect Bio-Formats-readable roots late in auto-detection.
- `get_grid_dimensions(...)` should return `(1, 1)` unless plate/site geometry is reliably present.
- `get_pixel_size(...)` should return Bio-Formats physical pixel size where present, otherwise fallback.
- `get_channel_values(...)` should return channel names where present.
- `get_well_values(...)`, `get_site_values(...)`, and `get_z_index_values(...)` should return values only when inferred.
- `get_image_files(...)` should return normalized virtual keys.

### Auto-Detection Policy

Detection order must be conservative:

1. `openhcsdata`
2. Explicit vendor handlers with strong metadata sidecars: ImageXpress/MetaXpress, Opera Phenix, OMERO, BBBC handlers, future Cell Observer.
3. `bioformats`

Reason:

- A Bio-Formats reader may be able to open an ImageXpress or Opera Phenix file, but it should not override handlers that encode OpenHCS-specific workflow semantics such as virtual folder flattening, field remapping, or missing-image policy.

Add a mechanism to mark broad fallback handlers:

```python
class BioFormatsHandler(MicroscopeHandler):
    _microscope_type = "bioformats"
    detection_priority = "fallback"
```

If adding a general priority system is too large, hardcode late ordering for `bioformats` in `_auto_detect_microscope_type(...)` as a first pass and add a refactor task.

### Source Schema Integration

The handler should produce normalized source identities that can feed existing source-binding semantics:

- well
- site
- channel
- z_index
- timepoint
- source_path
- series_index
- plane_index

Do not duplicate `PipelineImageSchema` or `StepSourceBindingsConfig`. Instead:

- Let microscope handlers normalize raw acquisition layout into file/plane sources.
- Let source schemas/bindings map those sources into named workflow roles when native workflows need explicit source aliases.
- Let CellProfiler imports keep `.cppipe` loading semantics separate.

### GUI Integration

Update the GUI only after the backend/handler behavior is stable.

Required GUI surfaces:

- Add `BioFormats` to microscope selection enum/dropdown.
- For `AUTO`, show detection results clearly: explicit vendor handler vs Bio-Formats fallback.
- If Bio-Formats cannot infer plate semantics, show a source-schema dialog instead of proceeding silently.
- Metadata viewer should display Bio-Formats-discovered dimensions, channel names, pixel size, and warnings.

## Implementation Sequence

### Pass 1: Dependency Spike

Deliverables:

- Choose Bio-Formats access library and optional dependency group.
- Add a tiny adapter spike that opens one representative dataset and reports series count, dimensions, channel names, pixel size, and plane coordinate mapping.
- Document JVM lifecycle implications if Java-backed.

Verification:

- Unit test with mocked adapter results.
- Optional integration smoke guarded by `OPENHCS_ENABLE_BIOFORMATS_TESTS=1`.

### Pass 2: Adapter Records

Deliverables:

- Add immutable records for dataset discovery and plane entries.
- Add fail-loud result types for ambiguous semantics.
- Add adapter tests using fake Bio-Formats metadata, not real jars.

Verification:

- Tests for C/Z/T projection.
- Tests for missing well/site behavior.
- Tests for channel names and pixel-size extraction.

### Pass 3: Storage Backend

Deliverables:

- Add `Backend.BIOFORMATS`.
- Add Bio-Formats backend registration path.
- Load one plane by structured reference.
- Batch-load planes.

Verification:

- Unit tests with a fake reader.
- Integration smoke with a small public or generated supported file if feasible.

### Pass 4: Handler

Deliverables:

- Add `BioFormatsFilenameParser`.
- Add `BioFormatsMetadataHandler`.
- Add `BioFormatsHandler`.
- Add `Microscope.BIOFORMATS`.
- Add late auto-detection behavior.

Verification:

- Unit tests for registry discovery.
- Unit tests for explicit `create_microscope_handler("bioformats", ...)`.
- Unit tests that vendor handlers win before Bio-Formats fallback.
- Unit tests for normalized image-file listing.

### Pass 5: Runtime Integration

Deliverables:

- Ensure `PipelineOrchestrator.initialize(...)` can initialize a Bio-Formats handler.
- Ensure compiler read-backend validation accepts `Backend.BIOFORMATS`.
- Ensure `PatternDiscoveryEngine` can discover normalized Bio-Formats virtual keys.
- Ensure `bulk_preload_step_images(...)` can load Bio-Formats-backed refs.

Verification:

- Minimal pipeline reads a Bio-Formats-backed source and executes a no-op or simple function.
- Test both explicit `Microscope.BIOFORMATS` and `Microscope.AUTO` fallback.

### Pass 6: GUI And Source Binding

Deliverables:

- Add GUI microscope option.
- Add warning/metadata display for fallback inference.
- Add source-binding fallback when well/site cannot be inferred.

Verification:

- Unit tests for enum/dropdown presenter behavior.
- GUI smoke if existing test harness supports it.

### Pass 7: Paper Evidence Gate

Deliverables:

- Replace `TODO: Bio-Formats...` in the paper with implementation evidence.
- Add exact list of tested Bio-Formats datasets/formats.
- Add a limitations sentence: Bio-Formats-backed discovery is broad but not magic; vendor handlers remain preferred for known plate-layout quirks.

Verification:

- Cite Bio-Formats for broad file readability.
- Cite OpenHCS tests for source identity normalization.
- Do not claim full automatic plate semantics for all Bio-Formats formats.

## Test Matrix

Required tests:

- `tests/unit/test_bioformats_adapter.py`
- `tests/unit/test_bioformats_microscope_handler.py`
- `tests/unit/test_bioformats_storage_backend.py` if backend is local.
- Integration test with a small fixture dataset.
- Auto-detection precedence test.
- Failure-mode test for ambiguous well/site semantics.
- Source-binding handoff test for ambiguous datasets.

Suggested fixture classes:

- Single multi-plane file with C/Z/T and no plate semantics.
- Folder with multiple files where filenames expose well/site.
- Simulated vendor dataset that Bio-Formats can read but explicit handler should own.
- Dataset with channel names and physical pixel size.

## Risks And Design Boundaries

### Do Not Overclaim Bio-Formats

Bio-Formats is a reader and metadata normalization layer. It does not guarantee workflow-level plate intent. The handler must distinguish:

- readable pixels,
- available C/Z/T metadata,
- inferred well/site identities,
- explicit user-provided source schema.

### Avoid Silent Misgrouping

Wrong image-set grouping is worse than a clear failure. If a dataset lacks stable well/site/channel semantics, stop and ask for a source schema or metadata mapping.

### Keep Vendor Handlers First

Bio-Formats fallback must not erase value from explicit handlers. ImageXpress and Opera Phenix are examples where OpenHCS has domain logic beyond opening pixels.

### Keep JVM Optional

Bio-Formats support should be optional. Core OpenHCS should not require Java startup or Bio-Formats jars.

### Keep Source Binding Separate From CellProfiler Import

Imported `.cppipe` workflows preserve their encoded loading semantics. The Bio-Formats handler is for native/source-driven workflows and broad dataset discovery, not a replacement for CellProfiler import semantics.

## Code Review Checklist Before Implementation

- Re-open `openhcs/microscopes/microscope_base.py` before editing auto-detection.
- Re-open `openhcs/microscopes/microscope_interfaces.py` before implementing metadata handler methods.
- Re-open `openhcs/core/orchestrator/orchestrator.py` before changing workspace initialization.
- Re-open `openhcs/core/pipeline/compiler.py` before adding read backend validation.
- Re-open `openhcs/core/steps/function_io.py` and `openhcs/core/steps/function_execution.py` before changing load paths or pattern discovery.
- Re-open `pyproject.toml` before adding optional dependencies.
- Fix or account for `validate_backend_compatibility(...)` referencing `supported_backends`; the current handler API exposes `compatible_backends`.
- Run focused microscope/source tests before broader integration tests.

## Plan Review Against Current Code

Reviewed against the current codebase on 2026-05-18.

Confirmed existing seams:

- `MicroscopeHandler` is already a registered family keyed by `_microscope_type`, so `BioFormatsHandler` should be a normal handler rather than a side runner.
- `create_microscope_handler(...)` and `_auto_detect_microscope_type(...)` are the correct entry points for explicit selection and automatic fallback detection.
- `initialize_workspace(...)`, `_build_virtual_mapping(...)`, and `post_workspace(...)` are the current acquisition-layout normalization seams.
- Runtime source discovery uses `auto_detect_patterns(...)` and `path_list_from_pattern(...)`, so normalized virtual keys are the least disruptive integration path.
- `ImageXpressHandler` and `OperaPhenixHandler` already prove the handler layer is responsible for vendor layout quirks, not just opening image pixels.
- `pyproject.toml` has `tifffile`, `ome-zarr`, and Fiji/scyjava extras, but no Bio-Formats-specific dependency group.
- `Backend` currently has no `BIOFORMATS` value; one-file-many-plane reads need a new backend or a materialization cache.

Implementation-critical gaps:

- Current virtual workspace metadata records path-like mappings. Structured Bio-Formats refs may require extending `VirtualWorkspaceBackend` or adding a dedicated Bio-Formats virtual backend before the handler can avoid materializing all planes.
- Base `post_workspace(...)` assumes files can be listed, parsed, renamed, and represented as image paths. A Bio-Formats handler should override this path if it uses structured refs.
- `validate_backend_compatibility(...)` currently checks `handler.supported_backends`, while handlers define `compatible_backends`. Fix this separately or as part of the backend integration pass.
- Bio-Formats fallback detection needs explicit late priority; relying on registry insertion order is too fragile for a broad reader.
- The plan should not claim automatic plate semantics for all Bio-Formats-readable datasets. The adapter must surface uncertainty and require explicit source schema input when well/site identity cannot be inferred.
