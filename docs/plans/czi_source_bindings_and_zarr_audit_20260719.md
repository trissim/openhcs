# CZI Source Bindings and Zarr Audit

**Date:** 2026-07-19
**Status:** unified store-emitted planes, saved source-state rebuild, canonical
biologist knowledge registration, and focused real-format acceptance are verified;
coordinated live UI/code-mode acceptance remains with its active MCP owner
**Scope:** store-emitted addressable planes, cross-store sample-group aggregation,
named source binding, VFS loading, and Zarr materialization/reload for CZI,
OME-TIFF, OME-Zarr, TIFF, PNG, and mixed-folder inputs

## Status

- **Complete in this lane:** manifest-backed knowledge retrieval, dead-helper
  deletion, one/multiple-container and collision regressions, and focused real
  CZI/OME-TIFF acceptance. Canonical public docstrings and the single
  biologist-facing source guide cover every requested store family. The unified
  nominal store collector, Java, OME-Zarr, ordinary-image leaves, Zarr conversion,
  and saved-config projection rebuild are implemented and covered.
- **Latest gate:** the combined format/store/knowledge/runtime-preload suite passes
  (`74 passed in 16.26s`), the guide validates with two executable Python blocks,
  and scoped
  syntax/Ruff/JSON/diff/deletion gates are clean. The durable real-format mixed
  NGFF + scalar TIFF + scalar PNG
  regression emits both `ome_zarr` and `disk` refs, preserves three named aliases,
  and reloads exact arrays through `virtual_workspace`. Public OME-TIFF companion
  files and the licensed CZI fixture also pass positive Java declaration, unified
  projection, and physical plane loading. The post-fix focused source/metadata
  suite passes (`33 passed in 5.20s`; scoped Ruff and syntax clean). The current
  built-in ImageXpress plate opens through 216 strict typed source refs and an exact
  virtual-workspace plane load. Saved binding edits invalidate and rebuild the
  projection with all canonical coordinates, aliases, and backend identities; the
  broader source/orchestrator gate is `64 passed in 10.85s`. A fresh licensed CZI
  probe binds alias `DNA` and VFS-loads one `3648x3648 uint16` plane; the public
  OME-TIFF companion emits five planes and VFS-loads one `96x96 int8` plane.
- **Remaining coordinated step:** the active MCP worker owns the default-240-second
  live save -> reinitialize -> visible-state rehearsal and its process lease. This
  lane does not start a second UI/ZMQ/Napari process while the parent reserves the
  official30 lock.
- **Boundaries:** no CellProfiler or code-mode peer-owned files are being edited;
  no unrelated changes are reverted, and no commit or push will be made.

## Authoritative Source-System Boundary

- CZI, OME-TIFF, OME-Zarr, individual TIFF/PNG, and mixed folders are storage
  backends only.
- Each store emits typed addressable planes plus exact metadata through one generic
  plane-source contract: `SourcePixelRef`, `SourceCandidate`, source projection,
  and VFS backend/dataset metadata are the smallest load-bearing authorities.
- An aggregate dataset/plate is a collection of sample groups across stores.
  Source bindings select and declare artifacts at the plate/sample-group boundary;
  they do not traverse, infer, or reinterpret format-internal axes.
- `WELL` is the canonical generic sample identity: the embedded plate well when one
  exists, otherwise the explicit sample/container identity. `SITE`, `CHANNEL`,
  `Z_INDEX`, and `TIMEPOINT` are exact per-plane coordinates. Human labels remain
  metadata and never replace coordinate identity.
- Explicit embedded metadata wins. An absent axis normalizes to coordinate `"1"`
  only when the store declares it absent or singleton. Multiple samples/scenes
  remain distinct, and exact identity collisions fail during dataset aggregation.
- Generic source matching contains no format policies. Store-specific code is
  limited to decoding its format and emitting the common plane declarations.
- The implementation must delete superseded format-level aggregation/projection,
  duplicated metadata writers, and forwarding wrappers. Completion requires a
  before/after authority inventory and production diffstat demonstrating a reduced
  production surface, not parallel CZI/OME-TIFF/OME-Zarr paths.
- The collector circular import is resolved first by moving the generic candidate
  authority to its owning core projection module and removing the eager dependency;
  lazy imports, `try`/`except` fallbacks, and compatibility aliases are prohibited.
- Acceptance uses the actual running PyQt code-mode UI through the MCP dev bridge
  and normal UI -> ZMQ compile/execute/materialize/reopen path for every supported
  store family with an available fixture. No format-specific UI/runtime shortcut is
  permitted.

## Ownership

- This audit owns this report and, only if executable evidence needs a durable regression, a new narrowly scoped CZI/Zarr test file.
- Existing edits in `openhcs/core/source_matching.py` and `openhcs/microscopes/bioformats.py` predate this audit and are externally owned.
- Do not edit CellProfiler `pipeline_import.py`, module artifact declaration/contract files, or files owned by `docs/plans/code_mode_paths_and_artifact_tab_plan_20260719.md` unless a verified defect has no disjoint owner and the parent is notified first.
- Do not revert, rewrite, or clean unrelated worktree changes.
- The resumed worker owns the Bio-Formats/source-binding implementation in
  `bioformats_adapter.py`, `bioformats_spw_projector.py`, and `bioformats.py`, the
  narrow generic candidate/handler capability hooks required to connect existing
  nominal owners, focused CZI tests, and this ledger.
- Live UI/code-mode acceptance must use the existing MCP dev client, plate manager,
  code-mode, and ZMQ compilation surfaces. No CZI-specific UI route is in scope.

## Active Coordination

- `.agents/cp-advanced-segmentation-object-source-conflict.md` is complete. Its former `source_bindings.py` and focused source-binding test surfaces are restore-only and remain untouched.
- `.agents/artifact-contract-collapse.md` owns the current CellProfiler declaration/import/runtime migration and is excluded from this audit.
- `.agents/official30-knowledge-examples.md` actively owns
  `knowledge_base_service.py`, knowledge manifest entries, and MCP knowledge tests.
  This plan owns the canonical source-store docstrings/RST and will provide one
  exact document path plus acceptance queries to that owner; it will not duplicate
  the knowledge service, manifest table, or prose.
- `.agents/global-ui-zmq-config-tabs.md` owns the live PyQt PID, bridge descriptor,
  and any ZMQ/Napari children. This plan owns the typed source-ref fixture migration
  and handler projection rebuild semantics only; it will not start, reuse, stop, or
  mutate the leased process. Pauli owns live save -> reinitialize -> visible-state
  proof after the focused source gates pass.
- `docs/plans/code_mode_paths_and_artifact_tab_plan_20260719.md` is in progress. It explicitly excludes source-binding URI semantics and backend addressing; this audit does not edit its source generation, compilation path, or UI files.
- No worker-spawn capability is available in this session, so the audit is executed serially through this ledger.
- 2026-07-19 07:10 EDT UI-acceptance dependency from
  `.agents/global-ui-zmq-config-tabs.md`: the real built-in ImageXpress plate
  reaches `OpenHCSMicroscopeHandler.initialize_workspace()` but fails in
  `VirtualWorkspaceBackend._load_mapping()` because its unchanged
  `openhcs_metadata.json::workspace_mapping` values are legacy strings and
  `SourcePixelRef.from_workspace_mapping()` now requires structured mappings.
  Exact terminal error: `TypeError: SourcePixelRef workspace mapping must be
  structured.` This occurs before ZMQ startup. Kepler retains ownership of the
  source-ref/metadata migration and should resolve or explicitly classify this
  built-in-plate compatibility boundary without a UI fallback or copied metadata
  cache. The UI-config worker retains the real MCP code-mode save -> reinitialize
  -> visible component-domain/microscope-metadata proof and same-path rollback once
  current source can initialize the existing plate.

## Authority Inventory

- `ImageFileFormat.__registry__` in `openhcs/core/image_file_serialization.py` owns OpenHCS disk serialization formats. CZI is not an OpenHCS serialization target and should not be added there merely to make it source-readable.
- Polystore's format registry, queried by `openhcs.core.source_matching.is_image_path()`, owns whether an extension is a loadable pixel source for `IS_IMAGE` filters.
- `BioFormatsMetadataAdapter.__registry__` and `BioFormatsCompositeAdapter` own Bio-Formats dataset discovery. The registered adapter order is manifest, Java, then filename layout.
- `BioFormatsDatasetAuthority`, `BioFormatsSPWProjector` / `BioFormatsLayoutProjector`, and `BioFormatsDatasetCompletenessValidator` own conversion from reader metadata to complete HCS plane entries.
- `BioFormatsWorkspaceMetadataWriter` owns normalized virtual TIFF paths and structured `BioFormatsPlaneRef(source_path, series_index, plane_index)` addresses. The physical CZI suffix is intentionally absent from source-binding matching after projection.
- `FileManager` plus `BioFormatsStorageBackend` and `VirtualWorkspaceStorageBackend` own physical plane loading from those structured references.
- `SourceBindingDeclarationsMixin`, source metadata roles, and source matching own binding and image-set construction over normalized virtual paths and their component metadata.
- `ZarrConfig`, the polystore Zarr backend, `function_io` materialization helpers, OpenHCS metadata, and `OpenHCSMicroscopeHandler` own Zarr write/read/reload behavior.

## Executable Plan

- [x] Read root `AGENTS.md`, current worktree state, active ownership notes, and the code-mode plan.
- [x] Search nominal registries/MRO authorities before proposing changes.
- [x] Trace the initial source-filter, Bio-Formats projection, structured plane-address, and virtual workspace path.
- [x] Confirm installed polystore/Bio-Formats/Zarr capability and dependency state.
- [x] Identify a small explicitly licensed CZI fixture and report source and size before download.
- [x] Probe one CZI through detection, workspace projection, source matching/binding, and physical plane load.
- [x] Probe multiple CZI files under one plate root and prove whether all files survive discovery and matching.
- [x] Probe Zarr materialization and reload from the projected CZI plate where supported.
- [x] Add only the narrow regression test justified by observed behavior.
- [x] Run focused suites and finalize the support matrix, architectural gap owners, next steps, and changed-file list.

### Resumed Implementation Checklist

- [x] Re-read `AGENTS.md`, active ownership notes, current diffs, and this ledger.
- [x] Build an AST inventory of Bio-Formats metadata/projection declarations and constructor sites.
- [x] Identify the existing generic alias/artifact and typed filter-path authorities.
- [x] Add nominal OME plate/container identity and exact independent-container aggregation.
- [x] Reject duplicate/conflicting image, WellSample, series, and plane identities at projection time.
- [x] Project Bio-Formats plane candidates through declared source bindings without path/extension/order inference.
- [x] Preserve physical source paths as typed filter provenance distinct from backend addresses.
- [x] Add focused one-CZI, aggregation, collision, provenance, VFS, and Zarr reload regressions.
- [x] Run scoped Ruff and focused unit/integration tests.
- [ ] Exercise normal plate-manager/code-mode APIs through the MCP dev/UI bridge and ZMQ compiler path.
- [ ] Record MCP actions, visible state, server logs, timings, failures, and final changed files.

### Expanded OME Scope

- [x] Eliminate the collector import cycle introduced by the Bio-Formats writer.
- [x] Inventory OME-TIFF and OME-Zarr/NGFF dataset and source-projection owners.
- [x] Prove the Bio-Formats path projects OME-TIFF without format-specific matching.
- [x] Add nominal OME-Zarr dataset/source candidate projection at its owning boundary.
- [x] Preserve aliases and well/site/channel/Z/time identities through
  CZI, OME-TIFF, and OME-Zarr materialization/reopen.
- [ ] Exercise every feasible format through live code mode -> ZMQ acceptance.

### Generic Plane-Source Refactor

- [x] Record the authoritative store/plane/sample-group boundary before production
  implementation.
- [x] Finish moving `SourceCandidate` to the core source-projection owner and prove
  the collector import graph is stable with no lazy dependency.
- [x] Replace Bio-Formats SPW/layout projection and format-level workspace
  orchestration with one store-emitted `SourceCandidate` dataset.
- [x] Make CZI and OME-TIFF emit exact OME identities and plane coordinates through
  that contract; reject explicit identity/address collisions.
- [x] Make OME-Zarr/NGFF emit the same contract from its embedded plate/well/image,
  multiscale-axis, and channel metadata.
- [x] Make individual TIFF/PNG and mixed folders emit explicit generic sample/plane
  declarations without extension or filename semantics in source matching.
- [x] Add one mixed-store aggregation regression using multiple storage backends.
- [x] Prove named aliases, all component identities, grouping, VFS load,
  materialization, and Zarr reopen through focused tests.
- [x] Delete superseded authorities/helpers and record before/after production line,
  declaration, and diffstat inventories.
- [x] Run scoped Ruff, syntax, import, and focused test gates.
- [ ] Run live MCP PyQt code-mode -> ZMQ compile/execute/materialize/reopen
  acceptance and record exact actions, state, logs, timings, and failures.

### Biologist-Facing Discoverability

- [x] Put concise public docstrings on the load-bearing source-binding and
  store-projection entry points. Explain named sources, sample/well identity,
  site/channel/Z/time coordinates, singleton-axis defaults, and explicit
  collision failures without exposing storage-adapter internals.
- [x] Add or update the canonical RST user documentation with executable code-mode
  examples for ordinary TIFF/PNG files, CZI, OME-TIFF, OME-Zarr, and mixed stores.
  Examples must use the same `PipelineConfig` source-binding API exercised by the
  UI and ZMQ compiler; do not add a format-specific tutorial API.
- [x] Register that canonical documentation through the existing knowledge-base
  manifest/owner so MCP knowledge search retrieves it for plain-language queries
  from a biologist. Do not duplicate the prose or add a parallel document table.
- [x] Add focused knowledge retrieval tests for queries such as loading CZI,
  combining multiple image stores, assigning named channels, and interpreting
  well/site/channel/Z/time metadata.
- [ ] Exercise the MCP knowledge and function-detail capabilities before the live
  UI run, then follow the retrieved instructions through code mode and verify the
  resulting named sources and component identities in the UI.

### Source-State Acceptance

- [x] Migrate the built-in ImageXpress fixture producer and current fixture from
  string workspace mappings to exact serialized `SourcePixelRef` values; do not
  weaken typed deserialization or add an ImageXpress/runtime compatibility branch.
- [x] Add a focused regression proving the built-in plate initializes and loads a
  plane through `VirtualWorkspaceBackend` using the same typed mapping path.
- [x] Prove that saving an edited `PipelineConfig.source_bindings_config` invalidates
  and rebuilds the selected plate's microscope-handler projection during normal
  initialization.
- [x] Prove canonical plate state receives every resolved sample/well, site,
  channel, Z, timepoint, named alias, and store-backed source identity from that
  rebuilt projection, with no UI-owned metadata mirror.
- [x] Hand the focused source-state evidence to the live UI lease owner; Pauli owns
  save -> reinitialize -> visible-state acceptance in the running UI.

## Current Support Matrix

| Flow | Executable evidence | Status |
|---|---|---|
| Physical `.czi` -> store/source filter | CZI is intentionally absent from `ImageFileFormat`; positive Java declaration emits typed Bio-Formats candidates whose filter provenance retains the original relative and absolute CZI paths | Supported through the Java plane-store leaf without adding CZI as a serialization target |
| One actual CZI -> Bio-Formats pixels | Zeiss Quick Start CZI reader returned one `3648x3648 uint16` virtual-workspace plane with sum 2,748,293,312 | Supported by the declared optional Java runtime |
| One actual CZI -> automatic HCS projection | Fixture declares zero OME Plates, so the current Java leaf emits an exact non-plate container/sample identity rather than inventing a well | Not demonstrated for plate-bearing CZI; correctly unavailable for this non-SPW fixture |
| Explicit SPW manifest -> one/multiple CZI virtual plate | Four physical CZI paths become explicit normalized virtual planes in deterministic fixture coverage | Supported as a fixture declaration only; it is not evidence of embedded plate semantics |
| Projected CZI -> named source binding | Fresh licensed-fixture handler initialization binds the physical master filename to alias `DNA`; deterministic multi-container tests preserve exact source paths | Supported |
| Projected CZI -> source path filters | Every Java candidate carries relative plus resolved physical path identities separately from its opaque `BioFormatsPlaneRef` backend address | Supported |
| Multiple independent plate-bearing CZI -> automatic Java SPW aggregation | Deterministic two-container regression merges equal explicit Plate IDs, preserves both container identities, and a sibling regression rejects duplicate exact plane addresses | Implemented and focused-tested; no licensed independent plate-bearing CZI pair is available for physical proof |
| Multiple files in downloaded fixture | Four physical files are one multipart dataset; `(1)`-`(3)` redirect to the master and automatic collection emits one plane | Multipart deduplication supported; not independent-container evidence |
| CZI plane -> FileManager/VFS | Fresh typed virtual-workspace mapping loads the real `3648x3648 uint16` plane exactly | Supported |
| Bio-Formats direct backend as directory/preload backend | Corrected runtime integration preloads through `virtual_workspace`; the direct backend remains an opaque single-plane reader | Supported through the declared virtual-workspace owner; direct directory listing is intentionally unsupported |
| CZI-derived plate -> Zarr | Four virtual planes saved with the Zarr backend and reloaded exactly | Supported |
| Zarr -> OpenHCS metadata reload | Initially failed because a manual parser catalog omitted `BioFormatsFilenameParser`; registry fix restores the parser and selects Zarr | Fixed and regression-tested |
| Reopened Zarr -> named source selectors | Site 1-4 selector matrix was diagonal; all `.tif` logical paths satisfy `IS_IMAGE` | Supported for normalized component selectors |
| OME-TIFF companion -> Java/VFS | Public companion fixture emits Plate:0, five exact planes across A02/B01/B03/C02, and loads one `96x96 int8` plane with sum -1,076,724 | Supported |
| OME-Zarr/NGFF -> store/VFS | Generated real NGFF plate emits exact embedded axes/channel labels and loads through the `ome_zarr` ref/backend | Supported |
| Ordinary TIFF/PNG -> store/VFS | Registered scalar 2D formats emit exact `disk` refs; richer OME-TIFF is positively diverted to Java | Supported |
| Mixed NGFF + TIFF + PNG -> aliases/Zarr reopen | Durable test preserves three aliases, both physical backends, exact components, typed conversion metadata, and exact reopened arrays | Supported |

## Progress Ledger

### 2026-07-19 05:02 EDT - Initial ownership and authority audit

- Read `AGENTS.md`, all active coordination filenames, the relevant source-conflict ownership/status, and the code-mode plan.
- Confirmed a heavily shared dirty tree. Existing modifications to `openhcs/core/source_matching.py` and `openhcs/microscopes/bioformats.py` are preserved as external work.
- Searched the required registry and strategy terms and identified the existing nominal authorities listed above.
- Found no `.czi` file under `/home/ts`, `/tmp`, `/home/ts/code`, or `/home/ts/Downloads` with the audit search commands.
- Existing Bio-Formats tests use a manifest plus `.npy` fixture; they prove virtual workspace mechanics but do not prove actual CZI decoding.
- Next: inspect the installed polystore package and Java runtime, then identify a licensed small CZI fixture before downloading data.

### 2026-07-19 05:09 EDT - Dependency, filter, and fixture-source audit

- Confirmed that importing OpenHCS from the source checkout activates `external/PolyStore/src`; tests therefore exercise the local PolyStore source, not the installed PyPI copy.
- The local polystore `FileFormat` owner has no CZI declaration. `get_format_from_extension('.czi')` raises `ValueError`, so `IS_IMAGE` does not match a physical CZI path. This does not by itself block the intended Bio-Formats flow, because successful handler preparation exposes normalized virtual `.tif` paths to source bindings.
- `BioFormatsStorageBackend` is present in the local PolyStore source. It is intentionally a direct structured-address reader: directory `exists`/listing through that backend is unsupported.
- The current environment has Zarr 2.18.7 and ome-zarr installed. It does not yet have the OpenHCS `bioformats` extra (`pyimagej` and `scyjava`). Java 26 is installed.
- Focused existing tests: `14 passed, 1 failed in 7.01s`. The failure is the integration preload test checking a plate directory through the direct-address Bio-Formats backend, which returns false and raises `ValueError: Directory does not exist`. This boundary is under further audit before assigning a fix.
- Licensed fixture candidate identified before download: [Zenodo DOI 10.5281/zenodo.8263451](https://doi.org/10.5281/zenodo.8263451), "CZI file examples" by Nicolas Chiaruttini, CC BY 4.0, explicitly published for CZI reader testing. Four multipart camera-noise files are 7.4 MB each. The 302 MB and 4.5 GB files are excluded. Attribution and checksums will be retained in the audit cache.
- Next: install only the declared Bio-Formats optional runtime, fetch the 7.4 MB fixture set, verify checksums, and probe reader metadata before deciding whether it is suitable for HCS source-binding projection.

### 2026-07-19 05:13 EDT - Real CZI reader and projection probes

- Installed the already-declared Bio-Formats optional runtime into `.venv`: pyimagej 1.8.0, scyjava 1.12.5, and jpype1 1.7.1. No dependency declaration changed.
- Downloaded only the four 7,395,712-byte multipart files to `/tmp/openhcs-czi-audit/fixture` (29,582,848 bytes total). All four MD5 values match the Zenodo record.
- Bio-Formats selected `Zeiss CZI (Quick Start)` and decoded one `3648x3648 uint16` plane. The fixture reports one image/series and zero OME plates/screens.
- Automatic dataset projection failed because the fixture has neither OME Plate metadata nor a registered HCS filename layout. This does not disprove the single-container OME-SPW path; a licensed small plate-bearing CZI fixture was not located.
- A temporary explicit `bioformats_spw.json` projected all four physical paths to `A01` sites 1-4. `VirtualWorkspaceStorageBackend` loaded every projected path through its structured `BioFormatsPlaneRef`.
- The files are one multipart dataset: opening `(1)`, `(2)`, or `(3)` redirects the reader to the master. All four projected arrays are consequently identical. This is valid multi-file addressing coverage, but it is not evidence for multiple independent CZI containers.
- Named source-binding matching failed for all four sites because `SourcePlaneProjection.source_alias` is unset while `SourceProjection.matches_binding()` requires exact alias identity. File filters see serialized plane-ref JSON because `SourcePatternResolutionContext.from_projection()` maps virtual paths to `backend_address`.

### 2026-07-19 05:17 EDT - Multi-container and Zarr probes

- A two-file Java-adapter probe returned after visiting `plate-a.czi`; `_BioFormatsJavaAdapter.discover()` does not aggregate multiple independently plate-bearing containers.
- Through a real `PipelineOrchestrator` and `ProcessingContext`, loaded the four CZI-derived virtual paths, wrote them with `save_materialized_data(..., backend='zarr')`, and called `update_metadata_for_zarr_conversion()`.
- Raw Zarr reads matched every source array exactly. Metadata marked `.` as `main=false`, `zarr` as `main=true`, and retained `BioFormatsFilenameParser` for both subdirectories.
- Auto-reload initially selected `OpenHCSMicroscopeHandler` and the `zarr` subdirectory, but parser restoration failed: `_get_available_filename_parsers()` manually listed only ImageXpress, Opera Phenix, and source-schema parsers.
- Replaced that mirrored catalog with `FilenameParser.__registry__.values()` and added a focused Zarr-main reload regression. After the fix, auto-reload selected backend `zarr`, found all four logical TIFF paths, and reproduced all four source arrays exactly.
- Direct matching on reopened Zarr paths produced the expected diagonal site 1-4 matrix. Matching against the original Bio-Formats projection remained all false, confirming that Zarr storage is not the owner of the source-alias gap.
- Next: no further production edits in this audit. The unresolved changes cross existing Bio-Formats projection and source-provenance ownership and should be designed there, not patched in Zarr, source filters, or the format registry.

### 2026-07-19 05:30 EDT - Resumed implementation authority audit

- Re-read the current shared tree and preserved all concurrent edits. The files
  `source_binding_workspace.py`, `microscope_base.py`, `bioformats.py`, and generic
  source projection/metadata modules already contain substantial external work;
  this implementation will make only scoped additions at the required boundaries.
- AST inventory covered every definition and constructor call for
  `BioFormatsMetadata`, `BioFormatsPlate`, `BioFormatsWell`,
  `BioFormatsWellSample`, `BioFormatsImage`, `BioFormatsImageEntry`,
  `BioFormatsDataset`, and `SourcePlaneProjection` in the adapter, projector,
  writer, and focused tests.
- Existing authority confirmed: `SourceBindingWorkspaceProjector` already owns
  selector evaluation and exact `NamedSourceBinding` -> `SourceProjection`
  conversion; `SourceCandidate` owns backend refs and filter identities;
  `SourceFilterPathMetadata` owns serialized physical filter provenance;
  `SourceProjection.matches_binding()` consumes exact projected alias/artifact
  identity; and `VirtualWorkspaceBackend` remains the structured-ref dispatcher.
- Planned production shape: expose the existing generic candidate projection method,
  add a microscope-handler capability hook so an explicitly selected Bio-Formats
  handler may consume source declarations, and have the Bio-Formats writer submit
  plane candidates to that generic projector. No alias or source path is inferred.
- Planned aggregation shape: OME Plate ID is the nominal dataset identity; the
  exact Bio-Formats used-file set is the nominal container identity. Equal
  container identities are multipart duplicates, while distinct containers merge
  only when their explicit Plate IDs agree. Projector validation owns all image,
  WellSample, series, plane, and projected-address collision rejection.
- Next: implement the identity records and generic projection connection, then add
  deterministic focused regressions before the live MCP/UI acceptance pass.

### 2026-07-19 05:40 EDT - Nominal identity and binding implementation

- Added explicit `BioFormatsPlateIdentity` and exact used-file-set
  `BioFormatsContainerIdentity` records at the adapter boundary. The registered Java
  adapter now visits every candidate, deduplicates only equal container identities
  (multipart members), and merges distinct containers only when their sole declared
  OME Plate IDs agree.
- Added required OME Plate, Well, WellSample, and WellSample.Index identities to the
  adapter records. The SPW projector now uses `WellSample.Index + 1` directly for
  site identity and rejects duplicate/conflicting Plate, Well, WellSample, Image,
  container/series, plane-coordinate, reader-plane, and projected-address records.
- Exposed the existing generic `projection_set_for_candidates()` method and added
  candidate component labels. `BioFormatsWorkspaceMetadataWriter` now submits exact
  structured refs, canonical component metadata, and physical source path identities
  to `SourceBindingWorkspaceProjector`; that existing authority assigns declared
  aliases/artifact kinds and emits typed `SourceFilterPathMetadata`.
- Added a nominal handler capability method. An explicitly selected Bio-Formats
  handler now consumes the same `SourceBindingsConfig` passed by normal
  orchestrator/UI setup, while handlers that do not declare the capability retain
  the existing generic source-bindings handler behavior.
- Syntax/diff gate passed, then the updated existing Bio-Formats fixture/projector/
  Java/handler tests passed: `22 passed in 3.64s`.
- Next: add focused deterministic CZI aggregation/collision/VFS/Zarr regressions,
  run the scoped suite, then exercise the running UI through the MCP dev client.

### 2026-07-19 05:44 EDT - Collector cycle fixed; OME scope expanded

- Reproduced the reported import chain at the ownership boundary. The new
  top-level `bioformats.py -> source_binding_workspace.py` edge re-entered the
  eagerly discovered microscopes package while `source_binding_workspace.py` was
  only partially initialized.
- Matched the established `SourceBindingsHandler` import boundary: Bio-Formats
  retains type-only candidate imports and resolves the required core projector in
  the writer execution method. This is an unconditional dependency with no
  try/except, alternate implementation, or fallback chain.
- Exact stability command:
  `.venv/bin/python -c 'import benchmark.adapters.cellprofiler; import openhcs.core.source_binding_workspace; import openhcs.microscopes.bioformats; print("imports-stable")'`
  -> `imports-stable`.
- Follow-up syntax plus focused suite:
  `.venv/bin/python -m py_compile openhcs/microscopes/bioformats.py openhcs/core/source_binding_workspace.py && .venv/bin/pytest -q tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_java_adapter.py tests/unit/test_bioformats_microscope_handler.py`
  -> `22 passed in 5.39s`.
- Scope now includes OME-TIFF through the same Bio-Formats OME-SPW path and
  OME-Zarr through the existing NGFF/Zarr/VFS ownership. No format branch will be
  added to generic source matching.
- Next: inventory OME-Zarr dataset identity and candidate projection, then complete
  format-neutral focused regressions before live UI/ZMQ acceptance.

### 2026-07-19 06:00 EDT - Generic authority moved; eager cycle eliminated

- Moved `SourceCandidate` to `openhcs.core.source_projection`, alongside
  `SourcePixelRef`-based source projections, and removed the duplicate declaration
  from `source_binding_workspace.py`.
- The proven eager cycle was broader than the temporary Bio-Formats local import:
  generic core workspace code imported both `microscopes.openhcs` for metadata
  writing and `microscopes.source_schema` for filename serialization. Importing any
  microscope submodule executes package discovery and therefore re-enters
  `bioformats.py`.
- Moved the OpenHCS metadata file schema, field identities, path resolver, and atomic
  writer into the existing core `virtual_workspace_metadata.py` owner. Added the
  small core `SourcePlaneFilenameCodec` used only by generic projection
  serialization. Core source projection/workspace modules now have zero microscope
  imports; microscope handlers consume those core declarations.
- Exact syntax/import gate:
  `.venv/bin/python -m py_compile openhcs/core/source_projection.py openhcs/core/source_binding_workspace.py openhcs/core/virtual_workspace_metadata.py openhcs/microscopes/openhcs.py && .venv/bin/python -c 'import benchmark.adapters.cellprofiler; import openhcs.core.source_binding_workspace; import openhcs.microscopes.bioformats; print("imports-stable")'`
  -> `imports-stable` with exit code 0.
- No lazy import, `try`/`except`, compatibility implementation, or fallback masks
  the dependency. Next: make Bio-Formats import the completed core projector normally
  while deleting its format-level dataset projector/writer layers.

### 2026-07-19 06:07 EDT - Adapter collapse at import-coherent checkpoint

- Deleted `bioformats_spw_projector.py`, all registered filename-layout parsers, the
  layout adapter, `BioFormatsDatasetAuthority`, completeness validator, and separate
  Bio-Formats workspace writer. `BioFormatsStoreMetadata.source_dataset()` now emits
  exact generic `SourcePlaneDataset` / `SourceCandidate` records directly; the
  handler only serializes that generic projection.
- Removed every production, unit, and integration import of
  `BioFormatsCompositeAdapter`, `BioFormatsSPWProjector`, and the deleted module.
  No alias, forwarding wrapper, or compatibility export was restored.
- Structured stale-consumer inventory command:
  `rg -n 'BioFormatsCompositeAdapter|bioformats_spw_projector|BioFormatsDatasetAuthority|BioFormatsWorkspaceMetadataWriter|BioFormatsSPWProjector|BioFormatsLayoutProjector|BioFormatsImageEntry|BioFormatsMetadata\\b|BioFormatsPlateIdentity' --glob '*.py' .`
  -> only the unrelated fake class name `FakeBioFormatsMetadata` remains.
- Syntax/import/collection checkpoint:
  `.venv/bin/python -m py_compile openhcs/core/source_projection.py openhcs/core/source_binding_workspace.py openhcs/core/virtual_workspace_metadata.py openhcs/microscopes/openhcs.py openhcs/microscopes/bioformats_adapter.py openhcs/microscopes/bioformats.py tests/unit/test_bioformats_java_adapter.py tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_microscope_handler.py && .venv/bin/python -c 'import benchmark.adapters.cellprofiler; import openhcs.core.source_binding_workspace; import openhcs.microscopes.bioformats; print("imports-stable")' && .venv/bin/pytest --collect-only -q tests/unit/test_bioformats_java_adapter.py tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_microscope_handler.py tests/integration/test_bioformats_imagexpress_synthetic.py`
  -> `imports-stable`; 16 tests collected; exit code 0.
- First execution run failed 9 tests at one exact boundary:
  `BioFormatsWellKeyAuthority` exposes `key_from_one_based`, while OME Well.Row and
  Well.Column are zero-based. Converted the explicit OME coordinates with `+1` at
  the Bio-Formats store decoder and retained the existing well-key authority.
- Focused rerun:
  `.venv/bin/pytest -q tests/unit/test_bioformats_java_adapter.py tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_microscope_handler.py`
  -> `15 passed in 4.41s`.
- Next: add NGFF store emission, cross-backend aggregation, named-binding/collision
  regressions, and Zarr materialization/reopen evidence through the same contract.

### 2026-07-19 06:15 EDT - Generic abstraction review and reduction

- Audited `SourcePlaneFilenameCodec` and found no independent state, dispatch, or
  policy. Deleted it. `OpenHCSPlaneAddress` now owns canonical filename construction
  and round-trip parsing; the registered `SourceSchemaFilenameParser` delegates to
  that address owner and retains only `FilenameParser` registry/interface behavior.
- Deleted `_ADDRESS_COMPONENTS`; address parsing/construction now iterates the
  authoritative `AllComponents` enum, while `OpenHCSPlaneAddress.component_values()`
  remains the typed address field projection.
- Deleted `_METADATA_COMPONENT_KEYS`; each existing enum-keyed
  `SourceComponentProjectionStrategy` leaf now declares its unique OpenHCS metadata
  collection field (`wells`, `sites`, `channels`, `z_indexes`, `timepoints`). The
  serializer queries the registered nominal leaf for every `AllComponents` member.
- A source workspace projector no longer invents a default parser. Materialization
  requires its caller's registered parser explicitly; core source-candidate and
  projection operations remain parser-independent.
- Deletion gate:
  `rg -n 'SourcePlaneFilenameCodec|_ADDRESS_COMPONENTS|_METADATA_COMPONENT_KEYS' openhcs tests`
  -> no matches.
- Syntax/import/test gate:
  `.venv/bin/python -m py_compile openhcs/core/source_projection.py openhcs/core/source_metadata.py openhcs/core/source_binding_workspace.py openhcs/microscopes/source_schema.py openhcs/microscopes/bioformats_adapter.py openhcs/microscopes/bioformats.py external/PolyStore/src/polystore/ome_zarr_storage.py && .venv/bin/python -c 'import benchmark.adapters.cellprofiler; import openhcs.core.source_binding_workspace; import openhcs.microscopes.bioformats; print("imports-stable")' && .venv/bin/pytest -q tests/unit/test_bioformats_java_adapter.py tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_microscope_handler.py tests/unit/test_source_projection.py`
  -> `imports-stable`; `27 passed in 6.27s`.

### 2026-07-19 06:29 EDT - Resume diff, peer, and advisor audit

- Re-read root `AGENTS.md`, the status/ownership/active-peer/executable-plan/parent-
  note sections of every non-complete `.agents` plan, the adjacent code-mode plan,
  and this ledger. No active peer owns `source_projection.py`, the store adapters,
  or the Bio-Formats handler; UI/code-mode and CellProfiler artifact/runtime files
  remain excluded.
- Audited the live owner-slice diff before editing. It currently modifies generic
  source projection/matching, collapses the Bio-Formats handler/adapter, deletes
  `bioformats_spw_projector.py`, and already contains an unverified
  `OmeZarrStoreAdapter`. `git diff --check` is clean; no unrelated worktree change
  was reverted or rewritten.
- Rebuilt the AST declaration inventory. The live store family is one
  `SourcePlaneStoreAdapter.__registry__` with manifest, OME-Zarr, and Java leaves;
  the generic aggregation authority is `SourcePlaneDataset.aggregate()`. There is
  no justification for a second format registry, source model, or axis catalog.
- Ran NominalRefactorAdvisor `e8a3c50` over the complete six-file owner slice. The
  package-context run timed out after 180 seconds without output; the exact
  no-auto-context scan completed in 5.14 seconds with 100 raw findings. Most are
  dataclass/opaque-annotation noise for this task. Actionable evidence is limited
  to the fail-soft `_relative_path()` exception fallback, manifest `.get()`
  defaults, and NGFF helpers that should live on the existing OME-Zarr store leaf.
  The advisor does not support adding a registry, wrapper, mirrored axis catalog,
  or format branch in generic matching.
- Next: inspect current store discovery and candidate aggregation against ordinary
  image and mixed-store fixtures, then make the smallest owner-local correction and
  add focused tests before UI/ZMQ acceptance.

### 2026-07-19 06:47 EDT - Unified store composition implementation checkpoint

- Replaced exclusive one-adapter selection with one nominal
  `SourcePlaneStoreAdapter` collection operation. Registered leaves now emit zero or
  more `SourcePlaneDataset` values; only the explicit fixture manifest may claim a
  collection exclusively, and `SourcePlaneDataset.aggregate()` remains the sole
  cross-store identity/collision authority.
- Added the minimum owner hooks needed for positive dispatch:
  `ImageFileFormat.requires_plane_store_decoder()` distinguishes ordinary TIFFs
  from metadata-rich OME-TIFF at the existing image-format owner, and PolyStore's
  `BioFormatsJavaContext.declares_path()` queries `ImageReader.isThisType()` while
  closing the probe reader. Java discovery no longer tries every file and ignores
  decoder errors.
- Added `ImageFileStoreAdapter`, which iterates `ImageFileFormat.__registry__`, emits
  exact scalar 2D disk planes for ordinary images, and rejects ambiguous stacks or
  color arrays. No image extension or filename policy was added to generic source
  matching.
- OME-Zarr traversal, topmost-store selection, plate/well/image traversal, and axis
  projection remain inside `OmeZarrStoreAdapter`. NGFF axes are validated by
  `ome_zarr.axes.Axes` and `format_from_version`; the former local supported-axis
  catalog was removed. Absent labels now remain `None` instead of synthesized
  channel/site/Z/time labels.
- Non-plate Java stores use the exact encoded container path as `WELL` and reader
  series as `SITE`, preventing repeated OME `Image:0` identities in independent CZI
  or TIFF containers from colliding. Generic collision keys are scoped by physical
  store identity, and collection-root identities are rebound only by the aggregate
  owner.
- Removed the advisor-identified `_relative_path()` fail-soft absolute-path return
  and manifest `.get(..., default)` reads. Paths outside the submitted collection
  now fail, while optional manifest values use explicit field-presence branches.
- Focused syntax/diff/test checkpoint:
  `python -m py_compile ...`; `git diff --check ...` -> clean;
  `pytest -q tests/unit/test_source_projection.py tests/unit/test_bioformats_spw_projector.py tests/unit/test_bioformats_java_adapter.py`
  -> `17 passed in 0.73s`.
- Real NGFF/VFS checkpoint on `/tmp/openhcs-czi-audit/fixture/zarr`: the adapter
  emitted one `Polystore_Plate` dataset with four exact `A01` sites, pixel size 1.0,
  and four `ome_zarr` refs. Normal `BioFormatsHandler` workspace initialization and
  `virtual_workspace` reads returned four `3648x3648 uint16` planes, each with sum
  2,748,541,748. This also exposed and fixed the valid case where one NGFF image's
  `field` axis spans several sample groups by including the field slice in the
  store-local image/series identity.
- Concrete changed files at this checkpoint:
  `openhcs/core/source_projection.py`,
  `openhcs/core/image_file_serialization.py`,
  `openhcs/microscopes/bioformats_adapter.py`,
  `external/PolyStore/src/polystore/bioformats_java.py`,
  `tests/unit/test_bioformats_java_adapter.py`, and
  `tests/unit/test_bioformats_spw_projector.py`. Existing unrelated hunks in these
  shared files remain preserved.
- Next executable step: add a durable real-format mixed PNG/TIFF/NGFF collection
  regression that verifies exact addresses, multiple backends, named binding, VFS
  reads, materialization, and reopen; then run the public OME-TIFF companion and
  real CZI fixtures through the same positive Java leaf before UI/ZMQ acceptance.

### 2026-07-19 06:58 EDT - Real positive leaves and mixed materialization checkpoint

- Added `tests/unit/test_plane_store_sources.py` with generated real NGFF plus
  scalar TIFF and PNG data. The unified collection emits exact logical wells and
  both `ome_zarr` and `disk` refs, routes automatic named bindings through
  `BioFormatsHandler`, and loads all arrays through the normal virtual workspace.
- Public OME-TIFF companion probe: all six physical TIFFs positively declare via
  `ImageReader.isThisType`; the collection emits `Plate:0`, five exact candidate
  planes, and five `96x96 int8` virtual reads with sum `-1076724`. Missing physical
  calibration is normalized by the Java store owner to explicit unit spacing;
  declared nonpositive calibration remains invalid.
- Licensed CZI probe: all four multipart paths positively declare; the collection
  emits one exact non-plate candidate at pixel size `1.8220764071160431`, preserves
  the encoded master-container identity, and loads a `3648x3648 uint16` plane with
  sum `2748293312`.
- Automatic microscope routing now gives a physically detected capable owner first
  refusal and uses `SourceBindingsMicroscopeHandler` only when no physical owner can
  project the declared bindings. Mixed NGFF/TIFF/PNG with aliases therefore stays
  on `BioFormatsHandler`.
- Materializing all three mixed planes to Zarr now succeeds using injective physical
  NGFF row/column coordinates owned by `BioFormatsFilenameParser`. The subsequent
  metadata update exposed one owner-local defect: the conversion generator
  rediscovers the entire mutable collection root after the derived `zarr` store
  exists, then correctly rejects conflicting source and derived dataset identities.
- Latest changed files additionally include `openhcs/microscopes/microscope_base.py`,
  `openhcs/microscopes/bioformats.py`,
  `tests/unit/test_bioformats_microscope_handler.py`, and
  `tests/unit/test_plane_store_sources.py`.
- Verification at this checkpoint: mixed VFS `1 passed in 0.83s`; Java/mixed pixel
  normalization `3 passed in 0.89s`; automatic-routing selection `4 passed, 40
  deselected in 2.60s`; scoped Ruff clean. Two broader handler assertions were
  stale expectations for intentionally absent labels and exact non-plate container
  identity; those tests are updated and await the next focused rerun.
- Next executable step: pass explicit pre-conversion grid/pixel metadata into the
  conversion metadata request, preserve exact alias/provenance mapping while
  rebasing to the materialized paths, and extend the durable mixed test through
  metadata update and exact reopen.

### 2026-07-19 07:03 EDT - Typed conversion metadata implementation

- Extended the existing `SourceProjectionMetadataSerializer` with a path prefix,
  so the nominal projection owner emits internally consistent `image_files`,
  `workspace_mapping`, `source_metadata`, and `source_projection` paths for a
  materialized subdirectory without caller-side field copying.
- Zarr conversion now reads the existing OpenHCS metadata owner, rebuilds every
  currently materialized typed source projection with an exact `zarr` ref, and
  writes the complete rebased metadata plus main-subdirectory transition in one
  atomic update. Output paths outside the declared store and outputs without a
  source projection fail explicitly.
- Inputs without typed source projections retain the ordinary metadata generator,
  but conversion supplies both grid dimensions and pixel size explicitly from the
  existing OpenHCS metadata. It no longer invokes format discovery on the mutable
  collection root, and incremental conversion no longer skips updates merely
  because one channel was already present.
- Post-edit gate:
  `.venv/bin/pytest -q tests/unit/test_plane_store_sources.py
  tests/unit/test_source_projection.py tests/unit/test_bioformats_java_adapter.py
  tests/unit/test_bioformats_spw_projector.py
  tests/unit/test_bioformats_microscope_handler.py tests/unit/test_function_io.py`
  -> `33 passed in 5.20s`; scoped Ruff, `py_compile`, and `git diff --check` pass.
- Changed files at this checkpoint additionally include
  `openhcs/core/steps/function_io.py` and `openhcs/microscopes/openhcs.py`.
- Next executable step: extend `test_plane_store_sources.py` through real Zarr
  writes, metadata transition, OpenHCS reopen, alias/projection assertions, and
  exact pixel reload; implement any owner-local failure it exposes.

### 2026-07-19 07:08 EDT - Discoverability scope and peer coordination

- Accepted the new mandatory `Biologist-Facing Discoverability` checklist before
  closure. The public surface will describe only `PipelineConfig`, named source
  bindings, sample/component identity, collision behavior, and automatic
  store-backed loading; adapter classes and storage traversal remain internal.
- Re-read `.agents/official30-knowledge-examples.md`. That active peer owns
  `knowledge_base_service.py`, manifest/docs registration entries, DTO/rendering
  tests, and MCP knowledge retrieval tests. This plan will implement the canonical
  source-store docstrings and one RST document, then communicate its path and
  plain-language retrieval queries through the peer plan rather than editing the
  same owner concurrently.
- Current implementation batch remains the durable mixed-store Zarr transition:
  extend the existing real NGFF/TIFF/PNG test through materialization, typed
  metadata rebasing, OpenHCS reopen, named aliases/components, and exact pixel
  reads. Documentation inventory and edits begin only after that gate is green.

### 2026-07-19 07:04 EDT - Built-in typed fixture and source-state gate

- Read `.agents/global-ui-zmq-config-tabs.md` and
  `.agents/builtin-testplate-mcp-e2e.md`. The live owner retains PyQt PID `1873825`,
  bridge port `7891`, its descriptor, and any future ZMQ/Napari children. This plan
  will not use or stop that process.
- The leased UI's normal `init_plate` failed before ZMQ in 3.24 seconds because the
  unchanged built-in ImageXpress metadata stores 216 string-valued
  `workspace_mapping` entries. `SourcePixelRef.from_workspace_mapping()` correctly
  rejects those legacy values; deserialization will remain strict.
- The synthetic fixture generator already has concurrent uncommitted changes that
  serialize `SourcePixelRef.to_workspace_mapping()` and a focused generated-fixture
  assertion. Preserve those changes. The current ignored/generated built-in plate
  is stale and must be regenerated or migrated at its fixture owner, then opened
  through the real typed virtual workspace in a focused regression.
- Added source-state acceptance: normal selected-plate initialization after saved
  `PipelineConfig.source_bindings_config` edits must rebuild the handler projection
  and project all component, alias, and backend-owned source identities into the
  existing canonical plate state. Pauli owns only the live UI save/reinitialize/
  visible-state proof; this plan owns rebuild semantics and focused tests.
- Next executable step: inspect the built-in fixture lifecycle, migrate/regenerate
  the stale metadata through that owner, add the typed-open regression, and report
  the green gate to the leased UI plan before auditing source-config invalidation.

### 2026-07-19 07:07 EDT - Built-in typed workspace gate passed

- Preserved the concurrent generator change that serializes every synthetic
  ImageXpress Z-stack mapping through `SourcePixelRef.to_workspace_mapping()`.
  Mechanically migrated the current ignored/generated built-in plate's 216 mapping
  values to that exact three-field wire shape; no deserializer or UI path changed.
- Extended the existing focused generator regression to initialize the generated
  plate with `OpenHCSMicroscopeHandler`, select `virtual_workspace`, list both
  virtual planes, and compare one virtual load exactly with its disk source.
  Result: `1 passed, 5 deselected in 0.73s`; scoped Ruff and diff checks pass.
- Direct probe of the exact built-in UI plate selected by the lease owner:
  `OpenHCSMicroscopeHandler`, input root `zstack_plate`, backend
  `virtual_workspace`, 216 listed planes, first load `256x256 uint16`, exact disk
  equality `True`.
- No live PyQt, bridge, ZMQ, or Napari process was started, reused, mutated, or
  stopped. The fixture gate is ready for Pauli's leased-process rerun.
- Changed files at this checkpoint include the focused additions in
  `tests/unit/test_synthetic_imagexpress_bioformats_compatibility.py`; the migrated
  current fixture is under ignored `tests/integration/tests_data/` and is runtime
  acceptance state, not a tracked production change.
- Next executable step: audit orchestrator/ObjectState source-binding configuration
  identity across save and initialization, implement handler/projection rebuild at
  that owner, and prove the complete canonical projection reaches plate state.

### 2026-07-19 07:13 EDT - Saved source state rebuild implemented

- `PipelineOrchestrator.apply_pipeline_config()` now compares the canonical saved-
  resolved `source_bindings_config`. A semantic change clears handler, input,
  component, and metadata state and returns the orchestrator to `CREATED`; changing
  source bindings during execution fails before mutating config.
- Invalidation retains only the nominal class of a handler that declares
  `projects_declared_source_bindings()`. The next ordinary `initialize()` rebuilds
  that class with the saved bindings, preventing its generated OpenHCS sidecar from
  changing auto-detection to a stale handler. No format name, UI branch, source
  field mirror, or fallback was added.
- Added a real mixed-store regression: initialize NGFF/TIFF/PNG with three aliases,
  save three renamed bindings, observe `CREATED` and cleared handler, reinitialize,
  and assert a new `BioFormatsHandler` publishes all sample/well, site, channel, Z,
  timepoint, alias, `disk`, `ome_zarr`, and distinct backend-address identities
  through `orchestrator.source_workspace_projection()` and component caches.
- Focused test: `1 passed, 1 deselected in 3.83s`. Broader source/orchestrator gate:
  `64 passed in 10.85s`. Test-file Ruff, production/test `py_compile`, and scoped
  `git diff --check` pass.
- Changed files at this checkpoint additionally include
  `openhcs/core/orchestrator/orchestrator.py`; the source-state regression is in
  `tests/unit/test_plane_store_sources.py`.
- Next executable step: hand this focused gate to Pauli's live lease, then return to
  the mixed Zarr materialization/reopen regression before beginning canonical
  docstring/RST discoverability edits.

### 2026-07-19 07:18 EDT - Durable mixed Zarr reopen gate passed

- Extended `tests/unit/test_plane_store_sources.py` through a real
  `PipelineOrchestrator` context and the production `save_materialized_data()` /
  `update_metadata_for_zarr_conversion()` path. NGFF, scalar TIFF, and scalar PNG
  planes are written into one Zarr plate under their exact canonical virtual paths.
- The conversion owner rebuilds the complete typed projection set with `zarr` refs,
  atomically marks the physical source subdirectory non-main, retains all three
  named aliases, and writes no caller-side source metadata copy.
- Normal auto reopen selects `OpenHCSMicroscopeHandler`, resolves `zarr` as the
  primary backend, exposes three typed source projections whose backend address
  equals each canonical virtual path, and reloads all `3x4 uint16` arrays exactly.
- Focused result: `1 passed, 2 deselected in 4.19s`; test Ruff, production/test
  `py_compile`, and scoped `git diff --check` pass.
- Next executable step: inventory the canonical user-doc and knowledge manifest
  owners, add concise load-bearing public docstrings and one executable RST source,
  then communicate that path/queries to the active knowledge owner before retrieval
  and live code-mode acceptance.

### 2026-07-19 07:20 EDT - Discoverability authority inventory

- Re-read the active knowledge and live-UI plans. The knowledge worker still owns
  `knowledge_base_service.py`, the existing JSON manifest, and MCP retrieval tests;
  Pauli owns the current GUI/bridge/ZMQ process and live save/reinitialize proof.
  This batch will not edit either peer's production or test boundary.
- `PipelineConfig.source_bindings_config`, `SourceBindingsConfig`,
  `NamedSourceBinding`, and `SourceSelector` are the public declaration path.
  `PipelineOrchestrator.source_workspace_projection()` is the canonical resolved
  plate-state view. `SourcePlaneDataset` and its nominal store leaves remain
  internal owners and will not become a format-specific tutorial API.
- The existing architecture source-model document is already registered in the
  knowledge manifest but is developer-oriented. One new canonical
  `guide_for_biologists/image_sources.rst` document will own the operational prose
  and executable declaration examples for TIFF/PNG, CZI, OME-TIFF, OME-Zarr, and
  mixed stores; its toctree will link that same file rather than copying content.
- Changed files remain the implementation/test files listed below plus this plan;
  no documentation or knowledge-owner file has been edited in this batch yet.
- Next executable step: edit the public docstrings and canonical RST/toctree, then
  run code-block syntax and focused docs gates before sending its exact path and
  retrieval queries to `.agents/official30-knowledge-examples.md`.

### 2026-07-19 07:22 EDT - Canonical source guide and docstrings implemented

- Expanded only the public `SourceSelector`, `NamedSourceBinding`,
  `SourceBindingsConfig`, and
  `PipelineOrchestrator.source_workspace_projection()` docstrings. They describe
  named sources, exact sample/component/store identity, owner-declared singleton
  axes, collision failures, and the one canonical UI/compiler/runtime projection;
  no adapter or backend API became public.
- Added one canonical biologist-facing document,
  `docs/source/guide_for_biologists/image_sources.rst`, and linked that same file
  from the existing biologist toctree. One executable declaration block constructs
  TIFF/PNG, CZI, OME-TIFF, OME-Zarr, and mixed-store `PipelineConfig` values through
  the identical source-binding API used by code mode. A second executable block
  selects exact well/channel coordinates through `ComponentSelector`.
- Focused documentation validation passes for one RST file and two Python blocks;
  both blocks also compile and execute against the current checkout. Production
  `py_compile` and scoped diff checks pass. Whole-file Ruff on the shared
  orchestrator still reports seven pre-existing unused-import/forward-annotation
  findings outside the docstring hunk; `source_bindings.py` will be checked
  independently in the next static batch.
- Changed files at this checkpoint additionally include
  `openhcs/core/source_bindings.py`,
  `docs/source/guide_for_biologists/image_sources.rst`, and the surgical toctree
  edit in `docs/source/guide_for_biologists/index.rst`.
- Next executable step: write the exact document path, manifest metadata, and four
  plain-language acceptance queries into the active knowledge owner's Parent Notes,
  then poll that owner and run focused retrieval/function-detail evidence without
  editing its active service/manifest/test boundary concurrently.

### 2026-07-19 16:02 EDT - Store/format and knowledge lane completed

- Re-read root guidance, every current active ownership boundary, this complete
  ledger, and the assigned `.agents/source-store-format-completion.md` plan. A
  fresh stdlib-AST inventory parsed all owned production and focused-test files and
  recorded every format/store declaration, helper, method, and focused call site.
- Deleted the remaining owner-local ceremony: the module-level component collector,
  duplicate metadata-handler-class assignment, one-use NGFF dataset and physical
  source-path forwarders, and the optional Java scalar wrapper nominal. The four
  production owners fell from 2,177 lines / 71 top-level declarations to 2,135 /
  69; the previously deleted SPW projector/writer/authority family remains absent.
- Added deterministic regressions for one CZI container, two independent CZI
  containers sharing an explicit Plate ID, physical provenance, exact
  cross-container address collision rejection, and OME-TIFF diversion from the
  ordinary TIFF leaf. Existing real mixed NGFF/TIFF/PNG tests cover named aliases,
  VFS, exact component identity, typed Zarr transition, and reopen.
- Registered `docs/source/guide_for_biologists/image_sources.rst` once through the
  canonical knowledge manifest. Six retrieval queries cover CZI, OME-TIFF,
  OME-Zarr/NGFF, mixed stores, named channels, and component metadata; the retrieved
  source is the exact guide, not copied prose.
- Real fixture acceptance used the documented CC BY 4.0 CZI multipart set and the
  public OME-TIFF companion set. The CZI collection emits one non-plate candidate,
  binds alias `DNA`, and VFS-loads `3648x3648 uint16` pixels with sum 2,748,293,312.
  The OME-TIFF collection emits five Plate:0 candidates across A02/B01/B03/C02 and
  VFS-loads `96x96 int8` pixels with sum -1,076,724. The four CZI files are multipart
  members and declare no OME Plate, so neither automatic plate-bearing discovery nor
  a real independent-container pair is claimed.
- The complete owned format/store/knowledge batch passes (`67 passed in 8.53s`),
  the corrected virtual-workspace runtime preload integration passes, six
  validation/function-I/O tests pass, the guide validates both executable blocks,
  and scoped syntax/Ruff/JSON/diff/deletion gates are clean. A separate synthetic
  ImageXpress Java integration still fails when its generated `plate.HTD` opens with
  a Java `NullPointerException`; no fallback or generic runtime edit was made.
- Live UI/ZMQ/Napari acceptance remains with the active MCP worker. Parent Notes
  reserve the official30 lock for the remaining suite; this lane records but does
  not run `.venv/bin/python scripts/mcp_thesis_demo_live.py --max-run-seconds 240`.
- Final combined rerun: scoped `py_compile`, Ruff, manifest JSON, guide execution,
  parent/submodule diff checks, and `74 passed in 16.26s`.

#### Final Production Inventory

The pre-collapse inventory captured at 05:30/05:40 was 1,960 format-level lines:
`bioformats_adapter.py` 1,048, `bioformats_spw_projector.py` 364, and
`bioformats.py` 548. Final format/store code is 1,691 lines:
`bioformats_adapter.py` 1,292, deleted projector 0, `bioformats.py` 270, and the new
PolyStore OME-Zarr leaf reader 129. This is a net reduction of 269 lines while
adding NGFF decoding, ordinary TIFF/PNG store emission, cross-store aggregation,
typed provenance, and independent-container validation. The final assigned
four-owner cleanup inventory (including `image_file_serialization.py` and the Java
bridge) is 2,135 lines / 69 top-level declarations, down from 2,177 / 71 at worker
start; scoped diff and stale-symbol gates are clean.

## Commands and Results

- `sed -n '1,260p' AGENTS.md` -> read complete repository guidance.
- `find .agents -maxdepth 1 -type f -print | sort` -> 27 active coordination files inventoried.
- `git status --short --branch` -> shared tree has extensive pre-existing edits; no cleanup or revert performed.
- `rg -n '__registry__|AutoRegisterMeta|RegistryConfig|RegistryFamily|MostDerivedContextStrategyMixin|NominalTypeKeyedStrategyMixin|EnumKeyedStrategyMixin' openhcs` -> existing format, microscope, metadata, source, artifact, and strategy authorities located.
- `find /home/ts -type f -iname '*.czi' -print` -> no local CZI fixture found.
- Source/Bio-Formats code reads -> confirmed normalized virtual paths plus structured physical plane references and manifest-only current tests.
- `.venv/bin/pytest -q tests/unit/test_bioformats_storage_backend.py tests/unit/test_bioformats_microscope_handler.py tests/integration/test_bioformats_handler_runtime.py` -> `14 passed, 1 failed in 7.01s`; exact preload failure recorded above.
- OpenHCS-first polystore probe -> local source package selected; `.czi` is unknown to `FileFormat`; Bio-Formats backend imports successfully.
- Java/dependency probe -> OpenJDK 26 present; `pyimagej`/`scyjava` absent; Zarr 2.18.7 and ome-zarr present.
- `uv pip install --python .venv/bin/python 'pyimagej>=1.4.1' 'scyjava>=1.9.1'` -> installed the declared optional runtime; final versions pyimagej 1.8.0, scyjava 1.12.5, jpype1 1.7.1.
- `wc -c -- *.czi && md5sum -- *.czi` under the audit cache -> each file 7,395,712 bytes; MD5 values `0540181a...`, `78c5a509...`, `82f314db...`, and `d76f5843...`, all matching Zenodo.
- Java reader metadata probe -> Zeiss Quick Start reader, one series/image/plane, zero plates, `3648x3648 C1 Z1 T1`; plane load `uint16`, min 138, max 308, sum 2,748,541,748.
- `BioFormatsDatasetAuthority().project(master_or_directory)` without the temporary manifest -> `BioFormatsAdapterUnavailableError`: no manifest, no Java OME-SPW plate, and no supported filename layout.
- Explicit-manifest multi-file probe -> virtual paths `A01_s001...tif` through `A01_s004...tif`, four distinct structured source refs, four successful VFS loads.
- Source-binding probe over the Bio-Formats projection -> aliases `[None, None, None, None]`; filter paths are plane-ref JSON; all 16 candidate/binding comparisons false.
- Two-file Java-adapter probe -> `{'visited': ['plate-a.czi'], 'returned_plate': 'plate-a.czi'}`.
- Production-context Zarr command -> four raw exact comparisons true; metadata main flags `{'.': false, 'zarr': true}`; auto-reload uses `OpenHCSMicroscopeHandler`, input `/tmp/openhcs-czi-audit/fixture/zarr`, backend `zarr`; four reload exact comparisons true.
- Reopened-Zarr binding probe -> `is_image_path` true for all four logical TIFF paths and a diagonal four-site match matrix.
- `.venv/bin/pytest -q tests/unit/test_czi_zarr_source_audit.py` -> `1 passed in 0.38s`.
- `.venv/bin/pytest -q tests/unit/test_czi_zarr_source_audit.py tests/unit/test_bioformats_*.py tests/unit/test_function_io.py` -> `32 passed in 3.90s`.
- `.venv/bin/pytest -q tests/unit/test_czi_zarr_source_audit.py tests/unit/test_bioformats_microscope_handler.py tests/unit/test_bioformats_storage_backend.py tests/integration/test_bioformats_handler_runtime.py` -> `15 passed, 1 failed in 5.29s`; only the known direct-backend preload test failed.
- `.venv/bin/ruff check tests/unit/test_czi_zarr_source_audit.py` -> all checks passed. Whole-file lint of `openhcs.py` still reports pre-existing unused import, undefined annotation, and late-import findings outside this audit's hunk.

## Remaining Verified Limits and Owners

- **Live code-mode/ZMQ acceptance.** Owner:
  `.agents/source-store-code-mode-zmq-acceptance.md`, coordinated with the active
  MCP process owner. The focused decoder, projection, knowledge, and non-live wire
  surfaces are available; live UI/ZMQ/Napari execution remains serialized behind
  the parent's official30 lock and must not be duplicated by this lane.
- **Physical CZI fixture limit.** The licensed real CZI decoder fixture is one
  multipart, non-plate container. No available small licensed CZI declares an OME
  Plate or supplies two independent plate-bearing containers. The deterministic
  equal-Plate-ID aggregation and collision tests are green, but automatic physical
  plate-bearing CZI discovery must not be claimed until such fixtures exist.
- **Java runtime warning.** pyimagej downloads Java 11 while the resolved Fiji
  `DefaultFijiService` is class-file version 65 (Java 21). Fiji service
  initialization logs errors, although the real CZI and OME-TIFF readers and plane
  loads complete successfully. Packaging/version alignment remains an external
  environment-support issue.
- **Synthetic ImageXpress fixture.** The generated `plate.HTD` integration is
  positively detected by Bio-Formats but currently opens with a Java
  `NullPointerException`. This is separate from the requested CZI/OME/TIFF/PNG/NGFF
  acceptance and received no fallback or generic runtime patch.

## Changed Files

- `docs/plans/czi_source_bindings_and_zarr_audit_20260719.md` (new audit ledger).
- `openhcs/core/source_projection.py` (generic collection identity rebinding,
  store-scoped collision validation, exact filename-safe component tokens, and
  projection-owned materialized path prefixes).
- `openhcs/core/steps/function_io.py` (typed source-projection rebasing for Zarr
  conversion and explicit conversion calibration).
- `openhcs/core/orchestrator/orchestrator.py` (saved source-binding changes
  invalidate and rebuild the existing nominal projection-capable handler; its
  public projection docstring defines the canonical consumer view).
- `openhcs/core/source_bindings.py` (canonical public selector, named-binding, and
  pipeline source-config docstrings; concurrent source-model edits preserved).
- `openhcs/core/image_file_serialization.py` (existing image-format owner now
  declares when embedded TIFF metadata requires a plane-store decoder).
- `openhcs/microscopes/bioformats_adapter.py` (unified collection composition and
  Java, OME-Zarr, ordinary-image store leaves).
- `openhcs/microscopes/bioformats.py` (injective NGFF physical coordinates for
  exact logical source-container identities).
- `openhcs/microscopes/microscope_base.py` (physical store owner gets first refusal
  before the explicit source-binding handler override).
- `openhcs/microscopes/openhcs.py` (metadata generation request accepts paired,
  explicit grid/pixel calibration for conversion without source rediscovery).
- `external/PolyStore/src/polystore/bioformats_java.py` (positive Bio-Formats path
  declaration using the owning Java reader; optional scalar wrapper ceremony
  collapsed into the existing projector).
- `openhcs/microscopes/openhcs.py` (this audit changed only parser discovery to iterate the authoritative `FilenameParser.__registry__`; unrelated pre-existing edits in this file were preserved).
- `tests/unit/test_bioformats_java_adapter.py` and
  `tests/unit/test_bioformats_spw_projector.py` (focused positive-dispatch and
  non-plate container/series regressions).
- `tests/unit/test_bioformats_microscope_handler.py` (exact absent-label and
  non-plate container expectations).
- `tests/unit/test_plane_store_sources.py` (real generated mixed NGFF/TIFF/PNG
  aggregation, aliases, routing, VFS, and saved-config handler-rebuild regression).
- `tests/unit/test_synthetic_imagexpress_bioformats_compatibility.py` (built-in
  generator's typed virtual-workspace open/load regression; concurrent fixture
  serialization edits preserved).
- `tests/unit/test_czi_zarr_source_audit.py` (new regression for Bio-Formats parser restoration from a Zarr-main OpenHCS plate).
- `tests/integration/test_bioformats_handler_runtime.py` (preload through the
  virtual-workspace directory/listing owner rather than the opaque plane reader).
- `docs/source/guide_for_biologists/image_sources.rst` (single canonical
  biologist-facing store/source-binding guide and executable code-mode examples).
- `docs/source/guide_for_biologists/index.rst` (links the canonical image-source
  guide from the existing toctree).
- `docs/source/development/mcp_knowledge_base_manifest.json` (one canonical image
  source guide registration) and `tests/unit/agent/test_knowledge_base_service.py`
  (six store/source retrieval queries plus exact document assertions).
- `.agents/source-store-format-completion.md` (live completion evidence and active
  process/ownership coordination).
- `.venv` received already-declared optional Bio-Formats packages; it is ignored and no dependency file changed.
- `/tmp/openhcs-czi-audit/fixture` contains the downloaded fixture, temporary manifest, generated OpenHCS metadata, and generated Zarr store; none are repository files.

## Verification

- Real CZI decoding, four-file structured addressing, virtual VFS reads, Zarr writes, raw reads, metadata transition, OpenHCS reload, and post-reload selector matching were all executed.
- The current unified OME-Zarr leaf emitted and loaded all four planes from the real
  generated NGFF fixture through the normal handler and virtual workspace.
- Current focused generic/store tests pass: `17 passed in 0.73s`; syntax and scoped
  diff checks are clean.
- All six public OME-TIFF companion paths and all four licensed CZI multipart paths
  pass positive Java declaration; unified projection and real plane reads pass with
  the shapes, pixel sizes, identities, and sums recorded at 06:58.
- Mixed NGFF/TIFF/PNG aggregation, named aliases, physical auto-routing, and virtual
  reads, Zarr writes, typed metadata transition, auto reopen, and exact Zarr reads
  pass in the durable mixed regression.
- Post-conversion-owner focused gate: `33 passed in 5.20s`; scoped Ruff,
  `py_compile`, and `git diff --check` pass.
- Saved source-state gate passes for the mixed store and built-in fixture; broader
  source/orchestrator suite: `64 passed in 10.85s`.
- The canonical image-source guide passes `scripts/validate_docs.py` (`1 files, 2
  Python blocks`), and both blocks compile and execute against the live API.
- The final combined format/store/knowledge/runtime-preload gate passes `74 passed
  in 16.26s`;
  scoped syntax, Ruff, JSON, parent/submodule diff, and stale-helper deletion gates
  pass. The corrected runtime preload integration and six supplemental
  Bio-Formats/function-I/O tests pass separately.
- Fresh licensed CZI VFS acceptance loads `3648x3648 uint16` with sum
  `2748293312`; fresh public OME-TIFF companion VFS acceptance loads `96x96 int8`
  with sum `-1076724` from a five-plane Plate:0 projection.
- The direct-address backend/directory mismatch is closed by the corrected
  virtual-workspace runtime preload integration; the remaining synthetic
  ImageXpress Java fixture failure is recorded under external limits above.
- No CellProfiler pipeline import/module artifact files, code-mode worker files, or
  source-binding workspace files were changed in this resumed batch.
