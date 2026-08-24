# Changelog

All notable changes to OpenHCS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Closing the GUI while its execution endpoint is still starting now cancels
  and reaps the exact connection attempt without leaving catalog processes or
  reporting owner-requested teardown as an asynchronous failure.

## [0.7.26] - 2026-08-24

### Changed

- The package-index README is now a required source-backed documentation-audit
  surface, including validation of its Python examples and repository links.

### Fixed

- The project overview no longer advertises the removed embedded LLM assistant;
  agent-assisted authoring is presented through the maintained MCP surface.

## [0.7.25] - 2026-08-24

### Changed

- Agent-assisted pipeline authoring now uses the same local MCP capabilities as
  external clients, keeping generated pipelines visible and reviewable through
  the existing GUI and Python document model.

### Fixed

- macOS installer cancellation now waits for and terminates the exact worker
  process deterministically instead of racing installer completion.
- Local IPC cleanup tolerates libzmq removing its socket path between endpoint
  shutdown and filesystem cleanup.

### Removed

- Removed the embedded LLM chat panel and host-side LLM service registry from
  OpenHCS and PyQT-reactive.

## [0.7.24] - 2026-08-23

### Changed

- Runtime framework selection, callable discovery, memory types, image axes,
  artifacts, and backend replay now derive from their typed declarations across
  source, compiled, worker, viewer, UI, and MCP boundaries.
- ZMQ compilation and execution use one retained client lifecycle, bounded
  submission deadlines, cooperative terminal cancellation, and snapshot-owned
  endpoint presentation.
- Viewer lifecycle ownership now preserves persistent viewers across execution
  teardown while retaining exact non-persistent process cleanup.
- Source checkouts, built wheels, and extracted packages now have explicit
  dependency-source boundaries, published release floors, and installed-package
  acceptance tests.
- Documentation now has source-backed editorial proofs and one canonical
  development setup procedure.
- Release CI now requires exact-commit documentation and integration evidence,
  source-inclusive coverage, immutable action pins, and native Windows and macOS
  installer acceptance before publication.

### Fixed

- Externally terminated execution and viewer children are observed and reaped
  by their exact ZMQRuntime process owner without waiting for later polling.
- CPU-only startup no longer performs GPU-backed function discovery.
- Windows and macOS installers, staged desktop updates, and state-preserving
  restarts retain the installed environment authority across process changes.
- Pipeline code documents preserve declaration identity and saved ObjectState
  history across complete-document edits and live window updates.
- Large MCP responses stream without line-size truncation, and headless jobs
  retain execution progress through terminal status.
- Bio-Formats runtimes are disposed before process exit, portable Numba math is
  selected before compilation, and Zarr discovery remains backend-owned.
- Cold Fiji startup retries bounded artifact-resolution failures before the JVM
  starts, while preserving the original failure once the retry budget is spent.

### Removed

- Removed the deprecated terminal UI and its version mirrors.
- Removed dataset-specific BBBC microscope handlers superseded by source
  bindings.

## [0.7.23] - 2026-08-14

### Changed

- Function discovery now prewarms the execution endpoint at application startup
  and projects the remote catalogue asynchronously, keeping the selector
  responsive during cold imports.
- Napari layers retain one declaration-owned semantic axis layout across every
  streamed pipeline route, with singleton slots for dimensions reduced by a
  processing step.

### Fixed

- Step code mode preserves live callable values, and pipeline code mode commits
  complete-document edits directly to the saved ObjectState baseline.
- Reset and provenance feedback reaches the nested input whose value changed,
  rather than flashing only its containing form.
- Reapplying an unchanged Qt theme avoids redundant native repolishing, fixing
  a Windows application-stylesheet crash.
- Napari sliders preserve the actual plate domains for site, channel, Z index,
  timepoint, and well instead of allowing a reduced route to inflate unrelated
  axes.

## [0.7.22] - 2026-08-13

### Changed

- Complete pipeline code documents now reconcile steps by their declarations,
  preserving the matching step and function state across reordering and edits
  while replacing stale or removed bodies.
- Configuration help, dimensionality guidance, and measurement capabilities are
  projected consistently across the desktop UI, documentation, and MCP tools.

### Fixed

- Managed Windows launches now resolve the active replacement environment at
  startup and restart through the stable native launcher, preventing shortcuts
  from retaining deleted environment paths after an in-application update.
- Windows desktop startup uses the windowed Python runtime and one Qt-owned
  startup window, avoiding a console window and duplicate loading windows.
- The GUI detects incompatible ZMQ server versions and offers a state-preserving
  server and application restart; code-mode pipelines without an explicit
  ``PipelineConfig`` use the declared default configuration.
- Callable identity now owns backend and CellProfiler classification, preventing
  unrelated functions with similar names from being compiled as CellProfiler
  modules.
- Special artifacts remain outside image-axis projection, and template-matching
  outputs retain their complete declared result columns.

## [0.7.21] - 2026-08-08

### Fixed

- Viewer streaming now consumes the single transport mode owned by ZMQRuntime's
  endpoint declaration, restoring Windows installer workflows after the
  endpoint-lifecycle unification without a compatibility mirror.

## [0.7.20] - 2026-08-08

### Fixed

- Log-tail shutdown now uses one monotonic stop-request authority, preventing
  rapid log switches or application cleanup from deadlocking during QThread
  startup on macOS.

## [0.7.19] - 2026-08-08

### Changed

- Endpoint discovery, startup presentation, the server browser, and the footer
  indicator now derive from one immutable observation snapshot instead of copied
  connection flags.
- Transport, connection, shutdown, startup-phase, and execution-status behavior
  is owned by nominal declarations in the generic runtime.
- Compilation and execution progress visibility now derives from the latest
  runtime projection and clears as soon as no active plate work remains.

### Fixed

- Execution servers appear in the browser while they are still importing and
  preparing their function catalog, before their control endpoint answers PING.
- Busy local execution servers remain connected when a periodic PING times out,
  using exact process identity as the liveness proof; killing that exact endpoint
  removes the row and disconnects the client and footer together.
- Control responses and terminal execution-state checks no longer duplicate wire
  fields or status membership in consumer-owned tables.

## [0.7.18] - 2026-08-08

### Changed

- ZMQ endpoint startup now reports typed preparation and connection phases, and
  the GUI presents those phases while cold function catalogs are prepared.
- Log and server browsers now correlate live process identities with their
  owning log files instead of selecting an unrelated first match.
- Native Qt theming now covers inherited menus, previews, progress bars, and
  boolean editors; the system monitor layout and brand assets were refined for
  the release desktop.

### Fixed

- Windows managed environments now use compact transactional identities directly
  under the stable install root so the full desktop dependency set installs under
  the default Windows path limit. Updates still recognize and remove environments
  created by older installers.
- Desktop launchers route Numba's generated runtime cache through the stable,
  compact install root instead of writing beside long installed module names.
- Desktop launch and update flows preserve GUI-subsystem startup, repair stale
  projections, and avoid eager application catalog construction.

## [0.7.17] - 2026-08-06

### Fixed

- Installer-facing extracted-package dependencies now use stable releases, and
  release validation rejects prerelease floors that installer-owned `uv` cannot
  resolve during an update without an explicit prerelease policy.
- The Windows installer smoke test now exercises the same installer-owned `uv`
  update-resolution boundary used by previously installed OpenHCS versions.
- Fresh MCP clients no longer initialize optional array runtimes through
  PolyStore's package exports, keeping stdio startup within its timeout budget.
- Redocking a resized fixed-height system monitor now reapplies its embedded
  constraint before restoring the saved workspace geometry.

## [0.7.16] - 2026-08-05

### Added

- The connected ZMQ execution server now owns function-catalog discovery and
  exposes typed catalog, search, and detail requests to the GUI. Endpoint and
  catalog revisions invalidate the GUI's derived presentation automatically.
- Managed Windows installs now use one stable native GUI-subsystem launcher,
  one authoritative current-environment pointer, and a continuous native-to-Qt
  startup splash handoff without opening a console window.

### Changed

- Code editors carry the authored declaration's nominal type through generic
  editor, LLM, and serialization protocols instead of maintaining a parallel
  set of string document kinds.
- Library registries declare discovery configuration on their nominal root.
  Cache validation now detects added, changed, and removed source files.
- GUI startup no longer initializes its own runtime function catalog. Desktop
  installation prepares the persistent endpoint catalog, while a genuinely cold
  server performs discovery in an isolated helper process and reports typed
  preparation progress without blocking its control endpoint.

### Fixed

- Persisted custom-function additions, edits, and removals now reconcile the
  live execution-server catalog from the custom-function source authority.
- Lazy package submodules and lazy module exports are imported through Python's
  module protocol, restoring scikit-image 0.26 discovery in clean installs.
- Windows updater and reinstall flows repair the stable native launcher,
  desktop shortcut, MCP launcher, and current-environment pointer while keeping
  progress visible outside the environment being replaced.
- Windows launcher compilation now uses the PowerShell 5.1 CodeDOM contract
  available on supported Windows hosts.
- Runtime-tested library discovery now retains only callables whose canonical
  main output satisfies the authoritative array-payload contract, preventing
  plotting and other non-image utilities from entering worker registries.

## [0.7.1] - 2026-07-30

### Added

- The desktop GUI can update its exact virtual environment in place, restart,
  and restore the complete plate-manager document, UI configuration, and typed
  ObjectState history. The detached updater runs outside the target
  environment so Windows can replace active environment files safely.
- The update check is available through the existing typed main-window MCP
  action surface.

### Changed

- UI configuration forms now retain annotated validation metadata while using
  dedicated key-sequence and finite color controls instead of free-text
  editing.
- Placeholder and enabled-state styling is applied as fields materialize
  instead of appearing only after a large form finishes constructing.

### Fixed

- ObjectState 1.1 history persistence now preserves paths, enums, dataclasses,
  callable objects, shared identity, the active timeline position, and unsaved
  typed state across application restarts.
- Function-pattern add and reset operations no longer fail on invalid transient
  editor text, and selectors support registered plate-scoped functions that do
  not consume image arrays.
- Compact UI and MCP parameter help render the owning type of annotated
  parameters without exposing raw `typing.Annotated` representations.
- ZMQ submission timeouts now cover progress-stream registration as well as
  execution dispatch, preventing installer smoke runs and slow-starting
  runtimes from falling back to an unrelated five-second handshake limit.

## [0.7.0] - 2026-07-30

### Changed

- Closed configuration and registered-function choices now use their nominal
  enum types throughout the UI, schema, and runtime. CellProfiler text
  conversion is confined to its import and source-setting boundaries.
- Napari and Fiji display configuration is declared by concrete typed
  dataclasses that also own viewer wire projection. Redundant display payload
  adapters and generated field-description factories were removed.
- Inert or misleading public options were removed instead of retaining
  compatibility shims, including unused visualization dtype, streaming batch
  size, Fiji executable path, custom aggregation, and singleton database
  choices.
- ObjectState now owns direct-child topology and structural reindexing across
  updates, checkpoints, and time travel. Synthesized lazy and injected config
  classes preserve their authoritative declaration documentation.
- PolyStore viewer backends now consume the typed display contract directly
  rather than reconstructing backend-specific payload mirrors.

### Added

- MCP architecture-symbol errors now explain that the symbol namespace is
  curated, provide live near matches, and direct clients to architecture topic
  discovery.
- MCP config-patch errors now report unknown nested fields with suggestions
  derived from the reflected dataclass authoring schema.
- Generic annotation validation now rejects invalid worker counts, viewer
  ports, blank hosts and registry names, and non-positive Z spacing at the
  owning configuration boundary.

### Performance

- Configuration form construction now indexes ObjectState topology once and
  reuses immutable construction metadata. In the representative 148-field
  benchmark, median declared-field completion fell from 400.57 ms to
  330.20 ms.
- The first-party Napari ROI manager now uses a virtual table over the native
  Shapes layer instead of rebuilding eager table widgets. Binding 4,097 ROIs
  fell from 490 ms to 1.07 ms in the focused benchmark, while preserving native
  geometry, features, colors, and selection ownership.

### Fixed

- Napari streaming now applies configured colormap and component display
  semantics without duplicating ROI or layer state.
- Dynamic configuration classes retain docstrings and validation metadata, so
  UI and MCP help describe the actual declarations rather than generated
  placeholders.
- Raw ObjectState reconstruction now preserves registered lazy runtime types,
  preventing unresolved inherited viewer settings from being validated as
  concrete values during pipeline compilation.
- The portable installed neurite demo now supplies typed percentile settings,
  and wheel integration fixtures preserve nested lazy-config identity.
- CellProfiler grid imports now retain their nominal shape and diameter choices
  through artifact-contract planning.

## [0.4.0] - 2025-11-05

### Added

#### Sequential Component Processing
- **Complete implementation of pipeline-wide sequential processing** ([#43](https://github.com/trissim/openhcs/pull/43))
  - Process images across multiple component combinations (e.g., all channels × all z-slices)
  - Per-process backend isolation for parallel execution safety
  - Automatic conflict detection and filtering for variable components
  - Memory clearing between combinations to prevent data leakage
  - Comprehensive validation tests and API improvements
  - Moved configuration from per-step to global `PipelineConfig` level
  - Self-describing backend pickling architecture eliminating duck typing

#### GUI Enhancements
- **Multi-directory plate selection** - Select and process multiple plate directories simultaneously in PyQt GUI
- **Git worktree testing documentation** - Comprehensive guide for testing with git worktrees

#### Configuration System
- **Lazy config merging improvements** - Proper None value resolution in nested dataclasses
- **GlobalPipelineConfig vs PipelineConfig field defaults** - Fixed inheritance and default value handling
- **Placeholder resolution fixes** - Multiple improvements to lazy config context propagation
- **List[Enum] placeholder styling** - Visual feedback when values match inherited defaults

### Fixed

#### Critical Bug Fixes
- **ZMQ viewer instance browser** - Fixed lazy config port scanning to properly detect streaming ports from `PipelineConfig`
- **Race condition in LazyDiscoveryDict** - Fixed concurrent plate initialization issues
- **Pipeline step deletion** - Properly handles duplicate step names
- **Registry isolation** - Multiple fixes for proper component registry isolation
- **Sequential processing component mismatch** - Fixed validation and execution bugs
- **Missing image detection** - Improved error handling and reporting

#### Configuration & UI Fixes
- **Reset parameter not updating placeholders** - Fixed nested form placeholder refresh
- **List[Enum] checkbox race conditions** - Fixed save/load issues and glitchy behavior
- **Component button syncing** - Fixed live form value propagation and group_by changes
- **Lazy resolution for live form values** - Use simple temp object instead of dataclass replace
- **Checkbox group placeholder comparison** - Use enum names instead of values
- **Enabled field styling** - React to context changes like placeholders do

#### Backend & Processing Fixes
- **OMERO backend reconstruction** - Fixed pickling and worker process issues
- **Backend inheritance hierarchy** - Eliminated problematic multiple inheritance
- **Nested dataclass config merging** - Resolve None values before converting to base
- **Enum cache** - Include SequentialComponents in cache
- **Sequential processing memory clearing** - Only clear files from current combination

#### Code Quality & Architecture
- **Eliminated duck typing** - Refactored to use ABCs and explicit interfaces throughout
- **Hardcoded config field accesses** - Replaced with generic merged config pattern everywhere
- **Module-level executable code** - Wrapped in `__main__` guards
- **Code export** - Handle enum lists properly

### Changed

#### Architecture Improvements
- **Backend pickling** - Self-describing architecture with explicit `PicklableBackend` ABC
- **Configuration hierarchy** - Proper GlobalPipelineConfig → PipelineConfig → StepConfig inheritance
- **Registry system** - Improved isolation and caching (9.7x startup improvement)
- **Enum generation caching** - 2800x speedup for colormap generation

#### Refactoring
- **Processing config consolidation** - Unified into single `ProcessingConfig` dataclass
- **Sequential component validation** - Filter conflicts instead of raising errors
- **Compiler config resolution** - Use merged config for global well_filter_config

### Documentation
- Added comprehensive development style review and extraction recommendations
- Updated code review documentation to reflect ABC usage instead of Protocol
- Improved API compatibility documentation after processing_config refactor

---

## [0.3.15] - 2024-10-XX

### Fixed
- Windows unicode escape issue in napari and fiji detached process spawning
- Improved test assertions based on code review feedback
- Applied black formatting and fixed ruff linting issues

---

## Previous Versions

See git history for changes in versions 0.3.14 and earlier.

[0.4.0]: https://github.com/trissim/openhcs/compare/v0.3.15...v0.4.0
[0.3.15]: https://github.com/trissim/openhcs/releases/tag/v0.3.15
[0.7.0]: https://github.com/OpenHCSDev/OpenHCS/compare/v0.6.17...v0.7.0
[0.7.24]: https://github.com/OpenHCSDev/OpenHCS/compare/v0.7.23...v0.7.24
[0.7.25]: https://github.com/OpenHCSDev/OpenHCS/compare/v0.7.24...v0.7.25
[0.7.26]: https://github.com/OpenHCSDev/OpenHCS/compare/v0.7.25...v0.7.26
