# CellProfiler-Like Debug Mode Plan

## Goal

Add an OpenHCS debug mode that gives users the CellProfiler-style development loop:

- Pick a well/image set.
- Run one module/step/invocation at a time.
- Pause before selected modules.
- Inspect inputs, outputs, labels, measurements, and timing.
- Change settings and rerun from the affected point.
- Keep the same UI/runtime path used by normal OpenHCS pipelines.

This should not be a CellProfiler-only execution engine. CellProfiler should get richer module-specific displays through the same debug substrate that can also support Fiji, napari, OMERO, and custom OpenHCS functions.

## What CellProfiler Test Mode Means

CellProfiler Test Mode is not just verbose logging. It is an interactive execution mode for pipeline construction:

- A user enters test mode from the GUI.
- The pipeline can step to the next module.
- A module can have a pause marker.
- The user can run until the next pause.
- The user can choose a different image set or group.
- Module output windows show intermediate results.
- Module settings can be changed and the pipeline rerun from an earlier point.

The OpenHCS equivalent should map this onto existing OpenHCS concepts:

- CellProfiler module -> OpenHCS function invocation inside a `FunctionStep`.
- CellProfiler image set/group -> OpenHCS well/axis, source-binding axis, pattern group.
- CellProfiler module outputs -> OpenHCS artifacts and source-bound runtime values.
- CellProfiler display window -> OpenHCS debug view backed by napari/Qt/table widgets.

## Current OpenHCS Surfaces To Reuse

### Runtime Boundary

The best runtime seam is the function invocation boundary inside `FunctionStep` execution:

- `FunctionStepExecutor` owns step-level orchestration.
- `FunctionStepExecutionPlan` already exposes the compiled function pattern, artifact inputs, artifact outputs, source-binding plan, streaming configs, axis, step name, and step index.
- `PatternGroupRuntime` loads source images, executes the compiled invocation chain, validates/unpacks outputs, saves outputs, and cleans up.

Debug mode should instrument this path. It should not fork a separate CellProfiler runner.

Fact-check: step-level instrumentation alone is not sufficient. The executable invocation list lives in `CompiledFunctionPattern` / `CompiledFunctionInvocation`, and `PatternGroupRuntime` is where pattern groups are loaded, executed, validated, saved, and cleaned up. A CellProfiler-like "module" maps most cleanly to a compiled invocation, not necessarily to a `FunctionStep`.

### Artifact Boundary

OpenHCS already has typed artifact plans:

- Image artifacts.
- Object label artifacts.
- Measurement artifacts.
- Relationship artifacts.
- Materialization policy.

Debug snapshots should store references to artifacts and small preview summaries, not blindly retain full arrays in memory.

### UI Boundary

The PyQt GUI already has:

- Pipeline editor.
- Step editor.
- Function pattern editor.
- Plate/image browser.
- Plate viewer.
- Streaming integration with napari/Fiji.

Debug mode should add a debug panel/window to these surfaces rather than building a CellProfiler clone.

### Current GUI/Worker Communication Path

The normal PyQt execution path is already ZMQ-centric:

- `PlateManager.action_run_plate()` delegates to `BatchWorkflowService.run_plates()`.
- `BatchWorkflowService` connects a `ZMQExecutionClient` to the execution server on port `7777`.
- The execution server compiles and executes the submitted pipeline.
- The execution server starts worker processes through `PipelineOrchestrator.execute_compiled_plate()`.
- Workers emit typed `ProgressEvent` dictionaries into a multiprocessing queue.
- `ZMQExecutionServer._forward_worker_progress()` validates worker ownership, enriches topology metadata, and forwards events to the server progress queue.
- The ZMQ execution server publishes progress to connected GUI clients.
- `BatchWorkflowService._on_progress()` and `ZMQServerManagerWidget._on_progress()` parse `ProgressEvent` and register it in the shared progress registry.
- The GUI updates from the shared progress registry/projection.

Workers should not call Qt widgets directly. In normal GUI execution, worker-to-GUI communication should go through the ZMQ execution server or through a separate viewer server.

### Current Viewer Communication Path

napari and Fiji are already separate viewer/server processes:

- Step streaming configs are collected during compilation and stored on the compiled step plan.
- The orchestrator creates viewer instances via `get_or_create_visualizer()`.
- Streaming viewers are managed through `zmqruntime.get_or_create_viewer(...)`.
- The worker writes display payloads through the filemanager using the viewer backend, e.g. `napari_stream` or `fiji_stream`.
- The viewer server receives ZMQ payloads and renders them.
- The PyQt image browser already uses the same route for manually streaming selected images/ROIs to napari/Fiji.

Debug visualization should reuse this for image/label layers. The debug inspector should remain the semantic source of truth; napari should render selected snapshot layers.

## Proposed Core Abstractions

### `DebugSession`

A `DebugSession` is a short-lived controller for one pipeline, one selected plate/well/image-set scope, and one execution cursor.

Fields:

- `pipeline_id`
- `plate_path`
- `axis_id`
- `selected_source_group`
- `cursor`
- `breakpoints`
- `snapshot_store`
- `dirty_from_cursor`

Responsibilities:

- Start/stop debug mode.
- Step one invocation.
- Step one `FunctionStep`.
- Run until pause/breakpoint.
- Select next/random/chosen image set.
- Recompile or partially invalidate after setting edits.

### `DebugCursor`

The cursor should be invocation-aware, not only step-aware:

- `step_index`
- `group_key`
- `invocation_key`
- `pattern_group_identity`

This matters because a single `FunctionStep` can contain a chain or grouped pattern of multiple functions. CellProfiler imports may map one module per step today, but OpenHCS debug mode should support native OpenHCS function patterns correctly.

### `DebugSnapshot`

A snapshot records what happened at one execution boundary.

Fields:

- `cursor`
- `step_name`
- `callable_name`
- `kwargs_summary`
- `axis_id`
- `source_paths`
- `input_artifact_refs`
- `output_artifact_refs`
- `measurement_summary_refs`
- `relationship_summary_refs`
- `timing`
- `exception`
- `preview_refs`

The snapshot should store artifact references and compact previews. Full arrays should be loaded lazily on demand.

The snapshot must distinguish three payload classes:

- Source inputs selected by `StepSourceBindingsConfig`.
- Runtime artifact inputs/outputs managed by artifact plans.
- Pure measurement/relationship records that may not have image-like previews.

That distinction matters because CellProfiler modules often produce measurements without producing a new image or object label artifact.

### `DebugEventSink`

Runtime code should emit typed events:

- `before_step`
- `before_invocation`
- `after_invocation`
- `after_step`
- `artifact_produced`
- `measurement_written`
- `exception`

The default sink is no-op. Debug mode installs a sink that records snapshots and forwards UI updates.

This keeps production runtime clean and avoids debug conditionals scattered through CellProfiler code.

The sink should be attached to execution context or compiled execution request state, not global module state. Worker processes and ZMQ server runs may overlap, so debug state must be keyed by `execution_id` / `debug_session_id`.

## Storage Strategy

Use OpenHCS storage concepts instead of an ad hoc cache.

The debug store should support:

- In-memory small previews.
- VFS/PolyStore-backed artifact references for large arrays.
- TTL cleanup.
- Per-session namespace.
- Optional persistence for “save debug report”.

Do not force every intermediate output to become a normal pipeline materialized output. Debug materialization is different: it is ephemeral, selective, and viewer-oriented.

Status: the first storage and transport slice is implemented. `DebugSnapshotStore` is the nominal store family, with `LocalDebugSnapshotStore` for local filesystem metadata and `FileManagerDebugSnapshotStore` for VFS/FileManager-backed metadata. `ProcessingContext` explicitly registers as the debug snapshot FileManager context, so worker policies can choose the shared store without structural probing.

Snapshot readback now has two concrete paths. Local and shared-namespace callers can read through the store abstraction directly. Remote GUI/server callers can use the ZMQ control-channel `DebugSnapshotReadRequest`/`DebugSnapshotReadResponse` path, which asks the execution server to read the snapshot from the declared local or FileManager-backed store and return metadata to the GUI. Preview payload materialization remains artifact/viewer work, not a progress-event concern.

Recommended shape:

```python
@dataclass(frozen=True, slots=True)
class DebugArtifactRef:
    kind: ArtifactKind
    name: str
    scope: DebugCursor
    storage_ref: object
    preview_ref: object | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
```

## CellProfiler-Specific Views

Add a registered renderer family for CellProfiler debug displays:

```python
class CellProfilerDebugView(metaclass=AutoRegisterMeta):
    module_name: ClassVar[str | None] = None

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        ...
```

Examples:

- `IdentifyPrimaryObjects`: show input image, label image, object outlines, object count, threshold stats.
- `IdentifySecondaryObjects`: show parent labels, child labels, propagation/mask overlay.
- `CorrectIlluminationCalculate`: show source image, illumination function, corrected preview when available.
- `CorrectIlluminationApply`: show before/after image.
- `Align`: show fixed/moving images, aligned output, displacement/quality measurements.
- `MeasureObjectIntensity`: show table summary and optional object-colored overlay.
- `MeasureTexture` / `MeasureGranularity`: show measurement table summary and selected feature plots.
- `RelateObjects`: show parent/child overlay and relationship counts.
- `FilterObjects`: show before/after labels and removal summary.
- `MaskObjects`: show source labels, mask, masked labels.
- `MeasureColocalization`: show image pair preview, scatter plot, coefficient table.

The default view should still work for any OpenHCS function:

- Inputs.
- Outputs.
- Artifact table.
- Timing.
- Logs/errors.

CellProfiler renderers should be additive.

Renderer input should be a generic `DebugSnapshot`, plus optional CellProfiler module provenance. The renderer registry should never execute CellProfiler code or read `.cppipe` files directly; execution and source lowering remain in the runtime/compiler layers.

## UI Design

### Pipeline Editor Additions

Add a “Debug/Test Mode” toggle.

When enabled:

- Show a cursor marker beside the next invocation/step.
- Show pause markers beside steps; individual function-pattern invocation markers are a presentation extension over the implemented invocation cursor.
- Disable full production run controls that conflict with the active debug session.
- Add controls: `Step`, `Run`, `Run to Next Pause`, `Restart`, `Choose Well/Image Set`, `Random Image Set`.

The controls should live in the existing pipeline editor/plate workflow area, not in a disconnected window:

- A compact debug toolbar at the top or bottom of `PipelineEditorWidget`.
- Per-step pause/cursor affordances in the existing step list.
- Optional expansion for per-function-invocation pause markers inside the function pattern editor.

The pipeline editor already persists steps through `ObjectState` and stable step scope IDs. Debug state should key against those step scope IDs so editing, undo/redo, and selection remain coherent.

### Debug Inspector

Add a dock/window with:

- Current step/invocation.
- Input artifacts.
- Output artifacts.
- Measurements.
- Relationships.
- Timings.
- Exception traceback.
- Module-specific display tabs.

For CellProfiler pipelines, this becomes the CP-like module output window. For native OpenHCS functions, it is still useful.

The inspector should be a separate dockable/managed window, but launched and controlled from the pipeline editor:

- Left/top: cursor and execution controls.
- Middle: invocation list and snapshots.
- Right/tabs: rendered views for images, labels, measurements, relationships, timing, logs.
- “Send to napari” / “Send to Fiji” buttons for selected debug artifacts.

This avoids overcrowding the step list while keeping the debugger attached to the pipeline editor workflow.

### Viewer Integration

Use existing viewers:

- Qt widgets for tables, scalar summaries, and thumbnails.
- napari for images, labels, overlays, and multi-layer inspection.
- Fiji as optional image display, not the primary state authority.

The debug inspector should own the semantic state. Viewers should render selected snapshot layers.

Do not make napari mandatory. Recommended split:

- Qt inspector is primary for cursor, settings, measurements, relationships, timings, and errors.
- napari is primary for interactive image/label/overlay visualization.
- Fiji is optional for users who prefer ImageJ-style inspection or ROI workflows.

Debug mode can auto-open napari when enabled, but should also work headless or Qt-only by storing snapshot refs/previews.

## Debug Transport Design

Progress and debug payloads should be related but not identical.

Use `ProgressEvent` for lightweight execution state:

- Current debug cursor.
- Step/invocation started/completed.
- Pause/breakpoint reached.
- Error state.
- Small message strings.

Do not put full snapshots or arrays into progress events. `ProgressEvent.context` can carry small cursor metadata, but not image/table payloads.

Use a separate debug snapshot store/query path for payloads:

- Worker records snapshot artifacts/previews into a debug store namespace.
- Worker emits a progress/debug event containing only the `debug_session_id`, `snapshot_id`, and cursor.
- GUI receives the event through the existing ZMQ progress client.
- GUI requests/loads the snapshot from the debug store when the inspector needs to render it.

The first implementation uses a local filesystem/VFS-backed debug store under the output/debug namespace and exposes an explicit ZMQ snapshot-read RPC for remote GUI/debugging reads. Large payloads still stay behind artifact refs and viewer/materialization paths.

## Worker Control Model

There are two different modes.

### Simple Debug Run

For the first implementation, debug mode should run a single selected axis/well with one worker and a debug cursor policy:

- Compile normally.
- Restrict execution to the selected well/image-set.
- Execute until the next requested boundary.
- Persist snapshots.
- Return control to the GUI.

Each `Step` command can be implemented as a short execution request with a starting cursor and stop cursor. This is simpler and safer than keeping a worker blocked while waiting for UI input.

### Interactive Persistent Worker

Later, a persistent debug worker could remain alive and wait for commands:

- `step`
- `run_until_pause`
- `select_image_set`
- `rerun_from_cursor`
- `stop`

That requires a command channel from GUI -> execution server -> worker. It should not be the first slice unless restart cost is unacceptable.

The execution server should remain the broker. The GUI should not connect directly to worker processes.

## Runtime Integration Plan

1. Add no-op debug event sink support to `ProcessingContext`.

Implemented. `openhcs.core.debug` now defines `DebugSession`, invocation-aware `DebugCursor`, `DebugSnapshot`, `DebugArtifactRef`, `DebugEvent`, and debug sinks. `ProcessingContext` owns a default `NO_OP_DEBUG_EVENT_SINK`, so normal execution remains inert unless a caller installs a real sink before context freeze.

2. Emit events in `FunctionStepExecutor` and `PatternGroupRuntime`.

Started at the real semantic boundary: compiled function invocation execution inside `openhcs.core.steps.function_runtime.execute_function_chain`. The runtime now emits `BEFORE_INVOCATION`, `AFTER_INVOCATION`, and `EXCEPTION` events with cursor identity, callable name, axis, and timing/traceback data. Step and pattern-group progress events can layer on top, but invocation events are the key boundary for CellProfiler-like module stepping.

3. Record debug snapshots.

The first implementation can capture:

- Step/invocation identity.
- Source paths.
- Artifact input/output refs.
- Timing.
- Exceptions.

Implemented. `DebugSnapshotStore` is now the store abstraction for snapshot metadata. `LocalDebugSnapshotStore` writes/reads JSON snapshot metadata under a per-session directory and maintains a manifest of snapshot IDs. `FileManagerDebugSnapshotStore` provides the same contract through an OpenHCS `FileManagerLike` backend. `LocalSnapshotProgressDebugEventSink` now accepts any `DebugSnapshotStore`, so worker debug events can write metadata snapshots through the local or FileManager-backed path before emitting progress with the resulting `snapshot_id`. This is intentionally metadata/ref-only; large arrays and tables should remain artifact refs or preview refs.

4. Add artifact preview extraction.

Build small preview records for images/labels/tables. Avoid full-array retention by default.

Implemented for metadata refs. Runtime debug events now project invocation-selected artifact plans into cursor-aware `DebugArtifactRef` values through `DebugArtifactRefProjection`. Snapshots therefore carry input refs, output refs, measurement refs, and relationship refs without loading or retaining artifact payload arrays. Rich thumbnail/table preview materialization remains viewer/export work layered on top of those refs.

5. Add PyQt debug session controls.

Start with invocation-level runtime events and present them through step-level GUI controls where that is clearer.

Started the inspector/control side. `DebugInspectorWindow` renders the renderer-independent debug view model and defaults to the CellProfiler debug-view registry. `DebugToolbarWidget` is now mounted in `PipelineEditorWidget` and emits typed `DebugCommand` values from the core debug model. The buttons are intentionally command-surface only for now; execution dispatch still goes through the bounded ZMQ debug request path.

Command transport has its first bounded execution slice: `DebugExecutionConfig` is carried through normal ZMQ execute params, and `submit_debug_pipeline(...)` installs a progress-backed debug sink in worker contexts. This intentionally reuses normal execution and progress transport rather than adding a CellProfiler-only runner.

The worker sink install path is now a nominal request boundary. `DebugSinkInstallRequest` owns the worker context, execution id, plate id, worker slot, and owned-well provenance needed by `DebugExecutionPolicy.install_context_sink(...)`, so no-op and progress policies consume one coherent request instead of parallel semantic parameters.

The first GUI bridge is now wired for bounded debug run submission. `PlateManagerWidget.action_run_debug_plate(...)` creates a `DebugSession`, selects a local `.openhcs_debug` snapshot store beside the plate, and delegates to `BatchWorkflowService.run_debug_plate(...)`. `PipelineEditorWidget` routes toolbar `Step`, `Run`, `Run to Pause`, `Restart`, `Choose Source Group`, and `Random Source Group` through that same plate-manager boundary with a typed `DebugCommandType` carried in `DebugExecutionConfig`; `Stop` uses the existing stop/force-kill path. Cursor-limited replay semantics can now be added inside the bounded execution policy without changing GUI or ZMQ request shape.

Snapshot readback now has a local, store-generic, and server-mediated GUI path. `DebugInspectorWindow.load_snapshot(...)` reads metadata from `LocalDebugSnapshotStore`, while `load_snapshot_from_store(...)` accepts any `DebugSnapshotStore`, including FileManager/VFS-backed stores supplied by a host that owns the same namespace. For remote GUI/server separation, `ZMQExecutionClient.get_debug_snapshot(...)` sends `DebugSnapshotReadRequest` over the control channel and receives a typed `DebugSnapshotReadResponse`. `BatchWorkflowService` attaches that snapshot to `DebugSnapshotAvailableNotification` when available, so `PipelineEditorWidget.show_debug_snapshot(...)` does not have to reconstruct server-local storage paths.

Selected snapshot artifacts now have typed viewer/export handoffs. The inspector builds an artifact action section from output, preview, and input artifact refs, discovers napari/Fiji targets from the existing streaming registry, and emits `DebugArtifactOpenRequest` values for viewers plus `DebugArtifactMaterializeRequest` values for host-side export/materialization. `PipelineEditorWidget` connects those inspector signals when the reusable inspector is created; export requests ask for a destination directory and route through `PlateManagerWidget.action_export_debug_artifact(...)` to the existing ZMQ debug artifact export control path. Viewer open requests are now a host-level typed seam; deeper viewer-specific streaming behavior can be added without changing the inspector model.

GUI progress handling now has a typed snapshot-availability seam. `BatchWorkflowService` still registers every debug `ProgressEvent` in the shared progress tracker, but it also parses `DebugProgressContext` and emits `DebugSnapshotAvailableNotification` to listeners when a snapshot ID is present. This keeps snapshot subscription out of raw ZMQ dictionaries and gives the inspector/controller a nominal event to consume.

The inspector loop is now wired through the normal widgets. `PlateManagerWidget` exposes `debug_snapshot_available` as a Qt signal, and `PipelineEditorWidget.show_debug_snapshot(...)` consumes that nominal notification, uses the attached server-read snapshot when present, otherwise falls back to store loading, and reuses a single `DebugInspectorWindow`. Bounded `STEP` and `RUN_TO_PAUSE` now have worker-policy semantics: `DebugStepStopStrategy` is the registered command strategy family, `StepDebugStepStopStrategy` stops after the configured start step, and `RunToPauseDebugStepStopStrategy` stops at step indices marked with `FunctionStep.debug_pause`.

Bounded `STEP` also works inside native OpenHCS function patterns, not only one-function imported CellProfiler steps. `DebugInvocationExecutionStrategy` is the registered command strategy family for invocation execution. For `DebugCommandType.STEP`, `StepDebugInvocationExecutionStrategy` skips invocations until the optional `start_after_invocation_key`, executes one compiled invocation, emits its `AFTER_INVOCATION` event/snapshot, then stops the chain. `PipelineEditorWidget` threads the current cursor's invocation key through `DebugExecutionConfig.start_after_invocation_key`, so repeated `Step` commands advance through functions inside the same `FunctionStep`.

Debug axis selection is now owned by `DebugExecutionPolicy`, not by GUI-side filtering. Bounded debug executions compile and run one axis by default, or the explicit `selected_source_group` when provided. The same `DebugPlateRunRequest` config is submitted for compile-before-run and execution, so compile artifact signatures match and debug mode does not accidentally compile every well before executing a one-axis command.

The remaining gap is richer preview payload rendering and GUI ergonomics. Cursor-bounded short executions are implemented: `DebugExecutionConfig.start_step_index` and `start_after_invocation_key` let restart/step commands resume at the correct step/invocation boundary without requiring GUI-side filtering or a CellProfiler-only runner. Warm replay now validates skipped upstream artifact outputs before claiming reuse, can hydrate local or FileManager/VFS-backed outputs from prior snapshot artifact refs by logical artifact identity, verifies producer/settings identity, records content digests in snapshot refs when payloads are readable, rejects stale payloads when content changes, and fails loudly when required warm outputs are missing.

During verification, a registry-state leak surfaced after `.cppipe` corpus imports initialized the global OpenHCS function registry: direct calls to registry-wrapped `PURE_2D` CellProfiler functions were forced through stack slicing. The fix is at the contract seam, not the module test: `Pure2DInputSlicer` now recognizes already-2D arrays and image payloads, and `LibraryRegistryBase._execute_pure_2d(...)` bypasses slice/restack for single-plane inputs. This keeps direct backend calls and pipeline stack execution consistent.

6. Add CellProfiler debug view registry.

Implement default views first, then add high-value CP renderers for segmentation, measurement, correction, alignment, and relationships.

Implemented for the generic view substrate and CellProfiler renderer families. `openhcs.core.debug_views` now owns the renderer-independent `DebugViewModel`/section/table row types. `DebugViewTable.from_artifact_refs(...)` is the shared artifact-ref table projection, and `DebugViewTable.from_invocation_parameters(...)` renders JSON-safe invocation kwargs captured in snapshots. `openhcs.interop.cellprofiler.debug_views` defines only the registered `CellProfilerDebugView` family and declarative `CellProfilerDebugSectionSpec` section tables. The default renderer exposes source paths, invocation parameters, input refs, output refs, preview refs, measurements, relationships, timing, and errors from generic snapshots. CellProfiler category renderers now include artifact-overview tables for image/object/display/export families so thumbnail/table-heavy modules get useful summaries without module-specific runtime code. Renderers consume generic `DebugSnapshot` only; they do not execute CellProfiler code or parse `.cppipe` files.

The debug model now also has nominal class-family witnesses for the common runtime boundary and control-request families. `DebugBoundaryState` is an `AutoRegisterMeta` root for `DebugEvent` and `DebugSnapshot`, and `DebugSessionRequest` is an `AutoRegisterMeta` root for snapshot read, artifact export, worker command, and progress-event request records. This makes the debug model family membership explicit instead of relying on implicit dataclass inheritance.

7. Add rerun/invalidation.

When a setting changes, mark snapshots downstream of the edited step/invocation dirty. Allow rerun from the dirty cursor.

Implemented for bounded replay. `DebugSession` now owns immutable cursor updates and `dirty_from_cursor` invalidation through `with_cursor(...)` and `mark_dirty_from_cursor()`. The pipeline editor keeps the current debug session state after loading a debug snapshot and marks it dirty when the pipeline changes, so stale downstream snapshots are explicitly identified instead of silently reused. `Restart` uses the dirty cursor's step index as `DebugExecutionConfig.start_step_index`, and `Step` uses the current cursor plus `start_after_invocation_key` to advance through function-pattern invocations. Reusing prior artifact materialization is now explicit: warm replay validates expected upstream outputs and may hydrate local files or FileManager/VFS payloads from prior snapshots with matching artifact identity before skipping a step.

## Minimal First Slice

The smallest useful version:

- Run one selected well/image set.
- Step by compiled invocation inside a `FunctionStep`.
- Record before/after invocation events and exceptions.
- Record artifact refs in a debug snapshot store.
- Show outputs in a debug inspector.
- Stream selected image/label outputs to napari.
- Preserve normal pipeline execution semantics.

The implementation started directly at invocation-level events because that is the real OpenHCS semantic boundary. Step-level controls can still present a coarser UI by grouping invocation cursors under a step.

## Risks

- Memory blowup if debug mode stores full arrays. Mitigation: lazy refs plus previews.
- Confusing semantics if debug mode only works for generated CellProfiler pipelines. Mitigation: core `DebugSession` plus CP-specific views.
- Incorrect rerun behavior if upstream settings change but downstream artifacts remain stale. Mitigation: explicit dirty cursor and downstream invalidation.
- Overcoupling to napari. Mitigation: snapshots are the source of truth; napari is one renderer.

## Acceptance Criteria

- Normal runs have no behavior change when debug mode is disabled.
- A user can choose a well/image set and step through a pipeline at compiled-invocation granularity, including multiple functions inside one `FunctionStep`.
- The UI shows which step/invocation will run next.
- Intermediate images, labels, measurements, and relationships can be inspected.
- CellProfiler modules get familiar output displays for common module families.
- Editing a step marks downstream debug snapshots dirty and reruns from the edited step boundary; artifact-reuse warm replay validates/hydrates local and FileManager/VFS upstream outputs before skipping reused steps.
- The system works for native OpenHCS functions, not only CellProfiler modules.
