# Debug Mode GUI/Worker Integration Plan

## Goal

Add a CellProfiler-like debug/test mode to OpenHCS without creating a second execution stack.

The debugger should:

- Integrate into the existing PyQt pipeline editor.
- Communicate through the existing ZMQ execution server path.
- Reuse existing napari/Fiji viewer servers for image/label visualization.
- Store debug snapshots through an OpenHCS/VFS-backed debug store.
- Work for CellProfiler pipelines and native OpenHCS function patterns.

## Existing Communication Model

Current GUI execution already has the right basic shape:

- `PlateManager.action_run_plate()` calls `BatchWorkflowService.run_plates()`.
- `BatchWorkflowService` connects a `ZMQExecutionClient` to the execution server.
- `ZMQExecutionServer` compiles and executes the pipeline.
- `PipelineOrchestrator.execute_compiled_plate()` starts worker lanes.
- Workers emit `ProgressEvent` dictionaries into a multiprocessing progress queue.
- `ZMQExecutionServer._forward_worker_progress()` validates/enriches worker progress and forwards it to ZMQ clients.
- PyQt receives progress through `BatchWorkflowService._on_progress()` and registers it in the shared progress registry.

The debugger should use this same route. Workers should never call Qt widgets directly.

### Fact-check notes

The named seams exist today:

- `openhcs/runtime/zmq_execution_server.py` defines `ZMQExecutionServer` and `_forward_worker_progress`.
- `openhcs/runtime/zmq_execution_client.py` defines `ZMQExecutionClient`.
- `openhcs/core/orchestrator/orchestrator.py` defines `PipelineOrchestrator.execute_compiled_plate`.
- `openhcs/core/progress/types.py` defines the serializable `ProgressEvent`.
- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py` parses incoming progress with `ProgressEvent.from_dict`.

Do not assume debug payload transport is solved by progress forwarding alone. The current path is verified for progress/status events. Snapshot retrieval now has a concrete local store path and a FileManager/VFS-backed store abstraction, but remote GUI/server reads still need an explicit shared namespace or RPC path.

## Existing Viewer Model

Viewer streaming is already separate from GUI execution:

- Streaming configs are collected during compilation.
- The orchestrator creates/reuses viewer processes with `get_or_create_visualizer()`.
- napari/Fiji viewer processes are managed through `zmqruntime`.
- Workers stream image/ROI payloads to viewer backends through the filemanager.
- The PyQt image browser already manually streams selected images/ROIs to napari/Fiji.

Debug mode should reuse this. napari is a renderer for selected debug artifacts, not the debugger state owner.

### Fact-check notes

The existing viewer path is real:

- `PipelineCompiler._collect_streaming_configs` stores streaming configs on compiled step plans.
- `PipelineOrchestrator.get_or_create_visualizer` calls `zmqruntime.get_or_create_viewer` for streaming visualizers.
- `openhcs/core/steps/function_outputs.py` and `function_artifact_materialization.py` write streaming outputs through the filemanager.
- `openhcs/ui/shared/streaming_service.py` is the reusable GUI-side service used by the image browser for manual streaming.

The first debug implementation should reuse the GUI-side streaming service for "send selected snapshot artifact" when possible. If the snapshot is only available in a worker-private debug namespace, add a resolver that can materialize or copy it into a path the streaming service can read.

## UI Design

### Pipeline Editor

Add a compact debug toolbar to `PipelineEditorWidget`:

- `Debug/Test Mode` toggle.
- `Step`.
- `Run`.
- `Run to Pause`.
- `Restart`.
- `Choose Well/Image Set`.
- `Random Image Set`.
- `Stop`.

Status: the command surface exists. `openhcs.pyqt_gui.widgets.debug_toolbar.DebugToolbarWidget` renders these controls and emits typed `DebugCommand` objects. `PipelineEditorWidget` mounts the toolbar and enables it only when the current plate is initialized, matching the existing pipeline-edit constraints. Toolbar `Step`, `Run`, `Run to Pause`, `Restart`, `Choose`, and `Random` route through `PlateManagerWidget.action_run_debug_plate(...)` with their `DebugCommandType` preserved in `DebugExecutionConfig`. The first command creates a `DebugSession` and submits a persistent-paused-worker debug run; later `Step`, `Run`, and `Stop` commands for the same plate route to the active worker through the typed `WORKER_COMMAND` control path, and `Stop` clears the active session. This command-lifetime behavior now has GUI-level unit coverage in addition to the ZMQ control-loop tests.

Bounded debug compile/run identity is now one nominal request, `DebugPlateRunRequest`, in the GUI workflow service. It produces the `DebugExecutionConfig` used for both compile-before-run and execution submission, so the ZMQ compile artifact signature is stable and debug executions do not silently reuse an all-well compile artifact for a one-axis debug command.

Add visual annotations to the existing step list:

- Current cursor marker.
- Completed/dirty status.
- Pause marker.
- Error marker.

Use existing `ObjectState`/step scope IDs to associate debug state with steps. Do not key debug state only by list index; list index changes under editing/reordering.

Status: pause markers are represented on `FunctionStep.debug_pause` and displayed in the step list. Dirty debug state has a nominal core authority: `DebugSession.mark_dirty_from_cursor()` records the cursor invalidated by subsequent pipeline edits, and `PipelineEditorWidget.on_pipeline_changed(...)` updates the active debug session state after a snapshot has established a cursor. Rich per-step dirty badges and rerun-from-cursor are still replay/UI slices rather than transport gaps.

### Function Pattern Editor

The runtime slice now steps at compiled-invocation granularity inside a `FunctionStep`; the GUI can still display that as a step-level command when the function pattern has only one invocation.

Current function-pattern editor presentation work:

- Per-invocation model/text badges are implemented for the pipeline preview and function-pattern editor.
- Invocation kwargs are now carried in `DebugSnapshot.invocation_parameters` and rendered by the inspector.
- Future polish is richer visual badge styling and per-invocation pause editing inside the function-pattern editor.

This matters because a native OpenHCS `FunctionStep` can contain a chain or grouped pattern of many callables.

### Debug Inspector

Add a docked/managed `DebugInspectorWindow`.

Tabs:

- `Summary`: cursor, step, invocation, well/image set, timing.
- `Inputs`: source paths and artifact inputs.
- `Outputs`: artifact outputs and preview buttons.
- `Images`: thumbnails and “send to napari/Fiji”.
- `Objects`: label previews, counts, outlines.
- `Measurements`: table previews and export/open actions.
- `Relationships`: relationship summaries.
- `Logs`: progress messages and exceptions.

The inspector should subscribe to debug/progress state and request snapshots lazily.

## Runtime Model

### Debug Session

Add a `DebugSession` model:

- `debug_session_id`
- `execution_id`
- `plate_id`
- `axis_id`
- `selected_image_set`
- `cursor`
- `breakpoints`
- `snapshot_store_ref`
- `dirty_from_cursor`

Status: implemented as the core model in `openhcs.core.debug`.

### Debug Cursor

Cursor should support both coarse and fine stepping:

- `step_index`
- `step_scope_id`
- `group_key`
- `invocation_key`
- `pattern_group_identity`

Status: implemented as `DebugCursor`, with `DebugCursor.from_invocation(...)` deriving stable invocation identity from `CompiledFunctionInvocation`.

### Debug Snapshot

Snapshot should record references, not full payloads:

- `snapshot_id`
- `cursor`
- `source_paths`
- `input_artifact_refs`
- `output_artifact_refs`
- `preview_refs`
- `measurement_refs`
- `relationship_refs`
- `timing`
- `exception`

Large arrays/tables stay in VFS/local debug storage and are loaded only when the inspector/viewer needs them.

Status: snapshot metadata, artifact-ref types, and store/write paths exist.

Update: `DebugSnapshotStore` is the nominal store family. `LocalDebugSnapshotStore` provides the local filesystem-backed write/read path for snapshot metadata, and `FileManagerDebugSnapshotStore` provides the same contract through OpenHCS `FileManagerLike` backends. `LocalSnapshotProgressDebugEventSink` accepts the store abstraction: when a bounded debug execution includes a `snapshot_store_ref`, worker debug events are written as metadata snapshots and progress events announce the `snapshot_id`. Shared VFS readback is available through the store abstraction when the GUI has the same FileManager namespace. Remote GUI/server readback is implemented through the ZMQ control-channel `DebugSnapshotReadRequest`/`DebugSnapshotReadResponse` path.

## Transport Model

### Progress Events

Use `ProgressEvent` for lightweight UI state:

- Debug session started/stopped.
- Cursor moved.
- Step/invocation started.
- Step/invocation completed.
- Pause reached.
- Snapshot available.
- Error occurred.

Use `ProgressEvent.context` only for small metadata:

```python
{
    "debug_session_id": "...",
    "snapshot_id": "...",
    "cursor": {...},
}
```

Do not send images, labels, or measurement tables through progress events.

Status: `DebugProgressContext` now owns this typed context payload and round-trips through `ProgressEvent.context`. It carries only `debug_session_id`, optional `snapshot_id`, cursor, event type, and optional snapshot-store ref. `DebugProgressEventRequest` builds a lightweight `ProgressEvent` from a `DebugEvent` plus execution/plate/worker identifiers so the worker/server transport can announce debug state without embedding snapshots or payload arrays. `ProgressDebugEventSink` installs this bridge on worker contexts when a `DebugExecutionConfig` is present.

Update: GUI-side progress consumption now exposes a typed snapshot notification seam. `BatchWorkflowService` parses debug progress contexts after normal progress registration and notifies subscribed listeners with `DebugSnapshotAvailableNotification` only when a concrete `snapshot_id` is present. That lets a debug inspector or controller subscribe to snapshot availability without parsing arbitrary progress dictionaries.

Update: the PyQt bridge is now wired end to end for bounded debug snapshots. `PlateManagerWidget` exposes a `debug_snapshot_available` Qt signal backed by the `BatchWorkflowService` listener seam, and `PipelineEditorWidget.show_debug_snapshot(...)` loads the announced snapshot into a reusable `DebugInspectorWindow`. When the snapshot event comes from a remote execution server, `BatchWorkflowService` asks `ZMQExecutionClient.get_debug_snapshot(...)` for typed metadata first and attaches that snapshot to the notification. If no attached snapshot is available, the inspector falls back to `LocalDebugSnapshotStore`/`DebugSnapshotStore` loading. This keeps GUI code out of server-local path reconstruction.

### Snapshot Retrieval

First implementation:

- Worker writes snapshots/previews into a debug store under the output/debug namespace.
- Progress event announces `snapshot_id`.
- GUI loads snapshot metadata from the debug store using a read path that is explicitly available to the GUI process.

Remote/server implementation:

- `ZMQExecutionClient.get_debug_snapshot(...)` sends a typed control-channel request.
- `ZMQExecutionServer` resolves the declared local or FileManager-backed store and returns snapshot metadata.
- Large preview/image/table payloads remain artifact/viewer refs; only snapshot metadata moves through this RPC.

### Snapshot retrieval boundary

The plan must not handwave "GUI uses the filemanager" unless the same filemanager/VFS namespace is available in the GUI process. Local debugging uses a filesystem-backed debug store under the run output directory. Shared-process or shared-namespace callers can read through `DebugSnapshotStore`. Remote/server debugging uses the explicit ZMQ snapshot-read RPC for metadata. Selected local and FileManager/VFS artifact payloads now also have a server-side `DebugArtifactExportRequest`/`DebugArtifactExportResponse` path so the worker/server can materialize payloads into a GUI-readable export root when namespaces differ. Snapshot artifact refs carry content digests when the payload is readable through local or FileManager/VFS storage, so replay/export code can reject stale payloads instead of matching only by artifact name/kind/group.

## Execution Model

### First Slice: Short Debug Executions

Use short execution requests instead of a long-lived paused worker.

For each debug command:

- Compile or reuse a compiled pipeline artifact.
- Restrict execution to one selected axis/well.
- Run from current cursor until requested stop boundary.
- Write snapshots.
- Return control to GUI.

This avoids a worker blocking while waiting for UI commands and fits the current execution server model.

Short executions have a real limitation: if execution restarts from the beginning for every step, debug mode may be correct but slow. The implementation uses explicit cursor bounds and can later add artifact reuse:

- Slice 1: restart from the requested step/invocation boundary for the selected image set and stop at the requested cursor; simplest and safest.
- Slice 2: reuse prior debug snapshots/artifacts as starting inputs when upstream settings are unchanged.
- Slice 3: persistent debug worker only if restart/replay overhead remains unacceptable.

### Later Slice: Persistent Debug Worker

Only add if restart overhead is too high.

Persistent worker command flow:

- GUI sends debug command to execution server.
- Execution server forwards command to debug worker.
- Worker executes `step`, `run_until_pause`, `select_image_set`, or `stop`.
- Worker writes snapshot refs and emits progress events.

The execution server remains the broker. The GUI still does not talk directly to workers.

## Viewer Integration

napari:

- Primary viewer for image/label/overlay layers.
- Debug inspector sends selected snapshot artifacts to napari through existing streaming backend.
- Optional auto-open when debug mode starts.

Status: the inspector-to-viewer/export seam is now explicit. `DebugInspectorWindow` derives a `DebugArtifactActionsModel` from the current snapshot, discovers viewer targets from the existing `StreamingService` registry, and emits `DebugArtifactOpenRequest` values when a user chooses a viewer. It also emits `DebugArtifactMaterializeRequest` values for explicit export actions. `PipelineEditorWidget` owns the host-side signal wiring for the reusable inspector, and export actions now use a directory picker plus `PlateManagerWidget.action_export_debug_artifact(...)` / `BatchWorkflowService.export_debug_artifact(...)` to call the existing ZMQ debug artifact export request. The inspector does not implement its own streaming or copy path; the host GUI remains responsible for satisfying typed requests through the existing streaming service or debug artifact export control path.

The streaming service also now has nominal request/context records for viewer streaming: `ViewerStreamingContext`, `ImageStreamingRequest`, and `RoiStreamingRequest`. Image-browser streaming call sites pass those records instead of re-threading the same viewer/config/callback bundle.

Fiji:

- Optional viewer for image/ROI workflows.
- Same “send selected snapshot artifact” pattern.

Qt:

- Primary for tables, scalar measurements, relationships, timings, errors, and controls.

## CellProfiler-Specific Layer

Add a renderer registry:

```python
class CellProfilerDebugView(metaclass=AutoRegisterMeta):
    module_name: ClassVar[str | None] = None

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        ...
```

Initial renderers:

- `IdentifyPrimaryObjects`: input image, labels, outlines, object count, threshold stats.
- `IdentifySecondaryObjects`: parent/child overlays.
- `CorrectIlluminationCalculate`: source image and illumination function.
- `CorrectIlluminationApply`: before/after preview.
- `Align`: fixed/moving/aligned preview.
- `FilterObjects`: before/after labels and filtered count.
- `MaskObjects`: source labels, mask, masked labels.
- `RelateObjects`: relationship counts and parent/child overlay.
- `MeasureColocalization`: image pair, scatter preview, coefficient table.

Default renderer works for all non-CellProfiler functions.

Status: implemented for the generic view substrate and first CellProfiler renderer. `openhcs.core.debug_views` owns the renderer-independent view model types and the shared artifact-ref table projection. The CellProfiler module owns only the registered renderer family and declarative section specs. The registry returns a default view for unknown modules and an initial specialized view for `IdentifyPrimaryObjects`; the default view renders source paths, artifact refs, preview refs, measurements, relationships, timing, and errors from the same `DebugSnapshot` structure used by PyQt, CLI diagnostics, and future report export.

## Implementation Slices

1. Add debug dataclasses and no-op debug sink.

Files likely touched:

- `openhcs/core/debug.py` or `openhcs/core/debug/`
- `openhcs/core/context/processing_context.py`

Implemented. `ProcessingContext` now carries a default no-op debug sink, and tests verify that installing a recording sink emits invocation events without changing normal execution output.

2. Emit debug events at step boundaries.

Files likely touched:

- `openhcs/core/steps/function_execution.py`
- `openhcs/core/steps/function_runtime.py`
- `openhcs/core/progress/types.py` only if a new progress phase is needed.

Implemented at invocation boundaries in the invocation execution helper. This is intentionally lower than `FunctionStepExecutor`, because one `FunctionStep` can contain multiple callables. Step/pattern-group events can be added later as presentation-level grouping events if the UI needs coarser markers.

3. Add debug snapshot store.

Files likely touched:

- `openhcs/core/debug/store.py`
- Filemanager/VFS integration if needed.

Implemented in `openhcs.core.debug`. The snapshot metadata format and progress context carry `snapshot_store_ref`; local GUI readback is wired through `LocalDebugSnapshotStore`; shared FileManager/VFS-backed stores are represented by `FileManagerDebugSnapshotStore`; remote server-mediated metadata readback is implemented through the typed ZMQ snapshot-read control request.

Debug snapshots now receive metadata artifact refs from the invocation planner itself. `DebugArtifactRefProjection` converts compiled artifact input/output plans into cursor-aware refs and separates measurement and relationship refs by `ArtifactKind`. This closes the first preview substrate without storing full images, labels, tables, or measurements in progress events.

4. Add PyQt debug inspector window.

Files likely touched:

- `openhcs/pyqt_gui/windows/debug_inspector_window.py`
- `openhcs/pyqt_gui/windows/managed_windows.py`

Implemented for the first inspector slice. `openhcs.pyqt_gui.windows.debug_inspector_window.DebugInspectorWindow` renders the core `DebugViewModel` sections/tables/text. It accepts a snapshot renderer callback and defaults to the CellProfiler renderer registry, so Qt remains a presentation layer over the core debug model instead of owning CellProfiler/runtime semantics.

5. Add pipeline editor debug toolbar and step markers.

Files likely touched:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`

6. Wire debug commands through execution server.

Files likely touched:

- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py`
- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/runtime/zmq_execution_client.py`

This should start with one command type that runs a bounded debug execution and returns when the boundary is reached. Do not add a paused-worker command protocol in the first slice.

Started. `DebugExecutionConfig` is carried through existing execute `config_params`, `ZMQExecutionClient.submit_debug_pipeline(...)` creates those params, and the execution server installs a debug sink on worker contexts for that run. The config now preserves `DebugCommandType`, optional selected source group, pause-step indices, snapshot store metadata, `start_step_index`, and `start_after_invocation_key`, so the GUI command surface, ZMQ request, compiler selection, and worker policy share one typed command envelope. Worker sink installation now uses `DebugSinkInstallRequest` instead of re-threading execution id, plate id, worker slot, and owned wells through sibling policy signatures. `BatchWorkflowService.run_debug_plate(...)` owns the GUI-side bounded path: compile one selected plate with the same debug config that will be used for execution, submit it through `submit_debug_pipeline(...)`, and keep using normal progress/completion polling. When `snapshot_store_ref` is present the worker sink writes local metadata snapshots before emitting progress; otherwise it emits progress only. `STEP` and `RUN_TO_PAUSE` stop through the registered `DebugStepStopStrategy` family, and `STEP` advances within native function patterns through the registered `DebugInvocationExecutionStrategy` family. Persistent paused-worker control now exists through `DebugPausedWorkerController`, `DebugPausedWorkerRegistry`, and the ZMQ `WORKER_COMMAND` control message, with live server/client command-loop coverage.

The pipeline editor now has the first real command bridge: toolbar run-family commands delegate to `PlateManagerWidget.action_run_debug_plate(...)`, which creates a `DebugSession`, chooses a local `.openhcs_debug` snapshot-store root beside the active plate, and submits the bounded debug run through `BatchWorkflowService`. Step-stop and one-axis selection behavior are implemented in the bounded execution policy, not in GUI command routing.

Snapshot availability is now connected back to the inspector for the bounded path. `PlateManagerWidget.debug_snapshot_available` carries nominal `DebugSnapshotAvailableNotification` values to `PipelineEditorWidget.show_debug_snapshot(...)`, which uses an attached server-read snapshot when available, falls back to store loading otherwise, and raises the reusable inspector window. This closes the GUI loop without introducing a CellProfiler-specific runner or a separate progress parser.

7. Add artifact streaming from debug inspector to napari/Fiji.

Files likely touched:

- `openhcs/pyqt_gui/windows/debug_inspector_window.py`
- Existing image browser streaming service can be reused or extracted.

8. Add invocation-level stepping.

Files likely touched:

- `openhcs/core/function_patterns.py`
- `openhcs/core/steps/function_runtime.py`
- Function pattern editor UI.

Implemented at the runtime and command-envelope level. `execute_function_chain(...)` consults the installed `DebugEventSink` before each compiled invocation and after each `AFTER_INVOCATION` event. `DebugInvocationExecutionStrategy` is the nominal command family that owns invocation skipping/stopping, and the `STEP` strategy uses `DebugExecutionConfig.start_after_invocation_key` to advance past the current function inside a `FunctionStep`. The current pipeline editor bridge supplies that cursor key from the active debug session snapshot, so function-pattern stepping is not limited to generated CellProfiler one-module steps. Rich per-invocation markers in the function-pattern editor remain presentation work, not runtime plumbing.

9. Add CellProfiler debug renderers.

Files likely touched:

- `openhcs/interop/cellprofiler/debug_views.py`

## Non-Goals For First Slice

- Do not implement a long-lived paused worker.
- Do not make napari mandatory.
- Do not send large arrays through progress events.
- Do not create a CellProfiler-only runner.
- Do not require all intermediate outputs to be normal materialized pipeline outputs.

## Acceptance Criteria

- User can enter debug mode from the pipeline editor.
- User can select one well/image set.
- User can step through at least one `FunctionStep` and through multiple compiled function invocations inside a native function pattern.
- GUI receives cursor/snapshot availability through existing ZMQ progress path.
- Debug inspector displays source paths, output artifact refs, timings, and errors.
- Selected image/label outputs can be sent to napari using existing viewer infrastructure.
- Normal pipeline execution is unchanged when debug mode is disabled.
