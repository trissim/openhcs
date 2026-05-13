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

Do not assume debug payload transport is already solved by this. The current path is verified for progress/status events. Snapshot retrieval still needs a concrete store/query path.

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

Add visual annotations to the existing step list:

- Current cursor marker.
- Completed/dirty status.
- Pause marker.
- Error marker.

Use existing `ObjectState`/step scope IDs to associate debug state with steps. Do not key debug state only by list index; list index changes under editing/reordering.

### Function Pattern Editor

First slice can step at `FunctionStep` granularity.

Second slice should step at function-invocation granularity:

- Add cursor markers for individual function-pattern items.
- Add pause markers for individual invocation items.
- Show invocation kwargs in the debug inspector.

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

### Debug Cursor

Cursor should support both coarse and fine stepping:

- `step_index`
- `step_scope_id`
- `group_key`
- `invocation_key`
- `pattern_group_identity`

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

### Snapshot Retrieval

First implementation:

- Worker writes snapshots/previews into a debug store under the output/debug namespace.
- Progress event announces `snapshot_id`.
- GUI loads snapshot metadata from the debug store using a read path that is explicitly available to the GUI process.

Later implementation:

- Add explicit ZMQ debug RPCs if remote GUI debugging needs server-mediated snapshot reads.

### Snapshot retrieval gap to close

The plan must not handwave "GUI uses the filemanager" unless the same filemanager/VFS namespace is available in the GUI process. For local debugging, a filesystem-backed debug store under the run output directory is enough. For remote/server debugging, snapshot reads need one of:

- A ZMQ debug RPC: `get_debug_snapshot(debug_session_id, snapshot_id)`.
- A shared VFS URI resolver that both server and GUI can open.
- A server-side export/materialize command that writes selected snapshot payloads into a GUI-readable path before viewer streaming.

First slice should implement the local filesystem-backed path and keep the RPC interface in the dataclass/API design so remote support does not require redesign.

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

Short executions have a real limitation: if execution restarts from the beginning for every step, debug mode may be correct but slow. The implementation needs an explicit resume policy:

- Slice 1: restart from pipeline start for the selected image set and stop at the requested cursor; simplest and safest.
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

## Implementation Slices

1. Add debug dataclasses and no-op debug sink.

Files likely touched:

- `openhcs/core/debug.py` or `openhcs/core/debug/`
- `openhcs/core/context/processing_context.py`

The sink should be optional and default to no-op so production execution does not pay UI/debug cost.

2. Emit debug events at step boundaries.

Files likely touched:

- `openhcs/core/steps/function_execution.py`
- `openhcs/core/steps/function_runtime.py`
- `openhcs/core/progress/types.py` only if a new progress phase is needed.

Invocation-level events likely belong in `PatternGroupRuntime._execute_pattern` or the invocation execution helper, not only `FunctionStepExecutor`, because one `FunctionStep` can contain multiple callables.

3. Add debug snapshot store.

Files likely touched:

- `openhcs/core/debug/store.py`
- Filemanager/VFS integration if needed.

4. Add PyQt debug inspector window.

Files likely touched:

- `openhcs/pyqt_gui/windows/debug_inspector_window.py`
- `openhcs/pyqt_gui/windows/managed_windows.py`

5. Add pipeline editor debug toolbar and step markers.

Files likely touched:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`

6. Wire debug commands through execution server.

Files likely touched:

- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py`
- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/runtime/zmq_execution_client.py`

This should start with one command type that runs a bounded debug execution and returns when the boundary is reached. Do not add a paused-worker command protocol in the first slice.

7. Add artifact streaming from debug inspector to napari/Fiji.

Files likely touched:

- `openhcs/pyqt_gui/windows/debug_inspector_window.py`
- Existing image browser streaming service can be reused or extracted.

8. Add invocation-level stepping.

Files likely touched:

- `openhcs/core/function_patterns.py`
- `openhcs/core/steps/function_runtime.py`
- Function pattern editor UI.

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
- User can step through at least one `FunctionStep`.
- GUI receives cursor/snapshot availability through existing ZMQ progress path.
- Debug inspector displays source paths, output artifact refs, timings, and errors.
- Selected image/label outputs can be sent to napari using existing viewer infrastructure.
- Normal pipeline execution is unchanged when debug mode is disabled.
