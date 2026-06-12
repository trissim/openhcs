# Live Measurement Table Plan - 2026-05-26

## Goal

Show measurements while a pipeline is running, next to the existing live image
streaming workflow, without creating a GUI-only measurement store or a second
execution path.

The live table should:

- update during execution as measurement artifacts are produced;
- preserve OpenHCS runtime artifact identity: artifact kind/name, axis, group
  key, VFS path, and backend;
- support CellProfiler-generated measurements and native OpenHCS measurement
  artifacts through the same path;
- display bounded previews in Qt while keeping VFS/runtime artifacts as the
  authority for full data;
- fail loudly for malformed live-measurement payloads, while treating payload
  absence as a normal non-measurement progress event.

## Current Architecture Facts

### Existing event plane

- Worker and compile phases already emit typed `ProgressEvent` dictionaries.
- `openhcs.core.progress.types.ProgressEvent` already carries a `context`
  mapping.
- `openhcs.runtime.zmq_execution_server.ZMQExecutionServer._forward_worker_progress`
  validates worker ownership and forwards enriched progress dictionaries to
  ZMQ clients.
- `openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service.ProgressWorkflowService.on_progress`
  parses incoming dictionaries with `ProgressEvent.from_dict`, registers them
  in the progress tracker, and notifies the debug progress service.

Conclusion: live measurement notifications should use the existing ZMQ progress
data plane. Workers should not call Qt, and Qt should not poll worker-private
state.

### Existing data authority

- `openhcs.core.runtime_stores.RuntimeValueStore` records typed runtime artifact
  writes as `StoredRuntimeValue` values.
- Measurement artifacts use `ArtifactKind.MEASUREMENTS`.
- `openhcs.core.runtime_values.MeasurementTable.from_runtime_value(...)`
  reconstructs the native measurement view from a stored runtime value.
- Native OpenHCS artifact writes go through
  `openhcs.core.steps.function_runtime._save_artifact_value`.
- CellProfiler adapter writes go through
  `openhcs.interop.cellprofiler.runtime.adapter` and record the same typed
  runtime values.
- Step execution in `openhcs.core.orchestrator.orchestrator._execute_single_axis_static`
  has one generic boundary after `step.process(...)`, before the
  `STEP_COMPLETED` event is emitted.

Conclusion: the step boundary should detect newly observed measurement records
from the runtime store and attach bounded previews to the existing
`STEP_COMPLETED` event. This avoids adding backend-specific GUI hooks.

### Existing UI surface

- `PlateManagerWidget` owns the batch workflow service and already receives
  progress.
- `BatchWorkflowService` owns the `ProgressWorkflowService` instance and debug
  snapshot listener registration.
- `DebugProgressNotificationService` is the closest existing pattern for
  context-derived notifications from progress events.
- `PlateViewerWindow` already groups image browsing and metadata in one tabbed
  window.
- Existing table rendering is local and duplicated in several widgets
  (`DebugInspectorWindow`, `ArtifactContractPreviewWidget`,
  `SourceBindingsEditor`).

Conclusion: add a live-measurement notification service parallel to debug
progress notifications, and add a dedicated live-results window/tab widget
rather than embedding decoding logic in `PlateManagerWidget`.

## Advisor Findings That Constrain This Work

Advisor scopes run:

- `nominal-refactor-advisor openhcs/core/progress`
- `nominal-refactor-advisor openhcs/core/orchestrator`
- `nominal-refactor-advisor openhcs/pyqt_gui/widgets/shared/services`
- `nominal-refactor-advisor openhcs/pyqt_gui/windows openhcs/pyqt_gui/widgets`

Relevant findings:

- `DebugProgressNotificationService._debug_context_from_event` currently has a
  fail-soft `return None` on decode errors. Do not copy this shape for live
  measurements.
- `BatchWorkflowService` is already a large single-consumer orchestration
  surface. Do not add live-measurement state directly to it beyond listener
  wiring.
- `ImageBrowserWidget` is already overgrown. Do not add measurement-table logic
  there.
- Existing Qt table rendering is repeated. A live table widget should own a
  small reusable table projection locally, and later that projection can move to
  `pyqt-reactive` if another table surface adopts it.
- Some advisor suggestions for generic `ExtractedBase` scaffolds around
  constructors/close events are noise for this feature and should not be
  followed.

## Target Design

### 1. Core live measurement progress contract

Add a nominal contract module, likely
`openhcs.core.progress.live_measurements`.

Types:

- `LiveMeasurementArtifactAddress`
  - `name`
  - `kind`
  - `axis_id`
  - `group_key`
  - `site`
  - `channel`
  - `z_index`
  - `timepoint`
  - `path`
  - `backend`

- `LiveMeasurementTablePreview`
  - `address`
  - `columns`
  - `rows`
  - `row_count`
  - `truncated_rows`
  - `truncated_columns`
  - `object_name`
  - `source_image_name`

- `LiveMeasurementProgressPayload`
  - list of previews for one event;
  - `to_context()` and `from_context()` methods;
  - absence of `live_measurements` means "no live measurement payload";
  - malformed `live_measurements` means error.

Rules:

- bounded preview only, default around 50 rows and 64 columns;
- cells must be JSON-safe because the ZMQ progress channel serializes JSON;
- full data must remain in runtime artifacts/VFS, with the artifact address
  carried as provenance and later-resolution identity;
- a live event must cap preview count as well as row/column count, because one
  step can write multiple measurement artifacts;
- support `ColumnarRows`, row sequences, dataclass rows, and mapping-backed
  columnar shapes through existing runtime value abstractions.

Do not:

- put pandas or Qt dependencies in the core payload module;
- infer table identity from field names;
- use stringly "measurement-ish" checks; only `ArtifactKind.MEASUREMENTS`
  produces live measurement previews in this feature slice.

### 2. Runtime emission boundary

Add a small runtime-step helper, likely in a new module instead of expanding
`orchestrator.py`:

- `StepMeasurementObservation`
- `StepMeasurementPreviewEmitter`
- inputs:
  - runtime store before observation cursor/revision;
  - runtime store after step execution;
  - execution identity fields;
  - step progress fields;
- output:
  - optional progress `context` mapping for `STEP_COMPLETED`.

Implementation sketch:

1. Add a small nominal observation cursor to `RuntimeValueStore`, for example
   `RuntimeStoreObservationCursor(index=int, revision=int)` plus
   `observation_cursor()` and `observed_values_after(cursor)`.
2. Before `step.process(...)`, capture the cursor.
3. After `step.process(...)`, select records through
   `store.observed_values_after(cursor)`.
3. Filter to measurement records with `ArtifactKind.MEASUREMENTS`.
4. Build `LiveMeasurementProgressPayload`.
5. Attach `context=payload.to_context()` to the existing `STEP_COMPLETED`
   progress event.

This should happen once in `_execute_single_axis_static`, not in every backend.

Edge cases:

- Debug step reuse should not fabricate live measurement events unless reused
  artifacts are actually recorded into the runtime store during reuse.
- Empty measurement tables should not spam events unless the UI needs them for
  schema visibility. Initial slice should skip empty previews.
- Large object-measurement outputs must be truncated in the event but keep the
  full artifact address.
- A memory-backend path inside a worker is not automatically GUI-readable.
  Treat the address as identity/provenance in the first slice. Full-table open
  must require a shared VFS namespace, debug snapshot export, or materialized
  output.

### 3. Progress notification service

Add a service parallel to `DebugProgressNotificationService`:

- `LiveMeasurementAvailableNotification`
- `LiveMeasurementProgressNotificationService`
- listener registration/removal methods;
- `notify_from_progress_event(event)`.

Decode rules:

- if `event.context` has no live-measurement key, do nothing;
- if the live-measurement key exists and is malformed, raise/log a warning from
  the caller path rather than silently returning `None`;
- do not swallow schema violations inside the decoder.

Wire it through:

- `BatchWorkflowService` owns the service instance;
- `ProgressWorkflowService.on_progress` receives it as a dependency and calls
  it after registering the event;
- `PlateManagerWidget` subscribes with a Qt signal or model method.

Keep `BatchWorkflowService` as wiring only. Do not add the live table model to
the service.

### 4. UI model and widget

Add a Qt-side model that is independent of the table widget:

- `LiveMeasurementTableModel`
  - stores recent previews by artifact address and execution identity;
  - exposes available table labels;
  - exposes current rows/columns;
  - caps retained previews to avoid unbounded GUI memory.

Add a widget/window:

- `LiveMeasurementsWindow` or `LiveMeasurementsWidget`;
- table selector for artifact/axis/step;
- read-only `QTableWidget`;
- status label showing row count and truncation;
- clear button for a new batch.

Initial placement:

- add a PlateManager action button, for example `Results`, that opens the live
  measurements window for the selected/running plate;
- later integrate as a `PlateViewerWindow` tab if the viewer window is made
  aware of live execution services.

Reasoning:

- `PlateViewerWindow` currently receives only an orchestrator, not the running
  progress/event service. Forcing live progress into it would create a hidden
  dependency. A separate live-results window is cleaner for the first slice.

### 5. Full-data follow-up

The first feature should display live previews only. A follow-up should add
"Open full table" by resolving the artifact address through a confirmed shared
runtime store/VFS namespace, a debug snapshot/export path, or materialized
outputs.

Do not pretend the preview event is the full data source.

## Dry-Run Against Current Code

### Progress contract dry-run

`ProgressEvent.to_dict()` and `from_dict()` already preserve `context`.
No schema migration is needed.

Risk: `create_event(...)` does not currently accept `context`.

Plan update:

- either extend `create_event(..., context=None)` or construct `ProgressEvent`
  directly only where live measurement context is needed;
- prefer extending `create_event` because existing runtime code uses it for
  init events, and keeping the constructor surface complete reduces future
  ad-hoc usage.

### Runtime dry-run

`_execute_single_axis_static` already has access to:

- `frozen_context`
- `lane_context.identity.execution_id`
- `lane_context.identity.plate_id`
- `lane_context.worker_slot`
- `lane_context.owned_wells`
- `step_name`
- step index and total steps.

It can call `require_runtime_value_store(frozen_context, owner_name=...)`.
The store exists on `ProcessingContext`.

Risk: acquiring runtime store for every step adds overhead.

Plan update:

- acquire the store once before the loop;
- add cursor-based observation methods to `RuntimeValueStore` instead of
  repeatedly materializing `observed_values` tuples for length/slicing;
- only compute a live measurement payload from the cursor delta after each step;
- event payload is bounded, so progress traffic does not scale with full table
  size.

Risk: one step can write many measurement records.

Plan update:

- `LiveMeasurementProgressPayload` needs a `preview_count` and
  `truncated_previews` flag;
- default event cap should be conservative, for example 8 table previews per
  progress event;
- the UI should show that additional tables were produced but not included in
  the live preview event.

### UI dry-run

`ProgressWorkflowService.on_progress` currently catches all exceptions around
parsing/registering/debug notification, logs a warning, and still marks dirty.

Risk: live-measurement decode errors could be swallowed too broadly.

Plan update:

- live measurement notification should raise a specific decode exception;
- `ProgressWorkflowService` should log it with enough context but still not
  crash the GUI event thread;
- unit tests should cover malformed payload logging and absence as no-op.

`PlateManagerWidget.BUTTON_CONFIGS` and `ACTION_ROUTES` are the existing button
extension point. Adding a `Results` action there is straightforward.

Risk: another top-level button may crowd the PlateManager.

Plan update:

- use a compact label (`Results`) and tooltip;
- do not add a landing-page style description inside the widget.

Risk: `PlateViewerWindow` sounds like the natural "alongside images" placement,
but it currently has no dependency on live progress services.

Plan update:

- first implementation opens a `LiveMeasurementsWindow` from PlateManager;
- the window can be positioned alongside the existing viewer by the user/window
  manager;
- move into `PlateViewerWindow` only after there is a typed viewer-session
  context that can safely carry progress subscriptions.

### Table rendering dry-run

Several existing widgets configure `QTableWidget` directly. A new widget can be
self-contained for this first feature. Moving a generic table browser to
`pyqt-reactive` should wait until at least two OpenHCS consumers can share the
same abstraction.

Plan update:

- keep only live-measurement-specific table projection in OpenHCS now;
- if this overlaps with debug inspector measurement tables during a later pass,
  move a read-only dynamic table widget to `external/pyqt-reactive`.

## Tests

Core unit tests:

- serialize/deserialize live measurement payloads through `ProgressEvent`
  context;
- build preview from a `RuntimeValueStore` measurement record;
- truncate rows/columns correctly;
- truncate preview count correctly;
- preserve artifact address fields.

Runtime unit tests:

- `RuntimeValueStore` observation cursor returns only records written after the
  cursor and rejects stale/conflicting cursors if needed;
- executing a step that writes a measurement artifact emits `STEP_COMPLETED`
  with live measurement context;
- steps with no measurement writes emit no live-measurement context;
- native OpenHCS measurement artifacts and CellProfiler measurement artifacts
  use the same payload builder at the runtime-store record level.

GUI/service unit tests:

- `ProgressWorkflowService.on_progress` notifies live-measurement listeners;
- absence of live-measurement context is a no-op;
- malformed live-measurement context logs/fails as designed, not silently
  `return None`;
- `LiveMeasurementTableModel` updates labels, rows, and truncation state.

Smoke tests:

- `QT_QPA_PLATFORM=offscreen openhcs` starts without import/runtime errors;
- focused PyQt tests for the new window/widget;
- run advisor on changed folders, not single files:
  - `openhcs/core/progress`
  - `openhcs/core/orchestrator`
  - `openhcs/pyqt_gui/widgets/shared/services`
  - `openhcs/pyqt_gui/widgets`
  - `openhcs/pyqt_gui/windows`

## Implementation Sequence

1. Add core live measurement progress contract and tests.
2. Add runtime step-boundary helper and tests.
3. Add live measurement progress notification service and wire it through
   `ProgressWorkflowService`/`BatchWorkflowService`.
4. Add Qt model/widget/window.
5. Add PlateManager `Results` action.
6. Run focused unit tests.
7. Run advisor on changed folders and resolve feature-relevant findings.
8. Run GUI startup smoke test.

## Explicit Non-Goals For This Slice

- no VFS polling loop;
- no direct worker-to-Qt calls;
- no full-table transfer through progress events;
- no CellProfiler-specific measurement side channel;
- no new measurement semantics inferred from column names;
- no broad refactor of `ImageBrowserWidget` or `BatchWorkflowService` beyond
  necessary listener wiring;
- no move to `pyqt-reactive` until there is a real second consumer for the same
  dynamic read-only table abstraction.

## Review Verdict

The target architecture is coherent if the implementation keeps the data-plane
split strict:

- ZMQ progress: live notification and bounded preview.
- Runtime store/VFS: full measurement data authority.
- UI model: retained presentation state only.

The main architectural risk is allowing `context` to become an untyped bag. The
mitigation is a nominal live-measurement payload module with explicit
encode/decode methods and tests. The second risk is expanding existing GUI hub
classes. The mitigation is a new notification service plus a small model/widget,
with `BatchWorkflowService` and `PlateManagerWidget` doing only wiring.
