# Runtime Viewer And Streaming Protocol Cleanup - 2026-05-18

## Full-Scan Evidence

The full scan shows high active finding density in:

- `openhcs/runtime/napari_stream_visualizer.py`
- `openhcs/runtime/napari_viewer_server.py`
- `openhcs/runtime/fiji_viewer_server.py`
- `openhcs/runtime/fiji_stream_visualizer.py`

Finding families include:

- closed-family string dispatch;
- manual class-marker membership;
- bidirectional registries;
- repeated threaded parameter families;
- oversized viewer/server orchestration methods;
- unreferenced private functions.

## Current State

The first-wave Napari campaign removed duplicated data-type handler tables.
Remaining debt is deeper: viewer/server code still recovers runtime roles from
strings, markers, and mirrored maps.

## Target Shape

Create typed viewer protocol records:

- `ViewerIdentity`
- `ViewerLifecycleState`
- `ViewerDimensionUpdate`
- `ViewerLayerUpdate`
- `ViewerConnectionRegistry`
- `ViewerEventRoute`
- `StreamingViewerBackend`

Napari/Fiji-specific code should implement these protocols rather than each
server maintaining its own string/marker/bidirectional logic.

## Phases

1. Characterize viewer import behavior with mocked optional dependencies.
2. Extract shared lifecycle/identity records without changing viewer startup.
3. Replace duplicated dimension/layer dispatch with route tables.
4. Replace manual bidirectional registries with one typed connection registry.
5. Split large listener/update methods into event-route stages.
6. Delete or make explicit unreferenced private viewer helpers.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
.venv/bin/python - <<'PY'
import openhcs.runtime.napari_stream_visualizer
import openhcs.runtime.napari_viewer_server
import openhcs.runtime.fiji_stream_visualizer
import openhcs.runtime.fiji_viewer_server
PY
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/runtime/napari_stream_visualizer.py \
  openhcs/runtime/napari_viewer_server.py \
  openhcs/runtime/fiji_stream_visualizer.py \
  openhcs/runtime/fiji_viewer_server.py
```

## Completion Criteria

- Viewer lifecycle and connection identity are typed.
- Napari/Fiji route differences are explicit backend variants.
- Optional GUI/runtime imports remain lazy and smoke-testable.

## Execution Log

### Checkpoint 1

Implemented:

- `NapariLayerUpdateAuthority`
- `NapariLayerUpdateRequest`
- `NapariViewerServerRequest`
- shared use of `ViewerQtEnvironmentPolicy`

Changed behavior boundaries:

- Napari image/shapes/points layer replacement mechanics now have one shared
  owner in `napari_streaming_handlers.py`.
- `napari_stream_visualizer.py` and `napari_viewer_server.py` keep their public
  helper wrappers but no longer duplicate the layer replacement mechanics.
- Dead `NapariStreamVisualizer._prepare_data_for_display` was deleted after
  repository-wide call-site verification.
- Repeated Qt platform setup now uses `ViewerQtEnvironmentPolicy`.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed
```

Advisor status:

- Cleared duplicated Napari create/update layer helper findings.
- Cleared dead `_prepare_data_for_display` finding.
- Cleared Napari platform dispatch findings.
- Remaining: component/shape dispatch, Napari viewer-server role quotient,
  Fiji dimension-listener hub, Fiji dimension context record, and public
  Napari/Fiji process signature-family cleanup.

### Checkpoint 2

Implemented:

- Napari `StreamingDataType` route table for layer update execution.
- Deletion of dead `_parse_component_info_from_path` helpers from both Napari
  runtime modules.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed
```

Advisor status:

- Cleared Napari data-type strategy ladder.
- Cleared unreferenced private component parser findings.
- Remaining high-value work is now the larger decompositions: component/shape
  dispatch, Napari server role quotient, Fiji dimension-listener hub, Fiji
  dimension context record, and public process signature-family cleanup.

### Checkpoint 3

Implemented:

- `ComponentDimensionLabelPolicy` in `viewer_protocol.py`.
- Shared Napari dimension-label construction in both Napari runtime modules.
- Unit coverage for channel, well, generic metadata, and metadata-ignored
  fallback label behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 6 passed
```

Advisor status:

- Cleared the Napari component-label string dispatch findings.
- Remaining viewer protocol findings are shape rasterization dispatch, stream
  visualizer lifecycle/membership boundaries, Napari/Fiji server role
  quotients, Fiji dimension context records, process signature records, and
  platform strategy ladders in the shared viewer protocol.

### Checkpoint 4

Implemented:

- `NapariShapeKind`, `NapariShapePaintContext`, and
  `NapariShapeLabelRasterizer` as the shared dense-label conversion authority.
- Direct rasterizer calls from both Napari runtime modules, deleting the
  forwarding `_shapes_to_labels` wrappers.
- `NapariLayerLogPolicy` for counted vs uncounted layer creation logging.
- Unit coverage for polygon/path raster projection and legacy point extent
  behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 8 passed
```

Advisor status:

- Cleared Napari shape-kind string dispatch.
- Cleared Napari `_shapes_to_labels` identity-forwarding shells.
- Cleared inline layer-kind subset logging policy.
- Remaining viewer work is now lifecycle/membership boundaries,
  request-builder projection records, Napari/Fiji server role quotients, Fiji
  dimension context records, process signature records, and shared viewer
  platform strategy ladders.

### Checkpoint 5

Implemented:

- `ViewerProcessHandle` as the single subprocess/multiprocessing liveness and
  termination adapter.
- `ManagedViewerLifecycleMixin` as the shared `is_running` algorithm for
  Napari and Fiji stream visualizers.
- Napari/Fiji cleanup, liveness, stop, PID logging, and viewer-running checks
  now route through the nominal process handle instead of structural probes.
- Unit coverage for real subprocess lifecycle wrapping and rejection of
  structural process lookalikes.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py tests/unit/test_viewer_protocol.py -q
# 10 passed
```

Advisor status:

- Cleared repeated `is_running` method skeleton.
- Cleared direct `hasattr(self.process, ...)` process-type probes.
- Remaining lifecycle findings are broader class-marker/membership signals
  around viewer state, plus request-builder records, Napari/Fiji server role
  quotients, Fiji dimension context records, process signature records, and
  shared viewer platform strategy ladders.

### Checkpoint 6

Implemented:

- `ViewerControlPingMode`, `ViewerControlPingPolicy`, and
  `ViewerControlPingRequest` for viewer control-port ping/readiness checks.
- Replaced repeated `ping_control_port(...)` argument bundles in Napari/Fiji
  stream visualizers.
- Replaced Napari's manual quick-ping ZMQ implementation with the same typed
  request path used by Fiji.
- Unit coverage for quick vs existing-viewer ping projection.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py tests/unit/test_viewer_protocol.py -q
# 11 passed
```

Advisor status:

- Cleared repeated `ping_control_port` projection findings.
- Cleared the intermediate constructor-variant findings by making quick vs
  existing-viewer ping behavior a typed policy table.
- Remaining viewer work is viewer state membership, Napari/Fiji server role
  quotients, dynamic/private entry witnesses, process signature records, and
  shared viewer platform strategy ladders.

### Checkpoint 7

Implemented:

- `VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME` for system-name to platform
  projection.
- `ViewerQtPlatformEnvironmentPolicy` and `VIEWER_QT_ENVIRONMENT_POLICIES`
  for per-platform Qt environment mutation.
- Unit coverage for Linux, Linux with preexisting `QT_QPA_PLATFORM`, Darwin,
  and Windows policy rows.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 4 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/viewer_protocol.py
# shared viewer platform strategy ladder findings cleared
```

Advisor status:

- Cleared `ViewerProcessPlatform.current` enum/string ladder.
- Cleared `ViewerQtEnvironmentPolicy.apply_to` enum ladder.
- Remaining viewer-protocol-local finding is broader lifecycle membership in
  `ManagedViewerLifecycleMixin`.

### Checkpoint 8

Implemented:

- `NapariViewerServerRequest.from_legacy_signature(...)` as the sole field-map
  authority for the current public Napari server/process signatures.
- Routed both Napari runtime modules through the request builder.
- Unit coverage for legacy-signature projection.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/viewer_protocol.py
# repeated NapariViewerServerRequest constructor field mapping cleared
```

Advisor status:

- Cleared repeated request-constructor field mapping.
- Remaining request-related finding is the larger public legacy signature
  family itself; that should be handled with an explicit API/process-entry
  migration rather than more wrapper forwarding.

### Checkpoint 9

Implemented:

- Routed Napari and Fiji `_send_ack(...)` status values through
  `ViewerProtocolStatus`.
- Added module-local `_ACK_SUCCESS` and `_ACK_ERROR` constants where repeated
  acknowledgments are emitted.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/fiji_viewer_server.py
# repeated viewer status literal findings cleared
```

Advisor status:

- Cleared repeated `error` status literals in both Napari runtime modules.
- Cleared repeated `success` status literals in Fiji viewer handling.
- Follow-up: `NapariStreamVisualizer._quick_ping_check` should be made a
  nominal lifecycle hook because the mixin calls it dynamically.

### Checkpoint 10

Implemented:

- `ManagedViewerLifecycleMixin` is now an ABC with an explicit
  `check_connected_viewer()` hook.
- Napari/Fiji stream visualizers implement the public lifecycle hook instead
  of relying on an unwitnessed private `_quick_ping_check` method.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# dangling private _quick_ping_check finding cleared
```

Advisor status:

- Cleared private lifecycle hook residue.
- Remaining lifecycle findings are the broader viewer state-membership model,
  not an unwitnessed hook.

### Checkpoint 11

Implemented:

- `NapariLayerStateStore` owns layer objects, dimension-label maps, and
  debounced pending-update timers as one per-layer state authority.
- Both Napari runtime modules route layer/dimension/timer operations through
  the state store.
- Unit coverage for layer, label, and pending-timer state operations.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 9 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# main Napari layers/dimension_labels/pending_updates registry findings cleared
```

Advisor status:

- Cleared the main Napari per-layer state dictionary ownership findings.
- Remaining Napari viewer-state finding is the separate batch-processor store
  in `napari_viewer_server.py`, plus larger server role quotient and process
  entry migration work.

### Checkpoint 12

Implemented:

- `NapariBatchProcessorStore` owns lazy batch processor creation by layer key.
- `napari_viewer_server.py` no longer carries its own `_batch_processors`
  registry and lock.
- Unit coverage verifies one processor is created per layer and reused.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 10 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# batch processor registry finding cleared
```

Advisor status:

- Cleared the remaining Napari batch-processor store finding.
- Remaining Napari findings are the larger process-entry/public-signature
  migration and server facade role quotient.

### Checkpoint 13

Implemented:

- `NapariDetachedProcessRequest` and `NapariViewerProcessEntrypoint` now own
  detached Napari launch code generation, log-file routing, and conversion to
  `DetachedViewerProcessRequest`.
- `napari_viewer_server.py` exposes a public process entrypoint and constructs
  `NapariViewerServer` from `NapariViewerServerRequest`.
- `napari_stream_visualizer.py` no longer carries its own private Napari
  process-entry implementation or ad hoc launch script.
- Unit coverage verifies generated process code imports the public entrypoint
  and carries the intended log/cwd/transport values.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 17 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# private process-entry, embedded launch payload, and repeated signature findings cleared
```

Advisor status:

- Remaining viewer findings are now lifecycle membership semantics and the
  duplicated Napari server role quotient.

### Checkpoint 14

Implemented:

- `ViewerLifecycleState` and `ViewerLifecycleMode` now own viewer lifecycle
  membership instead of paired `_is_running` / `_connected_to_existing`
  booleans.
- Napari and Fiji stream visualizers mark lifecycle transitions through the
  nominal state object.
- `ManagedViewerLifecycleMixin.is_running` evaluates the nominal state and
  resets it on stale external viewers, dead owned processes, or process-status
  errors.
- `napari_stream_visualizer.py` now explicitly registers its global cleanup
  callback.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 18 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py openhcs/runtime/napari_viewer_server.py
# lifecycle membership and unreferenced cleanup callback findings cleared
```

Advisor status:

- Remaining focused viewer finding is the duplicated Napari server role
  quotient in `napari_stream_visualizer.py` and `napari_viewer_server.py`.

### Checkpoint 15

Implemented:

- Removed the duplicate `NapariViewerServer` implementation from
  `napari_stream_visualizer.py`.
- `napari_stream_visualizer.py` now re-exports the canonical server from
  `napari_viewer_server.py` while retaining only client/process-manager
  responsibilities.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 18 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py
# stream-visualizer server role quotient cleared
```

Advisor status:

- Remaining focused viewer finding is the canonical `NapariViewerServer` role
  quotient in `napari_viewer_server.py`.

### Checkpoint 16

Implemented:

- `NapariComponentValueTracker` owns global component value accumulation and
  indexed-axis expansion.
- `NapariViewerServer` now delegates component value tracking to the tracker.
- Removed the no-op `_setup_ack_socket` override and dead server-side detached
  spawn helper.
- Unit coverage verifies indexed-axis expansion and missing-axis defaults.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 19 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# setup role and dead detached-spawn findings cleared; server quotient reduced to update/control/message roles
```

Advisor status:

- Remaining canonical server finding is the layer-update/control/message role
  quotient.
