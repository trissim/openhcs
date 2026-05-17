# ZMQ Server And Runtime Transport Decomposition Plan

## Goal

Complete the ZMQ/server-side decomposition without regressing the established
protocol, debug control loop, progress events, or orchestrator execution path.

The original `ZMQExecutionServer` split is mostly complete. The next large
campaign is not more arbitrary extraction from that file; it is consolidating
adjacent runtime transport/viewer systems that still carry duplicated protocol,
viewer, process-launch, and platform-dispatch logic.

## Verified Current State

Core ZMQ files now exist:

- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/runtime/zmq_debug_control.py`
- `openhcs/runtime/zmq_execution_signature.py`
- `openhcs/runtime/zmq_progress.py`
- `openhcs/runtime/zmq_orchestrator_environment.py`
- `openhcs/runtime/zmq_compilation.py`
- `openhcs/runtime/zmq_worker_execution.py`
- `openhcs/runtime/zmq_server_hooks.py`

Prior verification recorded:

- `tests/unit/test_debug_runtime.py`: passing
- `tests/unit`: passing
- advisor-clean extracted ZMQ service files

Fresh runtime advisor scan shows that the biggest remaining runtime debt is now
outside the core ZMQ execution server:

- duplicated Napari viewer/server code
- duplicated Fiji/Napari streaming server code
- string/numeric dispatch for viewer protocols
- manual process-launch keyword bundles
- platform dispatch by raw strings
- manual bidirectional registries in viewer state
- repeated `_create_pong_response`
- class-marker membership checks in stream visualizers

This means the next ZMQ/runtime campaign should be named more accurately:
`runtime transport and viewer server consolidation`.

## Target Architecture

### ZMQ Execution Core

Keep:

- `ZMQExecutionServer` as protocol/socket facade.
- `ZMQExecutionClient` as client protocol facade.
- `ZMQProgressEmitter` as semantic progress event builder.
- `DebugControlMessageStrategy` as debug control router.
- `ZMQWorkerExecutionRequest` as worker execution boundary.
- `ZMQCompilationRequest` as compile/cache boundary.
- `ZMQOrchestratorEnvironmentRequest` as environment setup boundary.

Do not reopen these unless a focused advisor or test shows real debt.

### Viewer/Transport Runtime

Introduce:

- `ViewerType`
  - enum for `napari`, `fiji`, and any future viewer identity
- `ViewerProtocolStatus`
  - enum/value object for `success`, `error`, `complete`, etc.
- `StreamingDataKind`
  - image, points, shapes, ROI payloads
- `ViewerLayerKind`
  - image, labels, points, shapes, ROI layer behavior
- `ViewerProcessLaunchRequest`
  - command, cwd, env, stdout/stderr policy, platform flags
- `ViewerPlatformPolicy`
  - Linux/Darwin/Windows launch-env and process-spawn behavior
- `ViewerHeartbeatResponse`
  - shared `_create_pong_response` payload for Fiji/Napari servers
- `ViewerStateRegistry`
  - owns layer/window/pending-update bidirectional registries
- `StreamingVisualizerLifecycle`
  - replaces class-marker checks in `is_running` and connect/reuse paths

## Non-Goals

- Do not change ZMQ wire protocol without compatibility tests.
- Do not rewrite Napari/Fiji feature behavior in the same pass.
- Do not merge GUI server browser refactors into runtime server refactors.
- Do not add generic wrappers around `Popen`; create a typed process launch
  request that owns platform behavior.

## Implementation Passes

### Pass 1: ZMQ Core Regression Gate

Before touching runtime viewer transport, run:

```bash
.venv/bin/python -m pytest tests/unit/test_debug_runtime.py -q
.venv/bin/python -m nominal_refactor_advisor openhcs/runtime/zmq_execution_server.py
```

If ZMQ core findings appear, fix those first. Otherwise leave ZMQ core stable.

### Pass 2: Viewer Protocol Domain

1. Add `ViewerType`, `ViewerProtocolStatus`, `StreamingDataKind`, and
   `ViewerLayerKind`.
2. Replace repeated hardcoded strings in Napari/Fiji viewer server responses.
3. Replace string dispatch for data/layer kinds with typed strategies.

Affected files:

- `openhcs/runtime/fiji_viewer_server.py`
- `openhcs/runtime/napari_stream_visualizer.py`
- `openhcs/runtime/napari_viewer_server.py`

### Pass 3: Process Launch Domain

1. Add `ViewerProcessLaunchRequest`.
2. Add `ViewerPlatformPolicy`.
3. Route Napari and Fiji detached process launching through typed requests.
4. Preserve exact environment variables and platform flags.

Advisor targets:

- repeated `Popen(...)` keyword bundles
- raw `platform.system()` string dispatch
- repeated `cocoa` / `xcb` strings

### Pass 4: Shared Viewer Server Base

1. Move `_create_pong_response` into the shared streaming viewer server base.
2. Replace concrete constant residue with `ViewerType`.
3. Add tests for Fiji and Napari pong response payloads.

### Pass 5: Viewer State Registry

1. Introduce `ViewerStateRegistry` or per-viewer typed registries.
2. Replace manual mirrored dictionaries for:
   - layers
   - dimension labels
   - pending updates
   - hyperstack metadata
3. Preserve update/removal semantics with tests.

### Pass 6: Visualizer Lifecycle

1. Replace class-marker membership checks with a nominal lifecycle state object.
2. Remove structural `hasattr` checks by introducing typed optional capability
   adapters for subprocess, socket, viewer process, and control channel.
3. Add tests for running, stopped, reused, and failed viewer lifecycle states.

### Pass 7: Integrated Runtime Gate

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_debug_runtime.py -q
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m nominal_refactor_advisor openhcs/runtime
```

Manual smoke, if viewer dependencies are available:

```bash
.venv/bin/python -m pytest tests/pyqt_gui/integration -q
```

## Completion Criteria

- Core ZMQ execution files remain stable and tests pass.
- Viewer protocol strings are represented by nominal enums/value objects.
- Detached process launching is owned by typed launch requests.
- Napari/Fiji duplicate server code is reduced through shared base behavior.
- Runtime advisor findings are materially reduced without weakening protocol
  compatibility.
