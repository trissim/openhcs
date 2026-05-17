# Debug GUI End-to-End Workflow Test Plan

## Goal

Add heavier debug workflow tests that prove the GUI, ZMQ server/client, paused
worker, snapshot store, inspector, and artifact export path work together over
multiple commands.

Current unit coverage is strong, but it is still split across mocked GUI seams,
debug runtime unit tests, and one live ZMQ command-loop smoke. The missing proof
is an end-to-end host workflow where the GUI drives debug commands and observes
snapshot/export state through the same public boundaries a user would exercise.

## Verified Current State

Existing lower-level coverage in `tests/unit/test_debug_runtime.py` includes:

- debug cursor and invocation identity
- debug config serialization
- debug event emission
- local and FileManager snapshot store round trips
- progress debug event sink
- warm replay hydration and stale artifact rejection
- artifact export plan materialization
- ZMQ snapshot read control request
- ZMQ artifact export control request
- live ZMQ debug worker command-loop status round trip
- paused worker controller stepping after snapshot boundary
- ZMQ paused-worker command routing

Existing GUI/unit coverage includes:

- `tests/unit/pyqt_gui/test_debug_toolbar.py`
- debug inspector window request creation
- GUI command-path workflow with faked batch/ZMQ seams

Missing coverage:

- live GUI-to-ZMQ command loop
- multiple debug commands across one persistent paused worker from GUI-facing
  services
- snapshot inspection through GUI window/service boundaries while the live
  server is active
- artifact export/open destination workflow through host GUI
- worker/session lifetime after stop and after command errors

## Target Test Matrix

### Unit GUI Workflow

Fast, default unit test:

1. Create minimal pipeline/debug session using fake services.
2. Start debug run.
3. Step.
4. Continue/run to pause.
5. Load snapshot into inspector.
6. Request export/open.
7. Stop.

This already mostly exists and should be preserved as the fast regression gate.

### Live ZMQ Workflow

Integration or marked-slow test:

1. Start `ZMQExecutionServer` on an unused local port.
2. Build a minimal native pipeline with two function-pattern invocations.
3. Submit a debug run through the GUI-facing service or plate-manager workflow.
4. Wait for first paused snapshot progress event.
5. Send step command through `ZMQExecutionClient`.
6. Send continue/run command.
7. Read snapshot through server control request.
8. Export one artifact through server control request.
9. Stop worker.
10. Assert final worker status and no leaked session.

### GUI Host Workflow

Integration test using Qt harness:

1. Instantiate `PipelineEditorWidget`, `PlateManagerWidget`, or a smaller host
   workflow harness.
2. Wire to a live or fake `ZMQClientService`.
3. Trigger toolbar actions rather than calling worker internals.
4. Assert visible GUI state and typed requests.

## Required Test Fixtures

Add or consolidate:

- unused TCP port fixture
- live server fixture with deterministic shutdown
- minimal debug pipeline fixture
- temporary snapshot root fixture
- GUI host fixture with `qtbot`
- fake plate/orchestrator fixture with one well/sample
- artifact export destination fixture

Avoid global ports such as `7777` in tests unless the test explicitly owns the
server process.

## Non-Goals

- Do not run CellProfiler official30 in debug GUI tests.
- Do not assert visual pixel details.
- Do not require a real display if Qt headless/offscreen works.
- Do not put slow live-server tests into the default unit suite unless they are
  stable and under a few seconds.
- Do not duplicate lower-level debug-runtime unit assertions.

## Implementation Passes

### Pass 1: Fixture Hardening

1. Add deterministic live ZMQ server fixture.
2. Add deterministic `ZMQExecutionClient` fixture.
3. Add snapshot root/export root fixtures.
4. Ensure shutdown closes server sockets and paused workers.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_debug_runtime.py -q
```

### Pass 2: Minimal Native Debug Pipeline

1. Create a tiny two-step or two-invocation native pipeline.
2. Ensure it emits snapshot refs without heavy image data.
3. Assert `DebugCursor` moves between invocations.

### Pass 3: Live Command Loop Test

1. Submit a debug run.
2. Poll progress until paused.
3. Send `STEP`, `RUN` or `CONTINUE`, and `STOP`.
4. Assert worker lifetime transitions.

This can live in `tests/unit/test_debug_runtime.py` only if fast; otherwise put
it under `tests/integration` or `tests/pyqt_gui/integration` with a clear mark.

### Pass 4: GUI Signal Path Test

1. Use toolbar or workflow service APIs rather than worker internals.
2. Assert the GUI emits/receives typed requests:
   - debug command request
   - snapshot inspection request
   - artifact export request
3. Keep any fake service seams typed and documented.

### Pass 5: Inspector Export/Open Flow

1. Add host destination selection seam.
2. Test export destination and viewer/open request creation.
3. Add negative tests for missing artifact, stale artifact identity, and
   unreadable VFS payload where applicable.

## Verification Gates

Fast:

```bash
.venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_debug_inspector_window.py \
  tests/unit/test_debug_runtime.py \
  -q
```

Full unit:

```bash
.venv/bin/python -m pytest tests/unit -q
```

Optional integration:

```bash
.venv/bin/python -m pytest tests/pyqt_gui/integration -q
```

## Completion Criteria

- One fast GUI command-path test remains in the default unit suite.
- One live ZMQ debug command-loop workflow proves persistent paused-worker
  behavior over multiple commands.
- Snapshot inspection is covered through GUI-facing boundaries.
- Artifact export/open is covered through typed GUI requests.
- Server and worker shutdown are deterministic with no leaked local processes.
