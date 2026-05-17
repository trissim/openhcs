# ZMQ Server Decomposition Plan

## Goal

Split `ZMQExecutionServer` along real transport and execution boundaries while
preserving the existing client/server protocol.

The server currently owns request normalization, execution signature building,
debug replay signature building, worker debug commands, snapshot reads, artifact
exports, orchestrator lifecycle, and progress emission. The refactor must name
those boundaries before touching the large orchestration method.

## Current Evidence

Fresh advisor spot-check on `openhcs/runtime/zmq_execution_server.py`: `10`
findings.

Main signals:

- attribute probes
- repeated `_emit_zmq_progress(...)` builder calls
- broad `ZMQExecutionServer` class role quotient
- `_execute_with_orchestrator` orchestration hub
- repeated semantic parameter family across request signature helpers

Relevant files:

- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/core/debug.py`
- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py`
- `tests/unit/test_debug_runtime.py`
- ZMQ-related integration tests under `tests/integration`

## Target Shape

- `ZMQExecutionSignatureBuilder`
  - owns run, compile, debug replay, and cache signature construction from one
    typed request record.
- `ZMQProgressEventEmitter` or `ZMQProgressEventFactory`
  - owns status/progress payload construction before sending through the server.
- `DebugControlRequestRouter`
  - owns snapshot read, artifact export, and worker command dispatch.
- `ZMQOrchestratorExecutionDriver`
  - owns the phase split currently inside `_execute_with_orchestrator`.
- `ZMQExecutionServer`
  - remains the socket/protocol facade and delegates to the above services.

## Non-Goals

- Do not change the wire protocol unless a compatibility test is updated first.
- Do not split `_execute_with_orchestrator` mechanically before typed request and
  progress records exist.
- Do not introduce structural `getattr`/`hasattr` replacements; define nominal
  request records instead.
- Do not mix with GUI widget decomposition in the same commit.

## Implementation Sequence

### Stage 1: Characterization

1. Add focused tests for request signature stability if missing.
2. Add focused tests for debug control routing using typed request/response
   records.
3. Snapshot representative progress payloads emitted during orchestrator phases.

### Stage 2: Debug Control Router

1. Move `_handle_debug_snapshot_read`, `_handle_debug_artifact_export`, and
   `_handle_debug_worker_command` behavior into a router object.
2. Keep server methods as compatibility delegators only if tests or external
   callers need them.
3. Run debug runtime tests and advisor.

### Stage 3: Signature Builder

1. Introduce a typed execution signature input record.
2. Move `_build_request_signature` and `_build_debug_replay_signature` logic into
   the builder.
3. Ensure compile/run/debug replay signatures remain byte-for-byte stable where
   existing cache behavior depends on it.

### Stage 4: Progress Event Factory

1. Introduce typed progress request records for queued/running/completed/error
   states.
2. Replace repeated `_emit_zmq_progress(...)` keyword bundles with request
   objects.
3. Keep `_emit_zmq_progress` as the socket-send boundary.

### Stage 5: Orchestrator Driver

1. Split `_execute_with_orchestrator` by phases after stages 2-4 have removed
   repeated parameter threading.
2. Preserve cleanup/finalization guarantees.
3. Run focused and full unit gates.

## Verification

Focused:

```bash
.venv/bin/python -m pytest tests/unit/test_debug_runtime.py -q --tb=short --disable-warnings
```

Broader:

```bash
.venv/bin/python -m pytest tests/unit -q --tb=short --disable-warnings
```

Advisor:

```bash
.venv/bin/python -m nominal_refactor_advisor openhcs/runtime/zmq_execution_server.py --json --min-hardcoded-string-sites 3 --min-builder-keywords 3
```

## Completion Criteria

- Debug control routing is separated from socket facade concerns.
- Signature construction is centralized and typed.
- Repeated progress payload construction is removed.
- `_execute_with_orchestrator` is smaller because real phase collaborators own
  behavior, not because code was relocated into anonymous helpers.

