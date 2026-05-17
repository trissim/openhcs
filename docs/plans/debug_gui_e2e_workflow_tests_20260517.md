# Debug GUI End-to-End Workflow Test Plan

## Goal

Add heavier debug GUI workflow coverage for the behavior that unit/control tests
do not fully prove: pause, step, continue, stop, snapshot inspection, export, and
worker lifetime across multiple commands.

## Current Evidence

Existing coverage includes:

- typed debug models and stores in `tests/unit/test_debug_runtime.py`
- live ZMQ debug worker command-loop status test
- paused-worker controller behavior test
- GUI command routing tests in `tests/unit/pyqt_gui/test_debug_toolbar.py`
- inspector/export request unit coverage

Missing coverage:

- a host GUI workflow that exercises the same `PipelineEditorWidget` and
  `PlateManagerWidget` signal path over multiple debug commands.
- snapshot inspection and export through the GUI path while a server/worker is
  alive.

## Target Shape

One or more tests should drive:

1. Start debug session.
2. Step to first invocation.
3. Continue/run to pause.
4. Inspect announced snapshot in the debug inspector path.
5. Export or open a selected artifact via typed request.
6. Stop session.
7. Assert worker/session lifetime transitions across commands.

## Non-Goals

- Do not run official30 or heavy CellProfiler pipelines in this test.
- Do not require a real display if the existing Qt test harness can run
  headless.
- Do not duplicate lower-level control-channel tests.
- Do not assert visual pixel details.

## Implementation Sequence

### Stage 1: Test Harness Inventory

Inspect:

- existing PyQt test fixtures
- `tests/unit/pyqt_gui/test_debug_toolbar.py`
- `tests/unit/test_debug_runtime.py`
- any existing ZMQ server fixtures

Choose the lightest harness that exercises the actual GUI signal path.

### Stage 2: Minimal Native Pipeline

Use a tiny native OpenHCS function-pattern pipeline with two invocations and
debug pause enabled. Prefer native functions over CellProfiler modules for this
test to isolate the debug substrate.

### Stage 3: GUI Command Workflow

Drive toolbar or command dispatch methods rather than calling worker internals.
Capture:

- active debug session id
- current cursor after step
- snapshot notification
- inspector loaded snapshot
- export/open typed request
- stopped/cleared session state

### Stage 4: Optional ZMQ Live Layer

If the existing unit GUI harness cannot drive a real server safely, keep the GUI
workflow test with fake service seams and add a separate live ZMQ workflow test
marked slow/integration.

## Verification

```bash
.venv/bin/python -m pytest tests/unit/pyqt_gui/test_debug_toolbar.py tests/unit/test_debug_runtime.py -q --tb=short --disable-warnings
```

If a new integration marker is needed, document it in the test name and avoid
running it in the default unit suite unless stable and fast.

## Completion Criteria

- At least one test covers multi-command debug GUI session lifetime.
- Snapshot inspection is verified through the GUI notification/inspector path.
- Export/open request creation is verified at the host GUI boundary.
- Lower-level control tests remain passing.

