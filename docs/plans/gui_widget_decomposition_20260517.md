# Old GUI Widget Decomposition Plan

## Goal

Decompose the older broad PyQt widgets without disturbing the newer service
boundaries that are already advisor-clean.

The target is not smaller files for their own sake. The target is stable,
testable service ownership for code execution, debug command bridging, preview
formatting, deletion workflows, progress wiring, and time-travel handling.

## Current Evidence

Fresh advisor spot-check:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`: `9` findings.
- `openhcs/pyqt_gui/widgets/plate_manager.py`: `5` findings.

Main signals:

- broad `class_role_quotient`
- dangling private methods
- old UI orchestration mixed with newer debug/source-binding services

Relevant files:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/shared/services/`
- `openhcs/pyqt_gui/widgets/shared/`
- `tests/unit/pyqt_gui/test_debug_toolbar.py`
- `tests/unit/pyqt_gui/test_source_bindings_editor.py`
- existing GUI unit tests touching pipeline editor and plate manager

## Target Shape

`PipelineEditorWidget` should compose:

- a step list model/controller
- a step preview formatter
- a debug command bridge
- a code execution result adapter
- a time-travel event adapter

`PlateManagerWidget` should compose:

- a plate deletion workflow service
- a debug run/export submission bridge
- a code execution result adapter
- a progress/snapshot wiring bridge
- a time-travel event adapter

The widgets remain Qt owners for signals, object lifetime, and layout. Services
own the repeatable state transitions and request/result records.

## Non-Goals

- Do not rewrite the GUI.
- Do not move methods into one-off helpers that only rename private methods.
- Do not change debug command semantics while moving GUI code.
- Do not mix this with ZMQ server decomposition or CP runtime compatibility
  deletion in the same commit.

## Implementation Sequence

### Stage 1: Behavior Pinning

Add or extend focused tests for:

- pipeline editor code execution success/failure dispatch
- pipeline editor debug command routing
- step preview formatting for source bindings and debug badges
- plate deletion validation and execution decision
- plate manager debug artifact export handoff
- time-travel completion handling, if currently exercised

### Stage 2: Pipeline Editor Services

1. Extract a `PipelineStepPreviewFormatter` or extend the existing formatter
   family if one already owns this behavior.
2. Extract a `PipelineDebugCommandBridge` that converts toolbar commands and
   cursor state into typed plate-manager requests.
3. Extract a code execution adapter for `_apply_executed_code` and
   `_handle_code_execution_error`.
4. Replace direct private-method logic with service calls while keeping Qt
   signal ownership in `PipelineEditorWidget`.

Verification:

- `tests/unit/pyqt_gui/test_debug_toolbar.py`
- focused pipeline editor tests
- advisor on `openhcs/pyqt_gui/widgets/pipeline_editor.py`

### Stage 3: Plate Manager Services

1. Extract a `PlateDeletionWorkflow` with typed validation and execution
   results.
2. Extract or reuse the debug workflow service for debug run/export submission.
3. Extract code execution and time-travel adapters only if they own reusable
   request/result semantics.
4. Keep `PlateManagerWidget` as the signal/layout/lifetime facade.

Verification:

- focused plate-manager tests
- debug toolbar/export tests
- advisor on `openhcs/pyqt_gui/widgets/plate_manager.py`

### Stage 4: Integration Gate

Run:

```bash
.venv/bin/python -m pytest tests/unit/pyqt_gui -q --tb=short --disable-warnings
.venv/bin/python -m pytest tests/unit -q --tb=short --disable-warnings
```

## Completion Criteria

- `PipelineEditorWidget` and `PlateManagerWidget` advisor findings are
  substantially reduced or the remaining findings are explicit Qt facade noise.
- Extracted classes own real GUI workflow semantics, not shallow forwarding.
- Existing debug/source-binding GUI behavior is unchanged.

## Progress: 2026-05-17

Completed decomposition slices:

- `PipelineEditorCodeWorkflow` owns edited pipeline-code application and legacy
  constructor migration fallback.
- `PipelineEditorDeletionWorkflow` owns atomic step deletion and backing
  `ObjectState` synchronization.
- `PipelineEditorListWorkflow` owns list preparation, reorder side effects, and
  time-travel restoration.
- `PipelineStepSaveWorkflow` owns edited-step replacement while preserving
  scope-token continuity.
- `PlateManagerCodeWorkflow` owns orchestrator-code application, plate entry
  creation, config propagation, pipeline data propagation, and compilation
  invalidation.
- `PlateManagerDeletionWorkflow` owns delete validation and plate/ObjectState
  cleanup.

Verification:

- `tests/unit/pyqt_gui/test_debug_toolbar.py tests/unit/pyqt_gui/test_source_bindings_editor.py`:
  `33 passed`.
- `tests/unit/pyqt_gui`: `89 passed`.
- Advisor:
  - `pipeline_editor_workflows.py`: `0`.
  - `plate_manager_workflows.py`: `0`.

Remaining:

- `PipelineEditorWidget` and `PlateManagerWidget` still expose private hook
  bridges required by the external `pyqt_reactive.AbstractManagerWidget`
  template contract. Eliminating the forwarding-wrapper findings properly
  requires a dependency-level refactor of that base contract to accept nominal
  hook providers instead of private subclass methods.
- Larger widget role-quotient findings remain for editor display formatting,
  debug command routing, time-travel handling, and plate-manager status/progress
  formatting. Continue extracting cohesive services rather than adding local
  private helpers.
