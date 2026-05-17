# Old GUI Widget Decomposition Plan

## Goal

Aggressively decompose the remaining large PyQt widgets while preserving Qt
ownership boundaries. The next campaign should reduce `PipelineEditorWidget`,
`PlateManagerWidget`, `SourceBindingsEditorWidget`, and debug inspector seams
into composable presenters/controllers/services, not smaller piles of private
methods.

The correct endpoint is not "no widget logic." Widgets should still own Qt
layout, signals, object lifetime, and direct child-widget wiring. Domain logic,
formatting, validation, workflow decisions, and request construction should be
owned by nominal services and presentation models.

## Verified Current State

Current file sizes:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`: 1468 lines
- `openhcs/pyqt_gui/widgets/plate_manager.py`: 1687 lines
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`: large but already
  structured around typed editor rows, codecs, dialogs, and view models
- `openhcs/pyqt_gui/windows/debug_inspector_window.py`: owns artifact action
  UI and snapshot rendering

Advisor scan over `openhcs/pyqt_gui` completed without filtered numbered
findings in the current baseline. That does not mean there is no work; it means
the remaining debt is architectural size/ownership and framework hook shape
rather than obvious local smells.

Existing extracted services include:

- `openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py`
- `openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py`
- `openhcs/pyqt_gui/widgets/shared/services/debug_workflow_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/execution_control_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/execution_submission_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/plate_status_presenter.py`
- `openhcs/pyqt_gui/widgets/shared/services/execution_server_status_presenter.py`
- `openhcs/pyqt_gui/widgets/shared/services/widget_action_dispatch.py`

Existing relevant tests:

- `tests/unit/pyqt_gui`
- `tests/pyqt_gui/integration/test_end_to_end_workflow_foundation.py`
- `tests/pyqt_gui/integration/test_reset_placeholder_simplified.py`

## Target Architecture

### Pipeline Editor

`PipelineEditorWidget` should compose:

- `PipelineEditorQtFacade`
  - Qt lifecycle, signal connections, layout, list widget ownership
- `PipelineStepListController`
  - selection, reorder, deletion, save/replace, `ObjectState` synchronization
- `PipelineStepPreviewPresenter`
  - preview formatting for function patterns, source bindings, debug badges,
    materialization flags, and validation state
- `PipelineDebugCommandBridge`
  - toolbar actions into typed plate-manager/debug workflow requests
- `PipelineCodeExecutionPresenter`
  - success/error application of generated code
- `PipelineTimeTravelAdapter`
  - time-travel restoration and event completion semantics

### Plate Manager

`PlateManagerWidget` should compose:

- `PlateManagerQtFacade`
  - Qt lifecycle, tree/list widgets, child window ownership
- `PlateListPresenter`
  - item text, icons/status, queue positions, preview text
- `PlateOperationValidator`
  - compile/run/debug/export eligibility
- `PlateExecutionRequestBuilder`
  - compile/run/debug request construction
- `PlateDebugSessionController`
  - active debug sessions, worker command routing, snapshot signals
- `PlateArtifactExportController`
  - debug inspector export/open destination workflow
- `PlateConfigWindowController`
  - config window construction, save, cache propagation
- `PlateTimeTravelAdapter`
  - orchestrator state restore and completion handling

### Source Bindings Editor

`SourceBindingsEditorWidget` already has good domain structure, but the next
polish pass should extract richer table-dialog cell widgets:

- enum combo cells for filter subject/operator
- metadata-field picker cells
- selector component picker cells
- match-dimension matrix widget
- validation hints backed by source schema/inventory

### Debug Inspector

`DebugInspectorWindow` should separate:

- snapshot loading
- view-model rendering
- artifact action presentation
- export/open request construction
- host destination selection

## Non-Goals

- Do not rewrite PyQt screens from scratch.
- Do not move Qt signal ownership into non-Qt services.
- Do not make one-off helper classes that are only renamed private methods.
- Do not combine GUI decomposition with CP runtime compatibility deletion or
  benchmark parity changes.
- Do not remove existing `pyqt_reactive.AbstractManagerWidget` hook bridges
  unless the base contract is refactored in the same pass.

## Implementation Passes

### Pass 1: Characterization Tests

Add or confirm tests for:

- pipeline step preview formatting with source-binding/debug badges
- pipeline code execution success/error
- pipeline step delete/reorder/save behavior
- plate compile/run/debug eligibility and request construction
- plate status/queue rendering
- debug inspector export/open request creation
- time-travel restoration for pipeline and plate state

Run:

```bash
.venv/bin/python -m pytest tests/unit/pyqt_gui -q
```

### Pass 2: Pipeline Editor Preview And Command Split

1. Extract `PipelineStepPreviewPresenter`.
2. Extract `PipelineDebugCommandBridge`.
3. Keep signal connections in `PipelineEditorWidget`.
4. Verify that function-pattern badge rendering is still present.

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  -q
```

### Pass 3: Pipeline Editor List/State Split

1. Move remaining list mutation and `ObjectState` synchronization into a
   controller that owns real state transitions.
2. Keep widget list item construction in the Qt facade unless presenter data is
   reusable.
3. Add tests for delete/reorder/save if missing.

### Pass 4: Plate Manager Presentation Split

1. Extract `PlateListPresenter` for item text/status/queue/preview.
2. Extract `PlateOperationValidator`.
3. Ensure existing `PlateManagerCodeWorkflow` and `PlateManagerDeletionWorkflow`
   remain the state-changing authorities.

Focused tests:

```bash
.venv/bin/python -m pytest tests/unit/pyqt_gui -q
```

### Pass 5: Plate Debug And Artifact Split

1. Extract `PlateDebugSessionController`.
2. Extract `PlateArtifactExportController`.
3. Wire `DebugInspectorWindow.artifact_export_requested` through typed host
   requests and destination selection.
4. Keep local/VFS materialization behavior in core/runtime debug services.

### Pass 6: Source Binding Dialog Polish

1. Replace text-area-plus-suggestions with row editors where the domain is
   typed.
2. Add validation hints for invalid selectors, filters, metadata fields, and
   match dimensions.
3. Keep serialization through `StepSourceBindingsConfig`.

### Pass 7: Integrated GUI Gate

Run:

```bash
.venv/bin/python -m pytest tests/unit/pyqt_gui -q
.venv/bin/python -m pytest tests/unit -q
```

Optional manual/integration:

```bash
.venv/bin/python -m pytest tests/pyqt_gui/integration -q
```

## Completion Criteria

- `PipelineEditorWidget` and `PlateManagerWidget` are mostly Qt facades.
- Extracted classes own real GUI workflow semantics.
- `tests/unit/pyqt_gui` and `tests/unit` pass.
- Any remaining widget-private methods are either Qt framework hooks or direct
  child-widget signal slots.
