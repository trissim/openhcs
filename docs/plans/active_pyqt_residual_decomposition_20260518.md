# Active PyQt Residual Decomposition - 2026-05-18

## Full-Scan Evidence

Active PyQt files with high non-TUI finding density:

- `openhcs/pyqt_gui/widgets/image_browser.py`
- `openhcs/pyqt_gui/widgets/shared/plate_view_widget.py`
- `openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py`
- `openhcs/pyqt_gui/windows/dual_editor_window.py`
- `openhcs/pyqt_gui/widgets/step_parameter_editor.py`
- `openhcs/pyqt_gui/services/llm_pipeline_service.py`
- `openhcs/pyqt_gui/testing/event_recorder.py`

Finding families include:

- class method-role quotient;
- bidirectional registries;
- reflective self-attribute access;
- enum subset guards;
- small repeated method templates;
- manual class-marker membership.

## Current State

The first-wave PyQt campaign extracted `PlateViewWidget.eventFilter`, but the
scan still identifies larger facade classes and mirrored registries.

Checkpoint 1:

- `ImageBrowserWidget` result-file double-click handling now routes through
  typed `ResultFileType` and `ResultFileAction` authorities instead of raw
  `"ROI"` / `"CSV"` / `"JSON"` string dispatch.
- Unreferenced CSV/JSON preview helpers were deleted after repository-wide
  call-site verification; CSV/JSON result files continue to open in the system
  default application.
- `filemanager` and streaming service access now derive from the current
  `orchestrator` through properties, removing stale derived state from
  `set_orchestrator`.

Checkpoint 2:

- `ProgressTreeBuilder` now uses typed `ProgressNodeType` and shared
  aggregation policy IDs instead of repeated node-type and policy strings.
- Progress-node creation is centralized through builder methods, with
  specialized step/well node constructors for the repeated well-detail rows.
- Execution-mode detection now uses `ProgressChannel.role` rather than a local
  hardcoded enum subset, and status predicates call the core progress semantics
  directly.

Checkpoint 3:

- `DualEditorWindow` now declares `header_label`,
  `_default_size_applied`, and `_save_button_base_style` during initialization
  instead of probing those attributes reflectively.
- Window-title and save-button updates now use direct fail-loud attributes for
  the declared UI contract.
- Removed the unreferenced `_get_current_plate_from_pipeline_editor` method,
  which was stale Textual-era structural probing residue.

Checkpoint 4:

- `StepParameterEditorWidget` hierarchy tree item handling now uses typed
  `TreeItemType` plus a handler table instead of raw string dispatch.
- Load/save step-settings file dialogs now route through
  `StepSettingsDialogRequest`, removing duplicated cached-dialog keyword
  mappings.
- Focused advisor output is reduced to the larger class decomposition and
  existing attribute-probe/template-method families.

Checkpoint 5:

- `LLMPipelineService` is now a generation/connection facade instead of also
  owning prompt catalog and function-documentation projection.
- Added `LLMPromptBuilder`, `LLMFunctionDocumentationBuilder`, and
  `LLMPromptResourceCatalog` so prompt assembly, registry documentation, and
  static/dynamic prompt resources have separate authorities.
- Focused advisor output for `llm_pipeline_service.py` is clean; the remaining
  active PyQt findings are `ImageBrowserWidget`, `PlateViewWidget`,
  `DualEditorWindow`, and the cross-widget initialization skeleton finding.

Checkpoint 6:

- Added `ImageBrowserFilterController` as the search/folder/well/column-filter
  authority for `ImageBrowserWidget`.
- Moved single-pass combined filtering, metadata display precomputation,
  column-filter construction, search-service ownership, and well-filter sync
  out of the widget facade.
- Added a controller-level test that pins folder/result-folder, well, and
  column-filter semantics without constructing the full widget.
- Focused advisor shows `ImageBrowserWidget` reduced from 50 methods to 43
  methods; remaining work is still the broader table/streaming/plate-view
  subsystem split.

Checkpoint 7:

- Added `PlateSelectionController` as the well-selection mutation, status, and
  well-filter synchronization authority for `PlateViewWidget`.
- Fixed the existing `PlateSelectionEventController` lifecycle boundary:
  `PlateSelectionInteractionLifecycle` no longer recursively constructs itself,
  and the event controller now owns one lifecycle instance explicitly.
- Moved rectangle selection, row/column axis toggles, inversion, public
  select/clear delegation, and filter sync out of the widget body.
- Added a Qt-offscreen selection-controller test covering select, invert, and
  clear behavior.
- Focused advisor shows `PlateViewWidget` reduced from 25 methods to 17
  methods; remaining work is subdirectory/grid rendering extraction.

Checkpoint 8:

- Added `PlateSubdirectoryController` as the subdirectory selector state and
  visibility authority for `PlateViewWidget`.
- Kept mode handling as a small table dispatch so the widget no longer owns the
  selector state machine, without introducing a fake strategy hierarchy for
  three local UI states.
- Focused Qt-offscreen plate tests pass, and focused advisor output for
  `plate_view_widget.py` is clean.

## Target Shape

Introduce active PyQt service boundaries:

- `ImageBrowserPlateViewController`
- `ImageBrowserDetachController`
- `PlateGridModel`
- `PlateSelectionModel`
- `ProgressTreeProjection`
- `DualEditorSessionModel`
- `StepParameterEditSession`
- `RecordedWidgetEvent`

Widgets should keep Qt ownership and signal wiring, while state transitions and
projection logic move into testable services/models.

## Phases

1. Add offscreen smoke tests for current widget/controller behavior.
2. Split `image_browser.py` plate-view/detach/filter sync into controllers.
3. Split `plate_view_widget.py` grid model, subdirectory model, and filter sync.
4. Convert `progress_tree_builder.py` marker checks into typed progress-node
   projection records.
5. Collapse `EventRecorder` repeated record methods into one typed event
   builder.
6. Audit `dual_editor_window.py` and `step_parameter_editor.py` for state
   models that can be tested without Qt event loops.

## Verification Gates

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
import openhcs.pyqt_gui.widgets.image_browser
import openhcs.pyqt_gui.widgets.shared.plate_view_widget
import openhcs.pyqt_gui.widgets.shared.server_browser.progress_tree_builder
PY
timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui
```

## Completion Criteria

- Active PyQt high-density findings are reduced without touching deprecated TUI.
- Qt widgets become facades over typed services/models.
- Offscreen smoke and PyQt unit tests pass.
