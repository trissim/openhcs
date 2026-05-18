# Remaining Debt Master Tracker - 2026-05-18

## Purpose

This tracker supersedes ad hoc remaining-debt discussion after the first
full-repo refactor campaign set. It is derived from a full `openhcs` advisor
scan, not file-specific discovery.

## Source Scan

Command:

```bash
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

Raw output:

- `/tmp/advisor_openhcs_remaining_20260518.txt`

Result:

- Findings: 1,133
- Current checkpoint before these plans: `adc98f9f Update full repo advisor triage`

## Exclusions

- Deprecated Textual TUI findings are excluded from active refactor campaigns
  unless they are handled by deletion/deprecation cleanup.
- Cleanup-only readability/blank-line findings are batched separately and must
  not be confused with architecture blockers.
- Known-noise entries in `advisor_known_noise.md` remain excluded unless the
  underlying architecture changes.

## Campaign Queue

| Order | Status | Campaign | Plan File | Primary Gate |
| --- | --- | --- | --- | --- |
| 9 | Complete | Orchestrator stage split continuation | `orchestrator_stage_split_continuation_20260518.md` | focused orchestrator/debug tests + advisor on orchestrator |
| 10 | Complete | Runtime viewer and streaming protocol cleanup | `runtime_viewer_protocol_cleanup_20260518.md` | mocked Napari/Fiji imports + runtime viewer tests |
| 11 | Active Focus | Active PyQt residual decomposition | `active_pyqt_residual_decomposition_20260518.md` | Qt offscreen smoke + PyQt focused tests |
| 12 | On Hold | Backend dimensional dispatch authority | `backend_dimensional_dispatch_authority_20260518.md` | focused backend tests + advisor on selected backend files |
| 13 | Active Focus | CellProfiler backend authority cleanup | `cellprofiler_backend_authority_cleanup_20260518.md` | CP compatibility/generated pipeline tests |
| 14 | Active Focus | Public API and export surface authority | `public_api_export_surface_authority_20260518.md` | import-surface tests + public API smoke |
| 15 | Pending | Active non-TUI cleanup batch | `active_non_tui_cleanup_batch_20260518.md` | targeted tests + full unit suite |

## Execution Rules

1. Start from the full-scan evidence in each plan.
2. Use file-specific advisor runs only as focused verification after selecting a
   campaign from the full scan.
3. Add characterization tests before changing risky runtime/compiler/GUI code.
4. Commit and push each completed campaign or coherent sub-campaign checkpoint.
5. Update this tracker with evidence, full unit results, and full advisor count.

## Full-Scan Evidence Summary

High-volume active, non-TUI areas from the full scan:

- CellProfiler backends: `thresholding.py`, `morphology.py`, `watershed.py`,
  `intensity_distribution.py`, `grid.py`, `zernike.py`, `illumination.py`,
  `colocalization.py`, `secondary.py`.
- Runtime viewers: `napari_stream_visualizer.py`, `napari_viewer_server.py`,
  `fiji_viewer_server.py`, `fiji_stream_visualizer.py`.
- Active PyQt: `image_browser.py`, `plate_view_widget.py`,
  `progress_tree_builder.py`, `dual_editor_window.py`,
  `step_parameter_editor.py`, `llm_pipeline_service.py`.
- Backend dimensional dispatch: `dxf_mask_pipeline.py`,
  `self_supervised_segmentation_3d.py`, `focus_torch.py`,
  `jax_nlm_processor.py`, `self_supervised_2d_deconvolution.py`,
  `self_supervised_3d_deconvolution.py`.
- Public/export surfaces and protocol probing: `__init__.py` modules,
  `unified_registry.py`, `func_registry.py`, `callable_contract.py`,
  `runtime_artifact_queries.py`.

## Current Focus Override - 2026-05-18

Backend dimensional dispatch is paused after checkpoint 4. The active focus
queue is now:

1. `active_pyqt_residual_decomposition_20260518.md`
2. `cellprofiler_backend_authority_cleanup_20260518.md`
3. `public_api_export_surface_authority_20260518.md`

These three plans should be worked before resuming
`backend_dimensional_dispatch_authority_20260518.md`.

## Execution Log

### Campaign 9 - Orchestrator Stage Split Continuation

Checkpoint:

- Extracted `CompiledContextLanePlanner` and `WorkerAssignmentPlan` for
  context grouping, worker assignment validation, and lane payload projection.
- Extracted `WorkerExecutorFactory` and `WorkerExecutorResources` for
  inline/thread/process/fork execution-mode selection and process-pool
  initializer wiring.
- Extracted `PooledWorkerLaneRunner` for executor submission, result
  collection, runtime observation merge, progress error emission, and
  fail-fast behavior.
- Extracted `ExecutorShutdownPlan`, `GpuCleanupPlan`,
  `AnalysisConsolidationPlan`, `ExecutionStateProjector`, and
  `ExecutionVisualizerCleanup` for finalization responsibilities.
- Added `tests/unit/test_orchestrator_lane_planning.py` characterization
  coverage for lane planning, executor construction, pooled lane collection,
  cleanup, consolidation skip behavior, state projection, and visualizer
  cleanup.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_orchestrator*.py tests/unit/test_debug*.py tests/unit/test_runner_cellprofiler_compatibility.py -q
# 76 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/core/orchestrator/orchestrator.py
# execute_compiled_plate oversized orchestration hub finding cleared
# remaining findings: pre-existing attribute-probe family and pipeline_config alias note
```

### Campaign 10 - Runtime Viewer And Streaming Protocol Cleanup

Checkpoint 1:

- Added `NapariLayerUpdateAuthority` and `NapariLayerUpdateRequest` to make
  Napari image/shapes/points create-or-replace behavior one shared authority.
- Routed both `napari_stream_visualizer.py` and `napari_viewer_server.py`
  through the shared layer authority.
- Deleted dead `NapariStreamVisualizer._prepare_data_for_display` residue after
  verifying it had no repository-visible call sites.
- Reused `ViewerQtEnvironmentPolicy` in Napari process setup and detached
  process launch instead of repeating platform string ladders.
- Added `NapariViewerServerRequest` as the shared process/server request record;
  public constructor/process signatures are still preserved for compatibility,
  so the advisor still reports the threaded signature family until the public
  API can move to the request object directly.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed

.venv/bin/python - <<'PY'
import sys
from types import ModuleType
class _DummyViewer: pass
napari = ModuleType('napari'); napari.Viewer = _DummyViewer
sys.modules.setdefault('napari', napari)
qtpy = ModuleType('qtpy'); qtcore = ModuleType('qtpy.QtCore'); qtwidgets = ModuleType('qtpy.QtWidgets')
class _DummyQTimer:
    @staticmethod
    def singleShot(*args, **kwargs): return None
qtcore.QTimer = _DummyQTimer
sys.modules.setdefault('qtpy', qtpy)
sys.modules.setdefault('qtpy.QtCore', qtcore)
sys.modules.setdefault('qtpy.QtWidgets', qtwidgets)
import openhcs.runtime.napari_stream_visualizer
import openhcs.runtime.napari_viewer_server
import openhcs.runtime.fiji_stream_visualizer
import openhcs.runtime.fiji_viewer_server
PY
# viewer imports ok

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/fiji_viewer_server.py
# cleared duplicated Napari layer-helper owner findings, platform dispatch findings,
# and dead _prepare_data_for_display finding
```

Checkpoint 2:

- Replaced the Napari `_execute_layer_update` enum ladder with a typed
  `StreamingDataType` route table on `NapariViewerServer`.
- Deleted unreferenced `_parse_component_info_from_path` helpers from both
  Napari runtime modules after repository-wide call-site verification.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/fiji_viewer_server.py
# cleared Napari data-type strategy ladder and unreferenced component parser findings
```

Checkpoint 3:

- Added `ComponentDimensionLabelPolicy` as the shared authority for
  human-readable Napari stacked-axis labels.
- Routed both Napari runtime modules through the shared policy instead of
  repeating channel/well/generic label branches.
- Added unit coverage for channel metadata labels, well labels, generic
  metadata labels, abbreviation fallback, and ignored `"None"` metadata.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 6 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/fiji_viewer_server.py openhcs/runtime/viewer_protocol.py
# cleared Napari component-label string dispatch findings
# remaining: shape rasterization dispatch, viewer lifecycle/membership,
# Napari/Fiji server role quotients, Fiji dimension context records,
# process signature records, and shared viewer platform strategy ladders
```

Checkpoint 4:

- Added `NapariShapeLabelRasterizer` plus `NapariShapeKind` and
  `NapariShapePaintContext` to own dense ROI-label projection.
- Replaced duplicated Napari `_shapes_to_labels` implementations with direct
  shared-rasterizer calls and deleted the forwarding wrappers.
- Added `NapariLayerLogPolicy` so counted layer logging is declared as a typed
  layer-kind policy instead of an inline enum subset.
- Added unit coverage for polygon/path rasterization and point extent behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 8 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/fiji_viewer_server.py openhcs/runtime/viewer_protocol.py
# cleared Napari shape-kind string dispatch
# cleared Napari _shapes_to_labels forwarding shells
# cleared inline layer-kind subset logging policy
```

Checkpoint 5:

- Added `ViewerProcessHandle` to centralize viewer process liveness,
  termination, forced-kill escalation, and PID formatting.
- Added `ManagedViewerLifecycleMixin` so Napari and Fiji stream visualizers
  share the same `is_running` algorithm.
- Removed structural `hasattr(self.process, ...)` process-type probes from
  Napari/Fiji liveness and stop paths.
- Added unit coverage for subprocess wrapping and fail-loud rejection of
  structural process lookalikes.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py tests/unit/test_viewer_protocol.py -q
# 10 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# cleared repeated is_running method skeleton
# cleared direct process hasattr probes
# remaining: broader viewer state membership, ping request projection,
# Napari server role quotient, dynamic/private entry witnesses, and
# shared viewer platform strategy ladders
```

Checkpoint 6:

- Added `ViewerControlPingMode`, `ViewerControlPingPolicy`, and
  `ViewerControlPingRequest` for quick and ready-required viewer control-port
  pings.
- Replaced repeated Napari/Fiji `ping_control_port(...)` projection bundles
  with the typed request object.
- Removed Napari's local quick-ping socket implementation in favor of the same
  control transport helper used by Fiji.
- Added unit coverage for the quick vs existing-viewer ping policy projection.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py tests/unit/test_viewer_protocol.py -q
# 11 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# cleared repeated ping_control_port projection finding
# cleared ping constructor-variant findings after introducing mode policy rows
```

Checkpoint 7:

- Replaced viewer process platform detection branches with
  `VIEWER_PROCESS_PLATFORM_BY_SYSTEM_NAME`.
- Replaced per-platform Qt environment branches with
  `ViewerQtPlatformEnvironmentPolicy` rows in
  `VIEWER_QT_ENVIRONMENT_POLICIES`.
- Added unit coverage for Linux, Darwin, Windows, and preconfigured Linux Qt
  environment behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 4 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/viewer_protocol.py
# shared viewer platform strategy ladder findings cleared
```

Checkpoint 8:

- Added `NapariViewerServerRequest.from_legacy_signature(...)` as the single
  projection authority for the current public Napari server/process signature.
- Routed both Napari runtime modules through that request builder and removed
  repeated request-constructor keyword maps.
- Added unit coverage for the legacy-signature request projection.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/viewer_protocol.py
# repeated NapariViewerServerRequest constructor field mapping cleared
# remaining request finding is the larger public legacy signature family
```

Checkpoint 9:

- Replaced repeated Napari/Fiji `_send_ack(...)` status string literals with
  `ViewerProtocolStatus`-derived module constants.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/fiji_viewer_server.py
# repeated viewer status literal findings cleared
```

Checkpoint 10:

- Made `ManagedViewerLifecycleMixin` an ABC with explicit
  `check_connected_viewer()` lifecycle hook.
- Replaced private Napari/Fiji `_quick_ping_check` implementations with the
  nominal hook required by the mixin.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py -q
# 5 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# dangling private _quick_ping_check finding cleared
```

Checkpoint 11:

- Added `NapariLayerStateStore` for layer objects, dimension-label maps, and
  pending debounce timers.
- Routed both Napari runtime modules through the state store.
- Added unit coverage for layer/label/timer state behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 9 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# main Napari layers/dimension_labels/pending_updates registry findings cleared
# remaining Napari state finding is the batch-processor store in napari_viewer_server.py
```

Checkpoint 12:

- Added `NapariBatchProcessorStore` for lazy per-layer batch processor
  ownership.
- Removed Napari viewer server's local `_batch_processors` registry and lock.
- Added unit coverage for one-processor-per-layer reuse.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 10 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# batch processor registry finding cleared
```

Checkpoint 13:

- Added `NapariDetachedProcessRequest` and `NapariViewerProcessEntrypoint` as
  the single detached Napari launch-code authority.
- Routed Napari process spawning and launch-command preview through the typed
  request.
- Removed the duplicate private process-entry implementation from
  `napari_stream_visualizer.py`.
- Made `napari_viewer_server.py` construct the server from
  `NapariViewerServerRequest`.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 17 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/viewer_protocol.py
# private process-entry, embedded launch payload, and repeated signature findings cleared
```

Checkpoint 14:

- Added `ViewerLifecycleState` / `ViewerLifecycleMode`.
- Replaced Napari and Fiji `_is_running` / `_connected_to_existing` flag pairs
  with nominal lifecycle transitions.
- Updated `ManagedViewerLifecycleMixin` to evaluate and reset lifecycle state.
- Registered the Napari stream visualizer global cleanup callback explicitly.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 18 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/fiji_stream_visualizer.py openhcs/runtime/viewer_protocol.py openhcs/runtime/napari_viewer_server.py
# lifecycle membership and unreferenced cleanup callback findings cleared
```

Checkpoint 15:

- Removed the duplicate `NapariViewerServer` implementation from
  `napari_stream_visualizer.py`.
- Re-exported the canonical server from `napari_viewer_server.py` for
  compatibility while leaving the stream visualizer as the process/client
  manager.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 18 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_stream_visualizer.py openhcs/runtime/napari_viewer_server.py
# stream-visualizer server role quotient cleared
```

Checkpoint 16:

- Added `NapariComponentValueTracker` for global component value accumulation
  and indexed-axis expansion.
- Routed `NapariViewerServer` through the tracker.
- Removed the no-op server `_setup_ack_socket` override and dead server-side
  detached spawn helper.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 19 passed

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py openhcs/runtime/napari_streaming_handlers.py
# setup role and dead detached-spawn findings cleared; server quotient reduced to update/control/message roles
```

Checkpoint 17:

- Removed the no-op `NapariViewerServer.handle_data_message(...)` method.
- Focused `napari_viewer_server.py` advisor scan is clean.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_viewer_protocol.py tests/unit/test_napari_streaming_handlers.py -q
# 19 passed

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/runtime/napari_viewer_server.py
# No refactoring findings.
```

## Campaign 11 - Active PyQt Residual Decomposition

Status: Active Focus

Checkpoint 1:

- Replaced `ImageBrowserWidget` result-file string dispatch with typed
  `ResultFileType` and `ResultFileAction` authorities.
- Deleted unreferenced CSV/JSON preview helpers after repository-wide call-site
  verification; CSV/JSON result double-clicks still open via the system default
  application.
- Converted `filemanager` and streaming-service access to orchestrator-derived
  properties so `set_orchestrator` no longer partially synchronizes derived
  state.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 91 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/image_browser.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/image_browser.py
# cleared file-type string dispatch, dangling CSV/JSON preview helpers, and
# orchestrator-derived state finding; remaining findings are class decomposition,
# viewer membership, and metadata/file registries.
```

Checkpoint 2:

- Replaced raw progress tree node-type strings with `ProgressNodeType`.
- Centralized aggregation policy IDs and progress-node construction.
- Replaced execution-mode enum subset checks with `ProgressChannel.role`.
- Removed local progress status predicate wrappers in favor of core progress
  semantic predicates.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui/test_progress_tree_aggregation.py tests/unit/pyqt_gui/test_execution_server_summary.py -q
# 16 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py
# reduced to the remaining class method-role quotient / subsystem extraction finding.
```

Checkpoint 3:

- Declared `DualEditorWindow` UI contract attributes during initialization
  instead of recovering them through `getattr` at use sites.
- Removed stale `_get_current_plate_from_pipeline_editor` structural probing
  residue after repository-visible call-site verification.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 91 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/windows/dual_editor_window.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/windows/dual_editor_window.py
# reflective self-attribute contract findings and dangling private method cleared;
# remaining finding is the broader attribute-probe/template-method bucket.
```

Checkpoint 4:

- Replaced `StepParameterEditorWidget` raw hierarchy item-type dispatch with
  typed `TreeItemType` and a handler table.
- Added `StepSettingsDialogRequest` as the single request shape for load/save
  cached file dialogs.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 91 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/step_parameter_editor.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/step_parameter_editor.py
# closed item-type dispatch and repeated dialog mapping cleared; remaining
# findings are class decomposition and attribute-probe/template-method families.
```

Checkpoint 5:

- Added `PlateGridModel` and `PlateGridBounds` as the pure coordinate authority
  for standard and supplied non-standard well IDs.
- Routed `PlateViewWidget` grid dimensions, row/column ranges, reverse lookup,
  and axis membership through the model instead of keeping those projections in
  QWidget state.
- Added `PlateSubdirectoryButtonRegistry` and `PlateWellButtonRegistry` so Qt
  button lookup/cleanup is owned by explicit registries rather than raw widget
  dictionaries.
- Removed the dead `_detect_dimensions` private wrapper and the trivial
  `set_well_filter_widget` transport method.
- Added pure model coverage for standard wells, supplied coordinates, explicit
  dimensions, and row/column membership.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 95 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/shared/plate_view_widget.py openhcs/pyqt_gui/widgets/image_browser.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/shared/plate_view_widget.py openhcs/pyqt_gui/widgets/image_browser.py
# cleared PlateViewWidget bidirectional registry findings for subdir_buttons
# and well_buttons, cleared the dead _detect_dimensions finding, and removed
# the well-filter transport wrapper finding.
```

Checkpoint 6:

- Added `PlateSelectionInteractionLifecycle` to own begin/update/finish
  transitions for button drag and rectangle selection.
- Reduced `PlateSelectionEventController` to event target routing and Qt event
  acceptance while the lifecycle object owns gesture state transitions.
- Focused advisor on `plate_view_widget.py` now reports only the broader
  `PlateViewWidget` facade quotient; the separate
  `PlateSelectionEventController` quotient is cleared.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 95 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/shared/plate_view_widget.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/shared/plate_view_widget.py
# cleared PlateSelectionEventController method-role quotient.
```

Checkpoint 7:

- Added `StreamingViewerField` and `ImageBrowserViewerControls` so viewer
  button construction, display names, enabled-state lookup, and selection-driven
  enablement are owned by one control authority.
- Added `ImageBrowserMetadataDisplayResolver`, `ImageBrowserImageCatalog`, and
  `ImageBrowserResultCatalog` so metadata display caching and image/result file
  catalogs are no longer raw mirrored dictionaries on `ImageBrowserWidget`.
- Removed the dead `_is_viewer_enabled` wrapper and reused private
  `_get_viewer_display_name` helper.
- Focused advisor on `image_browser.py` now reports only the broad
  `ImageBrowserWidget` facade quotient; viewer membership and bidirectional
  registry findings are cleared.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 95 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/image_browser.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/image_browser.py
# only ImageBrowserWidget class method-role quotient remains.
```

Checkpoint 8:

- Added `ProgressTreeStatusProjector` for percent aggregation and parent status
  projection.
- Added `ProgressNodeFactory` for model-node construction and
  `ProgressTreeNodeConverter` for PyQt-reactive `TreeNode` conversion.
- Routed execution-server tree syncing through the converter authority instead
  of static forwarding wrappers on `ProgressTreeBuilder`.
- Focused advisor on `progress_tree_builder.py` reports no findings.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui/test_progress_tree_aggregation.py tests/unit/pyqt_gui/test_execution_server_summary.py -q
# 16 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py openhcs/pyqt_gui/widgets/shared/server_browser/progress_projection.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py
# No refactoring findings.
```

Checkpoint 9:

- Replaced step/lazy-capability attribute probing with explicit step
  `__dict__` value reads and capability-name checks.
- Added `StepSettingsFileController` so load/save dialog handling and
  serialization behavior are no longer owned by `StepParameterEditorWidget`.
- Focused advisor on `step_parameter_editor.py` reports no findings.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
# 95 passed

.venv/bin/python -m py_compile openhcs/pyqt_gui/widgets/step_parameter_editor.py
# clean

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/pyqt_gui/widgets/step_parameter_editor.py
# No refactoring findings.
```

Remaining:

- Split `image_browser.py` plate-view/detach/filter/streaming roles into
  controllers if the remaining facade quotient should be eliminated.
- Split `plate_view_widget.py` selection mutation/filter-sync/status updates
  into smaller subsystems.
- Extract larger production PyQt subsystem boundaries for `image_browser.py`,
  `plate_view_widget.py`, `progress_tree_builder.py`, and
  `step_parameter_editor.py` in a later deeper GUI campaign.

## Campaign 12 - Backend Dimensional Dispatch Authority

Status: On Hold

Hold reason:

- User requested this campaign be paused after checkpoint 4.
- Remaining known item is the deconvolution blur-mode strategy family split.
- Resume only after the active focus queue completes:
  `active_pyqt_residual_decomposition_20260518.md`,
  `cellprofiler_backend_authority_cleanup_20260518.md`, and
  `public_api_export_surface_authority_20260518.md`.

Checkpoint 1:

- Added `DXFMaskStackProjection` in
  `openhcs/processing/backends/analysis/dxf_mask_pipeline.py`.
- Replaced local 3D/4D `image_stack.ndim` dispatch with the typed projection.
- Fixed the pre-existing unreachable registration/masking body caused by
  indentation under the invalid-dimension `else`.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/analysis/dxf_mask_pipeline.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/analysis/dxf_mask_pipeline.py
# No refactoring findings.
```

Checkpoint 2:

- Added `SegmentationVolumeProjection` in
  `openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py`.
- Replaced local 3D/4D/5D input projection and output restoration branches
  with the typed projection.
- Expanded compressed inline control flow in touched helpers.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py
# No refactoring findings.
```

Checkpoint 3:

- Added `LaplacianImageProjection`, `FocusStackProjection`, and
  `FocusSharpnessMethod` in
  `openhcs/processing/backends/enhance/focus_torch.py`.
- Replaced dimensional branches and sharpness-method string dispatch with typed
  projection/dispatch authorities.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/enhance/focus_torch.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/enhance/focus_torch.py
# No refactoring findings.
```

Checkpoint 4:

- Added typed input projection/restoration records for 2D and 3D
  self-supervised deconvolution.
- Added blur-mode enums and removed raw string checks from blur setup/apply
  branches.
- Removed dead 3D Gaussian-conv blur helper.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/enhance/self_supervised_2d_deconvolution.py openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py
# clean

git diff --check
# clean
```

Remaining:

- Split deconvolution blur-mode behavior into nominal strategy families.

## Campaign 13 - CellProfiler Backend Authority Cleanup

Checkpoint 1:

- Refactored robust-background center strategies in
  `openhcs/processing/backends/cellprofiler/thresholding.py` so repeated center
  mechanics live on `RobustBackgroundCenterStrategy` while explicit nominal
  subclasses remain registered/debuggable authorities.
- Added `CellProfilerThresholdProfiler` as the bound threshold timeline logging
  authority, replacing repeated manual `log_profile` calls in
  `cellprofiler_threshold`.
- Derived robust-background threshold kwargs from the dataclass field authority
  instead of hand-maintaining a semantic dict bag.
- Replaced the transient binned-mode forwarding helper with a callable center
  helper that owns provider-backed primitive lookup.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_library_loading.py tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_generated_pipeline_execution.py tests/unit/test_runner_cellprofiler_compatibility.py -q
# 402 passed, 5 warnings

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/cellprofiler/thresholding.py
# Cleared robust center helper wrappers, threshold profiler call-family finding,
# binned-mode forwarding wrapper, and robust-background semantic dict bag.
```

Remaining:

- Public/export `__all__` derivation belongs to Campaign 14.
- `backend_key` registry-key repetition needs a package/advisor-level
  AutoRegister key-boilerplate fix; replacing the literal with a local constant
  currently hides the stable key axis from the advisor.
- Keep explicit robust-background center subclasses unless the registry stops
  consuming nominal class identity.
- Continue CP backend authority cleanup in `intensity_distribution.py`,
  `watershed.py`, and adjacent CellProfiler backend files.

Checkpoint 2:

- Added `RadialDistributionArrays.empty` and
  `RadialDistributionArrays.from_components` as the array construction
  authority for intensity-distribution results.
- Moved repeated `measure_from_centers` and `measure_self_centered`
  orchestration from native/Numba radial-distribution subclasses into
  `RadialDistributionBackendStrategy`, leaving subclasses responsible for the
  backend-specific `measure` implementation.
- Removed repeated empty-result constructors and backend-specific self-centered
  skeletons from `openhcs/processing/backends/cellprofiler/intensity_distribution.py`.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_library_loading.py tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_generated_pipeline_execution.py tests/unit/test_runner_cellprofiler_compatibility.py -q
# 402 passed, 5 warnings

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/cellprofiler/intensity_distribution.py
# Cleared repeated self-centered algorithm skeleton and repeated
# RadialDistributionArrays constructor findings.
```

Remaining:

- Public/export `__all__` derivation belongs to Campaign 14.
- Collapse the remaining radial-distribution threaded parameter family into a
  nominal request/context record.
- Replace `measure_object_intensity_distribution` profile call repetition with
  a bound profiler object.

Checkpoint 3:

- Added `RadialDistributionMeasureRequest` and routed backend-specific radial
  measurement implementations through `_measure_request`, preserving the public
  `measure(...)` signature while giving backend implementations one nominal
  request authority.
- Added `IntensityDistributionProfiler` and routed
  `measure_object_intensity_distribution` phase logging through the bound
  profiler instead of repeated `_log_profile` call records.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_measureobjectintensitydistribution.py tests/unit/test_cellprofiler_library_loading.py tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_generated_pipeline_execution.py tests/unit/test_runner_cellprofiler_compatibility.py -q
# 408 passed, 10 warnings

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/cellprofiler/intensity_distribution.py
# Cleared measure_object_intensity_distribution profile call-family finding.
# Remaining radial parameter-family finding is now primarily public wrapper
# compatibility plus the Numba kernel boundary.
```

Remaining:

- Public/export `__all__` derivation belongs to Campaign 14.
- Decide whether to add a new public request-style radial API and deprecate the
  long-form compatibility wrappers, or leave the remaining wrapper finding as
  compatibility debt.

Checkpoint 4:

- Added `WatershedProfiler` and routed CellProfiler 4 watershed runtime phase
  logging through bound profiler methods.
- Added `LegacyWatershedRequest` for validated legacy watershed inputs and
  collapsed whole-volume/plane-wise helper parameter threading into one nominal
  request authority.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_library_loading.py tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_generated_pipeline_execution.py tests/unit/test_runner_cellprofiler_compatibility.py -q
# 402 passed, 5 warnings

timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/cellprofiler/watershed.py
# Cleared CellProfiler4WatershedRuntimeStrategy profile call-family finding and
# legacy watershed helper parameter-family finding.
```

Remaining:

- Public/export `__all__` derivation belongs to Campaign 14.
- Distance-initial watershed profile calls still need a deeper phase-spec
  extraction if we want to remove the remaining profile call-family finding
  without obscuring the real algorithm.
- Heap push/pop parameter threading remains in private Numba-compatible heap
  helpers.
