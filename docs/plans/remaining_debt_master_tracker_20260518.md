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
| 10 | In Progress | Runtime viewer and streaming protocol cleanup | `runtime_viewer_protocol_cleanup_20260518.md` | mocked Napari/Fiji imports + runtime viewer tests |
| 11 | Pending | Active PyQt residual decomposition | `active_pyqt_residual_decomposition_20260518.md` | Qt offscreen smoke + PyQt focused tests |
| 12 | Pending | Backend dimensional dispatch authority | `backend_dimensional_dispatch_authority_20260518.md` | focused backend tests + advisor on selected backend files |
| 13 | Pending | CellProfiler backend authority cleanup | `cellprofiler_backend_authority_cleanup_20260518.md` | CP compatibility/generated pipeline tests |
| 14 | Pending | Public API and export surface authority | `public_api_export_surface_authority_20260518.md` | import-surface tests + public API smoke |
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
