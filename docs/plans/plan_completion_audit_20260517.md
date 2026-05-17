# Plan Completion Audit - 2026-05-17

## Scope

This audit reconciles the current plan files against the current implementation.
It is intentionally stricter than prior progress notes: a plan is only marked
complete when the code seam exists, focused tests or advisor scans support the
claim, and remaining work is either explicit polish or a separate queued
refactor.

## Current Verified Checkpoint

- Parent repo checkpoint: `09e51bd8 Eliminate runtime equivalence advisor debt`.
- Submodule checkpoint: `external/metaclass-registry` at `d4b5f5d Add strategy key registry attribute`.
- Last full unit gate before this audit: `tests/unit`: `1485 passed, 10 warnings`.
- Last metaclass-registry gate before this audit: `29 passed`.
- Fresh advisor spot-checks in this audit:
  - `openhcs/core/runtime_equivalence.py`: `0`.
  - `openhcs/interop/cellprofiler/pipeline_generator.py`: `0`.
  - `openhcs/interop/cellprofiler/runtime/module_execution.py`: `0`.
  - `openhcs/core/pipeline/path_planner.py`: `0`.
  - `openhcs/pyqt_gui/widgets/source_bindings_editor.py`: `0`.
  - `openhcs/core/debug.py`: `0`.
  - `openhcs/core/debug_views.py`: `0`.

## Plan Status Summary

| Plan | Status | Current conclusion |
| --- | --- | --- |
| `cellprofiler_boilerplate_elimination.md` | Mostly complete | Generated boilerplate removal target is implemented. Invocation-aware artifact declarations, product-owned runtime wrapping, sidecar persistence, and generated-source tests exist. Remaining work is parity hardening and deeper compatibility-path deletion, not the main boilerplate elimination. |
| `cellprofiler_gui_source_bindings.md` | First implementation complete | The typed model, preview, inline PyQt editor, structured selector dialogs, VFS inventory, enum cells, and generic dataclass-widget integration exist. Remaining work is source-assignment model cleanup and UI polish, not a missing core feature. |
| `cellprofiler_like_debug_mode.md` | Core substrate mostly complete | Debug sessions/cursors/snapshots/stores, invocation-level stepping, persistent paused worker control, warm replay, artifact refs, CP renderer families, inspector actions, and dirty cursor support exist. Remaining work is richer module-specific views and heavier end-to-end GUI workflow coverage. |
| `debug_mode_gui_worker_integration.md` | Mostly complete | Toolbar, typed commands, bounded and persistent-paused worker routing, snapshot read RPC, artifact export RPC, debug progress bridge, and inspector wiring exist. Remaining work is UX hardening plus ZMQ server decomposition. |
| `debug_source_binding_followthrough_20260516.md` | Mostly complete | The explicitly listed source-binding dialogs, artifact replay identity, inspector typed requests, paused-worker command loop, and focused tests now exist. Remaining work is polish and broader live GUI workflow testing. |
| `architectural_debt_refactor_sequence_20260515.md` | Partially complete | Sequences 3 and 4 are clean at the advisor baseline. Sequence 5 is functionally far along. Sequences 1 and 2 still have old GUI widget decomposition debt. |
| `cellprofiler_runtime_boundary_decomposition_20260516.md` | Complete for current advisor baseline | `pipeline_generator.py`, `module_execution.py`, and related planner files now scan clean in the current spot-check. Deeper compatibility deletion remains tracked under architectural debt, not this decomposition pass. |
| `cellprofiler_runtime_deep_refactor_remaining_findings.md` | Stale, superseded for runtime equivalence | The old `55 findings` checkpoint is obsolete. Runtime equivalence is now at `0` findings. Keep only as historical rationale unless rewritten. |
| `runtime_artifact_semantics_consolidation_20260516.md` | Complete for current planner/artifact baseline | `path_planner.py` is advisor-clean and invocation-artifact planning is now provider-owned. Remaining artifact work belongs to replay identity and compatibility deletion. |
| `registry_key_declaration_refactor_20260516.md` | Implemented for immediate boilerplate issue | `RegistryFamily` plus `RegistryKeyAttribute.STRATEGY_KEY` removes the repeated key boilerplate in the touched runtime-equivalence families. Broader metaclass-registry ergonomics can continue, but this no longer blocks the current branch. |
| `advisor_guided_adjacent_refactor_overview_20260516.md` | Needs refresh | It accurately framed the refactor sequence, but should be regenerated from the current findings because several listed risky boundaries are now clean. |

## Remaining Work Queue

### Queue 1: Source Assignment Model Cleanup

Evidence:

- `openhcs/core/source_bindings.py`: `4` advisor findings.
- `openhcs/core/pipeline_image_schema.py`: `3` advisor findings.
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`: `0` advisor findings.

Work:

1. Decide whether `SourceRole` metadata leaves in `pipeline_image_schema.py` should be generated leaf declarations or remain explicit registered leaves.
2. Consolidate repeated validation loops in source-binding dataclasses only if the new object owns typed validation semantics.
3. Replace property aliases such as `SourceFilterMatchType.requires_value` and `SourceArtifactAssignment.artifact_kind` only if callers can consume the nominal field directly without compatibility churn.

Stop condition:

- Do not add generic private validators that only hide repeated loops. The extraction must name the source-binding validation domain.

### Queue 2: Old GUI Widget Decomposition

Evidence:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`: `9` advisor findings.
- `openhcs/pyqt_gui/widgets/plate_manager.py`: `5` advisor findings.

Work:

1. Split `PipelineEditorWidget` by stable responsibilities: step list model, code execution adapter, debug command bridge, step preview formatting, and time-travel handling.
2. Split `PlateManagerWidget` by stable responsibilities: plate deletion workflow, debug run submission, code execution adapter, progress wiring, and time-travel handling.
3. Remove or route dangling private methods through real service objects only after tests pin existing behavior.

Stop condition:

- Do not move methods into one-off helper classes unless those helpers own a reusable request/result model or service boundary.

### Queue 3: ZMQ Execution Server Decomposition

Evidence:

- `openhcs/runtime/zmq_execution_server.py`: `10` advisor findings.
- Main signals: attribute probes, `_execute_with_orchestrator` orchestration hub, repeated progress-builder calls, threaded parameter family across signature/debug replay/execution helpers.

Work:

1. Extract a typed progress emission request/factory for orchestrator progress states.
2. Extract request signature and replay signature builders into one nominal execution-signature service.
3. Extract debug control handlers from the server into a command router that owns snapshot read, artifact export, and worker command dispatch.
4. Split `_execute_with_orchestrator` into orchestration phases only after request/signature/progress objects exist.

Stop condition:

- Do not start by splitting `_execute_with_orchestrator` mechanically. First name the typed records that remove repeated parameters and progress payload construction.

### Queue 4: Debug UX End-To-End Coverage

Evidence:

- Unit/control-channel coverage exists, including live ZMQ debug worker command-loop status and paused-worker controller tests.
- Remaining plan language asks for heavier GUI workflow tests around pause, step, continue, stop, inspect, export, and worker lifetime.

Work:

1. Add a live GUI/ZMQ workflow test that executes the full command sequence against a real server process or in-process server thread.
2. Verify snapshot inspection and export through the same `PipelineEditorWidget`/`PlateManagerWidget` signal path used by the GUI.
3. Keep this test isolated from official30 and mark it appropriately if it requires a display or slow integration resources.

Stop condition:

- Do not duplicate the current control-channel unit tests. The missing coverage is the host GUI command path across multiple commands.

### Queue 5: CP Runtime Compatibility Deletion And Parity Recheck

Evidence:

- Generated boilerplate removal is complete, but older compatibility paths remain deliberately retained.
- Plans require official30 parity before deleting deeper CP runtime compatibility.

Work:

1. Inventory remaining compatibility exports/import shims in `openhcs/interop/cellprofiler/runtime`.
2. Delete one compatibility path at a time only after a generated-pipeline test proves it is unused.
3. Re-run targeted generated-pipeline integration tests and the relevant official30 cached parity slice after each deletion batch.

Stop condition:

- Do not remove compatibility paths based only on advisor cleanliness. These paths are migration semantics, not just code smell.

### Queue 6: Plan Docs Refresh

Work:

1. Replace stale status blocks in `cellprofiler_runtime_deep_refactor_remaining_findings.md`, especially the `55 findings` runtime-equivalence note.
2. Add a short "current as of 2026-05-17" block to each older plan that points to this audit.
3. Keep historical rationale, but separate it from active work queues so future agents do not chase already-completed findings.

## Recommended Execution Order

1. Refresh stale plan docs so the work queue is not misleading.
2. Source assignment model cleanup: small, high leverage, close to the source-binding plan.
3. ZMQ server decomposition: high risk but now bounded by typed debug/control seams.
4. GUI widget decomposition: broad but mostly older debt; do after ZMQ so debug command ownership is clearer.
5. Debug end-to-end workflow tests: best after the ZMQ/router boundaries stabilize.
6. CP compatibility deletion and official30 parity: do last because it is behavior-sensitive and benchmark-expensive.

