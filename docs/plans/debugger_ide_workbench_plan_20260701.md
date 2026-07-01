# Debugger IDE Workbench Plan

Date: 2026-07-01

## Purpose

Make OpenHCS debug mode feel like a real IDE debugger instead of a toolbar plus
an ad hoc inspector window.

This plan builds on:

- `docs/plans/debugger_runtime_projection_api_plan_20260701.md`
- `docs/plans/debugger_session_ux_rework_plan_20260630.md`

The runtime/progress/debug projection work is the semantic foundation. This plan
is the UI and agent-facing workbench layer that composes those projections into
the familiar debugger concepts: controls, current frame, call stack/frame path,
breakpoints, timeline, variables/artifacts, and source navigation.

## Live Evidence From Current UI

Observed with the live MCP dev client against the running UI:

- `ui_live_overview.state` reports one selected compiled plate and
  `phase=terminal_cancelled`.
- `pipeline_debug_toolbar.session` reports actions and current/last frame
  summaries, but not the full debug timeline.
- `pipeline_editor.state` reports 18 pipeline steps and no selected debugger
  frame path.
- The visible top-level debugger window is `Debug Inspector`, which renders
  snapshot tabs and artifact actions but no stack, timeline, controls, or source
  navigation.
- The Pipeline Editor shows debug buttons and selected/dirty-looking list rows,
  but the row styling does not communicate debugger state explicitly.

Concrete UX failure: the user sees progress/status text and can press `Step`,
but cannot visually answer:

- Where am I stopped?
- What has already run?
- What will run next?
- What did this frame produce?
- Where are my stop points?
- Which runtime values/artifacts belong to the selected frame?

## Observability Is Not A Breakpoint

The current `debug_pause` field is a poor user-facing debugger concept if it is
treated as "the step where debug info exists." That is not how an IDE debugger
works.

Correct semantics:

- debug mode records and projects live boundary state for every executed
  invocation that passes through the debug event sink;
- `Step` advances one invocation and updates the current frame even if no step
  has a stop point;
- `Continue`/`Run` update timeline/progress while they execute, independent of
  stop points;
- `FunctionStep.debug_pause` is only a persistent step stop point used by
  `RUN_TO_PAUSE`;
- `DebugSession.breakpoints` is the future runtime cursor-level stop-point
  authority;
- no inspector, frame, timeline, or current-row rendering is gated by
  `FunctionStep.debug_pause`.

Implementation consequence: the UI should label this concept as a breakpoint or
stop point, not as "debug pause." A missing stop point may disable `Run to
Breakpoint`, but it must never hide current frame, timeline, artifacts, runtime
values, or step-forward feedback.

## Current Implementation Audit

This section is the post-audit contract. If it conflicts with earlier wording in
this file, this section wins.

### Runtime Recording

The runtime side already has the correct observability boundary:

- `openhcs/core/steps/function_runtime.py::FunctionGroupRuntimeScope.execute_chain`
  records `BEFORE_INVOCATION`, `EXCEPTION`, and `AFTER_INVOCATION` whenever the
  installed debug sink captures invocation events.
- `FunctionGroupRuntimeScope.execute_chain` asks
  `debug_sink.should_stop_after_invocation(...)` only after recording the
  boundary event. Recording is not gated by `FunctionStep.debug_pause`.
- `openhcs/core/orchestrator/worker_execution.py::execute_worker_axis` stops at
  step boundaries through `DebugExecutionPolicy.step_stop_strategy()`.
- `openhcs/core/debug.py::RunToPauseDebugStepStopStrategy` is the only current
  runtime consumer of the step stop-point indices.
- `ProgressDebugEventSink` and `LocalSnapshotProgressDebugEventSink` both emit
  progress events for recorded debug boundaries; the latter also writes
  snapshots.

Do not refactor runtime event capture in the first workbench slice. The missing
piece is projection and presentation of the already-recorded events.

### Existing Projection And Transport Gap

`openhcs/core/progress/debug_projection.py::DebugRuntimeProjection` already owns
`current_frame`, `last_frame`, `timeline`, and `records`. Its builder already
projects debug records from the progress stream.

The current UI bridge drops most of that projection:

- `openhcs/agent/dto/ui_bridge.py::UiPipelineDebugSessionState` exposes only
  `current_frame` and `last_frame`.
- `openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py::PipelineDebugSessionStateSurfaceProvider._state`
  serializes only those two frame fields from `DebugRuntimeProjection`.
- `openhcs/mcp/dev_client_renderers/ui_bridge.py::PipelineDebugSessionStateSurfaceRenderer`
  renders only phase, target/session summary, cursor, current frame, last frame,
  and actions.

The workbench implementation must bridge the whole core projection, especially
timeline and per-step invocation state. It must not build a second timeline from
raw progress events in Qt or MCP.

### Existing Partial Workbench Code

There is already partial per-invocation presentation in
`openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py`:

- `PipelineEditorFunctionPresentation.invocation_badges(...)` scans a function
  pattern and marks current/dirty invocations from the editor debug session.
- `PipelineEditorFunctionPresentation.badge_provider(...)` exposes badge text to
  the existing function renderer.

That code is not a new authority; it is a UI-side local scan that should be
replaced by `DebugWorkbenchProjection`. Do not duplicate its badge logic in a new
widget. Either delete it after migration or make it consume the core workbench
projection.

### Stop Command Mismatch

Normal debugger Stop is currently split:

- `PlateManagerWidget.action_run_debug_plate(...)` can send
  `DebugCommandType.STOP` to an active paused worker through the ZMQ debug worker
  command path.
- `DebugPausedWorkerController.apply_command(DebugCommandType.STOP)` cleanly
  marks the paused worker stopped.
- `PipelineEditorDebugWorkflow.stop_command()` currently bypasses that path and
  calls `PlateManagerWidget.action_stop_execution(force=True)`.

This is a concrete implementation bug. The workbench implementation must route
the visible debugger Stop action through the declared `DebugCommandType.STOP`
worker command path whenever an active debug session exists. Force kill remains
an explicit emergency Plate Manager action, not the default Pipeline Editor
debugger Stop.

### Breakpoints Status

`DebugSession.breakpoints` currently exists as session data but is not consumed by
`DebugInvocationExecutionStrategy`, `DebugStepStopStrategy`, or the paused-worker
control path. It is not a functioning runtime breakpoint system yet.

First slice: expose only `FunctionStep.debug_pause` as the persistent step stop
point for `RUN_TO_PAUSE`. Do not expose cursor-level breakpoint toggles as
working UI until the runtime strategy consumes `DebugSession.breakpoints`.

### Action And Phase Authorities

The current implementation already has the right nominal owners:

- core command behavior: `DebugCommandDeclarationBase`,
  `DebugInvocationExecutionStrategy`, `DebugStepStopStrategy`;
- UI action presentation and editor dispatch:
  `PipelineDebugActionDeclarationBase`;
- phase title/detail and phase selection:
  `DebugSessionPhaseDeclarationBase`;
- inspector table/section schema: `DebugViewModel`,
  `DebugViewSectionDeclarationBase`, and
  `DebugViewTableProjectionDeclarationBase`.

The workbench can add composition classes, but any new branch that decides
command behavior, phase meaning, table columns, or inspector section semantics
belongs on one of these existing authorities.

## Non-Negotiable Boundaries

Do not add another debugger model.

Allowed authorities:

- `openhcs.core.debug.DebugCommandType`
- `openhcs.core.debug.DebugCommandDeclarationBase`
- `openhcs.core.debug.DebugBoundaryEventDeclarationBase`
- `openhcs.core.debug.DebugCursor`
- `openhcs.core.debug.DebugSession`
- `openhcs.core.debug.DebugTerminalSummary`
- `openhcs.core.debug.DebugSnapshot`
- `openhcs.core.debug.DebugTimelineNodeState`
- `openhcs.core.debug_session_projection.DebugSessionProjectionContext`
- `openhcs.core.debug_session_projection.DebugSessionPhaseDeclarationBase`
- `openhcs.core.progress.debug_projection.DebugRuntimeProjection`
- `openhcs.core.progress.debug_projection.DebugRuntimeFrame`
- `openhcs.core.progress.debug_projection.DebugTimelineNode`
- `openhcs.core.progress.runtime_tree.RuntimeTreeProjection`
- `openhcs.core.progress.runtime_tree.RuntimeTreeNodeDeclarationBase`
- `openhcs.core.steps.abstract.AbstractStep.debug_pause`
- `openhcs.core.steps.function_step.FunctionStep`
- `openhcs.processing.func_pattern.normalize_function_pattern`
- `openhcs.core.callable_contract.CallableContract`
- `openhcs.core.debug_views.DebugViewModel`
- `openhcs.core.debug_views.DebugViewSectionDeclarationBase`
- `openhcs.core.debug_views.DebugViewTableProjectionDeclarationBase`
- `openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions.PipelineDebugActionDeclarationBase`
- `openhcs.pyqt_gui.services.pipeline_object_state_binding.PipelineObjectStateBinding`

Forbidden patterns:

- no UI-owned event/status semantics;
- no second breakpoint store;
- no MCP-specific debugger DTO hierarchy that has different semantics from Qt;
- no parsing `DebugCursor.invocation_key` except through cursor-owned helpers or
  function-pattern normalized keys;
- no hardcoded command lists outside `PipelineDebugActionDeclarationBase`;
- no table/list column strings declared in renderers when an existing record
  dataclass can own the projected fields;
- no widget-tree scraping to decide debugger state;
- no CellProfiler module names in generic debugger workbench code;
- no action dispatch `if` ladder where command/action declarations can own it.

## Target UX

The Pipeline Editor should enter a debugger workbench mode, not just show a
small debug button cluster.

```text
Pipeline Editor
+----------------------------------------------------------------------------+
| Debug: Paused after invocation                                             |
| Plate BBBC022 / Axis A01 / Source group default                            |
| [Continue] [Step] [Run to Breakpoint] [Restart] [Stop] [Source Group...]   |
+--------------------------------+-------------------------------------------+
| Pipeline                       | Frame / Timeline                          |
| * 1 CorrectIlluminationApply   | Frame path                                |
|     done default[0] correct... |   plate -> A01 -> step 1 -> default[0]    |
| o 2 IdentifyPrimaryObjects     |                                           |
|     pending identify...        | Timeline                                  |
| o 3 IdentifySecondaryObjects   |   done CorrectIlluminationApply/default[0]|
|                                |   now  IdentifyPrimaryObjects/default[0]  |
+--------------------------------+-------------------------------------------+
| Inspector                                                                  |
| [Summary] [Runtime Values] [Input Artifacts] [Output Artifacts] [Params]   |
| rows for the selected frame, with Open in Napari/Fiji actions where valid  |
+----------------------------------------------------------------------------+
```

The existing `Debug Inspector` can remain as a detachable/details window, but
the first-class view should be embedded in the Pipeline Editor.

## Core Projection

Add a pure composition projection. It must not own new semantics; it only joins
existing authorities into a shape that Qt and MCP can render.

File: `openhcs/core/debug_workbench.py`

Draft:

```python
@dataclass(frozen=True, slots=True)
class DebugWorkbenchSource:
    session_context: DebugSessionProjectionContext
    runtime_projection: DebugRuntimeProjection
    pipeline_steps: tuple[FunctionStep, ...]


@dataclass(frozen=True, slots=True)
class DebugPipelineInvocationNode:
    step_index: int
    step_scope_id: str | None
    group_key: str
    position: int
    function_name: str
    cursor: DebugCursor
    state: DebugTimelineNodeState
    is_current: bool
    is_dirty_replay_start: bool


@dataclass(frozen=True, slots=True)
class DebugPipelineStepNode:
    step_index: int
    step_scope_id: str | None
    step_name: str
    has_step_stop_point: bool
    state: DebugTimelineNodeState
    invocations: tuple[DebugPipelineInvocationNode, ...]


@dataclass(frozen=True, slots=True)
class DebugFramePathEntry:
    label: str
    cursor: DebugCursor | None
    frame: DebugRuntimeFrame | None


@dataclass(frozen=True, slots=True)
class DebugWorkbenchProjection:
    session_context: DebugSessionProjectionContext
    runtime_projection: DebugRuntimeProjection
    pipeline_nodes: tuple[DebugPipelineStepNode, ...]
    frame_path: tuple[DebugFramePathEntry, ...]
    timeline: tuple[DebugTimelineNode, ...]
```

Builder rules:

- derive invocations from `normalize_function_pattern(step.func)`;
- derive callable names through `CallableContract.from_callable`;
- derive current/dirty state from `DebugSession.cursor`,
  `DebugSession.dirty_from_cursor`, and `DebugTerminalSummary.cursor`;
- derive timeline state through `DebugRuntimeProjection.node_state_for_cursor`;
- derive persistent step stop-point state from `FunctionStep.debug_pause`;
- derive runtime cursor stop-point state from `DebugSession.breakpoints`;
- derive frame/timeline observability from `DebugRuntimeProjection.records`,
  never from stop-point state;
- do not re-read progress events in this builder; use `DebugRuntimeProjection`;
- do not decide phase; use `DebugSessionProjectionContext.phase`.

### Timeline State Display

`DebugTimelineNodeState` is currently a wire enum. If display severity, order,
or icon role is needed, add nominal declarations:

```python
class DebugTimelineNodeStateDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "state"
    __skip_if_no_key__ = True

    state: ClassVar[DebugTimelineNodeState | None] = None
    sort_order: ClassVar[int]
    display_label: ClassVar[str]
    display_role: ClassVar[str]
```

This is allowed because no existing authority owns display/order for
`DebugTimelineNodeState`. Do not put this table in Qt, MCP, or enum tuple
payloads.

## UI Integration

### Pipeline Editor Workbench

Files:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/widgets/debug_toolbar.py`
- `openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py`
- new `openhcs/pyqt_gui/widgets/debug_workbench.py`

Replace the current "three button panels plus list" debugger experience with a
workbench widget inside the Pipeline Editor:

- header: phase title/detail from `DebugSessionPhaseDeclarationBase`;
- controls: rendered from `PipelineDebugActionDeclarationBase`;
- pipeline list overlay: current frame, completed invocations, failed
  invocations, dirty replay start, and stop-point markers from
  `DebugWorkbenchProjection.pipeline_nodes`;
- frame path panel: selected/current frame hierarchy from
  `DebugWorkbenchProjection.frame_path`;
- timeline panel: `DebugWorkbenchProjection.timeline`;
- inspector tabs: existing `DebugViewModel` rendering, initially embedded and
  optionally detachable through the existing `DebugInspectorWindow`.

Implementation note: keep `DebugToolbarWidget` as a command surface or fold it
into `DebugWorkbenchWidget`, but do not let it own phase/action semantics.
Command behavior stays in `DebugCommandDeclarationBase`,
`DebugInvocationExecutionStrategy`, `DebugStepStopStrategy`, and
`PipelineDebugActionDeclarationBase`.

### Pipeline List Gutter

Use a small Qt delegate or item decoration, not text prefixes, for debugger
meaning:

- stop point from `FunctionStep.debug_pause`;
- current invocation from `DebugPipelineInvocationNode.is_current`;
- dirty replay start from `is_dirty_replay_start`;
- completed/failed/pending state from `DebugTimelineNodeState`;
- selected ObjectState row remains visually separate from current debugger
  frame.

The gutter shows stop points, current frames, dirty replay starts, and timeline
states as separate visual channels. A stop point is not the current frame.

The current text current-frame badge can remain as a fallback, but the semantic
source must be `DebugWorkbenchProjection`, not a local scan in
`PipelineEditorFunctionPresentation`.

### Inspector

Keep `DebugViewModel` as the inspector data authority.

Change the UX:

- selecting a timeline/frame loads that frame's `DebugViewModel`;
- runtime inspection uses the active paused worker only when no snapshot frame
  is selected;
- artifact actions remain declared by `DebugViewTableProjectionDeclarationBase`
  and `StreamingConfig.supported_config_keys()`;
- no new table schema for artifacts/runtime values.

## UI Bridge And MCP

Do not add a second debugger surface first.

Extend the existing `pipeline_debug_toolbar.session` surface to carry the
workbench projection because it is already the single debugger state surface
and is bound to `PipelineDebugToolbarWidgetIdentity`.

Files:

- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py`
- `openhcs/mcp/dev_client_renderers/ui_bridge.py`

Add payload fields:

```python
workbench: UiDebugWorkbenchState
```

Where `UiDebugWorkbenchState` is a DTO serialization of
`DebugWorkbenchProjection`, not a new semantic model.

Do not keep independently constructed top-level `current_frame` and `last_frame`
fields beside `workbench`. Preferred implementation: make `workbench` the primary
debug-frame payload and remove top-level frame fields from the new schema. If an
intermediate compatibility period is unavoidable, derive any legacy
`current_frame`/`last_frame` fields mechanically from the same `workbench`
instance in the same provider method, and mark them for deletion. Never maintain
two frame builders.

The state-surface revision token must include the workbench payload through a
single serialization/digest path. Do not extend the current hand-maintained tuple
with another long list of workbench fields.

Renderer output should be compact and debugger-shaped:

```text
Pipeline debug: paused after invocation
Frame: step=0 CorrectIlluminationApply invocation=default[0] correct_illumination_apply
Stop points: 2 steps
Timeline: 1 completed, 0 failed, 17 pending
Actions: Continue enabled, Step enabled, Run to Breakpoint disabled (...)
Next: select a frame or open Debug Inspector for artifacts/runtime values.
```

This makes agents immediately aware of state without asking for widget trees.

## Stop Points / Breakpoints

Do not invent breakpoint storage.

Current authorities:

- persistent step stop point: `FunctionStep.debug_pause`;
- runtime cursor stop points: `DebugSession.breakpoints`;
- run-to-pause command policy: `DebugStepStopStrategy` and
  `DebugExecutionConfig.pause_step_indices`.

Stop points only affect stopping. They must not affect whether the workbench
shows frame/timeline/inspection state.

First slice:

- expose `FunctionStep.debug_pause` as the visible gutter stop point;
- label it as `Breakpoint` or `Stop point` in the UI, not `debug_pause`;
- make `Run to Breakpoint` a label alias for `RUN_TO_PAUSE` if the UX chooses
  IDE vocabulary;
- keep command wire token `DebugCommandType.RUN_TO_PAUSE`;
- show disabled reason from `RunToPauseDebugAction.availability_override`.

Later slice:

- use `DebugSession.breakpoints` for invocation-level stop points;
- the UI toggles a cursor-level stop point only after a selected invocation has
  a stable `DebugCursor`;
- first wire `DebugSession.breakpoints` into a nominal invocation or step stop
  strategy before exposing the toggle as functional;
- no new global breakpoint registry.

## Real IDE Command Vocabulary

Current command semantics:

- `RUN`: start/continue;
- `STEP`: resumes from cursor and executes one compiled invocation window;
- `RUN_TO_PAUSE`: run until a `debug_pause` step;
- `RESTART`: restart from the dirty cursor or session start;
- `STOP`: stop execution;
- `CHOOSE_SOURCE_GROUP`: choose debug source group.

UX labels should reflect these semantics:

- `Start Debug` / `Continue`
- `Step Invocation` or `Step`
- `Run to Breakpoint` if using gutter stop points
- `Restart`
- `Stop`
- `Source Group`

Do not rename the command enum just for presentation. Presentation belongs to
`PipelineDebugActionDeclarationBase`.

Debugger Stop must use the debug command route, not Plate Manager force kill, when
an active debug session exists. The exact first-slice fix is:

- `StopDebugSessionAction.dispatch_editor(...)` remains declaration-owned.
- `PipelineEditorDebugWorkflow.stop_command()` sends `DebugCommandType.STOP`
  through `PlateManagerWidget.action_run_debug_plate(...)` or an equivalent
  declared active-session command method.
- `PlateManagerWidget.action_stop_execution(force=True)` is only used for the
  emergency force-kill path.

## Dry Runs

### Terminal Cancelled After One Step

Inputs observed live:

- phase: `terminal_cancelled`
- terminal summary cursor: step 0, invocation
  `default:0:correct_illumination_apply`
- current frame: none
- last frame: `CorrectIlluminationApply / correct_illumination_apply`

Expected workbench:

- header says `Debug Cancelled`;
- frame path uses terminal summary/last frame;
- pipeline row 0 is marked as last frame;
- timeline has the completed/terminal node(s);
- inspector shows the last snapshot if available;
- disabled runtime inspection says active session required.

### Paused Active Session

Inputs:

- `DebugSessionProjectionContext.active_session` is not `None`;
- `DebugRuntimeProjection.current_frame` is not `None`;
- paused worker can answer runtime inspection.

Expected workbench:

- header says `Debug Active`;
- `Continue`, `Step`, `Stop`, and `Inspect Runtime` are enabled as declared;
- pipeline gutter marks current invocation even when no stop point is present;
- inspector defaults to current snapshot if one exists, otherwise runtime-value
  inspection;
- timeline click can change selected frame without changing execution state.

### Step Without Breakpoints

Inputs:

- no `FunctionStep.debug_pause=True`;
- `STEP` selected repeatedly.

Expected workbench:

- each `STEP` advances one invocation or step according to the core debug
  command strategy;
- header, current frame, timeline, and inspector update after each step;
- `Run to Breakpoint` is disabled with the declaration-owned disabled reason;
- no workbench data is hidden because no stop point exists.

### Run To Breakpoint

Inputs:

- at least one `FunctionStep.debug_pause=True`;
- `RUN_TO_PAUSE` selected.

Expected workbench:

- gutter shows stop-point markers before execution;
- progress tree and timeline update while running;
- when stopped, current frame is the invocation/step that reached the stop
  point;
- disabled reason remains declaration-derived if no stop points exist.

### Exception

Inputs:

- `DebugBoundaryEventDeclarationBase.for_event_type(EXCEPTION)` marks failed
  boundary;
- snapshot exception text exists.

Expected workbench:

- timeline node is failed;
- pipeline row is failed;
- inspector opens Error tab first;
- MCP live overview flags an error item with source surface id.

## Implementation Phases

0. Stop command correction
   - route Pipeline Editor debugger Stop through `DebugCommandType.STOP`;
   - keep Plate Manager force kill as a separate emergency action;
   - add a focused test that `StopDebugSessionAction` does not call
     `action_stop_execution(force=True)` when an active debug session exists.

1. Core workbench projection
   - add `openhcs/core/debug_workbench.py`;
   - build from `DebugSessionProjectionContext`, `DebugRuntimeProjection`, and
     `FunctionStep` sequence;
   - consume `normalize_function_pattern` and `CallableContract` rather than
     `UiPipelineEditorStepState.function_names`;
   - migrate `PipelineEditorFunctionPresentation` badge logic to consume this
     projection or delete that local badge path;
   - add unit tests with synthetic steps and debug records.

2. State surface payload
   - replace the primary debug-frame payload in `UiPipelineDebugSessionState`
     with `workbench`;
   - project through `PipelineDebugSessionStateSurfaceProvider`;
   - avoid an independently maintained `current_frame`/`last_frame` payload;
   - derive the revision token from one workbench serialization/digest path;
   - update MCP renderer to summarize frame, timeline, and stop points.

3. Pipeline Editor workbench UI
   - add `DebugWorkbenchWidget`;
   - compose existing command projection, pipeline list state, timeline panel,
     and inspector tabs;
   - keep `DebugToolbarWidget` only as a command-rendering subcomponent or
     remove it after replacing all callers.

4. Frame/timeline navigation
   - selecting a timeline node selects the corresponding pipeline row and
     inspector frame;
   - selecting a pipeline invocation filters/highlights the matching timeline
     nodes;
   - no execution command is dispatched by navigation alone.

5. Stop-point gutter
   - expose `FunctionStep.debug_pause` as a gutter toggle;
   - keep ObjectState as the mutation path;
   - route all provenance/dirty effects through existing ObjectState machinery.

6. Live validation
   - restart UI and execution server;
   - load the advanced CellProfiler pipeline;
   - compile;
   - run `Step`, `Continue`, `Run to Breakpoint`, and `Stop`;
   - verify `ui_live_overview.state`, `pipeline_debug_toolbar.session`, and the
     screenshot agree.
   - verify `Step` updates frame/timeline state with zero stop points enabled.

## Tests

Add or update:

- `tests/unit/progress/test_debug_projection.py`
  - timeline/current/last frame remain correct.
- new `tests/unit/test_debug_workbench_projection.py`
  - pipeline nodes derive from `FunctionStep` and function patterns;
  - current frame and dirty replay markers derive from `DebugSession`;
  - stop points derive from `debug_pause`;
  - frame/timeline/inspector state exists without `debug_pause`;
  - no progress-event reread.
- `tests/unit/pyqt_gui/test_debug_toolbar.py`
  - actions still render from declarations after workbench refactor.
  - debugger Stop dispatches through `DebugCommandType.STOP`, not force kill,
    when an active debug session exists.
- `tests/unit/pyqt_gui/test_ui_agent_bridge.py`
  - `pipeline_debug_toolbar.session` includes compact workbench payload.
  - bridge state does not maintain separate frame builders beside workbench.
- optional screenshot/manual MCP smoke:
  - `python -m openhcs.mcp.dev_client state-surface pipeline_debug_toolbar.session`
  - `python -m openhcs.mcp.dev_client state-surface ui_live_overview.state`
  - `python -m openhcs.mcp.dev_client window-snapshot pipeline_editor`

## Smell Audit Checklist

Before implementation is accepted:

- all command controls come from `PipelineDebugActionDeclarationBase`;
- all command behavior comes from core debug command/strategy declarations;
- all phase text comes from `DebugSessionPhaseDeclarationBase`;
- all timeline state text/order comes from a nominal declaration if needed;
- pipeline invocation rows come from `normalize_function_pattern` and
  `CallableContract`;
- stop points are `FunctionStep.debug_pause` or `DebugSession.breakpoints`;
- no observability path checks stop-point state;
- runtime/artifact/parameter tables come from `DebugViewModel`;
- MCP renders the same state Qt renders;
- no new dict/list of command names, phase names, section names, or module names.
