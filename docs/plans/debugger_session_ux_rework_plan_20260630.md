# Debugger Session UX Rework Plan

Date: 2026-06-30

## Status

Concrete implementation spec plus follow-up SSOT corrections. The first UX
slice was implemented in the current worktree on 2026-06-30, but this document
now also records required cleanup discovered during the 2026-07-01 debugger
runtime projection review: command policy rows, phase enum display dictionaries,
debug-view enum property dictionaries, and table projection mapping dictionaries
must be replaced by nominal declarations before the debugger work is considered
architecturally complete.

This file is the current focused plan for making the Pipeline Editor debugger
understandable and usable. It supersedes the stale implementation-order parts of
`docs/plans/debugger_session_ux_dynamic_menu_plan_20260630.md`, which still
describes moving toolbar action specs out of `DebugToolbarWidget`. That migration
has already happened in the current worktree through nominal debugger action
declarations.

The next implementation pass must start with
`docs/plans/debugger_runtime_projection_api_plan_20260701.md`. The UX panel,
toolbar labels, inspector state surface, and MCP rendering are consumers of the
unified runtime projection; they must not continue adding local phase,
progress, or debug-view rules while core progress/runtime declarations still own
mirrored tables.

2026-07-01 correction: this UX plan predates the stricter non-UI authority
boundary. Any references below that place phase semantics in
`openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py` must be
read as a UI adapter over core `openhcs/core/debug_session_projection.py`.
Debugger phase selection, runtime tree node semantics, and runtime/debug
projection state belong in `openhcs.core.*`; PyQt, MCP, and agent DTO code poll
or render those projections generically.

## User-Visible Problem

The debugger has the right pieces, but the experience is opaque:

- the active `DebugSession` is not presented as a coherent session;
- the current cursor, axis, source group, invocation, and terminal state are not
  visible near the pipeline list;
- the Pipeline Editor, Plate Manager, bottom debug toolbar, and Debug Inspector
  can tell different stories at the same time;
- `Debug`, `Step`, `Pause`, `Restart`, and `Inspect` do not make the command
  semantics obvious;
- `Inspect` is a bucket menu for unrelated commands;
- empty runtime/artifact tables look broken rather than intentionally empty;
- when a bounded debug command completes, the last cursor/snapshot disappears
  and stale status text can remain visible.

The target UX is a debugger session panel: the user immediately understands
whether the selected plate is ready, starting, paused at a boundary, running,
completed, failed, or stopped; what invocation is current or was last executed;
what can be inspected; and which command will advance execution.

## Non-Negotiable Architecture Boundaries

Do not fix the UX by adding another semantic mirror.

Allowed:

- derive actions from `PipelineDebugActionDeclarationBase.__registry__`;
- keep `DebugCommandType` as the closed command wire token;
- derive command behavior from a nominal core `DebugCommandDeclarationBase`
  keyed by `DebugCommandType`, plus `DebugInvocationExecutionStrategy` and
  `DebugStepStopStrategy`;
- derive UI action presentation from `PipelineDebugActionDeclarationBase`;
- derive session/cursor state from `DebugSession`, `DebugCursor`,
  `DebugProgressContext`, `DebugSnapshot`, and Plate Manager session ownership;
- derive inspector sections from `DebugViewModel`, `DebugViewSection`,
  `DebugViewTable`, `DebugViewSectionDeclarationBase`, and
  `DebugViewTableProjectionDeclarationBase`;
- reuse pyqt-reactive chrome primitives for layout only.

Not allowed:

- no action-id string lists;
- no per-step debug DTO fields that mirror `DebugSession.cursor`;
- no second debug-session registry;
- no MCP-specific runtime-values DTO;
- no title-string matching to decide debugger section semantics;
- no toolbar/bridge-specific command semantics;
- no serialized phase strings as local decision points; strings appear only
  as final bridge/MCP ABI values emitted from a nominal phase declaration;
- no enum property dictionaries for phase titles/details or debug view
  empty-state text;
- no projection-spec mapping dictionaries for debug view table rendering;
- no `getattr`/`hasattr` probing;
- no if/elif command dispatch where a nominal declaration or strategy owns the
  branch.

## Current Authorities

### Runtime and Debug Semantics

- `openhcs.core.debug.DebugCommandType`
- `openhcs.core.debug.DebugCommandDeclarationBase` (new; replaces
  `DebugCommandPolicyRow`)
- `openhcs.core.debug.DebugInvocationExecutionStrategy`
- `openhcs.core.debug.DebugStepStopStrategy`
- `openhcs.core.debug.DebugPausedWorkerController`
- `openhcs.core.debug.DebugSession`
- `openhcs.core.debug.DebugCursor`
- `openhcs.core.debug.DebugProgressContext`
- `openhcs.core.debug.DebugSnapshot`

These own command behavior, cursor identity, debug progress events, snapshot
identity, and bounded replay behavior.

Current gap: `DebugCommandPolicyRow` is a dataclass table hidden behind
`DebugCommandType.advances_one_boundary`. Replace it with a nominal
`DebugCommandDeclarationBase`:

```python
class DebugCommandDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Core semantic declaration for one DebugCommandType."""

    __registry_key__ = "command_type"
    __skip_if_no_key__ = True

    command_type: ClassVar[DebugCommandType | None] = None
    advances_one_boundary: ClassVar[bool] = False

    @classmethod
    def require_command_type(cls) -> DebugCommandType: ...

    @classmethod
    def for_command_type(
        cls,
        command_type: DebugCommandType,
    ) -> type["DebugCommandDeclarationBase"]: ...
```

`DebugCommandType` remains a string enum. Do not move behavior into tuple enum
payloads. `PipelineDebugCommandActionDeclaration` points at the same
`DebugCommandType` and may query the core command declaration when it needs core
command policy.

### Inspector Model

- `openhcs.core.debug_views.DebugViewModel`
- `openhcs.core.debug_views.DebugViewSection`
- `openhcs.core.debug_views.DebugViewTable`
- `openhcs.core.debug_views.DebugViewTableProjection`
- `openhcs.core.debug_views.DebugViewSectionDeclarationBase` (new)
- `openhcs.core.debug_views.DebugViewTableProjectionDeclarationBase` (new)
- `openhcs.core.debug.DebugArtifactRef`

Current gap: `DebugViewSection` only has `title`, `table`, and `text`.
Tabbed/empty-state rendering needs a nominal section kind. `DebugViewSectionKind`
and `DebugViewTableProjection` remain wire tokens only. Titles, empty messages,
row builders, and projection columns live on nominal declaration classes, not
enum property dictionaries or `DEBUG_VIEW_TABLE_PROJECTIONS` mappings.

### GUI Session Lifecycle

- `PlateManagerWidget._active_debug_sessions`
- `PlateManagerWidget.action_run_debug_plate(...)`
- `PlateManagerWidget.action_inspect_debug_runtime(...)`
- `PlateManagerWidget._clear_debug_session_for_plate(...)`
- `PipelineEditorWidget.debug_session_state`
- `PipelineEditorDebugWorkflow`

Current gap: `_clear_debug_session_for_plate()` clears
`PipelineEditorWidget.debug_session_state` immediately on terminal completion,
so the UI/bridge lose the last cursor and snapshot. Add a typed terminal summary
owned by the debug/session/progress layer before or while clearing the active
session.

### UI Action and State Projection

- `PipelineDebugActionDeclarationBase`
- `DebugToolbarActionProjector`
- `PipelineDebugSessionContext`
- `PipelineDebugSessionStateSurfaceProvider`
- `PipelineDebugToolbarActionProvider`

These are the right synchronization path. Qt and MCP continue consuming
the same projected action/session models.

Current gap: the projection exposes action enablement and a phase string, but the
Qt UI does not render a session header, and the state surface does not expose a
terminal debug summary after the active session is cleared. A live dry run after a
bounded `Step` produced `phase=ready` with `terminal=complete`; this must become a
single nominal terminal phase rather than two fields that can contradict each
other.

### pyqt-reactive Layout Reuse

Use these as chrome, not semantic owners:

- `ButtonPanel`: primary declared debug action buttons;
- `FormWindowActionHeader`: compact session title/action strip;
- `ActionTabbedWindowBody`: Debug Inspector tabs with active-tab actions;
- `DetachableActionBar`: not used in the first implementation slice; add it
  only after `ActionTabbedWindowBody` exposes a typed tab-action slot.

## Concrete Edit Specification

This section is the implementation contract. An implementation that cannot
follow one subsection must first replace that subsection with the exact alternate
authority and call sites.

### 1. Core Debug Session Summary

File: `openhcs/core/debug.py`

Add immediately after `DebugSession`:

```python
@dataclass(frozen=True, slots=True)
class DebugTerminalSummary:
    """Terminal UI/debug summary for one completed debug session."""

    debug_session_id: str
    plate_id: str
    terminal_status: str
    cursor: DebugCursor | None = None
    command_type: DebugCommandType | None = None
    axis_id: str | None = None
    snapshot_id: str | None = None
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    step_name: str | None = None
    callable_name: str | None = None
    completed_at_unix: float | None = None

    @classmethod
    def from_session(
        cls,
        session: DebugSession,
        *,
        terminal_status: str,
        completed_at_unix: float | None = None,
    ) -> "DebugTerminalSummary":
        ...

    def with_snapshot(
        self,
        *,
        snapshot: DebugSnapshot | None,
        snapshot_id: str | None,
        snapshot_store_ref: str | None,
        snapshot_store_backend: str | None,
    ) -> "DebugTerminalSummary":
        ...
```

Extend `DebugSession` with:

```python
command_type: DebugCommandType | None = None

def with_command(self, command_type: DebugCommandType) -> "DebugSession": ...
```

Update `DebugSession.create(...)`, `with_cursor(...)`, and
`mark_dirty_from_cursor(...)` so `command_type` is preserved.

Reason: the active session and terminal summary are different concepts. The UI
must not keep a completed run as a fake active `DebugSession`, and it must not
lose the last cursor when `_clear_debug_session_for_plate()` runs.

### 2. Shared Debug Projection Types

Core file: `openhcs/core/debug_session_projection.py`

UI adapter: `openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py`

Add:

```python
class PipelineDebugSessionPhase(str, Enum):
    """Closed wire token for debugger session phase."""

    NO_PLATE = "no_plate"
    NEEDS_INITIALIZATION = "needs_initialization"
    NEEDS_COMPILE = "needs_compile"
    PENDING_EXECUTION = "pending_execution"
    ACTIVE_SESSION = "active_session"
    TERMINAL_COMPLETE = "terminal_complete"
    TERMINAL_FAILED = "terminal_failed"
    TERMINAL_CANCELLED = "terminal_cancelled"
    READY = "ready"


class DebugSessionPhaseDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal core/session declaration for one debugger phase."""

    __registry_key__ = "phase"
    __skip_if_no_key__ = True

    phase: ClassVar[PipelineDebugSessionPhase | None] = None
    priority: ClassVar[int]
    title: ClassVar[str]
    detail: ClassVar[str]

    @classmethod
    def require_phase(cls) -> PipelineDebugSessionPhase: ...

    @classmethod
    def matches(cls, context: "DebugSessionProjectionContext") -> bool: ...

    @classmethod
    def for_phase(
        cls,
        phase: PipelineDebugSessionPhase,
    ) -> type["DebugSessionPhaseDeclarationBase"]: ...

    @classmethod
    def for_context(
        cls,
        context: "DebugSessionProjectionContext",
    ) -> type["DebugSessionPhaseDeclarationBase"]: ...


class NoPlateDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class NeedsInitializationDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class NeedsCompileDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class PendingExecutionDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class ActiveDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class TerminalCompleteDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class TerminalFailedDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class TerminalCancelledDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
class ReadyDebugSessionPhase(DebugSessionPhaseDeclarationBase): ...
```

Rules:

- `PipelineDebugSessionPhase` is a string enum only.
- Phase title/detail come from
  `DebugSessionPhaseDeclarationBase.for_phase(phase)`.
- Phase selection comes from registered phase declarations sorted by
  `priority`, not an if/elif chain embedded in the enum.
- Terminal phase declarations match `TerminalExecutionStatus` via
  `parse_terminal_status(...)`; do not normalize arbitrary status strings in the
  phase enum.
- `PipelineDebugSessionContext` adapts selected UI state into
  `DebugSessionProjectionContext`; it must not own the phase decision.

Change `PipelineDebugSessionContext` to:

```python
@dataclass(frozen=True, slots=True)
class PipelineDebugPauseBoundaryState:
    pause_step_indices: tuple[int, ...]

    @property
    def has_pause_boundaries(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PipelineDebugSessionContext:
    target: PipelineDebugTargetState | None
    session: DebugSession | None
    terminal_summary: DebugTerminalSummary | None
    pause_boundaries: PipelineDebugPauseBoundaryState
    manager_execution_state: ManagerExecutionState
```

Change `DebugActionRenderModel` to add:

```python
phase: PipelineDebugSessionPhase
```

Change `DebugToolbarActionProjector.phase(...)` to return
`DebugSessionPhaseDeclarationBase.for_context(context).phase`, not
`str`. Bridge rendering serializes `phase.value`; Qt receives the enum and uses
the declaration for title/detail.

The concrete phase declarations must cover this effective priority order:

1. no target;
2. target not initialized;
3. target not compiled;
4. pending execution without active session;
5. active session;
6. terminal summary, matched by typed `TerminalExecutionStatus`;
7. ready.

Terminal must not be reported as `READY`.

### 3. Declaration-Owned Labels and Availability

File: `openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py`

This layer owns GUI action presentation and availability. It does not own core
command semantics. Any behavior that is true for a `DebugCommandType` outside
the Pipeline Editor UI, such as advancing one boundary, must live on
`DebugCommandDeclarationBase` in `openhcs.core.debug`.

Add to `PipelineDebugActionDeclarationBase`:

```python
@classmethod
def label_for(cls, context: PipelineDebugSessionContext) -> str:
    return cls.label

@classmethod
def tooltip_for(cls, context: PipelineDebugSessionContext) -> str:
    return cls.tooltip

@classmethod
def availability_override(
    cls,
    context: PipelineDebugSessionContext,
) -> DebugActionDisabledReason | None:
    return None
```

To avoid an import cycle, put the context import under `TYPE_CHECKING` and make
the runtime annotations string annotations.

Implement overrides:

- `StartOrContinueDebugAction.label_for(...)`
  - `Start Debug` for `READY` and terminal phases;
  - `Continue` for `ACTIVE_SESSION`;
  - `Debug` only for unavailable/no-target states.
- `RunToPauseDebugAction.label = "Run to Pause"` and
  `availability_override(...)` returns a disabled reason with code
  `debug_pause_boundary_required` when the selected pipeline has no
  pause-boundary steps.
- `InspectRuntimeValuesAction.label = "Inspect Runtime"`.
- `StopDebugSessionAction.toolbar_placement` remains `SESSION`, but the Qt
  widget must render `SESSION` placement as visible buttons, not only as menu
  items.

Change `DebugToolbarActionProjector.render_model(...)` to call
`declaration.label_for(context)`, `declaration.tooltip_for(context)`, and
`declaration.availability_override(context)` before generic active-session
guards.

### 4. Pipeline Editor Context Construction

File: `openhcs/pyqt_gui/widgets/pipeline_editor.py`

Add state:

```python
self.debug_terminal_summary: DebugTerminalSummary | None = None
```

Update `debug_session_context()`:

```python
return PipelineDebugSessionContext(
    target=target,
    session=self.debug_session_state,
    terminal_summary=self.debug_terminal_summary,
    pause_boundaries=PipelineDebugPauseBoundaryState(
        pause_step_indices=tuple(
            index
            for index, step in enumerate(self.pipeline_steps)
            if step.debug_pause
        )
    ),
    manager_execution_state=manager_execution_state,
)
```

When a new debug command is submitted from `PipelineEditorDebugWorkflow.run_command`,
clear `self.editor.debug_terminal_summary` before dispatching the command.

### 5. Plate Manager Session Ownership

File: `openhcs/pyqt_gui/widgets/plate_manager.py`

Add public typed accessors, not direct external reads of `_active_debug_sessions`:

```python
def debug_session_for_plate(self, plate_path: str) -> DebugSession | None:
    return self._active_debug_sessions.get(plate_path)

def debug_terminal_summary_for_plate(
    self,
    plate_path: str,
) -> DebugTerminalSummary | None:
    editor = self._plate_pipeline_editor
    if editor is None or editor.current_plate != plate_path:
        return None
    return editor.debug_terminal_summary
```

Update `action_run_debug_plate(...)`:

- new session:
  ```python
  session = DebugSession.create(
      plate_id=target_plate_path,
      command_type=command_type,
  )
  ```
- existing session:
  ```python
  session = session.with_command(command_type)
  self._active_debug_sessions[target_plate_path] = session
  ```

Update `_clear_debug_session_for_plate(...)` before clearing the editor session:

```python
terminal_status = self.plate_terminal_activity_status.terminal_status(plate_path)
if terminal_status is not None and session is not None:
    editor.debug_terminal_summary = DebugTerminalSummary.from_session(
        session,
        terminal_status=terminal_status.value,
        completed_at_unix=time.time(),
    )
editor.debug_session_state = None
```

Do not store terminal summaries in a second Plate Manager dict. The active
session map remains the active-session authority; the Pipeline Editor holds the
selected plate's terminal UI summary.

### 6. Snapshot Events Update Active and Terminal State

File: `openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py`

In `show_snapshot(...)`, build the editor session from the Plate Manager session
when available:

```python
active_session = None
if self.editor.plate_manager is not None:
    active_session = self.editor.plate_manager.debug_session_for_plate(
        notification.progress_event.plate_id
    )
session = active_session or DebugSession(...)
self.editor.debug_session_state = session.with_cursor(debug_context.cursor)
self.editor.debug_terminal_summary = None
```

After loading `snapshot`, if `self.editor.debug_terminal_summary` is not `None`
and its `debug_session_id` matches `debug_context.debug_session_id`, replace it
with `.with_snapshot(...)`.

This lets the terminal summary retain the last snapshot identity while still
clearing active session state on terminal completion.

### 7. Bridge DTOs

File: `openhcs/agent/dto/ui_bridge.py`

Add:

```python
@dataclass(frozen=True, slots=True)
class UiDebugTerminalSummaryState:
    debug_session_id: str
    plate_scope_id: str
    terminal_status: str
    command_type: str | None
    axis_id: str | None
    snapshot_id: str | None
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None
    step_name: str | None
    callable_name: str | None
    cursor: UiDebugCursorState | None
    completed_at_unix: float | None
```

Extend `UiPipelineDebugSessionState`:

```python
terminal_summary: UiDebugTerminalSummaryState | None
```

Extend `UiPlateManagerRowState`:

```python
debug_phase: str | None = None
debug_session_id: str | None = None
```

These are serialized projections only. They are populated from active
`DebugSession`, terminal summary, and `PipelineDebugSessionPhase`; they do not own
debugger state.

File: `openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py`

Add `_terminal_summary_state(...)` next to `_cursor_state(...)`, and include the
terminal summary in `_revision_token(...)`.

`phase` must be:

```python
phase=DebugToolbarActionProjector.phase(context).value
```

File: `openhcs/pyqt_gui/services/plate_manager_state_projection.py`

In `project_row(...)`:

```python
debug_context = manager.pipeline_editor.debug_session_context()
debug_phase = DebugToolbarActionProjector.phase(debug_context)
debug_session = manager.debug_session_for_plate(plate_key)
```

Only apply those values when the projected row matches the Pipeline Editor's
current plate. `status_prefix` uses a debug-specific prefix while
`debug_phase` is active/pending/terminal:

- `Debug paused` for `ACTIVE_SESSION`;
- `Debug starting` for `PENDING_EXECUTION`;
- `Debug complete`, `Debug failed`, `Debug cancelled` for terminal phases.

Add `PlateStatusPresenter.build_debug_status_prefix(...)` and keep all debug
prefix strings there. The bridge and row projection pass typed phase/status
values into the presenter; they do not hardcode display strings.

### 8. Qt Debug Session Panel

File: `openhcs/pyqt_gui/widgets/debug_toolbar.py`

Modify the existing widget instead of adding a parallel action surface.

Fields to add:

```python
self.phase_label: QLabel
self.cursor_label: QLabel
self.primary_panel: ButtonPanel
self.session_panel: ButtonPanel
self.inspector_panel: ButtonPanel
```

Layout:

- top row: `phase_label` and `cursor_label`;
- second row: three `ButtonPanel`s for `PRIMARY`, `SESSION`, and `INSPECTOR`
  placements.

Remove the single `Inspect` menu button from the primary UX. Update tests to
assert action render models and visible `ButtonPanel` contents directly.

Rendering:

```python
phase = DebugToolbarActionProjector.phase(context)
phase_declaration = DebugSessionPhaseDeclarationBase.for_phase(phase)
self.phase_label.setText(phase_declaration.title)
self.cursor_label.setText(DebugSessionPanelText.from_context(context).detail)
```

`DebugSessionPanelText` is a frozen dataclass in
`debug_session_projection.py`:

```python
@dataclass(frozen=True, slots=True)
class DebugSessionPanelText:
    title: str
    detail: str

    @classmethod
    def from_context(cls, context: PipelineDebugSessionContext) -> "DebugSessionPanelText":
        ...
```

The widget renders labels only; it does not decide phase semantics.

File: `openhcs/pyqt_gui/main.py`

Stop removing the debugger toolbar from the Pipeline Editor. Delete this block:

```python
pipeline_layout = self.pipeline_editor_widget.layout()
if pipeline_layout is not None:
    pipeline_layout.removeWidget(debug_toolbar)
bottom_control_layout.addWidget(debug_toolbar)
```

Do not add debugger controls to the main-window bottom bar in this
implementation. The Pipeline Editor panel is the single debugger control
surface.

### 9. Debug View Model and Inspector

File: `openhcs/core/debug_views.py`

Add wire-token enums plus nominal declarations:

```python
class DebugViewSectionKind(str, Enum):
    """Closed wire token for debug view section kind."""

    SUMMARY = "summary"
    SOURCES = "sources"
    ARTIFACT_OVERVIEW = "artifact_overview"
    AVAILABLE_ARTIFACTS = "available_artifacts"
    INPUT_ARTIFACTS = "input_artifacts"
    OUTPUT_ARTIFACTS = "output_artifacts"
    PREVIEW_ARTIFACTS = "preview_artifacts"
    INVOCATION_PARAMETERS = "invocation_parameters"
    RUNTIME_VALUES = "runtime_values"
    MEASUREMENTS = "measurements"
    RELATIONSHIPS = "relationships"
    TIMING = "timing"
    ERROR = "error"


class DebugViewSectionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal display/empty-state declaration for one section kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[DebugViewSectionKind | None] = None
    default_title: ClassVar[str]
    empty_message: ClassVar[str]

    @classmethod
    def require_kind(cls) -> DebugViewSectionKind: ...

    @classmethod
    def for_kind(
        cls,
        kind: DebugViewSectionKind,
    ) -> type["DebugViewSectionDeclarationBase"]: ...


class DebugViewTableProjection(str, Enum):
    """Closed wire token for debug table projection kind."""

    ARTIFACT_REFS = "artifact_refs"
    INVOCATION_PARAMETERS = "invocation_parameters"
    RUNTIME_VALUE_RECORDS = "runtime_value_records"


class DebugViewTableProjectionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal table projection declaration."""

    __registry_key__ = "projection"
    __skip_if_no_key__ = True

    projection: ClassVar[DebugViewTableProjection | None] = None
    columns: ClassVar[tuple[str, ...]]
    empty_message: ClassVar[str]
    supports_artifact_actions: ClassVar[bool] = False

    @classmethod
    def require_projection(cls) -> DebugViewTableProjection: ...

    @classmethod
    def row_for(cls, value: object) -> tuple[str, ...]: ...

    @classmethod
    def table_for(cls, values: tuple[object, ...]) -> "DebugViewTable": ...

    @classmethod
    def for_projection(
        cls,
        projection: DebugViewTableProjection,
    ) -> type["DebugViewTableProjectionDeclarationBase"]: ...
```

Concrete declarations replace `DebugViewSectionKind.default_title`,
`DebugViewSectionKind.empty_message`, `DebugViewTableProjectionSpec`, and
`DEBUG_VIEW_TABLE_PROJECTIONS`. Do not keep enum property dictionaries or a
mapping proxy after the declaration migration.

Extend:

```python
@dataclass(frozen=True, slots=True)
class DebugViewTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    projection: DebugViewTableProjection | None = None
    empty_message: str | None = None

@dataclass(frozen=True, slots=True)
class DebugViewSection:
    kind: DebugViewSectionKind
    title: str
    table: DebugViewTable | None = None
    text: str | None = None

    @property
    def is_empty(self) -> bool: ...
```

Update `to_json_dict()` and `from_json_dict()` to include `kind` and
`empty_message`. Since debug snapshots are ephemeral developer artifacts, update
callers instead of adding old string-shape parsing.

`DebugViewModel.from_runtime_value_store(...)` must use:

```python
DebugViewSectionDeclarationBase.for_kind(DebugViewSectionKind.RUNTIME_VALUES)
DebugViewTableProjectionDeclarationBase.for_projection(
    DebugViewTableProjection.RUNTIME_VALUE_RECORDS
)
```

It must not call enum `.default_title` or a projection-spec dictionary.

File: `openhcs/interop/cellprofiler/debug_views.py`

Change `CellProfilerDebugSectionSpec`:

```python
kind: DebugViewSectionKind
title: str | None = None
```

`section_for(...)` passes `kind=...` and defaults title from
`DebugViewSectionDeclarationBase.for_kind(kind).default_title`.

Tag the existing specs:

- `Summary` -> `SUMMARY`
- `Sources` -> `SOURCES`
- `Artifact Overview` -> `ARTIFACT_OVERVIEW`
- available artifacts -> `AVAILABLE_ARTIFACTS`
- input/input images/inputs -> `INPUT_ARTIFACTS`
- output/output images/object outputs -> `OUTPUT_ARTIFACTS`
- previews -> `PREVIEW_ARTIFACTS`
- invocation parameters -> `INVOCATION_PARAMETERS`
- runtime value store -> `RUNTIME_VALUES`
- measurements/measurement tables -> `MEASUREMENTS`
- relationships -> `RELATIONSHIPS`
- timing -> `TIMING`
- exception -> `ERROR`

File: `openhcs/pyqt_gui/windows/debug_inspector_window.py`

Replace the single `QScrollArea` section list with `ActionTabbedWindowBody`.

`set_view_model(view_model)`:

1. clear the tab body;
2. iterate `view_model.sections` in order;
3. build one tab per section;
4. render empty table bodies using `section.table.empty_message` or
   `DebugViewSectionDeclarationBase.for_kind(section.kind).empty_message`;
5. add artifact open/export action widgets only when
   `DebugViewTableProjectionDeclarationBase.for_projection(
   section.table.projection
   ).supports_artifact_actions` is true.

Do not use section titles to choose tabs or actions.

### 10. Tests To Add or Rewrite

Focused tests:

- `tests/unit/pyqt_gui/test_debug_toolbar.py`
  - ready context renders `Start Debug`;
  - active session renders `Continue`;
  - `Run to Pause` disabled when `pause_boundaries` is empty;
  - stop is visible/enabled during active/pending context;
  - toolbar phase label comes from
    `DebugSessionPhaseDeclarationBase.for_phase(phase).title`.
- `tests/unit/pyqt_gui/test_ui_agent_bridge.py`
  - `pipeline_debug_toolbar.session` payload has terminal phase and
    `terminal_summary` after `_clear_debug_session_for_plate`;
  - action labels in bridge match `DebugToolbarActionProjector.render_models`.
- `tests/unit/pyqt_gui/test_debug_inspector_window.py`
  - empty runtime table renders the model-owned empty message;
  - tabs are created from `DebugViewSection.kind`;
  - artifact actions still emit `DebugArtifactOpenRequest` and
    `DebugArtifactMaterializeRequest`.
- `tests/unit/pyqt_gui/test_pipeline_editor_widget.py`
  - debug toolbar remains mounted in Pipeline Editor after main-window setup.
- `tests/unit/pyqt_gui/test_plate_manager_state_projection.py`
  - create this focused test file when it does not already exist;
  - active debug session projects a debug prefix and debug phase;
  - terminal summary projects terminal debug prefix without `Pending`.

Verification commands:

```bash
source .venv/bin/activate
ruff check \
  openhcs/core/debug.py \
  openhcs/core/debug_views.py \
  openhcs/agent/dto/ui_bridge.py \
  openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py \
  openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py \
  openhcs/pyqt_gui/widgets/debug_toolbar.py \
  openhcs/pyqt_gui/windows/debug_inspector_window.py \
  openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py \
  openhcs/pyqt_gui/services/plate_manager_state_projection.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/interop/cellprofiler/debug_views.py \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_debug_inspector_window.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py
pytest -q \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_debug_inspector_window.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_pipeline_editor_widget.py
```

## Implementation Order

### Slice 1: Projection and Model Contract

Edit exactly these production files:

- `openhcs/core/debug.py`
- `openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py`
- `openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py`
- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py`

Required result:

- `DebugSession` carries `command_type`.
- `DebugTerminalSummary` exists and is populated before active session clear.
- `PipelineDebugSessionContext` includes active session, terminal summary,
  pause-boundary state, target state, and manager execution state.
- `DebugSessionPhaseDeclarationBase.for_context(...)` is the only phase
  decision point.
- `PipelineDebugActionDeclarationBase` owns context-aware labels, tooltips, and
  declaration-specific disabled reasons.
- `pipeline_debug_toolbar.session` serializes enum `.value` and includes
  `terminal_summary`.

Stop condition for this slice:

```bash
source .venv/bin/activate
ruff check \
  openhcs/core/debug.py \
  openhcs/agent/dto/ui_bridge.py \
  openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py \
  openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/shared/services/pipeline_editor_workflows.py \
  openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py
pytest -q \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py
```

### Slice 2: Pipeline Editor Debug Panel

Edit exactly these production files:

- `openhcs/pyqt_gui/widgets/debug_toolbar.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/main.py`

Required result:

- The existing `DebugToolbarWidget` becomes the session panel.
- It renders `phase_label`, `cursor_label`, `primary_panel`, `session_panel`,
  and `inspector_panel`.
- `DebugToolbarWidget` consumes `DebugToolbarActionProjector.render_models(...)`
  and `DebugSessionPanelText.from_context(...)`; it owns no action ids and no
  phase strings.
- `Stop` and `Restart` render as visible session controls during active or
  pending phases.
- The Pipeline Editor keeps the debug panel mounted in its own layout.
- The main-window bottom control no longer steals the only debugger widget.

Stop condition for this slice:

```bash
source .venv/bin/activate
ruff check \
  openhcs/pyqt_gui/widgets/debug_toolbar.py \
  openhcs/pyqt_gui/widgets/pipeline_editor.py \
  openhcs/pyqt_gui/main.py
pytest -q \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_pipeline_editor_widget.py
```

### Slice 3: Debug Inspector Tabs and Empty States

Edit exactly these production files:

- `openhcs/core/debug_views.py`
- `openhcs/interop/cellprofiler/debug_views.py`
- `openhcs/pyqt_gui/windows/debug_inspector_window.py`

Required result:

- `DebugViewSectionDeclarationBase` is the section semantic authority.
- `DebugViewTable.projection` preserves the table projection identity.
- `DebugViewTable.empty_message` carries renderer-independent empty-state text.
- `CellProfilerDebugSectionSpec` stores `kind`, not title-only semantics.
- `DebugInspectorWindow` renders one `ActionTabbedWindowBody` tab per
  `DebugViewSection`.
- Artifact actions are attached only for
  `DebugViewTableProjectionDeclarationBase.supports_artifact_actions`.

Stop condition for this slice:

```bash
source .venv/bin/activate
ruff check \
  openhcs/core/debug_views.py \
  openhcs/interop/cellprofiler/debug_views.py \
  openhcs/pyqt_gui/windows/debug_inspector_window.py
pytest -q tests/unit/pyqt_gui/test_debug_inspector_window.py
```

### Slice 4: Plate Manager Debug Status Projection

Edit exactly these production files:

- `openhcs/pyqt_gui/services/plate_manager_state_projection.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/agent/dto/ui_bridge.py`

Required result:

- `UiPlateManagerRowState` exposes `debug_phase` and `debug_session_id`.
- `PlateStatusPresenter.build_debug_status_prefix(...)` owns debug display text.
- Plate rows derive debug state from the selected Pipeline Editor context and
  Plate Manager active sessions.
- The bridge stores no copied per-row debug registry.
- Terminal debug state wins over stale submission text.

Stop condition for this slice:

```bash
source .venv/bin/activate
ruff check \
  openhcs/pyqt_gui/services/plate_manager_state_projection.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/agent/dto/ui_bridge.py
pytest -q \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_progress_tree_aggregation.py
```

### Slice 5: Live MCP/UI Stress Pass

Run against a restarted UI/runtime:

```bash
source .venv/bin/activate
python -m openhcs.mcp.dev_client --timeout-seconds 2 ui-status
python -m openhcs.mcp.dev_client --timeout-seconds 3 state-surfaces
python -m openhcs.mcp.dev_client --timeout-seconds 3 state-surface pipeline_debug_toolbar.session
python -m openhcs.mcp.dev_client --timeout-seconds 3 actions pipeline_debug_toolbar
python -m openhcs.mcp.dev_client --timeout-seconds 3 selected-workflow init_plate --no-confirmation
python -m openhcs.mcp.dev_client --timeout-seconds 3 selected-workflow compile_plate --no-confirmation
python -m openhcs.mcp.dev_client --timeout-seconds 3 invoke-action pipeline_debug_toolbar step --no-confirmation
python -m openhcs.mcp.dev_client --timeout-seconds 3 state-surface pipeline_debug_toolbar.session
python -m openhcs.mcp.dev_client --timeout-seconds 3 state-surface plate_manager.state
python -m openhcs.mcp.dev_client --timeout-seconds 3 window-snapshot pipeline_editor
```

Required observations:

- Before compile, action disabled reasons mention initialization or compilation.
- After compile, `phase` is `ready` and the primary action label is
  `Start Debug`.
- While active, `phase` is `active_session`, `cursor` is present, and visible
  session controls include `Stop`.
- After bounded completion, `phase` is one terminal enum value and
  `terminal_summary` is present.
- Plate Manager state and Pipeline Editor state report the same debug phase.
- The screenshot shows a readable session panel in the Pipeline Editor.

## Implementation Decisions

1. Terminal summaries live on `PipelineEditorWidget.debug_terminal_summary`.
   They are cleared when the selected plate changes or a new debug command is
   submitted. They are not ObjectState.

2. The Pipeline Editor owns the primary debugger panel. The bottom status bar
   does not host debugger controls in this plan.

3. `Run to Pause` is disabled when
   `PipelineDebugPauseBoundaryState.has_pause_boundaries` is false. The disabled
   reason code is `debug_pause_boundary_required`.

4. Runtime-value inspection after terminal completion is disabled. Snapshot and
   artifact inspection remain available through `DebugTerminalSummary` snapshot
   refs.

5. Debug Inspector tabs are ordered by `DebugViewModel.sections`. Section kind
   controls tab label defaults and empty-state text. No tab renderer branches on
   title strings.

6. Plate Manager debug text is built only by
   `PlateStatusPresenter.build_debug_status_prefix(...)`.

7. Bridge DTO fields are ABI projections. They never become authorities for
   phase, cursor, action availability, section kind, or terminal status.

## Dry Run 1: User Workflow

Initial state:

- selected plate exists;
- no active debug session;
- Pipeline Editor has compiled target.

Expected state surface:

```text
phase=ready
session=null
terminal_summary=null
primary action label=Start Debug
```

After invoking `pipeline_debug_toolbar step`:

```text
phase=active_session or pending_execution
session.debug_session_id=<non-empty>
session.cursor=<present once first snapshot/progress arrives>
Stop action visible=true
Inspect Runtime visible=true
```

After the bounded step completes:

```text
phase=terminal_complete
session=null
terminal_summary.debug_session_id=<previous session id>
terminal_summary.cursor=<last cursor>
terminal_summary.command_type=step
stale text "Submitting debug step..." absent from the visible panel
```

Inspector state:

```text
tabs derive from DebugViewSection.kind
empty runtime table uses DebugViewTable.empty_message or section declaration empty_message
artifact open/export actions render only when the table projection declaration supports them
```

## Dry Run 2: Semantic Ownership Walk

State-aware action labels:

- Authority: `PipelineDebugActionDeclarationBase`.
- Projector: `DebugToolbarActionProjector.render_model(...)`.
- Consumers: `DebugToolbarWidget`, `PipelineDebugToolbarActionProvider`, MCP
  action renderer.
- Illegal implementation: a Qt dict from action id to display label.

Debugger phase:

- Authority: `DebugSessionPhaseDeclarationBase.for_context(...)`.
- Inputs: `PipelineDebugSessionContext`.
- Consumers: Qt panel, UI bridge session state, Plate Manager row projection.
- Illegal implementation: string comparisons against serialized phase values.

Terminal state:

- Authority: `DebugTerminalSummary`.
- Creation point: `PlateManagerWidget._clear_debug_session_for_plate(...)`.
- Snapshot enrichment: `PipelineEditorDebugWorkflow.show_snapshot(...)`.
- Illegal implementation: keeping completed runs in `_active_debug_sessions`.

Pause-boundary availability:

- Authority: `PipelineDebugPauseBoundaryState`.
- Producer: `PipelineEditorWidget.debug_session_context()`.
- Consumer: `RunToPauseDebugAction.availability_override(...)`.
- Illegal implementation: toolbar directly scanning pipeline steps.

Inspector section semantics:

- Authority: `DebugViewSectionDeclarationBase` and
  `DebugViewTableProjectionDeclarationBase`.
- Producer: debug view builders.
- Consumer: `DebugInspectorWindow`.
- Illegal implementation: matching `section.title` or branching directly on a
  projection enum value for behavior.

Plate row debug status:

- Authority: Plate Manager active sessions, Pipeline Editor context, terminal
  summary, and `PlateStatusPresenter`.
- Consumer: `PlateManagerStateProjectionService.project_row(...)`.
- Illegal implementation: copied row-level debug cache.

## Dry Run 3: Failure Mode Walk

Failure: Qt label says `Debug`, MCP says `Start Debug`.

- Cause: label logic exists outside `PipelineDebugActionDeclarationBase`.
- Fix: remove local label source and call declaration hook through projector.
- Test: compare Qt button text and bridge action labels for ready context.

Failure: state surface reports `phase=ready` with `terminal=complete`.

- Cause: terminal summary is separate from phase decision.
- Fix: `DebugSessionPhaseDeclarationBase.for_context(...)` maps terminal
  summary to a terminal enum before `READY`.
- Test: clear active session after terminal completion and assert terminal phase.

Failure: Inspector hides useful artifact actions in a generic empty tab.

- Cause: tab/action logic uses section titles or row presence.
- Fix: table projection declaration carries `supports_artifact_actions=True`.
- Test: artifact-ref table with zero or more rows still renders the correct tab
  action strip.

Failure: Plate Manager says `Pending` after debug run completed.

- Cause: generic execution prefix wins over terminal debug summary.
- Fix: row projection computes debug phase and asks
  `PlateStatusPresenter.build_debug_status_prefix(...)` first.
- Test: terminal debug summary row status contains terminal debug prefix and no
  pending prefix.

Failure: `Run to Pause` runs but no debug-pause boundary exists.

- Cause: action projection lacks pause-boundary state.
- Fix: `PipelineDebugPauseBoundaryState` is part of context and the action
  declaration returns `debug_pause_boundary_required`.
- Test: empty pause-boundary context disables `Run to Pause`.

## Acceptance Criteria

- `pipeline_debug_toolbar.session` alone explains the selected plate's debug
  phase, active session, terminal summary, cursor, and available actions.
- The Pipeline Editor visibly shows phase, cursor/last cursor, and next debug
  actions without opening the inspector.
- Qt and MCP action labels, enabled flags, and disabled reasons come from
  `DebugToolbarActionProjector`.
- The Debug Inspector uses tabs from `DebugViewSection.kind` and empty states
  from the debug view model.
- Plate Manager, Pipeline Editor, Debug Inspector, and MCP state surfaces agree
  on active, pending, terminal, and ready states.
- No new semantic mirror registries, title-string dispatch, action-id string
  lists, or per-row copied debug caches are introduced.
