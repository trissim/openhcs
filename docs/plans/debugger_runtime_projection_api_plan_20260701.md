# Debugger Runtime Projection API Plan

Date: 2026-07-01

## Purpose

Turn the current debug controls and inspector into a generic debugger by
projecting the existing progress/debug hook stream into a typed debugger model.

This plan is intentionally API-first. It names the nominal authorities,
class roots, inheritance structure, and bridge DTOs that implementation must use.
If implementation needs a manual list, dict, or string parser, stop and move the
missing fact onto one of the authorities below.

## Boundaries

Allowed authorities:

- `openhcs.core.progress.ProgressEvent`
- `openhcs.core.debug.DebugProgressContext`
- `openhcs.core.debug.DebugCursor`
- `openhcs.core.debug.DebugEventType`
- `openhcs.core.debug.DebugSession`
- `openhcs.core.debug.DebugTerminalSummary`
- `openhcs.core.debug.DebugSnapshot`
- `openhcs.core.progress.projection.ExecutionRuntimeProjection`
- `openhcs.core.progress.runtime_tree.RuntimeTreeNodeDeclarationBase`
- `openhcs.core.progress.debug_projection.DebugRuntimeProjection`
- `openhcs.core.debug_session_projection.DebugSessionPhaseDeclarationBase`
- `openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions.PipelineDebugActionDeclarationBase`
- `openhcs.pyqt_gui.widgets.shared.services.debug_session_projection.PipelineDebugSessionContext`
- `openhcs.pyqt_gui.services.pipeline_object_state_binding.PipelineObjectStateBinding`
- `openhcs.core.debug_views.DebugViewModel`
- `openhcs.agent.ui_bridge_identities.UiStateSurfaceIdentityDeclarationBase`

Forbidden implementation patterns:

- no second debug event bus;
- no UI-owned runtime/debug semantics;
- no UI-side cursor registry;
- no MCP-specific debugger model;
- no parsing `DebugCursor.invocation_key` to discover callable semantics;
- no widget-tree scraping to decide debugger state;
- no action string lists;
- no node-type to behavior dictionaries;
- no mirroring `PipelineDebugActionDeclarationBase.__registry__`;
- no direct CellProfiler module names in generic debugger projection;
- no large runtime values or arrays in `ProgressEvent.context`.

Progress events own "where is execution". The paused worker/debug snapshot APIs
own "what data exists at that location".

Layer rule:

- `openhcs.core.*` owns runtime/debug meaning, phase decisions, progress tree
  node semantics, debug boundary semantics, and projection state.
- `openhcs.pyqt_gui.*`, `openhcs.agent.*`, and `openhcs.mcp.*` are polling,
  rendering, invocation, and DTO layers. They may compose or serialize core
  projections, but they must not decide what a progress event, debug phase,
  tree node, or debug boundary means.
- If a UI service needs a new branch to decide runtime/debug behavior, stop and
  move that branch onto a core declaration or projection first.

## Architecture Review Tightening

Review against the current checkout found several places where the first draft
was still too eager to create a new declaration or DTO instead of coupling to an
existing owner. Implementation must apply these tightenings before writing code:

1. `DebugEventType` remains the closed wire token for debug boundary
   events. Move `reports_output_artifacts`, progress status, and timeline
   outcome onto nominal debug-boundary event declaration classes keyed by
   `DebugEventType`; do not hide the same table inside enum tuple payloads.
2. `ExecutionRuntimeProjection` and the `zmqruntime.progress` adapter path are
   the authority for plate/axis runtime aggregation. `DebugRuntimeProjection`
   attaches debug frames to that projection; it does not recalculate plate,
   axis, or terminal runtime state.
3. `PipelineDebugSessionContext`, `DebugToolbarActionProjector`, and the
   existing `UiPipelineDebugSessionState` surface are UI adapters, not semantic
   owners. Extract phase decisions and debugger session state into core
   `DebugSessionPhaseDeclarationBase` / `DebugSessionProjection` types, then
   let the existing UI context poll and serialize that core projection. UI
   action declarations may own labels, tooltips, and dispatch affordances, but
   they must query core command/session declarations for behavioral meaning.
4. `PipelineEditorBridgeProviderSet` is the current registration owner for
   pipeline-editor/debug-toolbar surfaces. Add the debugger surface there, or
   first refactor provider-set registration through `UiBridgeProviderSetABC`.
   Do not add a second top-level composition list.
5. Runtime/progress tree node declarations must live in core and own
   aggregation/display policy. Replacing `_NODE_AGGREGATION_POLICY_BY_TYPE` is
   insufficient if `ProgressTreeStatusProjector` still has node-type sets or
   branches. Do not create a UI-side declaration family and do not wrap
   `pyqt_reactive` aggregation policy classes as OpenHCS semantic authorities.
6. The debugger state-surface identity is owned by the existing debug toolbar
   widget declaration. Bind the new surface to
   `PipelineDebugToolbarWidgetIdentity`; do not add a
   `PipelineDebuggerWidgetIdentity` in this slice and do not bind debugger state
   to `PipelineEditorWidgetIdentity`.
7. Runtime value/table inspection is already represented by `DebugViewModel`,
   live-measurement payloads, and snapshot artifact refs. The debugger state
   surface can link to or summarize those authorities; it must not create a
   second runtime-value/table schema.

## Proper Progress/Runtime Unification

The accepted refactor scope is larger than a debugger add-on. The debugger
forces a proper unification of progress semantics, runtime aggregation,
runtime tree rendering, debug boundaries, and agent/UI state surfaces.

The target architecture is:

```text
ProgressEvent stream
  -> nominal progress phase/status/channel declarations
  -> ExecutionRuntimeProjection
  -> core RuntimeTreeProjection / core DebugRuntimeProjection / live-measurement notices
  -> Qt widgets and UI bridge state surfaces polling those projections
```

There must be one semantic layer for "what does this progress event mean?" and
all consumers must query it. Debugger code must not add a parallel answer.

### Core Progress Semantics

File: `openhcs/core/progress/types.py`

Keep these as wire tokens:

```python
class ProgressPhase(Enum): ...
class ProgressStatus(Enum): ...
class ProgressChannel(Enum): ...
class ProgressChannelRole(Enum): ...
```

Do not use tuple enum payloads to attach behavior to those tokens. Replace
`ProgressChannel.__new__`, `ProgressSemantics._PHASE_TO_CHANNEL`,
`_TERMINAL_PHASES`, `_TERMINAL_STATUSES`, `_FAILURE_STATUSES`,
`_FAILURE_PHASES`, and `_SUCCESS_TERMINAL_PHASES` with nominal declarations:

```python
class ProgressChannelDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one ProgressChannel."""

    __registry_key__ = "channel"
    __skip_if_no_key__ = True

    channel: ClassVar[ProgressChannel | None] = None
    role: ClassVar[ProgressChannelRole]

    @classmethod
    def require_channel(cls) -> ProgressChannel: ...

    @classmethod
    def for_channel(
        cls,
        channel: ProgressChannel,
    ) -> type["ProgressChannelDeclarationBase"]: ...


class ProgressPhaseDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one ProgressPhase."""

    __registry_key__ = "phase"
    __skip_if_no_key__ = True

    phase: ClassVar[ProgressPhase | None] = None
    channel: ClassVar[type[ProgressChannelDeclarationBase]]
    is_terminal: ClassVar[bool] = False
    is_failure: ClassVar[bool] = False
    is_success_terminal: ClassVar[bool] = False

    @classmethod
    def for_phase(
        cls,
        phase: ProgressPhase,
    ) -> type["ProgressPhaseDeclarationBase"]: ...


class ProgressStatusDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one ProgressStatus."""

    __registry_key__ = "status"
    __skip_if_no_key__ = True

    status: ClassVar[ProgressStatus | None] = None
    is_terminal: ClassVar[bool] = False
    is_failure: ClassVar[bool] = False

    @classmethod
    def for_status(
        cls,
        status: ProgressStatus,
    ) -> type["ProgressStatusDeclarationBase"]: ...
```

Wrapper functions remain as the public API, but they become registry queries:

```python
def phase_channel(phase: ProgressPhase) -> ProgressChannel:
    return ProgressPhaseDeclarationBase.for_phase(phase).channel.require_channel()


def is_terminal_event(event: ProgressEvent) -> bool:
    return (
        ProgressPhaseDeclarationBase.for_phase(event.phase).is_terminal
        or ProgressStatusDeclarationBase.for_status(event.status).is_terminal
    )


def is_failure_event(event: ProgressEvent) -> bool:
    return (
        ProgressPhaseDeclarationBase.for_phase(event.phase).is_failure
        or ProgressStatusDeclarationBase.for_status(event.status).is_failure
    )
```

Concrete phase/status/channel declaration classes are the exhaustive list. Tests
must assert declaration coverage equals the enum members. Do not keep a shadow
set, dict, or tuple payload "for convenience."

### Runtime State Semantics

File: `openhcs/core/progress/projection.py`

Keep `PlateRuntimeState` as a wire token:

```python
class PlateRuntimeState(str, Enum):
    IDLE = "idle"
    COMPILING = "compiling"
    COMPILED = "compiled"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"
```

Move terminal policy to declarations:

```python
class PlateRuntimeStateDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one PlateRuntimeState."""

    __registry_key__ = "state"
    __skip_if_no_key__ = True

    state: ClassVar[PlateRuntimeState | None] = None
    is_terminal: ClassVar[bool] = False

    @classmethod
    def for_state(
        cls,
        state: PlateRuntimeState,
    ) -> type["PlateRuntimeStateDeclarationBase"]: ...
```

`PlateRuntimeProjection.is_terminal` is a property that delegates to the
state declaration. Do not put `is_terminal` back onto the enum.

The `OpenHCSProjectionAdapter` remains the bridge to `zmqruntime.progress`, but
it must call the progress declarations for channel/failure/success semantics.
If it stays private, that privacy is only module encapsulation. It must not own
semantic tables.

### Unified Projection Build

`ProgressWorkflowService.rebuild_runtime_projection()` currently builds only
`ExecutionRuntimeProjection` and lets other services parse progress events
independently. Replace that with one projection rebuild path:

```python
@dataclass(frozen=True, slots=True)
class RuntimeProjectionBundle:
    execution: ExecutionRuntimeProjection
    debug: DebugRuntimeProjection
    # optional, bounded indexes only; heavy values stay behind their services
    live_measurements_available: tuple[ProgressIdentity, ...] = ()


class RuntimeProjectionBuilder:
    """Build all runtime projections from one tracked ProgressEvent snapshot."""

    def build(
        self,
        source: RuntimeProjectionSource,
    ) -> RuntimeProjectionBundle: ...
```

`RuntimeProjectionSource` carries the event snapshot plus typed context
providers:

```python
@dataclass(frozen=True, slots=True)
class RuntimeProjectionSource:
    events_by_execution: Mapping[str, Sequence[ProgressEvent]]
    debug_source: DebugProjectionSessionSource | None = None
    step_index: PipelineDebugStepIndex | None = None
```

The builder order is:

1. build `ExecutionRuntimeProjection` using the progress adapter;
2. build `DebugRuntimeProjection` from the same event snapshot and the already
   built execution projection;
3. collect lightweight live-measurement identities only when a UI state surface
   consumes those identities.

`ProgressWorkflowService` stores the bundle on the host:

```python
self._runtime_projection_bundle = self._runtime_projection_builder.build(source)
self._host.runtime_progress_projection = bundle.execution
self._host.debug_runtime_projection = bundle.debug
```

This replaces independent ad hoc parse paths for debugger UI state. It does not
move heavy snapshot/runtime-store reads into progress projection.

### Runtime Tree Projection

`progress_tree_builder.py` becomes a consumer of the same nominal progress
and runtime declarations. The refactor extracts a pure runtime tree
projection in core that is not coupled to server-browser Qt widgets:

```python
@dataclass(frozen=True, slots=True)
class RuntimeTreeProjection:
    roots: tuple[RuntimeTreeNode, ...]


class RuntimeTreeProjectionBuilder:
    def build(
        self,
        *,
        events_by_execution: Mapping[str, Sequence[ProgressEvent]],
        runtime_projection: ExecutionRuntimeProjection,
        topology: RuntimeExecutionTopology,
    ) -> RuntimeTreeProjection: ...
```

`ProgressTreeBuilder` then becomes a Qt rendering adapter over
`RuntimeTreeProjection`. It preserves the widget boundary but owns no runtime
semantics. Debugger timeline nodes reuse the same node declaration root where
the concepts overlap.

`RuntimeExecutionTopology` is a typed projection of worker assignments, known
wells, and step names. It replaces loose parallel dictionaries in the public
builder signature:

```python
@dataclass(frozen=True, slots=True)
class RuntimeExecutionTopology:
    worker_assignments: Mapping[tuple[str, str], Mapping[str, tuple[str, ...]]]
    known_wells: Mapping[tuple[str, str], tuple[str, ...]]
    step_names: Mapping[tuple[str, str, str], Mapping[int, str]]
```

The topology object is a transport shape, not a semantic owner. Node semantics
still come from declarations.

Core location:

- `openhcs/core/progress/runtime_tree.py`

PyQt location:

- `openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py`
  becomes a Qt adapter that converts `RuntimeTreeProjection` to `TreeNode`
  records. It must not retain aggregation, status, or display policy maps.

Do not import `pyqt_reactive` from core. Aggregation behavior belongs on core
runtime tree node declarations through core mixins such as mean aggregation and
explicit-percent aggregation. `pyqt_reactive` may render the resulting tree; it
does not own OpenHCS runtime semantics.

### What This Refactor Must Not Do

- Do not change the worker progress wire format unless a test proves a required
  fact is absent.
- Do not add another debug progress bus.
- Do not make CellProfiler a dependency of core progress/debug projection.
- Do not move snapshot payloads or runtime stores into progress context.
- Do not preserve old dicts/sets as "compatibility shims"; this worktree is
  already in a breaking internal refactor.

### Current Checkout Smell Inventory

This inventory was verified with an AST pass against the current worktree. These
are not optional cleanups; they are the concrete semantic mirrors that make the
debugger feel ad hoc.

Files and owners to change:

- `openhcs/core/progress/types.py`
  - `ProgressChannel.__new__` currently stores `ProgressChannelRole` as tuple
    enum payload.
  - `ProgressSemantics._PHASE_TO_CHANNEL`,
    `ProgressSemantics._TERMINAL_PHASES`,
    `ProgressSemantics._TERMINAL_STATUSES`,
    `_FAILURE_STATUSES`, `_FAILURE_PHASES`, and
    `_SUCCESS_TERMINAL_PHASES` are mirrored semantic tables.
  - Public helpers are allowed to remain, but only as declaration queries.
- `openhcs/core/progress/projection.py`
  - `PlateRuntimeState.__new__` currently stores terminal policy as tuple enum
    payload.
  - `_OpenHCSProjectionAdapter` is allowed to remain as the generic
    `zmqruntime.progress` adapter, but it must call declaration-owned helpers.
- `openhcs/core/debug.py`
  - `DebugEventType.reports_output_artifacts` is enum-owned behavior.
  - `DebugCommandPolicyRow` is a command semantic table.
  - `DebugProgressStatus._STATUS_BY_EVENT_TYPE` is a debug-event to progress
    status mirror.
  - `DebugInvocationExecutionStrategy`, `DebugStepStopStrategy`, and
    `DebugExecutionPolicy` are already correct nominal owners; keep using those
    patterns.
- `openhcs/core/debug_views.py`
  - `DebugViewSectionKind.default_title` and `.empty_message` are enum property
    dictionaries.
  - `DebugViewTableProjectionSpec` plus `DEBUG_VIEW_TABLE_PROJECTIONS` is a
    projection-behavior mapping.
- `openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py`
  - `ProgressNodeType` is acceptable only as the final Qt node-type token at
    the widget boundary; aggregation/display policy must move to core
    `RuntimeTreeNodeDeclarationBase`.
  - `_NODE_AGGREGATION_POLICY_BY_TYPE` and `_TREE_AGGREGATION_REGISTRY` are
    UI-layer policy registries parallel to core declarations.
  - `ProgressTreeStatusProjector.apply_node_percent_text(...)` branches on node
    kinds instead of asking node declarations.
- `openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py`
  - `PipelineDebugSessionPhase.for_context(...)`,
    `.from_terminal_status(...)`, `.is_terminal`, `.title`, and `.detail`
    encode phase semantics in a UI service. Move this authority to core
    `openhcs/core/debug_session_projection.py`; the UI module becomes an
    adapter over the core projection.
- `openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py`
  - `PipelineDebugActionDeclarationBase` is the correct GUI action declaration
    family.
  - `DebugToolbarAuxiliaryAction` is only acceptable as a final action-id token
    if all action behavior stays on action declaration classes.

### Implementation Sequence

Do this as one unification, but land it in dependency order. Each slice must
delete the old mirror before moving to the next slice.

1. Core progress declarations.
   - Add `ProgressChannelDeclarationBase`, `ProgressPhaseDeclarationBase`, and
     `ProgressStatusDeclarationBase` in `openhcs/core/progress/types.py`.
   - Change `ProgressChannel` to plain string enum values.
   - Reimplement `phase_channel`, `is_terminal_event`, `is_execution_phase`,
     `is_failure_event`, and `is_success_terminal_event` as declaration queries.
   - Delete `ProgressSemantics`, `_PROGRESS_SEMANTICS`, and all phase/status
     sets.
   - Add tests that every `ProgressPhase`, `ProgressStatus`, and
     `ProgressChannel` has exactly one declaration.

2. Runtime state declarations.
   - Add `PlateRuntimeStateDeclarationBase` in
     `openhcs/core/progress/projection.py`.
   - Change `PlateRuntimeState` to plain string enum values.
   - Add `PlateRuntimeProjection.is_terminal` delegating to the state
     declaration.
   - Keep `_OpenHCSProjectionAdapter`, but verify it only calls public
     progress helpers and does not own semantic maps.
   - Update status presenters/tests that read `PlateRuntimeState.is_terminal`.

3. Debug command and boundary declarations.
   - Add `DebugCommandDeclarationBase` in `openhcs/core/debug.py`.
   - Delete `DebugCommandPolicyRow`.
   - Change `DebugCommandType.advances_one_boundary` call sites to use the
     declaration. Prefer deleting the property entirely if no public API needs
     it; do not leave a forwarding property that makes the enum look
     authoritative.
   - Add `DebugBoundaryEventDeclarationBase` and concrete boundary
     declarations.
   - Delete `DebugEventType.reports_output_artifacts` and
     `DebugProgressStatus._STATUS_BY_EVENT_TYPE`.
   - Update `DebugProgressEventRequest.to_progress_event()` and snapshot/output
     artifact decision points to query boundary declarations.

4. Debug-view declarations.
   - Add `DebugViewSectionDeclarationBase` and
     `DebugViewTableProjectionDeclarationBase` in
     `openhcs/core/debug_views.py`.
   - Delete `DebugViewSectionKind.default_title`,
     `DebugViewSectionKind.empty_message`, `DebugViewTableProjectionSpec`, and
     `DEBUG_VIEW_TABLE_PROJECTIONS`.
   - Update `DebugViewTable.from_projection(...)`,
     `DebugViewModel.from_runtime_value_store(...)`, and CellProfiler debug
     section specs to query declarations.
   - Keep CellProfiler-specific section ordering/selection in
     `openhcs/interop/cellprofiler/debug_views.py`, but make each section spec
     carry a `DebugViewSectionKind` and ask the core declaration for default
     title/empty text.

5. Debug session phase declarations.
   - Add `DebugSessionPhaseDeclarationBase` in
     `openhcs/core/debug_session_projection.py`.
   - Move `PipelineDebugSessionPhase` wire values to core as
     `DebugSessionPhase` or alias the UI name from the core module during
     migration.
   - Delete enum-owned `for_context`, `from_terminal_status`, `is_terminal`,
     `title`, and `detail` from the UI module.
   - Convert `DebugToolbarActionProjector.phase(...)`,
     `DebugSessionPanelText`, `PlateStatusPresenter`, and state-surface
     renderers to query core phase declarations.
   - Terminal matching must use typed `TerminalExecutionStatus` parsing, not
     string normalization inside the enum.

6. Shared runtime projection bundle.
   - Add `RuntimeProjectionSource`, `RuntimeProjectionBundle`, and
     `RuntimeProjectionBuilder` in the core progress projection layer
     (`openhcs/core/progress/projection.py` or a sibling imported by that
     module).
   - Make `ProgressWorkflowService.rebuild_runtime_projection()` call the
     builder once and store both `runtime_progress_projection` and
     `debug_runtime_projection`.
   - The builder owns the event snapshot for the current coalesced update. No
     UI service separately rescans progress events for debugger state
     after this slice.

7. Runtime tree projection.
   - Add core `RuntimeTreeNodeDeclarationBase` in
     `openhcs/core/progress/runtime_tree.py`.
   - Do not add `RuntimeTreeAggregationPolicyDeclarationBase`; aggregation is a
     behavior on the node declaration, factored through core mixins where useful.
   - Replace `_NODE_AGGREGATION_POLICY_BY_TYPE`,
     `_TREE_AGGREGATION_REGISTRY`, and hardcoded node-kind percent branches with
     node declaration methods.
   - Extract `RuntimeExecutionTopology` and `RuntimeTreeProjectionBuilder`.
   - Leave `ProgressTreeBuilder` as a Qt adapter over `RuntimeTreeProjection`
     with no semantic policy maps.

8. Debug runtime projection and UI surface.
   - Add `openhcs/core/progress/debug_projection.py` only after the core
     progress/debug declarations exist.
   - Build `DebugRuntimeProjection` from the same `RuntimeProjectionSource` used
     for `ExecutionRuntimeProjection`.
   - Add/repair the debugger state surface through the existing state-surface
     provider path. It reads the projection; it does not parse events or widget
     text.

### Required Deletion Checks

The implementation is incomplete if any of these remain outside tests or
historical docs:

```bash
rg -n "ProgressSemantics|_PHASE_TO_CHANNEL|_TERMINAL_PHASES|_TERMINAL_STATUSES|_FAILURE_STATUSES|_FAILURE_PHASES|_SUCCESS_TERMINAL_PHASES" openhcs tests
rg -n "DebugCommandPolicyRow|_STATUS_BY_EVENT_TYPE|reports_output_artifacts" openhcs tests
rg -n "DEBUG_VIEW_TABLE_PROJECTIONS|DebugViewTableProjectionSpec|default_title|empty_message" openhcs tests
rg -n "_NODE_AGGREGATION_POLICY_BY_TYPE|_TREE_AGGREGATION_REGISTRY|ProgressNodeType\\.|RuntimeTreeAggregationPolicyDeclarationBase" openhcs tests
rg -n "PipelineDebugSessionPhase\\.(for_context|from_terminal_status|title|detail|is_terminal)" openhcs tests
rg -n "PipelineDebugSessionPhase|DebugSessionPhaseDeclarationBase|RuntimeTreeNodeDeclarationBase" openhcs/pyqt_gui openhcs/mcp openhcs/agent
```

Expected for the final command:

- UI/MCP/agent may import and serialize core `DebugSessionPhase` values and
  core runtime tree node projections.
- UI/MCP/agent must not define phase declarations, tree node declarations,
  phase-selection methods, node aggregation policies, or node-kind behavior.

Use an AST check in addition to `rg` before committing:

```bash
source .venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

targets = [
    Path("openhcs/core/progress/types.py"),
    Path("openhcs/core/progress/projection.py"),
    Path("openhcs/core/progress/runtime_tree.py"),
    Path("openhcs/core/progress/debug_projection.py"),
    Path("openhcs/core/debug.py"),
    Path("openhcs/core/debug_session_projection.py"),
    Path("openhcs/core/debug_views.py"),
    Path("openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py"),
    Path("openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py"),
]

def contains_enum_reference(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
        for child in ast.walk(node)
    )


def semantic_container(node: ast.AST) -> bool:
    if isinstance(node, (ast.Dict, ast.Set)):
        return contains_enum_reference(node)
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "MappingProxyType"
            and any(semantic_container(arg) for arg in node.args)
        )
    return False


for path in targets:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            enum_like = any(
                getattr(base, "id", None) == "Enum"
                or getattr(base, "attr", None) == "Enum"
                for base in node.bases
            )
            if enum_like:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        raise SystemExit(f"{path}:{item.lineno}: tuple enum payload")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.startswith("_")
                    and semantic_container(node.value)
                ):
                    raise SystemExit(
                        f"{path}:{node.lineno}: private enum-keyed semantic container {target.id}"
                    )
print("semantic mirror check passed")
PY
```

The AST check can be refined during implementation, but it must stay targeted at
semantic ownership, not cosmetic naming.

### Test Gates

Run focused gates after each slice:

```bash
source .venv/bin/activate
pytest tests/unit/progress/test_projection.py tests/unit/test_zmq_progress.py
pytest tests/unit/test_debug_runtime.py
pytest tests/unit/pyqt_gui/test_debug_toolbar.py tests/unit/pyqt_gui/test_plate_status_presenter.py
pytest tests/unit/pyqt_gui/test_progress_tree_aggregation.py
pytest tests/unit/pyqt_gui/test_ui_agent_bridge.py -k "debug or state_surface or live_overview"
```

Before declaring the debugger work ready for UI testing, also run the full
CellProfiler parity benchmark path that has been used for the official 30
pipelines. The debugger must not regress normal execution or pycodify
round-tripping.

## Carrier And Inheritance Rule

Before adding new debugger projection classes, factor repeated semantic fields
onto nominal carriers or compose the existing authority object.

Core carrier cleanup:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class DebugSessionIdentityCarrier(ABC):
    """Nominal carrier for a debug session id."""

    debug_session_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DebugSnapshotStoreRefCarrier(ABC):
    """Nominal carrier for snapshot-store location identity."""

    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
```

Existing classes that already carry these concepts must inherit them instead
of independently redeclaring the same fields:

```python
class DebugSessionRequest(DebugSessionIdentityCarrier, ABC, metaclass=AutoRegisterMeta):
    ...

class DebugProgressContext(
    DebugSessionIdentityCarrier,
    DebugSnapshotStoreRefCarrier,
):
    ...

class DebugSession(
    DebugSessionIdentityCarrier,
    DebugSnapshotStoreRefCarrier,
):
    ...

class DebugTerminalSummary(
    DebugSessionIdentityCarrier,
    DebugSnapshotStoreRefCarrier,
):
    ...

class DebugExecutionConfig(
    DebugSessionIdentityCarrier,
    DebugSnapshotStoreRefCarrier,
):
    ...
```

Use keyword-only carrier bases for constructor-order compatibility. Do not fall
back to duplicated fields.

Progress identity rule:

- `ProgressIdentity` owns `execution_id`, `plate_id`, `axis_id`, and
  `step_name`.
- `ProgressExecutionContext` owns `execution_id` and `plate_id` for execution
  requests.
- New debug projection records carry `ProgressEvent` or
  `ProgressIdentity`, not redeclare those fields.

Cursor rule:

- `DebugCursor` owns `step_index`, `step_scope_id`, `group_key`,
  `invocation_key`, and `pattern_group_identity`.
- New core models carry `DebugCursor`.
- Agent DTOs reuse `UiDebugCursorState`; they do not re-declare cursor fields.

## Existing Gap

Current debug runtime information already flows through the normal progress
path:

```text
worker invocation
  -> DebugEvent
  -> DebugProgressEventRequest
  -> ProgressEvent(context=DebugProgressContext)
  -> ZMQ progress stream
  -> ProgressWorkflowService / DebugProgressNotificationService
```

The UI currently renders this as percent progress plus a detached inspector.
There is no typed projection that says:

- current frame;
- current step row;
- current invocation inside the step;
- previous/completed/debug-failed invocations;
- terminal last frame;
- available debugger actions from the same context.

## Verified Dynamic Runtime Information

Yes, the debugger must expose runtime information that was not knowable at
compile time. The implementation must layer live runtime observations over the
compiled/object-state model instead of treating compile-time plans as the whole
truth.

Verified dynamic channels in the current codebase:

### 1. Lightweight Execution Progress

Authority:

- `openhcs.core.progress.types.ProgressEvent`
- `openhcs.core.progress.projection.build_execution_runtime_projection(...)`
- `openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service.ProgressWorkflowService.on_progress(...)`

Runtime facts available from `ProgressEvent`:

- `identity: ProgressIdentity` for execution/plate/axis/step label;
- `phase`, `status`, `percent`, `completed`, `total`;
- `timestamp`, `pid`, worker ownership metadata;
- `error` and `traceback` for runtime failures;
- `context` for typed extension payloads.

Concrete instruction:

- `DebugRuntimeProjection` must always be built from the same tracked
  `ProgressEvent` stream used by `ExecutionRuntimeProjection`.
- Runtime failures not caught by compilation are surfaced from
  `ProgressEvent.error`, `ProgressEvent.traceback`, and failure statuses/phases.
- Do not infer runtime completion/failure from compiled plans or ObjectState.

### 2. Lightweight Debug Boundary Context

Authority:

- `openhcs.core.debug.DebugProgressEventRequest.to_progress_event()`
- `openhcs.core.debug.DebugProgressContext`
- `openhcs.pyqt_gui.widgets.shared.services.debug_progress_service.DebugProgressNotificationService`

Runtime facts available from debug progress context:

- current `DebugCursor`;
- `DebugEventType` (`BEFORE_INVOCATION`, `AFTER_INVOCATION`, `EXCEPTION`);
- `debug_session_id`;
- optional `snapshot_id`;
- optional debug snapshot store reference.

Concrete instruction:

- `DebugProgressRecord.from_progress_event(event)` is the only parse entrypoint
  for debug progress context.
- `DebugRuntimeProjection.current_frame` and timeline state come from
  `DebugProgressContext`, not from the pipeline declaration.
- Runtime invocation exceptions are projected from `DebugEventType.EXCEPTION`
  and the surrounding `ProgressEvent.error` / `traceback` fields when present.
- Do not put full arrays, tables, or runtime stores into `ProgressEvent.context`.

### 3. Bounded Live Measurement Previews

Authority:

- `openhcs.core.progress.live_measurements.LiveMeasurementProgressPayload`
- `openhcs.core.progress.live_measurements.LiveMeasurementTablePreview`
- `openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service.LiveMeasurementProgressNotificationService`
- `openhcs.pyqt_gui.windows.live_measurements_window.LiveMeasurementTableModel`

Runtime facts available from live measurement progress:

- measurement artifact address (`name`, `kind`, `axis_id`, storage path/backend,
  group/source coordinates);
- bounded columns and preview rows;
- `row_count`;
- truncation flags for rows, columns, and preview count;
- object/source image names when present.

Concrete instruction:

- The debugger state can report that live measurement previews exist by
  retaining `LiveMeasurementProgressPayload` records keyed by the same
  `ProgressEvent` identity.
- The full debugger timeline must not copy preview rows. The state surface
  exposes only a bounded summary or a link/identity to the existing
  live-measurement model.
- Malformed live measurement context must remain fail-loud through
  `LiveMeasurementPayloadError`; do not swallow it as "no data".

### 4. Paused-Worker Runtime Store Inspection

Authority:

- `openhcs.core.debug.DebugPausedWorkerController.runtime_inspection_view()`
- `openhcs.core.debug.DebugRuntimeInspectionRequest`
- `openhcs.core.debug.DebugRuntimeInspectionResponse`
- `openhcs.runtime.zmq_debug_control.DebugRuntimeInspectionMessageStrategy`
- `openhcs.runtime.zmq_execution_client.ZMQExecutionClient.get_debug_runtime_inspection(...)`

Runtime facts available only while paused:

- current `RuntimeValueStore` contents rendered as a `DebugViewModel`;
- runtime artifact rows not necessarily knowable at compile time;
- value types, artifact kinds, axis/group context, backend/path.

Concrete instruction:

- The debugger projection may expose an action/state saying runtime inspection is
  available, but the runtime-value store itself stays behind
  `DebugRuntimeInspectionRequest`.
- `DebugRuntimeInspectionResponse.view_model` is the transport for inspected
  runtime data.
- Calling runtime inspection while the worker is not paused must keep raising
  the existing runtime error; do not fake an empty view.

### 5. Snapshot Metadata And Artifact Refs

Authority:

- `openhcs.core.debug.DebugSnapshot`
- `openhcs.core.debug.DebugSnapshotStore`
- `openhcs.core.debug.DebugSnapshotReadRequest`
- `openhcs.runtime.zmq_debug_control.DebugSnapshotReadMessageStrategy`

Runtime facts available through snapshots:

- input/output/preview/measurement/relationship artifact refs;
- runtime invocation parameters;
- timing;
- exception text and traceback-backed failure context;
- source paths.

Concrete instruction:

- `DebugRuntimeProjection` stores only `snapshot_id` and optional loaded
  `DebugSnapshot` on `DebugProgressRecord`.
- Large payloads stay behind artifact refs, viewer streaming, materialization,
  or runtime inspection.
- Do not duplicate snapshot artifact tables in MCP DTOs.

## Dynamic Versus Compile-Time Decision Rule

Use compile-time/object-state data only for stable structure:

- pipeline step order;
- step scope identity;
- declared function pattern shape;
- action declarations;
- configured pause markers.

Use runtime observations for live state:

- current frame/cursor;
- started/completed/failed invocation state;
- actual timing;
- runtime errors and tracebacks;
- emitted measurement previews;
- runtime value store contents;
- produced artifact refs and snapshot IDs.

If a field can change while the worker runs, it must come from
`ProgressEvent`, `DebugProgressContext`, `LiveMeasurementProgressPayload`,
`DebugSnapshot`, or `DebugRuntimeInspectionResponse`, not from ObjectState or
compiled plan declarations.

Verification anchors already present in tests:

- `tests/unit/test_debug_runtime.py::test_paused_worker_runtime_inspection_projects_runtime_value_store`
- `tests/unit/test_debug_runtime.py::test_runtime_inspection_control_payload_round_trips_view_model`
- `tests/unit/test_debug_runtime.py::test_debug_event_builds_metadata_snapshot`
- `tests/unit/test_debug_runtime.py::test_local_snapshot_progress_sink_writes_snapshot_and_emits_id`
- `tests/unit/progress/test_live_measurements.py::test_live_measurement_payload_round_trips_through_progress_context`
- `tests/unit/progress/test_live_measurements.py::test_live_measurement_payload_malformed_context_fails_loudly`
- `tests/unit/pyqt_gui/test_batch_workflow_compile_engine.py::test_on_progress_notifies_debug_snapshot_listeners`
- `tests/unit/pyqt_gui/test_batch_workflow_compile_engine.py::test_on_progress_notifies_live_measurement_listeners`

## New Core Projection Module

Add:

`openhcs/core/progress/debug_projection.py`

This module must not import PyQt, MCP, CellProfiler, or agent DTOs.

### Debug Progress Record

```python
@dataclass(frozen=True, slots=True)
class DebugProgressRecord:
    """One debug-aware progress event with parsed typed context."""

    event: ProgressEvent
    context: DebugProgressContext
    snapshot: DebugSnapshot | None = None

    @classmethod
    def from_progress_event(
        cls,
        event: ProgressEvent,
    ) -> "DebugProgressRecord | None": ...

    @property
    def session_id(self) -> str: ...

    @property
    def cursor(self) -> DebugCursor: ...

    @property
    def progress_identity(self) -> ProgressIdentity: ...

    @property
    def boundary(self) -> DebugBoundaryState | None: ...
```

Authority:

- `ProgressEvent` owns execution/plate/axis/timestamp/progress;
- `DebugProgressContext` owns session/cursor/snapshot/event type.

Rules:

- This replaces duplicated context parsing in UI services.
- This carries `ProgressEvent` and `DebugProgressContext`; it must not copy
  `execution_id`, `plate_id`, `axis_id`, `step_name`, or cursor fields into new
  stored fields.
- If a snapshot has been read, `boundary` returns the `DebugSnapshot`; otherwise
  it returns `None`. The record must not invent a partial boundary object.
- It may return `None` only when a normal non-debug progress event has no debug
  context.
- It must not catch arbitrary exceptions silently. Malformed debug context
  exposes a typed parse error in tests or logs.

### Shared Runtime Tree Node Declarations

Do not add debugger node declarations parallel to the existing progress-tree
node enum. First extract the progress tree node semantics into a shared
non-UI nominal declaration root in:

`openhcs/core/progress/runtime_tree.py`

```python
@dataclass(frozen=True, slots=True)
class RuntimeTreeNode:
    declaration: type["RuntimeTreeNodeDeclarationBase"]
    node_id: str
    label: str
    status: str
    info: str
    percent: float
    execution_id: str | None = None
    children: tuple["RuntimeTreeNode", ...] = ()


class RuntimeTreeNodeDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for runtime/progress tree node kinds."""

    __registry_key__ = "node_kind"
    __skip_if_no_key__ = True

    node_kind: ClassVar[str | None] = None
    sort_order: ClassVar[int] = 0

    @classmethod
    def require_node_kind(cls) -> str: ...

    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float: ...

    @classmethod
    def info_for(
        cls,
        *,
        node: RuntimeTreeNode,
    ) -> str: ...

    @classmethod
    def to_abi_node_type(cls) -> str: ...


class MeanPercentRuntimeTreeNode:
    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float: ...


class ExplicitPercentRuntimeTreeNode:
    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float: ...
```

Replace the existing `ProgressNodeType` enum plus
`_NODE_AGGREGATION_POLICY_BY_TYPE` and `_TREE_AGGREGATION_REGISTRY` with
declarations that own the current progress tree node semantics:

```python
class PlateProgressTreeNode(MeanPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase): ...
class WorkerProgressTreeNode(MeanPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase): ...
class WellProgressTreeNode(ExplicitPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase): ...
class StepProgressTreeNode(ExplicitPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase): ...
class CompilationProgressTreeNode(ExplicitPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase): ...
```

The aggregation mixins are core OpenHCS runtime tree semantics. They are not
wrappers around `pyqt_reactive` policies and they do not import PyQt. The final
`node_type` string is emitted only at the Qt/agent DTO boundary through
`declaration.to_abi_node_type()`.

Then add debugger timeline nodes as a subclass family of the same root:

```python
class DebugTimelineNodeDeclarationBase(RuntimeTreeNodeDeclarationBase):
    """Semantic declaration for debugger timeline node kinds."""
```

Concrete debug declarations:

```python
class DebugExecutionTimelineNode(DebugTimelineNodeDeclarationBase):
    node_kind = "execution"
    sort_order = 0

class DebugAxisTimelineNode(DebugTimelineNodeDeclarationBase):
    node_kind = "axis"
    sort_order = 10

class DebugStepTimelineNode(DebugTimelineNodeDeclarationBase):
    node_kind = "step"
    sort_order = 20

class DebugInvocationTimelineNode(DebugTimelineNodeDeclarationBase):
    node_kind = "invocation"
    sort_order = 30
```

No parallel `{"step": ...}` map is allowed. Builders iterate
`RuntimeTreeNodeDeclarationBase.__registry__.values()` for all runtime tree
nodes, or `DebugTimelineNodeDeclarationBase.__registry__.values()` when they
need only debugger nodes.

This extraction must also remove the existing display-policy mirrors in
`ProgressTreeStatusProjector.apply_node_percent_text(...)`. Percent text is a
node declaration behavior, not a hardcoded set of node-type strings.

`ProgressNode.aggregation_policy_id` may remain a string only as the final
pyqt-reactive ABI field, derived from
`node_declaration.aggregation_policy.require_policy_id()`. Local decisions must
use the policy declaration class.

### Debug Timeline DTOs

```python
@dataclass(frozen=True, slots=True)
class DebugTimelineNodeIdentity:
    progress_identity: ProgressIdentity
    cursor: DebugCursor | None = None


class DebugTimelineNodeState(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class DebugTimelineNode:
    declaration: type[DebugTimelineNodeDeclarationBase]
    identity: DebugTimelineNodeIdentity
    state: DebugTimelineNodeState
    label: str
    detail: str | None
    percent: float
    timestamp: float
    snapshot_id: str | None = None
    children: tuple["DebugTimelineNode", ...] = ()
```

For execution and axis nodes, `cursor` may be `None`. For step and invocation
nodes, use the exact `DebugCursor` carried by the progress context. Do not make a
new identity record with copied cursor fields.

The only string emitted from a declaration is final ABI/display text. Local
decision points use the declaration class, `DebugCursor`, and `DebugEventType`.

### Debug Boundary Event Declarations

Do not add a `DebugEventType -> ProgressStatus` or
`DebugEventType -> DebugTimelineNodeState` dictionary. Also do not encode those
fields as tuple payloads on `DebugEventType`. `DebugEventType` remains a closed
wire enum. The semantic owner is a nominal declaration family in
`openhcs.core.debug`:

```python
class DebugBoundaryOutcome(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class DebugEventType(Enum):
    BEFORE_INVOCATION = "before_invocation"
    AFTER_INVOCATION = "after_invocation"
    EXCEPTION = "exception"


class DebugBoundaryEventDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one DebugEventType boundary."""

    __registry_key__ = "event_type"
    __skip_if_no_key__ = True

    event_type: ClassVar[DebugEventType | None] = None
    progress_status: ClassVar[ProgressStatus]
    boundary_outcome: ClassVar[DebugBoundaryOutcome]
    reports_output_artifacts: ClassVar[bool]

    @classmethod
    def require_event_type(cls) -> DebugEventType: ...

    @classmethod
    def for_event_type(
        cls,
        event_type: DebugEventType,
    ) -> type["DebugBoundaryEventDeclarationBase"]: ...


class BeforeInvocationDebugBoundary(DebugBoundaryEventDeclarationBase):
    event_type = DebugEventType.BEFORE_INVOCATION
    progress_status = ProgressStatus.STARTED
    boundary_outcome = DebugBoundaryOutcome.STARTED
    reports_output_artifacts = False


class AfterInvocationDebugBoundary(DebugBoundaryEventDeclarationBase):
    event_type = DebugEventType.AFTER_INVOCATION
    progress_status = ProgressStatus.SUCCESS
    boundary_outcome = DebugBoundaryOutcome.COMPLETED
    reports_output_artifacts = True


class ExceptionDebugBoundary(DebugBoundaryEventDeclarationBase):
    event_type = DebugEventType.EXCEPTION
    progress_status = ProgressStatus.ERROR
    boundary_outcome = DebugBoundaryOutcome.FAILED
    reports_output_artifacts = True
```

`DebugProgressEventRequest.to_progress_event()` uses
`DebugBoundaryEventDeclarationBase.for_event_type(
self.debug_event.event_type
).progress_status` and delete
`DebugProgressStatus._STATUS_BY_EVENT_TYPE`.

Keep `DebugTimelineNodeState` in `openhcs.core.progress.debug_projection` for
this slice, and derive started/completed/failed state from
`DebugBoundaryEventDeclarationBase.for_event_type(event_type).boundary_outcome`.
Do not import the projection module into `openhcs.core.debug.py`; the dependency
direction remains projection importing core debug.

Any existing use of `DebugEventType.reports_output_artifacts` moves to the
same declaration lookup. Do not keep a forwarding property on `DebugEventType`;
that would make the enum look like the owner again.

### Debug Frame

```python
@dataclass(frozen=True, slots=True)
class DebugRuntimeFrame:
    """Debugger program-counter frame derived from one debug progress record."""

    record: DebugProgressRecord

    @property
    def cursor(self) -> DebugCursor: ...

    @property
    def event_type(self) -> DebugEventType: ...

    @property
    def snapshot_id(self) -> str | None: ...

    @property
    def progress_identity(self) -> ProgressIdentity: ...

    @property
    def boundary(self) -> DebugBoundaryState | None: ...

    @property
    def step_name(self) -> str: ...

    @property
    def callable_name(self) -> str | None: ...
```

Authority:

- `DebugProgressContext.cursor` owns cursor identity.
- `ProgressEvent.identity` owns execution/plate/axis/step-name identity.
- `DebugSnapshot` owns rich boundary refs and callable name when loaded.
- `step_name` is derived from `boundary.step_name` when a boundary exists,
  otherwise from `ProgressEvent.identity.step_name`.
- `callable_name` is derived only from `boundary.callable_name`; do not parse it
  from `invocation_key`.

### Debug Runtime Projection

```python
@dataclass(frozen=True, slots=True)
class DebugRuntimeProjection:
    """Generic debugger state for one execution/session/plate view."""

    runtime_projection: ExecutionRuntimeProjection
    session: DebugSession | None
    terminal_summary: DebugTerminalSummary | None
    current_frame: DebugRuntimeFrame | None
    last_frame: DebugRuntimeFrame | None
    timeline: tuple[DebugTimelineNode, ...]
    records: tuple[DebugProgressRecord, ...]

    @property
    def has_active_frame(self) -> bool: ...

    @property
    def debug_session_id(self) -> str | None: ...

    @property
    def current_progress_identity(self) -> ProgressIdentity | None: ...

    def node_state_for_cursor(
        self,
        *,
        cursor: DebugCursor,
    ) -> DebugTimelineNodeState: ...
```

`node_state_for_cursor()` must compare typed cursor objects. It must not parse
labels, invocation strings, or widget row text.

`debug_session_id` and current execution/plate identity are derived from
`session`, `terminal_summary`, or `current_frame`. They are not stored as a
parallel copy.

### Projection Builder

```python
@dataclass(frozen=True, slots=True)
class DebugProjectionSource:
    events_by_execution: Mapping[str, Sequence[ProgressEvent]]
    runtime_projection: ExecutionRuntimeProjection
    session: DebugSession | None = None
    terminal_summary: DebugTerminalSummary | None = None
    step_index: "PipelineDebugStepIndex | None" = None


class DebugRuntimeProjectionBuilder:
    """Build debugger state from normal progress and debug progress context."""

    def build(
        self,
        source: DebugProjectionSource,
    ) -> DebugRuntimeProjection: ...
```

The core builder takes core debug session objects, not
`PipelineDebugSessionContext`, so it remains PyQt-free. The PyQt layer unwraps
`PipelineDebugSessionContext.active_session` and `.terminal_summary` before
calling the builder.

The builder consumes `runtime_projection` as the authoritative plate/axis
runtime aggregate. Its only permitted `events_by_execution` scan produces
`DebugProgressRecord` objects and attaches debug frame/timeline state. It must
not recompute execution/plate/axis completion, failure, or percent rules already
owned by `build_execution_runtime_projection(...)`. If the debugger needs more
generic event topology, extract a public adapter hook from
`openhcs.core.progress.projection` or use `zmqruntime.progress` directly instead
of creating another traversal convention.

If another host needs to supply the same information, use a protocol in the UI
integration layer:

```python
class DebugProjectionSessionSource(Protocol):
    session: DebugSession | None
    terminal_summary: DebugTerminalSummary | None
```

Do not import that protocol into core unless a non-UI core caller also needs it.

## Step Index Authority

Do not derive step labels or row ownership from strings. Add a small typed
projection beside `PipelineObjectStateBinding`:

`openhcs/pyqt_gui/services/pipeline_object_state_binding.py`

```python
@dataclass(frozen=True, slots=True)
class PipelineDebugStepRecord:
    step_scope: StepEditorScope
    step: FunctionStep

    @property
    def step_index(self) -> int: ...

    @property
    def step_scope_id(self) -> str: ...

    @property
    def step_name(self) -> str: ...

    @property
    def debug_pause(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PipelineDebugStepIndex:
    records: tuple[PipelineDebugStepRecord, ...]

    def by_step_index(self, step_index: int) -> PipelineDebugStepRecord | None: ...

    def by_step_scope(
        self,
        step_scope: StepEditorScope,
    ) -> PipelineDebugStepRecord | None: ...
```

Add method:

```python
class PipelineObjectStateBinding:
    def debug_step_index_for_plate(
        self,
        plate_scope_id: str,
    ) -> PipelineDebugStepIndex: ...
```

Authority:

- `PipelineObjectStateBinding.steps_for_plate(...)` remains the source of step
  object-state scope/row identity.
- `StepEditorScope` / `FunctionStepScopeToken` remain the source of parsed step
  scope identity.
- `FunctionStep` remains the source of name/debug-pause fields.
- This is a typed projection of that authority, not a second step registry.

## Progress Hook Integration

Update:

`openhcs/pyqt_gui/widgets/shared/services/progress_workflow_service.py`

Current behavior:

```python
self._runtime_projection = build_execution_runtime_projection(events_by_execution)
self._host.runtime_progress_projection = self._runtime_projection
```

Target behavior:

```python
self._runtime_projection = build_execution_runtime_projection(events_by_execution)
self._debug_runtime_projection = self._debug_projection_context_provider.build(
    events_by_execution=events_by_execution,
    runtime_projection=self._runtime_projection,
)
self._host.runtime_progress_projection = self._runtime_projection
self._host.debug_runtime_projection = self._debug_runtime_projection
```

Do not have `ProgressWorkflowService` reach into arbitrary host/widget methods
or call the same context accessor repeatedly. Add a protocol constructor
dependency:

```python
class DebugProjectionContextProvider(Protocol):
    def build(
        self,
        *,
        events_by_execution: Mapping[str, Sequence[ProgressEvent]],
        runtime_projection: ExecutionRuntimeProjection,
    ) -> DebugRuntimeProjection: ...
```

The provider lives on the Plate Manager/Pipeline Editor integration side,
unwraps `PipelineDebugSessionContext`, obtains `PipelineDebugStepIndex`, and
calls the core `DebugRuntimeProjectionBuilder`.

Update:

`openhcs/pyqt_gui/widgets/shared/services/debug_progress_service.py`

Use `DebugProgressRecord.from_progress_event(event)` for parse/extract. Keep the
snapshot notification path, but do not make it own current-frame state.

## UI State Surface API

First tighten the existing state-surface identity pattern. The current code uses
module constants such as `PIPELINE_EDITOR_STATE_IDENTITY =
UiStateSurfaceProviderIdentity.from_declaration(..., title=...)`. That mirrors
declaration metadata into module state. The plan must not add another instance
of that pattern.

Move the human-facing title onto the nominal identity declaration and derive the
provider identity mechanically:

`openhcs/agent/ui_bridge_identities.py`

```python
class UiStateSurfaceIdentityDeclarationBase(UiOwnedByWidgetIdentityDeclaration):
    """Registered state surface identity declaration."""

    title: ClassVar[str | None] = None

    @classmethod
    def require_title(cls) -> str:
        if cls.title is None:
            raise ValueError(f"{cls.__name__} does not declare a title.")
        return cls.title


class PipelineDebuggerStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "pipeline_debugger.state"
    enum_member_name = "PIPELINE_DEBUGGER"
    title = "Pipeline debugger"
    widget_identity = PipelineDebugToolbarWidgetIdentity
```

The generic state-surface catalog already discovers
`UiStateSurfaceIdentityDeclarationBase.__registry__`. Do not add this value to a
separate list. Do not bind it to `PipelineEditorWidgetIdentity` just because the
editor hosts the controls.

Update `UiStateSurfaceProviderIdentity.from_declaration(...)` so it reads
`title=declaration.require_title()` and no longer accepts a separate `title`
argument. Then migrate existing module-level identity constants
(`PLATE_MANAGER_STATE_IDENTITY`, `PIPELINE_EDITOR_STATE_IDENTITY`,
`PIPELINE_DEBUG_SESSION_STATE_IDENTITY`, `LIVE_OVERVIEW_STATE_IDENTITY`) to the
same declaration-owned title pattern.

Add DTOs:

`openhcs/agent/dto/ui_bridge.py`

First factor existing UI debug identity/session concepts into one composed
projection DTO so the new surface does not repeat fields already present in
`UiPipelineDebugSessionState`, `UiDebugCursorState`, and
`UiDebugTerminalSummaryState`.

```python
@dataclass(frozen=True, slots=True)
class UiDebugSessionIdentityCarrier:
    debug_session_id: str


@dataclass(frozen=True, slots=True)
class UiDebugSnapshotStoreRefCarrier:
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None


@dataclass(frozen=True, slots=True)
class UiPipelineDebugSessionProjection:
    """Bounded wire projection of PipelineDebugSessionContext."""

    current_plate_scope_id: str | None
    pipeline_scope_id: str | None
    manager_execution_state: str
    initialized: bool
    compiled: bool
    phase: str
    active_session_id: str | None
    execution_id: str | None
    axis_id: str | None
    selected_source_group: str | None
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None
    terminal_status: str | None
    cursor: UiDebugCursorState | None
    terminal_summary: UiDebugTerminalSummaryState | None
    actions: tuple[UiDebugActionState, ...]


@dataclass(frozen=True, slots=True)
class UiProgressIdentityState:
    execution_id: str
    plate_id: str
    axis_id: str | None
    step_name: str


@dataclass(frozen=True, slots=True)
class UiDebugRuntimeFrameState(
    UiDebugSessionIdentityCarrier,
    UiDebugSnapshotStoreRefCarrier,
):
    progress_identity: UiProgressIdentityState
    cursor: UiDebugCursorState
    event_type: str
    callable_name: str | None
    snapshot_id: str | None
    timestamp: float


@dataclass(frozen=True, slots=True)
class UiDebugTimelineNodeState:
    node_kind: str
    node_id: str
    label: str
    detail: str | None
    state: str
    percent: float
    snapshot_id: str | None
    children: tuple["UiDebugTimelineNodeState", ...] = ()


@dataclass(frozen=True, slots=True)
class UiPipelineDebuggerState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    debug_session: UiPipelineDebugSessionProjection
    current_frame: UiDebugRuntimeFrameState | None
    last_frame: UiDebugRuntimeFrameState | None
    timeline: tuple[UiDebugTimelineNodeState, ...]
```

Conversion rules:

- `UiDebugRuntimeFrameState` is a bounded wire projection of
  `DebugRuntimeFrame`.
- `UiProgressIdentityState` is the bounded wire projection of core
  `ProgressIdentity`.
- `UiDebugRuntimeFrameState.cursor` reuses `UiDebugCursorState`; it must not
  repeat cursor fields.
- `UiPipelineDebugSessionState` and `UiPipelineDebuggerState` both compose
  `UiPipelineDebugSessionProjection`. Do not copy the field list into both DTOs
  and do not use a giant inheritance carrier to simulate sharing.
- `UiDebugTimelineNodeState.node_kind` comes from
  `DebugTimelineNode.declaration.require_node_kind()`.
- `actions` comes from `DebugToolbarActionProjector.render_models(context)`.
- No DTO owns new debugger semantics.

## UI Provider API

Add:

`openhcs/pyqt_gui/services/ui_bridge_pipeline_debugger.py`

```python
class DeclaredUiStateSurfaceProvider(UiStateSurfaceProviderABC):
    """Provider whose bridge identity is derived from a nominal declaration."""

    identity_declaration: ClassVar[type[UiStateSurfaceIdentityDeclarationBase] | None] = None

    @classmethod
    def require_identity_declaration(
        cls,
    ) -> type[UiStateSurfaceIdentityDeclarationBase]: ...

    @property
    def identity(self) -> UiStateSurfaceProviderIdentity:
        return UiStateSurfaceProviderIdentity.from_declaration(
            self.require_identity_declaration()
        )


class PipelineDebuggerStateSurfaceProvider(DeclaredUiStateSurfaceProvider):
    identity_declaration = PipelineDebuggerStateSurfaceIdentityDeclaration

    def read(
        self,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument: ...

    def _state(self) -> UiPipelineDebuggerState: ...
```

The provider must:

1. resolve the selected/current Pipeline Editor target through existing bridge
   selection code;
2. ask Plate Manager/Pipeline Editor for `PipelineDebugSessionContext`;
3. ask Plate Manager for `DebugRuntimeProjection`;
4. project actions through `DebugToolbarActionProjector`;
5. convert through DTO constructors.

The provider must not:

- inspect Qt widget text;
- call `ProgressEvent.from_dict` itself;
- parse object-state field paths;
- enumerate debug actions manually.

Before adding this provider, extract the existing
`PipelineDebugSessionStateSurfaceProvider._state()`, action projection, cursor
projection, and terminal-summary projection into a shared
`UiPipelineDebugSessionStateProjector`. The existing debug-session surface and
the new debugger surface both call that projector.

`DeclaredUiStateSurfaceProvider` lives with the existing UI bridge
contracts and used by existing state-surface providers. Do not leave the new
debugger provider as the only nominal one while older providers keep
module-level identity constants.

Register the provider through `PipelineEditorBridgeProviderSet`, the current
owner for pipeline-editor/debug-toolbar surfaces. If a broader provider-set
registration cleanup is done first, it must iterate
`UiBridgeProviderSetABC.__registry__` or a provider-set factory declared on the
provider-set classes; do not add another manual composition table.

## Pipeline Editor Visual Binding

Add a narrow Qt-facing model:

```python
@dataclass(frozen=True, slots=True)
class PipelineDebugRowState:
    step_record: PipelineDebugStepRecord
    timeline_state: DebugTimelineNodeState
    current_frame: DebugRuntimeFrame | None
```

Add method on the editor workflow/presenter layer, not raw widget code:

```python
class PipelineDebugRowStateProjector:
    def project_rows(
        self,
        *,
        step_index: PipelineDebugStepIndex,
        debug_projection: DebugRuntimeProjection,
    ) -> tuple[PipelineDebugRowState, ...]: ...
```

Qt rendering consumes `PipelineDebugRowState`:

- current row marker;
- completed/failed/pending styling;
- current invocation badge inside the selected step;
- terminal last-frame marker.

No row styling code queries `DebugSession.cursor` directly once this model
exists. That keeps Qt and MCP consuming the same debugger projection.

## Inspector Integration

Keep `DebugInspectorWindow` as a payload inspector, not the debugger authority.

Data flow:

```text
DebugRuntimeProjection.current_frame.snapshot_id
  -> existing DebugSnapshot read path
  -> DebugViewModel / DebugInspectorWindow
```

Runtime values:

```text
active DebugSession id
  -> DebugRuntimeInspectionRequest
  -> paused worker controller
  -> DebugRuntimeInspectionResponse
  -> DebugViewModel.from_runtime_value_store(...)
```

Progress cannot carry runtime stores. The projection exposes runtime inspection
availability through action enablement, but it must not embed store
contents.

## MCP Exposure

Do not add a debugger-specific MCP tool in the first slice.

Expose through the existing state-surface system:

```bash
python -m openhcs.mcp.dev_client ui-state \
  --surface-id pipeline_debugger.state
```

If the rendered output needs custom formatting, add:

```python
class PipelineDebuggerStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    output_contract = UiPipelineDebuggerState
```

This reuses the existing `UiStateSurfacePayloadRenderer` AutoRegisterMeta path.
Do not add a renderer name list or a surface-id switch.

## Dry Run 1: Normal Non-Debug Progress

Inputs:

- execution emits compile/step progress with no debug context.

Expected:

- `DebugProgressRecord.from_progress_event(event)` returns `None`;
- `ExecutionRuntimeProjection` still updates;
- `DebugRuntimeProjection.records == ()`;
- `pipeline_debugger.state.timeline == ()`;
- no snapshot readback is attempted.

No new branches are added outside `DebugProgressRecord`.

## Dry Run 2: First Debug Step

Inputs:

- worker emits `BEFORE_INVOCATION` then `AFTER_INVOCATION`;
- both events include `DebugProgressContext`;
- current Pipeline Editor has step records from `PipelineObjectStateBinding`.

Expected:

- `DebugRuntimeProjection.current_frame` points at the latest debug record;
- timeline contains execution -> axis -> step -> invocation nodes;
- step label comes from `PipelineDebugStepIndex` or `ProgressEvent.step_name`;
- current row highlight comes from `PipelineDebugRowStateProjector`;
- `pipeline_debugger.state.current_frame` reports callable, step, axis, and
  snapshot id.

No code parses `invocation_key`.

## Dry Run 3: Repeated Step Across Invocations

Inputs:

- active session has `cursor.invocation_key`;
- user invokes `DebugCommandType.STEP`;
- runtime skips until cursor and emits the next debug progress record.

Expected:

- command behavior remains owned by `DebugInvocationExecutionStrategy`;
- projection updates from the new progress event;
- UI moves current invocation marker inside the same step or next step;
- MCP state surface changes without any MCP-specific polling logic.

No projection code decides execution behavior.

## Dry Run 4: Terminal Completion

Inputs:

- bounded debug execution completes;
- Plate Manager clears active session and keeps `DebugTerminalSummary`.

Expected:

- `PipelineDebugSessionContext.active_session is None`;
- `DebugRuntimeProjection.current_frame is None`;
- `DebugRuntimeProjection.last_frame` remains populated;
- `pipeline_debugger.state.phase == "terminal_complete"` from the shared
  debug-session projection;
- timeline keeps last completed frame;
- actions are projected from `PipelineDebugActionDeclarationBase`.

No fake active session is kept alive for display.

## Dry Run 5: Runtime Value Inspection

Inputs:

- active session is paused;
- user invokes `Inspect Runtime`.

Expected:

- action enablement comes from `InspectRuntimeValuesAction`;
- request goes through `DebugRuntimeInspectionRequest`;
- `DebugInspectorWindow` renders `DebugViewModel`;
- `DebugRuntimeProjection` remains only identity/timeline/current-frame state.

No runtime values are copied into progress projection or MCP state.

## Dry Run 6: Runtime Failure Not Caught At Compile Time

Inputs:

- compilation succeeds;
- worker emits a progress event with failure status/phase, `error`, and
  `traceback`;
- if the failure occurs at an invocation boundary, the event also carries
  `DebugProgressContext(event_type=DebugEventType.EXCEPTION)`.

Expected:

- `ExecutionRuntimeProjection` marks the plate/axis failed from progress;
- `DebugRuntimeProjection` marks the current/last frame failed from
  `DebugBoundaryEventDeclarationBase.for_event_type(event_type).boundary_outcome`;
- `pipeline_debugger.state` reports failure from runtime progress, not from
  ObjectState or the compiled artifact plan;
- the inspector can read failure context from the loaded `DebugSnapshot` when a
  snapshot id exists;
- no code invents a compile-time validation error for this runtime-only failure.

## Dry Run 7: Live Measurement Preview During Execution

Inputs:

- worker records measurement runtime values;
- progress context contains
  `LiveMeasurementProgressPayload.to_context()`;
- `ProgressWorkflowService.on_progress(...)` receives the event.

Expected:

- `LiveMeasurementProgressNotificationService.notify_from_progress_event(...)`
  emits `LiveMeasurementAvailableNotification`;
- `LiveMeasurementsWindow` / `LiveMeasurementTableModel` remains the owner of
  retained preview rows;
- `DebugRuntimeProjection` may link the observation to the same progress
  identity but does not duplicate preview rows into the debugger timeline;
- `pipeline_debugger.state` may expose a bounded "live measurements available"
  summary or related surface id, not a second measurement table schema.

## Implementation Slices

### Slice 1: Core Projection

- Add `DebugBoundaryOutcome`.
- Add `DebugBoundaryEventDeclarationBase` and concrete boundary declarations.
- Keep `DebugEventType` as a wire enum with string values only.
- Delete `DebugProgressStatus._STATUS_BY_EVENT_TYPE`.
- Add `openhcs/core/progress/debug_projection.py`.
- Move debug context parsing into `DebugProgressRecord`.
- Add timeline declarations and projection builder.
- Unit tests with synthetic `ProgressEvent` and `DebugProgressContext`,
  including `DebugProgressEventRequest.to_progress_event()` proving it reads
  status from `DebugBoundaryEventDeclarationBase`.

Tests:

```bash
source .venv/bin/activate
pytest -q \
  tests/unit/test_debug_runtime.py \
  tests/unit/test_zmq_progress.py \
  tests/unit/progress/test_live_measurements.py
```

### Slice 2: Progress Workflow Store

- Add `debug_runtime_projection` host attribute.
- Rebuild debug projection beside `ExecutionRuntimeProjection`.
- Add `DebugProjectionContextProvider` as the way
  `ProgressWorkflowService` obtains session/step-index context.
- Update `DebugProgressNotificationService` to use `DebugProgressRecord`.

Tests:

```bash
source .venv/bin/activate
pytest -q \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_batch_workflow_compile_engine.py \
  tests/unit/test_zmq_progress.py
```

### Slice 3: State Surface

- Add `PipelineDebuggerStateSurfaceIdentityDeclaration` and bind it to the
  existing `PipelineDebugToolbarWidgetIdentity`.
- Extract `UiPipelineDebugSessionStateProjector` from the existing
  debug-session surface provider.
- Add DTO carriers/core DTOs so `UiPipelineDebugSessionState` and
  `UiPipelineDebuggerState` share session/action fields.
- Add `PipelineDebuggerStateSurfaceProvider`.
- Add `PipelineDebuggerStateSurfaceRenderer(UiStateSurfacePayloadRenderer)` in
  `openhcs/mcp/dev_client_renderers/ui_bridge.py`, keyed by
  `PipelineDebuggerStateSurfaceIdentityDeclaration`.
- The renderer registers through `UiStateSurfacePayloadRenderer.__registry__`;
  do not add a renderer dict or manual dispatch branch.
- Register through `PipelineEditorBridgeProviderSet`; do not add a new bridge
  composition list.

Tests:

```bash
source .venv/bin/activate
pytest -q tests/unit/pyqt_gui/test_ui_agent_bridge.py tests/unit/agent/test_mcp_server.py
```

### Slice 4: Pipeline Editor Row Binding

- Replace `ProgressNodeType`, `_NODE_AGGREGATION_POLICY_BY_TYPE`, and
  node-type display branches with runtime tree node declarations.
- Add `PipelineDebugStepIndex`.
- Add `PipelineDebugRowStateProjector`.
- Bind row styling to projection.
- Keep existing toolbar action declarations.

Tests:

```bash
source .venv/bin/activate
pytest -q tests/unit/pyqt_gui/test_pipeline_editor_widget.py tests/unit/pyqt_gui/test_debug_toolbar.py
```

### Slice 5: Live Stress Pass

Use a compiled plate and a CellProfiler pipeline only as an integration case.
Generic projection tests must not depend on CellProfiler.

Manual checks:

```bash
source .venv/bin/activate
python -m openhcs.mcp.dev_client ui-state --surface-id pipeline_debugger.state
python -m openhcs.mcp.dev_client ui-overview
```

Expected UI:

```text
Debugger: A01 paused

> 1 CorrectIlluminationApply
  > default[0] correct_illumination_apply    after invocation
o 2 IdentifyPrimaryObjects
  o default[0] identify_primary_objects

Current frame:
  axis: A01
  step: CorrectIlluminationApply
  callable: correct_illumination_apply
  snapshot: <id>
```

## Audit Checklist

Before committing implementation:

- `rg -n "pipeline_debugger.state|pipeline_debug_toolbar.session" openhcs`
  must show identity declarations/providers/renderers, not independent
  surface lists.
- `rg -n "invocation_key.*split|split\\(.*invocation|startswith\\(.*debug" openhcs`
  must have no generic debugger projection matches.
- `rg -n "DebugCommandType\\." openhcs/pyqt_gui openhcs/agent openhcs/mcp`
  must route through `PipelineDebugActionDeclarationBase` or core command
  policies, not switch locally.
- `rg -n "_NODE_AGGREGATION_POLICY_BY_TYPE|class ProgressNodeType" openhcs`
  must be empty after the shared runtime tree-node declaration extraction.
- `rg -n "runtime values|Runtime Values" openhcs/mcp openhcs/agent`
  must not introduce an MCP-only runtime-values DTO.
- `rg -n "CellProfiler|cellprofiler" openhcs/core/progress/debug_projection.py`
  must be empty.

## Acceptance

The implementation is acceptable when:

- progress remains the only transport for debug location updates;
- snapshot/runtime inspection remains the only path for heavy data;
- Pipeline Editor, Debug Inspector, MCP state surface, and live overview all
  agree on the current frame;
- adding a new debug action requires only one declaration subclass;
- adding a new debugger timeline node kind requires only one declaration
  subclass;
- adding a new UI/MCP state surface requires one identity/provider/DTO class
  chain discovered by existing registries;
- no generic debugger code knows about CellProfiler leaf modules.
