# OpenHCS MCP UI Code Document Bridge Plan

## Status

Drafted 2026-06-17 after investigating how the PyQt plate manager code mode
renders and applies orchestrator configuration code, and how the MCP agent layer
currently exposes headless OpenHCS services.

Dry-run audited 2026-06-17 against the current code paths for MCP registration,
agent services, plate-manager code execution, ObjectState snapshots, ZMQ
transport, and main-window lifecycle.

Second dry-run audited 2026-06-17 against nested ObjectState snapshot behavior,
Qt thread dispatch requirements, ZMQ client auto-spawn behavior, capability
metadata, GUI config ownership, and shutdown ordering.

Third dry-run audited 2026-06-17 against MCP tool wrapper shape, source
selection semantics, cross-process payload envelopes, local bridge
authentication, snapshot identity, revision-token canonicalization, and
control-socket lifecycle details.

Fourth dry-run audited 2026-06-17 against UI bridge discovery from a separate
MCP process, descriptor-file policy, multiple running UI instances, timed
confirmation behavior, concurrent write serialization, GUI-config propagation,
and operation timeout semantics.

This is a concrete implementation plan. As of 2026-06-17, Phases 1-4 are
implemented for the first UI-owned code document slice: DTOs, agent services,
capability/MCP registration, the PyQt in-process provider, UI-thread dispatch,
ObjectState snapshot projection/restore, success-only apply snapshots,
cross-process ZMQ bridge server/client, descriptor discovery, and bridge
lifecycle wiring exist.

Phase 0 advisor cleanup has been re-run on the MCP/agent/UI bridge folders and
the touched GUI integration files. The current scans report no findings for:

```bash
timeout 180 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/agent openhcs/mcp \
  openhcs/pyqt_gui/services/ui_agent_bridge.py \
  openhcs/pyqt_gui/services/ui_bridge_server.py \
  openhcs/pyqt_gui/services/ui_thread_dispatch.py \
  openhcs/pyqt_gui/config.py \
  openhcs/pyqt_gui/main.py
```

Focused Phase 5 unit smoke is passing:

```bash
XDG_CACHE_HOME=/tmp/openhcs_test_cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_ui_bridge_lifecycle_config.py \
  tests/unit/pyqt_gui/test_main_config_propagation.py \
  -q
```

The remaining Phase 5 item is a GUI-capable smoke against a live OpenHCS main
window and standalone MCP process.

Related plans:

- `docs/plans/openhcs_mcp_server_plan_20260616.md`
- `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`
- `docs/plans/openhcs_mcp_implementation_blueprint_20260616.md`

## Goal

Expose selected running-UI state to agents through MCP by reusing OpenHCS code
mode as the semantic boundary.

The first supported document is the plate manager orchestrator configuration
code document: the same source the user sees when selecting plates and pressing
`Code` in the plate manager.

The agent should be able to:

1. Discover whether a running OpenHCS UI bridge is available.
2. List UI code documents exposed by the UI.
3. Read the plate-manager orchestrator code document for selected or all plates.
4. Validate an edited document without applying it.
5. Apply an edited document through the existing PyQt workflow, on the Qt UI
   thread, with revision protection.
6. Create a normal ObjectState snapshot when MCP applies a code document.
7. List the same ObjectState history the UI timeline/snapshot browser sees.
8. Restore the running UI to a snapshot after an agent edit.

## Non-Goals

- Do not let MCP introspect arbitrary PyQt widgets.
- Do not expose raw `QWidget`, `ObjectState`, `WindowManager`, or
  `ServiceRegistry` objects as MCP values.
- Do not expose raw `Snapshot`, `Timeline`, or ObjectState history dicts as MCP
  values.
- Do not make standalone MCP import PyQt.
- Do not add a second ad hoc path for mutating plate manager state.
- Do not create a second undo stack for MCP edits.
- Do not expose arbitrary Python execution as a normal headless MCP tool.
  Applying UI code documents is a UI-owned, local-only, opt-in bridge action.
- Do not implement generic widget automation in the first slice.

## Existing Architecture To Reuse

### MCP Agent Boundary

Existing MCP work uses this split:

```text
OpenHCS internals
  -> openhcs.agent services and DTOs
    -> openhcs.mcp transport adapter
```

The UI bridge must preserve that split. `openhcs.mcp` should register tools only;
bridge policy and DTOs belong under `openhcs.agent`.

### Plate Manager Code Mode

Current read path:

- `openhcs/pyqt_gui/widgets/plate_manager.py`
  - `OrchestratorCodeSource`
  - `PlateManagerWidget.action_code_plate`
  - collects `plate_paths`, `global_config`, `per_plate_configs`, and
    `pipeline_data`
  - renders Python source with `pycodify`

Current apply path:

- `openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py`
  - `PlateManagerCodeNamespaceField`
  - `PlateManagerOrchestratorCodePayload`
  - `PlateManagerCodeWorkflow.apply_namespace`
  - updates plate entries, global config, per-plate configs, pipeline data,
    compilation state, signals, and GUI event bus broadcasts

This is the correct load-bearing mutation path. MCP must call this workflow
instead of writing plate manager fields directly.

### UI Lookup

Current lookup path:

- `external/pyqt-reactive/src/pyqt_reactive/services/service_registry.py`
  - `ServiceRegistry.get(PlateManagerWidget)` finds the embedded plate manager.

Advisor debt found here:

- `ServiceRegistry._services` is typed as an opaque object bag.
- `AutoRegisterServiceMixin` dispatches `SERVICE_TYPE` cases inline.

Implementation should either clean this first or avoid depending on the
problematic internals by registering a dedicated bridge service explicitly.

Dry-run decision: avoid `ServiceRegistry` for the MCP bridge. The main window
already owns `embedded_widgets.plate_manager` and `embedded_widgets.pipeline_editor`.
Construct a dedicated UI bridge service from those explicit fields and pass that
service to the UI bridge server.

### Source DTO Shape

Current agent source shape:

- `openhcs/agent/dto/common.py`
  - `RenderedSource`
- `openhcs/agent/services/source_rendering_service.py`
  - `PythonSourceAssignment`
  - `PythonSourceAssignmentKind`

The UI bridge should reuse this source-document style instead of inventing an
unstructured code blob.

### ObjectState Snapshot And Time Travel

Current ObjectState authority:

- `external/ObjectState/src/objectstate/object_state_registry.py`
  - `ObjectStateRegistry.atomic(label, scope_id=None)` coalesces mutations into
    one undo step.
  - `ObjectStateRegistry.record_snapshot(label, scope_id=None)` records the
    shared system snapshot and fires history callbacks.
  - `ObjectStateRegistry.get_history_info(filter_fn=...)` returns current branch
    history for UI display.
  - `ObjectStateRegistry.time_travel_to(index)`,
    `time_travel_to_snapshot(snapshot_id)`, and `time_travel_to_head()` restore
    UI state.
  - `ObjectStateRegistry.list_branches()`, `switch_branch(name)`, and
    `get_current_branch()` expose the branch layer.
  - `ObjectStateRegistry.get_token()` is the cache invalidation token.

Current UI consumers:

- `openhcs/pyqt_gui/widgets/shared/time_travel_widget.py`
  - shows a filtered history using hidden scopes `{"", "__plates__"}`;
  - uses `time_travel_to(...)`, `time_travel_back()`,
    `time_travel_forward()`, and `time_travel_to_head()`;
  - subscribes with `add_history_changed_callback(...)`.
- `openhcs/pyqt_gui/windows/snapshot_browser_window.py`
  - uses the same hidden-scope filter;
  - double-click restores by history index;
  - branch dropdown uses `list_branches()` and `switch_branch(...)`.

MCP must project this same authority through typed DTOs. It must not return the
raw `get_history_info()` dicts, because the advisor correctly flags that as an
opaque boundary. The bridge should introduce a small snapshot projection service
that converts ObjectState history into named agent DTOs on the UI thread.

Default history filtering should match the UI timeline:

```python
HIDDEN_OBJECT_STATE_SCOPES = frozenset(("", "__plates__"))
```

MCP can expose `include_system_scopes=True` for diagnostics, but normal agent
calls should see the same semantic history as the UI.

## Public API Shape

### MCP Tools

Add these tools to `openhcs/mcp/server.py`:

```text
openhcs_ui_bridge_status()
openhcs_ui_list_code_documents()
openhcs_ui_get_code_document(document_id, selection_mode="selected", clean=True)
openhcs_ui_validate_code_document(document_id, source, revision_token)
openhcs_ui_apply_code_document(document_id, source, revision_token, require_confirmation=True)
openhcs_ui_list_snapshots(include_system_scopes=False)
openhcs_ui_restore_snapshot(snapshot_id=None, index=None, branch=None, require_confirmation=True, allow_auto_branch=False)
openhcs_ui_time_travel_head(require_confirmation=True)
openhcs_ui_list_branches()
openhcs_ui_switch_branch(branch, require_confirmation=True, allow_auto_branch=False)
openhcs_ui_get_operation_status(operation_id)
```

Every UI bridge MCP tool should accept the same optional connection override
arguments, even when the short signature above omits them for readability:

```text
host=None
port=None
transport_mode=None
timeout_ms=None
auth_token=None
descriptor_path=None
bridge_instance_id=None
```

The wrappers should pass those values to `UiBridgeService.connection_from_args(...)`
or an equivalent typed helper. They should not resolve environment variables,
descriptor files, or defaults themselves.

`openhcs_ui_get_code_document` returns source plus metadata. It does not apply
anything.

`openhcs_ui_validate_code_document` parses and validates the code document
through the UI bridge source policy and returns the normalized namespace
summary. It does not mutate UI state.

Validation must not run arbitrary Python source. If construction of approved
OpenHCS dataclass/config objects requires evaluation, the source policy must
first prove the document matches the approved declarative shape.

`openhcs_ui_apply_code_document` applies through the registered UI document
provider and must be marked mutating in `AgentCapabilitySpec.side_effects`.

`openhcs_ui_restore_snapshot`, `openhcs_ui_time_travel_head`, and
`openhcs_ui_switch_branch` are also mutating because they change the running UI
state, even though they do not edit code.

### Capability Registry Entries

Add `AgentCapabilitySpec` entries in `openhcs/agent/capabilities.py`:

```text
openhcs_ui_bridge_status
  service: ui_bridge
  output_type: UiBridgeStatus

openhcs_ui_list_code_documents
  service: ui_bridge
  output_type: UiCodeDocumentCatalog

openhcs_ui_get_code_document
  service: ui_bridge
  input_type: UiCodeDocumentRequest
  output_type: UiCodeDocument

openhcs_ui_validate_code_document
  service: ui_bridge
  input_type: UiCodeDocumentValidationRequest
  output_type: UiCodeDocumentValidationResult

openhcs_ui_apply_code_document
  service: ui_bridge
  side_effects: ("mutates_running_ui_state",)
  input_type: UiCodeDocumentApplyRequest
  output_type: UiCodeDocumentApplyResult

openhcs_ui_list_snapshots
  service: ui_bridge
  input_type: UiSnapshotListRequest
  output_type: UiSnapshotCatalog

openhcs_ui_restore_snapshot
  service: ui_bridge
  side_effects: ("mutates_running_ui_state", "time_travels_ui_state")
  input_type: UiSnapshotRestoreRequest
  output_type: UiSnapshotRestoreResult

openhcs_ui_time_travel_head
  service: ui_bridge
  side_effects: ("mutates_running_ui_state", "time_travels_ui_state")
  input_type: UiTimeTravelHeadRequest
  output_type: UiSnapshotRestoreResult

openhcs_ui_list_branches
  service: ui_bridge
  output_type: UiBranchCatalog

openhcs_ui_switch_branch
  service: ui_bridge
  side_effects: ("mutates_running_ui_state", "time_travels_ui_state")
  input_type: UiBranchSwitchRequest
  output_type: UiSnapshotRestoreResult

openhcs_ui_get_operation_status
  service: ui_bridge
  input_type: operation_id
  output_type: UiBridgeOperationRef
```

Second dry-run correction: `AgentCapabilitySpec` currently has side effects and
extras, but no formal way to say "this tool requires a running local UI bridge".
Add a typed runtime requirement field instead of leaving that as prose:

```python
runtime_requirements: tuple[str, ...] = ()
data_exposure: tuple[str, ...] = ()
security_requirements: tuple[str, ...] = ()
```

Use `("running_openhcs_ui_bridge",)` for every UI bridge tool. Keep
`requires_network=False` for localhost/IPC bridge operations unless remote TCP
support is explicitly enabled later. This lets blind agents distinguish an
unavailable runtime service from a missing MCP capability.

Use `data_exposure=("local_paths_in_source",)` for code-document read/validate/apply
capabilities that expose plate paths inside the source document. Use
`security_requirements=("ui_bridge_auth_token",)` for every operation that
requires the per-session token.

## DTOs

Create `openhcs/agent/dto/ui_bridge.py`.

In the actual file, define snapshot DTOs before code-document DTOs or enable
postponed annotations.

```python
@dataclass(frozen=True, slots=True)
class UiBridgeConnectionSpec:
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    persistent: bool = True
    timeout_ms: int = 5000
    auth_token: str | None = None
    descriptor_path: str | None = None
    bridge_instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class UiBridgeStatus:
    schema_version: str
    reachable: bool
    service: str = "openhcs.ui_bridge"
    bridge_instance_id: str | None = None
    auth_required: bool = True
    descriptor_path: str | None = None
    descriptor_status: str | None = None
    descriptors: tuple["UiBridgeDescriptorSummary", ...] = ()
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorFile:
    """Internal descriptor-file payload; never returned directly by MCP."""

    schema_version: str
    bridge_protocol_version: str
    bridge_instance_id: str
    pid: int
    started_at_unix: float
    host: str
    port: int
    transport_mode: str | None
    auth_token: str
    descriptor_path: str


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorSummary:
    schema_version: str
    bridge_instance_id: str
    pid: int
    started_at_unix: float
    descriptor_path: str
    status: str
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


class UiCodeDocumentId(str, Enum):
    PLATE_MANAGER_ORCHESTRATOR = "plate_manager.orchestrator_config"


class UiCodeDocumentSelectionMode(str, Enum):
    SELECTED = "selected"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class UiSnapshotRef:
    schema_version: str
    snapshot_id: str
    index: int
    branch: str
    parent_snapshot_id: str | None
    timestamp_unix: float
    timestamp: str
    label: str
    num_states: int
    is_current: bool
    is_head: bool
    uri: str


@dataclass(frozen=True, slots=True)
class UiBranchRef:
    schema_version: str
    name: str
    head_snapshot_id: str
    base_snapshot_id: str
    description: str
    is_current: bool


@dataclass(frozen=True, slots=True)
class UiCodeDocumentRef:
    schema_version: str
    document_id: str
    title: str
    widget_id: str
    readable: bool
    writable: bool
    supported_selection_modes: tuple[str, ...] = ()
    current_selection_count: int = 0
    total_scope_count: int = 0


@dataclass(frozen=True, slots=True)
class UiCodeDocumentCatalog:
    schema_version: str
    documents: tuple[UiCodeDocumentRef, ...]
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiCodeDocumentRequest:
    document_id: str
    selection_mode: str = UiCodeDocumentSelectionMode.SELECTED.value
    clean: bool = True


@dataclass(frozen=True, slots=True)
class UiCodeDocument:
    schema_version: str
    ref: UiCodeDocumentRef
    source: str
    mime_type: str
    size_bytes: int
    sha256: str
    revision_token: str
    current_snapshot: UiSnapshotRef | None
    selection_mode: str
    selected_scope_ids: tuple[str, ...]
    warnings: tuple[AgentWarning, ...] = ()
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationRequest:
    document_id: str
    source: str
    revision_token: str | None = None


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationResult:
    schema_version: str
    document_id: str
    valid: bool
    normalized_scope_ids: tuple[str, ...] = ()
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyResult:
    schema_version: str
    document_id: str
    applied: bool
    previous_revision_token: str
    outcome: str = "not_applied"
    operation_id: str | None = None
    new_revision_token: str | None = None
    pre_apply_snapshot: UiSnapshotRef | None = None
    post_apply_snapshot: UiSnapshotRef | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyRequest:
    document_id: str
    source: str
    revision_token: str
    require_confirmation: bool = True
    snapshot_label: str | None = None
    apply_if_time_traveling: bool = False


@dataclass(frozen=True, slots=True)
class UiSnapshotCatalog:
    schema_version: str
    current_branch: str
    current_snapshot_index: int
    object_state_token: int
    is_time_traveling: bool
    snapshots: tuple[UiSnapshotRef, ...]
    branches: tuple[UiBranchRef, ...]
    warnings: tuple[AgentWarning, ...] = ()
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class UiSnapshotListRequest:
    include_system_scopes: bool = False


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreRequest:
    snapshot_id: str | None = None
    index: int | None = None
    branch: str | None = None
    include_system_scopes: bool = False
    require_confirmation: bool = True
    allow_auto_branch: bool = False


@dataclass(frozen=True, slots=True)
class UiTimeTravelHeadRequest:
    require_confirmation: bool = True


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreResult:
    schema_version: str
    restored: bool
    target_snapshot: UiSnapshotRef | None
    current_snapshot: UiSnapshotRef | None
    catalog: UiSnapshotCatalog | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiBranchCatalog:
    schema_version: str
    current_branch: str
    branches: tuple[UiBranchRef, ...]


@dataclass(frozen=True, slots=True)
class UiBranchSwitchRequest:
    branch: str
    require_confirmation: bool = True
    allow_auto_branch: bool = False


@dataclass(frozen=True, slots=True)
class UiBridgeOperationRef:
    schema_version: str
    operation_id: str
    request_id: str | None
    operation_name: str
    status: str
    target_id: str | None
    started_at_unix: float
    completed_at_unix: float | None = None
    outcome: str | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()
```

Bounded-payload rule:

- `UiCodeDocument.source` is allowed because code mode is the semantic document
  boundary, but it must be bounded.
- Add a UI bridge config value `max_code_document_bytes` and reject larger
  reads/applies with a structured `document_too_large` error.
- Always include `size_bytes` and `sha256` so agents can reason about document
  identity without rereading the whole source.
- The revision token still includes the source hash; the DTO hash is for
  observability and agent-side diagnostics.

Local-path exposure rule:

- The plate-manager code document intentionally contains `plate_paths`, so it
  exposes local host paths. This is a deliberate exception to the normal
  opaque-id agent boundary because code mode is the user-facing source of truth.
- Mark read/apply capabilities with a side-effect/fact such as
  `exposes_local_paths` or add a dedicated capability metadata field if one is
  introduced.
- Keep the UI bridge localhost/IPC-only and opt-in while code documents expose
  paths.
- All non-source DTO fields should still use opaque snapshot/document ids where
  possible. Do not add extra host-path fields outside the code document unless a
  workflow requires them.

## Agent Service Layer

Create `openhcs/agent/services/ui_bridge_service.py`.

```python
class UiBridgeGatewayABC(ABC):
    @abstractmethod
    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus: ...

    @abstractmethod
    def list_documents(self, connection: UiBridgeConnectionSpec) -> UiCodeDocumentCatalog: ...

    @abstractmethod
    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument: ...

    @abstractmethod
    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult: ...

    @abstractmethod
    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult: ...

    @abstractmethod
    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog: ...

    @abstractmethod
    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult: ...

    @abstractmethod
    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult: ...

    @abstractmethod
    def list_branches(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBranchCatalog: ...

    @abstractmethod
    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult: ...

    @abstractmethod
    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        operation_id: str,
    ) -> UiBridgeOperationRef: ...
```

Implement:

- `ZMQUiBridgeGateway`: talks to a running UI process.
- `UiBridgeDescriptorResolver`: resolves descriptor files into an authenticated
  endpoint plus token for internal use, and into token-free summaries for status
  responses.
- `UiBridgeService`: owns user-facing validation, request construction, and
  error DTO conversion.
- `InProcessUiBridgeGateway`: optional test helper that calls a fake provider
  registry without sockets.

Register `UiBridgeService` in `openhcs/agent/services/__init__.py` and
`openhcs/mcp/context.py`.

The service should expose snapshot helpers from the same `ui_bridge` service
rather than a separate `object_state` service. The reason is practical: these
operations are only valid against a running UI process and must share the same
connection, dispatcher, and local security policy as code-document apply.

Connection resolution:

- `UiBridgeService` owns the default `UiBridgeConnectionSpec` resolution from
  environment and constructor defaults.
- MCP tool wrappers may accept optional `host`, `port`, `transport_mode`, and
  `timeout_ms`, `auth_token`, `descriptor_path`, and `bridge_instance_id`
  arguments, but should immediately pass them into
  `UiBridgeService.connection_from_args(...)` or an equivalent typed helper.
- Do not duplicate connection default logic in each MCP tool.
- `status()` should never raise for connection failure; it returns
  `reachable=False` with an `AgentError`.
- Resolution order:
  1. explicit `host`/`port`/`transport_mode` plus explicit `auth_token`;
  2. explicit `descriptor_path`, validated by `UiBridgeDescriptorResolver`;
  3. explicit `bridge_instance_id`, resolved from the configured descriptor
     directory;
  4. exactly one live descriptor in the configured descriptor directory;
  5. environment host/port/transport/auth overrides;
  6. default localhost port with unauthenticated status probe only.
- If more than one live descriptor is available and no explicit descriptor path
  or instance id is supplied, return `ambiguous_ui_bridge` with token-free
  `UiBridgeDescriptorSummary` entries.
- If a descriptor resolves but its authenticated ping fails, return
  `stale_ui_bridge_descriptor` or `ui_bridge_auth_failed` according to the
  failure instead of falling through to a different endpoint.

## PyQt UI Bridge Layer

Create `openhcs/pyqt_gui/services/ui_agent_bridge.py`.

### Provider Contract

```python
class UiCodeDocumentProviderABC(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "document_id"
    __skip_if_no_key__ = True

    document_id: ClassVar[str | None] = None

    @abstractmethod
    def ref(self) -> UiCodeDocumentRef: ...

    @abstractmethod
    def read(self, request: UiCodeDocumentRequest) -> UiCodeDocument: ...

    @abstractmethod
    def validate(
        self,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult: ...

    @abstractmethod
    def apply(
        self,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult: ...
```

### Plate Manager Provider

Create `PlateManagerOrchestratorCodeDocumentProvider`.

Responsibilities:

1. Resolve the live plate manager through the main-window-owned
   `UiAgentBridgeService`, backed by `embedded_widgets.require_plate_manager()`.
2. Reuse the same collection logic as `PlateManagerWidget.action_code_plate`.
3. Render with `OrchestratorCodeSource`.
4. Execute and validate edited code through the shared code-execution service.
5. Parse with `PlateManagerCodeNamespace.from_namespace`.
6. Apply with `PlateManagerCodeWorkflow.apply_namespace`.
7. Return structured DTOs.

Important refactor:

- Extract the source collection from `PlateManagerWidget.action_code_plate` into
  a reusable method or service:

```python
def plate_manager_code_context(
    manager: PlateManagerWidget,
    selection_mode: UiCodeDocumentSelectionMode,
) -> PlateManagerCodeDocumentContext:
    ...
```

`action_code_plate` should then call this same service. That prevents the MCP
bridge and the button from diverging.

Selection semantics:

- Do not copy the current GUI button's implicit no-selection fallback into the
  MCP API. `PlateManagerWidget.action_code_plate()` currently falls back from no
  selection to all plates for user convenience.
- For MCP, `selection_mode="selected"` with no selected plates returns a
  structured `no_selection` error. Agents must ask for `selection_mode="all"`
  explicitly to read all plates.
- The reusable collection service may accept an explicit
  `empty_selection_policy` so the GUI button can preserve its current UX while
  the MCP provider remains fail-loud.
- `UiCodeDocumentRef.current_selection_count`, `total_scope_count`, and
  `supported_selection_modes` let a blind agent choose deliberately before
  calling `get_code_document`.
- Apply must derive its semantic target from the validated `plate_paths` payload
  in the source, not from whatever the current UI selection is at apply time.

### Code Execution Reuse

The dry run found that the normal GUI code path is not just `exec(...)` plus
`PlateManagerCodeWorkflow.apply_namespace(...)`.

Current manager code path:

```text
AbstractManagerWidget._handle_edited_code(...)
  -> ManagerActionController.apply_edited_code(...)
    -> operations.pre_code_execution()
    -> with operations.patch_lazy_constructors(): exec(code, namespace)
    -> operations.migrate_code_namespace(...)
    -> operations.apply_code_namespace(namespace)
    -> operations.post_code_execution()
```

For the plate manager, `pre_code_execution()` opens/ensures the pipeline editor,
and `post_code_execution()` increments the ObjectState token. If MCP bypasses
that path, pipeline data can fail to apply when the editor is missing and
revision tokens may not advance.

Create a reusable `UiCodeDocumentExecutionService` or equivalent method that
owns:

1. execute edited source with the same lazy-constructor patching;
2. run the workflow-specific migration hook;
3. validate the namespace payload without mutation for validation requests;
4. run the workflow apply path for apply requests;
5. call pre/post hooks only for apply requests;
6. return structured namespace summaries and errors.

Both `ManagerActionController.apply_edited_code(...)` and
`PlateManagerOrchestratorCodeDocumentProvider.apply(...)` should use this shared
service. If extracting from `ManagerActionController` is too invasive for the
first slice, the provider may call the manager's existing action operations
directly, but tests must prove `pre_code_execution`, lazy-constructor patching,
migration, apply, and post-token increment all ran.

Source safety and validation semantics:

- Do not claim validation is non-mutating if it simply runs arbitrary `exec(...)`.
  Python source can perform side effects before the namespace is parsed.
- Add a `UiCodeDocumentSourcePolicy` before validation/apply execution. For the
  first plate-manager document, accept only the declarative pycodify shape:
  imports from approved OpenHCS modules, assignments to the expected document
  fields, literals/containers, enum references, and constructor calls needed for
  `GlobalPipelineConfig`, `PipelineConfig`, lazy configs, source bindings, and
  `FunctionStep`.
- Reject function definitions, class definitions, loops, comprehensions with
  side effects, attribute assignment, subscripting assignment, calls to
  unapproved modules, `open`, `exec`, `eval`, `__import__`, and filesystem or
  subprocess modules.
- Track imported names and aliases. Function references in `FunctionStep`
  declarations must resolve to approved OpenHCS registry functions, not merely
  any same-named Python object in the execution namespace.
- After namespace parsing, normalize and validate pipeline payloads with
  `FunctionStepTransportAuthority.normalize_pipeline(...)` so CellProfiler
  module/function references go through the same callable normalization already
  used for transport.
- The source policy should reuse the existing pycodify/source-rendering
  authority where possible by validating the expected generated AST shape rather
  than inventing a string parser.
- GUI code mode can remain the broader trusted local editor path. MCP
  validation/apply is a narrower code-document protocol unless the user enables
  an explicit unsafe development mode.
- Tests must include side-effecting source that validates in neither dry-run nor
  apply mode.

### ObjectState Snapshot Provider

Create `UiObjectStateSnapshotProvider` in the same module or a focused sibling
module.

Responsibilities:

1. Run only on the Qt UI thread.
2. Convert ObjectState branch/history dicts into `UiSnapshotRef` and
   `UiBranchRef`.
3. Apply the same hidden-scope filter as `TimeTravelWidget` by default.
4. Resolve revision anchors from unfiltered branch history, not the filtered UI
   display list.
5. Restore by `snapshot_id` with `time_travel_to_snapshot(...)` when provided.
6. Restore by `index` with `time_travel_to(...)` when provided.
7. Switch branch before restore when `branch` is provided.
8. Return the fresh post-restore catalog so MCP clients can verify the UI moved.
9. Never expose raw `Snapshot` or `ObjectState` instances.

Snapshot identity rules:

- `UiSnapshotRef.index` is the unfiltered branch-history index accepted by
  `ObjectStateRegistry.time_travel_to(index)`. It is not a filtered display-row
  index.
- Filtered UI history can skip snapshots, so display order and branch-history
  indexes can diverge. Prefer restore by `snapshot_id` in MCP examples.
- Build `UiSnapshotRef` from `ObjectStateRegistry.get_branch_history(...)` plus
  the UI hidden-scope filter, not from `get_history_info(...)` alone. The raw
  `Snapshot` provides `timestamp`, `parent_id`, and stable identity that the
  display dict does not expose.
- Include both `timestamp_unix` and a formatted timestamp string. The current UI
  display only exposes time-of-day; that is not stable enough for agent logs.

Restore request validation:

- Exactly one target selector should be supplied: `snapshot_id`, `index`, or
  `branch`.
- `branch` alone means switch to the branch head.
- `branch` plus `snapshot_id` or `index` is rejected in the first slice. Add
  branch-scoped restore only after tests prove the selection semantics.
- Restore by `snapshot_id` should reject ids outside the current branch in the
  first slice unless a branch-scoped selector is added. This avoids surprising
  cross-branch time travel while the branch semantics are still minimal.
- Branch switch while time-traveled can auto-create a future-preserving branch
  in current ObjectState. The MCP first slice should either require
  `allow_auto_branch=True` on `UiSnapshotRestoreRequest` or
  `UiBranchSwitchRequest` for that case, or return a structured
  `time_travel_branch_switch_requires_confirmation` error.
- Snapshot restore, time-travel head, and branch switch should default
  `require_confirmation=True`. The confirmation may be disabled only through the
  request plus explicit trusted-dev bridge config; otherwise declined/timeout
  confirmation returns a non-mutated result.
- Missing history returns a structured `no_snapshots` error.
- Unknown snapshot or branch returns a structured `not_found` error.

### Apply Creates Snapshots

`PlateManagerOrchestratorCodeDocumentProvider.apply(...)` must use the shared
ObjectState history:

```python
pre_snapshot = snapshot_provider.current_snapshot()
pre_head_id = snapshot_provider.current_branch_head_snapshot_id()
validated = code_execution_service.validate_source(request.source)
snapshot_scope = snapshot_provider.mutation_scope_for_payload(validated.payload)

if ObjectStateRegistry.is_time_traveling() and not request.apply_if_time_traveling:
    return stale_time_travel_error(...)

label = request.snapshot_label or f"edit {request.document_id} via MCP"
snapshot_provider.ensure_pre_apply_baseline()
code_execution_service.before_apply()
with ObjectStateRegistry.atomic_success(label, scope_id=snapshot_scope):
    code_execution_service.apply_payload(validated.payload)
code_execution_service.after_apply()

post_head_id = snapshot_provider.current_branch_head_snapshot_id()
if post_head_id == pre_head_id:
    return snapshot_not_recorded_error(...)
post_snapshot = snapshot_provider.current_snapshot()
```

Notes:

- Validation-only requests must not call `atomic(...)` or `record_snapshot(...)`.
- Stale revision rejection must not create a snapshot.
- Default apply labels must start with `"edit"`. ObjectState creates the initial
  baseline snapshot only for first edit labels, and MCP edits must preserve that
  ability to revert to the pre-agent state.
- Add `ObjectStateRegistry.atomic_success(...)` or an equivalent public
  success-only atomic context before implementation. The existing
  `atomic(...)` records in `finally`, so it can create a snapshot even when the
  apply body raises.
- `atomic_success(...)` must share the same `_atomic_depth`, `_atomic_label`,
  and `_atomic_triggering_scope` machinery as `atomic(...)`, and differ only in
  exception handling: on an exception it unwinds and clears outermost atomic
  state without recording. Nested ordinary `atomic(...)` blocks inside a
  success-only outer block must still defer into the outer snapshot.
- This matters because `PlateManagerCodeWorkflow.ensure_plate_entries(...)`
  currently opens `ObjectStateRegistry.atomic("register orchestrators")`.
  During MCP apply, that inner registration must coalesce into the outer
  `"edit ... via MCP"` snapshot instead of creating a separate visible
  `"register orchestrators"` snapshot.
- `ensure_pre_apply_baseline()` should create a baseline snapshot when history is
  empty before mutating state. That makes first-agent-edit rollback explicit
  instead of relying on label side effects inside `record_snapshot(...)`.
- `ensure_pre_apply_baseline()` must inspect unfiltered branch history, not the
  UI-filtered timeline display. If it creates a baseline, it should do so
  through a public `ObjectStateRegistry.ensure_baseline_snapshot(...)` API or
  equivalent, so the implementation does not call private snapshot internals
  from the UI bridge.
- If a baseline is created explicitly before the outer edit snapshot, the later
  `"edit ..."` snapshot must not create a second implicit baseline. Tests should
  assert the first MCP edit yields exactly one baseline and one edit snapshot.
- Do not compare `get_current_snapshot_index()` to detect snapshot creation.
  That method returns `-1` at live HEAD, so it stays `-1` before and after a
  successful new head snapshot.
- Do not call `record_snapshot(...)` after the atomic block as a fallback. The
  outer `atomic_success(...)` block is the single snapshot authority for the
  apply operation.
  A missing new head is an error to return, not a reason to create a second
  snapshot.
- Call `ObjectStateRegistry.increment_token()` after a successful apply,
  matching the existing manager code-editor path. In the pseudo-code above this
  is owned by `code_execution_service.after_apply()`.
- Do not use a synthetic `mcp.ui_bridge` triggering scope. The main window uses
  the snapshot triggering scope to decide which dirty ObjectState windows to
  reopen during time travel. Use the single edited plate/scope when there is
  exactly one semantic target; use `None` for multi-scope document edits so the
  existing UI includes all dirty states.
- Derive that mutation scope from the validated namespace payload, not from
  stale UI selection metadata. For the plate-manager document, a single
  `plate_paths` entry is the scope; zero or many entries means `None`.
- The apply path should not call ObjectState APIs from the bridge thread; the
  whole block runs inside `UiThreadDispatcher.call(...)`.
- The apply path must not let `PlateManagerCodeWorkflow.apply_namespace(...)`
  call UI methods from the bridge thread indirectly. The dispatcher boundary is
  around validation, workflow apply, ObjectState mutation, confirmation dialogs,
  and post-apply UI refresh.
- Serialize mutating bridge operations with a `UiBridgeMutationGate` or
  equivalent. Only one apply/restore/branch-switch operation may be in flight at
  a time for the running UI.
- Apply must recompute the current document and revision token inside the
  UI-thread mutation gate immediately before mutation. A token checked earlier
  by a bridge worker thread is advisory only.
- If a second mutating request arrives while one is running, return
  `ui_bridge_busy` or queue only when the request explicitly allows queuing.
  The first slice should fail-loud rather than queue hidden writes.
- Validation requests may run concurrently only if they do not execute source or
  mutate UI state. If validation needs UI state snapshots, gather immutable
  context on the UI thread and run expensive pure validation outside the mutation
  gate.
- ObjectState itself creates an auto-branch if a snapshot is recorded while
  time-traveled. The MCP first slice should fail-loud while time-traveled unless
  `apply_if_time_traveling=True`; this prevents agents from accidentally
  diverging the user's UI state.
- A future `branch_before_apply` option can be added after the UI exposes this
  choice explicitly.

## UI Thread Execution

All provider read/apply operations must run on the Qt UI thread.

Create `openhcs/pyqt_gui/services/ui_thread_dispatch.py`:

```python
@dataclass(frozen=True, slots=True)
class UiThreadRequest(Generic[T]):
    operation_name: str
    callback: Callable[[], T]
    timeout_ms: int = 5000


class UiThreadDispatcher(QObject):
    def call(self, request: UiThreadRequest[T]) -> T:
        ...
```

Rules:

- If already on the GUI thread, run directly.
- If called from the bridge server thread, queue to the GUI thread and wait for
  result or timeout.
- Return structured timeout and exception errors.
- Never mutate UI state from the bridge thread.
- Treat ObjectState as UI-thread-owned. Snapshot list, restore, branch switch,
  and code apply all go through this dispatcher.

Second dry-run dispatcher requirements:

- Construct the dispatcher object on the GUI thread after `QApplication` exists.
- Do not pass arbitrary Python return values through an untyped Qt signal
  signature. Use a request/result carrier owned by the dispatcher, with a
  `threading.Event` or equivalent wait primitive for the bridge server thread.
- The queued slot that executes callbacks must be a `QObject` slot living on
  the GUI thread. Use a signal or `QMetaObject.invokeMethod` only as the wake-up
  mechanism; the result carrier remains the typed Python boundary.
- Add an `is_shutting_down` flag. Calls received after shutdown begins return a
  structured `ui_bridge_shutting_down` error instead of hanging behind a closing
  Qt event loop.
- Timeouts must include the operation name and document/snapshot id where
  available. They should not leave partially-applied UI work running silently;
  apply operations that time out are reported as unknown outcome unless the GUI
  callback has not started yet.
- Track every dispatched operation by `operation_id`. If the bridge thread times
  out while a GUI callback is already running, the response outcome is
  `unknown_outcome` and the operation record remains queryable until completion.
- Add `openhcs_ui_get_operation_status(operation_id)` for active/recent UI
  bridge operations. The operation record should include request id, operation
  name, target id, status, outcome, and structured errors/warnings. Agents can
  re-read the document or snapshot catalog after a completed operation to verify
  final state; the status record does not need to carry large source payloads.
- Shutdown should first reject new operations, then wait briefly for the active
  operation to finish, then stop the server loop. It must not destroy the main
  window while an apply callback is still mutating UI state.

## UI Bridge Transport

Create `openhcs/pyqt_gui/services/ui_bridge_server.py`.

Use the existing ZMQ runtime pattern where practical:

- Reuse `OPENHCS_ZMQ_CONFIG`.
- Implement a small `UiBridgeZMQServer` on top of `zmqruntime.server.ZMQServer`,
  not `ExecutionServer`.
- Own a small server loop thread that calls `process_messages()` until stopped.
  `ZMQServer.start()` binds sockets; it does not by itself run a background
  message loop.
- Implement a small `UiBridgeControlClient` or gateway on top of the same
  REQ/REP control socket protocol, not `ExecutionClient`.
- Do not subclass `zmqruntime.client.ZMQClient` for the UI bridge control
  client. `ZMQClient.connect()` can auto-spawn a server process and kill stale
  endpoints. The UI bridge client must be probe-only: send a bounded control
  request to an already-running UI bridge and return `unreachable` on failure.
- Use a dedicated default UI bridge port, for example `7888`, plus the existing
  control-port offset.
- Expose request types:
  - `ping`
  - `list_code_documents`
  - `get_code_document`
  - `validate_code_document`
  - `apply_code_document`
  - `list_snapshots`
  - `restore_snapshot`
  - `time_travel_head`
  - `list_branches`
  - `switch_branch`

Do not reuse the execution server class directly. It owns execution semantics,
job queues, and progress events. The UI bridge should be a small peer service
with a compatible ping/control style.

Dry-run reason: `ExecutionServer.handle_control_message(...)` validates message
types against execution enums, and `ExecutionClient` owns execution/progress
submission behavior. UI bridge requests are synchronous UI document/control
requests, not queued execution jobs.

Protocol rules:

- Request and response envelopes must be typed at the UI bridge layer before
  invoking provider code. Unknown request types return a structured
  `unknown_ui_bridge_request` error.
- Transport payloads should be JSONable dictionaries at the process boundary,
  even if the first ZMQ implementation still uses pickle for framing. Do not
  pickle live dataclass instances, Qt objects, ObjectState objects, functions,
  or provider instances across the bridge.
- Add explicit envelopes:

```python
@dataclass(frozen=True, slots=True)
class UiBridgeRequestEnvelope:
    schema_version: str
    bridge_protocol_version: str
    request_id: str
    operation: str
    auth_token: str | None
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class UiBridgeResponseEnvelope:
    schema_version: str
    bridge_protocol_version: str
    request_id: str
    ok: bool
    payload: JsonObject
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()
```

- The UI bridge server decodes `payload` into the operation-specific request
  DTO on the UI side. The agent gateway decodes the response payload into the
  operation-specific response DTO before returning to MCP.
- Envelope DTOs belong in `openhcs.agent.dto.ui_bridge` because they are part of
  the stable agent-facing bridge protocol, not a PyQt implementation detail.
- The base `ZMQServer` currently serializes control payloads with pickle. Keep
  the first slice localhost/IPC-only and opt-in. Do not enable remote TCP for
  this bridge while pickle is the control serialization.
- Add a per-request `request_id` and include it in responses and logs.
- Add `max_request_bytes` and `max_response_bytes` policy checks before
  unpickled DTO payloads are accepted into the UI bridge provider layer. If
  enforcing pre-unpickle size is not possible with the reused server base, the
  implementation must either add that support to the base control receive path
  or use a focused JSON control socket instead of the base pickle handler.
- The UI bridge is control-only. `zmqruntime.server.ZMQServer` always binds both
  a data socket and a control socket, so implementation must either extract a
  reusable control-only server primitive or explicitly justify the unused data
  socket. Prefer a focused `UiBridgeControlServer`/`ZMQControlServer` if the
  extraction is small.
- The server loop must use a stop event plus poll/sleep/backoff so it does not
  busy-spin around nonblocking `process_messages()`. `stop()` must signal the
  loop, join the thread with a timeout, then close sockets. Socket-close races
  during shutdown should be logged at debug level, not reported as UI bridge
  crashes.

Startup:

- Add a disabled-by-default nominal GUI config object:

```python
@dataclass(frozen=True)
class AgentUiBridgeConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7888
    transport_mode: str = "tcp"
    timeout_ms: int = 5000
    descriptor_dir: str | None = None
    descriptor_path: str | None = None
    max_code_document_bytes: int = 2_000_000
    max_request_bytes: int = 4_000_000
    max_response_bytes: int = 4_000_000
    confirmation_timeout_ms: int = 30_000
    require_confirmation_for_mutations: bool = True
    allow_unsafe_code_documents: bool = False
```

- Attach it to `PyQtGUIConfig` as
  `agent_bridge: AgentUiBridgeConfig = field(default_factory=AgentUiBridgeConfig)`.
  Do not add loose plugin settings or environment-only behavior as the public
  config authority.
- `PyQtGUIConfig` is currently not passed into `OpenHCSPyQtApp` or
  `OpenHCSMainWindow`; only `GlobalPipelineConfig` is. The implementation must
  add explicit GUI config construction in `launch.py`, pass it into
  `OpenHCSPyQtApp`, then into `OpenHCSMainWindow`, and finally into embedded
  widgets that already accept `gui_config`.
- Avoid calling `get_default_pyqt_gui_config()` independently in multiple
  bridge components. Resolve one GUI config at app startup and pass it through
  the existing ownership graph.
- Add an environment override:
  - `OPENHCS_ENABLE_UI_BRIDGE=1`
- Add explicit bridge connection config:
  - `OPENHCS_UI_BRIDGE_HOST`
  - `OPENHCS_UI_BRIDGE_PORT`
  - `OPENHCS_UI_BRIDGE_TRANSPORT_MODE`
  - `OPENHCS_UI_BRIDGE_TIMEOUT_MS`
  - `OPENHCS_UI_BRIDGE_DESCRIPTOR`
  - `OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR`
- Start the bridge during `OpenHCSMainWindow.deferred_initialization` only when
  enabled.
- Store the running bridge on the main window or lifecycle workflow.
- Stop the bridge from `MainWindowLifecycleWorkflow.close()` so `closeEvent` and
  application cleanup share the same shutdown path.
- Stop must be idempotent; app cleanup may call close after the window has
  already begun closing.
- `OpenHCSPyQtApp` currently creates `OpenHCSMainWindow(self.global_config)`
  without passing `PyQtGUIConfig`. The implementation must either pass an
  explicit GUI config into the main window or have the main window resolve the
  default GUI config once. The bridge enable flag should not be read from
  unrelated global pipeline config.
- `MainWindowLifecycleWorkflow` is currently a frozen dataclass constructed
  before any deferred bridge startup. Either include a mutable bridge lifecycle
  dependency at construction time, or keep the bridge lifecycle as an explicit
  main-window-owned service with `start()`/`stop()` called from
  `deferred_initialization()` and `MainWindowLifecycleWorkflow.close()`. Do not
  rely on assigning new fields to the frozen workflow after startup.

Security:

- Bind to localhost/IPC only by default.
- Do not expose TCP remote access unless explicitly configured.
- Generate a per-UI-session bridge auth token at startup.
- Allow `ping`/status to report `auth_required=True` without exposing the token.
- Require the auth token for document reads, validation, apply, snapshot list,
  restore, branch switch, and time-travel head.
- Resolve the token for MCP through `OPENHCS_UI_BRIDGE_AUTH_TOKEN`, an explicit
  user-readable-only descriptor file (`OPENHCS_UI_BRIDGE_DESCRIPTOR`), or the
  validated descriptor directory (`OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR` /
  runtime default). Descriptor files must be created with permissions
  equivalent to `0600` and should live under `XDG_RUNTIME_DIR` when available.
- Do not print the token in logs, status DTOs, or exception messages.
- Require revision tokens for apply.
- Default `require_confirmation=True` for every mutating UI bridge operation:
  code apply, snapshot restore, time-travel head, and branch switch.
- Add a UI confirmation dialog for mutating operations unless disabled by
  explicit dev config.
- Confirmation is part of the UI-thread operation. A declined confirmation
  returns the operation-specific non-mutated result with
  `confirmation_declined`; an unanswered confirmation returns
  `confirmation_timeout` and must not mutate state.

Descriptor bootstrap:

- The UI writes a `UiBridgeDescriptorFile` JSON file when the bridge starts,
  and removes it on clean shutdown. This file contains the auth token and is an
  internal connection artifact, not a public MCP result DTO.
- Descriptor identity:
  - generate a per-process `bridge_instance_id` at bridge startup;
  - if `$OPENHCS_UI_BRIDGE_DESCRIPTOR` or `AgentUiBridgeConfig.descriptor_path`
    names an explicit file, write that file and fail loudly if another live UI
    already owns it;
  - otherwise write one file per instance under a descriptor directory using a
    filename such as `ui_bridge_<bridge_instance_id>.json`;
  - never use a single default descriptor filename for all UIs, because that
    lets one UI overwrite another and hides the multi-UI ambiguity.
- Default descriptor directory:
  - `$OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR` when set;
  - otherwise `$XDG_RUNTIME_DIR/openhcs/ui-bridge/` when `XDG_RUNTIME_DIR`
    exists;
  - otherwise `/tmp/openhcs-ui-bridge-$UID/`.
- Descriptor writes must be atomic: write to a temp file in the same directory,
  fsync when practical, then rename.
- Descriptor permissions must be user-only (`0600` file, parent directory not
  group/world writable unless it is a sticky runtime directory such as `/tmp`;
  instance directories created by OpenHCS should be `0700`).
- The agent-side resolver must validate owner uid, permissions, schema version,
  bridge protocol version, pid liveness, and a successful authenticated ping.
  Stale descriptors return structured `stale_ui_bridge_descriptor`, not a raw
  file or socket error.
- Do not route descriptor reads through broad `AgentPathPolicy` roots. Add a
  focused `UiBridgeDescriptorResolver` that only reads the explicit descriptor
  file or the configured runtime descriptor directory, then validates
  ownership/mode before using it. This prevents solving bridge discovery by
  expanding general agent read access.
- Multiple running UI instances require explicit selection. If more than one
  live descriptor is discovered, status returns `ambiguous_ui_bridge` with
  `UiBridgeDescriptorSummary` entries excluding tokens. The agent must retry
  with `descriptor_path` or `bridge_instance_id`.
- Explicit descriptor-path mode is still supported for tests and advanced local
  workflows, but it is not the default discovery path because it cannot
  represent multiple UIs safely.

Timed confirmation:

- Do not use a plain blocking `QMessageBox.exec()` for bridge confirmation.
  Existing `PyQtServiceAdapter.show_dialog(...)` blocks indefinitely and has no
  timeout/cancel state.
- Add a `UiBridgeConfirmationService` that owns a timed, parented confirmation
  dialog on the GUI thread.
- Confirmation must close with `confirmation_timeout` when its timer expires,
  `confirmation_declined` when the user rejects it, and
  `ui_bridge_shutting_down` when shutdown starts.
- Confirmation text must include document id, selected scope count, snapshot
  label/source hash for code apply, or target snapshot/branch for time-travel
  operations. It must not include auth tokens or large source text.

## Revision Tokens

Revision tokens prevent an agent from applying edits to stale UI state.

First implementation:

```text
revision_payload = {
  "schema_version": schema_version,
  "document_id": document_id,
  "provider_version": provider_version,
  "source_policy_version": source_policy_version,
  "selected_scope_ids": selected_scope_ids,
  "object_state_token": ObjectStateRegistry.get_token(),
  "current_branch": ObjectStateRegistry.get_current_branch(),
  "snapshot_anchor": snapshot_provider.current_snapshot_id_or_branch_head_id(),
  "history_length": snapshot_provider.current_history_length(),
  "is_time_traveling": ObjectStateRegistry.is_time_traveling(),
  "source_sha256": sha256(source),
}
revision_token = sha256(canonical_json(revision_payload))
```

Do not use `get_current_snapshot_index()` as the main version anchor. At live
HEAD it returns `-1`, so it does not distinguish one head snapshot from the
next. Use the snapshot provider's current visible snapshot id or branch head id,
plus history length.

Use canonical JSON with sorted keys and fixed separators before hashing. Do not
hash raw string concatenation; adjacent values can collide and make debugging
stale-token behavior harder.

Apply behavior:

1. Read current document.
2. Compare current token with request token.
3. If mismatched, return `applied=False` with `stale_revision`.
4. If matched, validate and apply.
5. Snapshot the apply through ObjectState.
6. Return new revision token and pre/post snapshot refs from a fresh read.

## Dry-Run Findings

The 2026-06-17 dry run found these implementation gaps and plan corrections:

1. Missing DTOs: the initial plan referenced `UiBridgeStatus`,
   `UiCodeDocumentCatalog`, and `UiCodeDocumentRequest` without defining them.
   They are now part of the DTO slice.
2. Validate/apply request conflation: validation should not carry
   `require_confirmation`, `snapshot_label`, or `apply_if_time_traveling`.
   The plan now uses separate validation and apply request DTOs.
3. First-edit baseline: ObjectState only auto-creates the initial baseline for
   labels beginning with `"edit"`. MCP apply labels must keep that prefix.
4. Snapshot creation detection: `get_current_snapshot_index()` returns `-1` at
   live HEAD, so it cannot prove a new head snapshot was created. Use branch
   head snapshot id and history length instead.
5. Snapshot authority: current `ObjectStateRegistry.atomic(...)` records on the
   outermost exit, including exception exits. MCP apply needs a public
   success-only atomic variant, and must not call `record_snapshot(...)` as a
   fallback after apply.
6. First-edit rollback: relying on `record_snapshot(...)` label side effects is
   too implicit for MCP. The bridge should explicitly ensure a pre-apply
   baseline when history is empty.
7. Code execution reuse: the GUI code path runs pre hooks, lazy-constructor
   patching, migration, workflow apply, and post-token increment. MCP must share
   that path or an extracted equivalent.
8. ZMQ class selection: `ExecutionServer` and `ExecutionClient` are execution-job
   abstractions. The UI bridge should use the lower-level `ZMQServer` control
   pattern with a small UI-specific client/gateway.
9. UI lookup: the main window already owns embedded plate manager and pipeline
   editor widgets. Prefer an explicit main-window-owned bridge service over
   generic `ServiceRegistry` lookup.
10. Lifecycle: bridge shutdown belongs in `MainWindowLifecycleWorkflow.close()`
   with idempotent `stop()`, not as an isolated close-event side effect.
11. Triggering scope: a synthetic MCP scope would prevent existing time-travel
   window navigation from finding dirty edited scopes. Apply snapshots must use
   the edited scope or `None` for multi-scope documents.
12. Advisor result: existing ObjectState and UI code contain reflection/dict
   boundaries. New MCP code should not copy those; it should project typed DTOs
   at the bridge boundary.

The second 2026-06-17 dry run found these additional gaps:

13. Capability metadata gap: the current capability registry cannot formally
    express "requires a running UI bridge". Add `runtime_requirements` instead
    of relying on descriptions.
14. Validation side-effect gap: raw `exec(...)` can mutate state even during
    validation. MCP must add a declarative source policy before validation and
    apply.
15. Nested snapshot gap: plate-manager apply can open inner ObjectState atomic
    blocks while adding plate entries. MCP apply must coalesce those into the
    outer edit snapshot.
16. Success-only atomic gap: a naive `atomic_success(...)` wrapper is not
    enough. It must share the existing atomic depth/label/scope machinery and
    cleanly unwind after exceptions.
17. Baseline duplication gap: explicit first-edit baseline creation can interact
    with ObjectState's implicit `"edit"` baseline behavior. Tests must prove
    exactly one baseline is created.
18. Dispatcher gap: a plain QObject helper is underspecified. The dispatcher
    needs GUI-thread ownership, typed result carriers, timeouts, and shutdown
    semantics.
19. ZMQ client gap: `ZMQClient.connect()` can spawn or kill endpoints. UI bridge
    clients must be probe-only and never start or kill the GUI.
20. Serialization/security gap: reused ZMQ control sockets currently use pickle.
    The UI bridge must remain localhost/IPC-only and bounded, or use a focused
    JSON control socket before remote TCP is considered.
21. Config authority gap: the UI bridge flag belongs under `PyQtGUIConfig`, not
    global pipeline config or loose plugin settings.
22. Lifecycle gap: `MainWindowLifecycleWorkflow` is frozen and constructed
    before deferred bridge startup. Bridge lifecycle must be injected explicitly
    or owned by a separate main-window service.
23. Payload-size gap: code documents need `size_bytes`, `sha256`, and configured
    request/response size limits.
24. Confirmation gap: mutating MCP calls need explicit confirmation declined and
    timeout states, and those states must not mutate ObjectState.

The third 2026-06-17 dry run found these additional gaps:

25. Selection fallback gap: the GUI code button silently falls back from no
    selection to all plates. MCP must not inherit that fallback; agents must ask
    for `selection_mode="all"` explicitly.
26. Discovery gap: code-document refs need dynamic selection counts and
    supported selection modes so blind agents can choose safely.
27. Source identity gap: function names imported in generated code must resolve
    through approved imports and the OpenHCS function registry, then be
    normalized with `FunctionStepTransportAuthority`.
28. Snapshot identity gap: `get_history_info(...)` is display-oriented and lacks
    parent/raw timestamp data. Snapshot DTOs need unfiltered branch-history
    indexes, parent ids, and raw timestamps.
29. Restore semantics gap: filtered display indexes and branch-history indexes
    can diverge. MCP should prefer snapshot ids and define index as an
    unfiltered branch-history index.
30. Branch switch gap: ObjectState can auto-create a branch when switching while
    time-traveled. MCP needs an explicit opt-in or structured error for that
    side effect.
31. Revision hash gap: raw string concatenation is not a robust token format.
    Use canonical JSON with provider/source-policy version and source hash.
32. Transport payload gap: process-boundary messages need JSONable request and
    response envelopes, not pickled live dataclass/provider objects.
33. Control socket gap: the UI bridge is control-only, while `ZMQServer` binds a
    data socket too. Extract or justify a control-only primitive.
34. Server loop gap: a nonblocking `process_messages()` loop needs stop-event,
    poll/backoff, join timeout, and benign shutdown-race handling.
35. Auth gap: localhost is not enough for mutating UI state. Add a per-session
    auth token distributed through env or a user-only descriptor file.
36. Local-path exposure gap: plate-manager code documents intentionally expose
    `plate_paths`; this must be capability metadata and local-only policy, not
    an accidental leak.

The fourth 2026-06-17 dry run found these additional gaps:

37. Descriptor bootstrap gap: a separate MCP process may not inherit UI bridge
    env vars. The UI must write a validated descriptor file and the agent must
    resolve it through a focused descriptor resolver.
38. Path-policy gap: broadening `AgentPathPolicy` to read runtime descriptors
    would widen unrelated file access. Descriptor reads need a narrow
    owner/mode-validating resolver.
39. Multi-UI gap: more than one OpenHCS UI may be running. Descriptor status
    needs bridge instance identity and stale/ambiguous descriptor handling.
40. GUI-config propagation gap: `PyQtGUIConfig` is not currently passed through
    app/main-window construction. Agent bridge config must be propagated
    explicitly instead of being env-only.
41. Confirmation gap: existing `PyQtServiceAdapter.show_dialog(...)` is a
    blocking `QMessageBox.exec()` without timeout/shutdown semantics. Bridge
    confirmation needs its own timed service.
42. Concurrent write gap: revision tokens are not enough if two mutating bridge
    requests validate concurrently. Mutating operations need a UI-wide mutation
    gate and in-gate revision recomputation.
43. Timeout outcome gap: a bridge-thread timeout during an already-running GUI
    callback cannot safely be reported as failed. It needs `unknown_outcome` and
    an operation status query.
44. Shutdown ordering gap: UI shutdown must reject new bridge operations and
    wait briefly for any active mutation before destroying widgets.

## Gap Resolution Matrix

Every dry-run finding now has a concrete solution authority in the plan:

| # | Status | Solution authority |
|---|---|---|
| 1 | Solved | DTOs section defines `UiBridgeStatus`, code-document, snapshot, branch, descriptor-summary, and operation DTOs. |
| 2 | Solved | Separate validation/apply request DTOs keep confirmation and mutation fields out of validation. |
| 3 | Solved | Apply snapshot rules require default labels beginning with `"edit"` and explicit pre-apply baseline handling. |
| 4 | Solved | Snapshot creation checks use branch head id plus history length, not `get_current_snapshot_index()`. |
| 5 | Solved | Apply uses `ObjectStateRegistry.atomic_success(...)` as the single snapshot authority. |
| 6 | Solved | `ensure_pre_apply_baseline()` creates a rollback point through public ObjectState API before first mutation. |
| 7 | Solved | `UiCodeDocumentExecutionService` reuses manager pre-hook, lazy-constructor patching, migration, apply, and post-hook behavior. |
| 8 | Solved | Transport uses a UI bridge control gateway/server, not `ExecutionServer` or `ExecutionClient`. |
| 9 | Solved | Provider resolves widgets through main-window-owned `UiAgentBridgeService` and explicit embedded-widget requirements. |
| 10 | Solved | Startup/shutdown lives in main-window lifecycle with idempotent bridge stop. |
| 11 | Solved | Apply derives ObjectState triggering scope from validated payload and never uses synthetic MCP scopes. |
| 12 | Solved | Phase 0 records advisor debt to clean or quarantine; new bridge code must use typed DTO/projection boundaries. |
| 13 | Solved | `AgentCapabilitySpec.runtime_requirements` marks every UI bridge capability as requiring a running local UI bridge. |
| 14 | Solved | `UiCodeDocumentSourcePolicy` validates declarative AST/source shape before any execution/evaluation. |
| 15 | Solved | Success-only outer atomic coalesces inner plate-registration atomics into one visible edit snapshot. |
| 16 | Solved | `atomic_success(...)` must share atomic depth/label/scope machinery and unwind without recording on exception. |
| 17 | Solved | Baseline tests require exactly one baseline and one edit snapshot for the first MCP edit. |
| 18 | Solved | `UiThreadDispatcher` has GUI-thread ownership, typed carriers, timeout, operation tracking, and shutdown semantics. |
| 19 | Solved | UI bridge client is probe-only and must not subclass/use `ZMQClient.connect()` auto-spawn or kill behavior. |
| 20 | Solved | Envelopes are JSONable and bounded; pickle reuse remains localhost/IPC-only until replaced by focused JSON control. |
| 21 | Solved | `AgentUiBridgeConfig` under `PyQtGUIConfig` is the config authority; env vars are overrides only. |
| 22 | Solved | Bridge lifecycle is either injected into the frozen workflow up front or owned as an explicit main-window service. |
| 23 | Solved | Code documents and transport envelopes have `max_*_bytes`, `size_bytes`, and `sha256` checks. |
| 24 | Solved | All mutating operations default to timed UI confirmation with declined/timeout/shutdown non-mutation outcomes. |
| 25 | Solved | MCP selected reads fail with `no_selection`; only explicit `selection_mode="all"` reads all plates. |
| 26 | Solved | Document refs expose supported selection modes, current selection count, and total scope count. |
| 27 | Solved | Source policy resolves imported functions through approved OpenHCS function semantics and normalizes with `FunctionStepTransportAuthority`. |
| 28 | Solved | Snapshot DTOs are built from unfiltered branch history and include parent ids plus raw timestamps. |
| 29 | Solved | Restore uses snapshot ids by preference and defines index as unfiltered branch-history index. |
| 30 | Solved | Restore/branch DTOs carry `allow_auto_branch`; implicit auto-branching is rejected. |
| 31 | Solved | Revision tokens use canonical JSON over schema/provider/source-policy/source/snapshot/branch fields. |
| 32 | Solved | Process-boundary request/response envelopes are JSONable DTO payloads, not live dataclass/Qt/ObjectState objects. |
| 33 | Solved | Plan requires a control-only server primitive or explicit justification for unused `ZMQServer` data socket. |
| 34 | Solved | Server loop uses stop event, poll/backoff, join timeout, and benign socket-close shutdown handling. |
| 35 | Solved | Per-session auth token is generated by the UI, required for non-status operations, and never logged or returned in status. |
| 36 | Solved | Local path exposure is deliberate code-document behavior and declared in capability `data_exposure`. |
| 37 | Solved | Descriptor bootstrap uses token-bearing internal `UiBridgeDescriptorFile` plus resolver-backed connection discovery. |
| 38 | Solved | `UiBridgeDescriptorResolver` handles descriptor reads; `AgentPathPolicy` is not widened. |
| 39 | Solved | Instance-specific descriptor files avoid overwrites; multiple live descriptors return `ambiguous_ui_bridge`. |
| 40 | Solved | `PyQtGUIConfig.agent_bridge` is passed through launch, app, main window, and bridge construction. |
| 41 | Solved | `UiBridgeConfirmationService` replaces blocking `QMessageBox.exec()` for bridge confirmations. |
| 42 | Solved | `UiBridgeMutationGate` serializes mutating operations and recomputes revision state inside the gate. |
| 43 | Solved | Started-but-timed-out callbacks return `unknown_outcome` and are queryable through operation status. |
| 44 | Solved | Shutdown rejects new operations, waits briefly for active mutation, stops the loop, and removes/invalidates descriptors. |

## MCP Registration

Update `openhcs/mcp/server.py`:

- Add tools as thin one-line wrappers over `ctx.ui_bridge_service`.
- Do not import PyQt.
- Return `to_jsonable(...)` DTOs.
- Keep function signatures scalar/list/dict-friendly for FastMCP. Construct
  typed request DTOs inside `UiBridgeService`, not directly in the tool wrapper
  unless the wrapper already has all arguments.
- Accept optional connection overrides consistently across UI bridge tools:
  `host`, `port`, `transport_mode`, `timeout_ms`, `auth_token`,
  `descriptor_path`, and `bridge_instance_id`. Delegate default/auth/descriptor
  resolution to `UiBridgeService`.

Update `openhcs/mcp/context.py`:

- Add `ui_bridge_service`.
- Instantiate with `UiBridgeService()`.

Update `openhcs/agent/capabilities.py`:

- Add all UI bridge capabilities.
- Mark apply as mutating.
- Mark restore/head/branch switch as mutating time-travel operations.
- Mark UI bridge tools as requiring a running local UI bridge.
- Mark code-document tools with `data_exposure=("local_paths_in_source",)` and
  token-protected tools with `security_requirements=("ui_bridge_auth_token",)`.

## Tests

### Unit Tests

Add `tests/unit/agent/test_ui_bridge_service.py`.

Coverage:

- service returns structured unavailable status when no UI bridge responds;
- status reports `auth_required` without leaking the auth token;
- service gateway does not subclass/use `ZMQClient.connect()` auto-spawn or
  kill behavior;
- service resolves connection defaults once through a typed helper, not in each
  MCP tool wrapper;
- service resolves auth token from explicit args, environment, or descriptor
  file without logging it;
- descriptor resolver rejects world-readable, wrong-owner, stale, or dead-pid
  descriptor files with structured errors;
- descriptor resolver reports `ambiguous_ui_bridge` when more than one live UI
  descriptor is available and no explicit descriptor/instance is supplied;
- explicit descriptor path plus bridge instance id selects the intended running
  UI without broadening `AgentPathPolicy`;
- service sends and receives JSONable `UiBridgeRequestEnvelope` and
  `UiBridgeResponseEnvelope` payloads;
- service lists documents through fake gateway;
- service reads fake `UiCodeDocument`;
- service rejects stale revision apply;
- service maps gateway exceptions to `AgentError`;
- service exposes snapshot catalog through fake gateway;
- service rejects restore requests with multiple selectors;
- service propagates restore/head/branch confirmation and `allow_auto_branch`
  request fields through the gateway;
- service exposes `openhcs_ui_get_operation_status` through the fake gateway;
- validation request DTO does not expose apply-only confirmation fields.

Add `tests/unit/pyqt_gui/test_ui_agent_bridge.py`.

Coverage:

- plate-manager provider returns `plate_manager.orchestrator_config`;
- selected-vs-all selection changes `selected_scope_ids`;
- `selection_mode="selected"` with no selected plates returns `no_selection`;
- `selection_mode="all"` is required to read all plates when nothing is
  selected;
- document refs report supported selection modes, current selection count, and
  total scope count;
- rendered source contains `plate_paths`, `global_config`,
  `per_plate_configs`, and `pipeline_data`;
- rendered source reports `size_bytes` and `sha256`;
- oversized source/read payloads return `document_too_large`;
- source policy rejects side-effecting Python before validation/apply;
- source policy rejects imported function names that do not resolve through
  approved OpenHCS function registries;
- parsed pipeline payloads are normalized through
  `FunctionStepTransportAuthority.normalize_pipeline(...)`;
- validation rejects missing `plate_paths` or `pipeline_data`;
- apply calls `PlateManagerCodeWorkflow.apply_namespace`;
- apply records one visible edit snapshot on success;
- first MCP edit also preserves/creates a pre-edit baseline snapshot;
- first MCP edit creates exactly one baseline and one edit snapshot, not two
  baselines;
- nested `ObjectStateRegistry.atomic("register orchestrators")` calls inside
  plate-manager apply coalesce into the outer MCP edit snapshot;
- failed apply inside the mutation block does not record a new snapshot;
- apply does not create a second fallback snapshot after `atomic_success(...)`;
- failed apply inside `atomic_success(...)` unwinds atomic state and does not
  poison the next snapshot label/scope;
- stale revision and validation failure record no snapshot;
- concurrent mutating requests are rejected with `ui_bridge_busy` unless an
  explicit queuing mode is later added;
- apply recomputes the current document and revision token inside the UI-thread
  mutation gate immediately before mutation;
- apply result includes pre/post `UiSnapshotRef`;
- snapshot provider lists history matching the UI hidden-scope policy;
- snapshot refs use unfiltered branch-history indexes and include parent id plus
  raw timestamp;
- snapshot provider restores by id and by index;
- snapshot provider rejects current-slice cross-branch restore by snapshot id;
- branch switch while time-traveled requires explicit opt-in or returns a
  structured auto-branch warning/error;
- restore/head/branch operations require confirmation by default and record no
  mutation on confirmation declined or timeout;
- revision token changes when branch head changes, even when
  `get_current_snapshot_index()` remains `-1`;
- apply invokes the same pre/post manager code-execution hooks as the GUI code
  button;
- single-scope apply records the edited scope as ObjectState triggering scope;
- multi-scope apply records no synthetic MCP triggering scope;
- current plate pipeline refresh signals are emitted;
- dispatcher executes read/apply/restore on the Qt UI thread;
- dispatcher returns `ui_bridge_shutting_down` after shutdown begins;
- dispatcher timeout after the GUI callback has started returns
  `unknown_outcome` plus an operation id, not a false failure;
- operation status reports active/completed/failed/timed-out bridge operations
  without exposing live Qt or ObjectState objects;
- confirmation declined/timeout returns structured non-applied results without
  mutation;
- timed confirmation uses the bridge confirmation service, not blocking
  `QMessageBox.exec()`;
- descriptor file creation uses user-only permissions when auth descriptor mode
  is enabled.
- descriptor files are instance-specific by default and cannot overwrite another
  live UI descriptor.

Add `tests/unit/mcp/test_ui_bridge_tools.py`.

Coverage:

- MCP server registers UI bridge tools when optional `mcp` is installed;
- capability registry includes all UI bridge tools;
- UI bridge capabilities declare `runtime_requirements`;
- read/apply document capabilities declare local path exposure metadata;
- `openhcs_ui_apply_code_document` declares side effects;
- `openhcs_ui_get_operation_status` is registered and declared read-only;
- snapshot restore/head/branch tools declare mutating time-travel side effects;
- UI bridge transport tests use the UI bridge gateway/client, not
  `ExecutionClient`;
- MCP tool wrappers delegate connection argument resolution to
  `UiBridgeService`, not per-tool ad hoc defaults;
- UI bridge control client never sends or receives live dataclass instances over
  the process boundary; it uses JSONable envelopes.
- bridge server stop joins the loop thread and treats socket-close races during
  shutdown as benign.

### Integration Tests

Add `tests/integration/mcp/test_ui_bridge_smoke.py`.

Coverage:

- launch PyQt test app with UI bridge enabled;
- discover the bridge through a validated descriptor without relying on inherited
  environment variables;
- stale descriptors are ignored or returned as structured unavailable status;
- multiple live descriptors require explicit descriptor path or bridge instance;
- explicit descriptor path and bridge instance id select the intended UI;
- add a synthetic or temp plate entry;
- MCP client reads plate-manager code document;
- MCP client validates the document;
- MCP client applies a small safe edit, such as toggling
  `napari_streaming_config.enabled`;
- UI reflects the change through existing config/pipeline signals;
- MCP client lists snapshots and sees the apply snapshot;
- MCP client restores the pre-apply snapshot;
- UI reflects the reverted code/config state;
- app shutdown stops the UI bridge without leaving a server thread/socket.
- app shutdown removes the descriptor or marks it stale before the process exits.

## Implementation Phases

### Phase 0: Advisor Cleanup Gate

Run the advisor on all files to touch:

```bash
env XDG_CACHE_HOME=/tmp/openhcs_test_cache .venv/bin/python -m nominal_refactor_advisor \
  openhcs/agent/capabilities.py \
  openhcs/mcp/context.py \
  openhcs/mcp/server.py \
  openhcs/pyqt_gui/app.py \
  openhcs/pyqt_gui/config.py \
  openhcs/pyqt_gui/launch.py \
  openhcs/pyqt_gui/main.py \
  openhcs/pyqt_gui/services/main_window_workflows.py \
  openhcs/pyqt_gui/services/service_adapter.py \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py \
  openhcs/pyqt_gui/widgets/shared/time_travel_widget.py \
  openhcs/pyqt_gui/windows/snapshot_browser_window.py \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/manager_action_controller.py \
  external/ObjectState/src/objectstate/object_state_registry.py \
  external/zmqruntime/src/zmqruntime/client.py \
  external/zmqruntime/src/zmqruntime/server.py
```

Before feature implementation, resolve or explicitly quarantine:

- `external/pyqt-reactive/src/pyqt_reactive/services/service_registry.py`
  advisor findings if the bridge will rely on generic service lookup internals.
- `ObjectStateRegistry.get_history_info(...)` returns opaque dicts; the MCP
  bridge must project those into `UiSnapshotRef`/`UiSnapshotCatalog` at the
  boundary instead of propagating dicts.
- `TimeTravelWidget` and `SnapshotBrowserWindow` have presentation-level advisor
  debt. Do not copy those patterns into new bridge code; reuse ObjectState
  semantics through typed projection services.
- Advisor also flags fallback-heavy history navigation in those widgets. The
  MCP snapshot provider should compute current/display positions explicitly from
  `get_branch_history(...)` and snapshot ids instead of using display-widget
  `next(..., default)` fallbacks as authority.
- `ManagerActionController` currently owns useful code-execution semantics but
  its operation port is broad. If behavior is extracted, make the extracted
  service a typed code-document execution boundary rather than another
  `Any`/dict operation bag.
- `ZMQClient` auto-spawn/kill behavior is valid for execution clients but not
  for the UI bridge. The bridge client must be a new probe-only boundary.
- Raw `exec(...)` is too broad for MCP validation/apply. Add a source policy
  gate before exposing a write-capable UI code document.
- `ObjectStateRegistry.atomic(...)` is exception-recording by design today.
  Add and test success-only atomic behavior before any MCP apply path records
  snapshots.
- `OpenHCSMainWindow.setup_status_bar` currently uses reflective fallback for
  `debug_toolbar`; do not copy that pattern into the bridge. Bridge dependencies
  should be required explicitly or fail loudly.
- `openhcs/pyqt_gui/launch.py` has reflective/default fallback around logging
  and CLI config. Bridge enablement must be resolved through a typed GUI config
  loader with classified defaults, not scattered environment/argument fallback.
- `openhcs/pyqt_gui/services/main_window_workflows.py` uses opaque `object`
  annotations for main-window and splitter carriers. If bridge lifecycle wiring
  touches these paths, introduce a narrow nominal host/protocol for the bridge
  dependency instead of widening generic `object` plumbing.
- `external/pyqt-reactive/.../manager_action_controller.py` exposes
  `CodeEditorPayload.data: object`. The extracted code-document execution
  service should replace that at the new boundary with typed payload variants
  instead of reusing the opaque carrier as MCP API surface.
- `PlateManagerWidget.action_code_plate()` has a user-convenience selected-to-all
  fallback. The extracted collection service must make that policy explicit so
  MCP can fail loudly on empty selected reads.
- `ObjectStateRegistry.switch_branch(...)` can auto-branch while time-traveled.
  MCP branch tools must expose or block that side effect explicitly.
- `external/zmqruntime` has reflective registration, fallback-heavy IPC checks,
  and execution-client auto-spawn/kill behavior. The UI bridge may reuse low-level
  socket conventions, but its public gateway must be a typed probe-only control
  boundary and should not depend on those execution-client behaviors.

Preferred avoidance:

- create and register a dedicated `UiAgentBridgeService` instead of
  expanding use of `ServiceRegistry._services`.

### Phase 1: DTO And Agent Service

Files:

- add `openhcs/agent/dto/ui_bridge.py`
- add `openhcs/agent/services/ui_bridge_service.py`
- update `openhcs/agent/services/__init__.py`
- update `openhcs/mcp/context.py`

Acceptance:

- unit tests pass with fake gateway;
- snapshot DTOs provide a typed replacement for ObjectState history dicts;
- capability DTOs expose `runtime_requirements`;
- code-document DTOs expose bounded payload metadata;
- UI bridge envelope DTOs are JSONable and versioned;
- connection DTOs support auth token resolution without exposing tokens in
  status payloads;
- connection DTOs support descriptor path and bridge instance id without
  exposing auth tokens in status payloads;
- descriptor resolution is owned by a focused resolver with owner/mode/pid
  checks, not by broadening `AgentPathPolicy`;
- descriptor DTOs distinguish internal token-bearing descriptor files from
  public token-free descriptor summaries;
- operation-status DTOs expose active/recent operation state without leaking
  live implementation objects;
- advisor clean on new files.

### Phase 2: PyQt In-Process Document Provider

Files:

- add `openhcs/pyqt_gui/services/ui_agent_bridge.py`
- add or extract `UiCodeDocumentExecutionService`;
- add `openhcs/pyqt_gui/services/ui_thread_dispatch.py`
- add `UiObjectStateSnapshotProvider` under the same UI bridge service area;
- add `ObjectStateRegistry.atomic_success(...)` or equivalent success-only
  public atomic support before using ObjectState for MCP apply snapshots;
- modify `openhcs/pyqt_gui/widgets/plate_manager.py` to extract reusable
  code-document context creation;
- modify the manager code-editor path only enough to share execution behavior,
  preserving button behavior;
- keep `action_code_plate` behavior unchanged.

Acceptance:

- pressing `Code` in the plate manager preserves the existing user workflow
  while emitting the long-term `per_plate_configs` document shape;
- unit tests can read and validate a document through provider API;
- MCP selected reads fail loudly when no plates are selected;
- validation rejects non-declarative side-effecting source before `exec(...)`;
- validation confirms imported function names and normalized FunctionStep
  payloads through existing function semantics;
- unit tests can list and restore ObjectState snapshots through provider API;
- provider apply runs pre-code hook, lazy-constructor patching, migration hook,
  workflow apply, and post-token increment;
- nested plate-entry registration snapshots coalesce into one MCP edit snapshot;
- failed apply inside success-only atomic records no snapshot and leaves the
  next snapshot label/scope correct;
- mutation gate serializes apply/restore/branch-switch operations and recomputes
  revision tokens inside the gate;
- dispatcher shutdown returns structured errors instead of hanging;
- dispatcher records operation ids and returns `unknown_outcome` for callbacks
  that time out after starting;
- confirmation uses a timed bridge-owned service with declined, timeout, and
  shutdown outcomes;
- restore/head/branch operations carry typed confirmation request DTOs and
  `allow_auto_branch` is required before branch switching while time-traveled;
- advisor clean on touched files.

### Phase 3: UI Bridge Server

Files:

- add `openhcs/pyqt_gui/services/ui_bridge_server.py`
- add bridge config/env handling in `openhcs/pyqt_gui/config.py`,
  `openhcs/pyqt_gui/launch.py`, `openhcs/pyqt_gui/app.py`, and the main-window
  startup/shutdown path;
- add `ZMQUiBridgeGateway` under `openhcs/agent/services/ui_bridge_service.py`
  or a small sibling module.
- add `UiBridgeControlClient` or equivalent ZMQ control gateway that does not
  subclass `ExecutionClient`;
- wire bridge shutdown through `MainWindowLifecycleWorkflow.close()`.

Acceptance:

- UI bridge starts only when enabled;
- bridge enablement is sourced from `PyQtGUIConfig.agent_bridge` plus explicit
  environment overrides;
- `PyQtGUIConfig` is propagated through launch, app, main window, and bridge
  construction; bridge components do not independently fetch defaults;
- bridge auth token is generated and exposed only through env/descriptor paths;
- bridge writes a user-only descriptor containing instance id, pid, endpoint,
  auth metadata, and schema version;
- bridge writes instance-specific descriptors by default and rejects explicit
  descriptor-path collisions with another live UI;
- bridge status reports stale and ambiguous descriptors structurally;
- standalone MCP can read bridge status;
- standalone MCP can list UI snapshots through the bridge;
- standalone MCP can query operation status after an unknown-outcome timeout;
- bridge client does not auto-spawn, kill, or subclass execution clients;
- bridge transport uses JSONable envelopes and does not pickle live OpenHCS/Qt
  objects across process boundaries;
- control server loop has stop-event, backoff/polling, and join-timeout tests;
- bridge shutdown rejects new operations, waits briefly for an active mutation,
  stops the loop, and removes or invalidates the descriptor.

### Phase 4: MCP Tools

Files:

- update `openhcs/mcp/server.py`
- update `openhcs/agent/capabilities.py`
- add MCP unit tests.

Acceptance:

- `openhcs://capabilities` lists UI bridge tools;
- apply tool declares `mutates_running_ui_state`;
- operation-status tool is registered as read-only and returns
  `UiBridgeOperationRef`;
- snapshot restore tools declare `time_travels_ui_state`;
- standalone MCP imports without PyQt.

### Phase 5: End-To-End Smoke

Run:

```bash
env OPENHCS_ENABLE_UI_BRIDGE=1 XDG_CACHE_HOME=/tmp/openhcs_test_cache \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/mcp/test_ui_bridge_tools.py \
  -q
```

If a GUI-capable test environment is available, also run:

```bash
env OPENHCS_ENABLE_UI_BRIDGE=1 XDG_CACHE_HOME=/tmp/openhcs_test_cache \
  .venv/bin/python -m pytest tests/integration/mcp/test_ui_bridge_smoke.py -q
```

## Acceptance Criteria

- Standalone MCP can report whether a running UI bridge exists.
- Standalone MCP can discover a running UI through a validated descriptor file
  without broadening general path-read policy.
- Multiple or stale UI descriptors produce structured status/errors instead of
  silently choosing one.
- Descriptor discovery uses instance-specific files by default, and public
  status never exposes the descriptor auth token.
- Standalone MCP can read the same plate-manager orchestrator source the GUI
  code button shows.
- UI bridge capabilities formally declare their runtime requirement for a
  running local UI bridge.
- Code-document reads and writes are bounded and include `size_bytes` and
  `sha256`.
- Code-document reads that expose local paths are opt-in, local-only, and marked
  in capability metadata.
- The GUI code button and MCP read tool share the same source generation code.
- MCP selected reads do not silently expand to all plates.
- MCP apply uses `PlateManagerCodeWorkflow`, not direct field mutation.
- MCP validation/apply rejects non-declarative side-effecting source before
  execution.
- MCP validation resolves imported functions through existing OpenHCS function
  semantics and normalizes FunctionStep payloads through the existing transport
  authority.
- MCP apply creates a normal ObjectState snapshot that appears in the UI
  timeline and snapshot browser.
- First MCP edit creates a reversible baseline without duplicate baseline
  snapshots.
- Nested ObjectState registrations during apply coalesce into one visible edit
  snapshot.
- MCP can list and restore those snapshots through typed DTOs.
- Snapshot refs include raw timestamps, parent ids, and unfiltered branch
  history indexes.
- Snapshot restore uses `ObjectStateRegistry.time_travel_*` on the UI thread,
  not custom field mutation.
- Branch switching while time-traveled does not silently auto-branch without
  explicit opt-in.
- Snapshot restore, time-travel head, and branch switch are confirmed by
  default, same as code apply.
- Validation failures and stale revision failures do not create snapshots.
- Apply failures inside the mutation block do not create snapshots.
- Applying while time-traveled fails unless explicitly requested.
- Apply is guarded by revision token and local UI bridge enablement.
- Mutating operations are serialized by a UI-wide mutation gate and revalidate
  revision state inside that gate.
- Mutating bridge calls either receive UI confirmation or fail with structured
  confirmation errors without mutation.
- UI confirmation is timed and shutdown-aware; it does not rely on blocking
  `QMessageBox.exec()`.
- Timed-out operations that already started in the GUI thread return
  `unknown_outcome` with an operation id, and MCP can query final status.
- The UI bridge client never auto-spawns or kills a GUI process.
- UI bridge process-boundary payloads are versioned JSONable envelopes.
- UI bridge read/mutate operations require a per-session auth token that is not
  logged or exposed in status.
- Dispatcher calls fail cleanly during UI shutdown instead of hanging, and
  shutdown drains or marks active mutations before widgets are destroyed.
- Standalone MCP still imports without PyQt.
- All touched files pass advisor scan.
- Unit tests cover DTOs, service, provider, MCP registration, validation, and
  apply behavior.

## Open Questions

1. Should write apply require a visible UI confirmation every time, or should
   trusted local development mode allow silent apply?
2. Should the UI bridge share the execution ZMQ browser, or appear as a separate
   "UI Bridge" server type?
3. Should pipeline-editor code mode become the second document after plate
   manager, or should config-window code mode come next?
4. Should MCP expose branch creation now, or only list/switch branches and let
   ObjectState auto-branch when explicit time-traveled apply is enabled later?
