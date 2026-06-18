# OpenHCS MCP UI Capability Registry Plan

Status: Fifth dry-run audited; gaps filled.
Owner: OpenHCS agent/MCP/UI bridge architecture.
Date: 2026-06-17.

## Goal

Expose the running OpenHCS UI to MCP agents through a generic, load-bearing
capability registry instead of hardcoding individual widgets into the bridge.

The bridge should let a blind agent discover:

- which UI state surfaces can be read or polled;
- which code documents can be read, validated, applied, and snapshot-restored;
- which UI actions can be invoked safely;
- which windows/scopes are open and focusable;
- which ObjectState scopes exist, what owns them, and what snapshots can restore
  them.

The public contract remains:

`PyQt UI internals -> openhcs.pyqt_gui bridge providers -> openhcs.agent DTOs/services -> openhcs.mcp tools/resources`

The MCP server must not expose raw PyQt widgets, raw `ObjectState` instances,
raw `PipelineOrchestrator` objects, or arbitrary Python internals as first-class
agent API.

## Non-Goals

- Do not make MCP scrape widget trees.
- Do not make MCP execute arbitrary UI Python.
- Do not replace the existing ZMQ bridge transport.
- Do not block compile/run tools until full pipeline completion.
- Do not couple MCP directly to `WindowManager`, `PlateManagerWidget`, or
  `ObjectStateRegistry` internals.

## Current Architecture Facts

- `OpenHCSMainWindow._start_ui_bridge_if_enabled()` currently constructs
  `UiAgentBridgeService(plate_manager=self.plate_manager_widget)` directly in
  `openhcs/pyqt_gui/main.py`.
- `UiAgentBridgeService` currently owns separate dictionaries for code document
  providers and state providers in `openhcs/pyqt_gui/services/ui_agent_bridge.py`.
- The current provider registration is PlateManager-specific:
  `PlateManagerOrchestratorCodeDocumentProvider` and
  `PlateManagerStateSurfaceProvider`.
- `UiStateSurfaceProviderABC.read()` currently returns `UiPlateManagerState`,
  so a generic request type has a concrete PlateManager response type.
- `openhcs/agent/dto/ui_bridge.py` has enum identities only for
  `plate_manager.orchestrator_config` and `plate_manager.state`.
- `UiBridgeOperationName`, `UiBridgeRequestOperation`, the gateway, and MCP tools
  already form a registered operation chain. That is good machinery to extend.
- `PlateManagerAction` and `PlateManagerWidget.ACTION_ROUTES` already define a
  nominal set of UI actions. They should be reused as the PlateManager action
  provider's source of truth.
- `dispatch_widget_action()` currently returns `None`, and async widget actions
  are handed to an async runner without returning an operation receipt. That is
  insufficient for MCP action invocation semantics.

## Target Architecture

Introduce a UI bridge capability registry at the PyQt bridge boundary:

```text
OpenHCSMainWindow
  -> OpenHCSUiBridgeCompositionRoot
    -> UiBridgeSurfaceRegistry
      -> UiCodeDocumentProviderABC
      -> UiStateSurfaceProviderABC
      -> UiActionProviderABC
      -> UiWindowProviderABC
      -> UiObjectStateSurfaceProviderABC
  -> UiAgentBridgeService
  -> UiBridgeControlServer
  -> openhcs.agent UiBridgeGatewayABC / UiBridgeService
  -> openhcs.mcp tools/resources
```

The registry owns discovery and routing. `UiAgentBridgeService` owns UI-thread
dispatch, mutation gating, operation tracking, and snapshot integration. The ZMQ
server stays a process-boundary adapter.

## New Production Modules

### `openhcs/pyqt_gui/services/ui_bridge_contracts.py`

Add cycle-free provider contracts and identity carriers:

- `UiCodeDocumentProviderABC`
- `UiStateSurfaceProviderABC`
- `UiActionProviderABC`
- `UiWindowProviderABC`
- `UiObjectStateSurfaceProviderABC`
- provider identity dataclasses;
- provider result/registration protocol helpers.

This module must not import `UiAgentBridgeService`, `UiBridgeControlServer`, the
main window, or concrete widgets. It may import agent DTOs and pure-Python bridge
helpers only.

### `openhcs/pyqt_gui/services/ui_bridge_registry.py`

Add nominal bridge registration machinery:

- `UiBridgeSurfaceRegistry`
- `UiBridgeRegistrationContext`
- `UiBridgeProviderSetABC`
- `UiBridgeProviderSetRegistration`
- `UiRegisteredSurfaceKind`

The registry stores providers by typed identity and exposes typed catalog
methods. It should reject duplicate ids at registration time.

The registry imports provider contracts from `ui_bridge_contracts.py`; it must
not import `ui_agent_bridge.py`.

### `openhcs/pyqt_gui/services/ui_bridge_composition.py`

Add the main-window composition root:

- accepts `OpenHCSMainWindow`, `EmbeddedWidgetRefs`, `WindowManager`, and the
  default `UiObjectStateSnapshotProvider`;
- registers embedded widget provider sets;
- registers window and ObjectState projection providers;
- returns a ready `UiAgentBridgeService`.

`OpenHCSMainWindow._start_ui_bridge_if_enabled()` should call this composition
root and should no longer know that PlateManager has code/state providers.

### `openhcs/pyqt_gui/services/ui_bridge_plate_manager.py`

Move PlateManager-specific bridge providers out of `ui_agent_bridge.py`:

- `PlateManagerOrchestratorCodeDocumentProvider`
- `PlateManagerStateSurfaceProvider`
- `PlateManagerActionProvider`
- any PlateManager provider-set registration object

This prevents `UiAgentBridgeService` from staying coupled to PlateManager while
the registry becomes generic.

### `openhcs/pyqt_gui/services/ui_bridge_windows.py`

Add a bounded projection layer over `WindowManager`:

- `UiWindowProjectionService`
- `WindowManagerWindowProvider`

It reports opaque `window_id`, title, visibility, scope id if available, and
supported navigation/focus capabilities. It must not return `QWidget` objects.

### `openhcs/pyqt_gui/services/ui_bridge_object_state.py`

Add a bounded projection layer over `ObjectStateRegistry`:

- `UiObjectStateProjectionService`
- `ObjectStateSurfaceProvider`

It reports scope id, object type name, current snapshot refs, branch/time-travel
state, dirty/revision tokens when available, and bounded code-document links for
editable scopes. It must not return raw object values.

## Provider Contracts

### Code Documents

Keep the existing behavior but move provider registration through
`UiBridgeSurfaceRegistry`.

Required methods:

- `summary() -> UiCodeDocumentSummary`
- `read(UiCodeDocumentRequest) -> UiCodeDocument`
- `validate(UiCodeDocumentValidationRequest) -> UiCodeDocumentValidationResult`
- `apply(UiCodeDocumentApplyRequest) -> UiCodeDocumentApplyResult`

### State Surfaces

Make state surfaces payload-generic. The response must become a wrapper such as:

```python
UiStateSurfaceDocument(
    summary=UiStateSurfaceSummary(...),
    payload_schema="openhcs.ui.plate_manager_state.v1",
    payload={...},
    current_revision_token=...,
    current_snapshot=...,
    selected_scope_ids=(...),
)
```

`UiPlateManagerState` can remain as the typed payload DTO, but the gateway and
MCP boundary should handle the generic state-surface document.

### Actions

Add an action provider contract:

```python
class UiActionProviderABC(ABC):
    identity: UiActionProviderIdentity

    def summary(self) -> UiActionSummary: ...
    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult: ...
```

Action summaries must include:

- `action_id`
- `widget_id`
- display title
- enabled/disabled state
- selection mode
- side-effect classification
- confirmation requirement
- whether invocation is synchronous or returns an async operation receipt
- related state surface ids to poll after invocation

`PlateManagerActionProvider` should derive PlateManager action ids from
`PlateManagerAction` and route invocation through the existing PlateManager
action semantics. The provider is responsible for returning an MCP-useful
receipt.

### Windows

Add a window provider contract:

```python
class UiWindowProviderABC(ABC):
    def list_windows(self) -> UiWindowCatalog: ...
    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult: ...
```

Window focus should use existing `WindowManager` behavior on the UI thread.

### ObjectState

ObjectState exposure is a state surface, not a raw-object API. It should expose:

- scope ids;
- object type labels;
- ownership hints;
- snapshot ids;
- dirty/revision info;
- links to code documents when a scope has a safe code mode;
- current branch/time-travel state.

Code edits through MCP must create snapshots through the same
`UiObjectStateSnapshotProvider` used by the UI.

## DTO Additions

Extend `openhcs/agent/dto/ui_bridge.py` with string identities instead of
central closed enums:

- `UiSurfaceIdentity`
- `UiActionIdentity`
- `UiWindowIdentity`
- `UiObjectStateScopeIdentity`

Add DTOs:

- `UiBridgeCatalog`
- `UiBridgeConnectionRequest`
- `UiPageRequest`
- `UiCatalogPageMetadata`
- `UiStateSurfaceDocument`
- `UiActionSummary`
- `UiActionCatalog`
- `UiActionInvokeRequest`
- `UiActionInvokeResult`
- `UiActionInvocationStatus`
- `UiWindowSummary`
- `UiWindowCatalog`
- `UiWindowFocusRequest`
- `UiWindowFocusResult`
- `UiObjectStateScopeSummary`
- `UiObjectStateCatalog`
- `UiPathProjection`
- `UiMutationRequestToken`
- `UiMutationReceipt`

Keep existing `UiCodeDocumentIdentity` and `UiStateSurfaceIdentity` as
backward-compatible aliases or focused subclasses if that is less disruptive.

## Transport Additions

Extend `UiBridgeOperationName` with:

- `LIST_UI_CAPABILITIES`
- `LIST_ACTIONS`
- `INVOKE_ACTION`
- `LIST_WINDOWS`
- `FOCUS_WINDOW`
- `LIST_OBJECT_STATE_SCOPES`
- `GET_OBJECT_STATE_SCOPE`

Add corresponding `UiBridgeRequestOperation` subclasses in
`ui_bridge_server.py`. This keeps the registered operation pattern.

Add matching methods to:

- `UiBridgeGatewayABC`
- `UnavailableUiBridgeGateway`
- `ZMQUiBridgeGateway`
- `InProcessUiBridgeGateway`
- agent-facing `UiBridgeService`

`UiBridgeStatus` should also report protocol capabilities, including supported
operation names and live provider catalog schema versions. MCP tools that need a
new operation should return a typed unsupported-operation error when connected
to an older UI bridge.

Every transport operation phase must update the complete operation spine in the
same change:

- `UiBridgeOperationName`
- `UiBridgeOperationRequestPayload`
- `UiBridgeOperationDispatchResult`
- `UiBridgeRequestOperation` subclass registration
- gateway ABC and all concrete gateways
- MCP tool/resource wrapper
- static capability registry entry
- live `supported_operations` status projection

Unsupported but syntactically valid operation names are not invalid requests.
They must return a nominal `unsupported_ui_bridge_operation` error/result.

## MCP Additions

Add tools in `openhcs/mcp/server.py`:

- `openhcs_ui_list_bridges`
- `openhcs_ui_list_actions`
- `openhcs_ui_invoke_action`
- `openhcs_ui_list_windows`
- `openhcs_ui_focus_window`
- `openhcs_ui_list_object_state_scopes`
- `openhcs_ui_get_object_state_scope`

Keep existing tools as stable wrappers:

- `openhcs_ui_list_code_documents`
- `openhcs_ui_get_code_document`
- `openhcs_ui_validate_code_document`
- `openhcs_ui_apply_code_document`
- `openhcs_ui_list_state_surfaces`
- `openhcs_ui_get_state_surface`
- snapshot and branch tools

`openhcs_ui_list_bridges` is MCP-local and uses the descriptor resolver; it does
not require connecting to a specific bridge. All bridge-targeted tools share one
connection-request DTO/authority so host/port/token/descriptor arguments do not
become repeated per-tool policy.

## Action Invocation Semantics

Compile/run/init are long-running UI workflows. MCP invocation should not block
until the pipeline completes. It should return:

- `mutation_request_token` or `idempotency_key` supplied by the client for
  retries of accepted mutating actions;
- `bridge_operation_id` for the UI action dispatch;
- action id and target selection snapshot;
- selection revision token used for the action;
- whether the UI accepted the action;
- `workflow_status_surface_ids` to poll;
- `workflow_scope_ids` such as plate ids when available;
- optional runtime execution ids or submitted job ids if immediately available;
- recommended poll interval;
- errors and warnings.

The agent should poll `plate_manager.state` for real workflow status and poll
`openhcs_ui_get_operation_status` only for the bridge dispatch status.

This distinction must be explicit in DTO field names and tool docstrings.
Action providers must only enqueue or submit long-running work while inside the
bridge request path. They must return within the UI bridge timeout and must not
wait for compile/run completion inside `UiThreadDispatcher.call(...)`.

Mutating action requests must not rely only on "whatever is currently selected"
unless the action summary declares that behavior safe. The request should carry
explicit target scope ids or the state-surface revision/selection token the agent
observed.

## Snapshot Policy

Mutating UI operations must integrate with ObjectState snapshots:

- code document apply creates a snapshot before mutation and records the
  resulting snapshot id;
- snapshot restore/time travel/branch switch keep using the existing snapshot
  provider;
- mutating actions that only trigger workflows record an action snapshot when
  they change ObjectState-backed configuration;
- non-configuration actions such as compile/run return action receipts but do
  not create fake config snapshots.

## Safety Policy

- Every mutating action declares side effects.
- Every mutating action supports confirmation requirements.
- Every mutating action has idempotency semantics for retries.
- Every payload is bounded.
- Paths are projected through existing agent path policy where the path leaves
  the UI process boundary.
- UI state surfaces return projected path objects or redacted paths, not raw
  local paths as the default public representation.
- The bridge keeps token auth through the descriptor file.
- Every UI call runs through `UiThreadDispatcher`.
- Shutdown removes descriptor files and does not leave stale bridge entries in
  the ZMQ browser.
- UI bridge service shutdown is terminal for that service instance: close the
  dispatcher, reject new calls, and create a new service/server object for any
  later restart.
- The operation tracker has bounded retention by count and age.
- The PyQt GUI local profile starts the token-bearing localhost bridge by
  default unless the user explicitly disables it.

## Implementation Phases

### Phase 0: Pay Down Support Debt First

Before MCP builds on the window/action support layers, fix the production debt
that would otherwise become public API debt:

- Replace `ScopeWindowRegistry`'s `(regex, Callable[..., Any])` handler tuples
  with nominal window route declarations.
- Replace `WindowFactory.create_window_for_scope(..., object_state: Any)` with
  an explicit request/result carrier.
- Split `WindowManager._deferred_navigate()` into named navigation stages or a
  nominal navigation authority so MCP focus/navigation does not add more
  branches to the same control hub.
- Replace scattered concrete `isinstance(window, ...)` navigation recovery with
  one nominal navigation witness/adaptor.
- Replace `WidgetActionRoute.resolve_callable: Callable[..., object]` and
  `AsyncActionRunner = Callable[..., object]` with a result-carrying action
  route contract.

This phase is not optional. The dry run showed that these files are the support
seams MCP would rely on.

### Phase 0.5: Split Bridge Contracts Before Registry Work

- Add `ui_bridge_contracts.py`.
- Move provider ABCs and provider identity carriers out of `ui_agent_bridge.py`.
- Move PlateManager concrete providers into `ui_bridge_plate_manager.py`.
- Leave `UiAgentBridgeService` as an orchestration service that imports contracts
  and the registry, but does not import concrete widget providers.
- Keep `InProcessUiBridgeGateway` on the PyQt/test side; do not move it into
  `openhcs.agent.services`.
- Use direct module imports for bridge internals, not
  `from openhcs.pyqt_gui.services import ...`, to avoid package-level import
  side effects.

### Phase 1: Registry With No Behavior Change

- Add `ui_bridge_registry.py`.
- Add provider-set registration for existing PlateManager code/state providers.
- Change `UiAgentBridgeService` to accept a registry.
- Keep `plate_manager=` constructor compatibility temporarily for tests and
  current call sites.
- Update `OpenHCSMainWindow` to use the composition root.

### Phase 2: Generic State Surface Document

- Add `UiStateSurfaceDocument`.
- Have `PlateManagerStateSurfaceProvider.read()` return a generic state-surface
  document with a typed PlateManager payload.
- Keep compatibility conversion for existing MCP clients during the transition.
- Add shared catalog/page metadata before adding larger action/window/ObjectState
  catalogs.
- Add path projection/redaction for path fields in state payloads before exposing
  new generic state surfaces.

### Phase 3: Action Catalog and Invocation

- Add action DTOs and transport operations.
- Add `PlateManagerActionProvider`.
- Introduce a nominal action invocation result instead of relying on
  `dispatch_widget_action()` returning `None`.
- Keep `PlateManagerAction` and `ACTION_ROUTES` as the source of action identity,
  but change the dispatch layer so async actions return a UI action receipt.
- Add idempotency/selection-token handling for mutating action requests.
- Wire MCP list/invoke tools.

### Phase 4: Window Projection

- Add `UiWindowProjectionService`.
- Add list/focus window operations and tools.
- Project open ConfigWindow, DualEditorWindow, and embedded widgets by stable
  ids, not raw `QWidget` values.

### Phase 5: ObjectState Projection

- Add ObjectState scope catalog/surface provider.
- Link ObjectState scopes to code documents where safe.
- Expose current branch, head snapshot, and time-travel state.

### Phase 6: Live Smoke and Advisor Audit

- Start the UI with the bridge enabled.
- Confirm the UI bridge appears in the ZMQ browser.
- Confirm the default PyQt GUI configuration starts exactly one UI bridge server
  and removes its descriptor/browser entry on close.
- Use MCP to list live UI bridges before targeting a specific bridge.
- Use MCP to list actions, windows, state surfaces, code documents, ObjectState
  scopes, snapshots, and branches.
- Invoke a selected PlateManager action and confirm the receipt plus
  `plate_manager.state` polling path.
- Send one future/unsupported operation name through the ZMQ bridge and confirm
  the client receives `unsupported_ui_bridge_operation`, not a generic invalid
  request or timeout.
- Run `nominal_refactor_advisor` on production files touched by the
  implementation. Do not run advisor on tests.

## Acceptance Criteria

- The main window no longer hardcodes PlateManager provider registration.
- A blind MCP agent can discover available UI capabilities without prior widget
  names.
- PlateManager remains the first provider but is no longer privileged in the
  bridge core.
- Compile/run/init are invokable through explicit UI action DTOs and return
  action receipts.
- ObjectState snapshots and branch/time-travel are visible through MCP.
- Existing code document and snapshot MCP tools remain compatible.
- Advisor reports no findings on touched production bridge files, or any
  findings are fixed before feature work continues.
- MCP bridge tools do not repeat connection parsing logic; they all share the
  same connection request/descriptor resolution authority.
- Large or potentially growing UI catalogs are bounded and expose page/truncation
  metadata.
- Local paths in UI state payloads are projected through the agent path policy or
  redacted by default.

## Dry-Run Audit

### Code Paths Audited

- `openhcs/pyqt_gui/main.py`
- `openhcs/pyqt_gui/services/ui_agent_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_server.py`
- `openhcs/pyqt_gui/services/ui_thread_dispatch.py`
- `openhcs/pyqt_gui/services/plate_manager_state_projection.py`
- `openhcs/pyqt_gui/widgets/shared/services/widget_action_dispatch.py`
- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/agent/services/ui_bridge_service.py`
- `openhcs/agent/services/ui_bridge_transport.py`
- `openhcs/mcp/server.py`
- `external/pyqt-reactive/src/pyqt_reactive/services/window_manager.py`
- `external/pyqt-reactive/src/pyqt_reactive/services/scope_window_factory.py`
- `external/ObjectState/src/objectstate/object_state_registry.py`

### What Fits Cleanly

- The ZMQ transport/server already has registered operation dispatch. New action
  and window operations can extend `UiBridgeOperationName` and add one
  `UiBridgeRequestOperation` subclass per operation.
- `UiThreadDispatcher` already provides the correct UI-thread boundary. New
  providers should keep using it through `UiAgentBridgeService`.
- `UiObjectStateSnapshotProvider` already centralizes snapshot, branch,
  revision-token, and time-travel projection. ObjectState scope exposure should
  extend this provider or compose it, not add a parallel snapshot service.
- `PlateManagerStateProjectionService` is already a good pattern for bounded,
  DTO-shaped state projection.
- `PlateManagerAction` and `ACTION_ROUTES` already provide nominal action
  identity. MCP should reuse that action set instead of creating a second list.
- The main window already owns the composition context needed for registration:
  embedded widgets, `WindowManager`, ObjectState callbacks, and bridge lifecycle.

### What Does Not Fit Yet

- `UiAgentBridgeService` is still initialized with `plate_manager=...`, which
  hardcodes a specific widget at the bridge core.
- State surface reads are typed to `UiPlateManagerState`, so the transport,
  gateway, agent service, and MCP tool all know a generic state surface is a
  PlateManager payload.
- Existing DTO ids are closed enums with only PlateManager values. That blocks
  window/object-state/action providers from registering without editing the DTO
  enum every time.
- The action dispatcher returns `None`. For MCP, that loses whether the UI
  accepted the action, whether it started async work, which state surface should
  be polled, and which operation id refers to bridge dispatch.
- Compile/run/init are async UI workflows. The current route map can click them,
  but cannot produce an agent-grade receipt.
- `WindowManager` has useful primitives but also concentrates navigation
  branching and `isinstance` recovery. MCP should not widen that control hub.
- `ScopeWindowRegistry` is regex/callable/Any infrastructure with only narrow
  current fanout. If MCP relies on it directly, it would promote a weak internal
  matcher into public infrastructure.
- ObjectState has the right SSOT APIs (`get_all`, `get_by_scope`, token,
  snapshots, branches, time travel), but no bounded public scope catalog yet.
- `openhcs.config_framework` is a compatibility shim over vendored ObjectState.
  New UI bridge modules should import through the existing OpenHCS shim where
  the rest of PyQt code does, but the plan should avoid adding more compatibility
  responsibilities to that shim.

### Advisor Results

Command:

```bash
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/pyqt_gui/services/ui_agent_bridge.py \
  openhcs/pyqt_gui/services/ui_bridge_server.py \
  openhcs/agent/dto/ui_bridge.py \
  openhcs/agent/services/ui_bridge_transport.py \
  openhcs/agent/services/ui_bridge_service.py \
  openhcs/mcp/server.py \
  openhcs/pyqt_gui/main.py \
  openhcs/pyqt_gui/widgets/shared/services/widget_action_dispatch.py \
  openhcs/pyqt_gui/services/plate_manager_state_projection.py \
  openhcs/pyqt_gui/services/ui_thread_dispatch.py \
  external/pyqt-reactive/src/pyqt_reactive/services/window_manager.py \
  external/pyqt-reactive/src/pyqt_reactive/services/scope_window_factory.py
```

Advisor findings that affect the plan:

- `ScopeWindowRegistry` exposes opaque handler/callable/Any boundaries.
- `WindowFactory` is under-amortized matcher infrastructure and exposes opaque
  object-state parameters.
- `WidgetActionRoute` and `AsyncActionRunner` expose opaque object-like return
  contracts.
- `WindowManager._deferred_navigate()` and
  `WindowManager.position_window_near_cursor()` contain branch-heavy staged
  behavior.
- `WindowManager` scatters concrete `isinstance` recovery across navigation
  family types.
- `WindowManager` has unclassified fallback/default sites in navigation and
  positioning.

The plan now makes these cleanup items Phase 0 rather than optional follow-up.

## Gaps Filled After Dry Run

### Gap 1: Generic State Surface Return Type

Problem: The current `UiStateSurfaceProviderABC.read()` returns
`UiPlateManagerState`.

Resolution: Add `UiStateSurfaceDocument` as the bridge/gateway/MCP return type.
The document carries a `payload_schema` string and a bounded JSON payload. Keep
`UiPlateManagerState` as the typed payload during migration, but stop making the
transport return type PlateManager-specific.

Compatibility: Keep `openhcs_ui_get_state_surface` returning the old shape only
for `plate_manager.state` until clients migrate, or add a `compat_payload=True`
argument defaulting to current behavior for one release window.

### Gap 2: Closed Enum Identities

Problem: `UiCodeDocumentId` and `UiStateSurfaceId` currently contain only
PlateManager ids. Adding windows/actions/ObjectState by editing central enums
would recreate hardcoded registration.

Resolution: Use string identity dataclasses for provider registration. Keep the
enums only as constants for built-in provider ids, not as the authoritative set
of legal ids.

### Gap 3: Main-Window Bridge Composition

Problem: The bridge starts with `UiAgentBridgeService(plate_manager=...)`.

Resolution: Add `OpenHCSUiBridgeCompositionRoot` that receives the main window
context and registers provider sets. `UiAgentBridgeService` receives a registry.
The temporary `plate_manager=` constructor can stay only as migration glue and
should be removed after tests and call sites use the registry.

### Gap 4: Result-Losing Action Dispatch

Problem: `dispatch_widget_action()` returns `None`; async runners lose the
operation shape MCP needs.

Resolution: Introduce a nominal action dispatch contract:

```python
class WidgetActionInvokerABC(ABC):
    def invoke(self, request: WidgetActionInvokeRequest) -> WidgetActionInvokeResult:
        ...
```

`WidgetActionInvokeResult` records accepted/rejected, sync/async, bridge
operation id, target scope ids, related state surfaces, and warnings/errors.
The existing button code can ignore the result, while MCP uses it.

### Gap 5: Async Workflow Semantics

Problem: Compile/run/init cannot be represented as synchronous success/failure
of a button click.

Resolution: `UiActionInvokeResult` distinguishes:

- bridge dispatch completed;
- UI action was accepted or rejected;
- workflow was started;
- runtime operation id is known or not yet known;
- state surfaces to poll.

Agents poll `plate_manager.state` for workflow status. The bridge operation
status only reports the dispatch lifecycle.

### Gap 6: WindowManager Debt

Problem: Window projection/focus would build on branch-heavy, concrete-type
navigation code.

Resolution: Phase 0 introduces a nominal navigation witness or adapter layer
before MCP adds window tools. MCP calls `UiWindowProjectionService`, not
`WindowManager` directly. `UiWindowProjectionService` uses only bounded
WindowManager operations and projects DTOs.

### Gap 7: Scope Window Factory Debt

Problem: `ScopeWindowRegistry` is regex/callable/Any based and currently has
limited fanout.

Resolution: Replace it with nominal route declarations before using it for MCP
window discovery. At minimum, add:

- `ScopeWindowRoute`
- `ScopeWindowRequest`
- `ScopeWindowResult`
- `ScopeWindowRouteHandlerABC`

This allows MCP window focus/navigation to refer to route identity instead of
regex implementation details.

### Gap 8: ObjectState Scope Projection

Problem: ObjectState is the UI state SSOT, but the bridge exposes snapshots and
branches, not the registered scope catalog itself.

Resolution: Add `UiObjectStateProjectionService` backed by
`ObjectStateRegistry.get_all()`, `get_token()`, branch/time-travel APIs, and the
existing snapshot provider. It reports scope metadata and links to code
documents; it never returns raw ObjectState or object instances.

### Gap 9: Capability Registry Authority

Problem: Adding MCP tools manually can drift from actual providers.

Resolution: Provider registration writes capability metadata. MCP tool catalogs
and `openhcs://capabilities` read from the same registry-backed agent service.
Each mutating action declares side effects, network/runtime requirements, and
confirmation policy.

### Gap 10: Shutdown and Descriptor Correctness

Problem: Live UI bridge visibility depends on descriptor files and ZMQ browser
scans. Stale descriptors or shutdown races confuse agents.

Resolution: Keep `UiBridgeControlServer` as the descriptor owner. The composition
root must not own descriptor lifecycle. `UiAgentBridgeService.close()` should
close its dispatcher and reject new calls during shutdown. The ZMQ browser should
continue scanning the configured bridge port.

### Gap 11: Import Boundary

Problem: Some code imports `objectstate` directly and some imports through
`openhcs.config_framework.object_state`.

Resolution: New OpenHCS UI bridge modules should follow the local PyQt service
pattern and import through `openhcs.config_framework.object_state` where possible.
Do not add MCP-specific import shims.

### Gap 12: Testing Order

Problem: A broad implementation would touch DTOs, transport, UI thread dispatch,
MCP tools, and GUI services at once.

Resolution: Land phases in order. After each phase:

- run focused production import/compile checks;
- run relevant non-GUI unit tests;
- run advisor on touched production files only;
- do a live UI smoke after action/window/object-state tools exist.

## Second Dry-Run Audit

### Additional Code Paths Audited

- `openhcs/agent/capabilities.py`
- `openhcs/agent/dto/__init__.py`
- `openhcs/agent/dto/common.py`
- `openhcs/pyqt_gui/services/service_adapter.py`
- `openhcs/pyqt_gui/services/async_service_bridge.py`
- `external/ObjectState/src/objectstate/object_state.py`

### Second-Pass Findings

- `UiBridgeResponseEnvelope.payload` is a `JsonObject`, and
  `UiBridgeRequestDispatcher._result_payload()` rejects non-object operation
  results. Any generic UI state/action/window payload must serialize as a JSON
  object wrapper. Raw lists, scalars, or unwrapped arbitrary payloads will fail.
- `UiPlateManagerState` is assumed by DTO exports, server dispatch result
  unions, ZMQ gateway hydration, agent service error builders, MCP tool output,
  and `openhcs/agent/capabilities.py`. The generic state-surface migration needs
  a compatibility seam, not a single broad replacement commit.
- `UiBridgeMutationGate` currently classifies outcomes only for
  `UiCodeDocumentApplyResult` and `UiSnapshotRestoreResult`. Action invocation
  must either generalize the mutation gate result protocol or use a separate
  `UiBridgeActionGate`.
- `PyQtServiceAdapter.execute_async_operation()` submits a future but does not
  return it. MCP action invocation cannot reuse it unchanged because the agent
  needs an accepted/rejected receipt and, where possible, an async operation
  handle.
- `ObjectState` has enough public state to build a scope catalog, but
  `ObjectState.parameters` and `get_current_values()` can contain arbitrary
  Python objects. The ObjectState MCP projection must expose bounded metadata,
  field names, counts, dirty/signature-diff sets, and safe previews only.
- The current `openhcs://capabilities` registry is a static MCP/agent registry.
  It cannot directly be the live UI provider registry because live providers are
  in a separate running UI process and may not exist when the MCP server starts.
  Static capability discovery and live UI provider discovery must be distinct.
- `AgentDtoJsonCodec` can hydrate dataclass request/response DTOs, tuples, lists,
  mappings, and enums, but dynamic payload shapes should be modeled as
  `JsonObject` fields inside wrapper DTOs rather than unions of many possible
  widget-specific payload DTOs.

### Second Advisor Pass

Command:

```bash
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/agent/capabilities.py \
  openhcs/pyqt_gui/services/service_adapter.py \
  openhcs/pyqt_gui/services/async_service_bridge.py \
  openhcs/pyqt_gui/widgets/shared/services/widget_action_dispatch.py \
  openhcs/pyqt_gui/services/ui_agent_bridge.py \
  openhcs/agent/dto/ui_bridge.py \
  openhcs/agent/services/ui_bridge_transport.py \
  openhcs/pyqt_gui/services/ui_bridge_server.py
```

Relevant result:

- `WidgetActionRoute`, `AsyncActionRunner`, and
  `is_widget_action_dispatch_export` still expose opaque object-like action
  boundaries. This confirms Phase 0 must replace the action route contract before
  MCP action invocation is added.

Non-blocking for this MCP plan:

- `PyQtServiceAdapter` has unrelated dialog/helper debt. Do not put those helper
  paths on the MCP action invocation path unless they are cleaned or wrapped by a
  nominal action launcher.
- `ExternalEditorServiceBridge` has an unclassified fallback site unrelated to
  the UI bridge provider registry.

## Gaps Filled After Second Dry Run

### Gap 13: Object-Shaped Bridge Payloads

Problem: The bridge envelope and dispatcher require operation results to
serialize to JSON objects.

Resolution: Every new operation result must be a dataclass/object wrapper:

- `UiStateSurfaceDocument`
- `UiActionCatalog`
- `UiActionInvokeResult`
- `UiWindowCatalog`
- `UiWindowFocusResult`
- `UiObjectStateCatalog`

Do not return raw lists of actions/windows/scopes from any bridge operation.

### Gap 14: State-Surface Migration Order

Problem: `UiPlateManagerState` is embedded across DTO exports, transport,
gateway, service, MCP tools, and capability metadata.

Resolution: Migrate in this order:

1. Add `UiStateSurfaceDocument` and keep `UiPlateManagerState`.
2. Add a new `get_state_surface_document()` path internally.
3. Make `PlateManagerStateSurfaceProvider` produce the document wrapper.
4. Keep `openhcs_ui_get_state_surface` compatible for `plate_manager.state`.
5. Add `openhcs_ui_get_state_surface_document` or a compatibility flag.
6. Move capability output metadata from `UiPlateManagerState` to
   `UiStateSurfaceDocument` after clients have a stable generic path.

### Gap 15: Live Provider Capabilities vs Static MCP Capabilities

Problem: The static capability registry cannot be the live provider registry.

Resolution: Split the concepts:

- `openhcs://capabilities` remains the static MCP/agent tool contract registry.
- The running UI bridge exposes a live provider catalog through a bridge
  operation such as `LIST_UI_CAPABILITIES` or by enriching the existing
  code/state/action/window catalogs.
- Static capabilities describe generic tools like `openhcs_ui_list_actions`.
  Live capabilities describe current provider instances like
  `plate_manager.compile_plate`.

### Gap 16: Async Action Launcher

Problem: The current async launcher drops the submitted future and exposes no
receipt.

Resolution: Introduce a result-carrying launcher before MCP actions:

```python
class UiAsyncActionLauncherABC(ABC):
    def launch(self, request: UiAsyncActionLaunchRequest) -> UiAsyncActionLaunchResult:
        ...
```

The PlateManager button path can continue ignoring the result, but the MCP
provider must receive:

- accepted/rejected;
- launch error if any;
- future/thread/job identity when available;
- related state surfaces to poll;
- initial target scope ids.

### Gap 17: Generalized Bridge Operation Outcomes

Problem: `UiBridgeMutationGate` knows only code-document and snapshot restore
result types.

Resolution: Add a nominal outcome protocol before action mutation support:

```python
class UiBridgeOperationOutcomeCarrier(ABC):
    def bridge_outcome(self) -> str:
        ...
```

Implement it for existing apply/restore results or adapt them through an
authority. New action results then participate in the same operation tracker
without widening `UiBridgeMutationOutcome.from_result()` with more
`isinstance` cases.

### Gap 18: Bounded ObjectState Field Projection

Problem: ObjectState values can be arbitrary Python objects and may include
callables or local paths.

Resolution: ObjectState scope projection must expose:

- scope id;
- object type name and module;
- has delegate flag;
- parameter count;
- sorted field names;
- dirty field names;
- signature-diff field names;
- user-set field names;
- optional safe scalar previews with truncation;
- current token, branch, and snapshot refs.

It must not expose full `parameters`, `object_instance`, `saved_object`, or
resolved values by default.

### Gap 19: DTO Export and Capability Metadata Step

Problem: Adding DTOs without updating `openhcs/agent/dto/__init__.py` and
`openhcs/agent/capabilities.py` leaves MCP tools discoverable but undocumented or
incorrectly typed.

Resolution: Every DTO/operation phase includes:

- add DTOs in `openhcs/agent/dto/ui_bridge.py`;
- export them from `openhcs/agent/dto/__init__.py`;
- add/update static capability specs in `openhcs/agent/capabilities.py`;
- add static side-effect and data-exposure declarations for mutating/read-local
  tools;
- verify `validate_capability_registry()` still passes.

## Third Dry-Run Audit

### Additional Code Paths Audited

- `openhcs/mcp/context.py`
- `openhcs/agent/services/__init__.py`
- `openhcs/pyqt_gui/services/__init__.py`
- `openhcs/pyqt_gui/services/main_window_workflows.py`
- `openhcs/pyqt_gui/config.py`
- `openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py`
- `openhcs/pyqt_gui/services/ui_agent_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_server.py`

### Import Boundary Check

Command:

```bash
.venv/bin/python - <<'PY'
import sys
import openhcs.mcp.context
pyqt_modules = sorted(
    name
    for name in sys.modules
    if name.startswith('PyQt6') or name.startswith('openhcs.pyqt_gui')
)
print('mcp context import ok')
print('pyqt module count', len(pyqt_modules))
PY
```

Result:

```text
mcp context import ok
pyqt module count 0
```

This is a hard invariant for the implementation: importing the MCP context must
not import PyQt or `openhcs.pyqt_gui`.

### Third-Pass Findings

- `ui_agent_bridge.py` currently contains provider ABCs, concrete PlateManager
  providers, the bridge service, mutation/snapshot helpers, and the in-process
  gateway. If `ui_bridge_registry.py` imports provider ABCs from
  `ui_agent_bridge.py` while `UiAgentBridgeService` imports the registry, the
  implementation will create an avoidable cycle.
- Concrete PlateManager providers currently live next to the generic bridge
  service. A generic registry should not keep PlateManager as a privileged import
  inside the bridge core.
- `openhcs.pyqt_gui.services.__init__` imports `PyQtServiceAdapter`,
  `AsyncServiceBridge`, and `UiAgentBridgeService`. Bridge internals should avoid
  importing through that package barrel because it widens import side effects and
  makes circular imports harder to diagnose.
- `UiBridgeStatus` does not currently expose supported operation names or live
  provider-catalog schema versions. Adding new bridge operations without feature
  discovery means a new MCP client connected to an old UI bridge would only learn
  about unsupported operations after a request fails.
- The descriptor protocol version is strict (`openhcs.ui_bridge.v1`), but the
  status payload has no feature list. The plan needs explicit feature
  negotiation rather than relying on protocol-version equality alone.
- Current PyQt UI bridge tests instantiate `UiAgentBridgeService(plate_manager=...)`.
  The registry migration needs a compatibility constructor or a staged test
  harness update.

## Gaps Filled After Third Dry Run

### Gap 20: Cycle-Free Provider Contracts

Problem: If provider ABCs stay in `ui_agent_bridge.py`, the new registry and the
bridge service will naturally import each other.

Resolution: Add `ui_bridge_contracts.py` as the root for provider ABCs,
provider identities, provider set contracts, and small bridge-side request/result
protocols. `ui_bridge_registry.py`, concrete providers, and
`UiAgentBridgeService` all import contracts from that module. Contracts do not
import services, widgets, or the registry.

### Gap 21: PlateManager Provider Extraction

Problem: `UiAgentBridgeService` currently creates PlateManager providers itself,
which keeps PlateManager privileged in the generic bridge core.

Resolution: Move concrete PlateManager bridge providers into
`ui_bridge_plate_manager.py`. The composition root registers the PlateManager
provider set. `UiAgentBridgeService` receives a registry and never imports
PlateManager-specific providers.

### Gap 22: Headless Import Guard

Problem: It would be easy for new shared DTO/registry code to accidentally make
`openhcs.mcp` or `openhcs.agent` import PyQt.

Resolution: Add a focused import-boundary test or smoke command to the
implementation checklist:

```python
import sys
import openhcs.mcp.context
assert not any(
    name.startswith("PyQt6") or name.startswith("openhcs.pyqt_gui")
    for name in sys.modules
)
```

Only `openhcs.pyqt_gui` may import PyQt bridge provider implementations.

### Gap 23: Protocol Feature Negotiation

Problem: New MCP tools can connect to an older running UI bridge that speaks the
same strict protocol version but does not support new operations.

Resolution: Extend `UiBridgeStatus` with object-shaped capability fields:

- `supported_operations: tuple[str, ...]`
- `provider_catalog_schema_versions: tuple[str, ...]`
- `bridge_features: tuple[str, ...]`

Add `LIST_UI_CAPABILITIES` as the live bridge catalog operation. New MCP tools
should either call status first or convert unsupported operation failures into a
typed `unsupported_ui_bridge_operation` result with a hint to restart/update the
UI.

### Gap 24: Package Barrel Import Discipline

Problem: `openhcs.pyqt_gui.services.__init__` imports several concrete services,
including bridge and adapter classes. Importing through it can hide side effects.

Resolution: New bridge internals must import direct modules:

- use `from openhcs.pyqt_gui.services.ui_bridge_contracts import ...`;
- use `from openhcs.pyqt_gui.services.ui_bridge_registry import ...`;
- use `from openhcs.pyqt_gui.services.ui_agent_bridge import UiAgentBridgeService`;

Do not add the new registry/composition/provider modules to
`openhcs.pyqt_gui.services.__init__` until their import behavior is explicitly
tested.

### Gap 25: Test Harness Migration

Problem: Existing PyQt bridge tests build `UiAgentBridgeService` with
`plate_manager=FakePlateManager()`.

Resolution: Keep `plate_manager=` as a temporary compatibility constructor
during Phase 1, but implement it by building a registry with the PlateManager
provider set. Add direct registry-based test fixtures before removing
`plate_manager=`.

### Gap 26: In-Process Gateway Placement

Problem: The in-process gateway adapts a PyQt `UiAgentBridgeService`, so moving
it into `openhcs.agent.services` would violate the headless boundary.

Resolution: Keep `InProcessUiBridgeGateway` in PyQt-side bridge code or move it
to a PyQt test/support module. Agent services should know only the abstract
`UiBridgeGatewayABC` and the ZMQ gateway.

## Fourth Dry-Run Audit

### Additional Code Paths Audited

- `openhcs/pyqt_gui/services/ui_bridge_server.py`
- `openhcs/agent/services/ui_bridge_transport.py`
- `openhcs/agent/services/ui_bridge_service.py`
- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/pyqt_gui/services/ui_agent_bridge.py`
- `openhcs/pyqt_gui/services/ui_thread_dispatch.py`
- `openhcs/pyqt_gui/services/main_window_workflows.py`
- `openhcs/pyqt_gui/config.py`

### Fourth-Pass Findings

- `UiBridgeRequestDispatcher._operation_result()` currently parses with
  `UiBridgeOperationName(request.operation)`. A new MCP client connected to an
  older UI bridge would see that as `ValueError -> invalid_ui_bridge_request`,
  not as an unsupported-operation compatibility response.
- `UiBridgeOperationRequestPayload` and `UiBridgeOperationDispatchResult` are
  manual unions. New operations can compile at the enum/server layer while still
  missing the typed payload/result spine unless this is an explicit checklist
  item.
- `AgentDtoJsonCodec.dataclass_from_json()` tolerates missing fields only when
  dataclass defaults exist. New status/capability fields must therefore default
  to empty tuples or optional values for backward-compatible hydration.
- `MainWindowUiBridgeLifecycle.close()` centralizes shutdown through
  `UiBridgeControlServer.stop()`, but `UiAgentBridgeService` currently has no
  explicit `close()` method. Descriptor removal alone is not enough for a
  terminal bridge lifecycle.
- The ZMQ browser sees the UI bridge through a separate control socket and a
  small pong payload. Capability discovery should stay on the bridge API unless
  the browser explicitly opts into a new field.
- `UiBridgeOperationTracker` stores operation refs without a retention policy.
  Compile/run action receipts would make that a growing runtime map.
- `UiThreadDispatcher.call(...)` and the ZMQ client both default to five-second
  request timeouts. Compile/run actions must return launch receipts quickly and
  move long-running status to state-surface polling.
- `AgentUiBridgeConfig.enabled` currently defaults false. That keeps the normal
  GUI from being discoverable by a blind local MCP agent unless an environment
  variable or config override is present.

## Gaps Filled After Fourth Dry Run

### Gap 27: Unsupported Operation Error Contract

Problem: Unknown operation names currently become `ValueError` and are classified
as `invalid_ui_bridge_request`.

Resolution: Add a nominal operation parsing authority such as
`UiBridgeOperationAuthority.parse(...)` and a specific
`UiBridgeUnsupportedOperationError`. The dispatcher must classify this as
`unsupported_ui_bridge_operation`. A missing registered handler for a valid enum
member must use the same error class. Invalid envelopes, invalid protocol
versions, and unsupported operations remain distinct error codes.

### Gap 28: Operation Spine Exhaustiveness

Problem: Adding a bridge operation requires edits across enum, request payload
union, dispatch result union, server operation, gateway ABC, concrete gateways,
MCP wrapper, and capability metadata.

Resolution: Make the operation spine an implementation checklist and add a small
assertion/smoke check that registered server operations and
`UiBridgeStatus.supported_operations` match the intended public operation set.
No operation phase is complete until the request/result unions and all gateways
are updated.

### Gap 29: Backward-Compatible Capability Defaults

Problem: New `UiBridgeStatus` capability fields would break hydration from older
bridges if they are required dataclass fields.

Resolution: All newly added status/capability fields must have safe defaults:
empty tuples for lists/catalog hints, optional `None` for unknown scalar values,
and object-shaped nested DTOs only when their own defaults are also safe. MCP
clients should treat missing/empty capability fields as "not supported" and then
return a typed unsupported-operation response.

### Gap 30: Terminal Bridge Lifecycle

Problem: The descriptor owner is the server, but the dispatch service also owns
runtime resources such as `UiThreadDispatcher`. Stopping only the ZMQ sockets can
leave dispatch resources open.

Resolution: Add an idempotent `UiAgentBridgeService.close()` that closes the
dispatcher and rejects new calls. `MainWindowUiBridgeLifecycle.close()` must use
a terminal server/service close path. A stopped bridge service is not restarted;
the composition root creates a fresh service/server instance for any later UI
bridge restart.

### Gap 31: ZMQ Browser Payload Stability

Problem: The UI bridge appears in the existing ZMQ server browser through a
separate browser-control socket. Mixing MCP capability catalogs into that pong
payload would couple browser discovery to agent API evolution.

Resolution: Keep the browser pong small and stable: process/server identity,
port/control port, readiness, log path, schema/protocol version, and bridge
instance id. Live UI capabilities are exposed through `STATUS` and
`LIST_UI_CAPABILITIES`. Any future browser metadata must be optional and
backward-compatible.

### Gap 32: Workflow Status Surface Split

Problem: `get_operation_status` reports bridge dispatch status, not compile/run
workflow completion.

Resolution: Action invocation results use explicit names:
`bridge_operation_id`, `workflow_status_surface_ids`, `workflow_scope_ids`,
`runtime_execution_ids`, `submitted_job_ids`, and `poll_after_ms`. MCP docs must
tell agents to poll `plate_manager.state` or another returned state surface for
workflow progress, and to use `get_operation_status` only for the bridge dispatch
receipt.

### Gap 33: Operation Tracker Retention

Problem: `UiBridgeOperationTracker` is currently an unbounded dict.

Resolution: Add bounded retention by maximum operation count and completed-age
TTL. Prune on `start()` and opportunistically on `get()`. Unknown or pruned
operations continue returning the existing `not_found` operation ref shape.

### Gap 34: Request Timeout Discipline

Problem: UI bridge request dispatch has short timeouts. If an action provider
waits for compile/run completion inside the request path, the MCP client will see
timeouts even though the UI may be working.

Resolution: All long-running actions must submit/enqueue work and return a
receipt inside the bridge timeout. If submission cannot be completed quickly,
return an action-level launch error with `accepted=False`; do not keep the ZMQ
request open while waiting for workflow completion.

### Gap 35: Local UI Bridge Default Enablement

Problem: A blind local MCP agent cannot discover the running UI if the GUI starts
with `AgentUiBridgeConfig.enabled=False` and no environment override.

Resolution: Make the PyQt GUI local profile enable the token-bearing localhost
UI bridge by default, while preserving an explicit disable switch. Keep token
auth in the descriptor, bind to localhost by default, and keep mutation
confirmation policy enabled. The live-smoke checklist must verify exactly one UI
bridge appears in the ZMQ server browser during normal GUI startup.

## Fifth Dry-Run Audit

### Additional Code Paths Audited

- `openhcs/mcp/server.py`
- `openhcs/mcp/context.py`
- `openhcs/agent/capabilities.py`
- `openhcs/agent/path_policy.py`
- `openhcs/agent/services/ui_bridge_service.py`
- `openhcs/agent/dto/ui_bridge.py`
- `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`
- `docs/plans/openhcs_mcp_server_plan_20260616.md`

### Advisor Result

Command:

```bash
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/mcp/server.py \
  openhcs/mcp/context.py \
  openhcs/agent/capabilities.py \
  openhcs/agent/path_policy.py \
  openhcs/agent/services/ui_bridge_service.py \
  openhcs/agent/dto/ui_bridge.py
```

Result: no refactoring findings.

### Fifth-Pass Findings

- The MCP server already repeats UI bridge connection fields across every bridge
  tool. The generic UI tools should not multiply that boilerplate; connection
  resolution must stay centralized.
- `UiBridgeDescriptorResolver` already handles none/one/many live bridge
  descriptors and returns an `ambiguous_ui_bridge` result. A blind agent still
  needs an explicit `list_bridges` tool/resource before choosing a target.
- Current UI state DTOs expose path-like fields such as `plate_root` and
  `cppipe_path` directly. The older MCP exposition plan requires server-side path
  policy and opaque/projected path handling.
- Existing catalog DTOs do not share a page/truncation envelope. Function catalog
  has a bounded page pattern, but UI catalogs for windows/actions/ObjectState can
  grow and should not invent separate ad hoc limit fields.
- Confirmation is represented as a boolean policy carrier today. For MCP retry
  behavior, mutating actions also need idempotency/request tokens and explicit
  selection revision data.
- Static `openhcs://capabilities` is currently present as a resource and tool,
  but the UI plan should keep static MCP capability metadata separate from live
  UI provider/action metadata.
- The current MCP implementation has resources but no prompts. The older MCP
  exposition plan included reviewer/triage prompts. UI bridge prompts are not
  required for the registry refactor, but the capability registry should not
  prevent prompts from being added later.

## Gaps Filled After Fifth Dry Run

### Gap 36: Explicit Live Bridge Discovery

Problem: `openhcs_ui_bridge_status` can surface ambiguous descriptors, but a
blind agent needs a direct discovery call before it targets a bridge.

Resolution: Add `openhcs_ui_list_bridges` backed by the existing
`UiBridgeDescriptorResolver.live_descriptors()` and descriptor summaries. It
returns `UiBridgeCatalog` with bridge instance id, descriptor path, PID, start
time, public connection fields, descriptor status, and errors for stale
descriptors. Bridge-targeted tools keep accepting `bridge_instance_id` or
`descriptor_file_path`.

### Gap 37: Shared MCP Bridge Connection Request

Problem: Every current UI bridge MCP tool repeats host/port/transport/token/
descriptor arguments and manually constructs `UiBridgeConnectionToolArgs`.

Resolution: Introduce a single MCP-facing connection request DTO or helper
authority used by all UI bridge tools, including future action/window/ObjectState
tools. Descriptor resolution remains in `UiBridgeService`; MCP handlers only
project tool arguments into that request carrier.

### Gap 38: Catalog Paging and Truncation Envelope

Problem: Future action, window, ObjectState, snapshot, and state-surface catalogs
can grow. Without a shared page envelope, each provider will invent limits
differently.

Resolution: Add `UiPageRequest` and `UiCatalogPageMetadata` with limit, cursor or
offset, returned count, total count when cheap, `truncated`, and `next_cursor`.
Every potentially growing UI catalog includes the metadata. Defaults must be
bounded and deterministic.

### Gap 39: Path Projection for UI State

Problem: Existing state rows expose raw local paths. That is useful locally but
violates the agent boundary when paths become public MCP payloads.

Resolution: Add `UiPathProjection` and a path projection authority under
`openhcs.agent` that uses `AgentPathPolicy`. State surfaces return projected path
objects with `path_id`, display label, readability/writability flags, and optional
raw path only when policy allows it. Existing compatibility payloads can retain
raw path fields for one transition window, but new generic state surfaces default
to projected/redacted paths.

### Gap 40: Action Idempotency and Selection Tokens

Problem: MCP tools can be retried by clients or users. A mutating action that
targets "current selection" can run twice or run against a changed selection.

Resolution: Mutating `UiActionInvokeRequest` includes a
`mutation_request_token`/idempotency key and either explicit target scope ids or
the state-surface selection revision token observed by the agent. Providers keep
a short-lived accepted-token cache and return the original receipt for duplicate
accepted requests.

### Gap 41: Live Action Metadata Must Satisfy Static Policy Shape

Problem: Static MCP capability specs declare side effects and security
requirements, but live UI actions will have their own provider-level side
effects.

Resolution: Define a shared side-effect/security vocabulary used by both
`AgentCapabilitySpec` and `UiActionSummary`. Live action catalogs must include
side effects, confirmation requirements, runtime requirements, data exposure, and
related state surfaces. MCP invocation refuses mutating actions whose live summary
does not satisfy this metadata shape.

### Gap 42: Prompt Surface Compatibility

Problem: The earlier MCP exposition plan includes prompt-level workflows, but the
UI registry plan only discusses tools/resources.

Resolution: Keep prompt support out of the initial UI registry implementation,
but reserve capability-registry support for `CapabilityKind.PROMPT` and add a
future UI prompt set after the tool/resource contract is stable:
UI state triage, CP conversion triage from an open UI, and bug-report context
from live UI state. Prompts must call discovery tools first and must not depend
on raw widget internals.
