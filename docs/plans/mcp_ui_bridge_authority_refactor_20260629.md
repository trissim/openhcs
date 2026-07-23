# MCP UI Bridge Authority Refactor

Date: 2026-06-29

## Problem

The UI bridge layer now exposes useful agent operations, but several semantic
families are mirrored outside their existing owners:

- `UiCodeDocumentId`, `UiStateSurfaceId`, `UiWidgetId`, and
  `UiSelectedPlateWorkflowKind` in `openhcs/agent/dto/ui_bridge.py` duplicate
  provider/action identities.
- `UiObjectStateFieldFilter.matches()` embeds ObjectState field semantics in a
  DTO enum.
- `UiBridgeOperationRequestPayload` in
  `openhcs/agent/services/ui_bridge_transport.py` manually lists every request
  DTO.
- `UiBridgeRequestOperation` subclasses in
  `openhcs/pyqt_gui/services/ui_bridge_server.py` manually repeat operation
  name, request DTO, and bridge method.
- `UiCodeDocumentSourcePolicy` in `openhcs/pyqt_gui/services/ui_agent_bridge.py`
  duplicates the plate-manager code document namespace.
- `PLATE_MANAGER_ACTION_SIDE_EFFECTS`,
  `PLATE_MANAGER_ACTION_OPERATIONS`, and
  `PLATE_MANAGER_CONFIRMED_ACTIONS` duplicate `PlateManagerAction` semantics.
- `ManagedWindowAction`, `MANAGED_WINDOW_ACTION_SPECS`, and helper functions in
  `ui_bridge_windows.py` duplicate managed-window capability semantics.

## Existing Authorities

- `UiBridgeProviderIdentity` classes in `ui_bridge_contracts.py` already define
  provider identities.
- `UiBridgeProviderSetABC` and `AutoRegisterMeta` in `ui_bridge_registry.py`
  already provide provider discovery.
- `PlateManagerCodeNamespaceField`,
  `PlateManagerCodeNamespace`, and
  `PlateManagerOrchestratorCodePayload` in
  `widgets/shared/services/plate_manager_workflows.py` own code-document
  namespace and payload semantics.
- `PipelineEditorAction` already shows the right pattern: the action enum owns
  side effects, confirmation requirement, and target mode.
- `PlateManagerAction` is the right owner for plate-manager action semantics,
  but it currently only owns values.
- `ManagedWindowActionCapabilities` and `BaseFormDialog`/managed-window methods
  own managed-window action availability and dispatch behavior.
- `UiBridgeRequestOperation` already uses `AutoRegisterMeta`; the missing piece
  is a typed operation contract consumed by client and server.

## Target Shape

### UI bridge operation contracts

Create a single operation contract family, for example
`UiBridgeOperationContract`, in the agent UI bridge service/DTO layer. It must
be importable without PyQt.

Each operation contract owns:

- final operation ABI name;
- request DTO type, or `None`;
- response DTO type;
- `requires_auth`;
- mutating flag and side-effect labels if applicable;
- status feature tag if needed.

Client and server consume the same contract:

- `ZMQUiBridgeGateway` builds the wire envelope from the contract and a typed
  request DTO.
- `UiBridgeRequestDispatcher` hydrates request DTOs from the contract.
- PyQt server execution calls the bridge service through the contract binding
  or through a thin registered executor bound to the contract.
- `status_result()` derives supported operation names and feature tags from the
  operation contract registry.

The current one-class-per-operation server handlers can remain only if each
class inherits the shared contract and adds real behavior. Trivial handlers
should collapse into a generic executor once the contract has request and
response types.

### Code document source policy

Move code-document source validation back to the code-document authority:

- Expected assignment names come from `PlateManagerCodeNamespaceField`.
- Required payload shape comes from `PlateManagerOrchestratorCodePayload`.
- Removed fields come from `RemovedPlateManagerCodeNamespaceField`.
- Approved step/function constructors come from
  `FunctionStepTransportAuthority` or the registered function transport layer,
  not from `UiCodeDocumentSourcePolicy.approved_factory_names`.
- Import-root policy can stay as a security policy, but it should be named as a
  security policy and not own code-document semantics.

After the refactor, the validator asks the namespace/payload authorities whether
an assignment is permitted and whether the final namespace is complete.

### PlateManager action semantics

Refactor `PlateManagerAction` to match `PipelineEditorAction`.

Add declaration-owned fields or methods:

- `side_effects`
- `confirmation_required`
- `target_mode` or selection requirement
- optional `plate_operation` for init/compile/run actions
- `disabled_hint` or hint policy if the hint is semantic rather than display
  copy

Then delete:

- `PLATE_MANAGER_CONFIRMED_ACTIONS`
- `PLATE_MANAGER_ACTION_SIDE_EFFECTS`
- `PLATE_MANAGER_ACTION_OPERATIONS`

`PlateManagerActionProvider` should query the action declaration.

### Managed-window action semantics

Do not leave managed-window action semantics in `ui_bridge_windows.py`.

Move the action declaration to the managed-window owner:

- preferred: a pyqt-reactive declaration beside
  `ManagedWindowActionCapabilities` and `BaseFormDialog` methods;
- acceptable if external change is not possible in the same batch: an OpenHCS
  declaration subclassing or wrapping the managed-window capability type, with
  no duplicate availability rules.

The declaration owns:

- action value;
- title;
- side effects;
- required capability flag;
- dispatch method.

`ui_bridge_windows.py` should only project declared actions for a window.

### ObjectState field filters

Move `UiObjectStateFieldFilter.matches()` out of the DTO enum.

The DTO enum can remain an API vocabulary, but the predicate must live with the
ObjectState field projection authority, likely in
`ui_bridge_object_state.py` beside `ObjectStateFieldSemanticProjection` or a
small `ObjectStateFieldFilterPolicy` there.

The field projection provider should apply the predicate before constructing the
DTO page. MCP and dev-client code should not reimplement these tests.

## Deterministic Steps

1. Add operation contract registry and tests.
   - Define contract base with `AutoRegisterMeta`.
   - Move operation name/request/response/auth metadata out of transport/server
     tables.
   - Add test that client and server registries expose the same names.

2. Replace `UiBridgeOperationRequestPayload`.
   - The transport layer accepts `UiBridgeOperationContract` plus optional
     dataclass request payload.
   - Delete the union.
   - The serializer validates payload type against the contract.

3. Collapse server operation handlers.
   - For each current subclass of `UiBridgeRequestOperation`, move request DTO
     and response DTO into the shared contract.
   - Keep custom executor only where execution is not the default
     `bridge.<operation>(payload)`.
   - Remove repeated `operation_name = UiAgentBridgeService.method.__name__`
     from leaf classes once the contract owns it.

4. Refactor code-document source policy.
   - Replace `expected_assignments` with
     `PlateManagerCodeNamespaceField.allowed_assignment_names()`.
   - Replace `approved_factory_names` with transport-authority query.
   - Add tests for all namespace fields and removed fields.

5. Move PlateManager action semantics.
   - Extend `PlateManagerAction` in `widgets/plate_manager.py` using the same
     enum construction pattern as `PipelineEditorAction`.
   - Update `PlateManagerActionProvider` to query the action.
   - Delete side-effect/operation/confirmation maps from
     `ui_bridge_plate_manager.py`.

6. Move managed-window action semantics.
   - Add or reuse a managed-window action declaration beside the window
     capability owner.
   - Update `ManagedWindowActionProvider` to query declarations.
   - Delete local specs and support/dispatch helper functions from
     `ui_bridge_windows.py`.

7. Move ObjectState filter predicates.
   - Keep API vocabulary if needed.
   - Move matching logic into ObjectState field projection/provider.
   - Add a test that DTO enum has no `.matches()` method and all filtering
     happens through the provider.

## AST Removal Gates

```bash
rg -n "UiBridgeOperationRequestPayload" openhcs/agent openhcs/pyqt_gui
rg -n "operation_name = UiAgentBridgeService\\.|request_payload\\(Ui[A-Za-z]+Request" openhcs/pyqt_gui/services/ui_bridge_server.py
rg -n "expected_assignments|approved_factory_names" openhcs/pyqt_gui/services/ui_agent_bridge.py
rg -n "PLATE_MANAGER_CONFIRMED_ACTIONS|PLATE_MANAGER_ACTION_SIDE_EFFECTS|PLATE_MANAGER_ACTION_OPERATIONS" openhcs/pyqt_gui/services/ui_bridge_plate_manager.py
rg -n "MANAGED_WINDOW_ACTION_SPECS|_supports_save|_dispatch_save|_dispatch_discard" openhcs/pyqt_gui/services/ui_bridge_windows.py
rg -n "def matches\\(self, field:.*UiObjectStateFieldSummary" openhcs/agent/dto/ui_bridge.py
```

Expected result: no matches.

## Tests

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_config_window_state_resolution.py \
  tests/unit/pyqt_gui/test_object_state_callable_preview.py \
  tests/unit/agent/test_mcp_server.py
```

## Implementation Progress

2026-06-29 partial implementation:

- `PlateManagerAction` now owns action `side_effects`,
  `confirmation_required`, and optional `plate_operation` semantics, matching
  the existing `PipelineEditorAction` pattern.
- `PlateManagerActionProvider` now queries the action declaration instead of
  provider-local side-effect, confirmation, and operation maps.
- `PLATE_MANAGER_CONFIRMED_ACTIONS`, `PLATE_MANAGER_ACTION_SIDE_EFFECTS`, and
  `PLATE_MANAGER_ACTION_OPERATIONS` were removed from
  `ui_bridge_plate_manager.py`.
- `UiObjectStateFieldFilter.matches()` was removed from the DTO enum. The enum
  is now API vocabulary only; filtering is owned by
  `ObjectStateFieldFilterDeclaration` in the headless
  `object_state_field_projection` service and is reused by both the PyQt
  ObjectState provider and MCP wrapper.
- Generic managed-window action semantics moved to
  `pyqt_reactive.widgets.shared.ManagedWindowAction`, beside
  `ManagedWindowActionCapabilities` and `BaseManagedWindow` dispatch methods.
  `ui_bridge_windows.py` now projects that declaration instead of owning
  support/dispatch helper tables.
- Code-document source validation now asks the namespace and transport owners:
  `PlateManagerCodeNamespaceField.allowed_assignment_names()` owns permitted
  assignment names, and
  `FunctionStepTransportAuthority.approved_code_document_factory_names()` owns
  approved helper factory names. `UiCodeDocumentSourcePolicy` keeps only import
  root security policy.
- A shared `UiBridgeOperationContract` declaration family now lives in the
  headless UI bridge service layer. Transport and server dispatch use the same
  contract list for operation names, request DTO types, response DTO types, and
  auth policy. `UiBridgeOperationRequestPayload` was removed, transport
  validates payload type through the contract, and server request hydration uses
  contract-declared DTO types.

Verified gate:

```bash
rg -n "PLATE_MANAGER_CONFIRMED_ACTIONS|PLATE_MANAGER_ACTION_SIDE_EFFECTS|PLATE_MANAGER_ACTION_OPERATIONS" openhcs/pyqt_gui/services/ui_bridge_plate_manager.py
rg -n "def matches\(self, field:.*UiObjectStateFieldSummary" openhcs/agent/dto/ui_bridge.py
rg -n "MANAGED_WINDOW_ACTION_SPECS|_supports_save|_dispatch_save|_dispatch_discard" openhcs/pyqt_gui/services/ui_bridge_windows.py
rg -n "expected_assignments|approved_factory_names" openhcs/pyqt_gui/services/ui_agent_bridge.py
rg -n "UiBridgeOperationRequestPayload|operation_name = UiAgentBridgeService\.|request_payload\(Ui[A-Za-z]+Request" openhcs/agent openhcs/pyqt_gui/services/ui_bridge_server.py
```

These commands currently return no matches.

Verified tests:

```bash
QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  -k "plate_manager_action or selected_workflow or actions_command"
```

This selected 3 tests and all passed. A broader PyQt action selection without
`QT_QPA_PLATFORM=offscreen` aborted in a live widget action test; the enum
contract and provider-map removal were verified independently.

Additional verified tests:

```bash
QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -k "object_state"

XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k "object_state"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -k "managed_window or window"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k "window or widget"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -k "code_document"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k "code_document"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/agent/test_mcp_server.py \
  -k "(ui_bridge or code_document or object_state or window or widget or action) and not descriptor_resolution_uses_process_advertised_descriptor"

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  -k "(bridge or code_document or object_state or managed_window or window or action) and not control_server"
```

These selected 13, 19, 11, 12, 3, 10, 63, and 50 tests respectively; all
passed. The excluded tests are environment-bound in this sandbox: one tries to
unlink stale descriptors under `/run/user`, and two construct an IPC bridge
server under `~/.openhcs`.

2026-06-29 widget identity provider-id follow-up:

- `UiWidgetIdentityDeclaration` now owns the stable action-provider id
  convention through `action_provider_id()`.
- `UiActionProviderIdentity.from_widget_declaration()` defaults the action
  provider id from that widget declaration. PlateManager, PipelineEditor, and
  managed-window action providers no longer redeclare `*.actions` strings.
- `ManagedWindowWidgetIdentity` was added to the UI widget identity declaration
  family, so the managed-window action provider uses the same nominal authority
  as PlateManager and PipelineEditor.
- PlateManager and PipelineEditor provider-set registry keys now derive from
  their widget identity declarations instead of repeating widget id strings.
- User-facing hints that mention `plate_manager.state` now use the
  `PlateManagerStateSurfaceIdentityDeclaration` value.

Verified AST gate:

```bash
python - <<'PY'
import ast
from pathlib import Path

files = [
    "openhcs/pyqt_gui/services/ui_bridge_plate_manager.py",
    "openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py",
    "openhcs/pyqt_gui/services/ui_bridge_windows.py",
]
for file in files:
    tree = ast.parse(Path(file).read_text())
    print("##", file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [ast.unparse(target) for target in node.targets]
            value = ast.unparse(node.value)
            if "registry_key" in targets or any(
                "ACTION_PROVIDER_IDENTITY" in target for target in targets
            ):
                print(" ", node.lineno, targets, "=", value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (
                node.value in {"plate_manager", "pipeline_editor", "managed_window"}
                or node.value.endswith(".actions")
            ):
                print("  string", node.lineno, repr(node.value))
PY
# openhcs/pyqt_gui/services/ui_bridge_plate_manager.py:
#   action provider identity from UiActionProviderIdentity.from_widget_declaration(...)
#   registry_key = PlateManagerWidgetIdentity.require_value()
# openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py:
#   action provider identity from UiActionProviderIdentity.from_widget_declaration(...)
#   registry_key = PIPELINE_EDITOR_WIDGET_ID
# openhcs/pyqt_gui/services/ui_bridge_windows.py:
#   action provider identity from UiActionProviderIdentity.from_widget_declaration(...)
#   registry_key = MAIN_WINDOW_PROVIDER_ID
# no hardcoded widget id strings or *.actions strings are emitted by the gate
```

Verified tests:

```bash
. .venv/bin/activate
pytest tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_mcp_server.py -q
# 304 passed
```

2026-06-29 ObjectState field filter authority correction:

- Current checkout still had `UiObjectStateFieldFilter.matches()` and consumer
  calls to `field_filter.matches(...)`; that contradicted the intended boundary
  above.
- Added `ObjectStateFieldFilterDeclaration`, an AutoRegisterMeta-backed nominal
  declaration family where each filter value owns its predicate.
- Added `ObjectStateFieldListProjector` so field-list result projection and
  value compaction live beside the filter policy instead of on the DTO query.
- Updated `UiBridgeService`, the PyQt ObjectState provider, and MCP test
  doubles to consume the projector/policy.

Verified AST gate:

```bash
rg -n "def matches\(self, field:.*UiObjectStateFieldSummary|field_filter\.matches|UiObjectStateFieldFilter\.matches|query\.project_catalog" \
  openhcs/agent/dto/ui_bridge.py openhcs/agent/services openhcs/pyqt_gui/services tests/unit -g '*.py'
# no matches
```

Verified focused tests:

```bash
. .venv/bin/activate
pytest tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  -k "object_state or field_filter or get_object_state_fields" -q
# 34 passed, 270 deselected
```

2026-06-29 gateway-method authority follow-up:

- UI bridge operation leaves no longer declare
  `name = UiBridgeGatewayABC.<method>.__name__`.
- Added nominal `UiBridgeGatewayMethod` references. The operation registry key
  derives from the declared gateway method object through
  `__key_extractor__`, so the operation name is not copied into each leaf.
- No-payload and payload operation bases invoke through typed
  `UiBridgeNoPayloadGatewayMethod` / `UiBridgePayloadGatewayMethod` wrappers.
  The wrapper carries the method identity for registry/name semantics and the
  typed dynamic-dispatch callable required to call the concrete gateway.
- The PyQt server remains generic: `UiBridgeRequestOperation` looks up the
  contract by registry key and executes it mechanically.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/agent/services/ui_bridge_service.py").read_text())
name_mirrors = []
gateway_call_assignments = []
direct_abstract_methods = []
wrapped_methods = []
for node in tree.body:
    if not (
        isinstance(node, ast.ClassDef)
        and node.name.startswith("UiBridge")
        and node.name.endswith("Operation")
    ):
        continue
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "name"
                    and "UiBridgeGatewayABC" in ast.unparse(stmt.value)
                ):
                    name_mirrors.append((node.name, stmt.lineno))
                if isinstance(target, ast.Name) and target.id == "gateway_call":
                    gateway_call_assignments.append((node.name, stmt.lineno))
                if isinstance(target, ast.Name) and target.id == "gateway_method":
                    value = ast.unparse(stmt.value)
                    if value.startswith("UiBridgeGatewayABC."):
                        direct_abstract_methods.append((node.name, stmt.lineno))
                    if value.startswith(
                        (
                            "UiBridgeNoPayloadGatewayMethod(",
                            "UiBridgePayloadGatewayMethod(",
                        )
                    ):
                        wrapped_methods.append((node.name, stmt.lineno))
print("name_mirrors", name_mirrors)
print("gateway_call_assignments", gateway_call_assignments)
print("direct_abstract_methods", direct_abstract_methods)
print("wrapped_method_count", len(wrapped_methods))
PY
# name_mirrors []
# gateway_call_assignments []
# direct_abstract_methods []
# wrapped_method_count 26

source .venv/bin/activate && python - <<'PY'
from openhcs.agent.services.ui_bridge_service import (
    UiBridgeOperationContractABC,
    UiBridgeStatusOperation,
)
print("status_name", UiBridgeStatusOperation.name)
print("status_abstract", UiBridgeStatusOperation.__abstractmethods__)
print("registry_count", len(UiBridgeOperationContractABC.__registry__))
PY
# status_name status
# status_abstract frozenset()
# registry_count 26
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/services/ui_bridge_service.py \
  openhcs/pyqt_gui/services/ui_bridge_server.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py::test_ui_bridge_control_server_round_trips_documents_through_descriptor \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py::test_ui_bridge_control_server_preserves_bad_auth_error -q
# 2 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -q
# 66 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed

git diff --check
# no output
```

2026-06-29 UI identity declaration-owner follow-up:

- Concrete UI bridge identity declarations moved out of
  `openhcs.agent.dto.ui_bridge` into `openhcs.agent.ui_bridge_identities`.
  The DTO module now imports and re-exports those declarations for API
  compatibility, but it no longer defines the identity authority.
- `UiSelectedPlateWorkflowKind` is now projected from
  `PlateManagerAction.plate_operation` declarations in
  `openhcs.agent.ui_bridge_actions`, so selected-plate workflow API values are
  derived from the PlateManager action owner.
- PyQt providers, MCP dev-client commands/renderers, and selected-plate service
  import identity declarations from the declaration module instead of from DTOs.
- `PlateManagerAction` and `PlateOperation` live in the agent-facing action
  declaration module, while `plate_manager.py` re-exports them through normal
  imports for existing widget callers.
- Fixed the real undefined `set_saved_global_config` reference in
  `plate_manager.py` while this file was in the ruff slice, and removed stale
  imports surfaced by the focused lint pass.

Verified AST gates:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations
import ast
from pathlib import Path

identity_names = {
    "ManagedWindowWidgetIdentity",
    "PipelineEditorStateSurfaceIdentityDeclaration",
    "PipelineEditorWidgetIdentity",
    "PlateManagerOrchestratorCodeDocumentIdentity",
    "PlateManagerStateSurfaceIdentityDeclaration",
    "PlateManagerWidgetIdentity",
    "UiBridgeIdentityDeclaration",
    "UiCodeDocumentIdentityDeclaration",
    "UiStateSurfaceIdentityDeclarationBase",
    "UiWidgetIdentityDeclaration",
}
violations = []
for path in sorted(Path("openhcs").rglob("*.py")):
    if path == Path("openhcs/agent/dto/ui_bridge.py"):
        continue
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "openhcs.agent.dto.ui_bridge":
            hits = [alias.name for alias in node.names if alias.name in identity_names]
            if hits:
                violations.append(f"{path}:{node.lineno}: {hits}")
print("\n".join(violations) if violations else "no dto identity import crossings")
PY
# no dto identity import crossings

.venv/bin/python - <<'PY'
from __future__ import annotations
import ast
from pathlib import Path

for path in sorted(Path("openhcs").rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "openhcs.pyqt_gui.widgets.plate_manager":
            hits = [alias.name for alias in node.names if alias.name in {"PlateManagerAction", "PlateOperation"}]
            if hits:
                print(f"{path}:{node.lineno}: {hits}")
PY
# no output
```

Verified checks:

```bash
.venv/bin/python -m ruff check \
  openhcs/agent/ui_bridge_actions.py \
  openhcs/agent/ui_bridge_identities.py \
  openhcs/agent/dto/__init__.py \
  openhcs/agent/dto/ui_bridge.py \
  openhcs/agent/services/selected_plate_service.py \
  openhcs/mcp/dev_client.py \
  openhcs/mcp/dev_client_core.py \
  openhcs/mcp/dev_client_commanding.py \
  openhcs/mcp/dev_client_commands \
  openhcs/mcp/dev_client_rendering.py \
  openhcs/mcp/dev_client_renderers \
  openhcs/pyqt_gui/widgets/plate_manager.py \
  openhcs/pyqt_gui/services/ui_agent_bridge.py \
  openhcs/pyqt_gui/services/ui_bridge_contracts.py \
  openhcs/pyqt_gui/services/ui_bridge_windows.py \
  openhcs/pyqt_gui/services/ui_bridge_plate_manager.py \
  openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_plate_manager_widget.py \
  --select F401,F821,F811
# All checks passed

.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_plate_manager_widget.py -q
# 319 passed

git diff --check
# no output
```
