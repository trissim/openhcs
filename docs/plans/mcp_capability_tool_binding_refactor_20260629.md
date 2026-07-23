# MCP Capability and Tool Binding Refactor

Date: 2026-06-29

## Problem

`openhcs/agent/capabilities.py`, `openhcs/mcp/server.py`, and
`openhcs/mcp/dev_client.py` currently repeat the same semantic facts in several
forms:

- `AgentContractName` is a parallel enum of DTO class names.
- `CAPABILITIES` manually maps tool/resource names to string input/output
  contract names, service names, side-effect tuples, and runtime requirements.
- `MutatingCapabilityNamePolicy` infers mutation from name tokens.
- `openhcs/mcp/server.py` repeats transport wrappers and sometimes performs
  projection or validation logic that belongs in agent services.
- `openhcs/mcp/dev_client.py` repeats MCP tool names in command specs and
  response renderers.

This is semantic mirroring. The capability registry is the agent-facing
authority for exposed tools, but it should be typed and derived from existing
service/DTO declarations rather than being a second table of names.

## Existing Authorities

- DTO classes in `openhcs.agent.dto.*` are the input/output contract classes.
- Agent service methods own business behavior.
- `AgentCapabilitySpec` is the correct public capability record shape, but its
  `input_type` and `output_type` fields are currently strings.
- FastMCP registration in `openhcs/mcp/server.py` is only a transport adapter.
- Dev-client command specs are UX affordances, not public API authorities.

## Target Shape

Introduce a typed capability declaration family and make the current registry a
projection from those declarations.

Required properties:

- Each capability declaration has:
  - `name`: final MCP/resource ABI string.
  - `kind`: resource, tool, or prompt.
  - `input_contract`: DTO type or `None`.
  - `output_contract`: DTO/result type or `None`.
  - `service_binding`: a typed service method binding, not a service-name
    string used for dispatch.
  - `side_effects`, `runtime_requirements`, `data_exposure`, and
    `security_requirements`: explicit declaration-owned tuples.
- `AgentCapabilityRegistry` is generated from declarations.
- MCP tool wrappers bind to declarations.
- Dev-client command specs reference capability declarations, not raw tool-name
  strings.
- Mutation is declared, not inferred from words in a name.

## Deterministic Steps

1. Replace `AgentContractName`.
   - Delete the enum.
   - Add `AgentContractProjection` or equivalent helper that returns
     `contract_type.__name__` from a DTO class.
   - Change `AgentCapabilitySpec.input_type` and `output_type` from strings to
     projected strings at serialization time, not stored string authority.

2. Add typed capability declarations.
   - Keep this in `openhcs/agent/capabilities.py` or a sibling module such as
     `openhcs/agent/capability_declarations.py`.
   - Use `AutoRegisterMeta` only if declarations become one class per
     capability. Otherwise, a tuple of typed `AgentCapabilityDeclaration`
     instances is acceptable because the capability registry itself is the
     intended agent API authority.
   - Do not create a parallel enum of capability names.

3. Move explicit mutation metadata into declarations.
   - Delete `MutatingCapabilityNamePolicy` and
     `MUTATING_CAPABILITY_NAME_POLICY`.
   - Every mutating capability declares side effects.
   - Validation fails if a capability has a mutating service binding but no
     side effects. The mutating status must come from the declaration or from a
     typed service operation contract, not from name tokens.

4. Bind MCP tools from declarations.
   - Keep FastMCP function signatures scalar/list/dict friendly.
   - Each wrapper should do only:
     - accept transport parameters,
     - build the declared request DTO,
     - call the declared agent service binding,
     - serialize the declared result DTO.
   - If a wrapper contains non-transport projection logic, move it into the
     agent service first.

5. Delete duplicate contract strings from tests.
   - Update capability tests to assert against DTO classes or generated schema
     names.
   - Add a test that every capability output contract resolves to an importable
     DTO/result class unless it is explicitly `None`.
   - Add a test that every registered MCP tool has exactly one capability
     declaration.

6. Add a capability-to-MCP binding test.
   - `openhcs_list_capabilities` must be generated from the same declarations as
     the registered FastMCP tool list.
   - The test should fail if a tool is added to `server.py` without a
     capability declaration or vice versa.

## Dev-Client Boundary

The dev client can keep command names such as `knowledge-search` or `ui-status`
because those are CLI UX names. It cannot own MCP tool names. For every command
that calls a single tool:

```python
class KnowledgeCommandSpec(SingleToolCommandSpec):
    capability = ListKnowledgeDocumentsCapability
```

`SingleToolCommandSpec` should derive the tool name from `capability.name`.

Command-specific renderers should bind to output DTO type or capability
declaration. They should not rediscover payload semantics from arbitrary dict
keys when a typed response DTO exists.

## AST Removal Gates

```bash
rg -n "class AgentContractName|MutatingCapabilityNamePolicy|MUTATING_CAPABILITY_NAME_POLICY" openhcs/agent/capabilities.py
rg -n "output_type=AgentContractName|input_type=\"[A-Za-z].*\"|output_type=\"[A-Za-z].*\"" openhcs/agent/capabilities.py
rg -n "tool_name = \"openhcs_" openhcs/mcp/dev_client.py
```

Expected result:

- no `AgentContractName`;
- no mutation token policy;
- no dev-client raw MCP tool-name constants except in capability declarations or
  final transport assertions.

## Tests

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_agent_serialization.py
```

## Implementation Progress

2026-06-29 partial implementation:

- `AgentContractName`, `MutatingCapabilityNamePolicy`, and
  `MUTATING_CAPABILITY_NAME_POLICY` were removed.
- `AgentCapabilitySpec` now stores typed `input_contract` and
  `output_contract` references and projects public `input_type` /
  `output_type` values for compatibility.
- Capability registry serialization is registered explicitly through
  `to_jsonable`, so contract types do not leak onto the wire.
- Mutation is explicit through `AgentCapabilitySpec.mutating`; validation no
  longer infers mutation from capability-name tokens.
- Dev-client single-tool command specs now hold `AgentCapabilitySpec`
  references and derive MCP tool names from capability declarations.

Verified gates:

```bash
rg -n "class AgentContractName|MutatingCapabilityNamePolicy|MUTATING_CAPABILITY_NAME_POLICY|output_type=AgentContractName|input_type=\"[A-Za-z].*\"|output_type=\"[A-Za-z].*\"" openhcs/agent/capabilities.py
rg -n "tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"" openhcs/mcp/dev_client.py
```

Both commands currently return no matches.

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_agent_serialization.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
```

The second command passes with 168 selected tests. The excluded fresh-process
stdio smoke still times out during MCP `initialize` even with a 30 second
timeout and only emits the component cache log line on stderr. That remains a
transport/startup investigation item, not a schema-binding gate.

2026-06-29 capability namespace follow-up:

- Added `AgentCapabilityNamespace`, generated directly from `CAPABILITIES`.
- Dev-client command specs now reference `agent_capabilities.<name>` instead of
  calling `get_agent_capability("openhcs_*")` at declaration time.
- The generic `call` command still resolves a user-supplied final tool name
  through `get_agent_capability(tool_name)`, which is runtime transport lookup,
  not a declaration mirror.

Verified gates:

```bash
rg -n "get_agent_capability\\(\"openhcs_|tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"" \
  openhcs/mcp/dev_client.py
# no matches

rg -n "class AgentContractName|MutatingCapabilityNamePolicy|MUTATING_CAPABILITY_NAME_POLICY|output_type=AgentContractName|input_type=\"[A-Za-z].*\"|output_type=\"[A-Za-z].*\"" \
  openhcs/agent/capabilities.py
# no matches
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 66 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_agent_serialization.py
# 7 passed
```

2026-06-29 generated from_fields MCP binding follow-up:

- Added nominal `AgentCapabilityRequestInvocationABC` and
  `AgentFromFieldsServiceInvocation` declarations in
  `openhcs.agent.capabilities`.
- The 13 straight `from_fields()` MCP tools now declare their service
  invocation on the capability declaration itself:
  knowledge document, synthetic plate generation, plate inspect/query/sample/
  stream, add function step, orchestrator-session creation, source artifact
  plan inspection, and runtime-server scan/info/status.
- `openhcs.mcp.server` now has `GeneratedMcpFromFieldsToolBinding`, which
  hydrates the request DTO from the declared `from_fields()` signature and
  invokes the capability declaration mechanically.
- Removed the 13 explicit `McpFromFieldsToolBindingABC[...]` leaf classes.
  There are currently no explicit from-fields MCP binding leaves; this family
  is generated from capability declarations.
- Added `get_agent_capability_declaration()` so runtime adapters can ask for
  the declaration owner rather than treating the serialized
  `AgentCapabilitySpec` as execution authority.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
removed = {
    "GetKnowledgeDocumentMcpToolBinding",
    "InspectPlatePathMcpToolBinding",
    "GenerateSyntheticPlateMcpToolBinding",
    "QueryPlateFilesMcpToolBinding",
    "SamplePlateImageMcpToolBinding",
    "StreamPlateFilesToViewerMcpToolBinding",
    "AddFunctionStepMcpToolBinding",
    "CreateOrchestratorSessionMcpToolBinding",
    "CreateOrchestratorSessionFromPipelineSourceMcpToolBinding",
    "InspectPipelineSourceArtifactPlanMcpToolBinding",
    "ScanRuntimeServersMcpToolBinding",
    "GetRuntimeServerInfoMcpToolBinding",
    "GetRuntimeServerExecutionStatusMcpToolBinding",
}
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
from_fields_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if any(base.startswith("McpFromFieldsToolBindingABC") for base in bases):
            from_fields_leaves.append(node.name)
request_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "request_invocation":
                        request_invocations.append(node.name)
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_from_fields_leaves", from_fields_leaves)
print("request_invocation_count", len(request_invocations))
print("request_invocations", request_invocations)
PY
# removed_classes_still_present []
# explicit_from_fields_leaves []
# request_invocation_count 13
```

```bash
source .venv/bin/activate && python - <<'PY'
from openhcs.mcp import server

expected = {
    "openhcs_get_knowledge_document",
    "openhcs_generate_synthetic_plate",
    "openhcs_inspect_plate_path",
    "openhcs_query_plate_files",
    "openhcs_sample_plate_image",
    "openhcs_stream_plate_files_to_viewer",
    "openhcs_add_function_step",
    "openhcs_create_orchestrator_session",
    "openhcs_create_orchestrator_session_from_pipeline_source",
    "openhcs_inspect_pipeline_source_artifact_plan",
    "openhcs_scan_runtime_servers",
    "openhcs_get_runtime_server_info",
    "openhcs_get_runtime_server_execution_status",
}
actual = {declaration.name for declaration in server.generated_from_fields_capability_declarations()}
print("generated_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("extra_generated", sorted(actual - expected))
PY
# generated_count 13
# missing_generated []
# extra_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "knowledge or synthetic or inspect_plate or query_plate_files or sample_plate_image or stream_plate_files or runtime_scan or runtime_info or runtime_status or add_function_step or orchestrator_session or artifact_plan" -q
# 37 passed, 201 deselected

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

2026-06-29 resource binding follow-up:

- MCP resources are now bound from `CapabilityKind.RESOURCE` declarations using
  `GeneratedMcpResourceBinding`.
- Resource declarations own their no-argument invocation just like generated
  no-argument tools:
  - `CapabilitiesResourceCapability`
  - `KnowledgeResourceCapability`
  - `ArchitectureTopicsResourceCapability`
- `mcp/server.py` no longer declares manual `@server.resource(...)` functions
  for those URIs.
- `generated_no_argument_capability_declarations()` now filters
  `CapabilityKind.TOOL`, so resource declarations are not accidentally exposed
  as tools.

Verified gates:

```bash
rg -n "@server\\.resource|capabilities_resource|architecture_topics_resource|knowledge_resource" \
  openhcs/mcp/server.py
# no matches

XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m openhcs.mcp.dev_client tools
# Tools: matched=76 total=76 shown=76
# resource URIs are not listed as tools
```

Verified direct resource smoke:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python - <<'PY'
from __future__ import annotations
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openhcs.mcp"],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources = await session.list_resources()
            print("resources", [str(resource.uri) for resource in resources.resources])
            for uri in (
                "openhcs://capabilities",
                "openhcs://knowledge",
                "openhcs://architecture/topics",
            ):
                result = await session.read_resource(uri)
                print(uri, sorted(json.loads(result.contents[0].text).keys())[:6])

asyncio.run(main())
PY
# resources ['openhcs://capabilities', 'openhcs://knowledge', 'openhcs://architecture/topics']
# openhcs://capabilities ['capabilities', 'schema_version']
# openhcs://knowledge ['documents', 'errors', 'schema_version', 'warnings']
# openhcs://architecture/topics ['schema_version', 'topics']
```

Verified tests:

```bash
QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 260 passed
```

2026-06-29 generated dataclass-request MCP binding follow-up:

- Split generic request execution from MCP signature shape markers:
  `AgentRequestServiceInvocation` owns service+request execution, while
  `AgentFromFieldsServiceInvocation` and
  `AgentDataclassRequestServiceInvocation` mark the transport-facing request
  construction shape.
- Twelve direct dataclass request tools now declare dispatch on their capability
  declarations: function search/detail/custom registration, config source
  rendering, pipeline validation/source rendering, authoring context,
  orchestrator session lookup, compile/execute submission, execution status,
  and knowledge search.
- `openhcs.mcp.server` now has `GeneratedMcpDataclassRequestToolBinding`,
  which builds FastMCP parameters from dataclass request fields and invokes the
  capability declaration mechanically.
- Removed all explicit `McpDataclassRequestToolBindingABC` leaf classes.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {
    "SearchFunctionsMcpToolBinding",
    "DescribeFunctionMcpToolBinding",
    "RegisterCustomFunctionMcpToolBinding",
    "RenderConfigSourceMcpToolBinding",
    "ValidatePipelineMcpToolBinding",
    "RenderPipelineSourceMcpToolBinding",
    "GetAuthoringContextMcpToolBinding",
    "GetOrchestratorSessionMcpToolBinding",
    "SubmitCompileMcpToolBinding",
    "SubmitPipelineExecutionMcpToolBinding",
    "GetExecutionStatusMcpToolBinding",
    "SearchKnowledgeMcpToolBinding",
}
expected = {
    "openhcs_search_functions",
    "openhcs_describe_function",
    "openhcs_register_custom_function",
    "openhcs_render_config_source",
    "openhcs_validate_pipeline",
    "openhcs_render_pipeline_source",
    "openhcs_get_authoring_context",
    "openhcs_get_orchestrator_session",
    "openhcs_submit_compile",
    "openhcs_submit_pipeline_execution",
    "openhcs_get_execution_status",
    "openhcs_search_knowledge",
}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
dataclass_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if any(base.startswith("McpDataclassRequestToolBindingABC") for base in bases):
            dataclass_leaves.append(node.name)
dataclass_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "request_invocation" for t in stmt.targets):
                continue
            if isinstance(stmt.value, ast.Call):
                func_name = ast.unparse(stmt.value.func)
                if func_name == "AgentDataclassRequestServiceInvocation":
                    dataclass_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_dataclass_request_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_dataclass_leaves", dataclass_leaves)
print("dataclass_invocation_count", len(dataclass_invocations))
print("generated_dataclass_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_dataclass_leaves []
# dataclass_invocation_count 12
# generated_dataclass_count 12
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "functions or function or register_custom_function or render_config or validate_pipeline or render_pipeline or authoring_context or orchestrator_session or submit_compile or submit_pipeline or execution_status or knowledge_search or mcp_server_builds" -q
# 13 passed, 225 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated UI command-request MCP binding follow-up:

- Added nominal `CapabilityUiBridgeTimeoutProfile`.
- Extended `AgentConnectionRequestServiceInvocation` with a declaration-owned
  UI bridge timeout profile.
- Twelve direct command-timeout UI request tools now declare dispatch on their
  capability declarations: focus/navigate/close/snapshot window,
  time-travel/switch/restore snapshot branch, mutate ObjectState field, invoke
  UI action, selected-plate workflow, invoke widget action, and apply code
  document.
- `GeneratedMcpUiRequestToolBinding` now chooses
  `McpUiBridgeTimeoutPolicy` or `McpUiBridgeCommandTimeoutPolicy` from the
  capability invocation profile.
- Removed the 12 direct command UI request MCP leaf classes.
  `UiGetWidgetTreeMcpToolBinding` remains explicit because it has the MCP-only
  `compact_actions` parameter and compact payload projection.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.agent.capabilities import CapabilityUiBridgeTimeoutProfile
from openhcs.mcp import server

removed = {
    "UiFocusWindowMcpToolBinding",
    "UiNavigateWindowMcpToolBinding",
    "UiCloseWindowMcpToolBinding",
    "UiSnapshotWindowMcpToolBinding",
    "UiTimeTravelHeadMcpToolBinding",
    "UiSwitchBranchMcpToolBinding",
    "UiRestoreSnapshotMcpToolBinding",
    "UiMutateObjectStateFieldMcpToolBinding",
    "UiInvokeActionMcpToolBinding",
    "UiSelectedPlateWorkflowMcpToolBinding",
    "UiInvokeWidgetActionMcpToolBinding",
    "UiApplyCodeDocumentMcpToolBinding",
}
expected_command = {
    "openhcs_ui_focus_window",
    "openhcs_ui_navigate_window",
    "openhcs_ui_close_window",
    "openhcs_ui_snapshot_window",
    "openhcs_ui_time_travel_head",
    "openhcs_ui_switch_branch",
    "openhcs_ui_restore_snapshot",
    "openhcs_ui_mutate_object_state_field",
    "openhcs_ui_invoke_action",
    "openhcs_ui_selected_plate_workflow",
    "openhcs_ui_invoke_widget_action",
    "openhcs_ui_apply_code_document",
}
expected_default = {
    "openhcs_ui_get_state_surface",
    "openhcs_ui_get_code_document",
    "openhcs_ui_validate_code_document",
    "openhcs_ui_list_snapshots",
    "openhcs_ui_inspect_selected_plate_images",
    "openhcs_ui_query_selected_plate_files",
    "openhcs_ui_sample_selected_plate_image",
    "openhcs_ui_stream_selected_plate_files_to_viewer",
    "openhcs_ui_list_object_state_scopes",
    "openhcs_ui_get_object_state_fields",
    "openhcs_ui_describe_object_state_field",
}
expected = expected_command | expected_default
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
ui_request_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if any(base.startswith("McpUiRequestToolBindingABC") for base in bases):
            ui_request_leaves.append(node.name)
connection_request_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "connection_request_invocation":
                        connection_request_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_ui_request_capability_declarations()}
command_generated = sorted(
    declaration.name
    for declaration in server.generated_ui_request_capability_declarations()
    if declaration.connection_request_invocation.timeout_profile is CapabilityUiBridgeTimeoutProfile.COMMAND
)
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_ui_request_leaves", ui_request_leaves)
print("connection_request_invocation_count", len(connection_request_invocations))
print("generated_ui_request_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
print("missing_command_generated", sorted(expected_command - set(command_generated)))
print("unexpected_command_generated", sorted(set(command_generated) - expected_command))
PY
# removed_classes_still_present []
# explicit_ui_request_leaves ['UiGetWidgetTreeMcpToolBinding']
# connection_request_invocation_count 23
# generated_ui_request_count 23
# missing_generated []
# unexpected_generated []
# missing_command_generated []
# unexpected_command_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "focus_window or navigate_window or close_window or snapshot_window or time_travel or switch_branch or restore_snapshot or mutate_object_state or invoke_action or selected_plate_workflow or invoke_widget_action or apply_code_document or widget_tree or mcp_server_builds" -q
# 19 passed, 219 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated viewer request MCP binding follow-up:

- Added nominal `CapabilityViewerControlTimeoutProfile` and
  `AgentViewerWindowRequestServiceInvocation`.
- Eight standard viewer tools now declare request dispatch on their capability
  declarations: snapshot, state, payloads, image sample, ROI summary,
  navigation, layer isolation, and validation.
- Navigation and layer isolation declare the `COMMAND` viewer timeout profile;
  the other generated viewer tools use the default viewer timeout profile.
- `openhcs.mcp.server` now has `GeneratedMcpViewerRequestToolBinding`, which
  builds connection plus `from_fields()` request-option signatures from the
  declared request DTO and invokes the capability declaration mechanically.
- Removed the eight duplicated viewer MCP leaf classes. `ViewerProbeMcpToolBinding`
  remains explicit because it intentionally exposes only connection fields and
  builds a no-options state request internally.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.agent.capabilities import CapabilityViewerControlTimeoutProfile
from openhcs.mcp import server

removed = {
    "ViewerStateMcpToolBinding",
    "ViewerPayloadsMcpToolBinding",
    "ViewerNavigationMcpToolBinding",
    "ViewerLayerIsolationMcpToolBinding",
    "ViewerImageSampleMcpToolBinding",
    "ViewerRoiSummaryMcpToolBinding",
    "ViewerValidationMcpToolBinding",
    "ViewerSnapshotMcpToolBinding",
}
expected = {
    "openhcs_viewer_snapshot_window",
    "openhcs_get_viewer_window_state",
    "openhcs_get_viewer_window_payloads",
    "openhcs_sample_viewer_window_image",
    "openhcs_summarize_viewer_window_rois",
    "openhcs_navigate_viewer_window",
    "openhcs_isolate_viewer_window_layers",
    "openhcs_validate_viewer_window_state",
}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
viewer_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpViewerRequestToolBindingABC" in bases:
            viewer_leaves.append(node.name)
viewer_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "request_invocation" for t in stmt.targets):
                continue
            if isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "AgentViewerWindowRequestServiceInvocation":
                viewer_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_viewer_request_capability_declarations()}
command_timeout = sorted(
    declaration.name
    for declaration in server.generated_viewer_request_capability_declarations()
    if declaration.request_invocation.timeout_profile is CapabilityViewerControlTimeoutProfile.COMMAND
)
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_viewer_leaves", viewer_leaves)
print("viewer_invocation_count", len(viewer_invocations))
print("generated_viewer_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
print("command_timeout_generated", command_timeout)
PY
# removed_classes_still_present []
# explicit_viewer_leaves ['ViewerProbeMcpToolBinding']
# viewer_invocation_count 8
# generated_viewer_count 8
# missing_generated []
# unexpected_generated []
# command_timeout_generated [
#   'openhcs_isolate_viewer_window_layers',
#   'openhcs_navigate_viewer_window'
# ]
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "viewer or napari or payloads or rois or mcp_server_builds" -q
# 60 passed, 178 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated no-argument function MCP binding follow-up:

- Added nominal `AgentNoArgumentFunctionInvocation`.
- `openhcs_list_capabilities` now declares dispatch on its capability
  declaration instead of keeping a dedicated MCP leaf.
- `generated_no_argument_capability_declarations()` now accepts any
  declaration-owned no-argument invocation. Explicit leaves still win, so
  `HealthCheckMcpToolBinding` remains custom for stale-source diagnostics.
- Removed `ListCapabilitiesMcpToolBinding`; health is now the only explicit
  `McpNoArgumentToolBindingABC` leaf.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {"ListCapabilitiesMcpToolBinding"}
expected = {
    "openhcs_list_capabilities",
    "openhcs_list_knowledge_documents",
    "openhcs_list_architecture_topics",
    "openhcs_create_pipeline",
    "openhcs_ui_list_bridges",
}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
no_arg_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpNoArgumentToolBindingABC" in bases:
            no_arg_leaves.append(node.name)
no_arg_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "no_argument_invocation":
                        no_arg_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_no_argument_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_no_argument_leaves", no_arg_leaves)
print("no_argument_invocation_count", len(no_arg_invocations))
print("generated_no_argument_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_no_argument_leaves ['HealthCheckMcpToolBinding']
# no_argument_invocation_count 5
# generated_no_argument_count 5
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "list_capabilities or capabilities or health or mcp_server_builds" -q
# 5 passed, 233 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated UI scalar MCP binding follow-up:

- Added nominal `AgentCapabilityConnectionScalarInvocationABC` and
  `AgentConnectionScalarServiceInvocation` declarations.
- `openhcs_ui_get_operation_status` now declares dispatch on its capability
  declaration.
- `openhcs.mcp.server` now has `GeneratedMcpUiScalarInputToolBinding`, which
  builds the scalar input plus MCP `connection` signature from the capability
  declaration and invokes it mechanically.
- Removed all explicit `McpUiScalarInputToolBindingABC` leaf classes.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {"UiGetOperationStatusMcpToolBinding"}
expected = {"openhcs_ui_get_operation_status"}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
ui_scalar_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpUiScalarInputToolBindingABC" in bases:
            ui_scalar_leaves.append(node.name)
connection_scalar_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "connection_scalar_invocation" for t in stmt.targets):
                continue
            if isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "AgentConnectionScalarServiceInvocation":
                connection_scalar_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_ui_scalar_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_ui_scalar_leaves", ui_scalar_leaves)
print("connection_scalar_invocation_count", len(connection_scalar_invocations))
print("generated_ui_scalar_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_ui_scalar_leaves []
# connection_scalar_invocation_count 1
# generated_ui_scalar_count 1
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "operation_status or ui_bridge_operation or mcp_server_builds" -q
# 1 passed, 237 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated ConfigPatch MCP binding follow-up:

- Added nominal `AgentConfigPatchServiceInvocation` as a transport-shape marker
  over generic request execution.
- `openhcs_create_config` and `openhcs_validate_config_patch` now declare
  dispatch on their capability declarations.
- `openhcs.mcp.server` now has `GeneratedMcpConfigPatchToolBinding`, which
  preserves the existing MCP `values=None` to empty JSON-object coercion while
  invoking the capability declaration mechanically.
- Removed all explicit `McpConfigPatchToolBindingABC` leaf classes.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {"CreateConfigMcpToolBinding", "ValidateConfigPatchMcpToolBinding"}
expected = {"openhcs_create_config", "openhcs_validate_config_patch"}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
config_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpConfigPatchToolBindingABC" in bases:
            config_leaves.append(node.name)
config_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "request_invocation" for t in stmt.targets):
                continue
            if isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "AgentConfigPatchServiceInvocation":
                config_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_config_patch_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_config_patch_leaves", config_leaves)
print("config_patch_invocation_count", len(config_invocations))
print("generated_config_patch_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_config_patch_leaves []
# config_patch_invocation_count 2
# generated_config_patch_count 2
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "create_config or validate_config_patch or config_patch or mcp_server_builds" -q
# 1 passed, 237 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated UI request MCP binding follow-up:

- Added nominal `AgentCapabilityConnectionRequestInvocationABC` and
  `AgentConnectionRequestServiceInvocation` declarations.
- Eleven default-timeout UI request tools now declare dispatch on their
  capability declarations: selected-plate inspect/query/sample/stream,
  state-surface read, ObjectState scope/field/help reads, code-document read,
  code-document validation, and snapshot listing.
- `openhcs.mcp.server` now has `GeneratedMcpUiRequestToolBinding`, which binds
  default-timeout UI request tools from capability declarations and keeps the
  MCP `connection` parameter as transport-level plumbing.
- Removed the eleven explicit `McpUiRequestToolBindingABC[...]` leaf classes
  for those tools.
- Kept command-timeout mutating/window/action bindings explicit for now.
  `UiGetWidgetTreeMcpToolBinding` also remains explicit because it has an
  MCP-specific `compact_actions` projection parameter.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {
    "UiGetStateSurfaceMcpToolBinding",
    "UiGetCodeDocumentMcpToolBinding",
    "UiValidateCodeDocumentMcpToolBinding",
    "UiListSnapshotsMcpToolBinding",
    "UiInspectSelectedPlateImagesMcpToolBinding",
    "UiQuerySelectedPlateFilesMcpToolBinding",
    "UiSampleSelectedPlateImageMcpToolBinding",
    "UiStreamSelectedPlateFilesToViewerMcpToolBinding",
    "UiListObjectStateScopesMcpToolBinding",
    "UiGetObjectStateFieldsMcpToolBinding",
    "UiDescribeObjectStateFieldMcpToolBinding",
}
expected_capabilities = {
    "openhcs_ui_get_state_surface",
    "openhcs_ui_get_code_document",
    "openhcs_ui_validate_code_document",
    "openhcs_ui_list_snapshots",
    "openhcs_ui_inspect_selected_plate_images",
    "openhcs_ui_query_selected_plate_files",
    "openhcs_ui_sample_selected_plate_image",
    "openhcs_ui_stream_selected_plate_files_to_viewer",
    "openhcs_ui_list_object_state_scopes",
    "openhcs_ui_get_object_state_fields",
    "openhcs_ui_describe_object_state_field",
}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
ui_request_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if any(base.startswith("McpUiRequestToolBindingABC") for base in bases):
            ui_request_leaves.append(node.name)
connection_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "connection_request_invocation":
                        connection_invocations.append(node.name)
actual_capabilities = {declaration.name for declaration in server.generated_ui_request_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_ui_request_leaves", ui_request_leaves)
print("connection_request_invocation_count", len(connection_invocations))
print("generated_ui_request_count", len(actual_capabilities))
print("missing_generated_capabilities", sorted(expected_capabilities - actual_capabilities))
print("unexpected_generated_capabilities", sorted(actual_capabilities - expected_capabilities))
PY
# removed_classes_still_present []
# connection_request_invocation_count 11
# generated_ui_request_count 11
# missing_generated_capabilities []
# unexpected_generated_capabilities []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "state_surface or code_document or selected_plate or object_state or snapshots or ui_bridge_service" -q
# 55 passed, 183 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated scalar-input MCP binding follow-up:

- Added nominal `AgentCapabilityScalarInvocationABC` and
  `AgentScalarServiceInvocation` declarations.
- Three one-scalar tools now declare dispatch on their capability declarations:
  `openhcs_explain_architecture`, `openhcs_describe_internal_symbol`, and
  `openhcs_describe_config_schema`.
- `openhcs.mcp.server` now has `GeneratedMcpScalarInputToolBinding`, which
  builds the public tool signature from `AgentScalarInputContract` and invokes
  the capability declaration mechanically.
- Removed all explicit `McpScalarInputToolBindingABC` leaf classes.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {
    "ExplainArchitectureMcpToolBinding",
    "DescribeInternalSymbolMcpToolBinding",
    "DescribeConfigSchemaMcpToolBinding",
}
expected = {
    "openhcs_explain_architecture",
    "openhcs_describe_internal_symbol",
    "openhcs_describe_config_schema",
}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
scalar_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpScalarInputToolBindingABC" in bases:
            scalar_leaves.append(node.name)
scalar_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "scalar_invocation":
                        scalar_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_scalar_input_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_scalar_leaves", scalar_leaves)
print("scalar_invocation_count", len(scalar_invocations))
print("generated_scalar_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_scalar_leaves []
# scalar_invocation_count 3
# generated_scalar_count 3
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "architecture or internal_symbol or config_schema or mcp_server_builds" -q
# 3 passed, 235 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated UI connection MCP binding follow-up:

- Added nominal `AgentCapabilityConnectionInvocationABC` and
  `AgentConnectionServiceInvocation` declarations.
- Two connection-only UI tools now declare dispatch on their capability
  declarations: `openhcs_ui_bridge_status` and `openhcs_ui_list_branches`.
- `openhcs.mcp.server` now has `GeneratedMcpUiConnectionToolBinding`, which
  resolves the MCP `connection` argument once and invokes the capability
  declaration mechanically.
- Removed `UiBridgeStatusMcpToolBinding` and
  `UiListBranchesMcpToolBinding`.
- Kept the four UI catalog connection leaves explicit:
  code documents, state surfaces, actions, and windows. Those bindings still
  own MCP-specific identity flattening through `McpUiCatalogPayloadProjection`;
  moving them requires making that projection DTO/serialization-owned first.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path
from openhcs.mcp import server

removed = {"UiBridgeStatusMcpToolBinding", "UiListBranchesMcpToolBinding"}
expected = {"openhcs_ui_bridge_status", "openhcs_ui_list_branches"}
server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
ui_connection_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpUiConnectionToolBindingABC" in bases:
            ui_connection_leaves.append(node.name)
connection_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "connection_invocation":
                        connection_invocations.append(node.name)
actual = {declaration.name for declaration in server.generated_ui_connection_capability_declarations()}
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_ui_connection_leaves", ui_connection_leaves)
print("connection_invocation_count", len(connection_invocations))
print("generated_ui_connection_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("unexpected_generated", sorted(actual - expected))
PY
# removed_classes_still_present []
# explicit_ui_connection_leaves [
#   'UiListCodeDocumentsMcpToolBinding',
#   'UiListStateSurfacesMcpToolBinding',
#   'UiListActionsMcpToolBinding',
#   'UiListWindowsMcpToolBinding'
# ]
# connection_invocation_count 2
# generated_ui_connection_count 2
# missing_generated []
# unexpected_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "ui_bridge_status or ui_list_branches or branches or mcp_server_builds" -q
# 1 passed, 237 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 generated no-argument MCP binding follow-up:

- Added nominal `AgentCapabilityNoArgumentInvocationABC` and
  `AgentNoArgumentServiceInvocation` declarations.
- Four straight no-argument service tools now declare dispatch on their
  capability declarations: list knowledge documents, list architecture topics,
  create draft pipeline, and list UI bridges.
- `openhcs.mcp.server` now has `GeneratedMcpNoArgumentToolBinding`, which binds
  no-argument FastMCP tools from capability declarations.
- Removed the four explicit no-argument MCP leaf classes for those tools.
- Kept `ListCapabilitiesMcpToolBinding` and `HealthCheckMcpToolBinding`
  explicit because they are not simple context-service method calls; health
  also owns stale-source process diagnostics.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

server_tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
cap_tree = ast.parse(Path("openhcs/agent/capabilities.py").read_text())
removed = {
    "ListKnowledgeDocumentsMcpToolBinding",
    "ListArchitectureTopicsMcpToolBinding",
    "CreatePipelineMcpToolBinding",
    "UiListBridgesMcpToolBinding",
}
class_names = {node.name for node in server_tree.body if isinstance(node, ast.ClassDef)}
no_arg_leaves = []
for node in server_tree.body:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        if "McpNoArgumentToolBindingABC" in bases:
            no_arg_leaves.append(node.name)
no_arg_invocations = []
for node in cap_tree.body:
    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "no_argument_invocation":
                        no_arg_invocations.append(node.name)
print("removed_classes_still_present", sorted(removed & class_names))
print("explicit_no_argument_leaves", no_arg_leaves)
print("no_argument_invocation_count", len(no_arg_invocations))
print("no_argument_invocations", no_arg_invocations)
PY
# removed_classes_still_present []
# explicit_no_argument_leaves ['ListCapabilitiesMcpToolBinding', 'HealthCheckMcpToolBinding']
# no_argument_invocation_count 4
```

```bash
source .venv/bin/activate && python - <<'PY'
from openhcs.mcp import server

expected = {
    "openhcs_list_knowledge_documents",
    "openhcs_list_architecture_topics",
    "openhcs_create_pipeline",
    "openhcs_ui_list_bridges",
}
actual = {declaration.name for declaration in server.generated_no_argument_capability_declarations()}
print("explicit_no_arg_registry", len(server.McpNoArgumentToolBindingABC.__registry__))
print("generated_no_arg_count", len(actual))
print("missing_generated", sorted(expected - actual))
print("extra_generated", sorted(actual - expected))
PY
# explicit_no_arg_registry 2
# generated_no_arg_count 4
# missing_generated []
# extra_generated []
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/capabilities.py \
  openhcs/mcp/server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "knowledge or architecture or create_pipeline or ui_list_bridges or mcp_server_builds" -q
# 10 passed, 228 deselected

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
