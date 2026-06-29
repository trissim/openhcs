# MCP Plate, Viewer, and Runtime Projection Refactor

Date: 2026-06-29

## Problem

The agent services for plate inspection, synthetic plate generation, viewer
control, and runtime server status contain several semantic mirrors:

- `PlateInspectionComponentKind` duplicates component axes already exposed by
  `AllComponents.ordered_names()`.
- `PlateFileQueryKind` duplicates `PlateFileKind` plus an agent-local `all`
  sentinel.
- `SyntheticPlateGenerationFormat` duplicates the synthetic generator's
  supported microscope formats.
- `SyntheticPlateGenerationDefaults` and validation bounds live in the agent
  DTO/service even though they are generator semantics.
- `PlateInspectionFileQueryProjection.kinds()` maps an agent enum back into
  `PlateFileKind`.
- `ViewerControlField`, `ViewerLayerField`, `ViewerPayloadField`,
  `ViewerPayloadSummaryField`, and `ViewerDescriptorField` repeat viewer wire
  fields that are already declared in the runtime viewer protocol and
  zmqruntime.
- `RuntimeServerPongClassifier` classifies runtime servers from hardcoded
  server-name substrings.

These are read/projection features, but they still need single-source semantic
ownership. Agent DTOs should project backend declarations; they should not
declare parallel semantic vocabularies for component axes, file kinds, viewer
wire payloads, or runtime server identities.

## Existing Authorities

- `AllComponents`, `VariableComponents`, and `GroupBy` in
  `openhcs.constants.constants` own component-axis semantics.
- `PlateFileKind`, `PlateFileInventory`, and `PlateFileInventoryQuery` in
  `openhcs/core/plate_image_inventory.py` own file-record query semantics.
- `Microscope` and microscope handlers own microscope type identities.
- The synthetic generator currently lives in
  `openhcs/tests/generators/generate_synthetic_data.py`. There is no
  production generator profile authority yet. That must be created before the
  agent layer can stop owning synthetic defaults.
- `zmqruntime.viewer_protocol` owns viewer wire fields, reply headers, and
  reply payloads. `openhcs/runtime/viewer_protocol.py` owns OpenHCS control
  request DTOs such as `ViewerPayloadControlOptions`.
- `ZMQExecutionClient` and execution server status/pong payloads should own
  runtime server identity classification.

## Target Shape

### Plate inspection DTOs

`openhcs/agent/dto/plate.py` should carry stable agent DTOs, but not restate
backend enum values.

Replace:

- `PlateInspectionComponentKind` with a projection from `AllComponents`.
- `PlateFileQueryKind` with either:
  - direct use of `PlateFileKind` for concrete kinds plus a typed query object
    with `kinds=()`, or
  - an `AllPlateFiles` query sentinel declared next to `PlateFileInventoryQuery`.
- `SyntheticPlateGenerationFormat` with a production generator profile enum or
  supported-format query.
- `SyntheticPlateGenerationDefaults` with generator profile defaults/bounds.

`PlateInspectionFileQueryProjection` should not convert between agent and core
file-kind enums. It should receive a typed `PlateFileInventoryQuery`.

### Synthetic generation authority

Create a production profile for synthetic plate generation before changing MCP
DTOs. A minimal target shape:

```python
@dataclass(frozen=True, slots=True)
class SyntheticPlateGenerationProfile:
    default_request: SyntheticPlateGenerationParameters
    bounds: SyntheticPlateGenerationBounds
    supported_formats: tuple[Microscope | SyntheticMicroscopeFormat, ...]
```

The profile should live near the generator or in a production module that owns
test-data generation. The agent request DTO should import defaults from the
profile or expose a profile summary, not own them.

If the existing generator remains under `openhcs/tests`, the first refactor must
either move the generator authority to production or create a production wrapper
that owns the supported profile and calls the test generator internally.

### Viewer window service

`openhcs/agent/services/viewer_window_service.py` should parse viewer replies
through runtime protocol payload classes.

Required changes:

- Replace local `ViewerControlField` references with
  `ViewerControlResponseField` or `ViewerControlReplyPayload` accessors.
- Replace local layer/payload field classes with protocol-owned payload
  projection DTOs. If no such DTO exists for state/payload replies, add it in
  runtime viewer protocol, not in the agent service.
- Keep validation warning rules in the agent service only if they are
  agent-facing review policy. They may inspect typed viewer DTOs, not raw
  protocol dicts.
- Keep payload-size bounding in the agent service; it is API policy, not viewer
  wire semantics.

### Runtime server classifier

Move server identity classification out of
`RuntimeServerPongClassifier._EXECUTION_SERVER_NAMES`.

Target authority:

- execution server pong payload declares its server role/type; or
- `ZMQExecutionClient.scan_servers()` returns a typed result with
  `is_execution_server`; or
- a runtime server protocol enum declares supported server roles.

The agent service then projects the typed scan result. It must not classify by
matching display names.

## Deterministic Steps

1. Add plate query authority methods.
   - Add a method to `PlateFileInventoryQuery` or a sibling query type that
     represents all files without an agent-local enum.
   - Update `PlateFileQueryRequest` to carry core file kinds or query sentinel.
   - Delete `PlateFileQueryKind` once tests are updated.

2. Derive component summaries from `AllComponents`.
   - Replace `PlateInspectionComponentKind` members with component-axis values
     emitted by `AllComponents`.
   - Update `PlateInspectionComponentEntrySet` to store the backend component
     declaration or its final value.
   - Add a test that every reported component kind is in
     `AllComponents.ordered_names()`.

3. Create synthetic generation profile authority.
   - Move supported formats/defaults/bounds out of agent DTOs into generator
     profile code.
   - Update `SyntheticPlateGenerationRequest` defaults to use profile values.
   - Update service validation to call profile validation.
   - Delete `SyntheticPlateGenerationFormat` and
     `SyntheticPlateGenerationDefaults` from `agent.dto.plate`.

4. Move viewer reply parsing to runtime protocol.
   - Add typed payload classes for viewer state/payload replies if missing.
   - Make `ViewerWindowService` ask those classes to hydrate raw response dicts.
   - Delete local field-name classes.

5. Move runtime server classification to protocol.
   - Add server role/type field to pong/scan result if missing.
   - Update agent runtime service to consume typed role.
   - Delete `_EXECUTION_SERVER_NAMES`.

## AST Removal Gates

```bash
rg -n "class PlateInspectionComponentKind|class PlateFileQueryKind|class SyntheticPlateGenerationFormat|class SyntheticPlateGenerationDefaults" openhcs/agent/dto/plate.py
rg -n "PlateFileKind\\(query_kind.value\\)|PlateFileQueryKind\\(" openhcs/agent/services/plate_inspection_service.py openhcs/agent/services/plate_streaming_service.py
rg -n "class ViewerControlField|class ViewerLayerField|class ViewerPayloadField|class ViewerDescriptorField|class ViewerPayloadSummaryField" openhcs/agent/services/viewer_window_service.py
rg -n "_EXECUTION_SERVER_NAMES|ZMQExecutionServer|OpenHCSExecutionServer" openhcs/agent/services/runtime_server_service.py
```

Expected result: no matches outside tests that assert deletion.

## Tests

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_plate_streaming_service.py \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_mcp_server.py
```

Viewer-specific checks after protocol migration:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k viewer
```

## Implementation Progress

2026-06-29 runtime server classifier slice:

- Added `MessageFields.SERVER_TYPE` and `PongResponse.server_type` in
  `zmqruntime`, populated from the existing `ZMQServer.server_type()` runtime
  registration authority.
- The generic execution server pong now emits the same typed server role, so
  scans and direct pong reads carry the runtime server role instead of only a
  display class name.
- `RuntimeServerService` now validates execution endpoints by comparing the
  pong `server_type` to `ExecutionServer.server_type()`.
- Deleted the agent-local `_EXECUTION_SERVER_NAMES` class-name allow-list.

Evidence:

```bash
rg -n "_EXECUTION_SERVER_NAMES|ZMQExecutionServer|OpenHCSExecutionServer" \
  openhcs/agent/services/runtime_server_service.py
# no matches

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py -k runtime_server
# 3 passed

cd external/zmqruntime && ../../.venv/bin/python -m pytest tests/test_messages.py
# 4 passed
```

2026-06-29 plate file-kind query slice:

- Removed `PlateFileQueryKind` from `openhcs.agent.dto.plate`.
- Added core `PlateFileInventoryQuery` authority methods for query kind
  choices, ABI-value coercion, and all-file sentinel handling.
- `PlateFileQueryRequest`, `PlateFileStreamRequest`, and
  `PlateFileQueryRecordSummary` now carry `PlateFileKind | None` or
  `PlateFileKind` directly.
- `PlateInspectionService`, `PlateStreamingService`, MCP server tools, and the
  dev client now call `PlateFileInventoryQuery.kind_from_value()` /
  `kinds_for()` instead of converting through an agent enum.

Evidence:

```bash
rg -n "PlateFileQueryKind" openhcs tests
# no matches

rg -n "class PlateFileQueryKind|PlateFileKind\\(query_kind.value\\)|PlateFileQueryKind\\(" \
  openhcs/agent/dto/plate.py \
  openhcs/agent/services/plate_inspection_service.py \
  openhcs/agent/services/plate_streaming_service.py \
  openhcs/mcp/server.py \
  openhcs/mcp/dev_client.py \
  tests
# no matches

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_plate_streaming_service.py
# 20 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "plate_files or stream_selected_plate or query_selected_plate or agent_dto_package_exports"
# 15 passed
```

2026-06-29 plate component-axis slice:

- Removed `PlateInspectionComponentKind` from the agent DTO layer.
- `PlateInspectionComponentSummary` and inspection internals now use
  `AllComponents` directly.
- `MetadataComponentValueSet.values_for(AllComponents)` owns the projection
  from component declarations to named microscope metadata fields.
- Parsed filename component collection now iterates `AllComponents` instead of
  maintaining per-component accumulator fields.
- Added a regression assertion that every reported inspection component follows
  `AllComponents.ordered_names()`.

Evidence:

```bash
rg -n "class PlateInspectionComponentKind|PlateInspectionComponentKind" openhcs tests
# no matches

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_mcp_server.py::test_agent_dto_package_exports_mcp_debugging_contracts
# 16 passed
```

2026-06-29 synthetic generation profile slice:

- Added `openhcs.core.synthetic_plate_generation` as the production profile
  authority for synthetic plate defaults, validation bounds, and supported
  generation formats.
- Removed `SyntheticPlateGenerationFormat` and
  `SyntheticPlateGenerationDefaults` from `openhcs.agent.dto.plate` and the
  DTO export surface.
- `SyntheticPlateGenerationRequest` now takes defaults from
  `SYNTHETIC_PLATE_GENERATION_PROFILE.default_request`.
- `SyntheticPlateGenerationService` validates through the core profile and
  projects invalid-format error results through the profile default rather than
  re-parsing the same rejected value.
- MCP server and dev-client synthetic plate commands use the core profile for
  defaults, choices, and format coercion.

Evidence:

```bash
rg -n "SyntheticPlateGenerationDefaults|SyntheticPlateGenerationFormat" \
  openhcs tests -g '*.py'
# no matches

rg -n "class PlateInspectionComponentKind|class PlateFileQueryKind|class SyntheticPlateGenerationFormat|class SyntheticPlateGenerationDefaults" \
  openhcs/agent/dto/plate.py
# no matches

git diff --check -- \
  openhcs/core/synthetic_plate_generation.py \
  openhcs/agent/dto/plate.py \
  openhcs/agent/dto/__init__.py \
  openhcs/agent/services/synthetic_plate_service.py \
  openhcs/mcp/server.py \
  openhcs/mcp/dev_client.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_mcp_server.py
# clean

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_mcp_server.py \
  -k "synthetic or agent_dto_package_exports"
# 5 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py -k synthetic
# no selected tests in this checkout
```

2026-06-29 viewer response-field authority slice:

- Moved viewer control, layer, payload, payload-summary, and descriptor
  response-field declarations to `openhcs.runtime.viewer_protocol`.
- `ViewerWindowService` now imports those protocol-owned `str` enums instead
  of declaring local response-field classes.

Evidence:

```bash
rg -n "class ViewerControlField|class ViewerLayerField|class ViewerPayloadField|class ViewerDescriptorField|class ViewerPayloadSummaryField" \
  openhcs/agent/services/viewer_window_service.py
# no matches

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py -k viewer
# 23 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k viewer
# 57 passed
```
