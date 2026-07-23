# MCP Dev-Client Ergonomics Plan

Date: 2026-06-29

## Problem

The dev client is the easiest way to test fresh-process MCP behavior and agent
ergonomics. It should make common first-use commands obvious. For example,
`authoring-context first_use` should not fail if the intended command is
`authoring-context --kind first_use`.

This is dev-client UX friction. It should not change MCP server APIs or create
parallel tool semantics.

## Existing Authorities

- `AgentCapabilityDeclaration`
- capability `cli_command` declarations
- capability groups, roles, workflow stages, target contexts
- `McpDevCommandSpec`
- `SingleToolCommandSpec`
- `CapabilityBackedCommandSpec`
- command renderers
- request DTO `from_fields` constructors

## Target Shape

The dev client should derive command identity and request behavior from
capability and command declarations:

```text
AgentCapabilityDeclaration
    -> command spec
    -> argparse/CLI affordance
    -> request DTO construction
    -> renderer
```

It may add aliases and positional conveniences, but those conveniences belong to
command specs or CLI parsing policy. They must not alter server tool contracts.

## Nominal Iteration Authority

If implementation needs tool names, command names, workflow groups, target
contexts, or mutating/side-effect facts, iterate
`AgentCapabilityDeclaration.__registry__` through the existing capability
registry helpers. Do not duplicate capability names in command specs.

If implementation needs command behavior, iterate `McpDevCommandSpec`
declarations and `CapabilityBackedCommandSpec` mappings. Composite commands must
declare themselves as dev-client orchestration, not semantic authorities.
The 2026-06-29 dry run showed `McpDevCommandSpec.__registry__` only contains
`tools` before command modules are imported; implementation must import the
command modules or use capability-backed lookup instead of assuming the registry
is fully populated early in process startup.

If implementation needs request fields, iterate dataclass fields on the
capability input contract or call its `from_fields` constructor. Do not add
command-specific parsers for config/function/source semantics.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed `agent_capability_declarations()` returns the populated
capability list and `CapabilityBackedCommandSpec.for_capability_name` exists.
It also exposed the command-module import-order caveat above.

## Implementation Steps

1. Audit command specs with first-use dry runs.
   - `authoring-context first_use`
   - `knowledge-search "source bindings"`
   - `artifact-plan ...`
   - selected-plate UI commands
2. Fix argument affordances in command specs.
   - Add positional alias for `authoring-context KIND` if desired.
   - Keep `--kind` as canonical request field.
3. Derive tool names from capability declarations.
   - No command should hardcode MCP tool names when a capability declaration owns
     the name.
4. Keep renderers presentation-only.
   - Renderers should format response DTOs.
   - Renderers should not infer OpenHCS semantics.
5. Add CLI smoke tests.
   - Prefer parser/command-spec tests over live MCP where possible.
   - Keep one fresh-process smoke check for important first-use commands.

## Mirror Traps To Avoid

- Do not change MCP server APIs to accommodate CLI shorthand.
- Do not duplicate capability names in command specs when a declaration exists.
- Do not put semantic decisions in renderers.
- Do not add command-specific parsers for config/function/source semantics.
- Do not make fresh-process MCP diagnostics depend on a running UI bridge.

## Semantic Mirroring Audit

Audit questions:

- Does each simple command resolve its tool from an `AgentCapabilityDeclaration`
  or a `CapabilityBackedCommandSpec`?
- Are positional conveniences converted into request DTO fields without changing
  the MCP server contract?
- Do renderers only format response DTOs?
- Are composite commands explicitly marked as dev-client workflow UX rather than
  semantic authorities?

Hard failures:

- A command spec duplicates an MCP tool name already declared by a capability.
- A renderer decides source, artifact, config, function, or UI semantics.
- A CLI alias requires a new server tool or server-side compatibility shim.
- Fresh-process health checks assume a UI bridge is running.

AST/rg audit:

```bash
rg -n "openhcs_[a-z_]+|get_agent_capability\\(|CapabilityBackedCommandSpec|SingleToolCommandSpec" openhcs/mcp/dev_client_commands openhcs/mcp/dev_client_commanding.py
rg -n "if .*payload|if .*artifact|if .*source|if .*config|if .*function" openhcs/mcp/dev_client_renderers openhcs/mcp/dev_client_commands
rg -n "authoring-context|cli_command" openhcs/agent/capabilities.py openhcs/mcp/dev_client_commands
```

Allowed hardcoded command names are composite dev-client commands with no
one-to-one capability, and they must be documented as UX orchestration only.

## Verification

Search gates:

```bash
rg -n "openhcs_.*\\\"|get_agent_capability\\(\" openhcs/mcp/dev_client_commands openhcs/mcp/dev_client_commanding.py
rg -n "authoring-context|cli_command" openhcs/agent/capabilities.py openhcs/mcp/dev_client_commands
```

Expected result:

- Command specs either reference capability declarations or are explicitly
  non-tool composite commands.
- UX aliases are local to dev-client parsing.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py
```

Fresh-process checks:

```bash
.venv/bin/python -m openhcs.mcp.dev_client health
.venv/bin/python -m openhcs.mcp.dev_client authoring-context first_use
.venv/bin/python -m openhcs.mcp.dev_client authoring-context --kind first_use
```
