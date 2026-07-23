# MCP Function Contract Axis Semantics Plan

Date: 2026-06-29

## Problem

Agents need to know how each function or CellProfiler module treats axes:
pure 2D, full stack, flexible slice-by-slice/full-stack behavior, required
variable components, runtime-bound parameters, artifact inputs, and outputs.

This must come from function contracts and module declarations. MCP must not own
a registry of functions, modules, required variable components, or slice
behavior.

## Existing Authorities

- `CallableContract`
- function signatures in the unified registry
- nominal processing contract declarations
- `ProcessingContract` and `ProcessingContractDeclaration` in
  `openhcs.processing.backends.lib_registry.unified_registry`
- `runtime_bound_parameters`
- `required_variable_components`
- function/module artifact contracts
- `CellProfilerModule` declarations in
  `openhcs.processing.backends.cellprofiler.module_classes`
- `FunctionCatalogService._runtime_contract_summary`
- `FunctionRuntimeContractSummary`
- `FuncStepContractValidator`

## Target Shape

Function detail output should be the single agent-facing contract summary:

```text
registered function signature
    -> CallableContract
    -> processing contract declaration
    -> runtime-bound parameter declarations
    -> artifact input/output declarations
    -> CellProfiler module declaration when applicable
    -> FunctionRuntimeContractSummary
```

The catalog should clearly distinguish:

- execution semantics: pure 2D, full stack, or flexible;
- variable components required by the callable;
- `slice_by_slice` availability only where the contract supports flexible mode;
- batch execution as performance strategy, not semantic authority;
- artifact inputs and outputs with sidecar roles.

## Nominal Iteration Authority

If implementation needs functions, iterate the unified function registry through
the existing function catalog service. Do not maintain MCP-local function IDs or
signatures.

If implementation needs callable semantics, build `CallableContract` from the
registered callable and iterate its declared artifact specs,
runtime-bound-parameter declarations, required variable components, and
processing contract fields.

If implementation needs CellProfiler module-specific semantics, query the
registered `CellProfilerModule` declaration through
`CellProfilerModule.for_module(module_name)` or by iterating
`CellProfilerModule.__registry__.values()` where a full catalog is needed. Do
not maintain module-name lists in MCP, function catalog renderers, or authoring
context.

If implementation needs legal processing-contract variants, iterate the nominal
`ProcessingContract` enum from
`openhcs.processing.backends.lib_registry.unified_registry` and ask each member
for its `declaration`. Do not mirror pure-2D/full-stack/flexible as renderer
rules.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run corrected the authority path: `ProcessingContract` does not import
from `openhcs.core.callable_contract`; it imports from
`openhcs.processing.backends.lib_registry.unified_registry`. Current members are
`PURE_3D`, `PURE_2D`, `FLEXIBLE`, and `VOLUMETRIC_TO_SLICE`. The dry run also
confirmed `CellProfilerModule` is registry backed and exposes `for_module`.

## Implementation Steps

1. Audit current `FunctionRuntimeContractSummary`.
   - Identify missing fields for processing mode, variable component
     requirements, runtime-bound parameter declarations, and artifact roles.
2. Add missing DTO fields only as projections of `CallableContract` or module
   declarations.
   - Do not add new function/module lists.
3. Update `FunctionCatalogService._runtime_contract_summary`.
   - Query existing contract objects.
   - Query CellProfiler module declaration only through the registered module
     authority.
4. Update renderers and authoring context.
   - Make `describe-function` enough for an agent to choose
     `variable_components` and spot impossible steps.
5. Add compiler validation cross-link.
   - The MCP output should tell agents that compile-time validation enforces the
     same contract through `FuncStepContractValidator`.

## Mirror Traps To Avoid

- Do not add MCP-local lists of CellProfiler modules.
- Do not infer semantics from function names.
- Do not duplicate registered function signatures.
- Do not hardcode `slice_by_slice` behavior in renderers.
- Do not conflate batch execution with full-stack semantics.

## Semantic Mirroring Audit

Audit questions:

- Does each axis/stacking fact come from `CallableContract`, processing contract
  declarations, runtime-bound parameter declarations, or CellProfiler module
  declarations?
- Does function detail use registered signatures as the ABI authority?
- Does compile-time validation still use `FuncStepContractValidator` and the
  same contract fields shown to agents?
- Is `slice_by_slice` shown only when declared by the callable/module contract,
  not when guessed by name?

Hard failures:

- MCP/agent code owns a list of function IDs, CellProfiler modules, or required
  variable components.
- Renderer code decides pure-2D/full-stack/flexible semantics.
- A search alias or display label influences compiler/runtime behavior.
- `slice_by_slice` is treated as a generic kwarg rather than a contract-owned
  runtime parameter where applicable.

AST/rg audit:

```bash
rg -n "required_variable_components|declared_processing_contract|runtime_bound_parameters|CallableContract|FuncStepContractValidator" openhcs/agent openhcs/mcp openhcs/core
rg -n "CellProfiler.*= \\{|MODULE.*= \\{|slice_by_slice.*= \\{|function.*semantics.*= \\{" openhcs/agent openhcs/mcp
rg -n "if .*name.*slice|if .*function.*slice|if .*module.*slice" openhcs/agent openhcs/mcp
```

Allowed matches in MCP/agent code should be projection from contract objects or
display of DTO fields.

## Verification

Search gates:

```bash
rg -n "slice_by_slice|required_variable_components|declared_processing_contract" openhcs/agent/services/function_catalog_service.py openhcs/agent/dto/functions.py
rg -n "CellProfiler.*=.*\\{|MODULE.*=.*\\{|module.*registry" openhcs/mcp openhcs/agent
```

Expected result:

- Function contract data is projected from `CallableContract` and module
  declarations.
- No MCP file owns per-function or per-module semantic tables.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/test_cellprofiler_runtime_callable_introspection.py \
  tests/unit/agent/test_mcp_server.py
```
