Runtime System Assembly Rules
=============================

This guide is for agents assembling, debugging, or extending OpenHCS runtime
workflows. The goal is to choose the existing owner of each fact before adding
code, so pipeline generation, compilation, execution, viewers, and MCP agree.

Authority Order
---------------

Use this order when deciding where behavior belongs:

1. A concrete declaration owns its own semantics.
2. A nominal parent or mixin owns shared behavior for a real family.
3. Generic compiler, generator, runtime, and MCP code query those declarations.
4. Adapter code projects typed service results; it does not become a semantic
   owner.

For CellProfiler modules, the concrete owner is the backend
``CellProfilerModule`` declaration under
``openhcs.processing.backends.cellprofiler``. Module-specific settings,
function binding, artifact contracts, required processing axes, debug view
family, runtime policies, retained artifacts, measurement rows, relationship
rows, infrastructure behavior, and export behavior belong on that declaration
or on an inherited nominal policy family.

Do not add a parallel table, enum, all-caps setting constant, module-name
branch, or compatibility registry for a fact already represented by a module
type. If another system needs that fact, it should query the declaration or the
selected policy object.

FunctionStep Runtime Shape
--------------------------

``FunctionStep`` is the runtime declaration that tells the compiler how to
build and fan out work:

* ``variable_components`` are the axes actually stacked into the array passed to
  the callable.
* ``group_by`` is the partition or fanout axis for dictionary routing and
  grouped execution.
* source and output identity projection should be derived from actual
  stack/provenance metadata after compilation and execution.

Do not author or generate ``source_identity_stack_axes``,
``source_identity_stack_components``, or aliases of that concept. They are
parallel declarations and can disagree with the stack. If source identity is
needed, derive it from the resolved stack and runtime provenance.

Assembly Flow
-------------

Use the runtime systems in this order:

1. Resolve domain input through the microscope handler, source schema, or
   CellProfiler parser.
2. Build typed pipeline declarations, normally ``FunctionStep`` objects.
3. Compile the pipeline so ObjectState, source bindings, special I/O,
   materialization, memory contracts, and resource plans are resolved once.
4. Execute the compiled plan through the orchestrator or runtime server.
5. Project results through typed runtime records, viewer payloads, measurement
   tables, or MCP DTOs.

Agents should not skip the compiler by hand-matching paths, hand-building
payload records, or guessing source bindings. The compiler-owned plan is the
contract that runtime execution consumes.

Source Binding And Artifacts
----------------------------

Source binding is a compiler/runtime contract, not string matching in adapters.
Use the existing source binding records and artifact contracts:

* source schema compilers and pipeline importers declare available sources;
* module declarations declare required and produced artifact roles;
* the compiler resolves source bindings and materialization paths;
* runtime artifact lineage is derived from the compiled contract and observed
  execution records.

For CellProfiler imports, the pipeline generator lowers parser output into
OpenHCS declarations. It should not own module semantics. It asks the
``CellProfilerModule`` declaration for settings binding, function resolution,
artifact contract, invocation options, processing components, infrastructure
handling, and runtime export behavior.

Runtime Policy Selection
------------------------

Runtime policy families are allowed when they are nominal owners of one
mechanic, such as object input binding, special input binding, measurement row
projection, output recording, relationship rows, or execution mode. They should
be selected from the module declaration or a declaration-owned inherited policy.

The runtime executor should consume selected policy objects and execute a
generic plan. It should not repeatedly rediscover behavior through module-name
branches or duplicate registries.

MCP Usage Rules
---------------

MCP is the transport boundary over ``openhcs.agent`` services. When an agent is
building or debugging a workflow through MCP:

* query ``openhcs://knowledge`` or ``openhcs_search_knowledge`` before guessing
  architecture;
* use capability and DTO surfaces exposed by ``openhcs.agent`` services;
* use runtime-server tools only for discovery, bounded status, and execution
  projection;
* use UI bridge tools only for typed UI state surfaces, not raw widgets or
  process internals;
* follow returned source paths when a behavior needs code inspection.

MCP should expose current service projections. It should not expose raw
``PipelineOrchestrator``, ``FileManager``, ``ProcessingContext``, PyQt widgets,
arbitrary paths, or unbounded artifacts as first-class public objects.

Refactor Rules
--------------

When extending runtime behavior:

* Search for an existing typed owner first.
* Move module-specific facts onto the module declaration or an inherited
  nominal parent.
* Put shared mechanics in parent classes or existing policy families.
* Let generic generator/compiler/runtime code query the selected type.
* Delete obsolete mirrors after call sites have moved.
* Run focused tests for the touched runtime family and at least one importer or
  pipeline generation test when CellProfiler behavior changes.

If a new rule seems to need a global module-name table, stop and identify the
nominal type whose instances already define that family. The registry should be
the declaration registry, not a second catalog of the same semantics.
