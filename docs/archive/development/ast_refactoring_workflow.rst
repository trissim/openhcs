AST-Driven Refactoring Workflow
===============================

Use this workflow when a refactor touches many similar call sites, wrappers,
registries, or declaration projections. The goal is to make the structure of
the codebase drive the batch, instead of editing one instance at a time and
rediscovering the same decision repeatedly.

This is especially important in OpenHCS boundary code such as MCP adapters,
CellProfiler declarations, compiler contracts, and UI bridge projections. Those
layers must query existing nominal authorities. They must not grow parallel
tables, hardcoded string registries, or wrapper-local semantic branches.

Principles
----------

Start with an AST inventory, not manual search. ``rg`` is useful for locating
files, but AST is the right tool for grouping Python constructs by shape:
function definitions, class declarations, decorators, base classes, calls,
assignments, control flow, and constructor usage.

Use AST to build the work queue. The output of the first script should be a
bounded list of candidates with line spans, structural features, and suspected
owners. It should not be a one-off search result that is reinterpreted by hand
for each edit.

Classify before editing. A batch is only safe when every item in it has the
same structural role and the same semantic authority. For example, MCP wrappers
that only construct a request DTO and forward to a service can be moved to a
generated binding family. Wrappers that perform path-policy checks, payload
projection, UI selection resolution, or error composition stay separate until
that behavior has a typed owner.

Prefer declaration-owned generation. If a tool schema, runtime rule, default,
or option list can be derived from a dataclass, ``from_fields`` constructor,
``AutoRegisterMeta`` registry, function signature, or typed contract, do that.
Do not add a second dictionary that mirrors the same facts.

Use AST spans for removals. Once a group is selected, collect exact
``lineno``/``end_lineno`` spans from the parsed tree and delete whole
definitions. Avoid broad hand-matched hunks across unrelated code.

Keep strings at the ABI edge. Public MCP tool names, JSON field names, and
source-rendered names are strings by necessity. Decisions should be type-based
or declaration-based before that final boundary.

Do not treat AST shape as semantic authority. AST tells you that two blocks are
structurally similar. The batch is valid only after the actual authority is
identified: declaration class, DTO, function signature, config type, service
contract, or AutoRegisterMeta family.

The Loop
--------

Run refactors as a short feedback loop:

1. **Inventory**: parse the relevant files and report candidate definitions,
   decorators, bases, assignments, constructors, calls, and control flow.
2. **Partition**: split candidates by structural role and by semantic owner.
   If two candidates need different owners, they are different batches even if
   the code looks similar.
3. **Choose authority**: name the declaration, DTO, signature, or contract that
   owns each fact currently mirrored by the candidate code.
4. **Make authority load-bearing**: add the missing typed method, registry
   projection, dataclass field, or nominal base class before deleting mirrored
   code.
5. **Codemod mechanically**: use AST spans, AST-generated edits, or a small
   transformer over the selected batch. Avoid hand-editing each instance.
6. **Gate**: rerun AST inventory, focused tests, schema probes, and
   ``git diff --check``.

If a batch cannot name the authority in step 3, stop. The right next action is
an authority refactor, not a larger codemod.

Resume Protocol
---------------

AST-driven refactors should be resumable from repository artifacts, not from an
agent transcript. A later agent should be able to rerun the same query, see the
same candidate class, and understand why each candidate was edited, deferred, or
left explicit.

Before pausing or handing off, record four things in the relevant plan or dev
doc:

* the inventory query that produced the candidate set;
* the authority chosen for each batch;
* the exact AST gate that should be empty or should contain only named
  exceptions after the edit;
* the focused tests or schema probes that cover the public behavior.

When resuming, run the recorded inventory first. Do not start from the diff or
from a remembered list of files. The current checkout may have changed under a
parallel agent, and the AST inventory is the fastest way to distinguish already
completed work from remaining mirrors.

Then compare the current AST output to the handoff:

* candidates still emitted by the query remain in scope for that batch;
* candidates no longer emitted are already handled or moved and should be
  verified by the gate;
* new candidates emitted by the same query are only in scope if they share the
  same authority and structural role;
* candidates that require a new authority become a new batch.

This protocol keeps a resumed run mechanical. The agent does not need to infer
intent from local style or duplicate earlier reasoning. It reruns the query,
checks the authority ledger, applies the narrow batch, and proves the mirror is
gone with the same AST gate.

AST Inventory Pattern
---------------------

A useful first pass reports wrapper shape, constructors, service calls, and
control flow:

.. code-block:: bash

   . .venv/bin/activate
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/server.py").read_text())

   for node in ast.walk(tree):
       if not isinstance(node, ast.FunctionDef):
           continue
       if not node.name.startswith("openhcs_") or node.name == "openhcs_tool":
           continue

       calls = []
       constructors = []
       has_control_flow = False
       for child in ast.walk(node):
           if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.Match)):
               has_control_flow = True
           if isinstance(child, ast.Call):
               func = child.func
               if isinstance(func, ast.Name):
                   calls.append(func.id)
                   if func.id[:1].isupper():
                       constructors.append(func.id)
               elif isinstance(func, ast.Attribute):
                   calls.append(func.attr)

       print(
           node.lineno,
           node.end_lineno,
           node.name,
           "control_flow=", has_control_flow,
           "constructors=", constructors,
           "calls=", calls[:8],
       )
   PY

Use the output to form structural groups:

* ``direct_service_forward``: no control flow, returns ``to_jsonable`` around one
  service call.
* ``constructs_dto_then_forwards``: no meaningful branching, constructs a typed
  request/result DTO, then forwards to a service.
* ``stateful_or_projection_logic``: contains control flow, response reshaping,
  path-policy checks, UI selection resolution, or composed error/warning
  payloads.

Only the first two groups are candidates for immediate generated bindings. The
third group needs an owner decision first.

Authority Ledger Pattern
------------------------

Before editing, make the ownership explicit. A simple table in a scratch note,
plan, or script output is enough:

.. code-block:: text

   candidate                         mirrored fact                  authority
   -------------------------------   ----------------------------   -------------------------------
   openhcs_get_runtime_server_info   tool name/description          RuntimeServerInfoCapability
   openhcs_get_runtime_server_info   request fields/defaults        RuntimeServerInfoRequest
   openhcs_get_runtime_server_info   JSON output shape              RuntimeServerInfo DTO
   RuntimeInfoCommandSpec            renderer choice                RuntimeServerInfo DTO renderer

This prevents the common failure mode where a wrapper deletion leaves the same
metadata reintroduced as a manual dictionary somewhere else.

Authority Selection
-------------------

For each candidate group, identify the authority before coding:

* Tool identity and descriptions: ``AgentCapabilityDeclaration`` registry.
* Input schema and defaults: request DTO dataclass fields or
  ``Request.from_fields`` signatures.
* Enum/string coercion: the request DTO or the domain type that owns the enum.
* Function parameters: registered callable signatures and callable contracts.
* Viewer controls: ``Viewer*ControlOptions.from_overrides`` signatures.
* UI bridge connection fields: ``UiBridgeConnectionRequest`` and
  ``UiBridgeConnectionSpec``.
* Runtime submission timeouts: execution request DTOs and execution service
  constants.

If no existing owner exists, add the smallest request or policy DTO at the
agent boundary. Do not encode the same rule in the MCP wrapper.

Batch Editing Pattern
---------------------

For deletion batches, collect exact spans and apply them in reverse order:

.. code-block:: python

   import ast
   from pathlib import Path

   path = Path("openhcs/mcp/server.py")
   lines = path.read_text().splitlines()
   tree = ast.parse("\n".join(lines) + "\n")

   remove_names = {
       "openhcs_get_viewer_window_state",
       "openhcs_navigate_viewer_window",
   }
   spans = [
       (node.lineno, node.end_lineno, node.name)
       for node in ast.walk(tree)
       if isinstance(node, ast.FunctionDef) and node.name in remove_names
   ]

   for start, end, name in sorted(spans, reverse=True):
       del lines[start - 1:end]

   path.write_text("\n".join(lines) + "\n")

Prefer generating a patch from a local script and applying/reviewing the diff.
For repeated edits inside many classes, use an AST transformer or CST tool when
format preservation matters. The important property is that the edit is driven
by node identity and verified by a second AST pass.

Batch Refactor Pattern
----------------------

One clean MCP batch usually has these steps:

1. Add or reuse a request DTO or typed constructor that owns public defaults and
   coercion.
2. Add one generated binding family if the structural pattern repeats.
3. Add concrete binding declarations keyed by existing
   ``AgentCapabilityDeclaration`` classes.
4. Register the binding family once in ``build_server``.
5. Use AST to verify old ``openhcs_*`` wrapper definitions are gone and no
   duplicate tool names remain.
6. Run focused tests and schema smoke checks.

Deletion Gate
-------------

Before deleting a wrapper group, ask AST for exact spans:

.. code-block:: bash

   . .venv/bin/activate
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
   target_names = {
       "openhcs_get_viewer_window_state",
       "openhcs_navigate_viewer_window",
       "openhcs_get_viewer_window_payloads",
   }

   for node in ast.walk(tree):
       if isinstance(node, ast.FunctionDef) and node.name in target_names:
           print(node.name, node.lineno, node.end_lineno)
   PY

Delete whole definitions by those spans. Afterward, rerun the inventory and
assert the names are gone.

Verification Gate
-----------------

For MCP refactors, use this minimum loop:

.. code-block:: bash

   . .venv/bin/activate
   python -m py_compile openhcs/mcp/server.py openhcs/mcp/dev_client.py
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
   wrappers = [
       node.name
       for node in ast.walk(tree)
       if isinstance(node, ast.FunctionDef)
       and node.name.startswith("openhcs_")
       and node.name != "openhcs_tool"
   ]
   print("handwritten_openhcs_funcs", len(wrappers))
   PY
   pytest tests/unit/agent/test_mcp_server.py tests/unit/agent/test_capabilities.py -q
   git diff --check

When a binding changes public tool signatures, also build the FastMCP schema in
process and inspect the relevant tool properties. This catches drift in
required fields, defaults, and nested JSON schema generation.

Semantic Mirror Queries
-----------------------

Use small AST scripts to catch reintroduced mirrors. These are examples, not a
complete lint suite:

* Decorated ``openhcs_*`` functions in ``mcp/server.py`` after generated
  bindings should own those tools.
* All-caps assignment tables whose values are class lists, string names, or
  enum-kind maps next to AutoRegisterMeta families.
* Central command-name enums or string tables whose members are already declared
  on command/spec classes registered by AutoRegisterMeta.
* Command classes with identical ``render_response`` methods that select a
  renderer already implied by output DTO type.
* Operation namespaces that only point from an operation name to a contract
  class that already declares that operation name.
* ``if isinstance(...)`` ladders where a nominal family or declared method can
  own the behavior polymorphically.

The query result is not the refactor by itself. For each hit, name the real
authority and either route through it or document why the hit is an ABI edge.

Current MCP Cleanup Loop
------------------------

The MCP capability and dev-client cleanup uses the same loop, with concrete
authority gates:

* ``AgentCapabilityDeclaration`` owns MCP tool identity, CLI command metadata,
  input contract, and output contract.
* ``McpDevCommandSpec`` owns only CLI presentation mechanics: parser shape,
  renderer selection, and custom multi-call workflows.
* output DTO classes own renderer registry keys.
* UI bridge operation declarations own gateway calls and request/response
  contracts.

Start each pass by inventorying command classes by AST shape:

.. code-block:: bash

   . .venv/bin/activate
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())

   for node in tree.body:
       if not isinstance(node, ast.ClassDef) or not node.name.endswith("CommandSpec"):
           continue

       bases = [
           ast.unparse(base)
           for base in node.bases
       ]
       assignments = [
           target.id
           for stmt in node.body
           if isinstance(stmt, ast.Assign)
           for target in stmt.targets
           if isinstance(target, ast.Name)
       ]
       methods = [
           stmt.name
           for stmt in node.body
           if isinstance(stmt, ast.FunctionDef)
       ]

       print(node.name, node.lineno, bases, assignments, methods)
   PY

Then partition the result:

* capability-only leaves should be generated from capability declarations;
* no-input single-tool commands should share the generic generated command
  family;
* UI-bridge single-tool commands should share a generated UI-bridge command
  family selected by a declaration-owned command profile;
* commands with extra parser UX or multi-call behavior stay explicit until that
  behavior has a typed owner.

After a batch, run AST gates that prove the mirror did not move elsewhere:

.. code-block:: bash

   . .venv/bin/activate
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())

   local_command_metadata = []
   renderer_capability_keys = []

   for node in tree.body:
       if not isinstance(node, ast.ClassDef):
           continue
       has_capability = any(
           isinstance(stmt, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "capability" for target in stmt.targets)
           for stmt in node.body
       )
       for stmt in node.body:
           if not isinstance(stmt, ast.Assign):
               continue
           for target in stmt.targets:
               if has_capability and isinstance(target, ast.Name) and target.id in {"command", "aliases"}:
                   local_command_metadata.append((node.name, target.id, stmt.lineno))
               if (
                   isinstance(target, ast.Name)
                   and target.id == "output_contract"
                   and isinstance(stmt.value, ast.Attribute)
                   and isinstance(stmt.value.value, ast.Name)
                   and stmt.value.value.id == "agent_capabilities"
               ):
                   renderer_capability_keys.append((node.name, stmt.lineno))

   print("local_command_metadata", local_command_metadata)
   print("renderer_capability_keys", renderer_capability_keys)
   PY

Both lists should be empty. If either list has entries, the command layer is
still mirroring capability metadata or renderer ownership instead of querying
the declaration.

When removing explicit MCP wrappers, use the companion server gate:

.. code-block:: bash

   . .venv/bin/activate
   python - <<'PY'
   import ast
   from pathlib import Path

   tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
   wrappers = [
       node.name
       for node in ast.walk(tree)
       if isinstance(node, ast.FunctionDef)
       and node.name.startswith("openhcs_")
       and node.name != "openhcs_tool"
   ]
   print("handwritten_openhcs_funcs", wrappers)
   PY

The expected wrapper list is empty once the capability projection owns the MCP
tool surface.

Operational Discipline
----------------------

During an active shared refactor, use AST as the steering layer and patches as
the review layer.

* Use AST scripts for inventory, classification, span selection, and final
  gates. They answer structural questions quickly and keep the work from
  devolving into one-off string searches.
* Use ``apply_patch`` for committed source edits unless the edit is a purely
  mechanical generated rewrite. The reviewed diff is still the unit of change.
* Keep every batch tied to one authority migration: for example, "CLI argument
  defaults move from command classes to request DTOs" or "tool invocation moves
  from wrappers to capability bindings." Do not mix unrelated cleanup into the
  same AST pass.
* Leave a reproducible gate with the batch. A useful gate is a short AST query
  that would fail if the removed mirror reappears, plus focused tests for the
  public behavior touched by the batch.
* Treat failed gates as design feedback. If removing a mirror requires a manual
  allowlist, a duplicated field table, or an ``isinstance`` ladder over leaf
  behavior, the missing piece is usually a declaration-owned hook or a nominal
  base class.

For code in active collaboration, run the loop in small complete slices:

1. report the structural inventory and named authority;
2. add or strengthen the declaration/API that should own the behavior;
3. remove the mirrored consumer-side code by AST span or a narrow patch;
4. rerun the inventory as a regression gate;
5. run focused tests before moving to the next slice.

This keeps parallel agents from rediscovering the same smells in different
files and makes the remaining work obvious from the gate output.

Parallel-Agent Handoff Pattern
------------------------------

When multiple agents are working in the same boundary, the AST workflow should
leave resumable artifacts instead of prose-only intent.

Each handoff should include:

* the AST query used to build the candidate set;
* the exact candidate list emitted by that query;
* the semantic authority selected for the batch;
* the names intentionally left explicit and the reason they are not part of the
  batch;
* the AST gate that proves the mirror is gone after the edit;
* the focused tests that cover the public behavior.

This makes the next agent start from the same structural facts rather than from
a manual diff review. It also prevents accidental scope creep: if a candidate
is not emitted by the query or does not share the named authority, it belongs in
a separate batch.

For MCP binding work, a good handoff is concrete:

.. code-block:: text

   inventory: classes inheriting McpUiRequestToolBindingABC
   authority: AgentCapabilityDeclaration.connection_request_invocation
   generated: default-timeout UI requests with DTO input contracts
   explicit: command-timeout mutations and payload-projecting tools
   gate: removed class names absent; generated declarations match expected names
   tests: test_mcp_server.py selected UI request cases plus capability registry tests

The handoff should not say "delete similar wrappers" without the query and the
authority. Similar shape is only a starting signal; the authority is what makes
the batch correct.

Stop Conditions
---------------

Stop a batch and reassess when:

* a wrapper contains path-policy checks, UI selection resolution, or response
  projection;
* the candidate binding needs a manual map of tool names to service methods;
* the proposed request DTO would duplicate an existing domain contract;
* public defaults differ from lower-level domain defaults and the difference is
  not explicitly owned by an adapter policy;
* an edit requires ``getattr``/``hasattr`` or stringly ownership checks.

Those are signs that the next step is an authority refactor, not another MCP
wrapper deletion.
