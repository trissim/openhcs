Code/UI Interconversion System
==============================

Overview
--------

OpenHCS supports live bidirectional editing between UI state and Python code.
The goal is not to maintain two independent representations or to export a
script that is later re-imported into a passive GUI. The goal is to expose one
ObjectState-backed workflow through both interactive UI controls and reviewable
pycodified code documents.

The practical model is a reflected object system. Objects that are reflected in
the UI are backed by typed ObjectState scopes, fields, or code-document
projections. A user or agent can edit those objects through form controls,
field-level ObjectState tools, or code mode, and the accepted edit applies back
to the running UI state. Code-document apply has revision, confirmation, and
snapshot protection. Field-level mutation is a narrower operation with a
request token but no base-revision guard, so it requires a fresh read and
immediate post-mutation verification.

This lets a user or agent:

* inspect and edit pipeline/config state in the UI;
* read the same state as Python code;
* validate edited code without mutating UI state;
* apply code back into the running UI in real time with revision protection;
* preserve provenance through ObjectState snapshots, branches, dirty markers,
  and time travel.

Current Authority Model
-----------------------

The current model is:

.. code-block:: text

   ObjectState scopes and typed UI services
       -> code document projection
       -> reviewed Python edit
       -> validation
       -> revision-checked apply
       -> ObjectState update + snapshot/provenance

ObjectState is the UI state authority. Code documents are typed projections of
that state. Raw widgets and temporary editor files are not semantic
authorities.

If an object is reflected in the UI, agents should first look for its typed
ObjectState scope, state surface, code document, or semantic UI action. Generic
widget scraping is a fallback for interaction mechanics, not the model of the
application.

Code Documents
--------------

A code document is a named projection of an ObjectState-backed UI surface. The
Plate Manager orchestrator document, for example, projects selected plate paths,
pipeline data, and relevant configs into pycodified Python.

Every actual Code action has a nominal semantic document owned by the edited
surface. Pipeline code uses ``PipelineDocument``; configuration and individual
step editors use their config/step document authorities; Plate Manager owns a
multi-plate aggregate that composes per-plate pipeline values. Generic window
drivers, bridge DTOs, and editor widgets only project those documents.

Coverage is declaration-driven. Adding a Code action must add or inherit its
document contract at the same owning declaration boundary; generic catalogs
discover registered providers. Do not add a second button-to-document table,
copied export-name list, type switch, or fallback chain. A document's field
names, rendering, parsing, validation, and apply semantics come from that one
nominal owner.

Code mode is therefore a live editing surface over the same UI object graph. It
is useful when structured Python is clearer than clicking through nested forms,
but it still participates in ObjectState revision checks, validation,
snapshots, and UI status updates.

Agents should use the code-document tools rather than scraping widgets:

* list available code documents;
* read a code document, optionally as sparse clean source;
* validate edited source against the document policy;
* apply edited source with the observed revision token;
* inspect the returned snapshot, undo target, revision token, and status.

Revision tokens prevent stale edits from overwriting newer UI state. If the UI
has changed since a document was read, agents should re-read, re-validate, and
apply against the current revision.

ObjectState Scopes And Fields
-----------------------------

ObjectState stores UI-editable values in typed scopes. Field projections expose:

* raw stored values;
* resolved inherited values;
* dirty/default markers;
* provenance and scope identity;
* field descriptions from the underlying Python type.

ObjectState field mutation is the narrow projection for one field; a code
document is the revision-protected projection for a related configuration or
pipeline change that is clearer to review as Python.

These two write paths do not have identical concurrency contracts. A code
document apply requires its observed base revision, has an explicit confirmation
policy, rejects time-travel edits unless opted in, and returns pre/post/undo
snapshot facts. A field mutation has an idempotency request token but does not
carry a base revision. Before a field mutation, confirm the UI is at branch head,
re-read the exact field and scope, apply one small approved change, then re-read
the field and related state surface. Prefer a code document when several fields
must remain atomic or concurrent UI edits are plausible.

UI-owned and headless boundaries
--------------------------------

The UI bridge is the only agent path that projects mutations into the visible
desktop ObjectState. Its state surfaces, code documents, selection-revision
tokens, semantic actions, and operation status all refer to that same running
object graph. Keeping a visible workflow on this path preserves Plate Manager
rows, snapshots, selected state, status projection, and output auto-add
behaviour.

Headless sessions are different. They can compile and execute pipelines, but
they do not update the visible Plate Manager state unless the workflow is routed
through the UI bridge.

Round-Trip Expectations
-----------------------

Round-trip fidelity means:

* generated code uses stable OpenHCS imports and pycodified constructors;
* omitted lazy fields remain omitted rather than being materialised as concrete
  defaults;
* explicitly edited fields map back to typed ObjectState fields;
* validation catches syntax, import, type, and policy errors before mutation;
* apply returns enough snapshot/revision data to audit or undo the change.

If a field's meaning is unclear, use ObjectState field-help tools or config
schema reflection before editing. Do not infer behaviour from widget labels or
raw ``None`` values.

Related documentation
---------------------

- :doc:`../user_guide/code_ui_editing` for the task-oriented round trip
- :doc:`../user_guide/mcp_clients` for client setup and safe attached-agent use
- :doc:`../concepts/core_model`
- :doc:`system_overview`
- :doc:`pipeline_compilation_system`
- :doc:`../development/runtime_system_assembly_rules`
- :doc:`../guide_for_biologists/domain_expert_onboarding`

The installed MCP knowledge base projects these sources through its registered
document manifest; it is not a second prose authority.
