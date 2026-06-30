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
to the running UI state with revision and snapshot protection.

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

Use ObjectState field mutation for precise field-level edits. Use code
documents when pipeline/config changes are easier to review as Python.

UI-Owned Workflow
-----------------

When the running UI should visibly own the workflow, agents should keep all
state changes inside the UI bridge path:

1. Discover the bridge and list state surfaces.
2. Read the Plate Manager state surface.
3. Read the relevant code document.
4. Validate edited code.
5. Apply with the current revision token.
6. Dispatch init, compile, or run through the selected-plate workflow action.
7. Poll state surfaces and operation status.
8. Inspect selected/source/output plate files and viewer payloads.

This preserves Plate Manager rows, ObjectState snapshots, selected state,
status projection, and output auto-add behavior.

Headless sessions are different. They can compile and execute pipelines, but
they do not update the visible Plate Manager state unless the workflow is routed
through the UI bridge.

Round-Trip Expectations
-----------------------

Round-trip fidelity means:

* generated code uses stable OpenHCS imports and pycodified constructors;
* omitted lazy fields remain omitted rather than being materialized as concrete
  defaults;
* explicitly edited fields map back to typed ObjectState fields;
* validation catches syntax, import, type, and policy errors before mutation;
* apply returns enough snapshot/revision data to audit or undo the change.

If a field's meaning is unclear, use ObjectState field-help tools or config
schema reflection before editing. Do not infer behavior from widget labels or
raw ``None`` values.

Agent Rules
-----------

When editing UI-owned OpenHCS workflows:

* Treat ObjectState as the state authority.
* Treat code documents as live reviewable projections, not standalone scripts.
* Expect UI-reflected objects to be editable through typed fields, code
  documents, or semantic UI actions.
* Always validate before applying.
* Use revision tokens.
* Poll state surfaces after applying or dispatching workflow actions.
* Use snapshots, branches, and time-travel-head tools to understand provenance
  before mutating state.
* Prefer semantic UI actions and code documents over generic widget actions.

Related Knowledge
-----------------

Read these documents for the adjacent models:

* ``openhcs_core_model``
* ``openhcs_configuration_framework``
* ``openhcs_pipeline_compilation_system``
* ``openhcs_runtime_system_assembly_rules``
* ``openhcs_domain_expert_onboarding``
