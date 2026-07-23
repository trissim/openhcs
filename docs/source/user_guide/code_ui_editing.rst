Code and UI editing
===================

The desktop forms and generated Python are two projections of the same live
ObjectState-backed declarations. Pipeline code contains ``PipelineConfig`` and
``FunctionStep`` values; it is not a separate GUI model.

Use the code view attached to the relevant window or scope. A typical round
trip is:

1. edit a pipeline or configuration in the form;
2. open its code projection and inspect the generated declarations;
3. edit the code and apply it;
4. resolve any validation error before the live state is replaced;
5. compile again before execution.

Choose the document that owns the edited object. Pipeline Editor projects one
complete ``PipelineDocument``; a step editor projects one ``FunctionStep``; a
configuration window projects its typed config; Plate Manager projects the
multi-plate aggregate. The available Code controls and MCP document catalog are
discovered from those nominal owners rather than a copied button list.

Clean and resolved views
------------------------

A clean document keeps inherited/default lazy fields sparse, which is normally
the best form to edit and share. A resolved view includes effective inherited
values for inspection. Do not paste a resolved document back merely to change
one field: that can turn inherited values into explicit overrides.

Safe apply and recovery
-----------------------

Code-document reads return a revision token. Validate the edited source without
mutation, review the intended change, then apply against a freshly read token.
If the UI changed in the meantime, re-read and reapply the edit rather than
overwriting the newer state. Apply returns snapshot/revision facts that support
audit and undo; branches and time travel should be inspected before editing a
non-head state.

Dirty markers remain meaningful after a code edit: ``*`` marks unsaved state,
``_`` marks a value that differs from its default, and an inherited value can
have a resolved value even while its raw lazy field is ``None``. Use the owning
window's explicit save/commit action where persistence is required.

For an attached agent, the safe sequence is read, explain, obtain approval,
re-read, validate, apply with the fresh revision, retain the receipt, then poll
the relevant state surface. Small field-level ObjectState mutations have a
request token but no base-revision guard; use them only after a fresh field read
and verify immediately afterward. Prefer the code document for atomic related
changes.

pycodify owns generic Python source serialization and import/collision handling.
OpenHCS owns the mapping between UI scope identities, pipeline declarations,
and code documents. Applying code is therefore a validated state transition,
not arbitrary execution inside the widget.

See :doc:`../architecture/code_ui_interconversion` and the pycodify package
documentation for the two sides of the boundary.
