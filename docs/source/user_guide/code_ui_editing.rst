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

A Plate Manager document also carries its read-time selection scope. An
all-plates document represents the complete visible collection, so applying it
synchronises that collection. A selected document may replace only the named
plates and leaves every unselected plate unchanged. Its payload must contain the
same plate scope IDs that were read; read a fresh document to choose a different
scope.

Pipeline Editor code must define ``pipeline_steps``. You may omit
``pipeline_config`` when the pipeline uses the default ``PipelineConfig()``;
generated code includes the explicit assignment again. Provide
``pipeline_config`` when the pipeline needs non-default source, processing,
viewer, or materialisation settings.

Applying a complete Pipeline Editor or Plate Manager code document reconciles
the submitted declaration with the live ObjectState graph. Unchanged and
unambiguously edited occurrences retain their history across reordering; added
occurrences receive new scopes and omitted occurrences are removed. Ambiguous
duplicate edits receive new scopes instead of inheriting state by list position.

For the Pipeline Editor, a successful code apply is also the commit action. It
advances the saved baseline of the reconciled pipeline, each active step, and
each nested function parameter state. There is no transient post-apply pipeline
value waiting for a separate Save button. A step or configuration editor that
provides an explicit save action retains that smaller editor's normal dirty-state
workflow.

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
have a resolved value even while its raw lazy field is ``None``. Pipeline Editor
code apply clears the unsaved marker for its reconciled graph. On surfaces with
an explicit save or commit action, use that owning action where persistence is
required.

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
