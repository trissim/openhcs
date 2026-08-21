Compiler extensions
===================

Compiler phases operate on a ``CompilationSession`` and mutate typed
``CompiledStepPlan`` fields. They do not reinterpret ObjectState or maintain a
parallel dictionary plan.

Invariants
----------

``StepSnapshot`` binds one resolved step, compiler index, and ObjectState scope
identity. ``CompilationSession`` validates that steps, snapshots, states,
context, plans, source projection, and plate scope describe the same execution
axis.

Adding a decision
-----------------

1. Identify the declaration or previous typed plan field that supplies the
   input fact.
2. Add a compiler phase only if no existing phase owns the decision.
3. Store a result needed downstream in a typed plan field or typed plan object.
4. Validate the invariant when the result is constructed.
5. Make execution consume the compiled result without fallback discovery.

For framework resources, extend the callable or ArrayBridge declaration that
owns the memory role, then compile the derived binding onto
``CompiledStepPlan``. Do not restore a process-global GPU scheduler or validator
beside the typed plan.

Artifact planning uses exact ``ArtifactSpecRef`` identity. Source- or
metadata-satisfied inputs need not have runtime plans. A selector must only
exact-match a plan that is present.

Tests should cover the session invariant, the phase's typed output, one missing
or ambiguous declaration failure, and a minimal compile-to-bundle path.

See :doc:`../architecture/pipeline_compilation_system`.
