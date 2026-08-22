Source-binding extensions
=========================

Extend sources through typed declarations and existing nominal strategy roots.
Do not add filename guessing to a compiler, importer, or runtime adapter.

Declaration boundary
--------------------

Use ``SourceFilterClause``, metadata extraction rules,
``NamedSourceBinding``, ``ImagePlaneSource``, and ``SourceBindingsConfig`` for
pipeline-visible declarations. Step selection belongs in
``StepSourceBindingsConfig``.

Compilation boundary
--------------------

The compiler emits ``CompiledSourceUniversePlan`` and
``CompiledSourceBindingPlan`` on the typed step plan. A new fact required by
workers must be represented there, not rediscovered from paths.

Strategy selection
------------------

Existing families select source universes, matching behavior, component
projection, and workspace projection. Add a concrete strategy to the correct
root and declare its enum/type/context key. Generic consumers should continue
to use the root registry or its strategy mixin.

Virtual workspaces
------------------

PolyStore owns ``SourcePixelRef`` and generic backend addressing. OpenHCS owns
microscope metadata and ``VirtualWorkspaceSourceProjection``, which maps
virtual paths to source references and provenance for one compilation session.

Verify declaration validation, strategy selection, physical and virtual path
cases, metadata identity, and a compiled source-satisfied artifact input.
When a registered store emits exact candidates, also verify that filename or
folder metadata rules project the declared biological address without changing
the backend-owned source address, and that reopening the persisted workspace
reports the projected component values and labels. When a component selector
and component identity use different values, verify that the selector matches
the original candidate while the projected address, compiler group keys, and
runtime provenance use the declared component identity.

See :doc:`../architecture/source_model`.
