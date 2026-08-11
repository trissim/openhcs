Abstraction lattices: one owner or a mirror?
============================================

OpenHCS uses nominal declarations so each semantic question has one owner.
Registries make those owners discoverable; they do not make a second catalogue of
the same facts legitimate. This distinction matters because a mirror can look
well structured, typed, and thoroughly tested while still requiring two places
to agree forever.

This page helps maintainers distinguish a real abstraction from a parallel
authority. It complements :doc:`nominal_ownership`, which describes the current
selection mechanisms, and
:doc:`../development/respecting_codebase_architecture`, which gives the
task-oriented change workflow.

The ownership test
------------------

Start with the semantic question an abstraction answers. Examples include:

- Which artifact does this callable produce?
- Which CellProfiler module owns this parsed module name?
- Which runtime value projection applies to this nominal value type?
- Which microscope handler recognises and projects this source layout?

A valid owner contains facts that cannot be obtained from another declaration.
A mirror derives or copies its answer from an existing owner and then exposes
the copy as authoritative. Deleting a mirror leaves the original question
answerable; deleting the true owner does not.

``AutoRegisterMeta`` and ``__registry__`` establish discovery mechanics, not
ownership. A registered family is justified only when its leaves own distinct
semantics and generic consumers query that family directly.

Four failure shapes
-------------------

Parallel lattice
  A second package, pipeline model, source language, or catalogue answers a
  question already owned elsewhere. Both surfaces must be kept in sync.

Unnecessary nominal abstraction
  A registry family adds identity and discovery but no distinct behaviour. A
  value object, mapping, or direct call would preserve every fact.

Nominal mirror
  A typed declaration or registry copies fields from another nominal owner.
  Strong typing does not remove the duplicated authority.

Caller-side dispatch
  A generic consumer selects concrete leaves through class names,
  ``isinstance`` ladders, enum ``if``/``elif`` chains, or local priority
  numbers. The caller has become a second description of the leaf hierarchy.

Current owner map
-----------------

The following boundaries are verified by current declarations and tests:

``CellProfilerModule``
  Module leaves own module names, settings binding, callable resolution, source
  contribution, execution scope, and module-specific artifact derivation.
  ``pipeline_import`` and compiler providers query those leaves rather than a
  module-name policy table.

``CallableContract`` and ``ArtifactSpecRelation``
  Callable and cross-artifact semantics live on these declarations and their
  nominal leaves. ``CompiledStepPlan`` adds execution consequences without
  becoming a second declaration graph.

``MicroscopeHandler`` and ``SourceBindingsConfig``
  These are distinct owners, not duplicates. A handler recognises and projects
  a physical or virtual source layout; source bindings select, name, and assign
  component identity to sources. ``SourceBindingsHandler`` composes both for
  arbitrary declared files.

Strategy selection mixins
  ``EnumKeyedStrategyMixin``, ``NominalTypeKeyedStrategyMixin``, and
  ``MostDerivedContextStrategyMixin`` centralise selection by enum member,
  nearest nominal type, or most-derived matching context. Leaves own
  specialisation; consumers make one call through the root.

``AgentCapabilityDeclaration``
  MCP tool identity, contracts, effects, security facts, and surface
  availability belong to the capability declaration registry. Transport and
  packaging artifacts are projections, not independent tool catalogues.

Worked boundaries
-----------------

CellProfiler translation
~~~~~~~~~~~~~~~~~~~~~~~~

The historical converter accumulated a function catalogue, compatibility
runtime, module-role tables, settings file trees, and generated-pipeline
sidecars. Those surfaces answered questions now owned by
``CellProfilerModule`` leaves, ``CallableContract``, public ``FunctionStep``
declarations, and the ordinary compiler/runtime path.

The current boundary is:

.. code-block:: text

   parsed ModuleBlock
       -> CellProfilerModule.require_module(name)
       -> declaration-owned settings, callable, source, and artifact semantics
       -> PipelineConfig + list[FunctionStep]
       -> ordinary compilation and runtime

``tests/unit/test_cellprofiler_static_deletion_gates.py`` protects the retired
parallel paths and symbols. The test is a migration guard, not a replacement
semantic registry: current behaviour must still be read from the surviving
declarations.

Source layouts and source bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A microscope handler and source bindings sometimes mention the same files, but
they answer different questions. The handler determines how a layout becomes a
workspace and which storage backends it supports. Source bindings determine
which declared files satisfy named semantic inputs. Combining them in
``SourceBindingsHandler`` is therefore composition of two authorities, not a
mirror.

Collapsing every source concern into either owner would lose information. A
valid native vendor layout can supply metadata without handwritten bindings;
an arbitrary folder can use bindings without pretending to be a vendor format.

Strategy family or simple mapping?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A strategy family is appropriate when registered leaves own meaningful
behaviour selected through one shared contract. Current
``RuntimePlaneAxisStrategy`` leaves, for example, own plane-index resolution,
axis cardinality, and post-projection axis semantics for each
``RuntimePlaneAxis``.

A static scheme-to-value association with no polymorphic behaviour does not
gain an owner merely by becoming an auto-registering class hierarchy. Historical
GrayToColor resolver declarations were collapsed for this reason. The durable
question is not how many leaves exist, but whether the family owns behaviour
that generic consumers need to select polymorphically.

Owner-side specialisation
-------------------------

Generic consumers call the abstract owner. Leaves provide methods or declared
data used by a root template:

.. code-block:: python

   # Incorrect: the consumer restates the leaf hierarchy.
   if mode is Mode.FIRST:
       result = first_algorithm(value)
   elif mode is Mode.SECOND:
       result = second_algorithm(value)

   # Correct: selection and behaviour belong to the registered family.
   result = ModeStrategy.for_enum_member(mode).apply(value)

An ``isinstance`` check is not automatically a dispatch violation. A root
template may validate a value against a leaf-declared ``entry_type``. The
violation occurs when generic code names the concrete leaf set and thereby owns
the selection lattice itself.

Likewise, an enum payload can own a small closed association, while a strategy
family is better when behaviour has its own inheritance, validation, or
multiple related operations. The architecture does not require a strategy for
every enum or forbid one-method leaves categorically.

Migration evidence
------------------

The Git history records why these boundaries exist:

- ``63f0ede1`` moved CellProfiler semantics onto module declarations;
- ``a1cdbe0d`` collapsed a duplicate Napari stream server;
- ``f39e132b`` removed a CellProfiler contract wrapper around an existing
  authority;
- ``a5f9d99d`` and ``f9af176b`` moved contract derivation into ordinary
  compilation;
- ``9434d847`` established the shared registry-selection strategies;
- ``4b3e0ce9`` collapsed microscope and materialisation helper registries.

These commits are historical evidence, not current API documentation. Verify a
proposed change against the current declarations and tests before applying the
same refactoring shape.

Diagnostic questions
--------------------

Before introducing a type, registry, strategy, or catalogue, ask:

1. What single semantic question will it answer?
2. Which current declaration already answers all or part of that question?
3. Would deleting the proposed surface leave the answer available elsewhere?
4. Does a generic consumer call one root, or does it restate concrete leaves?
5. Are transport, UI, documentation, and benchmark forms projections of the
   owner, or independently maintained copies?
6. If a migration replaces an owner, can the mirror and its last consumer be
   deleted in the same change?

If no current owner exists, add one at the declaration boundary. If an owner
does exist, extend its leaf hook, contract, or registered selection surface
instead of building a synchronised neighbour.
