Nominal ownership and registry strategies
=========================================

OpenHCS attaches semantics to the declaration that owns them. Registries provide
discovery; they are not permission to copy the same facts into consumer-owned
tables.

Ownership rule
--------------

Module-specific behavior belongs on the authoritative module declaration.
Callable execution behavior belongs on its callable contract. Artifact and
measurement behavior belongs on the artifact type or measurement feature
declaration. Runtime projection behavior belongs on a registered strategy for
the nominal runtime value.

Generic code queries these declarations. It must not import concrete backend
modules merely to learn names, maintain copied feature lists, or add fallback
chains that guess the intended owner.

Registry roots
--------------

An ``AutoRegisterMeta`` family has a nominal root and an authoritative
``__registry__``. Consumers normally iterate ``Root.__registry__.values()`` or
use a shared selection mixin. ``RegistryConfig`` and ``RegistryFamily`` describe
generic registration mechanics and are owned by metaclass-registry.

OpenHCS uses three shared selection models:

``EnumKeyedStrategyMixin``
  A concrete strategy declares the enum member it owns. The mixin derives the
  stable key and performs exact enum selection.

``NominalTypeKeyedStrategyMixin``
  A concrete strategy declares its owning Python type. Selection follows the
  runtime type MRO and chooses the nearest registered type.

``MostDerivedContextStrategyMixin``
  Strategies declare whether they match a context and express precedence through
  inheritance. Selection retains the single most-derived match. Local priority
  numbers and repeated ``if``/``elif`` chains are unnecessary.

Adding behavior
---------------

1. Locate the nominal root and existing consumer.
2. Extend the owning module, callable, artifact, measurement, or strategy
   declaration.
3. Use inheritance when the behavior is a specialization of an existing owner.
4. Make generic consumers call the shared declaration or selection surface.
5. Test discovery, ambiguity handling, serialization identity, and the behavior
   itself.

If no authority exists, add one at the declaration boundary. Do not create a
consumer-local registry as a temporary semantic mirror.

Worked wrong versus right lattices from the ``benchmark-platform`` cutover —
including parallel catalogs, unnecessary ``AutoRegisterMeta`` families, and
nominal mirrors that copy another owner — are documented in
:doc:`abstraction_lattices`.

Examples in the current architecture
------------------------------------

- ``ArtifactType`` and module artifact partitions own artifact roles.
- ``RuntimeSliceProjectionStrategy`` selects behavior by nominal runtime type.
- runtime equivalence feature profiles select the most-derived semantic owner
  for a feature context.
- ``ProcessingContract`` enum members delegate execution to nominal contract
  declarations.
- ``CellProfilerModule`` subclasses own module names, settings binding,
  callables, artifacts, and source-setup behavior.

Extracted ownership
-------------------

Generic metaclass and cache details belong in metaclass-registry documentation;
generic lazy-configuration behavior belongs in ObjectState documentation.
OpenHCS documents which domain declarations use those mechanisms and the
invariants consumers must preserve.
