Respecting codebase architecture
================================

OpenHCS uses nominal declarations as semantic authorities. A change is
architecturally complete when a fact is declared at its owner and every generic
consumer derives that fact through the existing registry or strategy boundary.

Nominal registration alone is not enough. Before inventing a type, registry, or
``isinstance`` / ``elif`` dispatch ladder, classify it against the worked
examples in :doc:`../architecture/abstraction_lattices` (parallel lattice,
unnecessary nominal, nominal mirror, caller-side dispatch versus true owner
with leaf hooks and mixins).

Find the owner first
--------------------

Before changing architecture or semantics, search the domain terms and these
shared mechanisms:

.. code-block:: console

   rg -n "<domain terms>|__registry__|AutoRegisterMeta|RegistryConfig|RegistryFamily|MostDerivedContextStrategyMixin|NominalTypeKeyedStrategyMixin|EnumKeyedStrategyMixin" openhcs external

The likely owner is a module declaration, callable contract, artifact type,
runtime value type, measurement feature type, or strategy root. If no authority
exists, add it at that declaration boundary.

Registry discipline
-------------------

For a nominal family, the root registry is the catalog. Generic consumers
normally iterate ``Root.__registry__.values()`` or use the helpers in
``openhcs.core.registry_strategies``.

Do not add a second dictionary, set, ordered list, priority table, copied field
list, compatibility alias, or fallback chain for the same fact. Such mirrors
inevitably diverge from the types they describe.

Selection rules
---------------

Use the existing strategy mixin that matches the semantic key:

``EnumKeyedStrategyMixin``
   A declared enum member selects exactly one strategy.

``NominalTypeKeyedStrategyMixin``
   A runtime value's nominal type selects strategies by MRO specificity.

``MostDerivedContextStrategyMixin``
   A tuple of context types selects the most-derived compatible strategy.

Inheritance and MRO selection are preferable when behavior composes by type.
Do not replace them with name checks or manual priority numbers.

Package ownership
-----------------

Generic OpenHCS code must remain backend-blind. A compiler or executor should
not import a concrete CellProfiler, storage, array, or UI module merely to learn
its names or behavior. Put module-specific facts on the concrete declaration
and let generic code query the shared interface.

The extracted packages own generic configuration, memory conversion, storage,
registration, Qt reaction, signature analysis, transport, and code generation.
See :doc:`../architecture/external_foundations` before adding generic mechanics
to OpenHCS.

Failure behavior
----------------

Direct access is appropriate for fields guaranteed by a typed plan or nominal
contract. ``getattr(..., fallback)``, ``.get(..., fallback)``, path guessing,
and broad exception fallbacks hide invalid state. Validate once at the owning
boundary and let downstream code consume the validated type.

Review checklist
----------------

- Is there exactly one semantic authority?
- Does generic code query that authority without concrete imports?
- Is strategy selection type- or enum-driven instead of string-driven?
- Are compiler and runtime records typed instead of parallel dictionaries?
- Were obsolete mirrors removed after callers migrated?
- Do focused tests cover both registration and the generic consumer?
