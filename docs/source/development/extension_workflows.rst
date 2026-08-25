Extension workflows
===================

Choose an extension boundary by the semantic owner, not by the first caller
that needs the behavior.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Change
     - Owning boundary
   * - New processing callable
     - callable declaration and ``CallableContract``
   * - CellProfiler module behavior
     - ``CellProfilerModule`` declaration family
   * - New artifact kind
     - artifact type/spec and planning strategies
   * - New runtime payload
     - nominal value plus projection/store strategies
   * - New source selection behavior
     - source declaration or source strategy family
   * - New compiler decision
     - typed plan field and compiler phase
   * - Generic configuration mechanic
     - ObjectState
   * - Memory conversion
     - ArrayBridge
   * - Array framework, child import environment, or device discovery
     - ArrayBridge ``MemoryType`` declaration; OpenHCS projects its process
       environment and compiles callable roles into per-step device bindings
   * - Storage, format, or ROI
     - PolyStore
   * - Image pixel/container semantics
     - OpenHCS ``ImageFileFormat`` family
   * - Generic registration
     - metaclass-registry
   * - Generic Qt reaction
     - pyqt-reactive
   * - Signature analysis or generic callable-declaration projection
     - python-introspect
   * - Transport/process protocol
     - ZMQRuntime
   * - Python serialization
     - pycodify
   * - JSON transport projection
     - OpenHCS ``serialization.json.to_jsonable``

Common workflow
---------------

1. Search for the nominal root, its registry, and current consumers.
2. State the invariant at the declaration that owns it.
3. Add or extend the matching strategy family when behavior varies by enum,
   value type, or context types.
4. Carry the result through an existing typed plan. Add a typed field if the
   fact genuinely survives into compilation or execution.
5. Remove any mirror superseded by the authority.
6. Test the owner, selection rule, and one generic consumer.

Compiler/runtime split
----------------------

Mutable ObjectState is resolved once. Compiler phases consume the resolved
step through ``StepSnapshot`` and ``CompilationSession``. Workers consume
``CompiledExecutionBundle`` and must not reinterpret UI state or callable
attributes.

Framework devices follow the same split. Callable decorators declare input,
output, and execution memory roles. Compilation derives one framework-local
device map for the pipeline and selects the exact subset on each
``CompiledStepPlan``; workers consume that map without a global GPU scheduler.

Storage and transport split
---------------------------

Runtime values keep semantic identity independent of persistence and transport.
OpenHCS owns ``ArtifactKey`` and domain payload meaning. PolyStore owns generic
storage mechanics, ArrayBridge owns memory conversion, and ZMQRuntime owns
generic transport mechanics.

Serialization boundaries
------------------------

Python source generation and JSON transport are different contracts.
pycodify owns executable Python source, imports, and formatter composition.
OpenHCS ``to_jsonable`` owns the JSON-native projection used by agent
capabilities and UI-bridge transport. It uses ``singledispatch`` for mappings,
sequences, callables, registered nominal types, enums, paths, and scalars.

Add a ``to_jsonable.register`` implementation at the OpenHCS value type only
when that type has a stable public transport representation. Do not make
pycodify formatters or generic JSON consumers guess domain identity, and never
use either projection to reconstruct compiler semantics that belong in typed
plans.

Further guides
--------------

- :doc:`callable_artifact_authoring`
- :doc:`cellprofiler_module_authoring`
- :doc:`source_binding_extension`
- :doc:`compiler_extension`
- :doc:`runtime_value_extension`
