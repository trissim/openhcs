GPU resource management
=======================

GPU requirements originate in callable contracts. Memory decorators declare
input, output, and execution frameworks. Compilation validates the exact
declared footprint and records framework-local device bindings on
``CompiledStepPlan``.

Declaration-owned discovery
---------------------------

ArrayBridge's ``MemoryType`` members own optional-framework import policy,
device discovery, device scopes, and allocator cleanup. OpenHCS requests those
capabilities from the declarations rather than maintaining a second framework
registry. Device resolution imports only frameworks present in the compiled
pipeline. Catalogue preparation may import backend modules to discover their
callables; worker execution resolves transported callables directly instead of
preparing the complete catalogue.

The desktop ``--no-gpu`` option and ``OPENHCS_CPU_ONLY=true`` reject a GPU
footprint before optional frameworks are imported.

Compile-time assignment
-----------------------

``FrameworkDeviceResolver`` queries callable contracts and ArrayBridge
capabilities. Each GPU-backed input, output, or execution declaration receives
its own framework-local device binding or compilation fails. Identifiers from
different frameworks are not treated as proof that they refer to one physical
device.

``WorkerStartExecutionFacts`` projects only worker-start-relevant facts from
the typed plans. ``resolve_worker_start_context`` then enforces a CUDA-safe
start method.

Runtime boundary
----------------

Workers ask the compiled plan for the target framework's device during
conversion and for the execution framework's device scope during invocation.
Process-isolated and inline workers clean their compiled framework/device
footprint after each context. Thread-backed lanes share process-global storage
and allocator caches, so the parent cleans their merged footprint only after all
lanes have joined. OpenHCS owns compilation and worker lifecycle; it does not
copy ArrayBridge's framework table or inspect concrete array classes in the
compiler.

CPU-only discovery is selected with ``OPENHCS_CPU_ONLY=true`` or, for the
desktop application, ``--no-gpu``. A GPU-only pipeline must fail explicitly
when the capability is unavailable.

See :doc:`../guides/memory_type_integration` and
:doc:`../development/pyclesperanto_simple_implementation`.
