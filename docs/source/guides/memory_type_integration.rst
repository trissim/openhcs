Array and memory integration
============================

ArrayBridge owns framework detection, conversion, generic callable memory
metadata, framework-local device operations, stack utilities, dtype conversion,
and OOM policy. OpenHCS owns how those declarations participate in domain
callable contracts, compilation, and worker execution.

``openhcs.constants.MemoryType`` and ``openhcs.core.memory.MemoryType`` are
compatibility re-exports of ``arraybridge.MemoryType``. They must retain object
identity across those import paths so registry keys, annotations, serialized
declarations, and runtime dispatch never require an application-owned enum
translation.

Declaration boundary
--------------------

The decorators re-exported from ``openhcs.core.memory.decorators`` attach
ArrayBridge input, output, and execution memory metadata plus OpenHCS processing
semantics to a callable. Input and output declarations describe conversion
boundaries. ``execution_memory_type`` independently identifies the framework
that owns the function body; boundary types alone do not prove that ownership.
``CallableContract`` reads all three keys through ArrayBridge's
``MemoryContractAttribute`` owner. Generic metadata keys and conversion behavior
must not be copied into an OpenHCS registry.

.. code-block:: python

   from openhcs.core.memory.decorators import numpy
   from openhcs.processing.backends.lib_registry.unified_registry import (
       ProcessingContract,
   )

   @numpy(contract=ProcessingContract.PURE_2D)
   def normalize(image):
       return image

Compiler and runtime boundary
-----------------------------

During compilation, adjacent callable contracts determine required conversions.
Every invocation's input, output, and execution declarations contribute to the
step's framework footprint. The compiler resolves each required GPU framework
against its own device namespace and records typed conversion and
``FrameworkDeviceAssignment`` decisions on ``CompiledStepPlan``.

Workers consume those decisions and typed runtime image values. The execution
declaration selects the framework-owned device scope around the function body;
cleanup is applied to the exact compiled framework/device footprint. Workers do
not rediscover framework semantics from function names or use one application-
owned GPU registry for different framework namespaces.

``ProcessingContract`` is independent of memory type. It says whether a
callable is semantically plane-local or stack-dependent. Likewise,
``variable_components`` defines the transported stack axis and ``group_by``
defines post-stack grouping. See :doc:`../architecture/processing_semantics`.

Package ownership
-----------------

- :external+arraybridge:doc:`ArrayBridge documentation <index>`: framework
  conversion, callable memory metadata, and device utilities.
- :doc:`../architecture/external_foundations`: dependency boundaries.
- :doc:`../development/callable_artifact_authoring`: callable metadata in
  OpenHCS.
- :doc:`../architecture/gpu_resource_management`: OpenHCS worker planning.
