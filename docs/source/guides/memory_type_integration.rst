Array and memory integration
============================

ArrayBridge owns framework detection, conversion, device selection, stack
utilities, dtype conversion, and OOM policy. OpenHCS owns how those declarations
participate in callable contracts and compilation.

Declaration boundary
--------------------

The decorators re-exported from ``openhcs.core.memory.decorators`` attach
ArrayBridge input/output memory metadata plus OpenHCS processing semantics to a
callable. ``CallableContract`` is the compiler-facing reader of that metadata.
Generic conversion behavior must not be copied into an OpenHCS registry.

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

During compilation, adjacent callable contracts determine required conversions
and GPU availability. The compiler records typed conversion and device decisions
on ``CompiledStepPlan``. Workers consume those decisions and typed runtime image
values; they do not rediscover framework semantics from function names.

``ProcessingContract`` is independent of memory type. It says whether a
callable is semantically plane-local or stack-dependent. Likewise,
``variable_components`` defines the transported stack axis and ``group_by``
defines post-stack grouping. See :doc:`../architecture/processing_semantics`.

Package ownership
-----------------

- ArrayBridge documentation: framework conversion and memory utilities.
- :doc:`../architecture/external_foundations`: dependency boundaries.
- :doc:`../development/callable_artifact_authoring`: callable metadata in
  OpenHCS.
- :doc:`../architecture/gpu_resource_management`: OpenHCS worker planning.
