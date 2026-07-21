GPU resource management
=======================

GPU requirements originate in callable contracts. Memory decorators declare
input and output frameworks; compilation validates installed capabilities,
plans conversions, and assigns a device on ``CompiledStepPlan``.

Startup registry
----------------

``setup_global_gpu_registry`` initializes the process-wide GPU inventory once
from ``GlobalPipelineConfig``. It records detected device identifiers and the
configured concurrency capacity. The registry is initialization state, not a
runtime slot allocator.

Compile-time assignment
-----------------------

The compiler queries callable contracts and ArrayBridge capabilities. A plan
whose input or output memory type is GPU-backed receives an explicit ``gpu_id``
or fails validation. There is no runtime fallback to a different framework or
device.

``WorkerStartExecutionFacts`` projects only worker-start-relevant facts from
the typed plans. ``resolve_worker_start_context`` then enforces a CUDA-safe
start method.

Runtime boundary
----------------

Workers activate the compiled device and use ArrayBridge for conversion,
movement, OOM behavior, and framework cleanup. OpenHCS owns scheduling and
worker lifecycle; it does not copy ArrayBridge's framework table or inspect
concrete array classes in the compiler.

CPU-only discovery is selected with ``OPENHCS_CPU_ONLY=true``. A GPU-only
pipeline must fail explicitly when the capability is unavailable.

See :doc:`../guides/memory_type_integration` and
:doc:`../development/pyclesperanto_simple_implementation`.
