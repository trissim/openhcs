OpenHCS Concurrency Model
=========================

Overview
--------

OpenHCS implements a well-level parallelism model with thread isolation
and immutable compilation artifacts. This design provides performance
while maintaining thread safety through architectural constraints rather
than complex locking mechanisms.

**Note**: This document describes the actual concurrency implementation.
Some advanced features like runtime GPU slot management are planned for
future development.

Core Concurrency Philosophy
----------------------------

**Well-Level Parallelism**
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Parallel Unit**: Each well is processed independently in its own
   thread
-  **Sequential Within Well**: All steps for a well execute sequentially
   in the same thread
-  **No Cross-Well Dependencies**: Wells never share data or coordinate
   during execution

**Immutable Compilation Artifacts**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Compile Once, Execute Many**: Step plans compiled once, then
   immutable during execution
-  **Frozen Contexts**: ProcessingContext frozen after compilation,
   preventing state corruption
-  **Stateless Steps**: Step objects stripped of mutable state after
   compilation

Concurrency Architecture
------------------------

**Two-Phase Execution Model**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Phase 1: Compilation (Single-threaded)
   compiled_contexts = orchestrator.compile_pipelines(pipeline_definition, wells)
   # Result: Immutable ProcessingContext for each well

   # Phase 2: Execution (Multi-threaded)
   results = orchestrator.execute_compiled_plate(pipeline_definition, compiled_contexts)
   # Result: Parallel execution across wells

**Thread Pool Execution**
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def execute_compiled_plate(self, pipeline_definition, compiled_contexts, max_workers=None):
       """Execute with configurable parallelism."""

       actual_max_workers = max_workers or self.global_config.num_workers

       if actual_max_workers > 1 and len(compiled_contexts) > 1:
           # Choose executor type based on global config
           if self.global_config.use_threading:
               # ThreadPoolExecutor for debugging
               executor = concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers)
           else:
               # ProcessPoolExecutor for true parallelism
               executor = concurrent.futures.ProcessPoolExecutor(max_workers=actual_max_workers)

           with executor:
               future_to_well_id = {
                   executor.submit(self._execute_single_well, pipeline_definition, context, visualizer): well_id
                   for well_id, context in compiled_contexts.items()
               }

               for future in concurrent.futures.as_completed(future_to_well_id):
                   well_id = future_to_well_id[future]
                   try:
                       result = future.result()
                       execution_results[well_id] = result
                   except Exception as exc:
                       logger.error(f"Well {well_id} exception: {exc}", exc_info=True)
                       execution_results[well_id] = {"status": "error", "error": str(exc)}
       else:
           # Sequential execution
           for well_id, context in compiled_contexts.items():
               execution_results[well_id] = self._execute_single_well(pipeline_definition, context, visualizer)

**🔒 Thread Safety Mechanisms**
-------------------------------

**1. Immutable State Design**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Frozen ProcessingContext**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   class ProcessingContext:
       """Context with immutability enforcement."""
       
       def freeze(self):
           """Make context immutable after compilation."""
           self._is_frozen = True
       
       def __setattr__(self, name, value):
           """Prevent modification of frozen context."""
           if getattr(self, '_is_frozen', False) and name != '_is_frozen':
               raise AttributeError(f"Cannot modify '{name}' of frozen ProcessingContext")
           super().__setattr__(name, value)

**Thread Safety Guarantee**: Frozen contexts cannot be modified,
eliminating race conditions.

**Stateless Step Design**
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Step objects are designed to be stateless after compilation
   # They read configuration from immutable step_plans in ProcessingContext
   # No mutable state is stored in step objects during execution
   class FunctionStep(AbstractStep):
       def process(self, context):
           # Read configuration from immutable context
           step_plan = context.step_plans[self.step_id]
           # All execution state comes from context, not step object

**Thread Safety Guarantee**: Step objects with no mutable state can be
safely shared across threads.

**2. Thread-Local Resource Isolation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FileManager Per Thread**
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   class FileManager:
       """FileManager with strict thread isolation."""
       
       def __init__(self, registry):
           # Thread Safety:
           #   Each FileManager instance must be scoped to a single execution context.
           #   Do NOT share FileManager instances across pipelines or threads.
           #   For isolation, create a dedicated registry for each FileManager.
           self.registry = registry
           self._backend_cache = {}  # Per-instance backend cache

**Thread Safety Guarantee**: Each thread gets its own FileManager
instance with isolated backend cache.

**Backend Instance Isolation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def _get_backend(self, backend_name):
       """Get backend with per-FileManager caching."""
       # Thread Safety:
       #   This method is thread-safe for a single FileManager instance.
       #   Backend instances are NOT shared across FileManager instances.
       if backend_name not in self._backend_cache:
           backend_class = self.registry[backend_name]
           self._backend_cache[backend_name] = backend_class()  # New instance per FileManager
       
       return self._backend_cache[backend_name]

**Thread Safety Guarantee**: Backend instances are never shared between
threads.

**3. Global Resource Coordination**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Thread-Safe GPU Registry**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**See**: `GPU Resource Management <gpu-resource-management.md>`__ for
comprehensive GPU coordination details.

**Concurrency Aspects**: - GPU registry access is thread-safe with
atomic operations - GPU assignment happens at compilation time, not
runtime - No runtime slot acquisition/release needed in current
implementation - Registry status queries are protected by locks for
consistency

**Thread Safety Guarantee**: GPU registry access is atomic and
consistent across threads.

**Memory Backend Isolation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   class MemoryStorageBackend(StorageBackend):
       """Memory backend with per-instance storage."""
       
       def __init__(self):
           self._memory_store = {}  # Per-instance memory store
           self._prefixes = set()   # Per-instance namespace tracking

**Thread Safety Guarantee**: Each thread gets its own memory backend
instance with isolated storage.

**🔄 Execution Flow Thread Safety**
-----------------------------------

**Single Well Execution**
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def _execute_single_well(self, pipeline_definition, context, visualizer):
       """Execute pipeline for single well - thread-safe by design."""

       # 1. Context is frozen (immutable)
       assert context.is_frozen()

       # 2. Each thread has its own FileManager
       filemanager = context.filemanager  # Thread-local instance

       # 3. GPU assignment handled at compilation time
       # No runtime GPU slot management needed

       try:
           # 4. Sequential step execution within thread
           for step in pipeline_definition:
               step.process(context)  # Step is stateless, context is immutable

           return {"status": "success", "well_id": context.well_id}

       except Exception as e:
           logger.error(f"Pipeline execution failed for well {context.well_id}: {e}")
           return {"status": "error", "well_id": context.well_id, "error": str(e)}

**FunctionStep Thread Safety**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def process(self, context):
       """FunctionStep execution - thread-safe by design."""
       
       # 1. Read immutable step plan
       step_plan = context.step_plans[self.step_id]  # Immutable after compilation
       
       # 2. Use thread-local FileManager
       filemanager = context.filemanager  # Thread-local instance
       
       # 3. Load data using isolated backends
       for file_path in matching_files:
           image = filemanager.load_image(file_path, read_backend)  # Isolated backend
           raw_slices.append(image)
       
       # 4. Process data (pure computation)
       result = func(image_stack)  # Function operates on local data
       
       # 5. Save data using isolated backends
       for i, slice_2d in enumerate(output_slices):
           filemanager.save_image(slice_2d, output_path, write_backend)  # Isolated backend

**🎯 Concurrency Guarantees**
-----------------------------

**What is Thread-Safe:**
~~~~~~~~~~~~~~~~~~~~~~~~

**✅ Immutable Data Structures**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Frozen ProcessingContext**: Cannot be modified after compilation
-  **Step Plans**: Immutable dictionaries with execution configuration
-  **Stateless Steps**: No mutable state after attribute stripping

**✅ Thread-Local Resources**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **FileManager Instances**: One per thread, never shared
-  **Backend Instances**: Isolated per FileManager
-  **Memory Storage**: Separate memory store per backend instance

**✅ Atomic Global Operations**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **GPU Registry Access**: Protected by locks for atomic updates
-  **Configuration Access**: Immutable configuration objects

**What Requires Coordination:**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**🔒 GPU Resource Management**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**See**: `GPU Resource Management <gpu-resource-management.md>`__ for
complete GPU coordination architecture.

**Concurrency Considerations**: - Registry status queries use atomic
reads with locks - GPU assignment during compilation phase (thread-safe)
- Registry initialization is one-time with thread-safe checks

**🔒 Global Configuration Updates**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Live Config Changes**: Coordinated through orchestrator
-  **Registry Initialization**: One-time setup with thread-safe checks

**⚡ Performance Characteristics**
----------------------------------

**Scalability Model**
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Optimal parallelism calculation
   max_workers = min(
       num_wells,                    # Don't create more threads than wells
       global_config.num_workers,    # Respect configured limit
       available_gpu_slots           # Don't exceed GPU capacity
   )

**Resource Utilization**
~~~~~~~~~~~~~~~~~~~~~~~~

-  **CPU Cores**: One thread per core (configurable)
-  **GPU Devices**: Multiple pipelines per GPU (based on memory
   capacity)
-  **Memory**: Isolated per thread, no sharing overhead
-  **I/O**: Parallel disk access across threads

**Contention Points**
~~~~~~~~~~~~~~~~~~~~~

-  **GPU Registry**: Minimal contention (fast lock operations)
-  **Disk I/O**: Natural parallelism across different directories
-  **Memory Allocation**: Thread-local, no contention

**🚀 Advanced Concurrency Features**
------------------------------------

**Exception Isolation**
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Exceptions in one well don't affect others
   for future in concurrent.futures.as_completed(future_to_well_id):
       well_id = future_to_well_id[future]
       try:
           result = future.result()
           execution_results[well_id] = result
       except Exception as exc:
           # Exception isolated to this well
           logger.error(f"Well {well_id} exception: {exc}", exc_info=True)
           execution_results[well_id] = {"status": "error", "error": str(exc)}
           # Other wells continue processing

**Resource Cleanup**
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def _execute_single_well(self, pipeline_definition, context, visualizer):
       """Guaranteed resource cleanup per thread."""

       try:
           # Execute pipeline steps
           for step in pipeline_definition:
               step.process(context)

           return {"status": "success", "well_id": context.well_id}

       except Exception as e:
           # Exception handling and cleanup
           logger.error(f"Pipeline execution failed for well {context.well_id}: {e}")
           return {"status": "error", "well_id": context.well_id, "error": str(e)}

**Graceful Degradation**
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Automatic fallback to sequential execution
   if actual_max_workers <= 1 or len(compiled_contexts) <= 1:
       logger.info("Executing wells sequentially")
       for well_id, context in compiled_contexts.items():
           execution_results[well_id] = self._execute_single_well(pipeline_definition, context, visualizer)

**🎯 Why This Model is Brilliant**
----------------------------------

**1. Eliminates Complex Locking**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Immutable State**: No need to lock shared data structures
-  **Thread Isolation**: No shared mutable resources between threads
-  **Minimal Coordination**: Only GPU registry requires locking

**2. Excellent Error Isolation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Well-Level Failures**: One well failure doesn’t affect others
-  **Resource Cleanup**: Guaranteed cleanup per thread
-  **Exception Propagation**: Clear error reporting per well

**3. Predictable Performance**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Linear Scaling**: Performance scales with number of cores/GPUs
-  **No Lock Contention**: Minimal synchronization overhead
-  **Resource Efficiency**: Optimal utilization of available hardware

**4. Simple Mental Model**
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Easy to Reason About**: Each well is independent
-  **Debugging Friendly**: Clear thread boundaries and isolated state
-  **Maintainable**: No complex synchronization logic

**Current Implementation Status**
---------------------------------

**Implemented Features**
~~~~~~~~~~~~~~~~~~~~~~~~

-  ✅ Two-phase execution model (compilation + execution)
-  ✅ Well-level parallelism with ThreadPoolExecutor/ProcessPoolExecutor
-  ✅ ProcessingContext freezing for immutability
-  ✅ Thread-safe GPU registry with compilation-time assignment
-  ✅ FileManager thread isolation with per-instance backend cache
-  ✅ Exception isolation with per-well error handling
-  ✅ Graceful degradation to sequential execution

**Future Enhancements**
~~~~~~~~~~~~~~~~~~~~~~~

1. **Runtime GPU Slot Management**: Dynamic GPU slot acquisition/release
   during execution (see `GPU Resource
   Management <gpu-resource-management.md>`__)
2. **Work Stealing**: Dynamic load balancing between threads
3. **Pipeline Parallelism**: Parallel execution of steps within a well
4. **Distributed Processing**: Multi-node execution coordination
5. **Adaptive Threading**: Dynamic thread pool sizing based on workload
6. **Memory Pool Management**: Shared memory pools for large datasets

This concurrency model provides **solid parallel processing
architecture** that achieves good performance while maintaining
simplicity and thread safety through careful design rather than complex
synchronization.
