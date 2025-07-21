.. _pipeline-orchestrator:

====================
PipelineOrchestrator
====================

Role and Responsibilities
-------------------------

The `PipelineOrchestrator` is the central execution engine of OpenHCS. It is the primary user-facing class for running an entire end-to-end data processing workflow. It has three main responsibilities:

1.  **Environment Setup**: It inspects the input data, detects the microscope format, and initializes the Virtual File System (VFS) and GPU resources.
2.  **Pipeline Compilation**: It takes a user-defined list of `FunctionStep` objects and transforms them into a concrete, optimized, and executable plan. This is its most critical role.
3.  **Parallel Execution**: It executes the compiled plan across multiple wells of a plate in parallel, managing worker processes and resources.

.. code-block:: python

   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
   from openhcs.core.config import GlobalPipelineConfig

   # 1. Create orchestrator for a specific plate
   orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)

   # 2. Define the pipeline as a list of FunctionStep objects
   pipeline_steps = [ ... ]

   # 3. Run the three-phase execution workflow
   orchestrator.initialize()
   compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
   results = orchestrator.execute_compiled_plate(
       pipeline_definition=pipeline_steps,
       compiled_contexts=compiled_contexts,
       max_workers=global_config.num_workers
   )

The Compilation Process
-----------------------

The most important job of the orchestrator is **compilation**. The `compile_pipelines` method is a sophisticated 5-phase system that turns an abstract list of steps into a detailed, machine-readable execution plan for every well on the plate.

This compilation step is what makes OpenHCS so powerful, as it handles the complex dependency management, resource allocation, and workflow optimization automatically, before any processing begins.

The 5 phases are:

1.  **Step Plan Initialization**: Creates a basic plan for each step, resolving input/output paths within the VFS.
2.  **ZARR Store Declaration**: If Zarr is the materialization backend, this phase declares the necessary Zarr stores.
3.  **Materialization Planning**: Determines which steps require their output to be written to persistent storage.
4.  **Memory Validation**: Checks the memory requirements of the pipeline against available system resources.
5.  **GPU Assignment**: Assigns specific GPU devices to each processing task that requires one, ensuring balanced utilization.

By the end of compilation, the orchestrator has a `ProcessingContext` for each well—a complete, frozen set of instructions ready for execution.

Execution and Resource Management
---------------------------------

Once the pipeline is compiled, the `execute_compiled_plate` method takes over. It uses a process pool (or thread pool for debugging) to execute the compiled `ProcessingContext` for each well in parallel.

The orchestrator is responsible for:

-  **Distributing Work**: Sending the compiled plan for each well to a worker process.
-  **GPU Coordination**: Ensuring that multiple processes share GPU resources effectively and without conflicts.
-  **VFS Management**: Managing the lifecycle of the in-memory Virtual File System used for intermediate data.
-  **Error Handling**: Gracefully handling and reporting errors that occur within any worker process.

In essence, the `PipelineOrchestrator` frees the user from having to worry about the complex details of parallel processing, resource management, and file I/O. The user simply defines the *what* (the sequence of `FunctionStep` objects), and the orchestrator handles the *how*.
