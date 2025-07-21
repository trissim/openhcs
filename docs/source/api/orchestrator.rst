Pipeline Orchestrator
=====================

.. module:: openhcs.core.orchestrator.orchestrator

The PipelineOrchestrator is the main execution engine for OpenHCS pipelines. It coordinates pipeline compilation, resource allocation, and execution across multiple plates and wells.

PipelineOrchestrator Class
--------------------------

The PipelineOrchestrator is the main execution engine for OpenHCS pipelines.

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.config import GlobalPipelineConfig

    # Create orchestrator for single plate
    orchestrator = PipelineOrchestrator(
        plate_path='/path/to/plate1',
        global_config=GlobalPipelineConfig(num_workers=4)
    )

Key Features
------------

**Multi-Plate Processing**: Execute pipelines across multiple microscopy plates simultaneously.

**4-Phase Compilation**: Automatic path planning, materialization, contract validation, and GPU assignment.

**Resource Management**: Intelligent worker process coordination and GPU memory management.

**Backend Integration**: Seamless switching between disk, memory, and ZARR storage backends.

**Error Handling**: Comprehensive error reporting and recovery mechanisms.

Usage Examples
--------------

Basic Pipeline Execution
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig

    # Define pipeline steps
    steps = [
        FunctionStep(func=normalize_images, name="normalize"),
        FunctionStep(func=segment_cells, name="segment")
    ]

    # Create orchestrator for single plate
    orchestrator = PipelineOrchestrator(
        plate_path='/path/to/plate1',
        global_config=GlobalPipelineConfig(num_workers=4)
    )

    # Initialize and compile
    orchestrator.initialize()
    compiled_contexts = orchestrator.compile_pipelines(steps)

    # Execute pipeline
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=steps,
        compiled_contexts=compiled_contexts,
        max_workers=4
    )

Production Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.config import PathPlanningConfig, VFSConfig, ZarrConfig
    from openhcs.constants.constants import Backend, MaterializationBackend

    # Production-ready configuration
    global_config = GlobalPipelineConfig(
        num_workers=8,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_analysis",
            global_output_folder="/data/results/",
            materialization_results_path="results"
        ),
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=MaterializationBackend.ZARR
        ),
        zarr=ZarrConfig(
            compressor=ZarrCompressor.LZ4,
            chunk_strategy=ZarrChunkStrategy.ADAPTIVE
        )
    )

    orchestrator = PipelineOrchestrator(
        plate_path='/path/to/plate1',
        global_config=global_config
    )

Per-Plate Pipeline Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Different pipelines for different plates
    pipeline_data = {
        '/path/to/plate1': steps_for_plate1,
        '/path/to/plate2': steps_for_plate2
    }

    # Process each plate separately
    for plate_path, steps in pipeline_data.items():
        orchestrator = PipelineOrchestrator(
            plate_path=plate_path,
            global_config=global_config
        )
        orchestrator.initialize()
        compiled_contexts = orchestrator.compile_pipelines(steps)
        results = orchestrator.execute_compiled_plate(
            pipeline_definition=steps,
            compiled_contexts=compiled_contexts
        )

Parameters
----------

plate_path : str or Path
    Path to microscopy plate directory to process.

workspace_path : str or Path, optional
    Path to workspace directory. If None, defaults to plate_path parent with _workspace suffix.

global_config : GlobalPipelineConfig, optional
    Global configuration for execution, resource management, and storage. If None, uses default configuration.

storage_registry : optional
    Optional StorageRegistry instance for custom storage backends.

Execution Flow
--------------

1. **Initialization**: Validate plate paths and pipeline configuration
2. **Compilation**: 5-phase compilation for each well in each plate
3. **Resource Allocation**: Assign GPU resources and worker processes  
4. **Execution**: Execute compiled pipelines with progress monitoring
5. **Materialization**: Save results to configured storage backends

See Also
--------

- :doc:`../architecture/pipeline_compilation_system` - Compilation process details
- :doc:`../architecture/gpu_resource_management` - GPU resource allocation
- :doc:`../architecture/concurrency_model` - Multi-processing architecture
- :doc:`config` - Configuration system documentation
