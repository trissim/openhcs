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

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        plate_paths=['/path/to/plate1', '/path/to/plate2'],
        steps=steps,
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

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        plate_paths=['/path/to/plate1', '/path/to/plate2'],
        steps=steps,
        global_config=GlobalPipelineConfig(num_workers=4)
    )

    # Execute pipeline
    orchestrator.run()

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
        plate_paths=plate_paths,
        steps=steps,
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

    orchestrator = PipelineOrchestrator(
        plate_paths=list(pipeline_data.keys()),
        pipeline_data=pipeline_data,
        global_config=global_config
    )

Parameters
----------

plate_paths : list of str
    List of paths to microscopy plate directories to process.

steps : list of AbstractStep, optional
    Default pipeline steps to apply to all plates. Can be overridden by pipeline_data.

pipeline_data : dict, optional
    Per-plate pipeline configuration. Keys are plate paths, values are step lists.

global_config : GlobalPipelineConfig
    Global configuration for execution, resource management, and storage.

Execution Flow
--------------

1. **Initialization**: Validate plate paths and pipeline configuration
2. **Compilation**: 4-phase compilation for each well in each plate
3. **Resource Allocation**: Assign GPU resources and worker processes  
4. **Execution**: Execute compiled pipelines with progress monitoring
5. **Materialization**: Save results to configured storage backends

See Also
--------

- :doc:`../architecture/pipeline_compilation_system` - Compilation process details
- :doc:`../architecture/gpu_resource_management` - GPU resource allocation
- :doc:`../architecture/concurrency_model` - Multi-processing architecture
- :doc:`config` - Configuration system documentation
