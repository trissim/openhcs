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

**5-Phase Compilation**: Automatic step plan initialization, ZARR store declaration, materialization, memory validation, and GPU assignment.

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

    # Import processing functions
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.constants.constants import VariableComponents

    # Define pipeline steps with real functions
    steps = [
        FunctionStep(
            func=[(stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            })],
            name="normalize",
            variable_components=[VariableComponents.SITE]
        ),
        FunctionStep(
            func=[(tophat, {'selem_radius': 50})],
            name="preprocess",
            variable_components=[VariableComponents.SITE]
        ),
        FunctionStep(
            func=[create_composite],
            name="composite",
            variable_components=[VariableComponents.CHANNEL]
        ),
        FunctionStep(
            func=count_cells_single_channel,
            name="cell_count",
            variable_components=[VariableComponents.SITE]
        )
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

Gold Standard Example
^^^^^^^^^^^^^^^^^^^^^

Complete example from production OpenHCS script:

.. code-block:: python

    import os
    from pathlib import Path
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig, VFSConfig
    from openhcs.constants.constants import VariableComponents, Backend, MaterializationBackend

    # Import processing functions
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    def create_production_pipeline():
        """Create a complete production pipeline for neurite analysis."""

        # Configure global settings
        global_config = GlobalPipelineConfig(
            num_workers=4,
            path_planning_config=PathPlanningConfig(
                enable_path_planning=True,
                enable_smart_caching=True
            ),
            vfs_config=VFSConfig(
                intermediate_backend=Backend.MEMORY,
                materialization_backend=MaterializationBackend.ZARR
            )
        )

        # Define processing pipeline
        pipeline_steps = [
            # Step 1: Normalize images
            FunctionStep(
                func=[(stack_percentile_normalize, {
                    'low_percentile': 1.0,
                    'high_percentile': 99.0,
                    'target_max': 65535.0
                })],
                name="normalize",
                variable_components=[VariableComponents.SITE],
                force_disk_output=False
            ),

            # Step 2: Morphological preprocessing
            FunctionStep(
                func=[(tophat, {'selem_radius': 50})],
                name="preprocess",
                variable_components=[VariableComponents.SITE],
                force_disk_output=False
            ),

            # Step 3: Create composite images
            FunctionStep(
                func=[create_composite],
                name="composite",
                variable_components=[VariableComponents.CHANNEL],
                force_disk_output=False
            ),

            # Step 4: Cell counting analysis
            FunctionStep(
                func=count_cells_single_channel,
                name="cell_count",
                variable_components=[VariableComponents.SITE],
                force_disk_output=False
            ),

            # Step 5: Neurite tracing analysis
            FunctionStep(
                func=skan_axon_skeletonize_and_analyze,
                name="neurite_trace",
                variable_components=[VariableComponents.SITE],
                force_disk_output=False
            ),

            # Step 6: Tile position calculation
            FunctionStep(
                func=[(ashlar_compute_tile_positions_gpu, {
                    'overlap_ratio': 0.1,
                    'max_shift': 15.0,
                    'stitch_alpha': 0.2
                })],
                name="positions",
                variable_components=[VariableComponents.CHANNEL],
                force_disk_output=True
            ),

            # Step 7: Image assembly
            FunctionStep(
                func=[(assemble_stack_cupy, {
                    'blend_method': "fixed",
                    'fixed_margin_ratio': 0.1
                })],
                name="assemble",
                variable_components=[VariableComponents.CHANNEL],
                force_disk_output=True
            )
        ]

        return pipeline_steps, global_config

    def run_production_pipeline(plate_path: str):
        """Execute the complete production pipeline."""

        # Set subprocess mode for proper GPU handling
        os.environ["OPENHCS_SUBPROCESS_MODE"] = "1"

        # Create pipeline and configuration
        pipeline_steps, global_config = create_production_pipeline()

        # Setup GPU registry
        from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
        setup_global_gpu_registry(global_config=global_config)

        # Create orchestrator
        orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)

        # Execute 3-phase workflow
        orchestrator.initialize()
        compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
        orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline_steps,
            compiled_contexts=compiled_contexts,
            max_workers=global_config.num_workers
        )

    # Usage
    if __name__ == "__main__":
        plate_path = "/path/to/microscopy/plate"
        run_production_pipeline(plate_path)

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
