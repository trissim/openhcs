Complete OpenHCS Examples
=========================

This guide provides complete, working examples that demonstrate all major OpenHCS functionality. These examples are perfect for agents, developers, and users who want to understand practical usage patterns.

Complete Example Script
-----------------------

This section demonstrates a comprehensive OpenHCS workflow that covers all major functionality patterns.

**Script Features**:

**Configuration System**:
- GlobalPipelineConfig with sub-configurations
- VFS backend selection (memory + ZARR materialization)
- GPU resource management and parallel processing
- Compression and chunking strategies

**Function Patterns**:
- **List Pattern**: Sequential function chains with parameters
- **Dictionary Pattern**: Channel-specific processing (DAPI vs GFP)
- **Single Functions**: Simple function calls
- **Parameterized Functions**: Real parameter values

**Complete Workflow**:
- **Preprocessing**: Percentile normalization + morphological filtering
- **Composition**: Multi-channel composite creation
- **Stitching**: Position finding + image assembly
- **Analysis**: Cell counting + neurite tracing (channel-specific)

**System Features**:
- Large dataset handling with ZARR compression
- GPU acceleration with CuPy and PyTorch
- Multi-backend processing with automatic memory type conversion
- Parallel execution with worker scheduling

Key Code Patterns
------------------

**Complete Configuration**:

.. code-block:: python

    from pathlib import Path

    from openhcs.constants.constants import Backend, Microscope
    from openhcs.core.config import (GlobalPipelineConfig, PathPlanningConfig,
                                     VFSConfig, ZarrConfig, MaterializationBackend,
                                     ZarrCompressor, ZarrChunkStrategy)
    
    global_config = GlobalPipelineConfig(
        num_workers=5,
        materialization_results_path=Path("results"),
        path_planning_config=PathPlanningConfig(
            output_dir_suffix="_stitched",
            global_output_folder=Path("/data/output/"),
        ),
        vfs_config=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=MaterializationBackend.ZARR
        ),
        zarr_config=ZarrConfig(
            compressor=ZarrCompressor.ZSTD,
            compression_level=1,
            chunk_strategy=ZarrChunkStrategy.WELL
        ),
        microscope=Microscope.AUTO
    )

**Function Chain Pattern**:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import LazyProcessingConfig
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat
    
    # Sequential processing chain
    step = FunctionStep(
        func=[
            (stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            }),
            (tophat, {
                'selem_radius': 50,
                'downsample_factor': 4
            })
        ],
        name="preprocess",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
        ),
    )

**Dictionary Pattern for Channel-Specific Analysis**:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import LazyProcessingConfig
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel, DetectionMethod
    from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze, AnalysisDimension
    
    # Different analysis for different channels
    step = FunctionStep(
        func={
            '1': [  # DAPI channel - cell counting
                (count_cells_single_channel, {
                    'min_sigma': 1.0,
                    'max_sigma': 10.0,
                    'threshold': 0.1,
                    'detection_method': DetectionMethod.WATERSHED
                })
            ],
            '2': [  # GFP channel - neurite tracing
                (skan_axon_skeletonize_and_analyze, {
                    'min_branch_length': 10.0,
                    'analysis_dimension': AnalysisDimension.TWO_D
                })
            ]
        },
        name="analysis",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
        ),
    )

**GPU Stitching Workflow**:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import LazyProcessingConfig, LazyStepMaterializationConfig
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
    
    # GPU position calculation
    positions_step = FunctionStep(
        func=[(ashlar_compute_tile_positions_gpu, {
            'overlap_ratio': 0.1,
            'max_shift': 15.0,
            'stitch_alpha': 0.2
        })],
        name="find_positions",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
        ),
    )
    
    # GPU image assembly
    assembly_step = FunctionStep(
        func=[(assemble_stack_cupy, {
            'blend_method': "fixed",
            'fixed_margin_ratio': 0.1
        })],
        name="assemble",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
        ),
        step_materialization_config=LazyStepMaterializationConfig(enabled=True),
    )

**Complete Orchestrator Execution**:

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
    
    # Setup GPU registry
    os.environ["OPENHCS_SUBPROCESS_MODE"] = "1"
    setup_global_gpu_registry(global_config=global_config)
    
    # Execute pipeline
    orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)
    orchestrator.initialize()
    compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
    orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline_steps,
        compiled_contexts=compiled_contexts,
        max_workers=global_config.num_workers
    )

Running the Example
-------------------

**Prerequisites**:
- OpenHCS installed
- GPU with CUDA support (optional but recommended)
- Microscopy data in supported format

**Steps**:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/trissim/openhcs.git
    cd openhcs

    # View the complete script
    cat openhcs/debug/example_export.py

    # Modify plate path for your data
    # Edit line 33: plate_paths = ['/path/to/your/plate']

    # Run the complete pipeline
    python openhcs/debug/example_export.py

**Expected Output**:
- Preprocessed images with normalization and filtering
- Multi-channel composite images
- Stitched high-resolution images
- Cell count analysis results
- Neurite tracing analysis results
- All outputs saved with ZARR compression

For Agents and Developers
-------------------------

**This example script is the perfect reference for**:

🤖 **AI Agents**: Complete, working patterns for all OpenHCS functionality
🔧 **Developers**: Real parameter values and production configurations
🐛 **Debugging**: Working baseline to compare against
📚 **Learning**: Practical examples of every major feature
**Deployment**: Tested configuration and workflow patterns

**The script demonstrates**:
- Exact import paths that work
- Real parameter values for production use
- Complete configuration for 100GB+ datasets
- GPU acceleration patterns
- Error handling and signal management
- Production-grade logging and monitoring

See Also
--------

- :doc:`../getting_started/getting_started` - Getting started with the example
- :doc:`../api/index` - API reference (autogenerated from source code)
