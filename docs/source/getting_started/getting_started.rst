Getting Started with OpenHCS
============================

Installation
-----------

.. code-block:: bash

    pip install openhcs

Requirements:
- Python 3.8+
- For GPU acceleration: CUDA-compatible GPU with appropriate drivers

Basic Example
------------

This example shows the core OpenHCS workflow: creating a pipeline with processing steps and running it on microscopy data.

.. code-block:: python

    from openhcs.core.pipeline import Pipeline
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.constants.constants import VariableComponents

    # Define processing steps
    pipeline = Pipeline([
        # Normalize images
        FunctionStep(
            func=(stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0
            }),
            name="normalize",
            variable_components=[VariableComponents.SITE]
        ),

        # Count cells
        FunctionStep(
            func=(count_cells_single_channel, {}),
            name="count_cells",
            variable_components=[VariableComponents.SITE]
        )
    ])

    # Configure and run
    config = GlobalPipelineConfig(num_workers=2)
    orchestrator = PipelineOrchestrator(
        plate_path="/path/to/your/microscopy/data",
        global_config=config
    )

    # Execute pipeline
    orchestrator.run_pipeline(pipeline)

Understanding the Example
------------------------

The basic example demonstrates key OpenHCS concepts:

**Pipeline**: A list of processing steps that execute in sequence

**FunctionStep**: The basic processing unit that wraps a function with configuration

**Variable Components**: Defines how data is grouped for processing (SITE processes each imaging position separately)

**Orchestrator**: Manages pipeline execution across your dataset

Interactive Development
----------------------

For interactive pipeline building, use the TUI:

.. code-block:: bash

    openhcs-tui

This launches an interactive interface for:
- Selecting microscopy data directories
- Configuring processing pipelines
- Monitoring execution progress
- Viewing results

Next Steps
----------

After running the basic example, explore these areas:

**Core Concepts**: :doc:`../concepts/index`
  Understand pipelines, steps, function patterns, and data organization

**Function Library**: :doc:`../concepts/function_library`
  Learn about available image processing functions and backends

**Configuration**: :doc:`../concepts/storage_system`
  Configure storage backends, memory management, and output options

**Advanced Examples**: :doc:`../guides/index`
  Multi-channel analysis, GPU acceleration, and large dataset processing

Common Patterns
---------------

**Multi-Channel Analysis**:

.. code-block:: python

    # Different analysis for different channels
    FunctionStep(
        func={
            '1': (count_cells_single_channel, {}),  # DAPI channel
            '2': (trace_neurites, {})               # GFP channel
        },
        group_by=GroupBy.CHANNEL,
        variable_components=[VariableComponents.SITE]
    )

**Function Chains**:

.. code-block:: python

    # Sequential processing steps
    FunctionStep(
        func=[
            (gaussian_filter, {'sigma': 2.0}),
            (threshold_otsu, {}),
            (binary_opening, {'footprint_radius': 3})
        ],
        variable_components=[VariableComponents.SITE]
    )
