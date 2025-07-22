Getting Started with OpenHCS
============================

🚀 **Complete Production Example**
----------------------------------

**The best way to understand OpenHCS is through a complete, working example.**

We provide a **gold standard production script** that demonstrates every major OpenHCS feature:

📁 **Complete Example Script**: `openhcs/debug/example_export.py <https://github.com/trissim/toolong/blob/openhcs/openhcs/debug/example_export.py>`_

This script shows:

✅ **Complete Configuration**: GlobalPipelineConfig, VFS, ZARR, GPU settings
✅ **All Function Patterns**: List chains, dictionary patterns, single functions
✅ **Real Workflow**: Preprocessing → Stitching → Analysis (100GB+ datasets)
✅ **GPU Integration**: CuPy, PyTorch, GPU stitching algorithms
✅ **Production Settings**: Memory backends, compression, parallel processing

**Key Features Demonstrated**:

.. code-block:: python

    # Complete configuration system
    global_config = GlobalPipelineConfig(
        num_workers=5,
        vfs=VFSConfig(intermediate_backend=Backend.MEMORY),
        zarr=ZarrConfig(compressor=ZarrCompressor.ZSTD)
    )

    # Function chain pattern
    FunctionStep(func=[
        (stack_percentile_normalize, {'low_percentile': 1.0}),
        (tophat, {'selem_radius': 50})
    ])

    # Dictionary pattern for channel-specific analysis
    FunctionStep(func={
        '1': [count_cells_single_channel],      # DAPI channel
        '2': [skan_axon_skeletonize_and_analyze] # GFP channel
    })

Installation
-----------

.. code-block:: bash

    pip install openhcs  # Requires Python 3.8+

Running the Complete Example
---------------------------

**Download and run the production example**:

.. code-block:: bash

    # Clone the repository to access the example
    git clone https://github.com/trissim/toolong.git
    cd toolong

    # View the complete example script
    cat openhcs/debug/example_export.py

    # Run it (requires microscopy data)
    python openhcs/debug/example_export.py

**What the example demonstrates**:

🔬 **Complete Neurite Analysis Pipeline**:
1. **Preprocessing**: Percentile normalization + top-hat filtering
2. **Composition**: Multi-channel composite creation
3. **Stitching**: GPU-accelerated position finding + assembly
4. **Analysis**: Cell counting (DAPI) + neurite tracing (GFP)

🚀 **Production Features**:
- **100GB+ Dataset Handling**: ZARR compression with memory overlay
- **GPU Acceleration**: CuPy, PyTorch, GPU stitching algorithms
- **Multi-Backend Processing**: Automatic memory type conversion
- **Parallel Execution**: 5 workers with GPU scheduling

Interactive Development
----------------------

For interactive pipeline building, use the TUI:

.. code-block:: bash

    # Launch the interactive TUI
    openhcs-tui

    # Select your plate directory and configure pipeline
    # Real-time monitoring and professional log streaming
    # Works over SSH - no desktop required

Common Options
------------

.. code-block:: python

    stitch_plate(
        "path/to/plate",
        output_path="path/to/output",    # Custom output location
        normalize=True,                  # Enhance contrast
        flatten_z=True,                  # Convert Z-stacks to 2D
        z_method="max",                  # Z projection method
        well_filter=["A01", "B02"]       # Process specific wells
    )

Next Steps
---------

- Read :doc:`../user_guide/introduction` for an overview of OpenHCS concepts
- See :doc:`../user_guide/basic_usage` for detailed API usage
- Explore :doc:`../user_guide/intermediate_usage` for custom pipelines
- Check :doc:`../concepts/architecture_overview` for technical details
