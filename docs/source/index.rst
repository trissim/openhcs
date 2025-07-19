Welcome to OpenHCS Documentation
=================================

OpenHCS is a production-grade bioimage analysis platform designed for high-content screening datasets that break traditional tools. Built from the ground up for 100GB+ datasets, GPU acceleration, and remote computing environments.

Getting Started Quickly
-----------------------

The fastest way to get started with OpenHCS is through the terminal interface:

.. code-block:: bash

    # Install OpenHCS
    pip install openhcs

    # Launch the interactive TUI
    openhcs-tui

For Python API usage:

.. code-block:: python

    from openhcs import Pipeline, FunctionStep

    # Create a simple processing pipeline
    pipeline = Pipeline([
        FunctionStep(func="gaussian_filter", sigma=2.0),
        FunctionStep(func="binary_opening", footprint=disk(3)),
        FunctionStep(func="label", connectivity=2)
    ])

    # Process your data
    pipeline.run("path/to/microscopy/data")

For a complete guide including installation and examples, see :doc:`getting_started/getting_started`.

Key Features
------------

- **Unified GPU Function Registry**: 574+ functions from pyclesperanto, CuCIM, scikit-image with unified interfaces
- **Automatic Memory Management**: Memory type conversion between NumPy ↔ CuPy ↔ PyTorch ↔ TensorFlow
- **SSH-Native Terminal Interface**: Production-grade TUI that works over SSH without X11 forwarding
- **Intelligent Storage**: Memory overlay VFS with automatic backend selection (memory, disk, zarr)
- **GPU Acceleration**: Thread-safe GPU resource management with OOM recovery
- **Advanced Analysis**: HMM-based neurite tracing, cell counting, skeleton analysis
- **Multiple Microscope Support**: Works with ImageXpress and Opera Phenix microscopes
- **Production Architecture**: Handles 100GB+ datasets with comprehensive error handling

Key Resources
-----------

* :doc:`getting_started/getting_started` - Install OpenHCS and run your first pipeline in minutes
* :doc:`user_guide/basic_usage` - Learn about the terminal interface and Python API
* :doc:`concepts/architecture_overview` - Learn about OpenHCS's architecture
* :doc:`user_guide/intermediate_usage` - Create custom pipelines with FunctionSteps
* :doc:`user_guide/best_practices` - Learn best practices for using OpenHCS

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/introduction
   user_guide/basic_usage
   user_guide/intermediate_usage
   user_guide/advanced_usage
   user_guide/best_practices
   user_guide/integration

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/basic_microscopy
   concepts/architecture_overview
   concepts/pipeline_orchestrator
   concepts/pipeline
   concepts/pipeline_factory
   concepts/step
   concepts/function_handling
   concepts/processing_context
   concepts/directory_structure

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   appendices/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
