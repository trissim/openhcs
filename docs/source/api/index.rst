API Reference
=============

This section provides comprehensive API documentation for OpenHCS. Use this reference to understand the classes, functions, and modules available for building bioimage analysis workflows.

🚀 **Complete Working Example**
-------------------------------

**Before diving into the API, see our complete production example:**

📁 **Gold Standard Script**: `openhcs/debug/example_export.py <https://github.com/trissim/toolong/blob/openhcs/openhcs/debug/example_export.py>`_

This script shows **every API component in action** with real parameters and working code. Perfect for agents and developers to understand practical usage patterns.

**Quick Navigation**:
- **New to OpenHCS?** Start with :doc:`orchestrator` and :doc:`function_step`
- **Building pipelines?** See :doc:`processing_backends` for available functions
- **Managing data?** Check :doc:`io_storage` for file and storage operations
- **Advanced usage?** Explore :doc:`config` for configuration options

Essential Components
====================

Start here for the core OpenHCS API. These are the main classes you'll use to build bioimage analysis workflows.

.. toctree::
   :maxdepth: 2

   orchestrator
   function_step

**PipelineOrchestrator**: The main execution engine for running pipelines across multiple plates and wells.

**FunctionStep**: The building block for pipeline steps, supporting flexible function patterns and automatic GPU memory management.

Processing and Analysis
=======================

Discover the 574+ available processing functions organized by computational backend and functionality.

.. toctree::
   :maxdepth: 2

   processing_backends
   image_processing_functions
   stitching_functions

**Processing Backends**: GPU-accelerated functions for image processing, analysis, and assembly. Includes automatic memory type conversion between NumPy, CuPy, PyTorch, JAX, and pyclesperanto.

**Image Processing Functions**: Core image processing functions across 6 computational backends (NumPy, CuPy, PyTorch, TensorFlow, JAX, pyclesperanto) for filtering, normalization, morphology, and composition.

**Stitching Functions**: Complete stitching workflow with GPU-accelerated position generation and image assembly using Ashlar and MIST algorithms.

Data Management
===============

Handle file I/O, storage backends, and configuration for datasets from MB to 100GB+.

.. toctree::
   :maxdepth: 2

   io_storage
   config

**I/O and Storage**: Unified file operations with automatic backend selection (disk, memory, ZARR). Handles large dataset compression and VFS abstraction.

**Configuration**: Global pipeline configuration, GPU resource management, and system settings.

Pipeline Architecture
=====================

Core pipeline components for building and executing OpenHCS workflows.

.. toctree::
   :maxdepth: 2

   step_system
   pipeline
   tui_system

**Step System**: Complete step architecture including AbstractStep base class and FunctionStep implementation. Covers function patterns, variable components, and GPU integration.

**Pipeline**: Core pipeline execution and compilation system.

**TUI System**: Terminal User Interface for interactive pipeline building and execution.

Advanced Components
===================

Specialized components for advanced usage and system integration.

.. toctree::
   :maxdepth: 2

   file_system_manager
   microscope_interfaces
   microscopes

**File System Manager**: Low-level file operations and VFS backend management.

**Microscope Interfaces**: Interfaces for different microscopy platforms and file formats.

**Microscopes**: Specific microscope implementations and format parsers.
