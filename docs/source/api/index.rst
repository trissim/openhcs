API Reference
=============

This section provides comprehensive API documentation for OpenHCS. Use this reference to understand the classes, functions, and modules available for building bioimage analysis workflows.

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

**Processing Backends**: GPU-accelerated functions for image processing, analysis, and assembly. Includes automatic memory type conversion between NumPy, CuPy, PyTorch, JAX, and pyclesperanto.

Data Management
===============

Handle file I/O, storage backends, and configuration for datasets from MB to 100GB+.

.. toctree::
   :maxdepth: 2

   io_storage
   config

**I/O and Storage**: Unified file operations with automatic backend selection (disk, memory, ZARR). Handles large dataset compression and VFS abstraction.

**Configuration**: Global pipeline configuration, GPU resource management, and system settings.

Legacy Components
=================

These components are preserved for compatibility but may be deprecated in future versions.

.. toctree::
   :maxdepth: 2

   pipeline
   steps
   file_system_manager
