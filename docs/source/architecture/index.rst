====================
Architecture Reference
====================

This section provides detailed technical documentation of OpenHCS's architecture and design decisions. These documents are intended for developers who need to understand the internal implementation details.

**Prerequisites**: Familiarity with :doc:`../concepts/index` is recommended before reading these technical documents.

**For Integration**: See :doc:`../guides/index` for practical integration guides that combine these architectural concepts.

OpenHCS evolved from EZStitcher while preserving core architectural patterns and adding capabilities for GPU-native scientific computing.

Core Systems
============

.. toctree::
   :maxdepth: 2

   function_pattern_system
   memory_type_system
   pipeline_compilation_system

Advanced Features
================

.. toctree::
   :maxdepth: 2

   special_io_system
   function_registry_system
   tui_system
   code_ui_interconversion
   vfs_system

Core Infrastructure
===================

.. toctree::
   :maxdepth: 2

   gpu_resource_management
   memory_backend_system
   microscope_handler_integration
   configuration_management_system
   configuration-resolution
   lazy-class-system
   service-layer-architecture
   step-editor-generalization
   field-path-detection
   concurrency_model
   system_integration

Advanced Topics
===============

.. toctree::
   :maxdepth: 2

   compilation_system_detailed
   pattern_detection_system
   pipeline_debugging_guide
   dict_pattern_case_study
   ezstitcher_to_openhcs_evolution

Development and Research
========================

.. toctree::
   :maxdepth: 2

   parameter_analysis_audit
   unified_parameter_analyzer_migration
   research_impact

System Overview
===============

OpenHCS Architecture Components
-------------------------------

**Memory Type System**: Automatic conversion between NumPy↔CuPy↔PyTorch↔TensorFlow↔JAX with zero-copy GPU operations and compile-time validation.

**Function Registry**: Auto-discovery of 574+ GPU functions from pyclesperanto, scikit-image, CuCIM with decorator-based contracts.

**5-Phase Compilation**: Step plan initialization → ZARR store declaration → Materialization → Memory validation → GPU assignment for immutable execution contexts.

**Special I/O System**: Cross-step communication for analysis results (cell counts, measurements) using declarative decorators.

**TUI System**: Terminal interface for scientific computing with SSH compatibility and real-time monitoring.

**VFS Backend**: Memory overlay with ZARR storage for 100GB+ dataset handling and automatic materialization.

Preserved from EZStitcher
-------------------------

**Function Patterns**: Single, tuple, list, dict patterns with ``variable_components`` and ``group_by`` for intelligent processing.

**Pipeline Architecture**: PipelineOrchestrator → Pipeline → Step hierarchy with specialized step types.

**Microscope Integration**: Native support for Opera Phenix, ImageXpress with extensible handler system.

**Directory Management**: Automatic workspace creation and file organization with VFS abstraction.

Evolution Summary
================

OpenHCS represents the evolution of EZStitcher from a CPU-based stitching tool into a GPU-native bioimage analysis platform:

- **Preserved**: Core architectural patterns, function handling, pipeline concepts
- **Enhanced**: GPU processing, memory type safety, production TUI, large dataset support  
- **Added**: Special I/O, function registry, compilation system, advanced validation

This architecture enables researchers to build complex bioimage analysis workflows with the same ease as the original stitching operations, while gaining access to GPU acceleration and advanced tooling.
