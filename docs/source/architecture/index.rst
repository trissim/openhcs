======================
Architecture Reference
======================

This section provides detailed technical documentation of OpenHCS's architecture and design decisions. These documents are intended for developers who need to understand the internal implementation details.

**Prerequisites**: Familiarity with :doc:`../concepts/index` is recommended before reading these technical documents.

**For Integration**: See :doc:`../guides/index` for practical integration guides that combine these architectural concepts.

**Reading Guide**: Documents are organized in logical reading order, from foundational concepts to advanced implementation details. Start with Foundation, then proceed through Core Systems and Integration before diving into Advanced Topics.

OpenHCS evolved from EZStitcher while preserving core architectural patterns and adding capabilities for GPU-native scientific computing.

Foundation: Core Concepts
=========================

Start here to understand the fundamental building blocks of OpenHCS.

.. toctree::
   :maxdepth: 2

   function_pattern_system
   configuration_system_architecture
   ezstitcher_to_openhcs_evolution

Core Systems: Data and Processing
==================================

The heart of OpenHCS - how data flows through the system and gets processed.

.. toctree::
   :maxdepth: 2

   storage_and_memory_system
   pipeline_compilation_system
   function_registry_system

Integration: Cross-System Communication
=======================================

How the core systems work together and communicate with external components.

.. toctree::
   :maxdepth: 2

   special_io_system
   system_integration
   microscope_handler_integration

User Interface and Interaction
==============================

How users interact with OpenHCS and how the system presents information.

.. toctree::
   :maxdepth: 2

   tui_system
   code_ui_interconversion
   step-editor-generalization

Advanced Implementation Details
===============================

Deep dives into specific implementation aspects and advanced features.

.. toctree::
   :maxdepth: 2

   compilation_system_detailed
   gpu_resource_management
   pattern_detection_system
   concurrency_model
   service-layer-architecture

Development and Debugging
=========================

Tools and techniques for developing, debugging, and analyzing OpenHCS.

.. toctree::
   :maxdepth: 2

   pipeline_debugging_guide
   parameter_analysis_audit
   unified_parameter_analyzer_migration
   research_impact

Architecture Overview
=====================

OpenHCS Architecture Components
-------------------------------

**Foundation Layer**:
- **Function Patterns**: Single, tuple, list, dict patterns with ``variable_components`` and ``group_by`` for intelligent processing
- **Configuration System**: Hierarchical dataclass-based configuration with lazy resolution and thread-local context management
- **Evolutionary Design**: Built on proven EZStitcher patterns while adding GPU-native capabilities

**Core Systems Layer**:
- **Storage & Memory**: VFS backend with memory overlay, ZARR storage for 100GB+ datasets, and automatic NumPy↔CuPy↔PyTorch conversion
- **Pipeline Compilation**: 5-phase compilation (initialization → ZARR → materialization → memory validation → GPU assignment)
- **Function Registry**: Auto-discovery of 574+ GPU functions from pyclesperanto, scikit-image, CuCIM with decorator-based contracts

**Integration Layer**:
- **Special I/O System**: Cross-step communication for analysis results using declarative decorators and compiler-inspired namespacing
- **System Integration**: Coordination between VFS, memory types, and compilation for seamless data flow
- **Microscope Integration**: Native support for Opera Phenix, ImageXpress with extensible handler system

**User Interface Layer**:
- **TUI System**: Terminal interface for scientific computing with SSH compatibility and real-time monitoring
- **Code-UI Interconversion**: Bidirectional conversion between programmatic and interactive configurations
- **Step Editor**: Generalized editing system for pipeline components

Reading Path Recommendations
============================

**For New Developers**:
1. Start with :doc:`function_pattern_system` to understand the core abstraction
2. Read :doc:`configuration_system_architecture` to understand how the system is configured
3. Proceed through :doc:`storage_and_memory_system` and :doc:`pipeline_compilation_system` for core processing

**For System Integrators**:
1. Begin with :doc:`system_integration` for the big picture
2. Focus on :doc:`special_io_system` for cross-step communication
3. Review :doc:`microscope_handler_integration` for hardware integration

**For Advanced Developers**:
1. Study :doc:`compilation_system_detailed` for implementation specifics
2. Explore :doc:`gpu_resource_management` for performance optimization
3. Use :doc:`pipeline_debugging_guide` for troubleshooting techniques

**For UI Developers**:
1. Start with :doc:`tui_system` for interface architecture
2. Review :doc:`code_ui_interconversion` for configuration handling
3. Study :doc:`step-editor-generalization` for component editing

Evolution Summary
=================

OpenHCS represents the evolution of EZStitcher from a CPU-based stitching tool into a GPU-native bioimage analysis platform:

- **Preserved**: Core architectural patterns, function handling, pipeline concepts
- **Enhanced**: GPU processing, memory type safety, production TUI, large dataset support
- **Added**: Special I/O, function registry, compilation system, advanced validation

This architecture enables researchers to build complex bioimage analysis workflows with the same ease as the original stitching operations, while gaining access to GPU acceleration and advanced tooling.
