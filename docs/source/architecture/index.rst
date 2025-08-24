======================
Architecture Reference
======================

Technical documentation of OpenHCS's architecture for developers who need to understand internal implementation details.

**Prerequisites**: :doc:`../concepts/index` | **Integration**: :doc:`../guides/index`

Essential Architecture
======================

The core systems every OpenHCS developer needs to understand.

.. toctree::
   :maxdepth: 1

   function_pattern_system
   configuration_system_architecture
   storage_and_memory_system
   pipeline_compilation_system
   special_io_system
   function_registry_system
   microscope_handler_integration
   system_integration
   ezstitcher_to_openhcs_evolution

Implementation Details
======================

Deep technical dives for developers working on specific systems.

.. toctree::
   :maxdepth: 1

   compilation_system_detailed
   gpu_resource_management
   tui_system

Development Tools
=================

Practical guides for OpenHCS development workflows.

.. toctree::
   :maxdepth: 1

   code_ui_interconversion
   step-editor-generalization

Specialized Topics
==================

Advanced topics for specific use cases.

.. toctree::
   :maxdepth: 1

   pattern_detection_system
   concurrency_model
   service-layer-architecture

Quick Start Paths
==================

**New to OpenHCS?** Start with :doc:`function_pattern_system` → :doc:`configuration_system_architecture` → :doc:`storage_and_memory_system`

**System Integration?** Jump to :doc:`system_integration` → :doc:`special_io_system`

**Performance Optimization?** Focus on :doc:`gpu_resource_management` → :doc:`compilation_system_detailed`

**UI Development?** Begin with :doc:`tui_system` → :doc:`code_ui_interconversion`
