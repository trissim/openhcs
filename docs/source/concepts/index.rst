==========================
Core Concepts
==========================

This section explains the fundamental concepts of OpenHCS in user-friendly terms. Start here to understand what OpenHCS is and how it works before diving into technical details.

**Learning Path**: Read these concepts in order for the best understanding, then explore the :doc:`../architecture/index` for technical implementation details.

Fundamental Concepts
====================

Start with these core concepts to understand OpenHCS:

.. toctree::
   :maxdepth: 2

   basic_microscopy
   architecture_overview

**Basic Microscopy**: Understanding high-content screening and bioimage analysis fundamentals.

**Architecture Overview**: High-level overview of how OpenHCS components work together.

Pipeline Concepts
=================

Learn how to build and execute bioimage analysis workflows:

.. toctree::
   :maxdepth: 2

   pipeline_orchestrator
   pipeline
   step
   function_handling

**Pipeline Orchestrator**: The main execution engine that coordinates processing across plates and wells.

**Pipeline**: Sequences of processing steps that transform microscopy data.

**Step**: Individual processing operations within pipelines.

**Function Handling**: How OpenHCS supports flexible function patterns for different processing needs.

System Concepts
===============

Understand how OpenHCS manages data and resources:

.. toctree::
   :maxdepth: 2

   processing_context
   directory_structure
   module_structure
   storage_adapter

**Processing Context**: How OpenHCS manages execution state and configuration.

**Directory Structure**: How OpenHCS organizes input and output data.

**Module Structure**: How OpenHCS code is organized and how to import components.

**Storage Adapter**: How OpenHCS handles different storage backends and file formats.
