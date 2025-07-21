==========================
Core Concepts
==========================

This section provides a high-level, user-friendly guide to the core concepts of OpenHCS. It's the best place to start if you want to understand what OpenHCS does and how its major components fit together, without getting lost in deep technical details. The pages are designed to be read in order to build a strong foundational understanding.

Core Architecture
=================

Before you can build a pipeline, it's important to understand the main architectural ideas that make OpenHCS powerful and efficient. This section covers the "what" and "why" of the system's design.
.. toctree::
   :maxdepth: 2

   architecture_overview
   basic_microscopy

**Architecture Overview**: A 30,000-foot view of the system. Start here to see how all the pieces, from the user interface to the processing backends, connect to one another.

**Core Concepts of OpenHCS**: Dives one level deeper, explaining the main conceptual pillars of the system: the `PipelineOrchestrator` that runs everything, the `FunctionStep` that defines the work, and the Virtual File System (VFS) that manages data.


Building Pipelines
==================

Once you understand the architecture, this section shows you how to use it. These pages focus on the practical details of defining and running a data processing workflow.

.. toctree::
   :maxdepth: 2

   pipeline_orchestrator
   function_handling

**Pipeline Orchestrator**: The "brain" of the operation. This document explains how the orchestrator discovers your data, compiles your requested steps into an executable plan, and manages the entire process from start to finish.

**Function Handling**: The "muscle" of the operation. A `FunctionStep` is where the actual work happens, and this page details the powerful "function patterns" you can use to define everything from a single calculation to a complex, multi-stage processing chain.


Data and System Management
==========================

This final section covers the essential background components that make the pipeline work smoothly. You'll learn how OpenHCS handles files, memory, and its own internal state.

.. toctree::
   :maxdepth: 2

   directory_structure
   storage_adapter
   processing_context
   module_structure

**Directory Structure**: Explains how OpenHCS uses a Virtual File System (VFS) and symbolic links to create safe, efficient, in-memory workspaces, and how it handles final output to disk.

**Storage Adapter**: Details the seamless integration of different storage backends, allowing you to use memory, disk, or Zarr for your data without changing your pipeline code.

**Processing Context**: A look at the object that holds the "state" of a pipeline run. It's how configuration and data are made available to each step.

**Module Structure**: An overview of how the OpenHCS Python source code is organized, which is useful for advanced users who want to import specific components or extend the system.
