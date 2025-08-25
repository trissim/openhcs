.. _module-structure:

================
Module Structure
================

.. _module-overview:

Overview
--------

OpenHCS uses a carefully designed module structure that enforces clean architectural boundaries, prevents circular dependencies, and ensures doctrinal compliance. This document explains the organization of modules in OpenHCS and the principles behind this structure.

.. _module-basic-concepts:

Basic Module Concepts
--------------------

In OpenHCS, modules are organized according to these key principles:

* **Interface-Implementation Separation**: Interfaces are defined separately from their implementations
* **Schema-First Design**: Data structures are defined in schemas before they are used
* **Explicit Registration**: Components are registered explicitly, not implicitly on import
* **Unidirectional Dependencies**: Dependencies flow in one direction to prevent cycles

.. _module-directory-structure:

OpenHCS Module Structure
------------------------

OpenHCS follows a clean, hierarchical module organization optimized for GPU-accelerated bioimage analysis:

**Real Import Patterns** (from TUI-generated scripts):

.. code-block:: python

    # Core orchestration and pipeline components
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig

    # Constants and enums
    from openhcs.constants.constants import VariableComponents, Backend, Microscope

    # Processing functions by computational backend
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # Configuration classes
    from openhcs.core.config import PathPlanningConfig, VFSConfig, ZarrConfig
    from openhcs.core.config import MaterializationBackend, ZarrCompressor, ZarrChunkStrategy

**Module Organization**:

.. code-block:: text

    openhcs/
    ├── core/                       # Core orchestration and pipeline management
    │   ├── orchestrator/           # PipelineOrchestrator and execution engine
    │   ├── steps/                  # FunctionStep and step implementations
    │   ├── config/                 # Configuration classes and management
    │   └── context/                # ProcessingContext and state management
    ├── processing/                 # Integrated Python image processing libraries
    │   └── backends/               # Organized by computational backend
    │       ├── processors/         # Image processing functions
    │       ├── analysis/           # Analysis and measurement functions
    │       ├── assemblers/         # Image assembly and stitching
    │       └── pos_gen/            # Position generation algorithms
    ├── constants/                  # Enums and constant definitions
    └── utils/                      # Utility functions and helpers

**Import Patterns by Use Case**:

.. code-block:: python

    # Basic pipeline creation (most common)
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.constants.constants import VariableComponents

    # GPU processing functions
    from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize
    from openhcs.processing.backends.processors.cupy_processor import tophat

    # Analysis functions
    from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel

    # Configuration and enums
    from openhcs.core.config import PathPlanningConfig, VFSConfig, ZarrConfig
    from openhcs.constants.constants import Backend, Microscope

.. _module-key-directories:

Key Modules and Their Purpose
-----------------------------

core/
^^^^^

The ``core/`` module contains the fundamental orchestration and pipeline management components:

- **orchestrator/**: PipelineOrchestrator for multi-well execution
- **steps/**: FunctionStep implementation with GPU support
- **config/**: Configuration classes for system-wide settings
- **context/**: ProcessingContext for execution state management

processing/backends/
^^^^^^^^^^^^^^^^^^^^

The ``processing/backends/`` module provides seamless integration with major Python image processing libraries, organized by computational backend:

- **processors/**: Image processing functions (torch, cupy, numpy)
- **analysis/**: Analysis and measurement functions
- **assemblers/**: Image assembly and stitching algorithms
- **pos_gen/**: Position generation for stitching

constants/
^^^^^^^^^^

The ``constants/`` module defines enums and constants used throughout OpenHCS:

- **VariableComponents**: SITE, CHANNEL, TIME processing dimensions
- **Backend**: Memory backend types (MEMORY, DISK, ZARR)
- **Microscope**: Supported microscope types and detection

**Doctrinal Motivation**: Enforces clear separation of concerns, facilitates polymorphism, and is crucial for breaking import cycles. Implementations depend on these interfaces, not on each other directly. Supports ``Clause 21`` (Frontloaded Validation) by making dependencies explicit.

schemas/
^^^^^^^^

The ``schemas/`` directory contains all Pydantic models or other schema definitions used for configuration, data validation, and context management. These schemas define the structure of data that flows through the system.

**Doctrinal Motivation**: Enforces ``Clause 21`` (Frontloaded Validation) by providing a single source of truth for data structures. Promotes ``Clause 66`` (Context Immunity) by clearly defining the structure of context objects.

registries/
^^^^^^^^^^^

The ``registries/`` directory contains modules responsible for the registration and discovery of pluggable components (backends, handlers, steps). Each registry follows an ``initialize_foo()`` pattern for explicit, controlled initialization.

**Doctrinal Motivation**: Decouples component definition from usage. Prevents registration side-effects on module import. Ensures initialization is explicit and traceable, supporting testability and ``Clause 3`` (Statelessness) by controlling when stateful registries are populated.

backends/
^^^^^^^^^

The ``backends/`` directory contains concrete implementations of interfaces defined in ``interfaces/``. Each backend (e.g., MIST, Ashlar) resides in its own sub-package.

**Doctrinal Motivation**: Clear separation of implementation from interface. ``__init__.py`` in this directory and its subdirectories are minimal to prevent accidental registration on import.

io/backends/
^^^^^^^^^^^^

The ``io/backends/`` directory could contain different storage backend implementations.

**Doctrinal Motivation**: Similar to ``backends/``, separates storage interface implementations from their definition.

.. _module-initialization:

Initialization Discipline
------------------------

OpenHCS follows a strict initialization discipline to prevent side-effects on import and ensure explicit control over component registration:

1. **No Registration at Module Load**: Processing backends, GPU schedulers, and pipeline components are not registered when their respective modules are imported.

2. **Explicit Setup Pattern**: All registries provide explicit setup functions that perform the actual registration of available implementations.

3. **Import-Safe Initialization Points**:
   - TUI Application: The TUI calls necessary setup functions during application startup.
   - Script Generation: Generated scripts include proper initialization code.
   - Orchestrator Usage: Applications using OpenHCS as a library must call setup functions explicitly.

**Doctrinal Motivation**: Ensures that the application state (which components are available) is explicitly managed and not a side-effect of imports. This improves predictability, testability, and makes the set of available components deterministic at initialization.

.. _module-public-api:

Public API
---------

OpenHCS provides a stable public API through the ``openhcs`` package. This API is carefully designed to be safe to import without triggering side-effects:

.. code-block:: python

    # Safe to import - no side effects
    import openhcs

    # Initialize openhcs before using
    openhcs.initialize()

    # Now use the API
    config = openhcs.create_config(input_dir="path/to/images")
    results = openhcs.run_pipeline(config)

The public API is defined in ``openhcs/api.py`` and re-exported by ``openhcs/__init__.py``. This ensures that ``import openhcs`` is safe and does not trigger backend registrations or other internal initializations.

.. _module-doctrinal-compliance:

Doctrinal Compliance
-------------------

OpenHCS's module structure is designed to comply with the following doctrinal clauses:

- **Clause 3 (Statelessness)**: Explicit initialization of registries and clear separation of concerns help in designing components that are individually stateless or whose state is managed explicitly.

- **Clause 12 (Smell Intolerance)**: When fetching from a registry, if an item is not found, a deterministic error is raised. No trying alternative names or default fallbacks.

- **Clause 17 (VFS Exclusivity)**: The ``openhcs/core/io/file_manager.py`` module is the primary interaction point for file system operations, using the VFS system. Other modules depend on this for I/O.

- **Clause 21 (Frontloaded Validation)**: Interfaces define explicit contracts. Configuration classes define data dependencies. Registries make component availability explicit rather than implicit through imports.

- **Clause 65 (Absolute Execution)**: Clear interfaces and explicit registration reduce the need for ``hasattr`` or ``try-except`` blocks for probing capabilities.

- **Clause 66 (Context Immunity)**: The ``openhcs/core/context/processing_context.py`` manages context explicitly. Components declare their context needs via configuration classes.

- **Clause 77 (Rot Intolerance)**: The refactor provides an opportunity to identify and prune unused modules or consolidate overly fragmented ones. Clearer directory responsibilities make rot more apparent.

.. _module-best-practices:

Best Practices
------------

When working with OpenHCS's module structure, follow these best practices:

1. **Import from Public API**: Import from ``openhcs.core`` and ``openhcs.processing`` rather than directly from implementation modules.

2. **Use Configuration Classes**: Define data structures using configuration classes in ``openhcs.core.config`` before using them.

3. **Setup Components Explicitly**: Setup GPU registries and processing backends explicitly, not implicitly on import.

4. **Initialize Before Use**: Call appropriate setup functions before using GPU-accelerated components.

5. **Respect Unidirectional Dependencies**: Ensure dependencies flow in one direction to prevent cycles:
   - Abstract classes should not depend on implementations
   - Configuration should not depend on implementations
   - Implementations should depend on abstract classes and configuration
   - Registries should depend on interfaces, not implementations

6. **Use VirtualPath for I/O**: Always use ``VirtualPath`` for file system operations, not ``Path`` or ``str``.

7. **Declare Context Dependencies**: Use ``StepFieldDependency`` to declare context field dependencies, not direct access to context attributes.

8. **Avoid Runtime Flexibility**: Don't vary behavior based on field presence or state. Use explicit schemas and validation.

9. **Eliminate Dead Code**: Remove unused code, procedural glue, or legacy compatibility layers.

10. **Write Structural Tests**: Tests should enforce structure, not behavior. Use tests in ``tests/rot/`` to verify doctrinal compliance.