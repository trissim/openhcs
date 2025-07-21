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

Module Directory Structure
-------------------------

EZStitcher's module structure follows this organization:

.. code-block:: text

    openhcs/
    ├── __init__.py                 # Minimal; re-exports public API from openhcs.api
    ├── __main__.py                 # CLI entry point; calls initialize_foo() from registries
    ├── interfaces/                 # Abstract Base Classes and Interface definitions
    │   ├── __init__.py
    │   ├── compute.py              # Defines ComputeBackend, ImageAssembler interfaces
    │   ├── storage.py              # Defines StorageInterface
    │   ├── pipeline.py             # Defines PipelineStepInterface, PipelineInterface
    │   └── handlers.py             # Defines HandlerInterface
    ├── schemas/                    # All Pydantic/JSON schema definitions
    │   ├── __init__.py
    │   ├── config_schemas.py       # General application/pipeline config schemas
    │   ├── context_schemas.py      # ProcessingContext schema, StepState schemas
    │   └── backend_schemas.py      # Schemas for backend configurations
    ├── core/                       # Core orchestration, business logic, context management
    │   ├── __init__.py
    │   ├── pipeline_executor.py    # Orchestrates pipeline step execution
    │   ├── processing_context.py   # Manages processing state
    │   └── ...                     # Other core components
    ├── engine/                     # Low-level execution mechanics for pipeline steps
    │   ├── __init__.py
    │   └── step_execution.py       # Logic for executing individual steps
    ├── backends/                   # Concrete implementations of interfaces
    │   ├── __init__.py             # EMPTY or minimal; DOES NOT import specific backends
    │   └── mist/                   # MIST backend implementation
    │       ├── __init__.py
    │       ├── implementation.py   # Implements interfaces.compute.ComputeBackend
    │       ├── config.py           # MIST specific config logic
    │       └── ...                 # Other MIST-specific files
    ├── io/                         # Input/Output operations, VFS, data loading
    │   ├── __init__.py
    │   ├── file_manager.py         # VFS interactions (Clause 17)
    │   ├── image_io.py             # Image reading/writing utilities
    │   ├── ...                     # Other I/O utilities
    │   └── storage_adapters/       # Concrete storage implementations
    │       ├── __init__.py
    │       └── local_fs_adapter.py # Implements interfaces.storage.StorageInterface
    ├── registries/                 # Modules for registering and discovering components
    │   ├── __init__.py             # EMPTY or minimal
    │   ├── compute_backends.py     # Manages ComputeBackend registry
    │   ├── image_assemblers.py     # Manages ImageAssembler registry
    │   ├── storage_providers.py    # Manages StorageInterface registry
    │   └── pipeline_steps.py       # Manages PipelineStepInterface registry
    ├── microscopes/                # Microscope-specific data adapters/parsers
    │   ├── __init__.py
    │   └── ...                     # Microscope-specific implementations
    ├── materialization/            # Materialization logic
    │   ├── __init__.py
    │   └── ...                     # Materialization-specific implementations
    └── ez/                         # Public-facing simplified API layer
        ├── __init__.py
        ├── api.py                  # Consolidates user-facing functions and classes
        └── utils.py                # Publicly exposed utilities

.. _module-key-directories:

Key Directories and Their Purpose
--------------------------------

interfaces/
^^^^^^^^^^^

The ``interfaces/`` directory contains abstract base classes (ABCs) and protocol definitions that define the contracts for various components. These interfaces are the foundation of the system's architecture and ensure that implementations adhere to a consistent API.

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

io/storage_adapters/
^^^^^^^^^^^^^^^^^^^

The ``io/storage_adapters/`` directory contains concrete implementations of the ``StorageInterface`` defined in ``interfaces/storage.py``.

**Doctrinal Motivation**: Similar to ``backends/``, separates storage interface implementations from their definition.

.. _module-initialization:

Initialization Discipline
------------------------

EZStitcher follows a strict initialization discipline to prevent side-effects on import and ensure explicit control over component registration:

1. **No Registration at Module Load**: Backends, handlers, plugins, and pipeline steps are not registered when their respective modules are imported.

2. **initialize_foo() Pattern**: All registries provide an explicit initialization function (e.g., ``initialize_compute_backends()``) that performs the actual registration of available implementations.

3. **Import-Safe Initialization Points**:
   - CLI (``ezstitcher/__main__.py``): The main CLI entry point calls all necessary ``initialize_foo()`` functions at startup.
   - Test Bootstraps (``tests/conftest.py`` or specific test setups): Tests explicitly call ``initialize_foo()`` to set up the required components for a given test scenario.
   - Orchestrators/Application Entry Points: Any other application using ``ezstitcher`` as a library is responsible for calling these initialization functions.

**Doctrinal Motivation**: Ensures that the application state (which components are available) is explicitly managed and not a side-effect of imports. This improves predictability, testability, and helps avoid ``Clause 74`` (Runtime Flexibility Forbidden) by making the set of available components deterministic at initialization.

.. _module-public-api:

Public API
---------

EZStitcher provides a stable public API through the ``ezstitcher`` package. This API is carefully designed to be safe to import without triggering side-effects:

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

- **Clause 17 (VFS Exclusivity)**: The ``ezstitcher/io/file_manager.py`` module is the primary interaction point for file system operations, using ``VirtualPath``. Other modules depend on this for I/O.

- **Clause 21 (Frontloaded Validation)**: Interfaces define explicit contracts. Schemas define data dependencies. Registries make component availability explicit rather than implicit through imports.

- **Clause 65 (Absolute Execution)**: Clear interfaces and explicit registration reduce the need for ``hasattr`` or ``try-except`` blocks for probing capabilities.

- **Clause 66 (Context Immunity)**: Centralizing schemas in ``ezstitcher/schemas/`` (especially ``context_schemas.py``) and having ``ezstitcher/core/processing_context.py`` manage context explicitly helps. Components declare their context needs via these schemas.

- **Clause 77 (Rot Intolerance)**: The refactor provides an opportunity to identify and prune unused modules or consolidate overly fragmented ones. Clearer directory responsibilities make rot more apparent.

.. _module-best-practices:

Best Practices
------------

When working with EZStitcher's module structure, follow these best practices:

1. **Import Interfaces, Not Implementations**: Import from ``ezstitcher.interfaces`` rather than directly from implementation modules.

2. **Use Schemas for Data Validation**: Define data structures using schemas in ``ezstitcher.schemas`` before using them.

3. **Register Components Explicitly**: Register components using the appropriate registry's registration function, not implicitly on import.

4. **Initialize Before Use**: Call ``ezstitcher.initialize()`` before using any other functions in the API.

5. **Respect Unidirectional Dependencies**: Ensure dependencies flow in one direction to prevent cycles:
   - Interfaces should not depend on implementations
   - Schemas should not depend on implementations
   - Implementations should depend on interfaces and schemas
   - Registries should depend on interfaces, not implementations

6. **Use VirtualPath for I/O**: Always use ``VirtualPath`` for file system operations, not ``Path`` or ``str``.

7. **Declare Context Dependencies**: Use ``StepFieldDependency`` to declare context field dependencies, not direct access to context attributes.

8. **Avoid Runtime Flexibility**: Don't vary behavior based on field presence or state. Use explicit schemas and validation.

9. **Eliminate Dead Code**: Remove unused code, procedural glue, or legacy compatibility layers.

10. **Write Structural Tests**: Tests should enforce structure, not behavior. Use tests in ``tests/rot/`` to verify doctrinal compliance.