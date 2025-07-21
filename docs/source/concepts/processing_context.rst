===================
ProcessingContext
===================

Overview
--------

The ``ProcessingContext`` is the canonical state manager for OpenHCS pipeline execution. It maintains immutable execution state after compilation and provides structured access to configuration and resources.

**Key Characteristics**:

* **Immutable After Compilation**: Context is frozen after compilation to ensure thread-safe execution
* **VFS Integration**: All file operations go through the FileManager instance
* **Configuration Management**: Provides access to GlobalPipelineConfig and step-specific plans
* **Well-Specific State**: Each well gets its own context instance for parallel processing

**Real-World Usage** (from TUI-generated scripts):

.. code-block:: python

    # Context creation is handled by PipelineOrchestrator
    orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)
    orchestrator.initialize()

    # Compilation creates frozen contexts for each well
    compiled_contexts = orchestrator.compile_pipelines(steps)

    # Each context is well-specific and immutable
    for well_id, context in compiled_contexts.items():
        print(f"Well {well_id}: {context.well_id}")
        print(f"Frozen: {context.is_frozen()}")
        print(f"Step plans: {len(context.step_plans)}")

**Context Ownership**:

.. note::
   ProcessingContext instances may ONLY be created by PipelineOrchestrator. All other components must receive a context instance, never create one directly.

Context Attributes and Structure
---------------------------------

The ProcessingContext contains several key attributes for pipeline execution:

.. code-block:: python

    from openhcs.core.context.processing_context import ProcessingContext

    # Context attributes (read-only after freezing)
    context.well_id              # Well identifier (e.g., "A01")
    context.global_config        # GlobalPipelineConfig instance
    context.filemanager          # FileManager for VFS operations
    context.step_plans           # Dict mapping step IDs to execution plans
    context.outputs              # Step outputs (VFS-centric model)
    context.intermediates        # Intermediate results
    context.current_step         # Currently executing step ID

**Step Plan Access**:

.. code-block:: python

    # Access step-specific execution plans
    for step_id, plan in context.step_plans.items():
        print(f"Step {step_id}: {plan}")

    # Check if context is frozen (immutable)
    if context.is_frozen():
        print("Context is immutable - safe for parallel execution")

**Configuration Access**:

.. code-block:: python

    # Access global configuration through context
    num_workers = context.global_config.num_workers
    vfs_config = context.global_config.vfs
    zarr_config = context.global_config.zarr

    # Access VFS operations through filemanager
    context.filemanager.exists("path/to/file")
    context.filemanager.save_array(array, "output/path")

Context Lifecycle and Immutability
-----------------------------------

OpenHCS ProcessingContext follows a strict lifecycle to ensure thread-safe parallel execution:

**Phase 1: Creation and Configuration**:

.. code-block:: python

    # Created by PipelineOrchestrator during compilation
    context = ProcessingContext(
        global_config=global_config,
        well_id="A01",
        filemanager=filemanager
    )

    # Step plans are injected during compilation
    context.inject_plan("step_1", {
        "func": normalize_function,
        "parameters": {"low_percentile": 1.0},
        "variable_components": ["site"]
    })

**Phase 2: Freezing for Execution**:

.. code-block:: python

    # Context is frozen after compilation
    context.freeze()

    # After freezing, context becomes immutable
    try:
        context.well_id = "B01"  # This will raise AttributeError
    except AttributeError:
        print("Cannot modify frozen context")

**Phase 3: Execution Access**:

.. code-block:: python

    # During execution, functions access context read-only
    def processing_function(images, context):
        # Access configuration
        config = context.global_config

        # Access step-specific plans
        current_plan = context.step_plans.get("current_step_id")

        # Use filemanager for VFS operations
        context.filemanager.save_array(processed_images, "output/path")

        return processed_images

**Thread Safety**:

The frozen context ensures that multiple worker threads can safely access the same context instance without race conditions or data corruption.

See Also
--------

**Technical Details**:

- :doc:`../architecture/pipeline_compilation_system` - How contexts are compiled and frozen
- :doc:`../architecture/vfs_system` - VFS integration through FileManager

**Related Concepts**:

- :doc:`pipeline_orchestrator` - Context creation and management
- :doc:`step` - How FunctionSteps use context during execution
- :doc:`../api/config` - GlobalPipelineConfig structure and options
