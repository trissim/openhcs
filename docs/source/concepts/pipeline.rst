.. _pipeline-concept:

========
Pipeline
========

Overview
--------

In OpenHCS, a **pipeline** is a sequence of processing operations that are executed in order on your data. The concept is simple, powerful, and central to how the system works. Unlike in previous versions, a pipeline is not a complex object but simply a Python **list of `FunctionStep` instances**.

This list defines the *what* and in what order processing should occur. The :doc:`pipeline_orchestrator` then handles the *how*—compiling this list into an executable plan and running it efficiently across your dataset.

Key characteristics of an OpenHCS pipeline:

*   **It's just a list**: There is no special ``Pipeline`` class to instantiate. A pipeline is just `[step1, step2, step3, ...]`.
*   **Sequential Execution**: The steps in the list are executed sequentially for each unit of work (e.g., for each well). The output of one step becomes the input to the next.
*   **VFS-Powered**: The data flow between steps happens transparently within the in-memory Virtual File System (VFS), eliminating intermediate disk I/O.
*   **Compiled, then Executed**: The entire list of steps is passed to the orchestrator's ``compile_pipelines`` method, which validates the workflow and builds an optimized execution plan before any processing begins.

Creating a Pipeline
-------------------

Building a pipeline involves creating instances of ``FunctionStep`` and adding them to a list. Each `FunctionStep` encapsulates a specific task, such as image normalization, analysis, or assembly.

**Example Pipeline Definition**:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents
    from openhcs.processing.backends.processors.cupy_processor import tophat, create_composite
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # An OpenHCS pipeline is a simple Python list of steps
    pipeline_steps = [
        # Step 1: Preprocessing chain applied to each site
        FunctionStep(
            func=[
                (tophat, {'selem_radius': 50, 'downsample_factor': 4})
            ],
            name="preprocess",
            variable_components=[VariableComponents.SITE]
        ),

        # Step 2: Generate stitching positions for each site
        FunctionStep(
            func=ashlar_compute_tile_positions_gpu,
            name="find_stitch_positions",
            variable_components=[VariableComponents.SITE]
        ),

        # Step 3: Assemble the final stitched image for each well
        FunctionStep(
            func=assemble_stack_cupy,
            name="assemble",
            variable_components=[VariableComponents.SITE],
            force_disk_output=True  # Materialize the final result
        )
    ]

Executing a Pipeline
--------------------

The pipeline (your list of steps) is not run directly. Instead, it is passed to the ``PipelineOrchestrator``, which manages the entire three-phase workflow:

.. code-block:: python

    # (Assuming orchestrator and global_config are defined)

    # 1. Initialize
    orchestrator.initialize()

    # 2. Compile the list of steps into an executable plan
    compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)

    # 3. Execute the compiled plan
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline_steps,
        compiled_contexts=compiled_contexts,
        max_workers=global_config.num_workers
    )

This clean separation of concerns—where the user defines a simple list and the orchestrator handles the complex execution—is a core design principle of OpenHCS.
