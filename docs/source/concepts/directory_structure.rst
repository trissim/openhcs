.. _directory-structure:

===================
Directory Structure
===================

.. _directory-overview:

Overview
--------

EZStitcher uses a structured approach to directory management that balances automation with flexibility. This document explains how directories are managed, resolved, and customized in EZStitcher.

For information about how pipelines handle directories, see :doc:`pipeline`.
For information about how steps handle directories, see :doc:`step`.

.. _directory-basic-concepts:

Basic Directory Concepts
-----------------------

In EZStitcher, several key directories are used during processing:

* **Plate Path**: The original directory containing microscopy images
* **Workspace Path**: A copy of the plate path with symlinks to protect original data
* **Input Directory**: Where a step reads images from
* **Output Directory**: Where a step saves processed images
* **Positions Directory**: Where position files for stitching are saved
* **Stitched Directory**: Where final stitched images are saved

.. _directory-default-structure:

Default Directory Structure
-------------------------

When you run a pipeline, EZStitcher creates a directory structure as steps are executed:

.. code-block:: text

    /path/to/plate/                  # Original plate path
    /path/to/plate_workspace/        # Workspace with symlinks to original images
    /path/to/plate_workspace_out/    # Processed images (configurable suffix)
    /path/to/plate_workspace_positions/  # Position files for stitching (configurable suffix)
    /path/to/plate_workspace_stitched/   # Stitched images (configurable suffix)

This structure ensures that:

1. Original data is protected (via the workspace)
2. Processed images are kept separate from original images
3. Position files are stored in a dedicated directory
4. Stitched images are stored separately from individual processed tiles

.. _directory-resolution:

Directory Resolution
------------------

For detailed API documentation, see:

* :doc:`../api/pipeline_orchestrator`
* :doc:`../api/pipeline`
* :doc:`../api/steps`

EZStitcher automatically resolves directories for steps in a pipeline, minimizing the need for manual directory management. Here's how it works:

1. **Basic Resolution Logic**:

   .. code-block:: text

       Pipeline Input Dir → Step 1 → Step 2 → Step 3 → ... → Pipeline Output Dir
                            |         |         |
                            v         v         v
                         Output 1  Output 2  Output 3

   - Each step's output directory becomes the next step's input directory
   - If a step doesn't specify an output directory, it's automatically generated
   - The pipeline's output directory is used for the last step if not specified

2. **First Step Special Handling**:
   - If the first step doesn't specify an input directory, the pipeline's input directory is used
   - Typically, you should set the first step's input directory to ``orchestrator.workspace_path``

3. **Default Directory Generation**:
   - The first step always gets a new output directory (with "_out" suffix) if none is specified
   - This ensures we never modify files in the workspace path
   - Subsequent steps will use their input directory as their output directory (in-place processing) if no output directory is specified
   - This allows for more efficient processing by avoiding unnecessary file copying

4. **ImageStitchingStep Behavior**:
   - The ``ImageStitchingStep`` follows the standard directory resolution logic, using the previous step's output directory as its input
   - You can explicitly set ``input_dir=orchestrator.workspace_path`` to use original images for stitching instead of processed images
   - By default, its output directory is set to ``{workspace_path}_stitched``
   - This ensures stitched images are saved separately from processed individual tiles

.. _directory-example-flow:

Example Directory Flow
--------------------

Here's an example of how directories flow through a pipeline:

.. code-block:: text

    # Starting with a plate path: /data/plates/plate1

    orchestrator.workspace_path = /data/plates/plate1_workspace

    # Pipeline with 3 steps:

    Step 1 (Z-Stack Flattening):
      input_dir = /data/plates/plate1_workspace
      output_dir = /data/plates/plate1_workspace_out  # New directory to protect workspace

    Step 2 (Channel Processing):
      input_dir = /data/plates/plate1_workspace_out
      output_dir = /data/plates/plate1_workspace_out  # In-place processing

    Step 3 (Position Generation):
      input_dir = /data/plates/plate1_workspace_out
      output_dir = /data/plates/plate1_workspace_positions  # New directory for position files

    Step 4 (Image Stitching):
      input_dir = /data/plates/plate1_workspace_positions  # Uses previous step's output by default
      # Alternative: input_dir = /data/plates/plate1_workspace  # Can be set to use original images instead
      positions_dir = /data/plates/plate1_workspace_positions  # Same as input_dir
      output_dir = /data/plates/plate1_workspace_stitched  # New directory for stitched images

This automatic directory resolution simplifies pipeline creation and ensures a consistent directory structure.

.. _directory-step-initialization:

Step Initialization Best Practices
--------------------------------

When initializing steps, follow these best practices for directory specification:

1. **First Step in a Pipeline**:
   - Always specify ``input_dir`` for the first step, typically using ``orchestrator.workspace_path``
   - This ensures that processing happens on the workspace copies, not the original data
   - Specify ``output_dir`` only if you need a specific directory structure

   .. code-block:: python

       # First step in a pipeline
       first_step = Step(
           name="First Step",
           func=IP.stack_percentile_normalize,
           input_dir=orchestrator.workspace_path,  # Always specify for first step
           # output_dir is automatically determined
       )

2. **Subsequent Steps**:
   - Don't specify ``input_dir`` for subsequent steps
   - Each step's output directory automatically becomes the next step's input directory
   - Specify ``output_dir`` only if you need a specific directory structure

   .. code-block:: python

       # Subsequent step in a pipeline
       subsequent_step = Step(
           name="Subsequent Step",
           func=stack(IP.sharpen),
           # input_dir is automatically set to previous step's output_dir
           # output_dir is automatically determined
       )

3. **Specialized Steps**:
   - For ``PositionGenerationStep``, don't specify ``input_dir`` or ``output_dir`` unless needed
   - For ``ImageStitchingStep``, don't specify ``input_dir``, ``positions_dir``, or ``output_dir`` unless needed

   .. code-block:: python

       # Directories are automatically determined
       position_step = PositionGenerationStep()

       # Directories are automatically determined
       stitch_step = ImageStitchingStep(
           # Uncomment to use original images instead of processed images:
           # input_dir=orchestrator.workspace_path
       )

4. **Common Mistakes to Avoid**:
   - Specifying unnecessary directories, making the code more verbose
   - Forgetting to use ``orchestrator.workspace_path`` for the first step
   - Manually managing directories that could be automatically resolved

Following these best practices will make your code more concise and less error-prone, while taking full advantage of EZStitcher's automatic directory resolution.

.. _directory-custom-structures:

Custom Directory Structures
-------------------------

While EZStitcher's automatic directory resolution works well for most cases, you may sometimes need more control over where files are saved.

You can create custom directory structures by explicitly specifying output directories:

.. code-block:: python

    # Create a pipeline with custom directory structure
    pipeline = Pipeline(
        steps=[
            # First step: Save to a specific directory
            Step(
                name="Z-Stack Flattening",
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("/custom/output/path/flattened")
            ),

            # Second step: Save to another specific directory
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                group_by='channel',
                # input_dir is automatically set to the previous step's output_dir
                output_dir=Path("/custom/output/path/processed")
            ),

            # Image stitching step: Save to a specific directory
            ImageStitchingStep(
                # input_dir is automatically set to the previous step's output_dir
                # positions_dir is automatically determined
                output_dir=Path("/custom/output/path/stitched")
            )
        ],
        name="Custom Directory Pipeline"
    )

.. _directory-customizing-stitching:

Customizing ImageStitchingStep Directories
----------------------------------------

For more control over the ImageStitchingStep directories:

.. code-block:: python

    pipeline = Pipeline(
        steps=[
            # Processing steps...

            # Custom position generation step
            PositionGenerationStep(
                # input_dir is automatically set
                output_dir=Path("/custom/positions")  # Custom positions directory
            ),

            # Custom image stitching step
            ImageStitchingStep(
                input_dir=Path("/custom/input"),  # Custom input directory
                positions_dir=Path("/custom/positions"),  # Custom positions directory
                output_dir=Path("/custom/stitched")  # Custom output directory
            )
        ],
        name="Custom Stitching Pipeline"
    )

.. _directory-when-to-specify:

When to Specify Directories Explicitly
------------------------------------

1. **Always specify input_dir for the first step**:
   - Use `orchestrator.workspace_path` to ensure processing happens on workspace copies
   - This protects original data from modification

2. **Specify output_dir only when you need a specific directory structure**:
   - For example, when you need to save results in a specific location
   - When you need to reference the output directory from outside the pipeline

3. **Don't specify input_dir for subsequent steps**:
   - Each step's output directory automatically becomes the next step's input directory
   - This reduces code verbosity and potential for errors

4. **Don't specify directories for steps unless needed**:
   - `PositionGenerationStep` and `ImageStitchingStep` have intelligent directory handling
   - They automatically find the right directories based on the pipeline context

.. _directory-configuring-suffixes:

Configuring Directory Suffixes
-------------------------

EZStitcher allows you to configure the directory suffixes used for different types of steps through the `PipelineConfig` class:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig

    # Create a configuration with custom directory suffixes
    config = PipelineConfig(
        out_dir_suffix="_output",           # For regular processing steps (default: "_out")
        positions_dir_suffix="_pos",        # For position generation steps (default: "_positions")
        stitched_dir_suffix="_stitched"     # For stitching steps (default: "_stitched")
    )

    # Create an orchestrator with the custom configuration
    orchestrator = PipelineOrchestrator(config=config, plate_path=plate_path)

    # Now all pipelines run with this orchestrator will use the custom suffixes
    pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        name="Basic Pipeline",
        steps=[
            Step(name="First Step", func=IP.stack_percentile_normalize),
            PositionGenerationStep(),
            ImageStitchingStep()
        ]
    )

    # Run the pipeline
    orchestrator.run(pipelines=[pipeline])

This allows you to customize the directory structure to match your organization's naming conventions or to integrate with existing workflows.

.. _directory-best-practices:

Best Practices
------------

For comprehensive best practices for directory management, see :ref:`best-practices-directory` in the :doc:`../user_guide/best_practices` guide.
