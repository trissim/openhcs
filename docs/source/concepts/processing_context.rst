===================
ProcessingContext
===================

Overview
--------

The ``ProcessingContext`` is a crucial component that maintains state during pipeline execution. It:

* Holds input/output directories, well filter, and configuration
* Stores processing results
* Serves as a communication mechanism between steps

Creating a Context
----------------

The context is typically created by the pipeline, but you can create it manually for advanced usage:

.. code-block:: python

    from ezstitcher.core.pipeline import ProcessingContext

    # Create a processing context
    context = ProcessingContext(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator,  # Reference to the PipelineOrchestrator
        # Additional attributes can be added as kwargs
        positions_file="path/to/positions.csv",
        custom_parameter=42
    )

Accessing Context Attributes
--------------------------

Context attributes can be accessed directly:

.. code-block:: python

    # Access standard attributes
    print(context.input_dir)
    print(context.well_filter)

    # Access custom attributes
    print(context.positions_file)
    print(context.custom_parameter)

Accessing the Orchestrator
------------------------

The context provides access to the orchestrator, allowing steps to use plate-specific services. Here's how it's done in the actual Step class:

.. code-block:: python

    # From ezstitcher/core/steps.py - Step.process method
    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Process the step with the given context.
        """
        logger.info("Processing step: %s", self.name)

        # Get directories and microscope handler
        input_dir = self.input_dir
        output_dir = self.output_dir
        well_filter = self.well_filter or context.well_filter
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        microscope_handler = orchestrator.microscope_handler

        if not input_dir:
            raise ValueError("Input directory must be specified")

        # ... rest of the method ...

Steps use the orchestrator's high-level methods for specialized operations:

.. code-block:: python

    # From ezstitcher/core/steps.py - PositionGenerationStep.process method
    def process(self, context):
        # Get required objects from context
        well = context.well_filter[0] if context.well_filter else None
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        input_dir = self.input_dir or context.input_dir
        positions_dir = self.output_dir or context.output_dir

        # Call the generate_positions method
        positions_file, reference_pattern = orchestrator.generate_positions(well, input_dir, positions_dir)

        # Store in context
        context.positions_dir = positions_dir
        context.reference_pattern = reference_pattern
        return context

    # From ezstitcher/core/steps.py - ImageStitchingStep.process method
    def process(self, context):
        # Get orchestrator from context
        orchestrator = getattr(context, 'orchestrator', None)
        if not orchestrator:
            raise ValueError("ImageStitchingStep requires an orchestrator in the context")

        # Call the stitch_images method
        orchestrator.stitch_images(
            well=context.well,
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            positions_file=positions_file
        )

        return context
