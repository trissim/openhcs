API Reference
=============

The API reference documentation is currently being updated to reflect the latest OpenHCS architecture and remove outdated content.

For now, please refer to:

**Core Concepts**: :doc:`../concepts/index`
  Understanding pipelines, steps, function patterns, and system architecture

**Getting Started**: :doc:`../getting_started/getting_started`
  Basic examples and common usage patterns

**Function Library**: :doc:`../concepts/function_library`
  Available processing functions and library integrations

Key Classes
-----------

**PipelineOrchestrator**

.. code-block:: python

    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.config import GlobalPipelineConfig

    orchestrator = PipelineOrchestrator(
        plate_path="/path/to/data",
        global_config=GlobalPipelineConfig(num_workers=4)
    )

**FunctionStep**

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    step = FunctionStep(
        func=(processing_function, {'param': value}),
        variable_components=[VariableComponents.SITE],
        name="step_name"
    )

**Pipeline**

.. code-block:: python

    from openhcs.core.pipeline import Pipeline

    pipeline = Pipeline([step1, step2, step3])
    orchestrator.run_pipeline(pipeline)

The complete API reference will be restored with updated examples and correct module paths.
