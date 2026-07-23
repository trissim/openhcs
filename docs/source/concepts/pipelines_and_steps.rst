Pipelines and steps
===================

An OpenHCS pipeline is an ordered ``list[FunctionStep]`` interpreted together
with a ``PipelineConfig``. The declarations say what should happen; compilation
decides the exact work for sources, execution axes, artifacts, memory, and
workers.

FunctionStep
------------

``FunctionStep`` accepts a function pattern and step-level configuration:

.. code-block:: python

   from openhcs.constants import VariableComponents
   from openhcs.core.config import LazyProcessingConfig, ProcessingConfig
   from openhcs.core.steps.function_step import FunctionStep

   def normalize(image, *, low=1.0, high=99.0):
       return image

   processing = ProcessingConfig(
       variable_components=(VariableComponents.SITE,),
   )
   normalize_step = FunctionStep(
       func=(normalize, {"low": 1.0, "high": 99.0}),
       name="normalize",
       processing_config=LazyProcessingConfig.from_config(processing),
   )
   pipeline_steps = [normalize_step]

Processing and materialization fields are not direct ``FunctionStep`` keyword
arguments. They belong in the corresponding nested configuration object.

Function patterns
-----------------

The ``func`` field accepts:

- a callable;
- ``(callable, kwargs)``;
- a list containing either form, executed as a chain;
- a dictionary mapping compiled group keys to either form.

Callable contracts—not examples or UI labels—are the authority for accepted
parameters, runtime-injected parameters, required axes, processing behavior,
memory types, and artifacts.

Data flow
---------

By default a step consumes the previous step's main-flow result. Processing
configuration can select pipeline-start input, and source-binding configuration
can supply named semantic inputs. Artifact contracts can additionally consume
typed results from prior producers.

The compiler determines how each input is satisfied. Runtime code does not
search paths or infer dependency names again.

PipelineConfig
--------------

``PipelineConfig`` carries pipeline defaults and source declarations. Lazy
step-level configurations inherit through ObjectState from the pipeline and
global scopes. CellProfiler import returns a populated ``PipelineConfig`` plus
the ordered steps it lowered.

Compile before execution
------------------------

``PipelineOrchestrator.compile_pipelines(pipeline_steps)`` resolves the pipeline
once and compiles every selected execution axis. The returned
``execution_bundle`` contains runtime contexts, worker steps, and environment
decisions. The GUI uses the same boundary and compiles the complete selected set
before executing it.

Related pages
-------------

- :doc:`function_patterns`
- :doc:`data_dimensions`
- :doc:`../architecture/processing_semantics`
- :doc:`../architecture/pipeline_compilation_system`
