Public API orientation
======================

OpenHCS currently exposes a small declaration boundary and a lower-level
compile/execute boundary. This page intentionally documents those current
surfaces rather than a nonexistent ``Pipeline`` wrapper.

Read :doc:`../architecture/quick_start` first for the shared desktop, Python,
CellProfiler, and MCP route.

Pipeline declarations
---------------------

``FunctionStep`` wraps a callable, tuple-with-keyword-arguments, callable chain,
or dictionary function pattern. Step processing semantics belong in its nested
configuration objects.

.. code-block:: python

   from openhcs.constants import VariableComponents
   from openhcs.core.config import LazyProcessingConfig, ProcessingConfig
   from openhcs.core.steps.function_step import FunctionStep

   def normalize(image, *, scale=1.0):
       return image * scale

   processing = ProcessingConfig(
       variable_components=(VariableComponents.SITE,),
   )
   step = FunctionStep(
       func=(normalize, {"scale": 0.5}),
       name="normalize",
       processing_config=LazyProcessingConfig.from_config(processing),
   )
   pipeline_steps = [step]

Do not pass ``variable_components``, ``group_by``, or materialization fields
directly to ``FunctionStep``. They are owned by ``processing_config`` and the
relevant materialization configuration.

CellProfiler import
-------------------

.. code-block:: python

   from openhcs.interop.cellprofiler.pipeline_import import (
       import_cellprofiler_pipeline,
   )

   pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
       "analysis.cppipe",
       source_root="/data/plate",
   )

The result contains ordinary ``FunctionStep`` declarations and a
``PipelineConfig``. There is no generated runtime-pipeline object or semantic
sidecar.

Compilation and execution
-------------------------

``PipelineOrchestrator.compile_pipelines`` returns a compatibility result whose
``execution_bundle`` entry is the typed ``CompiledExecutionBundle`` consumed by
execution.

.. code-block:: python

   from pathlib import Path

   from objectstate import ensure_global_config_context
   from openhcs.core.config import GlobalPipelineConfig
   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

   plate_path = Path("/data/plate").resolve()
   ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
   orchestrator = PipelineOrchestrator(
       plate_path,
       pipeline_config=pipeline_config,
   ).initialize()

   compilation = orchestrator.compile_pipelines(pipeline_steps)
   execution_bundle = compilation["execution_bundle"]

   progress_context = {
       "execution_id": f"local::{plate_path}",
       "plate_id": str(plate_path),
       "axis_id": "",
   }
   progress_queue = (
       execution_bundle.runtime_environment.worker_start
       .multiprocessing_context()
       .Queue()
   )

   results = orchestrator.execute_compiled_plate(
       execution_bundle=execution_bundle,
       progress_queue=progress_queue,
       progress_context=progress_context,
   )

Applications should normally let the GUI or an execution service own progress
queue lifecycle, cancellation, and result presentation. The explicit call above
documents the current low-level boundary.

Primary public types
--------------------

The configuration classes are synthesized by ObjectState and should be
inspected through their nominal declarations, not frozen into generated field
pages. These are the stable import locations for the main integration surface:

``openhcs.core.steps.function_step.FunctionStep``
   A declarative processing step.

``openhcs.core.config.PipelineConfig``
   Pipeline-wide source, execution, and materialization configuration.

``openhcs.core.config.GlobalPipelineConfig``
   Process-wide defaults installed in the ObjectState context.

``openhcs.core.config.ProcessingConfig``
   Per-step axis and grouping semantics.

``openhcs.core.orchestrator.orchestrator.PipelineOrchestrator``
   The application-facing compile and execute coordinator.

``openhcs.core.compiled_execution.CompiledExecutionBundle``
   The typed product passed from compilation to execution.

``openhcs.interop.cellprofiler.pipeline_import.import_cellprofiler_pipeline``
   The supported CellProfiler ``.cppipe`` import boundary.

Architecture references
-----------------------

- :doc:`../concepts/core_model`
- :doc:`../concepts/pipelines_and_steps`
- :doc:`../architecture/quick_start`
- :doc:`../architecture/index`
