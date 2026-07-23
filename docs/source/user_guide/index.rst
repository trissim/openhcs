User guide
==========

The supported interactive interface is the PyQt desktop application launched by
``openhcs``. The former Textual terminal interface is deprecated and is not
covered by this guide.

Start here
----------

- :doc:`../getting_started/getting_started` — installation, first compilation,
  and CellProfiler import
- :doc:`../concepts/core_model` — the current OpenHCS data and execution model
- :doc:`../concepts/pipelines_and_steps` — pipeline declarations and step
  composition
- :doc:`../concepts/data_dimensions` — dimensions, stacking, and grouping

Task guides
-----------

.. toctree::
   :maxdepth: 2

   custom_functions
   custom_function_management
   code_ui_editing
   dtype_conversion
   cpu_only_mode
   analysis_consolidation
   experimental_layouts
   real_time_visualization
   log_viewer
   llm_pipeline_generation
   mcp_clients

Declaration model
-----------------

The public pipeline declaration is ``list[FunctionStep]`` plus
``PipelineConfig``. Step processing, source, materialization, and streaming
options live in nested configuration declarations. Compilation produces typed
plans and a runtime bundle before work begins.

Where to get technical detail
-----------------------------

- :doc:`../api/index` documents the current declaration and low-level execution
  calls.
- :doc:`../architecture/index` documents compiler and runtime ownership.
- :doc:`../development/index` covers extension and contribution workflows.
