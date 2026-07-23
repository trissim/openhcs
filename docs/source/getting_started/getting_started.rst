Getting started
===============

This page covers the supported installation and the two public ways to define a
pipeline: editing it in the desktop GUI or importing a CellProfiler ``.cppipe``
as ordinary OpenHCS declarations.

For the common desktop, Python, CellProfiler, and MCP mental model, start with
:doc:`../architecture/quick_start`.

Requirements
------------

- Python 3.11 or newer
- A local directory containing a supported microscopy plate, or an explicitly
  configured remote source
- A CUDA 12 compatible environment only when installing the optional ``gpu``
  dependencies

Install OpenHCS
---------------

For the desktop application:

.. code-block:: bash

   python -m pip install "openhcs[gui]"
   openhcs

The ``openhcs-gui`` command is an alias for the same application. The old
``openhcs-tui`` command is deprecated and is not included in the published
package.

Optional capabilities are installed independently:

.. code-block:: bash

   python -m pip install "openhcs[gui,napari]"
   python -m pip install "openhcs[gui,fiji]"
   python -m pip install "openhcs[gui,viz]"       # both viewers
   python -m pip install "openhcs[gui,gpu]"       # CUDA libraries
   python -m pip install "openhcs[gui,viz,gpu]"   # full desktop stack

The base package can be installed for headless imports and services with
``python -m pip install openhcs``. Do not install the ``gpu`` extra on a
CPU-only system. See :doc:`../user_guide/cpu_only_mode` for runtime controls.

Build a pipeline in the GUI
---------------------------

1. Launch ``openhcs``.
2. Add a plate directory in the Plate Manager.
3. Open the Pipeline Editor for that plate.
4. Add or import steps and edit their parameters.
5. Compile the selected plates. Compilation resolves configuration and checks
   sources, artifacts, memory contracts, and execution requirements before any
   plate is executed.
6. Run the compiled selection and inspect progress and materialized outputs.

The GUI stores a pipeline as a ``list[FunctionStep]`` plus a ``PipelineConfig``.
Code export and re-import use those same public declarations; there is no
separate GUI-only pipeline model.

To import an existing CellProfiler pipeline in the desktop application, select
the target plate, choose **File > Open Pipeline**, and select the ``.cppipe``.
The importer applies setup-module source bindings to that plate's
``PipelineConfig`` and replaces the editor contents with ordinary
``FunctionStep`` declarations. Review the source bindings and compiled artifact
plan before running; the imported pipeline does not retain a hidden
CellProfiler runtime or a second GUI-only representation.

Import a CellProfiler pipeline
------------------------------

``import_cellprofiler_pipeline`` parses a ``.cppipe`` and returns only public
OpenHCS declarations. CellProfiler setup modules contribute source bindings;
executable modules become ordinary ``FunctionStep`` instances.

.. code-block:: python

   from pathlib import Path

   from objectstate import ensure_global_config_context
   from openhcs.core.config import GlobalPipelineConfig
   from openhcs.interop.cellprofiler.pipeline_import import (
       import_cellprofiler_pipeline,
   )

   ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
   steps, pipeline_config = import_cellprofiler_pipeline(
       Path("analysis.cppipe"),
       source_root=Path("/data/plate"),
   )

   print(type(pipeline_config).__name__)  # PipelineConfig
   print([step.name for step in steps])

The declarations can be loaded into the GUI or passed to a
``PipelineOrchestrator``. Compilation is explicit:

.. code-block:: python

   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

   orchestrator = PipelineOrchestrator(
       Path("/data/plate").resolve(),
       pipeline_config=pipeline_config,
   ).initialize()

   compilation = orchestrator.compile_pipelines(steps)
   execution_bundle = compilation["execution_bundle"]

Execution requires a progress context and queue because the same boundary is
used by the GUI, local multiprocessing, and remote execution services. See
:doc:`../api/index` for the current low-level call and
:doc:`../concepts/core_model` for the lifecycle.

Important processing terms
--------------------------

``variable_components``
  Declares what the third array axis varies over, such as site, channel, or Z.

``group_by``
  Groups already assembled arrays. It creates callable fan-out only for a
  dictionary function pattern; it does not define the stack axis.

``ProcessingContract``
  Declares semantic locality. ``PURE_2D`` means per-plane semantics even when
  planes are transported as a 3D batch; ``PURE_3D`` means the result depends on
  the stack as a whole.

Next steps
----------

- :doc:`../architecture/quick_start`
- :doc:`../concepts/pipelines_and_steps`
- :doc:`../concepts/data_dimensions`
- :doc:`../user_guide/index`
- :doc:`../architecture/index`
