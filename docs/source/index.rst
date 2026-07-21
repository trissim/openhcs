OpenHCS documentation
=====================

OpenHCS is a bioimage-analysis platform for high-content screening. It turns
pipeline declarations into validated, per-plate execution plans and runs them
across microscopy datasets using CPU or GPU processing libraries.

Choose a path
-------------

**Using OpenHCS**
  Start with :doc:`getting_started/getting_started`, then use the
  :doc:`user_guide/index` for task-oriented workflows.

**Understanding the model**
  Start with the :doc:`architecture/quick_start`, then use
  :doc:`concepts/index` for pipelines, steps, dimensions, sources, and
  processing semantics.

**Extending or maintaining OpenHCS**
  Use :doc:`development/index` for contribution workflows and
  :doc:`architecture/index` for system boundaries and invariants.

Quick start
-----------

OpenHCS requires Python 3.11 or newer. Install and launch the desktop GUI with:

.. code-block:: bash

   python -m pip install "openhcs[gui]"
   openhcs

Viewer integrations are optional:

.. code-block:: bash

   python -m pip install "openhcs[gui,napari]"  # Napari
   python -m pip install "openhcs[gui,fiji]"    # Fiji/ImageJ
   python -m pip install "openhcs[gui,viz]"     # Both

The Textual terminal interface is deprecated and is not part of the published
package.

System at a glance
------------------

OpenHCS uses ordinary Python declarations at its public boundary:

.. code-block:: text

   PipelineConfig + list[FunctionStep]
       -> ObjectState resolution
       -> StepSnapshot + CompilationSession
       -> typed CompiledStepPlan objects
       -> CompiledExecutionBundle
       -> runtime values, artifacts, and materialized outputs

CellProfiler ``.cppipe`` files lower directly into the same
``PipelineConfig`` and ``FunctionStep`` declarations. The compiler resolves
configuration once and derives source, artifact, memory, and execution plans
from the authoritative declarations.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/getting_started
   guide_for_biologists/index

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/index

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Integration guides

   guides/index

.. toctree::
   :maxdepth: 2
   :caption: API and architecture

   api/index
   architecture/quick_start
   architecture/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index
   reference/index
   appendices/index

.. toctree::
   :hidden:

   installation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
