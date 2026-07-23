Current examples
================

The source-backed Official30 corpus is the authority for complete imported
pipelines. See :doc:`example_corpus_map` for the exact retrieval workflow and
for the distinction between current declarations and migration/reference
scripts.

Thirty complete imported pipelines
----------------------------------

Through MCP, search knowledge for a task name plus ``OpenHCS Python``. Retrieve
the returned section from ``openhcs_official30_benchmark_recipes`` with
``max_chars=50000``. For example, section
``examplehuman-openhcs-python`` defines both ``pipeline_config`` and
``pipeline_steps`` using the public API. Sections are generated lazily from the
manifest-resolved ``.cppipe`` through the canonical importer; they are not a
second checked-in set of generated scripts.

The same source is available to Python clients through
``KnowledgeBaseService`` and to MCP clients through
``openhcs_search_knowledge`` followed by ``openhcs_get_knowledge_document``.

Minimal declaration
-------------------

A pipeline is an ordered list of ``FunctionStep`` declarations. Callable
metadata is declared on the callable; step processing options live in the
nested step configuration.

.. code-block:: python

   from openhcs.core.memory.decorators import numpy
   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.processing.backends.lib_registry.unified_registry import (
       ProcessingContract,
   )

   @numpy(contract=ProcessingContract.PURE_2D)
   def rescale(image, *, gain=1.0):
       return image * gain

   pipeline_steps = [
       FunctionStep(func=(rescale, {"gain": 1.25}), name="rescale"),
   ]

Compilation
-----------

Create a ``PipelineOrchestrator`` for an absolute plate directory, initialize
it, and pass the declarations to ``compile_pipelines``. Compilation resolves
ObjectState-backed configuration, source bindings, callable and artifact
contracts, materialization, memory conversion, and worker requirements. The
result contains the typed execution bundle consumed by the execution boundary.

For a complete low-level compile/execute call, including the required progress
context, see :doc:`../api/index`. Most users should compile and run through the
desktop application, which uses the same declarations and compiler.

Typed native presets
--------------------

The current native MFD variants are declared once by ``MfdPresetKey`` and
materialized by ``build_mfd_preset`` in
``openhcs.processing.presets.mfd_specs``. The four corresponding
``10x_mfd_*.py`` modules are thin wrappers over that owner. Consult
:doc:`example_corpus_map` for their source index and portability caveats; do not
infer current API shape from every older file in the preset directory.

Loose Opera Phenix neurite outgrowth
------------------------------------

``openhcs/processing/presets/pipelines/loose_operaphenix_neurite_outgrowth.py``
is a complete, parameterized CellProfiler-backed example for selected Opera
Phenix TIFFs copied without ``Index.xml``. Edit its ``example_inputs`` boundary
for the plate path, exact Hoechst/MAP2/SMI312 filenames, well/site/Z/time
identities, output root, and viewer port.

The example uses MAP2 objects as neuronal seeds, enhances and skeletonizes
SMI312 neurites, measures topology per seed, and propagates seed identities into
one final ``UnifiedNeurons`` label result. It deliberately streams both useful
diagnostic layers and that final body-plus-neurite association; a skeleton by
itself is not the analysis result. Its top-level one-well filter bounds memory,
viewer/checkpoint filters inherit that scope, path-planning filter zero avoids
an unwanted ordinary final image copy, and typed object/measurement artifacts
plus selected checkpoints remain materialized.

For the same source identities behind a smaller public surface, use
``openhcs/processing/presets/pipelines/loose_operaphenix_neurite_outgrowth_metaxpress.py``.
It composes the registered CellProfiler-compatible leaves behind one
MetaXpress-style step while retaining typed measurements and unified neuron
labels.

Use the native ``Microscope.OPERAPHENIX`` handler when the complete plate and
``Index.xml`` are available. Source bindings are appropriate here because the
loose files no longer carry the plate-level metadata needed by that handler.

CellProfiler import
-------------------

Use ``openhcs.interop.cellprofiler.pipeline_import.import_cellprofiler_pipeline``
to lower a ``.cppipe`` into ``(steps, pipeline_config)``. The importer does not
create a parallel CellProfiler runtime model. See
:doc:`../architecture/cellprofiler_interop`.

Avoid obsolete examples
-----------------------

OpenHCS has no public ``Pipeline`` wrapper or ``run_pipeline`` helper. Direct
``variable_components`` or ``group_by`` arguments on ``FunctionStep`` and
string-keyed compiled plans are also obsolete. Use
:doc:`../concepts/pipelines_and_steps` for the current declaration shape.
