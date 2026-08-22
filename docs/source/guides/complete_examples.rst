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
:doc:`example_corpus_map` for their source index. Crop presets resolve
``templates/mfd_96_sobel_10x_whole_device.tif`` relative to each plate; place
the matching template there or edit that authored relative path before use.
Do not infer current API shape from every older file in the preset directory.

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

Master multi-plate lab-meeting showcase
---------------------------------------

``scripts/master_multi_plate_demo.py`` composes every declaration returned by
``scripts.mcp_assay_showcase.scenario_blueprints`` into one Plate Manager
document. It generates the bounded plates, registers every row in the running
desktop UI, then selects and initializes, compiles, and runs each plate in
sequence. Each plate has a deterministic dedicated Napari endpoint beginning at
port 5900. Its Napari window is framed with the exact scope accent projected by
the running Plate Manager, so the viewer can be matched to its plate and config
windows without a second color map.

First inspect the complete inventory and generated document without touching a
running UI:

.. code-block:: bash

   .venv/bin/python scripts/master_multi_plate_demo.py --dry-run

For the live showcase, start the OpenHCS desktop UI and pass its bridge
descriptor explicitly:

.. code-block:: bash

   .venv/bin/python scripts/master_multi_plate_demo.py \
       --descriptor-file-path /path/to/running-ui-bridge.json

The complete built-in inventory contains seven assay stories. The runner checks
each data and control endpoint before launch and reports a collision; it does
not silently move a plate to another port. A compile or runtime failure is
recorded for that plate and the next plate still runs. The summary and every MCP
command response are written under ``mcp_outputs/master_multi_plate_demo``.

Additional demos join only through an explicit contributor factory. For the
NeuronCyto II crossover example, point to the separately downloaded official
archive and name its preset-owned contributor:

.. code-block:: bash

   export OPENHCS_NEURONCYTO_II_TEST_ARCHIVE=/path/to/Testing\ image.zip
   .venv/bin/python scripts/master_multi_plate_demo.py \
       --descriptor-file-path /path/to/running-ui-bridge.json \
       --contributor openhcs.processing.presets.pipelines.neuroncyto_ii_crossover_neurite_outgrowth:neuroncyto_ii_crossover_demo_contribution

For the varied eight-plate lab-meeting sequence from a source checkout,
explicitly retain five curated built-ins, add the two repository-only
Official30 stories, and add NeuronCyto II:

.. code-block:: bash

   export OPENHCS_NEURONCYTO_II_TEST_ARCHIVE=/path/to/Testing\ image.zip
   .venv/bin/python scripts/master_multi_plate_demo.py \
       --descriptor-file-path /path/to/running-ui-bridge.json \
       --exclude-demo primary_object_segmentation \
       --exclude-demo nuclear_morphology \
       --contributor benchmark.demos.official30_lab_meeting:official30_lab_meeting_demo_contributions \
       --contributor openhcs.processing.presets.pipelines.neuroncyto_ii_crossover_neurite_outgrowth:neuroncyto_ii_crossover_demo_contribution

That source-checkout-only composition uses ports 5900 through 5907: five
curated built-in assays, Comet and wound-closure from Official30, and
NeuronCyto II. The benchmark package and Official30 manifest are not installed
with the OpenHCS wheel. The exclusions and contributors are command-line
choices, not a hidden alternate inventory.

Contributor factories receive ``session_root=Path`` and return a declared plate
path, pipeline config, steps, title, stable demo id, and optional preparation
callable.
The master still owns port assignment, UI registration, viewer launch, and
sequential execution; contributors do not duplicate that machinery.

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
