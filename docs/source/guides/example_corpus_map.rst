OpenHCS Example Corpus Map
==========================

Use this page when an agent needs practical starting material for a scientist
or imaging expert. OpenHCS already ships CellProfiler examples, native OpenHCS
examples, benchmark fixtures, and production-oriented docs. The agent-facing
task is to find and apply them, not to assume the project lacks examples.

CellProfiler Pipeline Corpus
----------------------------

The repository includes several CellProfiler-oriented sources that can anchor a
conversation with a domain expert:

* ``benchmark/cellprofiler_pipelines/`` contains in-tree ``.cppipe`` examples
  and converted OpenHCS Python examples, including ``ExampleHuman``,
  ``ExampleFly``, and ``BBBC021`` analysis/illumination pipelines.
* ``benchmark/native_refs/official30_scoped_rows/`` contains thirty scoped
  native CellProfiler reference runs. These include CellProfiler tutorials,
  example pipelines, sample image subsets, native outputs, and
  ``native_cellprofiler_headless/*.cppipe`` files.
* ``docs/pics/examples/`` contains captured CellProfiler example-page material
  and representative images for published example workflows.
* ``local/muecs_cp/`` may contain local CellProfiler pipeline sequences in
  developer checkouts. Treat local paths as optional evidence, not a portable
  public contract.

Treat existing CellProfiler pipelines as semantic evidence. They describe the
biological operation, module order, measurements, and expected artifacts. Before
claiming a direct MCP import path, verify the current tool surface and source
code, then convert or re-author through current OpenHCS pipeline-authoring
contracts.

CellProfiler To OpenHCS Translation
-----------------------------------

Agents should assume CellProfiler knowledge is useful in OpenHCS. A
CellProfiler pipeline's named images, objects, measurements, module order, and
export modules describe the semantic workflow that OpenHCS needs to preserve.

Use this translation map while reading examples:

* CellProfiler modules become OpenHCS ``FunctionStep`` declarations backed by
  absorbed or native functions.
* CellProfiler setup modules such as ``Images``, ``Metadata``, and
  ``NamesAndTypes`` become OpenHCS source schemas, source bindings, and virtual
  workspace names.
* CellProfiler image/object/measurement names become OpenHCS semantic artifact
  contracts and runtime values.
* CellProfiler ``SaveImages`` and table-export modules become materialization
  requirements when the result is externally required.

For the implementation boundary, use MCP architecture topic
``cellprofiler_translation``. For current authoring, use
``openhcs_get_authoring_context`` and then verify with artifact-plan and runtime
inspection tools.

Native OpenHCS Examples
-----------------------

Native examples show current OpenHCS concepts and code shapes:

* ``benchmark/pipelines/`` contains small benchmark pipelines such as BBBC021,
  BBBC022, CellProfiler-style nuclei segmentation, GPU variants, and
  preprocessing examples.
* ``openhcs/processing/presets/pipelines/`` contains production-style preset
  pipelines for ImageXpress 96-well neurite outgrowth, MFD crop/analyze, GPU
  stitching, and cell-count workflows.
* ``openhcs/debug/example_export.py`` and
  ``openhcs/debug/example_export_clean.py`` are complete exported scripts useful
  for checking imports and generated-code shape.
* ``docs/source/guides/complete_examples.rst`` and
  ``docs/source/user_guide/production_examples.rst`` document complete workflow
  patterns, including configuration, dictionary routing, GPU processing, zarr,
  and execution.

Operator Workflow
-----------------

When a scientist says "I have a folder of images, help me set this up", use the
examples as a retrieval and validation ladder:

1. Identify the source family: microscope layout, OMERO, Bio-Formats, native
   OpenHCS layout, or existing CellProfiler project.
2. Search the example corpus for the closest biological task: nuclei
   segmentation, illumination correction, colocalization, translocation,
   quality control, neurite outgrowth, worm analysis, wound healing, or
   CellProfiler-style measurement export.
3. Use CellProfiler examples to understand task semantics and expected outputs.
4. Use native OpenHCS examples to choose current imports, ``FunctionStep``
   structure, configuration, materialization, and viewer settings.
5. Build the smallest useful subset pipeline and inspect the artifact plan
   before running the full plate.
6. Validate outputs through structured status and viewer tools before relying on
   screenshots or subjective visual inspection.

MCP Search Terms
----------------

These searches should lead an agent to examples before it concludes that the
workflow needs to be invented from scratch:

``CellProfiler examples``, ``CellProfiler cppipe``, ``ExampleHuman``,
``ExampleFly``, ``BBBC021``, ``official30``, ``native CellProfiler reference``,
``native OpenHCS examples``, ``benchmark pipelines``, ``preset pipelines``,
``production examples``, ``complete examples``, ``nuclei segmentation``,
``illumination correction``, ``colocalization``, ``translocation``,
``quality control``, ``neurite outgrowth``.

Live Data Inspection
--------------------

Examples are semantic starting points, not substitutes for inspecting the
scientist's actual plate. Before authoring or running a pipeline, use
``inspect-plate`` and ``query-plate-files`` for explicit plate roots, or
``selected-plate-images`` and ``selected-plate-files`` when the Plate Manager is
open and a single row is selected. These tools report discovered wells, sites,
channels, Z planes, timepoints, files, virtual paths, source paths, handler
confidence, and warnings.

For UI-owned work, selected-plate review tools can target the selected, source,
or output plate row. Use those live inventories to validate whether an example
pipeline matches the user's data layout before relying on artifact plans,
viewer streaming, or measurement summaries.
