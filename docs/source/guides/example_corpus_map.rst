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

* ``benchmark/cellprofiler_pipelines/`` contains the in-tree ``.cppipe``
  fixtures used by the current benchmark adapter. It does not contain a
  parallel checked-in set of converted OpenHCS Python scripts.
* ``benchmark/native_refs/official30_scoped_rows/`` contains thirty scoped
  native CellProfiler reference runs. These include CellProfiler tutorials,
  example pipelines, sample image subsets, native outputs, and
  ``native_cellprofiler_headless/*.cppipe`` files.
* ``docs/pics/examples/`` contains captured CellProfiler example-page material
  and representative images for published example workflows.
* ``local/muecs_cp/`` may contain local CellProfiler pipeline sequences in
  developer checkouts. Treat local paths as optional evidence, not a portable
  public contract.

Treat existing CellProfiler pipelines as first-choice semantic evidence. They
describe the biological operation, module order, measurements, and expected
artifacts that OpenHCS preserves through compiler/runtime compatibility. A
missing high-level MCP shortcut should not be interpreted as missing
CellProfiler compatibility; use the current conversion, artifact-plan, runtime,
inventory, and viewer tools exposed by the checkout.

CellProfiler To OpenHCS Translation
-----------------------------------

Agents should assume CellProfiler knowledge is directly useful in OpenHCS. A
CellProfiler pipeline's named images, objects, measurements, module order, and
export modules describe the semantic workflow that OpenHCS compiles into its
own source bindings, ``FunctionStep`` declarations, artifact contracts, runtime
values, and materialized outputs.

Use this translation map while reading examples:

* CellProfiler modules become OpenHCS ``FunctionStep`` declarations backed by
  absorbed or native functions and runtime adapters.
* CellProfiler setup modules such as ``Images``, ``Metadata``, and
  ``NamesAndTypes`` become typed source bindings and virtual-workspace names.
* CellProfiler image/object/measurement names become OpenHCS semantic artifact
  contracts and runtime values.
* CellProfiler ``SaveImages``, ``ExportToSpreadsheet``, and
  ``ExportToDatabase`` become explicit executable steps. Plate-wide exporters
  consume merged typed runtime observations and emit declared file bundles.

For the implementation boundary, use MCP architecture topic
``cellprofiler_translation``. For current authoring, use
``openhcs_get_authoring_context`` and then verify with artifact-plan, runtime,
inventory, and viewer inspection tools.

Native OpenHCS Examples
-----------------------

The current checked-in native preset authority is
``openhcs/processing/presets/mfd_specs.py``. It owns the typed preset keys,
variant declarations, shared step templates, materializers, and
``build_mfd_preset``. These four small modules are current materialization
wrappers over that authority:

* ``openhcs/processing/presets/pipelines/10x_mfd_crop_analyze.py``
* ``openhcs/processing/presets/pipelines/10x_mfd_crop_analyze_dapi-fitc-cy5.py``
* ``openhcs/processing/presets/pipelines/10x_mfd_stitch_ashlar_cpu.py``
* ``openhcs/processing/presets/pipelines/10x_mfd_stitch_gpu.py``

The MCP knowledge renderer derives a ``Native Example Source Index`` from
exactly those declared Python paths, so agents can inspect the owning source and
thin wrappers without copying preset names or implementation into this page.

The larger, source-backed current corpus is
``openhcs_official30_benchmark_recipes``. Its thirty
``<case>-openhcs-python`` sections are converted lazily through the public
CellProfiler importer. Request source sections with ``max_chars=50000``; many
complete recipes are larger than the default bounded response.

Other scripts under the benchmark, preset-pipeline, and debug-export trees can
still be useful migration or backend evidence. They are not current API
examples unless a current test explicitly validates their imports and public
declaration shape. See :doc:`../getting_started/getting_started` and
:doc:`../concepts/function_patterns` for the supported declaration and callable
patterns.

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
4. Retrieve the closest Official30 ``OpenHCS Python`` section to inspect the
   current imported declarations; use the typed MFD authority when that preset
   family is the match.
5. Build the smallest useful subset pipeline and inspect the artifact plan
   before running the full plate.
6. Validate outputs through structured status and viewer tools before relying on
   screenshots or subjective visual inspection.

MCP Search Terms
----------------

These searches should lead an agent to examples before it concludes that the
workflow needs to be invented from scratch:

``CellProfiler examples``, ``CellProfiler cppipe``, ``ExampleHuman``,
``ExampleFly``, ``official30``, ``OpenHCS Python``,
``native CellProfiler reference``, ``native OpenHCS examples``,
``benchmark recipes``, ``MFD preset``,
``current examples``, ``complete examples``, ``nuclei segmentation``,
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
