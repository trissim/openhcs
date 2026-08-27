Domain Expert Onboarding
========================

Use this page when a scientist or imaging expert asks whether OpenHCS fits a
high-content microscopy workflow and what to try first.

When OpenHCS Fits
-----------------

OpenHCS is a good fit when the work is organized around microscopy plates,
wells, sites or fields of view, fluorescence channels, Z planes, or timepoints,
and the goal is a reproducible image-analysis pipeline rather than a one-off
manual viewer session.

It is especially useful for:

* high-content screening experiments with many wells or many images;
* plate-based fluorescence microscopy with multiple channels;
* workflows that need preprocessing, illumination correction, segmentation,
  cell counting, morphology, intensity measurement, or neurite-style analysis;
* workflows that should run the same way in the GUI and as Python code;
* datasets large enough to benefit from parallel execution, GPU backends, zarr
  storage, or real-time Napari/Fiji visualization.

OpenHCS is usually not the first tool for purely manual inspection, one
single-image notebook experiment, or a workflow where the plate layout and
image metadata cannot be described.

First Questions To Ask
----------------------

Before building a pipeline, collect these facts from the experiment:

* What is the plate root directory or OMERO source?
* Which microscope produced the data, such as ImageXpress, Opera Phenix,
  OMERO, Bio-Formats, or the native OpenHCS layout?
* How are wells, sites, fluorescence channels, Z planes, and timepoints encoded
  in filenames or metadata?
* Which channels correspond to biology, such as DAPI nuclei, GFP neurites,
  RFP marker intensity, or brightfield reference images?
* Which output is needed first: processed images, segmentation masks, ROI zip
  files, CSV/JSON measurements, or a quick visualization check?
* Can the first test use one or two wells from a representative subset instead
  of the full screen?

First Workflow
--------------

Start with a small dry-run plate subset.

1. Confirm the plate root and microscope/source type.
2. Confirm that OpenHCS can discover wells, sites, channels, Z planes, and
   timepoints from the data.
3. Build the smallest useful pipeline:

   * normalize or correct illumination;
   * optionally smooth or filter;
   * segment nuclei/cells or run the first biological measurement;
   * materialize outputs only where they are useful for inspection.

4. Compile or inspect the artifact plan before running the full screen.
5. Run one or two wells, inspect images and measurements, then scale up.

The pipeline concepts to read next are:

* :doc:`../guides/example_corpus_map` for existing CellProfiler pipelines,
  native OpenHCS examples, benchmark references, preset pipelines, and
  production example docs;
* :doc:`../concepts/data_dimensions` for plate, well, site, channel, Z, and
  timepoint grouping;
* :doc:`../concepts/pipelines_and_steps` for ``PipelineConfig``, ordered
  ``FunctionStep`` declarations, and compilation;
* :doc:`../concepts/function_patterns` for channel-specific routing;
* :doc:`../concepts/function_library` for segmentation, preprocessing, and
  measurement functions;
* :doc:`../user_guide/real_time_visualization` for Napari and Fiji streaming;
* :doc:`../user_guide/experimental_layouts` for well-to-condition mapping.

Agent And MCP Path
------------------

When an agent is helping through the MCP surface, use current tools before
assuming older planning documents are implemented:

1. ``openhcs_list_knowledge_documents`` to find source-backed docs.
2. ``openhcs_search_knowledge`` for domain terms such as ``microscopy``,
   ``plate layout``, ``well site channel``, ``segmentation``, ``fluorescence``,
   ``getting started``, ``CellProfiler``, ``Napari``, or ``zarr``.
3. ``openhcs_get_authoring_context`` for the current pipeline-authoring
   contract.
4. ``openhcs_search_functions`` for task terms such as ``normalize``,
   ``gaussian``, ``threshold``, ``watershed``, ``count cells``, or
   ``colocalization``.
5. ``openhcs_create_pipeline`` and ``openhcs_add_function_step`` for a draft
   only after the data layout and first biological goal are clear.
6. ``openhcs_inspect_pipeline_source_artifact_plan`` before a full run.

CellProfiler Mental Model
-------------------------

If the scientist or agent already knows CellProfiler, start there. OpenHCS uses
the same practical analysis ideas: ordered processing modules, named images,
named objects, measurements, and explicit result exports. CellProfiler
compatibility is integrated into the OpenHCS compiler/runtime model, so those
concepts are not just informal analogies. They compile into OpenHCS source
bindings, ``FunctionStep`` declarations, artifact contracts, runtime values,
materialization, and measurements.

The translation is:

* Executable CellProfiler modules lower to ordinary OpenHCS ``FunctionStep``
  declarations whose public callables remain visible to the GUI and code
  transport.
* CellProfiler ``Images``, ``Metadata``, and ``NamesAndTypes`` setup maps to
  OpenHCS source bindings, metadata rules, and virtual workspace projections.
* CellProfiler image names map to semantic source bindings or runtime image
  inputs.
* CellProfiler object names map to object-label runtime values or artifact
  contracts.
* CellProfiler measurement modules declare measurement artifacts and typed
  runtime values through their callable/module contracts.
* CellProfiler ``SaveImages`` and export modules become explicit steps; plate-wide
  exporters declare plate execution scope and materialized outputs directly.

Use CellProfiler examples to understand biological intent, module order, named
images/objects, measurements, and expected artifacts. Then compile and validate
the OpenHCS projection with artifact-plan, runtime, inventory, and viewer tools.
Over MCP, ``openhcs_explain_architecture`` with topic
``cellprofiler_translation`` provides the detailed internal map.

CellProfiler And Existing Pipelines
-----------------------------------

If the scientist already has a CellProfiler pipeline, treat it as valuable
semantic evidence. OpenHCS preserves CellProfiler-style image, object,
measurement, and export semantics by projecting them into OpenHCS compiler and
runtime authorities.

If the pipeline contains a module that requires interactive desktop input,
replace or remove that module before import. OpenHCS reports the unsupported
interaction instead of silently omitting it from the imported pipeline.

Do not start from a blank slate. The source-backed Official30 knowledge document
maps thirty CellProfiler recipes to lazily generated public OpenHCS declarations.
Search by biological task plus ``OpenHCS Python`` and retrieve the exact source
section with a 50,000-character bound. The typed MFD preset authority provides
the smaller current native preset family. Use
:doc:`../guides/example_corpus_map` to choose the closest verified workflow and
to distinguish current examples from older migration/reference scripts.

Practical Search Terms
----------------------

These terms should lead an agent to the right knowledge-base area:

``microscopy``, ``high-content``, ``screen``, ``plate layout``, ``plate root``,
``well``, ``site``, ``field``, ``channel``, ``fluorescence``, ``Z plane``,
``timepoint``, ``segmentation``, ``cell counting``, ``measurement``,
``getting started``, ``FunctionStep``, ``variable components``, ``group_by``,
``Napari``, ``Fiji``, ``zarr``, ``CellProfiler``, ``CellProfiler examples``,
``cppipe``, ``official30``, ``OpenHCS Python``, ``native OpenHCS examples``,
``MFD preset``.
