What is OpenHCS?
================

OpenHCS is a desktop and Python platform for building reproducible image-analysis
workflows over high-content microscopy plates. It combines microscope-aware
source discovery, visual pipeline editing, pre-execution validation, parallel
processing, typed analysis results, and optional viewers.

Good fits
---------

OpenHCS is useful when you need to:

- process many wells, sites, channels, Z planes, or time points consistently;
- stitch fields of view or run segmentation and measurement workflows;
- import a supported CellProfiler ``.cppipe`` and continue editing it as normal
  OpenHCS steps;
- combine compatible NumPy or optional GPU-backed processing functions;
- save selected outputs and analysis artifacts with provenance;
- inspect intermediate images in Napari or Fiji;
- share the same pipeline as editable UI state and Python declarations.

The available microscope formats and function libraries depend on the installed
version and optional dependencies. The application discovers them from their
registered declarations; use the Plate Manager and function browser as the
current capability catalog.

How a run works
---------------

You add and initialize a plate, build or import an ordered list of steps, and
compile it. Compilation combines the pipeline with that plate's metadata to
check sources, image dimensions, function requirements, artifacts, memory, and
outputs. Only a successfully compiled selection is executed.

Start with :doc:`installation_and_setup`, :doc:`basic_interface`, and
:doc:`intro_stitching`.
