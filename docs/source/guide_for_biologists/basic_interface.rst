Desktop interface
=================

The OpenHCS main window organizes plate, pipeline, configuration, diagnostics,
and analysis workflows. A ``?`` control beside a field opens help derived from
the current declaration when available.

Interface map
-------------

The desktop is a set of ObjectState-backed workflow windows, not one giant form:

.. code-block:: text

   Main window
       |- Plate Manager          datasets, selection, init/compile/run
       |- Pipeline Editor        PipelineConfig + ordered FunctionStep list
       |    `- Step editors      callable pattern + step-local config
       |- Global Configuration   application/session defaults
       |- Image/metadata tools   source and output inspection
       |- Log/progress/debug     operation status and diagnostics
       `- Analysis/tools         consolidation, layouts, custom functions

The Plate Manager is the operational hub. Other windows edit or inspect the
selected plate, pipeline, step, source projection, or application config. A
Napari or Fiji viewer is a separate runtime window launched only when requested.
In the supported PyQt desktop shell, Plate Manager and the ZMQ process manager
occupy the left workspace, Pipeline Editor occupies the right workspace, and
the system monitor sits below them in the outer splitter. Managed editors,
configuration windows, logs, and tools open or focus around that embedded
workspace. The PyQt desktop shell is the supported MCP-attached UI.

Plate Manager
-------------

Plate Manager is the starting point for a dataset. Add a local or configured
remote plate, initialize it, and select the plate before editing or running its
pipeline. Initialization discovers the microscope/source format and component
metadata. Compile and run actions are disabled until their prerequisites are
met.

Plate Manager code mode records whether its document represents all plates or
only the current selection. Applying a selected document changes only those
plates and preserves every unselected plate. Read a new document if you need to
change which plates the edit covers.

Pipeline Editor
---------------

The editor shows the ordered steps for the selected plate. Add, remove, reorder,
or edit a step; open the function-pattern editor to choose callables and
parameters. The code projection contains the same ``PipelineConfig`` and
``FunctionStep`` declarations and can be used to inspect or share the workflow.

The Pipeline Editor code document is a complete ``PipelineDocument``.
``pipeline_steps`` is required. ``pipeline_config`` may be omitted when the
default ``PipelineConfig()`` is sufficient. An individual step or config editor
has its own smaller nominal code document. Code mode is not a parallel script
format; applying it updates the same live ObjectState-backed object shown by the
forms. Applying a complete pipeline document synchronises that declaration.
Unchanged and unambiguously edited steps retain their live history while added
and omitted steps update the collection. Applying the document also advances
the saved baseline for the reconciled pipeline, steps, and nested function
parameters; the Pipeline Editor does not require a separate Save action.

The function browser reads the callable catalogue from the execution endpoint.
The desktop prepares that endpoint and begins loading its catalogue during
startup. If loading is still in progress when the browser opens, the browser
remains responsive and displays the request state until the shared catalogue is
ready.

Image and metadata browsing
---------------------------

The Image Browser filters source and output references by discovered plate
components. Viewer actions require the corresponding Napari or Fiji optional
dependency. Metadata views show the source projection used by compilation; fix
incorrect source metadata before running a pipeline.

Configuration and diagnostics
-----------------------------

Global Configuration sets application defaults. Plate/pipeline and step forms
can override inheritable fields. The Log Viewer, progress surfaces, debug
inspector, and optional system monitor help distinguish initialization,
compilation, and execution failures.

Tools
-----

The Tools menu includes custom-function management and analysis consolidation.
Viewer and LLM features appear only when their dependencies/services are
configured.

Agent navigation
----------------

An attached MCP agent should navigate semantically:

1. list running windows and ObjectState scopes;
2. read the Plate Manager state surface to identify selected/source/output rows;
3. open or focus the relevant window/scope;
4. list its code documents or fields rather than guessing widget labels;
5. validate code, then apply with the current revision token;
6. dispatch semantic init/compile/run actions and poll operation/state surfaces.

Window snapshots and the widget tree are useful for orientation or diagnosing a
missing semantic surface. They are not configuration authorities. The live
ObjectState field help, code documents, state surfaces, and declared UI actions
own meaning.

Do not confuse the **desktop window layout** with an **experimental plate
layout**. The latter maps wells to treatments, controls, concentrations, and
replicates after measurements have been produced.

Continue with :doc:`intro_stitching` and
:doc:`configuration_reference` for the workflow and configuration mental model;
use :doc:`../reference/configuration` when you need an exact field or default.
