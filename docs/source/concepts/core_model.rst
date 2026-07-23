OpenHCS core model
==================

OpenHCS is a compiler and runtime for high-content microscopy workflows. It
turns source data and analysis declarations into validated execution plans,
then records typed results and materialized outputs.

The model in one diagram
------------------------

.. code-block:: text

   microscope files, metadata, or remote sources
                     |
                     v
             SourceBindingsConfig
                     |
   PipelineConfig + ordered FunctionStep declarations
                     |
                     | resolve configuration once
                     v
         snapshots + compilation sessions
                     |
                     | validate sources, artifacts,
                     | memory, scope, and workers
                     v
           CompiledExecutionBundle
                     |
                     v
     runtime values -> stores -> exports/viewers

Declarations
------------

A pipeline is an ordered ``list[FunctionStep]`` plus ``PipelineConfig``. A step
contains a function pattern and nested configuration for processing, sources,
materialization, data types, filters, and viewers. There is no separate
``Pipeline`` wrapper class.

The GUI, Python code transport, agent/UI bridge, and CellProfiler importer all
produce the same declarations.

Sources
-------

Microscope handlers interpret plate layouts and metadata. Source bindings give
semantic aliases to selected images and metadata. The compiler resolves those
bindings against an explicit source universe and virtual-workspace projection;
runtime adapters do not rescan directories or reinterpret filenames.

Axes and processing
-------------------

OpenHCS transports image inputs as 3D arrays, including single-plane cases.
Three independent declarations matter:

- ``variable_components`` says what changes along the stack axis;
- ``group_by`` groups assembled arrays and routes dictionary patterns;
- ``ProcessingContract`` says whether a callable has per-plane, whole-stack,
  flexible, or volumetric-to-slice semantics.

Configuration and ObjectState
-----------------------------

Global, pipeline, and step configuration can inherit across context and class
hierarchies. ObjectState owns generic lazy resolution, live editing, provenance,
and snapshots. The compiler materializes resolved steps once; later stages read
the resolved declarations rather than repeatedly resolving individual fields.

Contracts and artifacts
-----------------------

Callable contracts declare memory types, required axes, allowed grouping,
processing contract, execution scope, runtime adapter, and artifact inputs and
outputs. CellProfiler module leaves derive invocation-specific ``ArtifactSpec``
inputs and outputs into that same callable contract. Compiler source plans,
input edges, and invocation output plans then own satisfaction and runtime
selection; those facts are not separate module-contract partitions.

Artifacts are semantic values—not filenames or Python output-slot names. The
artifact graph determines producers and dependencies; typed plans add runtime
addresses and materialization targets.

Compilation
-----------

Each execution axis receives a ``CompilationSession`` with resolved steps,
``StepSnapshot`` objects, source projection, and typed ``CompiledStepPlan``
objects. Plans contain source, artifact, function, memory, materialization,
streaming, GPU, and worker facts. Compilation emits a
``CompiledExecutionBundle`` only after these facts satisfy runtime invariants.

Runtime values
--------------

Runtime results include images, masks, object labels, measurements,
relationships, sparse labels, grids, tables, and external files.
``RuntimeValueStore`` records them under typed artifact keys and exact execution
scope. Nominal projection strategies preserve plane, object, source, and row
identity when values are projected or grouped.

CellProfiler
------------

A ``.cppipe`` is parsed into module records. Setup modules contribute ordinary
source bindings and executable modules contribute ordinary ``FunctionStep``
declarations. ``CellProfilerModule`` subclasses own settings, callables,
artifacts, processing semantics, and measurements. No generated semantic
sidecar is required to execute or reload the resulting public declarations.

Extracted foundations
---------------------

OpenHCS owns domain wiring. ObjectState, ArrayBridge, PolyStore,
metaclass-registry, pyqt-reactive, python-introspect, ZMQRuntime, and pycodify own
their reusable mechanisms. OpenHCS documentation describes integration with
those packages rather than duplicating their internal architecture.

Next reading
------------

- :doc:`pipelines_and_steps`
- :doc:`data_dimensions`
- :doc:`../architecture/system_overview`
- :doc:`../architecture/external_foundations`
