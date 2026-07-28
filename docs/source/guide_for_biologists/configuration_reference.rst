Configuration reference
=======================

OpenHCS configuration is hierarchical. Global defaults can be overridden for a
plate/pipeline, and step-level nested configurations can override the relevant
parts again. The desktop forms show the same ObjectState-backed declarations
used by generated Python code.

The mental model
----------------

There are three declaration scopes, followed by one resolution boundary:

.. code-block:: text

   GlobalPipelineConfig        concrete application/session defaults
              |
              v
   PipelineConfig              lazy per-pipeline overrides
              |
              v
   FunctionStep nested config  lazy per-step overrides
              |
              v
   ObjectState resolution at compile time
              |
              v
   immutable step snapshots and typed compiled plans

``None`` in a lazy field normally means *inherit*, not *disable*. Enableable
configs have a separate ``enabled`` field; resetting a lazy field returns it to
inheritance. Compilation resolves the hierarchy once. Runtime workers consume
the resolved snapshots and plans and never reinterpret the live forms.

Scope precedence is only one axis. Config dataclass inheritance is the other:
it determines which related policies share a default. The important well-filter
relationship is:

.. code-block:: text

   WellFilterConfig                 pipeline execution domain
      |-- PathPlanningConfig        automatic output-plate persistence
      `-- StepWellFilterConfig      step-policy default
             |-- StepMaterializationConfig
             `-- StreamingDefaults
                    `-- registered Napari/Fiji streaming config

The broad pipeline ``well_filter_config`` is applied before per-well contexts
compile, so it reduces image loading, memory use, and processing. Descendant
step policies inherit that selection unless explicitly narrowed. Sibling
overrides remain isolated: changing ``path_planning_config.well_filter`` changes
automatic main-flow persistence, not viewer eligibility.

The root configuration dataclasses and their generated lazy counterparts are
the authority for names, types, defaults, enum values, and help text. Through
MCP, call ``openhcs_describe_config_schema`` for ``global``, ``pipeline``, or
``step``; the no-prefix response is the current top-level family map. The step
root is reflected from the config-bearing keyword-only parameters declared by
``AbstractStep`` rather than a copied list. Call the tool again with a returned
``path_prefix`` to retrieve that family's generated dotted nested paths, field
help, enum and registry values, inheritance markers, declaring type, default
origin, and nested-schema route without loading the entire config tree. In the
UI, use ObjectState field help to see raw, resolved, inherited, dirty, and
default state before editing.

Common groups
-------------

``GlobalPipelineConfig``
  Concrete application and execution-environment defaults: worker/process
  policy, results location, output auto-add behavior, and the concrete default
  instance of every nested config family.

``PipelineConfig``
  The inheritable pipeline form generated from the same config declarations.
  Its nested ``Lazy*Config`` values carry only pipeline overrides; omitted
  values resolve from the surrounding global/ObjectState scope.

Processing and callable data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ProcessingConfig``
  Step processing semantics: ``variable_components``, ``group_by``, and
  ``input_source``. Respectively, they own the assembled array axis,
  post-assembly partitioning, and the previous-step versus pipeline-start
  main-flow choice. A separately named source is declared through the callable's
  artifact inputs and step source bindings, not another ``input_source`` value.
  These controls are independent of source discovery, dtype conversion,
  persistence, and viewer display.

``SourceBindingsConfig`` and ``StepSourceBindingsConfig``
  Pipeline source declarations and named semantic inputs for a step.

``DtypeConfig``
  Native-output versus preserve-input dtype behavior for decorated callables.

Sources and execution shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SequentialProcessingConfig``
  Chooses plate components that are processed through the whole pipeline one
  combination at a time to bound memory. It does not replace
  ``variable_components`` or ``group_by``.

``WellFilterConfig`` and ``StepWellFilterConfig``
  Reusable include/exclude selection for pipeline operations and individual
  steps. Step filtering changes which wells execute; it is not an image-source
  selector.

Storage, paths, and persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``VFSConfig`` and ``ZarrConfig``
  Read, intermediate, and persistent storage choices plus Zarr compression and
  chunking.

``PathPlanningConfig``
  Output workspace/root placement, directory naming, and which processed wells
  seed the automatic main-flow output plate. ``well_filter=0`` keeps that
  automatic output runtime-only. It does not disable explicitly requested step
  checkpoints or named-artifact materialization. Compiled plan inspection
  reports ``main_flow_axis_persistence_enabled`` for each step so this effective
  scope is visible before execution rather than inferred from the output plate.

``StepMaterializationConfig``
  Explicit per-step persistent copy of that step's ordinary main-flow result.
  Intermediate runtime flow is separate and does not imply a saved result. Its
  path and well-filter behavior compose through the owning config inheritance
  hierarchy. It does not mean "persist every named artifact".

Named artifact materialization
  Callable artifact outputs enter typed runtime dataflow independently of
  persistence. Explicit or compiler-added artifact materialization plus the
  compiled runtime-artifact materialization plan decides whether named images,
  labels, measurements, tables, or files persist. Inspect the compiled artifact
  plan rather than inferring persistence from a return value or filename.

``CompilationDebugConfig``
  Optional compiler-diagnostic bundle output. It is for debugging compilation,
  not normal result materialization.

Viewers
~~~~~~~

``StreamingDefaults``
  Shared viewer enablement, well filtering, batching, persistence, host, and
  transport behavior.

``NapariStreamingConfig`` and ``FijiStreamingConfig``
  Per-step streaming plus viewer-specific display configuration and ports.
  ``NapariDisplayConfig`` and ``FijiDisplayConfig`` provide inherited display
  defaults. Supported viewer config families come from the
  ``StreamingConfig`` registry; generic code does not maintain a viewer-name
  table. To display one exact step, leave viewer enablement false at broader
  scopes and override ``enabled=True`` only on that step's selected viewer
  config, optionally with a bounded ``well_filter`` and ``persistent=True``;
  then compile again.

Post-run analysis and metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnalysisConsolidationConfig``
  Optional post-plate consolidation, including well pattern, included
  extensions, exclusions, and output filenames.

``PlateMetadataConfig``
  Plate identity and acquisition metadata used by compatible consolidated
  outputs.

``ExperimentalAnalysisConfig``
  Experimental-design workbook, normalization, raw-result, and heatmap policy.
  It consumes consolidated measurements and plate layouts; it does not change
  pipeline axes or source bindings.

How scopes interact
-------------------

Pipeline-level nested configs provide defaults for every step. A
``FunctionStep`` can override the step-relevant families: processing, dtype,
named sources, step well filtering, materialization, shared streaming defaults,
and Napari/Fiji streaming. Root-only execution and post-run settings stay on
``GlobalPipelineConfig`` or ``PipelineConfig``.

Use reviewed Python when several related fields must change together. A config
document contains one typed config object; a complete pipeline document contains
``pipeline_config`` plus ``pipeline_steps``. In a running desktop session, apply
the relevant UI-owned code document so the visible ObjectState, snapshots, and
revision tokens remain authoritative.

How inheritance behaves
-----------------------

An inherited field is resolved once when the pipeline compiles. A later global
or UI edit does not mutate already compiled contexts; compile again to apply the
new declaration. The form preview and provenance controls show where a value is
coming from when a nested setting inherits.

For exact fields and defaults, use the tooltips generated from the current
dataclasses, ObjectState field help, or ``openhcs_describe_config_schema`` for
the installed version. Do not copy an old field list into a prompt. See
:doc:`../concepts/pipelines_and_steps`, :doc:`image_sources`,
:doc:`../user_guide/dtype_conversion`, and
:doc:`../architecture/code_ui_interconversion`.
