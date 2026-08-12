Architecture quick start
========================

This is the front door to the OpenHCS architecture for users, developers, and
agents. OpenHCS has one public pipeline model and several ways to operate it:
the desktop application, Python, CellProfiler import, and MCP all converge on
the same declarations, compiler, and runtime.

Choose the shortest route
-------------------------

**I have an image folder and want to use the desktop application**
  Start with :doc:`../getting_started/getting_started`. Add the plate, create
  or import a pipeline, compile it, and run a bounded selection before scaling
  up.

**I have a CellProfiler ``.cppipe``**
  Use the public importer shown in `CellProfiler pipelines`_ below. For working
  recipes, search the Official30 corpus before creating a translation from
  scratch.

**I am writing Python**
  Start with `The public declaration boundary`_ and :doc:`../api/index`.
  Configuration and processing semantics belong in their nominal nested
  declarations, not in duplicated keyword lists or application-specific
  wrappers.

**I am an MCP client or coding agent**
  Follow `First MCP session`_. The MCP quick start is
  ``openhcs_get_authoring_context(kind="first_use")``; this page is its
  source-backed architecture guide.

The public declaration boundary
-------------------------------

Every supported authoring route produces the same public declaration shape:

.. code-block:: text

   PipelineConfig + ordered list[FunctionStep]
       -> ObjectState resolves lazy configuration
       -> StepSnapshot + CompilationSession
       -> typed CompiledStepPlan objects
       -> CompiledExecutionBundle
       -> runtime values, artifacts, and materialized outputs

``FunctionStep`` owns the callable pattern and step-local nested configuration.
``PipelineConfig`` owns pipeline-wide configuration. Callables, modules,
artifact types, measurement feature types, sources, and strategies own their
respective semantics. Generic code queries those nominal authorities and their
registries; it does not maintain parallel name tables or backend-specific
fallback chains.

A minimal declaration uses an ordinary registered callable:

.. code-block:: python

   from openhcs.core.memory.decorators import numpy
   from openhcs.constants.input_source import InputSource
   from openhcs.core.config import (
       LazyProcessingConfig,
       LazyStepSourceBindingsConfig,
       PipelineConfig,
   )
   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.processing.backends.lib_registry.unified_registry import (
       ProcessingContract,
   )

   @numpy(contract=ProcessingContract.PURE_2D)
   def rescale(image, *, gain: float = 1.0):
       return image * gain

   pipeline_config = PipelineConfig()
   pipeline_steps = [
       FunctionStep(func=(rescale, {"gain": 1.25}), name="rescale"),
   ]

These assignments are one nominal ``PipelineDocument``. ``pipeline_steps`` is
required; an omitted ``pipeline_config`` resolves to ``PipelineConfig()`` for
older and default-only source documents. Configuration and steps are parsed,
validated, rendered, and transported atomically; a source route cannot replace
the pipeline config through a second config identifier.
``GlobalPipelineConfig`` remains outside the document because it is
execution-environment context rather than per-pipeline semantics.

See :doc:`../concepts/pipelines_and_steps` before adding axis, source-binding,
streaming, or materialization behavior. Those values belong in the appropriate
``Lazy*Config`` object; fields such as ``variable_components`` and ``group_by``
are not direct ``FunctionStep`` arguments.

First MCP session
-----------------

The server instructions and ``first_use`` context are intentionally small
discovery routes. An agent should not need a copied list of every OpenHCS tool.
Use this sequence:

1. Call ``openhcs_health_check``. Stop on bootstrap or stale-process errors.
2. Call ``openhcs_get_authoring_context`` with ``kind="first_use"``.
3. Call ``openhcs_search_capabilities`` with the workflow, target, or task text
   from the selected context and choose only tools returned by the active
   server profile. Request ``openhcs_list_capabilities`` only when the complete
   selected registry is required.
4. Search existing knowledge and recipes with ``openhcs_search_knowledge``
   before authoring. For a CellProfiler or benchmark task, include the task
   name and ``OpenHCS Python`` in the query.
5. Retrieve the matching section with ``openhcs_get_knowledge_document``. The
   Official30 source section ids end in ``-openhcs-python`` and are generated
   from the exact manifest-resolved ``.cppipe`` only when requested.
6. Before setting configuration values, call
   ``openhcs_describe_config_schema`` for ``pipeline``, ``global``, or ``step``
   and follow a returned nested ``path_prefix``. Use a field's
   ``authoring_value_path`` to construct the nested JSON accepted by mutations;
   the dotted ``path`` is for schema navigation. Search and describe registered
   functions before using non-default parameters. Create a pipeline-config draft
   first when needed, then pass its id to
   ``openhcs_create_pipeline``; rendering and source-backed execution preserve
   that same config inside the resulting ``PipelineDocument``.
7. Inspect real plate inventory, validate the declaration, inspect its artifact
   plan, and compile before execution. Begin with read-only operations and ask
   before mutation, execution, UI actions, viewer launch, or external access.

For example, search for ``ExampleHuman OpenHCS Python`` and retrieve document
``openhcs_official30_benchmark_recipes`` section
``examplehuman-openhcs-python``. That section defines an importable
``pipeline_config`` and ``pipeline_steps`` pair. The corpus contains 30 such
source-backed recipes; it is the broadest current end-to-end example set.

CellProfiler pipelines
----------------------

The public importer lowers a ``.cppipe`` directly to the same declarations:

.. code-block:: python

   from openhcs.interop.cellprofiler.pipeline_import import (
       import_cellprofiler_pipeline,
   )

   pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
       "analysis.cppipe",
       source_root="/data/plate",
   )

   from openhcs.core.pipeline_document import PipelineDocumentAuthority

   pipeline_document = PipelineDocumentAuthority.from_values(
       pipeline_config=pipeline_config,
       pipeline_steps=pipeline_steps,
   )

Setup modules contribute typed source bindings; executable modules become
``FunctionStep`` objects. There is no second runtime-pipeline model or generated
semantic sidecar. See :doc:`cellprofiler_interop` for the ownership boundary.

Configuration and processing semantics
--------------------------------------

OpenHCS configuration is ObjectState-backed and may inherit lazily across
global, pipeline, and step scopes. Inspect the reflected schema or the nominal
dataclass declarations rather than copying field names into documentation or
client code.

Keep these independent concepts separate:

``variable_components``
  Says what changes along the array's variable axis, such as site, channel, Z,
  or time.

``group_by``
  Groups already assembled arrays and selects branches only for dictionary
  function patterns. It does not define the stack axis.

``ProcessingContract``
  Declares callable locality. ``PURE_2D`` retains per-plane semantics even when
  planes travel in a batch; ``PURE_3D`` means the result depends on the stack.

Unsupported microscope folders
-------------------------------

``SourceBindingsConfig`` is the supported generic input boundary when a folder
does not match a registered microscope handler. A non-empty source-binding
declaration makes auto-detection select ``SourceBindingsHandler``. The agent or
user describes the files; processing callables remain independent of physical
filenames.

For files such as ``A01_s1_DNA.tif`` and ``A01_s1_GFP.tif``:

.. code-block:: python

   from openhcs.constants import AllComponents
   from openhcs.core.config import PipelineConfig
   from openhcs.core.source_bindings import (
       ComponentSelector,
       LazySourceBindingsConfig,
       MetadataExtractionRule,
       MetadataSource,
       NamedSourceBinding,
       SourceFilterClause,
       SourceFilterMatchType,
       SourceFilterSubject,
       SourceSelector,
   )
   from openhcs.processing.backends.processors.numpy_processor import (
       stack_percentile_normalize,
   )

   def channel_binding(alias: str, token: str, channel: str):
       return NamedSourceBinding(
           alias=alias,
           selector=SourceSelector(
               filters=(
                   SourceFilterClause(
                       subject=SourceFilterSubject.FILE,
                       match_type=SourceFilterMatchType.CONTAINS,
                       value=token,
                   ),
               ),
           ),
           component_identity=(
               ComponentSelector(AllComponents.CHANNEL, channel),
           ),
       )

   pipeline_config = PipelineConfig(
       source_bindings_config=LazySourceBindingsConfig(
           source_filters=(
               SourceFilterClause(
                   subject=SourceFilterSubject.EXTENSION,
                   match_type=SourceFilterMatchType.IS_IMAGE,
               ),
           ),
           metadata_rules=(
               MetadataExtractionRule(
                   source=MetadataSource.FILE_NAME,
                   pattern=(
                       r"^(?P<Well>[A-H][0-9]{2})_"
                       r"s(?P<Site>[0-9]+)_(?P<Stain>[^.]+)"
                   ),
               ),
           ),
           bindings=(
               channel_binding("DNA", "DNA", "1"),
               channel_binding("GFP", "GFP", "2"),
           ),
       ),
   )

   pipeline_steps = [
       FunctionStep(
           name="Normalize DNA",
           func=stack_percentile_normalize,
           processing_config=LazyProcessingConfig(
               input_source=InputSource.PIPELINE_START,
           ),
           source_bindings=LazyStepSourceBindingsConfig(
               enabled=True,
               bindings=(NamedSourceBinding(alias="DNA"),),
           ),
       ),
   ]

The two assignments are one complete ``PipelineDocument`` and can be sent as
reviewed Python through the source-backed MCP route or applied through Pipeline
Editor code mode. Do not send ``pipeline_config`` through a separate side
channel. Pipeline-level bindings own filtering, metadata, alias matching,
explicit planes, imported metadata, grouping, and source stack composition. A
step's ``source_bindings`` chooses which named sources it consumes. Compilation turns
those declarations into a typed source universe, binding plans, and a virtual
workspace; source-bound artifact inputs are satisfied there instead of through
invented runtime producers.

Before execution, inspect representative files and confirm required alias
counts, well/site/channel identities, stack axes, virtual paths, original source
paths, and source metadata. See :doc:`source_model` for the full declaration and
compiler model.

Compile before execution
------------------------

Compilation is the semantic boundary, not a convenience validation pass. It
resolves source workspaces, configuration, callable contracts, artifact edges,
materialization, memory conversion, worker requirements, and execution scope.
Runtime workers consume the resulting ``CompiledExecutionBundle`` and typed
runtime values; they do not rediscover those contracts from strings or files.

The desktop application and MCP execution services own progress, cancellation,
and result presentation. For the explicit lower-level Python boundary, see
:doc:`../api/index`.

Current example authorities
---------------------------

Use :doc:`../guides/example_corpus_map` as the retrieval map:

* ``openhcs_official30_benchmark_recipes`` is the current broad recipe corpus.
  Each exact ``*-openhcs-python`` section is generated through the public
  CellProfiler importer and returns ordinary public declarations.
* ``openhcs/processing/presets/mfd_specs.py`` is the typed authority for current
  MFD preset variants. The four ``10x_mfd_*.py`` modules are thin public
  materialization wrappers over that authority.
* Older benchmark scripts and debug exports can still explain a migration, but
  they are not current API examples unless a current test explicitly validates
  them.

Reading map
-----------

* :doc:`system_overview` expands the declaration-to-runtime path.
* :doc:`nominal_ownership` explains where semantics must live.
* :doc:`source_model` covers inventory, source bindings, and virtual workspaces.
* :doc:`../guide_for_biologists/configuration_reference` explains configuration
  scopes, inheritance, and the role of every nested config family.
* :doc:`artifact_contract_system` covers typed producer/consumer edges.
* :doc:`pipeline_compilation_system` covers compiler stages and authorities.
* :doc:`runtime_value_system` covers runtime values and execution products.
* :doc:`cellprofiler_interop` covers compatibility evidence and concrete
  CellProfiler Analyst SQLite/properties export.
* :doc:`code_ui_interconversion` and
  :doc:`../guide_for_biologists/basic_interface` cover UI/code ownership and the
  desktop window map.
* :doc:`../api/index` lists the public imports and low-level integration API.
* :doc:`../guides/example_corpus_map` maps practical examples and recipes.
