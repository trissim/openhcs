OpenHCS Core Model
==================

OpenHCS is a compiler/runtime for high-content microscopy workflows. It is not
just a GUI, a viewer, or a collection of image-processing functions. The core
job of OpenHCS is to turn microscopy sources and declared analysis intent into
compiled, executable, inspectable workflows.

For agents and developers, the important model is:

.. code-block:: text

   microscope data + metadata
       -> source model and virtual workspace
       -> FunctionStep pipeline declarations
       -> optional CellProfiler-compatible module/image/object/measurement semantics
       -> compile-time config, axis, artifact, and storage planning
       -> headless or UI-owned runtime execution
       -> inventory, materialized artifacts, viewer payloads, and measurements

Core Summary
------------

Use this summary before choosing tools:

If you do not already know OpenHCS, read the ``first_use`` authoring context
before choosing tools. It is the front-door model for the compiler/runtime, UI
bridge, CellProfiler compatibility, source universes, artifacts, and review
workflow.

* **Data/source model**: microscope folders and metadata become typed source
  inventory, source bindings, and virtual workspace paths.
* **Axis/component model**: wells, sites, channels, Z planes, and timepoints are
  semantic axes; ``variable_components`` controls stacking, while ``group_by``
  controls dictionary routing.
* **Pipeline/function model**: workflows are ordered ``FunctionStep``
  declarations over registry or custom functions; signatures and contracts are
  the kwargs authority.
* **CellProfiler compatibility model**: CellProfiler ``.cppipe`` modules,
  Images, Objects, Measurements, SaveImages, and exports are compiled into
  OpenHCS declarations, source bindings, artifact contracts, runtime values,
  materialization, and measurements.
* **Config/ObjectState model**: lazy configs inherit through global, pipeline,
  and step scopes; ObjectState owns UI state, resolved values, provenance,
  snapshots, branches, and dirty markers.
* **Compiler/artifact model**: compile resolves source bindings, config,
  axis/group semantics, artifact contracts, special IO, materialization,
  storage, memory contracts, and resources before runtime.
* **Source-universe model**: source bindings resolve against compile-planned
  file universes such as current step input, pipeline start, and load
  universes; runtime adapters consume those resolved universes instead of
  scanning paths independently.
* **Runtime artifact/sidecar model**: non-primary outputs are typed artifacts;
  sidecar artifacts, materialized exports, and generated contract sidecars are
  declared by artifact contracts and plans, not ad hoc files.
* **Runtime/UI model**: headless runs are automation jobs; UI-owned runs update
  Plate Manager rows, ObjectState snapshots, visible status, and output rows.
* **UI/code biconversion model**: UI-reflected objects can be edited through
  live ObjectState/code projections; code documents are typed pycodified
  projections with revision tokens, not export/import scripts.
* **Review model**: validate outputs through inventory, materialized artifacts,
  viewer layer/payload records, ROI summaries, sampled pixels, and measurements.

CellProfiler Compatibility Model
--------------------------------

OpenHCS has first-class CellProfiler compatibility integrated into the OpenHCS
compiler/runtime model. Agents that know CellProfiler should use that knowledge
up front: CellProfiler's ordered modules, named images, named objects,
measurements, and explicit exports map directly onto OpenHCS semantic
authorities.

The model is:

* ``.cppipe`` text is parsed into ordered module records.
* CellProfiler ``Images``, ``Metadata``, and ``NamesAndTypes`` setup becomes
  source schema, source bindings, metadata extraction, and virtual workspace
  names.
* CellProfiler module declarations become OpenHCS ``FunctionStep``
  declarations backed by absorbed or native runtime functions.
* CellProfiler image names become semantic source bindings, runtime image
  inputs, or artifact identities.
* CellProfiler object names become object-label runtime values or artifact
  contracts.
* CellProfiler measurement modules become OpenHCS runtime functions plus
  measurement/artifact materialization.
* CellProfiler ``SaveImages`` and table-export modules become materialization
  requirements when those external results are part of the workflow contract.

This is not a separate CellProfiler process bolted onto OpenHCS execution. The
compatibility layer preserves CellProfiler workflow semantics by compiling them
into OpenHCS source bindings, function declarations, artifact contracts,
compiled plans, runtime adapters, and materialized outputs.

The repository includes a CellProfiler corpus that agents should treat as
first-choice reference material: in-tree ``.cppipe`` examples, checked-in
OpenHCS equivalents for ExampleHuman and ExampleFly, and the official30 native
CellProfiler reference set. Use those examples before inventing a new workflow.

Data And Source Model
---------------------

OpenHCS starts from microscopy sources: local plate folders, microscope export
layouts, OMERO-style sources, or OpenHCS-native virtual workspaces. A microscope
handler and metadata layer interpret filenames, directory layout, image
metadata, wells, sites, channels, Z planes, timepoints, pixel size, and other
source facts.

Agents should inspect real inventory before authoring a pipeline. Use plate
inspection, file queries, and image sampling to verify what the folder contains.
Do not guess filename semantics or parse paths locally when source bindings,
metadata handlers, and inventory tools can expose the same facts as typed
records.

Axis And Component Model
------------------------

High-content microscopy data varies over semantic axes such as well, site,
channel, Z plane, and timepoint.

``variable_components`` says which axes are stacked into the array passed to a
callable. For example, site-variable processing usually means the callable sees
the image stack for one site at a time.

``group_by`` is the routing or fanout axis for dictionary function patterns.
For example, grouping by channel lets one ``FunctionStep`` route channel 1 to a
nuclei function and channel 2 to a neurite function.

These are pipeline semantics, not filename heuristics. Compile-time source and
axis planning should derive work units from the declared processing config and
resolved source inventory.

Pipeline And Function Model
---------------------------

A pipeline is an ordered set of ``FunctionStep`` declarations. A step declares
which registered callable or callable pattern should run, what user parameters
should be supplied, and which step-level configs control axes, source input,
materialization, viewers, and related behavior.

Functions come from the OpenHCS function registry or from the custom-function
manager. Function signatures, runtime contracts, artifact declarations, and
memory decorators are the authority for what an agent may pass as kwargs and
what OpenHCS supplies at runtime.

Configuration And ObjectState Model
-----------------------------------

Configuration is layered. ``GlobalPipelineConfig`` carries global or session
defaults. ``PipelineConfig`` and step-level lazy configs can inherit from higher
levels. A raw ``None`` may mean "inherit"; agents should inspect resolved values
through the typed config/ObjectState surfaces before assuming a field is unset
or inactive.

ObjectState is the UI and provenance authority for editable state. It owns
scopes, resolved values, dirty/default markers, snapshots, branches, and time
travel. Raw widgets are not the state model; they are projections over
ObjectState-backed data.

Compiler And Artifact Model
---------------------------

OpenHCS compiles before it executes. Compilation resolves source bindings,
lazy config inheritance, variable components, grouping, artifact inputs and
outputs, special IO, materialization, storage backends, memory contracts, and
resource assignment into execution plans.

Runtime code should consume compiled plans. It should not rediscover source
bindings, hand-match paths, or infer artifact identity from filenames after the
compiler has already produced the execution contract.

Artifact, Sidecar, And Source Universe Model
--------------------------------------------

Source bindings connect semantic source names to concrete files and axis
indices. They are resolved against a source universe: the set of candidate
files that a binding is allowed to inspect. The compiler stores source-binding
and source-universe decisions on the compiled step plan, including whether a
step needs current step-input selector resolution, the original pipeline-start
universe, or a load universe.

At runtime, registered ``SourceUniverseRequest`` and ``SourceUniverseStrategy``
types resolve those planned universes into concrete files, source metadata, and
virtual workspace projections. Runtime adapters should consume that resolved
state. They should not rescan physical folders, reinterpret filenames, or
silently choose a different source universe.

Runtime artifacts are typed non-primary-image values such as images, object
labels, measurements, relationships, tables, spatial grids, metadata, and
special values. ``ArtifactSpec`` declares the artifact name, kind,
materialization policy, required status, and optional sidecar role.

Sidecar artifacts are typed derivative artifacts, not loose companion files.
For example, a crop-mask sidecar is represented by ``ArtifactSidecarRole`` and
must match between producers and consumers during path planning. Runtime
artifact stores and output matching record the observed values against the
compiled artifact contract.

CellProfiler-generated pipelines may also emit JSON or Python semantic
contract sidecars. Those files preserve generated module artifact contracts for
review and reload, but they are generated projections of the same OpenHCS
artifact authority. They are not an independent semantic registry.

Runtime And UI Ownership Model
------------------------------

OpenHCS can run headlessly or through the visible UI.

Headless sessions are useful for automation, tests, and isolated execution.
They can compile, run, stream outputs, and write result plates, but they do not
update the Plate Manager selected rows, ObjectState snapshots, or visible UI
workflow state.

UI-owned workflows preserve user-visible state. When a user should see or
continue editing the work in OpenHCS, agents should use UI bridge code
documents, state surfaces, and selected-plate workflow actions.

UI And Code Biconversion Model
------------------------------

OpenHCS supports live bidirectional UI/code editing over the same reflected
state. The UI is not a separate legacy surface that exports scripts for later
re-import. UI-reflected objects are backed by ObjectState scopes, typed fields,
and code-document projections. If an object is reflected in the UI, agents
should expect to inspect or edit it through ObjectState tools, code documents,
or a semantic UI action.

Code documents are typed, pycodified projections of UI-owned state. Agents read,
validate, and apply code documents with revision tokens so edits mutate the
running UI state without overwriting newer user changes.

Reviewable Python is an interchange and provenance surface. ObjectState remains
the UI state authority. After applying code or dispatching UI-owned init,
compile, or run operations, agents should read state surfaces and operation
status to verify what the UI actually accepted.

Review Model
------------

Successful execution is not only "the command returned". Outputs should be
validated through structured evidence:

* plate and result inventory;
* materialized artifact records;
* sampled image statistics and pixels;
* viewer layer state and payload summaries;
* ROI summaries;
* measurement tables or exported records when relevant.

Screenshots can help humans, but agent validation should use the structured
viewer and inventory tools first.

Agent Operating Rule
--------------------

Use this order when helping a domain expert:

1. If you do not already know OpenHCS, read the ``first_use`` authoring
   context before choosing tools.
2. Inspect the user's real data source.
3. If the task resembles a CellProfiler workflow, search CellProfiler examples
   and official30 references before authoring.
4. Search examples and knowledge documents before inventing a workflow.
5. Search and describe registry functions before drafting steps.
6. Author the smallest useful pipeline or UI code-document edit.
7. Compile or inspect the artifact plan before a full run.
8. Execute a bounded validation.
9. Verify outputs through inventory, artifacts, viewer payloads, ROIs, and
   measurements.

Deep References
---------------

Use these knowledge-base documents for detail:

* ``openhcs_domain_expert_onboarding``
* ``openhcs_example_corpus_map``
* ``openhcs_official30_benchmark_recipes``
* ``openhcs_data_dimensions``
* ``openhcs_pipelines_and_steps``
* ``openhcs_function_patterns``
* ``openhcs_configuration_framework``
* ``openhcs_pipeline_compilation_system``
* ``openhcs_special_io_system``
* ``openhcs_code_ui_interconversion``
* ``openhcs_viewer_management``
* ``openhcs_runtime_system_assembly_rules``
