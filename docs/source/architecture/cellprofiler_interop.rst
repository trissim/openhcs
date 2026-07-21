CellProfiler interoperability
=============================

CellProfiler interoperability is a pure translation into the same public
declarations used by native OpenHCS pipelines. It does not introduce a second
runtime, generated pipeline class, source-schema layer, or semantic sidecar.

What compatibility means
------------------------

OpenHCS compatibility is semantic, not a promise to run CellProfiler itself as
a hidden subprocess. If you know CellProfiler, the familiar image names,
objects, measurements, relationships, groups, and export modules remain useful:
the importer lowers them to OpenHCS source bindings, ``FunctionStep`` objects,
artifact contracts, runtime values, and plate-scoped exporters.

The ``CellProfilerModule`` registry is the installed-version authority for
module support. The compatibility report derives module and setting coverage
from that registry and the selected ``.cppipe`` corpus; it distinguishes setup
modules absorbed into sources, executable modules, unsupported processing
modules, and settings that are bound, validated, or intentionally rejected.
Official30 recipes provide broad end-to-end import and parity evidence, but an
agent must not turn that corpus into the stronger claim that every historical
CellProfiler version, plugin module, and setting combination has been tested.
For an unfamiliar pipeline, import and compile the actual ``.cppipe`` and treat
an explicit module/setting error as the compatibility boundary.

Import flow
-----------

.. code-block:: text

   .cppipe
      -> CPPipeParser -> ModuleBlock records
      -> CellProfilerModule.require_module(name)
      -> setup modules contribute SourceBindingsConfig
      -> executable modules contribute ordinary FunctionStep declarations
      -> PipelineConfig.from_config(...)
      -> list[FunctionStep], PipelineConfig

``import_cellprofiler_pipeline`` resolves external files relative to the
pipeline directory by default or an explicit ``source_root``. Disabled modules
are ignored. Setup-only declarations do not emit a step; executable module
declarations emit one or more ordinary steps.

What the importer decides
-------------------------

The importer does encode the CellProfiler facts that are present in the parsed
pipeline and its owning module declarations. It derives source stack components
for ``variable_components``, applies callable-owned grouping constraints,
chooses previous-step versus pipeline-start main flow from source and producer
evidence, and resolves named images, objects, measurements, relationships, and
exports into exact callable artifact contracts. These are executable conversion
rules, not prose conventions.

That does not make the importer a universal native-pipeline intent oracle. The
``InputSource`` enum has only previous-step and pipeline-start main flow; an
additional named image is represented by an artifact input plus source binding
or prior producer. A ``.cppipe`` also does not choose arbitrary OpenHCS viewer
streaming, native checkpoint, VFS, or user review policy that it never declared.
Artifact presence in the runtime graph is not by itself persistent
materialization. Native authors still review those independent choices on the
ordinary ``PipelineConfig``, ``FunctionStep``, callable contract, and compiled
artifact/materialization plans.

Nominal module authority
------------------------

``CellProfilerModule`` is the auto-registering root for module semantics. A
module subclass owns:

- the exact CellProfiler module name
- settings binding and repeated-row interpretation
- whether the module emits a step
- its public processing callable or callable batch
- required processing axes and grouping constraints
- source-setup behavior
- callable artifact derivation, exact relations, and context advancement
- execution scope and runtime adapter behavior
- measurement and relationship declarations

Shared behavior composes through inheritance and module mixins. Import and
compiler code query the root registry; they do not maintain a parallel
module-name table.

Settings and public callables
-----------------------------

``SettingsBinder`` translates parsed CellProfiler settings into the public
callable's keyword arguments. The public callable remains visible to GUI/code
transport and signature analysis. During compilation an invocation-contract
provider may derive a runtime adapter or module executor, but that compile-only
object does not replace the public declaration format.

Sources
-------

CellProfiler input/setup modules contribute named image-plane sources, metadata
matching, grouping fields, and other source bindings. The importer builds
``SourceBindingsConfig`` directly and folds it into ``PipelineConfig``. Source
artifact inputs are then satisfied through the normal source planner.

Artifacts and exports
---------------------

Each executable module inherits ``CellProfilerModuleArtifactContracts``. For one
parsed invocation, that mixin resolves active ``SettingToKeywordBinding``
declarations and module leaf hooks into an ordinary ``CallableContract``. There
is no parallel module-contract object. Exact inputs and outputs include images,
object labels, measurements, relationships, grids, tables, and external
resources.

Export modules are explicit steps. Plate-wide exporters declare
``FunctionStepExecutionScope.PLATE`` rather than relying on an implicit
post-pipeline sidecar. Materialization and recording follow the normal artifact
graph and runtime-store paths.

CellProfiler Analyst export
---------------------------

``ExportToDatabase`` is a real plate-scoped OpenHCS step, not a placeholder or
post-run compatibility hook. Its owning module declaration selects the exact
image, object, measurement, relationship, thumbnail, and grouping artifacts.
At runtime:

.. code-block:: text

   typed RuntimeValueStore artifacts
       -> CellProfilerAnalystProjectionBuilder
       -> CPA image/object/relationship table projection
       -> CPASQLiteRenderer
       -> SQLite database + CellProfiler Analyst .properties files

The projection preserves declared table/column identities, image paths and file
names, object locations, image/object keys, group fields, relationships,
classifier metadata, channel display metadata, and optional thumbnails. The
module declares one typed file-bundle artifact; its materialized members are the
SQLite database and one or more ``.properties`` files.

OpenHCS currently implements the SQLite database and CPA ``.properties`` route.
It rejects non-SQLite databases and custom filter rows explicitly. The current
renderer does **not** emit a CellProfiler Analyst ``.workspace`` file and does
not implement every historical per-well aggregation, plate-filter, overwrite,
or workspace-measurement setting. Some of those settings can still be present
in an imported declaration, so their presence is a known parity gap rather than
evidence that the requested side output exists.

``ExportToSpreadsheet`` is also an executable plate-scoped exporter and emits
declared measurement files from the same runtime-store model. Agents should
inspect the imported export step and resulting artifact plan, report any
non-default unsupported CPA settings, then verify the materialized files. They
must never promise CPA or workspace output merely because upstream measurement
steps compiled.

Parity evidence and safe claims
-------------------------------

There are three distinct levels of evidence:

1. **Import coverage**: every enabled module and relevant setting in the
   concrete ``.cppipe`` lowers without an unsupported declaration error.
2. **Execution coverage**: the translated pipeline compiles and runs with
   satisfied typed source/artifact contracts.
3. **Result parity**: images, labels, measurements, relationships, spreadsheets,
   SQLite tables, and CPA properties are compared under their semantic
   equivalence policies.

Only the third level supports a parity claim for the tested case. Use the
Official30 corpus as examples and regression evidence, then validate the user's
actual pipeline and representative data before describing it as equivalent.

Adding a module
---------------

1. Add or extend the owning ``CellProfilerModule`` subclass.
2. Declare settings binding and the public callable.
3. Attach callable execution semantics: processing contract, required axes,
   allowed grouping, runtime parameters, image mode, and scope.
4. Declare exact module artifact inputs, outputs, relations, and source roles.
5. Add parser/lowering, public transport, compiler-plan, runtime, and parity tests.

Do not add module-specific conditionals to the importer, generic compiler,
runtime store, or equivalence engine.

Historical plans
----------------

The runtime unification and import/dispatch consolidation plans record the
migration that produced this boundary. They are not canonical current-state API
documentation and now live in the history archive. Unchecked items in those
snapshots describe migration-time follow-up, not current architectural
authority; current gaps must be tracked against this page and the source tree.
